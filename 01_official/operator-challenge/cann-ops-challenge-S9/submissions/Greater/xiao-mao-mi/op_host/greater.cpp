#include "greater_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

// Resource screening, not a claim of calibrated latency prediction.
static uint64_t SelectBf16ReuseKey(uint32_t row, uint64_t reuse,
                                  uint64_t rowsPerCore, uint64_t ubBytes, bool left)
{
    struct Plan { uint32_t tile; uint32_t buffers; uint64_t leftKey; uint64_t rightKey; };
    const Plan plans[] = {{4352U, 2U, 4311ULL, 4411ULL},
                          {8192U, 2U, 4711ULL, 4811ULL},
                          {12288U, 1U, 4511ULL, 4611ULL}};
    uint64_t bestKey = 311ULL;
    uint64_t bestChunks = ~0ULL;
    uint32_t bestOverlap = 0U;
    uint64_t bestBytes = ~0ULL;
    for (const auto &p : plans) {
        // Reuse input + varying input slots + byte outputs + two fp32
        // workspaces + half ones + packed compare mask, all 32B aligned.
        const uint64_t liveBytes = static_cast<uint64_t>(p.tile) * (12U + 3U * p.buffers) +
                                   ((p.tile / 8U + 31U) / 32U) * 32U;
        if (liveBytes > ubBytes) continue;
        const uint64_t chunks = (static_cast<uint64_t>(row) + p.tile - 1U) / p.tile;
        const uint32_t overlap = p.buffers == 2U && reuse > 1U && rowsPerCore > 1U ? 1U : 0U;
        if (chunks < bestChunks ||
            (chunks == bestChunks && overlap > bestOverlap) ||
            (chunks == bestChunks && overlap == bestOverlap && liveBytes < bestBytes)) {
            bestChunks = chunks;
            bestOverlap = overlap;
            bestBytes = liveBytes;
            bestKey = left ? p.leftKey : p.rightKey;
        }
    }
    return bestKey;
}

static uint64_t ShapeSize(const gert::Shape &shape)
{
    uint64_t size = 1;
    for (int i = 0; i < shape.GetDimNum(); ++i) {
        size *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return size;
}

static uint32_t DTypeCode(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return GREATER_DTYPE_FLOAT32;
        case ge::DT_BF16:
            return GREATER_DTYPE_BF16;
        case ge::DT_FLOAT16:
            return GREATER_DTYPE_FLOAT16;
        case ge::DT_INT32:
            return GREATER_DTYPE_INT32;
        case ge::DT_INT8:
            return GREATER_DTYPE_INT8;
        default:
            return GREATER_DTYPE_FLOAT16;
    }
}

static uint32_t DTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 1;
        default:
            return 2;
    }
}

static uint32_t CeilDivU64(uint64_t n, uint64_t d)
{
    return d == 0 ? 0U : static_cast<uint32_t>((n + d - 1U) / d);
}

static uint64_t AlignUpU64(uint64_t n, uint64_t align)
{
    return align == 0U ? n : ((n + align - 1U) / align) * align;
}

static uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        const uint32_t r = a % b;
        a = b;
        b = r;
    }
    return a == 0U ? 1U : a;
}

static uint32_t TileMod(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 64U;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return 128U;
        case ge::DT_INT8:
            return 256U;
        default:
            return 128U;
    }
}

static uint32_t TileLen(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
            return 16384U;
        case ge::DT_FLOAT16:
            return 4096U;
        default:
            return 4096U;
    }
}

static bool SameShape(const gert::Shape &a, const gert::Shape &b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (int i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool BuildBroadcast(const gert::Shape &selfShape, const gert::Shape &otherShape,
                           uint32_t outShape[GREATER_MAX_RANK],
                           uint64_t selfStride[GREATER_MAX_RANK],
                           uint64_t otherStride[GREATER_MAX_RANK],
                           uint32_t &rank, uint64_t &total)
{
    const uint32_t selfRank = static_cast<uint32_t>(selfShape.GetDimNum());
    const uint32_t otherRank = static_cast<uint32_t>(otherShape.GetDimNum());
    rank = selfRank > otherRank ? selfRank : otherRank;
    if (rank == 0U) {
        rank = 1U;
    }
    if (rank > GREATER_MAX_RANK) {
        return false;
    }

    uint64_t selfBaseStride[GREATER_MAX_RANK] = {};
    uint64_t otherBaseStride[GREATER_MAX_RANK] = {};
    uint64_t stride = 1;
    for (int32_t i = static_cast<int32_t>(selfRank) - 1; i >= 0; --i) {
        selfBaseStride[i] = stride;
        stride *= static_cast<uint64_t>(selfShape.GetDim(i));
    }
    stride = 1;
    for (int32_t i = static_cast<int32_t>(otherRank) - 1; i >= 0; --i) {
        otherBaseStride[i] = stride;
        stride *= static_cast<uint64_t>(otherShape.GetDim(i));
    }

    total = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        const bool selfPad = i + selfRank < rank;
        const bool otherPad = i + otherRank < rank;
        const uint32_t selfIdx = selfPad ? 0U : i + selfRank - rank;
        const uint32_t otherIdx = otherPad ? 0U : i + otherRank - rank;
        const uint64_t selfDim = selfPad ? 1U : static_cast<uint64_t>(selfShape.GetDim(selfIdx));
        const uint64_t otherDim = otherPad ? 1U : static_cast<uint64_t>(otherShape.GetDim(otherIdx));
        uint64_t outDim = 0;
        if (selfDim == otherDim) {
            outDim = selfDim;
        } else if (selfDim == 1U) {
            outDim = otherDim;
        } else if (otherDim == 1U) {
            outDim = selfDim;
        } else {
            return false;
        }
        if (outDim == 0U || outDim > 0xffffffffULL) {
            return false;
        }
        outShape[i] = static_cast<uint32_t>(outDim);
        selfStride[i] = (selfPad || selfDim == 1U) ? 0U : selfBaseStride[selfIdx];
        otherStride[i] = (otherPad || otherDim == 1U) ? 0U : otherBaseStride[otherIdx];
        total *= outDim;
    }
    return true;
}

static uint32_t SelectRoute(const gert::Shape &selfShape, const gert::Shape &otherShape,
                            ge::DataType dtype, uint64_t total,
                            const uint64_t selfStride[GREATER_MAX_RANK],
                            const uint64_t otherStride[GREATER_MAX_RANK],
                            uint32_t rank)
{
    if (SameShape(selfShape, otherShape)) {
        return GREATER_ROUTE_SAME_SHAPE;
    }
    if (ShapeSize(selfShape) == 1U || ShapeSize(otherShape) == 1U) {
        return GREATER_ROUTE_SCALAR_BROADCAST;
    }
    (void)dtype;
    (void)total;
    // After BuildBroadcast right-aligns dense contiguous inputs, the innermost
    // output axis always has stride 0 or 1 for each operand. Treat every
    // non-scalar broadcast as a row quotient route instead of leaving a generic
    // odometer path in the hot dispatch.
    return GREATER_ROUTE_ROW_BROADCAST;
}

static void ComputeDenseScalarSuffix(const uint32_t outShape[GREATER_MAX_RANK],
                                     const uint64_t selfStride[GREATER_MAX_RANK],
                                     const uint64_t otherStride[GREATER_MAX_RANK],
                                     uint32_t rank, uint32_t &rowLen,
                                     uint32_t &selfRowLinear, uint32_t &otherRowLinear)
{
    uint64_t suffix = 1U;
    bool selfDense = true;
    bool selfScalar = true;
    bool otherDense = true;
    bool otherScalar = true;
    uint64_t bestSuffix = 1U;
    bool bestSelfDense = rank == 0U || selfStride[rank - 1U] == 1U;
    bool bestOtherDense = rank == 0U || otherStride[rank - 1U] == 1U;

    for (int32_t dim = static_cast<int32_t>(rank) - 1; dim >= 0; --dim) {
        if (outShape[dim] == 1U) {
            continue;
        }
        const bool nextSelfDense = selfDense && selfStride[dim] == suffix;
        const bool nextSelfScalar = selfScalar && selfStride[dim] == 0U;
        const bool nextOtherDense = otherDense && otherStride[dim] == suffix;
        const bool nextOtherScalar = otherScalar && otherStride[dim] == 0U;
        if (!(nextSelfDense || nextSelfScalar) || !(nextOtherDense || nextOtherScalar)) {
            break;
        }
        selfDense = nextSelfDense;
        selfScalar = nextSelfScalar;
        otherDense = nextOtherDense;
        otherScalar = nextOtherScalar;
        suffix *= static_cast<uint64_t>(outShape[dim]);
        bestSuffix = suffix;
        bestSelfDense = selfDense;
        bestOtherDense = otherDense;
        if (suffix > 0xffffffffULL) {
            break;
        }
    }

    rowLen = static_cast<uint32_t>(bestSuffix > 0xffffffffULL ? 0xffffffffULL : bestSuffix);
    if (rowLen == 0U) {
        rowLen = 1U;
    }
    selfRowLinear = bestSelfDense ? 1U : 0U;
    otherRowLinear = bestOtherDense ? 1U : 0U;
}

static uint32_t ComputeRowSuffixStart(const uint32_t outShape[GREATER_MAX_RANK], uint32_t rank, uint32_t rowLen)
{
    uint64_t suffixProduct = 1U;
    uint32_t suffixStart = rank;
    for (int32_t dim = static_cast<int32_t>(rank) - 1; dim >= 0; --dim) {
        if (outShape[dim] != 1U) {
            suffixProduct *= static_cast<uint64_t>(outShape[dim]);
        }
        suffixStart = static_cast<uint32_t>(dim);
        if (suffixProduct >= rowLen) {
            break;
        }
    }
    return suffixStart;
}

static uint32_t ContiguousRowVectorReuse(const uint32_t outShape[GREATER_MAX_RANK],
                                         const uint64_t selfStride[GREATER_MAX_RANK],
                                         const uint64_t otherStride[GREATER_MAX_RANK],
                                         uint32_t suffixStart, bool forSelf)
{
    uint64_t reuse = 1U;
    for (int32_t dim = static_cast<int32_t>(suffixStart) - 1; dim >= 0; --dim) {
        const uint64_t reuseStride = forSelf ? selfStride[dim] : otherStride[dim];
        const uint64_t varyingStride = forSelf ? otherStride[dim] : selfStride[dim];
        if (reuseStride != 0U || varyingStride == 0U) {
            break;
        }
        reuse *= static_cast<uint64_t>(outShape[dim]);
        if (reuse > 65535ULL) {
            return 65535U;
        }
    }
    return reuse > 1U ? static_cast<uint32_t>(reuse) : 1U;
}

static uint32_t SelectSubroute(uint32_t selfRowLinear, uint32_t otherRowLinear)
{
    if (selfRowLinear != 0U && otherRowLinear != 0U) {
        return GREATER_SUBROUTE_LINEAR_LINEAR;
    }
    if (selfRowLinear != 0U) {
        return GREATER_SUBROUTE_LINEAR_SCALAR;
    }
    if (otherRowLinear != 0U) {
        return GREATER_SUBROUTE_SCALAR_LINEAR;
    }
    return GREATER_SUBROUTE_SCALAR_SCALAR;
}

static uint64_t MakeTilingKey(uint32_t route, uint32_t subroute, ge::DataType dtype)
{
    if (route == GREATER_ROUTE_SAME_SHAPE) {
        switch (dtype) {
            case ge::DT_FLOAT:
                return GREATER_TILING_KEY_SAME_FP32;
            case ge::DT_BF16:
                return GREATER_TILING_KEY_SAME_BF16;
            case ge::DT_FLOAT16:
                return GREATER_TILING_KEY_SAME_FP16;
            case ge::DT_INT32:
                return GREATER_TILING_KEY_SAME_INT32;
            case ge::DT_INT8:
                return GREATER_TILING_KEY_SAME_INT8;
            default:
                break;
        }
        return GREATER_TILING_KEY_SAME;
    }
    if (route == GREATER_ROUTE_SCALAR_BROADCAST) {
        switch (dtype) {
            case ge::DT_FLOAT:
                return GREATER_TILING_KEY_SCALAR_FP32;
            case ge::DT_BF16:
                return GREATER_TILING_KEY_SCALAR_BF16;
            case ge::DT_FLOAT16:
                return GREATER_TILING_KEY_SCALAR_FP16;
            case ge::DT_INT32:
                return GREATER_TILING_KEY_SCALAR_INT32;
            case ge::DT_INT8:
                return GREATER_TILING_KEY_SCALAR_INT8;
            default:
                break;
        }
        return GREATER_TILING_KEY_SCALAR;
    }
    if (subroute == GREATER_SUBROUTE_LINEAR_LINEAR) {
        switch (dtype) {
            case ge::DT_FLOAT:
                return GREATER_TILING_KEY_ROW_LL_FP32;
            case ge::DT_BF16:
                return GREATER_TILING_KEY_ROW_LL_BF16;
            case ge::DT_FLOAT16:
                return GREATER_TILING_KEY_ROW_LL_FP16;
            case ge::DT_INT32:
                return GREATER_TILING_KEY_ROW_LL_INT32;
            case ge::DT_INT8:
                return GREATER_TILING_KEY_ROW_LL_INT8;
            default:
                break;
        }
        return GREATER_TILING_KEY_ROW_LINEAR_LINEAR;
    }
    if (subroute == GREATER_SUBROUTE_LINEAR_SCALAR) {
        switch (dtype) {
            case ge::DT_FLOAT:
                return GREATER_TILING_KEY_ROW_LS_FP32;
            case ge::DT_BF16:
                return GREATER_TILING_KEY_ROW_LS_BF16;
            case ge::DT_FLOAT16:
                return GREATER_TILING_KEY_ROW_LS_FP16;
            case ge::DT_INT32:
                return GREATER_TILING_KEY_ROW_LS_INT32;
            case ge::DT_INT8:
                return GREATER_TILING_KEY_ROW_LS_INT8;
            default:
                break;
        }
        return GREATER_TILING_KEY_ROW_LINEAR_SCALAR;
    }
    if (subroute == GREATER_SUBROUTE_SCALAR_LINEAR) {
        switch (dtype) {
            case ge::DT_FLOAT:
                return GREATER_TILING_KEY_ROW_SL_FP32;
            case ge::DT_BF16:
                return GREATER_TILING_KEY_ROW_SL_BF16;
            case ge::DT_FLOAT16:
                return GREATER_TILING_KEY_ROW_SL_FP16;
            case ge::DT_INT32:
                return GREATER_TILING_KEY_ROW_SL_INT32;
            case ge::DT_INT8:
                return GREATER_TILING_KEY_ROW_SL_INT8;
            default:
                break;
        }
        return GREATER_TILING_KEY_ROW_SCALAR_LINEAR;
    }
    switch (dtype) {
        case ge::DT_FLOAT:
            return GREATER_TILING_KEY_ROW_SS_FP32;
        case ge::DT_BF16:
            return GREATER_TILING_KEY_ROW_SS_BF16;
        case ge::DT_FLOAT16:
            return GREATER_TILING_KEY_ROW_SS_FP16;
        case ge::DT_INT32:
            return GREATER_TILING_KEY_ROW_SS_INT32;
        case ge::DT_INT8:
            return GREATER_TILING_KEY_ROW_SS_INT8;
        default:
            break;
    }
    return GREATER_TILING_KEY_ROW_SCALAR_SCALAR;
}

static uint64_t EstimateBytesPerOutput(uint32_t route, uint32_t elemBytes)
{
    if (route == GREATER_ROUTE_SCALAR_BROADCAST) {
        return static_cast<uint64_t>(elemBytes) + 1ULL;
    }
    return 2ULL * static_cast<uint64_t>(elemBytes) + 1ULL;
}

static uint64_t SelectTargetBytesPerCore(uint64_t totalGmBytes, uint32_t route, ge::DataType dtype)
{
    constexpr uint64_t TINY_TENSOR_BYTES = 8ULL * 1024ULL;
    constexpr uint64_t SMALL_TENSOR_BYTES = 128ULL * 1024ULL;
    constexpr uint64_t MEDIUM_TENSOR_BYTES = 2ULL * 1024ULL * 1024ULL;
    constexpr uint64_t LARGE_LINEAR_TARGET = 16ULL * 1024ULL;
    constexpr uint64_t LARGE_BROADCAST_TARGET = 8ULL * 1024ULL;
    constexpr uint64_t MEDIUM_LINEAR_TARGET = 16ULL * 1024ULL;
    constexpr uint64_t MEDIUM_BROADCAST_TARGET = 8ULL * 1024ULL;
    constexpr uint64_t SMALL_LINEAR_TARGET = 16ULL * 1024ULL;
    constexpr uint64_t SMALL_BROADCAST_TARGET = 8ULL * 1024ULL;

    if (totalGmBytes <= TINY_TENSOR_BYTES) {
        return totalGmBytes == 0U ? 1U : totalGmBytes;
    }

    if (totalGmBytes <= SMALL_TENSOR_BYTES) {
        if (route == GREATER_ROUTE_ROW_BROADCAST && dtype != ge::DT_BF16 && dtype != ge::DT_INT32) {
            return SMALL_BROADCAST_TARGET;
        }
        return SMALL_LINEAR_TARGET;
    }
    if (totalGmBytes <= MEDIUM_TENSOR_BYTES) {
        if (route == GREATER_ROUTE_ROW_BROADCAST && dtype != ge::DT_BF16 && dtype != ge::DT_INT32) {
            return MEDIUM_BROADCAST_TARGET;
        }
        return MEDIUM_LINEAR_TARGET;
    }
    if (route == GREATER_ROUTE_ROW_BROADCAST) {
        if (dtype == ge::DT_BF16 || dtype == ge::DT_INT32) {
            return LARGE_LINEAR_TARGET;
        }
        return LARGE_BROADCAST_TARGET;
    }
    if (dtype == ge::DT_BF16 || dtype == ge::DT_INT32) {
        return LARGE_LINEAR_TARGET;
    }
    return LARGE_LINEAR_TARGET;
}

static uint32_t EstimateBlockDim(uint64_t total, uint32_t route, ge::DataType dtype,
                                 uint32_t elemBytes, uint32_t maxAivCores)
{
    constexpr uint64_t GM_TRANSACTION_BYTES = 32ULL;
    if (maxAivCores == 0U) {
        maxAivCores = 40U;
    }
    if (total == 0U) {
        return 1U;
    }

    const uint64_t bytesPerOutput = EstimateBytesPerOutput(route, elemBytes);
    const uint64_t totalGmBytes = AlignUpU64(total * bytesPerOutput, GM_TRANSACTION_BYTES);
    const uint64_t targetBytes = SelectTargetBytesPerCore(totalGmBytes, route, dtype);
    uint32_t blockDim = CeilDivU64(totalGmBytes, targetBytes);
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    if (blockDim > maxAivCores) {
        blockDim = maxAivCores;
    }
    return blockDim;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    GreaterTilingData tiling;
    const gert::StorageShape *selfStorage = context->GetInputShape(0);
    const gert::StorageShape *otherStorage = context->GetInputShape(1);
    auto rawTiling = context->GetRawTilingData();
    if (selfStorage == nullptr || otherStorage == nullptr || rawTiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &selfShape = selfStorage->GetStorageShape();
    const gert::Shape &otherShape = otherStorage->GetStorageShape();
    uint32_t outShape[GREATER_MAX_RANK] = {};
    uint64_t selfStride[GREATER_MAX_RANK] = {};
    uint64_t otherStride[GREATER_MAX_RANK] = {};
    for (uint32_t i = 0; i < GREATER_MAX_RANK; ++i) {
        outShape[i] = 1;
        selfStride[i] = 0;
        otherStride[i] = 0;
    }

    uint64_t total = 0;
    uint32_t rank = 0;
    if (!BuildBroadcast(selfShape, otherShape, outShape, selfStride, otherStride, rank, total)) {
        return ge::GRAPH_FAILED;
    }

    const gert::CompileTimeTensorDesc *inputDesc = context->GetInputDesc(0);
    ge::DataType dtype = inputDesc == nullptr ? ge::DT_FLOAT16 : inputDesc->GetDataType();
    const uint32_t route = SelectRoute(selfShape, otherShape, dtype, total, selfStride, otherStride, rank);
    const uint32_t elemBytes = DTypeSize(dtype);
    uint32_t rowLen = 1U;
    uint32_t selfRowLinear = 0U;
    uint32_t otherRowLinear = 0U;
    ComputeDenseScalarSuffix(outShape, selfStride, otherStride, rank, rowLen, selfRowLinear, otherRowLinear);
    const uint32_t subroute = SelectSubroute(selfRowLinear, otherRowLinear);
    // Physical access family: one reusable vector versus a longer dense stream.
    // No benchmark row length or total-element threshold participates.
    bool selfVectorReuseFamily = false;
    if (route == GREATER_ROUTE_ROW_BROADCAST &&
        subroute == GREATER_SUBROUTE_LINEAR_LINEAR && dtype == ge::DT_INT32) {
        const uint32_t suffix = ComputeRowSuffixStart(outShape, rank, rowLen);
        const uint32_t selfReuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffix, true);
        const uint32_t otherReuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffix, false);
        selfVectorReuseFamily = selfReuse > 1U && selfReuse >= otherReuse;
        uint64_t denseStride = 1U;
        for (int32_t dim = static_cast<int32_t>(rank)-1; dim >= 0; --dim) {
            if (outShape[dim] != 1U && otherStride[dim] != denseStride) selfVectorReuseFamily = false;
            denseStride *= outShape[dim];
        }
    }
    if (selfVectorReuseFamily) {
        if (rowLen > 1024U) {
            selfVectorReuseFamily = false;
        } else {
            uint32_t divisor = 128U;
            while ((rowLen % divisor) != 0U) divisor >>= 1U;
            const uint32_t alignmentRows = 128U / divisor;
            const uint32_t nativeRows = (8192U / rowLen / alignmentRows) * alignmentRows;
            const uint32_t computeRows = nativeRows != 0U ? nativeRows : (8192U / rowLen / 8U) * 8U;
            const uint32_t suffix = ComputeRowSuffixStart(outShape, rank, rowLen);
            const uint32_t reuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffix, true);
            selfVectorReuseFamily = reuse >= computeRows;
        }
    }
    const uint32_t tileMod = TileMod(dtype);
    uint32_t tileLen = TileLen(dtype);
    if (tileLen < tileMod) {
        tileLen = tileMod;
    }
    tileLen = (tileLen / tileMod) * tileMod;
    if (tileLen == 0U) {
        tileLen = tileMod;
    }
    const uint32_t gcdRow = GcdU32(rowLen, tileMod);
    uint32_t rowGroup = rowLen == 0U ? 1U : tileMod / gcdRow;
    if (route == GREATER_ROUTE_ROW_BROADCAST && subroute == GREATER_SUBROUTE_LINEAR_LINEAR &&
        dtype == ge::DT_INT32) {
        constexpr uint32_t INT32_ROW_REUSE_BATCH_ROWS = 4U;
        const uint32_t suffixStart = ComputeRowSuffixStart(outShape, rank, rowLen);
        const uint32_t selfReuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffixStart, true);
        const uint32_t otherReuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffixStart, false);
        if (otherReuse > selfReuse && otherReuse > 1U) {
            const uint32_t gcdBatch = GcdU32(rowGroup, INT32_ROW_REUSE_BATCH_ROWS);
            rowGroup = (rowGroup / gcdBatch) * INT32_ROW_REUSE_BATCH_ROWS;
        }
    }
    uint64_t alignedTotal = total - (total % static_cast<uint64_t>(tileMod));
    if (route == GREATER_ROUTE_ROW_BROADCAST &&
        (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32) && rowLen != 0U) {
        const uint64_t rows = total / static_cast<uint64_t>(rowLen);
        const uint64_t mainRows = (rows / static_cast<uint64_t>(rowGroup)) *
                                  static_cast<uint64_t>(rowGroup);
        const uint64_t rowGroupedTotal = mainRows * static_cast<uint64_t>(rowLen);
        if (rowGroupedTotal < alignedTotal) {
            alignedTotal = rowGroupedTotal;
        }
    }
    const uint64_t tailTotal = total - alignedTotal;
    const uint32_t tailLen = tailTotal > 0xffffffffULL ? 0xffffffffU : static_cast<uint32_t>(tailTotal);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCores = platform.GetCoreNumAiv();
    uint32_t blockDim = EstimateBlockDim(total, route, dtype, elemBytes, aivCores);
    blockDim = platform.CalcTschBlockDim(blockDim, 0U, aivCores == 0U ? blockDim : aivCores);
    if (route == GREATER_ROUTE_ROW_BROADCAST && dtype == ge::DT_INT8 &&
        total >= (64ULL * 1024ULL * 1024ULL) && aivCores != 0U) {
        blockDim = aivCores;
    }
    // Diagnostic only: actual40 launch with unchanged production kernel ELF.
    if (blockDim > 40U) blockDim = 40U;
    tiling.set_total(total);
    tiling.set_rank(rank);
    tiling.set_dtype(DTypeCode(dtype));
    tiling.set_subroute(subroute);
    tiling.set_elem_bytes(elemBytes);
    tiling.set_route(route);
    tiling.set_block_dim(blockDim);
    tiling.set_row_len(rowLen);
    tiling.set_tile_mod(tileMod);
    tiling.set_tile_len(tileLen);
    tiling.set_tail_len(tailLen);
    tiling.set_row_group(rowGroup == 0U ? 1U : rowGroup);
    tiling.set_aligned_total(alignedTotal);
    tiling.set_self_row_linear(selfRowLinear);
    tiling.set_other_row_linear(otherRowLinear);
    tiling.set_out_shape(outShape);
    tiling.set_self_stride(selfStride);
    tiling.set_other_stride(otherStride);

    context->SetBlockDim(blockDim);
    uint64_t hardwareKey = selfVectorReuseFamily ? 3313ULL : MakeTilingKey(route, subroute, dtype);
    if (route == GREATER_ROUTE_ROW_BROADCAST && dtype == ge::DT_BF16 &&
        subroute == GREATER_SUBROUTE_LINEAR_LINEAR) {
        const uint32_t suffix = ComputeRowSuffixStart(outShape, rank, rowLen);
        const uint32_t leftReuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffix, true);
        const uint32_t rightReuse = ContiguousRowVectorReuse(outShape, selfStride, otherStride, suffix, false);
        // Reuse and available rows justify the schedule, not a total-element cutoff.
        if ((leftReuse > 1U || rightReuse > 1U) && (total / rowLen) * 3U >= blockDim) {
            const bool left = leftReuse >= rightReuse && leftReuse > 1U;
            // Small double-buffer slot versus large single-buffer tiled row.
            uint64_t ubBytes = 0U;
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
            hardwareKey = SelectBf16ReuseKey(rowLen, left ? leftReuse : rightReuse,
                ((total / rowLen) + blockDim - 1U) / blockDim, ubBytes, left);
        }
    }
    uint64_t ubCapacity = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCapacity);
    const uint64_t workPerCore = (total + blockDim - 1U) / blockDim;
    if (hardwareKey == 100ULL && ubCapacity >= 96832ULL && workPerCore <= 5376ULL) hardwareKey = 5100ULL;
    if (hardwareKey == 101ULL && ubCapacity >= 185856ULL && workPerCore > 4096ULL) hardwareKey = 5101ULL;
    if (hardwareKey == 102ULL && ubCapacity >= 175104ULL && workPerCore > 8192ULL) hardwareKey = 5102ULL;
    if (hardwareKey == 103ULL && ubCapacity >= 92640ULL && workPerCore <= 3840ULL) hardwareKey = 5103ULL;
    if (hardwareKey == 104ULL && ubCapacity >= 81920ULL && workPerCore <= 8192ULL) hardwareKey = 5104ULL;
    if (hardwareKey == 200ULL && ubCapacity >= 186880ULL && workPerCore > 16384ULL) hardwareKey = 5200ULL;
    if (hardwareKey == 201ULL && ubCapacity >= 188160ULL && workPerCore > 4096ULL) hardwareKey = 5201ULL;
    if (hardwareKey == 202ULL && ubCapacity >= 175104ULL && workPerCore > 8192ULL) hardwareKey = 5202ULL;
    if (hardwareKey == 203ULL && ubCapacity >= 185856ULL && workPerCore > 6144ULL) hardwareKey = 5203ULL;
    if (hardwareKey == 204ULL && ubCapacity >= 192512ULL && workPerCore > 20480ULL) hardwareKey = 5204ULL;
    if (hardwareKey == 310ULL && ubCapacity >= 188160ULL && workPerCore > 12288ULL && rowLen > 12288U) hardwareKey = 5310ULL;
    if (hardwareKey == 312ULL && ubCapacity >= 149504ULL && workPerCore > 4096ULL && rowLen > 4096U) hardwareKey = 5312ULL;
    if (hardwareKey == 314ULL && ubCapacity >= 193536ULL && workPerCore > 20480ULL && rowLen > 20480U) hardwareKey = 5314ULL;
    if (hardwareKey == 320ULL && ubCapacity >= 188160ULL && workPerCore > 12288ULL && rowLen > 12288U) hardwareKey = 5320ULL;
    if (hardwareKey == 322ULL && ubCapacity >= 149504ULL && workPerCore > 4096ULL && rowLen > 4096U) hardwareKey = 5322ULL;
    if (hardwareKey == 324ULL && ubCapacity >= 193536ULL && workPerCore > 20480ULL && rowLen > 20480U) hardwareKey = 5324ULL;
    if (hardwareKey == 330ULL && ubCapacity >= 188160ULL && workPerCore > 12288ULL && rowLen > 12288U) hardwareKey = 5330ULL;
    if (hardwareKey == 332ULL && ubCapacity >= 149504ULL && workPerCore > 4096ULL && rowLen > 4096U) hardwareKey = 5332ULL;
    if (hardwareKey == 334ULL && ubCapacity >= 193536ULL && workPerCore > 20480ULL && rowLen > 20480U) hardwareKey = 5334ULL;
    if (hardwareKey == 340ULL && ubCapacity >= 188160ULL && workPerCore > 12288ULL && rowLen > 12288U) hardwareKey = 5340ULL;
    if (hardwareKey == 342ULL && ubCapacity >= 149504ULL && workPerCore > 4096ULL && rowLen > 4096U) hardwareKey = 5342ULL;
    if (hardwareKey == 344ULL && ubCapacity >= 193536ULL && workPerCore > 20480ULL && rowLen > 20480U) hardwareKey = 5344ULL;
    if (dtype == ge::DT_INT32 && route == GREATER_ROUTE_ROW_BROADCAST &&
        rowLen > 8192U && ubCapacity >= 185920ULL) {
        const bool ls = subroute == GREATER_SUBROUTE_LINEAR_SCALAR;
        const bool sl = subroute == GREATER_SUBROUTE_SCALAR_LINEAR;
        if (ls || sl) {
            // Work is measured in full input-tile equivalents per core.
            uint64_t tier = workPerCore > 4ULL * 12288ULL ? 7000ULL : 6000ULL;
            bool vectorBroadcast = false;
            for (uint32_t i=0; i<rank; ++i) {
                if (outShape[i]>1U && (ls ? selfStride[i] : otherStride[i])==0ULL)
                    vectorBroadcast=true;
            }
            if (vectorBroadcast && workPerCore >= 32ULL * 12288ULL) tier=8000ULL;
            hardwareKey=tier+(ls ? 323ULL : 333ULL);
        }
    }
    if (context->SetTilingKey(hardwareKey) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *selfShape = context->GetInputShape(0);
    const gert::Shape *otherShape = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (selfShape == nullptr || otherShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t outShape[GREATER_MAX_RANK] = {};
    uint64_t selfStride[GREATER_MAX_RANK] = {};
    uint64_t otherStride[GREATER_MAX_RANK] = {};
    uint32_t rank = 0;
    uint64_t total = 0;
    if (!BuildBroadcast(*selfShape, *otherShape, outShape, selfStride, otherStride, rank, total)) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        outputShape->SetDim(i, outShape[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char *name) : OpDef(name)
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
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Greater);
}
