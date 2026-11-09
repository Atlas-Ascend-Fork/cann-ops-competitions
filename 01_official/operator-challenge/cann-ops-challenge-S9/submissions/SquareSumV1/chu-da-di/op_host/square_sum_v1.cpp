#include "square_sum_v1_tiling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

#ifndef OP_LOGE
#define OP_LOGE(...) ((void)0)
#endif
#ifndef OP_CHECK_IF
#define OP_CHECK_IF(condition, log_statement, return_statement) \
    do {                                                        \
        if (condition) {                                        \
            log_statement;                                      \
            return_statement;                                   \
        }                                                       \
    } while (false)
#endif
#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    do {                                             \
        if ((pointer) == nullptr) {                  \
            return ge::GRAPH_FAILED;                 \
        }                                            \
    } while (false)
#endif

namespace s9squaresum {
namespace {
constexpr uint64_t kDefaultUbBytes = 192 * 1024;
constexpr uint64_t kUbReserveBytes = 16 * 1024;
constexpr uint64_t kMaxTileBytes = 64 * 1024;
constexpr uint64_t kFastArMaxReduceElements = 64;
constexpr uint64_t kFastArBytesPerCoreTarget = 64 * 1024;
constexpr uint64_t kMaxStridedVectorTileElements = 16 * 1024;
constexpr uint64_t kLongArPartialWidth = 64;
constexpr uint64_t kLongArRowsPerBatch = 64;
constexpr uint64_t kLongArBatchElements = kLongArPartialWidth * kLongArRowsPerBatch;
constexpr uint64_t kPrefixSmallInnerMax = 16;
constexpr uint64_t kPrefixGatherMaxRows = 2048;
constexpr uint64_t kReduceAllMultiCoreMinBytes = 1024 * 1024;
constexpr uint64_t kReduceAllPartialStrideBytes = 32;
// Hardware synchronization is enabled by MIX_AIV_1_0 and schedule mode 1.
constexpr bool kEnableReduceAllMultiCore = true;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

uint64_t Gcd(uint64_t left, uint64_t right)
{
    while (right != 0) {
        const uint64_t remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

uint64_t Product(const std::vector<uint64_t>& values, size_t begin, size_t end)
{
    uint64_t result = 1;
    for (size_t i = begin; i < end; ++i) {
        if (values[i] != 0 && result > UINT64_MAX / values[i]) {
            return 0;
        }
        result *= values[i];
    }
    return result;
}


uint64_t FloorPowerOfTwo(uint64_t value)
{
    uint64_t power=1;while(power<=value/2)power*=2;return value?power:0;
}
uint64_t RarBufferBytes(uint64_t rows,uint64_t inner)
{
    const uint64_t count=rows*inner,pitch=AlignUp(CeilDiv(rows,16),16);
    return 4*AlignUp(count,16)+2*AlignUp(count+128,16)+4*AlignUp(rows,8)+32*pitch+2*AlignUp(rows,16)+8*AlignUp(rows,8);
}

uint32_t DataTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
            return 4;
        default:
            return 0;
    }
}

bool ReadAxes(gert::TilingContext* context, uint32_t rank, std::vector<int32_t>& axes)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const auto* value = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (value == nullptr || value->GetSize() == 0 || value->GetSize() > rank) {
        return false;
    }
    const int64_t* data = static_cast<const int64_t*>(value->GetData());
    for (size_t i = 0; i < value->GetSize(); ++i) {
        int64_t axis = data[i];
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return false;
        }
        axes.push_back(static_cast<int32_t>(axis));
    }
    std::sort(axes.begin(), axes.end());
    return std::adjacent_find(axes.begin(), axes.end()) == axes.end();
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<SquareSumV1TilingData>();
    const auto* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *tiling = {};

    const auto& shapeView = xShape->GetStorageShape();
    uint32_t rank = static_cast<uint32_t>(shapeView.GetDimNum());
    OP_CHECK_IF(rank == 0 || rank > kMaxRank,
        OP_LOGE(context->GetNodeName(), "rank must be in [1, %u], got %u", kMaxRank, rank),
        return ge::GRAPH_FAILED);

    std::vector<uint64_t> shape(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        const int64_t extent = shapeView.GetDim(axis);
        OP_CHECK_IF(extent <= 0, OP_LOGE(context->GetNodeName(), "all dimensions must be positive"),
            return ge::GRAPH_FAILED);
        shape[axis] = static_cast<uint64_t>(extent);
    }
    std::vector<int32_t> axes;
    OP_CHECK_IF(!ReadAxes(context, rank, axes), OP_LOGE(context->GetNodeName(), "invalid or duplicate axis"),
        return ge::GRAPH_FAILED);

    // Unit dimensions carry no address information. Collapse maximal equal-role
    // runs after removing them; output shape inference still uses original axes.
    std::vector<uint64_t> effectiveShape;
    std::vector<int32_t> effectiveAxes;
    bool previousReduced = false;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] == 1) continue;
        const bool reduced = std::binary_search(axes.begin(), axes.end(), static_cast<int32_t>(axis));
        if (!effectiveShape.empty() && reduced == previousReduced) {
            OP_CHECK_IF(effectiveShape.back() > UINT64_MAX / shape[axis],
                OP_LOGE(context->GetNodeName(), "shape product overflow"), return ge::GRAPH_FAILED);
            effectiveShape.back() *= shape[axis];
        } else {
            if (reduced) effectiveAxes.push_back(static_cast<int32_t>(effectiveShape.size()));
            effectiveShape.push_back(shape[axis]);
        }
        previousReduced = reduced;
    }
    if (effectiveShape.empty()) {
        effectiveShape.push_back(1); effectiveAxes.push_back(0);
    } else if (effectiveAxes.empty()) {
        // Reducing only unit dimensions is elementwise squaring.
        effectiveAxes.push_back(static_cast<int32_t>(effectiveShape.size()));
        effectiveShape.push_back(1);
    }
    shape = std::move(effectiveShape); axes = std::move(effectiveAxes);
    rank = static_cast<uint32_t>(shape.size());

    std::vector<int32_t> nonReduce;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (!std::binary_search(axes.begin(), axes.end(), static_cast<int32_t>(axis))) {
            nonReduce.push_back(axis);
        }
    }

    const uint32_t dtypeBytes = DataTypeBytes(context->GetInputDesc(0)->GetDataType());
    OP_CHECK_IF(dtypeBytes == 0, OP_LOGE(context->GetNodeName(), "unsupported input dtype"),
        return ge::GRAPH_FAILED);
    const uint64_t total = Product(shape, 0, shape.size());
    OP_CHECK_IF(total == 0, OP_LOGE(context->GetNodeName(), "shape product overflow"),
        return ge::GRAPH_FAILED);

    bool consecutive = true;
    for (size_t i = 1; i < axes.size(); ++i) {
        consecutive = consecutive && axes[i] == axes[i - 1] + 1;
    }
    Mode mode = Mode::kARA;
    if (axes.size() == rank) {
        mode = Mode::kReduceAll;
    } else if (consecutive && axes.back() == static_cast<int32_t>(rank - 1)) {
        mode = Mode::kAR;
    } else if (consecutive && axes.front() == 0) {
        mode = Mode::kRA;
    }

    // A non-consecutive reduction whose preserved axes form one middle run
    // has the collapsed layout [reduceOuter, output, reduceInner].  Each
    // output value is therefore a regular 2-D GM gather, not a genuinely
    // arbitrary index problem.  Preserve this predicate for a dedicated UB
    // vector reduction below.
    bool splitReduce = false;
    if (mode == Mode::kARA && !consecutive && !nonReduce.empty()) {
        bool nonReduceConsecutive = true;
        for (size_t i = 1; i < nonReduce.size(); ++i) {
            nonReduceConsecutive =
                nonReduceConsecutive && nonReduce[i] == nonReduce[i - 1] + 1;
        }
        splitReduce = nonReduceConsecutive && nonReduce.front() > 0 &&
                      nonReduce.back() < static_cast<int32_t>(rank - 1);
    }
    // A consecutive reduction block strictly between preserved prefix and
    // suffix dimensions has layout [outer, reduce, inner].  It is distinct
    // from splitReduce: a 2-D gather here collects reduction rows, after
    // which the kernel performs a vector MulAdd over each inner column tile.
    const bool middleReduce = consecutive && axes.front() > 0 &&
                              axes.back() < static_cast<int32_t>(rank - 1);
    // For a truly non-consecutive axis set, a preserved trailing suffix is
    // still physically contiguous.  Vectorize that entire suffix and pay the
    // multi-axis div/mod cost once per reduction slice/tile instead of once
    // per output element. Example: [A,B,C,D], axes=[0,2] reduces to BxD.
    const bool genericPreservedSuffix =
        mode == Mode::kARA && !consecutive && !splitReduce &&
        axes.back() < static_cast<int32_t>(rank - 1);

    const bool genericReducedSuffix = mode == Mode::kARA && !consecutive &&
        !splitReduce && axes.back() == static_cast<int32_t>(rank - 1);

    tiling->rank = rank;
    tiling->mode = static_cast<uint32_t>(mode);
    tiling->dtypeBytes = dtypeBytes;
    tiling->axisCount = static_cast<uint32_t>(axes.size());
    tiling->outputRank = static_cast<uint32_t>(nonReduce.size());
    const auto* keepDims = context->GetAttrs()->GetAttrPointer<bool>(1);
    tiling->keepDims = keepDims != nullptr && *keepDims ? 1 : 0;
    tiling->totalElements = total;
    tiling->reduce = 1;
    for (int32_t axis : axes) {
        tiling->reduce *= shape[axis];
    }
    tiling->outputElements = total / tiling->reduce;

    uint64_t stride = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 1; axis >= 0; --axis) {
        tiling->shape[axis] = shape[axis];
        tiling->inputStride[axis] = stride;
        stride *= shape[axis];
    }
    for (size_t i = 0; i < axes.size(); ++i) {
        tiling->axes[i] = axes[i];
    }
    for (size_t i = 0; i < nonReduce.size(); ++i) {
        tiling->nonReduceAxes[i] = nonReduce[i];
    }

    if (mode == Mode::kReduceAll) {
        tiling->outer = 1;
        tiling->inner = 1;
    } else if (consecutive) {
        tiling->outer = Product(shape, 0, axes.front());
        tiling->inner = Product(shape, axes.back() + 1, shape.size());
    } else if (splitReduce) {
        tiling->outer = Product(shape, 0, nonReduce.front());
        tiling->inner = Product(shape, nonReduce.back() + 1, shape.size());
    } else if (genericPreservedSuffix) {
        size_t suffixStart = shape.size();
        while (suffixStart > 0 &&
               !std::binary_search(axes.begin(), axes.end(),
                                   static_cast<int32_t>(suffixStart - 1))) {
            --suffixStart;
        }
        tiling->inner = Product(shape, suffixStart, shape.size());
        tiling->outer = tiling->outputElements / tiling->inner;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubBytes = kDefaultUbBytes;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t usableBytes = ubBytes > kUbReserveBytes ? ubBytes - kUbReserveBytes : ubBytes;
    const uint64_t blockElements = 32 / dtypeBytes;
    const uint64_t rawTileElements = std::min(kMaxTileBytes, usableBytes / 3) / dtypeBytes;
    tiling->tileElements = std::max<uint64_t>(
        blockElements, rawTileElements / blockElements * blockElements);

    uint64_t tilingKey = static_cast<uint64_t>(tiling->mode) * 10 + dtypeBytes;
    uint64_t userWorkspaceBytes = 0;
    uint32_t blockDim = mode == Mode::kReduceAll
        ? 1
        : static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreNum, tiling->outputElements)));

    // WholeReduceSum<float> handles at most 64 values per repeat on A2.  A
    // short contiguous suffix therefore admits a much cheaper path: pad a
    // batch of rows in UB, promote once, square vectorially, and reduce every
    // row in one multi-repeat instruction.  The condition depends only on the
    // normalized AR layout and hardware limits, never on a published shape.
    if (mode == Mode::kAR ||
        (mode == Mode::kReduceAll && tiling->reduce <= kFastArMaxReduceElements)) {
        const uint64_t rawAlignment = 32 / dtypeBytes;
        const uint64_t reduceAligned = AlignUp(tiling->reduce, rawAlignment);
        if (reduceAligned != 0 && reduceAligned <= kFastArMaxReduceElements) {
            const uint64_t bytesPerRow = reduceAligned * (dtypeBytes + sizeof(float)) +
                                         sizeof(float) + dtypeBytes;
            const uint64_t maxRowsByUb = bytesPerRow == 0 ? 0 : usableBytes / bytesPerRow;
            const uint64_t rowTile = std::min<uint64_t>({tiling->outputElements, maxRowsByUb, 255});
            if (rowTile != 0) {
                tiling->reduceAligned = reduceAligned;
                tiling->rowTile = rowTile;
                const uint64_t inputBytes = tiling->totalElements * dtypeBytes;
                const uint64_t requestedCores = std::max<uint64_t>(
                    1, CeilDiv(inputBytes, kFastArBytesPerCoreTarget));
                blockDim = static_cast<uint32_t>(std::min<uint64_t>({
                    static_cast<uint64_t>(coreNum), tiling->outputElements, requestedCores}));
                // A short reduction can still have hundreds of independent
                // output rows.  The byte-volume heuristic leaves those rows
                // on one AIV, so the kernel executes several 255-repeat
                // batches serially.  Once one AIV cannot cover all rows in a
                // single batch, distribute 32-byte output ownership units
                // across the available AIVs.  Unit-aligned ownership also
                // prevents two AIVs from performing short GM writes into the
                // same cache line.
                if (mode == Mode::kAR && tiling->outputElements > rowTile) {
                    const uint64_t outputBlockElements = 32 / dtypeBytes;
                    const uint64_t outputBlocks =
                        CeilDiv(tiling->outputElements, outputBlockElements);
                    blockDim = static_cast<uint32_t>(std::min<uint64_t>(
                        static_cast<uint64_t>(coreNum), outputBlocks));
                    tiling->tileElements = outputBlockElements;
                    tiling->reserved0 = 2;
                }
                tilingKey = dtypeBytes == 2 ? kTilingKeyArVector16 : kTilingKeyArVector32;
            }
        }
    }
    // Longer contiguous rows (and reduce-all, which is one such row) still
    // admit a workspace-free vector path.  Stream 64x64 elements per batch,
    // square in FP32, reduce each 64-element lane, then reduce the at-most-64
    // partials.  This removes one scalar GM transaction per input element
    // without introducing a second kernel launch or inter-core atomics.
    if ((mode == Mode::kAR && tiling->reduce > kFastArMaxReduceElements) ||
        (mode == Mode::kReduceAll && tiling->reduce > kFastArMaxReduceElements)) {
        tiling->reduceAligned = kLongArPartialWidth;
        tiling->rowTile = kLongArRowsPerBatch;
        const uint64_t inputBytes = tiling->totalElements * dtypeBytes;
        const uint64_t requestedCores = mode == Mode::kReduceAll
            ? 1
            : std::max<uint64_t>(1, CeilDiv(inputBytes, kFastArBytesPerCoreTarget));
        blockDim = static_cast<uint32_t>(std::min<uint64_t>({
            static_cast<uint64_t>(coreNum), tiling->outputElements, requestedCores}));
        tilingKey = dtypeBytes == 2 ? kTilingKeyLongArVector16 : kTilingKeyLongArVector32;
        // Reduce-all has only one logical output, but its input is freely
        // partitionable.  For large tensors every AIV writes one FP32 partial
        // to a cache-line-disjoint workspace slot, then a hardware barrier and
        // a 48-value final fold replace the old single-core full scan.
        if (kEnableReduceAllMultiCore && mode == Mode::kReduceAll && coreNum > 1 &&
            tiling->totalElements >= CeilDiv(kReduceAllMultiCoreMinBytes, dtypeBytes)) {
            blockDim = coreNum;
            userWorkspaceBytes =
                static_cast<uint64_t>(blockDim) * kReduceAllPartialStrideBytes;
            tilingKey = dtypeBytes == 2 ? kTilingKeyReduceAllMultiCore16
                                       : kTilingKeyReduceAllMultiCore32;
        }
    }
    // For [reduceOuter, output, reduceInner], gather one output's reduction
    // rectangle with 2-D MTE2 commands, then use the same two-level FP32
    // vector reduction as the long contiguous path.  A rectangle need not fit
    // in UB in one piece: rowTile describes one <=4096-element streaming
    // batch and the kernel folds successive batch sums in reduction order.
    // This removes per-element rank div/mod and scalar GM traffic for large
    // split reductions while retaining the generic path when even one padded
    // inner row cannot fit.
    if (splitReduce) {
        const uint64_t rawAlignment = 32 / dtypeBytes;
        const uint64_t innerAligned = AlignUp(tiling->inner, rawAlignment);
        const uint64_t rowsPerBatch = innerAligned == 0
            ? 0
            : kLongArBatchElements / innerAligned;
        const uint64_t batchRows = std::min<uint64_t>(tiling->outer, rowsPerBatch);
        const uint64_t vectorElements = AlignUp(
            batchRows * innerAligned, kLongArPartialWidth);
        const uint64_t rowBytes = tiling->inner * dtypeBytes;
        const bool sourceStrideFits = rowBytes <= static_cast<uint64_t>(UINT32_MAX) &&
            (tiling->outputElements <= 1 ||
             tiling->outputElements - 1 <= static_cast<uint64_t>(UINT32_MAX) / rowBytes);
        if (tiling->outer <= UINT16_MAX && rowsPerBatch != 0 &&
            vectorElements != 0 && vectorElements <= kLongArBatchElements &&
            sourceStrideFits) {
            tiling->reduceAligned = innerAligned;
            tiling->rowTile = vectorElements;
            const uint64_t inputBytes = tiling->totalElements * dtypeBytes;
            const uint64_t requestedCores = std::max<uint64_t>(
                1, CeilDiv(inputBytes, kFastArBytesPerCoreTarget));
            blockDim = static_cast<uint32_t>(std::min<uint64_t>({
                static_cast<uint64_t>(coreNum), tiling->outputElements, requestedCores}));
            // Each gathered output is independent, but its scalar result is
            // written through a 32-byte MTE block.  When address-generation
            // work dominates the byte-volume heuristic, expose one AIV per
            // complete output block and let the kernel partition those blocks
            // contiguously.  Adjacent AIVs therefore never update the same
            // short-write cache line, including the final partial block.
            const uint64_t outputBlockElements = 32 / dtypeBytes;
            const uint64_t outputBlocks =
                CeilDiv(tiling->outputElements, outputBlockElements);
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(static_cast<uint64_t>(coreNum), outputBlocks)));
            tiling->reserved0 = 3;
            tilingKey = dtypeBytes == 2 ? kTilingKeySplitReduceVector16
                                       : kTilingKeySplitReduceVector32;
        } else if (tiling->outer != 0 && tiling->inner > kLongArBatchElements) {
            // The same [reduceOuter, output, reduceInner] geometry is still
            // regular when one inner row no longer fits in UB.  Stream one
            // contiguous inner chunk at a time instead of falling back to
            // per-element rank div/mod and scalar GM loads.  A 4096-element
            // chunk is both a complete 256-byte vector-repeat multiple and
            // small enough for raw/FP32/partial buffers in the 192-KiB UB.
            tiling->reduceAligned = rawAlignment;
            tiling->rowTile = kLongArBatchElements;
            const uint64_t outputBlockElements = 32 / dtypeBytes;
            const uint64_t outputBlocks =
                CeilDiv(tiling->outputElements, outputBlockElements);
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(static_cast<uint64_t>(coreNum), outputBlocks)));
            tiling->reserved0 = 3;
            tiling->reserved1 = 1;
            tilingKey = dtypeBytes == 2 ? kTilingKeySplitReduceVector16
                                       : kTilingKeySplitReduceVector32;
        }
    }
    // Mixed reduction axes ending in a reduced suffix still admit strided
    // rectangles. The remaining reduced axes are decoded per DMA batch.
    if (genericReducedSuffix) {
        size_t suffixAxis = axes.size() - 1;
        while (suffixAxis > 0 && axes[suffixAxis - 1] + 1 == axes[suffixAxis]) { --suffixAxis; }
        const uint64_t suffix = Product(shape, axes[suffixAxis], shape.size());
        const uint32_t rowAxis = axes[suffixAxis - 1];
        const uint64_t rowPitch = tiling->inputStride[rowAxis] * dtypeBytes;
        if (rowPitch <= UINT32_MAX) {
            tiling->inner = suffix;
            tiling->outer = tiling->reduce / suffix;
            tiling->reduceAligned = AlignUp(suffix, 32 / dtypeBytes);
            if (tiling->reduceAligned <= kLongArBatchElements) {
                const uint64_t rows = std::max<uint64_t>(1, kLongArBatchElements / tiling->reduceAligned);
                tiling->rowTile = AlignUp(rows * tiling->reduceAligned, kLongArPartialWidth);
                tiling->reserved1 = 2;
            } else {
                tiling->rowTile = kLongArBatchElements;
                tiling->reserved1 = 3;
            }
            const uint64_t blocks = CeilDiv(tiling->outputElements, blockElements);
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreNum, blocks)));
            tiling->reserved0 = 3;
            tilingKey = dtypeBytes == 2 ? kTilingKeySplitReduceVector16 : kTilingKeySplitReduceVector32;
        }
    }
    // Consecutive middle reduction: gather all reduction rows for an inner
    // tile in one strided DMA, then square/add in UB.  This is especially
    // important for shapes such as [outer, 31, inner], where the scalar
    // fallback would otherwise issue 31 separate GM loads per output.
    if (middleReduce && tiling->inner != 0 && tiling->reduce <= UINT16_MAX) {
        const uint64_t alignment = std::max<uint64_t>(32 / dtypeBytes, 8);
        const uint64_t reduceAligned = AlignUp(tiling->reduce, alignment);
        const uint64_t bytesPerElement =
            reduceAligned * (dtypeBytes + sizeof(float)) + sizeof(float);
        uint64_t maxInnerTile = bytesPerElement == 0 ? 0 : usableBytes / bytesPerElement;
        maxInnerTile = std::min<uint64_t>(maxInnerTile, kMaxStridedVectorTileElements);
        maxInnerTile = maxInnerTile / alignment * alignment;
        // DataCopyPad stores every gathered reduction row on a 32-byte UB
        // boundary, and the vector loop addresses row r as r * rowTile.
        // Keep rowTile aligned even when the complete inner suffix fits in a
        // single tile (for example FP32 inner=513).  Using the raw inner size
        // here made row 1 start at a four-byte-aligned UB address and raised
        // vector exception 507035 on 910B-family devices.
        const uint64_t vectorTile = std::min<uint64_t>(
            AlignUp(tiling->inner, alignment),
            std::max<uint64_t>(alignment, maxInnerTile));
        if (vectorTile >= alignment && reduceAligned != 0) {
            tiling->reduceAligned = reduceAligned;
            tiling->rowTile = vectorTile;
            const uint64_t tasks = tiling->outer * CeilDiv(tiling->inner, vectorTile);
            // There is one independent task per outer row/tile. The gathered
            // task is large enough that all independent rows are useful even
            // when total input bytes are below the generic 64 KiB/core rule.
            blockDim = static_cast<uint32_t>(std::min<uint64_t>(
                static_cast<uint64_t>(coreNum), std::max<uint64_t>(1, tasks)));
            // Keep the middle implementation on the already registered
            // strided-vector entry.  CANN 8.5's binary-entry table is
            // sensitive to adding a second family of ad-hoc keys; the
            // algorithm choice itself is a one-time tiling-data flag and
            // therefore has no per-element cost.
            tiling->reserved0 = 1;
            // When one logical inner row is not a whole 32-byte output block,
            // assigning rows/tiles round-robin lets different AIVs issue
            // short MTE3 writes into the same block.  Partition the flattened
            // output in complete block units and split logical rows only
            // inside the owning AIV.
            if (tiling->inner % blockElements != 0) {
                const uint64_t outputBlocks =
                    CeilDiv(tiling->outputElements, blockElements);
                blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                    1, std::min<uint64_t>(coreNum, outputBlocks)));
                tiling->tileElements = blockElements;
                tiling->reserved1 = 4;
            }
            tilingKey = dtypeBytes == 2 ? kTilingKeyStridedVector16
                                       : kTilingKeyStridedVector32;
        }
    }
    // Arbitrary non-consecutive reduction with a contiguous preserved suffix.
    // Each task owns an output suffix tile, computes one base address for its
    // preserved prefix, then streams a contiguous vector for every reduction
    // coordinate. This turns scalar GM traffic into DMA + SIMD while retaining
    // the fully general rank/axis mapping at tile granularity.
    if (genericPreservedSuffix && tiling->inner != 0) {
        const uint64_t alignment = std::max<uint64_t>(32 / dtypeBytes, 8);
        const uint64_t bytesPerElement = dtypeBytes + 2 * sizeof(float);
        uint64_t maxVectorTile = usableBytes / bytesPerElement;
        maxVectorTile = std::min(maxVectorTile, kMaxStridedVectorTileElements);
        maxVectorTile = maxVectorTile / alignment * alignment;
        const uint64_t inputBytes = tiling->totalElements * dtypeBytes;
        const uint64_t requestedCores = std::min<uint64_t>({
            static_cast<uint64_t>(coreNum), tiling->outputElements,
            std::max<uint64_t>(1, CeilDiv(inputBytes, kFastArBytesPerCoreTarget))});
        const uint64_t groupsPerOuter =
            std::max<uint64_t>(1, CeilDiv(requestedCores, tiling->outer));
        uint64_t vectorTile =
            AlignUp(CeilDiv(tiling->inner, groupsPerOuter), alignment);
        vectorTile = std::min(vectorTile, maxVectorTile);
        if (vectorTile >= alignment) {
            tiling->rowTile = vectorTile;
            const uint64_t tasks =
                tiling->outer * CeilDiv(tiling->inner, vectorTile);
            blockDim = static_cast<uint32_t>(std::min<uint64_t>({
                static_cast<uint64_t>(coreNum), requestedCores, tasks}));
            tiling->reserved1 = 1;
            // A one-tile suffix is an independent output group.  The generic
            // 64 KiB/core heuristic leaves small, reduction-heavy tensors on
            // one AIV even though their address-generation loops dominate.
            // Split contiguous groups across AIVs instead, with every inter-
            // core boundary on a 32-byte GM block so short output DMAs cannot
            // race on a shared tail block.
            if ((tiling->inner <= vectorTile && tiling->reduce > 1) ||
                tiling->inner % blockElements != 0) {
                const uint64_t outputBlockElements = 32 / dtypeBytes;
                const uint64_t alignedBlocks =
                    CeilDiv(tiling->outputElements, outputBlockElements);
                const uint64_t alignedCores = std::min<uint64_t>(
                    static_cast<uint64_t>(coreNum), alignedBlocks);
                if (alignedCores > 1 || tiling->inner % blockElements != 0) {
                    blockDim = static_cast<uint32_t>(alignedCores);
                    // reserved1==3 reinterprets tileElements as the number of
                    // output values in one 32-byte ownership unit.  Unlike
                    // whole-group ownership, a unit may end inside a suffix;
                    // the kernel splits that unit at logical group boundaries.
                    tiling->tileElements = outputBlockElements;
                    tiling->reserved1 = 3;
                }
            }
            tilingKey = dtypeBytes == 2 ? kTilingKeyStridedVector16
                                       : kTilingKeyStridedVector32;
        }
    }
    // A very long reduced prefix followed by only a few preserved columns is
    // mathematically a small collection of strided dot products:
    //   y[j] = sum_r x[r, j]^2.
    // The ordinary strided path issues one tiny MTE command and one hard
    // dependency per r.  Instead, load a large contiguous [rows, inner]
    // slab, Gather each column in UB, and reduce 64 values per vector repeat.
    // The predicate is geometry-only and covers any rank collapsed to RA.
    bool prefixSmallInner = false;
    if (mode == Mode::kRA && tiling->inner != 0 &&
        tiling->inner <= kPrefixSmallInnerMax && tiling->reduce >= 64) {
        const uint64_t conversionBytes = dtypeBytes == sizeof(float)
            ? 0
            : sizeof(float);
        const uint64_t bytesPerRow = tiling->inner * dtypeBytes + dtypeBytes +
            conversionBytes + sizeof(uint32_t) + 1;
        uint64_t rows = bytesPerRow == 0 ? 0 : usableBytes / bytesPerRow;
        rows = std::min<uint64_t>(rows, kPrefixGatherMaxRows);
        rows = rows / kLongArPartialWidth * kLongArPartialWidth;
        rows = std::min<uint64_t>(rows, AlignUp(tiling->reduce, kLongArPartialWidth));
        if (rows >= kLongArPartialWidth) {
            tiling->rowTile = rows;
            tiling->reduceAligned = tiling->inner;
            const uint64_t outputBlocks =
                CeilDiv(tiling->outputElements, blockElements);
            blockDim = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(static_cast<uint64_t>(coreNum), outputBlocks)));
            tilingKey = dtypeBytes == 2 ? kTilingKeyPrefixSmallInner16
                                       : kTilingKeyPrefixSmallInner32;
            prefixSmallInner = true;
        }
    }

    // A consecutive reduced prefix/middle collapses to [outer, reduce,
    // inner].  Each task owns an inner interval, keeps its FP32 accumulator
    // in UB, and streams one contiguous slice per reduction step.  Choose the
    // tile structurally from UB and requested input bytes per core.
    if (!prefixSmallInner && !splitReduce && !middleReduce &&
        !genericPreservedSuffix && !genericReducedSuffix &&
        (mode == Mode::kRA || mode == Mode::kARA) && tiling->inner != 0) {
        const uint64_t alignment = std::max<uint64_t>(32 / dtypeBytes, 8);
        const uint64_t bytesPerElement = dtypeBytes + 2 * sizeof(float);
        uint64_t maxVectorTile = usableBytes / bytesPerElement;
        maxVectorTile = std::min(maxVectorTile, kMaxStridedVectorTileElements);
        maxVectorTile = maxVectorTile / alignment * alignment;
        const uint64_t inputBytes = tiling->totalElements * dtypeBytes;
        const uint64_t requestedCores = std::min<uint64_t>({
            static_cast<uint64_t>(coreNum), tiling->outputElements,
            std::max<uint64_t>(1, CeilDiv(inputBytes, kFastArBytesPerCoreTarget))});
        const uint64_t groupsPerOuter = std::max<uint64_t>(1, CeilDiv(requestedCores, tiling->outer));
        uint64_t vectorTile = AlignUp(CeilDiv(tiling->inner, groupsPerOuter), alignment);
        vectorTile = std::min(vectorTile, maxVectorTile);
        if (vectorTile >= alignment) {
            tiling->rowTile = vectorTile;
            const uint64_t tasks = tiling->outer * CeilDiv(tiling->inner, vectorTile);
            blockDim = static_cast<uint32_t>(std::min<uint64_t>({
                static_cast<uint64_t>(coreNum), requestedCores, tasks}));
            tilingKey = dtypeBytes == 2 ? kTilingKeyStridedVector16 : kTilingKeyStridedVector32;
        }
    }
    // Batch 64 reduction rows for each owned contiguous column interval.
    // The byte stride and padded UB pitch are bounded independently.
    if (consecutive && axes.back() < static_cast<int32_t>(rank - 1) &&
        tiling->inner > kPrefixSmallInnerMax && tiling->reduce >= 16 &&
        tiling->totalElements >= 4096 &&
        tiling->inner <= static_cast<uint64_t>(UINT32_MAX) / dtypeBytes) {
        const uint64_t units = CeilDiv(tiling->outputElements, blockElements);
        blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreNum, units)));
        const uint64_t owned = AlignUp(CeilDiv(tiling->outputElements, blockDim), blockElements);
        tiling->rowTile = std::min<uint64_t>(256, owned);
        tilingKey = dtypeBytes == 2 ? 162 : 164;
        // A small output cannot expose enough independent columns. Split
        // the long reduction as a second task dimension, keep FP32 partials,
        // and fold them after a hardware barrier without floating atomics.
        if (units < coreNum && tiling->reduce >= 2048 &&
            tiling->totalElements * dtypeBytes >= (1ULL << 20)) {
            const uint64_t parts = std::min<uint64_t>(coreNum / units, CeilDiv(tiling->reduce, 512));
            if (parts >= 2) {
                blockDim = static_cast<uint32_t>(units * parts);
                tiling->rowTile = blockElements;
                tiling->reserved0 = static_cast<uint32_t>(parts);
                tilingKey = dtypeBytes == 2 ? 182 : 184;
                userWorkspaceBytes = AlignUp(tiling->outputElements, 16) * parts * sizeof(float);
            }
        }
    }
    if (prefixSmallInner && tiling->totalElements * dtypeBytes >= (1ULL << 20)) {
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, CeilDiv(tiling->reduce, 512)));
        if (blockDim > 1) {
            tiling->rowTile = std::min<uint64_t>({tiling->rowTile, 2048,
                AlignUp(CeilDiv(tiling->reduce, blockDim), 64)});
            userWorkspaceBytes = blockDim * 16 * sizeof(float);
            tilingKey = dtypeBytes == 2 ? 202 : 204;
        }
    }
    // The scalar fallback writes output through MTE3.  Partitioning individual
    // elements lets two AIVs own different values in the same 32-byte GM
    // block; their short writes can then overwrite each other.  Preserve
    // parallelism by partitioning complete output blocks instead.  reserved0
    // is consumed only by the default KernelSquareSumV1 entry.
    const uint64_t defaultTilingKey = static_cast<uint64_t>(mode) * 10 + dtypeBytes;
    if (tilingKey == defaultTilingKey) {
        const uint64_t outputBlocks = CeilDiv(tiling->outputElements, blockElements);
        blockDim = static_cast<uint32_t>(std::max<uint64_t>(
            1, std::min<uint64_t>(coreNum, outputBlocks)));
        tiling->rowTile = blockElements;
        tiling->reserved0 = 4;
    }
    // Stream contiguous rows through two input buffers. Short-output tensors
    // expose their reduction tiles as independent, deterministic partial tasks.
    if ((mode == Mode::kReduceAll || mode == Mode::kAR) && tiling->reduce >= 16384) {
        const uint64_t units = CeilDiv(tiling->outputElements, blockElements);
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, units));
        userWorkspaceBytes = 0;
        tilingKey = dtypeBytes == 2 ? 212 : 214;
        if (tiling->outputElements < coreNum * blockElements && tiling->totalElements >= 65536) {
            const uint64_t wanted = std::min<uint64_t>(coreNum,
                std::max<uint64_t>(tiling->outputElements, CeilDiv(tiling->totalElements, 8192)));
            uint64_t parts = std::max<uint64_t>(1, std::min<uint64_t>(wanted / tiling->outputElements,
                                                    CeilDiv(tiling->reduce, 8192)));
            if (mode == Mode::kAR && parts > 1) {
                const uint64_t tiles = CeilDiv(tiling->reduce, 8192);
                const uint64_t criticalTiles = CeilDiv(tiles, parts);
                // Minimum participants attaining the same maximum tile count.
                // Extra participants on this plateau only increase rendezvous work.
                parts = CeilDiv(tiles, criticalTiles);
            }
            if (parts >= 1) {
                blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, tiling->outputElements * parts));
                tiling->reserved0 = static_cast<uint32_t>(parts);
                userWorkspaceBytes = tiling->outputElements * parts * 32;
                tilingKey = dtypeBytes == 2 ? 222 : 224;
            }
        }
    }
    if (genericPreservedSuffix && tiling->reduce >= 16 && tiling->totalElements >= 4096 &&
        tiling->inner <= static_cast<uint64_t>(UINT32_MAX) / dtypeBytes) {
        const uint64_t units = CeilDiv(tiling->outputElements, blockElements);
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, units));
        const uint64_t owned = AlignUp(CeilDiv(tiling->outputElements, blockDim), blockElements);
        tiling->rowTile = std::min<uint64_t>(256, owned);
        // The final consecutive reduced run is physically a regular 2-D rectangle.
        uint32_t firstReduced = static_cast<uint32_t>(axes.back());
        while (firstReduced > 0 && std::binary_search(axes.begin(), axes.end(), firstReduced - 1))
            --firstReduced;
        tiling->reduceAligned = Product(shape, firstReduced, axes.back() + 1);
        userWorkspaceBytes = 0;
        tilingKey = dtypeBytes == 2 ? 232 : 234;
        if (units < coreNum && tiling->reduce >= 2048 && tiling->totalElements >= 65536) {
            const uint64_t parts = std::min<uint64_t>(coreNum / units, CeilDiv(tiling->reduce, 512));
            if (parts > 1) {
                blockDim = static_cast<uint32_t>(units * parts);
                tiling->rowTile = blockElements;
                tiling->reserved0 = static_cast<uint32_t>(parts);
                userWorkspaceBytes = AlignUp(tiling->outputElements, 16) * parts * sizeof(float);
                tilingKey = dtypeBytes == 2 ? 262 : 264;
            }
        }
    }
    if ((splitReduce || genericReducedSuffix) && tiling->inner > 0 &&
        tiling->inner <= 512 && tiling->outputElements >= 2 && tiling->totalElements >= 4096) {
        uint32_t suffixBegin = static_cast<uint32_t>(axes.size()) - 1;
        while (suffixBegin > 0 && axes[suffixBegin - 1] + 1 == axes[suffixBegin]) --suffixBegin;
        const uint64_t rowStride = tiling->inputStride[axes[suffixBegin - 1]];
        if (rowStride <= static_cast<uint64_t>(UINT32_MAX) / dtypeBytes) {
        const uint64_t units = CeilDiv(tiling->outputElements, blockElements);
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, units));
        uint32_t endPreserved = static_cast<uint32_t>(nonReduce.back());
        uint32_t firstPreserved = endPreserved;
        while (firstPreserved > 0 && !std::binary_search(axes.begin(), axes.end(), firstPreserved - 1))
            --firstPreserved;
        tiling->reduceAligned = Product(shape, firstPreserved, endPreserved + 1);
        userWorkspaceBytes = 0;
        tilingKey = dtypeBytes == 2 ? 242 : 244;
        const uint64_t outerReduction = tiling->reduce / tiling->inner;
        if (units < coreNum && outerReduction >= 64 && tiling->totalElements >= 65536) {
            const uint64_t parts = std::min<uint64_t>(coreNum / units, CeilDiv(outerReduction, 32));
            if (parts > 1) {
                blockDim = static_cast<uint32_t>(units * parts);
                tiling->reserved0 = static_cast<uint32_t>(parts);
                userWorkspaceBytes = AlignUp(tiling->outputElements,16) * parts * sizeof(float);
                tilingKey = dtypeBytes == 2 ? 252 : 254;
            }
        }
        }
    }
    // A 65..511-element suffix is padded to eight 64-lane partials,
    // allowing many output rows to be reduced in two vector instructions.
    if (mode == Mode::kAR && tiling->reduce > 64 && tiling->reduce < 512) {
        const uint64_t units = CeilDiv(tiling->outputElements, blockElements);
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, units));
        userWorkspaceBytes = 0;
        tilingKey = dtypeBytes == 2 ? 282 : 284;
    }

    // Whole contiguous rectangles avoid narrow strided DMA commands. Group
    // adjacent outer rows so independent owners never share a 32-byte store.
    if (consecutive && (mode == Mode::kRA || mode == Mode::kARA) && tiling->inner >= 32 &&
        tiling->inner <= 2048 && tiling->reduce >= 256 && usableBytes >= 8192 * 16 &&
        tiling->totalElements * dtypeBytes >= (8ULL << 20)) {
        const uint64_t outerGroup = blockElements / Gcd(tiling->inner, blockElements);
        const uint64_t groups = CeilDiv(tiling->outer, outerGroup);
        const uint64_t owners = std::min<uint64_t>(coreNum, groups);
        const uint64_t parts = std::max<uint64_t>(1, std::min<uint64_t>(
            coreNum / owners, CeilDiv(tiling->reduce, 512)));
        const uint64_t columns = AlignUp(tiling->inner, blockElements);
        uint64_t rows = 1;
        while (rows * 2 * columns <= 8192 && rows < 512) rows *= 2;
        // Bound serial FP32 accumulation; larger reductions retain the existing path.
        const uint64_t serialTiles = CeilDiv(CeilDiv(tiling->reduce, parts), rows);
        const uint64_t serialLimit = dtypeBytes == 4 ? 1024 : 16384;
        if (serialTiles <= serialLimit) {
            tiling->rowTile = 8192;
            tiling->reduceAligned = outerGroup;
            tiling->reserved0 = static_cast<uint32_t>(parts);
            blockDim = static_cast<uint32_t>(owners * parts);
            userWorkspaceBytes = parts > 1 ? AlignUp(tiling->outputElements, 16) * parts * 4 : 0;
            tilingKey = dtypeBytes == 2 ? 502 : 504;
        }
    }


    // Pure RAR: choose the largest row batch fitting actual Unified Buffer space.
    if (splitReduce && rank == 3 && tiling->inner >= 8 && tiling->inner <= 512 &&
        tiling->outer >= 2 && tiling->outer <= 256 &&
        tiling->outputElements >= 128 && total * dtypeBytes >= (1ULL << 20)) {
        const bool gather = tiling->inner % 8 != 0;
        uint64_t columns = tiling->inner;
        if (gather) { columns = 1; while (columns < tiling->inner) columns *= 2; }
        const uint64_t chunks = CeilDiv(tiling->inner, 64);
        const uint64_t fixed = 1024 + (dtypeBytes == 2 ? 512 : 0);
        const uint64_t perRow = columns * (2 * dtypeBytes + 4 + ((dtypeBytes == 2 || gather) ? 4 : 0) + (gather ? 4 : 0)) + chunks * 32;
        const uint64_t rows = usableBytes > fixed ? std::min<uint64_t>(240, (usableBytes - fixed) / perRow) / blockElements * blockElements : 0;
        if (rows > 0) {
            tiling->rowTile = rows * columns;
            tiling->reserved1 = 8;
            tiling->reserved0 = 0;
            userWorkspaceBytes = 0;
            blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, CeilDiv(tiling->outputElements, blockElements)));
            tilingKey = dtypeBytes == 2 ? 522 : 524;
        }
    }

    // Compute independent ARA outer rows before packing their ragged GM outputs.
    if (middleReduce && tiling->inner >= 32 && tiling->inner <= 2048 &&
        tiling->inner % blockElements != 0 && tiling->outer >= 4 &&
        tiling->reduce >= 256 && total * dtypeBytes >= (4ULL << 20) &&
        usableBytes >= 8192 * 16) {
        const uint64_t columns = AlignUp(tiling->inner, blockElements);
        uint64_t rows = 1;
        while (rows * 2 * columns <= 8192 && rows < 512) rows *= 2;
        if (CeilDiv(tiling->reduce, rows) <= (dtypeBytes == 4 ? 1024 : 16384)) {
            tiling->rowTile = 8192;
            tiling->reserved0 = 1;
            tiling->reduceAligned = blockElements / Gcd(tiling->inner, blockElements);
            blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, tiling->outer));
            userWorkspaceBytes = tiling->outer * columns * sizeof(float);
            tilingKey = dtypeBytes == 2 ? 542 : 544;
        }
    }

    // RAR with an unaligned short suffix: keep packed input/accumulators and
    // reuse a 64-column Gather panel. Selection depends on layout and device UB.
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16 &&
        splitReduce && rank == 3 && tiling->inner >= 65 && tiling->inner <= 127 &&
        tiling->inner % 8 != 0 && tiling->outer >= 8 && tiling->outer <= 256 &&
        tiling->outputElements >= static_cast<uint64_t>(coreNum) * 128 &&
        total * dtypeBytes >= (1ULL << 20)) {
        const uint64_t chunks = CeilDiv(tiling->inner, 64);
        const uint64_t extra = AlignUp(chunks * 64 - tiling->inner, 8) * sizeof(float);
        const uint64_t fixed = 1024 + 512 + extra;
        const uint64_t perRow = tiling->inner * (2 * dtypeBytes + 8) + 64 * 4 + chunks * 32;
        const uint64_t rows = usableBytes > fixed ? std::min<uint64_t>(240, (usableBytes - fixed) / perRow) / blockElements * blockElements : 0;
        if (rows != 0) {
            tiling->rowTile = rows * tiling->inner;
            tiling->reserved0 = 0; tiling->reserved1 = 8;
            userWorkspaceBytes = 0;
            blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, CeilDiv(tiling->outputElements, blockElements)));
            tilingKey = 562;
        }
    }



    // Algorithm selection follows normalized layout, vector limits and queried resources.
    // Streaming inputs are consumed once; cache hints are independent of device identity.
    const auto dtype=context->GetInputDesc(0)->GetDataType();
    const uint64_t deviceCores=std::max<uint64_t>(1,coreNum),budget=ubBytes>512?ubBytes-512:0;
    const uint64_t inputBytes=total*dtypeBytes;
    const uint32_t streamingHint=inputBytes>=std::max<uint64_t>(4ULL<<20,deviceCores*(128ULL<<10))?256U:0U;
    if(dtype==ge::DT_FLOAT16 && mode==Mode::kAR && tiling->reduce<=64){
        const uint64_t aligned=AlignUp(tiling->reduce,16);
        const uint64_t need=64*aligned*6+64*4;
        if(need<=budget){tiling->reduceAligned=aligned;tiling->rowTile=64;tilingKey=800+aligned/16-1;blockDim=std::min<uint64_t>(deviceCores,std::max<uint64_t>(1,CeilDiv(tiling->outputElements,32)));tiling->reserved1=0;userWorkspaceBytes=0;}
    }
    if(dtype==ge::DT_BF16 && (middleReduce || mode==Mode::kRA) && tiling->inner>=8 && tiling->reduce>=32 && inputBytes>=(128ULL<<10)){
        uint64_t rr=std::min<uint64_t>(128,FloorPowerOfTwo((32ULL<<10)/(tiling->inner*2)));
        while(rr>=16){
            const uint64_t ar=std::max<uint64_t>(8,rr/2),col=AlignUp(tiling->inner,16);
            const uint64_t acc=std::max<uint64_t>(AlignUp(ar*tiling->inner+16,8),16*col),conv=std::max<uint64_t>(AlignUp(rr*tiling->inner,8),16*col);
            const uint64_t need=2*AlignUp(rr*tiling->inner*2,32)+4*acc+4*conv+16*col*4+2*col*4;
            uint64_t parts=std::max<uint64_t>(1,(CeilDiv(tiling->reduce,rr)+4)/8);
            parts=std::min<uint64_t>(deviceCores,parts);
            if(need<=budget && CeilDiv(CeilDiv(tiling->reduce,parts),ar)<=1024){blockDim=std::min<uint64_t>(deviceCores,tiling->outer*parts);tilingKey=820;tiling->rowTile=ar;tiling->tileElements=rr;tiling->reserved0=parts;tiling->reserved1=streamingHint;tiling->reduceAligned=16/Gcd(tiling->inner,16);userWorkspaceBytes=tiling->outer*parts*col*4;break;}
            rr/=2;
        }
    }
    // Full output ownership removes GM partials when sufficient balanced work
    // remains per core. Choose cores from the output-task count and device limit.
    if(tilingKey==820 && tiling->outer>=2*deviceCores && CeilDiv(tiling->reduce,tiling->rowTile)<=1024){
        const uint64_t rounds=CeilDiv(tiling->outer,deviceCores);
        const uint64_t owners=CeilDiv(tiling->outer,rounds);
        if(owners*5>=deviceCores*4){
            blockDim=owners;tilingKey=821;tiling->reserved0=1;userWorkspaceBytes=0;
            if(tiling->rowTile>=16){
                const uint64_t ar=tiling->rowTile,col=AlignUp(tiling->inner,16);
                const uint64_t acc=AlignUp(std::max<uint64_t>(ar*tiling->inner,7*tiling->inner+col),8);
                const uint64_t conv=std::max<uint64_t>(AlignUp(ar*tiling->inner,8),16*col);
                for(uint64_t rr=tiling->tileElements+ar;rr<=128;rr+=ar){
                    const uint64_t need=2*AlignUp(rr*tiling->inner*2,32)+4*acc+4*conv;
                    if(need>budget)break;
                    tiling->tileElements=rr;tilingKey=823;
                }
            }
        }
    }
    if(dtype==ge::DT_BF16 && mode==Mode::kAR && tiling->reduce>64){
        const uint64_t base=tiling->reduce-1;
        if((base&(base-1))==0 && base>=64 && base/8<=255){
            const uint64_t rows=base>=256?16384/base:64,cols=rows,n=rows*(base+1),matrix=(rows+1)*cols;
            const uint64_t need=2*AlignUp(n,16)+4*AlignUp(rows,16)+4*AlignUp(n+cols+8,8)+12*matrix+4*rows*64+8*AlignUp(rows,8)+4*AlignUp(rows+1,8);
            if(need<=budget){uint32_t log=0;for(uint64_t v=base;v>1;v/=2)++log;tilingKey=810+log-6;tiling->rowTile=rows;tiling->reduceAligned=base;tiling->reserved1=streamingHint;blockDim=std::min<uint64_t>(FloorPowerOfTwo(deviceCores),std::max<uint64_t>(1,CeilDiv(tiling->outputElements,16)));userWorkspaceBytes=0;}
        }
    }
    if(dtype==ge::DT_FLOAT16 && splitReduce && rank==3 && tiling->inner>=64 && tiling->inner+16-Gcd(tiling->inner,16)<=128 && CeilDiv(tiling->outer,8)<=1024){
        uint64_t rows=std::min<uint64_t>(16*255,budget/(6*tiling->inner+14))/16*16;
        while(rows && RarBufferBytes(rows,tiling->inner)>budget)rows-=16;
        if(rows){blockDim=std::min<uint64_t>(deviceCores,std::max<uint64_t>(1,CeilDiv(tiling->outputElements,16)));tilingKey=830;tiling->rowTile=rows*tiling->inner;tiling->reserved0=0;tiling->reserved1=streamingHint;userWorkspaceBytes=0;}
    }
    if(dtype==ge::DT_FLOAT && mode==Mode::kRA && inputBytes>=(4ULL<<20) && tiling->totalElements<=UINT32_MAX-8){
        const uint64_t ar=8,col=AlignUp(tiling->inner,8),fixed=4*AlignUp(ar*tiling->inner+16,8)+4*col;
        for(uint32_t buffers: {4U,2U,1U}){
            const uint64_t tileBytes=budget>fixed?((budget-fixed)/buffers)/(8*1024)*(8*1024):0;
            if(!tileBytes || tileBytes<16*col*4)continue;
            const uint64_t parts=FloorPowerOfTwo(std::min<uint64_t>({deviceCores,CeilDiv(inputBytes,tileBytes),tileBytes/(64*4)}));
            if(parts && CeilDiv(CeilDiv(tiling->reduce,parts),ar)<=1280){blockDim=parts;tilingKey=840+buffers;tiling->rowTile=ar;tiling->tileElements=tileBytes/4;tiling->reserved0=parts;tiling->reserved1=streamingHint|3;tiling->reduceAligned=8/Gcd(tiling->inner,8);userWorkspaceBytes=parts*col*4;break;}
        }
    }

    // Dedicated short-row entries preserve the existing non-target device functions.
    if(tilingKey==800 || tilingKey==801)tilingKey+=180;

    if (userWorkspaceBytes != 0) {
        context->SetScheduleMode(1);
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        // GetUserWorkspace() skips the Ascend C system workspace.  The
        // framework allocates exactly the size reported here, so requesting
        // only the partial buffer makes the returned user pointer land past
        // the allocation (observed as an MTE DDR out-of-range fault on the
        // reference 910B-family device).
        // Hardware SyncAll also relies on this reserved system area.
        workspace[0] = userWorkspaceBytes == 0
            ? 0
            : userWorkspaceBytes + platform.GetLibApiWorkSpaceSize();
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* axisValue = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const auto* keepDims = attrs->GetAttrPointer<bool>(1);
    const auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisValue);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepDims);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);

    const uint32_t rank = static_cast<uint32_t>(input->GetDimNum());
    OP_CHECK_IF(axisValue->GetSize() == 0 || axisValue->GetSize() > rank,
        OP_LOGE(context->GetNodeName(), "axis must be non-empty"), return ge::GRAPH_FAILED);
    std::array<bool, kMaxRank> reduced{};
    const int64_t* axes = static_cast<const int64_t*>(axisValue->GetData());
    for (size_t i = 0; i < axisValue->GetSize(); ++i) {
        int64_t axis = axes[i] < 0 ? axes[i] + rank : axes[i];
        OP_CHECK_IF(axis < 0 || axis >= rank || reduced[axis],
            OP_LOGE(context->GetNodeName(), "invalid or duplicate axis"), return ge::GRAPH_FAILED);
        reduced[axis] = true;
    }
    output->SetDimNum(0);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (*keepDims) {
            output->AppendDim(reduced[axis] ? 1 : input->GetDim(axis));
        } else if (!reduced[axis]) {
            output->AppendDim(input->GetDim(axis));
        }
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace s9squaresum

IMPL_OP_INFERSHAPE(SquareSumV1).InferShape(s9squaresum::InferShape);

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> types = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
        const std::vector<ge::Format> formats(types.size(), ge::FORMAT_ND);
        this->Input("x").ParamType(REQUIRED).DataType(types).Format(formats);
        this->Output("y").ParamType(REQUIRED).DataType(types).Format(formats);
        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
        this->AICore().SetTiling(s9squaresum::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SquareSumV1);
}  // namespace ops
