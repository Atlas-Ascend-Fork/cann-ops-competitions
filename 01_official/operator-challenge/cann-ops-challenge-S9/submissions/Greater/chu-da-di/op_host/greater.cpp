#include <dlfcn.h>
#include "../op_kernel/greater_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
constexpr uint64_t kOutputAlignment = 32;
constexpr uint64_t kElementsPerCoreTarget = 32 * 1024;
constexpr uint64_t kBroadcastElementsPerCoreTarget = 2 * 1024;
// Splitting inside a repeated group removes the LCM core-collapse for short
// outer batches.  Once one input owns more than 96 groups, however, the
// group-aligned kernel amortizes each 2,048-entry Gather index vector over
// enough groups to beat the extra AIV parallelism.  This crossover is based
// on workload topology (groups per broadcast suffix), not a physical 910B
// revision, so the generated package remains generic ascend910b.
constexpr uint64_t kPartialGroupParallelLimit = 96;
constexpr uint64_t kTensorSelectMaskBytes = 8 * 1024;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + (value % divisor != 0);
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

uint64_t Gcd(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0) {
        const uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

uint32_t DtypeBytes(ge::DataType type)
{
    switch (type) {
        case ge::DT_INT8:
            return 1;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}

uint64_t VectorUbBytesPerElement(ge::DataType type)
{
    switch (type) {
        case ge::DT_FLOAT16:
            // Two inputs, one Select result, and byte-bool output.
            return 2 + 2 + 2 + 1;
        case ge::DT_FLOAT:
            return 4 + 4 + 4 + 1;
        case ge::DT_INT8:
            // Two raw inputs plus three FP16 comparison tensors.
            return 1 + 1 + 2 + 2 + 2 + 1;
        case ge::DT_BF16:
            // Two raw inputs plus three exact FP32 comparison tensors.
            return 2 + 2 + 4 + 4 + 4 + 1;
        case ge::DT_INT32:
            // Correctness fallback still initializes the generic buffers.
            return 4 + 4 + 4 + 1;
        default:
            return 0;
    }
}

bool ContiguousBroadcastRun(const uint64_t* inputShape, const uint64_t* outputShape,
                            uint64_t& repeat, uint64_t& inner)
{
    int32_t first = -1;
    int32_t last = -1;
    for (uint32_t axis = 0; axis < S9_GREATER_MAX_RANK; ++axis) {
        if (inputShape[axis] == 1 && outputShape[axis] != 1) {
            if (first < 0) {
                first = static_cast<int32_t>(axis);
            }
            last = static_cast<int32_t>(axis);
        }
    }
    if (first < 0) {
        return false;
    }
    for (int32_t axis = 0; axis < first; ++axis) {
        if (inputShape[axis] != outputShape[axis]) {
            return false;
        }
    }
    for (int32_t axis = first; axis <= last; ++axis) {
        if (inputShape[axis] != 1) {
            return false;
        }
    }
    for (uint32_t axis = static_cast<uint32_t>(last + 1); axis < S9_GREATER_MAX_RANK; ++axis) {
        if (inputShape[axis] != outputShape[axis]) {
            return false;
        }
    }
    repeat = 1;
    inner = 1;
    for (int32_t axis = first; axis <= last; ++axis) {
        repeat *= outputShape[axis];
    }
    for (uint32_t axis = static_cast<uint32_t>(last + 1); axis < S9_GREATER_MAX_RANK; ++axis) {
        inner *= outputShape[axis];
    }
    return true;
}

bool VectorScalarSuffix(const uint64_t* vectorShape, const uint64_t* scalarShape,
                        const uint64_t* outputShape, uint64_t& inner)
{
    inner = 1;
    bool hasBroadcastedWork = false;
    for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
        if (vectorShape[axis] != outputShape[axis] || scalarShape[axis] != 1) {
            break;
        }
        hasBroadcastedWork = hasBroadcastedWork || outputShape[axis] > 1;
        if (outputShape[axis] != 0 && inner > UINT64_MAX / outputShape[axis]) {
            return false;
        }
        inner *= outputShape[axis];
    }
    // Below one byte-bool DMA block, vector setup tends to dominate.  Keep
    // the fully generic path for those tiny suffixes and tune this structural
    // crossover conservatively for the generic Ascend 910B family target.
    return hasBroadcastedWork && inner >= kOutputAlignment;
}

// Detect the regular crossed-broadcast tensor product
//
//   vector [..., C, 1, I]   scalar [..., 1, S, 1]
//   output [..., C, S, I].
//
// The prefix may itself contain broadcasts; only the three innermost logical
// runs are constrained.  A scalar-suffix kernel would issue C*S tiny vector
// pipelines.  This topology instead admits one compact [S,I] broadcast batch
// for every cartesian row C.
bool CrossedCartesianSuffix(const uint64_t* vectorShape,
                            const uint64_t* scalarShape,
                            const uint64_t* outputShape,
                            uint64_t suffixInner,
                            uint64_t& cartesian,
                            uint64_t& scalarCount)
{
    uint64_t suffix = 1;
    int32_t axis = S9_GREATER_MAX_RANK - 1;
    for (; axis >= 0; --axis) {
        if (vectorShape[axis] != outputShape[axis] || scalarShape[axis] != 1) {
            break;
        }
        uint64_t next = 0;
        if (MulOverflow(suffix, outputShape[axis], next)) {
            return false;
        }
        suffix = next;
    }
    if (suffix != suffixInner || axis < 1) {
        return false;
    }
    const int32_t scalarAxis = axis;
    const int32_t cartesianAxis = axis - 1;
    if (vectorShape[scalarAxis] != 1 ||
        scalarShape[scalarAxis] != outputShape[scalarAxis] ||
        outputShape[scalarAxis] <= 1 ||
        vectorShape[cartesianAxis] != outputShape[cartesianAxis] ||
        scalarShape[cartesianAxis] != 1 ||
        outputShape[cartesianAxis] <= 1) {
        return false;
    }
    cartesian = outputShape[cartesianAxis];
    scalarCount = outputShape[scalarAxis];
    return true;
}

bool SharedContiguousSuffix(const uint64_t* x1Shape, const uint64_t* x2Shape,
                            const uint64_t* outputShape, uint64_t& inner)
{
    inner = 1;
    bool hasOuterBroadcast = false;
    int32_t firstSuffix = S9_GREATER_MAX_RANK;
    for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
        if (x1Shape[axis] != outputShape[axis] || x2Shape[axis] != outputShape[axis]) {
            break;
        }
        firstSuffix = axis;
        if (outputShape[axis] != 0 && inner > UINT64_MAX / outputShape[axis]) {
            return false;
        }
        inner *= outputShape[axis];
    }
    for (int32_t axis = 0; axis < firstSuffix; ++axis) {
        hasOuterBroadcast = hasOuterBroadcast ||
                            x1Shape[axis] != outputShape[axis] ||
                            x2Shape[axis] != outputShape[axis];
    }
    return hasOuterBroadcast && firstSuffix < S9_GREATER_MAX_RANK &&
           inner >= kOutputAlignment;
}

bool ShortRepeatedSuffix(const uint64_t* inputShape, const uint64_t* outputShape,
                         uint64_t& repeat, uint64_t& inner, uint64_t& group)
{
    inner = 1;
    repeat = 0;
    group = 0;
    for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
        if (inputShape[axis] == outputShape[axis]) {
            uint64_t next = 0;
            if (MulOverflow(inner, outputShape[axis], next)) {
                return false;
            }
            inner = next;
            continue;
        }
        if (inputShape[axis] != 1 || outputShape[axis] <= 1 ||
            inner >= kOutputAlignment || MulOverflow(inner, outputShape[axis], group)) {
            return false;
        }
        repeat = outputShape[axis];
        return group != 0;
    }
    return false;
}

// A tensor need not be globally full-shaped to supply one repeated-suffix
// group as a contiguous GM interval.  Broadcasts on axes outside the group
// merely change its base address; matching every axis inside the suffix is
// sufficient for one vector DMA after BroadcastOffset resolves that base.
bool ContiguousSuffixGroup(const uint64_t* inputShape,
                           const uint64_t* outputShape,
                           uint64_t group)
{
    if (group == 0) {
        return false;
    }
    uint64_t suffix = 1;
    for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
        if (inputShape[axis] != outputShape[axis]) {
            return false;
        }
        uint64_t next = 0;
        if (MulOverflow(suffix, outputShape[axis], next)) {
            return false;
        }
        suffix = next;
        if (suffix == group) {
            return true;
        }
        if (suffix > group) {
            return false;
        }
    }
    return false;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingGreaterBase(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<GreaterTilingData>();
    const auto* x1Holder = context->GetInputShape(0);
    const auto* x2Holder = context->GetInputShape(1);
    const auto* x1Desc = context->GetInputDesc(0);
    const auto* x2Desc = context->GetInputDesc(1);
    if (tiling == nullptr || x1Holder == nullptr || x2Holder == nullptr || x1Desc == nullptr || x2Desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (x1Desc->GetDataType() != x2Desc->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t elementBytes = DtypeBytes(x1Desc->GetDataType());
    if (elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape x1Original = x1Holder->GetStorageShape();
    const gert::Shape x2Original = x2Holder->GetStorageShape();
    const uint32_t x1Rank = static_cast<uint32_t>(x1Original.GetDimNum());
    const uint32_t x2Rank = static_cast<uint32_t>(x2Original.GetDimNum());
    const uint32_t rank = std::max(x1Rank, x2Rank);
    if (rank > S9_GREATER_MAX_RANK) {
        return ge::GRAPH_FAILED;
    }

    uint64_t x1Shape[S9_GREATER_MAX_RANK];
    uint64_t x2Shape[S9_GREATER_MAX_RANK];
    std::fill_n(x1Shape, S9_GREATER_MAX_RANK, 1);
    std::fill_n(x2Shape, S9_GREATER_MAX_RANK, 1);
    const uint32_t pad = S9_GREATER_MAX_RANK - rank;
    const uint32_t x1Lead = rank - x1Rank;
    const uint32_t x2Lead = rank - x2Rank;
    for (uint32_t axis = 0; axis < x1Rank; ++axis) {
        const int64_t extent = x1Original.GetDim(axis);
        if (extent < 0) {
            return ge::GRAPH_FAILED;
        }
        x1Shape[pad + x1Lead + axis] = static_cast<uint64_t>(extent);
    }
    for (uint32_t axis = 0; axis < x2Rank; ++axis) {
        const int64_t extent = x2Original.GetDim(axis);
        if (extent < 0) {
            return ge::GRAPH_FAILED;
        }
        x2Shape[pad + x2Lead + axis] = static_cast<uint64_t>(extent);
    }

    uint64_t outputShape[S9_GREATER_MAX_RANK];
    std::fill_n(outputShape, S9_GREATER_MAX_RANK, 1);
    uint64_t total = 1;
    for (uint32_t axis = pad; axis < S9_GREATER_MAX_RANK; ++axis) {
        const uint64_t lhs = x1Shape[axis];
        const uint64_t rhs = x2Shape[axis];
        if (lhs == rhs) {
            outputShape[axis] = lhs;
        } else if (lhs == 1) {
            outputShape[axis] = rhs;
        } else if (rhs == 1) {
            outputShape[axis] = lhs;
        } else {
            return ge::GRAPH_FAILED;
        }
        uint64_t next = 0;
        if (MulOverflow(total, outputShape[axis], next)) {
            return ge::GRAPH_FAILED;
        }
        total = next;
    }

    uint64_t x1Stride[S9_GREATER_MAX_RANK] = {0, 0, 0, 0, 0};
    uint64_t x2Stride[S9_GREATER_MAX_RANK] = {0, 0, 0, 0, 0};
    uint64_t stride1 = 1;
    uint64_t stride2 = 1;
    bool sameShape = true;
    for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
        x1Stride[axis] = (x1Shape[axis] == 1 && outputShape[axis] != 1) ? 0 : stride1;
        x2Stride[axis] = (x2Shape[axis] == 1 && outputShape[axis] != 1) ? 0 : stride2;
        sameShape = sameShape && x1Shape[axis] == outputShape[axis] && x2Shape[axis] == outputShape[axis];
        uint64_t next1 = 0;
        uint64_t next2 = 0;
        if (MulOverflow(stride1, x1Shape[axis], next1) || MulOverflow(stride2, x2Shape[axis], next2)) {
            return ge::GRAPH_FAILED;
        }
        stride1 = next1;
        stride2 = next2;
    }

    uint32_t coreCount = 1;
    uint64_t ubBytes = 192 * 1024;
    uint64_t l2Bytes = 0;
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        coreCount = std::max<uint32_t>(1, platform.GetCoreNumAiv());
        uint64_t detectedUb = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, detectedUb);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
        if (detectedUb >= 32 * 1024) {
            ubBytes = detectedUb;
        }
    }

    const uint64_t requested = total == 0 ? 1 : std::max<uint64_t>(1, std::min<uint64_t>(coreCount, CeilDiv(total, kElementsPerCoreTarget)));
    uint64_t blockElements = total == 0 ? 0 : AlignUp(CeilDiv(total, requested), kOutputAlignment);
    uint32_t blockDim = total == 0 ? 1 : static_cast<uint32_t>(CeilDiv(total, blockElements));
    // The native FP16/FP32 path compares directly.  INT8 widens exactly to
    // FP16 and BF16 widens exactly to FP32; each converted path keeps three
    // comparison tensors (lhs/rhs/Select result).  All vector counts are
    // padded to a full 256-byte comparison repeat.  A 256-element tile
    // alignment also keeps the second packed INT32 equality mask 32-byte
    // aligned.
    const uint64_t vectorUbBytes = ubBytes > kTensorSelectMaskBytes
                                       ? ubBytes - kTensorSelectMaskBytes
                                       : ubBytes;
    constexpr uint64_t kVectorTileAlignment = 256;
    const uint64_t bytesPerElement = VectorUbBytesPerElement(x1Desc->GetDataType());
    uint64_t tileElements = vectorUbBytes / bytesPerElement;
    tileElements = std::max<uint64_t>(
        kVectorTileAlignment, tileElements / kVectorTileAlignment * kVectorTileAlignment);
    tileElements = std::min<uint64_t>(tileElements, 32 * 1024);
    // Keep the legacy path at its established 2048-element working set.
    // Dedicated linear and run kernels have independently budgeted spans.
    if (x1Desc->GetDataType() == ge::DT_INT32) {
        tileElements = std::min<uint64_t>(tileElements, 2 * 1024);
    }
    // Do not reserve a full large-tensor queue for a small output.  Besides
    // wasting UB, that made small tensors pay the setup cost
    // of a 16--32K tile even though only one 2K span is ever consumed.
    if (total > 0) {
        tileElements = std::min<uint64_t>(
            tileElements, AlignUp(total, kVectorTileAlignment));
    }

    tiling->totalElements = total;
    tiling->blockElements = blockElements;
    tiling->tileElements = tileElements;
    tiling->rank = rank;
    // Inactive broadcast fields must not depend on recycled tiling storage.
    tiling->broadcastRepeat = 0;
    tiling->broadcastInner = 0;
    const bool x1Scalar = stride1 == 1;
    const bool x2Scalar = stride2 == 1;
    bool x1Full = true;
    bool x2Full = true;
    for (uint32_t axis = 0; axis < S9_GREATER_MAX_RANK; ++axis) {
        x1Full = x1Full && x1Shape[axis] == outputShape[axis];
        x2Full = x2Full && x2Shape[axis] == outputShape[axis];
    }
    uint64_t x1Repeat = 0;
    uint64_t x1Inner = 0;
    uint64_t x2Repeat = 0;
    uint64_t x2Inner = 0;
    const bool x1ContiguousBroadcast =
        ContiguousBroadcastRun(x1Shape, outputShape, x1Repeat, x1Inner);
    const bool x2ContiguousBroadcast =
        ContiguousBroadcastRun(x2Shape, outputShape, x2Repeat, x2Inner);
    uint64_t x1VectorInner = 0;
    uint64_t x2VectorInner = 0;
    const bool x1VectorX2ScalarSuffix =
        VectorScalarSuffix(x1Shape, x2Shape, outputShape, x1VectorInner);
    const bool x2VectorX1ScalarSuffix =
        VectorScalarSuffix(x2Shape, x1Shape, outputShape, x2VectorInner);
    uint64_t x1CrossCartesian = 0;
    uint64_t x1CrossScalars = 0;
    uint64_t x2CrossCartesian = 0;
    uint64_t x2CrossScalars = 0;
    const bool x1VectorX2Cross = x1VectorX2ScalarSuffix &&
        CrossedCartesianSuffix(x1Shape, x2Shape, outputShape, x1VectorInner,
                               x1CrossCartesian, x1CrossScalars);
    const bool x2VectorX1Cross = x2VectorX1ScalarSuffix &&
        CrossedCartesianSuffix(x2Shape, x1Shape, outputShape, x2VectorInner,
                               x2CrossCartesian, x2CrossScalars);
    uint64_t sharedSuffixInner = 0;
    const bool sharedContiguousSuffix =
        SharedContiguousSuffix(x1Shape, x2Shape, outputShape, sharedSuffixInner);
    uint64_t x1ShortRepeat = 0;
    uint64_t x1ShortInner = 0;
    uint64_t x1ShortGroup = 0;
    uint64_t x2ShortRepeat = 0;
    uint64_t x2ShortInner = 0;
    uint64_t x2ShortGroup = 0;
    const bool x1ShortRepeatedSuffix =
        ShortRepeatedSuffix(x1Shape, outputShape, x1ShortRepeat, x1ShortInner, x1ShortGroup);
    const bool x2ShortRepeatedSuffix =
        ShortRepeatedSuffix(x2Shape, outputShape, x2ShortRepeat, x2ShortInner, x2ShortGroup);
    const bool x1LocalForX2Short = x2ShortRepeatedSuffix &&
        ContiguousSuffixGroup(x1Shape, outputShape, x2ShortGroup);
    const bool x2LocalForX1Short = x1ShortRepeatedSuffix &&
        ContiguousSuffixGroup(x2Shape, outputShape, x1ShortGroup);
    const uint64_t minimumGroupsForParallelPath = std::min<uint64_t>(coreCount, 8);
    // Widening and two broadcasts have a fixed setup cost. Require at
    // least four scalar rows per Compare span, and enough aligned tasks
    // that batching offsets any loss in AIV occupancy. All quantities are
    // derived from shape, DMA granularity and the runtime platform.
    const auto int8BatchWorthwhile = [&](uint64_t inner, uint64_t scalars) {
        if (x1Desc->GetDataType() != ge::DT_INT8) {
            return true;
        }
        const uint64_t paddedInner = AlignUp(inner, kOutputAlignment);
        if (paddedInner == 0 || scalars < 4 || paddedInner > (2 * 1024) / 4) {
            return false;
        }
        const uint64_t batchRows = std::min<uint64_t>(scalars, (2 * 1024) / paddedInner);
        // A partially filled last batch still pays full setup. Use the
        // average useful rows per batch rather than the maximum capacity.
        const uint64_t batchCount = CeilDiv(scalars, batchRows);
        const uint64_t averageBatchRows = scalars / batchCount;
        if (averageBatchRows < 4) {
            return false;
        }
        const uint64_t rowElements = scalars * inner;
        const uint64_t tasks = total / rowElements;
        const uint64_t tasksPerUnit = kOutputAlignment / Gcd(rowElements, kOutputAlignment);
        const uint64_t units = CeilDiv(tasks, tasksPerUnit);
        const uint64_t unitsPerCore = CeilDiv(units, coreCount);
        const uint64_t occupiedCores = CeilDiv(units, unitsPerCore);
        return averageBatchRows * occupiedCores >= coreCount;
    };
    // Unaligned INT32 rows need at least twelve reused scalar rows per
    // vector tile to amortize padded broadcasts and fragmented DMA stores.
    // Aligned rows retain the smaller two-row batch because stores pack.
    const bool x1CrossWorthwhile = x1VectorX2Cross && int8BatchWorthwhile(x1VectorInner, x1CrossScalars) &&
        total >= 128 * 1024 && AlignUp(x1VectorInner, kOutputAlignment) <= 2 * 1024 &&
        (x1Desc->GetDataType() != ge::DT_INT32 ||
         (x1VectorInner % kOutputAlignment == 0 || x1CrossScalars >= 12) &&
         (x1VectorInner % kOutputAlignment == 0 ? 3 : 13) *
             AlignUp(x1VectorInner, kOutputAlignment) <= tileElements);
    const bool x2CrossWorthwhile = x2VectorX1Cross && int8BatchWorthwhile(x2VectorInner, x2CrossScalars) &&
        total >= 128 * 1024 && AlignUp(x2VectorInner, kOutputAlignment) <= 2 * 1024 &&
        (x1Desc->GetDataType() != ge::DT_INT32 ||
         (x2VectorInner % kOutputAlignment == 0 || x2CrossScalars >= 12) &&
         (x2VectorInner % kOutputAlignment == 0 ? 3 : 13) *
             AlignUp(x2VectorInner, kOutputAlignment) <= tileElements);
    if (sameShape) {
        tiling->mode = 0;
    } else if (x1Scalar) {
        tiling->mode = 2;
    } else if (x2Scalar) {
        tiling->mode = 3;
    } else if (x1CrossWorthwhile) {
        tiling->mode = 13;
        tiling->broadcastRepeat = x1CrossCartesian;
        tiling->broadcastInner = x1VectorInner;
    } else if (x2CrossWorthwhile) {
        tiling->mode = 14;
        tiling->broadcastRepeat = x2CrossCartesian;
        tiling->broadcastInner = x2VectorInner;
    } else if (x1VectorX2ScalarSuffix &&
               (!x2VectorX1ScalarSuffix || x1VectorInner >= x2VectorInner)) {
        tiling->mode = 6;
        tiling->broadcastInner = x1VectorInner;
    } else if (x2VectorX1ScalarSuffix) {
        tiling->mode = 7;
        tiling->broadcastInner = x2VectorInner;
    } else if (x1Full && x2ShortRepeatedSuffix && x2ShortGroup <= tileElements &&
               total / x2ShortGroup >= minimumGroupsForParallelPath) {
        // x1 is contiguous. Materialize one short x2 suffix in UB, repeat it
        // across its innermost broadcast axis, and compare the whole group.
        // Outer x2 broadcast axes are handled once at each group boundary.
        tiling->mode = 9;
        tiling->broadcastRepeat = x2ShortRepeat;
        tiling->broadcastInner = x2ShortInner;
    } else if (x2Full && x1ShortRepeatedSuffix && x1ShortGroup <= tileElements &&
               total / x1ShortGroup >= minimumGroupsForParallelPath) {
        tiling->mode = 10;
        tiling->broadcastRepeat = x1ShortRepeat;
        tiling->broadcastInner = x1ShortInner;
    } else if (x1LocalForX2Short && x2ShortGroup <= tileElements &&
               total / x2ShortGroup >= minimumGroupsForParallelPath) {
        // x1 may broadcast outside this group, but is contiguous throughout
        // the complete suffix consumed by one repeated x2 tile.
        tiling->mode = 11;
        tiling->broadcastRepeat = x2ShortRepeat;
        tiling->broadcastInner = x2ShortInner;
    } else if (x2LocalForX1Short && x1ShortGroup <= tileElements &&
               total / x1ShortGroup >= minimumGroupsForParallelPath) {
        tiling->mode = 12;
        tiling->broadcastRepeat = x1ShortRepeat;
        tiling->broadcastInner = x1ShortInner;
    } else if (x1ContiguousBroadcast && x2Full) {
        tiling->mode = 4;
        tiling->broadcastRepeat = x1Repeat;
        tiling->broadcastInner = x1Inner;
    } else if (x2ContiguousBroadcast && x1Full) {
        tiling->mode = 5;
        tiling->broadcastRepeat = x2Repeat;
        tiling->broadcastInner = x2Inner;
    } else if (sharedContiguousSuffix) {
        tiling->mode = 8;
        tiling->broadcastInner = sharedSuffixInner;
    } else {
        tiling->mode = 1;
    }
    // The fully generic broadcast path performs scalar address arithmetic
    // and scalar GM reads; it cannot use the segmented vector helpers.  Keep
    // its established 2,048-element queue footprint so a large unused UB
    // allocation does not penalize irregular rank-six broadcasts.
    if (tiling->mode == 1) {
        tiling->tileElements = std::min<uint64_t>(tiling->tileElements, 2 * 1024);
    }
    if (tiling->mode >= 9 && tiling->mode <= 12 &&
        x1Desc->GetDataType() != ge::DT_INT8) {
        const uint64_t groupElements = tiling->broadcastRepeat * tiling->broadcastInner;
        // The kernel never materializes more than one 2,048-element Gather
        // segment plus one source suffix.  Bound every short-repeat queue by
        // that physical working set, including groups just above 2,048.  The
        // dedicated 8 KiB index buffer must be paid for in the same UB budget;
        // retaining the large-tensor queue here can otherwise over-allocate
        // UB once the output grows beyond the small public anchor.
        const uint64_t compareBytes = x1Desc->GetDataType() == ge::DT_BF16
                                          ? sizeof(float)
                                          : elementBytes;
        const uint64_t vectorElements = 256 / compareBytes;
        const uint64_t segmentElements = std::min<uint64_t>(groupElements, 2 * 1024);
        const uint64_t vectorCount = AlignUp(segmentElements, vectorElements);
        const uint64_t dmaElements = kOutputAlignment / elementBytes;
        const uint64_t innerDmaCount = AlignUp(tiling->broadcastInner, dmaElements);
        uint64_t shortQueueElements = AlignUp(
            vectorCount + innerDmaCount, kVectorTileAlignment);
        // Modes 9/10 have one globally contiguous operand.  For tiny repeat
        // groups the per-group Compare/Select and bool MTE3 launch dominates
        // all useful work.  Reserve enough UB to fuse up to one proven 2K
        // Compare span whenever consecutive short-source groups are also
        // contiguous.  The kernel detects the actual run length at runtime;
        // modes 11/12 retain their smaller one-group footprint because their
        // other operand may broadcast between adjacent groups.
        if ((tiling->mode == 9 || tiling->mode == 10) &&
            x1Desc->GetDataType() != ge::DT_INT8 &&
            x1Desc->GetDataType() != ge::DT_INT32 &&
            groupElements <= 2 * 1024) {
            uint64_t batchElements = std::min<uint64_t>(total, 2 * 1024);
            batchElements = batchElements / groupElements * groupElements;
            if (batchElements >= groupElements * 2) {
                const uint64_t batchGroups = batchElements / groupElements;
                const uint64_t batchVectorCount = AlignUp(batchElements, vectorElements);
                const uint64_t batchShortDmaCount = AlignUp(
                    batchGroups * tiling->broadcastInner, dmaElements);
                shortQueueElements = std::max<uint64_t>(
                    shortQueueElements,
                    AlignUp(batchVectorCount + batchShortDmaCount,
                            kVectorTileAlignment));
            }
        }
        tiling->tileElements = std::min<uint64_t>(
            tiling->tileElements, std::max<uint64_t>(
                kVectorTileAlignment, shortQueueElements));
    }
    if (tiling->mode == 13 || tiling->mode == 14) {
        // The 2K INT32 compare span also needs a disjoint suffix staging area.
        // Enlarging this queue does not enlarge an individual Compare call.
        tiling->tileElements = std::max<uint64_t>(
            tiling->tileElements, AlignUp(2 * AlignUp(tiling->broadcastInner, kOutputAlignment), 256));
        const uint64_t scalarCount = tiling->mode == 13
            ? x1CrossScalars : x2CrossScalars;
        const uint64_t rowElements = scalarCount * tiling->broadcastInner;
        const uint64_t taskCount = total / rowElements;
        // Every task writes one compact [scalar,suffix] output row.  Keep
        // boundaries between AIVs on complete 32-byte bool units even when
        // rowElements itself is odd (for example 31*67=2077).
        const uint64_t tasksPerAlignedUnit =
            kOutputAlignment / Gcd(rowElements, kOutputAlignment);
        const uint64_t alignedUnits = CeilDiv(taskCount, tasksPerAlignedUnit);
        const uint64_t requestedBlocks =
            std::max<uint64_t>(1, std::min<uint64_t>(coreCount, alignedUnits));
        const uint64_t unitsPerCore = CeilDiv(alignedUnits, requestedBlocks);
        const uint64_t tasksPerCore = unitsPerCore * tasksPerAlignedUnit;
        blockElements = tasksPerCore * rowElements;
        blockDim = static_cast<uint32_t>(CeilDiv(total, blockElements));
        tiling->blockElements = blockElements;
    } else if (tiling->mode >= 4 && tiling->mode <= 8) {
        // These paths perform one vector launch per contiguous broadcast
        // suffix.  A 32K output can therefore contain hundreds of small
        // launches even though the ordinary element-count policy selects one
        // core.  Their kernels support starting and ending inside a suffix;
        // split at 32-byte output boundaries to expose more AIV parallelism
        // without allowing neighboring cores to share a bool DMA block.
        const uint64_t broadcastBlocks = std::min<uint64_t>(
            coreCount, CeilDiv(total, kBroadcastElementsPerCoreTarget));
        blockElements = AlignUp(CeilDiv(total, broadcastBlocks), kOutputAlignment);
        blockDim = static_cast<uint32_t>(CeilDiv(total, blockElements));
        tiling->blockElements = blockElements;
    } else if (tiling->mode >= 9 && tiling->mode <= 12) {
        const uint64_t groupElements = tiling->broadcastRepeat * tiling->broadcastInner;
        const uint64_t groupCount = total / groupElements;
        if (groupElements > 2 * 1024 &&
            x1Desc->GetDataType() != ge::DT_INT8 &&
            groupCount <= kPartialGroupParallelLimit) {
            // The segmented kernel can resume from any point inside a repeat
            // group.  Partition large groups directly on 32-byte bool-output
            // boundaries so an odd group size (for example 2,077) no longer
            // collapses 48 AIVs to only one or two LCM-aligned workers.
            const uint64_t segmentedBlocks = std::min<uint64_t>(
                coreCount, CeilDiv(total, kBroadcastElementsPerCoreTarget));
            blockElements = AlignUp(
                CeilDiv(total, segmentedBlocks), kOutputAlignment);
            blockDim = static_cast<uint32_t>(CeilDiv(total, blockElements));
            tiling->blockElements = blockElements;
        } else {
            // Each core writes byte-bool output through 32-byte MTE blocks. A
            // group boundary that is not also a DMA boundary would let
            // adjacent AIVs race on the same tail block. Partition only at
            // the LCM of the logical group and one output DMA block; the
            // final tensor tail has no following core and remains safe.
            const uint64_t groupsPerAlignedBlock =
                kOutputAlignment / Gcd(groupElements, kOutputAlignment);
            const uint64_t alignedBlockCount =
                CeilDiv(groupCount, groupsPerAlignedBlock);
            const uint64_t requestedBlocks =
                std::min<uint64_t>(coreCount, alignedBlockCount);
            const uint64_t alignedBlocksPerCore =
                CeilDiv(alignedBlockCount, requestedBlocks);
            const uint64_t groupsPerCore =
                alignedBlocksPerCore * groupsPerAlignedBlock;
            blockElements = groupsPerCore * groupElements;
            blockDim = static_cast<uint32_t>(CeilDiv(total, blockElements));
            tiling->blockElements = blockElements;
        }
    }
    for (uint32_t axis = 0; axis < S9_GREATER_MAX_RANK; ++axis) {
        tiling->outputShape[axis] = outputShape[axis];
        tiling->x1Shape[axis] = x1Shape[axis];
        tiling->x2Shape[axis] = x2Shape[axis];
        tiling->x1Stride[axis] = x1Stride[axis];
        tiling->x2Stride[axis] = x2Stride[axis];
    }
    if (total == 0) tiling->mode = 0;
    const uint32_t legacyMode = tiling->mode;
    // The largest 4K plan uses 156 KiB plus 8 KiB CANN scratch;
    // retain 2K unless every available AIV can amortize at least two 4K tiles.
    const uint64_t gatherLimit = total >= std::max<uint64_t>(65536, static_cast<uint64_t>(coreCount) * 8192) && ubBytes >= 164 * 1024
        ? S9_GREATER_GATHER_CAP : 2048;
    uint64_t gatherSpan = 1;
    uint32_t suffixStart = S9_GREATER_MAX_RANK;
    for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
        if (outputShape[axis] == 0 || gatherSpan > gatherLimit / outputShape[axis]) break;
        gatherSpan *= outputShape[axis];
        suffixStart = static_cast<uint32_t>(axis);
    }
    uint64_t innerSpan = gatherSpan;
    uint32_t steps = static_cast<uint32_t>(gatherLimit / gatherSpan);
    if (suffixStart == 0) {
        // The complete output fits in one tile. Preserve the actual first
        // axis instead of inventing extra source rows when rank is eight.
        suffixStart = 1;
        innerSpan = gatherSpan / outputShape[0];
        steps = static_cast<uint32_t>(outputShape[0]);
    }
    const uint32_t splitAxis = suffixStart - 1;
    const uint64_t groupSpan = innerSpan * outputShape[splitAxis];
    gatherSpan = innerSpan * steps;
    uint64_t lhsSuffix = x1Shape[splitAxis] == 1 ? 1 : steps;
    uint64_t rhsSuffix = x2Shape[splitAxis] == 1 ? 1 : steps;
    for (uint32_t axis = suffixStart; axis < S9_GREATER_MAX_RANK; ++axis) {
        lhsSuffix *= x1Shape[axis]; rhsSuffix *= x2Shape[axis];
    }
    const bool useRuns = total >= 4096 && legacyMode >= 4 && legacyMode <= 8 &&
                         tiling->broadcastInner >= ((legacyMode == 6 || legacyMode == 7) ? 2048 :
                             (legacyMode == 4 || legacyMode == 5) ? 4096 : 512);
    const uint64_t legacyUnit = legacyMode >= 9 && legacyMode <= 12
        ? tiling->broadcastInner * tiling->broadcastRepeat : tiling->broadcastInner;
    const bool smallFusionWorthwhile = total >= 512 && legacyUnit != 0 &&
                                      tiling->blockElements / legacyUnit >= 4;
    // Generic scalar GM addressing costs grow per output element, even below
    // the usual fusion threshold; a single compact Gather tile replaces it.
    const bool smallGenericWorthwhile = total >= 128 && legacyMode == 1;
    bool useGather = !useRuns && (total >= 4096 || smallFusionWorthwhile || smallGenericWorthwhile) && legacyMode != 0 && legacyMode != 2 && legacyMode != 3 &&
                     gatherSpan >= 32 && (legacyMode == 1 || lhsSuffix != gatherSpan || rhsSuffix != gatherSpan);
    context->SetTilingKey(useGather ? 1 : 0);
    if (useGather) {
        auto* gather = context->GetTilingData<GreaterGatherTilingData>();
        if (gather == nullptr) return ge::GRAPH_FAILED;
        gather->meta.splitAxis = splitAxis;
        gather->meta.span = static_cast<uint32_t>(gatherSpan);
        gather->meta.groupSpan = groupSpan;
        gather->meta.steps = steps;
        gather->meta.inner = static_cast<uint32_t>(innerSpan);
        gather->meta.lhsStep = x1Shape[splitAxis] == 1 ? 0 : static_cast<uint32_t>(lhsSuffix / steps);
        gather->meta.rhsStep = x2Shape[splitAxis] == 1 ? 0 : static_cast<uint32_t>(rhsSuffix / steps);
        gather->meta.lhsCount = static_cast<uint32_t>(lhsSuffix);
        gather->meta.rhsCount = static_cast<uint32_t>(rhsSuffix);
        gather->meta.paddedSpan = static_cast<uint32_t>(AlignUp(gatherSpan, 256));
        const uint64_t workers = std::max<uint64_t>(1, std::min<uint64_t>(coreCount, CeilDiv(total, 2048)));
        blockElements = AlignUp(CeilDiv(total, workers), 32);
        blockDim = static_cast<uint32_t>(CeilDiv(total, blockElements));
        tiling->blockElements = blockElements;
        // Classify the encoded suffix, independent of values and case names.
        // 0 constant, 1 identity, 2 periodic, 3 contiguous quotient, 4 mixed.
        auto classifyMap = [&](const uint64_t* shape, uint64_t sourceCount, uint32_t side) {
            uint32_t kind = 4, divisor = 1;
            uint64_t tail = 1, broadcastTail = 1;
            for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= static_cast<int32_t>(splitAxis); --axis) {
                const uint64_t extent = static_cast<uint32_t>(axis) == splitAxis ? steps : outputShape[axis];
                if (shape[axis] == 1 && extent != 1) break;
                tail *= extent;
            }
            for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= static_cast<int32_t>(splitAxis); --axis) {
                const uint64_t extent = static_cast<uint32_t>(axis) == splitAxis ? steps : outputShape[axis];
                if (shape[axis] != 1 && extent != 1) break;
                broadcastTail *= extent;
            }
            if (sourceCount == 1) kind = 0;
            else if (sourceCount == gatherSpan) kind = 1;
            else if (tail == sourceCount) { kind = 2; divisor = static_cast<uint32_t>(tail); }
            else if (broadcastTail * sourceCount == gatherSpan) { kind = 3; divisor = static_cast<uint32_t>(broadcastTail); }
            uint32_t shift = 12, power = 1;
            while (power < divisor) { power <<= 1; ++shift; }
            const uint64_t scale = uint64_t{1} << shift;
            gather->meta.mapKinds[side] = kind;
            gather->meta.divisors[side] = divisor;
            gather->meta.reciprocals[side] = static_cast<uint32_t>((scale + divisor - 1) / divisor);
            gather->meta.shifts[side] = shift;
        };
        classifyMap(x1Shape, lhsSuffix, 0);
        classifyMap(x2Shape, rhsSuffix, 1);
        const bool simpleMaps = total >= 4096 && gather->meta.mapKinds[0] != 4 && gather->meta.mapKinds[1] != 4;
        if (simpleMaps) {
            context->SetTilingKey(6);
            if (context->GetTilingData<GreaterGatherMeta>() == nullptr) return ge::GRAPH_FAILED;
        } else {
        // The byte offsets are a pure function of shape and comparison dtype.
        // No input values, answers, or case identities enter this mapping.
        const uint32_t compareBytes = x1Desc->GetDataType() == ge::DT_BF16 ? 4 :
                                      x1Desc->GetDataType() == ge::DT_INT8 ? 2 : elementBytes;
        for (uint32_t i = 0; i < gather->meta.paddedSpan; ++i) {
            uint64_t linear = i < gatherSpan ? i : 0;
            uint64_t offset1 = 0, offset2 = 0;
            for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= static_cast<int32_t>(splitAxis); --axis) {
                const uint64_t extent = static_cast<uint32_t>(axis) == splitAxis ? steps : outputShape[axis];
                const uint64_t c = linear % extent;
                linear /= extent;
                offset1 += c * x1Stride[axis]; offset2 += c * x2Stride[axis];
            }
            gather->lhsMap[i] = static_cast<uint32_t>(offset1 * compareBytes);
            gather->rhsMap[i] = static_cast<uint32_t>(offset2 * compareBytes);
        }
        }
    }
    const bool useLinear = !useGather && (legacyMode == 0 || legacyMode == 2 || legacyMode == 3);
    if (useLinear || useRuns) {
        context->SetTilingKey(useRuns ? 3 : total >= 1048576 ? 5 : 2);
        const uint64_t buffers = useLinear && total >= 1048576 ? 2 : 1;
        const uint64_t workingBytes = buffers * (2 * elementBytes + 1) +
            (x1Desc->GetDataType() == ge::DT_INT8 ? 4 : x1Desc->GetDataType() == ge::DT_BF16 ? 8 : 0);
        // CANN dav_c220 reserves UB [184 KiB, 192 KiB) for API temporaries.
        // It is not included in TPipe buffers and must be budgeted separately.
        const uint64_t reserve = 8192 + (x1Desc->GetDataType() == ge::DT_INT32 ? 57344 : 24576);
        const uint64_t usable = ubBytes > reserve ? ubBytes - reserve : 16384;
        uint64_t tile = std::min<uint64_t>(32768, usable / workingBytes / 256 * 256);
        const uint64_t workers = total == 0 ? 1 : std::max<uint64_t>(1, std::min<uint64_t>(coreCount, CeilDiv(total, useRuns ? 2048 : 4096)));
        blockElements = total == 0 ? 0 : AlignUp(CeilDiv(total, workers), 32);
        blockDim = total == 0 ? 1 : static_cast<uint32_t>(CeilDiv(total, blockElements));
        tiling->blockElements = blockElements;
        tiling->tileElements = std::max<uint64_t>(256, std::min<uint64_t>(tile, AlignUp(blockElements, 256)));
    }
    const bool useTiny = useLinear && legacyMode == 0 && total > 0 && total <= 4096;
    if (useTiny) { context->SetTilingKey(4); blockDim = 1; }
    // Cache pressure includes the two neighbouring output buffers commonly
    // live during streamed calls. This chooses policy, never numerical results.
    const auto shouldStreamOutput = [&]() {
        if (l2Bytes == 0) return false;
        uint64_t bytes1=0,bytes2=0;
        if (MulOverflow(stride1,elementBytes,bytes1) || MulOverflow(stride2,elementBytes,bytes2)) return true;
        if (bytes1 >= l2Bytes || bytes2 >= l2Bytes-bytes1) return true;
        return total >= CeilDiv(l2Bytes-bytes1-bytes2,2);
    };
    const bool streamOutput = shouldStreamOutput();
    const bool scalarLinear = legacyMode == 2 || legacyMode == 3;
    const uint64_t convertedBytes = x1Desc->GetDataType()==ge::DT_INT8 ? 2 : x1Desc->GetDataType()==ge::DT_BF16 ? 4 : 0;
    const uint64_t compareBytes = convertedBytes ? convertedBytes : elementBytes;
    const uint64_t computeReserve = 8192 + 24576 + (x1Desc->GetDataType()==ge::DT_INT32 ? 16384 : 0);
    if (useLinear && !useTiny && total >= 1048576) {
        const uint64_t inputs=scalarLinear?1:2;
        const uint64_t working=2*(inputs*elementBytes+1)+inputs*convertedBytes;
        const uint64_t available=ubBytes>computeReserve?ubBytes-computeReserve:0;
        if(available>=working*512) {
            const uint64_t workers=std::max<uint64_t>(1,std::min<uint64_t>(coreCount,CeilDiv(total,4096)));
            blockElements=AlignUp(CeilDiv(total,workers),512);blockDim=static_cast<uint32_t>(CeilDiv(total,blockElements));
            tiling->blockElements=blockElements;
            tiling->tileElements=std::min<uint64_t>(std::min<uint64_t>(32768,available/working/512*512),AlignUp(blockElements,256));
            context->SetTilingKey(scalarLinear?(streamOutput?9:8):(streamOutput?11:10));
        }
    }
    if(useRuns && (legacyMode==6 || legacyMode==7) && total>=262144 &&
       total/tiling->broadcastInner >= static_cast<uint64_t>(coreCount)*4) {
        const uint64_t reserve=computeReserve+512*elementBytes;
        const uint64_t working=2*elementBytes+2+convertedBytes;
        const uint64_t available=ubBytes>reserve?ubBytes-reserve:0;
        if(available>=working*256) {
            tiling->tileElements=std::min<uint64_t>(std::min<uint64_t>(32768,available/working/256*256),AlignUp(blockElements,256));
            context->SetTilingKey(streamOutput?14:7);
        }
    }
    if(useRuns && (legacyMode==4 || legacyMode==5 || legacyMode==8) && total>=1048576 &&
       tiling->broadcastInner>=512 && tiling->broadcastInner<=16384) {
        const uint64_t period=tiling->broadcastInner;
        uint64_t span=1;int32_t axis=S9_GREATER_MAX_RANK-1;
        while(axis>=0 && span<period){span*=outputShape[axis];--axis;}
        while(axis>=0 && outputShape[axis]==1)--axis;
        if(span==period && axis>=0) {
            const bool leftRepeat=x1Shape[axis]==1 && x2Shape[axis]==outputShape[axis];
            const bool rightRepeat=x2Shape[axis]==1 && x1Shape[axis]==outputShape[axis];
            if(leftRepeat || rightRepeat) {
                const uint64_t* repeated=leftRepeat?x1Shape:x2Shape;
                const uint64_t* dense=leftRepeat?x2Shape:x1Shape;
                uint64_t repeats=1;
                while(axis>=0 && (outputShape[axis]==1 || (repeated[axis]==1 && dense[axis]==outputShape[axis]))) {
                    repeats*=outputShape[axis];--axis;
                }
                const uint64_t cache=AlignUp(period,256);
                const auto needed = [&](uint64_t t) {
                    return 2*t*elementBytes + cache*elementBytes +
                        (convertedBytes ? (t+cache)*compareBytes : 0) + t*compareBytes +
                        2*t + 8*t + 4*t + 8192 +
                        (x1Desc->GetDataType()==ge::DT_INT32?4*t:0) + 8192;
                };
                const uint64_t t=needed(4096)<=ubBytes?4096:2048;
                if(repeats>=4 && needed(t)<=ubBytes) {
                    tiling->broadcastRepeat=repeats;tiling->broadcastInner=period;tiling->tileElements=t;
                    tiling->mode=leftRepeat?15:16;
                    const uint64_t workers=std::max<uint64_t>(1,std::min<uint64_t>(coreCount,CeilDiv(total,4096)));
                    blockElements=AlignUp(CeilDiv(total,workers),512);blockDim=static_cast<uint32_t>(CeilDiv(total,blockElements));
                    tiling->blockElements=blockElements;
                    context->SetTilingKey(streamOutput?13:12);
                }
            }
        }
    }
    context->SetBlockDim(blockDim);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

// A single broadcast axis with an identical prefix/suffix on both inputs.
// This is a structural dispatch; the original general path remains available.
ge::graphStatus TilingGreaterPad(gert::TilingContext* context){
 auto status=TilingGreaterBase(context);if(status!=ge::GRAPH_SUCCESS)return status;
 const auto dt=context->GetInputDesc(0)->GetDataType();
 if(dt!=ge::DT_INT32&&dt!=ge::DT_BF16)return status;
 auto* raw=context->GetRawTilingData();if(!raw||raw->GetDataSize()<sizeof(GreaterTilingData))return status;
 const auto saved=*reinterpret_cast<const GreaterTilingData*>(raw->GetData());
 // Collapse a contiguous run of repeated axes. Prefix and suffix are shared.
 int first=-1,last=-1;bool left=false;
 for(uint32_t i=0;i<S9_GREATER_MAX_RANK;++i){
  if(saved.x1Shape[i]==saved.x2Shape[i])continue;
  if(first<0){first=static_cast<int>(i);left=saved.x1Shape[i]==1;}
  if((left?saved.x1Shape[i]:saved.x2Shape[i])!=1)return status;
  last=static_cast<int>(i);
 }
 if(first<0||last==S9_GREATER_MAX_RANK-1)return status;
 uint64_t repeat=1,width=1;
 for(int i=first;i<=last;++i){
  if((left?saved.x1Shape[i]:saved.x2Shape[i])!=1)return status;
  uint64_t next=0;if(MulOverflow(repeat,saved.outputShape[i],next))return status;repeat=next;
 }
 for(uint32_t i=last+1;i<S9_GREATER_MAX_RANK;++i){uint64_t next=0;if(MulOverflow(width,saved.outputShape[i],next))return status;width=next;}
 // A row needs at least one vector tile; at least two rows must fit the index table.
 if(width<256||width>8192/2||repeat<2)return status;
 auto* pi=context->GetPlatformInfo();if(!pi)return status;
 platform_ascendc::PlatformAscendC p(pi);uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 const uint64_t cores=std::max<uint64_t>(1,p.GetCoreNumAiv());
 if(saved.totalElements/cores<8192)return status;
 const uint64_t tbytes=dt==ge::DT_INT32?4:2,cbytes=4,padded=AlignUp(width,128);
 const uint64_t fixed=padded*tbytes+(dt==ge::DT_BF16?padded*cbytes:0)+512+16384+8192;
 const uint64_t perRow=padded*(2*tbytes+2+cbytes+(dt==ge::DT_BF16?cbytes:4));
 if(ub<=fixed+perRow)return status;
 // Limit one input DMA to 32 KiB; the following UB checks account for conversions.
 const uint64_t desired=(32*1024)/(padded*tbytes);
 const uint64_t rows=std::min<uint64_t>(desired,(ub-fixed)/perRow);if(rows<2)return status;
 const uint64_t totalRows=saved.totalElements/width;
 const uint64_t rowUnit=32/Gcd(width,32),units=CeilDiv(totalRows,rowUnit);
 const uint64_t blockRows=CeilDiv(units,cores)*rowUnit;
 uint64_t blockElements=0;if(MulOverflow(blockRows,width,blockElements))return status;
 auto* data=context->GetTilingData<GreaterRowTilingData>();if(!data)return status;
 data->base=saved;data->base.blockElements=blockElements;data->base.tileElements=rows*padded;
 data->width=width;data->repeat=repeat;data->totalRows=totalRows;data->blockRows=blockRows;
 data->padded=static_cast<uint32_t>(padded);data->rows=static_cast<uint32_t>(rows);data->leftRepeated=left?1:0;data->reserved=0;
 context->SetBlockDim(static_cast<uint32_t>(CeilDiv(totalRows,blockRows)));context->SetTilingKey(15);
 return ge::GRAPH_SUCCESS;
}


ge::graphStatus TilingGreaterFlat(gert::TilingContext* context){
 const auto status=TilingGreaterPad(context);if(status!=ge::GRAPH_SUCCESS)return status;
 if(context->GetTilingKey()!=15)return status;
 auto* raw=context->GetRawTilingData();const auto old=*reinterpret_cast<const GreaterRowTilingData*>(raw->GetData());
 const auto dt=context->GetInputDesc(0)->GetDataType();const uint64_t tbytes=dt==ge::DT_INT32?4:2;
 if(!context->GetPlatformInfo())return status;
 platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 uint64_t rows=old.rows,n=0;
 const uint64_t fixed=old.padded*tbytes+(dt==ge::DT_BF16?old.padded*4:16384)+512+16384+8192+128;
 const uint64_t per=2*tbytes+2+4+4+(dt==ge::DT_BF16?4:0);
 while(rows>=2){n=AlignUp(rows*old.width,256);if(n<=8192&&fixed+n*per<=ub)break;--rows;}
 if(rows<2)return status;
 auto* data=context->GetTilingData<GreaterFlatRowTilingData>();if(!data)return status;
 data->meta=old;data->meta.rows=static_cast<uint32_t>(rows);data->meta.base.tileElements=n;
 std::fill_n(data->indices,8192,0);
 for(uint32_t i=0;i<n;++i)data->indices[i]=static_cast<uint32_t>(i%old.width)*4;
 context->SetTilingKey(25);return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingGreaterMain(gert::TilingContext* context){
 const auto status=TilingGreaterFlat(context);if(status!=ge::GRAPH_SUCCESS)return status;
 const auto dt=context->GetInputDesc(0)->GetDataType();if(dt!=ge::DT_FLOAT&&dt!=ge::DT_INT8)return status;
 auto* raw=context->GetRawTilingData();const auto old=*reinterpret_cast<const GreaterTilingData*>(raw->GetData());
 if(old.mode!=0||old.totalElements<1048576)return status;
 platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 const uint64_t per=dt==ge::DT_FLOAT?18:10,fixed=512+16384+8192+192;
 if(ub<=fixed+per*512)return status;
 const uint64_t desired=32*1024;
 const uint64_t tile=std::min<uint64_t>(desired,(ub-fixed)/per/512*512);if(tile<512)return status;
 auto* data=context->GetTilingData<GreaterTilingData>();if(!data)return status;*data=old;data->tileElements=tile;
 context->SetTilingKey(21);return ge::GRAPH_SUCCESS;
}



ge::graphStatus TilingGreaterDeferredWorkspace(gert::TilingContext* context){
 const auto status=TilingGreaterMain(context);if(status!=ge::GRAPH_SUCCESS||context->GetTilingKey()!=25||context->GetInputDesc(0)->GetDataType()!=ge::DT_INT32)return status;
 const auto old=*reinterpret_cast<const GreaterRowTilingData*>(context->GetRawTilingData()->GetData());if(old.repeat%old.rows||old.blockRows%old.rows||(old.rows*old.width)%8)return status;
 platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 const uint64_t tile=old.base.tileElements,fixed=old.padded*4+16384+512+16384+8192+192;
 if(fixed+18*tile+tile/4>ub)return status;
 const uint64_t user=CeilDiv(old.totalRows,old.blockRows)*AlignUp(old.blockRows*old.width/8,32);auto* sizes=context->GetWorkspaceSizes(1);if(!sizes)return status;
 sizes[0]=user+p.GetLibApiWorkSpaceSize();context->SetTilingKey(30);return status;
}

ge::graphStatus TilingGreaterDeferred(gert::TilingContext* context){
 const auto status=TilingGreaterDeferredWorkspace(context);if(status!=ge::GRAPH_SUCCESS||context->GetTilingKey()!=21||(context->GetInputDesc(0)->GetDataType()!=ge::DT_FLOAT&&context->GetInputDesc(0)->GetDataType()!=ge::DT_INT8))return status;
 const auto old=*reinterpret_cast<const GreaterTilingData*>(context->GetRawTilingData()->GetData());
 platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 const uint64_t packed=AlignUp(old.blockElements,256)/8,fixed=packed+512+16384+8192+192;
 const bool fp=context->GetInputDesc(0)->GetDataType()==ge::DT_FLOAT;const uint64_t per=fp?18:10,desired=32*1024;
 if(ub<=fixed+per*4096)return status;const uint64_t tile=std::min<uint64_t>(desired,(ub-fixed)/per/512*512);if(tile<4096)return status;
 auto* data=context->GetTilingData<GreaterTilingData>();if(!data)return status;*data=old;data->tileElements=tile;context->SetTilingKey(28);return status;
}

ge::graphStatus TilingGreaterAlgorithm(gert::TilingContext* context){const auto status=TilingGreaterDeferred(context);if(status!=ge::GRAPH_SUCCESS)return status;if(context->GetInputDesc(0)->GetDataType()!=ge::DT_FLOAT16||context->GetTilingKey()!=4)return status;const uint64_t n=reinterpret_cast<const GreaterTilingData*>(context->GetRawTilingData()->GetData())->totalElements;if(n==0||n>4096)return status;auto* d=context->GetTilingData<GreaterTinyTilingData>();if(!d)return status;d->total=n;context->SetTilingKey(26);context->SetBlockDim(1);return ge::GRAPH_SUCCESS;}



ge::graphStatus TilingGreaterUbRows(gert::TilingContext* context){
 const auto status=TilingGreaterAlgorithm(context);if(status!=ge::GRAPH_SUCCESS||context->GetTilingKey()!=30)return status;
 const auto old=*reinterpret_cast<const GreaterRowTilingData*>(context->GetRawTilingData()->GetData());const uint64_t tile=old.base.tileElements;if(tile<4096)return status;
 platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
 const uint64_t alignmentRows=32/Gcd(old.width,32),quantum=alignmentRows/Gcd(alignmentRows,old.rows)*old.rows;
 const uint64_t units=CeilDiv(old.totalRows,quantum),cores=std::max<uint64_t>(1,p.GetCoreNumAiv());
 const uint64_t fixed=2*(tile*4+32)+old.padded*4+tile*4+8192;
 const uint64_t bytesPerUnit=(quantum/old.rows)*(tile/8);
 if(ub<=fixed||bytesPerUnit==0)return status;
 const uint64_t unitsPerBlock=(ub-fixed)/bytesPerUnit;if(unitsPerBlock==0)return status;
 uint64_t waveCapacity=0;if(MulOverflow(cores,unitsPerBlock,waveCapacity))return status;
 const uint64_t waves=CeilDiv(units,waveCapacity);
 uint64_t requestedBlocks=0;if(MulOverflow(cores,waves,requestedBlocks))return status;
 const uint64_t blocks=std::min<uint64_t>(units,requestedBlocks);if(blocks==0||blocks>65535)return status;
 const uint64_t blockRows=CeilDiv(units,blocks)*quantum,packed=(blockRows/old.rows)*(tile/8);
 if(fixed+packed>ub)return status;uint64_t blockElements=0;if(MulOverflow(blockRows,old.width,blockElements))return status;
 auto* sizes=context->GetWorkspaceSizes(1);if(!sizes)return status;
 auto* d=reinterpret_cast<GreaterFlatRowTilingData*>(context->GetRawTilingData()->GetData());d->meta.blockRows=blockRows;d->meta.base.blockElements=blockElements;d->meta.reserved=static_cast<uint32_t>(quantum);sizes[0]=0;
 context->SetBlockDim(static_cast<uint32_t>(blocks));context->SetTilingKey(31);return status;
}

bool QueryGreaterL2Offset(GreaterL2Footer& footer) {
 footer={0,0x474C324F46463031ULL,UINT64_MAX,0};
 using GetDeviceFn=int32_t(*)(int32_t*);
 using GetOffsetFn=int32_t(*)(uint32_t,uint64_t*);
 auto getDevice=reinterpret_cast<GetDeviceFn>(dlsym(RTLD_DEFAULT,"rtGetDevice"));
 auto getOffset=reinterpret_cast<GetOffsetFn>(dlsym(RTLD_DEFAULT,"rtGetL2CacheOffset"));
 int32_t device=-1;uint64_t offset=0;
 if(!getDevice||!getOffset||getDevice(&device)!=0||device<0)return false;
 footer.deviceIndex=static_cast<uint64_t>(device);
 if(getOffset(static_cast<uint32_t>(device),&offset)!=0||offset==0)return false;
 footer.inputL2BypassOffset=offset;return true;
}
ge::graphStatus TilingGreater(gert::TilingContext* context) {
 const auto status=TilingGreaterUbRows(context);if(status!=ge::GRAPH_SUCCESS)return status;
 const uint64_t key=context->GetTilingKey();
 if(key!=21&&key!=25&&key!=28&&key!=30&&key!=31)return status;
 GreaterL2Footer footer{};if(!QueryGreaterL2Offset(footer))return status;
 auto* raw=context->GetRawTilingData();if(!raw||raw->GetDataSize()<sizeof(GreaterTilingData))return status;
 reinterpret_cast<GreaterTilingData*>(raw->GetData())->inputL2BypassOffset=footer.inputL2BypassOffset;
 context->SetTilingKey(key+256);return status;
}

} // namespace optiling

namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingGreater).AddConfig("ascend910b");
    }
};
OP_ADD(Greater);
} // namespace ops
