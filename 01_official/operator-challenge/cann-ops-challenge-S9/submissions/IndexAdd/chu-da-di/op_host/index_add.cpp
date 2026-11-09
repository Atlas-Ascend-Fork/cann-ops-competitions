#include <dlfcn.h>
#include "index_add_tiling.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef OP_LOGE
#define OP_LOGE(...) ((void)0)
#endif

namespace {
constexpr uint64_t kDmaBlockBytes = 32;
constexpr uint64_t kAccumulatorBytes = sizeof(int32_t);
constexpr uint64_t kUbSafetyReserve = 8 * 1024;
constexpr uint64_t kSparseRowPresenceMaxBytes = 4 * 1024;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

uint64_t MaximumLinearRows(uint64_t elements, uint64_t rowElements,
                           uint32_t valueBytes, uint32_t availableCores)
{
    const uint64_t blocks = CeilDiv(elements * valueBytes, kDmaBlockBytes);
    const uint64_t cores = std::min<uint64_t>(availableCores, blocks);
    const uint64_t perCore = blocks / cores;
    const uint64_t extra = blocks % cores;
    const uint64_t blockElements = kDmaBlockBytes / valueBytes;
    uint64_t maximum = 0;
    for (uint64_t core = 0; core < cores; ++core) {
        const uint64_t startBlock = core * perCore + std::min(core, extra);
        const uint64_t length = perCore + (core < extra ? 1 : 0);
        const uint64_t start = startBlock * blockElements;
        const uint64_t end = std::min(elements, (startBlock + length) * blockElements);
        const uint64_t visits = (end - 1) / rowElements - start / rowElements + 1;
        maximum = std::max(maximum, visits);
    }
    return maximum;
}

uint64_t Product(const gert::Shape& shape, size_t begin, size_t end)
{
    uint64_t result = 1;
    for (size_t axis = begin; axis < end; ++axis) {
        const int64_t extent = shape.GetDim(axis);
        if (extent < 0 || (extent != 0 && result > std::numeric_limits<uint64_t>::max() /
                                                static_cast<uint64_t>(extent))) {
            return 0;
        }
        result *= static_cast<uint64_t>(extent);
    }
    return result;
}

uint32_t DTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}

uint64_t CastBytesPerElement(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return sizeof(float);
    }
    // Atlas A2 does not expose a direct INT8 -> INT16 vector Cast.  Every
    // INT8 value is exact in FP16, so reserve one FP16 bridge and one INT16
    // source tile.  INT16 modular accumulation is sufficient because only
    // the low byte is observable at the INT8 output.
    if (dtype == ge::DT_INT8) {
        return sizeof(uint16_t) + sizeof(int16_t);
    }
    return 0;
}
}  // namespace

namespace optiling {
static ge::graphStatus IndexAddTilingBase(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputDesc(1) == nullptr || context->GetInputDesc(2) == nullptr ||
        context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& selfShape = context->GetInputShape(0)->GetOriginShape();
    const auto& indexShape = context->GetInputShape(1)->GetOriginShape();
    const auto& sourceShape = context->GetInputShape(2)->GetOriginShape();
    const size_t rank = selfShape.GetDimNum();
    if (rank == 0 || sourceShape.GetDimNum() != rank || indexShape.GetDimNum() != 1) {
        OP_LOGE(context, "IndexAdd expects self/source with equal positive rank and one-dimensional index.");
        return ge::GRAPH_FAILED;
    }

    const int64_t* dimPointer = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (dimPointer == nullptr) {
        OP_LOGE(context, "IndexAdd dim attribute is missing.");
        return ge::GRAPH_FAILED;
    }
    int64_t dim = *dimPointer;
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        OP_LOGE(context, "IndexAdd dim is outside the input rank.");
        return ge::GRAPH_FAILED;
    }

    for (size_t axis = 0; axis < rank; ++axis) {
        if (axis != static_cast<size_t>(dim) && selfShape.GetDim(axis) != sourceShape.GetDim(axis)) {
            OP_LOGE(context, "IndexAdd self/source shapes differ outside dim.");
            return ge::GRAPH_FAILED;
        }
    }

    const uint64_t indexCount = static_cast<uint64_t>(std::max<int64_t>(0, indexShape.GetDim(0)));
    if (sourceShape.GetDim(dim) != static_cast<int64_t>(indexCount)) {
        OP_LOGE(context, "IndexAdd index length must equal source.shape[dim].");
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    if (context->GetInputDesc(2)->GetDataType() != dtype ||
        context->GetInputDesc(1)->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context, "IndexAdd requires equal self/source dtype and int32 index.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t dtypeBytes = DTypeBytes(dtype);
    if (dtypeBytes == 0) {
        OP_LOGE(context, "IndexAdd received an unsupported dtype.");
        return ge::GRAPH_FAILED;
    }

    const uint64_t outer = Product(selfShape, 0, static_cast<size_t>(dim));
    const uint64_t destinationAxis = static_cast<uint64_t>(std::max<int64_t>(0, selfShape.GetDim(dim)));
    const uint64_t inner = Product(selfShape, static_cast<size_t>(dim) + 1, rank);
    const uint64_t selfElements = outer * destinationAxis * inner;
    const uint64_t sourceElements = outer * indexCount * inner;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (ubBytes <= kUbSafetyReserve || availableCores == 0) {
        OP_LOGE(context, "IndexAdd failed to query usable AIV/UB resources.");
        return ge::GRAPH_FAILED;
    }

    IndexAddTilingData* tiling = context->GetTilingData<IndexAddTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *tiling = {};
    tiling->outer = outer;
    tiling->destinationAxis = destinationAxis;
    tiling->indexCount = indexCount;
    tiling->inner = inner;
    tiling->selfElements = selfElements;
    tiling->sourceElements = sourceElements;
    tiling->dtypeBytes = dtypeBytes;

    if (selfElements == 0 || inner == 0 || destinationAxis == 0 || outer == 0) {
        tiling->mode = INDEX_ADD_EMPTY;
        tiling->usedCoreNum = 1;
        context->SetBlockDim(1);
        context->SetTilingKey(INDEX_ADD_EMPTY);
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace != nullptr) {
            workspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }

    // Index is at most 8000 int32 values in the published specification, so
    // cache it once per active AIV.  Allocate one DMA block even for M == 0.
    // INT32 CompareScalar consumes 256 bytes per full vector repeat.  Cache a
    // padded index vector once per AIV and reserve its packed one-bit match
    // mask.  ROW_OWNER can then test 8 indices per scalar UB byte instead of
    // issuing one scalar GetValue for every update.
    const uint64_t indexVectorElements = AlignUp(std::max<uint64_t>(indexCount, 1), 64);
    const uint64_t indexBufferBytes = indexVectorElements * sizeof(int32_t);
    const uint64_t matchMaskBufferBytes = AlignUp(indexVectorElements / 8, kDmaBlockBytes);
    if (indexBufferBytes + matchMaskBufferBytes + kUbSafetyReserve >= ubBytes) {
        OP_LOGE(context, "IndexAdd index tensor does not fit the UB cache.");
        return ge::GRAPH_FAILED;
    }
    const uint64_t usableBytes =
        ubBytes - indexBufferBytes - matchMaskBufferBytes - kUbSafetyReserve;


    // When the indexed axis is the last axis, each outer slice is one
    // contiguous destination row and one contiguous source row:
    //
    //   out[o, j] = self[o, j] + sum_{m:index[m]=j} source[o, m].
    //
    // The generic slab kernel materializes every scalar as an aligned vector
    // tile, causing O(outer * (D + M)) tiny GM transactions and full-pipeline
    // barriers.  For large M this is overwhelmingly latency-bound.  If both
    // rows fit in UB, cache them once and perform the ordered scatter-add on
    // the scalar pipeline.  FP32 and INT32 need no widening, so this removes
    // thousands of 4-byte MTE2/MTE3 transfers without changing arithmetic or
    // index order.  The shape/resource predicate is generic to 910B-family
    // devices and intentionally excludes small-M launch-bound cases.
    const bool directScalarType = dtypeBytes != 0;
    const uint64_t contiguousDestinationBytes =
        AlignUp(destinationAxis * inner * dtypeBytes, kDmaBlockBytes);
    const uint64_t contiguousSourceBytes =
        AlignUp(indexCount * inner * dtypeBytes, kDmaBlockBytes);
    // A task owns complete 32-byte output blocks. Wider scalar-cached
    // slices that serialize at least 16 outer rows per output block need
    // the full AIV set; otherwise retain the generic vector path.
    uint64_t cacheRowGcd = destinationAxis * inner * dtypeBytes;
    uint64_t cacheDivisor = kDmaBlockBytes;
    while (cacheDivisor != 0) {
        const uint64_t remainder = cacheRowGcd % cacheDivisor;
        cacheRowGcd = cacheDivisor;
        cacheDivisor = remainder;
    }
    const uint64_t cacheOuterRowsPerTask = kDmaBlockBytes / cacheRowGcd;
    const uint64_t cacheTaskCount = CeilDiv(outer, cacheOuterRowsPerTask);
    const bool enoughCacheParallelism = inner == 1 || cacheOuterRowsPerTask < 16 ||
        cacheTaskCount >= availableCores;
    // A short INT8 row owns a full 32-byte output group. Require enough
    // independent groups before replacing the existing small-input path.
    const bool contiguousAxisRowsFit =
        directScalarType && inner <= 8 &&
        (indexCount >= 32 || (indexCount > 0 && outer >= 128 && destinationAxis * inner <= 128 &&
            (dtype != ge::DT_INT8 || cacheTaskCount >= availableCores))) &&
        (enoughCacheParallelism || cacheTaskCount < availableCores) &&
        selfElements + sourceElements >= 4096 &&
        contiguousDestinationBytes + contiguousSourceBytes +
                indexBufferBytes + kUbSafetyReserve <=
            ubBytes;
    if (contiguousAxisRowsFit) {
        const IndexAddTilingData uncachedPlan = *tiling;
        const uint64_t outerRowsPerTask = cacheOuterRowsPerTask;
        const uint64_t taskCount = cacheTaskCount;
        const bool linearOwnership = taskCount < availableCores &&
            MaximumLinearRows(selfElements, destinationAxis * inner, dtypeBytes, availableCores) < outerRowsPerTask;
        const uint32_t usedCoreNum = static_cast<uint32_t>(
            std::min<uint64_t>(availableCores, linearOwnership ? CeilDiv(selfElements * dtypeBytes, 32) : taskCount));
        tiling->tileInner = outerRowsPerTask;
        tiling->tileInnerAligned = 1;
        tiling->innerTileCount = 1;
        tiling->taskCount = taskCount;
        tiling->tasksPerCore = taskCount / usedCoreNum;
        tiling->extraCoreTasks = static_cast<uint32_t>(taskCount % usedCoreNum);
        tiling->usedCoreNum = usedCoreNum;
        tiling->mode = linearOwnership ? INDEX_ADD_LINEAR_AXIS_OWNER : INDEX_ADD_CONTIGUOUS_AXIS_OWNER;
        tiling->accumulatorRows = static_cast<uint32_t>(destinationAxis);
        tiling->indexBufferBytes = static_cast<uint32_t>(indexBufferBytes);
        tiling->indexVectorElements = static_cast<uint32_t>(indexVectorElements);
        tiling->matchMaskBufferBytes = 0;
        tiling->rawBufferBytes = 0;
        tiling->castBufferBytes = 0;
        tiling->accumulatorBufferBytes =
            static_cast<uint32_t>(contiguousDestinationBytes);
        tiling->sourceCacheBufferBytes =
            static_cast<uint32_t>(contiguousSourceBytes);
        tiling->rowPresenceBufferBytes = 0;
        // Ordered inverse waves: the kth occurrence of every destination
        // forms one conflict-free vector addition. A bounded table fits UB;
        // excess occurrences remain in original order in a compact tail list.
        if (inner == 1 && destinationAxis < 65536 && indexCount < 65536) {
            const uint64_t lanes = AlignUp(destinationAxis, dtype == ge::DT_INT8 ? 128 : 64);
            const uint64_t rawSourceBytes = dtype == ge::DT_INT8 ? AlignUp(indexCount, 128)
                : contiguousSourceBytes + kDmaBlockBytes;
            const uint64_t widePitch = dtype == ge::DT_INT8 ? rawSourceBytes + 128 : rawSourceBytes / dtypeBytes;
            const double meanDepth = static_cast<double>(indexCount) / destinationAxis;
            const uint64_t waveLimit = 16;
            const uint64_t batchLimit = 4;
            const uint64_t preferredDepth = std::min<uint64_t>(waveLimit,
                static_cast<uint64_t>(std::ceil(meanDepth + 6.0 * std::sqrt(meanDepth) + 2.0)));
            for (uint64_t batchRows : {64ULL, 32ULL, 24ULL, 18ULL, 16ULL, 12ULL, 8ULL, 6ULL, 4ULL, 3ULL, 2ULL, 1ULL}) {
                if (batchRows > batchLimit) { continue; }
                const uint64_t flatGroup = 1;
                const uint64_t vectorElements = batchRows * lanes;
                const uint64_t destinationBytes = vectorElements * dtypeBytes;
                const uint64_t sourceBytes = batchRows * rawSourceBytes;
                const uint64_t convertBytes = dtype == ge::DT_BF16 ? 2 * vectorElements * sizeof(float)
                    : dtype == ge::DT_INT8 ? (2 * vectorElements + batchRows * widePitch) * sizeof(uint16_t) : 0;
                const uint64_t gatherBytes = dtype == ge::DT_INT8 ? vectorElements * sizeof(uint16_t) : destinationBytes;
                const uint64_t expandedOffsets = 0;  // Gather uses a per-row LocalTensor view.
                const uint64_t fixedBytes = indexBufferBytes + destinationBytes + gatherBytes + sourceBytes +
                    lanes * sizeof(uint32_t) + convertBytes + expandedOffsets + kUbSafetyReserve;
                const uint64_t capacity = fixedBytes < ubBytes
                    ? std::min<uint64_t>(waveLimit, (ubBytes - fixedBytes) / (lanes * sizeof(uint32_t) * flatGroup)) : 0;
                if (capacity < 2 || (batchRows > 1 && capacity < preferredDepth)) { continue; }
                tiling->csrBatchSize = flatGroup;
                tiling->inverseWaveCapacity = static_cast<uint32_t>(capacity);
                tiling->inverseAlignedDestinations = static_cast<uint32_t>(lanes);
                tiling->inverseSourceSentinel = static_cast<uint32_t>(dtype == ge::DT_INT8
                    ? rawSourceBytes : contiguousSourceBytes / dtypeBytes);
                tiling->inverseConvertBufferBytes = static_cast<uint32_t>(convertBytes);
                tiling->accumulatorBufferBytes = static_cast<uint32_t>(destinationBytes);
                tiling->sourceCacheBufferBytes = static_cast<uint32_t>(sourceBytes);
                tiling->inverseBatchRows = batchRows;
                tiling->inverseSourceRowElements = rawSourceBytes / dtypeBytes;
                tiling->inverseWideRowElements = widePitch;
                break;
            }
        }
        // Runtime index concentration selects independent destination waves
        // or independent outer-row lanes. Every buffer is charged to UB.
        uint64_t crossRows = std::max<uint64_t>(8, outerRowsPerTask);
        const uint64_t crossRowLimit = std::min<uint64_t>(64,
            std::min<uint64_t>(outer / availableCores, 128 / inner));
        while (crossRows * 2 <= crossRowLimit) { crossRows *= 2; }
        const uint64_t crossValueBytes = dtype == ge::DT_INT8 ? 2 : dtypeBytes;
        const uint64_t crossCountLanes = std::max<uint64_t>(tiling->inverseAlignedDestinations,
            (AlignUp(destinationAxis, 64)));
        const uint64_t scalarFallbackBytes = indexBufferBytes + contiguousDestinationBytes +
            contiguousSourceBytes + crossCountLanes * sizeof(uint32_t) + kUbSafetyReserve;
        bool crossConfigured = false;
        for (uint64_t trialRows = crossRows; trialRows >= outerRowsPerTask; trialRows /= 2) {
            uint64_t crossLanes = 1;
            while (crossLanes < std::max<uint64_t>(trialRows * inner, 32 / crossValueBytes)) { crossLanes *= 2; }
            const uint64_t crossSourceTile = std::min<uint64_t>(128, 4096 / crossLanes);
            const uint64_t crossElements = crossLanes * crossSourceTile;
            const uint64_t crossFixedBytes = indexBufferBytes + crossCountLanes * sizeof(uint32_t) +
                crossElements * (std::max<uint64_t>(dtypeBytes, 2) + crossValueBytes + 8 +
                    (dtype == ge::DT_INT8 ? 2 : 0)) + crossLanes * 32 +
                (dtype == ge::DT_BF16 ? crossLanes * 2 * sizeof(float) : 0) + kUbSafetyReserve;
            const uint64_t ioOffsetElements = crossLanes + (dtype == ge::DT_INT8
                ? AlignUp(trialRows * 32, 128) : trialRows * 32 / dtypeBytes);
            if (ioOffsetElements <= crossElements && inner <= 8 &&
                (tiling->inverseWaveCapacity > 0 || scalarFallbackBytes <= ubBytes) &&
                crossSourceTile >= 32 && (inner > 1 && indexCount >= 2 * destinationAxis ? outer >= trialRows : outer >= availableCores * trialRows) && crossFixedBytes + 32 < ubBytes) {
                const uint64_t capacity = std::min<uint64_t>(destinationAxis,
                    (ubBytes - crossFixedBytes - 32) / (crossLanes * crossValueBytes + sizeof(uint32_t)));
                const bool fitsAllActive = capacity >= std::min(destinationAxis, indexCount);
                // Preserve the previously supported concentrated-index route.
                // Only newly enabled narrow-inner plans require every active row to fit.
                const bool legacyPartialCross = inner <= 4 && indexCount >= 512 &&
                    outer >= availableCores * trialRows;
                if (capacity > 0 && (inner == 1 || fitsAllActive || legacyPartialCross) &&
                    (!crossConfigured || fitsAllActive)) {
                    tiling->inverseAlignedDestinations = crossCountLanes;
                    tiling->crossOuterRows = trialRows;
                    tiling->crossLanes = crossLanes;
                    tiling->crossSourceTile = crossSourceTile;
                    tiling->crossActiveCapacity = capacity;
                    tiling->mode = INDEX_ADD_ADAPTIVE_AXIS_OWNER;
                    crossConfigured = true;
                }
                // A diffuse narrow-inner input has no inverse-wave fallback.
                // Reduce the row batch when that allows all active rows in UB.
                if (capacity > 0 && (tiling->inverseWaveCapacity > 0 || fitsAllActive)) { break; }
            }
            if (trialRows == outerRowsPerTask) { break; }
        }
        if (inner <= 4 || crossConfigured) {
        context->SetBlockDim(usedCoreNum);
        // Preserve the complete original plan for runtime concentrated-index
        // fallback. The four mode-specific CSR fields carry the tiled plan.
        bool tiledAxis = false;
        if (inner == 1 && dtype != ge::DT_INT8 &&
            destinationAxis >= 4096 && destinationAxis < 65535 && destinationAxis % (32 / dtypeBytes) == 0 &&
            indexCount >= 512 && indexVectorElements <= 16320 && tiling->inverseWaveCapacity < 4) {
            uint64_t l2Bytes = 0;
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
            const bool smallWorkingSet = l2Bytes > 0 && (2 * selfElements + sourceElements) * dtypeBytes <= l2Bytes / 4;
            const uint64_t lanes = smallWorkingSet ? 512 : 2048;
            const uint64_t tiles = CeilDiv(destinationAxis, lanes);
            const uint64_t batch = smallWorkingSet ? 4 : 2;
            const uint64_t pitch = AlignUp(indexCount * dtypeBytes, 32) + 32;
            const uint64_t sourceBytes = std::max<uint64_t>(2 * indexBufferBytes, batch * pitch);
            const uint64_t destinationBytes = batch * lanes * dtypeBytes;
            const uint64_t gatherBytes = std::max<uint64_t>(destinationBytes, indexVectorElements);
            const uint64_t convertBytes = dtype == ge::DT_BF16 ? 2 * batch * lanes * 4 : 0;
            const uint64_t fixed = indexBufferBytes + sourceBytes + destinationBytes + gatherBytes + convertBytes + lanes * 4 + kUbSafetyReserve;
            const uint64_t capacity = fixed < ubBytes ? std::min<uint64_t>(16, (ubBytes - fixed) / (lanes * 4)) : 0;
            const uint64_t probeBytes = 2048 + AlignUp(CeilDiv(destinationAxis, 32) * 4, 32);
            const uint64_t waveScratch = 2 * matchMaskBufferBytes + indexVectorElements / 2;
            const uint64_t gatherScratch = indexVectorElements;
            if (tiles <= usedCoreNum && capacity >= 2 && capacity * lanes * 4 >= waveScratch &&
                gatherBytes >= gatherScratch && probeBytes + 1024 <= kUbSafetyReserve) {
                tiledAxis = true;
                tiling->csrHeadBytes = lanes;
                tiling->csrPrefixBytes = capacity;
                tiling->csrBatchSize = batch;
                tiling->csrSourceSlotBytes = sourceBytes;
            }
        }
        bool cooperative=false;
        uint64_t sharedBytes=0;
        if(!tiledAxis && inner==1 && tiling->inverseWaveCapacity>=2 && indexCount>=1024 && usedCoreNum>=8 &&
            tiling->inverseAlignedDestinations>=usedCoreNum && indexVectorElements>=8*usedCoreNum &&
            (tiling->mode!=INDEX_ADD_ADAPTIVE_AXIS_OWNER || indexCount<=destinationAxis*tiling->inverseWaveCapacity/2)) {
            const uint64_t lanes=tiling->inverseAlignedDestinations;
            const uint64_t waveBytes=lanes*tiling->inverseWaveCapacity*4;
            const uint64_t auxBytes=indexVectorElements+2*matchMaskBufferBytes;
            const uint64_t columns=CeilDiv(lanes/8,usedCoreNum)*8;
            const uint64_t localBytes=(columns*(1+tiling->inverseWaveCapacity)+8)*4;
            if(waveBytes>=2*indexBufferBytes &&
                (waveBytes>=2*indexBufferBytes+auxBytes || tiling->accumulatorBufferBytes>=auxBytes) &&
                tiling->sourceCacheBufferBytes>=localBytes) {
                cooperative=true;
                sharedBytes=AlignUp(waveBytes,512)+usedCoreNum*indexVectorElements*4+usedCoreNum*32;
            }
        }
        // Stable links remove the wave-capacity cliff; value order is unchanged.
        // Use links when the existing vector table cannot cover even the mean depth.
        // Keep cooperative and tiled waves when their UB capacity is sufficient.
        const uint64_t existingDepth=tiledAxis?tiling->csrPrefixBytes:tiling->inverseWaveCapacity;
        const bool waveCapacityCliff=existingDepth<=CeilDiv(indexCount,destinationAxis);
        bool linked=false;
        if(waveCapacityCliff&&inner==1&&outer>=128&&destinationAxis>=512&&destinationAxis<65535&&indexCount>=512&&indexCount<32768&&
            2048+AlignUp(CeilDiv(destinationAxis,32)*4,32)+1024<=kUbSafetyReserve){
            const uint64_t pitch=dtype==ge::DT_INT8?AlignUp(indexCount+1,128):AlignUp((indexCount+1)*dtypeBytes,32);
            const uint64_t fixed=AlignUp(indexCount+1,64)*4+pitch*(dtype==ge::DT_INT8?3:1)+kUbSafetyReserve;
            const uint64_t perLane=12+dtypeBytes+4+(dtype==ge::DT_BF16?8:dtype==ge::DT_INT8?4:0);
            const uint64_t lanes=fixed<ubBytes?std::min<uint64_t>(AlignUp(destinationAxis,128),(ubBytes-fixed)/perLane/128*128):0;
            if(lanes>=512){
                linked=true;cooperative=false;sharedBytes=0;
                tiling->csrHeadBytes=lanes;tiling->csrBatchSize=outerRowsPerTask;tiling->csrSourceSlotBytes=pitch;
                tiling->usedCoreNum=std::min<uint64_t>(availableCores,taskCount);
                context->SetBlockDim(tiling->usedCoreNum);
            }
        }
        context->SetTilingKey(linked ? INDEX_ADD_LINKED_AXIS : cooperative ? INDEX_ADD_COOPERATIVE_AXIS : tiledAxis ? INDEX_ADD_TILED_AXIS : tiling->mode);
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace != nullptr) {
            workspace[0] = sharedBytes ? sharedBytes + platform.GetLibApiWorkSpaceSize() : 0;
        }
        return ge::GRAPH_SUCCESS;
        }
        *tiling = uncachedPlan;
    }

    // Source-owned updates are useful only when each destination update is a DMA span.
    uint64_t atomicL2=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2,atomicL2);
    if ((dtype==ge::DT_INT8 || dtype==ge::DT_INT32) && inner>=32 &&
        inner%(32/dtypeBytes)==0 && indexCount>=512 && indexCount<=16320 && sourceElements>=65536 && indexCount>=32*destinationAxis &&
        inner*dtypeBytes>=512 && (dtype!=ge::DT_INT8 || (inner*dtypeBytes)%256==0) &&
        atomicL2>0 && selfElements*dtypeBytes<=atomicL2/8) {
        const uint64_t free=ubBytes>indexBufferBytes+kUbSafetyReserve?ubBytes-indexBufferBytes-kUbSafetyReserve:0;
        const uint64_t slot=std::min<uint64_t>(65536,free/2/32*32);
        if(slot>=256){
            const uint64_t width=std::min<uint64_t>(inner,slot/dtypeBytes);
            const uint64_t batch=std::min<uint64_t>(32,slot/(width*dtypeBytes));
            tiling->mode=INDEX_ADD_ATOMIC_SOURCE;tiling->tileInner=width;
            tiling->innerTileCount=CeilDiv(inner,width);tiling->csrBatchSize=batch;tiling->csrSourceSlotBytes=slot;
            tiling->indexBufferBytes=indexBufferBytes;tiling->usedCoreNum=availableCores;
            context->SetBlockDim(availableCores);context->SetTilingKey(INDEX_ADD_ATOMIC_SOURCE);
            size_t* ws=context->GetWorkspaceSizes(1);if(ws)ws[0]=platform.GetLibApiWorkSpaceSize();
            return ge::GRAPH_SUCCESS;
        }
    }

    // Vector-filter the index once for each core's output range. Only the
    // selected updates are materialized as stable per-destination lists.
    if (inner >= 5 && indexCount >= 64 && indexVectorElements <= 16320 &&
        destinationAxis < 65535 && (selfElements >= 32768 || sourceElements >= 32768)) {
        const uint64_t blocks = CeilDiv(selfElements * dtypeBytes, 32);
        const uint64_t cores = std::min<uint64_t>(availableCores, blocks);
        const uint64_t rows = std::min<uint64_t>(destinationAxis,
            CeilDiv(CeilDiv(blocks, cores) * (32 / dtypeBytes), inner) + 1);
        // Each core owns a finite GM32B interval; no tile can exceed its span.
        const uint64_t coreInner = std::min<uint64_t>(inner,
            CeilDiv(blocks, cores) * (32 / dtypeBytes));
        const uint64_t headBytes = AlignUp(rows * 4, 32);
        const uint64_t expandedMaskBytes = indexVectorElements / 2;
        const uint64_t fixedBytes = 3 * indexBufferBytes + headBytes +
            2 * expandedMaskBytes + 2 * matchMaskBufferBytes + kUbSafetyReserve;
        if (fixedBytes < ubBytes) {
            const uint64_t accBytes = dtype == ge::DT_BF16 ? 4 : dtype == ge::DT_INT8 ? 2 : dtypeBytes;
            const uint64_t alignment = dtype == ge::DT_INT8 ? 128 : std::max<uint64_t>(32 / dtypeBytes, 32 / accBytes);
            uint64_t batch = indexCount > 8 * destinationAxis ? 16 : 8;
            uint64_t tile = 0;
            for (;;) {
                const uint64_t perElement = 2 * batch * std::max<uint64_t>(dtypeBytes, 2) + accBytes + dtypeBytes +
                    (dtype == ge::DT_BF16 ? 4 * batch : dtype == ge::DT_INT8 ? 2 * batch : 0);
                tile = ((ubBytes - fixedBytes) / perElement / alignment) * alignment;
                if (tile >= AlignUp(coreInner, alignment) || batch == 1) { break; }
                batch /= 2;
            }
            tile = std::min<uint64_t>(tile, AlignUp(coreInner, alignment));
            if (tile >= alignment) {
                tiling->mode = INDEX_ADD_FILTERED_OWNER;
                tiling->tileInner = tile; tiling->tileInnerAligned = tile;
                tiling->indexBufferBytes = indexBufferBytes;
                tiling->indexVectorElements = indexVectorElements;
                tiling->matchMaskBufferBytes = matchMaskBufferBytes;
                tiling->csrHeadBytes = headBytes;
                tiling->csrPrefixBytes = expandedMaskBytes; // mode 9: expanded mask/offset bytes
                tiling->csrBatchSize = batch;
                tiling->csrSourceSlotBytes = batch * tile * std::max<uint64_t>(dtypeBytes, 2);
                tiling->castBufferBytes = dtype == ge::DT_BF16 ? 4 * batch * tile : dtype == ge::DT_INT8 ? 2 * batch * tile : 0;
                tiling->accumulatorBufferBytes = tile * accBytes;
                tiling->rawBufferBytes = tile * dtypeBytes;
                // Mode 9 only: one-pass input streams can avoid displacing L2 data.
                // This field is otherwise unused by the filtered kernel.
                tiling->rowPresenceBufferBytes = atomicL2 > 0 && tile * dtypeBytes >= 512 &&
                    (2 * selfElements + sourceElements) * dtypeBytes >= atomicL2 / 2 ? 1 : 0;
                tiling->usedCoreNum = cores;
                context->SetBlockDim(cores);
                context->SetTilingKey(INDEX_ADD_FILTERED_OWNER);
                size_t* workspace = context->GetWorkspaceSizes(1);
                if (workspace != nullptr) { workspace[0] = 0; }
                return ge::GRAPH_SUCCESS;
            }
        }
    }

    uint64_t alignmentElements = std::max<uint64_t>(kDmaBlockBytes / dtypeBytes,
                                                     kDmaBlockBytes / kAccumulatorBytes);
    // The INT8 path widens through an INT8 -> FP16 -> INT32 Cast.  On Atlas
    // A2 the Cast instruction is configured in 128-element repeats (the
    // same width used by the FP16 bridge in the Greater kernel).  A 64-element
    // tile is legal for GatherMask but is not a legal Cast repeat and causes
    // an AIV/MTE exception for narrow inner dimensions.  Keep every INT8
    // tile at the Cast width; GatherMask still handles the resulting two
    // repeats without a partial-read branch.
    if (dtype == ge::DT_INT8) {
        alignmentElements = std::max<uint64_t>(alignmentElements, 128);
    }
    const uint64_t castBytesPerElement = CastBytesPerElement(dtype);

    uint32_t mode = INDEX_ADD_SLAB_OWNER;
    uint64_t accumulatorRows = destinationAxis;
    uint64_t bytesPerInner = destinationAxis * kAccumulatorBytes + dtypeBytes + castBytesPerElement;
    uint64_t maximumAlignedTile = bytesPerInner == 0 ? 0 : usableBytes / bytesPerInner;
    maximumAlignedTile = maximumAlignedTile / alignmentElements * alignmentElements;

    const uint64_t initialSlabTile = std::min(inner, maximumAlignedTile);
    const uint64_t initialSlabInnerTiles = initialSlabTile == 0
        ? std::numeric_limits<uint64_t>::max()
        : CeilDiv(inner, initialSlabTile);
    // Keeping every destination row resident is legal for some large-M
    // shapes only by reducing the inner tile to one vector block.  That plan
    // replays every source update once per tiny inner tile.  When the padded
    // index is longer than the useful inner span, ROW_OWNER would instead
    // multiply the long index comparison by every destination row.  Grouped
    // destination slabs avoid both costs: a wide inner tile is processed
    // once, while a small deterministic set of destination rows shares one
    // scalar index scan.  The thresholds describe fragmentation and available
    // parallelism and are independent of a physical 910B revision.
    const bool fragmentedLargeIndexPrefersDestinationSlabs =
        maximumAlignedTile >= alignmentElements &&
        initialSlabInnerTiles >= 16 && indexVectorElements > inner &&
        destinationAxis > static_cast<uint64_t>(availableCores) * 4;

    if (maximumAlignedTile < alignmentElements ||
        fragmentedLargeIndexPrefersDestinationSlabs) {
        // A large destination axis does not fit as one monolithic UB slab.
        // Tiling it by *groups of rows* keeps O(rows*inner) data movement but
        // changes index traversal from one scan per destination row to one
        // scan per destination slab. For D=10000 this cuts the scan count by
        // hundreds of times while preserving deterministic ownership.
        mode = INDEX_ADD_DESTINATION_SLAB_OWNER;
        // Prefer a wide inner tile and derive the number of resident
        // destination rows from it.  The vector kernel performs several
        // ordered casts/barriers per row; reloading a row through many narrow
        // inner tiles is substantially more expensive than scanning the same
        // index list over additional destination slabs.  This shape-derived
        // choice is independent of the physical 910B revision.
        const uint64_t fixedBytesPerElement = dtypeBytes + castBytesPerElement;
        const uint64_t minimumBytesPerElement =
            fixedBytesPerElement + kAccumulatorBytes;
        uint64_t widestAlignedTile = usableBytes / minimumBytesPerElement;
        widestAlignedTile =
            widestAlignedTile / alignmentElements * alignmentElements;
        if (widestAlignedTile < alignmentElements) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t fullInnerAligned =
            inner > std::numeric_limits<uint64_t>::max() -
                        (alignmentElements - 1)
                ? widestAlignedTile
                : AlignUp(inner, alignmentElements);
        maximumAlignedTile = std::min(fullInnerAligned, widestAlignedTile);
        const uint64_t fixedTileBytes =
            maximumAlignedTile * fixedBytesPerElement;
        if (usableBytes <= fixedTileBytes) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t maximumRows = std::max<uint64_t>(
            1, (usableBytes - fixedTileBytes) /
                   (maximumAlignedTile * kAccumulatorBytes));
        accumulatorRows = std::min(destinationAxis, maximumRows);
        bytesPerInner =
            accumulatorRows * kAccumulatorBytes + dtypeBytes + castBytesPerElement;

        // If the wide tile would underfill the device, split D into enough
        // slabs to expose all AIVs.  Reducing the row count only frees UB, so
        // the selected inner width remains valid.
        const uint64_t initialTile = std::min(inner, maximumAlignedTile);
        const uint64_t initialTasks = outer * CeilDiv(destinationAxis, accumulatorRows) *
                                      CeilDiv(inner, initialTile);
        if (initialTasks < availableCores) {
            const uint64_t targetSlabs = std::min<uint64_t>(
                destinationAxis, CeilDiv(availableCores, outer));
            accumulatorRows = std::min<uint64_t>(
                accumulatorRows, CeilDiv(destinationAxis, targetSlabs));
            bytesPerInner = accumulatorRows * kAccumulatorBytes +
                            dtypeBytes + castBytesPerElement;
        }
    }
    if (mode == INDEX_ADD_SLAB_OWNER && destinationAxis > 1) {
        const uint64_t slabTile = std::min(inner, maximumAlignedTile);
        const uint64_t slabTasks = outer * CeilDiv(inner, slabTile);
        const uint64_t slabInnerTiles = CeilDiv(inner, slabTile);
        // A full-destination slab can make the inner tile extremely narrow:
        // every narrow tile then reloads and stores every destination row.
        // ROW_OWNER trades one vector index comparison per row for a single
        // wide, contiguous inner transfer.  Use that trade only when the
        // slab is materially fragmented, the padded index scan is no larger
        // than the useful inner span, and there are enough independent rows
        // to occupy the available AIVs.  This is a shape-derived cost rule;
        // it does not depend on a physical 910B revision or a published case.
        const bool fragmentedSlabPrefersRows =
            slabInnerTiles >= 4 && indexVectorElements <= inner &&
            outer * destinationAxis >= availableCores;
        const bool smallAxisNeedsParallelRows =
            destinationAxis <= static_cast<uint64_t>(availableCores) * 4 &&
            slabTasks < availableCores &&
            slabTasks * destinationAxis > slabTasks;
        // A tiny slab count leaves most AIVs idle (the public [32,128], M=120
        // INT8 case otherwise launches exactly one useful task).  Destination
        // row ownership preserves determinism and exposes D-way parallelism.
        if (fragmentedSlabPrefersRows || smallAxisNeedsParallelRows) {
            mode = INDEX_ADD_ROW_OWNER;
            accumulatorRows = 1;
            bytesPerInner = kAccumulatorBytes + dtypeBytes + castBytesPerElement;
            maximumAlignedTile = usableBytes / bytesPerInner;
            maximumAlignedTile = maximumAlignedTile / alignmentElements * alignmentElements;
        }
    }
    if (maximumAlignedTile < alignmentElements) {
        OP_LOGE(context, "IndexAdd cannot allocate a minimum aligned UB tile.");
        return ge::GRAPH_FAILED;
    }

    // A sparse row-owner would otherwise compare the complete index vector
    // once for every destination row, which is O(D*M) even though at most M
    // rows can receive an update.  Reserve one bit per destination inside the
    // UB safety budget.  The kernel builds that exact presence map once per
    // AIV, copies absent rows directly, and materializes a packed equality
    // mask only for rows that are actually present in index.
    uint64_t rowPresenceBufferBytes = 0;
    if (mode == INDEX_ADD_ROW_OWNER && dtype != ge::DT_INT8 &&
        indexVectorElements <= 512 && destinationAxis > indexCount * 4) {
        const uint64_t candidatePresenceBytes =
            AlignUp(CeilDiv(destinationAxis, 8), kDmaBlockBytes);
        if (candidatePresenceBytes <= kSparseRowPresenceMaxBytes) {
            mode = INDEX_ADD_SPARSE_ROW_OWNER;
            rowPresenceBufferBytes = candidatePresenceBytes;
        }
    }

    const uint64_t tileInner = std::min(inner, maximumAlignedTile);
    const uint64_t tileInnerAligned = AlignUp(tileInner, alignmentElements);
    const uint64_t innerTileCount = CeilDiv(inner, tileInner);
    uint64_t taskCount = outer * innerTileCount;
    if (mode == INDEX_ADD_ROW_OWNER || mode == INDEX_ADD_SPARSE_ROW_OWNER) {
        taskCount *= destinationAxis;
    } else if (mode == INDEX_ADD_DESTINATION_SLAB_OWNER) {
        taskCount *= CeilDiv(destinationAxis, accumulatorRows);
    }

    // The ordered list has O(M+D) scalar preprocessing per core. Use it
    // when the old slab under-occupies AIVs, or repeated packed-mask scans
    // across outer rows amortize that preparation. Sparse row copy and
    // destination-slab paths already avoid this cost and keep their plan.
    const bool csrHelpsSlabs = mode == INDEX_ADD_SLAB_OWNER && taskCount < availableCores;
    const bool csrHelpsRows = mode == INDEX_ADD_ROW_OWNER &&
        outer * destinationAxis * indexCount >
        24ULL * availableCores * (indexCount + destinationAxis);

    // Stable destination lists remove the old repeated full-index scans.
    // Packed prefixes are valid only within the proved 16-bit bound below.
    // If the index alone consumes a quarter of UB, full-index filtering leaves
    // little payload space. The compact ordered representation can still fit.
    const bool csrHelpsLong = indexBufferBytes >= ubBytes / 4;
    const uint64_t csrCountWeight = dtype == ge::DT_BF16 ? 4 : dtype == ge::DT_INT8 ? 2 : 1;
    // Prefix sum uses 16 bits; each per-destination count uses 15 bits.
    if ((csrHelpsSlabs || csrHelpsRows || csrHelpsLong) && inner >= 16 && indexCount >= 64 && indexCount < 32768 &&
        csrCountWeight * indexCount + 2 * destinationAxis < 65536 &&
        (sourceElements >= 32768 || selfElements >= 32768)) {
        const uint64_t headBytes = AlignUp(destinationAxis * 4, 32);
        const uint64_t prefixBytes = AlignUp((destinationAxis + 1) * 4, 32);
        const uint64_t fixedBytes = indexBufferBytes + headBytes + prefixBytes + kUbSafetyReserve;
        if (fixedBytes < ubBytes) {
            const uint64_t accBytes = dtype == ge::DT_BF16 ? 4 : dtype == ge::DT_INT8 ? 2 : dtypeBytes;
            const uint64_t alignment = dtype == ge::DT_INT8 ? 128 : std::max<uint64_t>(32 / dtypeBytes, 32 / accBytes);
            uint64_t batch = indexCount > 8 * destinationAxis ? 16 : indexCount > 2 * destinationAxis ? 8 : 4;
            uint64_t tile = 0;
            for (;;) {
                const uint64_t perElement = 2 * batch * std::max<uint64_t>(dtypeBytes, 2) + accBytes + dtypeBytes +
                    (dtype == ge::DT_BF16 ? 4 * batch : dtype == ge::DT_INT8 ? 2 * batch : 0);
                tile = ((ubBytes - fixedBytes) / perElement / alignment) * alignment;
                if (tile >= AlignUp(inner, alignment) || batch == 1) { break; }
                batch /= 2;
            }
            tile = std::min<uint64_t>(tile, AlignUp(inner, alignment));
            if (tile >= alignment) {
                tiling->mode = INDEX_ADD_ORDERED_CSR;
                tiling->tileInner = tile;
                tiling->tileInnerAligned = tile;
                tiling->indexBufferBytes = indexBufferBytes;
                tiling->csrHeadBytes = headBytes;
                tiling->csrPrefixBytes = prefixBytes;
                tiling->csrBatchSize = batch;
                tiling->csrSourceSlotBytes = batch * tile * std::max<uint64_t>(dtypeBytes, 2);
                tiling->castBufferBytes = dtype == ge::DT_BF16 ? 4 * batch * tile : dtype == ge::DT_INT8 ? 2 * batch * tile : 0;
                tiling->accumulatorBufferBytes = tile * accBytes;
                tiling->rawBufferBytes = tile * dtypeBytes;
                tiling->rowPresenceBufferBytes = atomicL2 > 0 && tile * dtypeBytes >= 512 &&
                    (2 * selfElements + sourceElements) * dtypeBytes >= atomicL2 / 2 ? 1 : 0;
                tiling->usedCoreNum = std::min<uint64_t>(availableCores, CeilDiv(selfElements * dtypeBytes, 32));
                context->SetBlockDim(tiling->usedCoreNum);
                context->SetTilingKey(INDEX_ADD_ORDERED_CSR);
                size_t* workspace = context->GetWorkspaceSizes(1);
                if (workspace != nullptr) { workspace[0] = 0; }
                return ge::GRAPH_SUCCESS;
            }
        }
    }

    uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(availableCores, taskCount));
    uint64_t tasksPerCore = taskCount / usedCoreNum;
    uint32_t extraCoreTasks = static_cast<uint32_t>(taskCount % usedCoreNum);

    tiling->tileInner = tileInner;
    tiling->tileInnerAligned = tileInnerAligned;
    tiling->innerTileCount = innerTileCount;
    tiling->taskCount = taskCount;
    tiling->tasksPerCore = tasksPerCore;
    tiling->extraCoreTasks = extraCoreTasks;
    tiling->usedCoreNum = usedCoreNum;
    tiling->mode = mode;
    tiling->accumulatorRows = static_cast<uint32_t>(accumulatorRows);
    tiling->indexBufferBytes = static_cast<uint32_t>(indexBufferBytes);
    tiling->indexVectorElements = static_cast<uint32_t>(indexVectorElements);
    tiling->matchMaskBufferBytes = static_cast<uint32_t>(matchMaskBufferBytes);
    tiling->rawBufferBytes = static_cast<uint32_t>(tileInnerAligned * dtypeBytes);
    tiling->castBufferBytes = static_cast<uint32_t>(tileInnerAligned * castBytesPerElement);
    tiling->accumulatorBufferBytes =
        static_cast<uint32_t>(accumulatorRows * tileInnerAligned * kAccumulatorBytes);
    tiling->sourceCacheBufferBytes = 0;
    tiling->rowPresenceBufferBytes =
        static_cast<uint32_t>(rowPresenceBufferBytes);
    // A long sequence of four-byte INT8 updates otherwise performs one tiny
    // GM transfer plus several padded vector operations for every match.
    // When the complete source fits comfortably beside the other UB buffers,
    // cache it once per active core and let the row-owner kernel accumulate
    // the four useful lanes on the scalar pipeline.
    if (mode == INDEX_ADD_ROW_OWNER && dtype == ge::DT_INT8 && outer == 1 &&
        inner <= 4 && indexCount >= 4096) {
        const uint64_t sourceCacheBytes = AlignUp(sourceElements * dtypeBytes, kDmaBlockBytes);
        const uint64_t allocatedBytes =
            indexBufferBytes + matchMaskBufferBytes + tiling->rawBufferBytes +
            tiling->castBufferBytes + tiling->accumulatorBufferBytes;
        if (sourceCacheBytes <= std::numeric_limits<uint32_t>::max() &&
            allocatedBytes + sourceCacheBytes + kUbSafetyReserve <= ubBytes) {
            tiling->sourceCacheBufferBytes = static_cast<uint32_t>(sourceCacheBytes);

            // Every active AIV loads the same index and source caches.  When
            // taskCount is not a multiple of the available core count, some
            // cores already execute two (or more) rows.  Compact the launch
            // to the smallest core count that preserves that exact maximum
            // rows-per-core bound; this removes redundant cache fills without
            // lengthening the critical task chain on any 910B-family device.
            const uint64_t maximumTasksPerCore = CeilDiv(taskCount, usedCoreNum);
            usedCoreNum = static_cast<uint32_t>(
                CeilDiv(taskCount, maximumTasksPerCore));
            tasksPerCore = taskCount / usedCoreNum;
            extraCoreTasks = static_cast<uint32_t>(taskCount % usedCoreNum);
            tiling->usedCoreNum = usedCoreNum;
            tiling->tasksPerCore = tasksPerCore;
            tiling->extraCoreTasks = extraCoreTasks;
        }
    }

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(mode);
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus IndexAddTiling(gert::TilingContext* context)
{
    const auto rc = IndexAddTilingBase(context);
    if (rc != ge::GRAPH_SUCCESS) return rc;
    auto* t = context->GetTilingData<IndexAddTilingData>();
    if(t->selfElements==0)return ge::GRAPH_SUCCESS;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    const uint64_t cores = platform.GetCoreNumAiv();
    uint32_t selectedFast=0;
    // Four-byte row starts allow an exact half-DMA-block boundary repair.
    // INT16 index conversion through FP16 is exact for this destination range.
    const uint64_t indexBytes = AlignUp(t->indexCount * 2, 32);
    const uint64_t reserved = indexBytes + 32 + 1024;
    const uint64_t chunk = ub > reserved ? (ub - reserved) / 8 / 128 * 128 : 0;
    if (dtype == ge::DT_INT32 && t->destinationAxis <= 2048 &&
        t->indexCount > 0 && t->selfElements > 0 &&
        t->selfElements <= UINT32_MAX && t->sourceElements <= UINT32_MAX &&
        t->inner % 4 == 0 && t->selfElements % 8 == 0 && t->sourceElements % 8 == 0 &&
        chunk >= 128 && t->inner >= (chunk + 13) / 14 &&
        2 * AlignUp(t->indexCount, 8) <= chunk && t->sourceElements >= cores * chunk) {
        t->tileInner = chunk;
        t->indexBufferBytes = indexBytes;
        t->usedCoreNum = std::min<uint64_t>(cores, CeilDiv(t->sourceElements, 128));
        context->SetBlockDim(t->usedCoreNum);
        context->SetTilingKey(INDEX_ADD_STREAMING_INT32);
        auto* ws = context->GetWorkspaceSizes(1);
        if (ws) ws[0] = platform.GetLibApiWorkSpaceSize() + 32;
    }
    // Small INT8 inner rows: widen a whole source slab once and reuse it.
    if (dtype == ge::DT_INT8 && t->inner > 0 && t->inner % 128 == 0 &&
        t->indexCount > 0 && t->indexCount <= 16320 && t->destinationAxis > 0) {
        const uint64_t n = AlignUp(t->indexCount, 128);
        const uint64_t slab = t->indexCount * t->inner;
        const uint64_t used = std::min<uint64_t>(cores, std::max<uint64_t>(1, CeilDiv(t->selfElements, 512)));
        const uint64_t rows = CeilDiv(t->outer * t->destinationAxis, used);
        const uint64_t raw = std::max(slab, rows * t->inner);
        const uint64_t bridge = std::max(raw * 2, n * 16);
        const uint64_t total = n * 4 + raw + bridge + slab * 2 + rows * t->inner * 2;
        if (total + kUbSafetyReserve <= ub && t->selfElements <= UINT32_MAX && slab <= UINT32_MAX) {
            t->usedCoreNum = used;
            t->indexVectorElements = n;
            t->indexBufferBytes = n * 4;
            t->rawBufferBytes = raw;
            t->castBufferBytes = bridge;
            t->sourceCacheBufferBytes = slab * 2;
            t->accumulatorBufferBytes = rows * t->inner * 2;
            context->SetBlockDim(used);
            context->SetTilingKey(INDEX_ADD_CACHED_INT8_ROWS);
            auto* ws = context->GetWorkspaceSizes(1); if (ws) ws[0] = 0;
        }
    }
    // A 128-lane resident BF16 tile keeps each ordered rounding step in UB.
    // Index filtering uses one 512-element vector tile; wider lists use CSR.
    if (dtype == ge::DT_BF16 && t->inner > 0 && t->inner <= 128 &&
        t->indexCount > 0 && t->indexCount <= 512 && t->destinationAxis < (1u << 24)) {
        uint64_t a = t->inner, b = 16;
        while (b) { const uint64_t c = a % b; a = b; b = c; }
        const uint64_t group = 16 / a;
        const uint64_t groups = CeilDiv(t->outer * t->destinationAxis, group);
        const uint64_t used = std::min(cores, groups);
        const uint64_t rows = CeilDiv(groups, used) * group;
        const uint64_t bytes = 2*128*256 + 16384 + 32768 + 2048*2 + 4096 + 2304 + 2048 + 1024 + 512 + 32;
        if (rows <= 64 && bytes + kUbSafetyReserve <= ub) {
            t->crossOuterRows = group;
            t->usedCoreNum = used;
            context->SetBlockDim(used);
            context->SetTilingKey(INDEX_ADD_RESIDENT_BF16);
            auto* ws = context->GetWorkspaceSizes(1); if (ws) ws[0] = 0;
        }
    }
    // FP16 16x16 transpose tiles turn scalar gathers into 32-byte block gathers.
    // All row pitches, sorting domains and UB offsets follow the input extents.
    if (dtype == ge::DT_FLOAT16 && t->inner == 1 && t->indexCount > 0 && t->destinationAxis > 0) {
        const uint64_t m=t->indexCount,d=t->destinationAxis,l=AlignUp(d,16),mp=AlignUp(m,16);
        uint64_t n=64,k=32,ns=6,ks=5;
        while(n<m+l){n*=2;++ns;} while(k<l){k*=2;++ks;}
        const uint64_t slot=std::max<uint64_t>(32*(mp+l),16*n);
        const uint64_t so=AlignUp((m+l)*4,4096);
        const uint64_t sort=std::max({32*n,so+slot,32*m});
        const uint64_t prefix=AlignUp(l+1,32);
        const uint64_t trans=AlignUp(slot,4096),meta=AlignUp((prefix+3*k+2*m)*4,32);
        const uint64_t bytes=sort+slot+trans+64*l+meta+224+4*l;
        if(l<=2040 && mp<=4080 && n<=4096 && bytes+1024<=ub &&
           t->outer>=cores*16 && (d+1)*n<(1u<<23)) {
            t->plan[0]=l;t->plan[1]=mp;t->plan[2]=n;t->plan[3]=k;
            t->plan[4]=ns;t->plan[5]=ks;t->plan[6]=so;t->plan[7]=sort;
            t->plan[8]=slot;t->plan[9]=trans;t->plan[10]=meta;t->plan[11]=prefix;
            t->usedCoreNum=cores;context->SetBlockDim(cores);context->SetTilingKey(INDEX_ADD_TRANSPOSE_WAVES);selectedFast=30;
            auto* ws=context->GetWorkspaceSizes(1);if(ws)ws[0]=0;
        }
    }
    // Cooperative inverse waves for FP32 contiguous axes. Capacity is the
    // largest table that fits the device UB; excess multiplicity falls back.
    if (dtype == ge::DT_FLOAT && t->inner == 1 && t->indexCount >= 32 &&
        t->destinationAxis > 0 && t->outer >= cores*2 && t->indexCount<=16320) {
        const uint64_t d=t->destinationAxis,m=t->indexCount,l=AlignUp(d,8),n=AlignUp(m,64);
        const uint64_t waveCores=std::min<uint64_t>(cores,l/8);
        uint64_t k=32,ks=5;while(k<l){k*=2;++ks;}
        const uint64_t pitch=64+128*CeilDiv(AlignUp(m,8)>64?AlignUp(m,8)-64:0,128),source=2*pitch*4,slot=source+2*l*4,pack=2*slot;
        const uint64_t fl=n*4,pos=fl+n*4,off=pos+n*4,exp=off+n/2;
        const uint64_t low=exp+n/2,high=low+AlignUp(n/8,32),lc=high+AlignUp(n/8,32);
        const uint64_t cols=CeilDiv(l/8,waveCores)*8,lw=lc+AlignUp(cols*4,32);
        for(uint64_t cap=31;cap>0;--cap) {
            const uint64_t status=AlignUp(lw+cap*cols*4,32),keys=AlignUp(cap*l*4,4096);
            const uint64_t maxpack=AlignUp(2*(l+m)+8*cap,8),back=pack+maxpack*4,values=back+2*l*4;
            const uint64_t counts=values+std::max<uint64_t>(maxpack*4,9*l*4),bytes=counts+256;
            const uint64_t wsstatus=AlignUp(cap*l,8),wscount=wsstatus+waveCores*8,wslen=wscount+l;
            if(status+32>pack || keys+6*k*4>pack || bytes+1024>ub || maxpack*4<8192 || k>2048)continue;
            uint32_t* p=t->plan;
            p[0]=l;p[1]=n;p[2]=cap;p[3]=pitch;p[4]=source;p[5]=slot;p[6]=pack;p[7]=back;
            p[8]=values;p[9]=counts;p[10]=bytes;p[11]=k;p[12]=ks;p[13]=fl;p[14]=pos;p[15]=off;
            p[16]=exp;p[17]=low;p[18]=high;p[19]=lc;p[20]=lw;p[21]=status;p[22]=keys;
            p[23]=wsstatus;p[24]=wscount;p[25]=wslen;p[26]=maxpack;
            t->usedCoreNum=waveCores;t->tasksPerCore=t->taskCount/waveCores;t->extraCoreTasks=t->taskCount%waveCores;context->SetBlockDim(waveCores);context->SetTilingKey(INDEX_ADD_ORDERED_WAVES);selectedFast=34;
            auto* ws=context->GetWorkspaceSizes(1);if(ws)ws[0]=platform.GetLibApiWorkSpaceSize()+wslen*4;
            break;
        }
    }
    // Capacity-bucket entry points keep buffer addresses constant while
    // dimensions, values, indices and rounding order remain fully dynamic.
    if(ub>=196352 && selectedFast==30){
        const uint32_t n=t->plan[2],sk=AlignUp(t->plan[8],4096)/1024,dk=AlignUp(t->plan[0]*32,4096)/1024;
        if(n==512 && sk==8 && dk==4 && t->plan[7]<=16384 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=153632){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(200);}
        if(n==512 && sk==8 && dk==8 && t->plan[7]<=16384 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=144928){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(201);}
        if(n==512 && sk==12 && dk==4 && t->plan[7]<=16384 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=145440){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(202);}
        if(n==512 && sk==12 && dk==8 && t->plan[7]<=16384 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=136736){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(203);}
        if(n==512 && sk==12 && dk==12 && t->plan[7]<=16384 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=128032){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(204);}
        if(n==1024 && sk==16 && dk==4 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=120864){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(205);}
        if(n==1024 && sk==16 && dk==8 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=112160){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(206);}
        if(n==1024 && sk==16 && dk==12 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=103456){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(207);}
        if(n==1024 && sk==16 && dk==16 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=94752){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(208);}
        if(n==1024 && sk==20 && dk==4 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=112672){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(209);}
        if(n==1024 && sk==20 && dk==8 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=103968){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(210);}
        if(n==1024 && sk==20 && dk==12 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=95264){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(211);}
        if(n==1024 && sk==20 && dk==16 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=86560){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(212);}
        if(n==1024 && sk==24 && dk==4 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=104480){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(213);}
        if(n==1024 && sk==24 && dk==8 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=95776){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(214);}
        if(n==1024 && sk==24 && dk==12 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=87072){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(215);}
        if(n==1024 && sk==24 && dk==16 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=78368){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(216);}
        if(n==1024 && sk==28 && dk==4 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=96288){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(217);}
        if(n==1024 && sk==28 && dk==8 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=87584){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(218);}
        if(n==1024 && sk==28 && dk==12 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=78880){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(219);}
        if(n==1024 && sk==28 && dk==16 && t->plan[7]<=32768 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=70176){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(220);}
        if(n==2048 && sk==32 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=55328){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(221);}
        if(n==2048 && sk==32 && dk==8 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=46624){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(222);}
        if(n==2048 && sk==32 && dk==12 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=37920){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(223);}
        if(n==2048 && sk==32 && dk==16 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=29216){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(224);}
        if(n==2048 && sk==36 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=47136){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(225);}
        if(n==2048 && sk==36 && dk==8 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=38432){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(226);}
        if(n==2048 && sk==36 && dk==12 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=29728){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(227);}
        if(n==2048 && sk==36 && dk==16 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=21024){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(228);}
        if(n==2048 && sk==40 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=38944){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(229);}
        if(n==2048 && sk==40 && dk==8 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=30240){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(230);}
        if(n==2048 && sk==40 && dk==12 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=21536){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(231);}
        if(n==2048 && sk==40 && dk==16 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=12832){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(232);}
        if(n==2048 && sk==44 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=30752){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(233);}
        if(n==2048 && sk==44 && dk==8 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=22048){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(234);}
        if(n==2048 && sk==44 && dk==12 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=13344){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(235);}
        if(n==2048 && sk==44 && dk==16 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=4640){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(236);}
        if(n==2048 && sk==48 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=22560){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(237);}
        if(n==2048 && sk==48 && dk==8 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=13856){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(238);}
        if(n==2048 && sk==48 && dk==12 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*512+2*t->indexCount)*4,32)<=5152){t->plan[3]=512;t->plan[5]=9;context->SetTilingKey(239);}
        if(n==2048 && sk==52 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=14368){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(240);}
        if(n==2048 && sk==52 && dk==8 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*256+2*t->indexCount)*4,32)<=5664){t->plan[3]=256;t->plan[5]=8;context->SetTilingKey(241);}
        if(n==2048 && sk==56 && dk==4 && t->plan[7]<=65536 && AlignUp((t->plan[11]+3*128+2*t->indexCount)*4,32)<=6176){t->plan[3]=128;t->plan[5]=7;context->SetTilingKey(242);}
    }
    if(ub>=196352 && selectedFast==34){
        const uint64_t sp=CeilDiv(t->plan[4],8192),dp=CeilDiv(2*t->plan[0]*4,2048);
        if(sp>=1&&sp<=6&&dp>=1&&dp<=8){
            const uint64_t src=sp*8192,dst=dp*2048,slot=src+dst;
            const uint64_t packed=(195328-2*slot-dst-256)/2/512*512;
            if(t->plan[26]*4<=packed && 9*t->plan[0]*4<=packed)context->SetTilingKey(64+(sp-1)*8+dp-1);
        }
    }
    t->sourceL2BypassOffset = 0;
    using GetDeviceFn = int32_t(*)(int32_t*);
    using GetOffsetFn = int32_t(*)(uint32_t, uint64_t*);
    auto getDevice = reinterpret_cast<GetDeviceFn>(dlsym(RTLD_DEFAULT, "rtGetDevice"));
    auto getOffset = reinterpret_cast<GetOffsetFn>(dlsym(RTLD_DEFAULT, "rtGetL2CacheOffset"));
    int32_t device = -1; uint64_t offset = 0;
    if (getDevice && getOffset && getDevice(&device) == 0 && device >= 0 &&
        getOffset(uint32_t(device), &offset) == 0) t->sourceL2BypassOffset = offset;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> valueTypes = {
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8};
        const std::vector<ge::DataType> indexTypes(valueTypes.size(), ge::DT_INT32);
        const std::vector<ge::Format> formats(valueTypes.size(), ge::FORMAT_ND);

        this->Input("self").ParamType(REQUIRED).DataType(valueTypes).Format(formats);
        this->Input("index").ParamType(REQUIRED).DataType(indexTypes).Format(formats);
        this->Input("source").ParamType(REQUIRED).DataType(valueTypes).Format(formats);
        this->Output("out").ParamType(REQUIRED).DataType(valueTypes).Format(formats);
        this->Attr("dim").AttrType(REQUIRED).Int();

        this->AICore().SetTiling(optiling::IndexAddTiling).AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}  // namespace ops
