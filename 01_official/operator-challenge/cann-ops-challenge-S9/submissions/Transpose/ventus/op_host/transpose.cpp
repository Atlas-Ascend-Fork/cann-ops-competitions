/**
 * Transpose 算子注册、置换验证与主机侧分块规划。
 * 合并可连续处理的维度，根据内存布局和 UB 预算选择连续搬运或二维分块转置。
 */

#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {
constexpr uint64_t kCacheLineBytes = 128;
constexpr uint64_t kWideStoreBytes = 128UL * 1024UL;
constexpr uint64_t kHwTileB16 = 128;
constexpr uint64_t kMaxStoreGroup = 8;
constexpr uint64_t kUbTotalBytes = 192ULL * 1024ULL;
constexpr uint64_t kUbBlockBytes = 32;
constexpr uint64_t kRunTileElements = 8192;
constexpr uint64_t kLargeRunTileElements = 24576;
constexpr uint64_t kMaxTile = 64;
constexpr uint64_t kWholeTileElements = 32768;
constexpr uint64_t kWholeTileMaxB = 256;

uint32_t GetAivCoreCount(const gert::TilingContext* context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return std::max<uint32_t>(1U, platform.GetCoreNumAiv());
}

uint64_t DataTypeSize(ge::DataType type)
{
    switch (type) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

bool ReadPerm(const gert::RuntimeAttrs* attrs, size_t rank, int64_t perm[S9_TRANSPOSE_MAX_RANK])
{
    if (attrs == nullptr) {
        return false;
    }
    const auto* dims = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (dims == nullptr || dims->GetSize() != rank) {
        return false;
    }
    const auto* values = reinterpret_cast<const int64_t*>(dims->GetData());
    bool seen[S9_TRANSPOSE_MAX_RANK] = {};
    for (size_t i = 0; i < rank; ++i) {
        int64_t axis = values[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank) || seen[axis]) {
            return false;
        }
        seen[axis] = true;
        perm[i] = axis;
    }
    return true;
}

struct Group {
    uint64_t len;
    uint64_t inStride;
    uint64_t outStride;
};

// 移除长度为 1 的维度，并合并在输入和输出中保持相邻顺序的维度。
size_t Canonicalize(const gert::Shape& shape, const int64_t* perm, size_t rank, Group groups[S9_TRANSPOSE_MAX_GROUP])
{
    uint64_t stride[S9_TRANSPOSE_MAX_RANK];
    uint64_t running = 1;
    for (size_t i = rank; i > 0; --i) {
        stride[i - 1] = running;
        running *= static_cast<uint64_t>(shape.GetDim(i - 1));
    }

    int64_t remap[S9_TRANSPOSE_MAX_RANK];
    uint64_t keptLen[S9_TRANSPOSE_MAX_RANK];
    uint64_t keptStride[S9_TRANSPOSE_MAX_RANK];
    size_t kept = 0;
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) == 1) {
            remap[i] = -1;
            continue;
        }
        remap[i] = static_cast<int64_t>(kept);
        keptLen[kept] = static_cast<uint64_t>(shape.GetDim(i));
        keptStride[kept] = stride[i];
        ++kept;
    }

    size_t count = 0;
    int64_t previous = -2;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t axis = remap[perm[i]];
        if (axis < 0) {
            continue;
        }
        if (count > 0 && axis == previous + 1) {
            groups[count - 1].len *= keptLen[axis];
            groups[count - 1].inStride = keptStride[axis];
        } else {
            groups[count].len = keptLen[axis];
            groups[count].inStride = keptStride[axis];
            ++count;
        }
        previous = axis;
    }

    uint64_t outStride = 1;
    for (size_t i = count; i > 0; --i) {
        groups[i - 1].outStride = outStride;
        outStride *= groups[i - 1].len;
    }
    return count;
}
}

namespace optiling {
// 选择连续片段搬运或二维转置分块，并计算行距、写回组大小和核数。
static ge::graphStatus TransposeTiling(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<TransposeTilingData>();
    const auto* storage = context->GetInputShape(0);
    const auto* desc = context->GetInputDesc(0);
    if (tiling == nullptr || storage == nullptr || desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shape = storage->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    if (rank == 0 || rank > S9_TRANSPOSE_MAX_RANK) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t elementSize = DataTypeSize(desc->GetDataType());
    if (elementSize == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling->storeGroup = 1;
    tiling->wholeTile = 0;
    tiling->aFastShift = 0;
    tiling->bypassOutput = 0;
    tiling->fullRowStore = 0;

    int64_t perm[S9_TRANSPOSE_MAX_RANK];
    if (!ReadPerm(context->GetAttrs(), rank, perm)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t total = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) < 0) {
            return ge::GRAPH_FAILED;
        }
        total *= static_cast<uint64_t>(shape.GetDim(i));
    }
    const bool largeRaggedFloat = desc->GetDataType() == ge::DT_FLOAT &&
        total >= (1ULL << 18U);
    const uint64_t runTileElements =
        desc->GetDataType() == ge::DT_FLOAT ||
        desc->GetDataType() == ge::DT_INT8 ?
        kLargeRunTileElements : kRunTileElements;

    const uint64_t packRunElements = elementSize == 1 ? 20480ULL : 8192ULL;
    uint64_t effRunTile = runTileElements;
    bool wideRaggedFloat = false;

    Group groups[S9_TRANSPOSE_MAX_GROUP];
    size_t count = Canonicalize(shape, perm, rank, groups);
    if (count == 0) {

        groups[0] = Group{1, 1, 1};
        count = 1;
    }

    const uint32_t cores = GetAivCoreCount(context);
    const size_t last = count - 1;
    size_t outerIndices[S9_TRANSPOSE_MAX_GROUP];
    size_t outerUsed = 0;

    if (groups[last].inStride == 1) {
        tiling->regime = S9_TRANSPOSE_REGIME_RUN;

        tiling->runLen = groups[last].len;

        const bool runGrainOk =
            tiling->runLen * elementSize >= 32ULL;

        const bool bypassRunOutput = runGrainOk &&
            total * elementSize > 48ULL * 1024ULL * 1024ULL;
        tiling->bypassOutput = bypassRunOutput ? 1U : 0U;
        if (count >= 2) {
            tiling->midLen = groups[last - 1].len;
            tiling->midInStride = groups[last - 1].inStride;
            for (size_t i = 0; i + 2 < count; ++i) {
                outerIndices[outerUsed++] = i;
            }
        } else {
            tiling->midLen = 1;
            tiling->midInStride = tiling->runLen;
        }
        if ((tiling->runLen * elementSize) % kUbBlockBytes != 0 &&
            tiling->midLen > 1 && tiling->runLen <= packRunElements) {
            effRunTile = packRunElements;
        }
        tiling->runChunk = std::max<uint64_t>(
            1, std::min<uint64_t>(tiling->runLen, effRunTile));
    } else {
        tiling->regime = S9_TRANSPOSE_REGIME_TILE;

        tiling->bypassOutput = 0U;
        size_t contiguous = last;
        for (size_t i = 0; i < count; ++i) {
            if (groups[i].inStride == 1) {
                contiguous = i;
                break;
            }
        }
        if (contiguous == last) {
            return ge::GRAPH_FAILED;
        }
        tiling->lenA = groups[contiguous].len;
        tiling->outStrideA = groups[contiguous].outStride;
        tiling->lenB = groups[last].len;
        tiling->inStrideB = groups[last].inStride;
        for (size_t i = 0; i < count; ++i) {
            if (i != contiguous && i != last) {
                outerIndices[outerUsed++] = i;
            }
        }
        uint64_t normalizedOuter = 1U;
        for (size_t i = 0; i < outerUsed; ++i) {
            normalizedOuter *= groups[outerIndices[i]].len;
        }
        wideRaggedFloat = largeRaggedFloat && tiling->lenB % 16U != 0U &&
            (CeilDiv(tiling->lenB, 16U) >= 128U || normalizedOuter >= 32U);

        const uint64_t minTile = elementSize == 4 ? 16 :
            std::max<uint64_t>(kUbBlockBytes / elementSize, 8);

        if (desc->GetDataType() == ge::DT_INT8) {
            tiling->tileA = 256;
            tiling->tileB = 64;
        } else if (desc->GetDataType() == ge::DT_FLOAT) {
            tiling->tileA = wideRaggedFloat ? 32U :
                ((largeRaggedFloat && tiling->lenB % 16U != 0U) ? 128U : 256U);
            tiling->tileB = 16;
        } else {
            const uint64_t tileCap = (elementSize == 2) ? 128 : kMaxTile;
            tiling->tileA = tileCap;
            tiling->tileB = tileCap;
        }
        while (tiling->tileA > minTile && tiling->tileA / 2 >= tiling->lenA) {
            tiling->tileA /= 2;
        }
        while (tiling->tileB > minTile && tiling->tileB / 2 >= tiling->lenB) {
            tiling->tileB /= 2;
        }
    }

    tiling->outerCount = static_cast<uint32_t>(outerUsed);
    tiling->outerTotal = 1;
    for (size_t i = 0; i < S9_TRANSPOSE_MAX_GROUP; ++i) {
        if (i < outerUsed) {
            const Group& group = groups[outerIndices[i]];
            tiling->outerLen[i] = group.len;
            tiling->outerInStride[i] = group.inStride;
            tiling->outerOutStride[i] = group.outStride;
            tiling->outerTotal *= group.len;
        } else {
            tiling->outerLen[i] = 1;
            tiling->outerInStride[i] = 0;
            tiling->outerOutStride[i] = 0;
        }
    }

    if (tiling->regime == S9_TRANSPOSE_REGIME_RUN) {

        auto refresh = [&]() {
            const uint64_t rowBytes =
                ((tiling->runChunk * elementSize + kUbBlockBytes - 1) / kUbBlockBytes) * kUbBlockBytes;
            tiling->midBatch = std::max<uint64_t>(
                1, std::min<uint64_t>(
                    tiling->midLen, (effRunTile * elementSize) / rowBytes));
            tiling->runChunkCount = CeilDiv(tiling->runLen, tiling->runChunk);
            tiling->midChunkCount = CeilDiv(tiling->midLen, tiling->midBatch);
            tiling->taskCount = tiling->outerTotal * tiling->midChunkCount * tiling->runChunkCount;
        };
        refresh();

        while (tiling->taskCount < cores && tiling->midBatch > 1) {
            tiling->midBatch = (tiling->midBatch + 1) / 2;
            tiling->runChunkCount = CeilDiv(tiling->runLen, tiling->runChunk);
            tiling->midChunkCount = CeilDiv(tiling->midLen, tiling->midBatch);
            tiling->taskCount = tiling->outerTotal * tiling->midChunkCount * tiling->runChunkCount;
        }
        while (tiling->taskCount < cores && tiling->runChunk > 256) {
            tiling->runChunk = (tiling->runChunk + 1) / 2;
            tiling->runChunkCount = CeilDiv(tiling->runLen, tiling->runChunk);
            tiling->midChunkCount = CeilDiv(tiling->midLen, tiling->midBatch);
            tiling->taskCount = tiling->outerTotal * tiling->midChunkCount * tiling->runChunkCount;
        }
        constexpr uint64_t kRunPackUbBytes = 180ULL * 1024ULL;

        tiling->tileA = 1;
        {
            const uint64_t exactB = tiling->runChunk * elementSize;
            const bool packed = tiling->runChunkCount == 1 &&
                (exactB % kUbBlockBytes) != 0 && tiling->midBatch > 1 &&
                (exactB % 4 == 0 || elementSize >= 2);
            if (packed && tiling->midChunkCount == 1 && tiling->outerCount == 1 &&
                tiling->outerInStride[0] == tiling->runLen &&
                tiling->outerOutStride[0] == tiling->midLen * tiling->runLen) {

                uint64_t g = exactB;
                uint64_t b = kUbBlockBytes;
                while (b != 0) { const uint64_t t = g % b; g = b; b = t; }
                const uint64_t kAlign = kUbBlockBytes / g;
                const uint64_t packUnitB = (exactB % 4 == 0) ? 4 : elementSize;
                const uint64_t perRow = exactB / packUnitB;

                auto ubBytes = [&](uint64_t kk) {
                    const uint64_t pitch =
                        CeilDiv(exactB * kk, kUbBlockBytes) * kUbBlockBytes;
                    return 2 * tiling->midBatch * pitch +
                           2 * kk * tiling->midBatch * exactB +
                           kk * tiling->midBatch * perRow * 4;
                };
                uint64_t k = 1;
                for (uint64_t cand = kAlign; cand <= tiling->outerTotal;
                     cand += kAlign) {
                    if (ubBytes(cand) > kRunPackUbBytes) { break; }
                    if (CeilDiv(tiling->outerTotal, cand) *
                        tiling->midChunkCount * tiling->runChunkCount < cores) {
                        break;
                    }
                    k = cand;
                }
                const uint64_t tasks =
                    CeilDiv(tiling->outerTotal, k) *
                    tiling->midChunkCount * tiling->runChunkCount;
                if (k >= 2 && tasks >= cores && ubBytes(1) <= kRunPackUbBytes) {
                    tiling->tileA = k;
                    tiling->taskCount = tasks;
                }
            }
        }
    } else {
        const uint64_t minTile = elementSize == 4 ? 16 :
            std::max<uint64_t>(kUbBlockBytes / elementSize, 8);

        if (elementSize == 2 && outerUsed == 0 && tiling->lenA % 16 == 0 &&
            tiling->lenB % 16 == 0 && tiling->lenB <= kWholeTileMaxB &&
            tiling->lenA * tiling->lenB <= kWholeTileElements) {
            tiling->wholeTile = 1;
            tiling->tileA = tiling->lenA;
            tiling->tileB = tiling->lenB;
        }
        tiling->tileCountA = CeilDiv(tiling->lenA, tiling->tileA);
        tiling->tileCountB = CeilDiv(tiling->lenB, tiling->tileB);
        tiling->taskCount = tiling->outerTotal * tiling->tileCountA * tiling->tileCountB;
        while (tiling->wholeTile == 0 && tiling->taskCount < cores &&
               (tiling->tileA > minTile || tiling->tileB > minTile)) {
            if (tiling->tileA >= tiling->tileB && tiling->tileA > minTile) {
                tiling->tileA /= 2;
            } else if (tiling->tileB > minTile) {
                tiling->tileB /= 2;
            } else {
                break;
            }
            tiling->tileCountA = CeilDiv(tiling->lenA, tiling->tileA);
            tiling->tileCountB = CeilDiv(tiling->lenB, tiling->tileB);
            tiling->taskCount = tiling->outerTotal * tiling->tileCountA * tiling->tileCountB;
        }

        const bool raggedB = tiling->lenB % tiling->tileB != 0U;
        const uint32_t candidateGroup = desc->GetDataType() == ge::DT_INT8 ? 4U :
            desc->GetDataType() == ge::DT_FLOAT ?
                (wideRaggedFloat ? 64U :
                 (raggedB && largeRaggedFloat ? 16U : 8U)) :
            desc->GetDataType() == ge::DT_INT32 ? 2U :
            (elementSize == 2 ? 4U : 1U);
        const bool raggedFusion = desc->GetDataType() == ge::DT_FLOAT16 ||
            (largeRaggedFloat && desc->GetDataType() == ge::DT_FLOAT);
        const bool alignedFusion = tiling->lenB % tiling->tileB == 0 &&
            tiling->tileCountB % candidateGroup == 0;
        if (tiling->wholeTile == 0 && candidateGroup > 1 &&
            tiling->tileCountB >= 2 && (raggedFusion || alignedFusion)) {
            const uint64_t targetTasks = std::min<uint64_t>(cores, tiling->taskCount);
            uint32_t group = candidateGroup;
            while (group > 1) {
                const uint64_t groupTasks = tiling->outerTotal * tiling->tileCountA *
                    CeilDiv(tiling->tileCountB, group);
                if (groupTasks >= targetTasks) {
                    tiling->storeGroup = group;
                    tiling->taskCount = groupTasks;
                    if ((tiling->tileCountA & (tiling->tileCountA - 1U)) == 0U) {
                        uint64_t countA = tiling->tileCountA;
                        while (countA > 1U) {
                            ++tiling->aFastShift;
                            countA >>= 1U;
                        }
                    }
                    break;
                }
                group >>= 1U;
            }
        }

        const bool frsFp16 = elementSize == 2 && tiling->tileB == kHwTileB16 &&
            (tiling->lenB % 16ULL) != 0ULL;
        const bool frsFp32 = desc->GetDataType() == ge::DT_FLOAT &&
            tiling->tileB == 16ULL &&
            (tiling->lenB * elementSize) % kCacheLineBytes != 0ULL;

        const bool frsInt8 = desc->GetDataType() == ge::DT_INT8 &&
            tiling->tileB == 64ULL && (tiling->lenB % 4ULL) == 0ULL &&
            (tiling->lenB % kUbBlockBytes) != 0ULL;

        const uint64_t maxSlots = elementSize == 2 ? kMaxStoreGroup :
            (desc->GetDataType() == ge::DT_INT8 ? 16ULL : 64ULL);
        if (tiling->wholeTile == 0 && (frsFp16 || frsFp32 || frsInt8) &&
            tiling->tileCountB >= 2 && tiling->tileCountB <= maxSlots &&
            tiling->outStrideA == tiling->lenB) {
            const uint64_t gran =
                desc->GetDataType() == ge::DT_INT8 ? 64ULL : 16ULL;
            const uint64_t rowLen =
                ((tiling->lenB + gran - 1ULL) / gran) * gran;

            const uint64_t L = kUbBlockBytes / elementSize;
            uint64_t gg = L;
            uint64_t xx = tiling->lenB % L;
            while (xx != 0ULL) { const uint64_t t = gg % xx; gg = xx; xx = t; }
            const uint64_t packRows = L / gg;
            const uint64_t unitBytes =
                (elementSize == 1 ||
                 (tiling->lenB * elementSize) % 4ULL == 0ULL) ? 4ULL : 2ULL;
            const uint64_t perRow = (tiling->lenB * elementSize) / unitBytes;

            const uint64_t minOffsetBytes = packRows * perRow * sizeof(int32_t);
            const uint64_t offsetBytes = minOffsetBytes > 32ULL * 1024ULL ?
                minOffsetBytes : 32ULL * 1024ULL;
            const uint64_t perA =
                2ULL * tiling->tileB * elementSize + rowLen * elementSize +
                tiling->lenB * elementSize;
            uint64_t tileACap = (kUbTotalBytes - offsetBytes) / perA;

            const uint64_t tileAGran = kUbBlockBytes / elementSize > 16ULL ?
                kUbBlockBytes / elementSize : 16ULL;
            tileACap = (tileACap / tileAGran) * tileAGran;
            if (tileACap > tiling->tileA) {
                tileACap = tiling->tileA;
            }

            const uint64_t tileAFloor = elementSize == 2 ? 16ULL :
                (elementSize == 1 ? 64ULL : 48ULL);
            const uint64_t countA = tileACap >= tileAFloor ?
                CeilDiv(tiling->lenA, tileACap) : 0;
            if (tileACap >= tileAFloor && tiling->outerTotal * countA >= cores) {
                tiling->tileA = tileACap;
                tiling->tileCountA = countA;
                tiling->storeGroup = static_cast<uint32_t>(tiling->tileCountB);
                tiling->taskCount = tiling->outerTotal * countA;
                tiling->fullRowStore = 2;
                tiling->aFastShift = 0;
            }
        }
        if (tiling->fullRowStore == 0U &&
            tiling->wholeTile == 0 && elementSize == 2 &&
            tiling->tileB == kHwTileB16 && tiling->tileCountB >= 2 &&
            tiling->tileCountB <= kMaxStoreGroup &&

            tiling->outStrideA == tiling->lenB &&
            (tiling->lenB * elementSize) % kUbBlockBytes == 0 &&
            (tiling->lenB * elementSize) % kCacheLineBytes != 0) {

            uint64_t tileACap = kWideStoreBytes / (tiling->lenB * elementSize);
            tileACap = (tileACap / 16) * 16;
            if (tileACap > tiling->tileA) {
                tileACap = tiling->tileA;
            }
            const uint64_t countA = tileACap >= 16 ?
                CeilDiv(tiling->lenA, tileACap) : 0;

            if (tileACap >= 16 && tiling->outerTotal * countA >= cores) {
                tiling->tileA = tileACap;
                tiling->tileCountA = countA;
                tiling->storeGroup = static_cast<uint32_t>(tiling->tileCountB);
                tiling->taskCount = tiling->outerTotal * countA;
                tiling->fullRowStore = 1;

                tiling->aFastShift = 0;
            }
        }

        if (tiling->fullRowStore == 0U && elementSize == 4 &&
            desc->GetDataType() == ge::DT_INT32 &&
            tiling->tileCountB == 1 && tiling->storeGroup == 1 &&
            tiling->outStrideA == tiling->lenB &&
            tiling->lenB <= tiling->tileB) {
            tiling->fullRowStore = 1;
        }

        if (tiling->fullRowStore == 0U && elementSize == 4 &&
            desc->GetDataType() == ge::DT_INT32 &&
            tiling->tileCountB >= 2 && tiling->storeGroup == 1 &&
            tiling->outStrideA == tiling->lenB) {

            const uint64_t budget =
                (kUbTotalBytes - 16ULL * 1024ULL) / (4ULL * elementSize);
            uint64_t tileACap = tiling->lenB == 0 ? 0 : budget / tiling->lenB;
            tileACap = (tileACap / 8ULL) * 8ULL;
            if (tileACap > tiling->tileA) {
                tileACap = tiling->tileA;
            }
            const uint64_t countA = tileACap >= 16ULL ?
                CeilDiv(tiling->lenA, tileACap) : 0;
            if (tileACap >= 16ULL && tiling->outerTotal * countA >= cores) {
                tiling->tileA = tileACap;
                tiling->tileCountA = countA;
                tiling->storeGroup = static_cast<uint32_t>(tiling->tileCountB);
                tiling->taskCount = tiling->outerTotal * countA;
                tiling->fullRowStore = 1;
                tiling->aFastShift = 0;
            }
        }
    }
    if (total == 0) {
        tiling->taskCount = 0;
    }

    const uint64_t usefulTasks = std::max<uint64_t>(1, tiling->taskCount);
    tiling->blockDim = static_cast<uint32_t>(std::min<uint64_t>(cores, usefulTasks));

    context->SetBlockDim(tiling->blockDim);

    const bool useF32WideKey = tiling->fullRowStore == 0U &&
        tiling->regime == S9_TRANSPOSE_REGIME_TILE &&
        largeRaggedFloat && tiling->lenB % tiling->tileB != 0U &&
        tiling->tileB == 16U &&
        tiling->storeGroup >= 16U && tiling->tileA == 32U;
    const bool useF32BulkKey = tiling->fullRowStore == 0U &&
        tiling->regime == S9_TRANSPOSE_REGIME_TILE &&
        largeRaggedFloat && tiling->lenB % tiling->tileB != 0U &&
        tiling->storeGroup == 16U && tiling->tileA == 128U &&
        tiling->tileB == 16U;
    const uint64_t tilingKey = tiling->wholeTile != 0 ? 1 :
        (tiling->regime == S9_TRANSPOSE_REGIME_RUN ? 2 :
         (useF32WideKey ? 4 : (useF32BulkKey ? 3 : 0)));
    context->SetTilingKey(tilingKey);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus TransposeInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(0);
    gert::Shape* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t rank = input->GetDimNum();
    if (rank == 0 || rank > S9_TRANSPOSE_MAX_RANK) {
        return GRAPH_FAILED;
    }
    int64_t perm[S9_TRANSPOSE_MAX_RANK];
    if (!ReadPerm(context->GetAttrs(), rank, perm)) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        output->SetDim(i, input->GetDim(perm[i]));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus TransposeInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> types = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("inputs").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("output").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Attr("dims").AttrType(REQUIRED).ListInt();
        this->SetInferShape(ge::TransposeInferShape)
            .SetInferDataType(ge::TransposeInferDataType);
        this->AICore().SetTiling(optiling::TransposeTiling).AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);
}
