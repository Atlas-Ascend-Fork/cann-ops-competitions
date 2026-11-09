/**
 * SquareSumV1 算子注册、归约形状推导与主机侧分块规划。
 * 合并相邻归约维度或保留维度，根据连续性和 UB 容量选择计算路径。
 */

#include "square_sum_v1_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {
constexpr uint32_t kTileElements = 8192U;

constexpr uint64_t kMinReduceSeg = 4096U;
constexpr uint32_t kPath4CompactTileElements = 15360U;
constexpr uint32_t kPath7CompactTileElements = 22528U;
constexpr uint64_t kMinInnerChunk = 256U;

uint32_t GetAivCoreCount(const gert::TilingContext* context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return std::max<uint32_t>(1U, platform.GetCoreNumAiv());
}

bool BuildReduceMask(const gert::RuntimeAttrs* attrs, size_t rank, bool mask[S9_SQ_MAX_RANK])
{
    for (size_t i = 0; i < rank; ++i) {
        mask[i] = false;
    }
    if (attrs == nullptr) {
        return false;
    }
    const auto* axes = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (axes == nullptr || axes->GetSize() == 0) {
        for (size_t i = 0; i < rank; ++i) {
            mask[i] = true;
        }
        return true;
    }
    const auto* values = reinterpret_cast<const int64_t*>(axes->GetData());
    for (size_t i = 0; i < axes->GetSize(); ++i) {
        int64_t axis = values[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return false;
        }
        mask[axis] = true;
    }
    return true;
}
}

namespace optiling {
// 合并相邻的归约或保留维度，计算任务数、归约切分和部分结果工作区。
static ge::graphStatus SquareSumV1Tiling(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<SquareSumV1TilingData>();
    const auto* storage = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    if (tiling == nullptr || storage == nullptr || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shape = storage->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    if (rank == 0 || rank > S9_SQ_MAX_RANK) {
        return ge::GRAPH_FAILED;
    }
    bool mask[S9_SQ_MAX_RANK];
    if (!BuildReduceMask(context->GetAttrs(), rank, mask)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t stride[S9_SQ_MAX_RANK];
    uint64_t running = 1;
    for (size_t i = rank; i > 0; --i) {
        stride[i - 1] = running;
        running *= static_cast<uint64_t>(shape.GetDim(i - 1));
    }

    uint64_t len[S9_SQ_MAX_RANK];
    uint64_t str[S9_SQ_MAX_RANK];
    bool red[S9_SQ_MAX_RANK];
    size_t count = 0;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (dim == 1) {
            continue;
        }
        if (count > 0 && red[count - 1] == mask[i]) {
            len[count - 1] *= static_cast<uint64_t>(dim);
            str[count - 1] = stride[i];
            continue;
        }
        len[count] = static_cast<uint64_t>(dim);
        str[count] = stride[i];
        red[count] = mask[i];
        ++count;
    }

    uint64_t outCount = 1;
    for (size_t i = 0; i < count; ++i) {
        if (!red[i]) {
            outCount *= len[i];
        }
    }

    tiling->innerLen = (count == 0) ? 1 : len[count - 1];
    tiling->innerIsReduce = (count == 0) ? 0U : (red[count - 1] ? 1U : 0U);
    const size_t prefixCount = (count == 0) ? 0 : count - 1;
    tiling->groupCount = static_cast<uint32_t>(prefixCount);
    uint64_t prefixOutCount = 1;
    uint64_t prefixRedCount = 1;
    for (size_t i = 0; i < S9_SQ_MAX_GROUP; ++i) {
        if (i < prefixCount) {
            tiling->groupLen[i] = len[i];
            tiling->groupStride[i] = str[i];
            tiling->groupIsReduce[i] = red[i] ? 1U : 0U;
            if (red[i]) {
                prefixRedCount *= len[i];
            } else {
                prefixOutCount *= len[i];
            }
        } else {
            tiling->groupLen[i] = 1;
            tiling->groupStride[i] = 0;
            tiling->groupIsReduce[i] = 0;
        }
    }
    tiling->outCount = outCount;
    tiling->prefixOutCount = prefixOutCount;
    tiling->prefixRedCount = prefixRedCount;
    const bool alignedInner = (tiling->innerLen % 8U) == 0U;
    if (alignedInner) {
        tiling->tileElements = inputDesc->GetDataType() == ge::DT_FLOAT ? 9216U : 11264U;
    } else {
        tiling->tileElements = kTileElements;
    }
    tiling->slabScratch = 0;
    if (tiling->innerIsReduce == 0U && prefixCount >= 3U &&
        red[prefixCount - 1U] && !red[prefixCount - 2U] &&
        str[prefixCount - 1U] == tiling->innerLen &&
        len[prefixCount - 1U] >= 64U) {
        tiling->slabScratch = 1;
    }

    {
        const uint64_t esz =
            inputDesc->GetDataType() == ge::DT_FLOAT ? 4U : 2U;
        if (tiling->slabScratch == 0U && tiling->innerIsReduce == 0U &&
            prefixCount >= 1U && red[prefixCount - 1U] &&
            str[prefixCount - 1U] == tiling->innerLen &&
            (tiling->innerLen * esz) % 32U != 0U) {
            tiling->slabScratch = 2U;
        }
    }

    uint32_t cores = GetAivCoreCount(context);

    if (cores < 2) {
        cores = 48;
    }
    tiling->redSegCount = 1;
    tiling->redSegLen = tiling->innerLen;
    tiling->keepRedSeg = 1;
    tiling->keepRedSegLen = tiling->prefixRedCount;
    if (tiling->innerIsReduce == 1U) {

        tiling->innerChunk = tiling->innerLen;
        tiling->innerChunkCount = 1;
        tiling->taskCount = tiling->outCount;

        if (tiling->outCount < cores && tiling->innerLen >= 2 * kMinReduceSeg) {
            uint64_t seg = (cores + tiling->outCount - 1) / tiling->outCount;
            const uint64_t maxSeg = tiling->innerLen / kMinReduceSeg;
            if (seg > maxSeg) {
                seg = maxSeg;
            }
            if (seg > 1) {
                tiling->redSegCount = seg;
                tiling->redSegLen = (tiling->innerLen + seg - 1) / seg;
                tiling->taskCount = tiling->outCount * seg;
            }
        }
    } else {

        uint64_t chunk = std::min<uint64_t>(tiling->innerLen, tiling->tileElements);
        const uint64_t wanted = static_cast<uint64_t>(cores);

        const bool contiguousReduce =
            prefixCount >= 1U && red[prefixCount - 1U] &&
            str[prefixCount - 1U] == tiling->innerLen;
        const bool reduceSplitFillsCores =
            tiling->innerLen <= tiling->tileElements &&
            tiling->prefixRedCount >= wanted &&
            (prefixOutCount == 1 ||
             (contiguousReduce && prefixOutCount * wanted <= (256UL * 1024UL) /
                                      (tiling->innerLen == 0U ? 1U : tiling->innerLen)));
        if (!reduceSplitFillsCores) {

            const uint64_t maxChunk =
                std::min<uint64_t>(tiling->innerLen, tiling->tileElements);
            const uint64_t minTasks = maxChunk == 0U ? 1U :
                prefixOutCount * ((tiling->innerLen + maxChunk - 1) / maxChunk);
            if (minTasks > 0U && prefixOutCount > 0U) {
                uint64_t waves = (minTasks + wanted - 1) / wanted;
                if (waves < 1U) {
                    waves = 1U;
                }
                const uint64_t want = waves * wanted;
                if (want >= prefixOutCount) {
                    uint64_t perPrefix = want / prefixOutCount;
                    if (perPrefix > tiling->innerLen) {
                        perPrefix = tiling->innerLen;
                    }
                    if (perPrefix >= 1U) {
                        uint64_t cand =
                            (tiling->innerLen + perPrefix - 1) / perPrefix;

                        const uint64_t lanes = 32U / static_cast<uint64_t>(
                            inputDesc->GetDataType() == ge::DT_FLOAT ? 4U : 2U);
                        cand = ((cand + lanes - 1U) / lanes) * lanes;
                        if (cand >= 1U && cand <= maxChunk) {
                            chunk = cand;
                        }
                    }
                }
            }
        }
        chunk = std::max<uint64_t>(1, chunk);
        tiling->innerChunk = chunk;
        tiling->innerChunkCount = (tiling->innerLen + chunk - 1) / chunk;
        tiling->taskCount = prefixOutCount * tiling->innerChunkCount;
        tiling->keepRedSeg = 1;
        tiling->keepRedSegLen = tiling->prefixRedCount;

        if (tiling->taskCount < cores && tiling->prefixRedCount >= 2 &&
            tiling->outCount * cores <= (256UL * 1024)) {
            uint64_t seg = cores / tiling->taskCount;
            seg = std::min<uint64_t>(seg, tiling->prefixRedCount);
            if (seg > 1) {
                tiling->keepRedSeg = seg;
                tiling->keepRedSegLen = (tiling->prefixRedCount + seg - 1) / seg;
                tiling->taskCount = tiling->taskCount * seg;
            }
        }
    }

    if (tiling->innerIsReduce == 0U && tiling->keepRedSeg > 1U &&
        tiling->innerChunkCount == 1U && tiling->innerLen > 0U) {
        const uint64_t esz2 =
            inputDesc->GetDataType() == ge::DT_FLOAT ? 4U : 2U;
        const uint64_t rowBytes2 = tiling->innerLen * esz2;
        uint64_t a2 = 32U, b2 = rowBytes2;
        while (a2 != 0U) { const uint64_t t2 = b2 % a2; b2 = a2; a2 = t2; }
        const uint64_t alignRows2 = b2 == 0U ? 1U : (32U / b2);
        const uint64_t superWidth2 = alignRows2 * tiling->innerLen;
        if (alignRows2 >= 2U && superWidth2 > 0U &&
            superWidth2 * 2U <= 16384U && superWidth2 * 2U > tiling->tileElements) {
            tiling->tileElements = static_cast<uint32_t>(superWidth2 * 2U);
        }
    }
    const uint64_t usefulTasks = std::max<uint64_t>(1, tiling->taskCount);
    tiling->blockDim = static_cast<uint32_t>(std::min<uint64_t>(cores, usefulTasks));
    context->SetBlockDim(tiling->blockDim);
    const uint64_t elementSize =
        inputDesc->GetDataType() == ge::DT_FLOAT ? 4U : 2U;

    const bool usePath4Cyclic = tiling->innerIsReduce == 1U &&
        tiling->redSegCount == 1U && tiling->prefixRedCount > 1U &&
        prefixCount >= 2U && !red[prefixCount - 1U] &&
        str[prefixCount - 1U] == tiling->innerLen &&
        (tiling->innerLen * elementSize) % 32U == 0U &&
        tiling->outCount >= cores && running >= (1ULL << 20U);
    const bool usePath4CompactHalf = tiling->innerIsReduce == 1U &&
        tiling->redSegCount == 1U && tiling->prefixRedCount > 1U &&
        prefixCount >= 2U && !red[prefixCount - 1U] &&
        str[prefixCount - 1U] == tiling->innerLen &&
        inputDesc->GetDataType() != ge::DT_FLOAT &&
        (tiling->innerLen % 8U) == 0U && (tiling->innerLen % 16U) != 0U &&
        tiling->outCount >= cores && running >= (1ULL << 20U);
    if (usePath4CompactHalf) {
        tiling->tileElements = kPath4CompactTileElements;
    }
    const bool usePath7CompactHalf = tiling->innerIsReduce == 0U &&
        tiling->keepRedSeg == 1U && tiling->prefixRedCount > 1U &&
        prefixCount >= 1U && red[prefixCount - 1U] &&
        str[prefixCount - 1U] == tiling->innerLen &&
        inputDesc->GetDataType() != ge::DT_FLOAT &&
        tiling->innerLen <= 2048U && (tiling->innerLen % 8U) == 0U &&
        (tiling->innerLen % 16U) != 0U && tiling->outCount >= cores &&
        running >= (1ULL << 20U);
    if (usePath7CompactHalf) {
        tiling->tileElements = kPath7CompactTileElements;
    }
    const bool usePath2CyclicFloat = tiling->innerIsReduce == 0U &&
        tiling->keepRedSeg > 1U && tiling->innerChunkCount == 1U &&
        prefixCount == 1U && red[0] && str[0] == tiling->innerLen &&
        inputDesc->GetDataType() == ge::DT_FLOAT &&
        tiling->innerLen <= tiling->tileElements &&
        (tiling->innerLen % 8U) == 0U && running >= (1ULL << 20U);
    const bool usePath2CyclicHalf = tiling->innerIsReduce == 0U &&
        tiling->keepRedSeg > 1U && tiling->innerChunkCount == 1U &&
        prefixCount == 1U && red[0] && str[0] == tiling->innerLen &&
        inputDesc->GetDataType() != ge::DT_FLOAT &&
        tiling->innerLen <= 2048U && (tiling->innerLen % 8U) == 0U &&
        (tiling->innerLen % 16U) != 0U && running >= (1ULL << 20U);
    if (usePath2CyclicHalf) {
        tiling->tileElements = kPath7CompactTileElements;
    }
    context->SetTilingKey(
        usePath2CyclicHalf ? 5U : (usePath2CyclicFloat ? 4U :
        (usePath7CompactHalf ? 3U :
        (usePath4CompactHalf ? 2U : (usePath4Cyclic ? 1U : 0U)))));
    if (tiling->redSegCount > 1) {
        context->GetWorkspaceSizes(1)[0] = 16UL * 1024 * 1024 +
            tiling->blockDim * 32 + tiling->outCount * tiling->redSegCount * 32 + 32;
    } else if (tiling->keepRedSeg > 1) {

        const uint64_t partialFloats = tiling->outCount * tiling->keepRedSeg;
        const uint64_t alignedPartialFloats = (partialFloats + 7U) & ~7UL;
        context->GetWorkspaceSizes(1)[0] = 16UL * 1024 * 1024 +
            tiling->blockDim * 32 + alignedPartialFloats * sizeof(float) +
            tiling->blockDim * tiling->tileElements * sizeof(float) + 32;
    } else if (tiling->slabScratch != 0U) {
        context->GetWorkspaceSizes(1)[0] = 16UL * 1024 * 1024 +
            tiling->blockDim * 32 +
            tiling->blockDim * tiling->tileElements * sizeof(float) + 32;
    } else {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus SquareSumV1InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(0);
    gert::Shape* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t rank = input->GetDimNum();
    if (rank == 0 || rank > S9_SQ_MAX_RANK) {
        return GRAPH_FAILED;
    }
    bool mask[S9_SQ_MAX_RANK];
    if (!BuildReduceMask(context->GetAttrs(), rank, mask)) {
        return GRAPH_FAILED;
    }
    const bool* keepDims = context->GetAttrs()->GetBool(1);
    const bool keep = (keepDims != nullptr) && *keepDims;

    output->SetDimNum(0);
    for (size_t i = 0; i < rank; ++i) {
        if (!mask[i]) {
            output->AppendDim(input->GetDim(i));
        } else if (keep) {
            output->AppendDim(1);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus SquareSumV1InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> types = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("input").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("output").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::SquareSumV1InferShape)
            .SetInferDataType(ge::SquareSumV1InferDataType);
        this->AICore().SetTiling(optiling::SquareSumV1Tiling).AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}
