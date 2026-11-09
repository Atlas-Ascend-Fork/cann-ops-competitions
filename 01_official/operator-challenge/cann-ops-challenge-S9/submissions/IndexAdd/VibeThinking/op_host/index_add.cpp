#include "index_add_tiling.h"

#include <algorithm>
#include <dlfcn.h>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t kRowStageBytes = 160 * 1024;
constexpr uint64_t kMinRowBytes = 64;
constexpr uint64_t kLayerUbBytes = 176 * 1024;
constexpr uint64_t kLayerCap = 64;
constexpr uint64_t kLaneUbBytes = 188 * 1024;
constexpr uint64_t kWaveCap = 24;
constexpr uint64_t kWaveHist = 32;
constexpr uint64_t kSlotCap = 20;
constexpr uint64_t kHistGroup = 4;
constexpr size_t kReservedWorkspace = 16 * 1024 * 1024;

constexpr uint32_t kL2BypassPaths = (1u << 8) | (1u << 9) | (1u << 10);
constexpr uint64_t kSlabUbBytes = 192 * 1024;
constexpr uint64_t kSlabRows = 64;
constexpr uint64_t kSelectLanes = 512;
constexpr uint64_t kSlabBurst = 40 * 1024;
constexpr uint64_t kMinBytesPerCore = 4096;

uint32_t ElementBytes(ge::DataType type)
{
    switch (type) {
        case ge::DT_INT8:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_INT32:
        case ge::DT_FLOAT:
            return 4;
        default:
            return 0;
    }
}
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* selfShape = context->GetInputShape(0);
    const auto* indexShape = context->GetInputShape(1);
    const auto* sourceShape = context->GetInputShape(2);
    const auto* selfDesc = context->GetInputDesc(0);
    const auto* attrs = context->GetAttrs();
    auto* raw = context->GetRawTilingData();
    if (selfShape == nullptr || indexShape == nullptr || sourceShape == nullptr || selfDesc == nullptr ||
        attrs == nullptr || raw == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& self = selfShape->GetStorageShape();
    const gert::Shape& index = indexShape->GetStorageShape();
    const gert::Shape& source = sourceShape->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(self.GetDimNum());
    const int64_t* dimAttr = attrs->GetInt(0);
    if (rank == 0 || dimAttr == nullptr || source.GetDimNum() != self.GetDimNum() || index.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    int64_t dim = *dimAttr;
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }
    // 将输入拆成 outer × 轴长度 × inner，沿指定轴按索引累加。
    uint64_t outer = 1;
    uint64_t inner = 1;
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t extent = self.GetDim(static_cast<size_t>(d));
        if (extent <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (d != dim && source.GetDim(static_cast<size_t>(d)) != extent) {
            return ge::GRAPH_FAILED;
        }
        if (d < dim) {
            outer *= static_cast<uint64_t>(extent);
        } else if (d > dim) {
            inner *= static_cast<uint64_t>(extent);
        }
    }
    const uint64_t selfLen = static_cast<uint64_t>(self.GetDim(static_cast<size_t>(dim)));
    const uint64_t indexLen = static_cast<uint64_t>(index.GetDim(0));
    if (static_cast<uint64_t>(source.GetDim(static_cast<size_t>(dim))) != indexLen) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType type = selfDesc->GetDataType();
    const uint64_t bytes = ElementBytes(type);
    if (bytes == 0 || outer * indexLen * inner > 0xFFFFFFFFULL || outer * selfLen * inner > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    // 将完整输出切片按对齐单位分给各核，保证各核写入范围互不重叠。
    const uint64_t sliceBytes = selfLen * inner * bytes;
    uint64_t unit = 1;
    while ((unit * sliceBytes) % 32 != 0) {
        ++unit;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t cores = std::max<uint64_t>(1, platform.GetCoreNumAiv());
    const uint64_t units = (outer + unit - 1) / unit;
    uint64_t blockDim = std::max<uint64_t>(1, std::min<uint64_t>(cores, units));

    // 根据数据类型、行宽和 UB 占用选择行缓存、分组累加或通用路径。
    const uint64_t pitch = (inner * bytes + 31) / 32 * 32;
    const bool addable = type == ge::DT_FLOAT16 || type == ge::DT_FLOAT || type == ge::DT_INT32;
    uint32_t path =
        (addable && inner * bytes >= kMinRowBytes && (selfLen + 1) * pitch <= kRowStageBytes) ? 1 : 0;

    uint64_t slabRows = 0;
    uint64_t workspaceBytes = 0;
    if (addable && inner == 1 && path == 0 && selfLen <= 4096) {
        const uint64_t align = 32 / bytes;
        const uint64_t permPitch = (selfLen + align - 1) / align * align;
        const uint64_t values = (indexLen + (align - 1) * kLayerCap + align + 7) / 8 * 8;
        const uint64_t bins = (selfLen + 63) / 64 * 64;

        const uint64_t fixed = (4 + kHistGroup + 4) * bins * 4 + 288 + values * 4 + 1024;
        const uint64_t perRow = (2 * selfLen + 2 * indexLen + values + 2 * permPitch) * bytes + 2 * permPitch * 4;
        if (fixed + indexLen * 4 + perRow <= kLayerUbBytes) {
            slabRows = std::min<uint64_t>((kLayerUbBytes - fixed) / perRow, outer);
            path = 2;
            workspaceBytes = ((blockDim + 2) * bins + values) * 4;
        }

        const uint64_t laneSelf = (selfLen + 15) / 16 * 16;
        const uint64_t laneSrc = (indexLen + 15) / 16 * 16;
        const uint64_t planBytes = 54 * bins + 288 + values * 4 + 1024;
        const uint64_t laneBytes = 3 * 32 * laneSrc + 5 * 32 * laneSelf;

        const uint64_t slotLanes = (indexLen + 63) / 64 * 64;
        const uint64_t slotScratch = 5 * ((slotLanes * 4 + 31) / 32 * 32) + (kSlotCap * 64 + 8) * 4;
        const uint64_t slotBytes = (kSlotCap * laneSelf + 8 * cores) * 4 + 32;
        if (path == 2 && laneSrc / 16 <= 255 && laneBytes + slotBytes + 32 <= kLaneUbBytes && selfLen <= 1024 &&
            (laneSelf / 8 + cores - 1) / cores * 8 <= 64 && 2 * 16 * laneSrc * bytes >= slotScratch &&
            laneSelf * 4 >= 2 * ((slotLanes / 8 + 31) / 32 * 32)) {
            path = 8;
            workspaceBytes = (kSlotCap * laneSelf + 8 * cores) * 4;
        }

        const uint64_t pairs = (outer + 1) / 2;
        const uint64_t waveCols = (permPitch / 8 + cores - 1) / cores * 8;
        const uint64_t lanes = (indexLen + 63) / 64 * 64;
        const uint64_t srcPitch = (indexLen * 4 + 31) / 32 * 32;
        const uint64_t slot = 2 * srcPitch + 32 + (2 * selfLen * 4 + 31) / 32 * 32;
        const uint64_t valueBytes = (2 * permPitch + 2 * indexLen + 8 * kWaveCap + 7) / 8 * 8 * 4;
        const uint64_t backBytes = 2 * permPitch * 4;
        const uint64_t arena = 2 * slot + 2 * valueBytes + 2 * backBytes;
        if (path == 2 && type == ge::DT_FLOAT && selfLen <= 1024 && indexLen < 16384 && pairs >= cores &&
            waveCols <= 64 && arena <= kLaneUbBytes && slot >= 2 * lanes * 4 &&
            valueBytes >= lanes * 4 + (kWaveCap * 64 + 32) * 4 && valueBytes >= 32768 + 2 * backBytes && valueBytes >= 4 * backBytes &&
            backBytes >= cores * kWaveHist * 4 && backBytes >= 2 * ((lanes / 8 + 31) / 32 * 32) &&
            kWaveCap * permPitch * 4 <= 2 * slot) {
            path = 10;
            blockDim = cores;
            workspaceBytes = (kWaveCap * permPitch + permPitch + cores * kWaveHist) * 4;
        }
    }

    uint32_t selfTile = 0;
    uint32_t sourceTile = 0;
    if (type == ge::DT_FLOAT && inner == 1 && outer >= kSlabRows && indexLen * 4 >= 1024 && selfLen >= 16 &&
        selfLen <= 0x7FFF) {
        uint64_t groupRows = 16;
        while (groupRows < kSlabRows && groupRows * cores < outer) {
            groupRows *= 2;
        }
        const uint64_t indexBytes = (indexLen * 2 + 31) / 32 * 32;
        const uint64_t poolElems = kSlabUbBytes > indexBytes + 32 ? (kSlabUbBytes - indexBytes - 32) / 4 : 0;
        const uint64_t selfCap = poolElems / (2 * groupRows) / 16 * 16;
        const uint64_t sourceCap = std::min(poolElems / 4, kSlabBurst / 4) / groupRows / 16 * 16;
        if (selfCap >= 16 && sourceCap >= 16 && indexLen <= poolElems) {
            const uint64_t selfTiles = (selfLen + selfCap - 1) / selfCap;
            const uint64_t sourceTiles = (indexLen + sourceCap - 1) / sourceCap;
            selfTile = static_cast<uint32_t>(((selfLen + selfTiles - 1) / selfTiles + 15) / 16 * 16);
            sourceTile = static_cast<uint32_t>(((indexLen + 15) / 16 + sourceTiles - 1) / sourceTiles * 16);
            const uint64_t groups = (outer + groupRows - 1) / groupRows;
            if (selfTile <= selfCap && sourceTile <= sourceCap && 2 * ((indexLen + 7) / 8 * 8) <= selfTile * groupRows && groups * groupRows * selfLen * 4 < 0xFFFFFFFFULL) {
                path = 12;
                blockDim = std::min<uint64_t>(cores, groups);
                slabRows = groupRows;
                workspaceBytes = groups * groupRows * selfLen * 4;
            }
        }
    }

    const bool orderFree = type == ge::DT_INT32 || type == ge::DT_FLOAT;
    const uint64_t deltaBytes = selfLen * pitch + indexLen * 4 + 1024;
    if (path == 1 && orderFree && deltaBytes + 2 * pitch <= kLayerUbBytes && outer * indexLen >= 2 * cores) {
        path = 4;
        slabRows = (kLayerUbBytes - deltaBytes) / pitch;
        blockDim = cores;
    }

    if (type == ge::DT_INT32 && path == 4 && (inner % 8 == 0 || inner % 8 == 4) &&
        2 * 8 * inner * 4 + indexLen * 4 + 1024 <= kLaneUbBytes) {
        path = 9;
    }

    uint64_t rowUnit = path == 12 ? slabRows : unit;
    uint64_t ownerCores = blockDim;

    if ((type == ge::DT_BF16 || type == ge::DT_INT8) && inner >= 2 && inner * bytes <= 1024) {
        const uint64_t rowBytes = inner * bytes;
        const uint64_t rowPitch = (rowBytes + 31) / 32 * 32;
        const uint64_t fixed = indexLen * 4 + selfLen * (rowBytes + rowPitch) + (rowBytes / 2 + 8) * 4 +
                               rowPitch + 8 * 512 + 1024;
        if (fixed + rowBytes <= kLayerUbBytes) {
            path = 3;
            slabRows = std::min<uint64_t>((kLayerUbBytes - fixed) / rowBytes, indexLen);
            rowUnit = 1;
            while ((rowUnit * rowBytes) % 32 != 0) {
                ++rowUnit;
            }
            const uint64_t rowUnits = (outer * selfLen + rowUnit - 1) / rowUnit;
            const uint64_t work = outer * (selfLen + indexLen) * rowBytes;
            ownerCores = std::max<uint64_t>(1, std::min<uint64_t>({cores, rowUnits, work / kMinBytesPerCore}));
        }
    }

    if (path == 3 && type == ge::DT_BF16 && selfLen <= 1024 && indexLen <= 2048) {
        const uint64_t lanes = (inner + 15) / 16 * 16;
        const uint64_t rowCap = (kLayerUbBytes - indexLen * 4 - 1024) / (12 * lanes);
        const uint64_t rowUnits = (outer * selfLen + rowUnit - 1) / rowUnit;
        uint64_t coresNeeded = ownerCores;
        while (coresNeeded < std::min<uint64_t>(cores, rowUnits) &&
               (rowUnits + coresNeeded - 1) / coresNeeded * rowUnit > rowCap) {
            ++coresNeeded;
        }
        const uint64_t perCore = (rowUnits + coresNeeded - 1) / coresNeeded * rowUnit;
        if (perCore <= rowCap && perCore <= 1024) {
            path = 7;
            ownerCores = coresNeeded;
            slabRows = perCore;
        }
    }

    if (path == 7 && outer >= 2 && outer <= 0xFFFF) {
        const uint64_t stackCores = std::min<uint64_t>(cores, selfLen);
        const uint64_t perCore = (selfLen + stackCores - 1) / stackCores;
        const uint64_t elems = perCore * outer * ((inner + 15) / 16 * 16);
        if (16 * elems + indexLen * 4 + 16 * 1024 <= kLayerUbBytes) {
            path = 11;
            ownerCores = stackCores;
            slabRows = elems;
            rowUnit = 1;
        }
    }

    if (type == ge::DT_INT8 && inner % 32 == 0 &&
        outer * (selfLen + indexLen) * inner + indexLen * 4 + 1024 <= kLayerUbBytes) {
        path = 5;

        ownerCores = std::max<uint64_t>(1, std::min<uint64_t>(cores, outer * indexLen / 30));
    }

    IndexAddTilingData data;
    data.set_outer(static_cast<uint32_t>(outer));
    data.set_selfLen(static_cast<uint32_t>(selfLen));
    data.set_inner(static_cast<uint32_t>(inner));
    data.set_indexLen(static_cast<uint32_t>(indexLen));
    data.set_unit(static_cast<uint32_t>(rowUnit));
    data.set_path(path);
    data.set_slabRows(static_cast<uint32_t>(slabRows));
    data.set_selfTile(selfTile);
    data.set_sourceTile(sourceTile);
    uint64_t l2Offset = 0;
    if (path < 32 && (kL2BypassPaths >> path) & 1u) {
        using GetDeviceFn = int32_t (*)(int32_t*);
        using GetOffsetFn = int32_t (*)(uint32_t, uint64_t*);
        const auto getDevice = reinterpret_cast<GetDeviceFn>(dlsym(RTLD_DEFAULT, "rtGetDevice"));
        const auto getOffset = reinterpret_cast<GetOffsetFn>(dlsym(RTLD_DEFAULT, "rtGetL2CacheOffset"));
        int32_t device = -1;
        if (getDevice == nullptr || getOffset == nullptr || getDevice(&device) != 0 || device < 0 ||
            getOffset(static_cast<uint32_t>(device), &l2Offset) != 0) {
            l2Offset = 0;
        }
    }
    data.set_l2Low(static_cast<uint32_t>(l2Offset));
    data.set_l2High(static_cast<uint32_t>(l2Offset >> 32));
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    if (workspaceBytes > 0) {
        size_t* workspace = context->GetWorkspaceSizes(1);
        if (workspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspace[0] = kReservedWorkspace + workspaceBytes;
    }
    context->SetBlockDim(static_cast<uint32_t>(ownerCores));
    context->SetTilingKey(path == 12 ? 1 : (path == 11 && indexLen <= kSelectLanes ? 2 : 0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (selfShape == nullptr || outShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outShape = *selfShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}
