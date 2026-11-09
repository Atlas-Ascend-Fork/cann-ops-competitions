#include "concat_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kMaxInputs = 256;
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kMaxCores = 40;
constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kDefaultUbBytes = 192 * 1024;
constexpr uint32_t kMaxDataCopyBlocks = 65535;
constexpr uint64_t kTinyOutputBytes = 2048;
constexpr int64_t kScheduleRows = 0;
constexpr int64_t kScheduleInputRows = 1;
constexpr int64_t kScheduleFlatContiguous = 2;
constexpr int64_t kScheduleInputFullOuter = 3;
constexpr int64_t kScheduleInputSegments = 5;
constexpr int64_t kSchedulePackedNarrowInputs = 6;
constexpr int64_t kSchedulePackedRaggedNarrowInputs = 7;
constexpr int64_t kSchedulePackedRaggedNarrowTailInputs = 8;
constexpr int64_t kSchedulePackedWideChunks = 9;
constexpr int64_t kScheduleFp16UnitTranspose = 12;
constexpr int64_t kScheduleBalancedGroups = 13;
constexpr int64_t kScheduleSmallInputDirectRows = 14;
constexpr int64_t kScheduleInputMajorBalanced = 15;
constexpr int64_t kScheduleSmallPacked = 16;
constexpr uint64_t kKeyPackedWideChunksFp32 = 50;
constexpr uint64_t kKeyBalancedGroups = 51;
constexpr uint64_t kKeySmallInputDirectRows = 52;
constexpr uint64_t kKeyInputMajorBalancedBase = 54;
constexpr uint64_t kKeyGenericInputReadBypassBase = 60;
constexpr uint64_t kKeyFp16UnitTranspose = 10771;
constexpr uint64_t kGenericInputReadBypassMinOutputBytes = 2 * 1024 * 1024;
constexpr uint64_t kInputMajorBalancedMinOutputBytes = 1 * 1024 * 1024;
constexpr uint64_t kInputMajorBalancedMinMaxRowBytes = 512;
constexpr int64_t kFlatLargeElemThreshold = 4096;
constexpr uint64_t kSmallRowMaxOutputBytes = 1024 * 1024;
constexpr uint64_t kSmallRowMaxSegmentBytes = 256;
constexpr uint64_t kSmallRowMaxOutputRowBytes = 2048;
constexpr int64_t kSmallRowMaxInputs = 16;
constexpr uint64_t kInputSegmentMinOutputBytes = 8 * 1024 * 1024;
constexpr uint64_t kInputSegmentMinLocalBytes = 256 * 1024;
constexpr int64_t kInputSegmentMaxOuter = 16;
constexpr int64_t kInputSegmentMaxInputs = 16;
constexpr uint32_t kInputSegmentMaxTasksPerCore = 8;
constexpr uint64_t kPackedNarrowMinOutputBytes = 8 * 1024 * 1024;
constexpr int64_t kPackedNarrowMinInputs = 32;
constexpr int64_t kPackedNarrowMaxSplit = 16;
constexpr int64_t kPackedNarrowMaxConcatDim = 1024;
constexpr uint32_t kPackedNarrowUbReserveBytes = 4096;
constexpr int64_t kPackedWideChunkElems = 256;
constexpr int64_t kPackedWideMaxSplit = 128;
constexpr uint64_t kFp16UnitTransposeMinOutputBytes = 32 * 1024 * 1024;
constexpr uint64_t kFp16UnitTransposeCoreBoundaryBytes = 64 * 1024;
constexpr int64_t kFp16UnitTransposeMinInputs = 32;
constexpr int64_t kBalancedGroupCount = 10;
constexpr int64_t kBalancedGroupArraySize = 21;
constexpr int64_t kBalancedGroupMinInputs = 40;
constexpr int64_t kBalancedGroupMaxInputsPerGroup = 24;
constexpr int64_t kBalancedGroupMinNonZeroInputs = 20;
constexpr int64_t kBalancedGroupMaxSplit = 256;
constexpr uint64_t kBalancedGroupMinOutputBytes = 32 * 1024 * 1024;
constexpr uint64_t kBalancedGroupWidthCost = 656;
constexpr uint64_t kBalancedGroupInputCost = 5248;
constexpr uint32_t kBalancedGroupPreferredRows = 16;
constexpr uint32_t kBalancedGroupMinRows = 8;
constexpr uint32_t kBalancedGroupUbReserveBytes = 3 * 1024;
constexpr int64_t kSmallInputDirectMaxInputs = 8;
constexpr uint64_t kSmallInputDirectMinOutputBytes = 1024 * 1024;
constexpr uint64_t kMaxDataCopyRows = UINT16_MAX;

constexpr ge::DataType kSupportedTypes[] = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8};
constexpr ge::Format kSupportedFormats[] = {ge::FORMAT_ND};

inline int64_t ElementBytes(ge::DataType dtype) {
    switch (dtype) {
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
            return 1;
        default:
            return -1;
    }
}

inline bool IsSupportedType(ge::DataType dtype) {
    for (const auto supported : kSupportedTypes) {
        if (dtype == supported) {
            return true;
        }
    }
    return false;
}

inline uint64_t TilingKeyFor(ge::DataType dtype, bool inputSegmentSchedule) {
    const uint64_t scheduleOffset = inputSegmentSchedule ? 10U : 0U;
    switch (dtype) {
        case ge::DT_FLOAT16:
            return scheduleOffset + 0U;
        case ge::DT_FLOAT:
            return scheduleOffset + 1U;
        case ge::DT_INT32:
            return scheduleOffset + 2U;
        case ge::DT_INT8:
            return scheduleOffset + 3U;
        default:
            return scheduleOffset + 0U;
    }
}

inline uint64_t PackedTilingKeyFor(ge::DataType dtype, bool ragged) {
    const uint64_t scheduleOffset = ragged ? 30U : 20U;
    switch (dtype) {
        case ge::DT_FLOAT16:
            return scheduleOffset + 0U;
        case ge::DT_FLOAT:
            return scheduleOffset + 1U;
        case ge::DT_INT32:
            return scheduleOffset + 2U;
        case ge::DT_INT8:
            return scheduleOffset + 3U;
        default:
            return scheduleOffset + 0U;
    }
}

inline uint64_t PackedTailTilingKeyFor(ge::DataType dtype) {
    return PackedTilingKeyFor(dtype, false) + 20U;
}

inline uint64_t InputMajorBalancedTilingKeyFor(ge::DataType dtype) {
    switch (dtype) {
        case ge::DT_FLOAT16:
            return kKeyInputMajorBalancedBase;
        case ge::DT_FLOAT:
            return kKeyInputMajorBalancedBase + 1U;
        case ge::DT_INT32:
            return kKeyInputMajorBalancedBase + 2U;
        case ge::DT_INT8:
            return kKeyInputMajorBalancedBase + 3U;
        default:
            return kKeyInputMajorBalancedBase;
    }
}

inline uint64_t GenericInputReadBypassTilingKeyFor(ge::DataType dtype) {
    return kKeyGenericInputReadBypassBase + TilingKeyFor(dtype, false);
}

inline bool ReadDimAttr(const gert::RuntimeAttrs *attrs, int64_t &dim) {
    if (attrs == nullptr) {
        return false;
    }
    const int64_t *dimPtr = attrs->GetInt(0);
    if (dimPtr == nullptr) {
        return false;
    }
    dim = *dimPtr;
    return true;
}

inline bool NormalizeAxis(int64_t rank, int64_t &axis) {
    if (rank <= 0) {
        return false;
    }
    if (axis < 0) {
        axis += rank;
    }
    return axis >= 0 && axis < rank;
}

inline int64_t ProductBefore(const gert::Shape &shape, int64_t axis) {
    int64_t value = 1;
    for (int64_t idx = 0; idx < axis; ++idx) {
        const int64_t dim = shape.GetDim(idx);
        if (dim < 0) {
            return -1;
        }
        value *= dim;
    }
    return value;
}

inline int64_t ProductAfter(const gert::Shape &shape, int64_t axis) {
    int64_t value = 1;
    for (int64_t idx = axis + 1; idx < shape.GetDimNum(); ++idx) {
        const int64_t dim = shape.GetDim(idx);
        if (dim < 0) {
            return -1;
        }
        value *= dim;
    }
    return value;
}

struct ConcatProfile {
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t axis = 0;
    int64_t rank = 0;
    int64_t inputCount = 0;
    int64_t outer = 1;
    int64_t inner = 1;
    int64_t concatDim = 0;
    int64_t outputElems = 0;
    int64_t elemBytes = 2;
    int64_t nonZeroCount = 0;
    int64_t nonZeroInput = 0;
    int64_t maxSplit = 0;
    uint16_t splits[kMaxInputs] = {};
};

struct SchedulePlan {
    int64_t mode = kScheduleRows;
    int64_t workItems = 1;
    int64_t rowsPerTask = 1;
    int64_t blockLimit = 0;
};

struct BalancedGroupPlan {
    uint16_t inputBounds[kBalancedGroupArraySize] = {};
    uint32_t outputOffsets[kBalancedGroupArraySize] = {};
    uint32_t maxWidth = 0;
    int64_t groupCount = 0;
};

inline uint32_t AlignUp(uint32_t value, uint32_t align) {
    return ((value + align - 1U) / align) * align;
}

inline bool LoadProfile(gert::TilingContext *context, ConcatProfile &profile) {
    int64_t axis = 0;
    if (!ReadDimAttr(context->GetAttrs(), axis)) {
        return false;
    }

    const size_t inputCount = context->GetComputeNodeInputNum();
    if (inputCount == 0 || inputCount > static_cast<size_t>(kMaxInputs)) {
        return false;
    }

    const gert::StorageShape *firstShapeHolder = context->GetDynamicInputShape(0, 0);
    const gert::Tensor *firstTensor = context->GetDynamicInputTensor(0, 0);
    if (firstShapeHolder == nullptr || firstTensor == nullptr) {
        return false;
    }

    const gert::Shape firstShape = firstShapeHolder->GetStorageShape();
    const int64_t rank = firstShape.GetDimNum();
    if (!NormalizeAxis(rank, axis)) {
        return false;
    }

    const ge::DataType dtype = firstTensor->GetDataType();
    const int64_t elemBytes = ElementBytes(dtype);
    if (!IsSupportedType(dtype) || elemBytes <= 0) {
        return false;
    }

    const int64_t outer = ProductBefore(firstShape, axis);
    const int64_t inner = ProductAfter(firstShape, axis);
    if (outer < 0 || inner < 0) {
        return false;
    }

    int64_t concatDim = 0;
    int64_t nonZeroCount = 0;
    int64_t nonZeroInput = 0;
    for (size_t idx = 0; idx < inputCount; ++idx) {
        const gert::StorageShape *shapeHolder = context->GetDynamicInputShape(0, idx);
        const gert::Tensor *tensor = context->GetDynamicInputTensor(0, idx);
        if (shapeHolder == nullptr || tensor == nullptr || tensor->GetDataType() != dtype) {
            return false;
        }
        const gert::Shape shape = shapeHolder->GetStorageShape();
        if (shape.GetDimNum() != rank) {
            return false;
        }
        for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
            const int64_t dimValue = shape.GetDim(dimIdx);
            if (dimValue < 0) {
                return false;
            }
            if (dimIdx != axis && dimValue != firstShape.GetDim(dimIdx)) {
                return false;
            }
        }
        const int64_t split = shape.GetDim(axis);
        if (split > UINT16_MAX) {
            return false;
        }
        profile.splits[idx] = static_cast<uint16_t>(split);
        if (split > 0) {
            ++nonZeroCount;
            nonZeroInput = static_cast<int64_t>(idx);
            profile.maxSplit = std::max(profile.maxSplit, split);
        }
        concatDim += split;
    }

    if (concatDim < 0) {
        return false;
    }

    profile.dtype = dtype;
    profile.axis = axis;
    profile.rank = rank;
    profile.inputCount = static_cast<int64_t>(inputCount);
    profile.outer = outer;
    profile.inner = inner;
    profile.concatDim = concatDim;
    profile.outputElems = outer * concatDim * inner;
    profile.elemBytes = elemBytes;
    profile.nonZeroCount = nonZeroCount;
    profile.nonZeroInput = nonZeroInput;
    return profile.outputElems >= 0;
}

inline int64_t MaxSplit(const ConcatProfile &profile) {
    int64_t maxSplit = 1;
    for (int64_t idx = 0; idx < profile.inputCount && idx < kMaxInputs; ++idx) {
        maxSplit = std::max<int64_t>(maxSplit, profile.splits[idx]);
    }
    return maxSplit;
}

inline uint32_t GetUbBytes(gert::TilingContext *context) {
    uint64_t ubBytes = kDefaultUbBytes;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    return static_cast<uint32_t>(std::max<uint64_t>(ubBytes, kAlignBytes));
}

inline uint32_t PickRowsPerTask(gert::TilingContext *context, const ConcatProfile &profile) {
    const int64_t maxSplit = MaxSplit(profile);
    const uint64_t rowBytes = static_cast<uint64_t>(maxSplit) * static_cast<uint64_t>(profile.inner) *
                              static_cast<uint64_t>(profile.elemBytes);
    const uint64_t alignedRowBytes = ((rowBytes + kAlignBytes - 1U) / kAlignBytes) * kAlignBytes;
    if (alignedRowBytes == 0) {
        return 1;
    }

    const uint32_t ubBytes = GetUbBytes(context);
    uint64_t maxRows = ubBytes / (alignedRowBytes * kBufferNum);
    if (maxRows == 0) {
        maxRows = 1;
    }
    maxRows = std::min<uint64_t>(maxRows, kMaxDataCopyBlocks);

    const uint64_t outputBytes = static_cast<uint64_t>(profile.outputElems) * static_cast<uint64_t>(profile.elemBytes);
    const uint64_t outputRowBytes = static_cast<uint64_t>(profile.concatDim) * static_cast<uint64_t>(profile.inner) *
                                    static_cast<uint64_t>(profile.elemBytes);
    if (outputBytes <= kSmallRowMaxOutputBytes && rowBytes <= kSmallRowMaxSegmentBytes &&
        outputRowBytes <= kSmallRowMaxOutputRowBytes && profile.inputCount > 1 &&
        profile.inputCount <= kSmallRowMaxInputs && profile.nonZeroCount > 1) {
        const uint64_t targetTasks = std::min<uint64_t>(kMaxCores, static_cast<uint64_t>(profile.inputCount));
        const uint64_t targetRows =
            (static_cast<uint64_t>(std::max<int64_t>(1, profile.outer)) + targetTasks - 1U) / targetTasks;
        return static_cast<uint32_t>(std::min<uint64_t>(targetRows, maxRows));
    }

    if (outputBytes > 0 && outputBytes <= kTinyOutputBytes && profile.inner >= 8) {
        return static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(std::max<int64_t>(1, profile.outer)), maxRows));
    }
    if (outputBytes > 0 && outputBytes <= kTinyOutputBytes && profile.inner == 1 && profile.inputCount < 8 &&
        profile.outer >= 16) {
        return static_cast<uint32_t>(std::min<uint64_t>(2, maxRows));
    }

    uint64_t targetRows = (static_cast<uint64_t>(profile.outer) + kMaxCores - 1U) / kMaxCores;
    targetRows = std::max<uint64_t>(1, targetRows);
    targetRows = std::min<uint64_t>(targetRows, maxRows);
    return static_cast<uint32_t>(targetRows);
}

inline bool IsTinyOutput(const ConcatProfile &profile) {
    if (profile.outputElems <= 0 || profile.elemBytes <= 0) {
        return false;
    }
    return static_cast<uint64_t>(profile.outputElems) * static_cast<uint64_t>(profile.elemBytes) <= kTinyOutputBytes;
}

inline bool PreferMergedTinyRows(const ConcatProfile &profile) {
    return profile.inner >= 8 && IsTinyOutput(profile);
}

inline uint64_t OutputBytes(const ConcatProfile &profile) {
    if (profile.outputElems <= 0 || profile.elemBytes <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(profile.outputElems) * static_cast<uint64_t>(profile.elemBytes);
}

inline uint64_t MinNonZeroInputLocalBytes(const ConcatProfile &profile) {
    uint64_t minBytes = UINT64_MAX;
    for (int64_t idx = 0; idx < profile.inputCount && idx < kMaxInputs; ++idx) {
        const int64_t split = profile.splits[idx];
        if (split <= 0) {
            continue;
        }
        const uint64_t bytes = static_cast<uint64_t>(split) * static_cast<uint64_t>(profile.inner) *
                               static_cast<uint64_t>(profile.elemBytes);
        minBytes = std::min(minBytes, bytes);
    }
    return minBytes == UINT64_MAX ? 0 : minBytes;
}

inline uint32_t PickFlatElemsPerTask(gert::TilingContext *context, const ConcatProfile &profile,
                                     uint32_t coreCount) {
    const uint64_t alignElems = std::max<uint64_t>(1, kAlignBytes / profile.elemBytes);
    const uint32_t ubBytes = GetUbBytes(context);
    uint64_t maxElems = ubBytes / (static_cast<uint64_t>(profile.elemBytes) * kBufferNum);
    maxElems = (maxElems / alignElems) * alignElems;
    if (maxElems == 0) {
        maxElems = alignElems;
    }

    const uint64_t requestedTasks = profile.outputElems >= kFlatLargeElemThreshold ? 8U : 16U;
    const uint64_t targetTasks = std::max<uint64_t>(
        1, std::min<uint64_t>({static_cast<uint64_t>(coreCount), requestedTasks,
                               static_cast<uint64_t>(profile.outputElems)}));
    uint64_t elemsPerTask = (static_cast<uint64_t>(profile.outputElems) + targetTasks - 1) / targetTasks;
    elemsPerTask = ((elemsPerTask + alignElems - 1) / alignElems) * alignElems;
    elemsPerTask = std::max<uint64_t>(alignElems, std::min<uint64_t>(elemsPerTask, maxElems));
    return static_cast<uint32_t>(std::min<uint64_t>(elemsPerTask, UINT32_MAX));
}

inline uint32_t PickInputSegmentElemsPerTask(gert::TilingContext *context, const ConcatProfile &profile) {
    const uint64_t alignElems = std::max<uint64_t>(1, kAlignBytes / profile.elemBytes);
    const uint64_t ubBytes = std::max<uint32_t>(GetUbBytes(context), kAlignBytes);
    uint64_t maxElems = ubBytes / (static_cast<uint64_t>(profile.elemBytes) * kBufferNum);
    maxElems = (maxElems / alignElems) * alignElems;
    if (maxElems == 0) {
        maxElems = alignElems;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(maxElems, UINT32_MAX));
}

inline int64_t CountInputSegmentWorkItems(const ConcatProfile &profile, int64_t elemsPerTask) {
    if (elemsPerTask <= 0) {
        return 0;
    }
    int64_t tasksPerOuter = 0;
    for (int64_t idx = 0; idx < profile.inputCount && idx < kMaxInputs; ++idx) {
        const int64_t split = profile.splits[idx];
        if (split <= 0) {
            continue;
        }
        const int64_t inputElemsPerOuter = split * profile.inner;
        tasksPerOuter += (inputElemsPerOuter + elemsPerTask - 1) / elemsPerTask;
    }
    return tasksPerOuter * profile.outer;
}

inline bool IsInputSegmentShape(const ConcatProfile &profile) {
    if (OutputBytes(profile) < kInputSegmentMinOutputBytes || profile.outer <= 0 ||
        profile.outer > kInputSegmentMaxOuter || profile.inputCount > kInputSegmentMaxInputs ||
        profile.nonZeroCount <= 1 || MinNonZeroInputLocalBytes(profile) < kInputSegmentMinLocalBytes) {
        return false;
    }
    return true;
}

inline bool InputSegmentWorkItemsFit(int64_t workItems, uint32_t coreCount) {
    const int64_t maxWorkItems = static_cast<int64_t>(std::max<uint32_t>(1, coreCount)) *
                                 kInputSegmentMaxTasksPerCore;
    return workItems > 0 && workItems <= maxWorkItems;
}

inline bool IsPackedNarrowInputShape(const ConcatProfile &profile, uint32_t coreCount) {
    return OutputBytes(profile) >= kPackedNarrowMinOutputBytes &&
           profile.inputCount >= kPackedNarrowMinInputs && profile.nonZeroCount == profile.inputCount &&
           profile.outer >= static_cast<int64_t>(coreCount) && profile.inner == 1 && profile.concatDim > 0 &&
           (profile.concatDim % 8) == 0 && profile.concatDim == profile.inputCount && MaxSplit(profile) == 1;
}

inline bool HasUnalignedUniformSplit(const ConcatProfile &profile) {
    const int64_t split = profile.splits[0];
    if (split <= 0 || split % (kAlignBytes / sizeof(int32_t)) == 0) {
        return false;
    }
    for (int64_t idx = 1; idx < profile.inputCount; ++idx) {
        if (profile.splits[idx] != split) {
            return false;
        }
    }
    return true;
}

inline bool IsPackedRaggedNarrowInputShape(const ConcatProfile &profile, uint32_t coreCount) {
    return OutputBytes(profile) >= kPackedNarrowMinOutputBytes &&
           profile.inputCount >= kPackedNarrowMinInputs && profile.nonZeroCount > 1 &&
           profile.outer >= static_cast<int64_t>(coreCount) && profile.inner == 1 && profile.concatDim > 0 &&
           (profile.concatDim % 8) == 0 && profile.concatDim <= kPackedNarrowMaxConcatDim &&
           MaxSplit(profile) <= kPackedNarrowMaxSplit && !HasUnalignedUniformSplit(profile);
}

inline bool IsPackedRaggedNarrowTailShape(const ConcatProfile &profile, uint32_t coreCount) {
    return OutputBytes(profile) >= kPackedNarrowMinOutputBytes &&
           profile.inputCount >= kPackedNarrowMinInputs && profile.nonZeroCount > 1 &&
           profile.outer >= static_cast<int64_t>(coreCount) && profile.inner == 1 && profile.concatDim > 0 &&
           profile.concatDim <= kPackedNarrowMaxConcatDim &&
           ((profile.concatDim % 8) != 0 || HasUnalignedUniformSplit(profile));
}

inline int64_t CountPackedWideChunks(const ConcatProfile &profile) {
    int64_t chunks = 0;
    int64_t width = 0;
    for (int64_t idx = 0; idx < profile.inputCount && idx < kMaxInputs; ++idx) {
        const int64_t split = profile.splits[idx];
        if (split <= 0) {
            continue;
        }
        if (width > 0 && width + split > kPackedWideChunkElems) {
            ++chunks;
            width = 0;
        }
        width += split;
        if (width == kPackedWideChunkElems) {
            ++chunks;
            width = 0;
        }
    }
    return chunks + (width > 0 ? 1 : 0);
}

inline int64_t MaxPackedWideChunk(const ConcatProfile &profile) {
    int64_t maxWidth = 1;
    int64_t width = 0;
    for (int64_t idx = 0; idx < profile.inputCount && idx < kMaxInputs; ++idx) {
        const int64_t split = profile.splits[idx];
        if (split <= 0) {
            continue;
        }
        if (width > 0 && width + split > kPackedWideChunkElems) {
            maxWidth = std::max(maxWidth, width);
            width = 0;
        }
        width += split;
        if (width == kPackedWideChunkElems) {
            maxWidth = std::max(maxWidth, width);
            width = 0;
        }
    }
    return std::max(maxWidth, width);
}

inline bool IsPackedWideChunkShape(const ConcatProfile &profile, uint32_t coreCount) {
    return profile.dtype == ge::DT_FLOAT && OutputBytes(profile) >= kPackedNarrowMinOutputBytes &&
           profile.inputCount >= kPackedNarrowMinInputs && profile.nonZeroCount > 1 &&
           profile.outer >= static_cast<int64_t>(coreCount) && profile.inner == 1 &&
           profile.concatDim > kPackedNarrowMaxConcatDim && MaxSplit(profile) <= kPackedWideMaxSplit;
}

inline uint64_t BalancedGroupUbBytes(uint32_t width, uint32_t rows, uint32_t groups);

inline bool TryBalancedGroupPlan(const ConcatProfile &profile, BalancedGroupPlan &plan, uint32_t groups) {
    const uint32_t maxInputs = groups == 5 ? 32 : kBalancedGroupMaxInputsPerGroup;
    if (profile.inputCount < groups ||
        profile.inputCount > groups * maxInputs) {
        return false;
    }

    uint32_t prefix[kMaxInputs + 1] = {};
    for (int64_t idx = 0; idx < profile.inputCount; ++idx) {
        prefix[idx + 1] = prefix[idx] + profile.splits[idx];
    }

    constexpr uint64_t kInf = std::numeric_limits<uint64_t>::max() / 4U;
    uint64_t costs[kBalancedGroupArraySize][kMaxInputs + 1];
    uint16_t previous[kBalancedGroupArraySize][kMaxInputs + 1] = {};
    for (int64_t group = 0; group <= groups; ++group) {
        for (int64_t end = 0; end <= profile.inputCount; ++end) {
            costs[group][end] = kInf;
        }
    }
    costs[0][0] = 0;

    for (int64_t group = 1; group <= groups; ++group) {
        for (int64_t end = group; end <= profile.inputCount; ++end) {
            const int64_t firstStart = std::max<int64_t>(group - 1, end - maxInputs);
            for (int64_t start = firstStart; start < end; ++start) {
                if (costs[group - 1][start] == kInf) {
                    continue;
                }
                const uint32_t width = prefix[end] - prefix[start];
                if (width == 0) {
                    continue;
                }
                if (BalancedGroupUbBytes(width, kBalancedGroupPreferredRows, groups) +
                        kBalancedGroupUbReserveBytes > kDefaultUbBytes) {
                    continue;
                }
                const uint64_t rowParts = kMaxCores / groups +
                                          (group - 1 < kMaxCores % groups ? 1 : 0);
                const uint64_t rawCost = static_cast<uint64_t>(width) * kBalancedGroupWidthCost +
                                         static_cast<uint64_t>(end - start) * (groups == 5 ? 4U * kBalancedGroupWidthCost : kBalancedGroupInputCost);
                const uint64_t groupCost = (rawCost * kMaxCores + rowParts - 1) / rowParts;
                const uint64_t candidate = std::max(costs[group - 1][start], groupCost);
                if (candidate < costs[group][end]) {
                    costs[group][end] = candidate;
                    previous[group][end] = static_cast<uint16_t>(start);
                }
            }
        }
    }

    if (costs[groups][profile.inputCount] == kInf) {
        return false;
    }
    int64_t end = profile.inputCount;
    plan.inputBounds[groups] = static_cast<uint16_t>(end);
    for (int64_t group = groups; group > 0; --group) {
        end = previous[group][end];
        plan.inputBounds[group - 1] = static_cast<uint16_t>(end);
    }
    plan.groupCount = groups;
    for (int64_t group = 0; group < plan.groupCount; ++group) {
        const uint32_t beginOffset = prefix[plan.inputBounds[group]];
        const uint32_t endOffset = prefix[plan.inputBounds[group + 1]];
        plan.outputOffsets[group] = beginOffset;
        plan.maxWidth = std::max(plan.maxWidth, endOffset - beginOffset);
    }
    plan.outputOffsets[plan.groupCount] = prefix[profile.inputCount];
    return true;
}

inline bool BuildBalancedGroupPlan(const ConcatProfile &profile, BalancedGroupPlan &plan) {
    return TryBalancedGroupPlan(profile, plan, 5) ||
           TryBalancedGroupPlan(profile, plan, kBalancedGroupCount);
}

inline uint64_t BalancedGroupUbBytes(uint32_t width, uint32_t rows, uint32_t groups) {
    const uint64_t storageWidth = AlignUp(width, kAlignBytes / sizeof(float));
    const uint64_t maxStagePadElems =
        ((groups == 5 ? 32U : kBalancedGroupMaxInputsPerGroup) - 1U) * (kAlignBytes / sizeof(float) - 1U);
    const uint64_t inputSlotBytes =
        AlignUp((rows * width + maxStagePadElems) * sizeof(float), kAlignBytes);
    const uint64_t inputBytes = (groups == 5 ? 1U : 2U) * inputSlotBytes;
    const uint64_t storeRows = groups == 5 ? 2 : (rows >= 16 ? 4 : rows);
    const uint64_t outputBytes = 2U * storeRows * storageWidth * sizeof(float);
    const uint64_t offsetBytes = (2U * storeRows + 1U) * storageWidth * sizeof(uint32_t);
    return inputBytes + outputBytes + offsetBytes;
}

inline uint32_t PickBalancedGroupRows(gert::TilingContext *context, const BalancedGroupPlan &plan) {
    const uint64_t ubBytes = GetUbBytes(context);
    for (uint32_t rows = kBalancedGroupPreferredRows; rows >= kBalancedGroupMinRows;
         rows -= kBalancedGroupMinRows) {
        if (BalancedGroupUbBytes(plan.maxWidth, rows, plan.groupCount) + kBalancedGroupUbReserveBytes <= ubBytes) {
            return rows;
        }
    }
    return 0;
}

inline bool IsBalancedGroupShape(gert::TilingContext *context, const ConcatProfile &profile,
                                 uint32_t coreCount) {
    if (coreCount != kMaxCores || profile.dtype != ge::DT_FLOAT ||
        profile.inputCount < kBalancedGroupMinInputs || profile.nonZeroCount < kBalancedGroupMinNonZeroInputs ||
        profile.outer < static_cast<int64_t>(coreCount) || profile.inner != 1 ||
        OutputBytes(profile) < kBalancedGroupMinOutputBytes || MaxSplit(profile) > kBalancedGroupMaxSplit) {
        return false;
    }
    BalancedGroupPlan plan;
    return BuildBalancedGroupPlan(profile, plan) && PickBalancedGroupRows(context, plan) >= kBalancedGroupMinRows;
}

inline void DirectQueueWidths(const ConcatProfile &profile, uint64_t &firstBytes, uint64_t &secondBytes) {
    int64_t nonZeroOrdinal = 0;
    for (int64_t idx = 0; idx < profile.inputCount; ++idx) {
        if (profile.splits[idx] == 0) {
            continue;
        }
        const uint64_t bytes = static_cast<uint64_t>(profile.splits[idx]) * profile.inner * profile.elemBytes;
        uint64_t &queueBytes = (nonZeroOrdinal++ & 1) == 0 ? firstBytes : secondBytes;
        queueBytes = std::max(queueBytes, bytes);
    }
}

inline bool IsSmallInputDirectRowsShape(const ConcatProfile &profile, uint32_t coreCount) {
    const uint64_t outputRowBytes = static_cast<uint64_t>(profile.concatDim) * profile.inner * profile.elemBytes;
    if (coreCount != kMaxCores || profile.inputCount < 2 || profile.inputCount > kSmallInputDirectMaxInputs ||
        profile.nonZeroCount < 2 || profile.outer < static_cast<int64_t>(coreCount) || profile.inner <= 0 ||
        OutputBytes(profile) < kSmallInputDirectMinOutputBytes || outputRowBytes > UINT32_MAX) {
        return false;
    }
    uint64_t firstBytes = 0;
    uint64_t secondBytes = 0;
    DirectQueueWidths(profile, firstBytes, secondBytes);
    return firstBytes > 0 && secondBytes > 0 && firstBytes <= UINT32_MAX && secondBytes <= UINT32_MAX;
}

inline uint32_t PickSmallInputDirectRows(gert::TilingContext *context, const ConcatProfile &profile) {
    uint64_t firstBytes = 0;
    uint64_t secondBytes = 0;
    DirectQueueWidths(profile, firstBytes, secondBytes);
    const uint64_t alignedFirstBytes = (firstBytes + kAlignBytes - 1U) / kAlignBytes * kAlignBytes;
    const uint64_t alignedSecondBytes = (secondBytes + kAlignBytes - 1U) / kAlignBytes * kAlignBytes;
    const uint64_t maxRows = GetUbBytes(context) / (alignedFirstBytes + alignedSecondBytes);
    const uint64_t rowsPerCore = (static_cast<uint64_t>(profile.outer) + kMaxCores - 1U) / kMaxCores;
    return static_cast<uint32_t>(std::min<uint64_t>({maxRows, rowsPerCore, kMaxDataCopyRows}));
}

inline bool IsFp16UnitTransposeShape(const ConcatProfile &profile) {
    const uint64_t outputBytes = OutputBytes(profile);
    return profile.dtype == ge::DT_FLOAT16 && outputBytes >= kFp16UnitTransposeMinOutputBytes &&
           profile.inputCount >= kFp16UnitTransposeMinInputs && profile.inner == 1 &&
           profile.nonZeroCount == profile.inputCount && profile.concatDim == profile.inputCount &&
           MaxSplit(profile) == 1;
}

inline uint32_t PickPackedNarrowRows(gert::TilingContext *context, const ConcatProfile &profile) {
    const uint64_t ubBytes = std::max<uint32_t>(GetUbBytes(context), kAlignBytes);
    const uint64_t usableUbBytes = ubBytes > kPackedNarrowUbReserveBytes ? ubBytes - kPackedNarrowUbReserveBytes
                                                                        : ubBytes;
    const uint64_t bytesPerElement = profile.dtype == ge::DT_INT8
                                         ? 2U * 2U + 1U + sizeof(uint32_t)
                                         : 2U * static_cast<uint64_t>(profile.elemBytes) + sizeof(uint32_t);
    const uint64_t bytesPerRow = static_cast<uint64_t>(profile.concatDim) * bytesPerElement;
    const uint64_t alignRows = std::max<uint64_t>(1, kAlignBytes / profile.elemBytes);
    uint64_t rows = bytesPerRow == 0 ? 1 : usableUbBytes / bytesPerRow;
    rows = (rows / alignRows) * alignRows;
    rows = std::min<uint64_t>(rows, static_cast<uint64_t>(std::max<int64_t>(1, profile.outer)));
    return static_cast<uint32_t>(rows);
}

inline uint32_t PickPackedTailRows(gert::TilingContext *context, const ConcatProfile &profile) {
    const uint64_t ubBytes = std::max<uint32_t>(GetUbBytes(context), kAlignBytes);
    const uint64_t alignElems = std::max<uint64_t>(1, kAlignBytes / profile.elemBytes);
    const uint64_t storageDim = AlignUp(static_cast<uint32_t>(profile.concatDim),
                                        static_cast<uint32_t>(alignElems));
    uint64_t fixedBytes = storageDim * sizeof(int32_t);
    uint64_t bytesPerRow = 0;
    if (profile.dtype == ge::DT_INT8) {
        bytesPerRow = static_cast<uint64_t>(profile.concatDim) * 2U + storageDim * 7U;
    } else {
        bytesPerRow = static_cast<uint64_t>(profile.concatDim) * profile.elemBytes +
                      storageDim * (profile.elemBytes + sizeof(uint32_t));
    }
    const uint64_t usableBytes = ubBytes > kPackedNarrowUbReserveBytes + fixedBytes
                                     ? ubBytes - kPackedNarrowUbReserveBytes - fixedBytes
                                     : 0;
    uint64_t rows = bytesPerRow == 0 ? 0 : usableBytes / bytesPerRow;
    const uint64_t gatherElemBytes = profile.dtype == ge::DT_INT8 ? sizeof(uint16_t) : profile.elemBytes;
    rows = std::min<uint64_t>(rows, (UINT8_MAX * 256U / gatherElemBytes) / storageDim);
    rows = (rows / alignElems) * alignElems;
    rows = std::min<uint64_t>(rows, static_cast<uint64_t>(std::max<int64_t>(1, profile.outer)));
    return static_cast<uint32_t>(rows);
}

inline uint32_t PickPackedWideRows(gert::TilingContext *context, const ConcatProfile &profile) {
    const uint64_t ubBytes = std::max<uint32_t>(GetUbBytes(context), kAlignBytes);
    const uint64_t alignElems = kAlignBytes / sizeof(float);
    const uint64_t chunkWidth = static_cast<uint64_t>(MaxPackedWideChunk(profile));
    const uint64_t storageWidth = AlignUp(static_cast<uint32_t>(chunkWidth), static_cast<uint32_t>(alignElems));
    const uint64_t fixedBytes = storageWidth * sizeof(int32_t);
    const uint64_t bytesPerRow = chunkWidth * sizeof(float) +
                                 storageWidth * (sizeof(float) + sizeof(uint32_t));
    const uint64_t usableBytes = ubBytes > kPackedNarrowUbReserveBytes + fixedBytes
                                     ? ubBytes - kPackedNarrowUbReserveBytes - fixedBytes
                                     : 0;
    uint64_t rows = bytesPerRow == 0 ? 0 : usableBytes / bytesPerRow;
    rows = std::min<uint64_t>(rows, (UINT8_MAX * 256U / sizeof(float)) / storageWidth);
    rows = std::max<uint64_t>(alignElems, rows / alignElems * alignElems);
    rows = std::min<uint64_t>(rows, static_cast<uint64_t>(std::max<int64_t>(1, profile.outer)));
    return static_cast<uint32_t>(rows);
}

inline uint32_t PickBalancedGroupRows(gert::TilingContext *context, const ConcatProfile &profile) {
    BalancedGroupPlan plan;
    if (!BuildBalancedGroupPlan(profile, plan)) {
        return 1;
    }
    return std::min<uint32_t>(PickBalancedGroupRows(context, plan),
                              static_cast<uint32_t>(std::max<int64_t>(1, profile.outer)));
}

inline uint32_t PickFp16UnitTransposeRows(gert::TilingContext *context, const ConcatProfile &profile) {
    const uint64_t ubBytes = std::max<uint32_t>(GetUbBytes(context), kAlignBytes);
    const uint64_t usableUbBytes = ubBytes > kPackedNarrowUbReserveBytes ? ubBytes - kPackedNarrowUbReserveBytes
                                                                        : ubBytes;
    const uint64_t paddedInputs = (static_cast<uint64_t>(profile.inputCount) + 15U) / 16U * 16U;
    const uint64_t bytesPerRow = 2U * paddedInputs * profile.elemBytes;
    const uint64_t alignRows = std::max<uint64_t>(16, kAlignBytes / profile.elemBytes);
    uint64_t rows = bytesPerRow == 0 ? 1 : usableUbBytes / bytesPerRow;
    rows = std::max<uint64_t>(alignRows, rows / alignRows * alignRows);
    rows = std::min<uint64_t>(rows, 4080U);
    rows = std::min<uint64_t>(rows, static_cast<uint64_t>(std::max<int64_t>(1, profile.outer)));
    return static_cast<uint32_t>(rows);
}

inline uint32_t PickFp16UnitTransposeCoreBoundaryRows(const ConcatProfile &profile) {
    const uint64_t outputRowBytes = static_cast<uint64_t>(profile.concatDim) * profile.elemBytes;
    if (outputRowBytes == 0) {
        return 1;
    }
    const uint64_t rowAlignment = std::max<uint64_t>(1, kAlignBytes / profile.elemBytes);
    uint64_t rows = std::max<uint64_t>(1, kFp16UnitTransposeCoreBoundaryBytes / outputRowBytes);
    rows = std::max<uint64_t>(rowAlignment, rows / rowAlignment * rowAlignment);
    return static_cast<uint32_t>(std::min<uint64_t>(rows, UINT32_MAX));
}

inline uint32_t PickTileElems(gert::TilingContext *context, const ConcatProfile &profile, int64_t rowsPerTask) {
    const uint64_t maxRowElems = static_cast<uint64_t>(MaxSplit(profile)) * static_cast<uint64_t>(profile.inner);
    if (maxRowElems == 0) {
        return 1;
    }

    const uint64_t alignElems = std::max<uint64_t>(1, kAlignBytes / profile.elemBytes);
    const uint64_t safeRows = static_cast<uint64_t>(std::max<int64_t>(1, rowsPerTask));
    const uint64_t ubBytes = std::max<uint32_t>(GetUbBytes(context), kAlignBytes);
    uint64_t maxTileElems = ubBytes / (static_cast<uint64_t>(profile.elemBytes) * kBufferNum * safeRows);
    maxTileElems = (maxTileElems / alignElems) * alignElems;
    if (maxTileElems == 0) {
        maxTileElems = alignElems;
    }
    maxTileElems = std::min<uint64_t>(maxTileElems, maxRowElems);
    return static_cast<uint32_t>(std::min<uint64_t>(maxTileElems, UINT32_MAX));
}

inline SchedulePlan MakeSchedule(gert::TilingContext *context, const ConcatProfile &profile, uint32_t coreCount) {
    SchedulePlan plan;

    if ((profile.inputCount == 1 || profile.nonZeroCount == 1) && profile.outputElems > 0) {
        plan.mode = kScheduleFlatContiguous;
        plan.rowsPerTask = PickFlatElemsPerTask(context, profile, coreCount);
        plan.workItems = (profile.outputElems + plan.rowsPerTask - 1) / plan.rowsPerTask;
        return plan;
    }

    if (profile.dtype == ge::DT_FLOAT16 && profile.inner == 1 &&
        profile.inputCount >= 4 && profile.inputCount <= 16 && profile.nonZeroCount > 1 &&
        profile.outer >= 16 && OutputBytes(profile) <= 256 * 1024 &&
        profile.concatDim >= 128 && profile.concatDim <= 512 && profile.concatDim % 16 == 0) {
        plan.mode = kScheduleSmallPacked;
        plan.rowsPerTask = AlignUp((4096 + profile.concatDim - 1) / profile.concatDim, 16);
        plan.workItems = (profile.outer + plan.rowsPerTask - 1) / plan.rowsPerTask;
        return plan;
    }
    if (IsFp16UnitTransposeShape(profile)) {
        plan.mode = kScheduleFp16UnitTranspose;
        plan.rowsPerTask = PickFp16UnitTransposeRows(context, profile);
        plan.workItems = coreCount;
        return plan;
    }

    if (profile.outputElems > 0 && IsInputSegmentShape(profile)) {
        const int64_t elemsPerTask = PickInputSegmentElemsPerTask(context, profile);
        const int64_t workItems = CountInputSegmentWorkItems(profile, elemsPerTask);
        if (InputSegmentWorkItemsFit(workItems, coreCount)) {
            plan.mode = kScheduleInputSegments;
            plan.rowsPerTask = elemsPerTask;
            plan.workItems = workItems;
            return plan;
        }
    }

    if (IsPackedNarrowInputShape(profile, coreCount)) {
        const uint32_t rows = PickPackedNarrowRows(context, profile);
        if (rows > 0) {
            plan.mode = kSchedulePackedNarrowInputs;
            plan.rowsPerTask = rows;
            plan.workItems = (profile.outer + rows - 1) / rows;
            return plan;
        }
    }

    if (IsPackedRaggedNarrowInputShape(profile, coreCount)) {
        const uint32_t rows = PickPackedNarrowRows(context, profile);
        if (rows > 0) {
            plan.mode = kSchedulePackedRaggedNarrowInputs;
            plan.rowsPerTask = rows;
            plan.workItems = (profile.outer + rows - 1) / rows;
            return plan;
        }
    }

    if (IsPackedRaggedNarrowTailShape(profile, coreCount)) {
        const uint32_t rows = PickPackedTailRows(context, profile);
        if (rows > 0) {
            plan.mode = kSchedulePackedRaggedNarrowTailInputs;
            plan.rowsPerTask = rows;
            plan.workItems = (profile.outer + rows - 1) / rows;
            return plan;
        }
    }

    if (IsBalancedGroupShape(context, profile, coreCount)) {
        plan.mode = kScheduleBalancedGroups;
        plan.rowsPerTask = PickBalancedGroupRows(context, profile);
        plan.workItems = coreCount;
        return plan;
    }

    if (IsSmallInputDirectRowsShape(profile, coreCount)) {
        const uint32_t rows = PickSmallInputDirectRows(context, profile);
        if (rows > 0) {
            plan.mode = kScheduleSmallInputDirectRows;
            plan.rowsPerTask = rows;
            plan.workItems = coreCount;
            return plan;
        }
    }

    if (IsPackedWideChunkShape(profile, coreCount)) {
        plan.mode = kSchedulePackedWideChunks;
        plan.rowsPerTask = PickPackedWideRows(context, profile);
        const int64_t rowTasks = (profile.outer + plan.rowsPerTask - 1) / plan.rowsPerTask;
        plan.workItems = rowTasks * CountPackedWideChunks(profile);
        return plan;
    }

    if (!PreferMergedTinyRows(profile) && profile.inputCount >= 8 && profile.outer < static_cast<int64_t>(coreCount) &&
        profile.inner == 1) {
        plan.mode = kScheduleInputFullOuter;
        plan.rowsPerTask = profile.outer;
        plan.workItems = profile.inputCount;
        if (profile.inputCount >= 32 && IsTinyOutput(profile)) {
            plan.blockLimit = 8;
        }
        return plan;
    }

    plan.rowsPerTask = PickRowsPerTask(context, profile);
    plan.workItems = (profile.outer + plan.rowsPerTask - 1) / plan.rowsPerTask;

    if (!PreferMergedTinyRows(profile) && profile.outer < static_cast<int64_t>(coreCount) && profile.inputCount > 1 &&
        profile.inner >= 8 && OutputBytes(profile) >= kInputMajorBalancedMinOutputBytes) {
        const uint64_t maxRowBytes = static_cast<uint64_t>(MaxSplit(profile)) * profile.inner * profile.elemBytes;
        plan.mode = OutputBytes(profile) >= kInputMajorBalancedMinOutputBytes &&
                            maxRowBytes >= kInputMajorBalancedMinMaxRowBytes
                        ? kScheduleInputMajorBalanced
                        : kScheduleInputRows;
        plan.rowsPerTask = 1;
        plan.workItems = plan.mode == kScheduleInputMajorBalanced ? coreCount : profile.inputCount * profile.outer;
    }
    return plan;
}

inline void StoreTiling(const ConcatProfile &profile, int64_t workItems, int64_t tasksPerCore,
                        int64_t rowsPerTask, int64_t tileElems, int64_t scheduleMode,
                        optiling::ConcatTilingData &tiling) {
    tiling.set_input_count(profile.inputCount);
    tiling.set_output_elems(profile.outputElems);
    tiling.set_outer_count(profile.outer);
    tiling.set_inner_span(profile.inner);
    tiling.set_concat_dim(profile.concatDim);
    tiling.set_element_bytes(profile.elemBytes);
    tiling.set_work_items(workItems);
    tiling.set_work_items_per_core(tasksPerCore);
    tiling.set_rows_per_task(rowsPerTask);
    tiling.set_tile_elems(tileElems);
    tiling.set_schedule_mode(scheduleMode);
    tiling.set_flat_input_index(profile.nonZeroInput);
    tiling.set_group_count(0);
    uint16_t splits[kMaxInputs] = {};
    for (int64_t idx = 0; idx < kMaxInputs; ++idx) {
        splits[idx] = profile.splits[idx];
    }
    tiling.set_splits(splits);
    if (scheduleMode == kScheduleBalancedGroups) {
        BalancedGroupPlan groupPlan;
        if (BuildBalancedGroupPlan(profile, groupPlan)) {
            tiling.set_group_input_bounds(groupPlan.inputBounds);
            tiling.set_group_output_offsets(groupPlan.outputOffsets);
            tiling.set_group_count(groupPlan.groupCount);
        }
    }
}

}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    ConcatProfile profile;
    if (!LoadProfile(context, profile)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreCount = std::max(1U, platform.GetCoreNumAiv());
    coreCount = std::min<uint32_t>(coreCount, kMaxCores);
    const SchedulePlan plan = MakeSchedule(context, profile, coreCount);
    const int64_t workItems = plan.workItems;
    const int64_t tileElems = plan.mode == kScheduleInputMajorBalanced
                                  ? PickInputSegmentElemsPerTask(context, profile)
                                  : plan.mode == kScheduleFp16UnitTranspose
                                  ? PickFp16UnitTransposeCoreBoundaryRows(profile)
                                  : (plan.mode == kSchedulePackedWideChunks
                                         ? kPackedWideChunkElems
                                         : PickTileElems(context, profile, plan.rowsPerTask));
    uint32_t blockDim = static_cast<uint32_t>(
        std::min<int64_t>(coreCount, std::max<int64_t>(1, workItems)));
    if (plan.blockLimit > 0) {
        blockDim = std::min<uint32_t>(blockDim, static_cast<uint32_t>(plan.blockLimit));
    }
    if (plan.mode == kScheduleInputMajorBalanced && profile.inputCount <= 32 &&
        OutputBytes(profile) <= 16 * 1024 * 1024) {
        const uint64_t coreGroupBytes = 8 * 384 * 1024;
        const uint32_t desiredCores = static_cast<uint32_t>(
            (OutputBytes(profile) + coreGroupBytes - 1) / coreGroupBytes * 8);
        blockDim = std::min(blockDim, std::max(16U, desiredCores));
    }
    const int64_t tasksPerCore = (workItems + static_cast<int64_t>(blockDim) - 1) / static_cast<int64_t>(blockDim);

    ConcatTilingData tiling;
    StoreTiling(profile, workItems, tasksPerCore, plan.rowsPerTask, tileElems, plan.mode, tiling);
    context->SetBlockDim(blockDim);
    uint64_t tilingKey = TilingKeyFor(profile.dtype, plan.mode == kScheduleInputSegments);
    if (plan.mode == kScheduleSmallPacked) {
        tilingKey = 70;
    } else if (plan.mode == kScheduleFp16UnitTranspose) {
        tilingKey = kKeyFp16UnitTranspose;
    } else if (plan.mode == kSchedulePackedNarrowInputs) {
        tilingKey = PackedTilingKeyFor(profile.dtype, false);
    } else if (plan.mode == kSchedulePackedRaggedNarrowInputs) {
        tilingKey = PackedTilingKeyFor(profile.dtype, true);
    } else if (plan.mode == kSchedulePackedRaggedNarrowTailInputs) {
        tilingKey = PackedTailTilingKeyFor(profile.dtype);
    } else if (plan.mode == kSchedulePackedWideChunks) {
        tilingKey = kKeyPackedWideChunksFp32;
    } else if (plan.mode == kScheduleBalancedGroups) {
        tilingKey = tiling.get_group_count() == 5 ? 72 : kKeyBalancedGroups;
    } else if (plan.mode == kScheduleSmallInputDirectRows) {
        tilingKey = kKeySmallInputDirectRows;
    } else if (plan.mode == kScheduleInputMajorBalanced) {
        tilingKey = InputMajorBalancedTilingKeyFor(profile.dtype);
    } else if (plan.mode != kScheduleInputSegments && OutputBytes(profile) >= kGenericInputReadBypassMinOutputBytes) {
        tilingKey = GenericInputReadBypassTilingKeyFor(profile.dtype);
    }

    if (plan.mode == kScheduleSmallInputDirectRows &&
        static_cast<uint64_t>(profile.concatDim) * profile.inner * profile.elemBytes <= UINT32_MAX / kMaxCores) {
        tilingKey = 74;
    }
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    int64_t axis = 0;
    if (!ReadDimAttr(context->GetAttrs(), axis)) {
        return ge::GRAPH_FAILED;
    }

    const size_t inputCount = context->GetComputeNodeInputNum();
    if (inputCount == 0 || inputCount > static_cast<size_t>(kMaxInputs)) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *firstShape = context->GetDynamicInputShape(0, 0);
    if (firstShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rank = firstShape->GetDimNum();
    if (!NormalizeAxis(rank, axis)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape output;
    int64_t concatDim = 0;
    for (size_t idx = 0; idx < inputCount; ++idx) {
        const gert::Shape *shape = context->GetDynamicInputShape(0, idx);
        if (shape == nullptr || shape->GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
            const int64_t dimValue = shape->GetDim(dimIdx);
            if (dimValue < 0) {
                return ge::GRAPH_FAILED;
            }
            if (dimIdx != axis && dimValue != firstShape->GetDim(dimIdx)) {
                return ge::GRAPH_FAILED;
            }
        }
        concatDim += shape->GetDim(axis);
    }

    for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        output.AppendDim(dimIdx == axis ? concatDim : firstShape->GetDim(dimIdx));
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = output;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const size_t inputCount = context->GetComputeNodeInputNum();
    if (inputCount == 0 || inputCount > static_cast<size_t>(kMaxInputs)) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = context->GetDynamicInputDataType(0, 0);
    if (!IsSupportedType(dtype)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t idx = 1; idx < inputCount; ++idx) {
        if (context->GetDynamicInputDataType(0, idx) != dtype) {
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Concat : public OpDef {
  public:
    explicit Concat(const char *name) : OpDef(name) {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("concat_dim").AttrType(OPTIONAL).Int(0);

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}
