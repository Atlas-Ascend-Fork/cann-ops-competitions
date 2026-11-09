#include "transpose_tiling.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint64_t kContiguousChunkBytes = 32 * 1024;
constexpr uint64_t kScalarChunkElements = 2048;

bool SafeMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

bool ReadPermutation(const gert::RuntimeAttrs *attrs, uint32_t rank,
                     int64_t *permutation)
{
    const gert::TypedContinuousVector<int64_t> *dims =
        attrs == nullptr ? nullptr : attrs->GetListInt(0);
    if (dims == nullptr || dims->GetSize() != rank) {
        return false;
    }
    bool seen[TRANSPOSE_MAX_DIMS] = {};
    const int64_t *data = dims->GetData();
    for (uint32_t axis = 0; axis < rank; ++axis) {
        int64_t inputAxis = data[axis];
        if (inputAxis < 0) {
            inputAxis += rank;
        }
        if (inputAxis < 0 || inputAxis >= rank || seen[inputAxis]) {
            return false;
        }
        seen[inputAxis] = true;
        permutation[axis] = inputAxis;
    }
    return true;
}

bool PrepareTiling(const gert::Shape &shape, const gert::RuntimeAttrs *attrs,
                   uint32_t elementBytes,
                   CompetitionTransposeTilingData &tiling)
{
    const size_t rankValue = shape.GetDimNum();
    if (rankValue == 0 || rankValue > TRANSPOSE_MAX_DIMS || elementBytes == 0) {
        return false;
    }
    tiling.rank = static_cast<uint32_t>(rankValue);
    tiling.elementBytes = elementBytes;

    int64_t permutation[TRANSPOSE_MAX_DIMS] = {};
    if (!ReadPermutation(attrs, tiling.rank, permutation)) {
        return false;
    }

    uint64_t inputStridesByAxis[TRANSPOSE_MAX_DIMS] = {};
    uint64_t inputElements = 1;
    for (uint32_t reverse = 0; reverse < tiling.rank; ++reverse) {
        const uint32_t axis = tiling.rank - 1 - reverse;
        const int64_t extent = shape.GetDim(axis);
        if (extent <= 0) {
            return false;
        }
        tiling.inputDims[axis] = static_cast<uint64_t>(extent);
        inputStridesByAxis[axis] = inputElements;
        if (!SafeMultiply(inputElements, static_cast<uint64_t>(extent),
                          inputElements)) {
            return false;
        }
    }
    tiling.totalElements = inputElements;

    for (uint32_t axis = 0; axis < tiling.rank; ++axis) {
        const uint32_t inputAxis = static_cast<uint32_t>(permutation[axis]);
        tiling.outputDims[axis] = tiling.inputDims[inputAxis];
        tiling.inputStrides[axis] = inputStridesByAxis[inputAxis];
    }

    // Remove singleton axes, then merge adjacent axes contiguous in both layouts.
    uint32_t compactRank = 0;
    for (uint32_t axis = 0; axis < tiling.rank; ++axis) {
        if (tiling.outputDims[axis] == 1) continue;
        if (compactRank && tiling.inputStrides[compactRank - 1] ==
            tiling.outputDims[axis] * tiling.inputStrides[axis]) {
            tiling.outputDims[compactRank - 1] *= tiling.outputDims[axis];
            tiling.inputStrides[compactRank - 1] = tiling.inputStrides[axis];
        } else {
            tiling.outputDims[compactRank] = tiling.outputDims[axis];
            tiling.inputStrides[compactRank++] = tiling.inputStrides[axis];
        }
    }
    if (!compactRank) {
        compactRank = 1;
        tiling.outputDims[0] = tiling.inputStrides[0] = 1;
    }
    tiling.rank = compactRank;
    uint64_t contiguous = 1;
    uint64_t expectedStride = 1;
    for (uint32_t reverse = 0; reverse < tiling.rank; ++reverse) {
        const uint32_t axis = tiling.rank - 1 - reverse;
        const uint64_t extent = tiling.outputDims[axis];
        if (extent == 1) {
            continue;
        }
        if (tiling.inputStrides[axis] != expectedStride) {
            break;
        }
        if (!SafeMultiply(contiguous, extent, contiguous) ||
            !SafeMultiply(expectedStride, extent, expectedStride)) {
            return false;
        }
    }
    tiling.contiguousElements = contiguous;
    // Original input dimensions have now been consumed. Reserve slot 5 for
    // the streaming pipeline cache policy without changing the tiling ABI.
    tiling.inputDims[5] = 0;
    tiling.inputDims[4] = 0;
    tiling.inputDims[6] = 0;
    tiling.inputDims[8] = 0;

    bool lastTwoSwap = tiling.rank >= 2;
    for (uint32_t axis = 0; axis + 2 < tiling.rank; ++axis) {
        lastTwoSwap = lastTwoSwap && permutation[axis] == axis;
    }
    if (lastTwoSwap) {
        lastTwoSwap = permutation[tiling.rank - 2] ==
                          static_cast<int64_t>(tiling.rank - 1) &&
                      permutation[tiling.rank - 1] ==
                          static_cast<int64_t>(tiling.rank - 2);
    }

    const uint64_t contiguousChunkElements = std::max<uint64_t>(
        1, kContiguousChunkBytes / elementBytes);
    // Batch contiguous blocks with strided DMA; no vector reorder is necessary.
    if (contiguous * elementBytes >= 32 && contiguous <= contiguousChunkElements &&
        inputElements != contiguous &&
        !(inputElements * elementBytes >= 1048576 && contiguous <= 1024 &&
          contiguous * elementBytes % 32 != 0)) {
        uint32_t fastAxis = 0;
        uint64_t outputStride = 1;
        for (uint32_t a = 0; a + 1 < tiling.rank; ++a)
            if (tiling.inputStrides[a] == contiguous) fastAxis = a;
        for (uint32_t a = fastAxis + 1; a < tiling.rank; ++a)
            outputStride *= tiling.outputDims[a];
        if (outputStride * elementBytes <= std::numeric_limits<uint32_t>::max()) {
            const uint64_t groupBytes = contiguous * elementBytes;
            const uint32_t rowAxis = tiling.rank - 2;
            if (groupBytes % 32 == 0 && groupBytes <= 4096 && inputElements >= 65536 &&
                tiling.inputStrides[rowAxis] * elementBytes <= std::numeric_limits<uint32_t>::max()) {
                uint32_t height = std::max<uint64_t>(1, 512 / groupBytes);
                while (height > 1 && tiling.outputDims[rowAxis] <= height / 2) height /= 2;
                const uint32_t width = std::min<uint64_t>(128, 32768 / height / groupBytes);
                tiling.strategy = 7;
                tiling.tileSide = height;
                tiling.inputDims[0] = fastAxis;
                tiling.inputDims[1] = outputStride;
                tiling.inputDims[2] = contiguous;
                tiling.inputDims[3] = rowAxis;
                tiling.inputDims[4] = width;
                if (inputElements * elementBytes >= 1048576 &&
                    outputStride * elementBytes % 128 == 0 &&
                    tiling.inputStrides[rowAxis] * elementBytes % 128 == 0)
                    tiling.inputDims[5] = 1;
                tiling.taskCount = inputElements / contiguous / tiling.outputDims[rowAxis] /
                    tiling.outputDims[fastAxis] * CeilDiv(tiling.outputDims[rowAxis], height) *
                    CeilDiv(tiling.outputDims[fastAxis], width);
                return true;
            }
            tiling.strategy = 5;
            tiling.inputDims[0] = fastAxis;
            tiling.inputDims[1] = outputStride;
            const uint64_t paddedBytes = CeilDiv(contiguous * elementBytes, 32) * 32;
            tiling.tileSide = std::min<uint64_t>(128, kContiguousChunkBytes / paddedBytes);
            tiling.taskCount = inputElements / contiguous / tiling.outputDims[fastAxis] *
                CeilDiv(tiling.outputDims[fastAxis], tiling.tileSide);
            return true;
        }
    }
    // A short unchanged suffix is a block of values, not millions of DMA tasks.
    if (contiguous <= 1024 && inputElements != contiguous) {
        const uint32_t rowAxis = tiling.rank - (contiguous > 1 ? 2 : 1);
        uint32_t fastAxis = 0;
        uint64_t outputStride = 1;
        for (uint32_t a = 0; a < tiling.rank; ++a)
            if (tiling.inputStrides[a] == contiguous) fastAxis = a;
        for (uint32_t a = fastAxis + 1; a < tiling.rank; ++a)
            outputStride *= tiling.outputDims[a];
        if (fastAxis != rowAxis &&
            outputStride * elementBytes <= std::numeric_limits<uint32_t>::max() &&
            tiling.inputStrides[rowAxis] * elementBytes <= std::numeric_limits<uint32_t>::max()) {
            tiling.strategy = 4;
            uint32_t side = 64;
            const uint32_t capacity = contiguous == 1 ? 4096 :
                (contiguous * elementBytes < 32 ?
                    (elementBytes == 1 ? 8192 : 32768 / elementBytes) : 16384);
            while (side * side * contiguous > capacity) side /= 2;
            tiling.tileSide = side;
            tiling.inputDims[0] = fastAxis;
            tiling.inputDims[1] = outputStride;
            tiling.inputDims[2] = contiguous;
            tiling.inputDims[3] = rowAxis;
            const uint64_t rows = tiling.outputDims[rowAxis];
            const uint64_t cols = tiling.outputDims[fastAxis];
            if (contiguous > 1 && contiguous * elementBytes >= 32) {
                const uint32_t height = std::min<uint64_t>(rows, side * 2);
                uint32_t width = std::min<uint64_t>(cols,
                    std::min<uint64_t>(64, capacity / height / contiguous));
                while (width > 1 &&
                    (height * CeilDiv(width * contiguous * elementBytes, 32) * 32 / elementBytes > capacity ||
                     width * CeilDiv(height * contiguous * elementBytes, 32) * 32 / elementBytes > capacity)) --width;
                tiling.tileSide = height;
                tiling.inputDims[4] = width;
                tiling.taskCount = inputElements / contiguous / rows / cols *
                    CeilDiv(rows, height) * CeilDiv(cols, width);
                return true;
            }
            if (contiguous == 1 && inputElements >= (elementBytes == 4 ? 1048576 : 65536)) {
                tiling.strategy = 6;
                uint32_t height = rows > 64 ? 128 : (rows > 32 ? 64 : 32);
                if (elementBytes >= 2 && rows <= 16) height = 16;
                if (elementBytes == 1 && rows == 16 && outputStride == 16) height = 16;
                uint32_t width = 128;
                if (inputElements >= 1048576 &&
                    ((elementBytes == 2 && rows == height) ||
                     (rows == 16 && height == 16)))
                    width = (elementBytes == 2 ? 32768 : 16384) / height;
                if (elementBytes == 2 && rows >= 256 && inputElements >= 1048576) {
                    height = 256;
                    width = 128;
                    if (rows >= 512 && outputStride * elementBytes % 32 == 0 &&
                        tiling.inputStrides[rowAxis] * elementBytes % 32 == 0 &&
                        outputStride * elementBytes % 128 != 0 &&
                        tiling.inputStrides[rowAxis] * elementBytes % 128 != 0) {
                        height = 512;
                        width = 64;
                    }
                }
                if (elementBytes == 4 && inputElements >= 1048576 && rows >= 256 &&
                    (outputStride * elementBytes % 512 != 0 ||
                     tiling.inputStrides[rowAxis] * elementBytes % 512 != 0)) {
                    height = 256;
                    width = 64;
                    // Narrow loads help when both pitches split 128-byte lines,
                    // but each row remains 32-byte DMA aligned. Keep ragged and
                    // mixed-alignment layouts on the established 256x64 path.
                    if (inputElements >= 16777216 && rows >= 4096 && cols >= 16 &&
                        outputStride * elementBytes % 32 == 0 &&
                        tiling.inputStrides[rowAxis] * elementBytes % 32 == 0 &&
                        outputStride * elementBytes % 128 != 0 &&
                        tiling.inputStrides[rowAxis] * elementBytes % 128 != 0) {
                        height = 1024;
                        width = 16;
                    }
                }
                // Native b8 needs no half intermediates, so a full 64 KB tile fits.
                if (elementBytes == 1 && inputElements >= 33554432 && rows >= 256 && cols >= 256) {
                    height = 256;
                    width = 256;
                    tiling.inputDims[5] |= 4;
                }
                tiling.tileSide = height;
                while (width > 32 && cols <= width / 2) width /= 2;
                tiling.inputDims[4] = width;
                if (inputElements * elementBytes >= 1048576 &&
                    outputStride * elementBytes % 128 == 0 &&
                    tiling.inputStrides[rowAxis] * elementBytes % 128 == 0)
                    tiling.inputDims[5] |= 1;
                tiling.taskCount = inputElements / rows / cols *
                    CeilDiv(rows, height) * CeilDiv(cols, width);
                // Merge a second input-contiguous axis into the tile's columns.
                // Output batches may be separated: the kernel writes each back
                // using its own output stride instead of assuming adjacency.
                // The fast output axis need not directly precede the row axis:
                // batchStride and outputStride retain both independent pitches.
                if (inputElements >= 1048576 &&
                    ((rows >= 128 && rows % 128 == 0) ||
                     ((rows == 16 || rows == 32 || rows == 64) &&
                      (elementBytes != 1 || rows >= 32 || outputStride == rows))) &&
                    tiling.inputStrides[rowAxis] * elementBytes % 128 == 0 &&
                    (cols == 8 || cols == 16 || cols == 32 || cols == 64)) {
                    for (uint32_t a = 0; a < tiling.rank; ++a) {
                        if (a == rowAxis || tiling.inputStrides[a] != cols) continue;
                        uint64_t batchStride = 1;
                        for (uint32_t b = a + 1; b < tiling.rank; ++b) batchStride *= tiling.outputDims[b];
                        tiling.inputDims[6] = a + 1;
                        tiling.inputDims[8] = batchStride;
                        height = elementBytes == 2 && rows % 256 == 0 ? 256 : 128;
                        width = 128;
                        // Short output rows otherwise dispatch many tiny tiles.
                        // Reuse batch-aware stores, keeping byte stores DMA-safe.
                        if (rows < 128) {
                            height = static_cast<uint32_t>(rows);
                            width = rows == 16 || elementBytes < 4 ? 16384 / height : 128;
                        }
                        tiling.tileSide = height;
                        tiling.inputDims[4] = width;
                        tiling.inputDims[5] = 1;
                        const uint64_t mergedCols = cols * tiling.outputDims[a];
                        tiling.taskCount = inputElements / rows / mergedCols *
                            CeilDiv(rows, height) * CeilDiv(mergedCols, width);
                        break;
                    }
                }
                return true;
            }
            tiling.taskCount = inputElements / contiguous / rows / cols *
                CeilDiv(rows, side) * CeilDiv(cols, side);
            return true;
        }
    }
    if (contiguous > 1) {
        tiling.strategy = 0;
        tiling.chunksPerRow = CeilDiv(contiguous, contiguousChunkElements);
        if (inputElements * elementBytes >= 1048576) {
            tiling.strategy = 8;
            tiling.chunksPerRow = CeilDiv(contiguous, 65536 / elementBytes);
        }
        tiling.taskCount = (inputElements / contiguous) * tiling.chunksPerRow;
    } else {
        uint32_t fastAxis = 0;
        uint64_t outputStride = 1;
        for (uint32_t a = 0; a < tiling.rank; ++a)
            if (tiling.inputStrides[a] == 1) fastAxis = a;
        for (uint32_t a = fastAxis + 1; a < tiling.rank; ++a)
            outputStride *= tiling.outputDims[a];
        if (fastAxis != tiling.rank - 1 &&
            outputStride * elementBytes <= std::numeric_limits<uint32_t>::max() &&
            tiling.inputStrides[tiling.rank - 1] * elementBytes <= std::numeric_limits<uint32_t>::max()) {
            tiling.strategy = 4;
            tiling.tileSide = 64;
            // Reuse inputDims for tile metadata; output-axis descriptors remain intact.
            tiling.inputDims[0] = fastAxis;
            tiling.inputDims[1] = outputStride;
            tiling.inputDims[2] = 1;
            tiling.inputDims[3] = tiling.rank - 1;
            const uint64_t rows = tiling.outputDims[tiling.rank - 1];
            const uint64_t cols = tiling.outputDims[fastAxis];
            tiling.taskCount = inputElements / rows / cols *
                CeilDiv(rows, 64) * CeilDiv(cols, 64);
            return true;
        }
        const uint64_t lastExtent = tiling.outputDims[tiling.rank - 1];
        const uint64_t stepBytes = tiling.inputStrides[tiling.rank - 1] *
                                   static_cast<uint64_t>(elementBytes);
        if (lastExtent > 1 && stepBytes >= elementBytes &&
            stepBytes - elementBytes <= std::numeric_limits<uint32_t>::max()) {
            tiling.strategy = 2;
            tiling.taskCount = inputElements / lastExtent;
        } else {
            tiling.strategy = 3;
            tiling.taskCount = CeilDiv(inputElements, kScalarChunkElements);
        }
    }
    return tiling.taskCount > 0;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *storage = context->GetInputShape(0);
    const gert::CompileTimeTensorDesc *desc = context->GetInputDesc(0);
    if (storage == nullptr || desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int32_t elementBytes = ge::GetSizeByDataType(desc->GetDataType());
    CompetitionTransposeTilingData tiling{};
    if (elementBytes <= 0 ||
        !PrepareTiling(storage->GetStorageShape(), context->GetAttrs(),
                       static_cast<uint32_t>(elementBytes), tiling)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t coreCount = static_cast<uint64_t>(
        std::max<int32_t>(1, platform.GetCoreNumAiv()));
    context->SetBlockDim(static_cast<uint32_t>(
        std::min<uint64_t>(coreCount, tiling.taskCount)));

    auto *raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetCapacity() < sizeof(tiling)) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &tiling, sizeof(tiling));
    raw->SetDataSize(sizeof(tiling));
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *input = context->GetInputShape(0);
    gert::Shape *output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    const uint32_t rank = static_cast<uint32_t>(input->GetDimNum());
    int64_t permutation[TRANSPOSE_MAX_DIMS] = {};
    if (rank == 0 || rank > TRANSPOSE_MAX_DIMS ||
        !ReadPermutation(context->GetAttrs(), rank, permutation)) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        output->SetDim(axis, input->GetDim(permutation[axis]));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> types = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("input").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("output").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Attr("dims").ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);
} // namespace ops
