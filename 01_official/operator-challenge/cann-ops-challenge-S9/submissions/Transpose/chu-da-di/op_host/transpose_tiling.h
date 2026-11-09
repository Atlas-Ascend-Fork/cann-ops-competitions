#ifndef S9_TRANSPOSE_TILING_H
#define S9_TRANSPOSE_TILING_H

#include <cstdint>

namespace s9transpose {
// The published shape uses an ellipsis before N5..N.  Eight dimensions keeps
// the tiling payload compact while covering ordinary PyTorch/ACLNN tensors and
// the explicitly listed rank-six case.
constexpr uint32_t kMaxRank = 8;

enum class Mode : uint32_t {
    kIdentity = 0,
    kContiguousSuffix = 1,
    kSwapLastTwo = 2,
    kGenericOutputOwner = 3,
    kPermutedMatrix = 4,
};

// Keep this a trivial standard-layout type.  The ASC CMake helper serializes
// the bytes written by GetTilingData and exposes the same layout to the kernel.
struct TransposeTilingData {
    uint32_t rank;
    uint32_t mode;
    uint32_t dtypeBytes;
    uint32_t fastSwap16;
    uint32_t rowOwnershipQuantum;
    uint64_t totalElements;
    uint64_t innerElements;
    uint64_t outputBlocks;
    uint64_t tileElements;
    uint64_t outer;
    uint64_t matrixH;
    uint64_t matrixW;
    uint64_t microCols;
    uint64_t tileRows;
    uint64_t tileCols;
    uint64_t tileBatchCols;
    uint64_t columnBatches;
    uint64_t shape[kMaxRank];
    uint64_t outputShape[kMaxRank];
    uint64_t inputStride[kMaxRank];
    int32_t perm[kMaxRank];
};
struct TransposeMacroTilingData {
    TransposeTilingData base;
    uint32_t rows;
    uint32_t cols;
    uint32_t permuted;
    uint32_t reserved; // cache policy bits: 1=input streaming, 2=output streaming
    uint64_t columnStride;
};
struct TransposeColumnOwnerTilingData {
    TransposeTilingData base;
    uint32_t rows;
    uint32_t cols;
    uint64_t totalColumns;
    uint64_t innerMatrices;
    uint64_t columnStride;
    uint32_t cachePolicy; // bit 0: input streaming; bit 1: output streaming
    uint32_t bufferCount; // zero retains the original two-buffer policy
    uint32_t rangeMode; // nonzero partitions complete output-column ranges
    uint32_t reserved;
    uint32_t rowMode, rowA, rowB, rowInner;
};
struct TransposeSuffixBlockHeader {
    TransposeTilingData base;
    uint32_t columns;
    uint32_t inputPitch;
    uint32_t outputPitch;
    uint32_t indexCount;
    uint32_t mapColumns, inputBuffers, outputBuffers, ownershipQuantum;
    uint32_t ownershipMode;
};
struct TransposeSuffixBlockTilingData {
    TransposeSuffixBlockHeader header;
    uint32_t indices[12288];
};
struct TransposeL2StagedHeader {
    TransposeTilingData base;
    uint32_t rows, columns, inputPitch, scratchPitch;
    uint32_t outputRows, rawBuffers, outBuffers, cachePolicy;
    uint64_t slotElements;
    uint32_t batchQuantum, linearCount, packCount, rawElements;
    uint32_t outputElements, logAlignment, reserved0, reserved1;
};
struct TransposeL2StagedTilingData {
    TransposeL2StagedHeader header;
    uint32_t indices[12288];
};
struct TransposeRowAlignedTilingData {
    TransposeColumnOwnerTilingData header;
    uint32_t masks[8192];
};
struct TransposeInt8PairsTilingData {
    uint64_t matrixH, matrixW, outer, totalElements;
    uint32_t tileBatchCols, tileCols, columnBatches, flags;
    uint32_t reserved[4];
};
struct TransposeDirectMaskHeader {
    TransposeTilingData base;
    uint32_t columns, inputPitch, outputPitch, inputRows;
    uint32_t maskWords, cachePolicy, inputBuffers, outputBuffers;
    uint64_t totalColumns;
    uint32_t rawPitch, inputMaskWords, reserved0, reserved1;
    uint32_t reserved2, reserved3, reserved4, reserved5;
};
struct TransposeDirectMaskTilingData {
    TransposeDirectMaskHeader header;
    uint32_t indices[12288];
};
struct TransposeStage {
    uint64_t outer;
    uint64_t matrixH;
    uint64_t matrixW;
    uint64_t tileBatchCols;
};
struct TransposeStagesTilingData {
    TransposeTilingData base;
    uint32_t stageCount;
    uint32_t reserved; // selected Gather stage + 1; zero means none
    uint32_t gatherBatchMatrices;
    uint32_t gatherCount;
    uint32_t wideStage;
    uint32_t wideTileCols;
    uint32_t wideGatherCount;
    uint32_t wideReserved;
    TransposeStage stages[kMaxRank - 1];
};
struct TransposeStagesGatherTilingData {
    TransposeStagesTilingData header;
    uint32_t indices[8192];
    uint32_t wideIndices[4096];
};
}  // namespace s9transpose

#endif  // S9_TRANSPOSE_TILING_H
