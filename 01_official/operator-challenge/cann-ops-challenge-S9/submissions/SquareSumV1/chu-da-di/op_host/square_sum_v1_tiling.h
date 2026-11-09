#ifndef S9_SQUARE_SUM_V1_TILING_H
#define S9_SQUARE_SUM_V1_TILING_H

#include <cstdint>

namespace s9squaresum {
constexpr uint32_t kMaxRank = 8;

enum class Mode : uint32_t {
    kAR = 0,        // preserved axes, then a contiguous reduced suffix
    kRA = 1,        // contiguous reduced prefix, then preserved axes
    kARA = 2,       // middle or non-contiguous reduction axes
    kReduceAll = 3,
};

constexpr uint64_t kTilingKeyArVector16 = 102;
constexpr uint64_t kTilingKeyArVector32 = 104;
constexpr uint64_t kTilingKeyStridedVector16 = 112;
constexpr uint64_t kTilingKeyStridedVector32 = 114;
constexpr uint64_t kTilingKeyLongArVector16 = 122;
constexpr uint64_t kTilingKeyLongArVector32 = 124;
constexpr uint64_t kTilingKeySplitReduceVector16 = 132;
constexpr uint64_t kTilingKeySplitReduceVector32 = 134;
constexpr uint64_t kTilingKeyReduceAllMultiCore16 = 142;
constexpr uint64_t kTilingKeyReduceAllMultiCore32 = 144;
constexpr uint64_t kTilingKeyPrefixSmallInner16 = 152;
constexpr uint64_t kTilingKeyPrefixSmallInner32 = 154;
struct SquareSumV1TilingData {
    uint32_t rank;
    uint32_t mode;
    uint32_t dtypeBytes;
    uint32_t axisCount;
    uint32_t outputRank;
    uint32_t keepDims;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t totalElements;
    uint64_t outputElements;
    uint64_t outer;
    uint64_t reduce;
    uint64_t inner;
    uint64_t tileElements;
    uint64_t reduceAligned;
    uint64_t rowTile;
    uint64_t shape[kMaxRank];
    uint64_t inputStride[kMaxRank];
    int32_t axes[kMaxRank];
    int32_t nonReduceAxes[kMaxRank];
};
}  // namespace s9squaresum

#endif  // S9_SQUARE_SUM_V1_TILING_H
