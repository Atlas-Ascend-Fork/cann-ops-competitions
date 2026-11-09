#ifndef S9_GREATER_TILING_H
#define S9_GREATER_TILING_H

#include <cstdint>

constexpr uint32_t S9_GREATER_MAX_RANK = 8;
constexpr uint32_t S9_GREATER_GATHER_CAP = 4096;

struct GreaterTilingData {
    uint64_t totalElements;
    uint64_t blockElements;
    uint64_t tileElements;
    uint32_t rank;
    uint32_t mode; // 0 same, 1 generic, 2/3 scalar, 4/5 run, 6/7 scalar suffix, 8 shared, 9..12 short repeat, 13/14 crossed cartesian suffix, 15/16 grouped periodic (keys 12/13)
    // Ordinary keys use broadcastRepeat; explicit-L2 keys use the same otherwise-unused word as the input alias offset.
    union { uint64_t broadcastRepeat; uint64_t inputL2BypassOffset; };
    uint64_t broadcastInner;
    uint64_t outputShape[S9_GREATER_MAX_RANK];
    uint64_t x1Shape[S9_GREATER_MAX_RANK];
    uint64_t x2Shape[S9_GREATER_MAX_RANK];
    uint64_t x1Stride[S9_GREATER_MAX_RANK];
    uint64_t x2Stride[S9_GREATER_MAX_RANK];
};

// Metadata is copied to scalar registers; shape-derived maps travel by DMA.
struct GreaterGatherMeta {
    GreaterTilingData base;
    uint32_t span;
    uint32_t lhsCount;
    uint32_t rhsCount;
    uint32_t paddedSpan;
    uint64_t groupSpan;
    uint32_t steps;
    uint32_t inner;
    uint32_t lhsStep;
    uint32_t rhsStep;
    uint64_t splitAxis;
    uint32_t mapKinds[2];
    uint32_t divisors[2];
    uint32_t reciprocals[2];
    uint32_t shifts[2];
};
static_assert(sizeof(GreaterGatherMeta) % 32 == 0, "index maps must be DMA aligned");
struct GreaterGatherTilingData {
    GreaterGatherMeta meta;
    uint32_t lhsMap[S9_GREATER_GATHER_CAP];
    uint32_t rhsMap[S9_GREATER_GATHER_CAP];
};

struct GreaterRowTilingData {
 GreaterTilingData base;
 uint64_t width,repeat,totalRows,blockRows;
 uint32_t padded,rows,leftRepeated,reserved;
};
static_assert(sizeof(GreaterRowTilingData)%32==0,"row tiling alignment");

struct GreaterFlatRowTilingData { GreaterRowTilingData meta; uint32_t indices[8192]; };
static_assert(sizeof(GreaterFlatRowTilingData)<65536,"flat row metadata capacity");

struct GreaterTinyTilingData { uint64_t total; };



struct GreaterL2Footer {
 uint64_t inputL2BypassOffset;
 uint64_t magic;
 uint64_t deviceIndex;
 uint64_t reserved;
};
struct GreaterL2DenseTilingData { GreaterTilingData data; GreaterL2Footer l2; };
struct GreaterL2FlatTilingData { GreaterFlatRowTilingData data; GreaterL2Footer l2; };
static_assert(sizeof(GreaterL2Footer)==32,"L2 footer size");
static_assert(sizeof(GreaterL2DenseTilingData)==sizeof(GreaterTilingData)+32,"dense prefix unchanged");
static_assert(sizeof(GreaterL2FlatTilingData)==sizeof(GreaterFlatRowTilingData)+32,"flat prefix unchanged");

#endif
