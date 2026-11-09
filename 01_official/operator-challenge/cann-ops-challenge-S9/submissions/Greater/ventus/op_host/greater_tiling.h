/**
 * Greater 主机侧与设备侧共享的分块数据结构。
 * 紧凑分块结构与通用结构的公共前缀必须保持一致，供对应 tiling key 解码。
 */

#ifndef S9_GREATER_TILING_H
#define S9_GREATER_TILING_H

#include <cstddef>
#include <cstdint>

constexpr uint32_t S9_GREATER_MAX_RANK = 16;
constexpr uint32_t S9_GREATER_BOX_ELEMENTS_FLOAT16 = 8192U;
constexpr uint32_t S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32 = 5120U;
constexpr uint32_t S9_GREATER_BOX_ELEMENTS_BFLOAT16 = 5120U;
constexpr uint32_t S9_GREATER_BOX_ELEMENTS_INT8 = 9216U;
constexpr uint32_t S9_GREATER_MATRIX_COL_BROADCAST = 1U;
constexpr uint32_t S9_GREATER_MATRIX_ROW_BROADCAST = 2U;
constexpr uint32_t S9_GREATER_ROW_STRIPE_MAX_ROWS = 128U;

struct GreaterRowSingleTilingData {
    uint64_t matrixRows;
    uint64_t matrixCols;
    uint64_t matrixTotalTiles;
    uint32_t matrixRowsPerTile;
    uint32_t matrixX1Pattern;
    uint32_t matrixX2Pattern;
};

struct GreaterSmallDenseTilingData { uint64_t totalElements; };
static_assert(sizeof(GreaterSmallDenseTilingData) == 8U);

struct GreaterCompactPeriodicTilingData {
    uint64_t matrixRows;
    uint64_t matrixCols;
    uint64_t matrixOuterCount;
    uint64_t totalElements;
    uint32_t matrixX1Pattern;
    uint32_t reserved;
};
static_assert(sizeof(GreaterCompactPeriodicTilingData) == 40U);

struct GreaterTilingData {

    uint64_t matrixRows;
    uint64_t matrixCols;
    uint64_t matrixTotalTiles;
    uint32_t matrixRowsPerTile;
    uint32_t matrixX1Pattern;
    uint32_t matrixX2Pattern;

    uint64_t totalElements;
    uint64_t rowElements;
    uint64_t boxElements;
    uint64_t boxCount;
    uint64_t matrixOuterCount;
    uint64_t outputDims[S9_GREATER_MAX_RANK];
    uint64_t x1Dims[S9_GREATER_MAX_RANK];
    uint64_t x2Dims[S9_GREATER_MAX_RANK];
    uint64_t x1Strides[S9_GREATER_MAX_RANK];
    uint64_t x2Strides[S9_GREATER_MAX_RANK];
    uint32_t rank;
    uint32_t outerRank;
    uint32_t blockDim;
    uint32_t x1InnerBroadcast;
    uint32_t x2InnerBroadcast;
    uint32_t boxStart;
    uint32_t boxEnabled;
    uint32_t matrixEnabled;
    uint32_t matrixRowStart;
    uint32_t matrixRowTiles;
    uint32_t rowStripeEnabled;
    uint32_t rowSlabEnabled;
    uint32_t rowSlabCols;
    uint32_t rowSlabRows;
    uint32_t rowSlabUbBytes;
    uint32_t rowSlabBufferNum;
};

static_assert(sizeof(GreaterRowSingleTilingData) == 40U,
              "single-outer row tiling prefix must remain 40 bytes");
static_assert(offsetof(GreaterTilingData, matrixRows) ==
                  offsetof(GreaterRowSingleTilingData, matrixRows) &&
              offsetof(GreaterTilingData, matrixCols) ==
                  offsetof(GreaterRowSingleTilingData, matrixCols) &&
              offsetof(GreaterTilingData, matrixTotalTiles) ==
                  offsetof(GreaterRowSingleTilingData, matrixTotalTiles) &&
              offsetof(GreaterTilingData, matrixRowsPerTile) ==
                  offsetof(GreaterRowSingleTilingData, matrixRowsPerTile) &&
              offsetof(GreaterTilingData, matrixX1Pattern) ==
                  offsetof(GreaterRowSingleTilingData, matrixX1Pattern) &&
              offsetof(GreaterTilingData, matrixX2Pattern) ==
                  offsetof(GreaterRowSingleTilingData, matrixX2Pattern),
              "compact row tiling must be an exact GreaterTilingData prefix");

#endif
