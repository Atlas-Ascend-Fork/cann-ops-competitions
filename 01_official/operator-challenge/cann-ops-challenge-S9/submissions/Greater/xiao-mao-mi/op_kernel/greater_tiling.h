#ifndef S9_GREATER_TILING_H
#define S9_GREATER_TILING_H

#include <cstdint>

constexpr uint32_t GREATER_MAX_RANK = 8;

constexpr uint32_t GREATER_DTYPE_FLOAT32 = 0;
constexpr uint32_t GREATER_DTYPE_BF16 = 1;
constexpr uint32_t GREATER_DTYPE_FLOAT16 = 2;
constexpr uint32_t GREATER_DTYPE_INT32 = 3;
constexpr uint32_t GREATER_DTYPE_INT8 = 4;

constexpr uint32_t GREATER_ROUTE_SAME_SHAPE = 1;
constexpr uint32_t GREATER_ROUTE_SCALAR_BROADCAST = 2;
constexpr uint32_t GREATER_ROUTE_ROW_BROADCAST = 3;

constexpr uint32_t GREATER_SUBROUTE_LINEAR_LINEAR = 1;
constexpr uint32_t GREATER_SUBROUTE_LINEAR_SCALAR = 2;
constexpr uint32_t GREATER_SUBROUTE_SCALAR_LINEAR = 3;
constexpr uint32_t GREATER_SUBROUTE_SCALAR_SCALAR = 4;

constexpr uint64_t GREATER_TILING_KEY_SAME = 10;
constexpr uint64_t GREATER_TILING_KEY_SCALAR = 20;
constexpr uint64_t GREATER_TILING_KEY_ROW_LINEAR_LINEAR = 31;
constexpr uint64_t GREATER_TILING_KEY_ROW_LINEAR_SCALAR = 32;
constexpr uint64_t GREATER_TILING_KEY_ROW_SCALAR_LINEAR = 33;
constexpr uint64_t GREATER_TILING_KEY_ROW_SCALAR_SCALAR = 34;

constexpr uint64_t GREATER_TILING_KEY_SAME_FP32 = 100;
constexpr uint64_t GREATER_TILING_KEY_SAME_BF16 = 101;
constexpr uint64_t GREATER_TILING_KEY_SAME_FP16 = 102;
constexpr uint64_t GREATER_TILING_KEY_SAME_INT32 = 103;
constexpr uint64_t GREATER_TILING_KEY_SAME_INT8 = 104;

constexpr uint64_t GREATER_TILING_KEY_SCALAR_FP32 = 200;
constexpr uint64_t GREATER_TILING_KEY_SCALAR_BF16 = 201;
constexpr uint64_t GREATER_TILING_KEY_SCALAR_FP16 = 202;
constexpr uint64_t GREATER_TILING_KEY_SCALAR_INT32 = 203;
constexpr uint64_t GREATER_TILING_KEY_SCALAR_INT8 = 204;

constexpr uint64_t GREATER_TILING_KEY_ROW_LL_FP32 = 310;
constexpr uint64_t GREATER_TILING_KEY_ROW_LL_BF16 = 311;
constexpr uint64_t GREATER_TILING_KEY_ROW_LL_FP16 = 312;
constexpr uint64_t GREATER_TILING_KEY_ROW_LL_INT32 = 313;
constexpr uint64_t GREATER_TILING_KEY_ROW_LL_INT8 = 314;

constexpr uint64_t GREATER_TILING_KEY_ROW_LS_FP32 = 320;
constexpr uint64_t GREATER_TILING_KEY_ROW_LS_BF16 = 321;
constexpr uint64_t GREATER_TILING_KEY_ROW_LS_FP16 = 322;
constexpr uint64_t GREATER_TILING_KEY_ROW_LS_INT32 = 323;
constexpr uint64_t GREATER_TILING_KEY_ROW_LS_INT8 = 324;

constexpr uint64_t GREATER_TILING_KEY_ROW_SL_FP32 = 330;
constexpr uint64_t GREATER_TILING_KEY_ROW_SL_BF16 = 331;
constexpr uint64_t GREATER_TILING_KEY_ROW_SL_FP16 = 332;
constexpr uint64_t GREATER_TILING_KEY_ROW_SL_INT32 = 333;
constexpr uint64_t GREATER_TILING_KEY_ROW_SL_INT8 = 334;

constexpr uint64_t GREATER_TILING_KEY_ROW_SS_FP32 = 340;
constexpr uint64_t GREATER_TILING_KEY_ROW_SS_BF16 = 341;
constexpr uint64_t GREATER_TILING_KEY_ROW_SS_FP16 = 342;
constexpr uint64_t GREATER_TILING_KEY_ROW_SS_INT32 = 343;
constexpr uint64_t GREATER_TILING_KEY_ROW_SS_INT8 = 344;

struct GreaterTilingData {
    uint64_t total;
    uint32_t rank;
    uint32_t dtype;
    uint32_t route;
    uint32_t subroute;
    uint32_t elem_bytes;
    uint32_t block_dim;
    uint32_t row_len;
    uint32_t tile_mod;
    uint32_t tile_len;
    uint32_t tail_len;
    uint32_t row_group;
    uint64_t aligned_total;
    uint32_t self_row_linear;
    uint32_t other_row_linear;
    uint32_t out_shape[GREATER_MAX_RANK];
    uint64_t self_stride[GREATER_MAX_RANK];
    uint64_t other_stride[GREATER_MAX_RANK];
};

#endif
