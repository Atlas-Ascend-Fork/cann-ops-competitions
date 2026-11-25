# 需求背景（required）

## 需求来源

CANN 社区任务 — `aclblasGemvStridedBatched` 算子开发与性能优化（Atlas A2 / Atlas A3，`arch22 / dav-2201`），对标 NVIDIA cuBLAS / 昇腾 `ops-blas` 批量步长偏移 GEMV 接口，补齐 Atlas A2/A3 系列产品多数据类型实数 StridedBatched 矩阵向量乘能力。

## 背景介绍

### aclblasGemvStridedBatched 算子实现目标

- 目标仓库：`https://gitcode.com/cann/ops-blas`（`master` 分支）
- 实现目录：`blas/gemv_strided_batched/arch22/`（`gemv_strided_batched_host.cpp`、`gemv_strided_batched_kernel.cpp`、`gemv_strided_batched_kernel.h`、`gemv_strided_batched_tiling_data.h`）
- 测试目录：`test/gemv_strided_batched/arch22/`
- 工程模式：Ascend C Kernel 直调（Host 侧参数校验 + 动态分核/Split-K Tiling + Device 侧纯 AIV 向量化双 Kernel 架构）

### 接口与语义分析

计算 $b = 0, \dots, \text{batchCount}-1$ 个独立的实数矩阵向量乘加：

```cpp
// 支持 FP16->FP16 (H), FP16->FP32 (HS), BF16->BF16 (B), BF16->FP32 (BS), FP32->FP32 (S) 五组接口
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* a, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

| 参数 | 输入/输出 | 数据类型 | 排布/维度 | 约束与异常行为 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | Host 句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | 输入 | `aclblasOperation_t` | `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m`, `n`, `batchCount` | 输入 | `int` | `>= 0` | `< 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；任一为 `0` 返回 `ACLBLAS_STATUS_SUCCESS` |
| `alpha`, `beta` | 输入 | `const float*` | Host FP32 标量指针 | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `a`, `lda`, `strideA` | 输入 | `const TIn*` | 列主序，$m \times n$，`lda >= max(1, m)`，`strideA > 0` | 非法值或 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x`, `incx`, `stridex` | 输入 | `const TIn*` | 逻辑长度 `N` 模式为 `n`、`T/C` 模式为 `m` | `incx == 0`/`INT_MIN`、`(xLen-1)*|incx| > INT_MAX`、`stridex <= 0` 或 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y`, `incy`, `stridey` | 输入/输出 | `TOut*` | 逻辑长度 `N` 模式为 `m`、`T/C` 模式为 `n` | `incy == 0`/`INT_MIN`、`(yLen-1)*|incy| > INT_MAX`、`stridey <= 0` 或 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2 / Atlas A3（`arch22`）上实现高性能、五精度组合（`FP16->FP16`、`FP16->FP32`、`BF16->BF16`、`BF16->FP32`、`FP32->FP32`）的 `aclblasGemvStridedBatched` 算子，内部统一 FP32 累加保证高精度，通过全量 1000+ 精度用例（含 50% `GAUSS_NORM_5_2` 正态分布）并逼近 NVIDIA A100 性能标杆。

## 需求拆解

1. **五精度与全步长完备**：覆盖 FP16/BF16/FP32 输入输出组合、任意正负非零 `incx/incy`、`(len-1)*|inc| <= INT_MAX` 跨度防溢出校验、`alpha == 0` / `beta == 0` 精确处理。
2. **零 `SyncAll` 发射开销的双 Kernel 分离**：将普通路径与 Split-K 归约路径拆分为两个独立 `__global__` Kernel，避免 `SyncAll()` 符号存在导致全量普通用例额外付出 `~2.8 us` 硬件同步启动税。
3. **N/T 双模式极致 AIV 向量化**：
   - `N` 模式（非转置）：采用 **`Brcb + Mul(src1BlkStride=0, src1RepStride=1) + Tree-Add`** 纯向量矩阵列加权归约，消除传统逐列 `GetValue` + `Axpy` 的标量发射瓶颈；
   - `T` 模式（转置）：采用 **1D 连续 `DataCopy` + `BroadcastMulCols` + `BlockReduceSum` + `WholeReduceSum`**，并在 `m <= 1024` 时单 Tile 直接归约消除外层循环开销。

# 详细设计（required）

## 算子分析

### 数学公式

对每个批次 $b \in \{0, \dots, \text{batchCount}-1\}$：

$$y^{(b)} \leftarrow \alpha \cdot \operatorname{op}\!\left(A^{(b)}\right) x^{(b)} + \beta \cdot y^{(b)}$$

其中：
- $A^{(b)}$ 起始地址为 `a + b * strideA`，列主序存储，第 $j$ 列偏移 `j * lda`；
- $x^{(b)}$ 起始地址为 `x + b * stridex`，步长 `incx`（负步长按 Netlib 规则逆序索引 `(-incx) * (len - 1 - k)`）；
- $y^{(b)}$ 起始地址为 `y + b * stridey`，步长 `incy`；
- 所有低精度（FP16/BF16）输入在 UB 内 Cast 为 FP32 累加，写回时再 Cast 为目标类型 `TOut`。

### 支持数据类型与形状

| 接口后缀 | `dtype` 编码 | `TIn` | `TOut` | 内部累加类型 |
| --- | --- | --- | --- | --- |
| `HgemvStridedBatched` | `0` | `FP16` | `FP16` | `FP32` |
| `HSgemvStridedBatched` | `1` | `FP16` | `FP32` | `FP32` |
| `BgemvStridedBatched` | `3` | `BF16` | `BF16` | `FP32` |
| `BSgemvStridedBatched` | `4` | `BF16` | `FP32` | `FP32` |
| `SgemvStridedBatched` | `2` | `FP32` | `FP32` | `FP32` |

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

1. **参数校验**（`ValidateGemvSbParams`）：
   - 严格校验 `handle`、`m/n/batchCount >= 0`（任一为 0 时直接 quick return）、`trans`、`lda >= max(1, m)`、`strideA/stridex/stridey > 0`、`incx/incy != 0 && != INT_MIN`、`int64` 跨度 `(len-1)*|inc| <= INT_MAX` 以及非空指针。
2. **动态输出分块与波次均衡（Wave-Balanced Tiling）**：
   - `incy == 1` 时按 AIV 物理核数 `coreNum` 与 `batchCount` 自适应切分输出维度 `outTile`（32B 对齐：`FP32` 对齐 8、`FP16/BF16` 对齐 16）。
   - `T` 模式下若 `uBatch < coreNum` 且总任务数不能整除 `coreNum`，自动在 `cand <= 6` 内搜索整除 `coreNum` 的最优分片数，消除尾波空闲核。
3. **条件 Split-K 派发**：
   - 当 `trans == ACLBLAS_OP_N && incy == 1 && *alpha != 0 && wsPtr != nullptr && baseTasks < coreNum && n >= 256` 且 workspace 足够容纳 `baseTasks * kSplits * wsTileStride * 4` 字节时，启用 `kSplits >= 2` 并派发至独立 Kernel `gemv_strided_batched_splitk_kernel_impl`；否则派发至零同步开销主 Kernel `gemv_strided_batched_kernel_impl`。

#### 3.2.2 Kernel 侧设计

1. **双 Kernel 分离消除 `SyncAll` 发射税**：
   - 主 Kernel `gemv_strided_batched_kernel_impl` 不含任何 `SyncAll()` 指令，小/中批量用例（`b=1..64`）端到端启动延迟直降 `~2.8 us`。
   - Split-K Kernel `gemv_strided_batched_splitk_kernel_impl` 在各 K 分片完成部分和写入 `workSpace` 后调用 `PipeBarrier<PIPE_ALL>() + SyncAll()`，再由各 `baseTask` 的 `splitId == 0` 核完成跨分片累加与 `beta * y` 融合写回。
2. **N 模式纯向量 `Brcb + Mul + Tree-Add` 流水**：
   - 当 `mCnt <= 512` 且每次批处理列块 `curCols = min(8, rem)` 时，先用 `Brcb` 将 8 个 `x` 标量广播到 `xBrcb`（每个 32B block 放 1 个标量），再配合 2D/1D 双缓冲（`evM2V0`/`evM2V1` 专用 `EVENT_ID2`/`EVENT_ID3`，同步等待独立使用 `EVENT_ID0` 杜绝硬件事件 ID 冲突）将 `curCols` 列 `A` 搬入 UB，通过 `Mul(..., src1BlkStride=0, src1RepStride=1)` 一次向量指令完成 8 列同步乘加，随后树形折半 `Add` 归约至 `acc`。
   - 对 32B 对齐的 `lda == m` 或 `lda % align == 0` 场景自动走 1D 连续 `DataCopy`，最大化 MTE2 带宽。
3. **T 模式宽行单 Tile 归约与 `InitYAcc`/`StoreStridedVec` 外提**：
   - `m <= 1024` 时整个 `x` 向量一次驻留 UB，直接完成 `BroadcastMulCols + BlockReduceSum + WholeReduceSum` 并批量写回 `y`，消除重复搬运与标量回写开销。
   - `InitYAcc` 与 `StoreStridedVec` 针对 `incy == 1`、`beta == 0`、`beta == 1` 做特化分支，跳过多余读取与乘法。
4. **UB 内存精确预算**（单核 UB 上限 192 KB）：
   - N 模式：`bufA0/bufA1`（双缓冲 64 KB）+ `bufX`（4 KB）+ `bufAcc`（8 KB）+ `bufTmp/bufXBrcb`（16 KB）+ `bufY`（8 KB），峰值 `~100 KB`；
   - T 模式：`bufA0/bufA1`（双缓冲 64 KB）+ `bufX`（8 KB）+ `bufProd/bufRed`（32 KB）+ `bufY`（4 KB），峰值 `~108 KB`，全路径留有 `> 80 KB` 安全余量。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 (910B3 / 910B4) | √ |
| Atlas 800I/T A3 (910_9362) | √ |

## 算子约束限制

- 支持五种实数精度组合：`FP16->FP16`、`FP16->FP32`、`BF16->BF16`、`BF16->FP32`、`FP32->FP32`；
- 要求 `lda >= max(1, m)`、`strideA/stridex/stridey > 0`、`incx/incy != 0 && != INT_MIN` 且 `(len-1)*|inc| <= INT_MAX`；
- Split-K 路径依赖 `handle` 已绑定的有效 workspace（无 workspace 时自动无缝回退至主 Kernel，不影响正确性）。

# 可维可测分析

## 精度标准/性能标准

- **精度标准**：与双精度 CPU Reference 逐元素对比，按官方精度判定（FP32 `atol/rtol = 1e-4`，FP16/BF16 匹配率与 ULP 门限），覆盖全量功能用例及 1000 条混合分布用例（含 50%+ `GAUSS_NORM_5_2` 正态分布）100% PASS。
- **性能标准**：对标盐城 NVIDIA A100-SXM4-80GB（cuBLAS，预热 5 次 + 有效 10 次 `aclrtEvent`/`cudaEvent` 均值），官方门限单用例 `>= 0.8× A100`。
- **实测结果（200 条官方 `TC_PF` 标杆，Atlas A3 910_9362 vs 盐城 A100-SXM4-80GB）**：
  - 总体中位数 **`0.99× A100`**、几何均值 **`0.99× A100`**、峰值 **`3.32× A100`**；
  - `N` 模式：中位数 **`1.04× A100`**，几何均值 **`1.01× A100`**，`>= 0.8×` 达标率 **76/96 (79.2%)**；
  - `T` 模式：中位数 **`0.98× A100`**，几何均值 **`0.97× A100`**，`>= 0.8×` 达标率 **77/104 (74.0%)**；
  - 总体达标率：**153/200 (76.5%) `>= 0.8×`**（优化前仅 19.0%，提升 **4.0×**），**133/200 (66.5%) `>= 0.9×`**，**97/200 (48.5%) `>= 1.0×`**；
  - 未达 0.8× 的 47 条用例均属于物理硬件边界：非 32B 对齐奇数 `m/lda`（如 `m=333, 777`，无法走 1D 连续搬运且实测 UB `Gather` 反慢 2×）、极小启动底噪用例（`b=1, m=64, n=64`：A3 `7.3 us` vs A100 `5.5 us`）及 `b=64` 48 核双波尾延迟。
