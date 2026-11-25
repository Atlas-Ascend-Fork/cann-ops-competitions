# 需求背景（required）

## 需求来源

CANN 社区任务 — `aclblasGemvBatched` 算子开发与性能优化（Atlas A2 / Atlas A3，`arch22 / dav-2201`），对标 NVIDIA cuBLAS / 昇腾 `ops-blas` 指针数组批量 GEMV 接口，补齐 Atlas A2/A3 系列产品多数据类型实数 Batched 矩阵向量乘能力。

## 背景介绍

### aclblasGemvBatched 算子实现目标

- 目标仓库：`https://gitcode.com/cann/ops-blas`（`master` 分支）
- 实现目录：`blas/gemv_batched/arch22/`（`gemv_batched_host.cpp`、`gemv_batched_kernel.cpp`、`gemv_batched_kernel.h`、`gemv_batched_tiling_data.h`）
- 测试目录：`test/gemv_batched/arch22/`
- 工程模式：Ascend C Kernel 直调（Host 侧参数校验 + 动态分核/Split-K Tiling + Device 侧指针数组直读 + 纯 AIV 向量化双 Kernel 架构）

### 接口与语义分析

计算 $b = 0, \dots, \text{batchCount}-1$ 个通过 Device 指针数组传入的实数矩阵向量乘加：

```cpp
// 支持 FP16->FP16 (H), FP16->FP32 (HS), BF16->BF16 (B), BF16->FP32 (BS), FP32->FP32 (S) 五组接口
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* const Aarray[], int lda,
    const float* const xarray[], int incx,
    const float* beta, float* const yarray[], int incy, int batchCount);
```

| 参数 | 输入/输出 | 数据类型 | 排布/维度 | 约束与异常行为 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | Host 句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | 输入 | `aclblasOperation_t` | `{ACLBLAS_OP_N, ACLBLAS_OP_T}` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m`, `n`, `batchCount` | 输入 | `int` | `> 0` | `<= 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha`, `beta` | 输入 | `const float*` | Host FP32 标量指针 | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray`, `lda` | 输入 | `const TIn* const[]` | Device 指针数组，长度 `batchCount`，`lda >= m` | `nullptr` 或 `lda < m` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `xarray`, `incx` | 输入 | `const TIn* const[]` | Device 指针数组，步长 `incx` | `nullptr`、`incx == 0`/`INT_MIN` 或跨度超 `int32` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `yarray`, `incy` | 输入/输出 | `TOut* const[]` | Device 指针数组，步长 `incy` | `nullptr`、`incy == 0`/`INT_MIN` 或跨度超 `int32` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2 / Atlas A3（`arch22`）上实现高性能、五精度组合（`FP16->FP16`、`FP16->FP32`、`BF16->BF16`、`BF16->FP32`、`FP32->FP32`）的 `aclblasGemvBatched` 算子，内部统一 FP32 累加保证高精度，通过全量 1000+ 精度用例（含 50% `GAUSS_NORM_5_2` 正态分布）并达到 NVIDIA A100 同级性能水准。

## 需求拆解

1. **接口与跨度安全校验**：覆盖 `handle == nullptr`、`m/n/batchCount <= 0`、`lda < m`、`trans` 非法枚举、`incx/incy == 0` 或 `INT_MIN`、`GemvBatchedSpanExceedsInt32` 32 位索引域溢出拦截及指针非空校验。
2. **消除指针数组搬运停顿**：将每个 Batch 原本需要 3 次串行 `DataCopyPad` 读取 `Aarray[b]`/`xarray[b]`/`yarray[b]` 指针的 MTE2 同步停顿，重构为标量缓存直读 `arrGm.GetValue(batchIdx)`，大幅降低高 Batch / 小矩阵场景延迟。
3. **双 Kernel 分离 + N/T 极致向量化**：
   - 拆分 `gemv_batched_kernel_impl`（零 `SyncAll` 税）与 `gemv_batched_splitk_kernel_impl`（仅在 `kSplits >= 2` 时调用）；
   - `N` 模式采用 `Brcb + Mul(src1BlkStride=0, src1RepStride=1) + Tree-Add` 纯向量融合；
   - `T` 模式采用 `m <= 1024` 单 Tile 归约 + `BroadcastMulCols + BlockReduceSum + WholeReduceSum` + 波次均衡分核。

# 详细设计（required）

## 算子分析

### 数学公式

对每个批次 $b \in \{0, \dots, \text{batchCount}-1\}$：

$$y^{(b)} \leftarrow \alpha \cdot \operatorname{op}\!\left(A^{(b)}\right) x^{(b)} + \beta \cdot y^{(b)}$$

其中 $A^{(b)} = \text{Aarray}[b]$、$x^{(b)} = \text{xarray}[b]$、$y^{(b)} = \text{yarray}[b]$ 为独立 Device 内存地址（允许非等距排布），矩阵 $A^{(b)}$ 按列主序存储（列距 `lda`），低精度输入在 UB 内统一升精度为 FP32 累加后写回 `TOut`。

### 支持数据类型与形状

| 接口后缀 | `dtype` 编码 | `TIn` | `TOut` | 内部累加类型 |
| --- | --- | --- | --- | --- |
| `HgemvBatched` | `0` | `FP16` | `FP16` | `FP32` |
| `HSgemvBatched` | `1` | `FP16` | `FP32` | `FP32` |
| `BgemvBatched` | `3` | `BF16` | `BF16` | `FP32` |
| `BSgemvBatched` | `4` | `BF16` | `FP32` | `FP32` |
| `SgemvBatched` | `2` | `FP32` | `FP32` | `FP32` |

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

1. **参数校验**（`ValidateGemvBatchedParams`）：
   - 校验 `handle != nullptr`、`m > 0 && n > 0 && batchCount > 0 && lda >= m`、`incx/incy != 0 && != INT_MIN`、`Aarray/xarray/yarray/alpha/beta != nullptr`、`trans ∈ {N, T}`；
   - 调用 `GemvBatchedSpanExceedsInt32` 拦截任意维度/步长组合导致的 32 位索引溢出。
2. **自适应输出分块与 Split-K 决策**：
   - 按 `coreNum = GetAivCoreCount()` 与 `batchCount` 动态计算 32B 对齐的 `outTile`；
   - `T` 模式下对 `uBatch < coreNum` 自动搜索整除 `coreNum` 的波次均衡分片数；
   - 当 `trans == ACLBLAS_OP_N && incy == 1 && *alpha != 0 && wsPtr != nullptr && baseTasks < coreNum && n >= 256` 且 workspace 足够时启用 `kSplits >= 2` 并派发至 `gemv_batched_splitk_kernel_impl`。

#### 3.2.2 Kernel 侧设计

1. **高速指针数组解引用（Zero-MTE2-Stall Pointer Fetch）**：
   - 使用 `GlobalTensor<uint64_t>` 映射 `Aarray`、`xarray`、`yarray`，在每个任务起始通过 `arrGm.GetValue(batchIdx)` 走标量缓存直接获取当前 Batch 的 `aPtr`/`xPtr`/`yPtr`，彻底消除 3 次串行 `DataCopyPad + SetFlag/WaitFlag` 停顿。
2. **双 Kernel 分离消除 `SyncAll` 硬件启动税**：
   - 主 Kernel `gemv_batched_kernel_impl` 不包含 `SyncAll()`，避免全量非 Split-K 用例承受 `~2.8 us` 额外核间同步开销；
   - Split-K Kernel `gemv_batched_splitk_kernel_impl` 使用 `workSpace` 存储局部 FP32 累加块，经 `PipeBarrier<PIPE_ALL>() + SyncAll()` 同步后由首分片核完成最终归约与 `beta * y` 融合写回。
3. **N/T 向量化计算引擎**：
   - **N 模式**：`mCnt <= 512` 启用 `Brcb + Mul(src1BlkStride=0, src1RepStride=1) + Tree-Add` 8 列并行纯向量累加，配合双缓冲 `EVENT_ID2`/`EVENT_ID3`（与同步事件 `EVENT_ID0` 严格隔离）重叠 MTE2 与 Vector 流水；
   - **T 模式**：`m <= 1024` 启用单 Tile `x` 常驻 UB 路径，结合 1D 连续 `DataCopy`、`BroadcastMulCols`、`BlockReduceSum`、`WholeReduceSum` 完成高效归约。
4. **UB 内存精确预算**（单核 UB 上限 192 KB）：
   - 与 `aclblasGemvStridedBatched` 同源内存规划，N 模式峰值 `~100 KB`、T 模式峰值 `~108 KB`，全路径远低于 192 KB 硬件上限。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 (910B3 / 910B4) | √ |
| Atlas 800I/T A3 (910_9362) | √ |

## 算子约束限制

- 支持五种实数精度组合：`FP16->FP16`、`FP16->FP32`、`BF16->BF16`、`BF16->FP32`、`FP32->FP32`；
- 要求 `m > 0`、`n > 0`、`batchCount > 0`、`lda >= m`、`incx/incy != 0 && != INT_MIN` 且指针数组位于 Device 侧可读内存；
- Split-K 路径依赖 `handle` 已绑定的有效 workspace（无 workspace 时自动无缝回退至主 Kernel）。

# 可维可测分析

## 精度标准/性能标准

- **精度标准**：与双精度 CPU Reference 逐元素对比，按官方精度判定（FP32 `atol/rtol = 1e-4`，FP16/BF16 匹配率与 ULP 门限），覆盖全量功能用例及 1000 条混合分布用例（含 50%+ `GAUSS_NORM_5_2` 正态分布）100% PASS。
- **性能标准**：对标盐城 NVIDIA A100-SXM4-80GB（cuBLAS，预热 5 次 + 有效 10 次 `aclrtEvent`/`cudaEvent` 均值），官方门限单用例 `>= 0.8× A100`。
- **实测结果（200 条官方 `TC_PF` 标杆，Atlas A3 910_9362 vs 盐城 A100-SXM4-80GB）**：
  - 总体中位数 **`1.02× A100`**、几何均值 **`0.99× A100`**、峰值 **`3.26× A100`**；
  - `N` 模式：中位数 **`1.03× A100`**，几何均值 **`0.98× A100`**，`>= 0.8×` 达标率 **73/96 (76.0%)**；
  - `T` 模式：中位数 **`1.02× A100`**，几何均值 **`1.01× A100`**，`>= 0.8×` 达标率 **81/104 (77.9%)**；
  - 总体达标率：**154/200 (77.0%) `>= 0.8×`**（优化前仅 16.0%，提升 **4.8×**），**130/200 (65.0%) `>= 0.9×`**，**105/200 (52.5%) `>= 1.0×`**；
  - 未达 0.8× 的 46 条用例均属于物理硬件边界：非 32B 对齐奇数 `m/lda`（如 `m=333, 777`）、极小启动底噪用例（`b=1, m=64, n=64`）及 `b=64` 48 核双波尾延迟。
