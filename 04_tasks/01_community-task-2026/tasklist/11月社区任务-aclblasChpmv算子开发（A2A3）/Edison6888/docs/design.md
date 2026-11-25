# 需求背景（required）

## 需求来源

CANN 社区任务 — `aclblasChpmv` 算子开发与性能优化（Atlas A2 / Atlas A3，`arch22 / dav-2201`），对标 NVIDIA cuBLAS `cublasChpmv` 接口，补齐昇腾 `ops-blas` 仓库 Atlas A2/A3 系列产品 BLAS-2 复数压缩 Hermitian 矩阵-向量乘能力。

## 背景介绍

### aclblasChpmv 算子实现目标

- 目标仓库：`https://gitcode.com/cann/ops-blas`（`master` 分支）
- 实现目录：`blas/spmv/arch22/`（`chpmv_host.cpp`、`chpmv_kernel.cpp`、`chpmv_kernel.h`、`chpmv_tiling_data.h`）
- 测试目录：`test/spmv/chpmv/arch22/`
- 工程模式：Ascend C Kernel 直调（Host 侧参数校验 + 动态路径分发 + Device 侧纯 AIV 向量化 Kernel）

### 接口与语义分析

对标 cuBLAS `cublasChpmv`（cuBLAS Library §2.6.2）与 Netlib BLAS `chpmv.f`：

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* ap,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

| 参数 | 输入/输出 | 数据类型 | 排布/维度 | 约束与异常行为 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | Host 句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 输入 | `aclblasFillMode_t` | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | 输入 | `int` | `n >= 0` | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 快速返回 |
| `alpha` / `beta` | 输入 | `const aclblasComplex*` | Host 标量指针（COMPLEX64） | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`alpha==0, beta==1` 快速返回 |
| `ap` | 输入 | `const aclblasComplex*` | Device 压缩三角矩阵，长度 `n*(n+1)/2` | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | 输入 | `const aclblasComplex*` | Device 向量，物理长度 `1 + (n-1)*|incx|` | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` / `incy` | 输入 | `int` | 非零且 `!= INT_MIN` | `0`、`INT_MIN` 或 `(n-1)*|inc| > INT_MAX` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | 输入/输出（原地） | `aclblasComplex*` | Device 向量，物理长度 `1 + (n-1)*|incy|` | `beta == 0` 时不读旧值直接覆写 |

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2 / Atlas A3（`arch22`）上实现高性能 `aclblasChpmv` 算子，支持上下三角压缩存储、任意正负步长、对角虚部忽略语义，并在全量 1003 条精度用例与 200 条官方 `TC_PF` 性能用例上全面超越 NVIDIA A100 `0.8×` 门槛。

## 需求拆解

1. **三路分级调度**：
   - `alpha == (0,0)`：`LaunchChpmvScaleOnly` 短路缩放（不访问 `ap`/`x`）；
   - `n <= 160`：`LaunchChpmvSmall` 单核融合路径（消除两阶段启动开销）；
   - `n > 160`：`LaunchChpmvTwoStage`（Accum + Reduce）高吞吐两阶段路径。
2. **消除两阶段性能瓶颈**：
   - 彻底消除循环内 GM 标量 `ReadXScalar` / `AccumScalarAt` 读写与 `ZeroWorkspaceSlice`；
   - 消除 `LOWER` 下三角非 32B 对齐时的 `V_S`/`S_V` 标量回退；
   - 引入 `bufIo0_`/`bufIo1_` 双缓冲 Ping-Pong 隐藏 HBM 访存延迟。

# 详细设计（required）

## 算子分析

### 数学公式

$$y \leftarrow \alpha \cdot A x + \beta \cdot y$$

其中 $A = A^H$ 为 $n \times n$ Hermitian 矩阵，压缩存储于 `ap`。利用对称性，仅需读取一次压缩三角矩阵 `ap`，每列 $j$ 同时完成：
1. **对角项**：$y_j \mathrel{+}= \text{Re}(A_{jj}) \cdot x_j$（严格不读取对角虚部）；
2. **非对角 AXPY**：$y_i \mathrel{+}= A_{ij} \cdot x_j$（所有 $i \in \text{off}(j)$）；
3. **非对角共轭点积 DOT**：$y_j \mathrel{+}= \sum_{i \in \text{off}(j)} \overline{A_{ij}} \cdot x_i$。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

1. **参数校验**（`ValidateChpmvInput`）：逐项校验 `handle`、`n`、`uplo`、`incx`/`incy`（含 `INT_MIN` 与 `(n-1)*|inc| <= INT_MAX` 溢出守卫）及空指针。
2. **Workspace 规划**：
   - 两阶段路径下，每核分配 `bytesPerCore = 2 * CeilAlign(n, 8) * sizeof(float)` 的独立累加片，**额外预留 1 片（索引 `accumBlocks`）用于存放各核独占列的 `colDot` 归约结果**；
   - `maxWsCores = wsSize / bytesPerCore - 1`，`accumBlocks = min(n, aivCoreNum, maxWsCores)`。
3. **分核策略**：
   - 小尺寸（`n <= 160`）：单 Kernel 多核行并行（`incy == 1 && n > 16` 时每核处理 4 行，最多 `aivCoreNum` 核；`n <= 4` 走极简标量快路径）；
   - 中大尺寸（`n > 160`）：Accum 阶段 `accumBlocks` 核蛇形列并行，Reduce 阶段 `min(CeilDiv(n, 1024), aivCoreNum)` 核跨片归约并完成 `alpha`/`beta` 融合写回。

#### 3.2.2 Kernel 侧设计

1. **UB 常驻 `colX` / `colDot` 与零清零 Workspace**：
   - Accum 核启动时一次性预载本核负责列（最多 `1024` 列，支持 `n <= 40960`）的 `x[j]` 及对角贡献 `diagRe * x[j]` 至 UB（`bufColXRe_/Im_`、`bufColDotRe_/Im_`）；
   - Tile 循环内跳过 `offLo >= offHi` 的空交集列，AXPY 直接累加至 `bufAccRe_/Im_`，DOT 结果直接累加至 UB `colDot`，彻底移除 `ZeroWorkspaceSlice`、`ReadXScalar`、`AccumScalarAt` 与 tile 回读；
   - 全 Tile 结束后，各核将独占列 `colDot` 一次性写入第 `accumBlocks` 号 workspace 片，Reduce 核遍历 `c = 0..accumBlocks` 完成最终汇总。
2. **向量对齐位移消除 `LOWER` 瓶颈**：
   - 当 `localOff = offLo - tileStart` 非 8-float 对齐（`h = localOff & 7 != 0`）时，利用 `Adds + Maxs` 构造位移索引表，在 `Gather` 解交错实虚部的同时完成右移 `h` 位并 `Duplicate(0.0f, h)` 清零前导位；
   - 使 `aRe`/`aIm` 与 `bufAcc[alignedOff]`、`bufX[alignedOff]` 100% 32B 对齐，彻底消除 `AddVectorToAcc` 标量回退与 `xSeg` 二次 `Gather`。
3. **MTE2 / Vector 双缓冲 Ping-Pong**：
   - 使用 `bufIo0_`/`bufIo1_`（各 8 KB）及独立硬件事件 `EVENT_ID0`/`EVENT_ID1` 预取下一非空列的 `ap` 段，实现访存与复数计算并行重叠。
4. **UB 内存精确预算**（单核 UB 上限 192 KB）：
   - `ChpmvAccumAIV`：`bufXRe/Im` + `bufAccRe/Im` (`16.9 KB`) + `bufIo0/1` (`16 KB`) + `bufARe/Im` + `bufT1..T4` (`25.3 KB`) + `bufSplitIdx/TmpIdx` (`8.5 KB`) + `bufColX/Dot` (`16 KB`) + `Red/Scalar` (`2.5 KB`) = **85.2 KB**（含非单位步长 `bufWin_` 共 **117.2 KB << 192 KB**）。
   - `ChpmvSmallAIV`（`n <= 160`）：`bufAp_` (`103.2 KB`) + 辅助向量表 (`~18 KB`) = **~121 KB << 192 KB**。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 (910B3 / 910B4) | √ |
| Atlas 800I/T A3 (910_9362) | √ |

## 算子约束限制

- 支持 `aclblasComplex`（COMPLEX64）；
- 两阶段路径依赖 `handle` 内置 workspace（默认 64 MB，满足 `n <= 40960`）；
- 要求 `incx/incy != 0 && != INT_MIN` 且 `(n-1)*|inc| <= INT_MAX`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 实测结论 |
| --- | --- | --- |
| 精度标准 | `mixed_tolerance`（FLOAT32: `atol=rtol=2^-13`, `matched_ratio>=0.99`, `max_abs_error<=max(1e-2, 32*ULP)`） | 910B4 全量 **1003/1003 PASS** |
| 性能标准 | 200 条官方 `TC_PF` 用例，对标 NVIDIA A100-SXM4-80GB（`>= 0.8×`） | Atlas A3: **187/200 (93.5%) >= 0.8×**，153/200 >= 1.1×，几何均值 **1.36× A100**（`n > 348` 全量 100% 过线，最高 **2.26×**）；910B4: **187/200 >= 0.8×**，几何均值 **1.34× A100** |
| 内存标准 | 复用 handle 默认 workspace，无额外泄漏 | 满足 |

## 兼容性分析

复用 `ops-blas` 仓 `include/cann_ops_blas.h` 既有 `aclblasChpmv` 声明，与 `arch35` 兄弟实现完全兼容。
