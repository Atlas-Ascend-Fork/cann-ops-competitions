# aclblasChbmv 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务 2026，任务书《aclblasChbmv 算子开发任务书》。目标代码仓为昇腾算子开源仓 `ops-blas`（https://gitcode.com/cann/ops-blas ），实现代码目录为 `blas/sbmv/arch22/`（与实数同族算子 `aclblasSsbmv` 同目录族），测试代码目录为 `test/sbmv/chbmv/arch22/`。

本任务为补齐 Blas 仓能力：在 Atlas A2/A3 系列产品上使用 Ascend C 编程语言开发单精度复数（complex64）Hermitian 带状矩阵-向量乘算子 `aclblasChbmv`，完成算子设计、开发、测试全流程。验收通过后合入昇腾算子开源仓。

### 背景介绍

`aclblasChbmv` 对标 cuBLAS `cublasChbmv` 与 Netlib `chbmv`，用于计算 Hermitian 带状矩阵与向量的乘积：

```text
y = alpha * A * x + beta * y
```

其中 `A` 为 n×n 复数 Hermitian 带状矩阵，包含 k 条次对角线与 k 条超对角线，按列主序带状格式压缩存储（大小为 lda×n），仅存储 `uplo` 指定的带形三角。未存储的三角由 Hermitian 共轭对称性隐含补全，主对角线元素的虚部假定为 0。`x`、`y` 为复数向量，`alpha`、`beta` 为复数标量。

当前 ops-blas 仓 `include/cann_ops_blas.h` 中尚无 `aclblasChbmv` 的声明与实现。本任务需新增接口声明，并在 `arch22` 目录内使用 Ascend C kernel 直调方式实现该算子，复用同族 `sbmv`（对称带状矩阵向量乘）的工程框架，完成 A2/A3 平台的适配与评测。

## 需求分析（required）

### 需求描述

在 Atlas A2/A3 上使用 Ascend C 实现 `aclblasChbmv`，接口签名、参数语义与 cuBLAS `cublasChbmv` 严格对齐，支持单精度复数（COMPLEX64）、上/下三角带状存储模式（`uplo`）、正负步长（`incx`/`incy`）、标量特殊值及 quick return 语义。需满足任务书规定的参数校验、边界行为、精度标准、性能标准及自验要求。

### 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 在 `include/cann_ops_blas.h` 新增 `aclblasChbmv` 声明，与其它产品线共用，不定义私有平行接口 | 代码审查；头文件比对 |
| R2 | 实现 `y = alpha * A * x + beta * y`，支持 UPPER/LOWER 带状存储 | 功能用例；golden 比对 |
| R3 | 正确解析 Hermitian 带状存储，仅引用指定三角；对角元素只取实部参与计算 | 代码审查；对角特殊值用例 |
| R4 | 边界行为：`n = 0` 为 quick return；`beta = (0,0)` 不读 y 旧值；`alpha = (0,0)` 时按 `y = beta * y` 计算 | 边界用例 |
| R5 | 参数校验及返回码符合任务书 §2.4/§2.5，特别关注防溢出处理（lda/incx/incy） | 负向用例；状态码断言 |
| R6 | 精度满足 COMPLEX64 实部/虚部 FLOAT32 判定标准，含 Inf/NaN 特殊值规则 | 自测报告；`cblas_chbmv` (fp64) golden |
| R7 | 性能达到 GPU 标杆 0.8 倍以上（平均单次耗时 ≤ GPU 标杆 1.25 倍） | msprof op 采样；自测报告 |
| R8 | 自验需覆盖正态分布补充用例（`GAUSS_NORM_5_2`）及全量边界负向场景 | 测试代码 + 自测报告 |

## 详细设计（required）

### 算子分析

#### 数学公式

```text
y = alpha * A * x + beta * y
```

对第 i 个元素（0-indexed）：

```text
y[i] = alpha * Σ_{j=max(0, i-k)}^{min(n-1, i+k)} A[i][j] * x[j] + beta * y[i]
```

#### 接口定义

在 `include/cann_ops_blas.h` 新增：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

#### 支持数据类型与形状

- 数据类型：COMPLEX64（`aclblasComplex`，实部/虚部各 FLOAT32）。
- 维度：`n ≥ 0`，`k ≥ 0`，有效带宽 `min(k, n-1)`。
- 前导维：`lda ≥ k+1`，且需在 int64 域计算防溢出。
- 步长：`incx ≠ 0`，`incy ≠ 0`，支持负步长，拒绝 `INT32_MIN`。

#### 带状存储解析

A 矩阵大小为 n×n，有效带宽为 k，物理存储大小为 `lda × n`，按列主序存储。

*   **UPLO = UPPER**：主对角存于第 k 行（0-indexed），第 j 列的元素 `A(i,j)` (i ≤ j) 存于 `A[k + i - j][j]`。左上角 k×k 区域不引用。
*   **UPLO = LOWER**：主对角存于第 0 行，第 j 列的元素 `A(i,j)` (i ≥ j) 存于 `A[i - j][j]`。右下角 k×k 区域不引用。

未存储的另一侧由 Hermitian 对称性补全：`A[i][j] = conj(A[j][i])`。特别地，主对角线元素 `A[i][i]` 的虚部假定为 0。

### 算子实现

#### Host 侧设计

1.  **参数校验**：
    *   `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
    *   `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 直接返回 `ACLBLAS_STATUS_SUCCESS` (quick return)。
    *   `k < 0`，`lda < k+1` 返回 `ACLBLAS_STATUS_INVALID_VALUE`（`k+1` 转 int64 防溢出）。
    *   `incx`/`incy` 为 0 或 `INT32_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；校验 `(n-1)*abs(incx)` 是否在 int32 范围内防溢出。
    *   指针 `alpha`/`beta`/`A`/`x`/`y` 判空。
    *   `uplo` 非法返回 `ACLBLAS_STATUS_INVALID_VALUE`（注：按任务书要求，与 `aclblasCsymv` 的 `INVALID_ENUM` 不同）。
2.  **Tiling 策略**：
    *   按行切分 y 向量的计算，将 n 个元素的计算任务均匀分配至多核。由于带状矩阵每一行的非零元素个数最多为 2k+1，数据量较小，优先考虑使用 Vector 核进行处理。
    *   对于正负步长：若 `incx < 0` 或 `incy < 0`，在 Host 侧计算反向起点偏移，并在 Kernel 内翻转遍历顺序，确保寻址正确。
    *   Host 侧将带宽 k、维度 n、存储格式 uplo 以及 alpha、beta 的值通过 TilingData 下发至 Kernel。

#### Kernel 侧设计

进行 Init 和 Process 两个阶段，Process 包含搬入、计算、搬出。

1.  **数据搬入**：
    *   计算第 i 行的 y 时，非零元素 `A[i][j]` 的 j 范围是 `[max(0, i-k), min(n-1, i+k)]`。
    *   根据 `uplo` 和 `j` 与 `i` 的关系，计算对应的带状数组实际物理偏移。若 `A[i][j]` 位于未存储侧，则需从对应的 `A[j][i]` 读取并求共轭。
    *   读取 `x[j]` 时需按 `incx` 偏移读取。
    *   读取主对角线元素时，强制将其虚部置 0。
2.  **复数乘加计算**：
    *   在 Vector 核内将读取的 `A` 和 `x` 进行复数乘法（实部/虚部分离计算），并对所有有效 j 进行累加。
    *   计算完 `alpha * A*x` 后，读取旧 `y[i]`（若 `beta != (0,0)`），按 `beta * y[i] + accum` 计算新值。
    *   若 `alpha = (0,0)`，则跳过 A 和 x 读取，直接计算 `beta * y[i]`。
3.  **数据搬出**：
    *   将计算得到的 `y[i]` 根据 `incy` 步长写回。注意原地更新。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备：Atlas 800T A2 / 910B3） | √ |
| Atlas A3 系列产品 | √ |

实现目录为 `blas/sbmv/arch22/`；README 产品支持表标注 Atlas A2/A3：支持。

### 算子约束限制

- 仅支持 COMPLEX64 类型；仅支持列主序。
- 数据存储必须遵循带状格式，且 `lda ≥ k+1`。
- `incx`/`incy` 不可为 0 或 `INT32_MIN`，不支持超出步长语义的非连续访问。
- `n=0` 为合法 no-op 返回 SUCCESS。
- 非法 `uplo` 统一返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 合入代码遵循 `CONTRIBUTING.md`，不得包含冗余文件。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 Netlib `cblas_chbmv`（fp64 真值计算后降精度至 float32）生成。COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol=atol=2^-13、required_matched_ratio=0.99、max_abs_error_limit = max(1e-2, 32*ULP)。支持 Inf/NaN 等效判定规则。需自行补充 `GAUSS_NORM_5_2` 正态分布用例。 | 任务书 §3.2 |
| 性能标准 | Atlas 800T A2 (910B3)，COMPLEX64 平均单次耗时 ≤ GPU 标杆 1.25 倍（≥0.8x GPU）。msprof op 5次 warmup，≥10次有效样本平均。 | 任务书 §3.3 |
| 内存标准 | 不涉及。 | 任务书 §3.4 |

### 兼容性分析

- 本算子为 ops-blas 仓内**新增接口**与**新增实现**：在 `include/cann_ops_blas.h` 新增 `aclblasChbmv` 声明，在 `blas/sbmv/arch22/` 新增实现。
- 复用同族 `sbmv` 框架，不改动既有框架默认行为，不影响其它产品线（如 950PR 的 arch35 路径）。
- 测试代码放在 `test/sbmv/chbmv/arch22/`，包含 CSV 及复现脚本。
- 接口签名、参数序列与 cuBLAS 对齐，供其他产品线共用，无破坏性变更。