# 需求背景（required）

## 需求来源

CANN 社区任务 — `aclblasChpr2` 算子开发与性能优化（Atlas A2 / Atlas A3，`arch22 / dav-2201`），对标 NVIDIA cuBLAS `cublasChpr2` 接口，补齐昇腾 `ops-blas` 仓库 Atlas A2/A3 系列产品 BLAS-2 复数压缩 Hermitian 秩-2 更新能力。

## 背景介绍

### aclblasChpr2 算子实现目标

- 目标仓库：`https://gitcode.com/cann/ops-blas`（`master` 分支）
- 实现目录：`blas/spr2/arch22/`（`chpr2_host.cpp`、`chpr2_kernel.cpp`、`chpr2_kernel.h`、`chpr2_tiling_data.h`）
- 测试目录：`test/spr2/chpr2/arch22/`
- 工程模式：Ascend C Kernel 直调（Host 侧参数校验 + Tiling 下发 + Device 侧纯 AIV 向量化 Kernel）

### 接口与语义分析

对标 cuBLAS `cublasChpr2`（cuBLAS Library §2.6.2）与 Netlib BLAS `chpr2.f`：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* ap);
```

| 参数 | 输入/输出 | 数据类型 | 排布/维度 | 约束与异常行为 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | Host 句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 输入 | `aclblasFillMode_t` | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | 输入 | `int` | `n >= 0` | `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` 校验通过后 quick return |
| `alpha` | 输入 | `const aclblasComplex*` | Host 标量指针（COMPLEX64） | `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`alpha == (0,0)` 短路返回 |
| `x` | 输入 | `const aclblasComplex*` | Device 向量，物理长度 `1 + (n-1)*|incx|` | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | 输入 | `int` | 非零且 `!= INT_MIN` | `0`、`INT_MIN` 或 `(n-1)*|incx| > INT_MAX` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | 输入 | `const aclblasComplex*` | Device 向量，物理长度 `1 + (n-1)*|incy|` | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incy` | 输入 | `int` | 非零且 `!= INT_MIN` | `0`、`INT_MIN` 或 `(n-1)*|incy| > INT_MAX` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ap` | 输入/输出（原地） | `aclblasComplex*` | Device 压缩三角矩阵，长度 `n*(n+1)/2` | `n > 0` 时 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2 / Atlas A3（`arch22`）上实现高性能 `aclblasChpr2` 算子，完整支持 `ACLBLAS_UPPER` / `ACLBLAS_LOWER`、任意正负步长 `incx`/`incy`、全值域含 `Inf`/`NaN` 的 IEEE 复数语义，并通过全量 1000+ 精度用例与 200 条 `TC_PF` 性能标杆对比（`>= 0.8×` NVIDIA A100）。

## 需求拆解

1. **接口与校验完备**：覆盖 `handle == nullptr`、`n < 0`、`uplo` 非法、`incx/incy == 0` 或 `INT_MIN`、`(n-1)*|inc| > INT_MAX` 跨度溢出、`alpha/x/y/ap == nullptr`、`n == 0` / `alpha == (0,0)` 快速返回。
2. **Hermitian 语义精确**：对角元素虚部必须强制清零（`Im(A[j,j]) = 0`），且当 `alpha.imag == 0` 输入含 `Inf` 时避免 `0.0f * inf -> NaN` 污染（`TC_FL_151/152` 专项修复）。
3. **高性能 AIV 向量化**：通过蛇形列分核（Snake-Striped Column Distribution）实现三角负载绝对均衡，单核内使用交错复数向量 `Muls + Axpy` 融合更新。

# 详细设计（required）

## 算子分析

### 数学公式

$$A \leftarrow A + \alpha \cdot x y^H + \bar{\alpha} \cdot y x^H$$

其中 $A$ 为 $n \times n$ 复数 Hermitian 矩阵，以列主序压缩三角格式存储在长度 $n(n+1)/2$ 的一维数组 `ap` 中：
- `ACLBLAS_UPPER`：列 $j$ 存储行 $0 \dots j$，起始偏移 $\text{colBase}(j) = j(j+1)/2$；
- `ACLBLAS_LOWER`：列 $j$ 存储行 $j \dots n-1$，起始偏移 $\text{colBase}(j) = j(2n - j + 1)/2$；
- 对角线元素 $A_{jj}$ 恒取实部更新且虚部置 `0.0f`。

### 支持数据类型与形状

- 数据类型：`aclblasComplex`（单精度复数 `COMPLEX64`，实虚交错存储，8 字节/元素）。
- 形状支持：任意合法 $n \ge 0$（运行时 Tiling，静态 Kernel 一次适配），支持正负任意非零步长 `incx`、`incy`。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

1. **参数校验层级**（`ValidateChpr2Body`）：
   - 先验 `handle != nullptr`（否则 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）；
   - 再验 `n >= 0`、`alpha != nullptr`、`uplo` 合法、`incx/incy != 0 && != INT_MIN`、`x/y/ap != nullptr`、`int64` 跨度 `(n-1)*|inc| <= INT_MAX` 及 `size_t` 字节长度无溢出；
   - `n == 0` 或 `alpha == (0, 0)` 直接返回 `ACLBLAS_STATUS_SUCCESS`。
2. **分核策略**：
   - 获取物理 AIV 核数 `aivCoreNum = GetAivCoreCount()`（910B4 为 40 核，Atlas A3 单 Die 为 48 核）；
   - 启动核数 `numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum)`，充分利用全芯片 AIV 算力与 HBM 带宽。
3. **Tiling 数据结构**（`Chpr2TilingData`）：
   - 传递 `n`、`uplo`、`numBlocks`、`alphaReal`、`alphaImag`、`incx`、`incy`，无动态 workspace 需求（0 字节额外 GM 开销）。

#### 3.2.2 Kernel 侧设计

1. **蛇形负载均衡分核（Snake-Striped Column Partitioning）**：
   - 压缩三角矩阵各列长度呈线性递增/递减（`1..n`）。第 `blockIdx` 核在第 `ord` 轮处理的列号为：
     `j = ord * numBlocks + ((ord & 1) ? (numBlocks - 1 - blockIdx) : blockIdx)`
   - 使每个 AIV 核承担的三角列总元素数近乎完全相等（不均衡度 `< 1%`）。
2. **Tile 化向量复用与交错复数乘加**：
   - 沿行维以 `CHPR2_ARCH22_TILE_ROWS = 1024` 分块；每块加载 `x[tileStart..tileEnd)` 和 `y[tileStart..tileEnd)` 一次，并预构建共轭虚部交换表 `xsprime = (-x_im, x_re)`、`ysprime = (-y_im, y_re)`。
   - 对列 $j$，标量预计算复系数 $t_1 = \alpha \bar{y}_j$、$t_2 = \overline{\alpha x_j}$；当 `alphaImag == 0` 时走无零乘分支（避免 `0.0f * inf = NaN`）。
   - 列段更新 `ComputeInterleavedMul`：用 `Muls(dst, raw, tr) + Axpy(dst, sprime, ti)`（当 `ti == 0` 跳过 `Axpy`）一次性完成交错复数向量乘加，并与 `ap` 的 `DataCopyPad` 搬入重叠执行。
   - 对角元素单独计算实部并强制 `imag = 0.0f` 后写回 GM。
3. **UB 内存精确预算**（单核 UB 上限 192 KB）：
   - `bufXraw_` / `bufXsprime_` / `bufYraw_` / `bufYsprime_`：`4 × 8 KB = 32 KB`
   - `bufAP_` / `bufD1_` / `bufD2_` / `bufWork_`：`4 × 8 KB = 32 KB`
   - 索引表 `bufShiftIdx_` / `bufSplitIdx_` / `bufPackIdx_`：`8 + 4 + 8 = 20 KB`
   - 标量/步长窗口 `bufScalar_` (256 B) + `bufWin_`（仅 `inc != 1` 时分配 32 KB）
   - **峰值 UB 占用**：单位步长 `84.3 KB`，非单位步长 `116.3 KB`（远低于 192 KB 上限）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 (910B3 / 910B4) | √ |
| Atlas 800I/T A3 (910_9362) | √ |

## 算子约束限制

- 仅支持 `aclblasComplex`（COMPLEX64）单精度复数；
- 要求 `incx != 0 && incx != INT_MIN`、`incy != 0 && incy != INT_MIN` 且 `(n-1)*|inc| <= INT_MAX`；
- 异步下发至 `handle->stream`，调用方读取结果前需调用 `aclrtSynchronizeStream`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 实测结论 |
| --- | --- | --- |
| 精度标准 | `mixed_tolerance`（FLOAT32: `atol=rtol=2^-13`, `matched_ratio>=0.99`, `max_abs_error<=max(1e-2, 32*ULP)`，含 Inf/NaN 等价规则） | 910B4 全量 **1003/1003 PASS**（含 `TC_FL_151/152` Inf 用例修复） |
| 性能标准 | 200 条官方 `TC_PF` 用例，单条耗时 `<= A100 / 0.8`（对标 NVIDIA A100-SXM4-80GB，warmup 5 + 有效采样取均值） | 910B4: **200/200 (100.0%) >= 0.8×**，191/200 >= 1.1×，几何均值 **4.04× A100**（最高 **9.50×**）；Atlas A3: 192/200 >= 0.8×，几何均值 **3.89× A100** |
| 内存标准 | 不涉及（0 字节额外 workspace） | 满足 |

## 兼容性分析

完全复用 `ops-blas` 仓 `include/cann_ops_blas.h` 既有 `aclblasChpr2` 接口声明，与 `arch35` 兄弟实现保持参数语义与返回码 100% 一致，无破坏性变更。
