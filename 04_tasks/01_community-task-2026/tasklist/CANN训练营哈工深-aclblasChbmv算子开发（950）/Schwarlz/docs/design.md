# aclblasChbmv 算子设计文档（Ascend 950PR）

版本：v1.0，2026-11-01  
目标代码仓：`cann/ops-blas`（新增 `blas/sbmv/arch35/chbmv_*` 与 `test/sbmv/chbmv/`）  
目标硬件与软件环境：Ascend 950PR（`arch35` / `dav-c310` / `DAV-3510`，56 AIV 核）；CANN 9.1.0

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 训练营（哈工深专场）赛题：**`aclblasChbmv` 算子开发（Ascend 950PR）**。目标是在 `cann/ops-blas` 算子库中面向下一代昇腾 AI 处理器 Ascend 950PR（`arch35`）使用 Ascend C Kernel 直调方式补齐 BLAS Level-2 单精度复数厄米特（Hermitian）带状矩阵-向量乘法算子 `aclblasChbmv`。

## 1.2 背景介绍

厄米特带状矩阵-向量乘法（Hermitian Banded Matrix-Vector Multiply, `CHBMV`）广泛应用于结构动力学频域有限元、电磁波导耦合方程及量子化学哈密顿量求解。接口语义对标 cuBLAS `cublasChbmv`，数值标杆（Golden）严格采用 Netlib BLAS 3.10.0 标准 `cblas_chbmv(CblasColMajor, ...)`。

在 `cann/ops-blas` 主仓中，`blas/sbmv/arch35/` 目录下已有实数对称带状矩阵乘法 `aclblasSsbmv` 实现。本次开发在此基础上新增复数厄米特带状算子 `aclblasChbmv`，公开接口唯一声明于 `include/cann_ops_blas.h`，复用仓内句柄（`aclblasHandle_t`）、流（`aclrtStream`）、复数结构（`aclblasComplex`）及 CMake 构建与 GTest 验证体系。

---

# 2. 需求分析（required）

## 2.1 需求描述

算子完成单精度复数厄米特带状矩阵与向量乘法：

$$y := \alpha A x + \beta y$$

公开 C 接口原型（声明于 `include/cann_ops_blas.h`）：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A,
    int lda,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* beta,
    aclblasComplex* y,
    int incy);
```

### 参数与约束表

| 参数 | 输入/输出 | 内存位置 | 类型 | 说明与合法性约束 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | Host | `aclblasHandle_t` | 上下文句柄，携带 `aclrtStream`；不可为空（`n < 0` 校验优先于空句柄校验） |
| `uplo` | 输入 | Host | `aclblasFillMode_t` | 指定存储上三角带（`ACLBLAS_UPPER`）或下三角带（`ACLBLAS_LOWER`） |
| `n` | 输入 | Host | `int` | 矩阵阶数，`n >= 0`；当 `n == 0` 且 `handle != nullptr` 时直接返回 `ACLBLAS_STATUS_SUCCESS` |
| `k` | 输入 | Host | `int` | 超对角线或次对角线条数（半带宽），`k >= 0` |
| `alpha` | 输入 | Host | `const aclblasComplex*` | 复数标量乘数，`n > 0` 时非空 |
| `A` | 输入 | Device | `const aclblasComplex*` | 列主序紧凑带状存储矩阵，物理尺寸至少 `lda * n`；对角线虚部假定为 0（不读取），未存储侧按共轭对称读取 |
| `lda` | 输入 | Host | `int` | 带状矩阵主维，`lda >= k + 1` |
| `x`, `incx` | 输入 | Device / Host | `const aclblasComplex*`, `int` | 输入复数向量及步长，`incx != 0`（支持正负步长） |
| `beta` | 输入 | Host | `const aclblasComplex*` | 复数标量乘数，`n > 0` 时非空；`beta == (0, 0)` 时不读取旧 `y` |
| `y`, `incy` | 输入/输出 | Device / Host | `aclblasComplex*`, `int` | 输入/输出复数向量及步长，`incy != 0`（支持正负步长），原地更新 |

## 2.2 需求拆解

1. **严格的参数校验顺序**：
   - `n < 0` $\rightarrow$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`（优先于 `handle == nullptr`）；
   - `handle == nullptr` $\rightarrow$ 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - `n == 0` $\rightarrow$ 立即返回 `ACLBLAS_STATUS_SUCCESS`（不校验后续参数，不启动 Kernel）；
   - `uplo` 非法、`k < 0`、`lda < k + 1`、`incx == 0`、`incy == 0` 或 `alpha / beta / A / x / y` 为空指针 $\rightarrow$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. **厄米特（Hermitian）共轭与实对角线契约**：
   - 对角元素 $A_{i,i}$ 仅读取其实部 $\text{Re}(A_{i,i})$，虚部强制按 `0` 处理且绝不发生内存读取（即便对角虚部内存中存放 `NaN` 或 `Inf` 也不影响输出）；
   - 未存储侧严格按共轭对称补全：$A_{i,j} = \overline{A_{j,i}} = \text{Re}(A_{j,i}) - i\cdot\text{Im}(A_{j,i})$。
3. **`alpha == 0` 与 `beta == 0` 的 IEEE-754 安全语义**：
   - 当 $\alpha = (0, 0)$ 时跳过 $A \cdot x$ 累加（不产生 $0 \times \text{NaN}$）；
   - 当 $\beta = (0, 0)$ 时直接覆写 $y_i = \alpha (Ax)_i$，不读取旧 $y_i$（防止旧 $y_i$ 中的 `NaN` 污染结果）。
4. **精度与性能双达标**：
   - 精度：1000 条功能/边界/异常用例 + 200 条性能尺寸用例 + 正态分布/随机复数标量/对角虚部毒值回归，实部与虚部独立满足 FLOAT32 混合容差（`rtol = atol = 2^-13`，匹配率 $\ge 99\%$，`max_abs_err <= max(1e-2, 32*ULP)`）及特殊值（`NaN`/`Inf`）等效规则。
   - 性能：基于 Ascend 950PR 真机 `msprof op` 采集全部 200 条 `TC_PF_1001 ~ TC_PF_1200` 的 `Task Duration(us)`，逐条满足 $\text{GPU\_H100\_us} / \text{NPU\_950\_us} \ge 0.40$。

---

# 3. 详细设计（required）

## 3.1 算子分析

### 数学公式与带状寻址映射

对于每一输出行 $i \in [0, n-1]$，有效列范围为 $j \in [\max(0, i-k), \min(n-1, i+k)]$：

$$y_{\text{idx}_y(i)} \leftarrow \alpha \sum_{j=\max(0, i-k)}^{\min(n-1, i+k)} \widetilde{A}_{i,j} x_{\text{idx}_x(j)} + \beta y_{\text{idx}_y(i)}$$

其中列主序带状紧凑存储的物理一维下标（以复数元素为单位）与共轭规则为：
- **对角线 ($j = i$)**：
  - `ACLBLAS_UPPER`：$\text{gmIdx} = k + i \cdot \text{lda}$；
  - `ACLBLAS_LOWER`：$\text{gmIdx} = i \cdot \text{lda}$；
  - 贡献项：$\text{Re}(A[\text{gmIdx}]) \cdot x_i$（虚部不读取）。
- **上三角存储 (`ACLBLAS_UPPER`)**：
  - 当 $i < j$（带内已存储上三角）：$\text{gmIdx} = (k + i - j) + j \cdot \text{lda}$，直接取 $A[\text{gmIdx}]$；
  - 当 $i > j$（未存储下三角）：$\text{gmIdx} = (k + j - i) + i \cdot \text{lda}$，取共轭 $\overline{A[\text{gmIdx}]}$。
- **下三角存储 (`ACLBLAS_LOWER`)**：
  - 当 $i > j$（带内已存储下三角）：$\text{gmIdx} = (i - j) + j \cdot \text{lda}$，直接取 $A[\text{gmIdx}]$；
  - 当 $i < j$（未存储上三角）：$\text{gmIdx} = (j - i) + i \cdot \text{lda}$，取共轭 $\overline{A[\text{gmIdx}]}$。
- **正负步长物理下标**：
  $$\text{idx}(m, \text{inc}) = \begin{cases} m \cdot \text{inc}, & \text{inc} > 0 \\ (n - 1 - m) \cdot (-\text{inc}), & \text{inc} < 0 \end{cases}$$

## 3.2 算子实现方案

### 3.2.1 Host 侧设计 (`blas/sbmv/arch35/chbmv_host.cpp`)

1. **参数契约校验**：严格按照 `n < 0` $\rightarrow$ `handle == nullptr` $\rightarrow$ `n == 0` $\rightarrow$ `ValidateChbmvParams(...)` 顺序执行。
2. **动态获取核数与自适应分块**：
   - 调用 `GetAivCoreCount()` 动态获取 Ascend 950PR 的 AIV 核心数（56 核），并作非零防御（满足编码规范 R2、R8）；
   - 当 `incx == 1 && k >= 8` 时，启用行内多线程协同归约，最小每核行数设为 `minRowsPerBlock = 8`，使中等规模（如 $n=512 \sim 1024$）也能跑满 56 个 AIV 核：`useNumBlocks = min(CeilDiv(n, minRowsPerBlock), aivCoreNum)`；
   - 计算每核负责的连续行窗口 `rowsPerBlock = CeilDiv(n, useNumBlocks)`，并根据半带宽 $k$ 选择每行协同线程数 $\text{tpr} \in \{1, 2, 4, 8, 16\}$，将 `numThreads` 向 `SIMT_MIN_THREAD_NUM (128)` 对齐且不超过 `SIMT_MAX_THREAD_NUM (2048)`。

### 3.2.2 Kernel 侧设计 (`blas/sbmv/arch35/chbmv_kernel.cpp`)

采用 Ascend C `arch35` 原生 SIMT 编程模型（`asc_vf_call`），设计双路径协同架构：

1. **通用路径 `ChbmvGm<UPLO_IS_UPPER>`**：
   - 处理 `incx != 1`、$n < 32$、$\alpha = 0$ 或局部 $x$ 窗口超出 UB 容量的情形；
   - 每个 SIMT 线程按 Grid-Stride（`row = blockIdx.x * blockDim.x + threadIdx.x`）独占负责输出行，无跨核原子操作。
2. **UB 滑动窗口缓存 + 行内多线程协同归约路径 `ChbmvUb<UPLO_IS_UPPER>`**：
   - **局部 $x$ 窗口协同加载**：每个 AIV Block 负责行区间 $[\text{rowStart}, \text{rowEnd})$，其所需的 $x$ 连续子区间仅为 $x[\max(0, \text{rowStart}-k) \dots \min(n, \text{rowEnd}+k))$（长度 `xLen <= 8192`）。Block 内所有线程并行将该窗口搬入片上 `__ubuf__ float xUb[2 * 8192]`（64 KB），随后执行 `asc_syncthreads()`，彻底消除带状卷积对 $x$ 的 $2k+1$ 次重复全局内存（GM）读取；
   - **行内多线程协同归约（Sub-Warp Row Reduction）**：当 $k \ge 8$ 时，自动为每一行分配 $\text{tpr} \in \{2, 4, 8, 16\}$ 个连续线程（`subTid = threadIdx.x & (tpr - 1)`，`localRow = threadIdx.x / tpr`）沿列方向交错累加带内元素，将部分和写入 `__ubuf__ float redUb[2 * 2048]`（16 KB），经 `asc_syncthreads()` 后由 `subTid == 0` 线程汇总并写回 `yGm`。
3. **编码规范合规性（R1 ~ R10）**：
   - 所有函数拆分精简，单函数非空非注释行数 `NBNC <= 46`（满足 R7 `<= 50`）、圈复杂度 `CCN <= 12`（满足 R5 `<= 20`）、最大嵌套深度 `<= 4`（满足 R6 `<= 5`），无成员变量 `TPipe`（R3），无 Tiling 数组（R4），通过 `chbmv_tiling_data.h` 声明入口（R10）。

---

# 4. 支持硬件与算子约束（required）

| 产品型号 | 架构目录 | 支持情况 |
| --- | --- | --- |
| Atlas A5 / Ascend 950PR (`dav-c310` / `DAV-3510`) | `arch35` | 支持（已真机验证通过） |

- **算子约束**：仅支持 `complex64`（`aclblasComplex`）列主序带状存储，`n >= 0, k >= 0, lda >= k + 1, incx != 0, incy != 0`。

---

# 5. 可维可测分析（required）

## 5.1 精度验证结果（Ascend 950PR 真机实测）

- **Golden 标杆**：Netlib CBLAS `cblas_chbmv(CblasColMajor, ...)`。
- **实测覆盖范围**：
  1. 随包 1000 条精度用例（`TC_L0_001 ~ TC_EX_1000`）：**1000 / 1000 全部 PASS**；
  2. 空句柄独立用例（`ChbmvArch35Test.NullHandle`）：**PASS**；
  3. 对角线虚部 `NaN / +Inf / -Inf / 999.0f` 毒值注入与分派边界用例（`ChbmvArch35Test.DiagonalImaginaryPoisonAndBoundaries`）：**PASS**；
  4. 正态分布（$\mu \in [-5.0, 5.0], \sigma \in [0.1, 2.0]$）与随机复数 $\alpha, \beta$ 标量 50%/50% 配对用例（`ChbmvArch35Test.NormalDistributionAndRandomScalars`）：**PASS**。

## 5.2 性能验证结果（Ascend 950PR 真机 `msprof op` 实测 vs NVIDIA H100）

在 Ascend 950PR（56 AIV 核，1650 MHz）上使用 `msprof op --warm-up=5 --aic-metrics=BasicInfo` 及 55 次 NPU Event 采样对全部 **200 条** `TC_PF_1001 ~ TC_PF_1200` 性能用例进行实测：

- **总体达标率**：**200 / 200 全部 PASS（100% 达标）**
- **最低加速比（vs H100）**：**`0.4324x`**（高于 `0.40x` 验收线）
- **最高加速比（vs H100）**：**`4.9404x`**
- **200 例平均加速比（vs H100）**：**`3.1858x`**（平均耗时仅为 H100 的 **31.4%**！）

### 任务书 4 个典型形状基准点实测对比表

| 用例编号 | `uplo` | `n` | `k` | `lda` | H100 标杆耗时 (`us`) | 验收上限 (`H100 / 0.4`, `us`) | Ascend 950PR 实测 (`msprof op`, `us`) | 相对 H100 倍率 (`GPU/NPU`) | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TC_PF_1001` | `UPPER` | 512 | 8 | 9 | 5.479 | 13.698 | **8.312** | **0.6592x** | **PASS** |
| `TC_PF_1002` | `UPPER` | 1024 | 16 | 17 | 11.128 | 27.820 | **10.181** | **1.0930x** | **PASS** |
| `TC_PF_1003` | `LOWER` | 2048 | 32 | 33 | 22.743 | 56.858 | **13.886** | **1.6378x** | **PASS** |
| `TC_PF_1004` | `LOWER` | 4096 | 64 | 65 | 43.828 | 109.570 | **17.120** | **2.5600x** | **PASS** |

---

# 6. 兼容性分析（required）

- 新增公开接口声明 `aclblasChbmv` 于 `include/cann_ops_blas.h`，与既有 `aclblasSsbmv`、`aclblasCsymv` 命名及参数风格完全一致，不修改任何既有 ABI 或数据结构。
- 算子实现隔离于 `blas/sbmv/arch35/`，不影响 `arch22` 或其他算子模块构建。
