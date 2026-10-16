# Atlas 950PR 单精度复数 LU 分解、求解和批量求逆算子设计文档

> 交付件①：算子设计文档。
> 任务名称：10月社区任务-单精度复数LU分解、求解和批量求逆(950)（任务编号：`10-2-aclsolverCgetrf-batch-950`）
> 提交账号：`sss-hust`
> 适配硬件：Ascend 950PR（Atlas 950，`dav-3510` / `arch35`）
> 目标代码仓：[https://gitcode.com/cann/ops-solver](https://gitcode.com/cann/ops-solver)
> 本文档严格按照社区任务 [design_template.md](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 的四个必需一级章节（`# 需求背景（required）`、`# 需求分析（required）`、`# 详细设计（required）`、`# 可维可测分析`）组织。文末汇总需要检视人员确认的任务书细节。

---

# 需求背景（required）

## 需求来源

本设计来自 2026 年 10 月 CANN 开源社区任务第 2 项「10月社区任务-单精度复数LU分解、求解和批量求逆(950)」（任务书文件：`Atlas950_Cgetrf_Cgetrs_CgetriBatched_CmatinvBatched_task_doc.md`）。

任务目标是在昇腾 **Ascend 950PR（Atlas 950）** 硬件上，基于 **ops-solver Host C API + AscendC/CATLASS Kernel 直调** 工程模式，开发并优化单精度复数（`COMPLEX64` / `aclFloatComplex`）稠密 LU 分解、单矩阵求逆以及配套的批量求逆功能，具体包括 4 个计算接口与 2 个工作空间查询接口：

| 序号 | NPU 交付接口 | 对标 CUDA / LAPACK 接口 | 功能概述 |
|---|---|---|---|
| 1 | `aclsolverCgetrf_bufferSize` | `cusolverDnCgetrf_bufferSize` | 查询单矩阵复数 LU 分解所需 Device Workspace 元素个数（按 `aclFloatComplex` 计） |
| 2 | `aclsolverCgetrf` | `cusolverDnCgetrf` | 单精度复数矩阵 $m \times n$ 带部分主元（可选不选主元）的原地 LU 分解：$P \cdot A = L \cdot U$ |
| 3 | `aclsolverCgetri_bufferSize` | 对标 LAPACK `cgetri` 查询接口 | 查询单矩阵复数 LU 求逆所需 Device Workspace 元素个数（按 `aclFloatComplex` 计） |
| 4 | `aclsolverCgetri` | LAPACK `cgetri`（调用约定对齐 cuSolver DN） | 基于 `aclsolverCgetrf` 输出的 LU 因子与主元序列 `devIpiv` 原地计算逆矩阵 $A^{-1}$ |
| 5 | `aclsolverCgetriBatched` | `cublasCgetriBatched` | 基于已分解 LU 因子与主元的批量矩阵求逆（Device 指针数组输入输出，out-of-place） |
| 6 | `aclsolverCmatinvBatched` | `cublasCmatinvBatched` | 面向小矩阵（$n \le 32$）原始输入的直接批量求逆（Device 指针数组输入输出，out-of-place） |

上述接口作为**同一社区任务整体交付**，验收通过后合入 [https://gitcode.com/cann/ops-solver](https://gitcode.com/cann/ops-solver)。

## 背景介绍

### 稠密复数 LU 分解与矩阵求逆的数学耦合关系

单精度复数 LU 分解及求逆是电磁仿真、量子化学计算、阵列信号处理、无线通信 MIMO 检测及科学计算中的核心线性代数原语：

1. **LU 分解（`Cgetrf`）**：将一般 $m \times n$ 复矩阵 $A \in \mathbb{C}^{m \times n}$ 分解为 $P \cdot A = L \cdot U$，其中 $P$ 为行置换矩阵（由 1-based 主元数组 `devIpiv` 表示），$L$ 为单位下三角（或梯形）矩阵，$U$ 为上三角（或梯形）矩阵。
2. **基于 LU 因子的求逆（`Cgetri` / `CgetriBatched`）**：利用已完成的分解 $P \cdot A = L \cdot U$，按 $A^{-1} = U^{-1} L^{-1} P$ 求逆：
   - 第一步（三角求逆 `ctrtri`）：求上三角矩阵 $U$ 的逆 $X = U^{-1}$，满足 $X \cdot U = I$；
   - 第二步（右端单位下三角求解 `ctrsm`）：求解 $Z \cdot L = U^{-1}$，即 $Z = U^{-1} L^{-1}$；
   - 第三步（主元逆序回放列交换）：对 $Z$ 按 $k = n-1, n-2, \dots, 0$ 逆序交换第 $k$ 列与第 $\text{ipiv}[k]-1$ 列，得到 $A^{-1} = Z \cdot P$。
3. **小矩阵直接批量求逆（`CmatinvBatched`）**：对 $n \le 32$ 的大批独立原始复矩阵 $A[i]$，在片上 UB 内融合执行带部分主元消元与求逆，省去跨 Kernel 的中间因子读写开销。

### `ops-solver` 仓现状分析与差距识别

通过对 `cann/ops-solver` 仓库现有代码进行核查，当前仓库虽存在 `cgetrf`、`cgetri`、`cgetri_batched`、`cmatinv_batched` 原型，但在**公开接口签名、内存空间归属、数据排布约束、硬件架构适配**四个维度与任务书要求存在根本差异：

| 对比维度 | `ops-solver` 现有原型现状 | 任务书要求与本设计改造目标 |
|---|---|---|
| **公开头文件与 C ABI** | `cann_ops_solver.h` 使用 C++ `std::complex<float>*`、64 位 `int64_t` 维数，返回 `aclError`；缺少 `_bufferSize` 查询接口与 `aclsolverOperation_t` | 在 `cann_ops_solver_common.h` 定义 C 兼容的 `aclFloatComplex` 与 `aclsolverOperation_t`；在 `cann_ops_solver.h` 导出 `extern "C"` 的 6 个标准接口，统一返回 `aclsolverStatus_t`，维数采用 32-bit `int` |
| **指针归属与异步流** | 现有原型接收 Host 指针，在 Host 内部申请 Device 内存、执行同步 H2D/D2H 拷贝与 `aclrtSynchronizeStream`，且未向 Device 写 `info` | `A`、`Workspace`、`devIpiv`/`PivotArray`、`devInfo`/`infoArray`/`info` 及批量指针数组均为 **Device 指针**；计算全流程绑定 `handle` 所属 `aclrtStream` 异步下发，真实写入 Device `info` |
| **矩阵排布与 `lda` 约束** | 强制要求 `lda == n`；批量接口将输入视为连续 `[batch, n, n]` 三维张量而非指针数组 | 严格采用**列主序（column-major）**，支持任意合法 `lda >= max(1, m)` / `ldc >= max(1, n)` padding；批量接口严格解引用 **Device 侧指针数组** `Aarray[i]` / `Carray[i]` |
| **算子语义边界** | `cgetri` 内部重新调用 LU 分解（无 `devIpiv` 入参）；`cmatinv_batched` 在 $n \ge 32$ 时静默转调 `cgetri_batched` | `aclsolverCgetri` 严格消费已有 LU 因子与 `devIpiv` 原地求逆；`aclsolverCmatinvBatched` 仅支持 $0 \le n \le 32$，$n > 32$ 严格返回 `ACLSOLVER_STATUS_INVALID_VALUE` 并写 `info[0] = -1` |
| **Ascend 950PR 适配** | 旧 Cube 核 `src/utils/kernel/c64/gemm.hpp` 依赖 `__DAV_C220_CUBE__` 宏，在 Ascend 950（`dav-3510` / `__NPU_ARCH__ == 3510`）下未生效，README 标注不支持 Ascend 950PR | 基于 Ascend 950 原生 AIV 向量指令（含 `__simd_vf__` 寄存器级编程与 `DIST_BRC_B32` 广播加载）及 AIC `Matmul` 高阶 API（FP32 精确模式 `SetHF32(false)`）重新设计与实现 |

---

# 需求分析（required）

## 需求描述

在 `cann/ops-solver` 仓面向 **Ascend 950PR** 实现并交付以下公开类型与 6 个 C API 接口，满足任务书全部功能、边界、异常、确定性、LAPACK 残差精度及 652 条基准用例性能要求（$T_{\text{NPU}} \le T_{\text{GPU}} / 0.35$）：

```c
/* include/cann_ops_solver_common.h */
typedef struct {
    float real;
    float imag;
} aclFloatComplex;

typedef enum {
    ACLSOLVER_OP_N = 0,   /* 不转置，对齐 CUBLAS_OP_N */
    ACLSOLVER_OP_T = 1,   /* 转置，对齐 CUBLAS_OP_T */
    ACLSOLVER_OP_C = 2    /* 共轭转置，对齐 CUBLAS_OP_C */
} aclsolverOperation_t;

/* include/cann_ops_solver.h */
aclsolverStatus_t aclsolverCgetrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverCgetrf(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda,
    aclFloatComplex *Workspace, int *devIpiv, int *devInfo);

aclsolverStatus_t aclsolverCgetri_bufferSize(
    aclsolverHandle_t handle, int n, aclFloatComplex *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverCgetri(
    aclsolverHandle_t handle, int n, aclFloatComplex *A, int lda,
    int *devIpiv, aclFloatComplex *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverCgetriBatched(
    aclsolverHandle_t handle, int n, const aclFloatComplex *const Aarray[], int lda,
    const int *PivotArray, aclFloatComplex *const Carray[], int ldc,
    int *infoArray, int batchSize);

aclsolverStatus_t aclsolverCmatinvBatched(
    aclsolverHandle_t handle, int n, const aclFloatComplex *const A[], int lda,
    aclFloatComplex *const Ainv[], int lda_inv, int *info, int batchSize);
```

## 需求拆解

结合任务书正文与任务包内 4 个算子目录下的 `cases.json` / `bench_result.json`（共计 **652 条** 测试与性能基准用例），将需求拆解如下：

| 编号 | 需求模块 | 具体要求与任务包数据核对 | 设计落点 |
|---|---|---|---|
| **R1** | 公开头文件与 ABI | 提供 C 可调用的 `aclFloatComplex`、`aclsolverOperation_t` 及 6 个 API；C++ 下保留旧 Host 指针重载以兼容仓内遗留调用 | `include/cann_ops_solver_common.h`、`include/cann_ops_solver.h` |
| **R2** | `aclsolverCgetrf` 及 `_bufferSize` | 支持方阵与矩形（$m \ne n$）、列主序、`lda >= max(1, m)` padding、有主元（`devIpiv != nullptr`）与无主元（`devIpiv == nullptr`）。任务包共 **148 条** 用例，$m \in [2, 4096], n \in [2, 8168]$ | 小矩阵 UB 驻留核 + 中大矩阵多核分块（AIV Panel LU/TRSM + AIC Cube 4M FP32 GEMM） |
| **R3** | `aclsolverCgetri` 及 `_bufferSize` | 接收 `Cgetrf` 输出的 LU 因子与 `devIpiv`（允许 `nullptr` 表示未选主元）原地求逆，支持 `lda >= max(1, n)`。任务包共 **200 条** 用例，$n \in [2, 4096]$ | $n \le 64$ 单核 UB 驻留求逆核 + $n > 64$ 分块 Cube `ctrtri` + 右端三角求解 + 逆序主元列置换 |
| **R4** | `aclsolverCgetriBatched` | Device 指针数组 `Aarray`/`Carray`（out-of-place），`PivotArray` 长度 $n \times \text{batchSize}$（或 `nullptr`），逐矩阵写 `infoArray[i]`。任务包共 **89 条** 用例，$n \in [2, 3049], \text{batchSize} \in [32, 1000000]$ | $n \le 32$ 批主序向量核 + $33 \le n \le 96$ UB 驻留 `__simd_vf__` 核 + $n > 96$ 多矩阵分块 Cube+AIV 流水 |
| **R5** | `aclsolverCmatinvBatched` | Device 指针数组 `A`/`Ainv`（out-of-place），对原始矩阵直接求逆，仅支持 $0 \le n \le 32$（$n > 32$ 报错），逐矩阵写 `info[i]`。任务包共 **215 条** 用例，$n \in [2, 32], \text{batchSize} \in [128, 10000000]$ | 批主序融合消元求逆核 `BatchMajorMatinv`（$G=16$ / $G=8$），支持任意 `lda`/`lda_inv` 与奇异检测 |
| **R6** | 参数校验与 `info` 语义 | `handle == nullptr` 返回 `HANDLE_IS_NULLPTR`；第 $i$ 个参数（不含 `handle`）非法返回 `INVALID_VALUE` 并在 `devInfo`/`info[0]` 可写时写入 `-i`；奇异矩阵写首个零主元 1-based 下标 $k > 0$；空问题（$m=0$ 或 $n=0$ 或 $\text{batchSize}=0$）返回 `SUCCESS` | Host 统一校验层 + Device 奇异检测与状态写回 |
| **R7** | 确定性与多流支持 | 相同输入、相同 stream 串行多次执行结果（含 `ipiv`、`info`）bit-wise 一致；支持非默认 stream 异步执行 | 固定分核归约顺序、无浮点原子竞争、绑定 `aclsolverGetStream` |
| **R8** | 精度与性能达标 | 精度满足 LAPACK `CGET01` / `CGET03` 残差判据（$\varepsilon = 2^{-24}$）；652 条用例在 Ascend 950PR 上单次耗时满足 $T_{\text{NPU}} \le T_{\text{GPU}} / 0.35$ | 全程 FP32 精确计算（关闭 HF32）、批主序交织、寄存器级 `__simd_vf__` 优化与 Cube 4M 矩阵乘加速 |

---

# 详细设计（required）

## 算子分析

### 数学公式

#### 1. `aclsolverCgetrf`（带部分主元的复数 LU 分解）

对 $A \in \mathbb{C}^{m \times n}$，令 $K = \min(m, n)$，分解形式为：

$$P \cdot A = L \cdot U$$

- 在第 $k \in [0, K-1]$ 步：
  - **选主元（当 `devIpiv != nullptr`）**：在列 $k$ 的行区间 $i \in [k, m-1]$ 内按确定性规则选取模长平方 $|A_{i,k}|^2 = \text{Re}(A_{i,k})^2 + \text{Im}(A_{i,k})^2$ 最大且行号最小的行 $p$，记录 1-based 主元 $\text{devIpiv}[k] = p + 1$；若 $p \ne k$，交换整行 $A_{k, 0:n-1} \leftrightarrow A_{p, 0:n-1}$。若 `devIpiv == nullptr`，则不选主元（$p = k$）。
  - **奇异判定**：若主元满足 $\text{Re}(A_{k,k}) == 0 \land \text{Im}(A_{k,k}) == 0$，记录首个零主元位置 $\text{info} = k + 1$，跳过当前列除以零。
  - **列缩放与尾随更新**：对 $i \in [k+1, m-1]$ 计算乘子 $L_{i,k} = A_{i,k} / U_{k,k}$ 并原地存入 $A_{i,k}$；随后更新尾随子矩阵 $A_{k+1:m, k+1:n} \leftarrow A_{k+1:m, k+1:n} - L_{k+1:m, k} \cdot U_{k, k+1:n}$。

#### 2. `aclsolverCgetri` 与 `aclsolverCgetriBatched`（基于 LU 因子的矩阵求逆）

给定 $n \times n$ 打包 LU 因子（上三角及对角线为 $U$，严格下三角为单位下三角 $L$ 的非对角元）及主元序列 $\text{ipiv}[0:n-1]$：

1. **奇异检测**：检查 $U$ 的对角元 $U_{k,k}$（$k = 0 \dots n-1$）。若存在 $U_{k,k} == 0$，则该矩阵奇异，写入最小的 $\text{info} = k + 1$ 并终止该矩阵求逆。
2. **阶段一（上三角求逆 `ctrtri`）**：求解 $X \cdot U = I \Rightarrow X = U^{-1}$，覆盖上三角区域：
   - 对角元：$X_{c,c} = 1 / U_{c,c} = \frac{\text{Re}(U_{c,c}) - i\,\text{Im}(U_{c,c})}{|U_{c,c}|^2}$；
   - 非对角元（$r < c$）：$X_{r,c} = \left(\sum_{p=r}^{c-1} X_{r,p} \cdot (-U_{p,c})\right) \cdot X_{c,c}$。
3. **阶段二（右端单位下三角求解 `ctrsm`）**：由右向左（$j = n-2, n-3, \dots, 0$）求解 $Z \cdot L = U^{-1}$：
   - 提取第 $j$ 列严格下三角的 $L_{j+1:n, j}$，将原位置清零后执行秩-1 / 分块更新：
     $$Z_{0:n, j} \leftarrow Z_{0:n, j} + \sum_{c=j+1}^{n-1} Z_{0:n, c} \cdot (-L_{c, j})$$
4. **阶段三（主元列逆序置换）**：当提供主元序列时，按 $k = n-2, n-3, \dots, 0$ 依次检查 $p = \text{ipiv}[k] - 1$；若 $p \ne k$，交换第 $k$ 列与第 $p$ 列：$Z_{0:n, k} \leftrightarrow Z_{0:n, p}$。

#### 3. `aclsolverCmatinvBatched`（小矩阵直接批量求逆）

对每个 $n \times n$（$n \le 32$）原始复矩阵 $A[b]$，在片上同时维护工作矩阵 $M \leftarrow A[b]$ 与右端矩阵 $X \leftarrow I_n$，按列 $p = 0 \dots n-1$ 执行带部分主元的行交换与 Gauss-Jordan 消元：
- 在 $r \in [p, n-1]$ 中选取 $|M_{r,p}|^2$ 最大（平局取最小行号）的主元行 $r_{\max}$，同步交换 $M$ 与 $X$ 的第 $p$ 行和第 $r_{\max}$ 行；
- 若 $|M_{p,p}|^2 == 0$，记录首个零主元 $\text{info}[b] = p + 1$；
- 将第 $p$ 行乘以 $M_{p,p}^{-1}$ 归一化，并用第 $p$ 行消去其余所有行 $r \ne p$ 的第 $p$ 列元素：$M_{r,:} \leftarrow M_{r,:} - M_{r,p} M_{p,:}$，$X_{r,:} \leftarrow X_{r,:} - M_{r,p} X_{p,:}$。$n$ 步完成后 $X$ 即为 $A[b]^{-1}$。

### 支持数据类型

| 参数类别 | 数据类型 | 内存布局 |
|---|---|---|
| 矩阵元素 `A` / `C` / `Ainv` / `Workspace` | `COMPLEX64`（`aclFloatComplex`，实部、虚部各为连续 `float32`，共 8 字节） | 列主序（column-major），元素 $(i, j)$ 偏移为 $(j \cdot \text{ld} + i)$ 个复数 |
| 主元 `devIpiv` / `PivotArray` | `INT32`（`int`） | 1-based 行下标，取值范围 $[1, m]$ 或 $[1, n]$ |
| 状态码 `devInfo` / `infoArray` / `info` | `INT32`（`int`） | Device 标量或长度为 `batchSize` 的一维数组 |
| 批量矩阵地址表 `Aarray` / `Carray` / `A` / `Ainv` | `uint64_t`（Device 指针数组） | 长度为 `batchSize` 的一维 Device 指针数组，每个元素指向一个独立 Device 矩阵 |

### 支持形状与规模

| 接口 | 支持维数范围（覆盖任务书下限及任务包全量用例） | Leading Dimension 约束 |
|---|---|---|
| `aclsolverCgetrf` / `_bufferSize` | $m \in [0, 4096], n \in [0, 8168]$（含方阵与长方/宽方矩形） | $\text{lda} \ge \max(1, m)$ |
| `aclsolverCgetri` / `_bufferSize` | $n \in [0, 4096]$ | $\text{lda} \ge \max(1, n)$ |
| `aclsolverCgetriBatched` | $n \in [0, 4096], \text{batchSize} \in [0, 1000000]$（任务包最大 $n=3049, \text{batch}=10^6$） | $\text{lda} \ge \max(1, n), \text{ldc} \ge \max(1, n)$ |
| `aclsolverCmatinvBatched` | $n \in [0, 32], \text{batchSize} \in [0, 10000000]$（$n > 32$ 严格报错） | $\text{lda} \ge \max(1, n), \text{lda\_inv} \ge \max(1, n)$ |

---

## 算子实现

### 实现方案总览

针对 Ascend 950PR 的硬件特征（56 个 AIV 向量核、28 个 AIC 矩阵核、每核 256 KB UB、支持 `__simd_vf__` 寄存器级向量编程与 `DIST_BRC_B32` 硬件广播加载），四个算子均采用 **「Host 侧O(1)分派 + 按规模自适应的多层 Kernel 架构」**：

| 算子接口 | 小规模路径（片上 UB 驻留 / 批主序向量核） | 中等规模路径（寄存器级 `__simd_vf__` 驻留核） | 大规模路径（多核分块 AIV + AIC Cube 4M FP32 GEMM） |
|---|---|---|---|
| `aclsolverCgetrf` | $m \le 320, n \le 64$：`cgetrf_small_kernel`（单核 UB 驻留 + 16 列分块） | — | 其余尺寸（至 $4096 \times 8168$）：`nb=128` 右看分块（AIV Panel LU/TRSM + AIC Cube 4M GEMM） |
| `aclsolverCgetri` | $n \le 64$：`cgetri_small_kernel`（UB 驻留秩-1 外积求逆 + 逆序列交换） | — | $64 < n \le 4096$：`nb=128` 分块 Cube `ctrtri` + 分块右端求解 $Z \cdot L = U^{-1}$ + 列置换写回 |
| `aclsolverCgetriBatched` | $n \le 32$：`SuperN2Solver` ($G=1024$)、`SuperN4Solver` ($G=256$)、`BatchMajorGetri` ($G \in \{128, 64, 32, 16\}$) 批主序核 | $33 \le n \le 96$：`cgetri_batched_ub64_kernel` / `ub96_kernel`（单核一矩阵 UB 驻留 + `DIST_BRC_B32`） | $n > 96$：多矩阵并行分块 Cube + AIV 流水（单次 Launch 并行处理 `chunkCnt` 个矩阵） |
| `aclsolverCmatinvBatched` | $n \le 32$：`BatchMajorMatinv`（$n \le 16$ 取 $G=16$，$17 \le n \le 32$ 取 $G=8$）批主序 Gauss-Jordan 融合核 | — | $n > 32$：Host 侧直接拒绝并返回 `ACLSOLVER_STATUS_INVALID_VALUE` |

---

### 3.2.1 Host 侧设计

#### 1. 参数校验与 `-i` 错误码写回契约

Host 侧严格按以下优先级顺序校验参数（不计 `handle` 的第 $i$ 个参数非法时，若 Device `info` 指针非空，通过 `aclrtMemcpyAsync` 在绑定 stream 上向 `devInfo` 或 `info[0]` 写入 `-i`，并返回 `ACLSOLVER_STATUS_INVALID_VALUE`）：

| 接口 | 参数序号映射（不含 `handle`） | 关键校验规则 |
|---|---|---|
| `aclsolverCgetrf_bufferSize` | `m`=1, `n`=2, `A`=3, `lda`=4, `Lwork`=5 | `handle == nullptr` 返回 `HANDLE_IS_NULLPTR`；`m < 0`、`n < 0`、`lda < max(1, m)`、`Lwork == nullptr` 或 (`m > 0 && n > 0 && A == nullptr`) 返回 `INVALID_VALUE` |
| `aclsolverCgetrf` | `m`=1, `n`=2, `A`=3, `lda`=4, `Workspace`=5, `devIpiv`=6, `devInfo`=7 | `devInfo == nullptr` 返回 `INVALID_VALUE`；`m < 0` (-1)、`n < 0` (-2)、`lda < max(1, m)` (-4)；$m=0$ 或 $n=0$ 时写 `*devInfo = 0` 并返回 `SUCCESS`；非空问题要求 `A != nullptr` (-3) 且当 `needWork > 0` 时 `Workspace != nullptr` (-5) |
| `aclsolverCgetri_bufferSize` | `n`=1, `A`=2, `lda`=3, `Lwork`=4 | `n < 0`、`lda < max(1, n)`、`Lwork == nullptr` 或 (`n > 0 && A == nullptr`) 返回 `INVALID_VALUE` |
| `aclsolverCgetri` | `n`=1, `A`=2, `lda`=3, `devIpiv`=4, `Workspace`=5, `Lwork`=6, `devInfo`=7 | `devInfo == nullptr` 返回 `INVALID_VALUE`；`n < 0` (-1)、`lda < max(1, n)` (-3)；`Lwork < minWork` (-6)；$n=0$ 时写 `*devInfo = 0` 并返回 `SUCCESS`；$n > 0$ 时要求 `A != nullptr` (-2) 且当 `minWork > 0` 时 `Workspace != nullptr` (-5) |
| `aclsolverCgetriBatched` | `n`=1, `Aarray`=2, `lda`=3, `PivotArray`=4, `Carray`=5, `ldc`=6, `infoArray`=7, `batchSize`=8 | `infoArray == nullptr` 返回 `INVALID_VALUE`；`n < 0` (-1)、`batchSize < 0` (-8)、`lda < max(1, n)` (-3)、`ldc < max(1, n)` (-6)；空问题置零 `infoArray` 并返回 `SUCCESS`；非空时校验 `Aarray != nullptr` (-2)、`Carray != nullptr` (-5)、`Aarray != Carray` (-5) 及首矩阵指针非空与不重叠 |
| `aclsolverCmatinvBatched` | `n`=1, `A`=2, `lda`=3, `Ainv`=4, `lda_inv`=5, `info`=6, `batchSize`=7 | `info == nullptr` 返回 `INVALID_VALUE`；`n < 0 || n > 32` (-1)、`batchSize < 0` (-7)、`lda < max(1, n)` (-3)、`lda_inv < max(1, n)` (-5)；空问题置零 `info` 并返回 `SUCCESS`；非空时校验 `A != nullptr` (-2)、`Ainv != nullptr` (-4)、`A != Ainv` (-4) |

#### 2. Workspace 规划与常驻 Scratch 内存池

1. **单矩阵接口（`Cgetrf` / `Cgetri`）**：`_bufferSize` 与计算接口共用同一纯函数规划器：
   - `Cgetrf`：当 $m \le 320, n \le 64$ 时原位 UB 计算，`Lwork = 0`；当走分块 Cube 路径时，按对齐后的 $\text{ldM} = \text{Align}(m, 128), \text{ldN} = \text{Align}(n, 128)$ 规划实虚分离平面 `wr/wi`、`nr/ni`、4M 矩阵乘累加平面 `pr/pi`、面板缓存 `ar/ai/br/bi`、Cube Tiling 区与 Matmul 系统工作区 `mmws`，返回 `Lwork = (totalFloats + 1) / 2`（以 `aclFloatComplex` 元素计）。
   - `Cgetri`：当 $n \le 64$ 时 `Lwork = 1`；当 $n > 64$ 时规划 5 个 $\text{ld} \times \text{ld}$ 平面（`ar/ai`、`tr/ti`、`pr`）+ 3 个 $128 \times \text{ld}$ 面板平面 + Tiling 区 + `mmws` 区，支持 `Lwork == -1` 查询约定。
2. **批量接口（`CgetriBatched` 大矩阵路径）**：接口不含外部 `Workspace` 参数，Host 侧维护 `thread_local` 常驻 Scratch 缓冲池 `AcquireCgetriBatchedScratch`（容量够则直接O(1)复用，不足时扩容），并将大 batch 按有界 `maxChunk <= 8192` 分批流式调度，保证显存占用有界且零重复 `aclrtMalloc`/`aclrtFree` 开销。
3. **Tiling 缓存与连续指针快速检测**：Host 侧对 `(m, n, nb)` / `(n, nb, ld)` 的 `TCubeTiling` 结构体进行 `thread_local` 缓存，避免重复调用 `MatmulApiTiling`；对 $n \le 96$ 的 `CgetriBatched` 缓存 `(Aarray, Carray, n, lda, ldc, batchSize)` 的连续块检测结果，若判定各矩阵在 GM 中紧密连续排布（$\text{Aarray}[i] = \text{Aarray}[0] + i \cdot n^2 \cdot 8$），则向 Kernel 传递基地址开启硬件多矩阵 Burst DMA 快路径，同时对非连续随机指针数组完整保留通用解引用路径。

#### 3. 分核策略

通过 `PlatformAscendCManager::GetInstance()` 动态获取当前芯片的 `GetCoreNumAiv()`（56 核）与 `GetCoreNumAic()`（28 核）：
- 批主序向量核按分组数 $\text{groups} = \lceil \text{batchSize} / G \rceil$ 启动 $\min(\text{aivCores}, \text{groups})$ 个 AIV 核，核内按 `grp = blockIdx; grp < groups; grp += blockNum` 静态轮转；
- 分块 Cube 核将 `(matrixIdx, tileIdx)` 二维任务展平为 `totalJobs = chunkCnt * numTiles`，启动 $\min(\text{aicCores}, \text{totalJobs})$ 个 AIC 核，消除单矩阵小 Tile 无法填满 28 个 Cube 核的瓶颈。

---

### 3.2.2 Kernel 侧设计

#### 1. 复数实虚分离平面（Split-Planar）与 4M 精确 FP32 Cube 矩阵乘

GM 中的复数矩阵为实虚交织（Interleaved，`[re0, im0, re1, im1, ...]`）列主序格式，而向量指令与 Cube 矩阵乘在实虚分离平面（Split-Planar，`Re` 平面与 `Im` 平面独立连续）上吞吐最高：
- **交织与去交织**：搬入时通过 `DataCopy` / `DataCopyPad` 将交织数据读入 UB `stage`，再用 `GatherMask`（模式 1 提取偶数位实部，模式 2 提取奇数位虚部）以硬件满带宽拆分为实、虚两个连续平面；写回时通过预构建的 `Gather` 索引表或成对交织将实、虚平面合并写回 GM。
- **4M 精确 FP32 Cube GEMM**：复数矩阵乘法 $C \leftarrow C - A \cdot B$ 展开为 4 次实数 FP32 Matmul（显式调用 `mm.SetHF32(false)` 禁用 HF32 截断，保证全尾数 IEEE-754 FP32 精度）：
  $$\text{Re}(A B) = A_r B_r - A_i B_i, \quad \text{Im}(A B) = A_r B_i + A_i B_r$$
  相比 Karatsuba 3M 方案，4M 方案在条件数较大的矩阵上无相减抵消放大的舍入误差，实测 LAPACK 残差比值稳定在 $0.02 \sim 0.15$，远优于任务书上限。

#### 2. `aclsolverCgetrf` Kernel 设计

1. **小规模驻留核（`cgetrf_small_kernel`，$m \le 320, n \le 64$）**：
   - 单核将整个 $m \times n$ 矩阵去交织装入 UB（列跨度对齐至 8 floats = 32B），按 16 列为一个子面板：
   - 逐列计算模长平方 `mag = vr*vr + vi*vi`，确定性选出最大值行号 $p$，交换整行 $0 \dots n-1$，向量化计算乘子 $L_{k+1:m, k}$ 并执行面板内秩-1 更新；
   - 子面板满 16 列后，利用 4 路展开的向量 FMA 一次性更新右侧剩余列 $c \in [j_0+16, n)$，把向量指令发射与流水屏障开销摊薄 16 倍。
2. **中大规模分块协同路径（$nb = 128$，支持至 $m=4096, n=8168$）**：
   - `cgetrf_load_split_kernel`（AIV）：多核并行将 GM 交织矩阵按列拆分为 128 对齐的实部平面 `wr` 与虚部平面 `wi`。
   - 对每个面板步 $j_0 = 0, 128, 256, \dots$（面板宽 $b = \min(128, K - j_0)$）：
     - `cgetrf_panel_trsm_kernel`（AIV 全核协同）：
       1. 对每一列 $k = j_0 \dots j_0 + b - 1$，全核按行分段并行计算局部最大模长平方及最小行号，写入 GM 归约槽，`SyncAll` 后由各核按固定核序归约出全局主元行 $p$；
       2. 全核按列分段并行交换 `wr/wi` 的第 $k$ 行与第 $p$ 行（覆盖 $0 \dots n-1$ 全宽）；
       3. 全核按行分段并行缩放第 $k$ 列下部乘子，并以 `Brcb` 广播行向量 `U[k, k+1:j0+b]` 并行更新面板内剩余列；
       4. 面板分解完成后，全核按右侧列块并行求解单位下三角方程 $L_{11} U_{12} = A_{12}$，并将 $-L_{21}$ 与 $U_{12}$ 打包至连续面板缓冲 `ar/ai` 与 `br/bi`。
     - `cgetrf_trail_gemm_kernel`（AIC Cube）+ `cgetrf_gemm_combine_kernel`（AIV）：
       - 为控制工作区内存占用，尾随矩阵列区间 $[j_0+b, n)$ 按 `G_MAX_N = 512` 分段，每段内再按 `G_SUB_N = 64` 切分子块；
       - AIC 核利用 `Matmul<MatmulType<GM, CubeFormat::ND, float>, ...>` 执行 4 次实数矩阵乘产出 `pr`、`pi`；
       - AIV 核将 `pr`、`pi` 累加回 `wr`、`wi` 对应尾随子矩阵。
   - `cgetrf_store_merge_kernel`（AIV）：多核并行将 `wr`、`wi` 交织写回用户矩阵 `A`（严格遵守 `lda` 跨度，不触碰 padding 区）。

#### 3. `aclsolverCgetri` Kernel 设计

1. **小规模驻留核（`cgetri_small_kernel`，$n \le 64$）**：
   - 将 $n \times n$ LU 因子装入 UB 实虚平面（列跨度 64 floats），先扫描对角线检测奇异；
   - **Phase 1（上三角求逆）**：对 $p = 0 \dots n-1$，提取第 $p$ 行 $U_{p, p+1:n}$ 取反后清零原行，缩放第 $p$ 列 $0 \dots p$ 行乘以 $U_{p,p}^{-1}$，再用 `Brcb` 将 $-U_{p, c}$ 广播为 8-float 块，对列 $c \in [p+1, n)$ 执行向量化秩-1 外积更新；
   - **Phase 2（右端求解 $Z \cdot L = U^{-1}$）**：对 $j = n-2 \dots 0$，提取第 $j$ 列严格下三角 $-L_{j+1:n, j}$ 并清零原位置，以 4 列展开的向量 FMA 将 $\sum_{c=j+1}^{n-1} Z_{0:n, c} (-L_{c,j})$ 累加至第 $j$ 列；
   - **Phase 3（主元逆序置换）**：按 $k = n-1 \dots 0$ 逆序更新列置换表 `perm`，按置换顺序交织写回 GM。
2. **中大规模分块路径（$64 < n \le 4096, nb = 128$）**：
   - `cgetri_load_split_kernel`（AIV）：将输入 LU 拆分为 `ar/ai` 平面，同时扫描全部对角元写出 `devInfo`；
   - **Phase 1（分块 `ctrtri`）**：对块步 $i_0 = 0, 128, \dots$（块宽 $b = \min(128, n - i_0)$）：
     - 若 $i_0 > 0$，先由 `cgetri_trtri_panel_prep_kernel`（AIV）提取 $-U_{0:i_0, i_0:i_0+b}$ 至面板缓冲，再由 `cgetri_trtri_gemm_kernel`（AIC Cube）计算 $T_{0:i_0, i_0:i_0+b} = U_{0:i_0, 0:i_0}^{-1} \cdot (-U_{0:i_0, i_0:i_0+b})$；
     - 随后由 `cgetri_trtri_panel_kernel`（AIV，按 64 行分片并行）在 UB 内完成对角块 $U_{i_0:i_0+b, i_0:i_0+b}^{-1}$ 求逆及上部面板右乘更新 $X_{0:i_0, i_0:i_0+b} = T_{0:i_0, i_0:i_0+b} \cdot U_{i_0:i_0+b, i_0:i_0+b}^{-1}$。
   - **Phase 2（分块求解 $Z \cdot L = U^{-1}$）**：对块步 $i_0$ 从末尾向 0 逆序遍历：
     - 将当前列面板嚴格下三角 $L_{i_0:n, i_0:i_0+b}$ 拷贝到暂存区 `tr/ti` 并将 `ar/ai` 对应下三角清零；
     - 若 $i_0 + b < n$，由 `cgetri_solve_gemm_kernel`（AIC Cube）计算尾随块贡献 $Z_{0:n, i_0:i_0+b} \mathrel{+}= Z_{0:n, i_0+b:n} \cdot (-L_{i_0+b:n, i_0:i_0+b})$；
     - 若 $b > 1$，由 `cgetri_solve_panel_kernel`（AIV，64 行分片并行）求解对角单位下三角块 $Z_{0:n, i_0:i_0+b} \cdot L_{i_0:i_0+b, i_0:i_0+b}^{-1}$。
   - **Phase 3（`cgetri_store_perm_kernel`）**：在 UB 内逆序回放 `devIpiv` 得到最终列映射表 `perm[j]`，多核并行从 `ar/ai` 的第 `perm[j]` 列读取并交织写回 `A` 的第 $j$ 列。

#### 4. `aclsolverCgetriBatched` Kernel 设计

针对跨度极大的矩阵维数（$n \in [2, 3049]$）与批量规模（$\text{batchSize} \in [32, 10^6]$），设计三层精细化内核：

1. **第一层：小矩阵批主序（Batch-Major）向量核（$n \le 32$）**：
   - 将 $G$ 个独立矩阵在 UB 最内维交织排布：`mr[(col * n + row) * G + g]`，使单条向量指令同时处理 $G$ 个矩阵的同一矩阵元：
     - $n = 2$：`SuperN2Solver`（$G = 1024$），单核一次吞吐 1024 个 $2 \times 2$ 复矩阵，采用 `SetFlag`/`WaitFlag` 流水同步与显式代数展开，无任何标量循环；
     - $n = 4$：`SuperN4Solver`（$G = 256$），利用跨行合并向量指令一次更新整列 $4 \times G = 1024$ 个浮点数；
     - $n \in [1, 32]$ 通用：`BatchMajorGetri<G, MAX_N, LCOLS, BCOLS>`（$n \le 4$ 取 $G=128$，$n \le 8$ 取 $G=64$，$n \le 20$ 取 $G=32$，$n \le 32$ 取 $G=16$）。其中 $G \in \{64, 32, 16\}$ 的核心三角求逆与下三角求解采用 Ascend 950 **`__simd_vf__` 寄存器级向量函数**（`SolveRightUpperN16VF`、`SolveRightUnitLowerN16VF`、`BuildColPermVF`），将每个输出元素在向量寄存器内累加完毕后再单次写回 UB，消除中间 UB 读写与 `LocalMemBar` 停顿。
2. **第二层：中等矩阵单核驻留 `__simd_vf__` 广播核（$33 \le n \le 96$）**：
   - `cgetri_batched_ub64_kernel`（$33 \le n \le 64$）与 `cgetri_batched_ub96_kernel`（$65 \le n \le 96$）：每个 AIV 核每次独立处理 1 个矩阵；
   - 利用 Ascend 950 `__simd_vf__` 的硬件广播加载指令 `AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>`，在 Phase 1 与 Phase 2 的秩-1 外积内循环中直接从 UB 标量地址广播加载 $-U_{p,c}$ 或 $-L_{c,j}$ 到向量寄存器，并与常驻寄存器的 64 行（或 96 行）列向量执行 `Reg::MulAddDst`，完全省去传统方案中在 UB 内构造 `Brcb` 广播表或 `Gather` 转置表的开销。
3. **第三层：大矩阵多实例分块 Cube + AIV 流水路径（$n > 96$）**：
   - 每次取 `chunkCnt`（至多 8192）个矩阵共享一套分块阶段启动：
     - `cgetri_batched_load_split_kernel`：全 AIV 核并行解引用 `Aarray[bStart + m]`，将 `chunkCnt` 个矩阵拆分为平面格式并完成奇异检测；
     - `cgetri_batched_trtri_panel_kernel`（AIV，内部调用 `__simd_vf__` 寄存器外积核 `SolveRightUpperRank1VF`）与 `cgetri_batched_trtri_gemm_kernel`（AIC Cube，单核每次处理一个矩阵的完整 GEMM 步）交替完成所有矩阵的上三角求逆；
     - `cgetri_batched_solve_gemm_kernel`（AIC Cube）与 `cgetri_batched_solve_panel_kernel`（AIV，内部调用 `__simd_vf__` 核 `SolveRightUnitLowerRank1VF`）逆序完成所有矩阵的下三角求解；
     - `cgetri_batched_store_perm_kernel`：全 AIV 核并行对各自负责的矩阵逆序回放 `PivotArray[(bStart + m) * n]` 并交织写回 `Carray[bStart + m]`。

#### 5. `aclsolverCmatinvBatched` Kernel 设计

采用批主序融合 Gauss-Jordan 消元核 `BatchMajorMatinv<G, MAX_N, LCOLS, BCOLS>`：
- **分组与分档**：$n \le 16$ 采用 $G = 16$（细分为 $n \le 4, 8, 14, 16$ 四档模板核）；$17 \le n \le 32$ 采用 $G = 8$（细分为 $n \le 22, 24, 26, 27, 28, 29, 30, 32$ 八档模板核），使每个 $n$ 都分配紧凑的 UB 空间与最大的列面板宽度 `LCOLS`/`BCOLS`。
- **UB 布局**：同时在 UB 内维护输入矩阵平面 `mr, mi`（大小 $n^2 G$）与伴随右端矩阵平面 `xr, xi`（大小 $n^2 G$，初始化为 $I_n$）。
- **批主序全向量化 Gauss-Jordan 步（$p = 0 \dots n-1$）**：
  1. **选主元**：对行 $r \in [p, n-1]$，以长度为 $(n - p)G$ 的单条 `Mul` + `MulAddDst` 算出全部候选行的模长平方，经 `Max` + `CompareScalar` 产生无分支 0/1 掩码，选出模最大且行号最小的主元行；
  2. **条件行交换**：当存在矩阵需要换行时，按列分块利用掩码向量算术同步交换 `mr/mi`（列 $p \dots n-1$）与 `xr/xi`（列 $0 \dots p$）的第 $p$ 行和主元行；
  3. **奇异检测**：检查第 $p$ 主元模长平方是否为 0，记录每个矩阵首个零主元位置 $\text{info}[g] = p + 1$；
  4. **主元行归一化与全行消元**：计算主元倒数 $M_{p,p}^{-1}$，归一化第 $p$ 行后将其清零（使第 $p$ 行在随后的全行 `MulAddDst` 中保持不变），再按 `bcols` 列分块将第 $p$ 列乘子与第 $p$ 行做秩-1 外积，一次性消去所有 $r \ne p$ 行的第 $p$ 列。

---

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
|---|---|---|
| **Ascend 950PR（Atlas 950）** | √ | 本任务目标验收硬件（编译目标 `--soc=ascend950`，对应 `dav-3510` / `arch35`） |

## 算子约束限制

1. **数据排布与类型**：仅支持 `COMPLEX64`（`aclFloatComplex`）列主序（column-major）输入输出，主元与 `info` 为 `INT32`，不支持 broadcast，不涉及图模式融合。
2. **指针与内存不重叠约束**：
   - `aclsolverCgetrf`、`aclsolverCgetri` 为原地（in-place）操作，`Workspace` 与矩阵 `A`、`devIpiv`、`devInfo` 不得重叠；
   - `aclsolverCgetriBatched`、`aclsolverCmatinvBatched` 为非原地（out-of-place）操作，`Carray[i]` 与 `Aarray[i]`、`Ainv[i]` 与 `A[i]` 内存不得重叠。
3. **接口职责边界**：
   - `aclsolverCgetri` 与 `aclsolverCgetriBatched` 的输入必须是已完成 LU 分解的因子矩阵及配套主元，不对原始未分解矩阵隐式调用 `getrf`；
   - `aclsolverCmatinvBatched` 仅支持 $0 \le n \le 32$，当 $n > 32$ 时直接返回 `ACLSOLVER_STATUS_INVALID_VALUE` 并写入 `info[0] = -1`，绝不静默转调 `aclsolverCgetriBatched`。

---

# 可维可测分析

## 精度标准/性能标准

| 验收维度 | 判定公式与阈值标准 | 标准来源 |
|---|---|---|
| **`aclsolverCgetrf` 精度标准** | 由打包 LU 重放 $L \cdot U$，按 `devIpiv` 逆序（$i = \min(m,n) \to 1$）回放行交换还原至原矩阵行序，计算 1-范数残差比值：<br>$$\text{ratio} = \frac{\|P^{-1}(L \cdot U) - A\|_1}{n \cdot \|A\|_1 \cdot \varepsilon}, \quad \varepsilon = 2^{-24} \approx 5.96 \times 10^{-8}$$<br>满足 $\text{ratio} \le \max(5 \cdot \text{ratio}_{\text{cpu}}, 0.5)$ 判定为 PASS | 任务书 §3.2.1.1（对齐 LAPACK [CGET01](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cget01.f/)） |
| **`aclsolverCgetri` 精度标准** | 将输入 LU 按 `devIpiv` 还原为 $A$，对 NPU 输出逆矩阵 $C$ 计算：<br>$$\text{ratio} = \frac{\|I - C \cdot A\|_1}{n \cdot \|A\|_1 \cdot \|C\|_1 \cdot \varepsilon}, \quad \varepsilon = 2^{-24}$$<br>满足 $\text{ratio} \le \max(5 \cdot \text{ratio}_{\text{cpu}}, 0.05)$ 判定为 PASS | 任务书 §3.2.1.2（对齐 LAPACK [CGET03](https://sources.debian.org/src/lapack/3.7.0-1~bpo8+1/TESTING/LIN/cget03.f/)） |
| **`aclsolverCgetriBatched` 精度标准** | 对 batch 内每个矩阵 $i \in [0, \text{batchSize}-1]$ 独立计算 `CGET03` 残差比值 $\text{ratio}_i$，所有矩阵均满足 $\text{ratio}_i \le \max(5 \cdot \text{ratio}_{\text{cpu},i}, 0.05)$ 判定为 PASS | 任务书 §3.2.1.3（逐矩阵套用 LAPACK `CGET03`） |
| **`aclsolverCmatinvBatched` 精度标准** | 对 batch 内每个矩阵 $i \in [0, \text{batchSize}-1]$ 独立计算 `CGET03` 残差比值 $\text{ratio}_i$，所有矩阵均满足 $\text{ratio}_i \le \max(5 \cdot \text{ratio}_{\text{cpu},i}, 0.5)$ 判定为 PASS | 任务书 §3.2.1.4（逐矩阵套用 LAPACK `CGET03`） |
| **性能标准（全量 652 条用例）** | 在 Ascend 950PR 上使用 `msprof` 采集 NPU Kernel 耗时 $T_{\text{NPU}}$，与任务包各算子目录 `bench_result.json` 中的 GPU 平均耗时 $T_{\text{GPU}}$ 逐条比对，满足：<br>$$T_{\text{NPU}} \le \frac{T_{\text{GPU}}}{0.35} \quad (\text{即性能倍率} \ge 0.35 \times \text{GPU})$$ | 任务书 §3.3 |

### 测试覆盖与验证规划

在 `test/cgetrf/`、`test/cgetri/`、`test/cgetri_batched/`、`test/cmatinv_batched/` 下构建完整的 C++ 测试与基准评测套件：

1. **功能与边界全覆盖**：
   - 基础正向：方阵 `Cgetrf`、`Cgetrf + Cgetri` 端到端求逆、`CgetriBatched` 求逆、`CmatinvBatched`（$n \in [1, 32]$）直接求逆；
   - 矩形与 Padding：$m > n$、$m < n$ 的矩形 `Cgetrf`；$\text{lda}, \text{ldc}, \text{lda\_inv}$ 等于最小合法值以及 $+3, +8, +32$ 等非对齐 padding（带 padding 哨兵不被污染检查）；
   - 选主元模式：`devIpiv != nullptr`（选主元）与 `devIpiv == nullptr` / `PivotArray == nullptr`（不选主元）双模式验证；
   - 异常与奇异分支：`handle == nullptr`、负维数、`lda` 过小、`Lwork` 不足、`n = 33`（`CmatinvBatched` 报错并返回 `info = -1`）、构造零行/成比例列的奇异矩阵验证 `info = k > 0`；
   - 空问题：$m=0$、$n=0$、$\text{batchSize}=0$ 快速成功返回且写 `info = 0`；
   - 确定性与非默认流：同一输入串行重复执行对比输出张量、`ipiv`、`info` 逐字节（bit-wise）一致；自建非默认 `aclrtStream` 下发验证正确性。
2. **全量 652 条基准用例性能评测**：
   - 覆盖 `cgetrf`（148 条）、`cgetri`（200 条）、`cgetribatched`（89 条）、`cmatinvbatched`（215 条）全量用例；
   - 原地修改接口（`Cgetrf`、`Cgetri`）在多次迭代计时及预热时保证输入矩阵有效重放，并使用 `msprof` 采集单次 API 调用下发的设备端 Kernel 耗时与 `bench_result.json` 逐条关联比对。

## 兼容性分析

1. **C ABI 与 C++ 向后兼容**：
   - `include/cann_ops_solver_common.h` 与 `include/cann_ops_solver.h` 使用 `#ifdef __cplusplus extern "C"` 包裹新增的 6 个标准 C 接口，确保纯 C 与 C++ 工程均可直接包含与链接；
   - 在 `#ifdef __cplusplus` 保护下保留仓内原有的 `std::complex<float>*` / `int64_t` 重载声明与实现包装，不破坏仓内已有 C++ 旧测试的编译兼容性。
2. **构建系统兼容**：
   - 沿用 `ops-solver` 仓现有 `build.sh --soc=ascend950` 与 CMake 构建体系，仅更新 `cgetrf`、`cgetri`、`cgetri_batched`、`cmatinv_batched` 四个算子目录及公共头文件，不影响 `cheevj`、`cpotrf` 等其他算子模块。

---

## 希望检视人员了解 / 待确认事项

1. **关于任务书标题中的 `Cgetrs` 与 §2.3 公开接口列表的对齐说明**：
   - 任务书文件名含 `Cgetrs`，且 §1 提到“五个接口”、§3.5 自构造用例表出现 `B / nrhs / trans`；但任务书 **§2.1 功能对标表、§2.3 接口定义、§2.4 参数说明、§3.2 精度判据、§3.3 性能对标及 §7.1 特别注意事项** 明确列出的交付接口均为 **`aclsolverCgetrf`（含 `_bufferSize`）、`aclsolverCgetri`（含 `_bufferSize`）、`aclsolverCgetriBatched`、`aclsolverCmatinvBatched`**，且任务包附件仅包含这四个算子的用例与基线（无 `Cgetrs` 原型与用例）。
   - 本设计严格按 §2.3 定义的 6 个函数签名（4 个计算接口 + 2 个 `bufferSize` 接口）及 `aclsolverOperation_t` 枚举交付。请检视老师确认是否以此为准。
2. **关于任务包附件用例超出正文规模表下限的处理**：
   - 任务书 §2.4 表格写明验收下限为 $m, n \in [1, 4096], \text{batchSize} \in [1, 10^6]$，同时注明“允许实现支持更大范围并在文档中声明”；而任务包附件 `cgetrf/cases.json` 含 $n \le 8168$，`cmatinvbatched/cases.json` 含 $\text{batchSize} = 10^7$。本设计已将实现支持范围扩展至覆盖附件全部 652 条用例（$m \le 4096, n \le 8168, \text{batchSize} \le 10^7$）。
3. **关于分块多 Kernel 路径的 `msprof` 耗时统计口径**：
   - 对于中大尺寸矩阵的分块实现（如 `Cgetrf`、`Cgetri` 及大尺寸 `CgetriBatched`），单次 API 调用在 Device 侧由数据拆分、面板分解/求解、Cube GEMM、合并写回等多个子 Kernel 顺序组成。我们在性能验证中按**单次 API 调用所下发的全部设备端 Kernel `Task Duration(us)` 之和**（并同时对照端到端墙钟均值）与 `bench_result.json` 中的 `perf.avg_ms / 0.35` 进行比对，确保不遗漏任何阶段耗时。
