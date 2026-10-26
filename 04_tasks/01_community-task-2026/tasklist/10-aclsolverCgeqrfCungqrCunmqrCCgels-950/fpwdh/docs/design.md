# 【社区任务】Atlas 950 单精度复数 QR 分解与最小二乘（Cgeqrf/Cungqr/Cunmqr/CCgels）算子设计文档

> **任务**：10月社区任务-单精度复数QR分解与最小二乘(950)
> **目标仓**：`cann/ops-solver`（Host C API + Ascend C/CATLASS Kernel 直调）
> **适配硬件**：Ascend 950PR（Atlas 950，dav-3510 / arch35）｜**CANN**：9.1.0
> **对标**：NVIDIA cuSolver DN `cusolverDnCgeqrf / Cungqr / Cunmqr / CCgels`
> **提交者**：fpwdh ｜ 四接口作为同一社区任务一并交付，不拆分验收

---

# 第 1 部分：需求背景

## 1.1 需求来源

本需求来自 CANN 社区 2026 年 10 月批次任务「单精度复数 QR 分解与最小二乘(950)」。任务要求在昇腾 NPU 上基于 Ascend C/CATLASS 开发稠密复数 QR 分解（`aclsolverCgeqrf`）、酉变换（`aclsolverCungqr`）、Q 乘（`aclsolverCunmqr`）、最小二乘求解（`aclsolverCCgels`）四接口，接口、功能、参数、约束与 cuSolver DN 对齐，精度达 CANN 生态算子开源标准，验收通过后合入 `cann/ops-solver`。

## 1.2 背景介绍

### 1.2.1 数学语义

| 接口 | 数学语义 | 对标 CUDA |
|---|---|---|
| `Cgeqrf` | A = Q·R（Householder 反射器原地存储：上三角= R，严格下三角= v(2:)，v[0]=1 隐式，TAU[0:min(m,n)]） | `cusolverDnCgeqrf` |
| `Cungqr` | 由 geqrf 输出 (v,tau) 生成前 n 列显式酉矩阵 Q（m×n，m≥n≥k≥0），原地覆盖 A | `cusolverDnCungqr` |
| `Cunmqr` | C := op(Q)·C（LEFT）或 C·op(Q)（RIGHT）；op(Q)=Q（N）或 Qᴴ（C）；**trans=T 必须报错** | `cusolverDnCunmqr` |
| `CCgels` | min ‖A·X−B‖_F（m≥n），X out-of-place 写独立缓冲，niter 报告迭代次数 | `cusolverDnCCgels` |

数据布局统一 **列主序（column-major）**，leading dimension 须满足对标约束（禁止要求 lda==m/n），COMPLEX64（与 `cuComplex` 同布局：两连续 FLOAT32 real/imag）。

### 1.2.2 算子现状分析

ops-solver 仓现有 solver 族为 LU（cgetrf/cgetri/sgetrf/sgetri）与 Cholesky 族（cheevj），**无复数 QR / gels 原型**，须全新实现。可复用资产：`src/utils/kernel/c64/`（gemm/trsm/pad/getf2 复数积木）、`lu_host_common.h`（gather/对齐常量）、cgetrf host 骨架（参数校验/cleanup/尺寸推导模式）。

### 1.2.3 硬件平台

Ascend 950PR（dav-3510）：AIC:AIV 异构，UB ~192KB/AIV，HBM 128GB。本设计四接口全部运行在 AI Core 上（任务书 §2.1.5 禁 CPU fallback）。

---

# 第 2 部分：需求分析

## 2.1 需求描述

| 项目 | 契约 |
|---|---|
| 输入 A / dA | COMPLEX64，列主序 ND，lda×n（有效 m×n），有限复数 |
| 输出 TAU / tau | COMPLEX64 [min(m,n)]（geqrf）/ [k]（ungqr/unmqr） |
| info / devInfo | INT32 [1]，Device 真实写入 0 / −i（1-based 参数序，handle 不计） |
| 空问题 | m=0/n=0/k=0/nrhs=0 成功返回 |
| 维数类型 | `int`（32-bit，对齐 CUDA legacy API） |
| 确定性 | 同输入同 stream 串行多次执行输出（含 tau/info）bit-wise 一致 |
| 性能 | T_NPU ≤ T_GPU(cuSolver)/0.35（msprof Task Duration 口径） |

## 2.2 需求拆解

1. **功能**：四接口数学语义与 cuSolver DN 对齐；Cunmqr trans=T 显式报错；CCgels 仅 m≥n。
2. **精度**：LAPACK 残差判据 CQRT01/02/03/16/17（任务书 §3.2.1），ratio ≤ max(5·ratio_cpu, 阈值)。
3. **性能**：P-01~P-12 全部达标（0.35× GPU），msprof 口径。
4. **工程化**：Device 指针语义（任务书 §2.3）、workspace 查询接口、bit-wise 确定性、非默认 stream 正确。

## 2.3 接口定义（按任务书 §2.3 原样签名）

```c
typedef struct { float real; float imag; } aclFloatComplex;
typedef enum { ACLSOLVER_OP_N=0, ACLSOLVER_OP_T=1, ACLSOLVER_OP_C=2 } aclsolverOperation_t;
typedef enum { ACLSOLVER_SIDE_LEFT=0, ACLSOLVER_SIDE_RIGHT=1 } aclsolverSideMode_t;

aclsolverStatus_t aclsolverCgeqrf_bufferSize(aclsolverHandle_t, int m, int n, aclFloatComplex *A, int lda, int *Lwork);
aclsolverStatus_t aclsolverCgeqrf(aclsolverHandle_t, int m, int n, aclFloatComplex *A, int lda,
                                  aclFloatComplex *TAU, aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCungqr_bufferSize(aclsolverHandle_t, int m, int n, int k, const aclFloatComplex *A,
                                             int lda, const aclFloatComplex *tau, int *lwork);
aclsolverStatus_t aclsolverCungqr(aclsolverHandle_t, int m, int n, int k, aclFloatComplex *A, int lda,
                                  const aclFloatComplex *tau, aclFloatComplex *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverCunmqr_bufferSize(aclsolverHandle_t, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                             int m, int n, int k, const aclFloatComplex *A, int lda,
                                             const aclFloatComplex *tau, const aclFloatComplex *C, int ldc, int *lwork);
aclsolverStatus_t aclsolverCunmqr(aclsolverHandle_t, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                  int m, int n, int k, const aclFloatComplex *A, int lda, const aclFloatComplex *tau,
                                  aclFloatComplex *C, int ldc, aclFloatComplex *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverCCgels_bufferSize(aclsolverHandle_t, int m, int n, int nrhs, aclFloatComplex *dA,
                                             int ldda, aclFloatComplex *dB, int lddb, aclFloatComplex *dX, int lddx,
                                             void *dwork, size_t *lwork_bytes);
aclsolverStatus_t aclsolverCCgels(aclsolverHandle_t, int m, int n, int nrhs, aclFloatComplex *dA, int ldda,
                                  aclFloatComplex *dB, int lddb, aclFloatComplex *dX, int lddx, void *dWorkspace,
                                  size_t lwork_bytes, int *niter, int *dinfo);
```

---

# 第 3 部分：详细设计

## 3.1 算子分析

### 3.1.1 数学公式（LAPACK CLARFG 约定）

单个 Householder 反射器：

```
H = I − τ·v·vᴴ,  v[0]=1（隐式）,  Hᴴ·x = β·e₁
τ = (β − α)/β,  v(2:m) = x(2:m)/(α − β)
β = −sign(Re(α))·√(Re(α)² + Im(α)² + ‖x(2:m)‖²)
```

QR 分解（geqrf）：对 j=0..min(m,n)−1 依次构造反射器并更新 A(j:m, j+1:n)：

```
A(j:m, j+1:n) −= conj(τ_j)·v_j·(v_jᴴ·A(j:m, j+1:n))    [Hᴴ 更新]
```

显式 Q 生成（ungqr）：Q = H₀·H₁···H_{k−1}（**τ 不共轭**），按列从右到左：

```
列 j = H_{min(j,k−1)}···H₀·e_j（降序应用最右先）；列 j 只需反射器 0..min(j,k−1)
（H_i·e_j = e_j for i > j，因 v_i 支撑 [i, m)）
```

Q 乘（unmqr）四组合反射器应用顺序（LAPACK CUNMQR）：

| side | trans | 应用顺序 | τ 变换 |
|---|---|---|---|
| LEFT | N | i = k−1 .. 0 | τ |
| LEFT | C | i = 0 .. k−1 | conj(τ) |
| RIGHT | N | i = 0 .. k−1 | τ |
| RIGHT | C | i = k−1 .. 0 | conj(τ) |

最小二乘（CCgels）组合：geqrf → unmqr(LEFT,C,k=n)（B := Qᴴ·B）→ 复数上三角回代 R·X = B[0:n,:]。

### 3.1.2 复数运算拆解

复数按 split-complex（real/imag 两平面）在 kernel 内计算（KB-0004：GM 交错存储 ↔ UB 拆平面，算完交织回写）。核心运算：

| 复数运算 | 实数指令实现 |
|---|---|
| 复数乘 (a+bi)(c+di) | (ac−bd) + i(ad+bc) |
| 复数除 (a+bi)/(c+di) | (ac+bd)/den + i(bc−ad)/den, den=c²+d² |
| conj(τ) | 虚部取反 |
| β = pythag3 | √(Re²+Im²+xnorm²)（防溢出 Newton invSqrt） |

### 3.1.3 host 校验矩阵（任务书 §2.4）

逐参数校验（1-based 参数序，handle 不计）：非法类型/shape/lda/枚举 → `ACLSOLVER_STATUS_INVALID_VALUE` + devInfo 写 −i；空问题成功返回；trans=T 拒绝（−2）；CCgels m<n 拒绝（−1）。

## 3.2 数据结构与 kernel 架构

```
cgeqrf_host.cpp   ──参数校验/workspace 查询──→ cgeqrf_kernel_do ──→ cgeqrf_kernel（unblocked Householder, 单核）
cungqr_host.cpp   ──参数校验/workspace 查询──→ cungqr_kernel_do ──→ cungqr_kernel（逐列反向累积, 单核）
cunmqr_host.cpp   ──参数校验/workspace 查询──→ cunmqr_kernel_do ──→ cunmqr_kernel（四组合反射器应用, 单核）
ccgels_host.cpp   ──参数校验/workspace 查询──→ ccgels_kernel_do ──→ geqrf + cunmqr + ctrsm_upper_kernel（组合）
```

**Device 指针语义**（任务书 §2.3）：A/TAU/Workspace/dA/dB/dX 全部为调用方 device buffer，host 层不做 H2D/D2H staging，走调用方 stream，禁无必要 Host 同步。

**workspace 布局**（CCgels）：前 2n floats = tau，其余 scratch；`_bufferSize` 以**字节**计（cuSolver 约定），其余三接口以 aclFloatComplex 元素计。

## 3.3 性能优化路线（M5 进行中）

当前 M2~M4 为单核标量 GM 版（正确性优先）。性能路线（按 KB-0003 三级 GEMM 更新范式）：

1. **blocked Householder**（geqrf）：panel 宽 nb=64，larft 生成块反射矩阵 T，larfb 用 `CMatmulCustom`（c64 gemm.hpp 复数四实 GEMM）做 rank-k 更新；
2. **UB/SIMD 化**：列向量/scratch 装入 UB（`TBuf<TPosition::VECCALC>` + `Get<float>()`），解交织用 VreduceV2；
3. **多核并行**：gels 的回代/ungqr 的列生成按列块分核（AIV 分核）；
4. **固定开销**：小形状（n≤256）走单核标量路径（launch 开销主导）。

---

# 第 4 部分：可维可测分析

## 4.1 精度验证（CQRT01/02/03/16/17）

| 接口 | 判据 | 判定阈 | 实测（M2~M4） |
|---|---|---|---|
| Cgeqrf | CQRT01: ‖R−QᴴA‖₁/(m‖A‖₁ε) | ≤max(5·ratio_cpu, 0.1) | 4/4 PASS（8×8~128×64） |
| Cungqr | CQRT02: ‖I−QᴴQ‖₁/(m·ε) | ≤max(5·ratio_cpu, 1.5) | 4/4 PASS（含 k<n） |
| Cunmqr | CQRT03: ‖op2−op1‖₁/(m‖C‖₁ε) | ≤max(5·ratio_cpu, 0.1) | 10/10 PASS（LEFT/RIGHT×N/C） |
| CCgels | CQRT16（方阵）/CQRT17（超定） | ≤max(5·ratio_cpu, 0.015) / ≤10·ratio_cpu | 4/4 PASS |

**golden 生成**：scipy.linalg.lapack COMPLEX64 版本（cgeqrf/cungqr/cunmqr/cgels），判据按 LAPACK 3.7.0 TESTING/LIN 源码（cqrt01.f~cqrt17.f）。

## 4.2 功能自验矩阵（任务书 §3.5）

基础功能（geqrf/ungqr 还原 Q/geqrf+unmqr/CCgels）、矩形（m>n/m=n/m<n 仅 geqrf）、side×trans（LEFT/RIGHT×N/C，T 报错）、CCgels（nrhs=1/多右端/m=n/m>n/m<n 报错）、padding（lda/ldc>+8/+32）、info（0/−i）、空问题、确定性、非默认 stream。

## 4.3 性能验证

P-01~P-12 全量 case，msprof `OpBasicInfo.csv` kernel Task Duration 平均口径，≤GPU/0.35。

## 4.4 可维护性

- 头文件 C 可调用（`extern "C"`），`aclFloatComplex` 与 `cuComplex` 同布局；
- 四接口独立 host 文件 + 独立 kernel 文件，复用 `qr_host_common.h` 校验宏；
- kernel 数学约定（CLARFG/顺序/共轭）在文件头注释固化，与 LAPACK 源码一一对应；
- workspace 查询接口与 compute 接口分离，便于调用方预分配。
