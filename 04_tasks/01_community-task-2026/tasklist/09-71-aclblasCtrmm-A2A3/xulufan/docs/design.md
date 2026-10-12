# aclblasCtrmm 算子设计文档（Atlas A2/A3 / arch22）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclblasCtrmm 算子开发(A2A3)」。在 `ops-blas` 仓
（https://gitcode.com/cann/ops-blas）为 Atlas A2/A3 系列（DAV-2201，`arch22`，覆盖
910B3/910B4/910_93 等）实现单精度复数（COMPLEX64）三角矩阵乘算子 `aclblasCtrmm`：
`C = alpha·op(A)·B`（side=LEFT）或 `C = alpha·B·op(A)`（side=RIGHT），列主序、
离席写 C，kernel 直调（Ascend C）。

代码交付 `blas/trmm/arch22/`，测试交付 `test/trmm/ctrmm/arch22/`；设计文档交付
`cann-ops-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/09-71-aclblasCtrmm-A2A3/`
（目录已存在）。

## 背景介绍

### 算子语义：GEMM 型算子 + 三角结构裁剪

A 为三角矩阵（uplo 指定上/下三角，全存储 m×m 或 n×n），op(A) ∈ {A, Aᵀ, Aᴴ}；
B、C 为一般矩形 m×n；diag=UNIT 时对角视为 1 且**不读取**。乘法语义（与 trsm 求解类
的根本区别）：无回代、无对角非零要求——NON_UNIT 时对角元允许为零（全零 A 合法）；
alpha=0 时 A/B 不被引用（可为 nullptr）、C 全部置零；m=0 或 n=0 为合法 no-op。

与 cuBLAS 对齐的两个语义要点（直接影响 kernel 正确性）：

1. **带外不引用**：A 的非三角区是调用方未定义内存（可能含 Inf/NaN 垃圾），任何
   情况下不得读入运算——这同时是 NaN 精度用例的硬约束：golden（Netlib `ctrmm`）
   只循环三角区，`0×NaN` 从不发生；若实现把带外当 0 参与全量 GEMM，垃圾 NaN 会
   扩散到全矩阵，与 golden 不一致。**「只算三角」既是减半 FLOPs 的性能手段，
   更是 NaN 语义的正确性要求。**
2. **B≡C 原地**：cuBLAS 允许 B 与 C 同址；除此之外 C 与 A/B 不允许重叠。

### ops-blas 现状分析

`blas/trmm/` 当前仅有 **arch35 的 `aclblasStrmm`（FP32）实现**（`strmm_host.cpp /
strmm_kernel.cpp / strmm_tiling_data.h`），基于 `master` 源码与 README 核对：

- `blas/trmm/README.md` 仅登记 `aclblasStrmm`，产品支持表中 Atlas A2/A3 系列标注
  「不支持」；**`aclblasCtrmm` 接口在仓内完全不存在**（README、
  `include/cann_ops_blas.h` 均无）——本任务是**新接口 + 新架构**双重新增；
- arch35 strmm 语义基线（README 核对）：m=0/n=0 不访问任何指针、不校验 ld（BLAS
  标准）；alpha==0 时 C 清零且 A/B 可为 nullptr、C 必须有效；C 为纯输出（无 beta·C
  项）；精度档 FLOAT32（rtol 2⁻¹⁰ / atol 2⁻¹⁶）——本任务按任务书 COMPLEX64 档
  （2⁻¹³/2⁻¹³）执行，严于 arch35 README 口径；
- 复数 GEMM 类先例：`blas/gemm3m/`（arch35）即复数 3 乘法（3M）GEMM 算法，仓内
  已有「复数 → 实数 GEMM 分解」的语义先例；
- arch22 侧先例：`blas/tbmv/arch22/`、`blas/trsv/arch22/`、`blas/tpmv/arch22/`
  （`*_host.cpp / *_kernel.cpp / *_kernel.h / *_tiling_data.h` 四件套 +
  `*_kernel_do` 启动约定）；**`blas/gemm/` 亦仅有 arch35，仓内无可复用的 arch22
  GEMM 基建**，cube 通路需在本算子内自建；
- 公共设施：`common/helper/{aclblas_handle_internal.h, kernel_constant.h,
  host_utils.h}`；构建分发按 SOC 将 `ascend910b*`、`ascend910_93*` 映射到 `arch22`
  目录，新增目录自动参与构建。

### 功能缺口

**缺口一：`aclblasCtrmm` 接口整体缺失。** 无声明、无 README、无实现、无测试。

**缺口二：arch22 缺失。** `blas/trmm/` 无 arch22 目录；且仓内无 arch22 GEMM/cube
先例，需要从零建立「复数三角乘 → 实数 GEMM 分解 + cube Matmul」的 arch22 通路。

**缺口三：复数语义。** arch35 strmm 是实数实现，OP_C 等同 OP_T；复数版需按
aclblasComplex 实虚部处理共轭，仓内 trmm 目录无先例（gemm3m 可作分解语义参照）。

### 功能分析：为什么「对齐 Netlib golden」决定了算法骨架

Netlib `ctrmm` 按三角区循环（结构零从不参与运算），因此 NPU 实现必须保证：
**结构零 × 任何值（含 NaN/Inf）从不被计算**。这排除了「A 物化为稠密含零矩阵 +
不带跳过的全量 GEMM」的朴素方案（0×NaN=NaN 会扩散），要求 kernel 的 K 循环对
「整体位于三角外的 K-tile」静态跳过（跳过表由 side/uplo/trans 在 Host 侧导出，
见 §3.3）。跳过同时把有效 FLOPs 减半，是性能达标的前置条件（§3.5）。

---

# 需求分析（required）

## 需求描述

在 `arch22` 上实现 `aclblasCtrmm`：COMPLEX64 三角矩阵乘，side×uplo×trans×diag
（2×2×3×2 = 24 组）全枚举组合语义对齐 cuBLAS `cublasCtrmm`；离席写 C、允许 B≡C；
alpha 为 Host 复数标量指针，alpha=0 时 C 清零且 A/B 不引用；m=0/n=0 no-op；
Inf/NaN 传播语义对齐 Netlib golden（带外与 UNIT 对角不读）；异步执行；5 条典型
性能 case 达标（GPU 实测 ÷ 0.8，msprof kernel 口径），全量精度用例满足 COMPLEX64
混合容差。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | arch22 全新实现：Host 分发 + Ascend C kernel（AIV 直算 + AIC cube Matmul 双路径）+ 构建 + 测试 | §1、§2.2、§5 | 新增（主体） |
| R2 | 新接口登记：`include/cann_ops_blas.h` 声明 + `blas/trmm/README.md` aclblasCtrmm 章节 | §4 | 新增 |
| R3 | side×uplo×trans×diag 24 组枚举全组合，语义对齐 cublasCtrmm（含 C 共轭转置、UNIT 对角不读） | §2.1 | 新增 |
| R4 | 离席写 C；允许 B≡C 原地，其余 A/B/C 重叠禁止；不返回视图 | §2.5 | 规定 |
| R5 | alpha Host 指针；alpha=0 → C 的 m×n 区域按 ldc 步长精确清零（位精确 +0）、A/B 不引用（可 nullptr）；m=0 或 n=0 no-op（不访问指针、不校验 ld） | §2.4、§2.5 | 规定 |
| R6 | 三角外与 UNIT 对角**不读取**（含垃圾 NaN/Inf 不读）→ NaN 语义对齐 Netlib golden | §2.1、§3.5 | 需保证并补测 |
| R7 | 精度：COMPLEX64 混合容差（rtol=atol=2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP）；alpha=0 全零 EXACT | §3.2 | 约束 |
| R8 | 性能：5 条典型 case 平均单次 kernel 耗时 ≤ GPU 实测 ÷ 0.8（msprof 口径，10 次有效采样） | §3.3 | 约束 |
| R9 | 异步语义：`aclblasSetStream` 绑定 stream，Host 侧不做流同步 | §2.5 | 规定 |
| R10 | 交付：README 支持矩阵、测试工程 `test/trmm/ctrmm/arch22/`、自测报告（910B3 + A3） | §4、§5 | 交付 |

---

# 详细设计（required）

## 算子分析

### 数学公式

```text
side=LEFT :  C = alpha · op(A) · B，  A 为 m×m
side=RIGHT:  C = alpha · B · op(A)，  A 为 n×n
op(A) = A (N) / Aᵀ (T) / Aᴴ (C)；B、C 为 m×n 一般矩阵，列主序
diag=UNIT 时 op(A) 对角恒为 1 且不读取 A 对角
```

复数分解（实虚部均 FP32）：设 P = op(A)、X = B，则

```text
C_r = α_r·(P_r·X_r − P_i·X_i) − α_i·(P_r·X_i + P_i·X_r)
C_i = α_r·(P_r·X_i + P_i·X_r) + α_i·(P_r·X_r − P_i·X_i)
```

即 4 个实数 GEMM（P_r·X_r、P_i·X_i、P_r·X_i、P_i·X_r）+ alpha 复数缩放尾乘。
侧向归一：RIGHT 视作 GEMM 中两操作数换位（X·P），复用同一套 4-GEMM 骨架。

### 支持数据类型

| 数据 | 类型 | 说明 |
|---|---|---|
| alpha / A / B / C | COMPLEX64（aclblasComplex） | 实/虚部 FP32；alpha 为 Host 指针 |
| m / n / lda / ldb / ldc | int（Host） | — |
| side / uplo / trans / diag | 枚举（Host） | 归一化为「三角侧 × 侧向 × 共轭 × 单位对角」 |

### 支持形状

m, n ∈ [0, INT32_MAX]（m=0 或 n=0 no-op）；side=LEFT 时 A 为 m×m、lda ≥ max(1,m)，
side=RIGHT 时 A 为 n×n、lda ≥ max(1,n)；ldb、ldc ≥ max(1,m)；C 与 A/B 除 B≡C 外
不重叠。支持方阵、fat（m≪n）、thin（m≫n）非方阵全谱。

## 算子实现

### 3.1 总体结构

```text
include/cann_ops_blas.h            新增 aclblasCtrmm 声明（供产品线共用）
blas/trmm/arch22/                  本任务新增：
  ctrmm_host.cpp                     校验 + alpha Host 读取 + 路径分派 + tiling + 启动
  ctrmm_tiling_data.h                arch22 tiling 结构（含三角跳过表）
  ctrmm_kernel.h                     各 *_kernel_do 声明
  ctrmm_kernel_aiv.cpp               路径 A：小规模 AIV 直算 kernel（含 alpha=0 清零）
  ctrmm_kernel_cube.cpp              路径 B：cube Matmul 4-GEMM kernel
  ctrmm_prepost.cpp                  路径 B 前后处理 kernel（物化/解交织/尾乘，
                                    可并入 AIV kernel 分支）
blas/trmm/README.md                新增 aclblasCtrmm 章节 + A2/A3 支持矩阵
test/trmm/ctrmm/arch22/            csv 驱动 GTest（镜像仓内 blas 测试结构）
```

### 3.2 Host 侧设计

#### Host 调用流程图

```text
┌──────────────────────────── Host ────────────────────────────┐
│  aclblasCtrmm(handle, side, uplo, trans, diag, m, n,          │
│               alpha, A, lda, B, ldb, C, ldc)                  │
│                                                               │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐  │
│  │ handle/  │──→│ 枚举/m/n/ld/ │──→│ m==0 || n==0          │  │
│  │ 枚举校验  │   │ alpha 非空    │   │ → SUCCESS（不碰指针） │  │
│  └──────────┘   └──────┬───────┘   └───────────┬───────────┘  │
│                        │                       │              │
│              ┌─────────▼─────────┐   ┌─────────▼───────────┐  │
│              │ 读 alpha（Host 内存│   │ alpha==0            │  │
│              │ 复数标量，无 D2H） │   │ → 启动清零 kernel    │  │
│              └─────────┬─────────┘   │ （ldc 步长列清零）    │  │
│                        │             └─────────────────────┘  │
│  ┌─────────────────────▼──────────────────────────────────┐   │
│  │ 归一化：三角侧 T(opA) / 侧向 / 共轭 / 单位对角            │   │
│  │ B≡C 检测（指针相等）→ 强制路径 B 或路径 A+B 拷贝          │   │
│  └─────────────────────┬──────────────────────────────────┘   │
│              ┌─────────▼─────────┐   ┌─────────────────────┐  │
│              │ 小规模判定          │   │ workspace 预算/分配  │  │
│              │ m·n·min(m,n) ≤ 阈值│   │ （仅路径 B，§3.6）    │  │
│              └─────────┬─────────┘   └──────────┬──────────┘  │
│              ┌─────────▼────────────────────────▼──────────┐  │
│              │ 路径 A: ctrmm_aiv_kernel_do(tiling, stream)  │  │
│              │ 路径 B: [前处理] → [cube 4-GEMM] → [尾乘]     │  │
│              │        三段 kernel 依序下发同一 stream         │  │
│              └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

要点：

1. **校验顺序**（launch 前全返回）：handle 非空 → side/uplo/trans/diag 枚举
   （`INVALID_VALUE`，任务书口径）→ m ≥ 0、n ≥ 0 → **m==0 或 n==0 直接 SUCCESS，
   不访问任何指针、不校验 ld**（BLAS 标准，arch35 strmm 同）→ alpha 非空 →
   lda/ldb/ldc 前导维约束 → alpha ≠ 0 时 A、B 非空、C 恒非空。
2. **alpha 为 Host 指针**（任务书 §2.4）：Host 直接读值经 tiling 传 kernel，无
   D2H 同步；alpha==0 走专用清零 kernel（A/B 不引用，允许 nullptr）。
3. **归一化**：side×uplo×trans → {物化后 op(A) 的三角侧 T ∈ {U,L}、侧向
   LEFT/RIGHT、共轭标志}；T 与侧向一起在 Host 导出 **K-tile 跳过表**（§3.3），
   kernel 零枚举分支。
4. **B≡C 检测**：指针相等时禁用路径 A 直写（防 tile 间读写竞争，见 §3.4），
   强制路径 B（B 在前处理阶段即被解交织消费，天然安全）或路径 A + B 拷贝。
5. **路径分派阈值**：`m·n·min(m,n) ≤ TH`（初值 2²⁴ ≈ 16.7M，约 256³/2·2）走
   路径 A（单 kernel 直算），否则路径 B；TH 上板标定后定版。
6. **tiling**：`CtrmmTilingData`（arch22 版）携带指针、m/n/sA（=m 或 n）、ld 三元组、
   归一化位、tile 形状（BM/BN/BK）、numM/numN/numTile、跳过表（或其紧凑编码）、
   alpha 实虚部、workspace 偏移。

### 3.3 Kernel 侧设计

#### 路径 B（大规模主路径）：复数 4-GEMM + 静态三角跳过

**阶段 1 前处理（AIV，单 kernel 两分支）**：

- **物化 op(A)** → 两个实数 FP32 矩阵 `P_r/P_i`（sA×sA，sA = m（LEFT）或 n
  （RIGHT））：只读 A 三角区；T 转置在写侧完成（写 `P(i,k) = A(k,i)`）；C 共轭
  取 `P_i = −A_i`；UNIT 对角直接写 1.0（**不读 A 对角**）；三角外写 0。
- **解交织 B** → `X_r/X_i`（两个实数 m×n）：交织 complex64 → 双实数平面。

**阶段 2 cube kernel（单 kernel，4 个 GEMM 通道共享 tile 装载）**：

```text
┌──────────── Device (AIC × cores, KERNEL_TYPE_AIC_ONLY) ───────────┐
│  grid: (m/BM) × (n/BN) tiles，core 循环认领 tile                   │
│                                                                    │
│  for tile (mi, nj) in 本核 tiles:                                  │
│    Cr = 0; Ci = 0                        ← 双累加器（或两遍扫描）  │
│    for kt = K-tiles（按跳过表裁剪后的范围）:                       │
│      ┌────────────┐  ┌────────────┐  ┌─────────────────────────┐  │
│      │ LoadData   │  │ LoadData   │  │ 4 路 MMA（同源 tile）：  │  │
│      │ P_r/P_i    │─▶│ X_r/X_i    │─▶│ Cr += P_r·X_r − P_i·X_i │  │
│      │ tile→UB/L0 │  │ tile→UB/L0 │  │ Ci += P_r·X_i + P_i·X_r │  │
│      └────────────┘  └────────────┘  └─────────────────────────┘  │
│    （物化矩阵已含零，跳过表只跳「整 tile 全零」的 K 段，           │
│      减半 FLOPs 且保证 0×NaN 从不发生）                            │
│    写 Cr/Ci 分量到 workspace（实数平面，连续写）                    │
└────────────────────────────────────────────────────────────────────┘
```

**K-tile 静态跳过表**（物化后的 op(A) 三角侧 T × 侧向；Host 侧导出，kernel 查表
得每个 M/N-tile 的 K 起始 tile 号）：

| 侧向 | T | 输出 tile | K 范围（非零区） | 跳过 |
|---|---|---|---|---|
| LEFT | U | 行块 [i0,i1) | k ≥ i0 | k1 ≤ i0 的 K-tile |
| LEFT | L | 行块 [i0,i1) | k ≤ i1−1 | k0 ≥ i1 的 K-tile |
| RIGHT | U | 列块 [j0,j1) | k ≤ j1−1 | k0 ≥ j1 的 K-tile |
| RIGHT | L | 列块 [j0,j1) | k ≥ j0 | k1 ≤ j0 的 K-tile |

（物化时 trans 已折叠进 T，故跳过表与 trans 无关；UNIT 对角为 1，不影响结构零。）

**阶段 3 尾乘（AIV）**：`C = alpha·(Cr + i·Ci)`，按 ldc 步长交织写回 complex64；
只写 m×n 逻辑区，**不触碰 ld 间隙**；B≡C 安全（阶段 2 只读 workspace）。

tile 形状初值 BM=BN=128、BK=64（FP32 cube），双累加器（Cr/Ci）与两遍扫描两方案
以 L0/UB 实测定版（探针项）；AIC 核数以运行时查询为准（910B3 约 20），tile 数不足
时启用 K 方向二次切分。

#### 路径 A（小规模）：AIV 直算单 kernel

```text
┌──────────── Device (AIV × cores, 单 kernel) ────────────┐
│  grid: 按 B/C 列条带划分（core j 处理列子集）            │
│                                                           │
│  for j in 本核列:                                         │
│    for i-block in 行方向（按三角 K 范围）:                │
│      装载 A 行切片（三角内，strided 读或对角带分块）       │
│      装载 B 的 K 行切片 → UB                             │
│      C(i,j) = alpha · Σ_tri op(A)(i,k)·B(k,j)            │
│      （UB 内复数乘加，向量 FMA，K 分块累加）              │
│    写回 C(:,j)（ldc 步长）                                │
└───────────────────────────────────────────────────────────┘
```

小规模下 A 的 strided 读与 B 重复装载代价可接受（≤ 256³ 量级）；**只读三角区**
同样由行块的静态 K 范围保证。B≡C 时 Host 先把 B 拷入内部缓冲（m·n·8B）再进 kernel。

#### 复数算术

- 前处理/cube 通路把复数问题完全化为实数矩阵运算，kernel 内复数算术只出现在
  路径 A 的 UB 乘加与阶段 3 尾乘：`(a·x)_r = a_r·x_r − a_i·x_i`、
  `(a·x)_i = a_r·x_i + a_i·x_r`，实虚解交织为独立 FP32 向量（双通路搬运或
  UB 内 Shuffle/Select）。
- alpha 复数乘在尾乘一次完成（每元素 1 次复乘），不折入 GEMM 标量（避免 4 路
  标量注入的分支与精度损失）。
- 累加全程 FP32；对 4096³/2 长度的 K 求和，逐元素混合容差（2⁻¹³、0.99 比例）
  与任务书「大数规约可放宽 2 ULP」的口径足够。

### 3.4 原地（B≡C）与重叠语义

| 场景 | 处理 |
|---|---|
| B ≡ C | 路径 B：B 在阶段 1 被解交织消费，C 只在阶段 3 写，天然安全；路径 A：Host 先拷 B 到内部缓冲 |
| C 与 A/B 部分重叠 | 任务书禁止，不检测（对齐 cuBLAS） |
| ld 间隙（ldc > m） | 尾乘/清零严格按列步长写 m×n 逻辑区，间隙不触碰 |
| alpha=0 清零 | 专用 kernel：每列 m 个 +0.0f（位精确），按 ldc 步长；A/B 不读（nullptr 合法） |

### 3.5 性能设计

任务书口径：NPU 平均单次 kernel 耗时（msprof `OpBasicInfo.csv` 的 `Task Duration`
平均）≤ `gpu_ms ÷ 0.8`；10 次有效采样、msprof 自带 5 次 warmup。
**注意：路径 B 为三段 kernel 流水，达标口径取同一次 `aclblasCtrmm` 调用触发的
全部 kernel 耗时之和**（与测试指导「同调用范围全部 kernel」一致）。

| case | m=n | side | uplo | trans | diag | GPU 实测 (us) | 达标预算 (us) | 三角有效 FLOP | 需有效算力 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | LEFT | UPPER | N | NON_UNIT | 83.7 | 104.6 | 67 MFLOP | 启动开销支配 |
| 2 | 512 | LEFT | LOWER | C | NON_UNIT | 188.1 | 235.2 | 0.54 GFLOP | 2.3 TFLOPS |
| 3 | 1024 | LEFT | UPPER | T | UNIT | 643.0 | 803.8 | 4.3 GFLOP | 5.4 TFLOPS |
| 4 | 2048 | RIGHT | LOWER | N | NON_UNIT | 2592.5 | 3241 | 34.4 GFLOP | 10.6 TFLOPS |
| 5 | 4096 | LEFT | UPPER | C | NON_UNIT | 15944.6 | 19931 | 274.9 GFLOP | 13.8 TFLOPS |

（FLOP 按 4 实 GEMM × 三角半区计；GPU 有效算力 0.8~17.2 TFLOPS，与 A100 FP32
CUDA 核 + 三角跳过的行为一致——基线不是张量核路径，NPU cube FP32 有真实胜算。）

**关键结论一：三角跳过是达标前置条件。** 不跳过则 FLOPs 翻倍，case 5 需有效
27.6 TFLOPS，风险不可控；跳过后 13.8 TFLOPS。910B3 FP32 cube 峰值以实板标定
（探针项），按有效利用率 35%–70% 评估 case 5 耗时 10–20 ms，预算内且有余量；
若实测峰值不足，候选优化：3M 分解（4 GEMM→3 GEMM，−25% FLOPs，gemm3m 语义先例）、
tile 形状调优、双累加器合批。

**关键结论二：小 case 是启动开销战。** case 1（256²）GPU 仅 83.7us，预算 104.6us；
路径 B 三段 kernel 串行下发 + 计算合计估 40–80us，压线可行但紧。对策：路径 A
单 kernel 覆盖小规模（256² 直算估 30–60us）；阈值 TH 上板用 case 1/2 实测标定，
两条路径都已实现，探针后定版。

**归因方法**：第一步即 `msprof --task-time=on` 拿逐 kernel 耗时（三段分别归因），
已知候选：阶段 1 物化与解交织合并度、tile/L0 命中率（P 通道 4 用）、K 跳过表
粒度（tile 边界损耗 ~B_K/2 每 tile）。

### 3.6 workspace 与内存

路径 B workspace：`P_r/P_i`（2·sA²·4B）+ `X_r/X_i`（2·m·n·4B）+ `Cr/Ci`
（2·m·n·4B），case 5（4096²）峰值 ~384 MB——远小于 HBM，且任务书 §3.4 内存
要求不涉及；优化项：`Ci` 与 `X_i` 缓冲别名复用（阶段 2 的 GEMM 消费顺序支持），
峰值可降 1/3。路径 A 无 workspace（B≡C 时 +m·n·8B 拷贝）。workspace 由 Host 按
tiling 预算 `aclrtMalloc`/`aclrtFree` 惰性管理，随调用释放，不跨调用持有。

## 支持硬件

| 芯片 | 支持情况 |
|---|---|
| Atlas 800I/T A2（910B3 / 910B4 等） | 支持（arch22 / DAV-2201），CANN 9.1.0+ |
| Atlas A3 训练/推理系列（910_93 等） | 支持（arch22 / DAV-2201） |

自验证优先 910B3（精度与性能）；交付含 910B3 与 A3 两个系列的功能/精度/性能报告
（任务书 §7：A2 与 A3 均需完成验收）。

## 算子约束限制

1. 仅 COMPLEX64；不支持其他 dtype。
2. side ∈ {LEFT, RIGHT}、uplo ∈ {UPPER, LOWER}、trans ∈ {N, T, C}、
   diag ∈ {NON_UNIT, UNIT}，非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. m ≥ 0、n ≥ 0；m=0 或 n=0 返回 SUCCESS 且不访问任何指针、不校验 ld；以下仅
   m>0 且 n>0 时生效：lda ≥ max(1, sA)、ldb ≥ max(1,m)、ldc ≥ max(1,m)。
4. alpha 不可为 nullptr（`INVALID_VALUE`）；alpha=0 时 A/B 不引用（可 nullptr）、
   C 清零仍需有效指针。
5. 三角外与 UNIT 对角不读取；不支持超出 lda/ldb/ldc 语义的非连续内存；不返回视图。
6. 允许 B≡C；除此之外 C 与 A/B 不允许内存重叠。
7. 异步执行：依赖 `aclblasSetStream`；Host 侧无隐式同步，读回 C 前须同步 stream。
8. 确定性：不要求（多核 tile 归约顺序不保证逐位一致，混合容差判定）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | COMPLEX64 混合容差：逐元素 `|actual−golden| ≤ atol + rtol×|golden|`，atol = rtol = 2⁻¹³（实/虚部分别按 FLOAT32 判定）；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP（大数规约误报时可放宽 2 ULP） | 任务书 §3.2 / opbase mixed_tolerance_standard |
| alpha=0 特判 | C 的 m×n 区域全零，位精确（EXACT） | 任务书 §2.5 / arch35 strmm 先例 |
| 性能标准 | 5 条典型 case（及 TC_PF 全 200 条关联比对）：NPU 平均单次 kernel 耗时 ≤ gpu_ms ÷ 0.8（同调用全部 kernel 之和），msprof 口径，10 次有效采样 | 任务书 §3.3 / 测试指导 |
| golden | Netlib BLAS `ctrmm` 单标杆生成，对 C（m×n）全矩阵验证 | 任务书 §3.1 |

## 测试策略

任务包提供 1200 条 csv 用例（`gen_csv.py` 固定种子 20260823，逐字节可复现）：
1000 精度（TC_L0 24 全枚举 / TC_SQ 23 尺寸 / TC_AB 24 alpha 特殊值 / TC_WS+TH 24
fat+thin 非方阵 / TC_LD 12 前导维 / TC_FL 13 填充含全零 A 与 Inf/NaN B /
TC_CV 24 / TC_ED 23 边界负向 / TC_EX 833 扩展）+ 200 性能（TC_PF，均纯连续访存）。
自补重点：

| 补充点 | 说明 |
|---|---|
| NaN 定向用例 | A 三角内 NaN、B NaN 的传播范围与 golden 逐元素对拍（验证「结构零×NaN 从不发生」）；A 带外预填 NaN 垃圾验证不读取 |
| B≡C 原地 | 路径 A/B 两阈值区间各覆盖（指针相等触发分支），结果与离席一致 |
| 路径分派边界 | TH 两侧同参数对拍（路径 A vs B 数值一致性，容差内） |
| UNIT 对角垃圾 | diag=UNIT 时 A 对角预填 Inf/NaN，结果不受影响（不读验证） |
| alpha=0 | EXACT 全零（含 ldc> m 间隙不被写、A/B 传 nullptr） |
| fat/thin 大偏态 | m=1×n=4096、m=4096×n=1 极端条带（tiling 退化路径） |
| guard 区 | C 缓冲列间与前后哨兵，检测 ld 间隙越界写 |

复现步骤（写入测试 README，验收人可复现）：

```bash
python gen_csv.py                                   # 生成 trmm_test.csv / gpu_baseline.csv
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --csv ./trmm_test.csv
msprof op --application="./build/test/trmm/ctrmm/ctrmm_test --gtest_filter=*TC_PF*" \
         --output=./prof_ctrmm                      # OpBasicInfo.csv 按 kernel 名平均
```

## 兼容性分析

- **arch22 为全新目录**，不触碰 arch35 strmm 任何行为。
- **新接口**：`include/cann_ops_blas.h` 新增 `aclblasCtrmm` 声明（与任务书 §2.3
  签名一致，供其他产品线共用）；`blas/trmm/README.md` 新增 aclblasCtrmm 章节，
  产品支持表标注 Atlas A2/A3 系列产品（含 Atlas 800I A2/A3）：支持；精度口径按
  任务书 COMPLEX64 档登记（严于 strmm 的 FLOAT32 档描述）。
- **与 950（arch35）任务的关系**：`09-56-aclblasCtrmm-950` 为 A5 侧任务；两者代码
  目录（arch22 vs arch35）与测试目录（test/trmm/ctrmm/arch22 vs arch35）互不重叠；
  若 950 侧也登记 `include/cann_ops_blas.h` 声明与 README 章节，**取齐到逐字相同
  以便自动合并**（公共文件仅此两处，冲突面可控）。
- **SOC 网关**：测试框架按 SOC 版本网关跳过不适型号（沿用仓内 `skipped_tests.list`
  机制），arch22 目标 SOC 为 `ascend910b*` 与 `ascend910_93*`。
- **版本兼容**：CANN 9.1.0 及后续配套。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | arch22 FP32 cube 实际峰值/利用率未知（仓内无 GEMM 先例） | case 4/5 性能 | 上板首日跑 4-GEMM 探针标定；3M 分解（−25% FLOPs）、tile 调优、双累加器三级候选；跳过表已把 FLOPs 减半 |
| 2 | 小 case 启动开销（路径 B 三段 kernel） | case 1 压线 | 路径 A 单 kernel 覆盖小规模；TH 阈值按 case 1/2 实测定版；阶段 1 两分支合 kernel |
| 3 | NaN 语义（0×NaN 扩散） | TC_FL 精度失败 | 只读三角 + K-tile 静态跳过从结构上排除；NaN 定向用例先行；物化通路零填充仅为冗余兜底 |
| 4 | 24 组枚举 × 跳过表/共轭/物化方向正确性 | 错一组即精度全红 | 归一化位 + Host 导出跳过表为唯一分派依据；TC_L0 24 组合先行；Host 单测直接对拍跳过表 |
| 5 | B≡C 原地与 ld 间隙写越界 | 用例间内存污染 | B≡C 强制安全路径；尾乘/清零严格列步长；guard 区用例 |
| 6 | workspace 峰值（case 5 ~384 MB） | 极端小内存环境 | HBM 余量充足且任务书内存不涉及；Ci/Xi 别名复用可降 1/3 |
| 7 | A3（910_93）环境晚到位 | 双系列验收进度 | 架构同源（arch22），先 910B3 全量自验，A3 到位后补跑 |
| 8 | 与 950 任务的 include/README 公共文件冲突 | 合入冲突 | 双方取齐逐字相同；提 PR 前 `git merge-tree` 预演 |

# 交付物

1. **算子设计文档**：本文件，PR 至 `cann-ops-competitions`
   `04_tasks/01_community-task-2026/tasklist/09-71-aclblasCtrmm-A2A3/<gitcode账号>/docs/design.md`。
2. **代码**（PR 至 `ops-blas` `master`，任务书 §5 结构）：`blas/trmm/arch22/`
   host/tiling/kernel（AIV + cube + 前后处理）；`include/cann_ops_blas.h` 新增
   `aclblasCtrmm` 声明；`blas/trmm/README.md` 新增章节与支持矩阵。
3. **自测用例与测试代码**：`test/trmm/ctrmm/arch22/`（csv 驱动 GTest，镜像仓内
   blas 测试结构）+ 任务包 1200 条 csv + 复现 README（环境/编译/步骤）。
4. **自测报告**：用例参数、精度对比结果及截图（1000 条精度全量 + 自补用例 +
   alpha=0 EXACT）、性能数据（5 条典型 case + TC_PF 200 条 msprof kernel 口径，
   10 次有效采样平均，路径 B 按三段 kernel 归因）及截图、内存说明（workspace
   峰值实测）、910B3 与 A3 双系列结果。
