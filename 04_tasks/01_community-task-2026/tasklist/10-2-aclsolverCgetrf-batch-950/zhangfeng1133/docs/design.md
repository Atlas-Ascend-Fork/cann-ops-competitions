# 需求背景（required）

## 需求来源

社区任务 10-2「单精度复数LU分解、求解和批量求逆(950)」，接口为 `aclsolverCgetrf`、`aclsolverCgetrs`、`aclsolverCgetri` 等。目标仓库 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，平台 Ascend 950PR（arch35）。验收以任务书（`Atlas950_Cgetrf_Getrs_Getri_task_doc.md` 及随附测试包）为准；本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿。

## 背景介绍

### 功能语义

| 接口 | 数学语义 | 说明 |
| --- | --- | --- |
| `aclsolverCgetrf` | `A = P·L·U`（部分主元，主元取模最大） | complex64；L 单位下三角、U 上三角，1-based `ipiv`；原地覆写 A |
| `aclsolverCgetrs` | 求解 `A·X = B` | 输入 LU 因子 + ipiv：laswp → 复数前代 `L·Y = P·B` → 复数回代 `U·X = Y`；B 原地覆写；共轭转置解法（`Aᴴ·X=B`）以任务书签名为准（cuSolver legacy `Cgetrs` 无 trans 参数，如任务书含 trans 则按扩展枚举实现） |
| `aclsolverCgetri` | 求 `A⁻¹` | trtri（复数 `U⁻¹`、`L⁻¹`）→ 复数乘积合并 → 按 ipiv 逆序行交换 |
| `aclsolverCgetriBatched` | 逐矩阵批量求逆 | 仓内已存在（`src/cgetri_batched/`，n≥32 LU 流程；`cmatinv_batched` 覆盖 n<32） |

### 仓内现状分析（工作量边界）

| 子项 | 现状 | 本任务动作 |
| --- | --- | --- |
| cgetrf | 已实现（`src/cgetrf/`，planar 拆分 + 双 kernel） | 任务书全量回归验收；缺口最小修复 |
| cgetri | 已实现（`src/cgetri/`） | 同上；与 cgetrf 联动验收 |
| **cgetrs** | **缺失** | 全新实现（本设计核心） |
| cgetriBatched / cmatinvBatched | 已有 | 回归验收；不重写 |

复数实现基础（仓内已验证模式，直接复用）：complex64 交错存储 ↔ 实部/虚部 planar 面之间用 `GatherMask` 拆分与回填（`cgetrf_kernel.cpp` 已有实现）；Cube 侧复数 GEMM 分解为实数 GEMM 组合；向量核上复数运算按 `(a_r·b_r − a_i·b_i, a_r·b_i + a_i·b_r)` 展开。

### 性能口径

门槛以任务书为准（历史 aclsolver 口径 NPU ≤ 0.35× GPU 950 实测）。计算量（复数按 4 倍实数乘加当量粗估，供策略分段）：

| 接口 | 计算量 | 主导资源 |
| --- | --- | --- |
| cgetrs | `8n²·nrhs` flops 当量（两趟复数三角求解） | 串行依赖 + planar 拆分带宽 ×2 |
| cgetri | `≈ 8/3·n³` flops 当量（实数版两倍） | Cube 算力 + planar staging 带宽 |
| cgetriBatched | 数据量主导 | GPU 批量弱项段，带宽下限自检 |

# 需求分析（required）

## 需求描述

交付 `aclsolverCgetrs` 新接口，并使 `aclsolverCgetrf/Cgetri/CgetriBatched` 通过任务书全量验收：Host C API + AscendC Kernel 直调，FLOAT/COMPLEX64 列主序，`aclError` + `int64_t` 维数 + `int32_t ipiv/info` 契约（与 `include/cann_ops_solver.h` 一致，不双轨）；计算走 handle stream，禁止 CPU fallback 与无必要 Host 同步；确定性 bit-wise 一致；dynamic shape 按次 tiling。

规模下限：`n ∈ [1, 8192]`、`nrhs ∈ [1, 128]`、`ldb ≥ max(1,n)`、输入内存 ≤ 4G（任务书另有约定以任务书为准）。

## 需求拆解

1. 新增 `aclsolverCgetrs`：planar 复数 laswp + 复数三角求解两趟，nrhs 列并行。
2. cgetrf/cgetri 回归验收（长方形 m≠n、padding、非默认 stream、奇异 info）。
3. 复数三角求解构件（trsm_lower/upper，planar 形态）收敛入 `src/utils/kernel/c64/`，与 `getri_custom.hpp` 互补，供 cgetrs/cgetri 复用。
4. 交付 `src/cgetrs/ + test/cgetrs/ + docs/zh/cgetrs.md` 及任务书提交物，完成评审验收。

# 详细设计（required）

## 算子分析

### 数学公式

- 复数前代（单位下三角）：`y_i = b_i − Σ_{j<i} l_ij·y_j`，`l_ij = (l_r, l_i)`，乘加按复数展开；
- 复数回代（非单位上三角）：`x_i = (y_i − Σ_{j>i} u_ij·x_j)/u_ii`，对角除法为复数除法 `z/w = (z·conj(w))/|w|²`，在向量核上展开为 6 次实数乘加，避免 NPU 复数除法指令缺失问题；
- getri 三相：复数 trtri → 复数乘积 `L⁻¹·U⁻¹` → laswp；复数乘积走 Cube 时分解为 3 次实数 GEMM（GEMM3M 恒等式：`(L_r+iL_i)(U_r+iU_i)` 只需 `L_r·U_r`、`L_i·U_i`、`(L_r+L_i)(U_r−U_i)` 三次，较 4M 省 25% Cube 周期）。

### 支持数据类型

A/B complex64（planar 计算形态），ipiv/info INT32，维数 int64_t。Cube fp32 GEMM `SetHF32(false)`。

### 支持形状

n×n 方阵、B 为 ldb×nrhs、列主序；对外交错 complex64 布局，kernel 内 planar。

## 算子实现

### 实现方案

新增 `src/cgetrs/`（`cgetrs_host.cpp + cgetrs_kernel.cpp + cgetrs_tiling.h`）。入口做一次全量 planar 拆分（B 与因子读入时），计算全程 planar，出口回填交错——相对逐步交错读写，GM 往返次数减少一半以上。

平台事实约束与应对（arch35，沿用仓内结论）：

| 平台事实 | 设计应对 |
| --- | --- |
| GatherMask 拆分/回填为带宽热点 | 拆分一次驻 planar staging（workspace 或原地因子面），三角求解按列窗流式 |
| 64 AIV + 28 AIC 分核 | 前代/回代小 n 全 AIV；大 n 的列窗更新 GEMM 走 AIC（实数面 Matmul） |
| launch ≈5.5µs | 小 n 拆分+求解 ≤3 kernel；大 n 按块数下发 |
| 零原子 | RHS 列区间、行交换区间分核不重叠 |

#### 3.2.1 host 侧设计

**校验链**：handle → `n/nrhs/ldb/lda ≥1` 且 `ldb≥max(1,n)`、`A/ipiv/B` 非空（n>0 时）→ 维数上限（对齐 cgetrf 现有契约）。任务书若要求 trans/side 扩展枚举，非法值在 Host 报 `ACL_ERROR_INVALID_PARAM`。

**tiling**：`{n, nrhs, ldb, path, b, coreRanges[]}`；路径选择：

| 路径 | 条件 |
| --- | --- |
| S-单 kernel 向量求解 | n ≤ 256 且 planar 两面驻 UB 预算内 |
| R-RHS 列切分 | active = min(nrhs, 64)，每核 ⌈nrhs/active⌉ 列 |
| L-大 n 分块求解 | n > 256：对角块 AIV 求解 + 块间 planar GEMM（AIC，3M 分解） |

**workspace**（若任务书签名含 bufferSize）：`Lwork ≈ 2·(n·b + b²) + 16`（planar 面板 staging + 标志区）；无 workspace 签名则复用因子原地面 + UB 列窗流式，恒 0。

#### 3.2.2 kernel 侧设计

| 步骤 | kernel | 核型 | 职责 |
| --- | --- | --- | --- |
| ① planar 化 + laswp | cgetrs_prep | AIV | B 交错→实/虚两面；按 ipiv 行交换（两面同步交换，索引一致） |
| ② 复数前代 | cgetrs_trsm_l | AIV/AIC | 单位下三角 planar 求解；列扫 + 已解列窗 4 实数 GEMM 组合更新（或 3M） |
| ③ 复数回代 | cgetrs_trsm_u | 同上 | 逆序列扫；对角复数除法按 `|u_ii|²` 预除（每列一次，向量化） |
| ④ 回填 | cgetrs_pack | AIV | planar → 交错写入 B（可与③按列流水合并） |

小 n 路径将①~④合并为单 kernel（拆分、求解、回填全在 UB 内），消除 staging GM 往返。

### 精度、确定性与测试

- 精度：任务书全量 case；本地对拍 CPU LAPACK（复数残差 `‖AX−B‖₂/（‖A‖₂‖X‖₂）` 双步判定）；planar 往返不引入额外舍入（仅重排，无数值变换）。
- info 语义：0/负参数；cgetrf 奇异正值回归；cgetrs 无奇异信息。
- 边界：n=1、nrhs=1、padding、实数解特例（虚部为 0 输入）、NaN/Inf 传播、B 与因子部分别名拒绝。
- 确定性：同输入同 stream 串行 3 次逐位一致（分核等分 + 余数前缀）。
- 性能：msprof kernel 口径对任务书 case 出比值表；planar 拆分开销单列（验证 ≤ 总预算 15%）。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 任务书 cgetrs 签名含 trans（≠cuSolver legacy） | 设计预留 trans 枚举位：conj 路径仅需因子面共轭化（虚部取负一次）复用同一求解核；评审时冻结 |
| planar 往返带宽抵消融合收益 | staging 驻留 + 列窗流式；若任务书允许原地因子 planar 化，则只拆 B（因子仅读一遍） |
| 复数除法精度（|w|² 溢出） | 对角元先按幅值缩放（mantissa/exponent 分解）再平方，极端用例单测覆盖 |
| 3M 恒等式中间量量级差导致精度损失 | 大 n 分块路径用 4M（两面对分别 GEMM），仅在小块内用 3M；以任务书精度用例为准择优 |
