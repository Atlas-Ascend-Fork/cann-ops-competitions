# 需求背景（required）

## 需求来源

社区任务 10-4「单精度复数QR分解与最小二乘(950)」，接口为 `aclsolverCgeqrf`、`aclsolverCungqr`、`aclsolverCunmqr` 等。目标仓库 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，平台 Ascend 950PR（arch35）。验收以任务书及随附测试包为准；本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿，是 10-3 实数 QR 设计的复数扩展，实数版已确立的 blocked Householder 框架、分核与 workspace 布局全部沿用，本文只展开复数差异项。

## 背景介绍

### 功能语义

| 接口 | 数学语义 | 说明 |
| --- | --- | --- |
| `aclsolverCgeqrf` | `A = Q·R`，complex64 | Householder QR：A 原地覆写为 R 与下方 Householder 向量 v(2)，`tau[i]`（**复数**，长度 min(m,n)） |
| `aclsolverCungqr` | 由 cgeqrf 产物生成 Q 的前 n 列 | 复数紧凑 WY：`Q = ∏(I − τ_j·v_j·v_jᴴ)` |
| `aclsolverCunmqr` | `C ← op(Q)·C`，op ∈ {N, C}（以任务书签名为准） | 隐式应用酉矩阵，`op=C` 时 `Qᴴ = conj(Q)ᵀ` |
| 最小二乘组合 | `min ‖Ax−b‖₂` | cgeqrf → cunmqr(C) → ctrsm 组合；任务书单列 `aclsolverCgels` 时按薄封装交付 |

### 仓内现状

无 QR 原型；复数基础设施已验证（`cgetrf` 的 GatherMask 交错↔planar 拆分回填、`src/utils/kernel/c64/` 复数构件、`getri_custom.hpp`）。实数版构件（larfg/larfb/T 构造/trsm）按 10-3 设计落地后，本任务以复数特化为主。实数版姊妹设计见 `10-3-aclsolverSgeqrf-leastsq-950/zhangfeng1133/docs/design.md`，分核框架、块宽与 workspace 布局的复用关系以该文档为准。

### 性能口径

门槛以任务书为准。复数 flop 当量（1 复乘 = 4 实乘 + 2 实加）为实数版 4 倍：cgeqrf `8mn² − 8/3·n³`；cungqr `16/3·mn² − 8/3·n³`；cunmqr `≈ 8mn·nrhs`。m=n=4096 时 cgeqrf ≈ 366 GFLOP 当量，走 3M 分解折合 3/4 实数 GEMM 量，Cube 预算内可行（0.35× 门槛下富余大）。

# 需求分析（required）

## 需求描述

交付 `aclsolverCgeqrf/Cungqr/Cunmqr`（+ bufferSize）：Host C API + AscendC Kernel 直调，COMPLEX64 列主序（对外交错、kernel 内 planar），`aclError` + `int64_t` + `int32_t info/tau 复数` 契约；handle stream、禁 CPU fallback、禁无必要 Host 同步、bit-wise 确定性、dynamic shape。规模下限同 10-3（`m,n ∈ [1,8192]`、`nrhs ∈ [1,128]`、padding 合法、内存 ≤ 4G）。

## 需求拆解

1. `cgeqrf`：复数 larfg（实 β 约定）+ planar 面板 + 尾随复数 GEMM（3M）。
2. `cungqr`：复数 WY 正向生成；`cunmqr`：op ∈ {N,C} 隐式应用（含 Vᴴ 与 conj 预处理）。
3. 复数构件入 `src/utils/kernel/c64/`：`clarfg`、`clacgv`（共轭）、复数 T 构造、planar 复数 Matmul 封装。
4. 最小二乘组合链（cgels 语义）端到端验收。
5. 交付 `src/<op>/ + test/<op>/ + docs/zh/<op>.md` 与任务书提交物。

# 详细设计（required）

## 算子分析

### 数学公式（复数差异项）

- **clarfg**：`β = −sign(Re(x_1))·‖x‖₂`（**β 取实数**、符号随实部，对齐 LAPACK zlarfg/clarfg 约定）；`v(2) = x(2:)/(x_1 − β)`；`τ = (β − x_1)/β` 为**复数**。`x` 为实非负向量时 `τ` 虚部为 0，`x_1=0` 时 `τ=0`。β 实数化保证 R 对角元实数、正交性误差不因相位放大。
- **复数块反射**：`A ← (I − V·Tᴴ·Vᴴ)·A`；递推 `T_j = τ_j·[T·(Vᴴv_j); 1]`，`T` 为上三角**复数**矩阵。
- **cungqr**：`Q = ∏(I − W·Yᴴ)`，块内 `Q ← Q − W·(Yᴴ·Q)`；`Yᴴ` 对 planar 面即"虚面取负后转置"（`clacgv`）。
- **cunmqr**：N 路径 `C ← (I − VTVᴴ)C`；C 路径逆序 + `Tᴴ` + `clacgv(V)`；planar 形态下共轭 = 虚面取负，为一次廉价 UB 操作。
- **复数 GEMM（Cube）**：planar 4M 与 3M 恒等式同 10-2；尾随更新用 **4M 为主**（精度稳健），仅对角块内小 GEMM 允许 3M（评审按任务书精度用例定案）。

### 支持数据类型

A/C/tau complex64；ipiv 不涉及；info INT32；维数 int64_t；`SetHF32(false)`；planar 往返仅重排、无舍入。

### 支持形状

同 10-3：m×n 列主序、C 为 m×nrhs（LEFT）/n×nrhs（RIGHT）、m<n 梯形覆盖。

## 算子实现

### 实现方案

新增 `src/cgeqrf/、src/cungqr/、src/cunmqr/`；入口一次性 planar 化（因子与 C），全程 planar，出口按需回填交错。分核框架、块宽 b、workspace 布局沿用 10-3（所有 staging 翻倍为实/虚两面）。

| 平台事实（arch35） | 复数应对 |
| --- | --- |
| ‖x‖₂ 规约对复数 = 实部²+虚部² 规约 | planar 两面同形状：双面平方和一次二叉规约，无额外 kernel |
| GatherMask 往返带宽 | 全程 planar 驻留，仅入口/出口各一次往返；小 n 单 kernel 路径 UB 内拆分零 GM 往返 |
| AIC 28 核 4M GEMM | 尾随 tile 分核；Vᴴ/Tᴴ 预处理（虚面取负 + 转置）在 Host staging 一次完成 |
| 零原子 | 分核区间不重叠，规约二叉树，同 10-3 |

#### 3.2.1 host 侧设计

**校验链 / tiling / 路径选择**：与 10-3 同构（S 单核小矩阵 / R 列切分 / L-blocked），差异仅：tau 为 complex64 输出（`std::complex<float>*`）；`op` 枚举 `CONJ_TRANS`；workspace 容量 ×2（两面）。

**Lwork**：`2×(h·b + b² + M0·N0) + 16`（两面 planar + 标志区，float 元素数口径按任务书裁定）。

#### 3.2.2 kernel 侧设计

| 步骤 | kernel | 核型 | 复数差异 |
| --- | --- | --- | --- |
| ① 面板 | cgeqrf_panel | AIV | 双面平方和规约 → clarfg（β 实数、τ 复数）→ 面板右邻列复数 GEMV 链更新（虚面同步更新） |
| ② T 构造 | cgeqrf_tbuild | AIV | `Vᴴv_j` = 实点积 − i·虚点积（两实数点积组合），T 复数上三角 |
| ③ 尾随 | cgeqrf_trailing | AIC | 4M：`VᴴA`（两面×2 次实 GEMM）→ 三角解（复数，同 10-2 trsm）→ `V·(·)` 减法更新 |
| ④ 回填 | cpack | AIV | planar → 交错（可与③按列合并） |

`op=C` 路径：Host 一次性生成 `Tᴴ`、`clacgv(V)` staging（虚面取负 + 索引转置），kernel 复用 N 路径。

### 精度、确定性与测试

- 精度：任务书全量 case + CPU LAPACK（zgeqrf 复数参考）双步判定（`‖QR−A‖/‖A‖`、`‖QᴴQ−I‖`、最小二乘残差）；β 实数约定与 LAPACK 一致性专项。
- 正交性/相位专项：复随机矩阵、纯虚/实特例、相位旋转不变性（`A→A·diag(e^{iθ})` 后 Q 相位一致）、零列、病态条件数。
- 确定性：同 10-3（固定规约树 + 等分前缀分核）。
- 性能：比值表 + planar 往返占比自检（≤15%）；4M/3M 占比按块记录。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| τ 复数相位约定与 golden 不一致（不同 LAPACK 实现历史上存在 ±i 相位分歧） | 开发初期与任务书 golden（CPU LAPACK reference）对拍冻结 β/τ 约定，写入头文件注释；不一致时以 golden 为准调整 clarfg |
| 4M 尾随带宽翻倍 | Vᴴ·A 阶段 K 维是块宽（小），M/N 维 tile 复用 V 驻留；带宽下限自检入测试报告 |
| op=C 转置索引错误 | clacgv/Tᴴ 生成做独立单测（小矩阵全枚举 op×side×trans 对照 CPU） |
| 实数版构件缺陷传导 | 10-3 先行评审合入；本任务仅叠加复数层，缺陷修复回灌实数版用同一 PR 链 |
