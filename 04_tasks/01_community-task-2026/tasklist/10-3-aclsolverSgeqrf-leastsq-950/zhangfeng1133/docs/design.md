# 需求背景（required）

## 需求来源

社区任务 10-3「单精度实数QR分解与最小二乘(950)」，接口为 `aclsolverSgeqrf`、`aclsolverSorgqr`、`aclsolverSormqr` 等。目标仓库 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，平台 Ascend 950PR（arch35）。验收以任务书及随附测试包为准；本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿。

## 背景介绍

### 功能语义（对齐 LAPACK/cuSolver）

| 接口 | 数学语义 | 说明 |
| --- | --- | --- |
| `aclsolverSgeqrf` | `A = Q·R`（m≥n 约化；m<n 时 `A = Q·R` 且 R 为 m×n 上梯形） | Householder QR：A 原地覆写为 R 与下方 Householder 向量 v(2)，`tau[i]`（长度 min(m,n)）记录尺度因子 |
| `aclsolverSorgqr` | 由 geqrf 产物显式生成 Q 的前 n 列 | `Q = H_1·H_2·…·H_n`，紧凑 WY（`Q = I − W·Yᵀ`）分块正向生成 |
| `aclsolverSormqr` | `C ← op(Q)·C`，op ∈ {N, T}（side/trans 以任务书签名为准） | 隐式应用 Q：对每个 Householder 因子按序做 `H_i·C` 或逆序 `H_iᵀ·C`，分块时用 `W` 累积 |
| 最小二乘（gels 组合语义） | `min ‖Ax−b‖₂` | `geqrf → ormqr(T)·b → trsm(R·x = b̂)` 三步组合；若任务书单列 `aclsolverSgels`，按薄封装复用三构件实现 |

### 算法选型

选 **blocked Householder（LAPACK geqrf 路线）**，弃 MGS/Cholesky QR：

- MGS 数值稳定性可接受但每列内点积串行、无法走 Cube GEMM，性能不达标；
- Cholesky QR 精度不满足任务口径（条件数平方放大）；
- blocked Householder 面板向量阶段（小计算、串行）+ 尾随更新（大 GEMM、Cube）与 950 AIV/AIC 分核天然匹配，flop 达到算力对抗段要求。

### 仓内现状

ops-solver 无任何 QR 原型（`src/` 仅 LU/Cholesky 设计稿、特征值 cheevj、求逆族），三接口全新实现；可复用公共构件：handle/stream、`SOLVER_ECHECK` 校验宏、`lu_host_common.h` 的分核等分与余数前缀调度、批量 staging 模式、Matmul 高阶 API 调用范式（Spotrf 设计稿已验证 AIC/AIV 分 kernel 同 stream 下发可行）。

### 性能口径

门槛以任务书为准（历史 aclsolver 口径 NPU ≤ 0.35× GPU 950 实测，msprof kernel 口径）。标准 flop（m≥n）：

| 接口 | 计算量 | m=n=4096 参考 | 策略判读 |
| --- | --- | --- | --- |
| sgeqrf | `2mn² − 2/3·n³` | ≈ 91.6 GFLOP | 算力对抗段：尾随更新占 ≈80%，必须走 Cube；面板阶段压 kernel 数 |
| sorgqr | `4/3·mn² − 2/3·n³` | ≈ 91.6 GFLOP | 同上，GEMM 为主 |
| sormqr | `≈ 2mn·nrhs`（经济 Q） | nrhs=32：≈ 4.3 GFLOP | 依赖链长、算力占比低：靠减少阶段数与列并行 |
| gels 组合 | 上述之和 + `2n²·nrhs` trsm | — | 三接口复用，不单独优化 |

# 需求分析（required）

## 需求描述

交付三个新接口（+各 bufferSize，若任务书签名含 workspace）：Host C API + AscendC Kernel 直调，FLOAT32 列主序，`aclError` + `int64_t` 维数 + `int32_t info` 契约（与 `include/cann_ops_solver.h` 现有风格一致）；计算走 handle stream，禁止 CPU fallback 与无必要 Host 同步；相同输入同 stream 串行执行输出 bit-wise 一致；dynamic shape 按次 tiling。

规模下限（以任务书为准）：`m ∈ [1, 8192]`、`n ∈ [1, 8192]`、`min(m,n) ≤ 8192`、`nrhs ∈ [1, 128]`、`lda ≥ max(1,m)` padding 合法、输入内存 ≤ 4G；m≥n 与 m<n 梯形形态均覆盖。

## 需求拆解

1. `aclsolverSgeqrf`：blocked Householder，面板 AIV + 尾随更新 AIC，tau 与 v(2) 落盘布局对齐 LAPACK。
2. `aclsolverSorgqr`：紧凑 WY 分块正向生成 Q（自块 k 向前），复用 geqrf 面板构件与 Cube GEMM。
3. `aclsolverSormqr`：op(Q)·C 隐式应用，N/T、LEFT/RIGHT 全枚举（以任务书签名裁剪）。
4. 公共构件：`larfg`（Householder 生成）、`larfb`（块反射应用）、trsm（复用 10-1 sgetrs 三角求解构件）入 `src/utils/`。
5. 最小二乘组合链验收（gels 语义）：geqrf→ormqr→trsm 端到端残差用例。
6. 交付 `src/<op>/ + test/<op>/ + docs/zh/<op>.md` 与任务书提交物，完成评审验收。

# 详细设计（required）

## 算子分析

### 数学公式

- Householder 因子：`H_i = I − τ_i·v_i·v_iᵀ`，`v_i = [1; v(2)]`；`larfg` 对列 `x`：`β = −sign(x_1)·‖x‖₂`、`τ = (β−x_1)/β`、`v(2) = x(2:)/(x_1−β)`；`x_1=0` 时 `τ=0`（该列已归约，跳过反射——零列/线性相关输入的信息语义入口）。
- geqrf 第 k 块：面板 `A[k:b, k]` 逐列 larfg 得 τ、v（A 原地存 v(2)）；尾随更新 `A[k:b, b:] ← (I − V·Tᵀ·Vᵀ)·A[k:b, b:]`，T 为上三角块（紧凑 WY：`T_j = τ_j·[T·(Vᵀv_j); 1]` 递推）。
- orgqr：`Q = ∏(I − W_j·Y_jᵀ)`，自最后一个块向前：先小块直接生成，再 `Q ← Q − W·(Yᵀ·Q)`（GEMM 主导）。
- ormqr（LEFT/N）：`C ← H_1·…·H_n·C`；T 路径逆序且因子转置共轭（实数即转置）；RIGHT 侧镜像。块化时 `C ← C − V·(Tᵀ·(Vᵀ·C))`，两次 GEMM 一次三角解。

### 支持数据类型

矩阵/tau/C 均 FLOAT32；info INT32；维数 int64_t；workspace 以 float 元素个数计。Cube fp32 `SetHF32(false)`。

### 支持形状

A 为 m×n 列主序（m<n 时 R 为上梯形、Q 为 m×m 语义由任务书裁定，实现按 LAPACK 全序列约定）；C 为 m×nrhs（LEFT）或 n×nrhs（RIGHT）。

## 算子实现

### 实现方案

新增 `src/sgeqrf/、src/sorgqr/、src/sormqr/` 各 `*_host.cpp + *_kernel.cpp + *_tiling.h`；`larfg/larfb/T 矩阵构造` 收敛 `src/utils/kernel/`，严控文件数。

平台事实约束（arch35）与应对：

| 平台事实 | 设计应对 |
| --- | --- |
| 64 AIV + 28 AIC，MIX 不可用 | 面板 larfg/向量内积 AIV；尾随更新/orgqr/ormqr 的 GEMM AIC；分 kernel 同 stream 下发，GM 标志位同步 |
| ‖x‖₂ 规约的串行敏感 | 面板列内二叉规约（UB 内分片 + 单核归并），单列规约 ≤2 kernel；多列间流水 |
| Cube fp32 需关 HF32 | 尾随更新 Matmul `SetHF32(false)` |
| launch ≈5.5µs | 块宽 b 按预算取大（128/256），n=4096 时 stage 数 ≈ n/b，下发开销 ≤ 预算 5% |
| UB ≈192KB/core | 面板高 h=min(m−k,b)、V/T 驻留按 `h·b` 预算；C 列窗流式 |
| 零原子 | 面板列、尾随列区间、C 行区间分核不重叠；规约走两叉树无原子 |

#### 3.2.1 host 侧设计

**校验链**：handle → `m,n ≥1`（或任务书允许 0 快返）、`lda ≥ max(1,m)`、A/tau/C 指针 → 维数上限。ormqr 校验 side/trans 合法、C 维数与 side 匹配。

**tiling**：`{m, n, nrhs, lda, ldc, op, b, h, coreAiv[], coreAic[], path}`。路径：

| 算子 | 路径 | 条件 |
| --- | --- | --- |
| sgeqrf | S-单核小矩阵 | m·n ≤ 64KB 预算（全矩阵驻 UB，纯 AIV，一步完成） |
| sgeqrf | L-blocked | 否则：b ∈ {128, 256}，尾块 SetTail；面板 1×AIV（列间流水）+ 尾随 28×AIC Matmul |
| sorgqr | S-直接生成 | n ≤ b（单块，AIV 直接正交化生成） |
| sorgqr | L-blocked WY | 自后向前块循环：`Yᵀ·Q` GEMM（AIC）→ `Q −= W·(·)`（AIC） |
| sormqr | C 列切分 | active = min(nrhs, 28)（AIC 各核领 C 列块，V/T 只读共享）；nrhs=1 退化为单核 GEMV 链（AIV） |

**Lwork 契约**（float 元素数，32B 对齐；sgeqrf 为例）：

```
[0, h·b)          面板 V staging（兼容任意 lda 的紧致化）
[h·b, h·b + b²)   T 矩阵（上三角紧凑）
[…, +M0·N0)       尾随 GEMM 输出 tile staging（若 Matmul beta 通路不可用）
[尾部, +16)        int32 标志区（早退/同步计数）
```

sorgqr/sormqr 复用同构布局；无 workspace 参数签名时改用 UB 列窗 + 因子原地面，Lwork 返回 0（以任务书签名为准，评审冻结）。

#### 3.2.2 kernel 侧设计

**sgeqrf（L 路径，每块 k 三步）**：

| 步骤 | kernel | 核型 | 职责 |
| --- | --- | --- | --- |
| ① 面板 | sgeqrf_panel | AIV×1（列流水） | 逐列：列向量入 UB → 二叉规约 ‖x‖₂ → larfg 得 β/τ → 回写 A[k:h,k] 与 tau[k]；即时更新面板右邻列（小 GEMV 链） |
| ② T 构造 | sgeqrf_tbuild | AIV | 由 V、τ 递推构造上三角 T（b×b，单核） |
| ③ 尾随 | sgeqrf_trailing | AIC（Matmul） | `A ← A − V·(Tᵀ·(Vᵀ·A))`：两次 GEMM + 一次三角解，行列 tile 分核 28 AIC |

**sorgqr（L 路径）**：自最后一块向前复用③同构 GEMM；首块由单位向量列直接构造；对角块 `Q` 小矩阵生成复用①的规约构件。

**sormqr**：V/T 只读，C 列块并行；`op=T` 时逆序块循环并预转置 T（Host 一次性生成 Tᵀ staging，避免 kernel 内转置）；`side=RIGHT` 时 C 行切分镜像。

**info 语义**：geqrf 正常恒 0（QR 对任意矩阵有定义，无奇异上报）；参数错误走 aclError 负值语义；orgqr/sormqr 同。不引入 LAPACK 之外的正值语义。

### 精度、确定性与测试

- 精度：任务书全量 case + 本地 CPU LAPACK 对拍双步判定（`‖QR−A‖/‖A‖`、`‖QᵀQ−I‖`、`Q·x=b` 残差）；重点验证 β 符号约定与 LAPACK 一致（`−sign(x_1)·‖x‖`），避免 `QᵀQ−I` 偏差。
- 正交性专项：随机矩阵、病态矩阵（条件数 1e8）、零列、重复列、m≫n / m<n 梯形。
- 确定性：规约二叉树固定顺序 + 分核等分余数前缀，串行 3 次逐位一致。
- 性能：任务书全部 case 比值表；面板/尾随耗时占比自检（尾随应 ≥70% 时间在 Cube）。
- gels 端到端：geqrf→ormqr(T)→trsm 组合链残差用例，覆盖超定/欠定。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 任务书 ormqr 签名枚举范围未知（是否含 RIGHT/side） | 实现按 LEFT/RIGHT × N/T 全枚举，任务书只测子集则全枚举不构成风险；签名差异评审冻结 |
| 面板阶段成为短板（串行规约） | 列间流水（前一列尾随更新与后一列规约重叠）；必要时面板列间并行 + 交叉规约 |
| 小矩阵 kernel 拆分开销反超收益 | S 路径单 kernel 全 UB 完成，阈值 spike 实测校准（预期 [64,128]） |
| lda≠n 紧致化额外搬运 | 仅面板阶段 staging，尾随 Matmul 直读原 stride（高阶 API 支持 stride） |
| T 矩阵构造错误难定位 | 单测先覆盖 T 递推与 `I−VTVᵀ` 正交性，再上端到端 |
