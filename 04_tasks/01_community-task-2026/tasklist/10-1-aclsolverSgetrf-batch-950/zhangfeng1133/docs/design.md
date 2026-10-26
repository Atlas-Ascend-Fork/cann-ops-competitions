# 需求背景（required）

## 需求来源

社区任务 10-1「单精度实数LU分解、求解和批量求逆(950)」，接口为 `aclsolverSgetrf`、`aclsolverSgetrs`、`aclsolverSgetri` 等（批量求逆对应 `aclsolverSgetriBatched`）。目标仓库 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，平台 Ascend 950PR（arch35）。验收以任务书（`Atlas950_Sgetrf_Getrs_Getri_task_doc.md` 及随附测试包）为准，本设计按[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)组织，为设计评审稿，不将设计目标表述为已完成的实测结论。

## 背景介绍

### 功能语义（对齐 LAPACK/cuSolver）

| 接口 | 数学语义 | 说明 |
| --- | --- | --- |
| `aclsolverSgetrf` | `A = P·L·U`，部分主元 LU 分解 | L 单位下三角、U 上三角，行交换记录于 1-based `ipiv`；原地覆写 A |
| `aclsolverSgetrs` | 求解 `A·X = B` | 输入 getrf 产出的 LU 因子与 ipiv：按 ipiv 施行行交换（laswp）→ 前代 `L·Y = P·B` → 回代 `U·X = Y`；B 原地覆写为解 |
| `aclsolverSgetri` | 求 `A⁻¹` | 输入 LU 因子与 ipiv：trtri（先求 `U⁻¹`，再求 `L⁻¹`）→ 合并 `L⁻¹·U⁻¹` → 按 ipiv 行交换 |
| `aclsolverSgetriBatched` | 逐矩阵求逆 | Device 指针数组（或连续批量布局，以任务书为准），逐矩阵执行 getrf+getri 流程，`infoArray[i]` 逐矩阵上报奇异 |

奇异/病态语义：getrf 主元为 0（或低于主元哨兵阈值）时 `info = i`（第 i 个主元恰为 0，U 第 i 个对角元为 0），因子仍保留前 i-1 列结果；getrs/getri 对奇异因子不保证数值结果，info 仅报参数级错误。

### 仓内现状分析（决定本任务工作量边界）

ops-solver master 已有：`aclsolverSgetrf`、`aclsolverSgetri`、复数族 `cgetrf/cgetri/cgetri_batched/cmatinv_batched`、公共 handle/stream 管理（`aclsolverCreate/Destroy/SetStream/GetStream`）。因此：

| 子项 | 现状 | 本任务动作 |
| --- | --- | --- |
| sgetrf | 已实现（`src/sgetrf/`，双 kernel 协同，20 block 硬编码） | 按任务书测试包全量回归验收；仅做达标必需的修复，不重写 |
| sgetri | 已实现（`src/sgetri/`） | 同上；另与 sgetrf 联动验证 `lda==n` 契约 |
| **sgetrs** | **缺失** | 全新实现（本设计核心） |
| **sgetriBatched（实数批量求逆）** | **缺失**（复数版已有，实数版无） | 全新实现（本设计核心） |
| 复数族 | 已有 | 不在本任务范围，仅作为构件参考 |

现有接口契约（`include/cann_ops_solver.h`）：`aclError` 返回 + `int64_t` 维数 + FLOAT32 列主序 + `int32_t ipiv/info`（1-based）+ 当前版本要求 `lda == n`、`m,n ∈ [1,8192]`、`m·n ≤ INT32_MAX`；info 遵循 LAPACK 语义、允许传 nullptr（issue #164/#180）。新增接口必须沿用同一契约，不引入 `aclsolverStatus_t`/`int` 维数双轨。

### 性能标杆口径

任务书给出 GPU 基线与 NPU 门槛（历史 aclsolver 任务口径为 NPU 耗时 ≤ 0.35× GPU 950 实测，msprof kernel 口径；以本次任务书数值为准）。理论计算量与带宽下限（用于反向核对门槛合理性、划分实现策略段）：

| 接口 | 计算量 | n=4096, nrhs=32 参考 | 主导资源 |
| --- | --- | --- | --- |
| sgetrs | `2n²·nrhs`（两趟三角求解） | ≈ 4.3 GFLOP；读 A 因子 64MB + B 2MB | 三角求解串行依赖 + 带宽 |
| sgetri | `≈ 4/3·n³` | ≈ 91.6 GFLOP | Cube 算力（0.35× 门槛下 Cube 占用 <20%，可行） |
| sgetriBatched | 逐矩阵 `4/3·n³` | 数据量主导：读写 ≈ 2·batch·n²·4B | 带宽下限 `2·batch·n²·4B/BW`；GPU 批量弱项是主要得分点 |

# 需求分析（required）

## 需求描述

在 ops-solver 上交付 `aclsolverSgetrs`、`aclsolverSgetriBatched` 两个新接口，并使 `aclsolverSgetrf/Sgetri` 通过任务书全量验收：Host C API + AscendC Kernel 直调（非 aclnn 两段式），计算走 handle 绑定 stream，禁止 CPU fallback 与无必要 Host 同步；相同输入相同 stream 串行执行输出（含 info）bit-wise 一致；dynamic shape 按次 tiling。

| 接口 | 输入 | 输出 | info 语义 |
| --- | --- | --- | --- |
| `aclsolverSgetrs_bufferSize`（如任务书要求 workspace 版） | n、nrhs、ldb | Host Lwork | float 元素个数 |
| `aclsolverSgetrs` | LU 因子 A（只读语义、可原位）、ipiv、B | B 原地覆写为 X | 仅 0 / 负参数（无奇异信息） |
| `aclsolverSgetriBatched` | Device Aarray（或连续布局）、lda、batchSize | 各矩阵原地逆矩阵 | `infoArray[i]`：0 / k（第 i 个矩阵第 k 主元为 0，该矩阵冻结、其余照常） |

规模与泛化（验收下限，以任务书为准）：`n ∈ [1, 8192]`（与仓内上限一致）、`nrhs ∈ [1, 128]`、`batchSize ∈ [1, 3000]`（对齐 cgetriBatched 现有范围；任务书若更大则按任务书扩展）；`ldb ≥ max(1,n)` 的 padding 合法；输入内存 ≤ 4G。

## 需求拆解

1. 新增 `aclsolverSgetrs`（+ bufferSize，若任务书签名含 workspace）：laswp + 双趟三角求解，支持 nrhs 维并行与原位 B。
2. 新增 `aclsolverSgetriBatched`：矩阵级分核、逐矩阵 info 隔离、小 n UB 路径与大 n Cube 路径自适应。
3. sgetrf/sgetri 按任务书测试包全量回归（精度 + info + 性能），补齐缺口用例（m≠n 长方形、padding、非默认 stream）。
4. 公共构件收敛：laswp 行交换、trsm 三角求解、批量分核调度入 `src/utils/`（lu_host_common.h 已有基础），供 sgetrs/batched 与后续复数 getrs 复用。
5. 交付 `src/<op>/ + test/<op>/ + docs/zh/<op>.md` 与任务书要求的提交物，完成社区评审与验收。

# 详细设计（required）

## 算子分析

### 数学公式

- getrs（LOWER 因子）：`B ← P(B)`（按 ipiv 从左到右行交换）→ `L·Y = B` 前代（单位下三角）→ `U·X = Y` 回代（非单位上三角）。UPPER 存储（U·L 形式）时镜像执行；本仓 getrf 产出标准 `P·A = L·U` 布局，按前式实现。
- getri 分解为三步（对齐 LAPACK dgetri）：
  1. trtri：`U⁻¹`（非单位上三角求逆，自对角起逐列）与 `L⁻¹`（单位下三角求逆，逐列右视）；
  2. 乘积：`invA ← L⁻¹·U⁻¹`（利用三角结构分块：上三角块直接由 `U⁻¹` 组合、下三角块走 GEMM）；
  3. laswp：按 ipiv 逆序行交换。
- getriBatched：对 i = 0..batchSize-1 独立执行 getrf（含主元检测）→ getri 流水。

### 支持数据类型

矩阵/右端项 FLOAT32；`ipiv/info/infoArray` INT32；维数 int64_t；Lwork 以 float 元素个数计。HF32 一律关闭（`SetHF32(false)`），Cube fp32 GEMM 全精度。

### 支持形状

n×n 方阵（getrs 的 B 为 ldb×nrhs），列主序，元素地址 `base + row + col·ld`。批量接口按任务书布局（Device 指针数组 `Aarray[i]` 或 `[batch,n,n]` 连续，实现层以 `Aarray` 指针数组优先，连续布局退化为 stride 计算，两者统一入 kernel 的 per-matrix base 寻址）。

## 算子实现

### 实现方案

总体架构：新增 `src/sgetrs/`、`src/sgetri_batched/` 各含 `*_host.cpp + *_kernel.cpp + *_tiling.h`；laswp、trsm、批量分核等跨算子构件放 `src/utils/`（Host 头 + kernel hpp），严控新增文件数。

平台事实约束（950PR / arch35，沿用仓内已验证结论）与应对：

| 平台事实 | 设计应对 |
| --- | --- |
| 64 AIV + 28 AIC，UB 物理隔离，MIX 不可用 | 向量阶段（laswp/小 n 三角求解）AIV kernel；GEMM/大 n 回代 AIC（Matmul 高阶 API）分 kernel 同 stream 下发 |
| launch ≈5.5µs/次 | getrs 全程压到 ≤3 kernel；小 n 单 kernel |
| UB ≈192KB/core | 单核 U 列窗高度按 `192KB/4B/核内列数` 预算 |
| atomic ≈200ns 串行化 | 全链路零原子：行交换区间、RHS 列区间分核不重叠 |
| TBuf 数组 N>1 Get() 重叠 | 单 TBuf 偏移切分 |

#### 3.2.1 host 侧设计

**校验链**（顺序短路；沿用 `SOLVER_ECHECK` 风格，aclError 返回，info 允许 nullptr）：

1. handle 空指针 / stream 获取失败；
2. sgetrs：`n<1 || nrhs<1 || ldb<max(1,n) || lda<max(1,n) || A/ipiv/B 空指针`（n>0 时）；sgetriBatched：`n<1 || batchSize<1 || Aarray/invA 指针非法`；
3. 维数超限（对齐仓内 INT32 与 8192 契约，任务书若放宽以任务书为准）。

**tiling 策略**：`{n, nrhs, batch, ldb, path, b, coreRanges[]}` 经 GM 传入（Host/Kernel 共用头文件），核数运行时查询。路径选择：

| 算子 | 路径 | 条件 |
| --- | --- | --- |
| sgetrs | S-单 kernel 向量前代/回代 | n ≤ 256 且 nrhs 小（U 列窗驻 UB） |
| sgetrs | R-RHS 列切分 | active = min(nrhs, 64)，每核 ⌈nrhs/active⌉ 列 |
| sgetrs | L-大 n 分块 | n > 256：分块回代，对角块 AIV、块间 GEMM AIC |
| sgetriBatched | BV 批量向量化 | n ≤ 64（整矩阵驻 UB，逐矩阵流水） |
| sgetriBatched | BL 批量分块 | 64 < n ≤ 512（矩阵级分核 + 块化 trtri/gemm） |
| sgetriBatched | BC 逐矩阵 Cube | n > 512（复用单矩阵 sgetri kernel，矩阵级并行在外层） |

**Lwork 契约**（sgetrs 若含 workspace）：`Lwork = n·b + 16`（面板 staging + int32 标志区）；纯向量路径返回 0。batched 无 Workspace 参数（对齐 cgetriBatched）。

#### 3.2.2 kernel 侧设计

**sgetrs**：

| 步骤 | kernel | 核型 | 职责 |
| --- | --- | --- | --- |
| ① laswp | sgetrs_laswp | AIV | B 按 ipiv 行交换；行区间连续分核，交换对在 UB 内完成，一趟 GM 读写 |
| ② 前代 | sgetrs_trsm_lower | AIV/AIC 按路径 | 单位下三角 `L·Y = B`：列扫 + 已解列窗 GEMM 更新（`Y_j = B_j − L[:, :j]·Y[:j, j]`） |
| ③ 回代 | sgetrs_trsm_upper | 同上 | 非单位上三角 `U·X = Y`：逆序列扫，对角除法后列窗更新 |

原位 B：三步均原地覆写，无全量中间张量；②③间用 stream 顺序 + GM 标志位（或独立 kernel 边界）保证依赖，不加 Host 同步。

**sgetriBatched**：

- BV：每核领取连续矩阵区间（等分 + 余数前缀，确定性），单矩阵全流程（LU 小消元 → 逆）在 UB 内串行完成，逐矩阵写 `infoArray[i]`，零核间同步；
- BL/BC：矩阵级分核同 BV；矩阵内按单矩阵 sgetri 的 blocked 三相（trtri → 乘积 → laswp）执行，GEMM 走 AIC Matmul，per-matrix base 寻址；
- 奇异隔离：主元哨兵触发时仅本矩阵停止（写 `infoArray[i]=k`），不设全局 stopFlag，其余矩阵照常。

### 精度、确定性与测试

- 精度：按任务书测试包全量 case；补充本地对拍（CPU LAPACK reference，`sgetrs` 残差 `‖AX−B‖/(‖A‖·‖X‖)`、`sgetri` 残差 `‖A·invA−I‖`，双步判定对齐仓内既有标准）。
- info 全语义覆盖：正常 0、参数负值、奇异正值（构造主元为 0 的矩阵）、batched 逐矩阵混合奇异。
- 边界：n=1、nrhs=1、lda=ldb=n（无 padding）与 padding、batch=1、B 与 A 别名（禁止，负参数）、NaN/Inf 传播。
- 确定性：同输入同 stream 串行 3 次逐位一致（分核等分 + 余数前缀保证）。
- 性能：msprof kernel 口径，对任务书全部性能 case 出比值表；批量 case 以带宽下限自检。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 任务书接口签名（是否含 bufferSize/workspace、批量布局）与仓内现有契约不一致 | 开发初期以任务书签名冻结 -i 参数序；差异点在设计评审提出，默认沿用仓内 `aclError + int64_t` 契约，不双轨 |
| sgetrs 大 n 回代串行依赖导致不达标 | 列扫改分块右视（block 回代 + 块间 GEMM），提高 Cube 占比；小 n 路径保证 benchmark 段不引入 kernel 拆分开销 |
| batched 指针数组与连续布局双形态 | kernel 统一 per-matrix base 寻址，Host 侧一次展开，两形态共用同一 kernel |
| sgetrf/sgetri 既有实现存在任务书口径缺口（如 padding/长方形） | 先跑全量测试包摸底，缺口清单化后按最小修复原则补丁，不重写 |
