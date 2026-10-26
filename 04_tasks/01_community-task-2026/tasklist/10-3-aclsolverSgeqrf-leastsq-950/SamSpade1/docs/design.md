# 需求背景（required）

## 需求来源

10 月社区任务「单精度实数 QR 分解与最小二乘（950）」。任务要求在 Ascend 950PR（Atlas 950）上基于 AscendC 实现稠密 QR 分解、正交变换与最小二乘求解，共 5 个接口，验收通过后合入 <https://gitcode.com/cann/ops-solver>。

5 个接口作为同一社区任务一并交付，不拆分验收：

| NPU 交付接口 | 对标 CUDA 接口 | 语义 |
| --- | --- | --- |
| `aclsolverSgeqrf` / `_bufferSize` | `cusolverDnSgeqrf` | Householder QR 分解 |
| `aclsolverSorgqr` / `_bufferSize` | `cusolverDnSorgqr` | 由反射器生成显式 Q |
| `aclsolverSormqr` / `_bufferSize` | `cusolverDnSormqr` | 用 Q 左乘或右乘 |
| `aclsolverXlarft` / `_bufferSize` | `cusolverDnXlarft` | 块反射器三角因子 T |
| `aclsolverSSgels` / `_bufferSize` | `cusolverDnSSgels` | 单精度最小二乘 |

## 背景介绍

### ops-solver 仓现状分析

ops-solver 当前已有算子为 `cgetrf`、`cgetri`、`cgetri_batched`、`cmatinv_batched`、`sgetrf`、`sgetri`、`cheevj`，均为 LU / 求逆 / 特征值方向，**无任何 QR 或 gels 原型**，本任务须全新实现。

可直接复用的仓内资产：

| 资产 | 路径 | 用途 |
| --- | --- | --- |
| Handle 管理 | `src/utils/aclsolver_auxiliary.cpp` | `aclsolverCreate` / `Destroy` / `SetStream` / `GetStream` |
| 参数校验宏 | `src/utils/assert.h` | `SOLVER_ECHECK`、`CHECK_ACLRT` |
| ~~fp32 cube GEMM~~ | ~~`src/utils/kernel/c32/gemm.hpp`~~ | **不可复用**，见下文「三处影响架构的实测结论」第三条，须自研 C310 版 |
| fp32 三角求解 | `src/utils/kernel/c32/trsm.hpp` | `SolveTrsm<float>`，`SSgels` 回代与 `larft` 内部三角运算 |
| 转置 / padding | `src/utils/kernel/c32/pad.hpp` | `TransPose<float>`、`PadCustom<float>` |

### 目标硬件实测分析

设计前已在 Ascend950PR 真机（CANN toolkit 9.1.0 / driver 25.7.rc1.10）上实测，以下数据来自运行时 `platform_ascendc` 查询与编译器预定义宏导出，非配置文件推断：

| 项 | 实测值 |
| --- | --- |
| `aclrtGetSocName()` | `Ascend950PR_9579` |
| Cube 核 / Vector 核 | 28 / 56 |
| L0A / L0B / L0C | 64 KB / 64 KB / 256 KB |
| L1 / UB / L2 | 512 KB / 248 KB / 128 MB |
| cube 主频 | 1650 MHz |
| cube fp32 矩阵乘 tile | `DT_FLOAT=16,8,16`（M=16, K=8, N=16），**原生支持 fp32** |

cube 原生支持 fp32 直接决定了本设计的精度路线：**Level-3 更新全程保持 FLOAT32，不采用 fp16 拆分凑精度的做法**，避免引入额外舍入误差，也使「相同输入 bit-wise 一致」的确定性要求更容易满足。

### 三处影响架构的实测结论

**（一）架构宏差异导致函数体被编译成空**

仓内既有 kernel 的函数体整体包在 `#ifdef __DAV_C220_VEC__ / #elif __DAV_C220_CUBE__` 中。实测编译器预定义宏为：

| | Ascend 910B | Ascend 950PR |
| --- | --- | --- |
| 向量核 | `__DAV_C220_VEC__` | `__DAV_C310_VEC__` |
| 矩阵核 | `__DAV_C220_CUBE__` | `__DAV_C310_CUBE__` |
| `__CCE_AICORE__` | 220 | 310 |
| `__NPU_ARCH__` | 2201 | 3510 |

在 950 上两个 C220 分支都不成立，函数体被编译为空，表现为 **kernel 正常下发、正常返回、exit 0，但输出等于输入，一个数都没算** —— 这是静默错误，比直接报错更难发现。

本设计的所有新增 kernel **一律使用架构无关的 `__DAV_VEC__` / `__DAV_CUBE__` 宏**（两种架构都会定义），不写死具体架构版本，从源头避免该问题。

**（二）`aclrtGetHardwareSyncAddr` 在 950PR 上不可用**

该接口在 950PR 上返回 `207000 ACL_ERROR_RT_FEATURE_NOT_SUPPORT`（尽管平台配置中 `hardware_sync_betweencore=1`）。仓内 6 个算子的 host 侧调用了它。经代码追踪，它取得的 `sync` 指针**是死参数** —— 一路传到 kernel 入口但不进入函数体，真正的跨核同步完全由 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 完成，该对原语在 950 上实测可用且不死锁。

本设计**不使用 `aclrtGetHardwareSyncAddr`**，跨核同步统一走 `CrossCoreSetFlag` / `CrossCoreWaitFlag`。

**（三）仓内 fp32 cube GEMM 在 950PR 上 trap，须自研 C310 版**

`src/utils/kernel/c32/gemm.hpp` 的 `MatmulCustom<float>` 在 950PR 上实测全部 cube 核报 `aicore error exception`（extend info `timeout or trap error, subErrType 0x4`）。根因是 cube 侧参数结构与搬运指令按架构分版本：

| 环节 | C220（910B） | C310（950） |
| --- | --- | --- |
| L0 分配 | `TQue<QuePosition::A2,2>` | 显式 `LocalTensor<T>(TPosition::A2, 字节偏移, 元素数)` |
| L1→L0 | `LoadData3DParamsV2Pro` + `Load3DSetFMatrix*` | `LoadData2DParamsV2` |
| Mmad | `MmadParams{m,n,k,unitFlag}` | 另需 `disableGemv=true`、`cmatrixSource=false` |
| L0C→GM | `Fixpipe(gm, cL0, FixpipeParamsV220)` | `DataCopy(gm, cL0, DataCopyCO12DstParams)` + `SetFixpipeNz2ndFlag` |
| 流水同步 | TQue 隐式 | 显式 `SetFlag`/`WaitFlag<HardEvent::…>` |

`FixpipeParamsV220` 含 `ndNum` / `srcNdStride` / `dstNdStride`，而 C310 版 `FixpipeParamsC310`（继承 `FixpipeParamsArch3510`）无这三个字段，改为 `dualDstCtl` / `subBlockId` / `params` / `isChannelSplit` 且按 `CO2Layout` 模板化。两者都能编译通过，但在 C310 上套用 V220 结构会生成错误的指令编码。

已按 CANN 自带的官方 arch35 实现（`opp/built-in/op_impl/.../mhc_pre_sinkhorn/arch35/mhc_pre_sinkhorn_cube_compute_arch35.h`）的指令序列完成自研 C310 GEMM 并在真机验证，详见 3.3 节性能实测。

此处另有一个易漏的硬件约束：**`SetFlag`/`WaitFlag` 的事件 ID 在不同 `HardEvent` 类型之间共享**，多种事件挤在同一 ID 上会互相覆盖状态。本实现的分配为 `MTE1_M` 用 0/1、`M_MTE1` 用 3/4、`MTE1_MTE2`/`MTE2_MTE1` 用 0/1、`FIX_M` 用 2（官方 arch35 实现同样特意把 `M_MTE1` 放在 3/4）。ID 冲突的故障特征具有欺骗性：k 方向分块步数 ≤ 2 时能正常通过、≥ 3 步才失败，M 或 N 单维度分块能通过、两者同时分块才失败，故单点用例通过不足以作为结论。

# 需求分析（required）

## 需求描述

使用 AscendC 实现上表 5 组接口（含 `_bufferSize`），参数名、顺序、含义、Device/Host 归属与对标 CUDA 接口严格一致，核心计算在 AI Core 完成，不允许 CPU fallback。

## 需求拆解

1. 数据布局为**列主序**，leading dimension 须支持 padding，禁止要求 `lda == m` 或 `lda == n`。
2. 全部计算接口返回 `aclsolverStatus_t`，状态码语义对齐 cuSolver。
3. `devInfo` / `dinfo` 真实写入：`0` 成功，`-i` 表示第 i 个参数非法（不计 handle）。
4. workspace 口径区分：`geqrf` / `orgqr` / `ormqr` 的 `Lwork` 以**元素个数**计；`Xlarft` / `SSgels` 以**字节**计。
5. 复用既有 handle 管理接口，计算走调用方 stream，禁止无必要的 Host 同步。
6. **确定性计算**：相同输入、相同 stream 串行多次执行，输出（含 `tau` / `info`）bit-wise 一致。
7. 空问题（`m=0` / `n=0` / `k=0` / `nrhs=0`）成功返回。
8. `Xlarft` 仅验收 FLOAT32 四元组组合，其他 dtype 组合返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`；`storev` 仅支持 `COLUMNWISE`。

# 详细设计（required）

## 算子分析

### 数学公式

**1. `aclsolverSgeqrf`** — A = Q · R，A 为 m×n。分解以 Householder 反射器形式原地写入 A：上三角（或梯形）为 R，严格下三角为反射器向量 v；`TAU` 长度 min(m,n)。

单个反射器：H_i = I − τ_i · v_i · v_iᵀ，Q = H_1 · H_2 · … · H_k。

**2. `aclsolverSorgqr`** — 由 (A, tau) 生成前 k 列正交矩阵 Q（m×n，m ≥ n ≥ k ≥ 0），原地覆盖 A。

**3. `aclsolverSormqr`** — C := op(Q)·C（side = LEFT）或 C := C·op(Q)（side = RIGHT），实数下 `T` 与 `C` 均为转置。

**4. `aclsolverXlarft`** — 由 V 中 k 个反射器与 tau 生成三角因子 T（k×k）：

H = I − V · T · Vᵀ

`FORWARD` 时 T 上三角，`BACKWARD` 时 T 下三角。

**5. `aclsolverSSgels`** — 求解 min ‖A·X − B‖_F（须 n ≤ m），解 X 为 n×nrhs，**out-of-place** 写入独立缓冲区。

### 支持数据类型

FLOAT32（`Xlarft` 的 `dataTypeV` / `dataTypeTau` / `dataTypeT` / `computeType` 四元组须全为 `ACLSOLVER_R_32F`，对齐 `CUDA_R_32F`，即 LAPACK SLARFT 语义）。

### 支持形状

列主序 ND，lda 支持 padding。验收下限：

| 接口 | 维数范围 |
| --- | --- |
| `aclsolverSgeqrf` | m, n ∈ [1, 4096]，覆盖方阵与矩形（含 m<n） |
| `aclsolverSorgqr` | m ∈ [1, 4096]，n, k ≤ m，覆盖 k=n 与 k<n |
| `aclsolverSormqr` | m, n ∈ [1, 4096]，side 覆盖 LEFT/RIGHT，trans 覆盖 N/T/C |
| `aclsolverXlarft` | n ∈ [1, 4096]，k ∈ [1, min(n, 256)]，FORWARD 与 BACKWARD |
| `aclsolverSSgels` | m ∈ [n, 4096]，n ∈ [1, 4096]，nrhs ∈ [1, 128] |

不支持 broadcast，不要求图融合。

## 算子实现

### 总体实现方案

5 个接口共享一套 Householder 原语，依赖关系如下，实现顺序亦按此推进：

```
            xlarft  (T 因子)
               │
               ▼
   sgeqrf ──► larfb (块反射器应用)
      │           │
      │           ├──► sorgqr   (反向累积生成 Q)
      │           └──► sormqr   (左乘/右乘 C)
      │
      └──────────────► ssgels = sgeqrf + sormqr(QᵀB) + trsm(R X = B₁)
```

新增共享 kernel 头 `src/utils/kernel/c32/householder.hpp`，内含：

| 原语 | 职责 | 执行单元 |
| --- | --- | --- |
| `GenHouseholder` | 单列生成 (v, τ)，含 2-范数与符号约定 | AIV |
| `PanelQr` | nb 宽面板的非分块 QR（逐列 geqr2） | AIV |
| `FormT` | 由面板 (V, tau) 生成 T 因子，即 xlarft 内核 | AIC + AIV |
| `ApplyLarfb` | C := (I − V T Vᵀ)ᵀ C 等四种组合 | AIC 为主 |

#### 分块策略

采用 LAPACK 标准的 blocked right-looking 算法，块宽 `nb` 由 host 按 L1/L0C 容量与规模选取：

- L0C 为 256 KB = 65536 个 fp32，cube 单次输出块上限 256×256；
- L1 为 512 KB，需同时容纳 A 面板片与 B 片；
- 取 `nb = 64`（n ≤ 512 时取 32，n ≥ 2048 时取 128），保证 T（nb×nb）与面板驻留 L1。

`sgeqrf` 主循环（j 为列块起点）：

1. **面板分解**：对 A[j:m, j:j+nb] 做非分块 QR，产出该面板的 v 与 τ。面板窄（nb ≤ 128），计算以向量指令为主，分配在 AIV 上，按行带切分到多核。
2. **形成 T**：由面板 (V, tau) 调 `FormT` 得 nb×nb 上三角 T。
3. **尾部更新**：A[j:m, j+nb:n] := (I − V T Vᵀ)ᵀ · A[j:m, j+nb:n]，拆为三次 Level-3 运算
   - W := Vᵀ · A_trail（GEMM，cube）
   - W := Tᵀ · W（三角乘，cube）
   - A_trail := A_trail − V · W（GEMM，cube）

尾部更新占总 flops 的主要部分，由 28 个 cube 核按列块并行，核间用 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 在「面板完成 → 尾部更新」处设栅栏。

#### 列主序与 GEMM 的对接

自研的 C310 GEMM 为 NT 型行主序语义（C = A·Bᵀ，A 为 M×K、B 为 N×K，A 与 B 在 L0 加载上完全对称）。列主序 m×n 矩阵（ld = lda）在内存上等价于行主序 n×m 矩阵（ld = lda），故列主序 GEMM

C_col = A_col · B_col

等价于行主序

C_rowᵀ = B_rowᵀ · A_rowᵀ

即交换两个输入操作数即可，无需显式转置搬运。本设计统一在 host 侧完成这层映射，kernel 内只处理行主序，避免在 kernel 中引入转置开销。

#### 确定性保证

- 尾部更新的归约顺序由固定的分块与核分配决定，不依赖运行时竞争；
- 禁止在跨核归约中使用 `SetAtomicAdd`（其累加顺序不确定），改为各核写独立分片后由固定顺序合并；
- Householder 符号约定取 `β = −sign(a_11)·‖a‖₂`，与 LAPACK SLARFG 一致，使 R 对角符号确定。

### host 侧设计

#### 参数校验与 info 写入

各接口按任务书 2.4 的值域表逐项校验，顺序与参数表一致，首个不合法参数的下标 i 以 `-i` 写入 `devInfo`（不计 handle），返回 `ACLSOLVER_STATUS_INVALID_VALUE`。`Xlarft` 的 dtype 组合不合法时返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`。空问题直接写 `info = 0` 并返回 `ACLSOLVER_STATUS_SUCCESS`，不下发 kernel。

#### workspace 计算

| 接口 | workspace 内容 | 口径 |
| --- | --- | --- |
| `Sgeqrf` | T（nb×nb）+ W（nb×n） | 元素个数 |
| `Sorgqr` | T（nb×nb）+ W（nb×n） | 元素个数 |
| `Sormqr` | T（nb×nb）+ W（nb×max(m,n)） | 元素个数 |
| `Xlarft` | 内部 Vᵀ·V 中间块（k×k） | 字节 |
| `SSgels` | tau（n）+ geqrf/ormqr workspace 并集 | 字节 |

`_bufferSize` 与计算接口共用同一套尺寸函数，避免两处口径漂移。

#### tiling 与下发

`nb`、核数（`GetCoreNumAic()` / `GetCoreNumAiv()` 实测取值，不写死）、各块偏移在 host 侧算好后随 kernel 参数下发，支持 dynamic shape（按本次调用维数生成）。kernel 下发使用调用方 stream，host 侧不做 `aclrtSynchronizeStream`。

### kernel 侧设计

#### 任务类型选择

实测结论：GEMM 类纯 cube 运算若用 `KERNEL_TYPE_MIX_AIC_1_2`，而函数体仅在 `__DAV_CUBE__` 守卫下有内容，AIV 侧会拿到空体但仍被拉起并参与 MIX 同步，真机报 `aivec error exception` / `aicore timeout`。

因此按算子性质选择任务类型：

| kernel | 任务类型 |
| --- | --- |
| `xlarft` | `KERNEL_TYPE_MIX_AIC_1_2`（T 因子含三角标量运算与 GEMM） |
| `sgeqrf` / `sorgqr` / `sormqr` / `ssgels` | `KERNEL_TYPE_MIX_AIC_1_2`（面板在 AIV、尾部更新在 AIC） |
| 纯 GEMM 子核 | `KERNEL_TYPE_AIC_ONLY` |

MIX 模式下 AIV 与 AIC 两侧的 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 严格配对，避免单侧空转导致超时。

#### 面板分解（AIV）

逐列处理，第 i 列：

1. `ReduceSum` 求 ‖a‖₂（先取绝对值最大元做缩放，防 fp32 溢出/下溢）；
2. 按 LAPACK SLARFG 约定定出 β 与 τ_i，归一化得 v；
3. 对面板内剩余列做 rank-1 更新 a_j := a_j − τ_i · v · (vᵀ a_j)。

面板按行带切分到多个 AIV 核，列内归约与 rank-1 更新之间用 pipe 同步。

#### T 因子（xlarft）

`FORWARD` 递推（列 j 从 1 到 k）：

- T[1:j-1, j] := −τ_j · V[:, 1:j-1]ᵀ · V[:, j]（GEMM，cube）
- T[1:j-1, j] := T[1:j-1, 1:j-1] · T[1:j-1, j]（上三角乘）
- T[j, j] := τ_j

`BACKWARD` 为对称的反向递推，T 取下三角。k ≤ 256 时 T 与 V 的列块可完整驻留 L1，整个递推在片上完成，仅最终写回 T。

#### 块反射器应用（larfb）

四种 (side, trans) 组合统一归约为三次 Level-3 运算，差别仅在操作数顺序与是否转置：

| side | trans | 计算 |
| --- | --- | --- |
| LEFT | N | C := C − V · T · (Vᵀ C) |
| LEFT | T/C | C := C − V · Tᵀ · (Vᵀ C) |
| RIGHT | N | C := C − (C V) · Tᵀ · Vᵀ |
| RIGHT | T/C | C := C − (C V) · T · Vᵀ |

转置通过前述列主序↔行主序等价关系在 host 侧的操作数映射完成，不做数据搬运。

#### sorgqr

反向累积：Q 初始化为单位矩阵的前 n 列，j 从最后一个块向前，每块调一次 `ApplyLarfb`（LEFT, N）。初始化单位矩阵复用仓内 `src/utils/eye_matrix.h` 的思路。

#### ssgels

1. `sgeqrf(A)` 得 (V, tau, R)；
2. `sormqr(LEFT, T)` 算 B := QᵀB；
3. `SolveTrsm` 解上三角 R·X = B[0:n, :]，结果写入独立的 dX（out-of-place）；
4. `niter` 置 0（直接主精度分解求解，不做迭代精化），对齐 cuSolver 允许的 0 值语义。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950（Ascend 950PR） | √ |

kernel 使用架构无关宏编写，源码在 **950 与 910B 两个 SOC 下都编译通过**（各验过一次）。
这一条是硬要求而不是加分项：`src/CMakeLists.txt` 对 `src/*` 做 glob，任何 SOC 的构建都会
编译全部算子的源码，而 `build.sh` 的默认 SOC 恰好是 `ascend910b` —— 评审者直接敲
`bash build.sh` 就会编到本任务的文件。

但"能编译"不等于"能算对"：cube 路径按 C310 的 mmad / fixpipe 参数结构写的，在 C220 上
语义不等价（`MmadParams` 没有 `disableGemv` 字段）。故 host 侧在下发前核对
`aclrtGetSocName()`，非 Ascend950 直接返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`，不静默算错。
本任务只在 950PR 上验收。

## 算子约束限制

1. 数据布局仅支持列主序 ND。
2. `Xlarft` 仅支持 `storev = ACLSOLVER_STOREV_COLUMNWISE`，仅支持 FLOAT32 四元组；`ROWWISE` 返回 `ACLSOLVER_STATUS_INVALID_VALUE`，非 FLOAT32 组合返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`。
3. `SSgels` 要求 n ≤ m，m < n 返回错误。
4. 不支持 broadcast。
5. 自构造用例规模保证输入数据占用内存在 4 GB 以内。
6. 仅支持 Ascend950 系列：cube 路径按 C310 的 mmad / fixpipe 参数结构实现，host 侧按 SOC 名拦截，非 Ascend950 返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 按 LAPACK 残差判据，见下表 | 任务书 3.2.1 |
| 性能标准 | 每个 case 满足 T_NPU ≤ T_GPU / 0.35 | 任务书 3.3 |

### 精度判据

| 接口 | 残差定义 | 判定 | 判据溯源 |
| --- | --- | --- | --- |
| `Sgeqrf` | ratio = ‖R − Qᵀ·A‖₁ / (m·‖A‖₁·ε) | ratio ≤ max(5·ratio_cpu, 0.1) | LAPACK SQRT01 RESULT(1) |
| `Sorgqr` | ratio = ‖I − Qᵀ·Q‖₁ / (m·ε) | 同上口径 | LAPACK SQRT02 RESULT(2) |
| `SSgels`（方阵） | ratio，R = B − A·X | ratio ≤ max(5·ratio_cpu, 0.015) | 任务书 3.2.1 |
| `SSgels`（超定） | ratio_opt = ‖Rᵀ·A‖₁ / (‖A‖₁·‖B‖₁·MAX(m,n,nrhs)·ε) | ratio_opt ≤ 10·ratio_opt_cpu | 任务书 3.2.1 |

ε = 2⁻²⁴ ≈ 5.96e-8。golden 构造中 Q 由 NPU 输出的 (v, tau) **升精度至 float64** 后调 CPU LAPACK 生成，使构造过程不引入额外舍入误差，残差只反映 NPU 实现本身的质量。

### 性能采集口径

参考 ops-solver 测试工程编译算子测试，用 msprof 对单条用例做 Profiling：

```bash
msprof op --application="${BUILD_DIR}/test/${op}/${op}_test" --output=./prof_${op}
```

在 `--output` 目录下 `OPPROF_*/OpBasicInfo.csv` 中按 kernel 名统计 `Task Duration(us)` 求平均，得 NPU 平均单次 kernel 耗时，与各算子目录 `bench_result.json` 中 `perf.avg_ms` 逐条关联比对。采集时收敛到单条用例，避免多条用例 kernel 在统计中混叠。

### 性能预算核算

按 `bench_result.json` 已有 GPU 实测值反推所需持续算力：

| case | GPU 实测 | 预算（/0.35） | 需要算力 |
| --- | --- | --- | --- |
| `sgeqrf` m=n=1024 | 3.794 ms | 10.8 ms | 0.13 TFLOPS |
| `sgeqrf` m=n=4096 | 25.450 ms | 72.7 ms | 1.26 TFLOPS |
| `sorgqr` m=n=k=4096 | 5.211 ms | 14.9 ms | 6.15 TFLOPS |
| `sormqr` m=n=k=2048 | 2.080 ms | 5.9 ms | 5.78 TFLOPS |
| `sgels` m=n=1024 | 4.353 ms | 12.4 ms | 0.17 TFLOPS |

门禁最紧的一条为 `sorgqr` m=n=k=4096，需 6.15 TFLOPS 持续 fp32 算力。

自研 C310 GEMM 已在 950PR 真机实测（28 cube 核，真 fp32，`SetHF32Mode(0)`）：

| 规模 | 耗时 | 吞吐 |
| --- | --- | --- |
| 1024³ | 0.1292 ms | 16.6 TFLOPS |
| 4096³ | 6.6176 ms | **20.8 TFLOPS** |

相对门禁需求有 3.4 倍余量，且该版本尚未做 L1 双缓冲等优化。精度上，128³ 至 4096³ 全规模通过校验，相对 1-范数残差约 0.03（相对 K·ε 归一），确认走的是真 fp32 而非 HF32 降精度路径 —— 官方 arch35 参考实现默认开启 `SetHF32Mode(1)`（类 TF32 降精度），本设计显式关闭，以满足按 LAPACK 残差判据的精度验收。

## 自验设计

功能自验覆盖（对应任务书 3.5）：

| 类别 | 必测场景 |
| --- | --- |
| 基础功能 | geqrf；geqrf+orgqr 还原 Q；geqrf+ormqr；Xlarft；SSgels |
| 矩形 | m>n、m=n、m<n（仅 geqrf）；SSgels 仅 m≥n |
| side / trans | ormqr 的 LEFT/RIGHT 与 N/T/C 全六种组合 |
| Xlarft | FORWARD/BACKWARD；非法 ROWWISE 报错；非 FLOAT32 返回 NOT_SUPPORTED |
| SSgels | nrhs=1 与多右端；m=n 与 m>n；m<n 报错 |
| padding | lda/ldc/ldv/ldt 取最小合法值及 +8 / +32 |
| info | 成功 info=0；非法参数 info=-i |
| 空问题 | m=0 或 n=0 或 k=0 或 nrhs=0 成功返回 |
| 确定性 | 同一输入重复执行 bit-wise 一致 |
| 流 | 非默认 stream 下结果正确 |

入参生成规则：A / C / B 取 50% 均匀 [-5,5]、50% 正态（μ∈[-5,5]、σ∈[0.1,2]）；m、n 覆盖 2 的幂与 2 的幂−1；nrhs 覆盖 1、8、32、128 及 0。

golden 由 NumPy / SciPy LAPACK 以 float64 生成。测试工程置于 `test/<op>/`，含 `data/gen_data.py`、`data/verify_result.py` 与 `README.md`，`README.md` 写明复现步骤。

**实测结果（已落地，详见代码 PR 里的 `docs/self_test_report.md`）**：
代码 PR <https://gitcode.com/cann/ops-solver/merge_requests/98>。

- 精度：随附用例包全量 **694 条跑完，692 条 PASS**（sgeqrf 174/174、sorgqr 165/165、
  sormqr 189/189、ssgels 164/166）。余下 2 条是超定分支纯相对判据在
  `ratio_opt_cpu` 小到 1e-7 时失去分辨力，按任务书 3.2.1 的注记走"举证实现无 bug"路径
  （两边绝对残差几乎逐位相同，报告 §3.2 给了分桶统计与迭代精化为何不可行的推导）。
- 性能：任务书 3.3 的 12 条全部达标，且按**更严的 min_ms 口径**自我要求每条余量 ≥ 1.3x
  （最紧的 P-08 为 1.34x、P-06 为 1.38x）。起点有四条不达标（P-09 曾超门禁 17.1 倍），
  推动过线的是三项改动：T 递推改分块 LARFT、修掉两处"扫整块"的标量循环、
  UB 内转置改用向量 `Gather`。
- 鲁棒性：关键规格各复跑 10 次，逐次核对精度与 bit-wise 确定性；交付件在洁净副本
  （只含 git 跟踪文件）里 950 与 910B 两个 SOC 都编译通过。

### 随附用例与性能表的对应缺口

核对任务随附的 4 组用例（`sgeqrf` 174 例、`sormqr` 189 例、`sorgqr` 165 例、`sgels` 166 例）与任务书 3.3 性能表后，以下缺口已识别，相应 case 将按 3.5 的自构造规则补齐，GPU 基线部分待官方补发：

1. `Xlarft` 无随附用例包，性能表 P-08 / P-09 无 GPU 基线。
2. P-03（`sgeqrf` m=2048, n=1024）、P-07（`sormqr` LEFT+trans=T）、P-11 / P-12（`sgels` nrhs=8 / 32）在随附用例中无对应条目。
3. `sgels/bench_result.json` 中 `failed` 为 108（共 166 条），这 108 条 `perf` 为空且全部为 m>n 的超定情形，无 CUDA 参照耗时。
4. 随附用例的 lda/ldc/ldda/lddb/lddx padding 均为 0，0 维用例 0 条，`sormqr` 的 side/trans 仅覆盖 (LEFT,N) 与 (RIGHT,T) 两种组合，`sgeqrf` 无 m<n 用例，`sgels` 无 m<n 报错用例且 nrhs 恒为 1。

精度自验不依赖 GPU 基线，上述缺口不阻塞精度验收；性能验收以官方补发的基线为准。

## 兼容性分析

新增算子，不修改任何既有算子的行为与接口，不涉及兼容性问题。

对公共头文件的改动均为**新增**：

- `include/cann_ops_solver_common.h`：新增 `aclsolverParams_t`、`aclsolverOperation_t`、`aclsolverSideMode_t`、`aclsolverDirectMode_t`、`aclsolverStorevMode_t`、`aclsolverDataType_t`，枚举取值与 cuBLAS / cuSolver 对应常量对齐。
- `include/cann_ops_solver.h`：新增 10 个 `aclsolverStatus_t` 接口声明。

新增算子目录由 `src/CMakeLists.txt` 的自动 glob 收集，无需修改构建脚本。`docs/api_list.md` 同步更新（根 `README.md` 只以「接口列表」链接到它，本身不逐个列算子，故无需改动）。
