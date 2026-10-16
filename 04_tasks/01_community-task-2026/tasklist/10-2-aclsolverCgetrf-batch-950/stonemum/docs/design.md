# Atlas 950PR 单精度复数 LU 分解、求解和批量求逆算子设计文档

> 任务名称：10月社区任务-单精度复数LU分解、求解和批量求逆（950）
>
> 任务编号：`10-2-aclsolverCgetrf-batch-950`
>
> 提交账号：`stonemum`
>
> 适配硬件：Ascend 950PR（Atlas 950，`dav-3510` / `arch35`）
>
> 目标代码仓：`cann/ops-solver`
>
> 本文档基于已落地的 Device C API 改造、Ascend 950PR Kernel 实现和真机验证方法编写，不是仅停留在算法推导阶段的理论方案。本文不包含验收阶段的具体精度和性能结果，实测数据将在自测报告中单独提交。

# 需求背景（required）

## 需求来源

本设计来自 2026 年 10 月 CANN 开源社区任务“单精度复数LU分解、求解和批量求逆（950）”。任务要求在 `cann/ops-solver` 中升级现有复数 LU 与求逆原型，使接口、数据归属、内存布局、错误码和数值语义与 cuSolver、cuBLAS 及 LAPACK 对标接口一致，并在 Ascend 950PR 上完成开发和验证。

交付范围包含四个计算接口与两个 workspace 查询接口：

| 类别 | 接口 | 对标接口 | 功能 |
| --- | --- | --- | --- |
| 查询 | `aclsolverCgetrf_bufferSize` | `cusolverDnCgetrf_bufferSize` | 查询 LU 分解所需 Device workspace |
| 计算 | `aclsolverCgetrf` | `cusolverDnCgetrf` | 对单精度复数矩阵执行带部分主元的原地 LU 分解 |
| 查询 | `aclsolverCgetri_bufferSize` | LAPACK `cgetri` workspace 查询语义 | 查询单矩阵求逆所需 Device workspace |
| 计算 | `aclsolverCgetri` | LAPACK `cgetri` | 消费 Cgetrf 输出的 LU 因子和主元，原地计算逆矩阵 |
| 计算 | `aclsolverCgetriBatched` | `cublasCgetriBatched` | 基于已分解 LU 的批量 out-of-place 求逆 |
| 计算 | `aclsolverCmatinvBatched` | `cublasCmatinvBatched` | 对 `n <= 32` 的原始小矩阵执行直接批量求逆 |

## 背景介绍

### 数学背景

对复数矩阵 $A \in \mathbb{C}^{m \times n}$，带部分主元的 LU 分解写为：

$$
P A = L U,
$$

其中 $P$ 为由行交换构成的置换矩阵，$L$ 为单位下三角矩阵，$U$ 为上三角或上梯形矩阵。对非奇异方阵，有：

$$
A^{-1} = U^{-1}L^{-1}P.
$$

因此，Cgetri 和 CgetriBatched 应直接消费 Cgetrf 已生成的 LU 因子和主元序列，而不是再次对输入执行 LU 分解。CmatinvBatched 面向小矩阵原始输入，可以采用融合 LU 与三角求解的直接路径。

### 现有原型与正式接口的差距

`ops-solver` 已存在 `cgetrf`、`cgetri`、`cgetri_batched` 和 `cmatinv_batched` 原型，但原型主要用于功能探索，存在以下差距：

1. 部分接口使用 Host 指针或连续 `[batch, n, n]` 内存，未采用 Device 指针数组；
2. 部分接口要求 `lda == n`，不能覆盖合法 padding；
3. `info` 语义不完整，未逐矩阵报告参数错误或奇异位置；
4. 单矩阵接口缺少调用方 workspace 查询或执行时仍进行内部内存分配；
5. 单矩阵求逆原型会重新分解输入，未严格消费 Cgetrf 的 LU 与主元；
6. CmatinvBatched 对超范围矩阵存在转调行为，与 `n <= 32` 的正式契约不一致；
7. 原型中的同步和 Host 搬运不适合 solver API 的异步 stream 组合方式。

本设计统一改造公开 C ABI、Device 内存语义和 Kernel 数据流，保留与仓库现有 handle/stream 管理接口的一致性。

### 任务包验证范围

任务正文中的规模表为验收下限。设计按任务包 `cases.json` 和成功的 `bench_result.json` 覆盖实际验证范围：

| 接口 | 用例数 | 实际覆盖范围 |
| --- | ---: | --- |
| Cgetrf | 148 | `m <= 4096`，`n <= 8168`，包含方阵、高矩阵和宽矩阵 |
| Cgetri | 200 | `n <= 4096` |
| CgetriBatched | 89 | batchSize 最大 1,000,000 |
| CmatinvBatched | 215 | `n <= 32`，batchSize 最大 10,000,000 |

四组用例合计 652 条，均纳入功能、精度和性能验证规划。

# 需求分析（required）

## 需求描述

使用 AscendC/CATLASS 能力在 Ascend 950PR 上实现单精度复数 LU 分解和矩阵求逆接口，满足以下统一要求：

- 使用 C ABI，返回 `aclsolverStatus_t`；
- `aclFloatComplex` 由相邻的 `real`、`imag` 两个 FLOAT32 组成；
- 矩阵采用列主序，支持合法的 leading dimension padding；
- 矩阵、workspace、主元、info 和批量指针数组均位于 Device；
- 计算在 handle 绑定的 stream 上异步执行；
- 不在正常计算路径中进行 Host 矩阵搬运、Host 逐矩阵处理或隐式同步；
- 主元选择和并行调度具备 bit-wise 确定性；
- 精度按 LAPACK CGET01/CGET03 残差公式验证；
- 性能按单次 API 发出的全部 NPU Kernel 时间之和与任务包 GPU 基准比较。

## 公开类型与接口

### 公共类型

```c
typedef struct {
    float real;
    float imag;
} aclFloatComplex;

typedef enum {
    ACLSOLVER_OP_N = 0,
    ACLSOLVER_OP_T = 1,
    ACLSOLVER_OP_C = 2
} aclsolverOperation_t;
```

### Cgetrf

```c
aclsolverStatus_t aclsolverCgetrf_bufferSize(
    aclsolverHandle_t handle,
    int m,
    int n,
    aclFloatComplex *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverCgetrf(
    aclsolverHandle_t handle,
    int m,
    int n,
    aclFloatComplex *A,
    int lda,
    aclFloatComplex *Workspace,
    int *devIpiv,
    int *devInfo);
```

### Cgetri

```c
aclsolverStatus_t aclsolverCgetri_bufferSize(
    aclsolverHandle_t handle,
    int n,
    aclFloatComplex *A,
    int lda,
    int *Lwork);

aclsolverStatus_t aclsolverCgetri(
    aclsolverHandle_t handle,
    int n,
    aclFloatComplex *A,
    int lda,
    int *devIpiv,
    aclFloatComplex *Workspace,
    int Lwork,
    int *devInfo);
```

### CgetriBatched

```c
aclsolverStatus_t aclsolverCgetriBatched(
    aclsolverHandle_t handle,
    int n,
    const aclFloatComplex *const Aarray[],
    int lda,
    const int *PivotArray,
    aclFloatComplex *const Carray[],
    int ldc,
    int *infoArray,
    int batchSize);
```

### CmatinvBatched

```c
aclsolverStatus_t aclsolverCmatinvBatched(
    aclsolverHandle_t handle,
    int n,
    const aclFloatComplex *const A[],
    int lda,
    aclFloatComplex *const Ainv[],
    int lda_inv,
    int *info,
    int batchSize);
```

## 需求拆解

1. **公开接口升级**：统一 C ABI、Device 指针归属、32-bit 维数类型和 solver 状态码；
2. **列主序与 padding**：支持 `lda/ldc/lda_inv >= max(1,n)`，只读写有效矩阵区域；
3. **Workspace 管理**：Cgetrf/Cgetri 提供无状态查询接口，执行接口只切分调用方 Device workspace；
4. **错误语义**：参数错误写 `-i`，奇异矩阵写首个正对角位置，成功写 0；
5. **确定性**：固定主元度量、tie-break、分核映射和 Kernel 发射顺序；
6. **批量扩展**：Kernel 直接读取 Device 指针数组，禁止 Host 逐矩阵循环；
7. **超大 batch**：采用 batch chunk 和有界临时空间，不按 batch 展开单位阵或 scratch；
8. **性能验证**：统计一次 API 调用中全部 Kernel 的 `Task Duration(us)`，对齐任务包 GPU 口径。

# 详细设计（required）

## 算子分析

### 数学公式

#### Cgetrf

对每个 panel 执行确定性部分主元选择，完成行交换和 panel 内消元，随后计算：

$$
U_{12} = L_{11}^{-1} A_{12},
$$

并通过 GEMM 更新 trailing matrix：

$$
A_{22} \leftarrow A_{22} - L_{21}U_{12}.
$$

#### Cgetri / CgetriBatched

对于 `P A = L U`：

$$
A^{-1} = U^{-1}L^{-1}P.
$$

实现时以置换后的单位阵为右端项，依次执行单位下三角前代和上三角回代，避免显式构造完整的 $L^{-1}$ 和 $U^{-1}$。

#### CmatinvBatched

对 `n <= 32` 的小矩阵，在单个批主序 Kernel 中融合主元搜索、LU 消元和单位阵求解，减少中间矩阵写回与 Kernel 发射次数。

### 复数运算

复数 GEMM 采用四个实数乘积组合：

$$
(A_r+iA_i)(B_r+iB_i)
=
(A_rB_r-A_iB_i)+i(A_rB_i+A_iB_r).
$$

内部将实部、虚部以及需要的负虚部拆为平面存储，使 AIV 上的复数标量操作与 AIC Cube FP32 GEMM 能够协同执行。

复数倒数采用缩放形式：先以 `max(abs(real), abs(imag))` 归一化，再计算平方和，避免有限大数平方溢出或有限小数平方下溢。

### 主元规则

主元度量与 LAPACK ICAMAX 兼容：

$$
metric(z)=|Re(z)|+|Im(z)|.
$$

当度量相同时选择原始行号较小者。输出主元为 1-based。`devIpiv == nullptr` 或 `PivotArray == nullptr` 表示输入来自 no-pivot LU 路径。

## 算子实现

### 实现方案总览

统一数据流如下：

```text
Host 参数校验 / workspace 规划
              |
              v
Device 列主序 AoS pack / 指针数组读取
              |
              v
内部 row-major / split-planar 工作布局
              |
              v
分阶段 producer / consumer Kernel
              |
              v
Device 列主序 finalize / 每矩阵 info 写回
```

对于中大矩阵，采用多个同 stream Kernel 分阶段执行。前一 Kernel 完成后，后一 Kernel 才开始读取其 Device 输出，使用运行时提供的 stream 顺序作为阶段完成边界。这样可以避免不同 AIV/AIC 分支依赖隐式参与者计数，同时保持接口正常路径异步。

### Host 侧设计

#### 参数校验

Host 立即校验可见参数：

- handle；
- `m/n/batchSize`；
- `lda/ldc/lda_inv`；
- 顶层 Device 指针；
- `Lwork` 与查询值；
- CmatinvBatched 的 `n <= 32` 约束。

空问题按接口契约成功返回并写 0。若非法 Host 参数要求写 `devInfo=-i`，在返回错误前完成该次 Device info 写入，避免调用方收到错误后立即释放 info 导致生命周期问题。

Device 矩阵内容、批量指针数组内部元素、主元内容和 NaN/Inf 属于调用方前置条件。正常路径不为了内容扫描同步 stream；`bufferSize` 只依赖 shape、leading dimension 和内部布局，不读取 A。

#### Stream 与生命周期

- 从 handle 获取 stream；
- 所有 Kernel 按同一 stream 顺序发射；
- 正常路径不调用 `aclrtSynchronizeStream`；
- 单矩阵接口不内部申请 Device workspace；
- 批量接口使用有界内部池或 stream-ordered 临时区，禁止通过每次同步后释放来管理生命周期。

#### Workspace 规划

Cgetrf workspace 主要包括：

1. 列主序 AoS 到内部 row-major AoS 的 packed A；
2. real/imag/negative-imag planar 工作区；
3. 主元与 panel workspace；
4. 转置/重排所需的小型 gather 表。

Cgetri workspace 主要包括：

1. 输入 LU 的 packed / planar 工作区；
2. 置换单位阵和求解结果的 real/imag/negative-imag 平面；
3. 主元重放或 row-map 工作区；
4. 上下三角分块求解所需的有界 scratch。

所有区域按字节对齐切分，查询结果换算为 `aclFloatComplex` 元素数。所有容量和偏移使用 `size_t/int64_t` 计算，并检查转为公开 `int Lwork` 时是否溢出。

#### 批量调度

CgetriBatched 与 CmatinvBatched 直接在 Kernel 中读取 Device 指针数组。Host 只按 `n`、`batchSize` 和硬件并行度选择 chunk 大小，不访问数组内部指针。

- 小矩阵以多个矩阵组成一个批主序处理组；
- 中大矩阵以“矩阵索引 × tile索引”构造网格；
- infoArray/info 按矩阵独立写入；
- 临时空间与并发 chunk 大小相关，不与总 batchSize 线性增长。

### Kernel 侧设计

#### 公共布局转换

公开 A 为列主序 AoS：

```text
A[row + column * lda]
```

prepare Kernel 只读取有效 `n` 或 `m x n` 区域，并转换为内部紧致 row-major / planar 布局。对齐区域采用确定性零扩展或块对角单位扩展。finalize Kernel 仅写有效区域，不修改 padding 行。

#### Cgetrf

Cgetrf 使用右看分块 LU：

1. **prepare**：列主序 Device A pack 到内部布局；
2. **panel factorization**：AIV 执行确定性 ICAMAX 主元搜索、行交换和 GETF2；
3. **pivot propagation**：将 panel 主元应用到 panel 外列；
4. **TRSM**：求解当前 panel 对应的 U 行块；
5. **trailing update**：AIC Cube 以 FP32 GEMM 更新剩余矩阵；
6. **finalize**：压缩主元为 1-based，扫描 U 对角并写 info，回写列主序 A。

中大矩阵的 producer/consumer 阶段使用同 stream 独立 Kernel，阶段间由 Kernel 完成语义保证可见性。AIC/AIV 混合 Kernel 保持 20 个逻辑 AIC block 与两个 AIV sub-block 的映射，AIV 逻辑编号采用 `coreIdx >>= 1`。

宽矩阵只分解 `min(m,n)` 个 panel 列。完成主元后，对右侧宽尾重建 $P A_{tail}$，再执行完整下三角前代，避免 Host 逐列处理。

#### Cgetri

Cgetri 输入为 Cgetrf 输出的 LU，不再次分解：

1. 校验并 pack LU，扫描 U 对角，首个精确零对角写正 info；
2. 根据 1-based `devIpiv` 构造置换后的单位阵；no-pivot 时直接使用单位阵；
3. 解 $L Y = P$；
4. 解 $U X = Y$；
5. 将 X 回写到原 A 的有效列主序区域。

下三角 solve 采用单位对角前代，上三角 solve 使用稳定复数倒数。奇异时 solve/finalize Kernel 读取 info 并提前退出，避免产生非有限输出。

#### CgetriBatched

CgetriBatched 的每个 `Aarray[i]` 已包含 LU 因子，`PivotArray[i*n : (i+1)*n]` 为主元。算法按 n 分层：

- 小矩阵：单个 AIV group 完成主元重放、前代和回代；
- 中矩阵：一个矩阵映射到一个或少量核心，LU 与 RHS 尽量驻留 UB；
- 大矩阵：复用单矩阵分块三角求解，以矩阵维度和 batch 维度联合分核。

输出写入独立的 `Carray[i]`，不修改输入 LU。每个矩阵独立写 `infoArray[i]`。

#### CmatinvBatched

CmatinvBatched 仅支持 `0 <= n <= 32`。`n > 32` 在 Host 立即返回参数错误，不转调 CgetriBatched。

对合法小矩阵，Kernel 执行：

1. 从 `A[i]` 指针读取原始矩阵；
2. 在 UB 中执行确定性部分主元 LU；
3. 同时维护单位阵/置换右端项；
4. 完成前代与回代；
5. 将结果写入 `Ainv[i]`，并写 `info[i]`。

对于超大 batch，单次 launch 网格覆盖多个矩阵，按硬件上限分 chunk，不创建 batchSize 份单位阵或中间矩阵。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Atlas 950（dav-3510） | √ |
| Atlas A2/A3（兼容路径） | √ |
| Ascend 310P | 不支持正式数值核心，返回架构不匹配 |

## 算子约束限制

- 数据类型仅支持 `COMPLEX64`；
- 矩阵采用列主序，不支持 broadcast；
- Cgetrf/Cgetri 的正常执行需要满足 bufferSize 查询所得 workspace；
- Cgetri/CgetriBatched 的 LU 与主元模式必须一致；
- CgetriBatched/CmatinvBatched 的输入输出矩阵不得重叠；
- CmatinvBatched 仅支持 `n <= 32`；
- Device 指针必须满足 CANN Runtime 的地址有效性和生命周期要求；
- Device 内容不在 Host 同步扫描，调用方需保证输入为有限复数并满足接口前置条件。

# 可维可测分析

## 精度标准/性能标准

### 精度标准

Cgetrf 使用 LAPACK CGET01 残差：

$$
ratio_{getrf}
=
\frac{\lVert LU-A\rVert_1}
{n\cdot\lVert A\rVert_1\cdot 2^{-24}}.
$$

矩形矩阵的 1 范数按每列的 m 行求和。构造 $LU$ 后按主元序列逆序回放行交换。判据为：

$$
ratio_{npu} \le \max(5\cdot ratio_{cpu}, 0.5).
$$

Cgetri、CgetriBatched 和 CmatinvBatched 使用 LAPACK CGET03：

$$
ratio_{getri}
=
\frac{\lVert I-C A\rVert_1}
{n\cdot\lVert A\rVert_1\cdot\lVert C\rVert_1\cdot 2^{-24}}.
$$

其中 Cgetri/CgetriBatched 的 A 由保存的 LU 和主元恢复，CmatinvBatched 使用原始输入 A。动态判据分别按任务书设置 floor。

CPU 参考同时保留两条链路：

- COMPLEX64 LAPACK 链路用于计算任务要求的动态 `ratio_cpu`；
- COMPLEX128 链路用于高精度结果诊断，不替代 COMPLEX64 动态阈值。

### 确定性验证

固定输入在同一 stream 上重复执行，逐次计算以下三个 SHA-256：

1. 矩阵输出；
2. 主元输出；
3. info 输出。

全部重复执行必须得到唯一 hash，同时每次残差均通过。验证覆盖单 panel、多 panel、128 边界、宽矩阵、高矩阵和大规模矩阵。该方法用于检查数值确定性和 Device 调度确定性，不以少量偶然通过代替长期稳定性。

### 性能标准与测量

性能目标按任务书第 3.3 节：

$$
T_{NPU} \le \frac{T_{GPU}}{0.35}.
$$

NPU 使用 `msprof op --aic-metrics=BasicInfo` 采集。对多 Kernel API，统计一次 API 调用发出的全部计算 Kernel `Task Duration(us)` 之和，包括布局转换、panel、TRSM、GEMM、批量调度和 finalize，不选取单个最快 Kernel。bufferSize、测试 H2D/D2H 和进程初始化不计入设备计算时间。

本文档不发布具体测试数值；完整数据、日志和截图在验收自测报告中提供。

## 测试覆盖规划

### 任务包全量

| 接口 | 用例数 | 验证内容 |
| --- | ---: | --- |
| Cgetrf | 148 | 方阵/矩形、对齐边界、主元、残差、性能 |
| Cgetri | 200 | Cgetrf→Cgetri、原地逆矩阵、CGET03、性能 |
| CgetriBatched | 89 | 指针数组、batch、逐矩阵CGET03/info、性能 |
| CmatinvBatched | 215 | `n<=32`、超大batch、逐矩阵CGET03/info、性能 |

### 补充语义用例

- `n/m=0` 空问题；
- `n=1`；
- 15/16/17、31/32/33、127/128/129、255/256/257、511/512/513；
- `lda=n`、`n+8`、`n+32`；
- pivot/no-pivot；
- 首个/中间/末尾精确零对角；
- 非默认 stream；
- Cgetrf 与 Cgetri 同 stream 连续调用，中间无 Host 同步；
- 批量矩阵异址分配、混合奇异和 padding canary；
- CmatinvBatched `n=33` 错误路径；
- bit-wise 重复执行。

## 兼容性分析

本任务在现有 `ops-solver` 原型基础上升级正式 C API。公共 handle/stream 接口保持不变；新增公共类型和正式 C 声明使用 `extern "C"`。历史 C++ 原型在迁移阶段隔离保留，避免无关接口一次性破坏；正式接口完成验证后，文档和测试统一迁移到 Device C API。

内部阶段化 Kernel、workspace 切分和批量 chunk 不改变公开参数语义。A2/A3 兼容路径保留原有架构实现，950PR 使用 `dav-3510` 专用数据搬运和 AIC/AIV 调度。

## 希望检视人员了解

1. 任务书个别位置提到“五个接口”或 Cgetrs，但第 2.3 节公开原型、任务包和验收用例明确对应四个计算接口与两个查询接口。本设计按第 2.3 节交付范围执行，请确认以该范围为准。
2. 任务包实际范围包含 Cgetrf `n=8168` 和 CmatinvBatched `batchSize=10000000`，本设计按全部 652 条附件用例规划，不将正文规模表的验收下限作为拒绝扩展用例的上限。
3. Device 内容检查若要求同步返回 Host 错误，将与正常异步 stream 语义冲突。本设计立即校验 Host 可见参数，Device 内容通过 info 和调用方前置条件约束，请确认该口径。
4. 多 Kernel 路径的性能按一次 API 调用全部设备 Kernel 时间之和与 GPU 基准比较。
