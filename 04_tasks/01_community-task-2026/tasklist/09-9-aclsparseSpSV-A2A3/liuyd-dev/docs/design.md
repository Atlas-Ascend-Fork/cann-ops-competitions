# aclsparseSpSV 算子设计文档（Atlas A2/A3，arch22）

| 项 | 内容 |
| --- | --- |
| 算子名 | `aclsparseSpSV`（`createDescr / destroyDescr / bufferSize / analysis / solve / updateMatrix`） |
| 任务 | 9 月社区任务 — aclsparseSpSV 算子开发（A2/A3） |
| 目标仓 | [ops-sparse](https://gitcode.com/cann/ops-sparse)，实现 `sparse/spsv/arch22/`，测试 `test/spsv/arch22/` |
| 对标基线 | cuSPARSE Generic API `cusparseSpSV` |
| 数据类型 | `ACL_FLOAT`（FP32）、`ACL_COMPLEX64`；Device 索引 I32 |
| 目标硬件 | Atlas A2 训练/推理系列、Atlas A3 训练/推理系列（arch22，DAV_2201） |
| CANN 版本 | 9.1.0 |
| 验证依据 | 于 Atlas 800T A2（**910B3**）与 **Atlas A3** 两款硬件各完整实测一轮：仓内 C++ UT/ST **304/304**、独立验证程序 **286/286**、`performance_cases.json` 全量 **206/206 达标**（门槛 0.25×，910B3 中位 11.97×、A3 中位 13.33×）、最大 workspace 48.0 MB = 实测 L2 的 25.0%。详见 §5.1 |

---

# 1 需求背景（required）

## 1.1 需求来源

9 月社区任务「aclsparseSpSV 算子开发（A2/A3）」。`ops-sparse` 仓当前只有
`sparse/spsv/arch35/` 一份面向 Ascend 950PR/DT 的实现，`sparse/spsv/README.md`
的产品支持表把 Atlas A2/A3 标为「不支持」；该实现同时只支持 `ACL_FLOAT`，
任务书要求的 `ACL_COMPLEX64` 也属于能力缺口。本任务为该算子补齐 arch22 实现。

## 1.2 背景介绍

稀疏三角求解（Sparse Triangular Solve，SpSV）是 `op(A)·Y = alpha·X` 的稀疏版本，
是不完全 LU/Cholesky 预条件子（ILU/IC）每次迭代都要执行的核心步骤，也是稀疏直接法
分解后回代的必经环节。它的特点与稠密 `trsm` 截然不同：

- **并行度由输入的依赖结构决定，而不是由形状决定**。行 `i` 的解依赖它在目标三角内
  引用的所有行的解，可并行性完全取决于依赖 DAG 的层宽。稠密 `trsm` 的层宽恒为块宽，
  SpSV 的层宽可以从 1（纯串行链）到 m（纯对角阵）跨五个数量级。
- **nrhs 恒为 1**，没有右端项维度可供并行。这是 SpSV 与同仓已合入的
  `aclsparseSpSM`（arch22）最本质的差别：SpSM 的并行维是右端项列数 nrhs，
  其「每层一次 kernel + 按列分核」的调度在 nrhs = 1 时退化为每层只有一个核在工作。
- **算术强度极低**。每个非零元只做一次乘加，纯内存与调度开销受限；性能工程的对象
  是 DMA 粒度与 kernel 下发次数，而不是 Cube 利用率（本算子不使用 Cube）。

### 1.2.1 arch35 现状分析

| 项 | `sparse/spsv/arch35/` 现状 | 本任务要求 |
| --- | --- | --- |
| 平台 | Ascend 950PR/DT（arch35，DAV_3510） | Atlas A2/A3（arch22，DAV_2201） |
| computeType | 仅 `ACL_FLOAT` | `ACL_FLOAT` + `ACL_COMPLEX64` |
| 稀疏格式 | CSR 为主 | CSR、CSC、COO、SLICED_ELL 四格式 |
| 编程模型 | SIMT（`__simt_vf__`） | Ascend C Vector API（`__aicore__`） |
| 索引宽度 | I32 / I64 | I32（任务书 2.4：Device 索引用 I32） |

arch35 的实现无法移植：SIMT 编程模型在 arch22 上不存在，其按 lane 组织的行内归约与
warp 级 shuffle 在本平台没有对应能力。因此 arch22 是一份独立实现，只共享公开接口与
Host 侧的描述符语义。

### 1.2.2 标杆接口（cuSPARSE SpSV）实现描述

cuSPARSE 的 `cusparseSpSV_bufferSize / analysis / solve` 把结构分析与数值求解分离：
analysis 阶段做依赖分析并把结果缓存在不透明的 `spsvDescr` 中，solve 阶段复用该缓存。
本设计沿用同一生命周期，并按任务书 2.3 增加 `aclsparseSpSV_updateMatrix`
（cuSPARSE 无等价接口，见 §2.3），使数值变化但 pattern 不变的场景可以跳过结构分析。

## 1.3 算子原型

公开接口与 `include/cann_ops_sparse.h` 的现有声明**逐字一致**，不新增、不修改签名：

```cpp
aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize);

aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr, void *externalBuffer);

aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType, aclsparseSpSVAlg_t alg,
    aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart);
```

---

# 2 需求分析（required）

## 2.1 需求描述

求解 `op(A)·Y = alpha·X`：A 为 m×m 稀疏三角方阵，X、Y 为稠密向量。
`op(A) = A`（`ACL_SPARSE_OP_NON_TRANSPOSE`）、`Aᵀ`（`..._TRANSPOSE`）或
`Aᴴ`（`..._CONJUGATE_TRANSPOSE`，complex64 下执行共轭转置语义）。
A 的存储格式为 CSR / CSC / COO / SLICED_ELL 之一，索引 base 0 或 1，
fill mode 为 LOWER / UPPER，diag type 为 UNIT / NON_UNIT。

## 2.2 需求拆解

| # | 要求 | 任务书出处 |
| --- | --- | --- |
| 1 | 四种格式 × `ACL_SPARSE_SPSV_ALG_DEFAULT` 的完整 Host/Kernel 路径 | 2.1-1 |
| 2 | LOWER/UPPER、UNIT/NON_UNIT、N/T/H 全组合；complex64 的 H 执行共轭转置 | 2.1-2 |
| 3 | handle 的 Host 与 Device pointer mode，`alpha` 在对应位置读取 | 2.1-3 |
| 4 | `BufferSize → Analysis → Solve → UpdateMatrix` 生命周期；BufferSize/Analysis 允许 vecX/vecY 描述符为 NULL；Analysis 缓存格式/三角属性/op/dtype/维度/索引与 workspace 绑定状态 | 2.1-4 |
| 5 | `GENERAL` 更新全部 values、`DIAGONAL` 更新对角 values，更新后复用同一描述符求解 | 2.1-5 |
| 6 | 未排序索引；确定的规范化与求解顺序 ⇒ bitwise deterministic；A/X 只读；Y 可与 X 同址原地求解 | 2.1-6 |
| 7 | 三角结构/对角属性/维度/索引范围/dtype/pointer mode/workspace 在 Host 侧校验并返回确定错误码；所有 kernel 走 handle 的 stream 异步执行 | 2.1-7 |
| 8 | 主求解与格式相关计算由 NPU kernel 执行，**禁止 CPU fallback** | 2.1-8 |
| 9 | 精度达《生态算子开源精度标准》；性能每场景 ≥ 0.25× 标杆；workspace ≤ L2 | 3.2 / 3.3 / 3.4 |
| 10 | C++ UT/ST + 格式转换规范化专项 UT + 端到端 ST + 性能脚本 + 可复现 README；Profiler 证据 | 3.5 |

## 2.3 与 cuSPARSE SpSV 的差异

| 项 | cuSPARSE `cusparseSpSV` | 本算子 | 原因 |
| --- | --- | --- | --- |
| 数值更新 | 无 `updateMatrix` 接口，数值变化须重跑 analysis | 提供 `aclsparseSpSV_updateMatrix`，`GENERAL`/`DIAGONAL` 两档 | 任务书 2.3 的接口原型已声明；性能收益见 §5.1.3 的 Update 列 |
| 稀疏格式 | SpSV 仅接受 CSR / COO（SLICED_ELL 不在 SpSV 支持列表） | CSR / CSC / COO / SLICED_ELL 四格式 | 任务书 2.1-1 |
| 索引宽度 | 支持 32/64 位 | 仅 I32（Host 尺寸元数据仍为 `int64_t`） | 任务书 2.4「Device row/column/vector index 使用 I32」 |
| 奇异矩阵 | 行为未定义 | `NON_UNIT` 下缺失或为零的对角按 IEEE-754 自然传播 INF/NAN，不拦截、不报错 | 任务书 2.4 约束「按任务接口定义传播 INF/NAN，并在接口文档与测试报告中记录」 |
| 确定性 | 未承诺 | 同一输入重复执行逐位一致，且与格式、核数、求解路径无关 | 任务书 2.1-6 |

---

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```
op(A) · Y = alpha · X
```

逐行展开（`deps(i)` 为行 i 在目标三角内引用的已解行集合）：

```
            alpha·X[i] − Σ_{j∈deps(i)} op(A)[i,j]·Y[j]
Y[i] = ───────────────────────────────────────────────    （NON_UNIT）
                        op(A)[i,i]

Y[i] = alpha·X[i] − Σ_{j∈deps(i)} op(A)[i,j]·Y[j]         （UNIT）
```

遍历方向由 `(opA, fillMode, format)` 共同决定的**有效 fill mode** 给出：有效 fill 为
LOWER 时前代（行号升序），UPPER 时回代（行号降序）。有效 fill 的推导见 §3.2.1。
`fillMode` 未选中的非目标三角项按三角语义忽略；`UNIT` 时对角视为 1 且不读取存储的对角值。

### 3.1.2 支持数据类型

| matA values | vecX | vecY | computeType | 索引 |
| --- | --- | --- | --- | --- |
| `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | `ACL_SPARSE_INDEX_32I` |
| `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_SPARSE_INDEX_32I` |

`computeType` 与 A/X/Y 的 dtype 必须一致，不一致返回 `ACL_SPARSE_STATUS_INVALID_VALUE`。
index base 支持 `ACL_SPARSE_INDEX_BASE_ZERO` / `..._ONE`。

### 3.1.3 支持形状

方阵 `0 ≤ m ≤ INT32_MAX`、`0 ≤ nnz ≤ INT32_MAX`，`m = 0` 时要求 `nnz = 0`；
向量长度 ≥ m。`m`、`nnz`、稀疏度、空行数量与对角密度均为运行时入参，
由 Host tiling 与运行时分层支持，不要求编译期已知。SLICED_ELL 的
`sliceWidth × numSlices × m` 槽位总数在 Host 侧做溢出校验，越界返回错误码。

### 3.1.4 硬件能力实测

分核与分块参数依赖的平台量全部由 `platform_ascendc::PlatformAscendCManager` 在运行时读出，
不引用文档值、不写死常量。两款硬件的实测值：

| 项 | Atlas 800T A2（910B3） | Atlas A3 |
| --- | --- | --- |
| AIV / AIC | 40 / 20 | 48 / 24 |
| UB | 196352 B（192 KB） | 196352 B（192 KB） |
| L1 | 524032 B | 524032 B |
| L0A / L0C | 65536 B / 131072 B | 65536 B / 131072 B |
| L2 | 201326592 B（192.00 MiB） | 201326592 B（192.00 MiB） |

两款硬件的 UB 与 L2 相同，差别只在核数；由于本算子以串行行扫描为主，耗时**与核数无关**：
两款硬件的逐用例 solve 耗时实测**逐条相同**——206 条的逐条比值中位 **1.000**，三个 P 场景
相差 ≤ 0.03%，AIV 40 与 48 给出同样的耗时。两机实测的 AIV 数（40 / 48）在 Profiler 的
`Block Num` 上得到交叉验证，见 §5.1.5；同型号跨实例对照与一台已识别的降速实例见 §5.1.6。

arch22 还有两条与功能正确性直接相关的硬件行为，经板上探针确认，它们塑造了 §3.2.2 与
§3.2.4 的设计：

1. `GlobalTensor::SetValue` 的标量 GM 写停留在本核数据缓存，后续 kernel 的 MTE2 读取不到；
   必须逐缓存行回刷，且 `CacheLine::ENTIRE_DATA_CACHE` 档在该平台**无效**。
2. 回刷以**整条数据缓存行**为单位，两个核的输出若落进同一缓存行会互相覆盖。

## 3.2 核心设计决策

### 3.2.1 op(A) 的 CSR 行视图：四格式 × 三 op 归一

核心抽象是 **op(A) 的 CSR 行视图**。Host 把任意 `(format, opA, fillMode, base)` 归结为
一组 `(rowPtr, colInd, values, base, upper, conjugate)`，kernel 只认这一种形态。

关键的代数事实是：**CSC 的 `colPtr/rowInd` 就是 Aᵀ 的 CSR**。因此 CSC 与「CSR + 转置」
互为对方的透传形态，由此得到三种**零拷贝**组合：

| format | opA | 行视图 | 是否物化 |
| --- | --- | --- | --- |
| CSR | N | 用户数组直接当 op(A) 的 CSR | **否（透传）** |
| CSC | T | 用户数组直接当 op(A) 的 CSR | **否（透传）** |
| CSC | H | 同上，共轭在求解时对虚部取反，不改存储 | **否（透传）** |
| CSR | T / H | Aᵀ 的 CSR 需要重排 | 是 |
| CSC | N | 同上 | 是 |
| COO / SLICED_ELL | N / T / H | 无行指针 | 是（全部） |

有效 fill mode 由两次翻转得到（`EffectiveUpper()`）：A 的 `fillMode` 先被 CSC 的行列互换
翻转一次，再被「是否需要转置」翻转一次。共轭只在 complex64 且 `opA = H` 时置位，
`ACL_FLOAT` 下 H 等价 T。

这一步把「四格式 × 三 op × 两 fill × 两 diag × 两 base = 96 种组合」收敛成
kernel 侧的 `(upper, unit, conjugate, base, complexT)` 五个标量字段，求解核只有一套。

### 3.2.2 无原子、与核数无关的规范化：count → 对齐扫描 → 按输出行归属散射

需要物化时，在 workspace 里建一份 0 基规范 CSR，分三趟 kernel：

```
mode=0 count    ：blocks 核并行，每核负责 op(A) 的一段输出行，扫描全部输入槽位，
                  只统计落进自己行段的条目 → rowPtr[i+1] 处写入行长
mode=1 scan     ：单核前缀和，并在核边界把偏移向上对齐到 64 个条目
mode=1 scatter  ：blocks 核并行，同样的行归属划分，把条目写到自己行段的游标位置
```

三条设计约束，每一条都对应一个被实测排除的替代方案：

- **按输出行归属分核，而不是按输入槽位分核**。按输入分核需要原子操作来竞争行游标，
  且写入次序随调度变化；按输出行归属时每行只有一个核会写它，**不需要任何原子操作**，
  条目在行内的次序等于输入扫描次序，与核数和调度无关。代价是每个核都要扫一遍全部
  输入槽位（O(blocks × slots) 的读），但读是成片 DMA 且命中 L2，实测
  `spsv_build_c64_kernel` 只占 P 用例总耗时的 4.7%~4.9%（§5.1.5）。
- **核边界按 64 个条目对齐**。见 §3.1.4 的硬件行为 2：标量 GM 写按整条缓存行回刷，
  若两个核的输出区落在同一缓存行里会互相覆盖。因此扫描阶段把每个核的起始偏移向上
  对齐到 64 个条目（int32/FP32 为 256 B、complex64 为 512 B，均为整数条缓存行），
  `colInd`/`values` 各多预留 `64 × blocks` 个槽位。
  **对齐产生的空槽会落在本核最后一行的 `[rowPtr[i], rowPtr[i+1])` 区间内**
  （向上取整等于让该行吞掉这些槽位），而 `aclrtMalloc` 不清零，因此 scatter 结束时必须把
  它们写成**越界哨兵列号 `-1`**（`PadTail()`，不随 base 偏移），求解与分层两侧统一按
  `col < 0 || col >= m` 过滤——这顺带硬化了对非法索引的处理。也正因为空槽落在行区间内，
  `diagOnly` 的结构判定必须数**有效条目**而不是数行区间长度（见 §3.4.5）。
- **转置路径按输入行升序扫描**，于是输出行内天然按列升序，无需额外排序。

未排序与重复坐标由这一趟规范化统一处理：次序由扫描次序唯一确定，重复坐标按出现顺序
各自保留为独立条目（与三角语义一致，均参与累加）。透传路径不做规范化，用户数组的原始
次序即求解次序——两条路径的结果因此可能在**累加顺序**上不同，这正是 §5.3 的格式等价性
专项按「输出行内非对角累加项个数」分判据的原因。

### 3.2.3 三条求解路径与路径选择

Analysis 阶段下发 `spsv_level_kernel` 做依赖分层，把层号、层内行表与一个 8 字节调度摘要
（层数 `numLevels`、最大层宽 `maxLevelWidth`、是否纯对角 `diagOnly`）写进 workspace 头部，
回读后在 Host 侧选路径：

| 路径 | 触发条件 | 分核 | 适用形态 |
| --- | --- | --- | --- |
| **纯对角批量** `spsv_diag_solve_*` | `diagOnly = 1`（每行恰好一个有效条目且为对角元） | 按行区间分核，满核 | 对角阵、ILU(0) 的对角块 |
| **按层并行** `spsv_level_solve_*` | `0 < numLevels ≤ 1024` 且 `m / numLevels ≥ 32` | 每层一次 kernel，层内行按核号跨步 | 层宽足够的宽浅系统 |
| **单核串行** `spsv_solve_serial_*` | 其余（层数为 0 表示超过 1024 层的长依赖链） | 单核 | 窄深系统、长依赖链 |

- 门槛 `kSpsvA2MinAvgWidth = 32` 的含义是「平均层宽不足 32 行时，每层一次 kernel 下发的
  固定开销超过并行收益」。任务书 3.3 的三个 P 场景实测均为长依赖链形态，走单核串行路径
  （Profiler 里 `spsv_solve_serial_*` 占 95%，见 §5.1.5）。
- 层数上限 `kSpsvA2LevelCap = 1024` 同时是 `levelPtr` 数组的长度；超限时分层 kernel 提前
  退出并把 `numLevels` 置 0，直接退回串行路径，不浪费后续计算。
- 层间可见性由同一 stream 的**下发顺序**保证，不在核内做跨核同步。这是刻意的选择：
  arch22 没有可用的跨核栅栏原语，软件栅栏需要 GM 标志位 + 逐行 dcci，其开销与收益在
  本算子的层宽分布下不成立。

### 3.2.4 arch22 的跨 kernel 可见性：逐缓存行 dcci

规范化与分层这两趟 kernel 的产出（`rowPtr`、`rowLevel`、`levelPtr`、`levelRow`、调度摘要）
都是标量 GM 写，必须在每次写后显式回刷，否则下游 kernel 读到旧值：

```cpp
gm.SetValue(idx, v);
AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                  AscendC::DcciDst::CACHELINE_OUT>(gm[idx]);
```

`CacheLine::ENTIRE_DATA_CACHE` 档在该平台实测无效（写仍停留在本核缓存），只能逐行回刷。
这条约束连同 §3.2.2 的 64 条目对齐，是本设计里唯一两处平台相关的硬性约束，二者都收敛在
`sparse/spsv/arch22/` 内部，不外泄到公共 Host 逻辑。

## 3.3 host 侧设计

### 3.3.1 函数划分

```
ValidateHandle / ValidateOpA / ValidateTypes / ValidateMatrixShape
  / ValidateMatrixLayout / ValidateCommon / ValidateVector      → 参数校验
NeedsTranspose / NeedsMaterialize / EffectiveUpper / SellSlots   → 行视图语义推导
BuildPlan / LayoutWorkspace                                      → 执行计划与 workspace 布局
ResolveRowView                                                   → 透传 / 物化的地址解析
LaunchBuild / LaunchScan / LaunchLevel                           → Analysis 的三趟 + 分层
ReadSchedule                                                     → 回读 8 字节调度摘要并选路
ValidatePlanConsistency                                          → Analysis → Solve 的一致性校验
FillSolveTiling / DispatchSolve / LaunchLevelSolve               → Solve 下发
aclsparseSpSV_*                                                  → 六个公开接口，只做调度
```

四个阶段的职责边界：

| 阶段 | 职责 |
| --- | --- |
| `bufferSize` | 纯 Host 算术，不下发 kernel、不访问 Device。校验 handle/描述符/枚举/dtype/方阵/索引宽度/base，推导行视图语义，按 §3.3.4 的布局算出字节数 |
| `analysis` | 需要物化时下发 count → scan → scatter 三趟；随后下发分层 kernel；**同步一次 stream** 回读 8 字节调度摘要决定求解路径；把计划（`Plan`）与绑定快照（`Binding`）写进描述符 |
| `solve` | 校验 + 一致性核对后按计划下发一条（或按层多条）求解 kernel，**全异步，不同步、不回读** |
| `updateMatrix` | 透传路径换 values 指针（`DIAGONAL` 只换对角来源，置 `splitDiag`）；物化路径重跑一趟 values scatter（`mode=2`）或仅对角 scatter（`mode=3`），**不重做结构分析** |
| `destroyDescr` | 释放定长 Host 元数据；Device 存储由调用方持有，不在此释放 |

`bufferSize` 与 `analysis` 允许 `vecX`/`vecY` 描述符本身为 NULL（任务书 2.1-4）：
两阶段都不需要向量的任何属性，向量校验只在 `solve` 里做。

### 3.3.2 参数校验顺序与错误码

顺序本身是语义的一部分（`m = 0` 且 `nnz = 0` 必须返回 SUCCESS，而 `m = -1` 必须返回
INVALID_VALUE），`ValidateCommon()` 的固定顺序：

1. `handle == nullptr` → `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR`
2. `matA / spsvDescr / alpha` 逐个判空 → `INVALID_VALUE`
3. `opA` 不在 N/T/H 内 → `INVALID_VALUE`
4. `computeType` 不在 {`ACL_FLOAT`, `ACL_COMPLEX64`} 内、`alg != ACL_SPARSE_SPSV_ALG_DEFAULT` → `INVALID_VALUE`
5. 非方阵、`m < 0`、`nnz < 0`、`m = 0 且 nnz > 0`、越 `INT32_MAX` → `INVALID_VALUE`
6. 格式不在四格式内、索引宽度非 I32、base 非 0/1、SLICED_ELL 几何非法或槽位溢出 → `INVALID_VALUE`
7. `solve` 另加：向量描述符为空、values 为空、长度 < m、dtype 与 computeType 不一致 → `INVALID_VALUE`
8. `solve` / `updateMatrix` 另加：未 Analysis → `ACL_SPARSE_STATUS_NOT_INITIALIZED`

### 3.3.3 分核策略

遵循「优先满核」与「核数运行时获取」两条仓规：

| kernel | 核数 | 依据 |
| --- | --- | --- |
| `spsv_build_*`（count / scatter） | `min(GetAivCoreCount(), m)` | 按输出行归属分段，段长 `ceil(m / blocks)`，余数落在前几个核；count 与 scatter 必须用**同一个** `blocks`，否则行归属划分不一致 |
| `spsv_scan_kernel` | 1 | 前缀和是沿行号的串行递推；同时负责核边界的 64 条目对齐 |
| `spsv_level_kernel` | 1 | 层号计算是沿求解方向的串行递推 |
| `spsv_diag_solve_*` | `min(GetAivCoreCount(), m)` | 行间无依赖，按行区间均分 |
| `spsv_level_solve_*` | `min(blocks, 本层行数)` | 层内行按核号跨步分配 |
| `spsv_solve_serial_*` | 1 | 该路径的输入本就没有可并行的层宽 |
| `spsv_empty_*` | 满核 | `nnz = 0` 的退化解，纯逐元素缩放 |

核数取自 `GetAivCoreCount()`（`sparse/common/aclsparse_host_utils.h` →
`platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv()`），
无硬编码。两机 Profiler 的 `Block Num` 分别为 40 与 48，与各自实测 AIV 数一致（§5.1.5）。

### 3.3.4 数据分块和内存优化策略（workspace 布局）

`LayoutWorkspace()` 的布局，基址 512 B 对齐、子区 64 B 对齐：

| 子区 | 大小 | 用途 | 条件 |
| --- | --- | --- | --- |
| 调度摘要头 | 512 B | `numLevels` / `maxLevelWidth` / `diagOnly` | always |
| `rowLevel` | `m × 4 B` | 每行的层号 | always |
| `levelPtr` | `1026 × 4 B` | 层偏移表（`kSpsvA2LevelCap + 2`） | always |
| `levelRow` | `m × 4 B` | 按层聚合的行号表 | always |
| `rowPtr` | `(m+1) × 4 B` | 规范 CSR 行指针 | 物化时 |
| `colInd` | `(nnz + 64×blocks) × 4 B` | 规范 CSR 列号，含核间对齐填充 | 物化时 |
| `values` | `(nnz + 64×blocks) × parts × 4 B` | 规范 CSR 值，`parts` = 1（FP32）/ 2（complex64） | 物化时 |

透传路径只需前四项：P-01（CSR + N，FP32，m = 65536）实测 **0.50 MB**；
物化路径的 P-03（CSR + H，complex64，nnz = 3932160）实测 **48.0 MB**，为实测 L2 的 25.0%。

UB 侧的分块（全部为编译期常量，不随运行时核数变化，避免 UB 分配随平台抖动）：

- **解向量常驻 UB**：串行路径把解按 window 行分块常驻 UB，并保留前一块做 ping-pong。
  `kSpsvA2WindowF32 = 8192` 行、`kSpsvA2WindowC64 = 4608` 行。跨度不超过 window 的依赖
  全部命中 UB；更远的依赖走一个小滑动窗口，仍是成片 DMA，**不做逐元素 GM 标量读**。
- **`colInd` / `values` 成片搬运**：按 `kSpsvA2EntryTile = 2048` 条目的窗口搬入；
  回代方向的窗口向低地址延伸，使跨行复用成立。
- **行指针按 `kSpsvA2RowTile = 1024` 行搬入**，对角值按 `kSpsvA2DiagTile = 256` 搬入。
- **长行向量点积** 的单块长度 `kSpsvA2VecTile = 512`，触发门槛 `kSpsvA2VecRowMin = 256`。
- **纯对角批量路径** 按 `kSpsvA2DiagRowTile = 2048` 行一批，x / values / y 三块 UB。

### 3.3.5 tilingkey 规划策略

**不使用 tilingkey 表**，改为按 dtype 实例化模板：每类 kernel 有 FP32 与 complex64 两个
`__global__` 入口（`spsv_*_f32_kernel` / `spsv_*_c64_kernel`）。理由是 dtype 决定 UB 布局与
标量算术形态（complex64 的交错实虚部、Smith 缩放除法），分支化会让两条路径互相拖累
寄存器压力；而其余语义全部是**不改变数据布局的标量条件**。

因此其余所有变体（`base`、`upper`、`unit`、`conjugate`、`deviceAlpha`、`splitDiag`、
`window`、`entryTile`、`entries`）作为 `SpsvA2SolveTilingData` 的字段按 const 引用下发，
kernel 侧按值接收。规范化侧另有 `SpsvA2BuildTilingData.mode` 四档
（0 = count、1 = scatter、2 = 仅刷新 values、3 = 仅刷新对角 values），
使 `updateMatrix` 复用同一份 scatter 代码。

### 3.3.6 Analysis → Solve 的一致性校验

任务书 2.1-4 要求「从 Analysis 到 Solve 期间，相关 matA/vecX/vecY 描述符、参数和
externalBuffer 保持一致，参数变化后重新 Analysis」。本设计不止于文档约定，而是在
`solve` / `updateMatrix` 入口逐项核对 Analysis 当时固化的快照，不一致即返回错误码而非
给出错误结果：

| 核对项 | 比对对象 |
| --- | --- |
| handle、matA 描述符地址 | `Binding.handle` / `Binding.matA` |
| `opA`、`computeType` | `Plan.opA` / `Plan.complexT` |
| `m`、`nnz`、`format` | `Plan.m` / `Plan.nnz` / `Plan.format` |
| index base、diag type、有效 fill mode | `Plan.base` / `Plan.unit` / `Plan.upper` |
| 索引数组指针（`ptrs` / `idxs`） | `Binding.ptrs` / `Binding.idxs`（被重绑定即拒绝） |
| pointer mode | `Binding.pointerMode` |
| workspace | `Binding.workspace` |

## 3.4 kernel 侧设计

### 3.4.1 kernel 清单

| kernel | 核数 | 职责 |
| --- | --- | --- |
| `spsv_build_f32_kernel` / `spsv_build_c64_kernel` | blocks | 规范化：`mode` 0 = 计数、1 = 落盘、2 = 仅刷新 values、3 = 仅刷新对角 values |
| `spsv_scan_kernel` | 1 | `rowPtr` 原地前缀和，并在核边界把偏移向上对齐到 64 条目 |
| `spsv_level_kernel` | 1 | 层号 `rowLevel`、层内行表 `levelPtr`/`levelRow` 与 8 字节调度摘要；层数超 1024 提前退出 |
| `spsv_solve_serial_f32_kernel` / `..._c64_kernel` | 1 | 单核前代/回代，解向量 UB ping-pong |
| `spsv_level_solve_f32_kernel` / `..._c64_kernel` | ≤ blocks | 单层并行求解，每层一次下发 |
| `spsv_diag_solve_f32_kernel` / `..._c64_kernel` | ≤ blocks | 纯对角阵批量求解 |
| `spsv_empty_f32_kernel` / `..._c64_kernel` | 满核 | `nnz = 0` 的退化解 |

### 3.4.2 串行求解：解向量 UB ping-pong

单核路径的瓶颈是「解 Y 的随机读」：行 i 要读 `deps(i)` 对应的 Y 分量，而 `deps(i)` 的
列号分布由输入决定。逐元素 GM 标量读在 arch22 上每次都要等 MTE2，实测不可接受。

本设计把 Y 按 window 行分块常驻 UB，并保留**前一块**做 ping-pong：

```
前代（有效 fill = LOWER，行号升序）：
  当前块 [w0, w0+window)   ← 正在求解并写入
  前一块 [w0-window, w0)   ← 只读，覆盖跨度 ≤ window 的依赖
  更远的依赖               ← 小滑动窗口，成片 DMA 搬入
回代（有效 fill = UPPER，行号降序）：窗口向低地址延伸，方向对称
```

实际稀疏矩阵（ILU 预条件子、带状系统）的依赖跨度绝大多数远小于 window，
因此命中率很高；即使未命中也走成片 DMA，不退化为逐元素标量读。
`colInd` / `values` 同样按 2048 条目的窗口搬入，回代方向的窗口向低地址延伸使跨行复用成立。

### 3.4.3 按层并行求解

`levelPtr`/`levelRow` 给出每层的行号列表。每层下发一次 kernel，层内行按核号跨步分配
（`for (r = levelPtr[L] + blockIdx; r < levelPtr[L+1]; r += blocks)`）。
层内无依赖，因此无需核间同步；层间可见性由同一 stream 的下发顺序保证。

每层一次下发的固定开销约为微秒量级，这是 `m / numLevels ≥ 32` 门槛的来源：
层宽不足 32 行时，下发开销超过并行收益，退回单核串行反而更快。

### 3.4.4 列连续长行的向量点积

**问题**：`skewed` / `power_law` / `one_long_row` 这类形态把绝大部分非零元集中在少数几行
（实测单行最多 6399 个元素）。按层求解时这一行由单个核逐元素标量累加，成为整个求解的
串行瓶颈——层宽再大也无济于事，因为瓶颈在单行内部而不在层间。

**解法**：检测「列连续」的长行并改走向量点积（`AccumulateContiguous()`）：

```
触发条件（全部满足才走向量路径，否则退回标量路径）：
  1. 行长 n ≥ kSpsvA2VecRowMin (256)
  2. colInd[s] .. colInd[e-1] 严格连续：colInd[e-1] - colInd[s] == n - 1
  3. 列号不越界：0 ≤ colInd[s] 且 colInd[e-1] < m
  4. 对角元落在区间端点：UPPER 时在区间首，LOWER 时在区间尾
```

满足时，Y 的对应区间可以**整段搬进 UB**，无需逐元素 gather、无需掩码，归约用
`Mul` + `ReduceSum` 按 `kSpsvA2VecTile = 512` 分块完成。complex64 在交错布局下用两张
常量符号表处理「实部点积要把虚部乘积取负」与「虚部点积要把实虚部对调」，表只在 Init 后建一次。

**效果**：同一形态相对标量路径，910B3 上实测快 7.8–20.3 倍（`one_long_row` 形态
3742.6 → 184.5 µs），这类形态在两款硬件上均达标。

设计上的取舍：条件 2–4 把适用面限制在「列连续且对角在端点」的长行，而不是对任意长行
做 gather。理由是 gather 路径需要逐元素的索引计算与掩码，在 arch22 上的收益远小于
整段 DMA；而实测中的长行形态（带状、幂律尾部）天然满足列连续。不满足时**无损退回**
标量路径，不改变数值结果。

### 3.4.5 纯对角批量路径

纯对角阵是另一个极端：完全没有行间依赖，本该是最快的输入，但走按层路径反而最慢——
每层下发一次 kernel、每行一次 DMA 与同步，而每行只有 1 个元素，耗时几乎全是固定开销。

Analysis 侧由分层 kernel 顺带判定 `diagOnly`（每行恰好一个**有效条目**且为对角元），
命中时 Solve 改走批量路径：按行区间分核，一次搬入整段 X 与对角值、在 UB 里逐行标量求解、
一次写回整段 Y。标量算术与按层/串行路径**完全相同**，结果逐位一致。

相对走按层路径，`diagonal` 家族 25 条在 910B3 上实测快 **4.7 倍**（中位；最大 12.7 倍，
m = 7168 的用例 105 µs → 14 µs）。

### 3.4.6 complex64 的除法与数值稳定

复数除法 `(ar + i·ai) / (dr + i·di)` 若直接按 `(dr² + di²)` 计算分母，在 `|d|` 接近
FP32 上界时中间量上溢、结果变成 NaN，而输入本身是有限的。本设计采用 **Smith 缩放**
（`sparse/spsv/arch22/spsv_math.h`），按 `|dr| ≥ |di|` 选分支：

```
|dr| ≥ |di| :  r = di/dr,  den = dr + di·r,  Y = ((ar + ai·r) + i·(ai − ar·r)) / den
|dr| <  |di| :  r = dr/di,  den = dr·r + di,  Y = ((ar·r + ai) + i·(ai·r − ar)) / den
```

两条分支的中间量都被缩放到 O(1)，有限输入不会产出 NaN。测试侧的同序 FP32 参考实现使用
**同一套** Smith 缩放，因此 NPU 与它的差异只反映舍入而非算法差异（§5.1.1）。

### 3.4.7 确定性（bitwise deterministic）

任务书 2.1-6 要求「以确定的规范化遍历与求解顺序获得 bitwise deterministic 结果」。
本设计的确定性由四条共同保证，每一条都可单独检验：

1. **同一行只由一个核写出**（规范化按输出行归属、求解按行分配），无写竞争；
2. **行内求和严格按存储顺序**，不使用浮点原子累加、不做树形归约重排；
3. **三条求解路径对同一行的算术序列相同**——纯对角与按层/串行路径共用同一段标量算术，
   长行向量点积只在列连续时启用且分块顺序固定；
4. **规范化的条目次序等于输入扫描次序**，与核数、调度无关（§3.2.2）。

因此结果与路径选择、核数、调度均无关。实测：每个用例连跑两次 solve 并逐字节比对，
全部一致（§5.1.2）。

需要显式说明的边界：**跨格式**不保证逐位一致。透传路径按用户数组的原始次序累加，
物化路径按规范化后的次序累加，当输出行内有 ≥ 2 个非对角累加项时两者的浮点累加顺序
可能不同。§5.3 的格式等价性专项因此按「输出行内非对角累加项个数」分成两档判据，
并以实测的单行最大非对角项数作为「可断言逐位一致」的前提。

## 3.5 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas 800I A2 推理产品 | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | - |
| Atlas 推理系列产品 | - |

A2 与 A3 同属 arch22 / DAV_2201，共用同一套源码：`--soc=ascend910b` 与
`--soc=ascend910_93` 经根 `CMakeLists.txt` 的 `get_soc_arch_dirs` 映射到同一个 arch22
目录。`build.sh` 的 SOC 短名按**前缀**匹配，因此 `ascend910b3` 与 `ascend910b4` 归一到
同一个 `ascend910b`、产出同一份二进制，即 910B3 与 910B4 共用同一套代码路径与同一份产物。

无硬编码核数，全部由 `GetAivCoreCount()` 运行时获取（§3.3.3），两机 Profiler 的
`Block Num` 40 / 48 交叉验证了这一点。arch35 路径保持不变：`sparse/CMakeLists.txt`
先 `REMOVE_ITEM` 掉全部 `/arch22/` 源码再只加回 `SOC_ARCH_DIRS`，因此 arch22 不会被
编进 ascend950 产物，构建日志中可见 `Filtering out arch22 sources`。

## 3.6 算子约束限制

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | 见 §3.3.2；全部在 Host 侧校验并返回确定错误码 |
| 索引宽度 | 仅 `ACL_SPARSE_INDEX_32I`；`m`、`nnz` ≤ `INT32_MAX`。Host 尺寸元数据仍用接口声明的 `int64_t` |
| 形状 | 二维三角方阵；`m = 0` 时 `nnz` 必须为 0；向量长度 ≥ m |
| 三角语义 | `fillMode` 之外的非目标三角项按三角语义忽略；`UNIT` 时对角视为 1 且不读取存储的对角值 |
| 奇异矩阵 | `NON_UNIT` 下缺失或为零的对角按 IEEE-754 自然传播 INF/NAN，不拦截、不报错 |
| 同步语义 | **Analysis 会同步一次 handle 的 stream** 以回读调度摘要；`bufferSize` 纯 Host；`solve` 与 `updateMatrix` 全异步 |
| 生命周期 | Analysis 到 Solve 期间 matA 的索引数组与 `externalBuffer` 必须保持有效且不被外部修改；参数变化须重新 Analysis（逐项核对见 §3.3.6） |
| `updateMatrix` | `DIAGONAL` 接受与 `GENERAL` 相同的 nnz 长数组，只取 `row == col` 的条目；不改变 pattern，不重做结构分析 |
| 原地与别名 | Y 可与 X 使用同一 Device values 指针（原地求解）；A、X 保持只读 |
| broadcast / dynamic shape | 不涉及 / 不要求（m、nnz 为运行时入参） |
| 确定性 | 同一输入重复执行逐位一致；**跨格式**仅在输出行内 ≤ 1 个非对角累加项时逐位一致，否则在精度判据内一致（§3.4.7） |
| 空 Tensor | `m = 0` 为合法 no-op；`nnz = 0` 走 `spsv_empty_*` 退化解 |

---

# 4 特性交叉分析

| 特性 | 与本算子的交叉 |
| --- | --- |
| 四格式 × N/T/H | 归一为「op(A) 的 CSR 行视图」；CSR+N / CSC+T / CSC+H 三组透传零拷贝，其余物化（§3.2.1） |
| `fillMode` × format × op | 有效 fill 由两次翻转推导（CSC 行列互换、是否转置），只影响遍历方向 |
| `diagType = UNIT` | 求解器内 2 处分支：跳过对角倒数缩放，且**不读取** A 的对角位置 |
| `opA = H` × dtype | complex64 下取共轭（读取 values 时虚部取反，不改存储）；`ACL_FLOAT` 下 H 等价 T |
| index base 0/1 | 作为 tiling 标量字段，读 `colInd` 时统一减 base；透传路径不改写用户数组 |
| pointer mode HOST/DEVICE | HOST 时 alpha 的实虚部写进 tiling；DEVICE 时把 alpha 的 Device 地址传给 kernel，由 kernel 搬入 |
| 未排序 / 重复坐标 | 规范化一趟统一处理；次序由扫描次序唯一确定，重复坐标按出现顺序各自保留并参与累加 |
| 空行 | 空行的层号为 0（无依赖），`NON_UNIT` 下对角缺失 ⇒ INF/NAN 传播；`UNIT` 下解为 `alpha·X[i]` |
| 纯对角阵 | `diagOnly` 批量路径，满核并行（§3.4.5） |
| 长依赖链（层宽 1） | 单核串行路径；任务书 3.3 的三个 P 场景均属此类 |
| 超长行 | 列连续时走向量点积，否则无损退回标量路径（§3.4.4） |
| 原地求解（X/Y 同址） | 求解按行推进且只写已读过的行，原地安全；UB ping-pong 不改变这一性质 |
| `updateMatrix` × 透传/物化 | 透传换指针（`DIAGONAL` 置 `splitDiag` 走独立对角数组）；物化重跑 values scatter（`mode` 2/3） |
| INF/NAN 输入 | 正常传播，不拦截；测试按「与同序 FP32 参考在数个 ULP 内一致且确实产生非有限值」判定 |
| 大幅值输入（1e20） | complex64 除法用 Smith 缩放，有限输入不产出 NaN（§3.4.6） |
| 核数差异（40 vs 48） | 分核数运行时获取；结果与核数无关（§3.4.7），两机逐用例结果一致 |

---

# 5 可维可测分析

## 5.1 精度标准 / 性能标准 / 内存标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FP32 用 float64 Golden、complex64 用 complex128 Golden，`rtol = 2⁻¹⁰`、`atol = 2⁻¹⁶`、`A = 1e-2`；逐元素 `\|actual − golden\| ≤ atol + rtol·\|golden\|` 的匹配率 ≥ 0.99，且每元素绝对误差 ≤ `max(A, 32 × ULP_fp32(golden))`；complex64 实部虚部分别适用全部规则 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)，任务书 3.2 |
| 性能标准 | 倍率 = 标杆接口 GPU 设备 Event 调用耗时 / NPU 同调用范围耗时，每个场景 ≥ **0.25×**；每个 case×dtype 预热 10 次、采样 30 次，报告中位数、P90、workspace、Analysis 与 Solve 时间 | 任务书 3.3 |
| 内存标准 | 两条满足其一。本算子适用第 2 条：**方案固有 workspace 绝对值不超过目标硬件 L2 Cache 容量**（任务书 3.4 明文：「无等价 GPU 接口的 case 按 workspace 不超过目标硬件 L2 Cache 容量验收」） | 任务书 3.4 |

内存口径的选择依据（为什么不走第 1 条的 GPU/NPU allocator 峰值对比）：任务包
`operator_adapter.py` 写明「PyTorch 没有与 SpSV analysis/update/solve 生命周期等价的
公开稀疏接口」，GPU 侧默认把这些 case 写入 `skipped_cases`；已采集的
`baseline_results/gpu_full_results.tsv` 是原生 cuSPARSE 口径，列里只有 `workspace_bytes`
与 `input_output_bytes`，没有 allocator 峰值列。另外输入输出总量最大约 50 MB，
也未触及第 1 条的 500 MB 前置条件。

### 5.1.1 精度判据的补充口径

1. **同序 FP32 参考**。除 FP64 Golden 外，另设一条与 kernel **完全同序**的 CPU FP32 实现
   （同样按存储顺序累加、同样的 Smith 缩放复数除法）。它的作用是把「实现误差」与
   「用例自身的条件数问题」分开：三角求解是长链规约，病态输入下 FP64 Golden 与任何
   FP32 实现都会有可观差异，此时唯一有意义的问题是「NPU 是否在做正确的 FP32 算术」。
   实测 NPU 与同序 FP32 参考的相对差 ≤ 2e-7（1–2 ULP）。
2. **奇异矩阵**。`NON_UNIT` 下缺失或为零的对角会传播 INF/NAN，与 FP64 Golden 比数值
   没有意义。这类用例改判「与同序 FP32 参考在数个 ULP 内一致**且确实产生非有限值**」——
   后半句是必要的，否则一个「悄悄把 0 对角当 1」的实现也能通过。
3. **绝对误差门** 取 `max(A, 32 × ULP_fp32(golden))`，即任务书 3.2-3 的原文口径，
   其中 ULP 按 **FP32** 的 ULP 计算（golden 虽为 FP64，但被比较的量是 FP32 结果）。

### 5.1.2 精度实测结果（910B3 / A3，CANN 9.1.0）

| 测试集 | 910B3 | A3 |
| --- | --- | --- |
| 仓内 C++ UT/ST（`build.sh --ops=spsv --soc=... --run`） | **304 / 304 通过** | **304 / 304 通过** |
| └ 参数化功能与精度 `Matrix/SpSVArch22.MatchesGolden/*` | 286 / 286 | 286 / 286 |
| └ 接口契约与异常 `SpSVArch22Param.*` | 15 / 15 | 15 / 15 |
| └ 格式转换/规范化专项 `SpSVArch22Normalize.*` | 3 / 3 | 3 / 3 |
| 独立验证程序（规模扫描与专项，286 个配置） | **286 / 286 通过** | **286 / 286 通过** |
| 门禁构建 `build.sh --soc=ascend910b --pkg` | **RC = 0** | — |
| 门禁构建 `build.sh --soc=ascend950 --pkg`（arch35 回归，在 Ascend950PR 上） | **RC = 0** | — |

两款硬件在同一版本上用例集一致、条数一致、全部通过，GTest 进程退出码均为 0。
**重复执行逐位一致**：每个用例连跑两次 solve 并逐字节比对，全部一致。
**无失败项。**

### 5.1.3 性能实测结果

口径：NPU 侧 `aclrtEvent` 计时，窗口内只有 `aclsparseSpSV_solve`，`updateMatrix` 与
values 恢复在窗口之外——这与官方 `common/benchmark_runner.py` 一致（该脚本在 Event 窗口
外调 `reset()`，窗口内只调 `invoke()`）。预热 10 次、采样 30 次。标杆 `median_us` 取自
任务包 `baseline_results/gpu_full_results.tsv`，即任务书 3.3 性能表直接引用的那组数。

**任务书 3.3 的三个达标场景**（门槛 0.25×）：

| 用例 | m / nnz / dtype | 910B3 median | **910B3 倍率** | A3 median | **A3 倍率** |
| --- | --- | --- | --- | --- | --- |
| P-01 base0 | 65536 / 524288 / FP32 | 34744.6 µs | **14.80×** | 34737.6 µs | **14.80×** |
| P-01 base1 | 65536 / 524288 / FP32 | 34726.2 µs | **14.81×** | 34723.3 µs | **14.81×** |
| P-02 base0 | 131072 / 1572864 / c64 | 122722.4 µs | **10.03×** | 122717.5 µs | **10.03×** |
| P-02 base1 | 131072 / 1572864 / c64 | 122720.8 µs | **10.01×** | 122717.8 µs | **10.01×** |
| P-03 base0 | 262144 / 3932160 / c64 | 309899.8 µs | **9.93×** | 309852.7 µs | **9.93×** |
| P-03 base1 | 262144 / 3932160 / c64 | 309895.2 µs | **9.93×** | 309851.5 µs | **9.93×** |

**`performance_cases.json` 全量 206 条**（6 条 P + 200 条参考）：

| 机器 | 达标 | 中位倍率 | 最差倍率 |
| --- | --- | --- | --- |
| 910B3 | **206 / 206** | **11.97×** | **4.03×**（`spsv-extra-184`，`power_law`） |
| A3 | **206 / 206** | **13.33×** | **4.03×**（`spsv-extra-184`，`power_law`） |

两款硬件最紧的用例是同一条（`spsv-extra-184`），对 0.25× 门槛仍有 **16.1 倍**余量。

**Analysis 与 Update 时间**（任务书 3.3-2 要求单独报告）：

| 用例 | 910B3 Analysis | 910B3 Update | A3 Analysis | A3 Update |
| --- | --- | --- | --- | --- |
| P-01 base0 / base1 | 809.0 / 300.3 µs | 1.7 / 1.6 µs | 680.0 / 327.0 µs | 3.0 / 3.1 µs |
| P-02 base0 / base1 | 93010.5 / 92931.0 µs | 45529.6 / 45530.0 µs | 90556.2 / 90087.5 µs | 43819.7 / 43814.3 µs |
| P-03 base0 / base1 | 226433.2 / 226861.8 µs | 111440.5 / 111370.9 µs | 218721.7 / 218446.0 µs | 107276.4 / 107307.5 µs |

读法：P-01 是 FP32 且 CSR+N，归一化走**透传**，Analysis 只做分层，因此只有几百微秒；
P-02/P-03 是 complex64 的 T/H，Analysis 含一次完整的 count/scan/scatter **物化**，
`updateMatrix` 复用同一趟 scatter 只重写 values，约为 Analysis 的一半——这正是
`updateMatrix` 接口存在的价值（cuSPARSE 无此接口，数值变化必须重跑 analysis）。
两者都是一次性成本，按任务书口径不计入 solve 的计时窗口。

### 5.1.4 内存 / workspace 实测

| 用例 | 路径 | workspace | 占实测 L2（192.00 MiB） |
| --- | --- | --- | --- |
| P-01（CSR + N，FP32） | 透传 | 528960 B = 0.50 MB | 0.26% |
| P-02（CSR + T，complex64） | 物化 | 20482688 B = 19.5 MB | 10.2% |
| P-03（CSR + H，complex64） | 物化 | 50367104 B = 48.0 MB | **25.0%** |

满足任务书 3.4 第 2 条：最大 48.0 MB，为实测 L2 容量 192.00 MiB 的 25.0%。
算子内零 `aclrtMalloc`，workspace 全部由调用方按 `bufferSize` 的返回值分配；
除描述符里的定长元数据外，Host 侧不保存与输入规模同量级的中间结果。

### 5.1.5 Profiler 证据

`msprof --ai-core=on --aicpu=on --application="spsv_check <dev> perf"`，采集 P 用例全程，
两款硬件各一次。910B3 侧：

| OP Type | Core Type | Count | Total Time(µs) | Ratio |
| --- | --- | --- | --- | --- |
| `spsv_solve_serial_c64_kernel` | AI_VECTOR_CORE | 52 | 11248173 | 87.93% |
| `spsv_solve_serial_f32_kernel` | AI_VECTOR_CORE | 26 | 903408 | 7.06% |
| `spsv_build_c64_kernel` | AI_VECTOR_CORE | 8 | 623991 | 4.88% |
| `spsv_scan_kernel` | AI_VECTOR_CORE | 4 | 13880 | 0.11% |
| `spsv_level_kernel` | AI_VECTOR_CORE | 6 | 2420 | 0.02% |

三条可直接核验的结论（A3 侧 kernel 组成与 task 数逐项一致，占比 88.09% / 7.07% / 4.71% / 0.11% / 0.02%；
两机的 `spsv_solve_serial_c64_kernel` 总耗时分别为 11248173 µs 与 11247508 µs，相差 0.006%）：

1. **96 个 task 全部为 AI_VECTOR_CORE，AI_CPU 为 0** ⇒ 主求解与格式规范化都在 NPU 执行，
   无 CPU fallback（任务书 2.1-8）。
2. **96 个 task 全在同一条 stream（ID 47）上** ⇒ 即调用方 handle 的 stream，
   算子内不自建 stream，顺序即下发顺序（任务书 2.1-7）。
3. 同一个 `spsv_build_c64_kernel` 在 **910B3 上 `Block Num = 40`、A3 上 `Block Num = 48`**，
   分别等于两机实测的 AIV 数 ⇒ 分核数是运行时取的、不是写死的常量。
   单机数据只能说明数值相符，**两机对照**才能排除「恰好写死成 40」。

### 5.1.6 同型号跨实例对照，与一台已识别的降速实例

§5.1.3 的 A3 数据取自 2026-10-19 的实测。更早一轮（2026-10-17）在另一台 A3 实例上
测得的逐用例 solve 耗时是本表的 **4.478 倍**，当时据此写下过「A3 的逐用例耗时约为
910B3 的 4.4 倍」并解释为架构差异。该解释**已被证伪**，本节给出判据，并保留原始记录。

用两台全新的 910B3 实例立基线，五份数据同源码（`tree 224e72f5…`）、同口径
（窗口只含 `solve`，预热 10 / 采样 30）：

| 数据集 | 物理卡 | P-01-base0 median | 逐条比值中位（对历史 A2） |
| --- | --- | --- | --- |
| 910B3 历史 | NPU 2，Bus `0000:81:00.0` | 34744.6 µs | 1.000（基准） |
| 910B3 复测 ① | NPU 4，Bus `0000:01:00.0` | 34746.1 µs | **1.000** |
| 910B3 复测 ② | NPU 6，Bus `0000:41:00.0` | 34744.0 µs | **1.000** |
| A3（本表所用） | Phy-ID 12/13 双 die | 34737.6 µs | **1.000** |
| A3 早先一轮 | Phy-ID 6 单 die | **155487.9 µs** | **4.366** |

四份互相吻合到 **0.006%**（三台 910B3 是三块不同物理卡，VDie ID 与 Bus-Id 各不相同），
只有早先那一轮 A3 整齐偏慢。四条判据：

1. **偏慢是均匀的，且覆盖全部器件侧计时**：solve 4.478×（P 用例 min 4.476 / max 4.478）、
   analysis 4.03×、update 4.04×。一个相同因子同时压住串行核（`Block = 1`）与
   并行核（`Block = 48`），核数、算法与计时口径都无法产生这种形态。
2. **方法学可以排除**：早先那轮的 910B3 与 A3 记录，`platform.log` 时间戳为
   `2026-10-17T14:26:35Z` 与 `14:26:37Z`——同一批、相差两秒。同一份源码、同一套脚本、
   同一天，910B3 正常而 A3 偏慢 4.478 倍，差异只能落在机器实例上。
3. **硬件类别可以排除**：本表 A3 数据的 `workspace_bytes` 与早先那轮 **206 / 206 全同**、
   与 910B3 有 **137 / 206 不同**——workspace 随 AIV 数（48 vs 40）排布，
   这证明本表确系 A3 实机数据。
4. **核数相同**：两轮 A3 的 AIV 均为 48（由 Profiler 的 `spsv_build_c64_kernel`
   `Block Num = 48` 实测），不是拿到了不同规格的 A3。

由此，§3 中关于平台差异的表述改为：本算子以串行行扫描为主，耗时**与核数无关**，
910B3 与 A3 的逐用例 solve 耗时实测逐条相同。「单核速度才是决定项」这一判断成立，
但当初据一台实例写下的「4.4 倍」数值不成立。

**限定：**

1. **降速的直接成因未实测。** 均匀因子同时作用于串行核与并行核是频率差的典型形态，
   但**没有量到频率**：容器内 `npu-smi info -t common` 不可用、`-t usages` 不含频率项，
   该实例已释放无法回查。本节只断言「该实例器件侧整体偏慢 4.0 ~ 4.5 倍」这一实测事实，
   **不断言成因**；引用时不得把「频率」写成已测结论。
2. **`gtest` 墙钟不可用于此判断。** 304 条 UT/ST 的总墙钟为早先那轮 5139 ms、
   本轮 14314 ms（本轮更慢），该墙钟由主机侧（算 golden、逐条建 handle）主导，
   不是器件侧计时。
3. **小用例倍率有抖动。** 逐条比值的 min / max 可到 0.50 / 1.78，来自 13 µs 量级的用例；
   判实例差异只看 34 ms ~ 310 ms 量级的 P 场景。三台 910B3 的「全量中位倍率」在
   11.97× ~ 13.00× 之间浮动 8%，浮动全部来自小用例，P 用例逐条只差 0.006%。
4. **两轮 A3 的原始记录均保留**，未以新数覆盖旧数。

## 5.2 兼容性分析

- **公开接口零改动**：六个接口的原型与 `include/cann_ops_sparse.h` 的现有声明逐字一致，
  不新增、不修改签名，因此对已合入的 arch35 调用方零影响。
- **源码目录隔离**：arch22 与 arch35 分别位于各自目录，由 `sparse/CMakeLists.txt` 的
  `SOC_ARCH_DIRS` 过滤，不会同时编译。已在 Ascend950PR 上实跑门禁命令
  `bash build.sh --soc=ascend950 --pkg` 得到 RC = 0，构建日志确认 arch22 源码被
  `Filtering out arch22 sources` 剔除、未参与编译。
- **公共 Host 逻辑解耦**：平台相关部分（逐行 dcci、64 条目对齐、UB 分块常量）全部收敛在
  `sparse/spsv/arch22/` 内；描述符语义与校验口径与 arch35 保持一致，
  可与 A5/950 路径共同维护（任务书 3.5 末条）。
- **产品支持表同步**：`sparse/spsv/README.md` 把 Atlas A2/A3 由「不支持」改为支持，
  并补充新增的 complex64、四格式与约束说明。

## 5.3 测试设计

测试工程位于 `test/spsv/arch22/`，三类入口：

| 文件 / 入口 | 项数 | 覆盖 |
| --- | --- | --- |
| `Matrix/SpSVArch22.MatchesGolden/*`（参数化功能与精度） | 286 | 四格式 × N/T/H × LOWER/UPPER × UNIT/NON_UNIT × FP32/complex64 × base 0/1 全组合，叠加 m = 1…30000 规模扫描与专项形态 |
| `SpSVArch22Param.*`（接口契约与异常） | 15 | 空指针、非法枚举/alg、dtype 与 computeType 不匹配、非方阵、未 Analysis 即 Solve、Analysis 后改参数、workspace 为空、异步完成后释放 workspace |
| `SpSVArch22Normalize.*`（格式转换与规范化专项） | 3 | 同一三角系统按四格式表达（COO 另打乱坐标），归一化后结果一致 |
| `spsv_a2_golden.h` | — | FP64/complex128 Golden（按行稀疏表组织，使 m = 30000 的用例不需要稠密 m×m 内存）+ 同序 FP32 参考 + 矩阵生成器 |

**数据生成规则**（任务书 3.5）：A/X 的每个 values 按 **70% `[-1,1]` 均匀 / 20% 正态
`N(0,1)` / 10% 规格允许的特殊值**逐值混合，特殊值档内「显式存储的 0」与 `±1` 单位幅值
等概率。实测抽样 37654 个非对角分量，恰为 0 的占 3.45%（期望 1/3 × 10% = 3.33%）。
`NON_UNIT` 对角在叠加 ±2 偏移后幅值恒 ≥ 1（实测越界个数 0），因此引入特殊值不会把正常
用例变成奇异矩阵——奇异由 `zeroDiag` / `singular` 专项单独构造。pattern 另覆盖空行、
长尾、重复与未排序坐标。

**格式等价性专项的分档判据**（这是 §3.4.7 的那条边界的直接检验）：

| 用例 | 矩阵 | 判据 | 成立条件 |
| --- | --- | --- | --- |
| `BitwiseBidiagAcrossFormats` | 双对角 m = 96，N/T/H × LOWER/UPPER × FP32/complex64 | **四格式逐位一致** | A 与 Aᵀ 的每行都最多一个非对角项（实测 maxOffPerRow = maxOffPerCol = 1） |
| `BitwiseNoTransAcrossFormats` | 随机 m = 96 / avgRow = 2，op = N | **四格式逐位一致** | 输出行即 A 的行，最多一个非对角项（实测 maxOffPerRow = 1） |
| `ToleranceAcrossFormats` | 随机 op = T/H，以及 m = 1024 / avgRow = 4 | 四格式在**精度判据内**一致 | op = T/H 的输出行汇聚 A 的一整列，实测单行最多 6 个（m = 96）/ 24 个（m = 1024）非对角项 |

**覆盖面对应任务书 3.5 的必测清单**：基础功能（四格式、LOWER/UPPER、UNIT/NON_UNIT、
FP32/complex64）、操作语义（N/T/H、complex64 共轭与正负抵消）、生命周期（Create、
NULL vecX/vecY 的 BufferSize/Analysis、Solve、GENERAL/DIAGONAL UpdateMatrix、Destroy、
Analysis→Solve 一致性、异步完成后释放 buffer）、边界与异常（m/nnz 为 0/1、未排序坐标、
空行、NON_UNIT 缺失或零对角的 INF/NAN 传播、索引/维度/dtype/workspace 异常）、
原地与指针（X/Y 同址、Host/Device pointer mode、重复调用、描述符变更、过早释放 buffer）。

性能与内存由 `test_cases/aclsparseSpSV_testCase/npu/` 的 C++ 程序采集，与任务包共用同一份
`performance_cases.json`、同样的矩阵生成规则、同样的 Event 窗口与预热/采样次数。

## 5.4 可测性

- 参数校验分支全部可由 `SpSVArch22Param.*` 或独立验证程序触达。
- Host 侧 `OP_LOGD` 打印执行计划的全部关键量（`m/nnz/format/opA/base/upper/unit/
  conjugate/transposed/materialize/blocks/numLevels/maxLevelWidth/useLevels/diagOnly/
  totalBytes`），现场可据此判断走的是哪条路径、为什么。
- 三条求解路径可由输入形态确定性地触发（纯对角阵、宽浅系统、长依赖链），
  便于逐路径回归；三者结果逐位一致，因此路径切换本身也是可检验的。
- 性能采集以 `msprof` 的 `op_summary_*.csv`（`Task Duration(us)`）为准，同时保留
  `op_statistic_*.csv` 用于核类型与调用流取证。
- 复现只需两条命令（见 `test/spsv/README.md` 与 `test_cases/aclsparseSpSV_testCase/` 的 README），
  脚本内部自行 `source set_env.sh` 并完成构建。

---

# 6 交付与合入

| 交付物 | 位置 |
| --- | --- |
| 接口声明 | `include/cann_ops_sparse.h`（原型已存在，本次**无改动**） |
| 算子实现 | `sparse/spsv/arch22/`：`spsv.h`、`spsv_tiling_data.h`、`spsv_math.h`、`spsv_kernel.h`、`spsv_host.cpp`、`spsv_build_kernel.cpp`、`spsv_kernel.cpp`、`spsv_level_kernel.cpp` |
| 算子文档 | `sparse/spsv/README.md`（产品支持表标注 A2/A3 支持、四格式与 complex64、约束与限制、复现说明） |
| 测试代码 | `test/spsv/arch22/`：`spsv_a2_golden.h`、`spsv_test.cpp`；`test/spsv/README.md`（目录结构、环境、编译运行、逐测试判据表、精度判定口径） |
| 专项测试与性能脚本 | `test_cases/aclsparseSpSV_testCase/`（含 `npu/` 的精度与性能程序、复现 README），公共运行组件 `test_cases/common/`，标杆结果 `test_cases/baseline_results/` |
| 自测报告与原始证据 | 环境实测、用例参数与输出结构、精度逐条结果、性能倍率与逐 case 原始数据、workspace、Profiler 原始导出、失败项与偏差说明；同一份材料内的表格取自同一批结果 |

合入路径：代码、测试与文档以 PR 提交至 `ops-sparse` 的 `master` 分支。

---

# 7 参考资料

1. Ascend C 算子开发文档、算子开发接口文档：<https://www.hiascend.com/document>
2. ops-sparse 开源仓：<https://gitcode.com/cann/ops-sparse>
3. 生态算子开源精度标准（experimental_standard）：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md>
4. cuSPARSE 官方文档（Generic API，`cusparseSpSV`）
