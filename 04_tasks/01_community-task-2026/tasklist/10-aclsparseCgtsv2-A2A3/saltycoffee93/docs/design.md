# aclsparseCgtsv2 算子设计文档

> 任务：《10月社区任务-aclsparseCgtsv2算子开发（A2/A3）》；目标代码仓 `cann/ops-sparse`（master）；本文提交仓 `cann/cann-ops-competitions`。
> 文档结构按官方 `04_tasks/01_community-task-2026/resources/design_template.md` 六段模板；接口、精度、性能、内存、交付条款全部可溯源到任务书 §编号。

# 需求背景（required）

## 需求来源

- 任务名称：10月社区任务-aclsparseCgtsv2算子开发（A2/A3），适用平台 Atlas A2/A3（DAV_2201，`arch22`）。
- 合入目标仓与目录：`cann/ops-sparse` `master` 分支，新增 `include/cann_ops_sparse.h` 声明、`sparse/cgtsv2/`、`sparse/cgtsv2/arch22/`、`test/cgtsv2/`、`test/cgtsv2/arch22/`（任务书 §5）。
- 设计文档提交位置：`cann/cann-ops-competitions` → `04_tasks/01_community-task-2026/tasklist/10-aclsparseCgtsv2-A2A3/saltycoffee93/docs/design.md`（按仓内 README 的 `{任务编号}-{算子名称}/{TeamName}/docs/design.md` 目录规约）。
- 验收流程引用：任务书 §3（软硬件/精度/性能/内存/自验）、§4（交付件）、§5（PR 合入）；社区流程讨论 https://gitcode.com/org/cann/discussions/39 。

## 背景介绍

### 算子功能定位

`aclsparseCgtsv2` 求解一个复数三对角线性方程组的多右端项问题 `A * X = B`，解 `X` 原地写回 `B`。

| 接口 | 对标 | 作用 |
|---|---|---|
| `aclsparseCgtsv2_bufferSizeExt` | `cusparseCgtsv2_bufferSizeExt` | 查询执行阶段所需 workspace 字节数 |
| `aclsparseCgtsv2` | `cusparseCgtsv2` | 带部分主元选取的复数三对角求解，`B` 原地更新为 `X` |

任务书 §7.4 明确：当前 ops-sparse **不存在** Cgtsv2 的公开接口或实现，`Sgtsv2`（实数 FP32 版）只作调用流程与算法结构参考。本任务新增独立 `aclsparseCgtsv2*` ABI 与公开复数类型，**不修改** `aclsparseSgtsv2(float*)` ABI（§7.2）。

### 业界对标

- 对标接口：NVIDIA cuSPARSE 13.3 Update 1 的 `cusparseCgtsv2` / `cusparseCgtsv2_bufferSizeExt`，C++ 接口语义以该版本为准（§7.3）。
- 语义差异点：
  - cuSPARSE 的 `handle` 属于各自运行时；本实现 `handle` 为 `aclsparseHandle_t`，内部携带 stream；求解在同一 stream 上异步提交。
  - `B` 双方均为原地输出；均无 `batchCount`、无独立 `X`；`n` 是 RHS 列数（§2.4）。
  - `bufferSizeExt` 返回值由本实现自定容量，调用方必须重新查询，不得硬编码历史容量。
- 算法基线：带**部分主元**选取的三对角消去（相邻行交换、维护第二上对角填充项）＋回代；官方明确禁止"仅实现无主元 Thomas"（§2.1），但允许数学等价的并行算法。

### 工程形态

- 工程模式：aclsparse Legacy C++ API 的 **Kernel 直调**模式（不走 aclnn 两段式，§2.2）：`include/cann_ops_sparse.h` 中定义公共复数类型与 C++ 接口，Host 侧完成参数校验、workspace 查询与 NPU 调度，Kernel 侧用 Ascend C 实现 `arch22`。
- 代码落位（§5 逐字）：`sparse/cgtsv2/`（公共 Host/工具代码 + README）、`sparse/cgtsv2/arch22/`（本平台 Kernel 与 Host 调度）、`test/cgtsv2/`（公共 Golden、参数、README、ATK 附件）、`test/cgtsv2/arch22/`（C++ UT/ST），根目录及各 `CMakeLists.txt` 按既有构建模式接入，不新增重复 build 目录。
- 仓内近亲：`sparse/gtsv2/arch22/`（实数 FP32 带主元版，2026-10-30 源码级核验：`gtsv2_host.cpp` 230 行 / `gtsv2_device.h` 381 行 / `gtsv2_device.cpp` 1649 行 / `gtsv2_kernel.h` / `gtsv2_tiling_data.h`）。复用方式见"详细设计"的复用/新写清单。
- 并行兄弟任务：A5（950/arch35）任务同期开发，公共 Host 代码按参数化 dtype 组织，复数特有机理（模长主元、复数补偿算术）隔离在 `sparse/cgtsv2/arch22/`，避免公共 Host 冲突（§5 要求后续 PR 基于已合入版本处理冲突并交叉回归）。

### 接口规格

公开类型（§2.1-2、§2.3）：

```c
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;
```

编译期断言：`sizeof(aclsparseComplex_t) == 8`、`alignof(aclsparseComplex_t) == alignof(float)`。

```c
aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, const aclsparseComplex_t *B,
    int ldb, size_t *pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t *dl, const aclsparseComplex_t *d,
    const aclsparseComplex_t *du, aclsparseComplex_t *B,
    int ldb, void *pBuffer);
```

参数规格表（逐行对齐任务书 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 | 非法值行为 |
|---|---|---|---|---|---|---|
| `handle` | aclsparse 上下文与 stream 句柄 | handle | 不涉及 | 有效已创建句柄 | 标量 | 空句柄返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| `m` | 三对角系统阶数 | int32 | 标量 | `m >= 3` | 标量 | `m < 3` 返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| `n` | RHS 列数（不是 batchCount） | int32 | 标量 | `n >= 0` | 标量 | `n < 0` 返回 `INVALID_VALUE`；`n = 0` 跳过求解 |
| `dl` | 下对角系数（只读） | complex64 | 连续一维 | `dl[0] = 0+0i` | `[m]` | `n > 0` 时空指针返回 `INVALID_VALUE` |
| `d` | 主对角系数（只读） | complex64 | 连续一维 | 有限值及规格允许的 Inf/NaN | `[m]` | `n > 0` 时空指针返回 `INVALID_VALUE` |
| `du` | 上对角系数（只读） | complex64 | 连续一维 | `du[m-1] = 0+0i` | `[m]` | `n > 0` 时空指针返回 `INVALID_VALUE` |
| `B` | RHS 矩阵（原地输入输出） | complex64 | 列主序连续 | `ldb >= max(1,m)`；`m<=i<ldb` padding 不变 | `[ldb, n]`，地址 `B[i+j*ldb]` | `n > 0` 时空指针、非法 `ldb` 返回 `INVALID_VALUE` |
| `ldb` | B 的 leading dimension | int32 | 标量 | `ldb >= max(1, m)` | 标量 | 不满足返回 `INVALID_VALUE` |
| `pBufferSizeInBytes` | 查询所需 workspace 字节数 | size_t* | 标量指针 | 非空 | 标量 | 空指针返回 `INVALID_VALUE` |
| `pBuffer` | 执行阶段临时工作区 | uint8 连续字节 | 连续字节地址 | 足量且 128 字节对齐；`n=0` 不要求 | `bufferSizeExt` 返回字节数 | `n > 0` 且需非零 workspace 时，空指针或未对齐返回 `INVALID_VALUE` |

补充约束（§2.4 逐条）：`n = 0` 时 `bufferSizeExt` 返回 0 且执行跳过计算；执行 ABI 不传 workspace 实际大小，调用方按查询值足量分配；Host 层校验 `m/n/ldb` 整数约束、空指针、workspace 对齐及乘法溢出；`m/n/ldb` 为运行时动态输入，每次执行内尺寸固定，不广播、不接受转置视图/任意 stride；raw C ABI 只传地址与整数，不能推断实际 dtype、分配长度或 stride，调用方/适配层负责 dtype/shape/布局转换与长度保证。

### 数学公式

以零基下标表示，`A[i,i] = d[i]`、`A[i,i-1] = dl[i]`（`1 <= i < m`）、`A[i,i+1] = du[i]`（`0 <= i < m-1`），其余元素为 0，边界 `dl[0] = du[m-1] = 0`。对每个 RHS 列 `0 <= j < n` 求解 `A X = B`：

```
d_0 * X[0,j] + du_0 * X[1,j] = B[0,j]
dl_i * X[i-1,j] + d_i * X[i,j] + du_i * X[i+1,j] = B[i,j]   (1 <= i < m-1)
dl_{m-1} * X[m-2,j] + d_{m-1} * X[m-1,j] = B[m-1,j]
```

# 需求分析（required）

## 需求描述

使用 C++ + Ascend C 在 ops-sparse 现有工程内新增 complex64（`aclsparseComplex_t`）三对角求解接口：带部分主元（复数模长比较、相邻行交换、第二上对角填充项）的消去与回代；一次因子分解供全部 RHS 复用，RHS 沿 `n` 分核并行；`B` 原地更新且 padding 不变；系数只读；`bufferSizeExt → 分配 128 字节对齐 workspace → 执行` 全流程；异步 stream 语义与错误码；确定性复跑逐位一致；精度达任务书 §3.2（rtol=atol=2^-13、0.99 匹配率、max_abs ≤ max(1e-2, 32 ULP)、NaN/Inf 分类语义）；性能达 §3.3（203 条用例逐条倍率 ≥ 0.25、均值 ≥ 0.35，仅 910B3 验收）；内存达 §3.4；核心计算全部在 Atlas A2/A3 NPU 上完成，禁止 CPU fallback。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可复用的代数量 | `A X = B`，A 为复三对角；消去/回代/主元交换的标准代数过程，可直接复用仓内 `Sgtsv2` 的因子存储与调度结构（§2.1） |
| 2 | 接口原型 | 两个 C 接口 + 一个公开类型，见"接口规格"；`aclsparseComplex_t{real,imag}`，`static_assert(sizeof==8 && alignof==alignof(float))`（§2.3） |
| 3 | 参数规格 | 上表逐行；`m>=3`、`n>=0`、`ldb>=max(1,m)`、`n>0` 时指针非空、workspace 128B 对齐且足量（§2.4） |
| 4 | 支持数据类型 | 仅 complex64（两分量 float）；不加 float 重载、不修改 `Sgtsv2` ABI（§2.3、§7.2） |
| 5 | 目标芯片与 arch 映射 | Atlas A2（910B3/910B4）与 Atlas A3 → `arch22` / DAV_2201，SOC `ascend910b*`（§2.1、§3.1） |
| 6 | 精度要求 | complex64 输出实/虚分别按 FLOAT32 混合容差 `|a-g| <= 2^-13 + 2^-13*|g|`、两部分匹配率 ≥ 0.99、`max_abs <= max(1e-2, 32*ULP(g_low))`；golden 为同输入升 complex128 的带主元带状求解；另有 NaN/Inf 分类语义套件（§3.2，混合容差标准 commit a5e8e71） |
| 7 | 性能预期 | 倍率 = GPU `cusparseCgtsv2` Kernel 中位数 / NPU 求解 Kernel 总耗时中位数；203 条每条 ≥0.25、均值 ≥0.35；主场景基准 732.032 / 717.120 / 1390.480 μs（§3.3） |
| 8 | 代码架构选型与可行性（含已排除的死路） | 复用 `sparse/gtsv2/arch22` 五件套结构：单核因子分解 + 多核按 RHS 列求解 + core0 扩展范围路径；**已排除**：① 无主元 Thomas（任务书明令禁止且不满足主元能力）；② 直接移植实数 Jacobi 快路（复数版需虚实分离向量化，UB 占用翻倍至 294 KiB 超 192 KiB，且直接路径实测余量充足，见"性能优化要点"）；③ CPU/NumPy 兜底（禁止 CPU fallback）；④ 用 `aclsparseComplex{x,y}` 复用现成类型（任务书要求 `real/imag` 字段的独立类型，不得修改既有类型语义） |
| 9 | 评测来源 | 官方 ATK 26.9.8（commit 9504bee6）：随包 200 条精度 + 验收时另生成 800 条 + 完整因子/特殊/合同配置；性能 203 条冻结清单与 GPU 基线 CSV（§3.5） |
| 10 | 交付与合入 | `task_submission/` 七件套（§4）；代码落点六项（§5）；A5 交叉回归（§5）；不提交无用代码/编译产物（§5） |

# 详细设计（required）

## 算子分析

### 数学公式

**消去（前向，带相邻行部分主元）**：主元比较使用复数模长（避免开方，比较模长平方 `|z|^2 = re^2 + im^2`，且在扩展指数表示下不溢出），相等时不交换（与 LAPACK `?gtsv` 的 `|dl_i| <= |d'_{i-1}|` 分支结构一致，NaN 比较为假时按"交换"分支走，保持非有限值的分类传播路径与批处理基准一致）。

对 `i = 1 .. m-1`（设当前主元行 `d'_{i-1}`、下对角 `dl_i`、当前主/上对角 `d_i`,`du_i`）：

```
若 |dl_i| <= |d'_{i-1}|（不交换）:
    fact   = dl_i / d'_{i-1}
    d'_i   = d_i - fact * du_{i-1}
    du'_i  = du_i              （du2' 位置填 0）
否则（交换行 i-1 与 i）:
    fact   = d'_{i-1} / dl_i
    d'_{i-1} = dl_i
    du'_{i-1} = d_i
    du2'_{i-1} = du_i          （交换引入的第二上对角填充项）
    d'_i   = -fact * du_i
    du'_i  = du_i              （交换后新的上对角）
记 swapped[i-1] = 是否交换
```

因子分解只依赖系数矩阵，与 `n` 无关，因此**一次分解供所有 RHS 复用**。首行 `d'_0 = d_0, du'_0 = du_0`。

**RHS 前代（按 swapped 传播行交换）**，对每列 `j`：

```
for i = 1 .. m-1:
    若 swapped[i-1] == 0:  b_i ← b_i - fact_{i-1} * b_{i-1}
    否则:                  swap(b_{i-1}, b_i);  b_i ← b_{i-1} - fact_{i-1} * b_i
```

**回代（含第二上对角填充项）**：

```
x_{m-1} = b_{m-1} / d'_{m-1}
for i = m-2 .. 0:
    num = b_i - du'_i * x_{i+1} - (i+2 < m ? du2'_i * x_{i+2} : 0)
    x_i = num / d'_i
```

**扩展范围路径**（病态/大动态范围/非有限输入的兜底，见"kernel 侧设计"）：在每分量"双 float 尾数（hi/lo）+ 独立 int64 指数"的表示下完成同一消去/回代，每行先按精确二次幂均衡，再做两轮扩展指数残差修正 `r = B - A x` 并将修正量回加。

### 支持数据类型

- 输入输出：`complex64`，以公开类型 `aclsparseComplex_t{float real; float imag;}` 表示；`sizeof == 8`、`alignof == alignof(float)` 由 `static_assert` 保证。
- 计算内部：FP32 复数运算（快速路径）；扩展范围路径为每分量两个 FP32 尾数 + 独立 int64 指数（不使用 FP64 指令）。
- 不支持：FP16/BF16/FP64、实数 ABI、复数双精度。

### 支持形状

| 接口 | 形状约束 | 退化/空操作 |
|---|---|---|
| `aclsparseCgtsv2_bufferSizeExt` | `m >= 3`、`n >= 0`、`ldb >= max(1,m)`（`n>0` 时校验 `m/ldb`；`n=0` 返回 0 字节） | `n = 0` → `*pBufferSizeInBytes = 0`，不校验 `dl/d/du` 指针 |
| `aclsparseCgtsv2` | 同上；`dl/d/du` 为 `[m]` 连续一维，`B` 为 `[ldb,n]` 列主序（地址 `B[i+j*ldb]`） | `n = 0` → 直接返回 SUCCESS，不要求系数/B/workspace |
| 全接口 | `m` 支持到 `2^20`；`m/n/ldb` 运行时动态，单次执行内固定；不支持广播/转置视图/任意 stride | `B` 的 `m <= i < ldb` padding 行保持原值不变 |

## 算子实现

### 实现方案

#### 总体架构与数据流

沿用仓内 `sparse/gtsv2/arch22` 已验证的三段式结构（一次分解 + 多 RHS 并行 + 兜底路径），按复数重写数值机械：

| 变体 | 触发条件 | 职责 |
|---|---|---|
| **V1** 因子分解 kernel（`<<<1>>>`，单核） | `n > 0` 且 `m <= kCgtsv2UbMaxM`（UB 可容纳 4 组复数因子 + 1 列 RHS） | UB 内完成复数带主元消去，产出 4 组复数因子（`d'/du'/du2'/multiplier`）与逐行 pivot 标记；同时按矩阵性质写 dispatch 字节 |
| **V1G** 因子分解 GM 版（单核） | `n > 0` 且 `m > kCgtsv2UbMaxM` | 因子按行分块在 GM↔UB 间搬运，标量主元消去，因子写回 workspace |
| **V2** 多 RHS 求解 kernel（`<<<blockNum>>>`） | `n > 1` | 每核负责一个列组：按行块搬入因子分片与 `B` 分片（实/虚分离），逐行做向量化前代 + 回代后回写；`m <= kCgtsv2UbMaxM` 时因子常驻 UB，否则按 `kCgtsv2ChunkRows` 行分块流式处理 |
| **V2S** 单 RHS 融合 kernel（`<<<1>>>`） | `n == 1` 且 `m <= kCgtsv2UbMaxM` | 分解 + 前代 + 回代在 UB 内一次完成，因子不落 workspace |
| **V3** 扩展范围路径（仅 core 0） | dispatch 字节置位：矩阵不满足快速判据（占优裕度不足/动态范围超界/含非有限值） | 双 float 尾数 + 独立指数的复数带主元 LU，行二次幂均衡，两轮残差修正；所有 RHS 列复用同一分解 |

数据流（UB 路径，`m <= kCgtsv2UbMaxM`）：

```
GM: dl/d/du ──DataCopyPad──▶ UB factorBuf(4×m×complex64) + pivotBuf(m×u8)
                                   │ 主元消去（逐行，标量）
                                   ▼
                     workspace: 复数因子区 + pivot 标记区 + (dispatch 字节)
                                   │ 求解核按行块取因子分片
GM: B[ldb,n] ──分列 DMA──▶ UB Btile[2][R][n]（实/虚分离）
                                    │ 逐行向量更新（一次处理该行全部 n 列）
                                    ▼
                              回写 B 对应行（padding 不动）
```

**求解侧为何按"行分块 + 列向量化"而不是近亲的"逐列标量"**：本机实测（910B3，2026-10-30）实数版 `Sgtsv2` 在 m=4096/n=64 上 `|B|≤5` 时为 971.18 μs、把 `B` 放大到 `|B|≤500` 迫使它退到逐列标量回代路径时为 961.62 μs——即逐列标量路径的每行处理成本约 200 周期，瓶颈在逐元素访问延迟而非算术。复数版若沿用逐列标量，按 2.5~3 倍实数代价外推，P-01 会逼近甚至超过 2 928 μs 的允许耗时。改为行分块后：一次向量更新处理该行全部 `n` 列（`n=64` 复数 = 128 个 float = 512 B = 2 条向量指令），每行开销降到 ~8 条向量指令量级，串行链长度仍为 `m`，但每步是大粒度向量操作，P-01 求解段可压到 0.25 ms 量级。

GM 路径（`m > kCgtsv2UbMaxM`）：因子与 B 均按行块（`kCgtsv2ChunkRows` 行）双向搬运，块内逐行向量更新；前代自首行向下、回代自末行向上各扫一遍，`B` padding 区不参与搬运与回写。

#### host 侧设计

- **分核策略**：
  - 因子分解固定**单核**（分解对全部 RHS 共享，多核无收益且会破坏"一次分解"语义），`<<<1>>>`。
  - 多 RHS 求解按 **`n` 分列组 × 行块流水**：`blockNum = min(ceil(n / kCgtsv2VecCols), GetAivCoreCount())`，其中 `kCgtsv2VecCols` 是单条向量指令能覆盖的复数个数（`n=64` 时 1 组即可占满 256 B 向量宽度，故 `blockNum` 多为 1~2）；每组列独立完成前代/回代，组间无通讯、无浮点归约 → 确定性成立。列组划分用 `ceil` 均分，末组为余块。
  - `n == 1` 走 V2S 融合核（单核，分解+求解都在 UB，因子不落 workspace）。
  - 当 `ldb * 8 % 32 != 0`（列步长非 32 字节对齐）时，V2 的列组退化为按列逐条 DMA 的单核路径，避免相邻列非对齐 DMA 边界互相覆盖（沿用近亲 `gtsv2_host.cpp:146-156` 的判定与注释理由）。
- **数据分块和内存优化策略**：
  - 因子常驻 UB 阈值：`41 * m` 字节（4 组复数因子 32m + pivot m + 单列 RHS 8m）对 DAV_2201 的 192 KiB UB 取安全阈值 `kCgtsv2UbMaxM = 4608`（188 928 B，留 ~7.7 KiB 给 32 B 对齐 padding 与框架预留）。超过阈值走 GM 流式路径。
  - 流式块行数 `kCgtsv2ChunkRows = 64`：因子分片 `4 × 64 × 8 = 2 KiB`，B 分片 `2 × 64 × n × 4` 字节（`n=64`→32 KiB，`n=128`→64 KiB），均远低于 UB 上限；块数与 `n` 解耦。
  - B 分片在 UB 内以**实/虚分离**的两个 `[R][n]` float 数组存放（列主序 DMA 分列搬入后即分离），使前代/回代的复数乘加变成纯实数向量指令（4 次 `Mul` + 2 次 `Sub`/`Add` / 行），避开复数交织布局的跨分量搬移。
  - 因子只写一次、求解期只读；`B` 仅原地读写，不额外分配独立 `X`（§3.4 内存口径）。
  - 不启用 double buffer：消去/回代是逐行数据依赖的串行链，乒乓收益低于寄存器压力。
- **tilingkey 规划策略**（以 `Cgtsv2TilingData` 的取值分支代替编译期 tilingkey，与近亲一致）：

| 分支 | 取值 | 含义 |
|---|---|---|
| `useUb` | 0 / 1 | `m > kCgtsv2UbMaxM` → 0（GM 流式） |
| `singleRhs` | 0 / 1 | `n == 1` → 1，走 V2S 融合核 |
| dispatch 字节（pivot 区末字节） | 0 / 1 | 因子核判断矩阵性质；1 → V3 扩展范围路径，仅 core 0 进入，其余核读到即退出 |
| `blockNum` | `min(ceil(n/VecCols), cores)` | 求解列组数 |
| `colGroup` / `colsThisCore` | 计算量 | 本核负责的列区间，末组余块 |

- **数据检测**（校验顺序与错误码，逐条对齐任务书 §2.4）：

| 顺序 | 检查项 | 返回 |
|---|---|---|
| 1 | `handle == nullptr` | `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` / `m < 3` / `ldb < max(1, m)`（`n > 0`） | `ACL_SPARSE_STATUS_INVALID_VALUE` |
| 3 | `n > 0` 且 `dl/d/du/B` 任一为空 | `INVALID_VALUE` |
| 4 | `n > 0` 且 `pBuffer == nullptr` | `INVALID_VALUE` |
| 5 | `pBuffer` 地址未 128 字节对齐 | `INVALID_VALUE` |
| 6 | workspace 字节与 `ldb*n` 元素数的乘法/加法溢出 | `INVALID_VALUE` |
| 7 | `n == 0`（`aclsparseCgtsv2`） | 直接 `SUCCESS`（不校验 3~6） |
| 8 | `aclsparseGetStream` 失败 / 平台核数为 0 | 透传库状态 / `ACL_SPARSE_STATUS_INTERNAL_ERROR` |

  `bufferSizeExt` 的检查为 1、2（仅 `n<0`）、`pBufferSizeInBytes` 空指针、`n>0` 时 `m/ldb/dl/d/du`、溢出（`B` 指针按近亲约定不参与查询校验，`n=0` 返回 0）。

#### kernel 侧设计

**Init/Process 两阶段**（与仓内同类算子一致）：

- `Init`：`SetGlobalBuffer` 绑定 `dl/d/du/B` 与 workspace 分区（复数因子区 / pivot 区 / 扩展范围区），按 `useUb` 预分配 `TBuf`。
- `Process`：按 `useUb` 与 dispatch 字节分支：

```
V1/V1G: Process() ─┬─ useUb ──▶ FactorInUb()   → 写 4 组复数因子 + pivot + dispatch 字节
                   └─ else  ──▶ FactorInGm()   → 分块搬运的主元消去
V2:     Process() ─┬─ dispatch==1 ──▶ 仅 core0 进 V3（扩展范围路径），其余核直接 return
                   ├─ useUb ──▶ SolveGroupUb(colGroup)（因子常驻 UB，行块向量化前代/回代）
                   └─ else  ──▶ SolveGroupGm(colGroup)（因子按行块流式搬入）
V2S:    Process() ──▶ 单核内 FactorInUb + 前代 + 回代（因子不落 workspace）
```

流程图（UB 路径）：

```
        ┌────────────── V1 factor kernel (1 core) ──────────────┐
        │ dl/d/du → UB[4m complex] → 逐行: 模长平方选主元/交换/填充 │
        │            → workspace[d',du',du2',mult] + pivot[] + flag│
        └───────────────────────────┬───────────────────────────┘
                                    │ (flag==0)
        ┌───────────────────────────▼───────────────────────────┐
        │ V2 solve kernel (blockNum 列组), 每组独立:              │
        │ 因子分片 + B 列组分片 → UB 实/虚分离 → 逐行向量前代/回代 │
        │ → B 同址回写（padding 行不搬运）                        │
        └───────────────────────────────────────────────────────┘
```

**V3 扩展范围路径（复数补偿算术）**：每个复数值表示为每分量 `(hi, lo)` 双 float 尾数 + 独立 `int64` 指数（实部、虚部分别维护指数，避免分量动态范围差异被共享指数截断）。基本构件：

| 构件 | 实数近亲 | 复数实现 |
|---|---|---|
| 无误差加法/乘法 | `Gtsv2TwoSum` / `Gtsv2TwoProduct`（拆分 12 位、舍入屏障 `volatile`） | 逐分量复用（real/imag 各一套），交叉项 `a_r b_i`、`a_i b_r` 分别做双 float 乘法后相加 |
| 复数加 | `Gtsv2ScaledAdd` | 分量双 float 相加 + 分量指数归一 |
| 复数乘 | `Gtsv2ScaledMultiply` | `(a_r b_r - a_i b_i) + i(a_r b_i + a_i b_r)`，4 次双 float 乘 + 2 次双 float 加减 |
| 复数除 | `Gtsv2ScaledDivide` | `a/b = a·conj(b) / (b_r^2 + b_i^2)`；分子分母全程扩展指数表示，平方不溢出，再逐分量除 |
| 模长比较 | `Gtsv2ScaledAbsGreaterEqual` | 比较 `|a|^2 = a_r^2 + a_i^2` 的扩展指数值（先比指数、再比尾数），等价于模长比较且无开方 |
| 行均衡 | `Gtsv2ScaledRowExponent`（取行内最大指数） | 取 `max_i ∈ {r,i} max(|d|,|dl|,|du|)` 的指数区间，按精确二次幂只调指数 |
| 残差修正 | 两轮 `b - A x`，复用所选 LU | 同结构，复数残差逐分量计算，写回前整列检查有限性 |

**非有限值语义**：任一构件遇到非有限分量即退回该步的普通 FP32 复数算术（`fromFloat/Add/Mul/Div` 的非有限分支），保持 Inf/NaN 沿消去-回代链的传播顺序；主元比较对 NaN 采用 IEEE 语义（比较为假即进入交换分支），与 LAPACK 带状求解的分支结构对齐。真正奇异（主元精确为 0）时不做特殊保护，按库既有 `Sgtsv2` 约定传播 Inf/NaN（§2.1-5 要求"不崩溃、不越界并记录返回码"，不要求 Host 检测奇异）。

**workspace 布局**（128 字节对齐，`n > 0` 时）：

| 区段 | 字节/行 | 大小 | 内容 |
|---|---|---|---|
| 复数因子区 | 32 | `align128(32*m)` | `d'`、`du'`、`du2'`、`multiplier` 各 m 个 `complex64` |
| pivot 标记区 | 1 | `align128(m)` | 逐行交换标记（u8）+ 末字节 dispatch 标志 |
| 扩展范围区 | 129 | `align128(129*m)` | 5 组复数扩展指数数组（每组每元素 4×float 尾数 16B + int64 指数 8B = 24B → 120B）、原始 RHS 副本 8B、pivot scratch 1B |

`bufferSizeExt` 返回值：

```
W(m, n) = n > 0 ? align128(32*m) + align128(m) + align128(129*m) : 0
```

实测值（本机 m 计算，单位字节）：`m=4096 → 663552`；`m=7168 → 1161216`；`m=8192 → 1327104`（≈1.27 MiB，§3.4 内存条款 2 要求 workspace 绝对值不超过目标硬件 L2 Cache，910B3 的 L2 容量在自测报告中实测记录）。

#### 性能优化要点（终态）

| 优化项 | 说明 | 效果/来源 |
|---|---|---|
| 一次分解、全 RHS 复用 | 消去成本与 `n` 无关，`n=64/128` 时分解只做一次 | 近亲结构（`gtsv2_device.cpp` 文件头注释），本任务沿用 |
| 沿 RHS 列分核 | 列之间完全独立，天然并行、无跨核通讯、无浮点归约 → 确定性天然成立 | §2.1-5 确定性要求 |
| 模长平方式主元比较 | 免开方，比较在扩展指数域进行不溢出 | 复数特有 |
| 单 RHS 融合核 | `n==1` 不写读因子，省一次 GM 往返 | 近亲同构 |
| 未采用：复数 Jacobi 快路 | 复数版需虚实分离向量化 + UB 需求翻倍（3 组系数 + 2 组迭代向量 = 48m ≈ 294 KiB @m=6144 > 192 KiB），且直接路径实测余量充足 | 见"需求拆解"第 8 行 |
| 未采用：双精度/CPU 回退 | 任务书禁止 CPU fallback；扩展指数已覆盖动态范围 | §3.4 |

性能预算校核（P-01，`m=8192, n=64, ldb=8192`）：允许耗时 = 4 × 732.032 μs = 2.93 ms；数据量 `(3m + ldb*n)*8 = 4,390,912 B`（与任务书 §3.4 的 `io_bytes` 一致），`B` 读改写按 4 遍计 ≈ 17.6 MB → 仅需 ≈ 6 GB/s，远低于 HBM 带宽；瓶颈是消去的逐行串行链。

**近亲实测参考**（本机 910B3，2026-10-30，`build_out/libops_sparse.so`，ACL Event 口径，warmup 10 / 采样 30，每次取样前恢复 `B`）：

| 形状 | 实数版 `Sgtsv2` Event 中位数 | 本任务允许耗时（4×GPU Kernel 中位数） | 实数版余量 |
|---|---|---|---|
| P-01 `m=8192,n=64,ldb=8192` | 1877.42 μs | 2928.1 μs | 1.56× |
| P-02 `m=4096,n=64,ldb=4096` | 971.18 μs | 2868.5 μs | 2.95× |
| P-03 `m=7168,n=128,ldb=7168` | 2990.91 μs | 5561.9 μs | 1.86× |

同一程序在**非严格对角占优**系数（`|d|` 仅比 `|dl|+|du|` 大 1）下实测 P-01 为 4,701,877 μs（4.70 s）：实数版会从向量 Jacobi 快路掉进单核扩展指数路径。**结论与设计约束**：① 复数版必须在因子核用**廉价矩阵性质判据**决定 dispatch，绝不把常规输入误判进单核兜底路径；② 本任务的 203 条性能用例分布为 `diag_dominant`（ATK `d = |raw|+|dl|+|du|+1`，`|dl|/|du|` 已乘 0.125），三条主场景均落在多核主路径；③ 复数版的多核主路径本身是**带主元** LU（对任意有限输入数值稳定），不依赖占优假设，因此不需要为性能复制 Jacobi；④ 扩展指数兜底路径只承接真正病态/超动态范围的输入，其耗时不进入性能验收。

复数主路径成本估算（P-01，`m=8192, n=64`）：求解段按行分块、每行对该行全部 64 列做复数乘加（实/虚分离后为 `2 × 64 = 128` 个 float = 512 B，约 8 条向量指令/行/遍），两遍扫描 → `8192 × 2 × 8 ≈ 13 万`条向量指令 ≈ 0.13 ms；因子段单核逐行消去（每行 ~60 周期）→ `8192 × 60 ≈ 49 万`周期 ≈ 0.27 ms；合计 ≈ 0.4~0.6 ms，相对 2.93 ms 允许耗时余量 ≥ 4.9×。P-02/P-03 与实测近亲同量级比例，同理留足余量。最终实测值以 `results/perf_gap.json` 为准。

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列（910B3 / 910B4） | ✔ 功能、精度、UT/ST；性能仅 910B3 验收 |
| Atlas A3 训练系列 / A3 推理系列 | ✔ 功能、精度（`arch22` / DAV_2201） |
| Ascend 950PR / 950DT | ✘ 本任务不含（A5 兄弟任务另立 `arch35`） |

## 算子约束限制

- 仅 `complex64`；无 float 接口重载；不修改 `aclsparseSgtsv2(float*)` ABI（§7.2）。
- `dl[0]`、`du[m-1]` 由调用方置 0（Host 不校验，raw ABI 约定）。
- 不接受广播、转置视图、任意 stride、非连续布局；`m/n/ldb` 单次执行内固定。
- 真正奇异系统不保证有限输出，按库既有传播约定（§2.1-5），不以非有限输出作为精度通过依据。
- 未明确支持的非有限输入不纳入常规数值精度验收；特殊值语义按 §3.2 分类表独立记录。
- 确定性：同设备/版本/输入/配置复跑逐位一致；不引入跨核浮点归约或原子累加。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度（常规） | complex64 输出的实部、虚部分别按 FLOAT32 混合容差验收：`|actual-golden| <= atol + rtol*|golden|`，`rtol = atol = 2^-13`，两部分匹配率均 ≥ 0.99；每分量最大有限绝对误差 ≤ `max(1e-2, 32*ULP(g_low))`，`g_low` 取该分量最大有限绝对误差点的 Golden 收窄为 float32 后的值，ULP 按 FLOAT32 计算 | 任务书 §3.2；混合容差标准 commit `a5e8e716` |
| 精度（golden） | 同一份 complex64 输入升 complex128，用 CPU 带主元带状求解（SciPy `solve_banded` / LAPACK `?gbsv`）作为 Golden；实/虚分位置比对 | 任务书 §3.1、§3.2 |
| 精度（溢出特例） | NPU 分量为 ±Inf、Golden 分量有限时，将 Golden 按 RNE 收窄为 FLOAT32，仅收窄后为同号 Inf 才通过 | 任务书 §3.2；混合容差标准 §2.3 |
| 精度（特殊值语义，独立记录） | NaN/Inf 分类 mask 一致、Inf 符号一致；NaN↔NaN 通过、同号 Inf 通过、其余失败；有限分量逐点 `|a-g| <= 2^-13 + 2^-13*|g|`；不适用 0.99 匹配率与 ULP 上限；含 NaN 一律不记常规精度通过 | 任务书 §3.2 分类表 |
| 精度（用例量） | 随包 200 条 + 验收时官方 ATK 另生成 800 条独立运行，两组均须通过；数量不替代完整因子覆盖审计 | 任务书 §3.2、§3.5 |
| 性能 | Kernel 总耗时中位数倍率 = cuSPARSE 求解调用 Kernel 中位数 / NPU 求解调用 Kernel 总耗时中位数；203 条（3 主 + 200 扩展）每条 ≥ 0.25、算术平均 ≥ 0.35；仅 910B3 验收；warmup ≥ 10、采样 ≥ 30，每次原地求解前恢复原始 B 并从恢复后开始计时，workspace 复用 | 任务书 §3.3 |
| 内存 | 满足其一：输入输出 > 500 MB 时 NPU 较 GPU 额外内存 ≤ 5%；或方案固有 workspace 绝对值 ≤ 目标硬件 L2 Cache。逐用例报告 `(3m + ldb*n)*8` 的 IO、workspace 查询值与实际对齐分配值，恢复副本单独列示 | 任务书 §3.4 |
| 确定性 | 同设备/版本/输入/配置，恢复原始 `B` 后复跑输出逐位一致 | 任务书 §2.1-5 |
| 合入 | `include/cann_ops_sparse.h` 新增类型与声明（不动 `Sgtsv2`）；`sparse/cgtsv2[/arch22]`、`test/cgtsv2[/arch22]` 落点；A5 交叉回归 | 任务书 §5 |

## 测试概览

| 组 | 入口 | 内容 | 状态 |
|---|---|---|---|
| 随包精度 200 条 | `test_cases/atk/`：`atk case -f case_designs/accuracy.yaml -p atk_plugins.py -s 1028` | 官方 ATK 原生生成，四 Tensor `[-5,5]` 均匀/正态各 50% 正交遍历；末尾含 4 条合法 `n=0` | 交付（§3.5） |
| 验收精度 800 条 | `case_designs/accuracy_acceptance_800.yaml`（验收方另选种子） | 独立生成、独立运行 | 由验收方生成并运行 |
| 完整因子域 | `case_designs/accuracy_full_coverage.yaml` + `audit_atk_cases.py` | `[1,2^20]` 幂次/幂次减一维度、padding 等价类、分布组合，经 `ldb<=2^20`、总元素 `<=2^34` 过滤 | 交付（审计报告） |
| 特殊值/合同 | `accuracy_special.yaml` / `accuracy_contract.yaml` | NaN/Inf 分类语义；非法参数拒绝（错误码以 `CGTSV2_INVALID_PARAMETER:` 标记） | 交付（独立结论） |
| 性能 203 条 | `test_cases/performance/collect_performance.py` + 冻结清单与 GPU 基线 | Kernel/Event 两口径、内存快照；GPU NCU 与 NPU Profiler 同清单同种子配对 | 交付（§3.3） |
| C++ UT/ST | `test/cgtsv2/arch22/`（CSV 驱动 + GTest） | 接口、Host 校验、Kernel 分支、复数类型布局、`bufferSizeExt → 执行 → 释放` 全流程、padding 不变、确定性复跑、奇异系统返回码 | 交付（§2.2、§5） |
| 跨平台回归 | 910B3 / 910B4 / A3 各一遍 | 功能与精度（性能仅 910B3） | 交付（§3.1） |

复现入口：交付件 `03-自测报告/1 自验证步骤说明.md` 给出环境、编译、ATK 生成/审计、NPU 精度运行、GPU/NPU Kernel 与 Event 采集、内存统计的完整命令序列。

## 兼容性分析

| 变更点 | 波及面 | 验证方式 |
|---|---|---|
| `include/cann_ops_sparse.h` 新增 `aclsparseComplex_t` + 两个声明 | 公共头文件；不修改 `aclsparseSgtsv2` 声明与 `aclsparseComplex{x,y}` 既有类型 | 全仓编译 + 既有 `gtsv2`/`gtsv2_nopivot`/`gtsv2_strided_batch` 回归全绿 |
| `sparse/cgtsv2/`、`test/cgtsv2/` 新增目录 | 由既有 `file(GLOB_RECURSE)` + arch 过滤机制自动纳入构建，不需要改公共构建逻辑 | `bash build.sh --ops=cgtsv2 --run` 干净目录构建通过 |
| A5（arch35）兄弟任务并行开发 | 公共 Host 代码按参数化 dtype 组织；复数特有机理只落在 `sparse/cgtsv2/arch22/` | 后合入方基于已合入版本 rebase 后做 910B3/A3 全量精度回归（任务书 §5 明确要求交叉回归） |
| workspace 容量变更 | 调用方须按 `bufferSizeExt` 查询值分配，不得硬编码既有 `Sgtsv2` 容量 | UT 校验查询值与实际需求一致（`HonorsQueriedWorkspaceBounds` 同类用例） |
