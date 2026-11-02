# 需求背景（required）

## 需求来源

《10月社区任务-aclsparseCgtsv2算子开发（A2A3）》：在 `cann/ops-sparse` 仓新增
complex64 三对角线性方程组求解接口，功能与 C++ 语义对标 cuSPARSE
`cusparseCgtsv2`。设计文档提交至 `cann/cann-ops-competitions` 的
`04_tasks/01_community-task-2026/tasklist/10-aclsparseCgtsv2-A2A3/` 下。

## 背景介绍

现有 `aclsparseSgtsv2` 只提供 `float*` ABI，且仅 arch35 有实现；A2/A3（DAV_2201，
arch22）需要独立的 complex64 接口。复数三对角系统在电磁场、频域电路和复数
预处理中常见；随机复系数矩阵即使输入、输出都有限，消元中间量也可能超出
FLOAT32 范围，或在抵消后损失超过混合容差。因此不能简单把 `Sgtsv2` 的 float
代码逐行改成两个 float。

上游 master 不存在 `Cgtsv2` 公开接口或实现；`Sgtsv2`（arch22/arch35）只作为
调用流程、校验顺序与事件编排的参考。

# 需求分析（required）

## 需求描述

新增 `aclsparseComplex_t`（连续 `float real; float imag;`，`sizeof==8`、
`alignof==alignof(float)` 由编译期断言冻结）和两个 Legacy C++ 接口：

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

不修改 `aclsparseSgtsv2(float*)` 声明。

## 需求拆解

| 编号 | 需求 | 设计落点 |
|---|---|---|
| R01 | 带部分主元的复数三对角求解，允许等价并行算法，禁止仅无主元 Thomas | §3.2 Kernel 侧：按模长比较的相邻行交换 + 第二上对角 fill-in |
| R02 | 新增 `aclsparseComplex_t` 及编译期尺寸/对齐断言 | §3.1 公共头 |
| R03 | `bufferSizeExt → 128B 对齐 workspace → 执行`，正确处理 stream/生命周期/错误码 | §3.1 Host 侧、§4 |
| R04 | 覆盖 `m/n/ldb`、单/多 RHS、对角占优、需 pivot、零主对角非奇异、病态/奇异、边界 | §5、`test/cgtsv2` 与 ATK 套件 |
| R05 | `dl/d/du` 只读，B 原地更新，无独立 X、无 batchCount | §3.2；UT 校验只读与 padding |
| R06 | 非奇异输入确定性复跑 | §3.2 纯确定性递推；UT 三次复跑 bitwise |
| R07 | 真正奇异系统不崩溃、不越界，记录返回码与输出 | §3.2 不设奇异保护；UT 与报告记录 |
| R08 | 公共算法与 A5 解耦，完成交叉回归 | §6 兼容性 |
| R09 | 精度：实/虚分量混合容差 `2^-13`、匹配率 ≥0.99、最大误差动态 ULP 上限 | §5 精度标准 |
| R10 | 性能：910B3 上 203 条逐例倍率 ≥0.25，算术平均 ≥0.35 | §5 性能标准 |
| R11 | 内存：workspace 绝对值不超过目标 L2，或大 IO 时额外内存 ≤GPU 5% | §4 内存口径 |
| R12 | 200+800 精度、8 特殊、12 合同、203 性能与内存全部走官方 ATK | §6 测试计划 |

# 详细设计（required）

## 算子分析

### 数学公式

`A[i,i]=d[i]`，`A[i,i-1]=dl[i] (i≥1)`，`A[i,i+1]=du[i] (i<m-1)`，其余为 0。
对每个 RHS 列 `j` 求解 `A X[:,j] = B[:,j]`。

前向消元对 `r=1..m-1`：

- 若 `|d'[r-1]| >= |dl[r]|`：`mult = dl[r]/d'[r-1]`，不交换，
  `d'[r] = d[r] - mult*du'[r-1]`，`du'[r] = du[r]`，`du2'[r-1] = 0`；
- 否则交换相邻行：`mult = d'[r-1]/dl[r]`，
  `d'[r-1] = dl[r]`，`du'[r-1] = d[r]`，`du2'[r-1] = du[r]`，
  `d'[r] = du'[r-1] - mult*d[r]`，`du'[r] = -mult*du[r]`。

对每个 RHS 列，前向同时应用交换与消元，再回代：

`x[i] = (b'[i] - du'[i]*x[i+1] - du2'[i]*x[i+2]) / d'[i]`。

主元比较使用复数模长（平方模比较，避免开方）；`|a|==|b|` 时固定选择不交换，
保证确定性。主元选择只依赖系数、与 RHS 无关，因此全部 RHS 复用同一份
`d'/du'/du2'/mult/pivot`。

### 支持数据类型

仅 complex64，公开入口只接受 `aclsparseComplex_t*`。不新增 float 重载，不支持
广播、转置视图或额外 stride。

### 支持形状

运行时动态 `m/n/ldb`；`dl/d/du` 为 `[m]`，`B` 为列主序 `[ldb,n]`，有效区
`[m,n]`，`B[i+j*ldb]`；`m<=i<ldb` 的 padding 不写。`n=0` 是合法空 RHS。

## 算子实现

### 实现方案

#### 数据流与职责

| 阶段 | 职责 | 输出/验证 |
|---|---|---|
| Host 查询 | 校验 handle、输出指针、`m/n/ldb`、必需系数指针；`n=0` 返回 0 | workspace 字节数，不读取设备数据 |
| Host 执行 | 校验尺寸、指针、128B 对齐、整数溢出；取 handle stream | 在 stream 上提交 factor→solve |
| Factor Kernel | 单核、只读系数；按 512 行 tile 流式分解 | 扩展精度因子 + pivot 字节写 workspace |
| Solve Kernel | 每核一组 RHS 列；前后向 tile 读因子 | B 有效区原地写 X，padding 不变 |
| 测试 | complex128 Golden、ATK、UT/ST、性能与内存采集 | 逐 SoC/提交记录通过、失败、未执行 |

#### Host 侧设计

校验顺序固定为：`handle → n==0 早退 → m/n/ldb → 空指针 → workspace 对齐 →
字节/索引溢出`（查询接口为 `handle → pBufferSizeInBytes → n<0 → n==0 →
m/ldb/系数指针 → 溢出`）。`n=0` 与 `Sgtsv2` 保持一致：执行直接 SUCCESS，
查询返回 0。

workspace 公式（查询与执行共用）：

```
alignedRows = align_up(m, 8)
slots       = min(n, 40)   # m <= 32768
            = 1            # m > 32768
bytes       = align_up((3 + slots) * 24 * alignedRows + alignedRows + 16, 128)
```

三个共享扩展复数数组（对角逆、第一上对角、乘数）各 `24*alignedRows` 字节
（4 个 float 有效位 + 2 个 int32 指数）；前向 RHS b' 按 **solve block** 分区，
每个 block 一份 `24*alignedRows`，不同核处理不同列时互不覆盖；pivot 1 字节/行，
末尾 16 字节保留。`n=0` 返回 0；性能主场景约 8.5 MB，`m=2^20` 且 n 很大时
`slots=1`，约 101.7 MB。8 行对齐保证每个平面 32B 对齐，可直接 `DataCopyPad`。

stream 从 handle 获取；两次 Kernel 顺序提交，不在接口内同步。Host 只返回参数
与启动状态，异步数值奇异在调用方同步后由输出体现。

#### Kernel 侧设计

**扩展精度表示。** 每个实/虚分量存为 `(hi, lo, exponent)`：`hi+lo` 是
error-free two-float 有效位，`exponent` 为独立二进制指数。加减用 two-sum
对齐指数；乘用 two-product 并累加低阶项；除用商 + 残差修正。这样有效位约为
FP32 的两倍（~48 bit），指数范围远超 FLOAT32，解决两类实测风险：中间量
超出 `FLT_MAX`、以及 `κ(A)` 放大后 complex64 误差超过 `2^-13`。

**Factor Kernel（1 核）。** 每 512 行为一个 tile：
1. DMA 三个系数 tile（complex64）到 UB；
2. 逐行执行 §3.2 递推，pivot 状态跨 tile 保存在寄存器；
3. 将 `invD/du'/mult` 的 6 个平面与 pivot 字节 DMA 到 workspace。
形状无关的尾行在循环后单独写并统一 `DataCacheCleanAndInvalid`。

**Solve Kernel（`blockNum` 核）。**
- `blockNum = min(n, AIV 核数)`；若 `ldb % 8 != 0`（相邻列共享 cache line 或
  32B DMA 边界），退化为 1 核，避免跨核写冲突。
- 每核按列循环：前向 tile 读 `mult/pivot/B`，把扩展 `b'` 写到 workspace 的
  `rhsPrime` 平面；后向 tile 读 `invD/du'/rhsPrime/pivot/原 du`，按行回代并
  四舍五入（RNE，含次正规与溢出到 Inf）写回 B。
- 前向 tile 向前多覆盖一行，使 `r==tile_start` 的交换可以安全改写上一 tile 的
  `b'[r-1]`；后向 tile 携带 `x[i+1], x[i+2]` 跨界状态。
- `du2'[i]` 不单独存储：当 `pivot[i]!=0` 时它等于原 `du[i+1]`，由输入重建，
  节省 24 B/row。

**非有限 RHS。** 前向扫描 B tile，一旦出现 NaN/Inf 就放弃该列的扩展路径，改用
plain complex64 递推。该路径前向跳过零乘数（对齐 zaxpy 的 alpha==0 早退），
回代无条件计算两个 ZAXPY 项（对齐 zgbtrs），从而复现 complex128 LAPACK 的
NaN/Inf 位置与符号分类。特殊值套件不参与常规数值精度。

## 支持硬件

| 平台 | 功能/精度 | 性能验收 |
|---|---|---|
| Atlas A2 910B3 | 是 | 是 |
| Atlas A2 910B4 | 是 | 否 |
| Atlas A3 任务环境具体型号 | 是 | 否 |

报告分别记录 SoC、CANN、驱动、固件、代码提交与实测结果；单平台结论不外推。

## 算子约束限制

无 batchCount、独立 X、转置视图或任意 stride；B 有效区允许原地覆盖；
workspace 在异步完成前有效；raw ABI 不能推断 dtype/长度/stride。
Inf/NaN 不作为常规精度通过；奇异系统不承诺具体错误码或有限解。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 输入升 complex128 带主元 Golden；实/虚分别 `atol=rtol=2^-13`、匹配率 ≥0.99、最大有限误差 ≤ `max(1e-2, 32*ULP(g_low))` | 任务书；opbase 混合容差标准 `a5e8e716…` |
| 性能 | 仅 910B3；203 条 Kernel 中位数倍率逐例 ≥0.25、算术平均 ≥0.35；预热 ≥10、采样 ≥30，排除 B 恢复 | 任务书与 performance 附件 |
| 内存 | 固有 workspace ≤ L2；或 IO>500 MB 时额外内存 ≤GPU 5%；逐例报告查询/对齐/峰值/设备快照 | 任务书 §3.4 |

## 测试计划

- 官方 ATK 26.9.8：200 发布精度、额外 800、8 特殊、12 合同、203 性能与内存；
  另有完整因子网格元数据审计，超资源项单列。
- C++ UT/ST：接口/边界/对齐/公式、系数只读、padding、三次复跑 bitwise、
  512 行 tile 边界、奇异系统内存安全、异步顺序。
- 交叉回归：与 `Sgtsv2` 共用同一份构建与测试框架，确认公共头只增不改、
  `Sgtsv2` 测试不回归。
- 变异测试：对关键分支人为破坏 ≥5 次（改 pivot 方向、删 fill-in、改 tile 边界、
  mask/对齐、golden 错位），每次 harness 必须报警。

## 兼容性分析

新增符号与类型，`Sgtsv2` 声明与实现不变；A2/A3 与 A5 各自独立目录，公共 Host
代码若并行变更以实际合入顺序解决冲突并做交叉回归。提交前运行
`git merge-base HEAD origin/master`，保证 PR 基于最新 master。

## 参考与未决事项

- 任务书：`aclsparseCgtsv2_A2A3_task_doc.md`。
- 参考实现：`sparse/gtsv2/arch22`（Host 校验/事件编排）与
  `sparse/gtsv2/arch35`（pivot 递推公式），不照抄其 float 数值路径。
- 未决：真实 910B3 上的精度/性能/内存数据在实现完成后按 ATK 与 C++ UT 采集；
  本文不预填实测结论。
