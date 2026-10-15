# MatmulLayerNormMatmul 算子设计文档（Ascend 950 / CATLASS）

> 提交位置：`cann-ops-competitions/04_tasks/01_community-task-2026/tasklist/09-45-MatmulLayerNormMatmul/{AtomGit账号名}/docs/design.md`
> （本设计对应任务「9月社区任务-MatmulLayerNormMatmul算子开发」，任务编号 09-45）
> PR 标题：`【CANN社区任务】MatmulLayerNormMatmul算子设计文档`

# 需求背景（required）

## 需求来源

CANN 社区任务（2026 年 9 月）：**MatmulLayerNormMatmul 算子开发**，适配 Ascend 950（28 cube /
56 vector，L2 128 MB），开发语言 Ascend C，开源仓 [cann/catlass](https://gitcode.com/cann/catlass)。

## 背景介绍

### MatmulLayerNormMatmul 算子说明

该算子把「矩阵乘 → 行归一化 → 矩阵乘」三段融合为一次计算：

```
C0      = A0 * B0            // (M0, N0)
C0norm  = LayerNorm(C0)      // 行归一化（gamma/beta 长度 N0，eps = 1e-6）
C1      = C0norm * B1        // (M0, M1)
```

### 现状分析（小算子拼接的问题）

参考实现用三个独立小算子拼接：`torch.mm` + `F.layer_norm` + `torch.mm`。其代价：

1. **中间矩阵两次落显存**：`C0` 由第一个 mm 写入显存，LayerNorm 再读一遍、写一遍（fp16），
   第二个 mm 又要把它读回 L1——3 次内核下发、约 4 次 M×N 规模的显存往返；
2. **下发与调度间隙**：116 条官方用例里，三段小算子之和与整链路标杆的差距平均约 11%
   （小形状每段 1–2 µs 的启动+调度开销无法摊薄）；
3. 逐段实现无法共享片上状态（L1/L0C 里的结果必须整块写回）。

融合算子的收益上界因此是：省下的下发开销 + 可以隐藏/减小的 LayerNorm 时间，
这两项在小形状上最显著；大形状上两段 Matmul 本身已接近硬件峰值，收益主要来自
LayerNorm 的搬运压缩。

# 需求分析（required）

## 需求描述

使用 Ascend C（CATLASS 框架）实现 MatmulLayerNormMatmul 融合算子：

1. **单次 kernel launch** 完成全部三段计算；
2. Mean 和 Variance 等中间结果由**算子内部 workspace** 管理，不对外暴露，唯一输出为 `C1`；
3. 精度满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)，
   基于 optest 任务测试集通过、ATK 泛化 ≥200 例精度通过；
4. 性能：116 条官方用例平均 `标杆时延 / 算子时延 > 1.1`（基准 = torch.mm + F.layer_norm + torch.mm
   拼接方案，标杆数据由任务测试集提供，msprof 采集）。

## 需求拆解

1. 算子签名与数据类型：`A0(M0,K0) fp16 RowMajor`、`B0(K0,N0) fp16 ColumnMajor`、
   `B1(N0,M1) fp16 ColumnMajor`、`gamma/beta(N0) fp32`、`C1(M0,M1) fp16 RowMajor`；
2. 内部 workspace：`M0 × RoundUp(N0,16) × 2 B`（fp16 暂存），最大任务形状 32 MiB，
   由 `GetWorkspaceSize` 申请、随 kernel 参数下发，调用方无感知；
3. 精度：三段数学等价，LayerNorm 的均值/方差在 fp32 下计算（见 §详细设计），
   ATK 253 条泛化中 227 条通过（其余 26 条为 fp16 逐元素阈值临界用例，见 §可维可测）；
4. 性能：逐形状瓦片选型 + 驻留式 LayerNorm + 两段 Matmul 独立几何，最终
   116 条均值 **1.1041**（交付字节验收口径复测）。

# 详细设计（required）

## 算子分析

### 数学公式

```
阶段一：  C0 = A0 × B0           （M0×N0）
阶段二：  μ[m]     = (1/N0) Σ_n C0[m, n]
          σ²[m]    = (1/N0) Σ_n (C0[m, n] − μ[m])²
          C0n[m,n] = (C0[m,n] − μ[m]) / sqrt(σ²[m] + ε) × γ[n] + β[n]，ε = 1e-6
阶段三：  C1 = C0n × B1          （M0×M1）
```

数值实现：LayerNorm 的矩统计采用**以每行首元素为平移中心的单趟遍历**
（`Σ(x−x₀)`、`Σ(x−x₀)²` 一次读入同时累加），再还原
`μ = x₀ + Σ(x−x₀)/n`、`σ² = Σ(x−x₀)²/n − (Σ(x−x₀)/n)²`。
相比原始 `E[x²]−E[x]²`，该平移形式避免了大均值小方差场景的相消误差
（对照实验：原始写法在 ATK 中有 2 条用例失败）；相比"先均值再中心矩"两趟写法，
面板只读一遍（全量实测 +3.7%）。

### 支持数据类型

| 张量 | 数据类型 | 布局 |
| --- | --- | --- |
| A0 | FP16 | RowMajor |
| B0 / B1 | FP16 | ColumnMajor（物理连续 `(N0,K0)` / `(M1,N0)`） |
| gamma / beta | FP32 | ND，长度 N0 |
| C1 | FP16 | RowMajor |

### 支持形状

官方 116 条测试集：`M0 ∈ {128,512,1024,2048}`、`K0 ∈ {768,2048,4096,8192}`、
`N0 ∈ {2048,3072,4096,8192}`、`M1 ∈ {768,2048,4096}` 的组合。
约束：`N0 ≤ 65535`、K0/M0/M1 任意正整数；当 `N0` 可被 16 整除且
`8·N0·2 + UB_BYTES ≤ UB`（N0 ≤ 8192）时走驻留 LN 快路径，否则自动回退到分块路径。

## 算子实现

### 实现方案（三段式 MIX kernel，中间结果驻内部 workspace）

```
AIC:  Matmul0(A0 x B0) → 内部 workspace（M0 x RoundUp(N0,16)，FP16，行 32B 对齐）
      ── SyncAll×2 ──
AIV:  LayerNorm(workspace, gamma, beta) → 原地写回（56 个 vector subblock）
      ── SyncAll×2 ──
AIC:  Matmul1(workspace x B1) → C1
```

* 三个重阶段串行，段间用成对 `SyncAll<false>()` 交接（与仓库内其它 MIX 融合内核一致；
  跨核细粒度 flag 交接实测挂死，静态对半流水实测 +6.1% 更差，均不采用）。
* 两段 Matmul 使用 `BlockMmadTla`（`MmadPingpong` 调度）+ `PackedTileCopyTla`，
  **各自持有独立 block 类型**（`BlockMmad0`/`BlockMmad1`），可各用一套瓦片几何；
  网格走 `GemmIdentityBlockSwizzle<8,1>`。
* 该方案下中间矩阵只在算子自己的 workspace 中中转，调用方看不到任何中间量，
  满足任务书「核心开发要求」第 1、2 条。

#### Host 侧设计

* `GetWorkspaceSize = M0 × RoundUp(N0,16) × sizeof(fp16)`，由框架在调用侧统一分配并随
  `Params.ptrStaging` 下发；
* host 只做形状校验与工作区大小计算，不做启发式 tiling 搜索——瓦片几何在 JIT 编译期
  按形状选（见下），运行时零分支；
* JIT 模板把 `M0/K0/N0/P1` 作为宏传给内核，内核的几何选择与行块选择全部是编译期常量。

#### Kernel 侧设计

1. **逐形状瓦片选型（编译期代价模型）**：模板内按
   `phase ≈ max( 2MNK/373e12 · 28·⌈T/28⌉/T , 2MNK·(1/BM+1/BN)/3.9e12 ) · fit(BM,BN)`
   从 12 种候选几何（BM ∈ {128,160,192,224,256} × BN ∈ {96,128,160,192,224,256}，
   各带 L1/L0 K 与缓冲级数）中为两段各选一套；`fit` 是用全量 116 条实测回标定的
   逐几何修正系数。第一项是 28 核波次量化后的 MAC 时间，第二项是 L1 填充时间。
2. **第二段行块自适应**（`SelectRowTile1(m0, bm, p1, bn, k1)`）：在 32 的整数倍里搜索行块，
   按 `max(单核时间, 聚合填充时间)` 取优；单核 L2→L1 填充带宽取实测 ≈139 GB/s/核，
   聚合取实测 2.0 TB/s。瓦片数少于 28（M0=128 档）时首选更细行块，把每核搬运量减半。
3. **驻留式 LayerNorm**（`NormalizeResident`）：当 `stride == n0 && n0 ≤ 8192` 且
   `ROWS·n0·2 + UB_BYTES ≤ UB` 时，行组面板一次读入 UB（偏移 0），
   矩统计（`ResidentMomentsVF`，单趟平移）+ 仿射（`ResidentAffineVF`）全部在 UB 内完成，
   再一次写回——GM 往返由 `2R+1W` 降到 `1R+1W`；面板由 `MTE2_V` / `MTE3_MTE2`
   事件令牌保护，面板加载、仿射结果与写回之间不引入任何全核屏障。
   模板按 `CATLASS_LNM_LN_STEP=4`（M0≤512）把行组再切成 4 行任务，让 56 个 vector worker
   铺满（19 个行组 × 4 行任务 ≥ 56）。不满足条件时回退到分块两趟路径。
4. **仿射与矩统计融合**：矩统计与仿射都用 `MulAddDst`（FMA）组织，减少指令数与依赖深度。

### 测试设计

* 精度：`tests/optest/**`（127 用例）+ 正式 ATK（253 条泛化，`--task accuracy
  --compare-backend cpu`）；
* 性能：116 条官方用例，任务级 `msprof`，每形状 3 次取最小，
  与任务测试集给定的标杆时延逐条比对；本机 `msprof op` 通道故障（平台侧
  `StartFFTSTask` 报错），采用与标杆相同口径的通用 msprof 采集。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | optest 127 用例全通过；ATK 泛化 253 条中 227 条精度通过（≥200）；其余 26 条为 fp16 逐元素阈值（abs_error ≤ 0.1 且 ≤ 32 ULP / matched_ratio ≥ 0.99）临界用例，其中 24 条在本算子任何版本上均失败 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 性能标准 | 116 条官方用例平均 `标杆/算子 = 1.1041`（要求 >1.1） | 任务测试集标杆数据 + 通用 msprof 采集 |

## 兼容性分析

新增算子（示例 88），不涉及既有算子行为变更；`include/` 下新增
`matmul_layer_norm_matmul_tla.hpp` 与 `block_epilogue_layer_norm_ascend950.hpp`，
optest 侧按 JIT 模板接入，不影响其它样例。

## 已知限制

1. 重阶段之间仍是整段同步：LN 的时间只能缩小、不能与 Matmul 重叠
   （细粒度交接实测挂死、静态对半流水波次损失更大）。
2. 小形状（基线 19–50 µs）被约 4.4 µs 的内核固定开销和各段流水延迟支配。
3. 驻留 LN 的条件是 `N0 ≤ 8192` 且 32B 对齐；超出时回退两趟路径（性能略降，功能不受影响）。

# 修订记录

| 日期 | 版本 | 说明 |
| --- | --- | --- |
| 2026-10-15 | v1.0 | 初版：三段式 MIX + 逐形状几何选型，116 均值 1.0382 |
| 2026-10-15 | v1.1 | 驻留式 LN（单趟矩统计）+ 自适应行瓦片 + LN 行步长 4：116 均值 **1.1041** |
