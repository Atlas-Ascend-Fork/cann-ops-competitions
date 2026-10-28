# MatmulLayerNormMatmul 算子设计文档

| 版本 | 日期 | 作者团队 | 说明 |
| --- | --- | --- | --- |
| v1.0 | 2026-10-26 | `lunhao2023` | 首版设计：需求分析、融合方案选型、CATLASS 组件选型、Host/Kernel 分层设计、片上资源与 workspace 核算、精度与测试设计 |

# 需求背景（required）

## 需求来源

| 项目 | 内容 |
| --- | --- |
| 任务书 | 9月社区任务-MatmulLayerNormMatmul算子开发任务书 |
| 任务测试集 | `test_case/MatmulLayerNormMatmul_测试集.csv`，共 116 条用例，标杆为 `torch.mm + F.layer_norm + torch.mm` 小算子拼接（Ascend 950PR 实测时延） |
| 目标开源仓 | [cann/catlass](https://gitcode.com/cann/catlass) |
| 适配硬件 | Ascend 950，性能测试使用 Ascend 950PR |
| 开发语言 | Ascend C，基于 CATLASS 分层组件（Arch/Block/Tile/Kernel）组装 |
| 精度标准 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 合入形态 | 参考 [CATLASS PR 678](https://gitcode.com/cann/catlass/pull/678) 的目录结构提交 PR |

## 背景介绍

### 计算模式

Matmul → LayerNorm → Matmul 是 Transformer 与 MLP 结构中常见的融合计算链（如归一化投影层）。独立执行时需要依次启动三个 kernel：第一个 Matmul 将中间矩阵写回显存，LayerNorm 读取该矩阵并将归一化结果再次写回显存，第二个 Matmul 再读出该结果。在任务测试集覆盖的小批量形状（M0 最小 128）上，两次额外 kernel launch 与中间矩阵 `C0`、`C0_norm` 的显存往返占总时延比例较高。

本任务在 CATLASS 仓中新增融合算子 `MatmulLayerNormMatmul`，单次 kernel launch 完成全部三段计算：先执行 `C0 = A0 × B0`，对 `C0` 逐行做 LayerNorm，再以归一化结果为左矩阵执行 `C1 = C0_norm × B1`。Mean、Variance 及中间矩阵均在算子内部管理，不对外暴露。

### 参数与输入输出规格

| 参数 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度 |
| --- | --- | --- | --- | --- | --- |
| A0 | 输入 | 第一个 Matmul 左矩阵，RowMajor | FP16 | ND | (M0, K0) |
| B0 | 输入 | 第一个 Matmul 右矩阵，ColumnMajor | FP16 | ND | (K0, N0) |
| B1 | 输入 | 第二个 Matmul 右矩阵，ColumnMajor | FP16 | ND | (N0, M1) |
| gamma | 输入 | LayerNorm 缩放参数 | FP32 | ND | (N0,) |
| beta | 输入 | LayerNorm 偏置参数 | FP32 | ND | (N0,) |
| C1 | 输出 | 唯一对外输出，RowMajor | FP16 | ND | (M0, M1) |

# 需求分析（required）

## 需求描述

使用 Ascend C 基于 CATLASS 组件实现 `MatmulLayerNormMatmul` 融合算子，满足：

1. 单次 kernel launch 完成 Matmul → LayerNorm → Matmul 全部三段计算。
2. Mean、Variance 及中间矩阵由算子内部 workspace/片上资源管理，不对外暴露。
3. Matmul 阶段支持多种实现方案（TileShape 档位、Swizzle 方向、SplitK），按 shape 分档选优以达成更好性能。
4. 精度满足生态算子开源精度标准。
5. 整体性能达成小算子拼接方案的 1.1 倍（平均标杆时延 / 测试时延 > 1.1，基于任务测试集 116 例，Ascend 950PR）。
6. 交付基于 CATLASS-optest 的测试件（116 例全通过），并基于 ATK 生成泛化测试（≥ 200 例）。

## 需求拆解

| 子需求 | 设计说明 |
| --- | --- |
| 第一个 Matmul | `C0 = A0 × B0`，FP16×FP16→FP32 累加，结果 FP16 落入内部 workspace |
| LayerNorm | 对 `C0` 每行沿 N0 维计算 mean/var（ε=1e-6），应用 γ/β 后得到 `C0_norm` |
| 第二个 Matmul | `C1 = C0_norm × B1`，左矩阵为归一化后的内部结果 |
| 数值精度 | Cube 侧 FP32 累加；LayerNorm 统计量与归一化计算在 FP32 域进行 |
| 中间结果 | `C0`/`C0_norm` 位于内部 workspace，Mean/Variance 在 UB 内就地维护 |
| 性能优化 | 多档 TileShape、Swizzle 方向、SplitK 按 shape 分档；跨核流水重叠 |
| 测试验证 | optest 精度测试 + ATK 泛化 + msprof op 性能采集 |

# 详细设计（required）

## 算子分析

### 数学公式

阶段一（第一个 Matmul）：

```text
C0 = A0 × B0          # (M0, K0) × (K0, N0) → (M0, N0)，RowMajor
```

阶段二（LayerNorm，对 C0 每行独立计算）：

```text
μ[m]     = (1 / N0) · Σ_n C0[m, n]
σ²[m]    = (1 / N0) · Σ_n (C0[m, n] − μ[m])²
C0_norm[m, n] = (C0[m, n] − μ[m]) / sqrt(σ²[m] + ε) · γ[n] + β[n]      # ε = 1e-6
```

阶段三（第二个 Matmul）：

```text
C1 = C0_norm × B1      # (M0, N0) × (N0, M1) → (M0, M1)，RowMajor
```

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| A0 / B0 / B1 / C1 | FP16 | 输入输出矩阵 |
| gamma / beta | FP32 | 可学习参数，直接以 FP32 参与 UB 内归一化计算 |
| Cube 累加与 LN 统计量 | FP32 | L0C 累加、mean/var/rstd 及归一化中间量均为 FP32 |

### 支持形状

任务测试集覆盖的维度域（共 116 例）：

| 维度 | 取值 |
| --- | --- |
| M0 | 128, 512, 1024, 2048 |
| K0 | 768, 2048, 4096, 8192 |
| N0 | 2048, 3072, 4096, 8192 |
| M1 | 768, 2048, 4096 |

维度特征分析：

- 四个维度均为 128 的倍数（768 = 6×128，其余为 256/1024 的倍数），主路径无尾块；kernel 侧仍保留尾块与对齐处理，保证维度域外的一般 2D shape 可正确执行。
- N0 ≥ 2048 且最大 8192：LayerNorm 的行归一化范围远大于单块 UB tile，行内统计必须跨 N 分块聚合。
- M0 最小 128：`(M0, N0)` 方向分块产生的任务块可能少于 AIC 核数，小 M0 场景需要 SplitK 提升并行度。

## 算子实现

### 实现方案选型

核心挑战：LayerNorm 是**整行依赖**的后处理——第 m 行的 mean/var 需要该行全部 N0 个元素；而 GEMM 按 `(M tile, N tile)` 二维分块并行，同一行的元素天然分散在多个任务块（最多 `N0/BN ≈ 32` 块）中。可选方案对比如下：

| 方案 | 思路 | 结论 |
| --- | --- | --- |
| 方案一：行带驻留 | 每个 AIC 独占一个 M 行带、内循环扫完全部 N0，行统计本地完成，无需跨核归约 | 并行度 = M0/BM（M0=128 时仅 1~2 块），AIC 利用率崩溃，不采纳；仅在 N0 很小的特殊场景有理论价值，本测试集 N0 ≥ 2048 不适用 |
| 方案二：三阶段 + 内部 workspace | 阶段1 AIC 完成 mm1 并将 C0(FP16) 写入内部 workspace；跨核同步；阶段2 AIV 逐行块在 UB 内完成 LN，原地写回 workspace；跨核同步；阶段3 AIC 以 workspace 为左矩阵完成 mm2 | 与仓库 87 号样例（batched_matmul_softmax）已验证的"两阶段 + CrossCore 同步"框架同构，仅需扩展为三阶段；C0 往返局限于紧邻阶段、命中 L2，Mean/Variance 全程 UB 内维护。**选定为主方案** |
| 方案三：UB 级 tile 流水 | 每块 C0 tile 经 Fixpipe 进 UB 由 AIV 即时消费，跨 tile 在线统计行状态 | N0 最大 8192 时单行 16KB(FP16)，整行无法驻留单核 UB；跨 AIC 的在线统计状态传递复杂度与出错风险高，不作为首选。其中"细粒度 flag 重叠相邻阶段"的子项作为 P1 优化吸收进方案二 |

选定方案二的关键依据：

1. 中间矩阵的显存往返无法真正消除（N0 最大 8192，C0 行数据横跨全部 N 分块核），但 workspace 化后 C0 的写与读发生在同一 kernel 的紧邻阶段间，数据驻留 L2（950 大容量 L2），实际 HBM 流量显著低于三 kernel 拼接。
2. 相对拼接基线的收益来源明确：kernel launch 3 → 1；LayerNorm 与两个 Matmul 之间无调度间隙；C0/C0_norm 不再以外部 tensor 形式落盘；小 M0 场景可叠加 SplitK。
3. Mean/Variance 属于行内归约，在 UB 内随行块一次完成，无需 workspace，满足"中间结果不对外暴露"的要求。

### CATLASS 组件选型

| 组件 | 选型 | 说明 |
| --- | --- | --- |
| ArchTag | `Arch::Ascend950` | 目标硬件 |
| DispatchPolicy | `Gemm::MmadPingpong<Arch::Ascend950, true, false, 1, false, 2, 2, 2, 2>` | 与 87 号样例在 950 上验证的配置一致；L1A/L1B/L0A/L0B 双 buffer，L0C 单缓冲，实测后允许按档位微调 |
| L1TileShape | 档位A `Shape<128, 256, 128>`；档位B `Shape<128, 128, 128>` | 按 taskBlocks 分档（见 Host 侧），mm1/mm2 可独立选档 |
| L0TileShape | 档位A `Shape<128, 256, 32>`；档位B `Shape<128, 128, 64>` | 与 L1 档位配套，容量核算见下文 |
| BlockMmad | `Gemm::Block::BlockMmadTla` | 复用，两段 Matmul 同型（A RowMajor × B ColumnMajor） |
| BlockScheduler | `Gemm::Block::GemmIdentityBlockSwizzle<3, 0/1>` | M ≥ N 取方向 0，M < N 取方向 1，两段分别判定 |
| LN Epilogue | 新增 `Epilogue::Block::BlockEpilogueLayerNorm` | AIV 侧行块 LayerNorm，风格对齐 `block_epilogue_batched_matmul_softmax.hpp` |
| Kernel | 新增 `Gemm::Kernel::MatmulLayerNormMatmulTla` | 组合两个 mmad 任务循环 + LN 阶段 + 两次跨核同步 |

Ascend 950 片上资源（`include/catlass/arch/arch.hpp`）：UB 248KB、L1 512KB、L0A/L0B 64KB、L0C 256KB；AIC 核数以运行时 `GetCoreNumAic()` 为准（950 为 24）。

### Host 侧设计

host 侧完成参数校验、tiling 解析、档位选择、workspace 计算与 kernel 下发：

1. 校验输入输出 shape、dtype、layout 与连续性（校验规则见"算子约束限制"）。
2. 由 M0/K0/N0/M1 计算两段 Matmul 的任务块数，选择 TileShape 档位与 Swizzle 方向。
3. 计算 LayerNorm 行块参数 `rowsPerTile`（见 Kernel 侧 LN 设计）。
4. 计算 workspace 大小并下发 kernel（blockNum = `min(aicCoreNum, max(mm1Tiles, mm2Tiles))`）。

**档位选择**（风格对齐 87 号样例的 `Run()` 分档）：

| 条件（按任务块数 t 与 AIC 数 a） | mm1 档位 | mm2 档位 | 说明 |
| --- | --- | --- | --- |
| t ≤ 2 | 128×128 | 128×128 | 小负载，小 tile 提升核间均衡 |
| 2 < t ≤ 20 | 128×128 或 128×256 | 同左 | 过渡档，按 msprof 实测微调边界 |
| t > 20 | 128×256 | 128×256 | 大负载，大 tile 提高吞吐 |
| mm1: `t < a/2` 且 `K0 > 5120` | 叠加 SplitK（K 切 2~4 段） | — | 典型命中：M0=128 且 K0=8192 |
| mm2: `t < a/2` 且 `N0 > 5120` | — | 叠加 SplitK（N0 切 2~4 段） | 典型命中：M0=128 且 N0=8192 |

具体分档边界为初始经验值，以 116 例 msprof 实测数据回填修正；每档对应的 TileShape、SplitK 段数与切换条件将在性能测试报告中备注（任务书性能要求）。

**TilingData**（host → kernel 传递）：

| 字段 | 说明 |
| --- | --- |
| m0, k0, n0, m1 | 问题规模 |
| mm1TileM/N/K, mm2TileM/N/K | 两段 Matmul 生效档位 |
| mm1SplitK, mm2SplitK | SplitK 段数（1 = 不切分） |
| swizzleDir1, swizzleDir2 | 两段 Swizzle 方向 |
| lnRowsPerTile, lnAlignedN | LN 行块行数与 N 对齐值 |
| blockNum | 启动核组数 |

**workspace**（`GetWorkspaceSize`，512B 对齐）：

| 区域 | 大小 | 用途 |
| --- | --- | --- |
| `wsC0` | `M0 × N0 × 2B` | 阶段1 mm1 输出（FP16），阶段2 原地归一化，阶段3 作为 mm2 左矩阵 |
| `wsC0Partial` | `M0 × N0 × 4B × (mm1SplitK − 1)`（仅 SplitK > 1 时） | mm1 SplitK 部分和（FP32） |
| `wsC1Partial` | `M0 × M1 × 4B × (mm2SplitK − 1)`（仅 SplitK > 1 时） | mm2 SplitK 部分和（FP32） |

最大规格（case 116：M0=2048, N0=8192, M1=4096，不分段）workspace ≈ 32MB。

### Kernel 侧设计

kernel 为 MIX 形态单次 launch，AIC 与 AIV 分支按阶段协同执行：

```text
AIC 分支:
  Phase1: for tile in mm1Tiles[本核]:  BlockMmad(A0, B0) --fixpipe FP16--> wsC0
  跨核同步①（AIC 间 Barrier + SetFlag AIC_DONE）
  等待 LN_DONE（跨核同步②）
  Phase3: for tile in mm2Tiles[本核]:  BlockMmad(wsC0, B1) --> C1（SplitK 时写 wsC1Partial）
  （SplitK > 1 时追加 AIV 归并子阶段，见下文）

AIV 分支:
  等待 AIC_DONE（跨核同步①）
  Phase2: for rowTile in lnRowTiles[本核]:
      GM(wsC0) --copy--> UB(src)
      FP32 域统计 mean/rstd（行内 reduce，跨 N 分块按行块整体加载，无需跨核归并）
      y = (x − μ)·rstd·γ + β；FP16 cast 后原地写回 GM(wsC0)
  SetFlag LN_DONE（跨核同步②）
```

**跨核同步**（复用 `catlass/arch/cross_core_sync.hpp` 原语，风格同 87 号样例）：

| 同步 | 方向 | 原语与模式 | PIPE |
| --- | --- | --- | --- |
| ① AIC_DONE | AIC → AIV | `CrossCoreBarrier<0>`（AIC 间）+ `CrossCoreSetFlag<0x2>` / `CrossCoreWaitFlag<0x2>` | Set: FIX，Wait: MTE2 |
| ② LN_DONE | AIV → AIC | 每个 AIV 组完成全部行块后 `CrossCoreSetFlag<0x2>`，AIC `CrossCoreWaitFlag<0x2>` | Set: MTE3，Wait: MTE2 |
| ③ MM2_DONE | AIC → AIV | 仅 mm2 SplitK > 1 时使用，Phase3 结束后通知 AIV 归并 | Set: FIX，Wait: MTE2 |

flag ID 预算 3 个（AIC_DONE=0，LN_DONE=1，MM2_DONE=2），低于同类型 ID ≤ 8 的硬件上限。语义说明：wsC0 的写（MTE3）与读（MTE2）跨 AIC/AIV，通过上述 PIPE 配对保证数据可见性；Mean/Variance 仅存在于 AIV 本地 UB，无跨核一致性问题。

**Phase2 LayerNorm 设计**：

以"若干整行 × alignedN"为 AIV 处理块（对齐 `block_epilogue_batched_matmul_softmax` 的行块模型），行块间按 `vectorCoreNum` 均匀分配，无跨 AIV 状态。UB 布局：

| UB 区域 | 数据类型 | 作用 |
| --- | --- | --- |
| srcUb | FP16 | 行块输入（自 wsC0 搬入） |
| dstUb | FP16 | 归一化结果（写回 wsC0 原位） |
| xFp32Ub | FP32 | src 的 FP32 展开（统计与归一化计算域） |
| gammaUb / betaUb | FP32 | γ/β 行块切片，循环外驻留、按 N0 视 UB 余量整载或分段 |
| tmpUb | uint8 | 高层 API / reduce 指令临时空间 |

计算流程（单行块）：

1. `GM → UB` 搬入 `rowsPerTile` 行，尾块行数不足时按实际行数处理；
2. cast 为 FP32，`ReduceSum` 求行和得 `μ[m]`，再 `ReduceSum((x−μ)²)` 得 `σ²[m]`，`rstd = 1/sqrt(σ² + 1e-6)`（等价地优先使用 AscendC 高层 API `LayerNorm` 一次完成统计与仿射变换，eps 由 tiling 传入；若该 API 在 950 上对当前 dtype/shape 组合受限，则回退到 Reduce 系指令手写路径，两者精度路径一致）；
3. 应用 `γ/β`，cast 回 FP16，原地写回 wsC0。

`rowsPerTile` 按 UB 预算选择。单行块 UB 占用为 `rowsPerTile × N0 × (2B src + 4B FP32 计算域 + 2B dst) = rowsPerTile × N0 × 8B`，另加 γ/β 驻留 `N0 × 8B` 与 tmp 预留 32KB：

| N0 | rowsPerTile | UB 合计 |
| --- | --- | --- |
| ≤ 2048 | 8 | 8×2048×8B + 16KB + 32KB = 176KB ✓ |
| 3072 ~ 4096 | 4 | 4×4096×8B + 32KB + 32KB = 192KB ✓ |
| 8192 | 2 | 2×8192×8B + 64KB + 32KB = 224KB ✓ |

LN 为带宽受限阶段且写回目的地与读取源相同（原地归一化），单缓冲 + MTE2/MTE3 交替即可，双缓冲收益有限；若实测表明搬入搬出可重叠，可在小 N0 档位回收 UB 预算改开双缓冲。M0 行数不被 rowsPerTile 整除时尾块按实际行数处理；N0 非 32B 对齐倍数时（维度域外）按 alignedN 补齐并在计算中屏蔽 padding 列。

**SplitK 变体**（小 M0 场景，按 Host 档位触发）：

- mm1 SplitK：AIC 按 K 段写 FP32 部分和到 `wsC0Partial`；同段归并改由**承担该 M 行带的 AIV 在 Phase2 入口完成**（读部分和累加 + cast FP16 得 C0 行块，随后正常 LN），不引入新的同步阶段；
- mm2 SplitK：Phase3 写 `wsC1Partial` 后追加子阶段——AIC SetFlag MM2_DONE，AIV 等待后归并部分和 + cast FP16 写 C1，kernel 退出。该变体增加一次 AIV 读写 C1 的开销，仅在小任务块场景下净收益为正时启用。

**P1 优化路径（实测不达标时启用）**：

当前主方案的 Phase2（纯 AIV 访存阶段）与 Phase3 之间为全量 barrier，大 shape 时 LN 串行段约占端到端 5%~10%。若实测显示该段成为瓶颈，改为按 **M-tile 粒度** 细粒度同步：AIV 每完成一个 M-tile 覆盖的行块即对该 M-tile 的原子计数器 +1，AIC 的 mm2 任务块在启动前仅等待其左矩阵行对应 M-tile 的计数到达，实现 LN 与 mm2 的行级流水重叠。该优化不改变 workspace 布局与正确性语义，作为独立开关由 tiling 控制。

**分支实例化条件与合法组合**：

| 分支条件 | 取值 |
| --- | --- |
| dtype | 固定 FP16 输入/输出（v1 不含 BF16，接口预留 dtype 参数） |
| Layout | 固定 A0/B0/B1 RowMajor/ColumnMajor/ColumnMajor，C0/C1 RowMajor |
| TileShape 档位 | mm1、mm2 各自独立 ∈ {128×128×128, 128×256×128} |
| Swizzle 方向 | mm1、mm2 各自独立 ∈ {<3,0>, <3,1>} |
| SplitK | mm1、mm2 各自独立 ∈ {1, 2, 4} |
| LayerNorm 路径 | {高层 API, 手写 Reduce}（编译期二选一，tilingkey 不区分） |

合法组合 = 各维取值的笛卡尔积，由 host 侧 `tilingkey`（档位/Swizzle/SplitK 编码）选择，kernel 模板按档位显式实例化，其余维度域外 shape 走默认档位 + 尾块处理。

### 资源与容量核算

TileShape 档位核算（FP16，2B）：

| 资源 | 档位A 128×256×128 | 档位B 128×128×128 | 上限 |
| --- | --- | --- | --- |
| L1（L1A+L1B，双 buffer） | 128×128×2×2 + 128×256×2×2 = 192KB | 128×128×2×2×2 = 128KB | 512KB ✓ |
| L0A（双 buffer） | 128×32×2×2 = 16KB | 128×64×2×2 = 32KB | 64KB ✓ |
| L0B（双 buffer） | 32×256×2×2 = 32KB | 64×128×2×2 = 32KB | 64KB ✓ |
| L0C（单缓冲，FP32） | 128×256×4 = 128KB | 128×128×4 = 64KB | 256KB ✓ |

LN 阶段 UB 核算见上文（最坏 224KB < 248KB）；硬件事件按 MTE2/V/MTE3 各 2 个、flag 3 个，均在同类型 ≤ 8 上限内。

性能可行性：任务验收口径为 116 例**平均**加速比 > 1.1。小 M0 例（基线 19~66μs）收益以 launch 次数 3→1 与中间张量免落盘为主，加速比弹性最大；大 shape 例（基线 100μs+）计算主导，收益取决于两段 GEMM 相对 torch.mm 的效率（CATLASS matmul 标杆区间 0.98~1.2×）叠加 LN 串行段缩减与 L2 局部性收益；SplitK 补齐小 M0 时的 AIC 利用率短板。预期整体平均 > 1.1，最终以 950PR 实测为准（数据随自验证报告提交）。

### 精度设计

| 环节 | 策略 | 与基线对齐 |
| --- | --- | --- |
| 两段 Matmul | L0C FP32 累加，Fixpipe 随路 cast FP16 | 同 torch.mm（FP32 累加） |
| LayerNorm 统计 | FP32 域求 μ/σ²/rstd，ε=1e-6 | 同 F.layer_norm（FP32 统计） |
| 归一化 | `(x−μ)·rstd·γ + β` 全程 FP32，γ/β FP32 原生参与，末级 cast FP16 | 同 F.layer_norm（fp16 输入 fp16 输出） |
| SplitK | 部分和 FP32 落盘、FP32 归并后 cast | 不引入额外精度损失 |

CPU golden（测试件内实现）：

```python
c0 = A0.astype(np.float32) @ B0.astype(np.float32)
mu = c0.mean(axis=-1, keepdims=True)
var = ((c0 - mu) ** 2).mean(axis=-1, keepdims=True)
norm = (c0 - mu) / np.sqrt(var + 1e-6) * gamma + beta
golden = (norm.astype(np.float16).astype(np.float32) @ B1.astype(np.float32)).astype(np.float16)
```

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950 | √ |

# 算子约束限制

| 约束项 | 说明 |
| --- | --- |
| 数据类型 | A0/B0/B1/C1 仅支持 FP16，gamma/beta 仅支持 FP32 |
| 数据布局 | A0 RowMajor、B0 ColumnMajor、B1 ColumnMajor、C1 RowMajor，ND 格式 |
| 连续性 | 仅支持连续 Tensor（任务书"非连续 Tensor"列为"—"） |
| 维度 | 输入输出均为 2 维（gamma/beta 为 1 维），不支持 batch 维 |
| 维度值 | 各维度 > 0；测试集维度域内为性能优化目标域，域外走默认档位 + 尾块处理 |
| 扩展功能 | 不支持 bias、eps 可配、elementwise_affine 关闭等 LayerNorm 扩展（eps 固定 1e-6，γ/β 必选） |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)；任务测试集 116 例精度全通过 | 任务书 |
| 性能标准 | 任务测试集上平均标杆时延 / 平均测试时延 > 1.1（标杆 `torch.mm + F.layer_norm + torch.mm`，Ascend 950PR） | 任务书 |

## 测试设计

**功能与精度（optest）**：按 PR 678 形态接入 CATLASS-optest，基于任务测试集 116 例生成 pytest 测试件（`catlass-example-to-pytest` skill），另补充边界用例（最小 shape、M0=128 小负载分档、SplitK 触发组合、N0=8192 大 LN 行块），预期 ≥ 130 例全通过。

**泛化测试（ATK）**：基于已接入 optest 的测试件，经 `catlass-atk-support` skill 生成 ATK 测试框架交付件，泛化 ≥ 200 例测试条目精度通过。

**性能测试**：`msprof op` 采集（参考 CATLASS 样例性能调试文档），逐例对比小算子拼接标杆；记录每类 case 实际使用的方案档位（TileShape/Swizzle/SplitK）并按要求备注；输出 116 例加速比分布与整体平均值。

**交付件结构**（对应 PR 合入目录）：

```text
.
├── experimental
│   └── matmul
│       └── ascend950_matmul_layer_norm_matmul
│           ├── CMakeLists.txt
│           ├── ascend950_matmul_layer_norm_matmul.cpp    # 样例入口（含 CPU golden 自校验）
│           ├── README.md
│           └── test_ascend950_matmul_layer_norm_matmul.py
├── include
│   └── catlass
│       ├── epilogue
│       │   └── block
│       │       └── block_epilogue_layer_norm.hpp          # 新增：AIV 行块 LayerNorm
│       └── gemm
│           └── kernel
│               └── matmul_layer_norm_matmul_tla.hpp       # 新增：三阶段融合 kernel
└── tests
    └── optest
        ├── include
        │   └── catlass_kernel_jit.h                       # 注册 Ascend950MatmulLayerNormMatmul 签名
        ├── kernels
        │   ├── ascend950_matmul_layer_norm_matmul
        │   │   ├── CMakeLists.txt
        │   │   ├── ascend950_matmul_layer_norm_matmul.cpp
        │   │   └── ascend950_matmul_layer_norm_matmul_impl.cpp
        │   ├── CMakeLists.txt
        │   └── common
        │       └── kernel_runner.h
        ├── README.md
        ├── src
        │   ├── catlass_ascend950_matmul_layer_norm_matmul.cpp
        │   └── include
        │       └── template
        │           └── ascend950_matmul_layer_norm_matmul.h
        └── torch_catlass
            ├── __init__.py
            └── ops
                ├── __init__.py
                └── ascend950_matmul_layer_norm_matmul.py
```

## 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 大 shape 下两段 GEMM 效率不及 torch.mm，平均加速比被摊薄 | 性能不达标 | 多档位 + Swizzle + SplitK 分档调优；P1 细粒度同步重叠 LN 与 mm2；必要时对 mm1 尝试 `MmadPingpongPreLoad` 变体对比 |
| Phase2 LN 为 AIV 串行访存段，大 N0 时占比升高 | 大 shape 时延增加 | rowsPerTile 与双缓冲调优；P1 行级流水重叠；γ/β UB 驻留避免重复加载 |
| mm2 左矩阵来自 workspace，MTE2 读带宽与 AIV 写带宽在阶段边界争用 | 阶段切换气泡 | 全量 barrier 语义下无并发争用；P1 重叠时通过 M-tile 粒度错峰 |
| AscendC `LayerNorm` 高层 API 在 950 上 dtype/shape 受限 | LN 路径受阻 | 保留手写 Reduce 路径为等价回退，精度路径一致 |
| workspace 受 L2 容量约束，超大 shape C0 往返落 HBM | 大 shape 收益缩水 | 两段 GEMM 分档优化读写带宽（Swizzle/ShuffleK 由 BlockMmad 继承）；该场景收益本就以计算主导 |

## 兼容性分析

新增融合算子与新增样例目录，不修改 CATLASS 既有组件接口与既有样例行为；对不满足 dtype/layout/连续性约束的输入，host 侧直接返回不支持，不进入 kernel。

# 参考资料

- 9月社区任务-MatmulLayerNormMatmul算子开发任务书及测试集
- [算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- CATLASS 仓库：<https://gitcode.com/cann/catlass>
- 先例样例：`examples/experimental 78_ascend950_dual_matmul_silu_mul`（AIC 双 Matmul + AIV UB 后处理）、`87_ascend950_batched_matmul_softmax`（Matmul + 行级 Softmax 跨核同步框架，本设计直接同构扩展）
- [CATLASS-optest 测试工程](https://gitcode.com/cann/catlass/blob/master/tests/optest/README.md)、[catlass-example-to-pytest](https://gitcode.com/cann/catlass/blob/master/.agents/skills/catlass-example-to-pytest/SKILL.md)、[catlass-atk-support](https://gitcode.com/cann/catlass/tree/master/.agents/skills/catlass-atk-support/SKILL.md)
- [参考合入 PR 678](https://gitcode.com/cann/catlass/pull/678)
- [msprof op 性能采集](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProf/docs/zh/profiling/msprof_cmd/general_collect_commands.md)、[CATLASS 样例性能调试](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/evaluation/performance_tools.md)
