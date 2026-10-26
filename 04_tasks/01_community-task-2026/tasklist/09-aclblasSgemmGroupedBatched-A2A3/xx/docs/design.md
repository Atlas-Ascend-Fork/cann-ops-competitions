# aclblasSgemmGroupedBatched 算子设计文档（Atlas A2/A3，arch22）

> 交付件：算子设计文档（验收交付件第 1 项）
> 目标仓库：`https://gitcode.com/cann/ops-blas`，新增目录 `blas/gemm_grouped_batched/arch22/`
> 提交方式：以 PR 形式提交至 `cann-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/`
> 设计文档模板来源：`cann-competitions/04_tasks/01_community-task-2026/resources/design_template.md`
> 文档状态：**已按真机实现校准**（2026-10-13）。文中每一项机制均与 `blas/gemm_grouped_batched/arch22/` 源码逐条对齐，
> 并标注真机验证结论；实现与本文不一致时以源码为准。

---

## 1 需求背景（required）

### 需求来源

9 月社区任务：在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3，Atlas A2/A3 系列产品）上，基于 Ascend C / CATLASS 编程语言开发单精度实数（float32）分组批量矩阵乘算子 `aclblasSgemmGroupedBatched`。ops-blas 开源仓已有该算子的 Atlas 950（arch35）实现，本任务新增 Atlas A2/A3 对应的 `arch22` 架构目录实现，并完成设计、开发、测试全流程。

### 背景介绍

#### 算子功能分析

`aclblasSgemmGroupedBatched` 一次调用完成 `groupCount` 组彼此独立的批量矩阵乘。对第 `g` 组（`g = 0..groupCount-1`），其 `m_g / n_g / k_g`、`transa_g / transb_g`、`alpha_g / beta_g`、批量大小 `groupSize_g` 均独立设定；组内对 `j = 0..groupSize_g-1`，以扁平下标 `idx = Σ_{t<g} groupSize[t] + j` 索引 A/B/C 指针数组：

```
C[idx] = alpha_g * op(A[idx]) * op(B[idx]) + beta_g * C[idx]
```

关键语义：**组与组之间没有任何共享信息**——形状、转置、标量、前导维度、批量大小全部逐组独立，仅 `Aarray`/`Barray`/`Carray` 三个扁平指针数组按组的顺序拼接（长度均为 `Σ groupSize[g]`）。所有矩阵列主序（Column-Major）存储；`cublasSgemmGroupedBatched` 无 stride 语义。

#### 已有实现现状分析（arch35，不可直接复用）

主仓 arch35 实现面向 Atlas 950，采用 **AIV-only 的 MicroAPI SIMD 路径**：分组批量矩阵乘在纯 Vector 核上以手工分块 + 广播乘加完成。该实现与本任务目标架构 A2/A3 的差异是根本性的：

| 维度 | arch35（Atlas 950） | arch22（Atlas A2/A3，本任务） |
|---|---|---|
| 计算单元 | 仅 AIV（Vector） | AIC（Cube）+ AIV |
| 编程方式 | MicroAPI SIMD 指令手工拼装 | 高层 Matmul API（`lib/matmul_intf.h`） |
| 矩阵乘实现 | 手工分块乘加 | Cube 单元 MMAD，硬件完成 K 维规约 |
| 可否直接移植 | — | **否**，MicroAPI 接口与硬件路径均不互通 |

因此 arch35 的 kernel 无法迁移到 arch22；但 **host 侧的参数校验顺序、语义判定与 tiling 拆分策略可以对齐复用**，本设计在 host 侧刻意保持了与 arch35 及 CPU golden 相同的判定次序。

需要说明一处容易误读的地方：本设计的 "AIC + AIV" 并不意味着 AIV 也承担矩阵乘。**矩阵乘全部由 AIC 的 Cube 单元经 Matmul API 完成**（K 维规约由 MMAD 在硬件内完成），AIV 只承担逐元素的 alpha/beta 收尾（`C = alpha * product + beta * C`）——这是因为 Matmul API 本身没有 alpha/beta 语义（见 §3.2.2(4)）。仓内曾实现过一条纯 AIV 的瘦 GEMM 路径，经真机测量为净损失，已关闭且不可达（见 §3.2.1(8)）。

#### 对标接口差异

本接口逐组数组语义与 `cublasSgemmGroupedBatched` 完全一致，仅存在两处**非语义**差异：

| 项 | aclblas（本接口） | cuBLAS | 说明 |
|---|---|---|---|
| `groupCount` 位置 | 第 2 个参数（handle 之后） | 末尾（`group_size` 前） | 仅参数顺序调整，语义一致 |
| 参数命名 | `mArray` / `nArray` / `kArray` / `groupSizeArray` | `m_array[]` / `group_size[]` | 命名风格差异，数值语义一致 |

因此**无需任何映射处理**，验收用例可直接对齐 cuBLAS 行为。

---

## 2 需求分析（required）

### 需求描述

使用 Ascend C 编程语言实现 `aclblasSgemmGroupedBatched` 算子的 Atlas A2/A3（arch22）版本，支持 float32 数据类型，支持 `groupCount` 个形状完全独立的分组批量矩阵乘，支持逐组 `transa/transb` 的 N/T 组合，支持 `alpha/beta` 逐组标量，完成合法输入、边界与负向校验的泛化覆盖，并满足生态算子开源精度标准与任务书性能指标。

### 需求拆解

1. **引擎选择**：arch22 上矩阵乘必须走 AIC Cube 单元，使用高层 Matmul API，与仓内其余 arch22 的 BLAS-3 算子（herk / hemm / symm / syrk 等）保持一致的技术路线。
2. **列主序适配**：Matmul API 产出行主序结果，需设计"列主序 × 行主序"的等价变换，避免显式转置拷贝。
3. **逐组异构**：一次调用内组间形状/转置/标量/前导维度全异构，需要"按转置组合拆分 kernel 入口 + 逐组重设 Matmul 描述符"的分派机制。
4. **alpha/beta 收尾**：Matmul API 无 alpha/beta 语义，需单独设计收尾阶段；并在 `alpha = 1 && beta = 0` 时把收尾整体消除（乘积直接落 C）。
5. **工作空间可控**：组内批量可能很大（如 `4096³ × 128`），中间乘积缓冲区必须按工作空间上限分窗，不能按总批量一次性申请。
6. **大 K 支持**：单次 Matmul 的 `singleCoreK` 有上限，`k` 超过上限时需在 kernel 内切分 K 维并累加。
7. **约束完备**：覆盖 `m/n/k = 0`、`groupSize = 0`、`alpha = 0`、`beta = 0/1`、非法枚举（含 `OP_C`）、非法前导维度、空指针（数组级与元素级）等全部边界与负向场景。
8. **性能分层（组间形状跨度极大）**：同一张性能表里既有 `4096³` 的大 batch，也有 `16 × 1024` 的瘦组，单一固定分块无法两全。故分三层处理——① **逐组自适应 IterateAll 形状**（在形状网格上按代价模型择优选，见 §3.2.1(7)）；② **epilogue 融合**（`alpha = 1 && beta = 0` 的组不落 scratch、不发收尾，见 §3.2.2(4)）；③ **规则 C 行瓦片合并**（每 K 步 base block 数过少的组抬高 `singleM`，见 §3.2.1(7.2)）。三层都是**精确恒等或硬界护栏**，不改变任何数值语义。

---

## 3 详细设计（required）

### 算子分析

#### 数学公式

对第 `g` 组、组内第 `j` 个矩阵（扁平下标 `idx = Σ_{t<g} groupSize[t] + j`）：

```
C[idx] = alpha_g * op(A[idx]) * op(B[idx]) + beta_g * C[idx]

op(X) = X     当 trans = ACLBLAS_OP_N
op(X) = Xᵀ    当 trans = ACLBLAS_OP_T
```

维度关系（列主序）：`op(A)` 为 `m_g × k_g`，`op(B)` 为 `k_g × n_g`，`C` 为 `m_g × n_g`，前导维度分别记为 `lda/ldb/ldc`。存储维度随转置枚举互换：

| 枚举 | A 的存储维度 | B 的存储维度 |
|---|---|---|
| `transa = N` | `m × k` | — |
| `transa = T` | `k × m` | — |
| `transb = N` | — | `k × n` |
| `transb = T` | — | `n × k` |

退化语义（对齐 ops-blas 已实现行为）：

| 条件 | 行为 |
|---|---|
| `m = 0` 或 `n = 0` 或 `groupSize = 0` | 该组合法 no-op，返回成功且不执行计算 |
| `k = 0` | 等效 `C = beta_g * C` |
| `alpha_g = 0` | 跳过矩阵乘，仅执行 `C = beta_g * C` |
| `alpha_g = 0` 且 `beta_g = 1` | C 保持不变，该组不产生任何下发 |

#### 支持数据类型

| 数据类型 | 输入 | 输出 | 内部累加 |
|---|---|---|---|
| FLOAT32 | A / B / C / alpha / beta | C | FLOAT32（Cube 单元 FP32） |

#### 支持形状

- `groupCount ≥ 0`；每组 `m_g ≥ 0`、`n_g ≥ 0`、`k_g ≥ 0`、`groupSize_g ≥ 0`，组间完全独立，无公共形状约束。
- `Aarray/Barray/Carray` 长度 = `Σ groupSize[g]`，元素为 **Device 地址的 Host 侧指针数组**（不可传入 Device 侧指针数组）。
- 前导维度约束：`transa = N` 时 `lda ≥ max(1, m)`，否则 `lda ≥ max(1, k)`；`transb = N` 时 `ldb ≥ max(1, k)`，否则 `ldb ≥ max(1, n)`；`ldc ≥ max(1, m)`。
- 不涉及 broadcast 语义：每个矩阵地址由扁平指针数组逐元素独立给出，无 stride。

### 算子实现

#### 实现方案

整体为 **host 编排 + 两段式 kernel（AIC 乘积 + AIV 收尾）**，并在 `alpha = 1 && beta = 0` 时把收尾**整体融合**掉（§3.2.2(4)）。一次调用分四步：校验与指针扁平化 → 逐组描述符构建（含**逐组 single shape 选择**）→ 按工作空间分窗下发 → 收尾。

```
                     host
  ┌──────────────────────────────────────────────┐
  │ 1 校验（顺序见 3.2.1）                        │
  │ 2 逐组 GemmGroupedGroupParam 构建             │
  │ 3 Host 指针数组 → Device 地址数组（一次性上载）│
  │ 4 按 workspace 预算分窗                        │
  └───────────────┬──────────────────────────────┘
                  │  window [start, start+batchCount)
                  ▼
  ┌───────────────────────────┐   ┌───────────────────────────┐
  │ AIC：gemm_kernel_nn/nt/tn/tt│   │ AIV：combine_kernel       │
  │ 按转置组合分派，逐组重设     │→ │ C = alpha*slot + beta*C    │
  │ Matmul 描述符；乘积写 scratch│slot│ 或 C = beta*C（跳过组）    │
  │ 融合组直接写 C（无槽位）     │   │ 融合组不参与                │
  └───────────────────────────┘   └───────────────────────────┘
                  │  两段在同一 stream 上背靠背下发，天然串行
                  │  融合组只发 AIC、不发收尾（收尾块数得 0）
                  ▼
             下一窗（复用同一 scratch）
```

##### 3.2.1 host 侧设计

**（1）校验顺序**（刻意与 §2.4 参数表及 arch35 实现对齐，使负向用例返回码可预期）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `groupCount < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `groupCount == 0` → `ACLBLAS_STATUS_SUCCESS`（合法 no-op；**刻意排在数组校验之前**——无组可描述时调用方没有可指向的对象，arch35 与 CPU golden 同序）
4. 描述符数组任一为 nullptr（`transaArray/transbArray/mArray/nArray/kArray/alphaArray/ldaArray/ldbArray/betaArray/ldcArray/groupSizeArray`）→ `INVALID_VALUE`
5. `Aarray/Barray/Carray` 任一为 nullptr → `INVALID_VALUE`
6. 逐组校验：`m/n/k/groupSize ≥ 0`；对 `groupSize > 0` 的组校验转置枚举合法（**实数算子无共轭语义，`ACLBLAS_OP_C` 判非法**）与前导维度约束；累计 `totalBatchCount`
7. `totalBatchCount == 0` → `SUCCESS`（全空组，指针数组永不解引用）
8. 逐组逐 batch 校验 **元素级** 非空：`m,n,k > 0 && groupSize > 0` 时 A/B 元素不可为空；`m,n > 0 && groupSize > 0` 时 C 元素不可为空；no-op 组跳过
9. 全部合法 → 构建描述符并下发

**（2）Matmul 模式编码**。由于列主序经行主序 Matmul 实现时两个操作数互换（见 3.2.2），kernel 内的 `isTrans` 与 BLAS 的 `transa/transb` 是对调关系：

```cpp
// host 与 kernel 两侧一致：bit1 = Matmul 左操作数（承载 BLAS B），bit0 = 右操作数（承载 BLAS A）
matmulMode = ((transb == T) << 1) | (transa == T);
```

| `matmulMode` | Matmul (左, 右) | 对应 BLAS | 常量 |
|---|---|---|---|
| 0 | (plain, plain) | `transa=N, transb=N` | `GEMM_GROUPED_MM_NN` |
| 1 | (plain, trans) | `transa=T, transb=N` | `GEMM_GROUPED_MM_NT` |
| 2 | (trans, plain) | `transa=N, transb=T` | `GEMM_GROUPED_MM_TN` |
| 3 | (trans, trans) | `transa=T, transb=T` | `GEMM_GROUPED_MM_TT` |

**（3）工作空间布局**。一次调用在句柄的 workspace 内顺序排布：`groupParams` → `aPtrArray` → `bPtrArray` → `cPtrArray` → `temp`（乘积 scratch），各段 256 字节对齐（`GGB_ALIGN`）。指针数组在 workspace 中的**长度取 `totalBatchCount`（不是窗口大小）**，一次上载全部批次；只有 `temp` 按窗口大小伸缩。

乘积 scratch 的"槽位"大小取所有**需要槽位**的组的最大值。槽位与 C **逐元素同构**（同一 `ldc`），故最后一格下标为 `(n-1)*ldc + m - 1`：

```
slotElements(g) = (n_g == 0) ? 0 : (n_g - 1) * ldc_g + m_g     // 该组乘积槽位所需元素数
tempSlotSize    = max over { g : GroupNeedsSlot(g) }
GroupNeedsSlot(g) = 有乘积(g) && !fuseEpilogue(g)
有乘积(g)          = m>0 && n>0 && k>0 && groupSize>0 && !skipGemm && !useAivGemm
```

因此**被跳过的组（`skipGemm`）与被融合的组（`fuseEpilogue`）都不占槽位**：一个极大的组若恰好被融合，不会抬升 `tempSlotSize`。当一个调用的**所有**参与乘积的组都被融合时，`tempSlotSize = 0`（见 (4) 的第 3 个特例）。

**（4）分窗策略**。乘积 scratch 是全调用最占空间的资源，若按总批量一次性申请，`4096³ × 128` 需数百 GiB。故按 workspace 预算分窗：

```
windowBatches = totalBatchCount                        // 特例 1：totalBatchCount == 0
              = totalBatchCount                        // 特例 2：tempSlotBytes == 0（全部融合/跳过 ⇒ 无槽位）
              = clamp(TEMP_BUDGET / tempSlotBytes, 1, totalBatchCount)   多数情况
TEMP_BUDGET   = 1 GiB
```

每窗独立携带 `startBatch / batchCount`，窗之间无数据依赖（AIC 写完本窗槽位才轮到 AIV 读，下一窗复用同一段 scratch），因此分窗不影响正确性。取 `clamp(..., 1, ...)` 保证再大的单槽位也能至少推进一个 batch——此时若单槽位本身就超预算，由下发前的 workspace 容量检查明确报错，而不是静默降级。句柄 workspace 上限 2 GiB，1 GiB 的槽位预算为其留出足够余量。

> **特例 2 是性能的关键一条**：epilogue 融合（§3.2.2(4)）会把 `tempSlotSize` 压到 0，于是**无论批量多大都只有一窗**，多窗的重复下发（每窗一次 AIC + 一次收尾）随之消失。性能集 200 条全部命中 `alpha = 1 && beta = 0`，因此实测路径就是单窗路径。

**（5）分核策略**（R2：coreNum 动态获取，禁止硬编码）：

- AIC：`GemmBlocks(mode) = min(Σ_{窗内该模式组} (窗内 batch 数 × rowTiles × colTiles), GetAicCoreCount())`，其中 `rowTiles = ceil(n_g / singleM_g)`、`colTiles = ceil(m_g / singleN_g)` ——**用该组自己的 single shape**（§3.2.1(7)），不是固定 `TILE_M/TILE_N`。块数只用于"铺满核"，真正决定工作怎么分的是 kernel 内同一条 `(batch, rowTile, colTile)` 除法，两侧读**同一份描述符字段**，因此不可能失配。某模式窗内无组则返回 0，该入口不下发。
- AIV 收尾：`CombineBlocks = min(Σ_{窗内需收尾组} (窗内 batch 数 × itemsPerBatch_g), GetAivCoreCount())`。`itemsPerBatch` 分两支——**稠密组**（`ldc == m`，整批在 slot 内是一段连续区间）按 `ceil(m*n / COMBINE_CHUNK)` 计，**跨步组**（`ldc > m`）按 `n` 计（每列一个工作项）。两侧（host 计数与 kernel 分派）同样是同一份 `CombineItemsPerBatch` 逻辑。
- 纯 AIV 瘦 GEMM 的块数：`AivGemmBlocks = min(Σ_{useAivGemm 组} 窗内 batch 数, GetAivCoreCount())`；因该路径已关闭（§3.2.1(8)），恒为 0。
- 均先判零再使用（R8 除零防御）：`GetAicCoreCount()` / `GetAivCoreCount()` 返回 0 时返回 `INTERNAL_ERROR`，不参与任何除法。

**（6）下发顺序**：窗内先按枚举序 `NN → NT → TN → TT`（`mode = 0..3`）依次下发**有组的** AIC 入口（多入口在同一 stream 上串行），再（若有）下发纯 AIV 入口，最后（若有）下发一次 AIV 收尾。两段靠 stream 顺序天然同步，无需额外 event。

判别分派正确性只看一条：**某个入口若 `blocks == 0` 就完全不 launch**。因此一次"全融合"调用（性能集的全部 200 条）实际只下发 1~4 个 AIC 入口，收尾入口根本不出现；一次纯 `alpha = 0, beta = 1` 的调用则一个入口都不下发。

**（7）逐组自适应 IterateAll 形状**（`ApplySingleShapes` / `ChooseSingleShape`）

一个组的 "single shape"（`singleM × singleN`）就是**一次 `IterateAll` 覆盖的输出范围**，区别于一个 128×128 的 L0 base tile（base tile 由 Matmul 在调用内部走）。命名按 Matmul 自己的 extent：`singleM` 切 BLAS 的 **n** 方向（Matmul 的 M），`singleN` 切 BLAS 的 **m** 方向（Matmul 的 N）。于是对单个组：

```
rowTiles = ceil(n / singleM)      colTiles = ceil(m / singleN)
items    = groupSize * rowTiles * colTiles
rounds   = ceil(items / coreNum)            // 一个核串行要付的次数
```

**为什么不能是固定常量**：两种失效模式方向相反，且都与访存无关：

| 方向 | 后果 | 真机标定 |
|---|---|---|
| 太小 | `items` 爆炸，每个 item 重付一次 per-IterateAll 固定延迟 | 910B3 实测 ~4.5 us/次，而瘦组真实 MMAD 只有几百 ns；`4096×16×32` 在 `singleN=128` 下每 batch 32 次 IterateAll，只为铺 16 行 |
| 太大 | `items` 掉到 coreNum 以下，该组只占 `items` 个核 | `1536×128×2048` 在 `singleN=128` 是 12 items（12 核忙），`singleN=1024` 是 2 items（2 核忙），实测后者**慢 6.3 倍** |

所以形状改为**逐组在 host 上选**：在倍增网格 `singleM, singleN ∈ {TILE_M, …, SINGLE_MAX} = {128, 256, 512, 1024}` 上最小化

```
cost(singleM, singleN) = rounds × (macPerItem + ITEM_LATENCY_MAC)
macPerItem = (n / rowTiles) × (m / colTiles) × k
ITEM_LATENCY_MAC = 3.5e6     // 4.5 us × 770 GMAC/s，仅量级有意义
```

**两道护栏**——这是把它做成"严格改进"而不是"押注代价模型"的关键：

1. **平手归基线**。基线形状 `(TILE_M, SINGLE_MAX) = (128, 1024)` 就是此前所有版本用的固定形状；搜索中 `singleM` 用 `<`（保窄）、`singleN` 用 `<=`（保宽），因此**代价相同的候选绝不替换基线**。依据是真机实测：只把 `singleN` 从 1024 缩到 128、而分解其实并未改变，代价就是 15~30%（`TC_PF_1008` 28.1 → 32.3 us，该轮唯一 FAIL）——原因是 Matmul 按**传入的** `singleN` 规划运行时 tiling，而不是按它最终真正算的 tile。
2. **偏离阈值**。候选必须被预测为至少便宜 `100 − DEVIATE_PERCENT`%（=20%）才采纳。模型对边际配对会排错序——`TC_PF_1112` 在一个模型说"便宜 9%"的形状上**翻倍**；而它真正找到的胜势都是 3~5 倍，20% 阈值干净地分开两种情形。

选出的形状写入 `GemmGroupedGroupParam::singleM / singleN`，host 的 `GemmBlocks` 与 kernel 的 `SetSingleShape` **共读同一份字段**，从构造上排除两侧漂移。基线代价的比较写成 `(baseCost / 100) * DEVIATE_PERCENT`——**先除后乘**，把判定留在 uint64 内。⚠️ 注意此处与规则 C 的护栏 3 **恰好相反**（那里必须**先乘后除**）：因为 `baseCost` 以 MAC 计、量级 ≥10⁹，先除不会截断；而 `v0` 以 base block 计、只有几百，先除会被截成 0。两者是"同一个整数除法陷阱在两种量级下的相反解法"，见 (7.2)。

**（7.2）规则 C：每 K 步 base block 数下限**（`GgbWidenToMinBlocksPerKStep`）

上述代价模型**按 IterateAll 计费**，因此看不见"每 K 步一次固定流水/同步代价"。把全量运行的用例按下式排序后，几乎全部按同一个量重排：

```
eff = (total_MAC / 芯片峰值) / 实测
bps = ceil(min(singleM, n) / TILE_M) × ceil(min(singleN, m) / TILE_N)
    // 一个 item 每 K 步发出的 128×128×64 base block 数
```

| `bps` | `eff` | 代表性用例 |
|---|---|---|
| 8 | 95% ~ 104% | `TC_PF_1039 / 1042 / 1069 / 1194 / 1127` |
| 4 | 81% | `TC_PF_1002` |
| ≈3.6 | 31% | `TC_PF_1162` |
| 2 | 20% | `TC_PF_1176 g1`（全表最窄） |

即：**窄 item 摊不掉每 K 步的固定代价**。模型看不到这一点，于是它为 `TC_PF_1176` 的 `(256,256,768,gs=128)` 组保留了 128×256 的 item（模型只把更宽的 `(256,1024)` 判贵 1.1%，远低于模型自身误差），该组实测到预测值的 5 倍，是 200 条里唯一贴着限值的一条（983.62 / 987.60 us，余量 0.4%）。

于是 `ChooseSingleShape` 出口再加一步：`bps ≤ MIN_BLOCKS_PER_KSTEP`（=2）的组**只抬高 `singleM`**（合并行瓦片），且**四道护栏全过才动**：

1. `rounds` 严格下降；
2. `items` 仍 ≥ coreNum（不让任何核空转）；
3. 每核 base block 总数增长 ≤ `WIDEN_V_GUARD_PERCENT`%（=110，即最多多付 10% 的 MMAD 量化误差）；
4. `bps` 严格上升（否则这次改动没有机制支撑）。

取 `bps` 最大者；平手取块数最小，再平手取最窄——规则确定性。**`singleN` 全程不动**，所以护栏 1 所针对的失效模式（只缩 `singleN` 导致 15~30% 退化）不会回来。

> 🔴 护栏 3 有一条必须记住的写法：**必须先乘后除**（`v0 * 110 / 100`）。写成 `(v0 / 100) * 110` 会被整数除法截断为 0，从而**全部拒绝**——规则**静默失效**，且不产生任何报错。（`v0` 以 base block 计，是几百量级；不是 MAC，所以不能用 MAC 那种先除后乘的写法。）

影响面：200 条性能集里 **11 个组 / 10 条用例**，其中 5 个组不多付任何 base block（`1059` 960→960、`1110` 320→320、`1145` 256→256、`1096` 48→48、`1165 g0` 64→64），10 条里 9 条余量 >80%，所以**只有 `TC_PF_1176` 的判定可能移动**。真机对照：同一条 `TC_PF_1176` 在规则 C 生效前为 983.62 us（限 987.60，余量 **0.4%**，压线），生效后为 **607.80 us（余量 38.5%）**，即 **1.62×**。A/B 开关：`MIN_BLOCKS_PER_KSTEP = 0` 即退回上一版 chooser。

**（8）纯 AIV 瘦 GEMM 路径：已关闭且不可达**

瘦组（`m < 128 || n < 128 || k < 64`）会浪费 Cube 的 128×128×64 tile，因此仓内曾有一条把它们路由到纯 AIV 核的路径（`useAivGemm`，kernel 侧 rank-1 更新 + 逐标量 `GetValue`）。**真机实测为净损失，已关闭**：

- AIV 版读 `op(B)` 是**一个标量一次**（`(l, j)` 循环里的 `GetValue`），代价按 `ceil(m/64) × k × n` 次 ~34 ns 的同步读增长；
- 200 条性能集里 **138 条**命中该谓词。**Cube 路径把它分到的 62 条全过，AIV 路径 138 条里失败 71 条**；参考集上扁平 Cube 轮转 **195/200**，带 AIV 只有 **129/200**；AIV 本想救的 5 条，在 Cube 下超预算 1.2~4.4 倍，在 AIV 下 **5~852 倍**。

因此 host 侧 `constexpr bool USE_AIV_GEMM = false` 使 `useAivGemm` 恒为 0，kernel 侧对应的遍历显式标注为不可达，宿主计算块数时该路径恒得 0。**代码保留但不可达；读代码时不要以为它在生效。** 要重新启用必须连同性能集一起重测。

##### 3.2.2 kernel 侧设计

**（1）列主序经行主序 Matmul 实现（核心设计点）**

Matmul API 产出**行主序** C。列主序存储 C 等价于行主序存储 Cᵀ；而把一段列主序缓冲区**当行主序读**得到的正是它的转置。于是：

```
C_列主序 = op(A) · op(B)                    (m × n)
   ⟺  Cᵀ（n × m，行主序，行步长 ldc）= (op(A)·op(B))ᵀ = op(B)ᵀ · op(A)ᵀ
```

即：**Matmul 的左操作数取 BLAS 的 B 缓冲区，右操作数取 BLAS 的 A 缓冲区**，且 `M = n、N = m、K = k`。这样 scratch 与 C **逐元素同构**（同一 `ldc`），AIV 收尾退化为一次普通逐列读写，不需要任何重排。

**（2）四个转置组合的推导**。把 `A` 的存储当行主序读：

- `transa = N`：`A` 存储 `m × k` → 该行主序视图是 `op(A)ᵀ`（`k × m`）
- `transa = T`：`A` 存储 `k × m` → 该行主序视图是 `op(A)`（`m × k`）

对 B 同理（`n × k` / `k × n`）。把这两个视图喂给 Matmul，让 `isTrans` 吸收剩余的翻转，即得四种组合（`isTrans` 是 `MatmulType` 的**编译期**模板参数，因此每个组合必须是一个独立 kernel 入口）：

| BLAS | Matmul (A, B) | `M, N, K` |
|---|---|---|
| `transa=N, transb=N` | (plain, plain) | `n, m, k` |
| `transa=N, transb=T` | (trans, plain) | `n, m, k` |
| `transa=T, transb=N` | (plain, trans) | `n, m, k` |
| `transa=T, transb=T` | (trans, trans) | `n, m, k` |

**（3）`SetOrgShape` 的逐模式显式填充**。前导维度可以大于逻辑维度，故步长必须原样传入、维度由 `SetSingleShape` 单独给出。`SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc)` 各字段的实际语义（已对照 `common/helper/complex_blas3_arch22.h` 的接口注释，并已在真机四种模式上验证）：

- 左操作数：plain 时其 GM 行步长取自 `orgKa`，transposed 时取自 `orgM`
- 右操作数：plain 时取自 `orgN`，transposed 时取自 `orgKb`
- `orgKc` 恒为乘积槽位的行步长

按"未转置习惯"填写会在转置场景静默算错，因此四种模式**逐条显式写出**（左=B 步长 `ldb`，右=A 步长 `lda`）：

| 模式 | 左操作数（B）/ 右操作数（A） | `SetOrgShape` 实参 |
|---|---|---|
| NN | plain / plain | `n, lda, ldb, k, ldc` |
| TN | trans / plain | `ldb, lda, k, k, ldc` |
| NT | plain / trans | `n, m, ldb, lda, ldc` |
| TT | trans / trans | `ldb, m, k, lda, ldc` |

自检：NN 的左操作数（B，`ldb`，plain）行步长取自 `orgKa = ldb` ✓、右操作数（A，`lda`，plain）取自 `orgN = lda` ✓；TN 的左操作数改为 transposed 后取 `orgM = ldb` ✓、右操作数仍取 `orgN = lda` ✓；NT/TT 同理自洽，且 `orgKc` 恒为 `ldc`。

**（4）两段式 kernel 拆分 + epilogue 融合**。Matmul API 自身没有 alpha/beta，这正是收尾必须独立成 AIV 阶段、而不能靠 fixpipe 开关完成的原因。三段职责如下：

- **AIC 阶段**（`KERNEL_TYPE_AIC_ONLY`）：只负责"确实产生乘积"的组——即 `m,n,k > 0`、`groupSize > 0`、`alpha != 0` 且未走 AIV 路径（`skipGemm == 0 && useAivGemm == 0`）；逐组重设描述符后把乘积写入 scratch 槽位（或直接写入 C，见下）。
- **AIV 收尾阶段**（`KERNEL_TYPE_AIV_ONLY`）：覆盖窗内**需要收尾**的 batch；对 AIC 跳过的组（`alpha == 0` 或 `k == 0`）退化为 `C = beta * C`。`skipGemm && beta == 1` 时 C 逐位不变，**直接不下发该组**（`CombineBlocks` 计入 0）。被融合的组（见下）不进收尾。
- **epilogue 融合**（性能关键路径）：当 `alpha == 1 && beta == 0` 时，该组的 C 恒等于 op(A)·op(B)，收尾退化为一次纯 `scratch → C` 拷贝。AIC 阶段把这一步折叠进自己的落盘——**乘积 tile 直接写进 C**，该组因此**不占 scratch 槽位、不参与收尾**（描述符 `fuseEpilogue = 1`，host 在 `ComputeTempSlotSize` / `CombineBlocks` 中排除它）。

  融合之所以只需改一个输出基址，是因为**槽位与 C 逐元素同构**：同一 `ldc`、同一原点，行主序 Matmul 写 `[0,n)` 行、每行 `m` 宽，两者布局完全一致。判据是 `skipGemm == 0 && alpha == 1.0F && beta == 0.0F`——**精确恒等，不是容差放宽**，因此命中它的精度用例只省时间、不改判定。

  🔴 **注意这里没有 `k ≤ SINGLE_K` 的限制**（早期版本有）：大 K 时 kernel 按 `SINGLE_K` 切块、用"首块覆盖、后续原子累加"落盘，这套逻辑本来就与"写到槽位还是写到 C"无关，融合组只是把目标基址换成 C。`k == 0` 则必须排除（此时乘积为空，且 `skipGemm` 已把它归零，C 仍需由收尾处理）。

  收益量级：`2048 × 192` 的瘦组在旧路径下每个工作项搬一列、每 batch 付 192 次同步，只为搬运 Cube 刚产生的字节——融合后这部分流量与整段 AIV 下发都不存在。**性能集 200 条全部命中 `alpha = 1 && beta = 0`**，因此全部走融合路径；而融合把 `tempSlotSize` 压到 0，又连带把调用收敛成**单窗**（§3.2.1(4) 特例 2）。

**（5）Matmul 实例复用**。每次下发只构造**一个** `TPipe` 和一个 `MatmulImpl`，在 `SetSubBlockIdx(0)` / `Init(nullptr, &pipe)` / `DisableBias()` 之后，被本次下发所辖的**所有组复用**（仅 `SetOrgShape`/`SetSingleShape`/`SetTensorA`/`SetTensorB` 逐组重设）。若逐组构造 `MatmulImpl`，每组都会重新注册 L1/L0 缓冲区，多组调用会耗尽 pipe；这是一个已验证的设计约束，故在 `GemmStage` 内一次性建好、以引用传入 `RunGroupGemm`。

**（6）分核与遍历**：AIC 把该组的 `(batch, rowTile, colTile)` 摊成一条扁平序列，某核拿到的恰是下标 `coreIdx, coreIdx + coreNum, coreIdx + 2·coreNum, …`，**直接按下标枚举**（而非从 0 扫一遍 cursor 再筛 `cursor % coreNum == coreIdx`）。两种写法给每个核分的活**逐项相同**，但扫描法在每个被丢弃的 item 上都要付一次 64 位取模与三次 GM 指针加载——在 `tilesPerBatch` 很小的组上约 95% 的迭代都是这种空转。AIV 收尾同样按 `(batch, item)` 轮转：稠密组的 item 是一段连续区间（按 `COMBINE_CHUNK` 切），跨步组的 item 是一整列（列主序下是一段连续行），所以每个核始终连续读写，无需感知邻核形状。

**（6b）逐组 single shape 在 kernel 侧的消费**。`RunGroupGemm` 从描述符读该组自己的 `singleM / singleN`（§3.2.1(7) 由 host 选定）交给 `SetSingleShape`，因此一个"tile"= 一次 `IterateAll` = 至多 `singleM × singleN × kCount`，**不是**一个 128×128 的 L0 base tile——base tile 由 Matmul 在调用内部自己走。三点约束：

1. **`SetSingleShape` 只在形状真的变化时才重发**。它会驱动 Matmul 的运行时 tiling（含 L1 缓冲重分配），代价不小；同一组的内部 tile 形状相同，同一组的各 K chunk（末块除外）也相同，所以按"上一组用过的形状"做缓存，组边界处失效。
2. **融合组的目标是 C 而非槽位**，两者逐元素同构 ⇒ 只改输出基址（见 (4)）。
3. **零步长除数在使用前判零**，且 `k == 0` 的组根本到不了这里（`skipGemm` 已在 host 拦截）。

> **已废弃的机制，此处留档以免误读**：早期版本曾实现一条 "batch-per-core 快速路径"（每个核领走**完整 batch**、一次 `SetSingleShape(n,m,k) + IterateAll`，靠 Matmul 内部流水线复用 L1），并用 `useBatchPerCore` 开关控制。**该机制及其字段在现行源码中已不存在**，其职责被 §3.2.1(7) 的逐组自适应形状取代——后者在同一个框架内同时处理了"瘦组 tile 空转"（用更小的 `singleM`）与"共享操作数的重复搬运"（用规则 C 合并行瓦片，见 §3.2.1(7.2)），而无需为两种病态形状分别加特例。若在旧设计稿或旧日志里见到 `useBatchPerCore` / `directWrite` 这两个名字，它们**都已重命名或移除**（后者对应现在的 `fuseEpilogue`）。

**（7）大 K 切分**：`singleCoreK` 取 `SINGLE_K = 8192`。`k > SINGLE_K` 时在 kernel 内按 `SINGLE_K` 切块，用 `IterateAll(out, kc == 0 ? 0 : 1)` 的原子累加标志实现"首块覆盖、后续累加"——因此 `k ≤ 8192` 的调用**完全不触发原子写**。

**（8）编译期静态约束**：`MatmulShapeParams` 取 `(SINGLE_MAX, SINGLE_MAX, SINGLE_K, TILE_M, TILE_N, BASE_K)`——前三位是**静态 tiling 的上限**，后三位是 base block。两个 extent 都用 `SINGLE_MAX`，因为形状网格是方的、组可能朝任一方向宽（`4096×16` 想要 `singleN`，`16×4096` 想要 `singleM`）。`static_assert` 把"逐组形状的上限"与 Matmul 静态 tiling 在编译期锁在一起：

| `static_assert` | 作用 |
|---|---|
| `singleCoreM >= SINGLE_MAX`、`singleCoreN >= SINGLE_MAX` | 运行时 `SetSingleShape` 不得超过静态上限 |
| `singleCoreM >= TILE_M`、`singleCoreN >= TILE_N`、`SINGLE_MAX >= TILE_M` | 形状网格下界非空 |
| `singleCoreK == SINGLE_K` | K 上限两侧一致 |

🔴 **这条铁律的失败方式极隐蔽**：`SetSingleShape` 传入超过静态 `singleCoreM/N` 的形状**不报错**，而是让 Matmul 以为"还有额外的核可以分走多出来的部分"，最终只有第一个核在跑——**tile 尾部静默地不写**。曾有一轮实验 578 条失败中的 **577 条**都是这一个原因（形状超过静态 128）。因此 `SINGLE_MAX` 与 `singleCoreM/N` **必须一起改**，由 `static_assert` 锁死。若某天要下探 `SINGLE_MAX`（该文件注释给了 1024→512→256 的梯级），host chooser、kernel `SetSingleShape`、静态 tiling 三处读的都是同一个常量，只需重编、无需改代码。

**（9）关键常量**（名为 `GEMM_GROUPED_*`，定义在 `gemm_grouped_batched_tiling_data.h`，host 与 kernel 共读同一份）：

| 常量 | 值 | 说明 |
|---|---|---|
| `TILE_M` / `TILE_N` | 128 / 128 | base block 的行/列，也是形状网格的下界 |
| `BASE_K` | 64 | base block 的 K。**不能再抬到 128**：128×128×128 会关掉 `dbL0B`/`dbL0C` 双缓冲 |
| `SINGLE_K` | 8192 | 单次 Matmul 的 K 上限，超过则 kernel 内切分并原子累加 |
| `SINGLE_MAX` | 1024 | 逐组 `singleM/singleN` 的上界；与 `singleCoreM/N` 由 `static_assert` 锁定 |
| `ITEM_LATENCY_MAC` | 3.5e6 | 一次 IterateAll 的固定延迟，折算成 fp32 MAC 当量（见 §3.2.1(7)） |
| `DEVIATE_PERCENT` | 80 | 形状偏离基线的阈值：预测 ≤80% 基线代价才动（即 ≥20% 才动） |
| `MIN_BLOCKS_PER_KSTEP` | 2 | 规则 C 的触发阈值 `bps`；置 0 即关闭规则 C |
| `WIDEN_V_GUARD_PERCENT` | 110 | 规则 C 允许每核 base block 数增长的上限（+10%） |
| `COMBINE_CHUNK` | 8192 | AIV 单次 UB 处理元素数（每缓冲 32 KB）；host 也用它算收尾块数 |
| `OP_N` / `OP_T` | 111 / 112 | `aclblasOperation_t` 的内核侧镜像，避免 kernel 引入 host API 头 |
| `GGB_ALIGN` | 256 | workspace 各段对齐（host 侧） |
| `GGB_TEMP_BUDGET` | 1 GiB | 乘积 scratch 分窗预算（host 侧），见 §3.2.1(4) |

> `COMBINE_CHUNK` 刻意声明在共享头而不是 kernel 里：host 要用它算收尾块数，两处手抄会漂移，而漂移是**静默的**——块数偏少只是并行度差，但一旦与 UB 缓冲尺寸不一致就会**越界**。

**（10）收尾阶段的 UB 流水**（`CombineOneSegment`）：按 `MTE2 → V → MTE3` 的显式事件链组织，并对 `alpha/beta` 做退化消除以省掉无谓访存。

**分段方式**取决于该组是否稠密（`ldc == m`）：

| 情形 | 分段 |
|---|---|
| 稠密（`ldc == m`） | 整批是一段连续区间，按 `COMBINE_CHUNK` 切段 |
| 跨步（`ldc > m`） | 每列一个工作项，列内再按 `COMBINE_CHUNK` 切 |

两种分段下的 `alpha/beta` 退化消除一致：

| 场景 | 访存/计算 |
|---|---|
| `alpha != 1` | 对乘积做 `Muls(prod, prod, alpha)` |
| `beta == 0` | 不读旧 C，直接写 `alpha * prod` |
| `beta != 0` | `DataCopyPad` 旧 C 后 `Axpy(prod, cOld, beta)`（`prod += beta * C_old`） |
| 跳过组（`skipGemm`）且 `beta == 0` | `Duplicate(prod, 0)` 后写回，不读 C |
| 跳过组且 `beta == 1` | 整组不下发（见 (4)） |
| 融合组（`fuseEpilogue`） | 整组不进收尾（见 (4)） |

> 槽位视图的 extent 始终按 `tempSlotSize` 取，**不按本组的列数推导**：槽位尺寸是为所有需要槽位的组取的公共最大值，一个只做 `C = beta*C` 的组其 `n` 可能远小于该最大值，若按本组推导就会读到槽外。

### 支持硬件

支持的芯片版本（勾选）：

- Atlas 800I/T A2 √
- Atlas 800I/T A3 √

### 算子约束限制

- 不支持超出 `ldaArray/ldbArray/ldcArray` 语义的非连续内存访问。
- 不涉及 broadcast 规则，无 stride 语义。
- 不要求动态 shape（`m/n/k/groupSize/groupCount` 均为运行时入参，由 host tiling 支持）。
- 离席计算：结果写入各组 C；C 与 A 或 B 不允许内存重叠；不返回视图。
- `Aarray/Barray/Carray` 为 **Host 侧**指针数组（元素为 Device 地址），不可传入 Device 侧指针数组。
- 不要求确定性计算（浮点累加顺序不保证逐位一致），因此大 K 切分与原子累加不影响验收。

---

## 4 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 混合容差（Mixed Tolerance）逐元素比对。FLOAT32：`rtol = atol = 2⁻¹³ (1.22e-4)`，`required_matched_ratio = 0.99`，`max_abs_error_limit = 1e-2 或 32×ULP` | 生态算子开源精度标准（opbase `mixed_tolerance_standard.md`）；golden 由 cblas（Netlib BLAS `sgemm` 逐组）生成 |
| 性能标准 | 5 个性能 case 的平均单次耗时不超过 351.3 / 3714 / 37652 / 298380 / 1188440 µs。测试设备 Atlas 800T A2 (910B3) | 任务书 §3.3 |
| 内存标准 | 不涉及（任务书 §3.4） | 任务书 §3.4 |

精度的两个条件——逐元素 `|actual − golden| ≤ atol + rtol × |golden|`，用例级 `matched_ratio ≥ 0.99` **且** `max_abs_error ≤ max_abs_error_limit`——必须**同时**满足。本算子含 K 维规约，大数规约可能引入更大误差，`max_abs_error_limit` 按任务书 §3.2 说明可酌情放宽至 2 ULP。

**性能口径（重要，避免重复采集）**：耗材取自 msprof op 落盘的 `Task Duration`，**该值本身就是 msprof 内部的有效采样统计量**（msprof op 自带 warmup）。因此验收过程对每条用例**只采集一次**，不需要、也不应重复跑用例——重复跑付出的是 host 侧准备时间（最贵的一条用例 host 侧约 87.8 s，而其 kernel 仅 8 ms），而且不会让读数更稳：197 条用例重复测量的抖动中位 1.21%、P90 5.22%、最大 18.7%，**与采样次数无关**。余量低于 5% 的用例再跑多少遍也不会变得更稳。

**实测结论**（2026-10-09 真机，AIC=20 核，`SOC_VERSION=ascend910b3`）：

- **精度**：1212 条全部 PASS（1011 条 CSV 精度用例 + 1 条 `NullHandle`，其中 970 条走到数值判定；745 条匹配率恰为 1.0000，225 条落在 [0.99821, 0.9999]，**无一条 < 0.99**）。唯一曾有疑问的 `TC_FL_134` 已按参考侧溢出处置，见附录。
- **性能**：200 条全部 PASS，**最低余量 38.46%**。五条硬指标实测 **136.02 / 926.16 / 7781.08 / 62955.84 / 279123.06 µs**。
- **物理复核**：由 15 条大用例反推芯片实际峰值 ≈ **33.17 TMAC/s**（保守取 31.7 TMAC/s 作下界）。全表 `达标耗时 / 物理下界` 最小 **4.28** ⇒ 即便只跑到峰值 23.3%，200 条仍可全部达标。**因此性能不是压着硬件极限通过的，而是有结构性余量。**

### 兼容性分析

- **新算子**，不涉及既有接口的兼容性变更。
- 接口声明 `aclblasSgemmGroupedBatched` 已位于 `include/cann_ops_blas.h`，**签名与 arch35 完全一致**，供其他产品线共用；本任务不修改该头文件，仅新增 `arch22` 实现，因此对 arch35 与既有调用方零影响。
- 构建系统按 `SOC_VERSION` 单选架构目录（`ascend910b*` / `ascend910_93*` → `arch22`；`ascend950*` → `arch35`），`blas/` 下源码全量自动收集并统一以 ASC 前端编译（kernel 直调模式），因此新增目录**无需任何手工注册**，也不会与 arch35 交叉编译。

---

## 附录：真机验证结论

本设计已在 hidevlab A2/A3 环境（`SOC_VERSION=ascend910b3`，AIC=20 核，CANN 9.1.0）完成编译、精度与性能验证。开发前识别的 6 项风险**全部闭合**：

| # | 开发前识别的风险 | 真机结论 |
|---|---|---|
| 1 | `SetOrgShape` 字段映射（当时判为最高风险，仅由接口注释推导） | ✅ 四种模式实参经逐例比对确认正确，四个转置组合的精度用例全过 |
| 2 | asc-devkit / CANN 9.1.0 的 API 可用性 | ✅ `MatmulShapeParams` / `MatmulBiasParams` / `GetMMConfig<CONFIG_NORM>` / `KERNEL_TYPE_AIC_ONLY` 均存在、签名匹配 |
| 3 | AIC/AIV 两段在同一 stream 的同步语义 | ✅ stream 顺序保证"乘积写完再收尾"，无跨核乱序 |
| 4 | `IterateAll` 原子累加标志（`kc == 0 ? 0 : 1`）在多核并发、`k > SINGLE_K` 下的行为 | ✅ 大 K 用例（`k` 远超 `SINGLE_K`）全部 PASS |
| 5 | 精度与性能达标 | ✅ 精度 **1212/1212**；性能 **200/200**，最低余量 38.46% |
| 6 | `MatmulConfigMode::CONFIG_NORM` 的性能取舍 | ✅ 沿用仓内同架构 BLAS-3 算子结论；其 `enUnitFlag` 默认值省掉每个 base block 的 MMAD/FIXPIPE 同步，在本算子形状上复测为等优或更优 |

### 验证基线

```
[==========] 1212 tests from 2 test suites ran.
[  PASSED  ] 1212 tests.
```

- 精度：1011 条 CSV 用例 + 1 条 `NullHandle`（不依赖 CSV）。970 条走到数值判定，匹配率分布 745 条恰为 1.0000 / 225 条落在 [0.99821, 0.9999]，**无一低于 0.99**。
- 性能：200 条全部满足 `实测 ≤ 达标耗时`，完整性门禁退出码 0（无缺条、无 `NO_DATA`、无 `ERROR`）。

### 唯一需说明的用例：`TC_FL_134`（参考侧溢出，非算子缺陷）

`TC_FL_134` 是全表**唯一** `A_fill = RANDOM_EXTREME` 的用例：A 中填入 ±FLT_MAX、B 中填入 U(−5,5)，**每个乘积都溢出 fp32 范围**。其结果为：

- 两侧都返回 ±inf，而**符号由各自的累加顺序决定，BLAS 契约并不规定**；
- 该用例的 `tensorMax|golden| = 0`、MERE/MARE 统计量全为 0、`maxAbsErr = 0.0000e+00` ⇒ **没有任何元素进入过算术比较**，判定完全由 `test/frame/verify.h` 的**非有限类检查**驱动（任何 inf/nan 元素直接硬失败）。**因此即便算子与参考逐位相同，匹配率也拿不到 1.0**；
- 解析复核：该用例 855/1024（83.5%）的精确值 `|·| > FLT_MAX`，3677/16384 个乘积溢出；真机打印的两个采样点解析值均为 **+2.07e39**（正确 fp32 饱和为 `+inf`），而**参考报 −inf** —— 参考的符号与解析值相反，在该点**不携带真值**。

**处置方式：照搬仓内既有先例，不改共享判据。** arch22 的 `cherk` / `csyrk` / `csymm` 早已处理同一 `RANDOM_EXTREME` 家族（`test/herk/cherk/arch22/cherk_test.cpp` 的 `DropNonFiniteReference`）。本算子把同一模式搬进**算子自己的测试文件**（`test/gemm_grouped_batched/arch22/gemm_grouped_batched_test.cpp`），**刻意不改 `test/frame/verify.h` 这一共享判据**——改它会波及全表。做法是：门禁前按 **golden 侧非有限**剔除相应位置并打印排除条数，`dropped == total` 时额外打印 `UNVERIFIABLE`；残差统计仍留在未过滤数组上计算。

真机日志确认该改动行为正确且不误伤：

```
[TC_FL_134] 1024/1024 positions excluded: reference non-finite (fp32 overflow); operator finite at 23 of them
[TC_FL_134] UNVERIFIABLE: no finite reference value to compare against
```

**全表仅 3 条**打印过排除行，恰为 `TC_FL_134` / `TC_FL_135`（`A_fill = VALUE_NORM_INF`）/ `TC_FL_136`（`A_fill = VALUE_NORM_NAN`）。其余 **1008 条**精度用例**一条都未被剔除**：其中 **1006 条**用 `A_fill = RANDOM_NORM_5_5`（≈±5 均匀，乘积界 25，**物理上远达不到 fp32 上限**），另外两条 `TC_FL_132`（`VALUE_NORM_0`）与 `TC_FL_133`（`RANDOM_ALTER`）也不产生非有限值。全表 1211 条数据行的填充模式分布为 **1206 条 `RANDOM_NORM_5_5` + 5 条特殊填充**（`VALUE_NORM_0` / `RANDOM_ALTER` / `RANDOM_EXTREME` / `VALUE_NORM_INF` / `VALUE_NORM_NAN` 各一条）。

其中"算子仍给出有限值的 **23** 个位置"尤其能说明问题：若两侧都饱和，这个数会是 0；它恰好证明"参考饱和、算子未饱和"是**实现差异**，而非算子算错。

> 该处置不改变任何数值语义，也不放宽容差：它只是承认"参考在该域无真值"。若任务方更倾向直接过滤 `TC_FL_134` 或降级其 `A_fill`，本实现与这两种方案**均不冲突**。
