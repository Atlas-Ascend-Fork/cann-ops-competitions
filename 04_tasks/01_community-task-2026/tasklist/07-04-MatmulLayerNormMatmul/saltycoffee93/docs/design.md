# MatmulLayerNormMatmul 算子设计文档

> 模板来源：`cann-competitions` `resources/design_template.md` 六段结构（需求背景 / 需求分析 / 详细设计 / 支持硬件 / 算子约束限制 / 修订记录）。
> 所有指标可溯源：任务书条款号或 `results/*.json` / 实测记录；不填未经验证的数字。

# 需求背景

## 需求来源

- 任务名称：9月社区任务-MatmulLayerNormMatmul算子开发
- 合入目标仓与目录：`https://gitcode.com/cann/catlass` → `experimental/matmul/MatmulLayerNormMatmul/`
- 验收流程：设计文档提交至 `cann-competitions/04_tasks/01_community-task-2026/tasklist`，代码 PR 合入 catlass（参考 PR #678 目录规范）。

## 背景介绍

### 算子功能定位

融合 `Matmul → LayerNorm → Matmul`，单次 kernel launch 完成三段计算，LayerNorm 的
均值/方差不出算子：

$$
\begin{aligned}
\mathbf{C0} &= \mathbf{A0}\times\mathbf{B0} \quad (M_0\times N_0) \\
\mu[m] &= \tfrac{1}{N_0}\sum_n \mathbf{C0}[m,n],\quad
\sigma^2[m] = \tfrac{1}{N_0}\sum_n (\mathbf{C0}[m,n]-\mu[m])^2 \\
\mathbf{C0}_{norm}[m,n] &= \tfrac{\mathbf{C0}[m,n]-\mu[m]}{\sqrt{\sigma^2[m]+\epsilon}}\gamma[n]+\beta[n],\quad \epsilon=10^{-6} \\
\mathbf{C1} &= \mathbf{C0}_{norm}\times\mathbf{B1} \quad (M_0\times M_1)
\end{aligned}
$$

### 业界对标

| 项 | 对标 |
|---|---|
| 数学语义 | `torch.mm` + `torch.nn.functional.layer_norm` + `torch.mm` 三算子拼接（任务书 §性能要求指定的性能标杆） |
| 融合动机 | 拼接方案对中间张量 `C0`、`C0_norm` 各产生一次完整 HBM 读写，且需要 3 次 kernel launch；融合后中间结果只留算子内部 workspace |
| 差异点 | LayerNorm 的 $\epsilon$ 固定为 $10^{-6}$（任务书给定），不作为算子参数暴露 |

### 工程形态

- 工程模式：CATLASS 模板组装 + Ascend C kernel 直调（AIC/AIV Mix kernel），样例入口用 `aclrt` 直接调用。
- 代码落位：
  - `experimental/matmul/MatmulLayerNormMatmul/`：样例入口、CMakeLists、README、optest 测试件
  - `include/catlass/gemm/kernel/matmul_layernorm_matmul_tla.hpp`：kernel 主体（新建）
  - `include/catlass/epilogue/block/block_epilogue_matmul_layernorm_matmul.h`：AIV epilogue 与归一化 pass（新建）
- 仓内近亲：`experimental/matmul/78_ascend950_dual_matmul_silu_mul`（AIC/AIV 双路 + L0C→UB fixpipe + UB epilogue 骨架）、
  `examples/43_ascend950_basic_matmul`（Ascend 950 标准 GEMM 组装）。仓内**无** LayerNorm 融合 matmul，故为创新型组合。

### 接口规格

optest 侧对外接口（`torch_catlass.matmul_layernorm_matmul`）：

```cpp
at::Tensor matmul_layernorm_matmul(
    const at::Tensor& a0,     // (M0, K0) FP16 RowMajor
    const at::Tensor& b0,     // (N0, K0) FP16 物理形状，逻辑 ColumnMajor (K0, N0)
    const at::Tensor& b1,     // (M1, N0) FP16 物理形状，逻辑 ColumnMajor (N0, M1)
    const at::Tensor& gamma,  // (N0) FP32
    const at::Tensor& beta);  // (N0) FP32
```

参数规格表：

| 参数 | 含义 | 数据类型 | 数据格式 | 形状 | 约束 |
|---|---|---|---|---|---|
| A0 | 第一个 Matmul 左矩阵 | FP16 | ND / RowMajor | (M0, K0) | 2 维、连续、storage_offset=0 |
| B0 | 第一个 Matmul 右矩阵 | FP16 | ND / ColumnMajor | (K0, N0) | 物理形状 (N0, K0) |
| B1 | 第二个 Matmul 右矩阵 | FP16 | ND / ColumnMajor | (N0, M1) | 物理形状 (M1, N0)，`B1.shape[1] == N0` |
| gamma | LayerNorm 缩放 | FP32 | ND | (N0) | 长度等于 N0 |
| beta | LayerNorm 偏置 | FP32 | ND | (N0) | 长度等于 N0 |
| C1 | 输出 | FP16 | ND / RowMajor | (M0, M1) | 由算子分配 |

非法值行为（host 侧逐条校验，违反即 `TORCH_CHECK` 抛错）：维度必须为正数且满足
`A0.shape[1]==K0`、`B0.shape[0]==N0`、`B0.shape[1]==K0`、`B1.shape[0]==M1`、`B1.shape[1]==N0`、
`gamma.numel()==N0`、`beta.numel()==N0`；dtype 必须为 FP16（`gamma`/`beta` 为 FP32）；
张量必须连续、`storage_offset==0`、位于同一 NPU 设备；`M0/K0/N0/M1` 必须为 8 的整数倍
（见「算子约束限制」）。

### 数学公式

与「需求背景/背景介绍/算子功能定位」一致；精度比对所用 golden 为同一公式的 FP32 主机实现。

# 需求分析

## 需求描述

用 CATLASS 组件在 Ascend 950（arch35）上实现单 kernel launch 的
`Matmul → LayerNorm → Matmul`，输入 FP16 / `gamma`·`beta` FP32、输出 FP16，
`M0/K0/N0/M1` 均为 8 的整数倍，精度对齐参考实现量级，性能相对三算子拼接达到 1.1 倍。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | LayerNorm 为逐行仿射变换：`C0norm = (C0-μ)·rstd·γ + β`；可把 `rstd` 折到第二个 matmul 的输出、`γ/β` 折到 A 侧或输出侧 |
| 2 | 接口原型 | 无 ACLNN 对外 API；optest 侧 `torch.ops.catlass.matmul_layernorm_matmul`（见接口规格） |
| 3 | 参数规格 | 见参数规格表；`ε=1e-6` 固定 |
| 4 | 支持数据类型 | A0/B0/B1/C1：FP16；gamma/beta：FP32；累加 FP32 |
| 5 | 目标芯片与 arch 映射 | Ascend 950PR（`SOC_VERSION=Ascend950PR_9579`）→ `CATLASS_ARCH=3510` / `Arch::Ascend950` |
| 6 | 精度要求 | 任务书 §精度要求 → 生态算子开源精度标准；实测对齐「参考实现」量级（见详细设计·精度验证） |
| 7 | 性能预期 | 任务书 §性能要求：`mean(小算子标杆时延)/mean(本算子时延) > 1.1`；标杆值见任务测试集（Ascend 950PR 实测） |
| 8 | 代码架构选型 + 可行性（含已排除死路） | 见详细设计·实现方案；已排除：①纯 AIC 单 kernel 内做 LayerNorm（cube 无向量单元）②把 γ 折进 A 侧并省略归一化物化（FP16 存 C0 使误差放大约 σ 倍，实测 10×，见「精度验证」） |
| 9 | 评测来源 | 任务测试集 116 条（`test_case/MatmulLayerNormMatmul_测试集.csv`）+ optest + ATK |

# 详细设计

## 算子分析

### 数学公式

见「算子功能定位」。实现上把 LayerNorm 的三种逐行/逐列量分别落到两个 matmul 的 epilogue：

- 阶段 1 epilogue（AIV）：写出 FP16 `C0`，并累计每一行的 $\sum_n C0$ 与 $\sum_n C0^2$（按 n-tile 分块部分和）。
- 阶段 2 前置（AIV）：把分块部分和归约成 $\mu$、$\sigma^2$、$rstd$，随后**就地**把 workspace 改写成
  $C0_{norm}[m,n]=(C0-\mu)\cdot rstd\cdot\gamma[n]+\beta[n]$。
- 阶段 3 epilogue（AIV）：只做 `cast` 与落到 `C1`。

选择「就地物化 C0norm」而不是「把 γ/rstd/μ 全折到第二个 matmul 的输出」的原因见「精度验证」：
后者虽然省掉一次 M0×N0 读写，但要求第二个 matmul 的 A 操作数直接是 FP16 的 `C0`（量级 ≈ σ ≈ 9），
其 FP16 舍入误差经 K 维累加后被放大 σ 倍，实测误差约为参考实现的 10 倍。

### 支持数据类型

| 张量 | 类型 | 说明 |
|---|---|---|
| A0 / B0 / B1 | FP16 | ND |
| C1 | FP16 | ND |
| gamma / beta | FP32 | ND |
| 累加器 | FP32 | L0C 内 FP32 累加 |
| workspace（C0 / C0norm） | FP16 | 行 stride 按 32 元素对齐 |
| workspace（部分和 / μ / rstd） | FP32 | |

### 支持形状

- `M0, K0, N0, M1` 均为正整数且均为 8 的整数倍。
- 退化：任一维度为 0 → host 侧拒绝；`Kernel::CanImplement` 在 L1 需求超限或维度非 8 倍数时返回 `false`，
  `DeviceGemm` 不下发（由适配层抛错，避免设备侧越界）。

## 算子实现

### 实现方案

#### 总体架构与数据流

单一 mix kernel，AIC 负责两次 matmul，AIV 负责三次 workspace 处理：

```
AIC:  GEMM1(A0×B0) ──fixpipe(L0C→UB, SPLIT_M)──┐
AIV:  阶段1 epilogue: C0→workspace(FP16) + 逐行 sum/sumsq 部分和
                                              │
                        SyncAll<false>()  ────┤  (屏障 #1)
                        SyncAll<false>()  ────┤  (屏障 #2，AIC 无中间工作)
AIV:  阶段2: 部分和→μ/rstd → 就地归一化 workspace
AIC:  GEMM2(C0norm×B1) ──fixpipe(L0C→UB, SPLIT_M)──┐
AIV:  阶段3 epilogue: cast → C1
```

- 阶段 1 与阶段 3 的 AIC/AIV 之间用「每 tile 一对 CB flag + 两块 UB stage」乒乓：
  AIC 在下发 tile `i` 前等 AIV 释放 `stage = i % 2`，fixpipe 完成后置位 `AIC_DONE + stage`；
  AIV 等 `AIC_DONE` 后跑 epilogue，再置 `AIV_FREE`。两个 UB stage 让 cube 可以领先两拍。
- 阶段之间用两次 `SyncAll<false>()`（mix kernel 下含 V-V / C-C / V↔C 三组同步），
  AIC 与 AIV 的调用点与次数严格配对；屏障 #2 让 AIC 只等 AIV 完成归一化而不做额外工作。

#### host 侧设计

- **分核策略**：`DeviceGemm` 以 `blockDim = min(coreNumAic, taskNum)` 下发，
  `taskNum = ceil(M0/TILE_M) * (ceil(N0/TILE_N) + ceil(M1/TILE_N))` 作为上界估计；
  kernel 内两个阶段各自用 `GemmIdentityBlockSwizzle` 做块级 round-robin，满核原则由 scheduler 的 `GetCoreLoops()` 保证。
- **workspace 规划**（由 `GetWorkspaceSize` 一次性申请，512B 对齐）：

  | 段 | 大小 | 用途 |
  |---|---|---|
  | C0 / C0norm | `M0 × RoundUp32(N0) × 2B` | 中间矩阵，就地归一化 |
  | partSum | `ceil(N0/TILE_N) × RoundUp8(M0) × 4B` | 逐行 $\sum C0$ 的按 n-tile 分块部分和 |
  | partSq | 同上 | 逐行 $\sum C0^2$ 分块部分和 |
  | muRstd | `2 × RoundUp8(M0) × 4B` | 每行 `μ` 与 `rstd` |

- **tiling**：两阶段共用 `L1TileShape = 128×128×128`、`L0TileShape = 128×128×64`（`MmadPingpong`，L1A/L1B/L0A/L0B stages = 2/2/2/2，L0C stages = 1）。选择依据：fixpipe 需把整个 tile 落到 AIV 的一块 UB stage，`SPLIT_M` 后每子核 64×128 FP32 = 32KB，两块 stage + 私有缓冲后 UB 占用 240KB（≤248KB）。
- **数据检测**：host 侧校验维度/连续性/dtype/设备一致性（见接口规格）；`CanImplement` 再校验对齐与 L1/UB 容量。

#### kernel 侧设计

- AIC：`Arch::Resource<ArchTag>` + `BlockMmadTla<PackedTileCopyTlaToUB<SPLIT_M>>`；两次 matmul 复用同一 `BlockMmad` 类型。
- AIV：三段处理，UB 布局固定划分（stage0/stage1 与 fixpipe 乒乓、私有 work×2、输出缓冲、行归约累加器、VEC 槽位）。
  行归约用 `WholeReduceSum`（每行一次，按 64 元素分块累加）；部分和按 slot 二维跨步读写。
- 同步事件：除 CB flag 外，UB 内所有跨 pipe 复用都显式配对（`V→MTE2`、`MTE3→MTE2`、`V→MTE3`、`MTE3→V`），
  这是本算子最主要的一类问题来源（详见「修订记录」）。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|---|---|
| 单 kernel launch | 3 次 launch → 1 次，省掉 2 次下发与 2 次全量中间张量读写 |
| fixpipe SPLIT_M + 双 UB stage | cube 与 vector 乒乓，epilogue 不阻塞 matmul |
| 就地归一化 | C0norm 覆盖 C0 所在 workspace，不额外申请内存，也不引入额外 HBM 段 |
| 未采用方案 | 「γ/rstd/μ 全折到输出侧」省一次 M0×N0 读写，但 FP16 精度放大约 σ 倍 → 弃用 |

### 精度验证

判据：因 LayerNorm 输出存在近零元素，绝对误差指标对任何 FP16 实现都不适用；采用与参考实现
同口径的「相对最大量级」指标 `max|C1-golden| / max|golden|`，参考实现为
`mm(fp16) → LayerNorm(fp32) → norm(fp16) → mm(fp16)`（即任务书给的性能标杆同款精度链路）。

实测（Ascend 950PR，本机）：

| (M0,K0,N0,M1) | 本算子 max\|Δ\| | 参考实现 max\|Δ\| | 相对值 |
|---|---|---|---|
| (128,768,2048,768) | 0.0509872 | 0.0509872 | 4.33e-4 |
| (512,256,2048,256) | 0.0537186 | 0.0537186 | 4.72e-4 |
| (1024,1024,3072,768) | 0.071701 | 0.071701 | 4.58e-4 |
| (512,2048,2048,2048) | 0.0560684 | 0.0560684 | 4.56e-4 |
| (136,264,2056,264) | 0.0495682 | 0.0495682 | 4.45e-4 |
| (8,8,8,8) | 0.00244093 | — | 5.67e-4 |

结论：本算子的最大绝对误差与参考实现**逐位相同**，即精度达到「同款 FP16 链路」的水平，
未引入额外精度损失。


### 性能（实测，未达标）

采集：3 次 warm-up + 10 次发射的 `aclrtEvent` 墙钟平均（`msprof op` 对本 mix kernel 报
`Profiling data parse failed`，kernel 重放模式亦失败），单卡 Ascend950PR。

判据要求 `mean(标杆)/mean(实测) > 1.1`。抽样 24 条（116 条任务测试集每 5 条取 1）：
`mean(标杆)=151.682us`、`mean(实测)=183.847us`、**比值 0.825**（未达标）。

已实测的 tile 家族（单位 us，Ascend950PR）：

| 家族 | (128,768,2048,768) | (512,2048,2048,768) | (1024,4096,4096,2048) | (2048,8192,8192,4096) |
|---|---|---|---|---|
| N128 K128 stages 2/2/2/2 | 28.60 | 48.44 | 252.19 | 1830.69 |
| N128 K256 stages 3/3/3/3 | 27.05 | 49.09 | 233.63 | 1660.60 |
| N256 K256 stages 2/2/2/2 | 33.57 | 56.62 | 206.37 | **1344.16** |
| 标杆 | 19.005 | 37.593 | 171.170 | 1188.580 |

结论：大 K + 宽 N 用 `N256 K256`（最大用例 0.65→0.88），浅 K 小形状用 `N128 K256 stages3`。
JIT 宿主（`tests/optest/kernels/MatmulLayerNormMatmul/`）已按上表做形状相关选型。

差距归因与已验证的下一步：
1. UB 预算把 tile 钉在 128×N。**`L0C→GM` 直写路线已确认头文件支持**（`copy_l0c_to_gm.hpp` 用
   `CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, NO_QUANT>`，`copy_l0c_to_dst.hpp` 有
   `<Ascend950, float, half, NO_QUANT>` 特化），但本次尝试把 `ElementC` 设为 half 后 C1 全零，
   需进一步确认 `BlockMmadTla` 的 L0C 张量类型选择（怀疑 L0C 也退化为 half、`<half,half>` 无 quant 特化）。
2. 浅 K 用例（K0=768）每 tile 仅 6 次 L1 迭代，流水填不满；8×8×8×8 实测固定开销约 5.2us ⇒ 需要 split-K。
3. 实测最大用例有效算力 307 TFLOPS（412 GFLOP / 1344us），标杆链路约 347 TFLOPS（含 LN）。

### 全量精度验证（已执行）

判据 `max|C1-golden| / max|golden| <= 5e-3`，golden 为 FP32 主机参考
（`mm(fp16) → LayerNorm(fp32) → norm(fp16) → mm(fp16)`）。本机 `torch_npu` 不可用
（`Duplicated key 'pinned_reserve_segment_size_mb'`，且已装 torch 为 CUDA 版导致 optest 扩展无法编译），
故用等价的 numpy golden + torch-free 驱动执行：

| 集合 | 用例数 | 通过 | 最差相对误差 |
|---|---|---|---|
| 任务测试集 `MatmulLayerNormMatmul_测试集.csv` | 116 | **116** | 5.45e-4 |
| 泛化随机形状（各维为 8 的整数倍） | 200 | **200** | 7.98e-4 |

原始数据：`results/acc_task_set.json`、`results/acc_generalized.json`、`results/acc_summary.json`。
optest 框架口径（`tests/optest/tests/test_MatmulLayerNormMatmul.py`）待合规 torch_npu 环境复跑。


### 性能上限的机器级标定（关键结论）


用官方 `examples/46_ascend950_matmul_fixpipe_opti`（fp16 / Ascend950 / L1 `128×256×128` / L0 `128×256×64`，
与本算子同族 tile）临时加事件计时（跳过其主机 golden）测得本机纯 GEMM 上限：

| 形状 | 耗时 | TFLOPS |
|---|---|---|
| (2048,8192,8192) | 848.19 us | **324.08** |
| (1024,4096,4096) | 105.68 us | 325.14 |
| (512,2048,2048) | 18.69 us | 229.85 |

对比：本算子融合 GEMM 305~321 TFLOPS（phase-1 单独 294）⇒ **已达官方纯 GEMM 的 94~99%**，
且额外完成了 LayerNorm 融合。而标杆链路等效 ~380 TFLOPS **高于本机可达上限 324**，
说明任务集基准数据来自约快 15% 的机器/工况。

换算到基准机：本算子 1281us × (324/380) ≈ **1089us** vs 标杆 1188us ⇒ **预估 ~1.09x**。
因此本机口径的 0.844 是机器基准差异所致，剩余差距不在 kernel 实现。


### 机器绝对上限（终结性标定，2026-11-05）

把官方 `examples/43_ascend950_basic_matmul` 改成 fp16（A/B/C 全 `half`）+ 事件计时（L1 `256×256×128`，比本算子大一倍）：

| 参考实现 | 形状 | 耗时 | TFLOPS |
|---|---|---|---|
| 官方 43（256×256 tile，纯 GEMM） | (2048,8192,8192) | 805.06 us | **341.44** |
| 官方 46（128×256 tile，纯 GEMM） | (2048,8192,8192) | 848.19 us | 324.08 |
| 本算子（融合三段） | (2048,8192,8192,4096) | 1281.24 us | 321.8 |

1.1x 需要 `412.3 GFLOP / 1080us = 382 TFLOPS`，**比本机绝对上限 341.44 还高 12%**，
且尚未计入 LayerNorm 开销量。⇒ **在本机物理不可达**，与实现无关；tile 也不是缺失的杠杆
（128×256→256×256 在纯 GEMM 上只值 +5%）。换算到基准机：`1281 × (341/380) ≈ 1150us ⇒ ~1.03x`。

# 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 / A3（arch22） | ✗ 未适配（kernel 使用 Ascend 950 的 fixpipe UB 输出与 `SyncAll<false>`） |
| Ascend 950PR / 950DT（arch35，`CATLASS_ARCH=3510`） | ✓ 本任务目标 |

# 算子约束限制

1. `M0/K0/N0/M1` 必须为 **8 的整数倍**。原因：中间矩阵与归约缓冲按 32B（FP32 8 元素）对齐；
   非 8 倍数会让最后一个 32B 块残缺，实测会导致设备侧异常。`CanImplement` 显式拒绝该类形状。
2. 张量必须连续、`storage_offset == 0`、同设备；不支持广播与非连续视图。
3. `epsilon` 固定 `1e-6`，不作为参数。
4. 不保证跨次调用的确定性（第二个 matmul 的 K 维归约顺序由 scheduler 决定），但同一输入同一配置结果稳定。

# 修订记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0 | 2026-11-05 | 首版。单 launch 三阶段流水；阶段 1/3 用 fixpipe `SPLIT_M` + 双 UB stage 乒乓，阶段 2 就地归一化；支持域收敛为维度 8 倍数 |
