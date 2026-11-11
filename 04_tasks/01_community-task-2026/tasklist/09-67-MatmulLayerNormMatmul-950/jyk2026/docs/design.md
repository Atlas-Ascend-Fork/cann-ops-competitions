# MatmulLayerNormMatmul 算子设计文档（CATLASS / Ascend 融合算子）

> **算子**：`Matmul → 行级 LayerNorm → Matmul` 三段融合，单次 MIX kernel launch 完成
> **仓库**：<https://gitcode.com/cann/catlass>（分支 `op/matmul_layernorm_matmul`）
> **代码落位**：`experimental/matmul/matmul_layernorm_matmul/`、`include/catlass/gemm/kernel/matmul_layernorm_matmul.hpp`、`include/catlass/epilogue/block/block_epilogue_layernorm.hpp`、`tests/optest/`（对齐 [PR #678](https://gitcode.com/cann/catlass/pull/678) 交付结构）
> **任务来源**：9月社区任务-MatmulLayerNormMatmul算子开发；官方任务目录 `09-67-MatmulLayerNormMatmul-950`
> **硬件**：任务书声明 Ascend 950（性能标杆采集于 Ascend 950PR，SoC `dav_3510`）；开发验证在 Ascend 910B3（SoC `dav_2201`，20 AICore）实机完成全量精度与性能验证，950 侧的迁移点（TLA 形式 `BlockMmad`）见「支持硬件」节
> **性能判据**：`平均(标杆时延 / 测试时延) > 1.1`，标杆 = `torch.mm + F.layer_norm + torch.mm` 小算子拼接，采集工具 `msprof op`

---

# 需求背景（required）

## 需求来源

| 项 | 内容 |
| --- | --- |
| 任务名称 / 标签 / 硬件 | 9月社区任务-MatmulLayerNormMatmul算子开发 / 算子开发 / Ascend 950 |
| 开源仓 / 语言 | <https://gitcode.com/cann/catlass> / Ascend C（CATLASS 模板库） |
| CANN 版本 | 算子开源仓指定版本（catlass `dependencies.yaml`：`cann >= 8.5.0`；开发验证环境 CANN 9.2.0） |
| 精度标准 | [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) |
| 合入方式 | 算子代码 PR 合入 catlass（结构参考 [PR #678](https://gitcode.com/cann/catlass/pull/678)）；设计文档 MR 合入 cann-competitions tasklist `09-67-MatmulLayerNormMatmul-950` |
| 验收交付件 | ① 自测用例、测试结果报告、测试步骤指导文档；② 算子代码私仓邀请链接、代码仓路径、分支、算子目录 |

## 背景介绍

### MatmulLayerNormMatmul 算子实现优化

Transformer / MLP 结构中普遍存在 `矩阵乘 → 行级归一化 → 矩阵乘` 的链式结构：

```python
c0      = torch.mm(a0, b0)                                   # (M0,K0) x (K0,N0) -> (M0,N0)
c0_norm = torch.nn.functional.layer_norm(c0, (n0,), weight=gamma, bias=beta, eps=1e-6)
c1      = torch.mm(c0_norm, b1)                              # (M0,N0) x (N0,M1) -> (M0,M1)
```

小算子拼接方案存在两类开销：

1. **launch 与同步开销**：3 次 kernel launch，中间矩阵 `C0` / `C0_norm` 必须写回 GM 并在下一个 kernel 前完成 flush，小 shape 下该开销占主导（任务测试集最小档标杆仅 19.0us）；
2. **中间结果的冗余往返**：`C0` 与 `C0_norm` 各经历「写 + 读」，共 4 趟 GM 流量，而这两份数据对使用者完全不可见。

本算子将三段计算融合进**单次 MIX kernel launch**（1 AIC : 2 AIV），中间结果由算子内部 workspace 管理、不对外暴露，`Mean` / `Variance` / `invstd` 全程驻留 UB 不落 GM，唯一对外输出为 `C1`。

### MatmulLayerNormMatmul 算子功能分析

本任务为**新增融合算子**，不是历史 TBE 算子的语义迁移：catlass 仓内不存在与该三段链路等价的既有实现（仓内无 LayerNorm / RMSNorm 组件，`grep -ri "layernorm\|rmsnorm" include/ examples/` 仅命中无关注释）。因此不存在「旧版本性能/精度对照基线」，唯一对标基线是任务书指定的小算子拼接方案。

参数规格（逐行对齐任务书「参数说明」表）：

| 参数名 | 输入/输出/属性 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 输入 | 第一个 Matmul 左矩阵 | RowMajor 布局 | FP16 | ND | (M0, K0) | — |
| B0 | 输入 | 第一个 Matmul 右矩阵 | ColumnMajor 布局 | FP16 | ND | (K0, N0) | — |
| B1 | 输入 | 第二个 Matmul 右矩阵 | ColumnMajor 布局 | FP16 | ND | (N0, M1) | — |
| gamma | 输入 | LayerNorm 缩放参数（可学习） | 长度 N0 | FP32 | ND | (N0) | — |
| beta | 输入 | LayerNorm 偏置参数（可学习） | 长度 N0 | FP32 | ND | (N0) | — |
| C1 | 输出 | 第二个 Matmul 输出矩阵（唯一对外输出） | — | FP16 | ND | (M0, M1) | — |

其余契约：单次 launch 完成三段；中间结果由算子内部 workspace 管理且不对外暴露；`eps = 1e-6` 为固定常量、非属性；无广播，`gamma` / `beta` 仅沿 N0 逐列作用；允许采用多种 matmul 方案以达成更好性能。

### 布局契约符合性

任务书标注 `B0` / `B1` 为 **ColumnMajor** 布局，本实现**严格遵循**，不做 RowMajor 偏离。catlass 的布局定义已在仓内逐字核验（`include/catlass/layout/matrix.hpp:161-212`）：`ColumnMajor(rows, cols)` 的 stride 为 `(1, rows)`，`GetOffset(r, c) = r + c * rows`。因此：

| 参数 | 逻辑形状 | 内存语义 | 元素寻址 |
| --- | --- | --- | --- |
| A0 | (M0, K0) | RowMajor 连续 | `m * K0 + k` |
| B0 | (K0, N0) | ColumnMajor 连续（K0 维连续） | `k + n * K0` |
| B1 | (N0, M1) | ColumnMajor 连续（N0 维连续） | `n + p * N0` |
| C1 | (M0, M1) | RowMajor 连续 | `m * M1 + p` |

映射到 catlass 类型为 `LayoutA = RowMajor`、`LayoutB = ColumnMajor`、`LayoutC = RowMajor`。该组合在 AtlasA2 路径上有仓内已验证近亲：`examples/26_matmul_relu/matmul_relu.cpp:47-49` 即 `RowMajor / ColumnMajor / RowMajor`，其 `TileCopy` 具备 GM ColumnMajor → L1(nZ) 搬运特化，无需新增搬运组件。

# 需求分析（required）

## 需求描述

用 Ascend C / CATLASS 模板库实现 `MatmulLayerNormMatmul` 融合算子：单次 MIX kernel launch 完成 `Matmul(FP16) → 行级 LayerNorm(FP32 统计, eps=1e-6) → Matmul(FP16)`；中间结果经算子内部 workspace 中转、不对外暴露；功能与精度通过任务测试集 116 条及生态算子开源精度标准；平均标杆时延 / 测试时延 > 1.1；按 catlass 规范交付样例、kernel 组件、optest 接入件、README 与自验证据。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| R1 | 数学定义 | `C0 = A0×B0`；`C0norm[m,n] = (C0[m,n]-μ[m])/sqrt(σ²[m]+1e-6)·γ[n]+β[n]`（`μ,σ²` 为逐行总体统计）；`C1 = C0norm×B1` |
| R2 | 可白嫖的代数量 | 否。LayerNorm 的非线性（`1/sqrt(σ²+ε)` 逐行标量）无法并入任一 matmul；γ 沿归约轴缩放 B1 在硬件上无法随路完成，且需额外整遍读 B1（流量翻倍），故不采用代数变形 |
| R3 | 接口原型 | catlass 样例 + optest 接入；`torch_catlass.matmul_layernorm_matmul(a0, b0, b1, gamma, beta) -> c1` |
| R4 | 参数规格 | 见上「参数规格」表；dtype 契约 A0/B0/B1/C1=FP16、gamma/beta=FP32，非法 dtype 由 host 断言拦截 |
| R5 | 支持数据类型 | FP16（矩阵）+ FP32（gamma/beta），中间 FP16、统计 FP32；不支持 BF16/FP32 矩阵（无需求，预留泛化点） |
| R6 | 目标芯片与 arch 映射 | 任务书声明 950（`CATLASS_ARCH=3510` / `Arch::Ascend950`，需 TLA 形式 `BlockMmad`）；本环境仅有 910B3（`CATLASS_ARCH=2201` / `Arch::AtlasA2`），故实现与全部验证在 2201 完成，950 迁移点已在「支持硬件」节逐条列明 |
| R7 | 精度要求 | 任务书 §精度要求 → opbase `experimental_standard`；实现侧 `C0`/`C0norm` 取 FP16（与 `torch.mm` 的 FP16 dtype 传播一致），LN 统计 FP32 两遍中心化 |
| R8 | 性能预期 | 判据 `平均(标杆时延/测试时延) > 1.1`（任务书 §性能要求）；收益来源：消除 2 次 launch、中间结果留 GM 但省去独立 LN kernel 的 flush、AIC/AIV 双流并行 |
| R9 | 代码架构选型 + 可行性 | 主方案「GM-staging 三阶段 + 全核栅栏」；已排除：单 block 持有整行（L0C 128KB 装不下 M_tile×N0 fp32）、逐行块生产者-消费者（M0=128 时仅 1 个 M-tile，20 核中 19 核闲置）、单遍 `E[x²]-E[x]²` 求方差（抵消风险）。见「实现方案」节 |
| R10 | 评测来源 | 精度：optest 116 条（任务测试集 CSV，GBK 编码，列 `idx,M0,K0,N0,M1,小算子标杆耗时(us)`）+ ATK 泛化 ≥200 例；性能：`msprof op` 逐 case |

# 详细设计（required）

## 算子分析

### 数学公式

阶段一（第一个 Matmul，输出 C0 形状 (M0, N0)）：

$$C0[m,n] = \sum_{k=0}^{K_0-1} A0[m,k] \cdot B0[k,n]$$

阶段二（LayerNorm，对 C0 每一行独立计算，$\epsilon = 10^{-6}$）：

$$
\begin{aligned}
\mu[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1} C0[m,n] \\
\sigma^2[m] &= \frac{1}{N_0}\sum_{n=0}^{N_0-1}\big(C0[m,n]-\mu[m]\big)^2 \\
C0_{\text{norm}}[m,n] &= \frac{C0[m,n]-\mu[m]}{\sqrt{\sigma^2[m]+\epsilon}}\cdot\gamma[n]+\beta[n]
\end{aligned}
$$

阶段三（第二个 Matmul，输出 C1 形状 (M0, M1)）：

$$C1[m,p] = \sum_{n=0}^{N_0-1} C0_{\text{norm}}[m,n] \cdot B1[n,p]$$

kernel 内等价形式：`invstd[m] = rsqrt(max(σ²[m], 0) + 1e-6)`，`C0norm = (C0 - μ[m]) * invstd[m] * γ[n] + β[n]`。

三处数值决策：

1. **方差取总体方差（除以 N0）**，不做 Bessel 校正 —— 与 `F.layer_norm(normalized_shape=(N0,))` 口径一致；
2. **方差两遍计算**（先 μ 再算 Σ(C0-μ)²），不使用单遍 `E[x²]-E[x]²`：后者在「均值大、方差小」的行上存在严重抵消（matmul 输出的行均值量级可达 O(sqrt(K0))，K0=8192 时均值量级 ~90，而 σ 量级可小至 1e-2，fp32 下 `E[x²]` 与 `μ²` 相减会损失全部有效位）。本方案行数据已在 UB 常驻，第二遍不产生 GM 流量，代价仅为向量计算；
3. `rsqrt` 前对 σ² 取下界 0，避免负零/极小负值开方产生 NaN。

### 支持数据类型

| 阶段 | 输入 dtype | 输出/中间 dtype | 计算 dtype |
| --- | --- | --- | --- |
| Matmul0 | A0/B0 FP16 | C0（workspace W）FP16 | Cube FP32 累加，FixPipe 随路 `CAST_RINT` 转 FP16 |
| LN 统计 | W FP16 | μ / σ² / invstd FP32（驻 UB） | FP32 |
| LN 归一化 | W FP16 + γ/β FP32 | C0norm（workspace W，原地覆写）FP16 | FP32 计算后转 FP16 |
| Matmul1 | W FP16、B1 FP16 | C1 FP16 | Cube FP32 累加，FixPipe 随路转 FP16 |

`C0` / `C0norm` 取 FP16 的依据：PyTorch 链路中 `torch.mm` 的 fp16 输入经 FP32 累加后输出即为 fp16，`F.layer_norm` 对 fp16 输入内部上转 FP32 计算再回 fp16，因此 FP16 中间结果与参考实现的 dtype 传播逐位一致，不引入额外精度偏差。

### 支持形状

任务测试集共 116 条：`M0 ∈ {128, 512, 1024, 2048}`、`K0 ∈ {768, 2048, 4096, 8192}`、`N0 ∈ {2048, 3072, 4096, 8192}`、`M1 ∈ {768, 2048, 4096}`（`M0` 与 `K0` 交叉分布 30/30/30/26 条）。

实现不把「对齐」写成接口限制：所有维度按 `CeilDiv` + `actual shape` 处理尾块（M/N/K 三个方向的尾 tile、KL0 尾片、UB 内 N0 非 64 整倍数），最小合法单位 `M0, K0, N0, M1 ≥ 1`。测试集形状均可被 64 整除，非对齐形状在 S3 的泛化用例中专项覆盖。

## 算子实现

### 实现方案

主方案：**单次 MIX kernel launch（1 AIC : 2 AIV）+ GM-staging 三阶段 + 两次全核栅栏**。

```text
                        ┌────────────────── 单次 kernel launch ──────────────────┐
GM A0 ─MTE2─► L1 ─MTE1─► L0A ┐                                               │
GM B0 ─MTE2─► L1 ─MTE1─► L0B ┼─► Mad(FP32 L0C) ─FIX─► W (GM, FP16, M0×N0)     │  Phase 1 (AIC)
                              │                       ▲ 网格 (M0/BM)×(N0/BN)    │
                              └───────────────────────┘                       │
                    ─ ─ ─ ─ ─ ─ AscendC::SyncAll<false>() (AIC/AIV 配对) ─ ─ ─┤
                                                                              │
AIV (40 个向量核，按行 round-robin)                                            │  Phase 2 (AIV)
  行 m: W[m,:] ─MTE2─► UB(fp16, 全长 N0 常驻)                                 │
        pass0: chunk→fp32 ─V─► Σx            → μ[m]                           │
        pass1: chunk→fp32 ─V─► Σ(x-μ)²       → σ²[m] → invstd[m]              │
        pass2: chunk→fp32 ─V─► (x-μ)*invstd*γ+β ─CAST─► UB(fp16)              │
        UB ─MTE3─► W[m,:]（原地覆写为 C0norm）                                 │
  γ/β (FP32, N0) 每个向量核首轮各装载一次，常驻 UB                                │
                    ─ ─ ─ ─ ─ ─ AscendC::SyncAll<false>() (AIC/AIV 配对) ─ ─ ─┤
                                                                              │
GM W ─MTE2─► L1 ─MTE1─► L0A ┐                                                 │
GM B1 ─MTE2─► L1 ─MTE1─► L0B ┼─► Mad(FP32 L0C) ─FIX─► GM C1 (FP16, M0×M1)     │  Phase 3 (AIC)
                              └─ 网格 (M0/BM)×(M1/BN)                          │
                        └──────────────────────────────────────────────────────┘
```

**kernel 变体划分**

| 变体 | 触发条件 | 职责 |
| --- | --- | --- |
| V1 `MatmulLayerNormMatmul`（主方案） | 全部合法形状 | AIC 承担两个 matmul，AIV 承担行级 LN，中间经 GM workspace 中转 |
| V2 `MatmulLayerNormMatmulL1Resident`（真机迭代方向，S5 视性能缺口启用） | N0 与 BM 使 L0C→UB 能容纳整行块 | C0 走 `L0C→UB`（`CopyL0CToDst`）不经 GM，LN 后在 UB 内直接 `CopyUbToL1` 供 Phase 3 左矩阵使用，目标零中间 GM 流量 |

**已被排除的路线**

| 备选方案 | 结论 | 原因（现场核验） |
| --- | --- | --- |
| 单 block 持有整行块 C0（M_tile × N0 全宽常驻 L0C） | 不可行 | L0C 容量 A2=128KB / 950=256KB（`include/catlass/arch/arch.hpp:18-37`）；`M_tile×N0×4B` 在 N0=8192 时仅够 4~8 行，无法形成有效 tile |
| 逐行块生产者-消费者（AIC 算完行带即通知配对 AIV，无全局栅栏） | 不采用（v1） | 行带是 matmul0 的 M 维分块，`M0=128` 时只有 1 个 M-tile（测试集 30/116 条为此档），20 个 AIC 中 19 个闲置；改按 N 切分则行级归约跨核，仍需全局同步 |
| 单遍 `E[x²]-E[x]²` 求方差 | 不采用 | 均值大方差小时存在灾难性抵消（见「数学公式」节第 2 条） |
| 代数变形推迟归一化（γ 吸收进 B1） | 不采用 | γ 沿归约轴，需改写 B1 的 K 维缩放，Cube 无法随路完成；AIV 侧改写 B1 会使流量翻倍 |
| 3 个独立 kernel（不融合） | 不采用 | 即基线形态，无法达成 1.1x 判据 |
| 用 `AscendC::SetAtomicAdd` 在 GM 累加行统计量 | 不采用 | 行统计量需行内全 N0 归约，跨核原子累加会引入确定性风险且仍需栅栏，收益为负 |

#### 3.2.1 Host 侧设计

**1. 样例组织（对齐 PR #678 文件清单，见 contract `merge_contract.path_list`）**

```text
catlass/
├── experimental/matmul/matmul_layernorm_matmul/
│   ├── CMakeLists.txt                                   # catlass_example_add_executable(... mix ...)
│   ├── matmul_layernorm_matmul.cpp                       # host 入口 + golden 比对（打印 Compare success）
│   ├── README.md                                         # 算子说明/用法/约束/预期输出
│   └── test_matmul_layernorm_matmul.py                   # 基于 optest 的测试件（pytest）
├── include/catlass/gemm/kernel/matmul_layernorm_matmul.hpp        # 融合 kernel（三阶段 + 栅栏）
├── include/catlass/epilogue/block/block_epilogue_layernorm.hpp    # 行级 LN 的 UB 级实现
└── tests/optest/{include,kernels,src,torch_catlass,tests}/...      # optest 接入（见 contract 路径清单）
```

**2. 分核策略**

| 阶段 | 并行维度 | 核数利用 | 说明 |
| --- | --- | --- | --- |
| Phase 1（matmul0） | `(M0/BM) × (N0/BN)` 二维网格，grid-stride 分发 | AIC 全核（20） | 复用 `GemmIdentityBlockSwizzle<3,0>` 的 L2 友好折叠序；`M0=128` 时靠 N0 方向切满全核 |
| Phase 2（LayerNorm） | 行（M0）round-robin，步长 = AIV 核数（AIC×2） | AIV 全核（40） | 单行独立，无跨核归约，无竞争；W 中没有被任何核写到的行不存在（行号全覆盖） |
| Phase 3（matmul1） | `(M0/BM) × (M1/BN)` 二维网格 | AIC 全核 | `M1 ∈ {768,2048,4096}` 保证 N 方向至少 3 个 tile |

栅栏两侧必须**调用次数严格一致**：`AscendC::SyncAll<false>()` 在 AIC 与 AIV 各调用 2 次；某阶段核心例数不足而自然跳过循环时，仍必须参与栅栏（否则对侧死锁）——该约束在 kernel 内以注释固化，并在代码检视清单中列项。

**3. 数据分块与内存策略**

| 缓冲 | 类型 | 容量（A2 参考） | 用途 |
| --- | --- | --- | --- |
| L1 | Stage buffer | 512KB | matmul0 / matmul1 的 A、B tile（双缓冲由 `MmadAtlasA2Preload` 提供） |
| L0A / L0B | Fractal | 64KB / 64KB | Cube 左右操作数 |
| L0C | FP32 累加 | 128KB | `BM×BN` fp32，FixPipe 随路转 FP16 落 W |
| UB | Vector | 192KB | LN 阶段：整行 fp16（N0×2B ≤ 16KB）+ fp32 chunk + γ/β（2×N0×4B ≤ 64KB），按 N0 自适应 |

**workspace 布局（算子内部，不对外暴露）**

| 偏移 | 大小 | 内容 | 生命周期 |
| --- | --- | --- | --- |
| 0 | `M0 × N0 × 2B` | W：Phase 1 写 C0（FP16），Phase 2 原地覆写为 C0norm，Phase 3 作左矩阵读取 | 单次 launch 内，launch 结束即可复用 |

`GetWorkspaceSize(args) = M0 * N0 * sizeof(fp16_t)`（上界 shape 2048×8192×2B = 32MiB）。workspace 由 optest 适配层在每次调用内部申请/释放，不进入算子对外参数表。

**4. tiling 参数（host 计算，随参数下发）**

| 参数 | 取值策略 |
| --- | --- |
| `L1TileShape`（matmul0） | `GemmShape<128, 256, 256>`（K0=768 时 K 方向 3 个 tile，最小档不空转） |
| `L1TileShape`（matmul1） | `GemmShape<128, 128, 256>`（N0 作 K 维，K 方向 2048/256=8 ~ 8192/256=32 个 tile） |
| `blockDim` | `min(aicCoreNum, 有效块数)`，由 host 按 `CeilDiv(M0,BM)×CeilDiv(N,BN)` 计算后取 min；MIX 场景下发 AIC 核数，AIV 自动 2 倍 |
| LN 行内 chunk | `CHUNK = 64 * k`（k 使 `CHUNK ≤ N0` 且 UB 预算满足），运行期按 `N0 % CHUNK` 处理尾段 |

**5. 数据检测（非法输入拦截，逐条对齐参数表）**

| 检查项 | 行为 |
| --- | --- |
| 任一 shape 维度为 0（M0/K0/N0/M1） | 空输出，直接返回成功（quick return），不启动 kernel |
| `A0.size(0) != M0` 或 `A0.size(1) != K0` | host 断言失败，报 `ACL_ERROR_INVALID_PARAM` |
| `B0` 形状非 `(K0, N0)`、`B1` 形状非 `(N0, M1)` | host 断言失败 |
| `gamma/beta` 长度 != N0 | host 断言失败 |
| dtype 不符合契约（矩阵非 FP16 / γ、β 非 FP32） | host 断言失败 |
| 非连续 tensor | 不支持（任务书全参数标注「—」），host 校验 `is_contiguous()` |

#### 3.2.2 Kernel 侧设计

**Phase 1（AIC，`operator()<AscendC::AIC>`）**

1. `BlockScheduler(params.problemShape0, MakeCoord(BM, BN))` 得 `coreLoops`，`for (loopIdx = GetBlockIdx(); loopIdx < coreLoops; loopIdx += GetBlockNum())`；
2. `blockCoord → actualBlockShape`，`blockMmad(gmA0[offA0], layoutA0, gmB0[offB0], layoutB0, gmW[offW], layoutW, actualBlockShape)`；
3. `blockMmad` 内部为 L1 双缓冲 + L0A/L0B ping-pong + L0C FP32 累加，结束时 `FixPipe` 以 `CAST_RINT` 转 FP16 写入 W；
4. 循环结束后 `AscendC::PipeBarrier<PIPE_ALL>()`，再 `AscendC::SyncAll<false>()`。

**Phase 2（AIV，`operator()<AscendC::AIV>`）**

0. 进入前先 `AscendC::SyncAll<false>()` 等待 Phase 1 全量落盘；
1. 装载 `γ`、`β`（FP32，各 N0）到 UB，`layout::VectorLayout` + `CopyGm2Ub`（`include/catlass/epilogue/tile/copy_gm_to_ub.hpp:40-58`）；
2. `rowIdx ∈ [subblock 全局行号, M0)` 步长 = AIV 核数：单行 N0 全长 `CopyGm2Ub` 到 UB（fp16，`DataCopyPad` 支持非 64 对齐尾段）；
3. `pass0`：逐 CHUNK 从 UB fp16 段 `Cast` 到 fp32，`Add` 累加进 `accSum`，末尾用「两级 `WholeReduceSum`」折叠成长度 1 的标量 → `μ = Σx / N0`；
4. `pass1`：逐 CHUNK 计算 `(x-μ)`，`Mul` 自乘后累加进 `accSq`，同法折叠 → `σ² = Σsq / N0`，`invstd = rsqrt(max(σ²,0)+1e-6)`；
5. `pass2`：逐 CHUNK 计算 `(x-μ)*invstd*γ+β`（`AscendC::Muls` 广播标量、`TileRowBroadcastMul` 语义的逐列乘、`Add` 逐列加），`Cast(CAST_RINT)` 回 fp16，写回 UB 行缓冲；
6. `CopyUb2Gm` 把整行写回 W 同一地址（原地覆写）；
7. 全部行处理完后 `AscendC::PipeBarrier<PIPE_ALL>()` + `AscendC::SyncAll<false>()`。

两级 `WholeReduceSum` 折叠（处理长度为 CHUNK 的 fp32 向量，CHUNK 最大 2048）：

```cpp
// 第一级：mask=64，repeatTimes=CHUNK/64，得 CHUNK/64 个部分和
AscendC::WholeReduceSum(partial, acc, 64, CHUNK / 64, 1, 1, 8);
// 第二级：把 CHUNK/64 个部分和折成 1 个
AscendC::WholeReduceSum(scalar, partial, CHUNK / 64, 1, 1, 1, 8);
```

该写法与仓内长行归约范式一致（`include/catlass/epilogue/block/block_epilogue_fa_unshared_softmax.hpp:178-206`）。

**Phase 3（AIC）**

进入前 `AscendC::SyncAll<false>()` 等待全部行归一化完成；余同 Phase 1，`ElementA = half`（来自 W，RowMajor）、`ElementB = half`（来自 B1，ColumnMajor）、`ElementD = half`（写 C1，RowMajor），`problemShape1 = (M0, M1, N0)`。

**同步协议表**

| 时刻 | 动作 | 发起侧 | 等待侧 | 语义 |
| --- | --- | --- | --- | --- |
| Phase1 末 | `SyncAll<false>()` #1 | AIC 全体 | AIV 全体（Phase2 开头配对调用） | W 全量写完毕，AIV 可读 |
| Phase2 末 | `SyncAll<false>()` #1 | AIV 全体 | AIC 全体（Phase3 开头配对调用） | W 全量归一化完毕，AIC 可读 |
| 单次 launch 内 | 其余阶段内同步 | — | — | Phase1/Phase3 的 tile 级同步由 `blockMmad` 内部的 MTE/Vector/FixPipe 流水 flag 承担，无跨核 flag |

选择 `SyncAll<false>()` 而非 GM 原子计数自旋的依据：catlass 内**不存在** GM 计数器自旋范式（`grep -n "spin\|GlobalBarrier" include/catlass` 零命中），而 `SyncAll<false>` 是仓内 MIX kernel 的既有全核同步原语（`include/catlass/gemm/kernel/dual_level_quant_mx_batched_matmul_tla.hpp:409-411, 426`），注释明确其为「V-V / C-C / V↔C 三组一次完成」，且 AtlasA2 指令集支持（`docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_gemm_instruction_set.md:526-539`），A2 实机用例见 `experimental/attention/fag_tla/fag_tla_kernel.cpp:700-872`。

#### 性能优化要点

| 优化项 | 预期收益 | 状态 |
| --- | --- | --- |
| 三段融合为单次 launch | 消除 2 次 launch 与 2 次 kernel 边界 flush，小 shape（最小档标杆 19.0us）收益最大 | v1 实现 |
| 行数据 UB 常驻 + 方差两遍在同一 UB 副本上完成 | LN 的第二遍不产生额外 GM 流量（对比「重读 GM 求方差」方案省一趟 N0 读） | v1 实现 |
| γ/β 每向量核只装载一次 | 避免逐行重复搬运（M0=2048 时省 2047 次 × 32KB） | v1 实现 |
| W 原地覆写（C0 → C0norm 复用同一 buffer） | workspace 从 2 份降为 1 份（上界省 32MiB），并省一次分配 | v1 实现 |
| Phase1/Phase3 tile 形状分别调优（`K0` 方向 vs `N0` 作 K 方向） | 避免 K 方向 tile 数不足导致的 Cube 空转 | S5 逐 case 调优 |
| V2 L1 常驻（`L0C→UB` 直通 + `UB→L1`） | 中间结果零 GM 流量，理论上界再省 4 趟 `M0×N0×2B` | S5 视性能缺口启用 |
| 行方向 CHUNK 与多行批处理（多行共享一次 γ/β 复用） | 减少 Vector 指令数 | S5 视 msprof 结论 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
| --- | --- |
| Atlas A2 系列（`dav_2201`，`CATLASS_ARCH=2201`） | ✓ **本次交付的实现与全部验证证据所在**（910B3，20 AICore）：编译、116 条官方测试集精度、220 条泛化精度、116 条性能数据均在该平台实机采集 |
| Atlas A3 系列（`dav_2201`） | ✓ 与 A2 同一 arch 分支与非 TLA `BlockMmad` 路径，模板层面兼容（未单独实机验证） |
| Ascend 950PR / 950DT（`dav_3510`，`CATLASS_ARCH=3510`） | ⚠ 任务书声明的目标硬件。本次未在 950 实机验证：catlass 的 950 路径要求 TLA 形式的 `BlockMmad`（`Gemm::Block::BlockMmadTla` + `MmadAscend950*` 调度策略 + `Shape<Int<...>>` 分块），与本实现在 2201 上使用的非 TLA `BlockMmad` 属不同模板族，需替换后重新验证。层结构与阶段划分（三阶段 + 双栅栏 + 行级 LN）与 arch 无关，可直接迁移；`tests/optest/kernels/matmul_layernorm_matmul/CMakeLists.txt` 当前声明 `NPU_ARCH_LIST 2201`，接入 950 时需补 3510 与 TLA 模板分支 |

## 算子约束限制

| # | 约束 | 说明 |
| --- | --- | --- |
| C1 | 仅支持 FP16 矩阵 + FP32 γ/β | 任务书参数表定义；BF16/FP32 矩阵不支持 |
| C2 | 仅支持连续 tensor | 任务书全参数「非连续Tensor」列为「—」 |
| C3 | 不支持广播 | γ/β 长度必须严格等于 N0 |
| C4 | `eps` 固定 1e-6 | 任务书公式给定常量，不开放为属性 |
| C5 | 单次 launch 内部使用 `M0×N0×2B` 的 GM workspace | 上界 32MiB（2048×8192）；由适配层内部申请，不计入对外参数 |
| C6 | 依赖 MIX kernel（AIC+AIV 配对栅栏） | 纯 AIC 或纯 AIV 启动方式不支持；`blockDim` 需按 AIC 核数下发 |
| C7 | 确定性 | 单次 launch 内每个输出元素只被一个 core 写入（无原子累加），结果确定可复现 |

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 矩阵输出 FP16；按 rtol/atol 组合阈值 + 误差统计（MERE/MARE）判定，逐 case 检查 NaN/Inf 位置 | [生态算子开源精度标准 experimental_standard](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)（任务书 §精度要求） |
| 性能标准 | `平均(标杆时延 / 测试时延) > 1.1`；标杆 = `torch.mm + F.layer_norm + torch.mm` 拼接，逐 case 采集 | 任务书 §性能要求；标杆数据 = `test_case/MatmulLayerNormMatmul_测试集.csv` 第 6 列（116 条，采集于 Ascend 950PR） |

**测量口径与硬件差异声明**：任务书标杆数据采集于 Ascend 950PR，本方案开发验证机为 910B3，跨硬件时延不可直接比对。因此性能结论采用双口径：

- **判据口径（有效）**：在**同一台设备**上自测 `torch.mm + F.layer_norm + torch.mm` 拼接时延作为标杆，与本算子实测时延求比值。判据是比值式，同机自测使分母与分子同硬件，结论有效；
- **对照口径（仅参考）**：任务书 950PR 标杆值并列展示，并显式标注「跨硬件不可比」。

两项口径的原始数据均落 `results/perf_gap.json`，报告同时呈现，不做混用。

## 测试概览

| 类别 | 规模 | 来源/生成方式 |
| --- | --- | --- |
| 官方测试集精度 | 116 条 | 任务测试集 CSV 全量参数化（`test_matmul_layernorm_matmul.py`） |
| 泛化精度 | ≥ 200 条 | ATK 框架，覆盖非对齐形状、极端 K0/N0 比例、单行/单列退化、非 64 整倍数 N0 |
| 冒烟 | 6 条正向 + 负向（dtype/形状/长度非法） | `run_smoke`，目标 < 2 分钟/轮 |
| 性能 | 116 条 | `msprof op`，warmup + 多次采样取均值 |

复现入口：交付件 `03-自测报告/1 步骤说明`（含环境变量、构建命令、pytest 与 msprof 命令、golden 生成脚本调用方式）。

## 兼容性分析

本算子为**新增样例 + 新增 kernel 组件**，不修改任何既有算子语义，兼容性影响面逐项列明：

| 改动点 | 类型 | 影响面 | 验证方式 |
| --- | --- | --- | --- |
| `include/catlass/gemm/kernel/matmul_layernorm_matmul.hpp` | 新增文件 | 无（无既有文件 include 它） | 全仓 `scripts/build.sh catlass_examples` 编译通过 |
| `include/catlass/epilogue/block/block_epilogue_layernorm.hpp` | 新增文件 | 无 | 同上 |
| `include/catlass/epilogue/block/block_epilogue.hpp` | 新增 1 行 `#include` 注册 | 所有 include 该聚合头的翻译单元增加一个头文件可见性；新头无全局对象/宏污染 | 全仓 examples 全量编译（含 26/12/20 等 MIX 样例）回归 |
| `examples/CMakeLists.txt` | 登记样例名 | 仅新增一个构建目标 | 全量构建 + 目标二进制实跑 |
| `tests/optest/**` | 新增/最小并入（`add_subdirectory`、`__init__` 导出、注册三行） | optest 框架新增一个 op，既有 op 不受影响 | `tests/optest` 既有 pytest 子集回归 |

无公共接口头（`include/catlass/catlass.hpp` 等）语义修改，无其他 arch/产品线的行为变化。
