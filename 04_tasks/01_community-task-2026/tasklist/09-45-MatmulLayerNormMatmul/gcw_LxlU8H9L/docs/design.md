# MatmulLayerNormMatmul 算子设计文档

> 版本：v1.0　日期：2026-10-26　参赛账号：gcw_LxlU8H9L
> 对应任务：9月社区任务-MatmulLayerNormMatmul 算子开发（09-45 / 09-67 同题）
> 代码仓库：gcw_LxlU8H9L/catlass（私有 fork，已邀 ascend-cann）

# 1 需求背景

## 1.1 需求来源

| 项目 | 内容 |
|---|---|
| 任务书 | 9月社区任务-MatmulLayerNormMatmul 算子开发任务书 |
| 测试集 | `test_case/MatmulLayerNormMatmul_测试集.csv`，116 条 |
| 目标仓库 | [cann/catlass](https://gitcode.com/cann/catlass) |
| 硬件 | Ascend 950（性能采集平台 950PR） |
| 语言 | Ascend C / CATLASS TLA |

## 1.2 背景介绍

### 1.2.1 算子定位与数学定义

`C1 = LayerNorm(A0·B0, γ, β, ε) · B1`

Transformer/MLP 投影层的常见融合计算链。独立执行时需依次启动 mm→LN→mm 三次 kernel，中间矩阵写回/读出 GM。融合算子在单次 MIX kernel launch 内完成全部三段。

### 1.2.2 小算子拼接基线开销

拼接方案（torch.mm + F.layer_norm + torch.mm）在 NPU 上产生：3 次 kernel launch、C0 两次 GM 写读（fp16）、LayerNorm 独立遍历。融合收益来源：消除 launch 开销、减少 GM 往返、LayerNorm 与 Cube 重叠。

## 1.3 目标仓库现状

| 组件 | 可用性 |
|---|---|
| Mmad/BlockMmad（cube 模板） | ✅ 复用 |
| CopyL0CToUBTla / CopyL0CToGmTla（fixpipe） | ✅ 复用 |
| SyncAll / CrossCoreFlag（跨核同步） | ✅ 复用 |
| LayerNorm/行归约 tile | ❌ 仓库无现成 → 自建 |
| Matmul1 A 操作数片上衔接 | ⚠️ 需适配 |

# 2 需求分析

## 2.1 需求描述

| 参数 | 类型 | 形状 | 布局 |
|---|---|---|---|
| A0 | 输入 fp16 | (M0, K0) | RowMajor |
| B0 | 输入 fp16 | (K0, N0) | ColumnMajor |
| B1 | 输入 fp16 | (N0, M1) | ColumnMajor |
| gamma/beta | 输入 fp32 | (N0,) | — |
| C1 | 输出 fp16 | (M0, M1) | RowMajor |

eps=1e-6；Mean/Variance 由内部 workspace 管理，不对外暴露。

## 2.2 需求拆解

1. 单次 kernel launch 完成三段计算（硬要求）
2. 中间结果不落对外 GM（W0/W1 由 launcher 内部申请）
3. 精度满足[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
4. optest 任务测试集通过 + ATK 泛化 ≥200 例
5. 性能达小算子拼接 1.1×

## 2.3 支持域

| 约束 | 值 |
|---|---|
| N0 | ≤8192（16 倍数为高效路径） |
| K0 | ≥16（K0=1 数值异常，不支持） |
| 值域 | 标准域（均匀 [-5,5] / 正态）；双侧大均值正态输入可使 C0 超 FP16 表示域（饱和→NaN，任何 fp16 实现同样触发，按 xfail 申报） |

# 3 详细设计

## 3.1 算子分析

### 3.1.1 数学公式

阶段一：C0 = A0 × B0（FP32 累加，FP16 落盘）
阶段二：μ[m] = mean(C0[m,:]); σ²[m] = var(C0[m,:]); C0_norm = (C0−μ)/√(σ²+ε)·γ+β
阶段三：C1 = C0_norm × B1（FP32 累加，FP16 输出）

### 3.1.2 数据类型与精度语义

- mm 段：FP16 输入 → FP32 累加 → FP16 输出
- LN 段：FP32 全程（gamma/beta FP32 直载，方差 max(var,0) 钳负）
- 无 FP16 中间下溢风险（FP16 staging 的固有舍入由精度标准 matched_ratio≥0.99 容忍）

## 3.2 算子实现

### 3.2.1 总体架构

单 `__mix__(1,2)` kernel launch（1 AIC : 2 AIV），行块流水 R=2：

```
AIC: mm0 blk(rb) → SyncAll(rb+1) → ... → mm1 blk(rb)
AIV:          LN blk(rb) ∥ AIC mm0 blk(rb+1) → SyncAll(rb+1) → ...
```

两侧 SyncAll 计数恒等 R+1，无跨核 flag（≤7/≤15 红线天然满足）。

### 3.2.2 片上生产消费链

```
GM A0/B0 → AIC L1 → L0A/L0B → MMAD(FP32 L0C)
    → Fixpipe(F322F16) → W0(GM workspace)
    → AIV UB → LN 向量三遍(统计→归一化) → W1(GM workspace)
    → AIC mm1: W1 → L1 → L0A → MMAD → Fixpipe → C1(GM 唯一对外输出)
```

关键 buffer 所有权：
- W0[rb]：AIC 写（fixpipe）→ SyncAll → AIV 读
- W1[rb]：AIV 写（DataCopy）→ SyncAll → AIC 读
- C1：AIC 写（fixpipe），唯一对外

### 3.2.3 Host 侧设计

**tilingKey 分派**：

| 条件 | mm0 tile | mm1 tile |
|---|---|---|
| m0≤128 | 64×256 | 64×256 |
| 大形状档 | 256×256 | 256×256 |
| 其他（512 档等） | 128×128 | 128×128 |

**workspace**：`2 × M0 × RoundUp(N0,16) × 2B`（W0/W1 行主序，pad 列不进统计分母），launcher 内部申请释放。

### 3.2.4 Kernel 侧设计

**mm0/mm1**：复用 CATLASS `BlockMmadTla`（MmadPingpong policy），FP32 累加 L0C，Fixpipe F322F16 直出 GM。

**LN**：自写 AIV 向量核（`KernelAdapter` 发射），chunk=2048 向量三遍：
1. pass0：逐 chunk DataCopy GM→UB，Cast fp32，ReduceSum 行累加
2. passA：重取 chunk，中心化平方累加（var = E[x²]−E[x]²，max(var,0) 钳负）
3. passB：归一化 + γ/β 仿射 → Cast fp16 → DataCopy 写 W1

gamma/beta FP32 常驻 UB。rsqrt 采用 Newton 迭代（aicore 无标量 sqrtf）。

**尾块**：actualM/actualN/curK 全程显式传递；padding 行列不进统计分母、不写 C1 有效区外。

### 3.2.5 与拼接标杆的差异

| 差异 | 原因 |
|---|---|
| LN 在 AIV 向量核（非独立 kernel） | 消除独立 launch + 与 Cube 重叠 |
| C0 不落对外 GM | 消除中间显存读写 |
| mm1 A 操作数从 W1(GM workspace) 读取 | 当前片上容量约束；后续可优化为 L1 直通 |

## 3.3 支持硬件

Ascend 950（dav_3510），950PR 采集。CANN 9.1.0，CATLASS 2.1.0。

## 3.4 算子约束

见 §2.3 支持域。

# 4 特性交叉分析

本算子为独立融合算子，无与其他算子的直接交互场景。在 GQA/MLP 推理链中，上游产出 A0/B0/B1（fp16），本算子消费并输出 C1 供下游使用。支持 stream 并发（workspace 由 launcher 按 stream 管理）。

# 5 可维可测分析

## 5.1 精度标准

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)（mixed_tolerance）：

| 参数 | FLOAT16 |
|---|---|
| 逐元素 | \|a−g\| ≤ atol+rtol·\|g\|，atol=rtol=2⁻¹⁰ |
| 用例通过 | matched_ratio≥0.99 且 max_abs_error≤max(0.1, 32·ULP) |

**golden 口径**：标准允许的"昇腾小算子拼接"（§2.3）：torch.mm→LN(fp32)→torch.mm 拼接链，含 FP16 staging。

## 5.2 测试结果

| 项 | 结果 |
|---|---|
| optest 任务测试集 | **116/116 PASS**（matched_ratio 全部=1.0） |
| pytest 附加用例 | 137 passed / 6 xfailed |
| 标准值域组 | uniform U[-5,5] 6/6 PASS |
| ATK 泛化 | **183 Pass / 230** |

## 5.3 性能结果

| 口径 | 定义 | 实测 |
|---|---|---|
| A | 逐例(标杆/实测)算术平均 | **0.6194** |
| B | 总时延之比 | **0.6741** |

测量协议：进程内热循环（预热丢弃），msprof kernel-only Task Duration，device 0。

**未达 1.1×**。差距主要来自融合上下文开销（workspace staging 读写、行块间栅栏），与 mm 段实现质量无关（standalone mm 已与标杆库平价）。

## 5.4 参考资料

1. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
2. [CATLASS 创新样例开发流程指南](https://gitcode.com/cann/catlass/blob/master/docs/zh/1_Practice/10_innovative_example_development_guide.md)
3. [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
