# 需求背景（required）

## 需求来源

HiAscend 10 月社区任务：**daVinci-MagiHuman 冗余 Cast 算子消除（Atlas A3）**。  
目标仓：`cann/cann-recipes-infer`，改动范围限定 `models/davinci-magihuman/`。  
讨论帖：https://gitcode.com/cann/cann-recipes-infer/discussions/18  

## 背景介绍

### daVinci-MagiHuman 冗余 Cast 现状

daVinci-MagiHuman 为单流 Transformer，将文本 / 视频 / 音频 token 联合去噪。仓内 BF16 推理路径中，线性层、Attention、RMSNorm、激活与残差边界存在大量 `tensor.to(...)` / Cast：

| 热点 | 现状问题 |
|---|---|
| `_BF16ComputeLinear` | 输入 / 权重 / bias / 输出无条件 `.to(compute_dtype|output_dtype)`，同 dtype 仍触发 NPU Cast |
| Attention Q/K/V | `flash_attn_*` 入口无条件转 bf16 再 restore |
| Attention 外层 | `normed.to(bf16)` 与 Linear `output_dtype` 交互易引入额外 Cast |
| MLP / SwiGLU | bf16↔fp32 往返以满足 `npu_clipped_swiglu` 历史契约 |
| `MultiModalityRMSNorm` | residual/x 无条件对齐 gamma；gamma 每步 fp32→bf16 |

部分 Cast 是算子契约必需；另一些在上下游已是目标 dtype 时成为冗余。任务书禁止仅文本搜索批量删 `.to()`，要求 A3 profiling + 数据流证据。

### 任务目标（摘要）

1. 形成 Cast 清单（位置、源/目标 dtype、生产者/消费者、次数与耗时、保留/删除依据）。  
2. 消除可证伪的冗余 Cast，保持算子契约与生成质量。  
3. 端到端生成中位耗时下降且超过基线波动；Cast 调用数与累计耗时均下降。  
4. 默认启用优化，无独立开关；不改公共模块、不做量化。

# 需求分析（required）

## 需求描述

在 Atlas A3 单卡、base 文生视频固定配置下，对 `models/davinci-magihuman` 的 DiT 前向 dtype 路径做证据驱动的冗余 Cast 消除，使：

- Cast 次数与累计耗时相对基线下降；  
- E2E 生成中位时延下降且超过基线 formal 波动（极差）；  
- 同 seed / 同 prompt 下质量无稳定可辨退化；  
- 峰值显存无未解释增长。

## 需求拆解

1. **基线采集**：`ENABLE_PROFILER=1` + `infer.sh base`，固定 448×256 / 5s / 32 steps / CFG=2 / seed=42 / NoCache。  
2. **Cast 清单**：优先 `_BF16ComputeLinear`、Attention、RMSNorm、MLP/激活、残差边界。  
3. **消除策略**：同 dtype 跳过（`_maybe_cast`）；只读 gamma bf16 缓存；优先 bf16 `npu_clipped_swiglu`；Linear 默认 `output_dtype=bf16` 避免外层删除后回灌 fp32。  
4. **自验**：精度盲测材料、性能 warmup1+formal≥3/5、内存连续 5 次、非法输入 / 中断日志。  
5. **交付**：私仓 `task_submission/` 七件套 + CAST_LIST + op_statistic；本 PR 仅提交设计文档。

## 功能目标

| 项 | 目标 |
|---|---|
| Cast count / time | 相对基线下降（实测方向：402→40，58304→6415 µs） |
| E2E gen median | 下降且 drop ≥ 基线 formal gen 极差 |
| 精度 | ≥3 组固定 prompt 两档 A/B；2 名指定验收人盲测 |
| 内存 | 记录加载后 / 峰值 / 连续 5 次；无未解释增长 |
| 工程 | 仅改模型目录；默认启用；无硬编码密钥 / 大文件 |

# 详细设计（required）

## 算子分析

> 本任务为**模型 dtype 路径优化**，不新增独立 Ascend C / aclnn 算子。下表将“算子分析”映射为 Cast 位点与契约分析。

### 数学 / 语义约束

不改变数值公式，仅去掉「源 dtype == 目标 dtype」或可合并的转换；保留算子强制契约 Cast。

### 支持数据类型与场景

| 场景 | dtype 策略 |
|---|---|
| DiT 主计算 | BF16 |
| RMSNorm weight | **保持 FP32**（`aclnnRmsNorm` 非残差路径拒 bf16 Parameter） |
| 残差路径 gamma | 缓存只读 `_gamma_bf16`，避免逐步 Cast |
| SwiGLU | 优先 bf16 `npu_clipped_swiglu`；失败回退 fp32 契约路径 |

### 接口定义

- 入口：`models/davinci-magihuman/infer.sh` / `sample_video.py`  
- 配置：`base.yaml` + `base_config.json` + `base_runtime.yaml`（NoCache）  
- 不新增对外算子 API；改动集中于 `inference/model/dit/dit_module.py`（及 load 后 gamma prewarm）

### Cast 清单（设计锁定）

| ID | 位置 | 原行为 | 决策 | 契约依据 |
|---|---|---|---|---|
| C01–C02 | `_BF16ComputeLinear` in/out | 无条件 `.to` | 同 dtype 跳过 | 权重/输入多为 bf16 |
| C03–C06 | flash / flex FA | QKV `.to(bf16)` + restore | 同 dtype 跳过 | FA 要 bf16；上游常已 bf16 |
| C07 | Attention 外层 | `normed.to(bf16)` | guarded + Linear `output_dtype=bf16` | 避免删外层后 matmul 回 fp32 再 Cast |
| C08 | MLP / SwiGLU | bf16↔fp32 | 优先 bf16 swiglu；同 dtype 跳过 | 去掉无意义往返 |
| C09 | RMSNorm residual | residual/x / gamma 逐步 Cast | guarded + `_gamma_bf16` 缓存 | weight 保持 fp32 |
| C10 | RoPE | `.to(bf16)` | 同 dtype 跳过 | adapter 已 bf16 则跳过 |

## 算子实现

### 实现方案

#### Host / 运行时路径设计（对应模板 Host/Tiling）

本任务无独立 tiling；“Host 侧”对应 **Python 推理路径上的 dtype 决策与缓存生命周期**：

1. **`_maybe_cast(x, dtype)`**  
   - 若 `x.dtype == dtype`：直接返回，避免 NPU Cast  
   - 否则：执行一次必要转换  

2. **分核 / 并行**  
   - 保持既有单卡 base 配置；不改 CP / 并行策略  

3. **workspace / 缓存**  
   - `_gamma_bf16`：模块级只读缓存，在 checkpoint load 后 prewarm  
   - 禁止每次 forward 新建长期缓存；异常中断后依赖进程退出 / `empty_cache` 释放  

4. **“tilingkey”类分支**  
   - SwiGLU：优先 bf16 路径；捕获契约失败则回退 fp32（可观测日志）  
   - Linear：`output_dtype` 默认 bf16，减少 Attention/MLP 边界回灌  

#### Kernel / 模块级设计（对应模板 Kernel）

无自定义 Ascend C kernel。映射为 DiT 模块前向三段：

1. **CopyIn（输入对齐）**  
   - 仅在 dtype 不匹配时 Cast；QKV / Linear 入参 guarded  

2. **Compute（主计算）**  
   - FA / Linear / RMSNorm / SwiGLU 按契约 dtype 执行  
   - 残差路径使用缓存 bf16 gamma，避免逐步 Cast  

3. **CopyOut（输出对齐）**  
   - Linear / FA 输出仅在需要 restore 时 Cast；同 dtype 跳过  

关键辅助：

```text
_maybe_cast(t, dt) -> t if t.dtype==dt else t.to(dt)
_gamma_bf16()     -> cached weight.to(bf16)  # weight 本体仍为 fp32 Parameter
```

默认启用上述路径，**不设功能开关**。

# 硬件与约束

| 项 | 要求 / 实际 |
|---|---|
| 硬件 | Atlas A3 单卡（Ascend910） |
| 任务书软件 | Python 3.12、CANN 9.0.0、PyTorch 2.10.0、torch_npu v26.0.0 |
| 实机记录 | 须在 RFC/验收中如实记录实际 CANN / torch_npu 版本；若与字面版本不一致，走讨论帖确认是否可验收 |
| 配置固定 | base、448×256、5s、32 steps、CFG=2、seed=42、`dit_cache.method=NoCache` |
| 改动边界 | 仅 `models/davinci-magihuman/`；禁止改公共模块 / 量化 / 关精度检查 |
| 禁止项 | 批量文本删除全部 `.to()`；强制全网同一 dtype；吞异常；提交权重 / 大数据集 |

# 可维可测

## 精度标准

- 算子级：被优化位点输入/输出及后续至少 1 个 DiT block 的 dtype 契约可复核  
- 视频级：≥3 组固定 prompt（单人动作 / 多人交互 / 复杂背景），基线 vs 优化同 seed  
- 验收：2 名指定验收人盲测无稳定可辨退化；辅以 A/B 视频 MD5 / 日志留存  

## 性能标准

- 协议：每档 warmup≥1 + formal≥3（建议 formal×5）；计时含完整 5s 生成并 NPU 同步  
- 通过：优化档 E2E 生成中位 < 基线，且下降幅度 ≥ 基线 formal 极差；Cast count 与 Cast total time 均下降  
- 产物：`op_statistic.csv`、逐次 wall/gen/step3、SUMMARY JSON  

## 内存与异常

- 记录加载后 / 生成峰值 / 连续 5 次后的 NPU 显存；峰值无未解释增长  
- 非法输入、异常中断、连续生成日志纳入 `task_submission/exception_*.log`  

## 测试与验收方案

| 阶段 | 内容 |
|---|---|
| 功能 | base 模式可跑通；优化默认生效 |
| 精度 | 3×2 视频 + 盲测签字模板 + 自动比对日志 |
| 性能 | profiling Cast 对比 + formal 中位/极差判定 |
| 内存 | mem5 峰值序列 |
| 工程 | `pre-commit` / CI（合入 `cann-recipes-infer` 时）；本设计文档 PR 仅含本文件 |

## 兼容性

- 上游固定 commit：`209209b7086eba2020c5439265221495a8357322`（非覆盖合并 `inference/`）  
- 与既有 MagiCompiler / FA / RMSNorm 契约兼容；RMSNorm weight 保持 fp32  
- 私仓自验材料路径：开发者私仓 `task_submission/`（七件套 + CAST_LIST + op_statistic）
