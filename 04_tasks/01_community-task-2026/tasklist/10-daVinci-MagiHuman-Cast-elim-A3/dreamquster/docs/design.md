# 需求背景（required）

## 需求来源

HiAscend 10 月社区任务：**daVinci-MagiHuman 冗余 Cast 优化（Atlas A3）**（任务书：《10月社区任务-daVinci-MagiHuman 冗余 Cast 优化任务书(A3)》）。
目标代码仓：`cann/cann-recipes-infer`，代码改动范围限定 `models/davinci-magihuman/`。
任务讨论帖：https://gitcode.com/cann/cann-recipes-infer/discussions/18

## 背景介绍

### daVinci-MagiHuman 冗余 Cast 现状

daVinci-MagiHuman 是将文本、参考图像、视频和音频 token 放入统一序列进行联合去噪的单流 Transformer。仓内 DiT 实现中，线性层（`_BF16ComputeLinear`）、Attention（`flash_attn_func`）、归一化（`MultiModalityRMSNorm`）与激活边界存在大量 BF16 与原始 dtype 之间的显式转换（`tensor.to(...)`）。

按性质这些转换可分为三类：

1. **算子契约所需**：如 `npu_add_rms_norm` 要求 gamma 与输入同 dtype，去掉会直接报错；
2. **精度策略所需**：如输出头 `final_norm_*` / `final_linear_*` 在 fp32 计算，属模型既定精度策略；
3. **冗余转换**：输入 dtype 已等于目标 dtype（PyTorch 中 `.to(同 dtype)` 为 no-op，不产生 Cast 算子），或同一常量派生量在每次 forward 被重复计算并转换。

第三类是本任务的对象。基线 profiling 显示，单 step 窗口内 `Cast` 算子 402 次、累计 62,180.3 µs（约占算子总耗时 1.56%），其中 98.8% 耗时来自 MLP 激活边界的 bf16↔fp32 往返，其余热点为 RMSNorm 缩放因子 `(weight+1).to(bf16)` 的重复转换（218 次/step）。

任务书禁止仅通过文本搜索批量删除 `.to()`，要求 Atlas A3 profiling + 数据流证据逐项支撑。

### 任务目标（摘要）

1. 形成 Cast 清单：位置、源/目标 dtype、生产者、消费者、调用次数、累计耗时及是否为算子契约所需；
2. 只删除满足条件的 Cast：输入 dtype 已等于目标 dtype，或能在不改变后续算子契约的前提下合并相邻转换；不得改变权重或 scale 语义；
3. 端到端生成中位耗时下降且超过基线波动范围；profiling（op statistics）可观测 Cast 调用数与累计耗时均下降；
4. 输出与基线一致、无质量损失；记录峰值显存用于回归检查；
5. 默认启用优化，不改公共模块、不做量化、不关闭精度检查。

# 需求分析（required）

## 需求描述

在 Atlas A3 单卡、base 文生视频固定配置下，对 `models/davinci-magihuman` 的 DiT 前向 dtype 路径做证据驱动的冗余 Cast 消除，使：

- profiling `Cast` 算子调用次数与累计耗时相对冻结基线下降；
- 端到端生成中位耗时下降，且降幅超过基线 formal 波动（极差）；
- 同 seed / 同 prompt 下输出与基线逐位一致，盲测无稳定可辨退化；
- 峰值显存无未解释增长（缓存类优化须给出精确内存核算）。

## 需求拆解

1. **基线冻结与采集**：`ENABLE_PROFILER=1 && bash infer.sh base`，448×256 / 5s / 32 steps / CFG=2 / seed=42 / NoCache；3 组固定 prompt（sha256 冻结）。
2. **Cast 清单与契约判定**：对 `dit_module.py` 全部显式 dtype 转换逐点给出"保留/消除"判定（C1–C20），辅以 profiling 逐实例归因与算子 dtype 契约实验（E1–E4）。
3. **消除设计**：O1 消除 MLP 激活岛 fp32 往返；O2 缓存 RMSNorm 缩放因子（只读缓存 + 失效检测）。
4. **自验**：精度（逐帧/音频/mp4 MD5 + 双盲）、性能（warmup 1 + formal 3 × 3 prompt 对称协议）、内存（峰值对比 + 缓存核算）、异常与工程规范。
5. **交付**：私仓 `task_submission/` 七件套 + Cast 统计 CSV；代码 PR 走 `cann/cann-recipes-infer`（另提）；本 PR 仅提交设计文档。

## 功能目标

| 项 | 目标 | 实测达成 |
|---|---|---|
| Cast 调用数 | 相对基线下降 | 402 → 24（−94.0%） |
| Cast 累计耗时 | 相对基线下降 | 62,180.3 → 577.3 µs（−99.1%） |
| E2E 生成中位耗时 | 下降且降幅 ≥ 基线 formal 极差 | −2.992 s（−2.20%），为极差 1.224 s 的 2.4 倍 |
| 精度 | 3 组 prompt A/B + 2 名验收人盲测 | 全部输出逐位一致（bitwise） |
| 内存 | 峰值无未解释增长 | 47.89 GB → 47.89 GB（0），缓存核算 7.27 MiB |
| 工程 | 仅改模型目录、默认启用、无大文件 | 改动集中于 `dit_module.py` 单文件 |

# 详细设计（required）

## 算子分析

> 本任务为**模型 dtype 路径优化**，不新增独立 Ascend C / aclnn 算子。下表将"算子分析"映射为 Cast 位点与契约分析。

### 数学 / 语义约束

不改变数值公式，仅消除「源 dtype == 目标 dtype」的转换、或可证明逐位等价的往返转换（fp32 计算在算子内部完成）；保留算子强制契约 Cast 与模型既定精度策略（fp32 输出头）。

### 支持数据类型与场景

| 场景 | dtype 策略 |
|---|---|
| DiT 主计算 | BF16 |
| rope cos/sin | fp32 生成后统一转 bf16 一次（`dit_model.py:1016`），使逐层转换为 no-op |
| RMSNorm weight（非残差分支） | **保持 FP32**（`npu_rms_norm` 下 fp32/bf16 gamma 输出不同，max_diff=1.25e-1，须维持原语义） |
| RMSNorm gamma（残差分支） | 缓存只读 bf16（`npu_add_rms_norm` 硬契约，E3） |
| SwiGLU / GELU | 直接以 bf16 调用 `npu_clipped_swiglu` / `npu_fast_gelu`（E1/E2 证明逐位一致） |
| 输出头 `final_norm_*` / `final_linear_*` | 保持 FP32（模型精度策略，不动） |

### 接口定义

- 推理入口：`models/davinci-magihuman/infer.sh`（base 模式）；不新增对外算子 API。
- 代码改动集中于 `inference/model/dit/dit_module.py`（基线 commit `bbd278fe5ef772daa310df680865a6826ae6d909`，下文行号以此为准）。

### Cast 清单（C1–C20 契约判定）

对 `dit_module.py` 中全部显式 dtype 转换逐点判定（✅ 保留，❌ 消除，⚠ 非热点不动）：

| # | 位置（基线行号） | 转换 | 消费者与契约 | 判定 |
|---|---|---|---|---|
| C1–C3 | `swiglu7`/`gelu7` 内 fp32 往返（57–67） | bf16→fp32→bf16 | 激活算子 | ❌ 见 O1（E1/E2 证明 bf16 直算逐位一致） |
| C4 | `apply_rotary_emb_torch` cos/sin `.to(x.dtype)`（154–155） | fp32→bf16 | `npu_rotary_mul` 要求同 dtype | ✅ base 路径不执行（rope 以 tuple 传入，已统一转 bf16） |
| C5 | `forward_multi_experts` `(w_chunk+1).to(bf16)`（292） | fp32→bf16 | `npu_add_rms_norm` gamma 硬契约（E3） | ❌ 见 O2（每 forward 重算，可缓存） |
| C6 | `forward_multi_experts` residual/x `.to(gamma.dtype)`（294） | bf16→bf16 | — | ✅ no-op |
| C7 | `forward_single_expert` `(weight+1)`→`.to(bf16)`（310–312） | fp32→bf16 | `npu_add_rms_norm` | ❌ 见 O2 |
| C8 | `forward_single_expert` residual/x `.to(gamma.dtype)`（314） | bf16→bf16 | — | ✅ no-op |
| C9 | `_BF16ComputeLinear` input/weight/bias→bf16、out→output_dtype（331–345） | bf16→bf16 | `torch.matmul` bf16 | ✅ 全部 no-op |
| C10 | `flash_attn_func` q/k/v→bf16、out→original_dtype（426–438） | bf16→bf16 | `npu_fused_infer_attention_score` | ✅ no-op（`.contiguous()` 有效，非 Cast） |
| C11 | `flex_flash_attn_func`（505–555） | 同 C10 | 同上 | ✅ base 模式未命中 `use_local_attn` |
| C12 | `*_with_cp` q/k/v→bf16（569/607） | bf16→bf16 | — | ✅ `cp_size=1` 仅透传 |
| C13/C14 | `Attention/MLP.forward` `normed.to(bf16)`（691/778） | bf16→bf16 | — | ✅ no-op |
| C15/C16 | `MLP.forward` `.to(float32)`/`.to(bf16)`（779–780） | bf16→fp32→bf16 | 激活函数 | ❌ 见 O1 |
| C17 | `x.to(params_dtype)`（1015） | fp32→bf16 | 主干 | ✅ 保留（每次前向 1 次） |
| C18 | `rope.to(bf16)`（1016） | fp32→bf16 | `npu_rotary_mul` | ✅ 保留（使 C4/C12 成为 no-op） |
| C19 | `x[mask].to(weight.dtype)`（1034/1038） | bf16→fp32 | fp32 输出头精度策略 | ✅ 保留（1 次/前向） |
| C20 | `vae2_2_module.py:487-489` | — | VAE 解码路径 | ⚠ 1 次/生成，非 DiT 热点，本次不动 |

**结论**：能通过源码改动消除、且能在 profiling 中体现为 Cast 算子数下降的，是激活岛 fp32 往返（C1–C3/C15/C16）与 RMSNorm 缩放因子重复转换（C5/C7）两类；其余 `.to(同 dtype)` 为 PyTorch no-op，删除无收益。

### 基线 profiling 逐实例归因

单 step 窗口（`kernel_details.csv` 逐实例归因）：

| 归因 | shape | 次数/step | 累计耗时 | 定位 |
|---|---|---|---|---|
| C-a FP32→BF16 | (5120) | 218 | 0.682 ms | `MultiModalityRMSNorm` 的 `(weight+1).to(bfloat16)`——冗余，可缓存 |
| C-b BF16→FP32 | (3928\|4102, 27304\|20480) | 80 | 35.9 ms | MLP 激活岛入口 |
| C-c FP32→BF16 | (3928\|4102, 13652\|10240) | 80 | 22.9 ms | MLP 激活岛出口 |
| C-d…C-g | 首帧/音频/Adapter/索引 | 24 | < 0.1 ms | 非热点 |

C-b + C-c 占 Cast 累计耗时 98.8%。反证 no-op：`_BF16ComputeLinear`、`flash_attn_func`、`Attention/MLP.forward` 中的 `.to(bf16)` 若产生真实算子，应额外出现 ~80×N 条 Cast——实测无此峰值，确认 PyTorch 同 dtype `.to()` 为 no-op。

### 算子 dtype 契约实验（实机）

| 实验 | 结论 |
|---|---|
| E1 `npu_clipped_swiglu` | bf16 输入被接受，与"先转 fp32 再调用"逐位一致（53,625,056 元素 max_abs_diff = 0）；2.994 ms vs 3.695 ms（快 19%） |
| E2 `npu_fast_gelu`（clamp 7.0） | bf16 输入被接受，逐位一致（80,445,440 元素 max_abs_diff = 0）；2.377 ms vs 2.776 ms（快 14%） |
| E3 `npu_add_rms_norm` gamma | fp32 gamma 报错 `aclnnAddRmsNorm error 161002` → bf16 gamma 为硬契约 |
| E4 `npu_rms_norm` gamma | fp32/bf16 均被接受但输出不同（max_diff = 1.25e-1）→ 非残差分支必须保持 fp32 gamma 语义 |

## 算子实现

### 实现方案

本任务无独立 kernel / tiling；"Host 侧"对应 **Python 推理路径上的 dtype 决策与缓存生命周期**，"Kernel 侧"对应 DiT 模块前向中的算子调用契约。默认启用，**不设功能开关**。

#### Host / 运行时路径设计：O1 消除 MLP 激活岛 fp32 往返（对应 C-b/C-c）

现状（基线 `MLP.forward`）：

```python
normed = self.up_gate_proj(normed, ...).to(torch.float32)   # bf16→fp32，真实 Cast
normed = self.activation_func(normed).to(torch.bfloat16)    # fp32→bf16，真实 Cast
```

而 `activation_func`（`swiglu7`/`gelu7`）内部还要再做一次 fp32 往返。E1/E2 已证明 `npu_clipped_swiglu`/`npu_fast_gelu` 直接接受 bf16 输入，且结果与显式 fp32 路径逐位一致（fp32 计算发生在算子内部），因此：

- `swiglu7`/`gelu7` 去掉内部 fp32 搬运，直接以 bf16 调用算子（`gelu7` 的 clamp 保持非原地形式，不修改调用方输入），仅在输出 dtype 与期望不符时回转；
- `MLP.forward` 去掉 `.to(float32)` / `.to(bf16)` 往返，线性层输出直接进入激活算子。

收益：每 step −160 次 Cast（−58.8 ms），数值逐位等价。

#### Host / 运行时路径设计：O2 缓存 RMSNorm 缩放因子（对应 C-a）

现状：`forward_multi_experts` / `forward_single_expert` 每次 forward 都计算 `gamma = (weight + 1)`（fp32）并转 bf16（残差分支），而 `weight` 在推理期不变。单模态层每层 2 次、多模态层每层 2×3 次，随 40 层线性放大为 218 次/step。

方案：新增 `_get_scaled_weight()`，将 `(weight + 1)`（fp32）与 `.to(bfloat16)` 结果作为只读缓存，首次调用时惰性计算一次。失效检测覆盖所有可从张量元数据观测到的权重变更路径：

- **对象身份**（持有上次缓存时的 weight 引用）：捕获 Parameter 对象整体替换（`load_state_dict(assign=True)`、`module.weight = ...`）；
- **`weight._version`**：捕获一切就地写入（`copy_`、`nn.init.*`，含 `no_grad`/`inference_mode` 下及经 `detach()` 共享版本计数器的写入）；
- **`weight.device` / `weight.data_ptr()`**：捕获同对象 storage 交换（`module.to(...)`、`param.data = ...`、`swap_tensors`）与设备迁移。

weight 引用经 `object.__setattr__` 写入，避免 `nn.Module` 将其注册进 `_parameters`/state_dict；该引用同时钉住旧张量，防止分配器将旧 `data_ptr` 复用给新张量造成失效误判。`reset_parameters()` 主动清缓存。

分支语义按 E3/E4 区分：残差分支用缓存 bf16 gamma（`npu_add_rms_norm` 硬契约），非残差分支保持 fp32 gamma（维持原语义）。

**前提声明（有效边界）**：本缓存假定权重在首次编译/forward 之后冻结——这正是本模型推理路径的实际形态：`TransformerBlock` 带 `@magi_compile` 装饰，首次 trace 之后走编译图 replay，Python 函数体（含本失效逻辑）根本不执行，因此 replay 期间的权重变更不会被这里察觉，其正确性由编译器自身的图权重绑定语义负责。上述失效检测保护的是 eager 路径（权重加载 / `load_state_dict` / offload 换装等在首次编译之前发生的变更）。若未来出现"编译后再改权重"的场景，不可依赖本缓存提供保护。

收益：每 step −218 次 Cast（−0.68 ms），数值逐位一致。

#### Kernel / 模块级设计

无自定义 Ascend C kernel。映射为 DiT 模块前向三段：

1. **CopyIn（输入对齐）**：仅在 dtype 不匹配时 Cast；QKV / Linear 入参保持 bf16 直通；
2. **Compute（主计算）**：FA / Linear / RMSNorm / 激活按契约 dtype 执行；残差路径使用缓存 bf16 gamma；
3. **CopyOut（输出对齐）**：Linear / FA 输出仅在需要 restore 时 Cast；同 dtype 不转换。

### 缓存内存核算（按实机权重 shape 精确计算）

| 模块 | 个数 | dim | num_modality | 元素数 |
|---|---|---|---|---|
| `attention.pre_norm` | 40 | 5120 | 3 | 614,400 |
| `attention.q_norm` / `k_norm` | 40+40 | 128 | 3 | 30,720 |
| `mlp.pre_norm` | 40 | 5120 | 3 | 614,400 |
| `final_norm_video` / `final_norm_audio` | 2 | 5120 | 1 | 10,240 |

合计 162 个模块 / 1,269,760 元素；每元素缓存 fp32（4B）+ bf16（2B）两份 = 6 B → **7,618,560 B ≈ 7.27 MiB**（相对 ~48 GB 峰值显存约 0.015%）。缓存随模块存活、不随 forward 增长；异常中断时随模块析构一并释放。实测两档 torch `max_allocated` 均为 47.89 GB，无内存回归。

# 硬件与约束

| 项 | 要求 / 实际 |
|---|---|
| 硬件 | Atlas A3（Ascend 910_93，单 chip HBM 64GB） |
| 实机环境 | driver/npu-smi 26.1.1，CANN 9.1.0，PyTorch 2.10.0，torch_npu 2.10.0.post6，Python 3.12.13，Ubuntu 22.04 aarch64 |
| 配置固定 | base、448×256、5s（25fps）、32 steps、CFG=2、seed=42、`NoCache`、单卡 `world_size=1`/`cp_size=1` |
| 上游模型代码 | `GAIR-NLP/daVinci-MagiHuman` @ `209209b7086eba2020c5439265221495a8357322`（`inference/` 非覆盖合并） |
| 固定 prompt | 3 组（单人动作 / 多人交互 / 复杂背景），sha256 冻结 |
| 改动边界 | 仅 `models/davinci-magihuman/`；不改公共模块、不量化、不关精度检查、不批量文本删 `.to()` |
| 可比性 | 基线与优化版同容器、同 venv、同权重，差异仅在 `dit_module.py` |

# 可维可测

## 精度标准

- 算子级：E1/E2 全元素 max_abs_diff = 0（bf16 直算 vs 显式 fp32 路径）；
- 视频级：3 组固定 prompt × 基线/优化，同 seed 逐帧 MD5 + 音频 MD5 + mp4 文件 MD5 全部一致；
- 验收：2 名指定验收人 A/B 匿名盲评（主体/语义/构图/动作/时序/色彩/伪影），无稳定可辨退化。

## 性能标准

- 协议：每档预热 1 次 + 正式 3 次 × 3 组固定 prompt，对称采集；计时含完整 5s 生成并 NPU 同步；
- 通过：优化档中位 < 基线，且降幅 ≥ 基线 formal 极差；`op_statistic.csv` 中 Cast 调用数与累计耗时均下降；
- 产物：`op_statistic.csv`、逐次 wall/gen 计时、Cast 统计 CSV。

## 内存与异常

- 记录基线与优化版峰值显存（torch `max_allocated`）；缓存类改动须附精确内存核算（见上 7.27 MiB）；
- 失效检测场景矩阵（对象替换 / 就地写入 / storage 交换 / 设备迁移 / 地址复用探针）在本地 CPU（torch 2.13）30/30 与部署容器（torch 2.10 + torch_npu 2.10.0.post6）31/31 全通过。

## 测试与验收方案

| 阶段 | 内容 |
|---|---|
| 功能 | base 模式跑通；优化默认生效；非法输入/异常中断日志留存 |
| 精度 | 3 组 prompt A/B 逐位比对 + 双盲签字 |
| 性能 | profiling Cast 对比 + 中位/极差判定 + 独立 spot 复测 |
| 内存 | 峰值对比 + 连续生成序列 |
| 工程 | 私仓 `task_submission/` 七件套；代码 PR 走 `cann-recipes-infer` 的 pre-commit / CI / 实机验证 |

## 实测结果（已验收）

| 维度 | 指标 | 基线 | 优化版 | 变化 |
|---|---|---|---|---|
| 性能 | 去噪段中位耗时（32 steps） | 135.840 s | 132.848 s | **−2.992 s（−2.20%）**，基线极差 1.224 s，降幅超极差 2.4× |
| 性能 | 流水线 e2e 中位（Step1→8） | 138.319 s | 135.392 s | −2.927 s（−2.12%） |
| 性能 | p2/p3 fresh 对称复测 base_sec 中位 | 43.11 s | 41.06 s | −2.05 s（−4.75%），极差 0.20 s |
| Cast | 调用数/step（纯 Cast 口径） | 402 | 24 | **−94.0%** |
| Cast | 累计耗时/step | 62,180.3 µs | 577.3 µs | **−99.1%** |
| 内存 | 峰值显存（torch allocated） | 47.89 GB | 47.89 GB | 0 |
| 精度 | 3 组 prompt 输出 vs 基线 | — | 逐帧/音频/mp4 文件 MD5 全部一致 | **bitwise 级** |

确定性留痕：p1 累计 22 份 mp4（跨 4 个代码版本、跨多日多时段实跑）MD5 全部一致；p2/p3 fresh 对称复测 16 份 mp4（8 base + 8 opt）逐位一致；独立 spot 复测（另一时段、代码换装实跑）Cast 402→24 完全复现。

## 兼容性

- 上游固定 commit：`209209b7086eba2020c5439265221495a8357322`；无接口变更、无配置变更；
- 与 MagiCompile 编译路径兼容：首编后 replay 期间 Python 失效逻辑不执行，权重绑定由编译器负责（见前提声明）；
- 私仓自验材料：fork `master` 分支 `task_submission/`（七件套 + Cast 统计 CSV）。

## 风险与回滚

| 风险 | 影响 | 应对 |
|---|---|---|
| 缓存失效场景遗漏（权重重载/offload 后结果错误） | 精度 | 失效检测覆盖对象替换/就地写入/storage 交换/设备迁移/地址复用；`reset_parameters()` 清缓存；双环境场景矩阵验证 |
| `.to(同 dtype)` 在图编译路径下被 lower 为 Cast | 误判收益 | 一律以 `op_statistic.csv` 实测为准 |
| 环境或权重 revision 不一致导致不可比 | 性能 | commit/权重/prompt/seed 全部冻结，同容器同 venv 对比 |

回滚：优化改动集中于 `models/davinci-magihuman/inference/model/dit/dit_module.py` 单文件、无接口变更，还原该文件即恢复基线行为。
