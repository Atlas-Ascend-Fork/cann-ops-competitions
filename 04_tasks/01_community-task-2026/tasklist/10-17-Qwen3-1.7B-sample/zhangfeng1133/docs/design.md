# 需求背景（required）

## 需求来源

社区任务 10-17「10月社区任务-Qwen3-1.7B 模型样例」，参照仓内 Qwen3-8B 样例，在 A2/A3 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) `models/qwen/` 新增 Qwen3-1.7B 样例。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，验收以任务书与官方 Qwen3-1.7B 权重/config 为准。

## 背景介绍

`models/qwen/` 已有 Qwen3-8B 样例：`configuration_qwen.py`、`modeling_qwen.py`（torch_npu 推理实现）、`config/`（yaml）、`README.md`。统一入口 `executor/scripts/infer.sh --model qwen --yaml <yaml>`，通过 HuggingFace `config.json` 自动识别模型变体（Qwen2/Qwen3 dense 共用一套建模代码）。Qwen3-1.7B 为同家族稠密小模型，与 8B 的差异在规模参数与 tie embedding，结构族一致（GQA、QK-Norm、SwiGLU、RoPE）。

因此本任务主体是**配置与验收的新增**而非建模重写：新增 1.7B 配置 yaml 与 README 章节，复用 modeling 实现；重点风险在 8B 实现里隐含的规模假设（硬编码维度、权重加载对层数/词表形状的假设、小模型特有的 tie embedding 路径）需逐一审计。审计结论见「算子分析」的差异审计清单。

# 需求分析（required）

## 需求描述

在 `models/qwen/` 下新增 Qwen3-1.7B 配置与 Offline 推理样例：官方权重审计式加载、A2/A3 BF16 推理、单/多轮对话与生成参数与官方对齐；精度对拍（官方 GPU/CPU golden）与性能基线满足任务书门槛；README 可复现，A2/A3 双平台实测交付。

## 需求拆解

1. 配置接入：新增 `config/qwen3_1_7b_1tp.yaml`（与 `qwen3_8b_1tp.yaml` 同构，`model_path` 指向 1.7B 权重），不改 8B/7B 既有 yaml 行为。
2. 建模审计：排查 modeling_qwen.py 的规模假设（维度、层数、cache 形状、norm epsilon、QK-Norm 路径），确认全部从 config 派生；补齐 tie embedding 路径与权重缺失审计。
3. 权重加载：官方 safetensors 审计式加载（name/shape/dtype 对照 index、tie 复用核对、缺失/孤儿参数 fail-fast），报告落盘。
4. 推理样例：复用统一入口 `infer.sh`（offline 模式），greedy/采样、单/多轮、chat template 官方渲染；A2/A3 双平台。
5. 验证：精度对拍 + 功能 + 性能三关；8B/Qwen2.5 路径零回归；README 复现。

## 口径锁定清单（开发前冻结，验收中途不改参）

| # | 锁定项 | 冻结口径 |
| --- | --- | --- |
| L1 | 模型 config 来源 | 官方 `Qwen/Qwen3-1.7B` 仓库 `config.json`（ModelScope 与 HF 镜像双渠道核对一致），见「模型结构」表 |
| L2 | 生成口径 | 与 8B Benchmark 表对齐：`input_truncated_len=4096`、`max_new_tokens=256`、`batch_size=1`、greedy 主口径（采样单列） |
| L3 | 精度门槛 | 逐 token argmax 一致率 ≥99%、末层 logits 余弦 ≥0.99 量级，任务书另有数值时以任务书为准 |
| L4 | 性能口径 | README Benchmark 同表格式回填（Prefill ms / Decode ms per step），A2/A3 各自同窗口连测，NPU idle |
| L5 | 平台与执行模式 | Atlas A2 / A3 单卡；执行模式 `ge_graph`（与 8B 基线一致），`eager` 作数值对拍辅助模式 |

# 算子分析

## 模型结构

Qwen3-1.7B 是 Qwen3 家族稠密 decoder-only 模型（`Qwen3ForCausalLM`），官方 config.json 实测（双渠道核对）：

| 参数 | 值 | 与 8B 对比 |
| --- | --- | --- |
| num_hidden_layers | 28 | 36 |
| hidden_size | 2048 | 4096 |
| num_attention_heads | 16 | 32 |
| num_key_value_heads | 8 | 8（同） |
| head_dim | 128 | 128（同） |
| intermediate_size | 6144 | 12288 |
| vocab_size | 151936 | 151936（同） |
| tie_word_embeddings | **true** | false |
| max_position_embeddings | 40960 | 40960（同） |
| rope_theta | 1e6 | 1e6（同） |
| rms_norm_eps | 1e-6 | 1e-6（同） |
| rope_scaling | 无 | 无 |
| QK-Norm | 有（per-head RMSNorm） | 有 |

单层 = GQA 注意力（QKV 2048→4096 合并投影 + O 投影，QK-Norm per-head）+ SwiGLU MLP（2048×6144×3）。参数量核算：embedding 151936×2048 ≈ 311M + 28 层 × ≈50M ≈ 1.41B，合计 **≈1.72B**（tie 单份计，与官方标称 1.7B 吻合）；逐张量构成以权重 index.json 审计为准。

```text
input_ids ─► embed_tokens(151936×2048, tied) ─► ×28 [ GQA attention(QK-Norm+RoPE) + SwiGLU MLP ] ─► RMSNorm ─► lm_head(=embed_tokens, tied) ─► logits fp32 ─► 采样
```

## 计算算子清单与 NPU 映射（全部复用 8B 既有实现）

| # | 计算类 | 仓内实现（8B 已验证） | NPU 算子路径 | 1.7B 适配点 |
| --- | --- | --- | --- | --- |
| 1 | QKV 合并投影 | `QKVParallelLinear`（merged，TP 感知） | NPU GEMM（ge_graph 下图融合） | 无（头数/维数由 config 驱动） |
| 2 | QK-Norm | `QwenRMSNorm(head_dim)` per-head，`use_qk_norm` config 驱动 | RMSNorm | 无（Qwen3Config 自动启用） |
| 3 | RoPE | `torch_npu.npu_apply_rotary_pos_emb`（TND 布局） | 融合旋转算子 | 无（theta/维度从 config） |
| 4 | KV Cache 读写 | paged `CacheEntry`（block_size 分块）+ `npu_scatter_nd_update_` | scatter 写入 paged 块 | 无（层数 28 由 config 驱动 cache 池） |
| 5 | 注意力核心 | `npu_fused_infer_attention_score_v2`（TND、sparse_mode=3、GQA、block_table） | 融合注意力算子 | GQA 16:8（2:1）在支持面内，见 Kernel 设计 |
| 6 | O 投影 / MLP | `RowParallelLinear` / merged `gate_up_proj`（SwiGLU）+ `down_proj` | NPU GEMM + 融合激活 | 无 |
| 7 | lm_head | `ColumnParallelLinear`（独立权重） | NPU GEMM | **tie：改为与 embed_tokens 权重共享** |
| 8 | 采样 | `logits.float()` 后 greedy/top-p | host 侧采样（与上游一致） | 无 |

## 形状与显存分析（单卡资源规划输入）

| 项 | 估算 | 说明 |
| --- | --- | --- |
| 权重（BF16） | 1.72B × 2 B ≈ **3.4 GB** | tie 单份计；全部驻留 NPU，无 offload |
| KV Cache | 2(K,V) × 28 层 × 8 头 × 128 head_dim × 2 B = **224 KiB/token** | 4096 输入 + 256 输出 ≈ **1.0 GB**；40960 满长上限 ≈ 8.8 GB（当前口径不触达） |
| 激活（BF16） | prefill 峰值 ≈ batch 1 × 4096 token × 2048 hidden 量级，百 MB 级 | paged 分块调度 |
| 峰值合计 | 权重 3.4 GB + KV/激活 < 2 GB，**< 6 GB** | A2/A3 单卡 64 GB HBM 满足，余量充足 |

## 1.7B 接入差异审计清单（实现前逐项核对官方 config）

| 审计项 | 仓内现状（8B 实现实测） | 风险 | 处置 |
| --- | --- | --- | --- |
| 层数/hidden/heads/kv_heads/head_dim/intermediate | configuration 全部 config 驱动，默认值仅为签名兜底 | 默认值（4096/36/32）误用于缺字段 config | 官方 config.json 全字段覆盖；加载前逐字段审计，缺失字段 fail-fast |
| **tie_word_embeddings** | `lm_head` 为独立 `ColumnParallelLinear`，无 tie 支持；`load_weights` 对 checkpoint 外参数名静默跳过 | **1.7B 官方权重无 `lm_head.weight` → lm_head 保持随机初始化，输出完全错误** | 建模唯一改动点：tie 路径（见 Kernel 设计与详细设计） |
| 权重缺失/孤儿审计 | `load_weights` 对 `params_dict` 外名字静默 `continue`，无加载覆盖率校验 | 缺权重静默通过 | 加载后断言 `params_dict - loaded_params` 为空（tie 参数豁免），报告落盘 |
| QK-Norm | `use_qk_norm` 由 config 驱动，per-head 形状随 head_dim | 无（head_dim 128 同 8B） | 单测覆盖 128 维形状 |
| RoPE（theta/max_pos） | theta/config 驱动，cos-sin cache 按需扩展 | 无（参数同 8B） | 长文本用例覆盖位置行为 |
| 词表/tokenizer | vocab 151936 同 8B | 无 | 直接加载官方 tokenizer，不自造 |

# Host/Tiling 设计

## Host 侧推理流水线

| 阶段 | 组件 | 职责 | 关键契约 |
| --- | --- | --- | --- |
| 1 入口与配置 | `executor/scripts/infer.sh` → `InferenceConfig` | 解析 `qwen3_1_7b_1tp.yaml`（model_path/exe_mode/data/scheduler） | yaml 与 8B 同构；不改 8B yaml |
| 2 变体识别 | `configuration_qwen.py` | 读官方 `config.json` 自动识别 `qwen3` 变体，实例化 1.7B 维度 | 逐字段审计（审计清单 L1），缺失 fail-fast |
| 3 权重加载 | `default_loader` + `load_weights` | safetensors index 驱动分片读取；q/k/v、gate/up 合并映射；**tie 共享** | 逐张量 name/shape/dtype 对照 index；覆盖率断言；审计报告落盘 |
| 4 预填充 | `QwenForCausalLM.forward`（prefill 元数据） | 一次性 prefill 建 paged KV，记录首 token 延迟 | prompt 按 `input_truncated_len=4096` 截断 |
| 5 解码循环 | scheduler（`max_new_tokens=256`、`batch_size=1`） | 逐 token 解码，paged cache 增量写入 | 每 token 实际 seq 长度传递（TND） |
| 6 后处理 | logits fp32 → 采样/detokenize | greedy 主口径；chat template 官方渲染；单/多轮 | 生成参数与 L2 锁定表一致 |
| 7 落盘 | 结果写入器 | 推理日志与结果存 `models/qwen/res/<日期>/<case_name>/` | 目录结构不变，1.7B case 独立命名 |

## 资源分块规划（Tiling 在 LLM 推理中的映射）

| 资源 | 规划 | 依据 |
| --- | --- | --- |
| batch | 评测批 = 1（与 8B 基线口径一致；多轮会话按轮次顺序） | L2 锁定 |
| KV paged 分块 | 沿用仓内 `scheduler_config.block_size` 分块与 slot 分配，cache 池按 28 层 × 8 头 × 128 维 × BF16 预分配 | 形状与显存分析；8B 同机制仅层数变化 |
| 输入截断 | `input_truncated_len=4096`，超长按首侧截断（与 8B 一致） | yaml `data_config` |
| 输出长度 | `max_new_tokens=256` | L2 锁定 |
| 精度切分 | 主干线性/注意力 BF16；logits 归一化与采样 fp32（仓内 `logits.float()` 既有行为） | 数值一致性设计 |
| TP 切分 | 1tp（attn/lmhead tp_size=1）；embed/lmhead 并行分组逻辑不动，tie 下 lmhead_tp>1 路径不启用 | 单卡任务范围 |
| stream | 单 stream 顺序执行（与 8B 一致）；seed 统一注入保证可复现 | 评测可复现要求 |

# Kernel 设计

## attention kernel 选型（复用 8B 已验证路径，不新增 kernel）

| 计算点 | 算子 | 1.7B 适配性 | 采用策略 |
| --- | --- | --- | --- |
| 注意力核心 | `npu_fused_infer_attention_score_v2`（TND、sparse_mode=3、paged block_table、`actual_seq_*` 变长） | GQA 16Q/8KV（2:1）、head_dim 128、BF16 均在算子常规支持面（8B 为 32Q/8KV 同 head_dim，比例更极端） | 直接复用；首跑探针确认无告警，异常切 `eager` 模式对拍定位 |
| RoPE | `torch_npu.npu_apply_rotary_pos_emb`（TND） | head_dim 128、theta 1e6 与 8B 一致 | 直接复用 |
| KV Cache 写入 | `npu_scatter_nd_update_`（paged 块） | 形状由 config 驱动 | 直接复用 |
| GEMM（QKV/O/MLP/lm_head） | `QKVParallelLinear` / `RowParallelLinear` / merged `gate_up_proj`；ge_graph 下图融合调度 | 维度缩小（2048/6144），tile 自动适配 | 直接复用 |
| Norm | `QwenRMSNorm`（hidden 2048 / per-head 128） | eps 1e-6 同 8B | 直接复用 |

## tie embedding 接入设计（建模唯一改动点）

官方 1.7B checkpoint 仅含 `model.embed_tokens.weight`，无 `lm_head.weight`；当前 `lm_head` 独立初始化会随机读取。设计：

1. `Qwen3Config.tie_word_embeddings=true` 时，`lm_head` 权重与 `embed_tokens` 共享：加载 embed_tokens 后将同一参数对象绑定到 `lm_head`（引用共享，不做拷贝），参数计数断言"总参数 < 独立 lm_head 版本 311M"。
2. 共享参数的布局约束：embed 读路径与 lm_head GEMM 读路径 dtype/布局一致（BF16），tie 下 `lmhead_tp_size` 固定 1（不与 TP 切分叠加）。
3. `load_weights` 补缺失审计：加载完成后 `params_dict - loaded_params` 必须为空集（tie 参数以共享豁免标注），否则 fail-fast 并落盘缺失清单——修复"静默随机权重"风险。
4. 8B/7B（tie=false）路径零改动：tie 分支以 config 判定，默认路径行为逐位不变。

## 数值一致性设计

- 主路径 BF16（与 8B 基线一致）；logits 采样前 fp32（仓内既有 `logits.float()`）。
- 对拍三级：① 权重加载审计（逐张量 name/shape/dtype 对照官方 index，tie 复用与覆盖率核对，NaN/Inf=0）；② 固定用例集 NPU vs 官方 transformers GPU/CPU golden（逐 token argmax 一致率 + 末层 logits 余弦，门槛 L3）；③ 端到端生成文本抽检（中英文 + ≥2k 长文本）。
- QK-Norm/RMSNorm 的 eps 与计算精度随仓内既有实现保持，不自写等价物，避免引入数值分叉源。
- `eager` 模式作为 ge_graph 异常时的数值对照与定位手段（同一代码路径关闭图融合）。

## 回退与降级矩阵

| 失效场景 | 降级路径 | 状态 |
| --- | --- | --- |
| 融合注意力 1.7B 形状探针异常 | 切 `eager` 模式复测，问题定位后报社区 | 显式标注，不静默 |
| BF16 数值不达 L3 门槛 | 定位分叉层（norm/attention 逐层对拍），必要时 logits/norm 层定点升级 fp32 | 报告标注 |
| 权重审计不通过 | fail-fast，不进入推理 | 硬失败 |
| OOM | 降 max_new_tokens → 降输入截断长度，并在报告标注；禁止静默重试到正确 | 显式标注 |

# 硬件与约束

| 项 | 口径 |
| --- | --- |
| 硬件 | Atlas A2 与 Atlas A3 单卡（README「支持的产品型号」既列两系）；双平台各自交付精度与性能 |
| 版本组合 | CANN 9.0.0+ / torch_npu v26.0.0（torch 2.8.0）/ transformers 5.0.0 / Python 3.11（README「软件版本」表） |
| 显存 | 峰值 < 6 GB（见形状分析），单卡 64 GB HBM 余量充足 |
| 数据与 License | 权重不入库（官方 Apache-2.0 下载指引写入 README）；模型路径在 yaml 占位 |
| 禁止项 | 不修改 8B/7B 既有 yaml 与行为、不静默吞权重加载缺失、不编造性能数据、生成参数不偏离 L2 锁定表 |

# 详细设计（required）

## 工程组织（统一入口 + 配置新增，建模最小改动）

```text
cann-recipes-infer/
├── executor/scripts/infer.sh              # 复用统一入口（--model qwen --yaml qwen3_1_7b_1tp.yaml）
├── models/qwen/
│   ├── config/qwen3_1_7b_1tp.yaml         # 新增：1.7B 单卡 yaml（与 qwen3_8b_1tp.yaml 同构）
│   ├── models/configuration_qwen.py       # 复用（审计确认 config 驱动，无改动；如需缺字段审计则最小增补）
│   ├── models/modeling_qwen.py            # 复用 + tie embedding 支持（tie 分支 + 加载覆盖率审计）
│   └── README.md                          # 增补 1.7B 章节（模型表/权重下载/部署命令/精度/性能基线）
```

不新增 sample 脚本与独立 infer.sh（v1 修正：统一入口已覆盖 1.7B，仅新增 yaml 即可启动）；结果目录沿用 `models/qwen/res/`。

## 精度对拍设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 加载审计 | 逐张量 name/shape/dtype 对照官方 index；tie 复用 + 覆盖率断言；NaN/Inf=0 | 全通过 |
| 精度对拍 | 官方 transformers GPU/CPU BF16 同输入：逐 token argmax 一致率、末层 logits 余弦；中英文 + 长文本(≥2k)用例 | L3（argmax 一致率 ≥99%、cos ≥0.99 量级，任务书为准） |
| 功能 | 多轮对话、chat template、greedy/采样、异常输入（超长截断/空输入） | 行为有定义、不崩溃 |
| 性能 | A2/A3 各自：Prefill 时延、Decode ms/step（README Benchmark 同口径） | 报告基线，任务书门槛为准 |
| 兼容 | 8B 与 Qwen2.5-7B 路径同 seed 输出与上游 master 逐位一致 | 零回归 |
| 可复现 | README 三条命令（环境 → 权重 → infer.sh） | 第三方可复现 |

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| tie 缺失导致静默随机 lm_head | 审计清单实锤 + tie 分支设计 + 加载覆盖率 fail-fast 断言 |
| GQA 比例 2:1 融合算子边界差异（8B 为 4:1） | 算子支持面核查 + 首跑探针 + eager 降级定位 |
| 8B 实现存在规模隐含假设，1.7B 下静默出错 | 审计清单逐项核对 + 形状断言前置；改动以"config 派生"为唯一原则 |
| A2 与 A3 算子行为差异 | 双平台各自对拍；平台相关分支收敛到显式 yaml 配置 |
| 小模型对采样参数敏感，评测结果与官方偏差被放大 | 生成参数按 L2 锁定；greedy 为主口径，采样口径单列 |

# 可维可测分析

## 精度标准/性能标准

- 精度：三级对拍全通过（加载审计 / logits 对比 L3 / 端到端文本抽检）；对拍脚本与用例随样例交付，第三方可复跑。
- 性能：A2/A3 各自回填 README Benchmark 表新增行（Prefill ms / Decode ms/step），注明测试窗口、版本组合、NPU idle 状态；只报告实测值，不做预测外推。

## 功能用例

| 维度 | 用例 | 预期 |
| --- | --- | --- |
| 生成模式 | greedy / top-p 采样 | 固定 seed 可复现 |
| 会话 | 单轮 / 多轮（历史拼接 + chat template） | 轮次上下文正确 |
| 输入 | 中文 / 英文 / ≥2k 长文本 / 超长截断 / 空输入 | 截断与报错行为有定义 |
| 平台 | A2 / A3 × ge_graph / eager | 四组合可跑通，ge_graph 为报告口径 |
| 回归 | 8B 1tp / 2tp、Qwen2.5-7B 1tp（同 seed） | 输出与上游 master 逐位一致 |

## 兼容性分析

- 新增 yaml 独立文件，8B/Qwen2.5 既有 yaml 不动；modeling 改动收敛在 `tie_word_embeddings=true` 分支，tie=false 路径零改动，以回归用例佐证。
- configuration 若增补缺字段审计，只加校验不改默认值语义。

## 交付与运行顺序

环境（CANN/torch_npu 版本表核对）→ 权重下载（README 指引）→ yaml 配置 → 加载审计 → 精度对拍 → 功能用例 → A2/A3 性能基线 → README/结果回填。每步产物（审计报告/对拍结果/性能数据）落 `models/qwen/res/` 对应目录。

# 支持硬件

Atlas A2 系列产品、Atlas A3 系列产品（单卡 BF16）。

# 约束限制

- 单卡推理，1tp；不涉及量化（W8A8 不在本任务范围）、多机与分布式、服务化部署。
- 上下文按官方 config 上限 40960，当前评测口径 4096 输入 + 256 输出。
- 权重与数据不入库；README 中路径占位符化。
