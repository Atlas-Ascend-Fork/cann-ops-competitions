# Gemma-4-E2B 纯文本推理接入开发计划与设计说明书（Atlas A3）

| 项目 | 内容 |
| --- | --- |
| **社区任务** | 10月社区任务-Gemma-4-E2B纯文本推理接入任务书（A3）（任务编号：`10-7`） |
| **贡献者 / 团队名** | `gcw_x0sve0ZQ`（Pinkish orange） |
| **设计状态** | RFC / 设计评审阶段；本文定义实现方案与验收契约，不预先宣称实机功能、精度或性能已通过 |
| **关联 RFC Issue** | [cann/cann-recipes-infer #434](https://gitcode.com/cann/cann-recipes-infer/issues/434) |
| **目标代码仓** | [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)（调研基线 commit：`bbd278fe5ef772daa310df680865a6826ae6d909`） |
| **设计提交仓** | [cann/cann-ops-competitions](https://gitcode.com/cann/cann-ops-competitions)（基线 commit：`83aeaa5887560b5d0036d01f2e2ae3b0d2b1a469`） |
| **官方对标模型** | [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)（固定 Revision：`d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`） |
| **官方对标源码** | [huggingface/transformers v5.5.0 `modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py) |
| **目标硬件与软件** | Atlas A3 单卡（1TP）；CANN 9.0.0（及目标 A3 实机配套版本）、PyTorch 2.8.0、torch_npu 2.8.0.post4（v26.0.0）、Python 3.12（实际安装版本另存锁定记录） |

---

# 需求背景（required）

## 需求来源

本设计文档来源于 **CANN 2026 年 10 月社区任务《Gemma-4-E2B 纯文本推理接入任务书（A3）》** 及其配套测试说明 `test_cases/README.md`：
- 任务书 `Gemma-4-E2B纯文本推理接入任务书.md` SHA256：`4e351c1c6bb012bbacd9c39b5db9112e51d8f6b8d41f8d2e960fcd71020f7731`
- 测试说明 `test_cases/README.md` SHA256：`cfb420fde881ec2111451e32aad18fd6ca838a24c31affc66fad5d69b2c31156`

## 背景介绍

### 1. 模型背景与任务边界

Gemma-4-E2B（[google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)）是 Google 开源的高效多模态底座模型，其语言解码器（`Gemma4TextModel`）采用 35 层混合注意力（4:1 Sliding/Full Attention）与高词表容量（`vocab_size=262144`）设计，并在标准 Transformer Decoder 基础上引入了三项关键轻量架构机制：
1. **逐层输入嵌入链路（Per-Layer Input Embeddings）**：通过独立的逐层词表嵌入 `embed_tokens_per_layer`（维度 `35 × 256 = 8960`）与主干输入嵌入线性投影 `per_layer_model_projection` 融合，在每个 Decoder Layer 尾部通过门控激活（`per_layer_input_gate` + `gelu_pytorch_tanh`）、逐元素相乘、投影与 RMSNorm 注入残差流。
2. **跨层 KV 缓存共享（Cross-Layer KV Sharing）**：配置 `num_kv_shared_layers = 20`，后 20 层（零基索引第 15～34 层）不再独立生成或维护 K/V 缓存，而是直接复用共享区之前最后一个同类型注意力层的 K/V（第 13 层为最后一层非共享 `sliding_attention`，第 14 层为最后一层非共享 `full_attention`），大幅降低推理期 KV Cache 显存占用与 K/V 投影开销。
3. **分层双宽 MLP（Double-Wide MLP）**：配置 `use_double_wide_mlp = True`，前 15 层（第 0～14 层）Dense MLP 中间维度为 `6144`，共享 KV 的后 20 层（第 15～34 层）Dense MLP 中间维度翻倍至 `12288`，以补偿共享 KV 层的表达能力。

本任务聚焦于在 `cann/cann-recipes-infer` 统一执行器框架（`executor/core/`）下，完成 **Gemma-4-E2B 纯文本、BF16、Offline、Atlas A3 单卡（1TP）** 推理接入，覆盖 `eager` 模式与 `npugraph_ex` Decode 图模式，并提供可复现的参考基线脚本、精度对拍、性能矩阵与内存生命周期验证。视觉/音频编码器、量化、MTP、Online PD 分离、多卡并行及同批次变长 Prefill 不在本期必做范围。

### 2. 仓内现状审计与差距分析

基于 `cann/cann-recipes-infer` 基线提交 `bbd278fe5ef772daa310df680865a6826ae6d909`，对仓内现有 `models/gemma_4/`（适配对象为 `google/gemma-4-26B-A4B` 稀疏 MoE 模型）及执行器框架进行静态审计，现状与差距如下：

| 审计模块 | 仓内现状（`models/gemma_4` & `executor/core`） | Gemma-4-E2B 接入差距与设计结论 |
| --- | --- | --- |
| **目录与模型注册** | `executor/core/support_models.py` 仅注册 `"gemma_4"`，指向 `models/gemma_4`（8 卡 EP MoE 配置） | 按 `test_cases/README.md` 规范，新建独立目录 `models/gemma_4_e2b/`，并在 `support_models.py` 独立注册 `"gemma_4_e2b"`，严禁混用 `models/gemma_4` 的 MoE 配置或基线 |
| **配置解析（`config.json`）** | `models/gemma_4/models/configuration_gemma4.py` 仅读取顶层扁平字段，且缺失 `hidden_size_per_layer_input`、`vocab_size_per_layer_input`、`num_kv_shared_layers`、`use_double_wide_mlp` 字段；默认 `layer_types` 按 5:1 推导 | 官方 `google/gemma-4-E2B/config.json` 顶层为 `Gemma4Config`（`model_type="gemma4"`），语言参数嵌套在 `text_config` 子字典中，且为 35 层 4:1 `layer_types`。需在 `Gemma4E2BConfig.from_pretrained` 中自动解包嵌套 `text_config` 并完整校验 E2B 字段 |
| **逐层输入链路** | `models/gemma_4` 的 `Gemma4TextModel` 与 `Gemma4DecoderLayer` 完全未实现 `embed_tokens_per_layer`、`per_layer_model_projection` 及层内 `per_layer_input_gate` | 必须完整补齐顶层逐层输入生成与每层门控投影残差链路，严格对齐官方 `v5.5.0` 的缩放常数、激活函数与计算顺序 |
| **跨层 KV 共享** | `models/gemma_4` 每层独立创建 `CacheEntry(needs_block=True)` 并各自执行 `k_proj/v_proj` 与 `_update_cache` | 必须实现 `num_kv_shared_layers=20` 的来源层推导（15～34 层复用到 13/14 层），打通 Prefill 全长 K/V 传递与 Decode 共享 Paged Cache 零拷贝读取，禁止消费层重复写缓存或分配冗余物理块 |
| **分层双宽 MLP** | `models/gemma_4` 的 `Gemma4MLP` 对所有层固定使用 `config.intermediate_size` | 需根据 `use_double_wide_mlp` 与 `is_kv_shared_layer` 动态设置每层 `intermediate_size`（0～14 层为 `6144`，15～34 层为 `12288`） |
| **权重加载审计** | `models/gemma_4` 的 `load_weights` 直接 `if name not in all_params: continue` 静默忽略未匹配权重 | 任务书 §2.2 & §3.2 严禁静默忽略未匹配权重。需建立精确审计：显式列出多模态跳过前缀，核对全部 600 个 `model.language_model.*` 权重与 buffer，缺失或未知权重立即报错 |

---

# 需求分析（required）

## 需求描述

基于 `cann/cann-recipes-infer` 统一执行器，在 Atlas A3 单卡上实现 `gemma_4_e2b` 纯文本 BF16 Offline 推理适配，支持 `eager` 与 `npugraph_ex` Decode 图模式，满足精度、性能、内存及代码规范验收要求。

## 需求拆解与基线锁定（RFC 必填项）

### 1. 代码、权重与环境基线锁定

| 基线项 | 锁定内容 |
| --- | --- |
| **代码仓基线** | `cann/cann-recipes-infer` commit：`bbd278fe5ef772daa310df680865a6826ae6d909` |
| **官方模型仓库与 Revision** | HuggingFace：[google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)，固定 Revision：`d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| **模型文件校验信息（SHA256）** | - `config.json`（4,914 B）：`e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8`<br>- `generation_config.json`（181 B）：`de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c`<br>- `tokenizer_config.json`（906 B）：`12754e3442e47c1a1c55d550b617fc47ae227e8156ed1b1de0d8428195768b5c`<br>- `processor_config.json`（1,689 B）：`32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c`<br>- `tokenizer.json`（32,170,070 B，LFS）：`12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`<br>- `model.safetensors`（10,246,621,918 B，LFS）：`76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57` |
| **对标参考环境** | 固定 `transformers==5.5.0`，按官方 `AutoProcessor` / `AutoModelForMultimodalLM` 纯文本路径建立参考；使用 `Gemma4ForCausalLM` 语言路径时先对照官方多模态类的纯文本调用验证等价性。参考与NPU环境都记录 Python、torch、torch_npu、Accelerate及完整 `pip freeze`；仓内 `transformers==5.0.0` 不能视为官方E2B参考已可运行的依据 |
| **生成与模板参数映射** | - 固定revision不含 `chat_template.jinja`，采用原始文本续写，两端 `add_special_tokens=True`、右侧截断，并核对输入token逐项相同；不添加角色或thinking标记，不修改checkpoint。<br>- 特殊 token：`bos_token_id=2`，`pad_token_id=0`，按固定 `generation_config.json` 使用 `eos_token_id=1`；不能自行加入106<br>- 确定性解码参数：`do_sample=False, temperature=0.0, top_k=1, top_p=1.0, max_new_tokens=128, seed=1234` |
| **软硬件目标环境** | Atlas A3 单卡（1 个逻辑 NPU 设备，`world_size=1`），实机验证时完整记录 `npu-smi info`（芯片/整机型号、可用 HBM）、`lscpu`、`uname -a`、驱动/固件版本、CANN 版本（参考起点 CANN 9.0.0）、PyTorch 2.8.0 与 `torch_npu` 2.8.0.post4（v26.0.0） |

### 2. 参数与接口规格（任务书 §2.5）

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | dtype 类型 | 数据排布格式 | 维度 (shape) | 值域范围 | 异常行为 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `model_config.model_name` | 属性 | 独立注册名 | string | 不涉及 | YAML | 标量 | `"gemma_4_e2b"` | 未注册名称明确报错 |
| `model_config.model_path` | 属性 | 官方配置、tokenizer、权重目录 | string | 语言权重 BF16，必要 buffer 按官方语义 | HF checkpoint | 单路径 | 固定 revision `d29ff6b4` | 缺失配置或必要权重明确失败 |
| `model_config.exe_mode` | 属性 | 执行模式 | string | 不涉及 | YAML | 标量 | `"eager"` / `"npugraph_ex"` | 不支持模式明确报错 |
| 文本样本 | 输入 | 按固定tokenizer处理的原始文本 | string/list | 不涉及 | 现有数据入口 | 首期 `batch=1`（可扩展等长 batch） | token 长度 $\le 2048$，ID $\in [0, 262144)$ | 空输入、越界 ID 或不支持的变长 batch 明确报错 |
| `data_config.input_truncated_len` | 属性 | 输入长度上限 | int | 不涉及 | YAML | 标量 | 正整数，默认 `2048` | 非法值（$\le 0$）报错，超长按右截断规则处理 |
| `scheduler_config.max_new_tokens` | 属性 | 最大输出长度 | int | 不涉及 | YAML | 标量 | 正整数，默认 `128` | 非法值报错，不越界生成 |
| `scheduler_config.batch_size` | 属性 | batch 上限 | int | 不涉及 | YAML | 标量 | 首期 `1`（支持等长 `batch=2` 扩展验证） | 变长 Prefill batch 抛出明确 `ValueError` |
| `parallel_config.*` | 属性 | 并行配置参数 | int | 不涉及 | YAML | 标量 | 单卡 1TP，各并行度均为 `1` | 非法组合启动失败 |
| E2B 结构配置 | 属性 | `layer_types`、`num_kv_shared_layers`、`hidden_size_per_layer_input`、`use_double_wide_mlp` | list/int/bool | 不涉及 | 官方 `text_config` | `len(layer_types)==35` | 与权重 shape 及来源层映射兼容 | 层数、来源层或权重 shape 不匹配时明确报错 |
| 生成 token/text | 输出 | 生成序列及解码文本 | list/string | token 为 int | 执行器输出格式 | 每请求独立序列 | 生成至 EOS 或 `max_new_tokens` | 失败返回非零退出码，绝不静默返回无效结果 |

---

# 详细设计（required）

## 模型结构与算子分析

### 1. 官方 `text_config` 结构规格核对

通过对 `google/gemma-4-E2B`（revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`）的 `config.json` 与 `model.safetensors` 头部元数据进行精确核对，E2B 语言模型核心参数如下：

| 配置项 | 官方数值 | 结构语义说明 |
| --- | --- | --- |
| `num_hidden_layers` | `35` | 共 35 个 Decoder Layer（零基索引 `0..34`） |
| `hidden_size` | `1536` | 主干隐藏状态维度 $H = 1536$ |
| `vocab_size` / `vocab_size_per_layer_input` | `262144` / `262144` | 主词表与逐层输入词表大小均为 262,144 |
| `hidden_size_per_layer_input` | `256` | 每层逐层输入嵌入维度 $H_{\text{pli}} = 256$，35 层展开总维度为 $35 \times 256 = 8960$ |
| `num_attention_heads` / `num_key_value_heads` | `8` / `1` | MQA/GQA 结构：8 个 Query 头，1 个 KV 头（`num_kv_groups = 8`） |
| `head_dim` / `global_head_dim` | `256` / `512` | `sliding_attention` 头维 `256`（Q 总维 `2048`，KV 总维 `256`）；`full_attention` 头维 `512`（Q 总维 `4096`，KV 总维 `512`） |
| `layer_types` | 4:1 循环（共 35 层） | 每 5 层为 4 个 `"sliding_attention"` + 1 个 `"full_attention"`：<br>- `full_attention`（7 层）：`[4, 9, 14, 19, 24, 29, 34]`<br>- `sliding_attention`（28 层）：其余 28 层 |
| `sliding_window` | `512` | `sliding_attention` 层有效滑窗长度为 512 tokens |
| `num_kv_shared_layers` | `20` | 后 20 层（`15..34`）为 KV 共享消费层，前 15 层（`0..14`）为非共享层 |
| `intermediate_size` & `use_double_wide_mlp` | `6144` & `True` | 前 15 层（`0..14`）MLP 中间维为 `6144`；后 20 层（`15..34`）双宽开启，MLP 中间维为 `12288` |
| `hidden_activation` | `"gelu_pytorch_tanh"` | Dense MLP 与逐层输入门控均使用 `gelu_pytorch_tanh` |
| `rope_parameters` | 双模式 RoPE | - `sliding_attention`：`rope_type="default", rope_theta=10000.0`（旋转全维 `256`）<br>- `full_attention`：`rope_type="proportional", rope_theta=1000000.0, partial_rotary_factor=0.25`（旋转维 `128`，频率分母按完整 `global_head_dim=512` 归一化） |
| `final_logit_softcapping` | `30.0` | LM Head 输出 logits 执行 $\text{tanh}(\text{logits} / 30.0) \times 30.0$ |
| `tie_word_embeddings` | `True` | `lm_head.weight` 与 `model.embed_tokens.weight`（形状 `[262144, 1536]`）共享同一权重 |

### 2. 数学公式与关键链路设计

#### 2.1 逐层输入链路（Per-Layer Input Embeddings）

设打包后的一维输入 token 序列为 $\mathbf{x} \in \mathbb{Z}^T$（$T$ 为总 token 数），计算步骤严格对齐官方 `transformers v5.5.0`：

1. **主干词嵌入与缩放**：
   $$\mathbf{h}_0 = \text{EmbedTokens}(\mathbf{x}) \times \text{BF16}\left(\sqrt{H}\right) \in \mathbb{R}^{T \times 1536}$$
   其中缩放因子 $\sqrt{1536} \approx 39.1918$ 按官方 `Gemma4TextScaledWordEmbedding` 语义先转为 `weight.dtype`（BF16，值为 `39.25`）后与嵌入相乘。

2. **逐层词嵌入查找与缩放**：
   $$\mathbf{E}_{\text{pli}} = \text{ Reshape}\left(\text{EmbedTokensPerLayer}(\mathbf{x}) \times \text{BF16}\left(\sqrt{H_{\text{pli}}}\right),\ [T, L, H_{\text{pli}}]\right) \in \mathbb{R}^{T \times 35 \times 256}$$
   其中 $\sqrt{256} = 16.0$，`embed_tokens_per_layer.weight` 形状为 `[262144, 8960]`。

3. **主干隐藏态向逐层空间投影、归一化与融合**：
   $$\mathbf{P}_{\text{raw}} = \text{Reshape}\left(\mathbf{h}_0 \mathbf{W}_{\text{model\_proj}}^\top \times H^{-0.5},\ [T, L, H_{\text{pli}}]\right) \in \mathbb{R}^{T \times 35 \times 256}$$
   $$\mathbf{P}_{\text{norm}} = \text{RMSNorm}_{\text{pli}}(\mathbf{P}_{\text{raw}}) \in \mathbb{R}^{T \times 35 \times 256}$$
   $$\mathbf{U} = \left(\mathbf{P}_{\text{norm}} + \mathbf{E}_{\text{pli}}\right) \times 2^{-0.5} \in \mathbb{R}^{T \times 35 \times 256}$$
   其中 $\mathbf{W}_{\text{model\_proj}} \in \mathbb{R}^{8960 \times 1536}$，$H^{-0.5} = 1536^{-0.5}$，$2^{-0.5} \approx 0.70710678$，`per_layer_projection_norm` 为维度 256 的 `Gemma4RMSNorm`（按官方顺序在FP32计算均方、eps、逆平方根、权重缩放，最后转回BF16；融合算子需独立精度验证后才能替换）。

4. **第 $i$ 层（$i \in \{0, \dots, 34\}$）末尾门控融合与 `layer_scalar` 缩放**：
   在第 $i$ 层完成 Self-Attention 残差与 Dense MLP 残差得到 $\mathbf{h}_i^{\text{mlp}} \in \mathbb{R}^{T \times 1536}$ 后，取第 $i$ 层逐层输入切片 $\mathbf{u}_i = \mathbf{U}[:, i, :] \in \mathbb{R}^{T \times 256}$：
   $$\mathbf{g}_i = \text{GELU}_{\text{tanh}}\left(\mathbf{h}_i^{\text{mlp}} \mathbf{W}_{\text{gate}, i}^\top\right) \odot \mathbf{u}_i \in \mathbb{R}^{T \times 256}$$
   $$\mathbf{p}_i = \text{RMSNorm}_{\text{post\_pli}, i}\left(\mathbf{g}_i \mathbf{W}_{\text{proj}, i}^\top\right) \in \mathbb{R}^{T \times 1536}$$
   $$\mathbf{h}_i^{\text{out}} = \left(\mathbf{h}_i^{\text{mlp}} + \mathbf{p}_i\right) \times s_i \in \mathbb{R}^{T \times 1536}$$
   其中 $\mathbf{W}_{\text{gate}, i} \in \mathbb{R}^{256 \times 1536}$，$\mathbf{W}_{\text{proj}, i} \in \mathbb{R}^{1536 \times 256}$，$s_i \in \mathbb{R}^1$ 为该层 buffer `layer_scalar`。

#### 2.2 跨层 KV 共享映射与生命周期设计

1. **来源层数学推导**：
   设总层数 $L = \text{num\_hidden\_layers} = 35$，共享层数 $L_{\text{shared}} = \text{num\_kv\_shared\_layers} = 20$，则首个共享消费层索引为：
   $$i_{\text{first\_shared}} = L - L_{\text{shared}} = 15$$
   对于任意层 $i \in \{0, \dots, 34\}$：
   - 当 $0 \le i < 15$ 时，该层为**独立/来源层**（`is_kv_shared_layer = False`），独立拥有物理 KV Cache。
   - 当 $15 \le i < 35$ 时，该层为**共享消费层**（`is_kv_shared_layer = True`），其共享来源层索引 $\text{src}(i)$ 定义为前 15 层（`0..14`）中最后一个具有相同 `layer_types[i]` 的层：
     $$\text{src}(i) = \max\left\{j \in \{0, \dots, 14\} \mid \text{layer\_types}[j] = \text{layer\_types}[i]\right\}$$
   在 E2B 官方配置下：
   - 前 15 层中最后一个 `sliding_attention` 层为 **$j = 13$**，故消费区全部 16 个 `sliding_attention` 层（`15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33`）满足 $\text{src}(i) = 13$。
   - 前 15 层中最后一个 `full_attention` 层为 **$j = 14$**，故消费区全部 4 个 `full_attention` 层（`19, 24, 29, 34`）满足 $\text{src}(i) = 14$。

2. **缓存分配与扇出绑定契约（零公共框架修改）**：
   ```
   KVCacheManager / allocate_cache_tensors()
   ├── Layer 0..12 (non-shared)  ──► 各自分配独立 k_cache / v_cache，仅绑定至本层
   ├── Layer 13 (sliding source) ──► 分配 1 份 sliding k_cache / v_cache
   │                                 └─► tensor_setter 同时绑定至 Layer 13 及 Layer {15..18, 20..23, 25..28, 30..33}
   └── Layer 14 (full source)    ──► 分配 1 份 full k_cache / v_cache
                                     └─► tensor_setter 同时绑定至 Layer 14 及 Layer {19, 24, 29, 34}
   ```
   - `Gemma4ForCausalLM.get_cache_info()` 仅遍历前 15 层（`0..14`）收集 `LayerCacheInfo` 并返回 `ModelCacheInfo(num_layers=15, layer_infos=layer_infos)`。
   - 经核查 `executor/core/kv_cache/cache_utils.py` 的 `validate_cache_info()`，框架仅校验 `cache_info.num_layers == len(cache_info.layer_infos)`，不要求等于 `config.num_hidden_layers`。因此消费层 `15..34` 无需声明 `CacheEntry`，既避免分配 20 层冗余 HBM 缓存，也无需改动 `executor/core/kv_cache/` 任何一行公共代码。

3. **Prefill 与 Decode 阶段读写衔接**：
   - **唯一写入者原则**：仅 `is_kv_shared_layer == False` 的第 `0..14` 层执行 `k_proj`、`v_proj`、`k_norm`、`v_norm`、K-RoPE，并调用 `_update_cache()`（`torch_npu.npu_scatter_nd_update_`）将本步新增 K/V 写入各自的 Paged Cache。
   - **消费层零写原则**：`is_kv_shared_layer == True` 的第 `15..34` 层在前向中仅计算 `query_states = RoPE(q_norm(q_proj(hidden_states)))`，**不执行 `k_proj/v_proj/k_norm/v_norm/K-RoPE`，更不调用 `_update_cache()`**。
   - **Prefill 全长 K/V 传递**：在 Prefill 阶段，当输入长度 $L > 512$ 时，`SlidingWindowManager` 仅为滑窗尾部保留物理块，头部块在 `block_table` 中为 `_null_block`。而在 Prefill 的因果滑窗注意力中，序列前中部 token 仍需访问各自窗口 `[max(0, t-511), t]` 内的 K/V。因此，来源层 `13` 与 `14`（`store_full_length_kv = True`）在 Prefill 前向中将已完成 `k_norm + RoPE` 的 `key_states` 与已完成 `v_norm` 的 `value_states` 存入当次 `Gemma4TextModel.forward` 的局部字典 `shared_kv_states[layer_idx] = (key_states, value_states)`；消费层 `15..34` 在 Prefill 阶段直接从 `shared_kv_states[self.kv_shared_layer_index]` 取出全序列 `(K, V)` 送入下述BF16/FP32显式注意力路径。`forward` 结束时该局部字典立即销毁，无跨请求残留。
   - **Decode 阶段 Paged Cache 直接读取**：在 Decode 阶段（每请求 1 token，在层间顺序执行），当执行流经过第 `13`、`14` 层时，新 token 的 K/V 已通过 `_update_cache()` 写入共享 `k_cache / v_cache` 的当前 `slot_mapping` 槽位；当执行流到达第 `15..34` 层时，消费层直接使用已绑定的 `self.k_cache / self.v_cache` 与同一份 `block_table["SlidingWindow" | "FullAttention"]` 按block table gather有效K/V，再执行BF16 QK、FP32 Softmax、BF16概率与PV；sliding有效范围包含当前token，历史最多511个token。保持缓存分页管理和官方舍入顺序，融合FA仅作为需单独验证的候选。
   - **图模式（`npugraph_ex`）兼容性**：保持分页KV与输出地址稳定；实际有效长度和滑窗分组在图外准备，通过现有图任务更新接口传递。必须实测捕获和重放次数、固定shape预热后的新增编译/捕获及跨请求输出；静态引用本身不能证明兼容。后端私有接口如需兼容处理，须记录版本、受影响范围与回归证据，依赖升级重新验证。

#### 2.3 分层双宽 Dense MLP（`use_double_wide_mlp`）

在 `Gemma4MLP.__init__(self, config, layer_idx)` 中：
```python
first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
is_kv_shared_layer = (layer_idx >= first_kv_shared_layer_idx > 0)
use_double_wide = getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)
```
- 第 `0..14` 层：`self.intermediate_size = 6144`
- 第 `15..34` 层：`self.intermediate_size = 12288`
- 前向计算：$\text{MLP}(\mathbf{x}) = \text{down\_proj}\left(\text{GELU}_{\text{tanh}}(\text{gate\_proj}(\mathbf{x})) \odot \text{up\_proj}(\mathbf{x})\right)$。

---

## 模型实现方案

### 1. 代码目录结构规划

严格遵循 `test_cases/README.md` 与任务书第 4 章交付要求，目录规划如下：

```text
executor/core/support_models.py                  # 新增注册 "gemma_4_e2b"
models/gemma_4_e2b/
├── README.md                                    # 环境准备、权重校验、运行指南与支持边界说明
├── requirements.txt                             # Python 依赖声明
├── config/
│   └── gemma_4_e2b_bf16_1tp.yaml                # 离线推理标准 YAML 配置（路径使用占位符）
├── models/
│   ├── __init__.py
│   ├── configuration_gemma4_e2b.py              # Gemma4E2BConfig（支持解包嵌套 text_config 与完整结构校验）
│   └── modeling_gemma4_e2b.py                   # Gemma4ForCausalLM 及 E2B 三大特性实现与权重审计
└── tests/
    ├── data/
    │   └── prompts.json                         # 10+ 固定评测样本（覆盖中/英/问答/摘要改写/推理）
    ├── reference_hf.py                          # 官方 HF 参考实现基线生成与 logits 导出脚本
    └── test_gemma4_e2b_verification.py          # 模块精度对齐、滑窗边界、KV 共享生命周期与异常输入自验脚本
docs/models/gemma_4_e2b/
├── gemma4_e2b_adaptation.md                     # 架构适配细节与跨层共享 KV 缓存契约文档
└── gemma4_e2b_validation.md                     # 精度、性能、内存、图模式及 CI 验收报告摘要
```

### 2. 权重加载与完整性审计设计（`load_weights`）

对官方 `model.safetensors`（共 2,011 个 Tensor）建立确定性分类审计机制：

1. **显式跳过多模态非语言前缀白名单（共 1,411 个 Tensor）**：
   - `model.audio_tower.*`：751 个 Tensor
   - `model.vision_tower.*`：658 个 Tensor
   - `model.embed_audio.*`：1 个 Tensor（`model.embed_audio.embedding_projection.weight`，shape `[1536, 1536]`）
   - `model.embed_vision.*`：1 个 Tensor（`model.embed_vision.embedding_projection.weight`，shape `[1536, 768]`）
   - `model.multimodal_projector.*`：0 个（保留兼容白名单）
2. **语言模型权重与 Buffer 精确映射（共 600 个 Tensor）**：
   - 将前缀 `model.language_model.` 映射为 `model.`。
   - **词表绑定（Tied Embeddings）**：`model.embed_tokens.weight`（`[262144, 1536]`）与 `lm_head.weight` 绑定为同一Parameter/底层存储；数值相等但复制为两份不算共享。
   - **逐层输入顶层权重（3 个）**：
     - `model.embed_tokens_per_layer.weight`（`[262144, 8960]`）
     - `model.per_layer_model_projection.weight`（`[8960, 1536]`）
     - `model.per_layer_projection_norm.weight`（`[256]`）
   - **每层权重与 Buffer（35 层 × 17 个 = 595 个）**：
     - Buffer：`model.layers.{i}.layer_scalar`（`[1]`，BF16）
     - 逐层输入层内权重：`per_layer_input_gate.weight`（`[256, 1536]`）、`per_layer_projection.weight`（`[1536, 256]`）、`post_per_layer_input_norm.weight`（`[1536]`）
     - 4 个主干 RMSNorm：`input_layernorm.weight`、`post_attention_layernorm.weight`、`pre_feedforward_layernorm.weight`、`post_feedforward_layernorm.weight`（均为 `[1536]`）
     - 3 个 MLP 权重：`mlp.{gate_proj,up_proj,down_proj}.weight`（0～14 层中间维 `6144`，15～34 层中间维 `12288`）
     - 6 个 Attention 权重：`self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight`。注：官方 checkpoint 在15～34共享层仍包含 `k_proj/v_proj/k_norm`，前向不参与计算；按完整语言权重加载方案保留并逐项审计，禁止静默忽略。若改为省略这些参数，应先明确必要性并取得方案确认。
3. **严格强校验**：
   - 若遇到既不在多模态跳过白名单、也不属于模型参数/审计列表的未知 Tensor，立即抛出 `ValueError`。
   - 若加载完成后，模型必需的任一 Parameter 或持久化 Buffer（含 `layer_scalar`）未被加载，立即抛出 `RuntimeError` 并列出缺失键名。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A3 系列产品（单卡 1TP） | √ |

## 约束与限制说明

1. 首期仅支持 **纯文本、BF16、Offline、单卡（1TP）** 推理；视觉与音频输入不在本任务范围。
2. 同批次多请求（如扩展验证 `batch_size=2`）要求 Prefill 阶段各请求输入 token 等长；若传入不等长变长 Prefill batch，模型在 `forward` 入口显式检测 `actual_seq_lengths_q` 并抛出可读 `ValueError`，拒绝静默错误计算。
3. 输入序列长度受 `data_config.input_truncated_len`（默认 2048）约束，总上下文长度不超过 `input_truncated_len + max_new_tokens`。

---

# 可维可测分析

## 精度标准/性能标准/内存标准

| 验收维度 | 具体验收标准与判定条件 | 标准来源 |
| --- | --- | --- |
| **1. 模块级与单步数值精度** | 对齐官方 `transformers v5.5.0` BF16 参考实现，逐项核对：(a) 逐层输入顶层与层内输出；(b) 第 13/14 来源层及第 15～34 消费层 Attention 输出；(c) 前 15 层与后 20 层双宽 MLP 输出；(d) `layer_scalar` 与 `final_logit_softcapping=30.0` 后的最终 logits。要求：无 NaN/Inf，余弦相似度 $\ge 0.999$，相对误差 $\text{NRMSE} \le 0.015$，单步贪婪 top-1 token 一致。 | 任务书 §3.2 / RFC 约定 |
| **2. 固定样本端到端生成对比** | 提供 `models/gemma_4_e2b/tests/data/prompts.json`（至少 10 条固定样本，覆盖中文、英文、知识问答、摘要/改写、简单推理各 $\ge 2$ 条），固定 `max_new_tokens=128, seed=1234` 及贪婪解码。要求：`eager` 与 `npugraph_ex` 输出 token 序列 **100% 完全一致**，且与官方 HF 参考实现生成文本语义及 token 序列对齐。 | 任务书 §3.2 & `test_cases/README.md` |
| **3. 公开数据集全量评测** | 固定采用 `cais/mmlu`，revision `c30699e8356da336a370243923dbaf21066bb9fe`，config=all、split=test，全量14042题，parquet SHA256=`74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`；按相同选项prompt与下一token选项评分规则完成accuracy评测，NPU 评测得分与官方参考实现得分绝对偏差 $\le 1.0\%$。 | 任务书 §3.2 & `test_cases/README.md` |
| **4. 滑窗边界与共享 KV 自验** | 覆盖输入长度 `511 / 512 / 513` token 的滑窗跨界测试，验证 Prefill 全序列计算与 Decode 跨块边界读取无精度退化、无 `_null_block` 越界访问。 | 任务书 §3.5 |
| **5. 图模式（`npugraph_ex`）验证** | 验证 `exe_mode: "npugraph_ex"` 下 Decode 图成功编译与预热，固定 shape 稳态推理阶段**零重编译**，输出与 `eager` 完全一致。 | 任务书 §3.5 & 新模型 Checklist |
| **6. 性能测试矩阵** | 覆盖 `batch_size=1`、输入长度 `128 / 512 / 1024 / 2048`、最大输出 `128`，分别采集 `eager` 与 `npugraph_ex` 模式；每组至少预热 1 次、正式运行 3 次，开启 `enable_profiler: true`，分开统计模型加载、预热/编译与稳态 Prefill/Decode 时延、吞吐、均值与中位数（本任务为新模型接入，首次实测结果作为后续基线）。 | 任务书 §3.3 & `test_cases/README.md` |
| **7. 内存预算与连续请求稳定性** | 在 `2048` 输入、`128` 输出、`batch_size=1` 下无 OOM（BF16语言权重约8.66 GiB，另计KV、激活、工作区、图缓存与allocator reserved；硬件容量和峰值以实测为准）；开启 `profile_memory` 记录权重加载后、Prefill 峰值、Decode 峰值及连续 20 个请求结束后的 Host/NPU 内存，确保无内存泄漏、无重复释放或跨请求状态污染。 | 任务书 §3.4 & §3.5 |

## 兼容性与回归分析

1. **共享改动与回归范围**：
   - 新模型、配置和测试放在 `models/gemma_4_e2b/`，文档放在 `docs/models/gemma_4_e2b/`；新增独立注册。分页KV优先复用现有接口，不建立模型私有调度框架。
   - 如生成上限、EOS、图更新需要共享scheduler或后端兼容改动，先固定改动前的既有Gemma-4-26B-A4B受影响路径功能与性能基线，再验证所有直接调用方。不得以“新增模型”推断零影响。
   - 26B回归在同硬件、权重、拓扑、batch及长度下各预热1次、测量至少3次，记录输出、KV回收、Prefill/Decode与吞吐。性能波动下界须在采集前与维护者确认；没有冻结容差时标记未判定，不能声称性能回归通过。
2. **工程门禁与复现脚本**：
   - 提供一键复现与自验脚本，覆盖 `pre-commit run --all-files`、静态检查、模块单测、端到端生成对比、性能与内存采集，任一步骤失败均返回非零退出码。
   - 严格遵守仓库规范：不提交模型权重、数据集、二进制或大型 profiling 原始产物，真实环境路径全部使用占位符脱敏。


# 开发里程碑与提交顺序

| 阶段 | 任务及产物 | 进入下一阶段的条件 |
| --- | --- | --- |
| 1. 计划评审 | 本文、RFC基线、配置/缓存契约、误差阈值、测试矩阵与风险清单 | 计划书PR合入且社区材料审核通过 |
| 2. 私有开发 | 独立注册、嵌套配置、逐层输入、共享KV、双宽MLP、语言权重审计 | 模块与非法输入测试通过；代码保持在私人仓库 |
| 3. A3验证 | 固定官方参考、12固定文本、完整MMLU、两模式八组性能与Profiler、内存及回归 | 同版本证据逐项通过，失败返回非零；问题修复后重采受影响基线 |
| 4. 材料与交付 | 用户/设计文档、验证摘要、原始证据清单与SHA256、复现命令、源码包 | 社区流程所需审核完成，取得提交代码授权 |
| 5. 代码合入 | 向 `cann/cann-recipes-infer` 提交实现PR，关联已合入计划书 | 目标仓CI、维护者评审及新模型Checklist通过 |

本计划书PR只提交当前 `design.md`，不包含实现文件、补丁、源码压缩包、模型权重、数据集或实机日志。实现代码在计划书合入、材料审核通过之前仅保留在私人仓库；不在本PR中公开代码仓访问地址或凭据。

## 明确的测量协议与证据管理

- 精度：12个固定自然语言样本，中文、英文、知识问答、摘要/改写和简单推理各不少于2个；保存两端输入ID、输出ID及完整文本。除Prefill外，在官方输出前缀下对齐teacher-forced Decode第1、2、7、31、63、127步（以实际可用输出步数为准），不以teacher forcing替代自由greedy生成。
- 数值：BF16模块输出与官方同dtype对照；FP32累积步骤按官方语义保留。要求双方finite，cosine≥0.999，且NRMSE≤0.015或max_abs_err≤0.05，最终logits top1一致。标准在正式测试前冻结，不根据失败结果放宽。
- 性能：128/512/1024/2048输入、最多128输出、batch1、eager/npugraph_ex；每组至少1预热+3正式，记录逐次值、均值、中位数、实际生成token、Prefill/Decode/TPOT与吞吐。计时与logits复制、Profiler、内存观察分开。
- Profiler：八组均启用 `profile_memory`，解析timeline、kernel、算子/API和内存表，保留完整产物清单与SHA256；仅生成原始目录不能判定通过。
- 内存：每种模式单独新进程，四shape各4次预热，在最大shape观察阶段峰值，再固定20次请求/观察器预热和20次测量。保留40条Host RSS与NPU allocated/reserved轨迹、连续输出和缓存回收。增长采用末5条中位数减前5条中位数，允许增长0 MiB；不延长预热、不强制GC/allocator trim/empty_cache。该协议测量推理进程，不声称全量评测工具本身无Host增长。
- 观察器：同步后一次读取公开 `memory_stats_as_nested_dict` 的allocated/reserved current/peak，Host使用psutil RSS；先验证与公开scalar计数等值，避免递归展平统计产生的观察器循环引用干扰测量。
- 环境：首次运行记录实际Atlas型号、board/chip/物理ID/容器可见索引映射、HBM、CPU/OS/驱动/固件、CANN与完整依赖。启动queue、allocator与线程变量使用白名单保存，各参考/benchmark/memory子进程一致；图模式启动前设置 `TASK_QUEUE_ENABLE=1`，配置校验后的实际值另记。
- 发布：验证摘要保留模块最差误差、完整固定样本、全量评分、逐次性能、内存轨迹和Profiler解析结果；数千条模块/Profiler明细留原始证据，正文关联路径和SHA256。证据封存后不修改fingerprint或拼接诊断，报告生成失败单独记录并修复。

## 需维护者确认的事项

1. 任务正文与测试说明目录存在差异，拟以测试说明的独立目录 `models/gemma_4_e2b/` 为准。
2. 采用固定revision原始文本续写、EOS=1；使用官方语言类时提供与官方多模态类纯文本调用的等价性证据。
3. 显式BF16 QK/PV与FP32 Softmax作为正确性起点；融合RMSNorm/RoPE/FA替换须独立对齐，不预先声称所有A3 shape可用或更快。
4. 26B共享路径代表性回归的拓扑和性能允许波动范围需先冻结；未确认时如实报告未判定。
5. 若A3资源不足或必要算子/图接口缺失，提交实测日志与预算并先确认替代方案，不能自行降低测试矩阵或改为其他硬件验收。
