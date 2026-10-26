# 需求背景（required）

## 需求来源

社区任务 10-7「10月社区任务-Gemma-4-E2B 推理样例」，在 Atlas A3 上复用 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 的 `models/gemma_4/`，接入 Gemma-4-E2B 纯文本 BF16 Offline 推理：按官方 `text_config` 支持**逐层输入、跨层 KV 共享与双宽 MLP**，并审计权重加载。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)；本设计为设计评审稿，验收以任务书与官方权重/配置为准。

## 背景介绍

### 模型与仓内现状

`models/gemma_4/` 已有目录：`README.md`、`baseline/`、`agentic/`、`config/`、`models/`、`chat_template.jinja`、`requirements.txt`。E2B 为 Gemma-4 系列的边缘小模型（~2B 有效参数），其 `text_config` 相对常规 Gemma-4 有三个结构特性，是本任务核心：

| 特性 | 语义 | 实现要点 |
| --- | --- | --- |
| 逐层输入 | text tower 按层接收输入/输出（layer-by-layer 调用形态，便于逐层 KV/图切分） | 模型 forward 暴露单层入口 `layer_forward(hidden, layer_idx, cache)` |
| 跨层 KV 共享 | 每 N 层共享一组 KV cache（KV head/层数少于层数，按 `text_config` 的共享映射） | cache 按"物理 KV 组"分配，共享组内各层读写同一物理 cache，不复制 |
| 双宽 MLP | FFN 宽度按官方 `text_config` 的双宽结构（gate/up 与 down 维度异形，非标准 4x/8x 单值） | 严格按 config 的 `intermediate_size(s)` 构建，不从 hidden 推导 |

本任务为纯推理样例接入：BF16 全程、Offline 推理形态（非服务化），不涉及训练与量化。

# 需求分析（required）

## 需求描述

在 `models/gemma_4/` 下新增 E2B 纯文本 BF16 Offline 推理样例：加载官方 E2B 权重与 `text_config`，实现逐层输入接口、跨层 KV 共享与双宽 MLP 结构，权重加载逐张量审计；通过生成功能、精度对拍与性能三关验证，A3 实测交付。

## 需求拆解

1. 配置接入：按官方 `text_config` 新增 E2B 模型配置（层数/heads/KV 共享映射/双宽 MLP 维度/归一化方式），与 config/ 既有家族配置同构。
2. 结构实现：逐层输入入口、物理 KV 组共享 cache、双宽 MLP、RMSNorm/GEGU 按 config;BF16 全程，异常 dtype 输入显式拒绝。
3. 权重加载审计：逐张量对照官方 safetensors index——名称、shape、dtype、非容差项为零;审计报告落盘。
4. 推理样例：Offline 生成脚本（chat_template.jinja 已有）、greedy 与采样、单/多轮;`infer.sh` 入口。
5. 验证：小模型 golden 对拍(logits 余弦/argmax 一致率)、性能(时延/吞吐)、README 复现说明。

# 详细设计（required）

## 模型分析

### 结构语义（按官方 text_config 落地，不自行推测）

- **跨层 KV 共享**：`text_config` 给出共享映射（如每层 i 使用 `kv_layer[i]` 组）。cache 容器按物理组数分配 `K/V [B, kv_heads, S, D]`；层 i 前向读写第 `map[i]` 组——共享组内后层不重复投影 KV 或直接复用前层 K/V 投影结果（以官方建模代码为准，两种形态审计后择一）。
- **双宽 MLP**：`gate/up: [d_ff_wide, hidden]`、`down: [hidden, d_ff_wide]`，`d_ff_wide` 按 config 双宽字段；激活按官方（GeGLU 系）；禁止用 `4*hidden` 兜底。
- **逐层输入**：外层 pipeline 逐层调用单层入口并显式传 position_ids/cache 指针;整模型入口保留（内部循环单层入口），两形态共享实现。
- **归一化**：RMSNorm eps、normalizer（Gemma 系 `(hidden * normalizer)` 缩放惯例）严格按 config。

### 权重加载审计（任务书点名项）

| 审计项 | 通过条件 |
| --- | --- |
| 名称 | safetensors index 全张量被消费，模型无孤儿参数（tie_word_embeddings 单独核对） |
| shape | 逐张量与模型参数严格相等;双宽 MLP 与 KV 共享层是审计重点 |
| dtype/数值 | 转换策略显式（BF16 cast 仅一次）;加载后逐张量 min/max/mean 抽样报告，NaN/Inf 计数=0 |
| 共享参数 | KV 共享组若对应共享权重（如共用 k/v projection），以引用而非拷贝加载，参数计数核对 |

## 接入与实现设计

### 工程组织

```text
models/gemma_4/
├── config/gemma4_e2b_text.yaml        # 按 text_config 的 E2B 配置
├── models/gemma4_e2b/                  # 建模代码（与既有 gemma_4 models 同构）
│   ├── configuration.py  modeling.py   # 单层入口 + KV组 cache + 双宽 MLP
│   └── weight_loader.py                # 审计式加载器（报告落盘）
├── baseline_e2b/generate.py            # Offline 推理样例（greedy/采样、单/多轮）
└── infer.sh                            # 入口（模型/权重/输出路径参数化）
```

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 加载审计 | weight_loader 报告逐项核对 | 全通过，无孤儿/缺张量 |
| 结构正确性 | KV 共享映射、双宽维度单测（构造 config 对照手算 shape） | 全通过 |
| 精度 | 官方参考实现（GPU/CPU BF16）同输入 logits 对拍：逐 token argmax 一致率 + 末层余弦相似度;长文本(≥2k token)专项 | 任务书门限（argmax 一致率 ≥ 99%，cos ≥ 0.99 量级，以任务书为准） |
| 功能 | 多轮对话、chat template 渲染、异常输入(超长/空)处理 | 不崩溃、行为有定义 |
| 性能 | A3 BF16 单卡:首 token 时延、decode 吞吐 | 报告基线值，门槛以任务书为准 |
| 复现 | README:权重下载、环境(torch_npu 版本)、三条命令跑通 | 第三方可复现 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| text_config 字段命名与 HF 版本漂移 | 以任务书附带的官方 config.json 为唯一事实源;加载器对未知字段报错不忽略 |
| KV 共享两种实现形态（复用投影 vs 重投影）与 golden 不一致 | 开发初期与官方 modeling 对拍冻结形态，写入配置注释 |
| 双宽 MLP 维度猜错 | shape 审计在加载期立即失败，不进入推理 |
| 逐层输入引入额外调度开销 | 层间无 Host 同步（cache 指针直传），profiling 核对逐层 gap |
