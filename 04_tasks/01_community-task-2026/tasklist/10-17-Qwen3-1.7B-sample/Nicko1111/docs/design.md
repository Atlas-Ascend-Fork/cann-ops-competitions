# Qwen3-1.7B 模型样例设计文档

## 需求背景

### 需求来源

- [10月社区任务：Qwen3-1.7B 模型样例](https://www.hiascend.com/activities/task-center/details/73500202c71f4d30bc512daeb749f00d)，任务编号 10-17。
- 参与账号：Nicko1111。
- 代码目标仓：[cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。
- 方案与基线：[RFC #470](https://gitcode.com/cann/cann-recipes-infer/issues/470)。
- 代码交付：[PR #1107](https://gitcode.com/cann/cann-recipes-infer/pull/1107)。

### 背景介绍

现有 `models/qwen/` 提供 Qwen3-8B、Qwen2.5-7B-Instruct 的统一推理实现。本任务在该实现上增加 Qwen3-1.7B BF16 单卡样例，使较小规模的官方 Qwen3 模型能够使用相同的离线入口、Packed Sequence、Paged KV Cache 和图执行能力。

本次复用既有 `QwenForCausalLM` 与 `Qwen3Config`，不另建模型 forward 或算子实现。模型参数和权重形状由官方配置驱动。

## 需求分析

### 需求描述

在 Atlas A2 单卡上运行官方 `Qwen/Qwen3-1.7B`，完成模型注册、BF16/1TP YAML、文本生成、自验脚本、首次性能与内存基线以及用户文档。

### 需求拆解

| 需求 | 实现位置或验证方式 |
| --- | --- |
| 独立注册名 `qwen3_1_7b` | `executor/core/support_models.py`，复用现有 Qwen 模型类和配置类 |
| BF16 单卡配置 | `models/qwen/config/qwen3_1_7b_1tp.yaml` |
| 官方配置和权重兼容 | 审计 HF config、分片索引、tensor 名称/shape/dtype 及运行时加载覆盖 |
| 离线 Prefill/Decode | 统一 `executor/scripts/infer.sh` 入口 |
| 多请求与缓存 | batch 1/2、不等长输入、Packed Sequence、KV 写入/读取/回收和请求隔离 |
| 图模式 | `npugraph_ex` 预热、Decode 图复用及稳态编译计数 |
| 文本质量与性能 | 五个固定自然语言样本；四种输入长度的 Prefill/Decode 统计 |
| 内存与公共路径回归 | 连续20批内存记录；Qwen3-8B、Qwen2.5-7B-Instruct 改动前后对照 |

交付范围为 BF16、A2、单卡、离线推理；量化、Online PD、多卡和长上下文扩展不在本题范围内。

## 详细设计

### 模型与权重

官方模型、tokenizer 和配置固定为 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。保留原始 BF16 safetensors，不做权重转换。模型资产逐文件 SHA256 随基线元数据提供。

核对 hidden/FFN size、层数、Q/KV heads、head_dim、vocab、RoPE、QK-Norm、attention bias、embedding/lm-head 和权重命名。既有模型依据 `Qwen3Config` 构造层与张量，复用融合 QKV、gate/up 和权重映射，不硬编码 1.7B 规模。

`checkpoint_audit.py` 核对名称、shape、BF16、分片索引、重复项和未预期项。运行时记录每个源权重到目标参数的映射与加载覆盖，防止静默漏载。固定官方 checkpoint 包含显式 `lm_head.weight`，同时设置 `tie_word_embeddings=true`；审计保留该事实并校验加载关系，不仅凭配置假定 lm-head 被省略。

### 执行流程

```text
官方 tokenizer / chat template
    -> 固定 input_ids 与约定左截断
    -> Scheduler 组织不等长请求
    -> Packed Prefill，分配并写入 Paged KV Cache
    -> npugraph_ex Decode，复用 warmup 图与 KV Cache
    -> 按 EOS / max_new_tokens 返回有效 token 与文本
    -> 回收请求持有的 KV 页
```

Prefill/Decode 与底层 NPU 算子调度均复用仓库既有实现，不增加自定义 tiling、kernel 或公共配置开关。

### 模型注册与 YAML

注册项分别指向 `models.qwen.models.modeling_qwen.QwenForCausalLM` 和 `models.qwen.models.configuration_qwen.Qwen3Config`。目录参数 `--model qwen` 与注册名 `qwen3_1_7b` 分别使用各自已有入口。

交付 YAML 如下，模型路径使用占位符：

```yaml
model_config:
  model_name: "qwen3_1_7b"
  model_path: "/path/to/Qwen3-1.7B"
  dtype: "bfloat16"
  exe_mode: "npugraph_ex"
  enable_profiler: False
  with_ckpt: True
  enable_cache_compile: False
  enable_static_kernel: False
data_config:
  dataset: "default"
  input_truncated_len: 2048
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  seed: 17
parallel_config:
  world_size: 1
  attn_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
scheduler_config:
  max_new_tokens: 128
  batch_size: 2
```

性能用例将 batch 调整为1；功能用例覆盖 batch 1/2、输出上限1/128及 eager/npugraph_ex，并分别保存实际配置。

### 输出长度边界

现有离线 Scheduler 为维持静态 batch，在 EOS 后继续执行固定 Decode 计算步数。Prefill 已产生首个 token，因此输出上限为1时，首次 Decode 可能使有效返回长度越界。

修复在离线完成分支中按全局与请求级输出上限裁剪 `valid_output_len`；若 EOS 位于裁剪范围外，则完成原因记为 `length`。保留原有计算步数、KV 管理及在线行为。测试记录同时保存完整计算 token 和有效返回 token，以区分固定计算耗时与实际生成结果。

### 支持硬件与环境

| 项目 | 实测配置 |
| --- | --- |
| 平台 | Atlas A2，Ascend 910B4 V1，单卡32 GiB HBM |
| 设备映射 | 物理卡4/chip0，torch 逻辑设备 `npu:0` |
| Host | Ubuntu 22.04.5，aarch64，容器32 GiB内存、32核CPU配额 |
| 驱动 / 固件 | 26.1.1 / 9.0.0.9.220 |
| CANN | 9.0.0 |
| Python | 3.12.13 |
| PyTorch / torch_npu | 2.8.0+cpu / 2.8.0.post4；模型实际在NPU运行 |
| Transformers | 5.0.0 |

其他依赖、实际有效配置及模型文件校验值见代码交付的 `models/qwen/agentic/baseline/qwen3_1_7b_metadata.json`。

### 约束限制

本次验证输入上限2048、输出上限128、batch最大2，使用官方 tokenizer、greedy、seed17、`enable_thinking=False` 和 `add_generation_prompt=True`。超长输入按约定左截断，参考实现和本实现接收相同的截断后 token。权重和 tokenizer 须来自上述同一 revision。

## 可维可测分析

### 功能与精度标准

1. 官方 Transformers 5.0.0 的同一1.7B BF16模型作为参考；不以8B模型输出作为精度标杆。
2. 固定中文、英文、知识问答、摘要/改写、简单推理五个样本，保存模板、参数、输入及完整输出 token/text。文本须相关、通顺，无乱码、异常重复、语义断裂或可稳定辨识的质量退化。
3. 对照 eager/npugraph_ex 生成结果；不同后端出现差异时，依据任务书3.2说明原因并逐例检查文本质量。两例差异位置及 BF16 近似并列候选记录随固定输出材料保存。
4. 覆盖 batch1/2、输出边界、EOS、短/最大/超长输入和非法配置，检查有限 logits、必需权重完整加载、KV页写入/读取/回收、同形状请求扰动及连续请求隔离。
5. `npugraph_ex` 首次预热完成，正式Decode复用图；分别记录原生capture/replay与Dynamo计数。

### 性能与内存标准

首次建立1.7B基线，不设置优化提升比例。输入128/512/1024/2048，每组batch1、输出上限128、预热1次、正式3次；分开记录加载、编译/预热、稳态Prefill/Decode、生成长度、吞吐、均值和中位数。独立Profiler采集Level1、PipeUtilization及内存，不将采集开销混入无Profiler基线。

RFC冻结内存预算为Host/NPU各24 GiB；最大输入2048、batch2、输出上限128，连续20批不得OOM。记录权重加载后、Prefill/Decode峰值、逐批保留/已分配内存和Host RSS。后5批相对第6至10批的中位数增长上限为NPU64 MiB、Host256 MiB，同时检查KV回收。

### 兼容性分析

模型注册和YAML为新增项。公共调度器变更先保留原版Qwen3-8B和Qwen2.5-7B-Instruct的功能、输出与代表性性能基线，再运行同配置回归。覆盖两种模式、有效token与完成原因、KV回收以及128/512输入的三次计时；性能界限使用RFC冻结的原版三次最慢值乘1.10。对调度器停止边界、批处理及在线直接调用方运行适用单元测试。

### 自验入口与交付记录

从模型目录执行：

```bash
cd models/qwen
python3 tests/checkpoint_audit.py --model-dir /path/to/Qwen3-1.7B --sha256
bash ../../executor/scripts/infer.sh --model qwen --mode offline --yaml qwen3_1_7b_1tp.yaml
python3 tests/qualify_functional.py --model-dir /path/to/Qwen3-1.7B \
  --assets /path/to/verified/assets.json --output-dir /path/to/new/functional-run \
  --mode npugraph_ex --batch-size 2 --max-new-tokens 128
```

`qualify_baseline.py --kind performance|memory|profile` 分别采集基线，使用同样的模型、资产索引及独立输出目录。每次运行由有限超时的监督进程管理，记录命令、PID、日志、进度和退出状态，异常清理整组进程并保留部分结果。

已采集的2026-10-17 A2结果为：八组144批功能检查完成；五样本文本质量对照完成；四种输入的Prefill均值分别为40.555/41.606/43.104/62.160 ms，Decode均值分别为9.864/9.970/10.379/10.682 ms/步；20批NPU分配峰值约4.71 GiB、Host RSS峰值约8.30 GiB；旧模型44例有效token和完成原因与原版一致。

小型记录、实际配置和完整五样本输出随代码交付。大型Profiler、权重与数据集不入Git，按报告索引和校验值获取。README提供环境、下载、配置、启动及结果说明。代码提交通过仓库CI与变更文件检查；全仓pre-commit中原仓既有格式/权限问题保留在原始记录中。
