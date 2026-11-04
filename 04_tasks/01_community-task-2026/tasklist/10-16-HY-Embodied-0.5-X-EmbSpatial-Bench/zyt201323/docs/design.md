# HY-Embodied-0.5-X NPU 接入与 EmbSpatial-Bench 评测设计文档

| 项目 | 内容 |
| --- | --- |
| 社区任务 | 10 月社区任务：新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测（tasklist `10-16-HY-Embodied-0.5-X-EmbSpatial-Bench`） |
| 贡献者 | zyt201323 |
| 设计仓 / 路径 | `cann/cann-ops-competitions` → `04_tasks/01_community-task-2026/tasklist/10-16-HY-Embodied-0.5-X-EmbSpatial-Bench/zyt201323/docs/design.md` |
| 实现仓 / 样例目录 | `cann/cann-recipes-embodied-ai` → `manipulation/hy-embodied-0.5-x/infer_with_torch/` |
| 优化/说明文档 | `docs/manipulation/hy-embodied-0.5-x/infer_with_torch/`（实现仓） |
| 核心门槛 | EmbSpatial-Bench human-verified evaluation 全量 **3640** 题，样本级 Accuracy **≥ 80.0%**（正确 ≥ 2912） |
| 说明 | 本任务为 **Recipes 样例接入**（非 TBE/Ascend C 单算子）。官方 `design_template.md` 中 Host Tiling / Kernel 章节按「环境加载侧 / NPU 执行侧」映射改写；注意力与 MoT 路由适配、评测协议与验收标准以任务书为准。 |

# 1. 需求背景（required）

## 1.1 需求来源

通过 CANN 社区任务，在 `cann-recipes-embodied-ai` 新增 HY-Embodied-0.5-X 的标准化 **NPU 推理样例**，并在锁定评测协议下完成 EmbSpatial-Bench 全量评测。任务书：《新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测》。

## 1.2 背景介绍

### 1.2.1 HY-Embodied-0.5-X 与上游推理栈

HY-Embodied-0.5-X 面向真实机器人「理解—推理—行动」闭环，基于 MoT（Mixture-of-Transformers）：总参数约 4B、单次激活约 2B，支持图像输入、thinking / direct 生成与 OpenAI 兼容服务。上游 CUDA 环境强依赖特定 Transformers 版本与 FlashAttention。

| 对象 | 路径 / 基线（以实现仓 RFC/实验记录最终锁定为准） |
| --- | --- |
| 上游模型仓 | `Tencent-Hunyuan/HY-Embodied-0.5-X` |
| 上游推理入口 | `src/hy_embodied/inference/`（`HyEmbodiedPipeline`）、`src/hy_embodied/cli/infer`、`docs/inference.md` |
| 模型结构 | transformers `hunyuan_vl_mot`（`modeling_hunyuan_vl_mot.py`）；PyPI 常规版本通常不含该模型，需源码安装锁定 commit |
| 模型权重 | HF `tencent/HY-Embodied-0.5-X`（safetensors 分片；**权重不入库**） |
| CUDA 侧关键依赖 | `flash_attn` 模块级硬导入；CUDA device / autocast / stream / RNG |

模型结构概要：

- **视觉塔**：HYViT2（动态分辨率），注意力调用 `flash_attn_func` / `flash_attn_varlen_func`；
- **语言主干**：MoT-2B，约 32 层，GQA；每层 text / vision 双路径（qkv / o / mlp / norm），由 modality mask 经 `_mask_apply` 路由；
- **MoT 注意力** `_flash_attention_forward_mot`：Step1 全序列因果注意力；Step2 对视觉 token 段做双向注意力并覆盖写回。

### 1.2.2 上游（CUDA）实现现状分析

| 参数 | 含义 | 上游支持 | 约束 |
| --- | --- | --- | --- |
| image | 输入图像 | PIL / 路径；官方 processor 动态分辨率 | 本期评测为单图 |
| prompt | 文本 | chat template；thinking 开关可控 | 评测只用白名单字段 |
| completion | 生成文本 | direct / thinking（含 `<think>` / `<answer>`） | 计分只取最终答案 |
| dtype | 精度 | bf16 直载直算 | 与上游一致 |
| 解码 | temperature 等 | CLI 可配 | 正式协议冻结后中途不改 |

与 NPU 迁移直接相关的事实：

1. `modeling_hunyuan_vl_mot.py` **顶层硬导入** `flash_attn`，且 MoT **不走 SDPA 开关绕过**——仅 `.to("npu")` 不够；
2. MoT 注意力两阶段语义必须保持（因果 + 视觉段双向覆盖）；
3. `_mask_apply` 每层多次调用，含 NonZero / IndexPut 类控制流，易成为 NPU 上 launch / Memcpy 热点。

### 1.2.3 EmbSpatial-Bench 基准现状

面向具身空间理解的多选题基准，六类 relation（egocentric）。官方主指标为**样本级 Accuracy**（禁止用六类宏平均替代总分）。本任务仅使用 human-verified evaluation（`embspatial_bench.json`），**禁止**使用 SFT split 或答案泄漏。

| relation | 样本数 | relation | 样本数 |
| --- | ---: | --- | ---: |
| close | 612 | under | 602 |
| far | 594 | left | 616 |
| above | 596 | right | 620 |
|  |  | **总计** | **3640** |

## 1.3 任务功能分析

样例功能 = **NPU 图文推理**（保持 direct / thinking 契约）+ **EmbSpatial 标准化评测**：

- 输入：本地图像、prompt（原问题 + 四候选项）、NPU 配置（权重路径、device、dtype、max tokens、temperature、seed、thinking）；
- 输出：原始 completion、解析答案、逐题 JSONL、汇总统计；
- 精度：`Accuracy = 正确数 / 3640 × 100%`，并报告六类分项。

# 2. 需求分析（required）

## 2.1 需求描述

在 `cann-recipes-embodied-ai` 新增 `manipulation/hy-embodied-0.5-x/infer_with_torch/`：将 CUDA device / FlashAttention / RNG 等适配到 NPU；MoT 路由、视觉编码器、语言主干、KV Cache、lm head **关键计算不得静默回落 CPU**；复用 `HyEmbodiedPipeline` 契约或可维护 patch，**不复制完整上游代码**；在 RFC 锁定协议下全量评测 Accuracy ≥ 80%，并交付可复现记录。

## 2.2 需求拆解

| 编号 | 需求 | 设计落点 |
| --- | --- | --- |
| RQ01 | 平台 / 版本 / 权重 / 数据 / 评测协议锁定 | §3.2.2、§3.3 |
| RQ02 | 图文推理；direct / thinking 格式正确 | §3.2.1–3.2.3 |
| RQ03 | 关键计算全程 NPU，无 CUDA 残留 / CPU 静默回退 | §3.2.2–3.2.3 |
| RQ04 | 全量 3640、无答案泄漏、六类分项 | §3.2.4 |
| RQ05 | Accuracy ≥ 80%；异常计错；不择优挑分 | §3.2.4、§4.1 |
| RQ06 | 断点续跑；同 qid 不重复计分 | §3.2.4 |
| RQ07 | ≥10 固定 smoke（覆盖六类 / thinking / 异常） | §4.1 |
| RQ08 | 延迟 / 吞吐 / 显存基线（无强制加速门槛） | §4.1 |

# 3. 详细设计（required）

## 3.1 任务分析

### 3.1.1 数学公式 / 语义基线

- 注意力：`Attention(Q,K,V)=softmax(QK^T/√d + M)V`（LM head_dim 与视觉 head_dim 按上游配置）；
- MoT：Step1 全序列因果；Step2 视觉段双向覆盖写回；
- 评测：`Accuracy = 正确数 / 3640 × 100%`。

### 3.1.2 支持数据类型

| 项 | 类型 | 说明 |
| --- | --- | --- |
| 权重 / 前向 / KV | bfloat16 | 与上游一致 |
| softmax / 部分 norm 累加 | float32 | 按上游 / FA 核内部行为 |
| 索引 / mask | int / bool | 控制张量可短时落 Host |
| 图像 | RGB → processor 张量 | 使用官方 image processor |

### 3.1.3 支持形状与输入规模

- 正式评测 **B=1** 顺序推理；
- 图像动态分辨率；本设计默认将 `max_pixels` 收敛到可复现且精度稳定的上限（小样本验证后冻结，见 §3.2.3）；
- `max_new_tokens` 在精度持平前提下取可控上限（小样本验证后冻结）。

## 3.2 样例实现

### 3.2.1 总体设计与目录结构

```text
manipulation/hy-embodied-0.5-x/infer_with_torch/
├── README.md
├── requirements.txt                 # 锁定依赖，禁止自动升级未验证版本
├── scripts/
│   ├── run_npu_server.py            # 可选：OpenAI-compatible server（先装 FA shim）
│   └── run_embspatial_eval.sh
├── src/
│   ├── compat/
│   │   ├── flash_attn_npu.py        # flash_attn → npu_fusion_attention shim
│   │   ├── mot_mask_apply.py        # B=1 mask_apply（CPU 索引 + 推理跳过空模态占位）
│   │   └── mot_fused_ops.py         # 可选：RMSNorm / SwiGLU / RoPE（默认关）
│   ├── npu_runtime.py               # 设备初始化、加载 Pipeline、适配开关
│   ├── npu_generate.py              # seed / greedy / </answer> 早停
│   ├── prompt_builder.py            # evalscope 等 prompt 协议
│   ├── answer_parser.py             # 仅 A/B/C/D（thinking 不泄漏）
│   ├── evaluate_embspatial.py       # 全量/子集评测 + resume + JSONL
│   ├── recompute_metrics.py         # 独立复算（不加载模型）
│   ├── bench_latency.py             # 延迟抽样基线
│   └── smoke_test.py
└── tests/                           # 解析等单测
```

设计原则：**复用不复制**——上游 Pipeline / modeling 尽量零拷贝；NPU 适配集中在 `compat/` 与 runtime 薄封装。

对应实现仓文档目录：`docs/manipulation/hy-embodied-0.5-x/infer_with_torch/`。

### 3.2.2 Host 侧设计（环境、加载与配置）

对应模板「host / tiling」角色：负责版本校验、设备与调度参数、权重加载、协议配置，不在 Host 上执行 MoT 主体 MatMul。

1. **版本 fail-fast**：校验 CANN / torch / torch_npu / transformers 与 RFC 锁定一致；不匹配立即报错，不自动升级；
2. **设备与精度**：`device=npu:0`，bf16 加载；加载后断言关键参数在 NPU；
3. **调度默认（可环境变量覆盖）**：
   - `TASK_QUEUE_ENABLE=2`
   - `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
   - `CPU_AFFINITY_CONF=1`
   - `torch.npu.set_compile_mode(jit_compile=False)`（动态 decode 形状下避免 MaskedScatter 编译失败）；
4. **随机性**：正式协议使用固定 seed + `force_greedy`，保证可复现；
5. **图像预处理**：官方 processor；`default_to_square=False`；`max_pixels` 默认 `655360`（小样本相对 1M/768k 精度持平且更快，可 env 覆盖）。

### 3.2.3 Kernel / NPU 执行侧设计

对应模板「kernel」角色：在 NPU 上完成视觉塔、MoT、生成。

#### （1）flash_attn → NPU shim（核心适配）

`compat/flash_attn_npu.py` 在导入 modeling 前注入 `sys.modules["flash_attn"]`：

- 实现 `flash_attn_varlen_func` / `flash_attn_func`；
- 后端：`torch_npu.npu_fusion_attention`，布局 TND / BSND；
- 因果：`sparse_mode=3`（right-down causal），`pre_tockens/next_tockens` 按 Ascend FA 推荐设置；
- B=1 时用 shape 推导 `actual_seq_*`，减少逐步 `tolist()` D2H；
- 每次构造**新的** seq-len list，避免绑定原地修改造成轨迹污染。

#### （2）MoT `_mask_apply` 图优化（B=1）

`compat/mot_mask_apply.py`（默认开启，可用 env 关闭）：

- 用 CPU `nonzero` 取索引 + `index_select` / `index_copy_`，削减 NPU 上 NonZero/IndexPut 碎片；
- **推理**时跳过空模态的 training-only placeholder MatMul（decode 多为纯文本 token，避免每层空跑 vision Linear/MLP）；
- 索引缓存挂在 **mask 张量对象**上（同 forward 跨层复用）；**禁止**按裸 `data_ptr` 跨步缓存（分配器复用会串内容）。

可选（默认关闭，需同输出长度 A/B）：`npu_rms_norm` / SwiGLU / RoPE、FA prefill 免 clone、稠密纯文本快路径等。

#### （3）生成与早停

`npu_generate.py`：

- `torch.inference_mode()`；
- `</answer>` 单 token 时注册为 `eos_token_id`，并保留 Python `StoppingCriteria` 兜底；
- 返回可选 `input_tokens` / `output_tokens` 与 NPU 显存快照。

#### （4）与上游差异（原因）

| # | 差异 | 上游 CUDA | 本设计 NPU | 原因 |
| --- | --- | --- | --- | --- |
| 1 | 设备 | cuda | npu | 平台要求 |
| 2 | 注意力 | flash_attn CUDA | `npu_fusion_attention` shim | 硬导入无法绕过 |
| 3 | mask 路由 | 布尔高级索引 | B=1 CPU 索引 + index_copy | 降低控制算子碎片 |
| 4 | 空模态 | placeholder.mean()*0 | 推理跳过 | 去掉 decode 无效 vision GEMM |
| 5 | max_pixels | 默认可达约 4M | 默认 640k（可覆盖） | 小样本精度持平、降低视觉 token |
| 6 | 随机性 | 可采样 | 正式 greedy+seed | 可复现验收 |

### 3.2.4 评测工具链设计

**拟锁定正式协议（开发/验收以 RFC 最终冻结为准；小样本已验证可复现）**：

| 项 | 值 |
| --- | --- |
| 方式 | generation + thinking |
| prompt | `evalscope`（含空间消歧；`evalscope_ego` 不默认） |
| 范围 | human-verified 全量 3640，原始顺序 |
| 解码 | `max_new_tokens=512`，`temperature=0.05`，`seed=42`，`force_greedy` |
| 早停 | `</answer>` EOS |
| 解析 | 仅 A/B/C/D（或约定索引）；不可解析计错；thinking 文本不泄漏 |
| 计分 | 正确数/3640；六类分项；禁止缩分母 / 人工改答案 / SFT 泄漏 |

工程要点：

- `evaluate_embspatial.py`：resume、逐题 JSONL（qid / relation / raw / pred / gt / correct / tokens / 耗时 / 可选显存）；
- `recompute_metrics.py`：只读 JSONL 复算，与模型解耦；
- OOM / 超时 / 异常计错并记录；同 qid 不重复计分。

### 3.2.5 流程示意

```mermaid
flowchart LR
  A["图像 + 问题 + 四选项"] --> B["官方 Processor + evalscope prompt"]
  B --> C["FA shim + MoT on NPU"]
  C --> D["generate + answer EOS"]
  D --> E["answer_parser → A/B/C/D"]
  E --> F["JSONL + 独立复算"]
```

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2（Ascend910B 类，单逻辑 NPU，HBM 64GB/die） | √ |
| Atlas A3 | 任务书允许；本设计以 A2 实机验证为主，A3 需同等协议复测后声明 |

拟锁定软件栈（以实现仓 RFC/实验记录为准）：

| 项 | 值 |
| --- | --- |
| OS | Ubuntu 22.04 aarch64 |
| CANN | 9.1.x |
| Python | 3.12.x |
| PyTorch / torch_npu | 与 CANN 9.1 匹配的配对版本（本机验证线：2.12 + 2.12.post2） |
| 精度 | BF16 |

## 3.4 约束限制

1. 仅推理样例：不交付 SFT/训练；不得用评测答案微调或硬编码冲 80%；
2. 不入库：权重、数据集、base64 图、缓存、凭据、大体量原始日志；
3. 正式计分仅锁定 generation+thinking 协议；其他协议不得混报；
4. transformers / CANN / torch_npu 版本不匹配时 fail-fast；
5. 可选融合算子默认关闭：须同输出 token 长度 A/B，避免 greedy 轨迹漂移；
6. 公共 `module/` 默认不改；若改公共代码须列直接调用样例并回归。

# 4. 可维可测分析

## 4.1 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 全量 3640 样本级 Accuracy ≥ 80.0%；六类分项公开；异常计错；可独立复算 | 任务书第 3 章 |
| 性能 | 不设强制加速比；须记录 TTFT / 吞吐 / 端到端 / NPU·Host 峰值内存，可作为后续优化基线 | 任务书 2.4 |

测试设计：

| 组别 | 覆盖 |
| --- | --- |
| 解析单测 | A/B/C/D、不可解析、thinking 不泄漏 |
| smoke ≥10 | 六类 relation、thinking、异常路径；无 NaN/乱码/跨样本污染 |
| 小样本正确性 | 固定 qid 子集可复现；失败即停修 |
| 延迟抽样 | `bench_latency.py`（交互默认小 warmup/sample；正式可用 10/60） |
| 全量 | 一次完整正式跑；多次则全部报告，不挑最高分 |
| 续跑 | 中断恢复不重复计分、不改分母 |

小样本阶段已观察到：仅个别难例（如 left 类 `mp3d_7`）在固定 greedy 协议下稳定错误；不以 prompt 刷分翻盘作为验收手段。

## 4.2 兼容性分析

新增独立样例目录，不改变其他样例默认行为。依赖安装于独立环境并由 requirements 锁定。若抽取公共能力进入 `module/`，须证明可复用并完成相关样例回归。

# 5. 参考资料

- 官方设计模板：`04_tasks/01_community-task-2026/resources/design_template.md`
- 设计文档 CheckList：腾讯文档（社区任务设计文档提交规范）
- 任务书：《新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测》
- 实现仓贡献指南：`cann-recipes-embodied-ai/CONTRIBUTION.md`
- 上游：https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X
- EmbSpatial-Bench：https://github.com/mengfeidu/EmbSpatial-Bench ；论文 https://arxiv.org/abs/2406.05756
