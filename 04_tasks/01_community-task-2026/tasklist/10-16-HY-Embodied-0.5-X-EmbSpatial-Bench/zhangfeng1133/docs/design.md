# 需求背景（required）

## 需求来源

社区任务 10-16「新增 HY-Embodied-0.5-X 并完成 EmbSpatial-Bench 评测」，在 Atlas A2/A3 单卡上基于 [cann/cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai) 新增标准化 NPU 推理样例，并按官方协议完成 EmbSpatial-Bench 全量评测（目标 Accuracy ≥ 80%）。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)、目标仓 [CONTRIBUTION.md](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/CONTRIBUTION.md) 与任务书 §2.1 的 RFC 机制；本设计为**设计评审稿 v2**，平台/版本/协议以开发前锁定的 RFC 为准，不将设计目标表述为已完成的实测结论。

**修订记录**：v1 自动检视未通过（70 分：Host/Tiling 设计 5/20、Kernel 设计 5/20），且存在多处事实错误，v2 逐项修正：

| v1 问题 | v2 落实 |
| --- | --- |
| 模型参数误写「约 0.5B 量级」 | 实为 **MoT-2B 架构：总参约 4B、单次激活约 2B**（任务书 §1 + 上游 README/config.json 双渠道核实），见算子分析 |
| 样例目录误设 `3d_vision/hy_embodied_0_5_x/` | 改为任务书 §4 指定的 `manipulation/hy-embodied-0.5-x/infer_with_torch/`（CONTRIBUTION 小写短中划线命名规范），见详细设计 |
| 空间关系误写「方位、距离、包含、遮挡」 | 实为 egocentric 六类空间关系（close/far/left/right/above/under，官方仓样例字段实证），见评测对象 |
| 缺 Host/Tiling 设计、Kernel 设计章节 | 新增独立章节（Host 侧流水线与资源分块规划；attention kernel 选型与算子替换 patch 矩阵） |
| 缺硬件与约束、可维可测分析 | 新增独立章节；评测/稳定性/交付逐条对齐任务书 §2、§3、§4 全条目 |
| 平台仅写 A3 | 改为任务书 §2.1 口径：Atlas A2 或 A3 单卡，RFC 锁定其一 |

## 背景介绍

### 模型对象（上游双渠道核实：GitHub README + HuggingFace config.json）

| 项 | 事实 | 来源 |
| --- | --- | --- |
| 架构 | MoT-2B（Mixture-of-Transformers 双模态塔），总参约 4B、单次激活约 2B，图像/视频输入，thinking/direct 两种生成模式，OpenAI 兼容服务 | 任务书 §1 + 上游 README |
| 模型类 | `HunYuanVLMoTForConditionalGeneration`（`model_type: hunyuan_vl_mot`）；transformers 自 commit `9293856` 起原生注册 `HunYuanVLMoT{Config,Model,Processor}`，**无需 trust_remote_code、无本地 modeling 代码** | 上游 docs/architecture.md + HF config.json `architectures` |
| 语言主干关键参数（config.json 实测） | hidden_size 2048、num_hidden_layers 32、num_attention_heads 16 / num_key_value_heads 4（GQA 4:1）、head_dim 128、intermediate_size 6144、silu、RMSNorm（eps 1e-5）、QK-Norm、RoPE 动态缩放（theta 10000）、max_position_embeddings 262144、vocab 120818、tie_word_embeddings=true、dtype bfloat16 | HF `tencent/HY-Embodied-0.5-X/config.json` |
| 视觉塔与投影 | 结构以权重仓 config/processor 配置为准（RFC 锁定项）；样例复用 `HunYuanVLMoTProcessor`，不重写视觉塔内部实现 | 上游 docs/architecture.md「no local modelling code」 |
| 推理入口契约 | `HyEmbodiedPipeline.from_pretrained(..., torch_dtype=bfloat16)` + `GenerationConfig(max_new_tokens=32768, temperature=0.05, enable_thinking=...)`；CLI `python -m hy_embodied.cli.infer`；可选 OpenAI 兼容 server（`hy_embodied.cli.server`）；thinking 模式输出 `<think>[推理]</think><answer>[答案]</answer>`，direct 模式仅输出答案 | 上游 README Quick Start |
| 上游依赖（NPU 适配对象） | CUDA 12.6、PyTorch 2.10.0、Python 3.12、`flash_attn==2.8.3`、`liger-kernel`、transformers 指定 commit、accelerate/deepspeed/timm；≥16 GB VRAM | 上游 README Prerequisites |
| License | 代码 Apache-2.0；权重/数据集 License 于 RFC 阶段核对登记（任务书 §2.1-4） | 上游 LICENSE + 任务书 |

### 评测对象（EmbSpatial-Bench 官方仓核实）

| 项 | 事实 |
| --- | --- |
| 构成 | 具身场景（mp3d / ai2thor 等来源）egocentric 视角图像 + 四选一空间关系多选题；**六类空间关系 close / far / left / right / above / under**（官方仓样例 `relation` 字段实证 close、under；全集以 RFC 锁定的 human-verified split 实际字段为准） |
| 数据格式 | JSONL：`question_id` / `data_source` / `scene_id` / `question` / `relation` / `image`(base64) / `answer_options`(4) / `answer`(索引) / `objects`(bbox)；图像与问题一一对应 |
| 评测协议 | 官方两种方式：**generation**（生成答案后解析）与 **likelihood**（逐选项打分）；主指标 Accuracy；默认采用官方多选生成评测 + thinking 开启（任务书 §2.1-3），换用 likelihood 须维护者确认且不得与其他协议结果混报 |
| 数据边界 | **仅使用 human-verified evaluation split**；EmbSpatial-SFT 数据、`answer`/`objects` 等标注字段严禁进入模型上下文（任务书 §2.3-1、§7） |
| 样本规模 | 以 RFC 锁定的 human-verified split 为准；全量按固定顺序推理，逐题留痕，不漏题不重复计分 |

### 仓内现状（工作量边界）

目标仓 `manipulation/` 现有 10 个样例（pi0、smolvla、openvla、Isaac-GR00T、spirit-v1.5 等），**均无 HY-Embodied，属全新接入**；`infer_with_torch` 惯例（参照 `manipulation/pi0/infer_with_torch/`）：modeling 封装 + 推理脚本 + 测试脚本 + README/README_en + 运行 shell + pyproject；样例名小写短中划线（如 spirit-v1.5）。运行时 patch 先例：`manipulation/Isaac-GR00T/adaptor_patches/`。公共能力上沉 `module/` 须证明可复用并回归全部直接调用样例（`module/` 现仅 `dit_cache/`，本任务默认不上沉，见兼容性分析）。EmbSpatial 数据按官方渠道获取，不自造题目。

### RFC 锁定清单（开发前冻结，验收中途不改参）

| 类别 | 锁定项 |
| --- | --- |
| 平台与版本 | Atlas A2 或 A3（单卡，二选一定稿）；CANN / torch_npu / PyTorch / Python 版本组合；精度类型（默认 BF16）与峰值显存目标 |
| 上游锚点 | HY 上游 commit/release tag；目标仓 cann-recipes-embodied-ai 基线 commit；`tencent/HY-Embodied-0.5-X` 权重 revision 与哈希；权重与数据集 License 核对结论 |
| 评测协议 | EmbSpatial 官方仓 commit、dataset revision、human-verified split、样本总数、评测方式（默认 generation + thinking，thinking 开关显式可配）、prompt 模板、图像预处理、解码参数（max_new_tokens / temperature / top-p / seed）、答案解析规则（A/B/C/D 或候选索引） |

# 需求分析（required）

## 需求描述

在 `manipulation/hy-embodied-0.5-x/infer_with_torch/` 新增 NPU 推理样例：复用上游 `HyEmbodiedPipeline`/CLI 契约（必要时以可维护 patch 适配，不复制完整上游代码），将 CUDA device、autocast、FlashAttention/Liger、stream、随机数与分布式逻辑适配到 NPU，MoT 路由、视觉编码器、语言主干、KV Cache、lm_head 五个关键计算点**无 CPU 静默回退**；按 RFC 冻结协议跑通 EmbSpatial-Bench human-verified 全量评测，样本级 Accuracy ≥ 80%，六类 relation 分类统计公开，逐题结果可独立复算，全程可复现可追溯。

## 需求拆解

1. **RFC 与锚点**：按上文清单锁定平台/版本/权重/数据/协议五类项，RFC 文档随 PR 提交关联。
2. **样例工程**：目录与配置（权重路径、设备、dtype、max token、temperature、top-p、seed、thinking 开关全部显式可配、路径占位符化）、CLI/单样本批/单入口、可选 OpenAI server。
3. **NPU 适配**：attention kernel 替换、liger 算子替换、autocast/device/stream 适配、随机数统一、依赖不匹配快速报错（不自动升级未验证版本）。
4. **评测工具**：数据加载（官方 split、dataset revision 可配）、固定顺序推理、答案解析（仅 RFC 约定格式）、逐题 JSONL、汇总 JSON/CSV、独立复算脚本（模型输出与计分解耦）。
5. **稳定性**：≥10 个固定 smoke 样本（六类关系、thinking/direct、异常图片/超长输出）、OOM/超时计错、断点续跑同 `question_id` 不重复计分、无状态污染与内存持续增长。
6. **性能基线**：首 token 延迟、输出 token 吞吐、端到端耗时、NPU/Host 峰值内存（本任务无强制性能收益门槛，结果作后续优化基线）。
7. **文档与交付**：README、优化过程文档、实验记录，对齐任务书 §4 八项交付件。

# 算子分析

## 模型结构

MoT（Mixture-of-Transformers）按模态解耦权重：文本/图像各自独立的注意力、MLP 与归一化参数分支，token 按模态路由到对应分支后共享序列语义——这是「总参 4B、每 token 仅激活约 2B」的来源。一次 EmbSpatial 问答的计算链路：

```text
图像 ─► 视觉编码器(patch 切分/多分辨率按 processor) ─► 投影对齐 ─┐
                                                                 ├─► MoT 语言主干 ×32 层 ─► lm_head(tied) ─► 采样 ─► <think>/<answer> 文本
问题文本 ─► 分词(vocab 120818) ──────────────────────────────────┘
```

关键结构参数（HF config.json 实测）：32 层、hidden 2048、GQA 16 查询头/4 KV 头（head_dim 128）、intermediate 6144、RMSNorm+QK-Norm、动态 RoPE、tie_embedding。单层语言主干 ≈ attention（QKV+O 投影，GQA 折算）+ MLP（3×h×i）+ 归一化，量级约 50M 参数/层；逐张量构成以权重 index.json 审计为准，本表仅作量级输入。

## 计算算子清单与 NPU 映射

| # | 计算类 | 上游实现 | NPU 实现方案（Kernel 设计详述） | 风险 |
| --- | --- | --- | --- | --- |
| 1 | 因果注意力（GQA 16:4、head_dim 128、QK-Norm、动态 RoPE） | `flash_attn==2.8.3` | torch_npu 融合注意力（`npu_fusion_attention`，原生支持 GQA/因果掩码/BF16）或 torch SDPA 兜底 | 数值差异以 logits 对拍把关 |
| 2 | RMSNorm | 自定义/liger-kernel 替换层 | 关闭 liger，回落 transformers 原生 RMSNorm（eps 1e-5） | 无自定义 kernel 依赖 |
| 3 | SwiGLU MLP | silu 门控（torch 原生） | torch 原生（NPU 自动下沉） | 低 |
| 4 | QKV/投影/o_proj/lm_head GEMM | `torch.nn.Linear`（BF16） | torch 原生 → NPU GEMM | tied embedding 读路径审计 |
| 5 | RoPE（dynamic scaling） | transformers 原生 | 原生实现跟随（位置计算 Host、旋转在 NPU） | mscale 长序列行为与上游一致 |
| 6 | 视觉 patch embedding / 多分辨率位置编码 | processor + 视觉塔 | 随 `HunYuanVLMoT` 原生实现运行，仅 device/dtype 适配 | 分辨率上限按 processor 冻结 |
| 7 | 采样（temperature/top-p/greedy） | torch 多项采样 | torch 原生（logits fp32 化后采样） | 固定 seed 可复现 |
| 8 | KV Cache 读写 | transformers `use_cache=true` | 原生 cache 张量驻留 NPU | 显存配额见 Host/Tiling 设计 |

## 形状与显存分析（单卡资源规划输入）

| 项 | 估算 | 说明 |
| --- | --- | --- |
| 权重（BF16） | 4B × 2 B ≈ **8 GB**（全部模态分支驻留） | 总参含非激活分支，不拆分不 offload |
| KV Cache | 2(K+V) × 32 层 × 4 KV 头 × 128 head_dim × 2 B = **64 KiB/token** | 3 万 token ≈ 1.9 GB；thinking 上限 32768 新 token |
| 典型 EmbSpatial 序列 | 图像 token（数百至 ~2k，按 processor）+ 问题 + 输出 ≤ 数千 token | 稳定态 KV 数百 MB 量级 |
| 激活（BF16） | prefill 峰值按最长图像序列核算，GB 内量级 | RFC 锁定卡型后按 profiling 实测回填 |
| 峰值合计 | 权重 8 GB + KV/激活 ≤ 4 GB，**< 16 GB**（上游最低显存口径），A2/A3 满足 | OOM 防线见稳定性设计 |

## 无 CPU 静默回退核查点（任务书 §2.2-2 点名五处）

| 核查点 | 机制 |
| --- | --- |
| MoT 路由 | 加载后逐模块 device 审计：全部参数张量 on NPU，否则 fail-fast |
| 视觉编码器 | 首次前向 hook 校验输入/输出 device 与 dtype |
| 语言主干 | 同上 + 显式禁用 CPU offload 类配置 |
| KV Cache | decode 循环中 cache 张量 device 抽查 |
| lm_head（tied） | 权重共享路径 device 一致性审计 |
| 全局 | 进程内 assert 无 `cuda` 设备引用残留（任务书 §3「无 CUDA 残留」） |

# Host/Tiling 设计

## Host 侧推理流水线

| 阶段 | 组件 | 职责 | 关键契约 |
| --- | --- | --- | --- |
| 1 输入准备 | 数据加载器 | 读官方 JSONL（base64 解码；仅取题目侧字段），按 question_id 固定顺序排序 | 禁止 `answer`/`objects` 入上下文（单测把守） |
| 2 预处理 | `HunYuanVLMoTProcessor`（上游原生） | resize/normalize/tokenize/chat template | processor 版本冻结；图像 token 数逐题落盘 |
| 3 预填充 | `HyEmbodiedPipeline`（或 patch 版） | 一次性 prefill 建 KV，记录首 token 延迟 | prompt 长度、prefill 耗时逐题记录 |
| 4 解码循环 | GenerationConfig | 逐 token 生成（RFC 锁定 temperature/top-p/max_new_tokens/seed） | 停止条件 = 上游默认 eos 或长度上限 |
| 5 后处理 | 解析器 | thinking 模式提取 `<answer>` 字段；direct 模式取全文本；推理文本不进入解析器（任务书 §2.2-3） | 解析规则 RFC 冻结 |
| 6 计分 | 独立 scorer（与生成解耦） | 仅接受 RFC 约定的 A/B/C/D 或候选索引；不可解析计错 | 输入=逐题 JSONL 原始输出，可独立复算 |
| 7 落盘 | 结果写入器 | 逐题 JSONL + 汇总 JSON/CSV + 断点索引 | 字段表见评测设计 |

## 资源分块规划（Tiling 在 LLM 推理中的映射）

| 资源 | 规划 | 依据 |
| --- | --- | --- |
| 批大小 | 评测批 = 1（逐题顺序，保证协议一致性与断点语义）；联调 smoke 可并行批 | 任务书 §2.3-1 固定顺序要求 |
| KV Cache 配额 | 最长序列（图像 token + prompt + 32768 新 token）× 64 KiB/token 预留；超限样本计错并记录 | 形状与显存分析 |
| max_new_tokens | RFC 锁定（上游默认 32768；EmbSpatial 答案短，可按 RFC 收敛省墙钟，收敛值须先确认不影响协议有效性） | 上游 README 默认值 |
| 精度切分 | 主干线性/注意力 BF16；logits 归一化与采样 fp32 | 数值一致性设计 |
| Host 内存 | base64 解码后即时释放；逐题结果流式写盘不全量驻留 | 长跑无内存增长（任务书 §2.4） |
| 断点粒度 | 单题（question_id 级）；续跑跳过已完成 id 且不改分母 | 任务书 §2.4-3 |
| stream | 评测路径单 stream 顺序执行（与上游一致）；随机数 CPU/NPU 两侧统一 seed 注入 | 任务书 §2.2-2 |

# Kernel 设计

## attention kernel 选型

| 方案 | 适配性 | 采用策略 |
| --- | --- | --- |
| torch_npu `npu_fusion_attention` | 原生支持 GQA、因果掩码、BF16；RoPE 由 transformers 原生逻辑在 attention 外前置完成 | **首选**；单样本探针 + 精度对拍通过后锁定 |
| torch `scaled_dot_product_attention` | 语义通用，速度低于融合算子 | 兜底与对拍基准（与 GPU golden 同源逻辑） |
| 上游 flash_attn | CUDA 专属，不移植 | 替换为上两行，替换点集中在 pipeline patch |

GQA 4:1、head_dim 128 属 torch_npu 融合注意力常规支持面；RFC 阶段以 1 样本探针确认目标 CANN 版本无已知限制，异常即切兜底方案并在报告标注，不影响评测协议。

## 算子替换与 patch 矩阵（全部为可维护 patch，不复制完整上游代码）

| # | 上游依赖 | 替换方式 | 输入/输出契约（不变量） | 验证 |
| --- | --- | --- | --- | --- |
| P1 | `flash_attn` 调用点 | 运行时替换为 torch_npu 融合注意力 / SDPA | (q,k,v,causal) → (hidden)，形状/dtype/因果语义不变 | 单样本 logits 余弦 + 文本对拍 |
| P2 | `liger-kernel`（RMSNorm 等替换层） | 关闭 liger，回落 transformers 原生实现 | 数学等价（eps/精度模式一致） | 抽样层输出对拍 |
| P3 | `torch.autocast("cuda")` | 改 `torch.autocast("npu", dtype=bfloat16)` | dtype 传播行为一致 | 全链路 dtype 审计 |
| P4 | device 语义（`cuda` 字符串/`.cuda()`） | 统一设备抽象（npu）+ 全局禁用 CUDA 残留 assert | 模型/输入/cache 全 NPU | 无回退核查五点 |
| P5 | 分布式逻辑（deepspeed/accelerate 分支） | 评测路径单卡，关闭分布式分支 | 单进程语义不变 | 代码审查 + smoke |
| P6 | 随机数（torch/numpy/random） | 统一 seed 注入（RFC 锁定值） | 同 seed 同输出 | 复跑一致性用例 |
| P7 | 依赖版本校验 | import 时检查 transformers commit 区间/torch_npu 版本，不符快速报错 | 阻断不兼容环境 | 环境自检脚本 |

patch 统一以「运行时替换 + 少量包装文件」实现（参照仓内 `manipulation/Isaac-GR00T/adaptor_patches/` 先例），上游代码零拷贝；每个 patch 文件头注明上游锚点 commit 与失效条件（transformers 升级时复核）。

## 数值一致性设计

- 主路径 BF16（RFC 锁定）；logits→softmax/采样前 fp32 化；likelihood 模式（如获准采用）逐选项 logprob 用 fp32 累计。
- 对拍三级：① 权重加载审计（逐张量 name/shape/dtype 对照官方 index，NaN/Inf=0）；② 固定 10 样本集 NPU vs 上游 GPU（文本一致率 + logits 余弦，门槛 RFC 锁定）；③ 全量评测 Accuracy 对照（门槛 = 任务书 80% + 与官方基线差距分析）。
- QK-Norm/RMSNorm 的 eps 与计算精度随 transformers 原生实现保持，不自写等价物，避免引入数值分叉源。

## 回退与降级矩阵

| 失效场景 | 降级路径 | 状态 |
| --- | --- | --- |
| 融合注意力探针失败 | SDPA 兜底 | 自动切换并在报告标注 |
| BF16 数值不达标 | logits/norm 层定点升级 fp32 | RFC 确认后启用 |
| 依赖版本漂移 | import 快速报错，禁止自动升级 | 硬失败（任务书 §2.2-5） |
| OOM | 降图像分辨率上限 → 降 max_new_tokens → 该题计错并记录；禁止静默重试到正确 | 任务书 §2.4-3 |

# 硬件与约束

| 项 | 口径 |
| --- | --- |
| 硬件 | Atlas A2 或 A3 单卡（RFC 定稿其一）；本任务验收面以图像输入为主，视频输入能力不进验收矩阵 |
| 版本组合 | CANN / torch_npu / torch / Python 以 RFC 锁定表为准（对齐目标仓既有样例支持矩阵）；transformers 锚定原生注册 HunYuanVLMoT 的 commit 区间（`9293856` 起） |
| 显存目标 | 权重 8 GB + KV/激活 ≤ 4 GB；RFC 锁定峰值上限并按实测回填 |
| 数据与 License | 权重/数据集/缓存图片/base64 大对象/服务凭据不入库；路径占位符化；权重 Apache-2.0、数据 License 于 RFC 登记 |
| 禁止项 | 禁用 SFT split/答案字段构造上下文、禁重试到正确/人工修正/删分母、禁自动升级未验证依赖、禁混报不同协议结果、禁择优报告（任务书 §2、§7 全条目） |

# 详细设计（required）

## 工程组织（任务书 §4 指定位置 + CONTRIBUTION 命名规范）

```text
manipulation/hy-embodied-0.5-x/
└── infer_with_torch/
    ├── README.md / README_en.md        # 模型/平台/权重 License/环境/安装/评测命令/结果/性能/限制
    ├── requirements.txt                # NPU 必需依赖（与上游对照依赖分组）
    ├── configs/
    │   ├── infer.yaml                  # 权重路径(占位)/设备/dtype/max_new_tokens/temperature/top-p/seed/thinking
    │   └── eval_embspatial.yaml        # 数据路径(占位)/dataset revision/split/thinking/解码参数/断点目录/输出目录
    ├── hy_embodied_npu/
    │   ├── pipeline_patch.py           # P1~P6 运行时替换（上游锚点 commit 注明）
    │   ├── device_audit.py             # 五点无回退审计 + 无 CUDA 残留 assert
    │   └── compat.py                   # 依赖版本探针（P7 快速报错）
    ├── infer.py                        # 单样本推理入口（本地图像 + prompt，批/单）
    ├── eval_embspatial.py              # 全量评测（固定顺序/断点续跑/逐题 JSONL）
    ├── score_embspatial.py             # 独立复算脚本（输入=逐题 JSONL，与生成解耦）
    ├── tests/                          # 任务书 §4-4 测试矩阵（见可维可测）
    └── scripts/prepare_data.py         # 官方数据获取引导（数据本体不入库）
```

公共能力（如评测 JSONL 读写器）默认留在样例目录内；仅当出现明确第二消费者时按任务书 §4-2 上沉 `module/` 并回归全部直接调用样例（当前仓内无第二消费者，避免无谓公共面扩散）。

## 推理链路设计（复用上游契约）

优先原样使用上游 `HyEmbodiedPipeline`（from_pretrained / generate / GenerationConfig 契约见背景表），仅在 device/dtype/attention/liger 注入点应用 P1~P6 patch；`enable_thinking` 语义与 `<think>/<answer>` 结构遵循上游格式定义，评测只依据最终答案字段；processor/chat template、图像尺寸语义与回答格式按上游冻结处理（EmbSpatial 为四选一选项格式，上游坐标输出格式不进本任务验收面，仅在 README 限制节说明）。可选 OpenAI-compatible server（`hy_embodied.cli.server`）作为增强交付，不进验收关键路径。

## 评测设计（协议冻结表）

| 项 | 冻结值（RFC 定稿） |
| --- | --- |
| 数据 | 官方 human-verified evaluation split 全量；按 question_id 固定顺序；问题、四候选项、图像一一对应 |
| 评测方式 | generation + thinking（任务书 §2.1-3 默认建议）；换 likelihood 须维护者确认，不混报 |
| prompt/选项 | 官方 `question` + `answer_options` 按官方模板拼装；标注字段（answer/objects）不进上下文 |
| 解码 | temperature 0.05 / top-p / max_new_tokens / seed（上游默认或 RFC 收敛值） |
| 答案解析 | 仅接受 RFC 约定的 A/B/C/D 或候选索引；不可解析计错，不重试/人工修正/删分母 |
| 指标 | Accuracy = 正确回答数 / 全部评测样本数 × 100%（样本级）；六类 relation 分别报正确数/总数/Accuracy，不用宏平均替代 |
| 逐题留痕 | question_id、relation、原始输出、解析答案、ground truth、是否正确、输入/输出 token 数、耗时 |
| 断点续跑 | question_id 级跳过已完成项；异常/OOM/超时样本计错并列报告；仅完整重跑同一协议才覆盖结果 |
| 多次运行 | 每次完整结果与汇总全部报告，不择优（任务书 §2.3-5） |

## 稳定性设计

- **Smoke 矩阵（≥10 固定样本，任务书 §2.4-1）**：六类 relation 各 ≥1、thinking/direct 各覆盖、异常图片（损坏/超分辨率）与超长输出用例；断言无 NaN/Inf、乱码、空答案、跨样本状态污染。
- **长跑监测**：完整评测全程记录 NPU/Host 内存曲线；持续增长趋势即告警定位。
- **异常计错**：OOM/超时/解析失败样本计错并列入报告（附原始输出），除非完整重跑同一固定协议。

## 性能基线（无强制收益门槛）

首 token 延迟、输出 token 吞吐、端到端耗时、NPU/Host 峰值内存四项逐题记录并汇总（P50/P95），作为后续优化基线随实验记录交付（任务书 §2.4-2）。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 数值 | 标准来源 |
| --- | --- | --- |
| EmbSpatial 总 Accuracy | ≥ 80.0%（官方样本级公式，逐题可复算） | 任务书 §3 精度指标 |
| 分类统计 | 六类 relation 正确数/总数/Accuracy 全公开 | 任务书 §3 |
| smoke | ≥10 固定样本全通过（格式/无污染） | 任务书 §2.4-1 |
| 单样本对拍 | NPU vs 上游 GPU：文本一致率 + logits 余弦，门槛 RFC 锁定 | 本设计数值一致性 |
| 性能 | 四项基线指标无门槛、全量记录 | 任务书 §2.4-2 |
| 精度类型 | BF16 主路径（FP16 对照为可选实验项） | RFC |

## 功能用例（任务书 §4-4 矩阵）

| 用例 | 断言 |
| --- | --- |
| 模型加载 | 审计式加载：张量清单零缺失、NaN/Inf=0、五点 device 审计通过 |
| 单图推理 | 本地图像+prompt 出合法文本；批/单入口等价 |
| thinking / direct | `<think>/<answer>` 结构解析正确；direct 无 think 段 |
| 答案解析四选项 | A/B/C/D、候选索引、带冗余文本仅取选项、全不合法计错 |
| 不可解析输出 | 计错且原始输出留痕 |
| 重复 question_id | 续跑不重复计分、分母不变 |
| 缺图/坏图 | fail-fast 报错并计错记录 |
| OOM/超时 | 计错 + 上报 + 继续后续样本 |
| 真实数据 smoke | 官方数据一小批端到端跑通 |
| 公共回归 | 若发生公共改动，全部直接调用样例回归通过 |

## 兼容性分析

| 维度 | 分析 |
| --- | --- |
| transformers | 锚定原生注册 `HunYuanVLMoT` 的 commit 区间；升级触发 P1/P2 patch 复核 |
| torch_npu / CANN | 与目标仓既有样例支持矩阵对齐，RFC 锁定；融合注意力探针用例先行 |
| 与仓内样例共存 | 全部改动限定在 `manipulation/hy-embodied-0.5-x/` 新目录，零公共文件改动则无回归面；一旦上沉 `module/` 即触发全样例回归 |
| 依赖隔离 | NPU 环境不安装 flash_attn/liger；requirements 分「NPU 必需」与「上游对照（GPU 复现用）」两组 |

## 交付与运行顺序

任务书 §4 八项交付件对照：①目录与配置（占位符化）②Python/patch 代码（P1~P7）③评测工具（eval/score 解耦 + 断点续跑 + 逐题 JSONL/汇总 JSON/CSV）④测试矩阵（上表）⑤README（中英）⑥优化过程文档（上游架构/依赖替换/NPU 适配点/精度定位/评测协议/各 relation 误差/优化前后实验）⑦实验记录（环境/提交号/权重与数据哈希/完整命令/全量结果/耗时内存）⑧回归与 PR（关联 RFC/Issue、pre-commit、CI、实机验收）。

运行顺序：RFC 冻结 → 环境安装（requirements + 版本探针）→ 权重/数据就位（占位符路径）→ smoke（10 样本）→ 全量评测 → 独立复算脚本核分 → 报告与实验记录归档。README 保证三条命令复现：环境安装、smoke、全量评测。

# 支持硬件

Atlas A2 / Atlas A3 单卡（RFC 锁定其一）；权重 8 GB + KV/激活峰值实测回填，满足上游 ≥16 GB 显存口径的卡型均在支持面内。

# 约束限制

权重、数据集、缓存图片、base64 大对象、完整输出大文件、服务凭据不入库；路径占位符化。评测协议（split/方式/thinking/prompt/预处理/解码/解析）验收中途不改参；不使用 EmbSpatial-SFT 或评测答案进行微调、提示泄漏或规则硬编码；不重试到正确、不人工修正、不从分母删除异常样本；依赖版本不符快速报错不自动升级；不同协议结果不混报、不择优。
