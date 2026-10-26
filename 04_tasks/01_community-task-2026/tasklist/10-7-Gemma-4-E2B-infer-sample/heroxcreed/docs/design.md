# Gemma-4-E2B 纯文本推理接入设计文档（Atlas A2）

| 项目 | 内容 |
| --- | --- |
| **社区任务** | 10月社区任务-Gemma-4-E2B 纯文本推理接入任务书（A2）（任务编号 `10-7`），[任务页](https://www.hiascend.com/activities/task-center/details/69423188360d481bab8669b7b9f817e6) |
| **贡献者** | `heroxcreed` |
| **设计状态** | 实现与 A2 实机自验已完成，本文数字均为实测；所有判定阈值均在对应全量运行之前登记，运行后未调整 |
| **关联 RFC Issue** | [cann/cann-recipes-infer #495](https://gitcode.com/cann/cann-recipes-infer/issues/495) |
| **目标代码仓** | [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)，基线 master `6d06031` |
| **官方对标模型** | [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B)，revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| **官方对标源码** | [transformers v5.5.0 `modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py) |
| **目标硬件与软件** | Atlas A2（Ascend 910B3）单卡 1TP；CANN 9.0.0、PyTorch 2.8.0、torch_npu 2.8.0.post4、Python 3.12.13 |

---

# 需求背景（required）

## 需求来源

HiAscend 10 月社区任务「Gemma-4-E2B 推理样例（A2）」：在 `cann/cann-recipes-infer` 统一执行器中，将 Gemma-4-E2B 作为独立新模型接入纯文本 BF16 Offline 推理，并提供可复现的配置、精度与性能基线。

## 背景介绍

Gemma-4-E2B 是 Google Gemma-4 家族的小尺寸多模态模型（权重含 vision tower 与 audio tower）。本样例只接入其文本解码器，与仓内 `gemma_4`（26B-A4B MoE）样例完全独立：独立目录 `models/gemma_4_e2b/`、独立注册名 `gemma_4_e2b`，不复用其模型实现、配置、权重映射或测试基线。

E2B 文本解码器相对仓内已有样例的三项特有结构：
1. **逐层输入（per-layer inputs）**：每层额外接收一份 256 维逐层嵌入，经门控、投影、归一化后残差融合。
2. **跨层 KV 共享**：后 20 层（15～34）不计算自身 K/V，读取共享区之前同类型最后一层的 K/V。
3. **双宽 MLP**：共享区层 FFN 中间维度加倍（6144 → 12288）。

---

# 需求分析（required）

## 需求描述

**支持范围（首期）**：BF16、纯文本、Offline、Atlas A2 单卡 1TP；batch=1 与等长 batch（已验证 bs=4）；变长 batch 明确报错。
**不在范围**：视觉/音频输入、量化、MTP、Online/PD 分离、多卡并行、同批次变长 Prefill、极限长上下文。

## 基线锁定

### 代码与权重

| 项 | 值 |
|---|---|
| 仓库基线 | `cann/cann-recipes-infer` master `6d06031` |
| 官方权重 | `google/gemma-4-E2B`，HF revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`（Apache-2.0） |
| `model.safetensors` | SHA256 `76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57`，10,246,621,918 B |
| `tokenizer.json` | SHA256 `12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6` |
| 获取方式 | `huggingface-cli download google/gemma-4-E2B --revision <上述>`；ModelScope 镜像无法锁 revision，下载后须按 README 全表核对 SHA256（实测 6 个公共文件与 HF revision 全部一致） |
| 加载范围 | 语言参数 540 个张量全部加载；`vision_tower`/`audio_tower` 等 1411 个多模态张量按前缀跳过；60 个共享层 k/v 投影权重在 checkpoint 中存在但按官方语义不使用（见 §3.2） |

### 对标实现与参数映射

| 项 | 参考实现 | 本实现 |
|---|---|---|
| 版本 | Transformers 5.5.0，`AutoProcessor` + `AutoModelForMultimodalLM`（官方入口；模块级对齐脚本另用 `AutoModelForCausalLM` 加载同一权重） | executor 统一入口 |
| 运行环境 | 同机 NPU，torch 2.8.0 / torch_npu 2.8.0.post4，依赖锁定在 `requirements-hf.txt` | `requirements.txt` |
| 模板 | 基座模型官方无 chat template（`add_bos_token=true`）：纯文本 + BOS | 框架离线入口固定 `is_chat=True`，本样例在运行期为 tokenizer 挂载**直通模板**（渲染==原文），权重目录保持官方原样；CPU 单测证明 10/10 样本 token 与纯文本+BOS 逐位相同 |
| EOS | `{1}` | 框架合并 config / generation_config / tokenizer 后同为 `{1}` |
| 采样 | greedy（`do_sample=False`） | 框架默认 greedy，不读取 generation_config 的采样字段 |
| thinking | 基座模型无 thinking 模式，不适用 | 不适用 |
| seed | 1234 | 1234 |

### 测试配置

- YAML：`models/gemma_4_e2b/config/gemma_4_e2b_bf16_1tp.yaml`（eager）、`_graph.yaml`（`exe_mode: npugraph_ex`，Decode 图模式）、`_prof.yaml`（`enable_profiler: True`），三者仅此一处差异；`config/ci/` 下为 CI 登记用例。`model_path` 使用 `/data/models/gemma-4-E2B`（仓库惯例占位路径）。
- 输入容量 2048 token（`input_truncated_len`），超长截断（实测截断后 2047 token）；batch=1；最大输出 128。
- 固定样本：`tests/data/prompts.json` 10 条（中 4 / 英 6；问答 3、摘要/改写 3、推理 3、代码 1；含 1 条 560 token 长样本），参考输出 `tests/data/hf_golden.json`。
- 数据集：`openai/openai_humaneval` revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`，test 164 题；基座续写、greedy、max_new_tokens 384、固定截断 stop 列表、独立子进程执行（rlimit 隔离，10 s 超时）。

---

# 详细设计（required）

## 模型结构与算子分析

### 官方 `text_config` 结构规格（代码从配置读取，无硬编码尺寸）

| 项 | 值 |
|---|---|
| 层数 / hidden | 35 / 1536 |
| attention | 8 头，KV 1 头（GQA）；sliding 层 head_dim 256，full 层 head_dim 512 |
| layer_types | `S S S S F` × 7（4:1），sliding_window 512 |
| 逐层输入 | `hidden_size_per_layer_input=256`，`vocab_size_per_layer_input=262144` |
| KV 共享 | `num_kv_shared_layers=20`，首个消费层 15 |
| MLP | GELU(tanh) 门控；层 0–14 中间维 6144，层 15–34 为 12288（`use_double_wide_mlp`） |
| RoPE | sliding：θ=1e4；full：proportional，partial_rotary_factor 0.25，θ=1e6 |
| 输出 | tie embedding；final_logit_softcapping 30.0；每层 `layer_scalar` buffer |

### 逐层输入链路

专用 `embed_tokens_per_layer` 产出每 token 35×256 的逐层嵌入，与主隐藏状态经 `per_layer_model_projection` + RMSNorm 的投影组合（缩放因子取自官方实现）；第 i 层取本层切片，经 `per_layer_input_gate`、GELU 门控、`per_layer_projection` 与 RMSNorm 后残差融合，层输出再乘 `layer_scalar`。计算顺序、激活与缩放与 Transformers 5.5.0 逐项对齐，模块级对拍通过（§4.1）。

### 跨层 KV 共享映射与缓存契约

- **来源推导**：由 `layer_types` 与 `num_kv_shared_layers` 计算：首个消费层 = 35 − 20 = 15；sliding 消费层（16 层）→ 层 13，full 消费层（19/24/29/34）→ 层 14。不写死层号。
- **存储**：框架 `KVCacheManager` 维持 sliding 与 full 两个 block pool，**只由层 0–14 分配和写入**。消费层在 `get_cache_info` 中声明一条 `needs_block=False` 的元数据 entry（满足 `validate_cache_info` "每层至少一条 entry"的契约，不分配显存；该 entry 携带与来源层同组一致的合法 `block_size`，以通过 `calculate_block_num()` 分页循环的组内 block_size 一致性检查），forward 时用自身 Q 读取来源层的 K/V 与 block table。
- **写入责任与生命周期**：只有来源层写 KV，消费层只读，杜绝重复写入；block 随请求由框架统一回收，共享层不持有独立生命周期，因此不会提前回收。实测 20 个连续请求后两池均回到空闲态（仅 null block），无泄漏、无跨请求污染（§4.3）。
- **内存收益**：KV 显存只按 15 层计，比不共享节省 20/35。

### Attention 与算子可用性

- 复用仓内 torch_npu 融合算子（FA、RoPE、RMSNorm），**未新增 kernel 或 aclnn 工程**。
- sliding 层 Prefill 用 TND band；full 层 head_dim=512，开发期 TND 实现在 warm-up 报 EZ9999，因此 full 层 Prefill 走 BNSD（等长 batch 批量计算）。
- **算子语义注意点**：band 模式下 `pre_tokens=512` 实际可见 513 个 key。我们用扰动锥探针（只改窗口外 token，检查第 0 层输出是否逐比特不变）发现并修正为 `pre_tokens = sliding_window − 1`；Prefill 与 Decode（含 block 回收后）窗口均与 HF 的 512 逐比特一致，511/512/513 与 1023/1024/1025 边界已验证。

### 双宽 MLP、权重加载审计与 softcap

按 `use_double_wide_mlp and is_kv_shared_layer` 选择中间维；共享 embedding/lm_head；`final_logit_softcapping` 在 lm_head 后执行 `30·tanh(x/30)`。权重审计：必需持久参数缺失 0、shape/dtype 不符 0，抽检 88 个张量与 checkpoint 逐比特一致；19 个 NaN 初值全部是运行期计算的非持久 buffer。

### 输入校验与异常行为

模型侧在加载权重前拒绝：26B-A4B 结构 config、`input_truncated_len` 越界、`max_new_tokens`/`batch_size` < 1、缺 safetensors 等（10 个非法用例全部非零退出且报错可读）。Prefill 入口检查 `actual_seq_lengths_q`，变长 batch 抛出明确的 `ValueError`（bs=1 不触发设备同步，Decode 图不受影响）。

## 模型实现方案

- 交付结构：`models/gemma_4_e2b/{models,config,tests,docs}`、`requirements.txt`、`requirements-hf.txt`、`README.md`；仓库其他位置仅 `executor/core/support_models.py` 一处纯新增注册。
- 测试入口：`bash models/gemma_4_e2b/tests/run_all.sh`（模块对齐、非法输入、全 logit、eager/graph、权重审计、20 连发、batch 形态、内存四时点，共 8 项，失败返回非零），另有 3 组 CPU 单测（无需 NPU）。
- pre-commit：本分支改动文件全部通过。`pre-commit run --all-files` 在干净的上游 `6d06031` 上即会修改 489 个既有文件并有 3 个 hook 失败，与本改动无关（证据随 PR 附上）。
- CI：已在 `config/ci/` 登记，`executor/scripts/test_all_case.sh` 本地跑通。附带发现：该脚本不检查子进程返回码，CANN 路径为占位符时 torch_npu 加载失败仍会打印 "all case pass"，建议另行修复。
- 大日志与 profiler 产物不入 Git，以获取地址与校验信息在验证报告中索引。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（Ascend 910B3 单卡 1TP，已实测） | √ |

| 项 | 值 |
|---|---|
| 硬件 | Atlas A2，Ascend 910B3 单卡（容器内逻辑卡号 3），64 GB HBM；Host aarch64、256 核、1.9 TiB 内存 |
| OS | Ubuntu 22.04.5（宿主内核 5.10，openEuler 22.03 SP4） |
| 驱动 / 固件 | 26.1.1 / 9.0.0.9.220 |
| CANN | 9.0.0（V100R001C10SPC001B250） |
| Python / torch / torch_npu | 3.12.13 / 2.8.0 / 2.8.0.post4 |
| transformers | 5.5.0 |

未在其他型号上实测，不声明支持。

## 约束与限制说明

1. 首期仅支持纯文本、BF16、Offline、单卡 1TP；视觉与音频输入不在范围。
2. 支持 batch=1 与等长 batch；不等长变长 Prefill 在模型入口检查 `actual_seq_lengths_q` 并抛出可读 `ValueError`。
3. 输入容量受 `data_config.input_truncated_len`（2048）约束，超长截断。
4. 已知问题：npugraph_ex 模式下 Prefill（不入图）比 eager 慢 16%–79%，需要低 Prefill 时延时用 eager。

---

# 可维可测分析

## 判定标准（阈值先于运行登记）

| 层级 | 度量 | 阈值 | 登记 |
|---|---|---|---|
| 单模块（21 个模块 + 组合输入边界，随机张量 BF16，seed 固定） | cos / relL2；norm·softcap | cos ≥ 0.9999、relL2 ≤ 1e-2；≤ 2 ulp | 先于运行 |
| 逐层（3 条真实 prompt × 35 层，含共享 KV 来源层 13/14） | cos / relL2 / 相邻层误差比 | ≥ 0.99 / ≤ 5e-2 / ≤ 3 | 先于运行 |
| 全 logit（teacher forcing 全位置） | 汇总 top-1 一致率；分歧处 HF top1−top2 gap；每条 max symKL | ≥ 99%；≤ 0.5；≤ 0.05 | 先于运行 |
| 生成 | input_ids、首 token | 10/10 | 先于运行 |
| 数据集 | HumanEval pass@1 | \|Δ\| ≤ 2 题 | RFC 约定 |
| eager vs npugraph_ex | 同条件生成文本 | 逐条一致 | — |
| 性能 | 预热 1 次 + 正式 3 次，报均值/中位数，两轮取范围；加载、编译、稳态分开统计 | 首次采集即基线 | — |
| 内存 | `npu-smi` HBM + `torch.npu` 统计 + Host RSS，四个时点 | 无 OOM、连续请求无增长 | — |

> 说明：99% 汇总阈值是在第一次小样本试跑之后、全量运行之前确定并提交的，此处如实披露。

## 自验结果（A2 实机）

### 精度

| 项 | 结果 |
|---|---|
| 单模块 / 组合边界 | 22/22 PASS（最差 cos 0.99999458，relL2 3.27e-3） |
| 逐层 35 层 × 3 prompt（含来源层 13/14 K/V） | 3/3 PASS |
| 10 样本 | input_ids 10/10、首 token 10/10、完整输出严格一致 7/10（按 EOS 归一后比较；原始逐 token 为 6/10）；3 条分叉处 HF top1−top2 gap 分别为 0.375、0.0、0.0 |
| teacher forcing 全位置 | 1158/1165 = 99.40%；分歧处最大 HF gap 0.125；max symKL 0.0345 → PASS |
| 全 logit 对拍 | prefill 8/8、decode 8/8，max symKL 0.00406 |
| HumanEval | NPU 42/164 = HF 42/164（pass@1 0.2561），逐题一致 160/164（互有 2 题差异，净 0） |
| eager vs npugraph_ex | 10/10 文本一致，图模式两次运行一致 |

### 性能（Atlas A2 910B3，bs=1，输出 128，开 profiler，两轮范围，单位 ms）

| 输入 | eager prefill | eager decode/token | graph prefill | graph decode/token |
|---|---|---|---|---|
| 128 | 47.94–49.04 | 45.85–46.29 | 75.03–82.95 | 16.49–16.59 |
| 512 | 47.08–48.56 | 45.12–45.69 | 73.98–84.47 | 16.50–16.81 |
| 1024 | 54.19–55.84 | 45.28–46.45 | 72.06–86.85 | 16.80–17.03 |
| 2048 | 74.84–75.52 | 45.20–46.86 | 87.35–89.13 | 16.78–16.91 |

- npugraph_ex Decode 为 eager 的 2.66–2.81 倍；24 组运行均无稳态重编译。
- **已知问题**：npugraph_ex 下 Prefill 比 eager 慢 16%–79%（Prefill 本身不入图），根因尚未定位，文档已声明，需要低 Prefill 时延时用 eager。
- 本仓无 E2B 历史基线，上表即首份基线。

### 内存与生命周期

- 20 个连续请求（长度 17–1700，覆盖滑窗边界，正序/逆序/重跑）：输出顺序无关、无 KV 泄漏、显存增长 0.0 MB，eager 与 graph 20/20 一致。
- 四时点 Host/NPU 内存（最大输入 2048 / 输出 128 / bs=1）：两模式均无 OOM，20 个请求均输出满 128 token。

  | 时点 | eager ΔHBM / allocated / Host RSS | npugraph_ex ΔHBM / allocated / Host RSS |
  |---|---|---|
  | 加载完成 | 20862※ / 9686 / 3541 MB | 10231 / 9687 / 3288 MB |
  | warm-up 编译后 | — | 10659 / 9687 / 3306 MB |
  | Prefill 峰值 | 11332 / 9722 / 3576 MB | 10659 / 9723 / 3306 MB |
  | Decode 峰值 | 10552 / 9686 / 3576 MB | 10659 / 9687 / 3306 MB |
  | 20 请求后 | 10552 / 9686 / 3576 MB | 10659 / 9687 / 3306 MB |

  ΔHBM 为 npu-smi 已用量减空卡基线。※ 该采样点落在权重装载临时缓冲释放前，下一时点即回落（瞬时 HBM 24.3 GB，远低于 64 GB）。稳态 ΔHBM：eager 10.6–11.1 GB，npugraph_ex 10.7 GB；`max_memory_allocated` 峰值 10.06 GB；20 请求后 allocated 增长 0.00 MB（阈值 16 MB 先于运行登记）。

## 兼容性与回归分析

1. **共享代码改动面**：对 `executor/` 仅 `executor/core/support_models.py` 一处纯新增 `gemma_4_e2b` 注册条目，不改动 `gemma_4` 注册项；**不修改公共 KV 缓存框架代码**（`executor/core/kv_cache/` 零改动），不建立模型私有调度框架。
2. **共享层占位 entry 的框架兼容性**：消费层声明的 `needs_block=False` entry 在张量分配、manager 创建、PD 传输等消费点均被既有框架逻辑跳过；`calculate_block_num()` 不检查 `needs_block`，因此该 entry 携带合法且组内一致的 `block_size`。其影响仅限在线模式按显存余量推算块数时的预算估算（偏保守）；离线模式块数由 `offline_max_len` 决定，不受影响。
3. **对 Gemma-4-26B-A4B 的影响**：由于不修改共享执行器与缓存逻辑，`gemma_4` 的模型、配置、注册与运行路径均无变化；任务书要求的"26B 受影响路径"为空。若维护者仍需确认，可在仓库 8 卡 CI 上运行一次 `gemma_4` 既有用例。
4. **工程门禁**：`bash models/gemma_4_e2b/tests/run_all.sh` 一键自验，任一步失败返回非零；交付文件 `pre-commit` 全部通过；已在 `config/ci/` 登记用例。权重、数据集、大日志与 profiler 产物不入 Git，以获取地址与 SHA256 在验证报告中索引。

---

# 开发里程碑与提交顺序

| 阶段 | 产物 | 状态 |
| --- | --- | --- |
| 1. 设计评审 | 本文 + RFC Issue #495 | 评审中 |
| 2. 实现 | `models/gemma_4_e2b/` 全部交付件 + 独立注册 | 已完成 |
| 3. A2 实机验证 | 模块/逐层/全 logit/10 样本/HumanEval/图模式/性能矩阵/内存/20 连发/边界 | 已完成（去除公共缓存改动后的回归复测随交付件提交） |
| 4. 提交验收 | task_submission + 自验证报告 | 设计评审通过后提交 |
| 5. 代码合入 | 向 `cann/cann-recipes-infer` 提交实现 PR，关联本设计与 #495 | 验收测试通过后提交 |

本 PR 只提交 `design.md`，不包含实现代码、权重、数据集或实机日志。

# 需维护者确认的事项

1. 判定阈值与 HumanEval 容差 |Δ| ≤ 2 题。
2. 性能首份基线采用上文口径（开 profiler、预热 1 + 正式 3、两轮范围）。
3. 不修改共享缓存代码、"26B 受影响路径为空"的回归结论是否认可。

# 参考

- 官方模型：https://huggingface.co/google/gemma-4-E2B
- 官方实现：https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py
- 仓内：`docs/common/new_model_checklist.md`、`docs/design/kv_cache_design.md`、`CONTRIBUTION.md`
