# 需求背景（required）

## 需求来源

本设计对应「10月社区任务-daVinci-MagiHuman FA MXFP8 量化任务书（A5）」及配套 `test_cases/README.md`，目标代码仓为 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)、[社区任务目录说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)与目标仓 `CONTRIBUTION.md`。

提交团队：`ji096929`。本设计文档由 AI 辅助整理（辅助范围：任务书核对、现状代码分析、实测数据归集与文档结构化；全部实机数据来自团队在 Ascend 950PR 上的自主运行记录）。关联 RFC/Issue：按任务书要求在 `cann/cann-recipes-infer` 创建并回填链接。设计 PR：[本 PR]。里程碑约定：本设计评审通过并冻结后，按锁定环境重采正式基线与优化数据，再提交代码 PR。

## 背景介绍

daVinci-MagiHuman 是联合生成视频与音频的多模态 DiT 模型（40 层 Transformer，hidden 5120，head_dim 128）。固定上游配置下，Q 为 40 heads，K/V 为 8 heads，属 GQA（40:8，分组比 5）。base 文生视频模式的 `local_attn_layers` 为空，全部层走标准稠密 Attention；`SR2_1080` 模式经 `local_attn_layers` 启用 flex 局部注意力（帧间局部窗口 + 视频/音文交叉区间，按 softmax LSE 合并多 K 区间结果）。

模型现有 `flash_attn_func`（BSND 标准路径）与 `flex_flash_attn_func`（BNSD 段级路径）均使用 BF16 `npu_fused_infer_attention_score`。仓内公共模块 `module/fa_quant/` 已提供 Q/K/V 动态 MXFP8 量化（`npu_mxfp8_quant`，封装 `npu_dynamic_mx_quant`）与 MXFP8 FA 封装（`npu_mxfp8_attn`，封装 `npu_fused_infer_attention_score_v2`），HunyuanVideo 已有 `fa-mxfp8` 接入样例。现有封装的直接调用方为 HunyuanVideo Attention 与 `module/unified_sp/core.py`，实施前将再次检索并冻结影响清单。

本任务仅量化 Attention 的 Q/K/V 激活，复用现有 CANN 算子；MatMul 权重量化、RoPE 融合、缓存算法与图模式收益不计入本任务。

## 实机环境确认（Ascend 950）

已完成目标软件组合实机确认与算子能力探针（详见「可维可测分析」）：

- 芯片：Ascend 950PR（Ascend950PR_9579 V100），单卡，HBM 131072 MB；
- 驱动 25.7.rc1.10；CANN 9.1.0（toolkit + 950 ops）；
- 锁定软件组合：Python 3.12 + PyTorch 2.10.0+cpu + torch_npu 2.10.0.post4（官方参考组合 `v26.0.0-pytorch2.10.0` 版本线）；
- `npu_dynamic_mx_quant` 与 MXFP8 `npu_fused_infer_attention_score_v2`（含 `return_softmax_lse=True`）三项能力探针实机全通过。

# 需求分析（required）

## 需求描述

在 Ascend 950 单卡上，使 daVinci-MagiHuman 的标准 Attention 与 flex 局部 Attention 路径真实执行 Q/K/V 动态 MXFP8 量化 + MXFP8 FA v2 计算；输出保持原 dtype/shape/GQA/局部区间语义；未支持组合明确回退或报错；通过中间张量比较、视频盲测、端到端性能与峰值内存验证后提交代码 PR。

## 需求拆解

| 工作 | 产物与通过条件 |
| --- | --- |
| 基线与 RFC | 固定源码、权重、完整配置、输入与随机种子；950 实机运行 BF16 base 基线与 profiling |
| 能力验证 | 实机验证真实 GQA 40/8/D128 的动态量化、scale、FA v2（含 LSE 与输出 dtype 恢复） |
| 公共接口 | 复用 `npu_mxfp8_attn`；扩展 LSE 返回能力时保持旧签名/旧行为兼容，新增参数 keyword-only |
| 模型接入 | 开关与回退贯通 YAML→CLI→模型 Attention；标准与 flex 双路径接入；保留 flex 区间与 LSE 合并语义 |
| 精度 | Q/K/V、scale、FA 输出、DiT block 输出达到预设门限；5 组固定样本盲测无稳定可辨识退化 |
| 性能与内存 | 每档 ≥1 次预热、≥3 次正式生成；FA 累计耗时、计入量化/layout 的 Attention 累计耗时、端到端耗时下降；峰值内存记录并解释 |
| 兼容与交付 | 公共模块全部直接调用方回归、HunyuanVideo 代表配置回归、pre-commit、仓库 CI、950 实机验证、文档与自验材料 |

# 详细设计（required）

## 算子分析

### 数学公式

对 query head \(h\)，对应 KV head 为 \(g(h)=\lfloor h/(H_q/H_{kv})\rfloor\)，原计算为：

\[
O_h=\operatorname{softmax}\left(Q_h K_{g(h)}^T / \sqrt{D} + M\right)V_{g(h)}.
\]

MXFP8 路径中，Q/K 按最后一维、V 按 token 维执行 `npu_dynamic_mx_quant`，分别得到 FP8 值与 E8M0 scale，scale 直接传入 `npu_fused_infer_attention_score_v2`（TND、quant mode 6/6/8、非 sparse）。运行路径不执行「反量化后调用 BF16 FA」。

flex 路径同一 Q 区间存在多个 K 区间时，按各区间 softmax LSE 合并：

\[
L=\operatorname{logsumexp}(L_1,\ldots,L_m),\qquad O=\sum_i \exp(L_i-L)O_i .
\]

MXFP8 路径要求每段 FA v2 调用可返回满足上述合并语义的 LSE。本设计的实现方案为：**flex 的每个段（单 K 区间）独立执行 Q/K/V 动态量化 + MXFP8 FA v2，并取 `return_softmax_lse=True` 的 LSE 参与合并**——即多 K 区间同样走 MXFP8，不以 BF16 回退。该方案已通过实机能力探针（见下），不依赖未经证实的算子行为。

### 支持数据类型

| 张量 | 设计契约 |
| --- | --- |
| 量化前 Q/K/V | 目标模型 Attention 当前使用 BF16；三者同 NPU、维度与长度合法、contiguous 化后量化 |
| 量化后 Q/K/V | `torch.float8_e4m3fn`，由 `npu_dynamic_mx_quant` 动态生成 |
| Q/K/V scale | 算子动态返回，FA 侧使用 `torch_npu.float8_e8m0fnu` 对应类型；禁止全 1 或伪造值 |
| FA 输出 | 与输入一致的 BF16；外层按原 dtype 恢复，行为与现 BF16 路径一致 |
| LSE | `return_softmax_lse=True` 返回，fp32；标准路径不使用，flex 路径用于区间合并 |

### 支持形状

标准路径输入为 Q `[B,Sq,40,128]`、K/V `[B,Skv,8,128]`（BSND），单卡 base 模式 B=1。BSND→TND 保持 batch 边界；`actual_seq_qlen`/`actual_seq_kvlen` 使用各样本真实累计长度；GQA 的 K/V head 数按 8 真实传入，不做 head 展开复制。

flex 路径输入为 B=1 的 BSND 或 TND 序列，由现有 Q/K range 切出真实子区间；段内以 singleton batch 包装为 BSND 后量化与调用。base 模式固定配置下，序列由视频 tokens（约 3.5k）、音频 tokens（126）与有效文本 tokens 组成（精确值以实机运行时记录为准并随验收报告提交）。空区间、非法 head 比、非连续输入、长度不齐等情况在进入 NPU 算子前检查或显式转换并计入成本。

## 算子实现

### host 侧设计

1. **配置与开关**：新增 `config/fa_mxfp8.yaml`（复制 `base.yaml` 全字段，增加 `model_args: fa-mxfp8: true`、`fa-mxfp8-fallback: true`）；`infer.sh` 增加 `fa_mxfp8` 模式入口；CLI 参数经 `entry.py` 解析后写入 `dit_module` 的模块级开关（默认关闭，不影响既有模式）。
2. **标准路径接入**：`flash_attn_func` 在开关开启且输入满足契约（4D BSND、head_dim 32 倍数、bf16）时，调用公共 `npu_mxfp8_attn`（BSND 输入、TND 计算、GQA 参数、真实长度、`1/sqrt(D)`），失败或契约不符时按回退开关走原 BF16 分支并记录审计日志。
3. **flex 路径接入**：`flex_flash_attn_func` 保持原「Q 区间切分 + 逐 K 区间计算 + LSE 合并」结构；MXFP8 分支对每段执行量化 + FA v2（`return_softmax_lse=True`），LSE 归一化为 `(N, S)` 后并入原合并逻辑（与原 BNSD v1 LSE 的合并路径保持同一数值语义）。
4. **公共接口扩展（向后兼容）**：`npu_mxfp8_attn` 新增 keyword-only 参数 `return_lse=False`；默认返回与现行为完全一致（单 tensor、原 shape/dtype）；`return_lse=True` 时额外返回 `(B,N,S)` fp32 LSE。既有调用方（HunyuanVideo、`unified_sp`）无需修改，签名与默认行为零变化。
5. **回退与错误策略**：CP world size > 1、非 bf16、head_dim 非 32 倍数等其他未覆盖组合：`fa-mxfp8-fallback` 开启时走原 BF16 并输出可审计回退日志（含原因与唯一性去重）；关闭时明确报错。参数类非法输入直接报错，不当作能力回退吞掉；设备/OOM 类异常保留原始上下文抛出。
6. **审计日志**：记录模式、分支（标准/flex/回退）、shape、heads、真实长度、dtype、scale shape、回退次数；正式性能运行不逐调用打印、不引入额外同步。

### kernel 侧设计

不新建 kernel 工程，不修改 CANN tiling。使用现有 `npu_dynamic_mx_quant` 与 `npu_fused_infer_attention_score_v2`，quant mode 沿用已验证公共实现的 6/6/8 契约。验收 profiling 必须同时出现三路动态量化与 MXFP8 FA v2，并结合调用参数、tensor dtype/scale shape 证据交叉确认（不以同名算子出现为唯一证据）。

## 支持硬件

| 项目 | 已确认条件 |
| --- | --- |
| 芯片 | Ascend 950PR_9579 V100，单卡，HBM 131072 MB |
| Host | x86_64 容器环境（Ubuntu 24.04），Host 内存 754 GiB |
| 驱动 | 25.7.rc1.10（npu-smi 记录随验收材料提交） |
| CANN | 9.1.0（toolkit + 950 ops，x86_64） |
| 锁定软件组合 | Python 3.12、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post4（对应官方 `v26.0.0-pytorch2.10.0` 版本线） |
| MagiCompiler | 上游源码本地构建（1.0.0），模型依赖 API 导入通过 |

## 算子约束限制

- 验收固定单卡 base T2V：448×256、5 秒、32 steps、CFG=2、seed=42、`dit_cache.method: NoCache`、固定 prompt；A2/A3 数据不作为 950 基线。
- 标准路径 case 为默认 base；flex 路径 case 按 RFC 指定配置触发（base 权重 + 独立测试配置显式 `local_attn_layers`，保持全部生成规格），并以同配置 BF16 对照；case 的触发与调用计数以运行日志留证。
- CP>1 的分布式数值与性能不在单卡验收覆盖内：默认回退 BF16 并记录原因，不宣称支持。
- 未覆盖的 sparse/layout/dtype 组合保持显式回退清单；构造输入的接口测试与真实模型 case 分开报告。
- 不删除 BF16 回退路径；不追求量化覆盖率牺牲精度。

# 可维可测分析

## 精度标准/性能标准

### 基线冻结与证据

- infer 仓基线 commit：`bbd278fe5ef772daa310df680865a6826ae6d909`；上游固定 commit：`209209b7086eba2020c5439265221495a8357322`（按模型 README 非覆盖合并，记录覆盖/补入文件清单）。
- 权重：ModelScope 官方来源（daVinci-MagiHuman base/turbo_vae、T5Gemma、Stable Audio、Wan2.2 VAE），逐文件 size/SHA256 校验后记录；大文件不随仓库提交，附获取位置与校验信息。
- 输入：官方 `example/assets/prompt.txt` 固定 prompt（记录内容 hash）、seed=42、完整解析后配置与有效 CLI 随日志存档。
- 两档（BF16/MXFP8）使用同一源码准备方式、依赖、权重、环境与生成输入。

### 设计可行性依据

设计阶段已在锁定环境（950PR + CANN 9.1.0 + 驱动 25.7.rc1.10 + Py3.12 + torch 2.10.0 + torch_npu 2.10.0.post4）完成算子能力验证，确认本设计的三个关键前提成立：

1. `npu_dynamic_mx_quant` 在真实 GQA 形态（按本设计契约的轴向）下可用，量化输出与 scale 的类型、形状符合「支持数据类型」契约；
2. FA v2 MXFP8 主链路（TND、quant mode 6/6/8、GQA 40/8）可用，输出 dtype 恢复行为正确；
3. **FA v2 的 `return_softmax_lse=True` 返回可用且语义满足 flex 区间合并需求**——flex 多 K 区间因此可按本设计采用逐段 MXFP8 + LSE 合并，无需 BF16 回退。

flex 每段为 singleton batch 的连续子区间；FA v2 对 varlen 单调用的段边界对齐约束已经实测确认，逐段调用方案不依赖该对齐。

以上验证只为确认设计可行性，不构成真实模型 case、视频质量或性能结论。

### 数值检查

对相同 Q/K/V 输入对比 BF16 与 MXFP8 的：量化前 Q/K/V 分布、scale（非全 1、非伪造，饱和率记录）、FA 输出、同一层 DiT block 输出。反量化仅用于离线误差检查，不进入推理路径。

指标：`NRMSE = ||X_test-X_ref||₂ / max(||X_ref||₂, ε)`，cosine 按 FP32 展平向量计算；同时记录 MAE、p50/p99、min/max、NaN/Inf 计数、FP8 饱和比例。零范数 reference 单独记录绝对误差。

| 对象 | 实验前提出的门限 |
| --- | --- |
| Q/K/V 量化重建 | 各自 NRMSE ≤ 0.08、cosine ≥ 0.995 |
| 相同输入 FA 输出 | NRMSE ≤ 0.10、cosine ≥ 0.99 |
| 同一输入 DiT block 输出 | NRMSE ≤ 0.05、cosine ≥ 0.995 |
| 稳定性 | 无 NaN/Inf；scale 动态产出非全 1；饱和率在实测范围 |

门限随本设计评审冻结，正式失败结果保留原始记录，不因量化结果放宽容忍。

### 最终质量

固定 5 组 prompt（覆盖单人、多人、肢体动作、镜头运动、复杂背景，含可检查对白/环境声），BF16/MXFP8 保持 seed、权重与生成参数一致；随机 A/B 标签 + 隐藏映射，由 2 名指定验收人独立盲测，记录主体、语义、构图、动作、时序、色彩、音画一致性与伪影；两者均未发现稳定可辨识退化方为通过。验收人待指定，不以自动指标代替盲测。

### 性能

每档 ≥1 次完整生成预热、≥3 次正式完整生成；计时起止处 NPU synchronize（tensor map 计时口径），模型加载不混入生成计时；正式计时不使用 profiler。分别记录 Base（去噪）、Decode 与 Total 的逐次值、平均值、中位数、极差。

判定：`median(BF16) - median(MXFP8) > range(BF16)`（range 定义为 BF16 三次正式 Total 的 max-min）；同时满足 FA 累计耗时下降、计入 Q/K/V 量化与 layout 转换成本后的 Attention 累计耗时下降。profiling 使用独立重复运行，按算子统计累计耗时与调用次数。

### 内存

两档分别记录：加载后、每次生成峰值、连续 5 次完整生成的 NPU `allocated`/`reserved` 与 `npu-smi` 使用量；同一环境、独立新进程、每次生成前重置峰值统计。

**口径说明（重要）**：本任务只量化 Q/K/V 激活，不量化权重。峰值内存由模型驻留（权重）与解码阶段临时量主导，Attention 激活（单层 MB 级、短生命周期）经分配器池化后对整机峰值贡献有限；因此同时报告三层口径并做归因：

- `allocated` 峰值（真实张量占用）；
- `reserved` 峰值与采样重复性（同配置多次采样的波动范围，说明"几十 MB 级差异不可分辨"）；
- `npu-smi` 整机值；
- Attention 算子级的峰值增量对比（MXFP8 路径与 BF16 路径的算子级内存行为）。

连续 5 次生成用于验证不泄漏（峰值稳定不增长）与资源释放；回退、非法参数异常后的资源释放单独验证。

## 兼容性分析

- `module/fa_quant` 既有接口（`npu_mxfp8_quant` / `npu_mxfp8_attn`）签名、默认返回值、dtype 行为全部保持；LSE 通过新增 keyword-only 参数开启，默认关闭；HunyuanVideo 与 `unified_sp` 无感。
- 公共接口修改前后检索全部直接调用方并留存清单；对每个调用方做 API/功能验证。
- HunyuanVideo `single_fp8.yaml` 代表配置在相同权重/参数/环境下执行修改前后的精度与 ≥1 预热 + ≥3 正式性能回归（仅用于模块回归，其权重量化收益不计入本任务）；回归性能不低于原基线波动范围，精度沿用调用方门限。
- 接口用例覆盖：标准 shape、不同的 Q/K 长度、TND/BSND、GQA head 比、非法 dtype/layout/长度、配置关闭、回退路径；构造输入测试标记为接口测试，不替代真实模型 case。

## 交付物

模型 README 与 `docs/models/davinci-magihuman/` 优化说明、公共模块文档、测试用例与脚本；`pre-commit run --all-files`、仓库 CI 与 950 实机结论；私仓 `task_submission/` 按任务书提交步骤说明与精度/性能/内存报告及原始日志；代码 PR 关联 RFC、本设计 PR 与全部自验结果，未完成项明确列出。
