# 需求背景（required）

## 需求来源

本设计对应「10月社区任务-daVinci-MagiHuman FA MXFP8 量化任务书（A5）」及配套 `test_cases/README.md`，目标代码仓为 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。流程采用[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与社区任务目录说明；目标仓贡献规范见其 `CONTRIBUTION.md`。

本设计提交团队为 `potatomato111`。需求基线、能力契约与判定标准已在 RFC（[Issue #437](https://gitcode.com/cann/cann-recipes-infer/issues/437)）中锁定；设计对应的实现与 950 实机自验（能力探针、单元用例、中间张量对齐、盲测、端到端性能与内存）已完成，代码与自验材料在开发者私仓 `potatomato111/cann-recipes-infer` 的 `a5-submission` 分支交付。

## 背景介绍

### daVinci-MagiHuman Attention 现状分析

daVinci-MagiHuman DiT 共 40 层，hidden size 5120，head_dim 128，Q 40 heads / KV 8 heads（GQA）。`flash_attn_func`（标准稠密路径）与 `flex_flash_attn_func`（局部/变长路径）当前使用 BF16 `npu_fused_infer_attention_score`；base 文生配置的 `local_attn_layers` 为空，标准 Attention 为唯一实际触发路径。局部路径使用 Q/K 区间划分与 softmax LSE 合并，不可用全序列稠密 Attention 直接替换。

仓内公共模块 `module/fa_quant/` 已提供 Q/K/V 动态 MXFP8 量化（`npu_dynamic_mx_quant`）与 `npu_fused_infer_attention_score_v2` 封装（`npu_mxfp8_attn`），HunyuanVideo 已有接入样例（`config/single_fp8.yaml`）。现有公共封装面向 TND、非 sparse 的基础路径；daVinci-MagiHuman 所需的 GQA、变长与 LSE 合并语义超出其直接契约，需要按公共接口方式扩展而不能在模型内复制实现。

本任务仅量化 Attention 的 Q/K/V 激活，复用仓内公共算子能力；MatMul 权重量化、RoPE 融合与缓存算法收益不纳入范围。

# 需求分析（required）

## 需求描述

在 Ascend 950 单卡上，使目标 Attention 路径真实执行动态 MXFP8 FA（Q/K/V 经 `npu_dynamic_mx_quant` 量化为 FP8 E4M3、scale 为 E8M0 动态产出，注意力经 `npu_fused_infer_attention_score_v2` 计算，输出恢复原 dtype），保持 GQA、变长/局部范围与输出语义；未支持组合必须显式回退 BF16（可审计日志）或 fail-fast。生成质量经固定样本盲测确认，端到端性能获得提升，公共模块既有调用方保持兼容。

## 需求拆解

| 工作 | 产物与通过条件 |
| --- | --- |
| 基线与 RFC | 固定代码基线（master `bbd278f`）、上游 commit、权重 revision/hash、软件矩阵、配置与输入；950 实机采集 BF16 base 基线 |
| 能力验证 | 实机验证 `npu_dynamic_mx_quant` 与 FA v2 MXFP8（quant mode 6/8）能力，锁定 MXFP8 契约（量化轴、scale 形状、LSE 形状） |
| 公共接口 | `module/fa_quant` 内纯新增公共接口，既有 `npu_mxfp8_attn`/`npu_mxfp8_quant` 签名与行为零变化；直接调用方 API/功能回归 |
| 模型接入 | `fa-mxfp8`/`fa-mxfp8-fallback` 开关贯通 CLI/配置/实际 Attention；不支持组合回退或 fail-fast |
| 精度 | 量化前 Q/K/V、scale、FA 输出与 DiT block 数值检查（预设门限）；scale 非全 1、无 NaN/Inf；≥5 组固定样本同 seed 双模式盲测 |
| 性能与内存 | 同机同配置预热≥1、正式≥3 次完整生成；Total 与 denoise 中位耗时下降超基线波动范围；FA 累计耗时下降，计入量化与 layout 转换后 Attention 总耗时仍下降；峰值 NPU 内存记录并与基线对比 |
| 兼容与交付 | 关闭特性恢复 BF16 基线；pre-commit 与仓库 CI 通过；文档更新；私仓 task_submission 自验材料 |

# 详细设计（required）

## 能力分析

### 数学公式

对 query head \(h\)，其对应 KV head 为 \(g(h)=\lfloor h/(H_q/H_{kv})\rfloor\)，原计算为：

\[
O_h=\operatorname{softmax}\left(Q_h K_{g(h)}^T/\sqrt{D}+M\right)V_{g(h)}
\]

量化路径中 Q/K 按最后一维（逐 token、D 维 32 元素组）、V 沿 token 维（每 64 token 一组）执行 `npu_dynamic_mx_quant`，得到 FP8 E4M3 值与 E8M0 scale，scale 随量化结果直接传入 FA v2（quant mode：query/key=6，value=8），不做「反量化后走 BF16 FA」。V 不存在逐 token scale 模式（实机穷举 value quant mode 1–6 均拒绝），64-token 组为 FA v2 的 V scale 契约。

flex 路径同一 Q 区间命中多个 K 区间时按 softmax LSE 合并：

\[
L=\operatorname{logsumexp}(L_1,\ldots,L_m),\qquad O=\sum_i \exp(L_i-L)\,O_i
\]

FA v2 的 LSE 形状为 (Lq, N, 1) fp32（与 v1 BNSD 的 (N, Lq) 不同），合并逻辑按新形状适配，不沿用 permute 假设。

### 契约表（实机定谳，冻结于 RFC）

| 项 | 契约 |
| --- | --- |
| 量化 dtype | Q/K/V 量化输出 `float8_e4m3fn`，scale 为 e8m0（`float8_e8m0fnu`），动态产出，禁止全 1/伪造 scale |
| Q/K 量化轴 | `npu_dynamic_mx_quant(axis=-1)`：逐 token、D 维 32 元素组 |
| V 量化轴 | `npu_dynamic_mx_quant(axis=0)`：沿 T 每 64 token 组，scale 形状 `[ceil(T/64), N_kv, D, 2]` |
| FA 调用 | `npu_fused_infer_attention_score_v2`，input_layout=TND，sparse_mode=0 |
| GQA | 支持（DiT 实形态 40:8 实测通过） |
| varlen 限制 | FA v2 TND varlen 单调用要求段边界 64 对齐，真实多模序列不满足 → 变长路径按段量化 |
| LSE 形状 | (Lq, N, 1) fp32 |

## 实现设计

### 公共模块扩展（`module/fa_quant/`，纯新增）

- `npu_mxfp8_quant_tnd`：TND 量化直通（axis=-1 / axis=0 两档，内部参数校验与错误信息与公共封装一致）；
- `npu_mxfp8_attn_tnd_dense`：单段 dense MXFP8 FA v2 封装，返回 (output, lse)，供模型侧标准与分段调用；
- 既有 `npu_mxfp8_attn`/`npu_mxfp8_quant` 签名与行为零变化；仅做语义等价的 host 侧结果缓存（避免每次调用重建公共张量），既有直接调用方（HunyuanVideo 等）兼容性经单元用例与 API 回归确认。

### 模型接入（`models/davinci-magihuman/`）

- 标准路径：`flash_attn_func` 复用公共接口完成量化与 FA v2 调用（GQA 参数逐次透传）；
- 变长/局部路径：`flex_flash_attn_func` 保持原段循环 + softmax LSE 合并结构；每个（Q 段 × K 段）内层调用替换为「Q/K 整段量化（逐 token scale，切片安全）+ V 按段量化并缓存 + FA v2 单段 dense」；
- 开关与回退：`config/fa_mxfp8.yaml`（自 base.yaml 复制全部字段附加 `fa-mxfp8: true` / `fa-mxfp8-fallback: true`），入口 `bash infer.sh fa_mxfp8`；head_dim 非 32 倍数、非 NPU 张量、CANN 版本低于能力要求、context-parallel（cp_size>1）等组合回退 BF16 并输出 `[FA_MXFP8] fallback: ...` 审计日志，`fallback=false` 时 fail-fast；
- 数值验证钩子：`FA_MXFP8_DEBUG_DUMP` 支持 dump 量化前 Q/K/V、FA 输出与 DiT block 输出，用于双跑数值对齐。

### 支持矩阵（base 文生配置实机定谳）

| 分支 | 命中情况 | 结论 |
| --- | --- | --- |
| `flash_attn_func`（dense，GQA 40:8） | 实机命中（80 call/步） | 已覆盖并通过数值对齐与盲测 |
| `flex_flash_attn_func`（局部/变长） | base 文生配置 `local_attn_layers` 为空，全程未触发（profiling 证据：命中日志仅出现 flash_attn_func） | 列为不支持项；该路径以合成变长/局部用例覆盖并验证 LSE 合并语义 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950（950PR 实测） | √ |

MXFP8 FA v2 为 CANN ≥ 9.1.0 的算子能力（9.0.0-beta.2 的 FA v2 tiling 仅支持 per-tensor quant，实机验证不支持本路径所需 quant mode）；实机软件矩阵：CANN 9.1.0（toolkit + 950-ops）+ torch 2.10.0 + torch_npu 2.10.0。

## 算子约束限制

- V 无逐 token scale 模式（value quant mode 1–6 实测拒绝），V 量化必须按 64-token 组；
- FA v2 TND varlen 单调用要求段边界 64 对齐，真实多模序列不满足，变长路径按段量化；
- context-parallel（cp_size>1）与 MXFP8 组合未验证，自动回退 BF16；
- 非 32 倍数 head_dim、非 NPU 张量回退 BF16（可配置为 fail-fast）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 判定 | 结果（950 实机） |
| --- | --- | --- |
| 能力探针 | `npu_dynamic_mx_quant` / FA v2 BF16 / FA v2 MXFP8 三项 PASS | 通过 |
| 单元/功能用例 | 标准 shape、变长/局部 shape、GQA、配置关闭、非法参数、BF16 回退 | 5/5 PASS |
| 中间张量门限 | 同输入首调 FA 输出 rel_mean ≤ 0.10 且 cos ≥ 0.98，无 NaN/Inf | rel 0.047 / cos 0.999；DiT block rel 0.062 / cos 0.9999 |
| scale 真实性 | 动态产出、非全 1 | Q/K/V scale unique 值 12/9/8，fp8 饱和率 ~2% |
| 盲测 | ≥5 组固定样本（单人/多人/肢体动作/镜头运动/复杂背景）同 seed 双模式 | 通过，无可稳定辨识退化 |
| 端到端性能 | 正式 3 次中位耗时下降超过基线波动范围 | Total 27.103→25.955 s（-4.24%，超极差 0.141 s）；denoise -4.40% |
| FA 算子累计 | 计入动态量化与 layout 转换后仍下降 | FA v1→v2+量化：-26%（含量化开销 0.5%） |
| 峰值内存 | 同配置记录并对比基线 | torch 峰值持平（权重驻留主导），reserved 略低；连续 5 次生成无增长、进程退出后资源可释放 |

性能计时口径为管线内置分段计时、无 profiler 干净口径（profiler 常驻插桩会引入墙钟开销并掩盖收益，仅用于算子级证据采集）。

## 兼容性分析

- 公共模块改动为纯新增，既有 `npu_mxfp8_attn`/`npu_mxfp8_quant` 调用方（HunyuanVideo 等）签名与行为不变，兼容性单元用例通过；
- 关闭 `fa-mxfp8` 开关后完全恢复 BF16 基线行为（等值验证用例通过）；
- HunyuanVideo `single_fp8.yaml` 代表性精度/性能回归按交付要求执行（其 Attention 使用既有 `npu_mxfp8_attn` 接口，与本任务新增接口无共享改动面）。
