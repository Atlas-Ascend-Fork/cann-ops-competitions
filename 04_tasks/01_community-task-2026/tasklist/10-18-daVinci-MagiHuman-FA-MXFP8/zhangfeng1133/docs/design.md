# 需求背景（required）

## 需求来源

社区任务 10-18「10月社区任务-daVinci-MagiHuman FA MXFP8 量化」，在 Ascend 950 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)（`module/fa_quant/` 公共模块与 950 平台算子）实现 daVinci-MagiHuman FA 的 MXFP8 量化。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，与同任务已提交设计（`ji096929`）任务口径一致、实现路线独立，验收以任务书为准。

## 背景介绍

### 模型与公共设施

daVinci-MagiHuman 为视频+音频联合 DiT（40 层，hidden 5120，Q 40 heads / K,V 8 heads GQA，head_dim 128），`base` 模式全层标准稠密 Attention（`flash_attn_func`，BSND），`SR2_1080` 模式经 `local_attn_layers` 启用 flex 局部注意力（`flex_flash_attn_func`，BNSD 段级 + softmax LSE 合并）。两者现均为 BF16 `npu_fused_infer_attention_score`。

公共模块 `module/fa_quant/` 已提供：Q/K/V 动态 MXFP8 量化（`npu_mxfp8_quant`，封装 `npu_dynamic_mx_quant`，输出 FP8 + E8M0 scale）与 MXFP8 FA 封装（`npu_mxfp8_attn`，封装 `npu_fused_infer_attention_score_v2`）;HunyuanVideo 已有 `fa-mxfp8` 接入样例、`module/unified_sp/core.py` 亦有直接调用。

### 任务边界

仅量化 Attention 的 Q/K/V 激活并走 MXFP8 FA 计算;MatMul 权重量化、RoPE、缓存算法、图模式收益均不在本任务;随机数生成、seed 管理不涉及。

# 需求分析（required）

## 需求描述

在 950 单卡上使 magihuman 标准 Attention 与 flex 局部 Attention 真实执行 Q/K/V 动态 MXFP8 量化 + MXFP8 FA v2 计算：输出保持原 dtype/shape/GQA/局部区间语义;开关与回退贯通;通过中间张量比较、视频/音频质量、端到端性能与峰值内存验证;A5/950 实测交付。

## 需求拆解

1. 能力探针：实机验证真实 GQA 40/8/D128 的 `npu_dynamic_mx_quant` 与 MXFP8 FA v2（含 `return_softmax_lse=True`）支持面。
2. 公共接口：复用 `npu_mxfp8_attn`;如需扩展 LSE 返回，新增 keyword-only 参数、默认行为零变化。
3. 模型接入：`flash_attn_func` 标准路径与 `flex_flash_attn_func` 段级路径双接入;开关 `fa-mxfp8` 贯通 YAML→CLI→模块;回退开关控制契约不符时的 BF16 回退与审计日志。
4. flex 语义：每段（单 K 区间）独立量化 + MXFP8 FA v2 + LSE 返回，按原 softmax-LSE 合并公式合并;多 K 区间不回退 BF16。
5. 精度：Q/K/V、scale、FA 输出、block 输出分层门限;固定样本质量盲测。
6. 性能与内存：FA 累计耗时、含量化/layout 的 Attention 累计耗时、端到端耗时、峰值内存对照 BF16 基线，门槛以任务书为准。
7. 兼容：既有调用方（HunyuanVideo、unified_sp）零回归;pre-commit 与仓内 CI 通过。

# 详细设计（required）

## 功能分析

### 计算语义

标准路径：Q/K 按最后一维、V 按 token 维动态 MXFP8 量化，scale（E8M0）直传 FA v2（TND 布局、quant mode 6/6/8、GQA 真实 head 数 8、非 sparse），输出 BF16——**不走"反量化后 BF16 FA"**。

flex 路径：保持原"Q 区间切分 + 逐 K 区间计算 + LSE 合并"结构;每段独立量化与 FA v2（`return_softmax_lse=True`），LSE 归一化后并入原合并逻辑（`L=logsumexp(L_i)`, `O=Σ exp(L_i−L)·O_i`），与原 BNSD v1 合并保持同一数值语义。

| 张量 | 契约 |
| --- | --- |
| 量化前 Q/K/V | BF16，维度长度合法、contiguous 化后量化 |
| 量化后 | `float8_e4m3fn` + E8M0 scale，由算子动态生成，禁止伪造 scale |
| FA 输出 | BF16,外层按原 dtype/shape 恢复 |
| LSE | fp32,仅 flex 路径消费 |

### 支持范围与回退

BSND 4D、head_dim 为 32 倍数、bf16、CP world size=1 为支持契约;契约不符或运行异常：`fa-mxfp8-fallback` 开启走 BF16 并记审计日志（含原因、去重计数），关闭则报错。参数类非法输入直接报错，不当能力回退吞掉。

## 接入与实现设计

### host 侧（工程接入）

1. **配置**：`config/fa_mxfp8.yaml`（`model_args: fa-mxfp8: true`、`fa-mxfp8-fallback: true`）;`infer.sh` 模式入口;CLI 经 `entry.py` 写入 dit_module 模块级开关（默认关闭）。
2. **标准路径**：`flash_attn_func` 开关开启且契约满足时调用 `npu_mxfp8_attn`（BSND 入、TND 算、GQA 40/8/128、真实 seq len、`1/sqrt(D)`）。
3. **flex 路径**：`flex_flash_attn_func` 保持切分结构,段内走量化 + FA v2 + LSE;LSE 合并数值语义与原实现对拍冻结。
4. **公共接口**：`npu_mxfp8_attn` 如扩展则 `return_lse=False` keyword-only,旧调用方签名/默认行为零变化。
5. **审计日志**：模式/分支/shape/heads/真实长度/dtype/scale shape/回退统计;正式性能运行不逐调用打印、不引入额外同步。

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 中间张量 | 固定输入：Q/K/V 量化结果、scale、FA 输出与 BF16 路径分层 cos-sim | 任务书分层门限 |
| 端到端质量 | 固定 seed 多组样本生成,视频/音频盲测 + 客观指标 | 无稳定可辨识退化（任务书口径） |
| 性能与内存 | 每档 ≥1 预热 + ≥3 正式;FA 累计/含量化 Attention 累计/端到端耗时、峰值内存 | 对照 BF16 出比值，门槛以任务书为准 |
| profiling 证据 | 同时出现三路动态量化与 MXFP8 FA v2,参数与 dtype/scale shape 交叉确认 | 不以同名算子出现为唯一证据 |
| 兼容 | HunyuanVideo 代表配置回归、既有调用方签名核对 | 零回归 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| flex 段内 LSE 合并语义与原 v1 实现有微差 | 合并公式单测（fp64 参考）+ 端到端质量门槛双卡点 |
| 量化/layout 转换开销吃掉 FA 收益 | 收益核算含量化与 layout 总成本;profiling 拆分报告 |
| 音频 token（126）段占比小、head 维度不规则 | 段级契约校验,不符回退并单独统计 |
| 与 matmul-int8 等组合量化级联 | 组合用例独立跑门槛;文档标注推荐组合与互斥项 |
