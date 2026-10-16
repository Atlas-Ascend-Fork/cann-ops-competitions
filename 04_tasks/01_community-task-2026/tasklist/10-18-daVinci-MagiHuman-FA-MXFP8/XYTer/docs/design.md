# 需求背景（required）

## 需求来源

- 任务：daVinci-MagiHuman FA MXFP8 量化（A5），[社区任务入口](https://www.hiascend.com/activities/task-center/details/491a4d30e2e549e19857d26df87da1e0)。
- 提交者：XYTer。
- 目标代码仓：[cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)。
- 关联 RFC：[MXFP8 FA 接口、数值门限与内存验收范围](https://gitcode.com/cann/cann-recipes-infer/issues/449)。
- 本文为设计方案。接口、支持范围及验证方法供评审确认。

## 背景介绍

daVinci-MagiHuman 将视频、音频和文本组成统一序列。标准 Attention 经 `flash_attn_func` 调用 BF16 FA；局部 Attention 经 `flex_flash_attn_func` 按区间计算，再合并结果。模型另有 context-parallel（CP，序列跨卡划分）路径。

仓内 `module/fa_quant/` 提供 `npu_mxfp8_quant` 和 `npu_mxfp8_attn`。本方案复用公共模块，将 Q/K/V 激活动态量化为 MXFP8，再调用 FA v2。模型接入和临时张量管理放在适配层；不新建独立 kernel 工程，不修改底层算子。

# 需求分析（required）

## 需求描述

在 Ascend 950 上为标准 Attention 和选定局部 Attention 接入 `fa_mxfp8` 开关。保持 GQA、序列边界、局部可见范围和输出 dtype 语义。通过中间张量检查、最终生成盲测、profiling、端到端性能与全流程峰值内存验证收益。

## 需求拆解

| 工作项 | 设计要求 |
| --- | --- |
| 基线与 RFC | 固定源码、权重、软硬件、完整配置、输入、随机种子和测量方法 |
| 路径梳理 | 记录标准、flex、局部及 CP 的 Q/K/V shape、layout、head、实际长度、sparse、调用数和耗时 |
| 公共接口 | 复用 `module.fa_quant.npu_mxfp8_attn`，通用扩展保持现有调用方兼容 |
| 量化接入 | 动态生成 Q/K/V 与 scale，正确传递 TND、累计长度、GQA/MQA 参数 |
| 配置与错误 | 默认关闭；未支持组合明确回退 BF16 或 fail-fast，记录原因 |
| 验证 | 检查 Q/K/V、scale、FA、DiT block、最终视频、FA/Attention/端到端耗时及峰值内存 |
| 交付 | 更新配置、模型与公共模块文档，提交正式回归和规定自验材料 |

# 详细设计（required）

## 算子分析

### 数学公式

对 query head h，令 g(h) = floor(h / (Hq / Hkv))：

```text
O_h = softmax(Q_h K_g(h)^T / sqrt(D) + M) V_g(h)
```

Hq 为 query head 数，Hkv 为 K/V head 数。模型目标形态为 Hq=40、Hkv=8、D=128，分组比为 5。M 表示原路径的可见区间约束。

MXFP8 流程为：输入检查 → layout 转换 → `npu_dynamic_mx_quant` 生成 Q/K/V 与各自 scale → `npu_fused_infer_attention_score_v2` → 恢复 layout 和 dtype。

公共接口拟提供可选 `scale_alg=0` 和 `center_key=False`，保持原调用默认行为。MagiHuman 显式选择 `scale_alg=1` 和 K 中心化。算法 1 向上选择 E8M0 scale，以控制量化裁剪；目标版本的支持情况须通过能力探针确认。

K 中心化使用每个 batch/head 的序列均值 μ：K'=K-μ。单段 softmax 对每行相同的 logit 平移不敏感。若局部路径使用多段结果，返回 LSE 时需加回 Qμ/sqrt(D)，再参与原区间合并。LSE 是 softmax 分母的对数。多段合并保持：

```text
L = logsumexp(L_1, ..., L_m)
O = sum_i exp(L_i - L) O_i
```

### 数据类型与接口契约

| 对象 | 契约 |
| --- | --- |
| 公共输入 | rank-4 BSND，Q/K/V 同 NPU、同 FP16 或 BF16 dtype；模型目标使用 BF16 |
| Q/K 量化 | 沿 head_dim 量化，数据为 E4M3FN |
| V 量化 | 沿 token 维量化，按算子要求转换 layout |
| scale | 动态算子真实生成；uint8 承载 E8M0 编码，调用 FA 时显式指定 scale dtype |
| FA 参数 | TND；Q/K/V quant mode 分别为 6/6/8，遵循公共模块及目标 CANN 契约，不伪造 scale |
| head | `num_heads=Hq`、`num_key_value_heads=Hkv`；Hq 是 Hkv 的整数倍，不复制 K/V head |
| 序列长度 | Host 累计结束位置；与有效 token 和 batch 边界一致，Q 与 KV 可不同长 |
| 输出 | 默认返回单一 BSND tensor，dtype 与输入一致 |
| 可选 LSE | `return_softmax_lse=True` 返回输出及 FP32 BSN LSE，包含中心化偏移恢复 |

检查 rank、dtype、设备、K/V shape、head_dim、head 比和空序列。非连续输入按需连续化，开销计入 Attention。输出 dtype 恢复仅转换 FA 结果；不采用量化后立即反量化再调用 BF16 FA 的计算路径。

### 支持形状与路径

| 路径 | 方案与边界 |
| --- | --- |
| 标准 | Q `[B,Sq,40,128]`，K/V `[B,Skv,8,128]`；BSND 转 TND，保留 batch 边界 |
| GQA/MQA | 检查合法整数 head 比；测试 Hkv=8 和 Hkv=1 |
| 局部/flex | 保留 Q/K range 调度；每个非空连续区间独立调用公共 MXFP8 FA，多 K 区间按 LSE 合并 |
| 变长 | 局部区间使用真实长度；多个样本打包变长不直接套用等长 BSND 展平逻辑 |
| CP、打包变长、SR | 本设计采用单卡验收；未覆盖组合回退 BF16 或 fail-fast，不宣称多卡能力 |
| sparse/mask | 不直接移植原 sparse/mask 参数；仅对语义等价的连续子区间启用 MXFP8 |

## 算子实现

### host/适配层设计

1. 增加 `fa_mxfp8` 模式。配置从 `base.yaml` 保留完整字段，提供 `bash infer.sh fa_mxfp8` 入口。配置键 `fa-mxfp8` 默认 false，`fa-mxfp8-fallback` 控制能力不足时的行为，开关经入口传至 DiT Attention。
2. 标准路径经公共 `npu_mxfp8_attn` 处理 layout、动态量化、scale、FA 与输出恢复。模型内只保留配置和路径选择。
3. flex 保留原区间索引、输出位置和 LSE 合并顺序。通用 LSE 返回及精度控制放入 `module/fa_quant/`，模型专属调度留在模型目录。
4. 原参数和默认返回值保持兼容。新增参数为可选参数；HunyuanVideo 和 UnifiedSPAttention 保持默认算法及非中心化行为。
5. 量化完成后缩短 BF16 Q/K/V 和原 QKV 的持有时间。小型 gate 使用独立缓冲，避免 view 延长大张量生命周期。归一化保留必要的 FP32 运算，结果及时恢复 BF16。释放点须通过数据依赖和异步执行验证。
6. 未支持组合在算子调用前选择 BF16 或报错。回退记录分支、shape、配置与原因，并去重。非法参数直接报错；OOM、设备错误和非法访问向上传递，不作为回退条件吞掉。

### kernel 侧设计

直接使用 CANN 的 `npu_dynamic_mx_quant` 与 `npu_fused_infer_attention_score_v2`，不新增 tiling 或 kernel。先在目标软件组合下验证动态量化、真实 GQA 和 LSE 能力；能力不足时保留明确回退或错误行为。

### 涉及目录

| 目录 | 预期改动 |
| --- | --- |
| `models/davinci-magihuman/config/` 与推理入口 | 配置、开关透传和使用说明 |
| `models/davinci-magihuman/inference/model/dit/` | 标准/局部接入、回退和临时张量生命周期 |
| `module/fa_quant/` | 公共检查、layout、LSE 和可选精度控制 |
| 仓库规定的测试目录 | 公共接口及模型路径的正式回归 |
| 模型 README 与 `docs/models/davinci-magihuman/` | 软件矩阵、启用方式、支持范围及故障排查 |

## 支持硬件

| 项目 | 目标与锁定方式 |
| --- | --- |
| 硬件 | Ascend 950 单卡；记录芯片、卡数、Host/NPU 内存、驱动和固件 |
| 软件候选 | CANN 9.1.0、Python 3.12、PyTorch 2.10.0、torch_npu 2.10.0.post4、MagiCompiler 1.0.0 |
| 环境确认 | 实机探针确认后在 RFC 固定完整版本；两档使用相同组合 |

## 算子约束限制

- 单卡为本次验证范围。CP 需梳理和明确边界，多卡功能与性能不作为本次通过声明。
- 目标为推理态；保持原局部范围、拼接顺序和 GQA 语义，保留必要 BF16 路径。
- MatMul 权重量化、RoPE 融合和缓存算法收益不计入本任务。
- 适配层局部节省不必然改变全流程峰值。全流程峰值覆盖加载、DiT、解码等阶段，以最大值判定。
- 如评审纳入独立 `turbo-vae-fused-rmsnorm` 解码配置，应单独开关、验证和归因；组合收益不作为 FA 单独收益。范围由 RFC 评审确认。

# 可维可测分析

## 基线与可复现性

RFC 固定 infer 基线 `bbd278fe5ef772daa310df680865a6826ae6d909`，模型上游提交 `209209b7086eba2020c5439265221495a8357322`，两者分别记录。权重记录官方来源、revision、逐文件大小及 SHA256；输入记录原文或获取位置及 SHA256。

采用 base 文生视频、batch=1、448×256、5 秒、32 steps、CFG=2、seed=42、NoCache。另用共同实际尺寸 768×416 比较高分辨率趋势。默认标准路径与显式 `local_attn_layers=[0]` 的局部配置分别对照，不将后者称为默认 base。其余参数、权重和环境保持一致。

## 精度标准/性能标准

### 数值与分布

使用相同输入比较量化前 Q/K/V、真实 scale、FA 和至少一个 DiT block 输出，并补充完整 block 链检查。记录 shape/dtype、min/max、均值、标准差、分位数、NaN/Inf、真实裁剪及量化端点比例。反量化仅用于离线误差分析。

工程门限采用 RFC 提案，评审确认后固定，不依据正式结果放宽：

| 对象 | cosine 下限 | relative RMSE 上限 | 归一化最大误差上限 | 真实裁剪比例上限 |
| --- | ---: | ---: | ---: | ---: |
| Q/K/V 量化重建 | 0.999 | 0.04 | 0.125 | 0.03125 |
| FA 输出 | 0.99875 | 0.05 | 0.25 | — |
| 单层 DiT block | 0.99875 | 0.05 | 0.25 | — |
| 完整 40 层 block 链 | 0.995 | 0.10 | 0.50 | — |

relative RMSE 为 `||test-ref||₂ / max(||ref||₂, ε)`。归一化最大误差为 `max(abs(test-ref)) / max(max(abs(ref)), ε)`。cosine 使用 FP32 展平向量。零范数参考单独判定绝对误差。scale 必须由输入动态产生，不使用全 1 替代。所有目标张量须无 NaN/Inf、异常饱和和维度错误。

### 最终生成质量

至少固定 5 组样本，覆盖单人、多人、手部动作、镜头运动和复杂背景。两档使用相同 seed 和生成参数。随机隐藏 A/B 标签，由 2 名指定验收人独立评价主体、语义、构图、动作、时序、色彩、音画一致性和伪影，保留原始意见。要求无稳定可辨识的退化；数值通过不替代盲测。

### profiling 与性能

1. 先采集 `bash infer.sh base` 基线，逐路径记录 shape、layout、head、实际长度、sparse、调用数和耗时。
2. 交叉核对 profiler 和调用参数。目标路径须出现真实动态量化和 MXFP8 FA v2，同时核对 dtype、scale 和长度；名称本身不作为唯一证据。
3. 每档至少 1 次预热、3 次正式生成；最终复测采用 3 次预热、5 次正式样本，交替运行。同步 NPU 后计时，纳入全部正式样本，单独采集 profiler。
4. 分别报告 FA 累计、包含量化/layout/局部合并的 Attention 累计，以及 Base/Decode/Total，保留原始值、平均、中位和极差。
5. 通过条件：FA 和完整 Attention 累计下降；同 batch 端到端中位下降超过 BF16 正式样本极差；高分辨率相对降幅更明显。量化开销须被 FA 收益覆盖。

### 峰值内存与归因

相同配置和窗口记录 `max_memory_allocated` 原始 bytes，补充 reserved 和设备观测。统一预热、清理、reset 和同步位置。分别报告加载后占用、Attention 增量峰值、DiT、解码和完整生成峰值，并检查连续 5 次生成、异常及回退后的 Host/NPU 资源释放，要求无持续增长。

内存通过条件为**同配置 MXFP8 的全流程 NPU 峰值低于 BF16 基线**，报告原始值与 `(BF16-MXFP8)/BF16`。Attention 局部下降单独呈现，不替代该条件。

若解码构成全局峰值，适配层释放 Q/K/V 后全流程峰值仍可能相同。可经 RFC 单独评审解码融合配置，并执行 BF16/MXFP8 × 原解码/融合解码四组消融。横向保持解码配置一致以核对 FA 收益，纵向计算解码收益。组合方案明确范围和来源，不把解码下降计入 FA。

## 兼容性分析

- 关闭 `fa-mxfp8` 恢复 BF16。覆盖标准、局部、GQA/MQA、不同 Q/KV 长度、非连续输入、非法参数、能力不足和回退。
- 检索并固定全部直接公共调用方。对 HunyuanVideo、UnifiedSPAttention 等执行 API/功能/精度回归；HunyuanVideo 代表配置完成完整生成和性能对照。
- 公共默认参数与返回值保持兼容。代表性性能不得低于基线波动下界；分支选择测试不视为多卡执行证据。
- 按贡献指南执行 `pre-commit run --all-files`，检查格式、License、错误处理和正式回归；设计 CI 与产品 CI 分别记录，最终代码须 Ascend 950 实机验证。

## 交付计划

设计文档提交本任务团队目录并关联 RFC。产品 PR 提交必要模型适配、公共接口、配置、说明和正式测试，并关联设计评审。原始日志、trace、权重、视频和临时脚本保留独立材料目录，不混入产品代码。按任务书在私仓 `task_submission/` 提交七件自验文件，保留原始结果和未通过项。
