# 需求背景（required）

## 需求来源

社区任务 10-8「10月社区任务-daVinci-MagiHuman MatMul INT8 量化」，在 Atlas A3 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 实现 daVinci-MagiHuman linear 层的 MatMul INT8 量化。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，验收以任务书为准，不将设计目标表述为已完成的实测结论。

## 背景介绍

daVinci-MagiHuman 为视频+音频联合生成的多模态 DiT 模型（40 层 Transformer，hidden 5120，head_dim 128），代码在 `models/davinci-magihuman/`（`inference/model/dit/dit_model.py`、`dit_module.py` 为 DiT 主体）。linear 层集中在各 DiT block 的 self/cross-attention q/k/v/o 投影与 FFN；`t5_gemma` 文本塔与 `vae2_2` 默认**不量化**。

量化语义与 10-5（Wan2.2-I2V）同构：**W8A8 动态量化**——权重离线 per-channel 对称 int8 + fp32 scale，激活 per-token 在线动态量化，`torch_npu.npu_quant_matmul` 计算、BF16 输出，无反量化中间张量。与 10-5 的差异仅在模型结构（GQA 40/8 注意力、音/文/视频 token 混排、双输出头），故本设计聚焦差异项，公共量化实现复用同一份 `module/quantization/int8.py`（与 10-5 共建，先合入者为准）。

# 需求分析（required）

## 需求描述

为 daVinci-MagiHuman DiT 提供 MatMul INT8 量化推理路径：量化范围/排除清单、开关与回退贯通 YAML→CLI→模型；block 级与端到端质量、端到端时延与内存满足任务书门槛；A3 实测交付。

## 需求拆解

1. 公共实现复用 `module/quantization/int8.py`（权重离线量化脚本、在线动态量化、`npu_quant_matmul` 封装）。
2. 模型接入：`dit_module.py` linear 替换点;量化排除清单（modulation/head/t5_gemma/vae2_2/timestep embedding）显式配置。
3. 权重产物缓存（int8 + scale + 配置指纹），量化误差统计报告。
4. 精度：block 级余弦相似度 + 端到端视频/音频质量双门槛;视频与音频 token 混排段专项对比。
5. 性能与内存：端到端时延、DiT 段耗时、峰值 HBM 对照 BF16 基线，门槛以任务书为准。
6. 兼容：BF16 路径零行为变化;与 fa-mxfp8（若同开）正交组合验证。

# 详细设计（required）

## 功能分析

### 量化语义（与 10-5 共用，差异项如下）

| 项 | 本模型契约 |
| --- | --- |
| 量化层 | 40 层 DiT block 的 q/k/v/o、FFN f1/f2;hidden 5120，K 维均为 5120 或其倍数 |
| 排除层 | modulation、head（视频/音频输出头）、t5_gemma、vae2_2、timestep/pos embedding |
| 音视频混排 | 单次前向内视频/音频/文本 token 同一激活张量，per-token 量化天然按 token 独立，无需分段;但质量评测须分模态统计（视频 PSNR/SSIM、音频频谱指标），防止某一模态劣化被平均掩盖 |
| GQA 40/8 | 与 linear 量化无关（FA 层不动），排除清单不涉及 |

### 支持范围

固定分辨率/时长配置下 token 数固定;动态 shape 按实际 token 数在线量化，权重 scale 不变。非 BF16 输入、shape 异常按回退开关走 BF16 并记审计日志。

## 接入与实现设计

### host 侧（工程接入）

1. **配置**：`config/matmul_int8.yaml`（`model_args: matmul-int8: true`、`matmul-int8-fallback: true`、排除层清单）;`infer.sh` 模式入口。
2. **替换点**：`dit_module.py` 内 linear 前向按 开关+层名+shape 判定;`dit_model.py` 传入层名上下文。
3. **权重准备**：量化脚本生成 `*.int8.pt` 缓存产物，配置指纹防错配;A3 HBM 充足时权重 int8 常驻，启动时一次性反量化表不落盘。
4. **审计日志**：量化/排除层清单、逐层误差、回退次数;正式运行不逐调用打印。

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| block 级精度 | 固定输入逐 block 余弦相似度（视频段/音频段 token 分组统计） | ≥0.999（首层）/≥0.99（40 层累计），以任务书为准 |
| 端到端质量 | 同 seed 三组样本：视频 PSNR/SSIM + 音频谱失真/盲听抽检 | 任务书门限;任一模态单独不达标即不达标 |
| 性能与内存 | ≥1 预热 + ≥3 正式;端到端与 DiT 段耗时、峰值 HBM 对照 BF16 | 门槛以任务书为准 |
| 兼容回归 | BF16 逐位一致;与 fa-mxfp8 组合用例 | 零回归 |

### 风险与对策

| 风险 | 对策 | 
| --- | --- |
| 音频 token 占比小、劣化被视频平均掩盖 | 质量门槛按模态分别判定 |
| modulation 未量化但输入来自量化层，误差经调制放大 | block 级监控调制输出;必要时收紧量化层范围 |
| 与 fa-mxfp8 并开的量化级联误差 | 组合用例单独跑门槛;文档标注推荐组合 |
| 5120 大 K 维激活 per-token 量化动态范围大 | 预留 per-tensor + 平滑配置;逐层误差定位劣化层 |
