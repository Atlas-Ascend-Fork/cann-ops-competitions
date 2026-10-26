# 需求背景（required）

## 需求来源

社区任务 10-5「10月社区任务-Wan2.2-I2V MatMul INT8 量化」，在 Atlas A3 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 实现 Wan2.2-I2V linear 层的 MatMul INT8 量化。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计按官方模板组织，为设计评审稿，性能/精度门槛以任务书为验收依据，不将设计目标表述为已完成的实测结论。

## 背景介绍

### 模型与量化范围

Wan2.2-I2V 为 DiT 架构图生视频模型（`models/wan2.2-i2v/`，`wan/modules/model.py` 含 WanAttention/WanMLP 等 block），linear 层集中在：

- Self-Attn / Cross-Attn 的 `q/k/v/o` 投影；
- FFN 的 `f1/f2`（gate-up / down）;
- 调制与输出层（`modulation`、`head`）——默认**不量化**，保持 BF16 以稳住首末级精度。

本任务对上述 DiT linear 层做 **W8A8 动态量化**：权重离线 per-channel 对称量化（int8 + fp32 scale），激活推理时 per-token 动态量化，计算走 `torch_npu.npu_quant_matmul`（int8 输入、int32 累加、BF16 输出），不引入 FP16 中转。

### 仓内现状（工作量边界）

| 现状 | 说明 | 本任务动作 |
| --- | --- | --- |
| `module/quantization/` | 已有 fp8.py / mxfp4.py / mxfp8.py（HunyuanVideo 等接入），**无 int8.py** | 新增 `int8.py`：权重离线量化 + 在线动态量化 + 反量化 scale 管理公共实现 |
| `module/linear.py` | 通用 linear 封装 | 增加 int8 分支（与 fp8 分支同构的替换点） |
| `models/wan2.2-i2v/` | 无任何量化开关 | YAML→CLI→模型贯通开关与回退 |
| HunyuanVideo fp8/mxfp8 接入 | 已有样例路径 | 作为接入模式参照，不回归范围 |

# 需求分析（required）

## 需求描述

为 Wan2.2-I2V 提供 MatMul INT8 量化推理路径：权重 int8 per-channel、激活 per-token 动态量化、`npu_quant_matmul` 计算，输出 BF16、shape/语义与 BF16 路径一致；通过开关启用/回退；满足任务书精度与性能门槛（视频质量 + 端到端时延），A3 实测交付。

## 需求拆解

1. 公共实现：`module/quantization/int8.py`（权重离线量化脚本 + 在线动态量化 + `npu_quant_matmul` 封装），与 fp8/mxfp8 同层复用。
2. 模型接入：`wan/modules/model.py` 的 WanT2V block 内 linear 替换点；开关 `matmul-int8` 贯通 `config/*.yaml → infer.sh → 模型`；量化排除清单（modulation/head/text encoder/VAE）显式配置。
3. 权重处理：离线量化产物（int8 权重 + scale）缓存复用，避免每次启动重算；量化误差统计（per-layer max/mean 相对误差）报告输出。
4. 精度：block 级输出余弦相似度门限 + 端到端视频质量（PSNR/SSIM 与 BF16 基线对比）双门槛；固定 seed 下采样对比。
5. 性能与内存：端到端时延、DiT 段耗时、峰值内存对照 BF16 基线；性能门槛以任务书为准。
6. 兼容与交付：BF16 路径零行为变化；README/配置样例；pre-commit 与仓内 CI 通过。

# 详细设计（required）

## 功能分析

### 量化语义

- 权重：`W_q = round(W / s_w)`，`s_w = max|W|_per_out_channel / 127`（fp32 scale，per-channel，out_features 维）；
- 激活：`X_q = round(X / s_x)`，`s_x = max|X|_per_token / 127`（在线动态，per-token，最后一维归约）；
- 计算：`Y = npu_quant_matmul(X_q, W_qᵀ, bias=None, scale_x=s_x, scale_w=s_w, dtype=BF16)`，scale 融合在算子内完成，全程无 int8→float 反量化中间张量。
- 非对称/分组量化不做（首版对称 per-channel/per-token，任务书若要求 SmoothQuant/分级再扩展，α 系数作为配置项预留）。

### 支持范围与排除

| 项 | 契约 |
| --- | --- |
| 量化层 | DiT self/cross-attn 的 qkvo、FFN f1/f2（shape `[out, in]`，in 为 hidden 或 hidden/2 倍数） |
| 排除层 | modulation、head、text_encoder、VAE、timestep embedding |
| dtype | 权重 int8 + fp32 scale；激活 BF16 → 动态量化 int8；输出 BF16 |
| shape | 固定分辨率配置下 token 数固定；动态 shape（可变分辨率）时激活按实际 token 数量化，权重 scale 不变 |

## 接入与实现设计

### host 侧（工程接入）

1. **配置**：`config/matmul_int8.yaml`（复制 base.yaml + `model_args: matmul-int8: true`、`matmul-int8-fallback: true`、排除层清单）；`infer.sh` 增加模式入口。
2. **替换点**：`WanLinear.forward` 按 开关+层名+shape 契约 判定走 int8 分支；不符契约（非 BF16 输入、shape 异常）或运行异常时按回退开关走 BF16 并记审计日志。
3. **权重准备**：独立量化脚本按模型配置生成 `*.int8.pt`（W_q + s_w），加载路径优先使用缓存产物；产物带配置指纹（层名/shape/seed）防错配。
4. **审计日志**：量化层数、排除层数、每层量化误差、回退次数；正式性能运行不逐调用打印。

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| block 级精度 | 固定输入下单 block 输出 BF16 vs INT8 余弦相似度 | ≥ 0.999（首层）/ ≥ 0.99（40 层累计），以任务书为准 |
| 端到端质量 | 同 seed 三组 prompt 生成视频，PSNR/SSIM 对 BF16 基线 | 任务书门限 |
| 性能 | 每档 ≥1 次预热、≥3 次正式生成；DiT 段与端到端耗时、峰值内存 | 对照 BF16 出比值，门槛以任务书为准 |
| 兼容回归 | BF16 路径同 seed 输出逐位一致；HunyuanVideo 不受影响 | 零回归 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| qkv/f1 大 K 维 per-token 量化误差累积 | block 级 cos-sim 逐层监控定位；必要时对注意力 q/k 层改 per-tensor + 平滑系数（预留配置，不静默启用） |
| npu_quant_matmul 动态 shape 下 scale 传播开销 | scale 沿 token 维融合在算子内；避免 python 端逐次 allclose/同步 |
| 首末层量化导致首帧伪影 | 排除清单兜底；评审时可按任务书收紧量化层范围 |
| 与 dit_cache（若同开）交互 | 两开关正交验证；联合开关用例入回归集 |
