# 需求背景（required）

## 需求来源

社区任务 10-10「10月社区任务-daVinci-MagiHuman RoPE优化」，在 Atlas A3 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 完成 daVinci-MagiHuman 的 RoPE 融合算子替换，并收敛到可复用的融合路径。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，验收以任务书为准。

## 背景介绍

daVinci-MagiHuman（40 层 DiT，Q 40 heads / K,V 8 heads GQA，head_dim 128）当前 RoPE 以基础算子链实现：cos/sin 表切片 → q/k 按 head 拆半/交错旋转（多次 mul/add/cat/neg）→ 拼回。每层每步对 Q/K 各执行一遍，产生大量小算子与中间张量，位于注意力关键路径上。

仓内现状：`models/davinci-magihuman/inference/model/dit/`（dit_model.py / dit_module.py）内含 RoPE 逻辑;`module/` 层**无公共 RoPE 融合封装**（fa_quant/dit_cache/quantization 均与本任务无关）。`torch_npu` 提供 `npu_rotary_pos_emb` / `npu_apply_rotary_pos_emb`（或 `npu_rotary_mul`，以实机版本能力探针为准）可单算子完成旋转位置编码应用。

任务目标：① 替换——RoPE 计算收敛到 NPU 融合算子;② 收敛——封装为 `module/` 层可复用路径（供 magihuman 及后续模型统一接入），非散点改写。

# 需求分析（required）

## 需求描述

将 magihuman DiT 全部层的 Q/K RoPE 应用替换为 NPU 融合算子路径，保持输出在任务书容差内、BSND/GQA 与位置偏移语义不变;封装 `module/rope.py` 公共入口（布局适配 + cos/sin cache 管理 + 融合算子调用 + eager 回退）;A3 实测端到端与 attention 段时延达成任务书降幅。

## 需求拆解

1. 现状审计：定位 RoPE 调用点与实现形态（拆半 vs 交错旋转、apply 于 Q/K 哪一侧、位置偏移规则），profiling 记录基线算子数与耗时。
2. 能力探针：实机验证目标融合算子对 BSND/GQA/head_dim 128/位置偏移切片的支持面与输出布局。
3. 公共封装：`module/rope.py`（cos/sin 预计算 cache、布局转换、融合调用、回退开关），magihuman 接入为首个调用方。
4. 精度：逐层 Q/K 旋转输出 cos-sim、端到端视频/音频质量对照;逐位不可期（算子内部实现差异），以容差口径验收。
5. 性能：RoPE 段耗时、attention block 耗时、端到端时延对照基线，门槛以任务书为准。

# 详细设计（required）

## 功能分析

### RoPE 语义（保持不变）

- 公式：`RoPE(x, pos) = x ⊙ cos(θ_pos) + rotate_half/interleave(x) ⊙ sin(θ_pos)`，θ 按 head_dim 128 与基频配置生成;实现形态（拆半/交错）与上游一致，替换不改数学语义。
- 位置规则：视频/音频/文本 token 共享序列位置（以现有 pipeline 的 position_ids 为唯一事实源），cache 化预计算按最大位置数一次性生成 BF16 表常驻。

### 支持范围

| 项 | 契约 |
| --- | --- |
| 布局 | 模型侧 BSND;融合算子如需 TND/BSH 由封装内转换，转换成本计入收益核算 |
| dtype | Q/K BF16;cos/sin 表 BF16 预计算（fp32 计算期仅构造时一次）;输出 BF16 |
| 形态 | 拆半/交错两种实现形态按模型现状接入;封装参数化，不隐式改语义 |
| 回退 | 融合算子不可用（探针失败/shape 不支持）时回退原算子链并记审计日志 |

## 接入与实现设计

### host 侧（工程接入）

1. **公共封装** `module/rope.py`：
   - `RopeCache(theta, max_pos, head_dim, dtype)`：cos/sin 表预计算与切片接口;
   - `apply_rope(q, k, cos, sin, layout, mode)`：布局校验 →（必要时）转换 → 融合算子调用 → 输出还原;`mode: fused | eager`，默认 fused、异常回退 eager。
2. **模型接入**：dit_module 中 RoPE 调用点改为 `module/rope.py` 入口;`config/rope_fused.yaml` 开关贯通（`model_args: rope-fused: true`），关闭时走原路径零行为变化。
3. **审计日志**：fused/eager 分支计数、布局转换开销、回退原因;正式运行不逐调用打印。

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 语义等价 | 单测：随机 Q/K 小规模对拍 eager 参考（fp64 host golden + fp32 交叉核对拆半/交错） | 容差内（任务书口径） |
| 精度 | 逐层旋转输出 cos-sim 抽层全量统计 + 同 seed 端到端视频/音频质量 | 任务书门限 |
| 性能 | ≥1 预热 + ≥3 正式;RoPE 段/attention 段/端到端三级耗时对照 | 降幅以任务书为准 |
| 可复用性 | 封装单测不依赖 magihuman;接口被第二个模型（如 wan 系列）冒烟接入 | 可独立使用 |
| 兼容 | 开关关闭逐位一致;与 matmul-int8/fa-mxfp8 组合用例 | 零回归 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| 融合算子布局/位置偏移契约与模型不一致（尤其交叉注意力文本段位置） | 能力探针先行;位置切片语义以 pipeline position_ids 对拍冻结 |
| 拆半与交错形态选错导致静默错位 | fp64 host golden 单测在先，形态参数显式配置并审计 |
| GQA 下 K/V 仅 8 head，融合算子调用批次小、收益打折 | Q/K 合并调用或 batched 接口探针验证;报告分别给出 Q/K 收益 |
| 表预计算内存（max_pos×128×2 表） | max_pos 按配置上限，表为 MB 级;动态超长场景回退在线计算 |
