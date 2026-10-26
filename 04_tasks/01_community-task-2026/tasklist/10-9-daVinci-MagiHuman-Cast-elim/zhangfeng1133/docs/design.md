# 需求背景（required）

## 需求来源

社区任务 10-9「10月社区任务-daVinci-MagiHuman冗余Cast算子消除」，在 Atlas A3 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 消除 daVinci-MagiHuman 推理链路中的冗余 Cast 算子。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，验收以任务书为准。

## 背景介绍

### 问题定义

daVinci-MagiHuman 为 BF16 主体模型，但链路中大量标量/小张量（位置编码、timestep、RoPE cos/sin 表、norm 的 fp32 累计路径、CFG 分支常量）与 NPU 算子 dtype 契约交互，产生三类 Cast：

| 类别 | 特征 | 处置 |
| --- | --- | --- |
| 冗余 | dtype 已与下游一致、或上游刚 cast 过又 cast 回（A→B→A 链）、或常量可在 Host 预转 | **消除** |
| 可合并 | 同一位置重复 cast 相同张量（多个消费方各自 cast） | **提升/共享，一次 cast** |
| 必要 | 算子 dtype 契约要求（如 norm 的 fp32 计算）、数值语义需要 | **保留** |

冗余 Cast 的代价：额外 kernel launch、多余 GM 读写、打断了下游可融合的机会。任务目标是在**不改变数值语义**（逐位或任务书容差内）的前提下消除冗余项。

### 仓内现状

`models/davinci-magihuman/`（`inference/model/dit/`、`inference/pipeline/`、`config/`）无 Cast 审计工具与 dtype 治理约定;同类先例：任务目录 `10-daVinci-MagiHuman-Cast-elim-A3` 已有他人设计稿（Light_Uu），本设计与其在方法上可对照、实现独立。

# 需求分析（required）

## 需求描述

建立"审计 → 分类 → 消除 → 回归验证"闭环：以 profiler/图 dump 为事实源统计 Cast 清单，分类后逐项消除冗余/合并可共享项，保持输出与基线一致（任务书容差内），Cast 算子数量与端到端时延达成任务书降幅，A3 实测交付。

## 需求拆解

1. 审计工具：脚本化采集指定 step 区间的算子清单（msprof/op summary），产出 Cast 清单（位置、入出 dtype、shape、频次、所属模块）。
2. 分类与消除：按上表三类逐项处理——Host 预转换（常量/表类）、模块边界 dtype 统一（消除边界往复）、消费方共享 cast（提升到公共上游）。
3. 数值安全：消除前后 block 级输出对拍（cos-sim + 逐位统计），必要项保留清单化。
4. 验收：Cast 数量降幅、端到端时延降幅达任务书门槛;BF16 路径行为零变化;README 与审计报告交付。

# 详细设计（required）

## 功能分析

### 审计方法

- 固定 prompt/seed/分辨率，跑 N 步去噪，采集 [step0, step1, stepT-1] 三个切片（首步含缓存/表初始化，须与稳态分开统计）;
- 产出矩阵：`Cast 调用点 × (模块, 输入 dtype, 输出 dtype, shape, 每步频次, 每次字节数)`;稳态每步重复的 cast 为主攻对象，一次性初始化类 cast 不计收益。

### 消除手法（映射到具体代码点）

| 手法 | 适用点（按模型结构预判） | 实现位置 |
| --- | --- | --- |
| Host 预转换 | RoPE cos/sin 表、position_ids、timestep 向量、CFG scale 等常量/低频更新量 | pipeline/模块构造期一次转成消费 dtype 常驻 |
| 边界 dtype 统一 | block 入口/出口因子模块 dtype 不一致产生的往复 cast | dit_module 边界契约:统一 BF16 进出，内部 fp32 仅限 norm 累计 |
| 共享提升 | 同一 hidden 被多个下游各自 cast | 提升到分支点一次 cast，下游直用 |
| 算子替换 | 上游算子自带 out_dtype 参数却走 cast 的（如 mul/silu 后 cast） | 改用带 out_dtype 的算子重载 |

### 支持范围

仅推理链路（pipeline + DiT + t5_gemma + vae2_2 的调用边界）;训练/反向不在范围。消除不得改变随机数消费序（不引入/删除任何依赖全局 RNG 的算子），保证同 seed 逐位可复现基线。

## 接入与实现设计

### host 侧（工程接入）

1. **审计脚本**：`scripts/cast_audit.py`（采集 + 分类 + 报告 markdown/CSV 落盘），基线报告随 PR 附上。
2. **代码改造**：按手法逐点 PR 内聚修改;每处消除在代码旁注明审计编号（对应报告行），可回溯。
3. **守护**：`config/cast_policy.yaml`（白名单必要 cast、消除项开关），开关关闭时可一键回到基线行为做 A/B。
4. **回归**：三档对拍——逐位（应逐位一致的子集）、容差（block cos-sim）、端到端（同 seed 视频/音频质量指标）。

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 数量 | 审计报告对照:稳态每步 Cast 调用次数 | 降幅达任务书门槛 |
| 时延 | ≥1 预热 + ≥3 正式，端到端与 DiT 段耗时 | 降幅达任务书门槛 |
| 数值 | 逐位子集 + block cos-sim + 同 seed 端到端质量 | 任务书容差内 |
| 兼容 | 白名单外无行为变化;开关关闭回基线 | 零回归 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| 误消"必要"cast 引入数值漂移 | 白名单机制 + 逐位/容差双重对拍;存疑项默认保留 |
| 消除后触发算子 dtype 契约报错 | 每处消除独立开关，A/B 定位;回退不连锁 |
| profiler 口径争议（kernel 名含 Cast 的杂项） | 报告同时给 op 数与字节数两个口径，评审冻结主口径 |
| 收益被 launch 开销噪声掩盖 | 以 N 步稳态累计均值统计，多次运行取中位数 |
