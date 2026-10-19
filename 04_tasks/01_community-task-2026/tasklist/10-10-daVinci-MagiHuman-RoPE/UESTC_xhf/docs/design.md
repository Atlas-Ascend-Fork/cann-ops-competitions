# 需求背景（required）

## 需求来源

- 任务：10月社区任务-daVinci-MagiHuman RoPE优化（A3）
- 任务 ID：`f144dedf075b451e830f4a21ab4ddc89`
- 任务页面：https://www.hiascend.com/activities/task-center/details/f144dedf075b451e830f4a21ab4ddc89?menu=tasks
- 代码仓：https://gitcode.com/cann/cann-recipes-infer
- 代码 PR：https://gitcode.com/cann/cann-recipes-infer/pull/1105

## 背景介绍

### daVinci-MagiHuman partial RoPE 现状

daVinci-MagiHuman 的 base 路径在注意力层中使用 half rotary 形式的 RoPE。真实模型的
head dimension 为 128，参与旋转的维度为 96，因此属于 partial RoPE。原实现先对前
96 维调用 `torch_npu.npu_rotary_mul`，再通过切片和 `torch.cat` 拼回未旋转的 32 维
尾部。该实现数值语义正确，但每个注意力层的 Q/K 都会增加尾部拼接工作。

### 优化目标

`npu_rotary_mul` 提供可选 `rotate` 矩阵。任务目标是用一个带尾部单位块的旋转矩阵，
把 96 维旋转区和 32 维恒等区放进一次融合调用，减少 partial RoPE 路径的切片和拼接，
同时保持公共接口、模型权重、推理配置和输出语义不变。

# 需求分析（required）

## 需求描述

1. 对 96/128 partial RoPE 构造完整 128×128 块旋转矩阵。
2. cos 的非旋转尾部补 1，sin 的非旋转尾部补 0。
3. Q/K 均通过一次 `npu_rotary_mul(..., rotary_mode='half', rotate=matrix)` 完成旋转。
4. 旋转矩阵随模型设备迁移，但不写入 checkpoint。
5. 保留原有无 `rotate` helper 路径，避免扩大变更范围。
6. 在 Atlas A3 单卡上完成实际 shape、完整生成、性能、内存与 Profiler 验证。

## 需求拆解

1. 新增旋转矩阵构造函数和参数校验。
2. 在 `DiTModel` 初始化阶段注册 non-persistent buffer。
3. 在 forward 中把 RoPE cos/sin 扩展到完整 head dimension。
4. 把 Attention 的 RoPE 参数由二元组扩展为 cos、sin、rotate 三元组。
5. 保证 Python 3.9 可解析，并保留激活函数原有映射和异常行为。
6. 固定环境、输入、预热与正式轮次，比较基线和候选版本。

# 详细设计（required）

## 算子分析

### 数学公式

设输入最后一维为 `D=128`，旋转维度为 `R=96`，`h=R/2`。half rotary 对旋转区的
变换为：

```text
rotate_half([x0, x1]) = [-x1, x0]
out = cos * x + sin * rotate_half(x)
```

其中 `x0`、`x1` 的长度均为 `h`。构造矩阵：

```text
M = [[ 0,  I_h, 0],
     [-I_h, 0,  0],
     [ 0,   0,  I_(D-R)]]
```

并令尾部 `cos=1`、`sin=0`，则：

```text
out = cos_full * x + sin_full * (x @ M)
```

前 96 维与原 half rotary 等价，后 32 维恒等传递。

### 支持数据类型

本模型路径使用 bfloat16。矩阵 dtype 与传入的 RoPE 张量匹配；实现接口仍接受调用者提供的
`torch.dtype`，便于独立数值测试覆盖 float16、bfloat16 和 float32。

### 支持形状

- 模型主路径布局：`(batch, sequence, heads, head_dim)`。
- 实测 sequence：32、1024、4025、4102。
- 实测 Q heads：40；K heads：8。
- 主任务配置：rotary dimension 96，head dimension 128。

## 算子实现

### 实现方案

#### host 侧设计

1. `build_partial_rotary_matrix(rotary_dim, head_dim, dtype)` 检查旋转维度为正偶数且不超过
   head dimension；非法输入抛出 `ValueError`。
2. 用两个单位矩阵块构造 half rotary 的交换/符号区域，并在非旋转尾部写入单位阵。
3. `DiTModel` 根据 rope bands 计算 `rotary_dim`，把矩阵注册为
   `persistent=False` 的 buffer。它会随模型迁移到目标设备，但不会改变 checkpoint。
4. forward 中把 cos 拼接为 `[cos, cos, ones_tail]`，把 sin 拼接为
   `[sin, sin, zeros_tail]`，并保持连续内存。

#### 调用侧设计

1. `apply_rotary_emb_npu` 增加可选 `rotate` 参数。
2. 提供矩阵时，直接调用一次 `npu_rotary_mul` 处理完整 head dimension。
3. 未提供矩阵时，保持原 full rotary 和 partial rotary 分支，减少兼容风险。
4. Attention 对 Q/K 复用同一 cos、sin 和 non-persistent buffer。
5. 为满足仓库 Python 3.9 检查，激活函数选择使用等价 `if` 分支；SWIGLU7、GELU7 和
   未知类型异常行为保持不变。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A3，1×Ascend 910C | √ |

验证环境为 CANN 9.0.0、Python 3.12、torch/torch_npu 2.10.0。Python 3.9 语法兼容性通过
独立编译和激活映射检查。

## 算子约束限制

- `rotary_dim` 必须大于 0、为偶数且不超过 `head_dim`。
- 当前模型接入只改变 half rotary 的 96/128 partial 路径。
- 不修改公共函数参数、模型配置、权重、attention/linear 结构或推理步数。
- 后续若代码 PR 发生 rebase、squash 或修订，必须从新的最终 HEAD 重新构建和复测。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实际 shape 与基线一致，partial tail 保持；不得出现 NaN/Inf | 任务书与官方测试材料 |
| 性能标准 | 同机同配置复刻官方预热、采样、同步与统计口径；性能收益须大于基线波动 | 任务书与官方测试材料 |
| 内存标准 | 连续生成无泄漏，记录 HBM、allocated、reserved 与 RSS | 任务书与自验证模板 |

## A3 实测结果

最终代码 PR HEAD 为 `c1bb10c71829e6bb34011600d18b50ca625e9acf`，tree 为
`88b9efc2ede088d0f837b10070696961034ba4b4`，目标文件 blob 为
`d7b6734745a4afcd587a41ff4cf46abe16c2676d`。

### 精度与边界

- S=32/1024/4025/4102、H=8/40 的 8 组实际 shape 全部 bit-exact，max abs error 为 0。
- 非旋转尾部逐位保持。
- 6 个正向场景与 4 个非法 rotary dimension 场景通过。
- 公开 base 32-step smoke 在最终 c1bb HEAD 通过。

### 性能

同一 A3 会话中每个变体执行 1 次预热和 4 次正式生成。热稳态 round3、round4、round5
连续通过；以 round5 的较差结果作为最终报告值：

| 指标 | baseline | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| Total 中位数 | 45.0471388060 s | 44.7270138540 s | -0.3201249520 s（-0.71064436%） |
| base 主阶段中位数 | 43.5481505 s | 43.2391395 s | -0.3090110 s（-0.70958467%） |
| Total 基线极差 | 0.0435756840 s | — | 收益/极差=7.346413 |
| 主阶段基线极差 | 0.0466520 s | — | 收益/极差=6.623746 |

### 内存与 Profiler

- 同一 pipeline 连续 5/5 次生成通过，无持续增长。
- driver/HBM 离散峰值不变；reserved 中位数下降 10 MiB；allocated 增加 33,280 bytes，
  已在报告中披露。
- c1bb shape-aware Profiler 全部状态为 0，保存 84 个原始 profile 文件。
- 父版本同口径可比 Profile 中，RoPE 相关累计时间 38280.955→28879.043 us，下降
  24.560286%；调用数 924→444。

## 兼容性分析

- base、base_ti2v、distill、distill_ti2v、sr_540p、sr_540p_ti2v、sr_1080p、
  sr_1080p_ti2v 八个真实模式在语义等价父版本完成 A3 推理和媒体校验。
- c1bb 只把激活选择的 `match/case` 改写为等价 `if` 分支；最终 HEAD 已补跑激活映射、
  公开 base smoke、actual-shape/edge、性能、内存与 shape-aware Profiler。
- 三类固定 prompt 的 A/B 视频均有效且每对逐字节相同。任务要求的两名指定验收人独立
  签署属于提交前外部门禁，不以机器哈希替代。

## 证据与复现

c1bb 原始证据归档 SHA-256 为
`45578ffaf0537d1cb4dc05b331b1a5e7d4bcf36644a9597a080982812a194ce4`。远端与本地整包
哈希一致，内部清单 636/636 通过。代码 PR 已显示 `ci-pipeline-passed` 和
`cann-cla/yes`；设计文档仅引用摘要和公开链接，不提交模型权重、登录凭据或机器私有路径。

## AI 辅助声明

本文档在整理、核对和提交阶段使用 OpenAI Codex（GPT-6）辅助；全部功能、精度、性能、内存和 Profiler 指标均来自已归档的 Atlas A3 原始证据，并由提交者复核。
