# 需求背景（required）

## 需求来源

社区任务 10-20「10月社区任务-Alpamayo-R1图模式与预取优化」，在 A2/A3 上基于 [cann/cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai) 的 Alpamayo-R1 样例（`navigation/alpamayo-r1/`）完成图模式接入和预取接入。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，验收以任务书为准。

## 背景介绍

### 样例现状

`navigation/alpamayo-r1/` 当前为最小骨架（`README.md` + `patches/`）。Alpamayo-R1 为自动驾驶 VLA/世界模型系样例：多传感器输入（相机/自车状态）→ 感知-预测-规划推理链。当前以 eager torch_npu 推理为主，存在两类开销：

1. **调度开销**：eager 模式逐算子下发，小算子密集段（norm/逐元素/attention 投影）Host 调度占比高;
2. **数据供给开销**：传感器数据预处理与 H2D 拷贝串行在推理前，GPU 等待数据。

任务两项：**图模式接入**（torchair/npu_graph 静态化，消除调度开销）、**预取接入**（数据异步 H2D 流水，掩盖供给开销）。

# 需求分析（required）

## 需求描述

Alpamayo-R1 在 A2/A3 上支持图模式推理与数据预取：图模式覆盖推理主链（感知-预测-规划），静态 shape 分桶或固定输入契约;预取使下一帧数据供给与当前帧推理重叠;输出与 eager 路径一致（容差内），端到端时延下降满足任务书门槛;A2/A3 双平台实测。

## 需求拆解

1. 图模式：推理主链静态化（shape 契约/分桶策略确定），torchair（或 npu_graph，以实机版本能力为准）接入，eager 回退开关。
2. 预取：dataloader 预处理与 H2D 异步流水（pin_memory + non_blocking + 独立拷贝流 + event 依赖），深度可配。
3. 一致性：图模式 vs eager 输出对拍（固定输入 allclose/容差），预取不改变数值。
4. 性能：端到端时延、单帧推理耗时、CPU 占用三档对照（eager / graph / graph+prefetch），门槛以任务书为准。
5. 交付：开关贯通、README 复现、A2/A3 双平台结果表。

# 详细设计（required）

## 功能分析

### 图模式契约

| 项 | 契约 |
| --- | --- |
| 覆盖范围 | 推理主链（视觉编码器 + 模型主体 + 解码头）；数据预处理留在图外（CPU） |
| shape 策略 | 输入分辨率/序列长度固定的场景直接静态;多分辨率场景按桶（bucket）编译多图，运行期按 shape 选图，未命中桶回退 eager 并计数 |
| 随机性/动态控制流 | 图内不含数据依赖的控制流与随机采样；采样类后处理留图外 |
| 缓存 | 编译产物按 shape 桶缓存复用；首帧编译时延单列不计入稳态性能 |

### 预取契约

- 流水：预处理（CPU 线程/进程）→ pinned buffer → 专用拷贝流 H2D（non_blocking）→ event → 推理流 wait event 使用；深度 N 可配（默认 2）;
- 一致性：预取只改变"何时拷贝"，不改变数值与顺序；buffer 池按桶尺寸预分配，循环复用带版本号，防覆盖未消费数据;
- 降级：pinned 分配失败/拷贝流异常时回退同步路径并记审计日志。

## 接入与实现设计

### 工程组织

```text
navigation/alpamayo-r1/
├── patches/                    # 既有补丁目录（沿用）
├── infer_with_torch/
│   ├── graph_mode.py           # torchair/npu_graph 封装：构建、分桶、选图、回退
│   ├── prefetch.py             # 预取管线：buffer 池、拷贝流、event、版本号
│   ├── infer.py                # 推理入口（--graph --prefetch-depth N 开关）
│   └── bench.py                # 三档对照基准脚本
├── config/graph_prefetch.yaml  # 开关与参数
└── README.md                   # 复现命令与结果表
```

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 一致性 | 固定输入集：图 vs eager 逐输出 allclose/容差；prefetch vs 同步逐位一致 | 任务书容差内 |
| 功能 | 多分辨率分桶选图、桶未命中回退、预取深度 1/2/4 | 行为有定义 |
| 性能 | 三档 ×（≥1 预热 + ≥3 正式）：端到端时延、单帧耗时、CPU 占用、峰值显存 | graph 与 prefetch 降幅分别达任务书门槛 |
| profiling | 调度 gap（图模式应显著缩短）、数据等待 gap（预取应趋近 0） | 拆解报告 |
| 兼容 | 全关开关 = 上游 eager 行为；A2/A3 双平台各自验证 | 零回归、双平台通过 |
| 稳定性 | 连续长跑（数百帧）无泄漏/无递增时延 | 通过 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| 模型含动态控制流无法整图静态化 | 控制流抽到图外（Python 侧分支），图只收静态主链；审计清单先行 |
| torchair 与 npu_graph 能力随版本差异 | 实机能力探针冻结选型；封装层隔离后端，切换不改调用方 |
| 分桶过多导致编译产物爆炸 | 桶集合白名单化（按实测输入分布），未命中回退 eager 并统计命中率 |
| 预取 buffer 被未完成推理覆盖 | 版本号 + event 生命周期断言；debug 模式下越界直接报错 |
| 图模式与既有 patches 冲突 | patches 分层加载顺序固定；组合冒烟入 CI |
