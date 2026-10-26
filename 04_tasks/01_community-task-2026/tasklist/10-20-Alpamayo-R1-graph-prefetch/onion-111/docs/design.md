# Alpamayo-R1 图模式与预取优化设计

## 需求背景

### 需求来源

本设计对应 CANN 社区任务《10月社区任务-Alpamayo-R1图模式与预取优化》，面向 Atlas A2 与 Atlas A3 上的 Alpamayo-R1 推理样例。任务要求在保持模型输入、输出及默认行为的前提下，探索图模式与权重预取优化，并按任务书自验流程报告精度、性能、显存及运行环境。

### 背景介绍

Alpamayo-R1 推理包含视觉编码、语言模型解码和轨迹扩散。完整 rollout 有变长 token 生成、EOS 判断、动态 KV cache 生命周期及随机采样，不适合捕获成固定静态图。本方案只捕获满足固定输入形状和执行状态的局部计算，其余流程保留 eager。预取只作用于 decoder layer 0 的 MLP 权重读取，不改变数学运算。

## 需求分析

### 需求描述

提供可选的图模式与预取开关，支持独立或组合启用且默认关闭。优化只安装在当前 Alpamayo-R1 模型实例上。遇到不支持的设备、输入或运行状态时回退到原始实现；无法安全回退的异步设备错误向上传播。

### 需求拆解

1. 视觉路径仅尝试捕获 vision block 0 的 MLP，attention 与其他视觉 block 保持 eager。
2. 在受控的原始视觉 forward 中使用固定 grid 元数据；设备 grid 与 CPU grid 的完整视觉输出不完全一致时，不启用该路径。
3. 对 decoder layer 0 MLP 的 down_proj.weight 提交只读 CMO；不新增计算、数据复制、同步或额外 stream。
4. 固定输入下对比 eager 与优化输出；不修改采样参数、数据处理或默认 test_inference.py。
5. 提供实例级安装/关闭句柄和运行统计，记录图捕获、重放、预取提交及回退原因。

## 详细设计

### 方案分析

实现入口为 alpamayo_r1.optimization.install_optimizations。调用方显式传入 enable_graph 和 enable_prefetch；两项默认 false。原有模型调用方式与默认推理入口不变。

| 模式 | 图模式 | 预取 | 执行路径 |
|---|---:|---:|---|
| baseline | 关闭 | 关闭 | 原始 eager |
| graph-only | 开启 | 关闭 | 视觉 block 0 MLP 图及受控元数据缓存 |
| prefetch-only | 关闭 | 开启 | decoder layer 0 MLP 权重预取 |
| graph+prefetch | 开启 | 开启 | 同时启用以上两项 |

### Host 侧设计

#### 图模式与边界

视觉 forward wrapper 检查模型处于 eval、未启用梯度、输入 grid 形状和类型受支持且位于 NPU。首个请求将设备 grid 转为 CPU 值，并在修改视觉 MLP 前分别使用设备 grid 与 CPU grid 执行原始完整视觉 forward；仅当最终输出及 deepstack 输出逐项完全一致时，才捕获局部图。

通过 torch_npu.npu.NPUGraph 捕获 vision block 0 的原始 MLP。attention、其余视觉 block、输出拼接和完整 rollout 保持原样。每个视觉实例只为一个已验证输入签名创建固定 buffer；捕获后的首次 replay 与 eager 参考输出逐项比较，通过后才复用。

#### Guard 与回退

重放前检查输入 shape、stride、dtype、device、stream、autocast 状态、模块结构、hooks、权重与 buffer 的对象和版本。grid 值变化、guard 失败或调用状态不适用时执行完整原始视觉 forward。捕获或 replay 发生设备错误时，不在可能残留部分设备任务的状态下静默重试。

元数据方法只在原始 visual.forward 的受控调用点读取固定 grid；方法、调用顺序、张量 identity/version 或 hook 检查不通过时关闭优化并回退。外部直接调用元数据方法仍使用原实现。关闭句柄时恢复本实例 wrapper 并释放图和缓存资源。

#### 预取插入点

只选择 decoder layer 0 MLP 的 down_proj.weight，最多预取 2 MiB。CMO 在原始 MLP forward 入口、当前 stream 上提交，之后按原顺序执行 gate/up、激活、逐元素乘法和 down projection。仅在 eval、无梯度、NPU 单 token (batch, 1, hidden) 输入、权重 contiguous 且输入与权重同设备时提交；prefill 和其他不匹配调用走原始 forward。

当前 torch_npu.npu_prefetch 实现忽略第二个参数，因此传入输入张量不构成显式生产者依赖。方案依赖同一当前 stream 上先提交 CMO、后执行原始 MLP 的顺序、只读权重及设备实现语义，不创建额外 stream 或同步。提交错误向上抛出，不重试可能部分提交的请求；若平台无法证明或支持该语义，则关闭此路径。

### Device 侧设计

图捕获与重放由 torch_npu.npu.NPUGraph 在当前 NPU 后端执行，输入和输出 buffer 在捕获前准备并保持生命周期稳定。图路径不包含随机采样、动态 token 生成、EOS 判断或 KV cache 更新。

预取通过 torch_npu.npu_prefetch 对目标线性层权重提交只读 CMO；原有线性层算子和权重访问保持不变。预取 API 次数仅表示 Host 提交次数，设备侧执行需要通过 NPU profiler 事件关联验证，不能将 Host 计数当作设备执行或有效重叠。

### 支持硬件

| 平台 | 验证要求 |
|---|---|
| Atlas A2 | 使用配套驱动、固件、CANN 与 torch-npu 环境实测 |
| Atlas A3 | 使用配套驱动、固件、CANN 与 torch-npu 环境实测 |

API 可导入不等于平台通过自验。A2、A3 分别记录设备与软件版本，并独立运行验证矩阵。

### 约束限制

- 优化默认关闭，仅影响显式安装的模型实例。
- 每个视觉实例最多保留一套图；不同 shape/stride/stream/autocast 签名不扩张图池。
- 训练、有梯度、非 NPU 输入、动态 grid、未验证 kwargs、模型结构或权重变化时不使用图优化。
- 预取仅面向 eval 的单 token decode MLP；不覆盖 prefill、多 token 输入或不连续/异设备权重。
- 图模式增加图内存，峰值显存按任务自验记录。
- CMO 同流顺序只说明提交顺序，不能单独证明带宽收益、缓存命中或有效重叠。
- 图和预取均不得改变 eager 数学路径、输入数据、随机种子、采样设置或评测指标。

## 可维可测分析

### 精度标准与测试方案

严格按任务书自验说明，在 A2 和 A3 分别对 1、3、5、10 条轨迹执行 baseline、graph-only、prefetch-only、graph+prefetch 四种配置。固定同平台模型权重、数据、seed、BF16、top-p、temperature、最大生成长度和扩散步数；计时不含模型加载、数据下载、输入预处理、磁盘读写和结果序列化。100 clips 精度评估使用指定 chunk_0000 数据与默认评测口径。

检查原始 xyz/rot 与 CoC 输出相对同平台 eager 的一致性，并按任务定义计算 minADE。10 轨迹门槛为 minADE ≤ 0.5676；每个平台独立报告。失败项保留原始日志并如实标记，不以单 clip 或其他轨迹数替代。

### 性能与资源标准

分别报告冷启动、捕获时间、稳态多次测量和中位数，记录每次 allocated/reserved/peak 显存。graph+prefetch 按任务书判据与同平台 eager 比较。四档消融用于定位差异；没有预取单项收益证据时，不把组合收益归因于预取。

使用 NPU profiler 检查图 replay、目标 MatMul、CMO 的 stream、任务和时间关系，并说明 profiler 版本及设备事件统计范围。区分 Host wrapper 时间、Host API 提交数和设备侧任务；不以 Python 调用次数替代设备执行证据，也不根据单一 PMU 计数推导实测 HBM 带宽。

### 兼容性分析

默认关闭时保持原始 eager 行为。实现通过实例级 monkey patch 接入，不修改模型公共类或共享模块；关闭句柄恢复原实例方法。A2 与 A3 使用各自支持的驱动、固件、CANN 与 torch-npu 组合独立验证并记录环境。guard 不满足时回退 eager；捕获/replay 或预取提交的异步设备错误按错误处理，不计作优化通过。