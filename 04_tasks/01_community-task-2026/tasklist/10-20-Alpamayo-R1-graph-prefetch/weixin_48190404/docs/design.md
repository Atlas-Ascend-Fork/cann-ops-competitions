# 需求背景（required）

## 需求来源

依据下发《Alpamayo-R1 图模式与预取优化(A2, A3)》和 `test_cases/README.md`。

- 目标代码仓：[cann/cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai)
- 样例目录：`navigation/alpamayo-r1/`
- 技术文档：`docs/navigation/alpamayo-r1/README.md`
- 冻结 RFC：`cann/cann-recipes-embodied-ai` **#145**
- 代码实现 PR：`cann/cann-recipes-embodied-ai` **#211**（分支 `feature/alpamayo-r1-graph-prefetch`）

提交人：weixin_48190404。任务：10月社区任务—Alpamayo-R1 图模式与预取优化。

## 背景介绍

### 样例现状

仓内 Alpamayo-R1 样例基于固定上游提交 `e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073`
（NVlabs/alpamayo），通过 `patches/ar-1.patch` 完成昇腾适配，已在 Atlas A2 上接入融合 Attention、
RMSNorm、`addcmul`、索引和内存分配等算子级优化。样例文档记录的 7.32 s 为历史参考值。

样例推理分两个阶段：

| 阶段 | 特征 | 是否可入图 |
| --- | --- | --- |
| VLM 自回归 rollout | 序列长度逐步增长；含 top-p/temperature 随机采样；`CoC` 文本生成 | ❌ 动态 seqlen + 随机算子 |
| diffusion action expert 去噪前向 | **10 个去噪步的形状 / dtype / 设备完全一致** | ✅ 稳定核心子图 |

### 本次优化目标

在不改变模型语义、权重、精度类型、输入、扩散步数、轨迹数量及 CoC 采样参数的前提下：

1. 将 diffusion 去噪段这一稳定子图接入图模式，实现真实的 compile/capture 与 replay；
2. 对 profiling 证明为 memory-bound 且存在安全依赖窗口的 MatMul 类权重热点使能预取。

两个开关（`enable_graph`、`enable_prefetch`）**默认均为 false**，关闭时不改变任何执行路径与输出。

### 技术难点预判

1. **图边界受限于随机采样**：VLM rollout 无法入图，收益上限由 diffusion 段占比决定；
2. **attention 算子的图兼容性**：既有实现经 `npu_fusion_attention` 调用融合 attention，
   该算子在图模式下可能不可用，需在图内替换为图可用的等价算子，且必须保证数值逐位一致；
3. **预取前提未必成立**：是否 memory-bound 需以 profiling 证据判定，不得预设；
4. **双平台一致性**：A2 与 A3 需使用同一套软件栈，且两平台结果独立报告、不跨平台计算收益。

# 需求分析（required）

## 需求描述

在 Alpamayo-R1 样例上交付图模式 wrapper、静态 buffer、预取插入与回退逻辑，形式为
**monkey patch（不改上游模型代码）**，并提供 `enable_graph`、`enable_prefetch` 两个独立开关
（默认关闭）、A2/A3 运行配置、测试与复现入口、README 与优化文档。

## 需求拆解

1. **开关与回退**：两个开关独立、默认 false；关闭后解除对应包装、恢复原调用路径；异常时可观测回退。
2. **四档覆盖**：eager（baseline）、graph-only、prefetch-only、graph+prefetch，四档共享权重、数据、
   数学路径与采样配置。
3. **图模式真实性**：至少一个稳定核心子图真实编译/捕获并 replay，有 `compiles`/`calls`/
   `eager_fallbacks` 计数证据；无持续 graph break 或异常重编译。
4. **预取真实性**：开关开启后预取**真实执行**（调用数需与 `层数 × 去噪步数 × 请求数` 自洽），
   无生命周期错误、无破坏收益的隐藏同步。
5. **精度不退化**：同平台固定输入与 seed 下，各优化档原始轨迹 Tensor 与 eager 逐元素一致，
   序列化轨迹与 CoC 文本一致；100 clips 的 1/3/5/10 轨迹 minADE 不劣于同平台 eager 且 ≤ 0.5676。
6. **性能收益**：双平台的 graph+prefetch 稳态端到端耗时均低于各自 eager 基线，且改善**超过基线波动**。
7. **稳定运行**：连续不少于 20 次推理无失败、无内存持续增长、无输出漂移。
8. **双平台同栈**：A2 与 A3 使用同一套软件栈（CANN / torch / torch_npu / python 等版本一致）。

# 详细设计（required）

## 方案分析

### 图边界选择

以相位拆分与 kernel 时间线为证据选择子图。design 阶段锁定的候选为 **diffusion action expert 的
去噪前向**，其静态签名如下（10 个去噪步恒定）：

```
x                   (10, 64, 2)     float32
t                   (10, 1, 1)      float32
future_token_embeds (10, 64, 2048)  float32
prompt cache        seq_len = 3027  （每步 crop 回 3027，恒定）
is_causal           False（expert_non_causal_attention = True）
expert              36 层 / hidden 2048 / intermediate 8256 / heads 16 / kv_heads 8 / head_dim 128
```

### 图可用的 attention 算子

`npu_fusion_attention` 在两条图模式路上均不可用（aclgraph 内部切流、GE 无对应实现），
故图内 attention 必须替换为图可用的等价算子。替换点定位在**各布局的统一入口**，
覆盖 **BSND（4-D）** 与 **TND/varlen（3-D + cu_seqlens）** 两种调用形态，
**causal 分支原样保留**（VLM 自回归使用带 mask 的 causal 路径），以确保 CoC 文本不变。

替换属**共享适配层**（四档共用），目的是让"优化档与 eager 逐元素一致"可被严格验证：
四档走同一 kernel，差值只会来自图/预取本身，而不会混入算子差异。

### 预取目标选择原则

目标与系数**由 profiling 决定**，不在 design 阶段预设：
- 先以访存 / 计算 / 时间线证据判定目标权重是否为 memory-bound；
- 预取位置须在权重就绪之后、计算消费之前，且存在可重叠计算窗口；
- 从**单目标 + 保守尺寸**起步（平台规范：单点验证、先保守后激进），确认有效后再扩展；
- 不按验收 clip 或输入内容特判。

## 实现方案

### host 侧（图外准备）

1. **静态 buffer**：按首次真实调用的 shape/dtype/device 预建输入/输出 buffer，每请求只做 `copy_`；
   图内 attention mask 在无 mask 场景下不建 buffer。
2. **prompt K/V 静态化**：通过只读 cache 包装层把 prompt K/V 暴露给子图，`update()` 只做拼接、
   从不写回原 cache；调用方的 `crop(prefill_seq_len)` 语义保持不变。
3. **随机状态**：编译/捕获前后对 RNG 状态做快照与还原，保证 capture 不消耗随机序列、replay 可复现。
4. **workspace 与内存池**：优先使用图引擎自身的 workspace；显式捕获路线使用共享内存池，
   并记录捕获用的流与内存池标识。
5. **图缓存**：按**输入签名**管理，限制生命周期；关闭开关时解除包装并释放缓存。
6. **配置与校验**：四档配置集中管理（`configs/a2.yaml`、`a3.yaml`、`baseline.yaml`），
   对非法组合（如预取未配多流、未知配置键）在构造期直接拒绝。

### device 侧（图内子图）

1. 图内仅包含 diffusion 去噪前向，VLM rollout 保持 eager；
2. 每个请求把新数据写入稳定地址后 replay，replay 完成后在 buffer 下一次复用前消费输出；
3. 编译后立即用同一组静态输入跑一次 eager 作参照并逐元素比对，不一致则**自动回退 eager**
   并记录 `verification_failures`，避免静默产出错误轨迹；
4. 结构化统计输出：`compiles` / `calls` / `eager_fallbacks` / `capture_failures` /
   `verification_failures` / `compile_seconds` / `replay_seconds` / `static_buffer_bytes`。

### 预取插入

在 decoder layer 的**输入处**插入（目标算子的直接前驱、同一关键路径），依赖节点为本层输入
hidden states；`size = numel × itemsize × ratio`，使用独立流并与计算重叠，
不引入隐藏的 Host/Device 同步；提供 `skipped` 计数以便核对插入点是否全部命中。

### 平台能力适配

图模式的可用性与 CANN/SoC 相关，故不预设某平台一定支持某种调用形态：
以**能力探测 + 可观测回退**实现（例如图内 attention 的入参集合按平台能力自动降级并缓存），
两平台最终都必须满足 `bitwise_equal = True`。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 / `Ascend910B3`（64 GB HBM） | √ |
| Atlas A3 / `Ascend910_9382`（64 GB HBM） | √ |

软件栈（两平台**完全一致**，由 RFC #145 锁定）：CANN 9.0.0、`npu-smi` 记录实际版本、
Python 3.11.15、torch 2.8.0+cpu、torch_npu 2.8.0、torchvision 0.23.0、transformers 4.57.1、
numpy 2.4.2，另需 `protobuf<5`、`attrs`。**不锁死具体 A2/A3 型号与驱动版本**。

## 约束限制

1. **图边界**仅覆盖 diffusion 去噪段；VLM 自回归 rollout 含随机采样算子，按平台规范不可入图，
   端到端收益存在结构性上限。
2. 图模式需将 prompt K/V 静态化，**额外占用显存**；64 GB 为设备总容量，10B 权重之外仍有
   KV cache、激活与图 workspace，须实测容量。
3. **首次编译耗时显著高于稳态**，须单列报告，不与稳态混淆。
4. **图按输入签名编译**：prompt 长度变化会触发新图，在线服务需按活跃签名数配置图缓存。
5. 不改权重、精度类型、轨迹数、扩散步数、数据集或采样参数换取收益；
   不新增量化、模型并行或自定义算子。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | ① 同平台固定输入与 seed 下，graph-only / prefetch-only / graph+prefetch 的原始轨迹 Tensor 与 eager **逐元素一致**，序列化轨迹与 CoC 文本一致；② 100 clips 的 **1/3/5/10** 轨迹 minADE **不劣于同平台 eager**，且 10 轨迹 **≤ 0.5676**；③ `failed=0`，不替换或筛除失败 clip | 任务书 §3 / §2.3 |
| 性能标准 | ① 每档预热 ≥1 次、正式运行 ≥3 次（计划扩大至 ≥10 次）；② 记录**首次编译耗时**与**稳态端到端耗时**；③ **A2/A3 的 graph+prefetch 稳态端到端均低于各自 eager 基线，且改善超过基线波动**；④ 不用历史 7.32 s 替代本机实测基线 | 任务书 §3 / §4.1 |
| 稳定性标准 | 连续不少于 20 次推理：无失败、无内存持续增长（显存斜率判定）、无输出漂移 | 任务书 §3 / test_cases |
| 内存标准 | 开启 `profile_memory` 的独立 profiling 运行，记录峰值、稳态占用与连续请求趋势；报告图缓存与 workspace 额外开销 | 任务书 §4.1 |
| 错误返回标准 | 精度、配置、运行或比较失败必须返回**非零状态**，不吞异常、不将无法比较记为通过 | 任务书 §4.1 |

**seed 与聚合口径**：每个评测 arm 在 clip 循环之前设置一次 seed，不在每个 clip 重置；
100 clips 的 minADE 为**等权平均**，单采样时间点、单轮运行、无过滤。

**性能测试方法**：同设备独占、相同同步边界；下载/权重加载/首次编译单列；
性能运行与 profiler 诊断运行分离；**同一进程内交错测量四档**，避免启动顺序与温度影响。

## 兼容性分析

| 维度 | 分析 |
| --- | --- |
| 上游兼容 | 不改上游模型代码；改动以 monkey patch 交付，未新增 `.patch` 文件（仅在既有 `ar-1.patch` 上追加入口接入段） |
| 公共代码影响 | 不修改公共 `module/` 或框架代码，故无需对既有样例做回归 |
| 默认行为 | 两个开关默认 false，关闭时执行路径与输出与原样例一致 |
| 硬件兼容 | A2 与 A3 双平台验证；平台差异以能力探测 + 回退处理，不维护独立代码分支 |
| 软件栈兼容 | 由 RFC #145 锁定 CANN / torch / torch_npu / python（及 transformers、numpy）版本；两平台同栈 |
| 输入兼容 | 保留 1/3/5/10 轨迹入口与原有输出格式；不改变默认输入与采样参数 |
| 可维护性 | 图模式统计、预取统计、三层输出比对与平台探针均随代码交付，可独立复现与定位 |
