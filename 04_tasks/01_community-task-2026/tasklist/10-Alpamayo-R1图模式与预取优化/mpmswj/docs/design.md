# Alpamayo-R1 图模式与预取优化设计方案（A2/A3）

# 需求背景（required）

## 需求来源

对应 HiAscend 任务“Alpamayo-R1 图模式与预取优化”：
https://www.hiascend.com/activities/task-center/details/9c55c937110d43af8c0bd984012bcda6

## 背景与目标

Alpamayo-R1 推理包含视觉语言模型生成和轨迹生成等阶段。小算子的重复调度、动态执行边界以及 MatMul 权重搬运可能限制端到端性能，拟分别通过稳定子图 capture/replay 和权重预取降低相关开销。

在保持模型权重、BF16、输入、Attention、batch 形状、采样参数和轨迹数不变的前提下，提供独立且默认关闭的图模式与预取开关，面向 A2/A3 验证精度、收益及稳定性。修改范围限定为 `navigation/alpamayo-r1` 及对应优化文档，不修改公共框架或其他样例。

# 需求分析（required）

## 基线与约束

- 软件基线拟固定为 Recipes `db9b2453ad70ac919f8c42e02a61d16db63f4e66`、NVlabs/alpamayo `e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073`；模型使用 `nvidia/Alpamayo-R1-10B`，固定 revision 并核对权重分片哈希。
- 目标平台为 Atlas A2/A3；拟统一 Python 3.11.13、CANN 9.1.0、torch 2.8.0、torch-npu 2.8.0、Transformers 4.57.1 等依赖，记录设备、驱动及完整环境指纹。
- 数据拟采用 PhysicalAI-Autonomous-Vehicles 25.10 的固定版本，按官方 `chunk_0000` 锁定 100 clips，保存 clip 顺序、时间点及清单哈希，禁止按指标挑选样本。
- 精度评估分别独立生成 1/3/5/10 条轨迹；默认 10 条轨迹、10 次扩散步、`top_p=0.98`、`temperature=0.6`、`max_generation_length=256`、`t0_us=5100000`。
- 数据集验证在整轮开始初始化一次 seed 42；预热保存并恢复 RNG。性能重复实验采用固定的逐请求 RNG 协议，分别标记，避免混用不同协议的结果。

## 需求拆解

1. 为适合静态执行的核心子图提供捕获、回放、关闭及回退能力，保留动态 Attention/KV cache 的原始语义。
2. 通过 profile 识别受权重访存限制的 MatMul，并在有真实依赖和安全时间窗口的位置插入预取。
3. 提供 eager、graph、prefetch、combined 四档，在相同计时边界下分别评估收益和交互影响。
4. 对同平台、同后端、同输入与采样协议进行严格输出比较，覆盖重复调用、状态变化及内存稳定性。

# 详细设计（required）

## 配置与接口

拟通过 `configure_npu_optimizations` 配置样例实例，并由 benchmark CLI 映射至相应参数。

| 配置 | 默认值及设计约束 |
| --- | --- |
| `enable_graph` / `--enable-graph` | 默认关闭；关闭时恢复该实例由优化逻辑安装的包装 |
| `enable_prefetch` / `--enable-prefetch` | 默认关闭；可独立于图模式启用 |
| `graph_boundary` / `--graph-boundary` | Python 默认 `norm-mlp`、CLI 默认 `mlp`；方案验证显式选择 `decoder-split`，避免依赖默认值 |
| `action_full` / `--graph-full-action` | 默认关闭；允许单独选择 Action 完整子图 |
| `graph_targets` / `--graph-targets` | 默认 `action,expert,vlm`，允许限制优化目标 |
| `cache_vision_positions` / `--cache-vision-positions` | 默认关闭；启用时要求同时开启图模式 |
| `prefetch_bytes` / `--prefetch-bytes` | 初始配置 1 MiB、offset 0；按权重边界截断并满足 512 B 对齐 |
| `copy_output` / `--borrow-graph-output` | Python 默认复制输出；CLI 显式借用时取反映射，仅用于受约束的内部激活 |
| `alias_graph_inputs` / `--alias-graph-inputs` | 默认关闭；仅允许匹配的活动生产者输出别名，并要求借用内部输出 |
| `expert_attention_backend` / `--expert-attention-backend` | 默认 `original`；主方案保留原 Attention，不以切换后端代替图优化 |

Python 与 CLI 均校验非法组合。严格验证模式下，捕获或状态恢复失败应返回非零，避免将失败静默计为成功。接口应支持实例级配置与恢复，不修改模型类的全局行为。

## 图模式与生命周期

拟采用 `decoder-split` 边界，将 RMS/QKV/norm/RoPE 与 projection/residual/postnorm/MLP 分段捕获；Attention 与 KV cache 调用保留在图外。Action 分支可捕获 Fourier/MLP/LayerNorm 组成的稳定子图。大 prefill 或不支持的输入签名回退至原执行路径。

在图外准备静态输入、输出及 workspace。保存和恢复预热相关 RNG，使用真实输入校验捕获路径的参考输出，并分别记录 warmup、capture、replay 和回退。

图缓存 key 包含 shape、stride、dtype、device、autocast、stream 及生产者身份；参数或 buffer 地址、版本、结构、forward 和 hook 变化应触发失效或回退。普通 wrapper 最多缓存 4 个签名，避免持续重捕获与无界显存增长。

默认输出归调用方所有。借用输出和输入别名仅适用于同流、同步单线程且立即消费的内部激活；最终轨迹保持自有存储，并通过跨请求保留旧输出的用例检查覆盖风险。

## 有界视觉位置复用

拟对固定 Transformers 版本的原视觉位置方法进行实例级包装，复用由原方法计算的插值和 rotary 位置张量，不改变图像 batch、Attention 或采样。

缓存保持私有，每个方法最多保存一个几何签名，向调用方返回 clone；其他几何或不支持的类型回退至原方法。签名包含实际 grid 值、设备、dtype、autocast、default device 和 stream，并核对位置模块、forward、参数与 buffer 版本及配置变化。

训练、grad、hooks 等不支持的路径不使用缓存。关闭或重新配置时恢复自身包装，不覆盖外部后来安装的方法。记录缓存张量的实际字节数；读取 grid 的 D2H、输出 clone 及相关同步均纳入端到端计时。

## MatMul 权重预取

首先使用 NPU profile 和算子形状分析确定访存热点。初始候选为 VLM decode 的 `up_proj.weight`，仅在 eval/no-grad、BF16 连续权重且 `M≤10` 的约束内评估。

拟在 post-attention residual 之后、postnorm 之前插入预取，以真实 residual 作为 dependency，使 RMSNorm 和 gate 计算形成独立窗口，随后由原 up projection 消费权重。预取地址与长度不得越界，图内调用位置固定，不新增显式预取同步。

分别记录 Python 发射、capture 负载和真实 replay；通过 CMO、时间戳、调用顺序及消费者 counter 交叉核对执行与依赖窗口。仅有 CMO 或局部访存比值不能证明 L2 命中率改善、全芯片带宽饱和或端到端净收益，最终以无 profiler 的四档性能比较判定。

## 实现组织与兼容性

图运行时、decoder 分段、位置复用和必要的后端适配拟封装为样例私有模块，随现有 `ar-1.patch` 集成。保留原适配逻辑，不修改公共 `module/`。开关默认关闭，未知签名、设备或不支持的状态按约束回退。

如扩展其他 Attention 后端，应单独标记其数值语义，只与该后端自身的 eager 比较，不将跨后端结果作为原 Attention 等值的证据。主方案不采用 batch 压缩、padding/truncation、减少轨迹数或调整采样参数来获取性能收益。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 计划判定方式 | 标准来源 |
| --- | --- | --- |
| 精度 | 保留任务书 `minADE≤0.5676` 要求及实际达标情况；各优化档对同平台、同后端、同协议 eager 的原始 XYZ/rot、序列化轨迹、CoC 与 RNG 严格一致，固定 100 clips 不退化 | 任务书及官方精度说明 |
| 真实执行 | capture/replay/回退可观测；预取有热点分析、真实 CMO、安全依赖窗口及关闭对照 | 任务书图模式与预取条款 |
| 性能 | A2/A3 combined 的端到端稳态耗时均优于本平台 eager，收益超过基线波动；同时报告图和预取的独立影响 | 任务书性能条款 |
| 稳定性及内存 | 连续至少 20 次无失败或输出漂移，无持续内存增长；准备、缓存和稳态峰值分开记录 | 任务书稳定性与存储条款 |
| 工程及交付 | 单样例范围、License、复现说明、全仓 pre-commit、外部 CI、双平台验证及社区评审流程 | 任务书与贡献指南 |

若固定环境 eager 与任务书参考精度存在差异，计划在验收材料中保留原始要求、逐 clip 指标、环境/协议差异及官方解释依据，明确优化前后是否退化；不自行设定数值容差，也不把内部一致性替代正式评审结论。实际结果和偏差说明归入验证报告。

## 验证计划

1. 固定模型、数据、依赖、采样配置和源码指纹；分别执行四档，独立生成 1/3/5/10 轨迹的固定 100 clips，严格比较原始输出、序列化、CoC 与 RNG。
2. 在无 profiler 条件下预热 3 次、连续测量至少 20 次，并执行前后 eager 对照和交替区组复测；保留全部样本，不删除慢值。
3. 统一计时边界：设备输入准备完成后开始，覆盖完整模型生成调用及结束同步；内部拷贝、位置 grid D2H 和输出 clone 计入。输入加载/processor、返回后的 CPU 拷贝、落盘及指标计算另计，首次图准备单列。
4. 使用独立的仪器化实验分析 kernel、CMO、依赖窗口和内存；不将 profiler 或嵌套 Event 时间作为正式稳态收益。
5. 覆盖开关恢复、参数/buffer/forward/hook 变更、动态签名回退、捕获失败、RNG 恢复和旧输出拥有权，并检查捕获次数及持续重捕获。
6. 验证 patch 安装、默认路径兼容性、定向与全仓代码检查；保存实际日志、环境/资产指纹及可离线比较的输出归档，权重、许可输入、凭据和大 trace 不入库。

## 风险与处理

动态图和 KV cache 可能限制覆盖率，采用稳定边界并记录回退原因；小子图调度、输出拷贝和位置缓存可能抵消收益，按完整端到端计时决定配置。预取可能与消费者争用带宽，需通过四档对照调整目标、位置和范围。缓存与图输出复用可能带来显存或生命周期问题，以有界缓存、失效检查和输出拥有权用例约束。

## 交付安排

按社区流程先提交设计评审，依据反馈完善方案，再开展实现与验证。交付包括仅涉及本样例的代码变更、配置及复现说明、精度/性能/内存报告和原始证据。设计 PR 评审合入后按要求提交平台测试，代码 PR 的 CI、评审和合入结果按实际流程记录。
