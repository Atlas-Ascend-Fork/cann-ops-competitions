# Alpamayo-R1 图模式与预取优化 设计文档

任务：10月社区任务-Alpamayo-R1图模式与预取优化（hiascend task-center ID `9c55c937110d43af8c0bd984012bcda6`）
目标代码仓：cann/cann-recipes-embodied-ai（代码 PR 另附，本目录仅设计文档）
代码 PR：https://gitcode.com/cann/cann-recipes-embodied-ai/merge_requests/213

## 1. 需求背景（required）

仓内 Alpamayo-R1 样例基于固定版本上游代码（NVlabs/alpamayo @ e0e2ac3）经 patch 完成昇腾适配，已在 Atlas A2 接入融合 Attention、RMSNorm、`addcmul`、索引与内存分配优化：单次生成 10 组轨迹 + CoC 文本约 7.32s，Physical-AI-AV `chunk_0000` 的 10 轨迹 minADE 为 0.5676。

本任务在**不改变模型语义和输出**的前提下：1）将稳定计算子图接入图模式；2）对 profiling 证明为 memory-bound、存在安全依赖窗口的 MatMul 类热点使能预取，在 A2、A3 获得可重复的端到端性能收益。

## 2. 功能描述（required）

### 2.1 图模式接入

- 提供 `enable_graph` 配置，**默认关闭**，关闭时完全恢复原路径；不修改默认输入、采样步数或轨迹数量换取收益。
- 图外完成输入 buffer、输出 buffer 与可复用 workspace 准备；warmup → capture → replay 生命周期可观测（`AR1_GRAPH_DEBUG=1` 打印各阶段耗时），capture 失败自动回退 eager；**随机状态**（扩散初始噪声）在**图外**一次性采样（`diffusion/flow_matching.py` 的 `sample()` 入口 `torch.randn(...)`），capture 区仅 10 步确定性 `step_fn`，区内不含随机数生成，每请求按固定 seed（`torch.npu.manual_seed_all(42)`）复现。

### 2.2 预取接入

- 提供独立 `enable_prefetch` 配置，默认关闭；图模式与预取组合时预取调用位置固定，capture/replay 生命周期无额外重编译。
- 四档消融（baseline / graph-only / prefetch-only / graph+prefetch）分别报告端到端耗时、目标模块耗时、kernel 变化与显存峰值。

### 2.3 精度与稳定性

- 固定输入、权重、seed：优化前后原始轨迹 Tensor、序列化轨迹与 CoC 文本一致。
- 复跑 `chunk_0000` 固定 100 clips 的 minADE 验证，保留 1/3/5/10 轨迹入口，证明图模式与预取不引入数据集级退化。

## 3. 详细设计（required）

### 3.1 候选子图选择与 RFC 锁定

推理链路分三段：

1. VLM 自回归 CoC 生成（`vlm.generate`）——数据相关长度、动态停止，不可 capture；
2. 轨迹 token 融合与后处理——形状随生成结果变化；
3. 扩散采样（Flow Matching，固定 10 步 Euler，每步 `step_fn(x, t)` 形状完全静态）——**选择扩散 step_fn 作为 RFC 锁定的稳定核心子图**。

### 3.2 图边界与 capture 约束（关键工程点）

aclgraph capture 要求 capture 区域内所有算子在 capture stream 上执行；任何内部派生未 join side stream 的融合算子都会使 `aclmdlRICaptureEnd` 失败（ACL 107025，GLOBAL/RELAXED 模式均不可绕过）。torch_npu 2.8.0 实测定位：

- `npu_fusion_attention` / `npu_fms_norm`（SDPA 长 KV 路径）内部派生 side stream → 不可 capture；
- 纯 torch 原语（matmul/softmax/layernorm/rms_norm/silu/切片赋值/addcmul 等）全部 capture 通过。

**处理**：expert（AR-1 轨迹专家）attention 切换为 transformers `eager`（matmul+softmax 原生实现），非因果语义以**显式全零 additive attention mask** 传入（eager 路径按 mask 计算，等价原 `is_causal=False`）；`npu_rms_norm` 同样替换为数学等价的纯 torch RMSNorm。基线与图模式走同一 attention 实现、同一 mask，配合固定 seed 保证四档输出逐元素一致。`AR1_LEGACY_OPS=1` 可切回官方原算子用于数值一致性 A/B（默认关闭）。

### 3.3 图生命周期（`ar1_graph.NpuGraphRunner`）

- 图外准备静态输入 buffer（x/t 每步拷贝进静态 buffer）；
- 侧流 warmup 2 次后 `torch.npu.graph` capture（首次 compile/capture 约 0.4s，一次性）；
- capture 失败自动回退 eager（不改变结果语义）；`enable_graph=False` 完全恢复原路径。

### 3.4 vision 塔元数据缓存（enable_graph 路径，默认开）

`Qwen3VLVisionModel.forward` 每请求重算 `fast_pos_embed_interpolate(grid_thw)` 与 `rot_pos_emb(grid_thw)`（grid 插值位置编码与 3D RoPE），二者均为 `grid_thw` 的纯函数。`VisionMetadataCache` 按 (方法名, grid 值, dtype, device) 缓存结果，仅 grid 变化时重算；两处消费均为只读（`hidden_states + pos_embeds`、`cat→cos/sin`）。缓存三路径（冷算/未命中/命中）输出 `torch.equal` 一致，整次推理 A/B 轨迹 Tensor `maxdiff=0`；单请求实测 9.00s → 8.15s（**−0.85s**，为端到端收益的主要来源）。

### 3.5 权重预取设计（官方 `torch_npu.npu_prefetch` CMO）

- **profiling 论证**：msprof（`--application` 模式）基线单次推理 NPU 算子时间 6.65s，其中 `MatMulV3` 1.92s（29%，avg 5.6ms）为权重读密集的 decode/prefill 投影，判定 memory-bound 热点。
- **插入点与安全依赖窗口**：在 decoder 层 `forward_pre_hook` 中，于独立 CMO 流上提交 `torch_npu.npu_prefetch(本层 down_proj 权重, None, nbytes)`（异步、无 dependency），权重搬运与本层 attention 计算重叠。单次上限 `AR1_PREFETCH_MAX_MB`（默认 2 MiB）；默认仅首层注册（`AR1_PREFETCH_LAYERS=1`）——实测全层注册（36 层 × 每步提交）会引入 host 提交开销（组合档 +0.17s），收敛为首层后恢复中性；异常/API 缺失静默回退。

### 3.6 静态 KV 与 capture 跨推理复用（AR1_STATIC_KV，默认开）

- `StaticCache` 桶化（bucket = ceil((prefill+64)/1024)×1024；实测 prefill≈3027 → bucket 4096）：expert 输入形状跨推理恒定；
- 每推理将 VLM prompt KV 拷入静态 buffer、尾部置零并以 -inf mask 屏蔽（语义与动态定长一致），显式 `cache_position`，`StaticLayer.update` 包装层做 dtype 对齐（expert 经 k_norm/rope 后为 float32，与动态路径 `cat` 隐式行为等价）；
- `NpuGraphRunner` 按 (b_star, bucket) 缓存于模型实例：每进程每桶仅 capture 一次，后续推理直接 replay；
- eager 与图模式共用同一静态路径，四档输出逐元素一致（torch.equal，maxdiff=0.000e+00）。

### 3.7 Cholesky 求解 NPU 原生实现

`action_space/utils.py` 的轨迹平滑最小二乘（`solve_xs_eq_y`/`dxy_theta_to_v*` 共 4 处）原使用 `torch.linalg.cholesky` + `torch.cholesky_solve`，NPU 不支持、每次调用 CPU 回退（D2H/H2D 同步）。替换为纯 torch 批量 Cholesky（逐列 batched 向量算子，同一分解算法），全部算子 NPU 执行、无 host 同步，eager 端到端 −60ms，minADE 不变。

### 3.8 精度保障机制

同一 attention 实现（eager + 显式全零 mask）+ 静态 KV 路径 + 固定 seed（`torch.npu.manual_seed_all(42)`）+ 确定性算子序列 → 四档原始轨迹 Tensor 逐元素一致（消融基准 clip 四档 minADE 均为 0.2315，逐次一致）；另有 `ar1_selftest.py` 作为可执行门禁（开关组合 / 连续 20 次 / 输入输出逐元素比对 / 关开关回退，任一不一致以非零状态退出）。

## 4. 测试方案（required）

| 测试项 | 方法 | 通过标准 |
|---|---|---|
| 四档消融 | `ar1_ablation.py`，A2 910B3 + A3，CANN 9.1.0；每档 1 次预热 + 3 次正式，另做 20 次稳态 | 报告逐次耗时与 minADE；**20 次稳态中位 A2：eager 7.680 / graph+prefetch 7.350s（−4.3%）；A3：6.790 / 6.525s（−3.9%）**；改善为各自 eager IQR 的 11 / 26 倍；四档 minADE 逐位一致（0.2315 / 0.2342） |
| 目标模块与 kernel 变化 | 四档逐档 `msprof --ai-core=on`（`AR1_ITERS=1`，2 次推理，op_statistic 口径） | 目标模块 `MatMulV3` 总耗时四档 4.211 / 4.208 / 4.208 / 4.208s（占比约 30%）；**算子种类均 84 种、无新增/消失** → 只改下发方式与访存调度，不改计算 kernel |
| 数据集精度 | `ar1_eval100.py`，官方 `chunk_0000`（= `get_clip_chunk()` 的 chunk id 3116）固定 100 clips，1/3/5/10 轨迹 | **10 轨迹 mean minADE A2 0.4828 / A3 0.4954（门槛 ≤0.5676）**；1/3/5 轨迹 A2 1.7244 / 0.9314 / 0.7556、A3 1.6155 / 0.9371 / 0.7712（仓内基线表 2.2026 / 0.9498 / 0.8514） |
| 输出一致 | 固定输入/seed，逐元素比较轨迹 Tensor、序列化轨迹、CoC 文本 | `torch.equal=True`、`maxdiff=0.000e+00`（三档 vs eager）；`ar1_selftest.py` 双平台 `SELFTEST_OK`、退出码 0 |
| 稳定运行 | 四档各连续 ≥20 次推理（`ar1_prof_mem.py`） | 20/20 成功；显存恒定（eager/prefetch allocated 26.42 GiB、graph/combo 26.68 GiB，峰值 35.15 / 35.41 GiB）；输出逐次一致 |
| 双平台 | A2 + A3 自验实机，同一份 patch | 双平台均跑通四档与 100 clips；工作树仅差 SPDX 注释头与默认关闭的 `AR1_LEGACY_OPS` 开关，`ar1_graph.py`、`models/alpamayo_r1.py` 逐字节相同 |

## 5. 复现口径（RFC 锁定）

Recipes @ db9b245；上游 alpamayo @ e0e2ac3；权重 nvidia/Alpamayo-R1-10B（bf16，5 分片，sha256 见验收材料）；A2 910B3（自验）/ A3（官方验收）；CANN 9.1.0 V100R001C11SPC001B243；torch 2.8.0 + torch_npu 2.8.0 + transformers 4.57.1；输入 `chunk_0000`（= `get_clip_chunk()` 的 chunk id 3116）固定 100 clips，t0=5.1s（脚本默认会先查 per-clip t0 表，本输入集全部未命中故回落 5.1s；`AR1_NO_T0MAP=1` 显式关闭查表后逐位相同）；1/3/5/10 轨迹；top_p=0.98、temperature=0.6；seed=42；`AR1_EXPERT_ATTN=eager`（默认值）。

## 6. 已知限制与实测结论（诚实声明）

1. **性能条款已达成**：核心收益来自 vision 塔元数据缓存（3.4 节，单请求 −0.85s）与静态 KV 支撑的扩散 step capture；20 次稳态中位 A2 7.680 → 7.350s（−4.3%）、A3 6.790 → 6.525s（−3.9%），均远超各自 eager 波动。**单开预取（官方 CMO）对端到端为中性**：目标模块为计算为主而非 launch-bound，权重搬运与 attention 重叠带来的收益落在波动内；预取仍按任务要求实装、可开关、无生命周期错误与隐藏同步。
2. **精度**：早期设计阶段曾用非官方分块（`clip_index[chunk==0]`）评测得 0.6323，与门槛不可比；改用官方 `chunk_0000`（chunk id 3116）后双平台 0.4828 / 0.4954，**低于门槛 0.5676**（详见代码 PR 文档第 9 节修订说明）。
3. **平台数值差异**：同一份代码同一 clip，A2 基准 clip minADE 0.2315、A3 0.2342，为两平台 kernel 数值差异（四档内部逐位一致不受影响）。
4. **图边界约束**：`npu_fusion_attention`（SDPA 长 KV 路径）与 `npu_rms_norm` 不可 capture，已用数学等价的 eager 实现替代（输出逐元素一致，含 A/B 实验）；后续 torch_npu 版本若支持 c 内部 side stream join，可移除该替换。
5. **A3**：已完成四档与 100 clips 实机自验；最终以官方实机验收为准。
