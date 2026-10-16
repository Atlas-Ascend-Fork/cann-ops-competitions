# Alpamayo-R1 图模式与权重预取设计

## 需求背景

本方案对应“10月社区任务-Alpamayo-R1图模式与预取优化”，提交账号为 `gcw_AFUK9jLY`。在固定权重、BF16、输入和采样配置下，使用稳定子图 capture/replay 与有依据的权重预取降低 A2/A3 的稳态端到端耗时。

方案讨论：[Recipes RFC #146](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/146)。[源码分支](https://gitcode.com/gcw_AFUK9jLY/cann-recipes-embodied-ai/tree/codex/alpamayo-graph-prefetch)包含实现、固定清单、测试和双平台优化说明。本文是供维护者评审的方案与实测摘要，精度协议仍需确认。

## 需求分析

1. `enable_graph` 与 `enable_prefetch` 独立、默认关闭，关闭时恢复原 eager 路径。保持默认 10 条轨迹、10 个扩散步、64 个 waypoint 和原 CoC 采样参数。
2. 同平台固定输入与随机状态，比对原始 xyz/rot Tensor 的 dtype 和逐元素值、序列化轨迹及 CoC；任何差异使评测返回非零。
3. A2/A3 分别完成 eager、graph、prefetch、combined 四档测试，单列首次准备、稳态耗时、kernel 与显存，连续不少于 20 次运行。
4. 固定 Physical-AI-AV 100 clips，保留 1/3/5/10 轨迹入口；各档不劣于同轨迹数 eager，10 轨迹另检查 minADE≤0.5676。

## 固定输入与环境

| 项目 | 配置 |
|---|---|
| Recipes 基线 | `db9b2453ad70ac919f8c42e02a61d16db63f4e66` |
| NVIDIA 上游 | `e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073`，应用样例 patch |
| 权重 | `nvidia/Alpamayo-R1-10B`，revision `dd4a24cacefc9a6477a6dfc7354de2443401409d` |
| A2 / A3 | 910B3 / Ascend910_9382；模型均使用单逻辑芯片、64 GB HBM |
| 软件 | CANN 9.0.0 对应平台算子包；driver 26.1.1；Python 3.11.13；torch 2.8.0；torch-npu 2.8.0.post4；transformers 4.57.1 |
| 性能输入 | 原样例 clip `030c760c-ae38-49aa-9ad8-f5650a545d26`，`t0_us=5100000` |
| 数据集 | Physical-AI-AV `25.10`，revision `2ae73f49ffd2b5db43b404201beb7b92889f7afc`，物理分片 chunk_0000 的固定 100 clips |
| 随机协议 | Python/NumPy/CPU/NPU seed 42，每请求重置；四档恢复同一请求前随机状态 |
| 清单 | `navigation/alpamayo-r1/configs/physical_ai_av_chunk_0000.json`，SHA-256 `deab873b934a6ab181516d79045db2aa52cf87d48366b2479298de1d35d7182e` |

权重通过 ModelScope 镜像下载，以下五个分片均与 HF 官方 SHA-256 一致：

| 文件 | SHA-256 |
|---|---|
| `model-00001-of-00005.safetensors` | `6aabd8d143cff0295a60b515dfcb5ba6a5b1b5acf7cea1d6c6254ed653d35965` |
| `model-00002-of-00005.safetensors` | `e346fcf2bf4ebd75853bd17b5744c4158cf621aba195aad7c7e2f1e484c1ae20` |
| `model-00003-of-00005.safetensors` | `7acad9950402f002825aa41048155ea1a8a2fb5f7a00501154d52828291171c9` |
| `model-00004-of-00005.safetensors` | `59f3ec7a6983ae2654b7ffe5d79f59d3d523b8a71a0ffc83362816886636ac0a` |
| `model-00005-of-00005.safetensors` | `5753b1fd57638b70db4882cc8061eedfa28c2748a37414eb4032e20b1bb2e4c1` |

## 详细设计

### 图边界与生命周期

捕获一次完整 diffusion expert 前向（36 层），在 10 个扩散步中重放。VLM 自回归、CoC 采样与动态图控制留在图外。原样例签名的 hidden 为 `[10,64,2048]`，KV 前缀长度为 3027；其他 clip 使用自己的实际签名。

图外准备 hidden、position ids、KV 前缀和输出 buffer。每个请求更新一次 KV 前缀与位置，每步只更新 hidden。私有 cache 保持固定前缀，各层仍按原语义拼接当前 token；不持久保存临时拼接 KV。最多缓存两张图，淘汰前同步并释放；输出在图外 clone，防止后续 replay 覆盖已返回结果。

当前 TorchNPU 的 `npu_fusion_attention(sync=False)` 会留下未合流辅助流。捕获期间在 dropout 为零时使用 `sync=True`，保持原融合算子及数值运算，已验证独立 Attention 变化输入与完整模型输出一致。输入判定在捕获外进行，不支持的输入走 eager。捕获体失败且 capture 正常结束时回退并禁用失败签名；capture end 本身失败时终止当前进程，避免继续使用未清理的后端状态。`close()` 释放资源并恢复原方法。

### 预取目标与依赖窗口

目标为 expert MLP 的 `gate_proj.weight`。BF16 gate/up MatMul 的输入形状为 `[640,2048] × [8256,2048]`。eager 的 MTE2/MAC ratio 在 A2 为 0.89587/0.68048，在 A3 为 0.95087/0.74319，支持将其作为搬运受限候选。

每层 self-attention 开始时异步预取后续 gate 权重的 4 MiB，预取流等待起始事件，进入 MLP 前等待完成事件。窗口覆盖 Q/K/V、位置变换与 KV 拼接；实际 CMO 在融合 Attention kernel 开始前已结束。图与 eager 各自持有事件，组合模式把 CMO 及依赖纳入捕获。

两平台组合模式每请求均有 360 次设备 CMO，全部在对应 gate MatMul 前完成；关闭预取的两档没有 CMO。组合 replay 没有额外 `npu::npu_prefetch` / `aclrtCmoAsync` 主机调用，四档主机同步 API 计数相同。独立预取收益尚未证实，当前端到端收益主要来自图模式。

### 交付边界

模型私有 wrapper、静态 buffer、预取及回退逻辑通过 `navigation/alpamayo-r1/patches/ar-1.patch` 交付，不修改公共 `module/` 或 NVIDIA 上游子模块。`benchmark.py`、固定清单和 `tests/check_npu_graph.py` 提供复现与故障恢复检查；样例 README 和优化文档记录支持矩阵、配置、命令与结果。

下述全量实验使用的正式优化模块 SHA-256 为 `7e10c0f6f96dfc51bf4f6187dab3014b8d1689fde6bfad942e8fb09170e11f2d`，评测入口为 `4b1561e42c87ef96311274c8e07c7c1efdf54045f7afbc727d615d2e795f1781`。

## 可维可测分析

### 已验证的性能与稳定性

原样例固定输入、每档预热一次、轮转测量八轮；性能采集时没有其他推理任务。端到端计时包括 processor、H2D、模型、CoC 解码和轨迹 D2H，排除磁盘读取、加载、seed 设置和序列化。

| 平台 | eager 中位数 | graph | prefetch | combined | 组合配对节省中位数 / eager IQR | 平均节省 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| A2 | 7.80313 s | 7.73333 s | 7.82452 s | 7.72266 s | 76.35 / 51.80 ms | [52.07,140.12] ms |
| A3 | 6.52641 s | 6.46102 s | 6.52015 s | 6.46040 s | 68.26 / 27.00 ms | [64.34,109.18] ms |

组合模式端到端中位耗时降低 1.03% / 1.01%，符合配对平均节省 CI 下界为正、配对中位节省超过 eager IQR 的判据。每平台 36 次输出精确匹配原 eager；32 次测量后的 allocated memory 均固定为 31,194,373,120 bytes。独立四档 profiler 的峰值 allocated 为 eager/预取 31.548 GB、graph/组合 36.022 GB。

每平台捕获两张图、重放 180 次，无同签名重复捕获。graph/组合首次准备在 A2 为 210.13/266.83 ms，在 A3 为 200.96/168.01 ms，均单独记录。NPU 小模型测试覆盖捕获体失败回退、20 次变化输入、旧输出所有权、签名淘汰、配置切换及关闭恢复。

### 数据集与待确认事项

10 轨迹下，两平台分别完成固定 100 clips、四档 400 次请求，优化输出与各自原 eager 逐元素一致。四档 minADE 均为 A2 **0.63017793**、A3 **0.62827477**，均高于 **0.5676**。A2 连续随机序列协议为 0.63712861；该对照也未复现绝对门槛。

请维护者确认 0.5676 对应的固定清单、原始评测入口、随机状态推进与软件栈，并确认本次 CANN 9.0.0 实测组合是否可作为验收环境。现有数值差异按原始结果报告；不把四档一致等同于绝对门槛已通过。1/3/5 轨迹也已完成完整清单与同轨迹数 eager 比较，双平台四种轨迹数共 3200 次请求全部精确一致，完整数值见代码分支的优化文档。

完整配置、逐次值、源码快照、输入/输出哈希及 profiler 摘要已保留。完整原始输出和 profiler 以归档及 SHA-256 保存，权重和传感器数据不入库。变更文件的 pre-commit 检查通过；全仓已有文件的检查问题单独记录，远端 CI 随代码 PR 核验。方案与验收口径确认后继续完成评审闭环。
