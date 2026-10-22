# LingBot-VA 昇腾推理与 LIBERO 四套件评测方案设计

## 需求背景（required）

### 需求来源

CANN 社区任务 2026，任务编号 **10-11**，任务名称 **LingBot-VA 模型接入与 LIBERO 评测**。

- 上游模型：<https://github.com/Robbyant/lingbot-va>
- 固定上游提交：`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- 目标代码仓：`cann/cann-recipes-embodied-ai`
- 目标样例目录：`manipulation/lingbot-va/infer_with_torch/`
- 仿真基准：LIBERO Spatial、Object、Goal、Long 四个官方 suite

### 背景介绍

LingBot-VA 是面向机器人操作的视频-动作世界模型。模型以视频动态预测作为中间表征，在同一条交错序列中完成未来视频与动作推断，并通过服务端-客户端架构支持 LIBERO 闭环控制。

与已有单帧视觉语言动作模型相比，本任务同时包含以下负载特征：

1. 模型主干、VAE、文本编码器、action head 和 KV Cache 多组件联合推理。
2. 双相机视频输入、连续动作 chunk、已执行步骤回灌和 episode reset。
3. 20 步视频推理与 50 步动作推理组合成的多次自回归前向。
4. 模型环境与 LIBERO 仿真环境依赖不同，需要进程和服务协议隔离。
5. 最终指标由数千次机器人闭环 rollout 决定，配置漂移会直接改变评测结果。

当前 `cann-recipes-embodied-ai` 尚无 LingBot-VA 样例。本任务需要在保持上游执行方式的基础上完成昇腾 NPU 接入，提供可复现的四套件评测，并证明四套件成功率的算术平均值不低于 98%。

### 上游现状与关键适配点

上游默认围绕 CUDA 运行。接入 NPU 时至少需要处理以下边界：

| 类别 | CUDA 默认行为 | 昇腾方案 |
| --- | --- | --- |
| 设备选择 | `cuda:{rank}`、`is_cuda` | 显式映射到 `npu:{local_rank}` 和 `is_npu` |
| 分布式 | NCCL / CUDA 进程组 | HCCL / NPU context |
| attention | flash attention、flex attention | 使用已验证的 NPU attention 或等价实现 |
| dtype | 部分路径隐式使用 CUDA 特性 | 明确 dtype，禁止核心计算静默转 CPU |
| 缓存释放 | CUDA cache API | `torch.npu.empty_cache()` 和进程级清理 |
| 服务状态 | 以 CUDA 进程为边界 | 明确 NPU、HCCL、KV Cache 和 VAE 状态生命周期 |

推理时 checkpoint 中 `transformer/config.json` 的 `attn_mode` 必须为 `torch`、`flashattn` 或经验证的 NPU 等价实现，不能保留训练模式 `flex`。

## 需求分析（required）

### 需求描述

在 `cann-recipes-embodied-ai` 中新增独立的 LingBot-VA 推理样例，使模型能够在锁定的 Atlas 平台上加载权重并生成有效动作，保持上游 server-client 协议，并在 LIBERO 四个官方 suite 上完成完整、可复现的闭环评测。

任务完成时，四项 suite 成功率按算术平均计算：

```text
Avg = (SR_spatial + SR_object + SR_goal + SR_long) / 4
```

验收条件为：

```text
Avg >= 98.0%
```

该平均值不得按 2,000 个 episode 重新加权，也不得选择最好一次运行作为结果。

### 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | RFC 锁定平台、软件、上游提交和四套 checkpoint | RFC 字段完整且后续可追溯 |
| R2 | 主干、VAE、文本编码器、action head、KV Cache 关键计算在 NPU | device 审计和运行日志无核心 CPU/CUDA 回退 |
| R3 | 保持 server-client 协议完整 | health、reset、双相机、prompt、action chunk、步骤回灌测试通过 |
| R4 | 支持 episode 和 task 状态隔离 | 同 suite 内 task A -> reset -> task B 测试通过 |
| R5 | 支持不同 suite 的 checkpoint 切换 | suite 串行重启 server、HCCL/NPU 释放和新 checkpoint 加载成功 |
| R6 | 四套件各覆盖 10 个任务 | 40 个任务任务映射完整，无遗漏 |
| R7 | 每任务执行 50 个 episode | 每 suite 500 条、总计 2,000 条唯一记录 |
| R8 | 失败样本不删除 | 超时、异常、无效动作等均计入分母 |
| R9 | 四套件平均成功率不低于 98% | 四项 SR 可由 JSONL 明细独立复算 |
| R10 | 性能、内存和稳定性可复现 | profiling、HBM/Host 内存和连续 20 episode 证据完整 |
| R11 | 样例工程可维护 | README、优化报告、License、测试、CI 和实机结果完整 |

### 范围与非目标

本任务包含：

- NPU 模型加载、forward、attention 和 KV Cache 适配；
- server/client 协议及 reset、timeout、异常退出处理；
- 四套件评测、结果校验、汇总和断点续跑；
- 性能优化、内存采样和稳定性验证；
- README、优化与验收文档。

本任务不包含：

- 将完整 LingBot-VA 上游源码复制到 Recipes 仓库；
- 将模型权重、LIBERO 资产、仿真镜像、凭据或 rollout 视频提交到仓库；
- 修改 LIBERO 成功判定以获得目标指标；
- 要求同一个 server 在运行中动态切换四个 suite 的 checkpoint；
- 在没有合法权重来源的情况下伪造 Spatial、Object、Goal 评测结果。

## 详细设计（required）

### 总体架构

```text
┌───────────────────────────────┐
│ LIBERO Client                 │
│ task / init state / reset     │
│ observation / action / result │
└───────────────┬───────────────┘
                │ configurable websocket protocol
┌───────────────▼───────────────┐
│ LingBot-VA NPU Server         │
│ prompt / dual camera / action │
│ KV Cache / session state      │
└───────────────┬───────────────┘
                │ torch_npu + HCCL
┌───────────────▼───────────────┐
│ Ascend NPU                    │
│ backbone / VAE / text encoder │
│ action head / attention       │
└───────────────────────────────┘
```

模型环境与 LIBERO 仿真环境独立安装、独立进程启动。模型环境负责 NPU 推理和缓存管理，仿真环境负责任务创建、观测采集、动作执行和成功判定。两端通过可配置地址和端口通信，不写入生产地址或凭据。

### RFC 与环境冻结

开发前创建 RFC，至少冻结以下内容：

| 类别 | 冻结字段 |
| --- | --- |
| 硬件 | Atlas 平台、卡型、卡数、可见设备、卡间拓扑 |
| 软件 | CANN、驱动、PyTorch、torch_npu、Python 版本 |
| 上游 | LingBot-VA commit、LIBERO commit、仓库地址 |
| 权重 | 每 suite checkpoint、来源、revision、License、SHA256 |
| 配置 | `va_libero_cfg.py` SHA256、suite 配置和 norm stats |
| 输入 | 双相机键、输入尺寸、相机翻转和 prompt 格式 |
| 动作 | 通道顺序、归一化统计、反归一化、控制频率 |
| 推理 | `attn_mode`、视频/动作步数、attention window、chunk size |
| 评测 | task 顺序、init states、seed、等待步数、最大环境步数 |
| 输出 | JSONL schema、唯一键、失败分类和续跑规则 |

必须锁定的模型配置包括：

- `action_snr_shift=0.05`；
- `used_action_channel_ids=[0..6]`；
- 双相机输入尺寸 `128x128`；
- `attn_window=30`；
- `frame_chunk_size=4`；
- 视频推理步数 `20`；
- 动作推理步数 `50`；
- guidance scale 和控制频率；
- checkpoint 内 action normalization 统计。

### 四套 checkpoint 设计

| suite | checkpoint | revision | transformer SHA256 | norm stats |
| --- | --- | --- | --- | --- |
| `libero_spatial` | RFC 冻结 | RFC 冻结 | RFC 冻结 | RFC 冻结 |
| `libero_object` | RFC 冻结 | RFC 冻结 | RFC 冻结 | RFC 冻结 |
| `libero_goal` | RFC 冻结 | RFC 冻结 | RFC 冻结 | RFC 冻结 |
| `libero_10` | RFC 冻结 | RFC 冻结 | RFC 冻结 | RFC 冻结 |

当前官方公开清单仅明确包含 Long 的 post-training 权重。Spatial、Object、Goal 必须在 Gate 0 前提供合法 checkpoint 或可复现 post-training 配方，并明确来源、License 和使用限制。若需要训练，应放在 `manipulation/lingbot-va/train/`，不得把未披露训练作为推理适配的隐藏前提。

### 代码组织

目标仓库目录如下：

```text
manipulation/lingbot-va/
├── README.md
├── infer_with_torch/
│   ├── README.md
│   ├── requirements-model.txt
│   ├── configs/
│   │   ├── suites.yaml
│   │   └── runtime.yaml
│   ├── patches/
│   │   └── lingbot_npu.patch
│   ├── server/
│   │   ├── launch_server.sh
│   │   ├── health_check.py
│   │   └── npu_runtime.py
│   ├── evaluation/
│   │   ├── run_suite.sh
│   │   ├── evaluate_libero.py
│   │   ├── validate_results.py
│   │   └── summarize_results.py
│   ├── tests/
│   │   ├── test_config.py
│   │   ├── test_protocol.py
│   │   └── test_result_schema.py
│   └── doc/
│       ├── design.md
│       └── optimization-and-validation.md
└── train/                         # 仅在需要 post-training 时增加
```

上游模型逻辑通过 patch、overlay 或 monkey patch 接入。模型私有逻辑只放在 `manipulation/lingbot-va/` 下；公共模块改动必须保持默认行为不变并完成直接调用样例回归。

### NPU 适配设计

#### 设备和进程

1. 启动脚本通过环境变量配置可见设备、进程数、端口和路径。
2. 多卡运行使用 HCCL 初始化，建立 local rank 到 NPU device 的明确映射。
3. 启动时校验主干、VAE、文本编码器、action head 和 KV Cache 所在设备。
4. device 审计发现 CUDA 或 CPU 核心计算时立即失败，不能静默继续。
5. server 退出时释放 HCCL 进程组、NPU context、内存池和临时文件。

#### 推理数据流

```text
双相机观测 + prompt
        -> 图像预处理
        -> VAE / 文本编码器
        -> 视频与动作自回归推理
        -> attention + KV Cache
        -> action head
        -> 动作反归一化
        -> action chunk
        -> LIBERO 执行并回灌已执行步骤
```

CPU 仅承担协议处理、图像格式转换、LIBERO 环境交互等明确记录的前后处理。核心模型计算不得因算子不支持或异常被自动映射到 CPU。

#### attention 与 KV Cache

1. 优先使用上游等价 attention 语义和已验证的 `torch_npu` 融合能力。
2. checkpoint 中的 `attn_mode` 由配置校验器检查，禁止 `flex` 进入正式推理。
3. KV Cache 的创建、追加、读取和 reset 使用显式接口，并通过 eager 与图模式路径分别验证。
4. 每个 episode 的缓存具有 session/task 边界，禁止跨 episode 复用未清理状态。
5. cache 释放失败、shape 不一致或 token 周期不完整时，当前 episode 标记失败并记录原因。

### Server-client 协议设计

协议必须保持上游语义并增加可诊断字段，至少支持：

| 能力 | 校验内容 |
| --- | --- |
| health | 服务就绪、模型加载、设备和配置摘要 |
| reset | 清空 session、prompt、帧缓存、动作缓存和 KV Cache |
| infer | 接收双相机图像和 prompt，返回 action chunk |
| 步骤回灌 | 接收已执行环境步和观测，更新时序缓存 |
| timeout | 请求超时返回明确错误码，不挂死事件循环 |
| shutdown | 停止接收请求并释放 NPU/HCCL 资源 |

reset 响应需要返回当前 suite、task、episode 和配置 hash，便于 client 检测状态错位。协议中不得丢失 prompt、任一相机图像或 reset 状态。

### 任务与 suite 切换设计

同一 suite 内必须支持：

```text
task A / episode N
    -> reset
task B / episode 0
    -> reset
task A / episode N+1
```

该测试用于验证 prompt、init state、双相机观测、KV Cache 和动作缓存没有跨任务污染。

不同 suite 使用不同 checkpoint 或 normalization 统计时，采用串行重启：

```text
Spatial server -> stop/release -> Object server
Object server  -> stop/release -> Goal server
Goal server    -> stop/release -> Long server
```

每次重启后重新执行 health 和 reset 检查，并记录新 suite 的 checkpoint/config hash。suite 串行切换只验证资源释放和新进程加载，不替代同一 server 内的跨 task 状态隔离测试。

### LIBERO 评测设计

正式评测矩阵：

| suite | 任务数 | 每任务 episodes | suite 总 episodes |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 10 | 50 | 500 |
| `libero_object` | 10 | 50 | 500 |
| `libero_goal` | 10 | 50 | 500 |
| `libero_10` | 10 | 50 | 500 |
| 合计 | 40 | - | 2,000 |

正式评测前依次完成：

1. server health 和 reset 检查。
2. 四 suite 各 1 个 single-episode smoke。
3. 至少一次同 suite 内跨 task 切换。
4. 连续 20 个 episode 状态和内存稳定性测试。
5. checkpoint、配置和 normalization 一致性校验。

正式流程按 suite 串行执行。每个 suite 内按冻结的 task 顺序执行 10 x 50 episode。每个 episode 完成后立即写 JSONL 并 flush，进程中断时保留已写记录。

### 结果结构与断点续跑

每条 episode 至少包含：

```json
{
  "suite": "libero_spatial",
  "task": 0,
  "episode": 0,
  "init_state": "...",
  "seed": 0,
  "checkpoint": "...",
  "checkpoint_sha256": "...",
  "config_sha256": "...",
  "success": true,
  "failure_type": null,
  "termination_reason": "libero_done",
  "env_steps": 113,
  "request_count": 14,
  "first_action_latency_s": 17.60,
  "action_chunk_latency_s": [1.2, 1.1]
}
```

唯一键为：

```text
(suite, task, episode)
```

以下情况均计为失败并保留在分母中：

- health 或 reset 超时；
- server 未就绪；
- 请求超时或服务异常；
- 仿真创建失败；
- action 无效；
- client/server 进程退出；
- episode 未正常完成。

续跑时检查 suite、task、episode、checkpoint hash、config hash、norm stats、seed 和任务顺序。checkpoint 或配置变化后，对应 suite 的旧记录不得与新配置混合统计，必须重新完整评测。

### 性能测试与优化设计

server 至少预热 1 个 episode，再对至少 5 个 episode 采集 profiling 数据。报告：

- 首 action 延迟的均值、P50、P95；
- action chunk 延迟的均值、P50、P95；
- 模型请求次数和请求吞吐；
- 每个 episode 的原始延迟数据。

性能对比固定 suite、task、init state 和 seed。优化前后各至少运行 5 个 episode，报告中位延迟、逐次原始值和预先记录的基线波动。

候选优化按证据推进：

1. attention 融合算子；
2. eval 图模式或等价 NPU 图能力；
3. 权重预取和常驻布局；
4. KV Cache 内存布局和复用；
5. 多卡通信和进程启动开销；
6. 输出 action chunk 处理；
7. 经验证不影响精度的量化或融合算子。

本任务没有绝对性能门槛，但必须提交可复现的性能结论；无收益的优化也要保留实验记录，不能只报告成功项。

### 内存和稳定性设计

开启 profiler memory 或等效采集，记录：

- server 权重加载后 Host/NPU 内存；
- warmup 后 Host/NPU 内存；
- 正式评测期间峰值内存；
- 每个 task 结束后的当前内存，共 40 个采样点；
- server 重启前后的资源释放情况。

扣除框架常驻缓存后，内存不得持续单调增长，完整评测不得 OOM。测试覆盖 reset、跨 task、跨 suite、请求超时、单 episode 异常、client 中断和 server 中断，确认 KV Cache、观测帧和动作缓存均已释放。

## 支持硬件（required）

正式目标平台由 RFC 锁定。样例代码通过配置支持单卡或多卡，不在业务代码中写死卡数、端口或本地路径。

当前工作区原型曾面向 Atlas 910 双卡环境进行验证设计，但该环境不能直接视为本任务正式验收平台。正式报告必须列出：

- 芯片型号和卡数；
- CANN、驱动、PyTorch、torch_npu、Python 版本；
- 模型和仿真环境安装方式；
- 实测 NPU 峰值内存和单 episode 耗时。

## 可维可测分析（required）

### 精度标准

| 验收项 | 要求 |
| --- | --- |
| 成功判定 | 仅使用 LIBERO 环境 `done` 或官方 evaluator |
| 评测规模 | 40 个 task，每 task 50 episode，共 2,000 条唯一记录 |
| SR 计算 | 每 suite 为成功数除以 500 |
| 最终指标 | 四项 SR 的算术平均 |
| 精度门槛 | `Avg >= 98.0%` |
| 失败处理 | 所有失败保留并计入分母 |
| 结果复算 | 从 JSONL 明细可独立复算四项 SR 和 Avg |

### 性能标准

| 指标 | 要求 |
| --- | --- |
| 预热 | 至少 1 个 episode |
| profiling | 至少 5 个 episode |
| 首 action | 均值、P50、P95 |
| action chunk | 均值、P50、P95 |
| 吞吐 | 模型请求吞吐 |
| 优化对比 | 相同输入条件下优化前后各至少 5 个 episode |

### 兼容性分析

| 对象 | 设计 |
| --- | --- |
| 上游代码 | 通过 patch/overlay 接入，不整体复制 |
| 公共框架 | 默认行为不变；若有修改，回归全部直接调用样例 |
| 模型权重 | 不入库，只记录来源、revision、hash 和 License |
| LIBERO 资产 | 不入库，只提供官方获取方式和版本 |
| 模型/仿真环境 | 独立 Python 环境，减少依赖冲突 |
| 单卡/多卡 | 通过运行配置区分，不修改模型业务逻辑 |
| 不同 suite | 独立 checkpoint/config，允许串行重启 server |

### 可诊断性

1. 每条 episode 记录配置 hash、checkpoint hash、seed、init state、请求次数和延迟。
2. server 日志记录 suite、task、episode、reset 和 cache 生命周期。
3. device 审计记录各模型组件所在设备。
4. 失败结果包含错误码、异常摘要和终止阶段。
5. 汇总脚本同时输出四项 SR、Avg、失败类型和覆盖率。
6. 性能脚本输出逐次原始值，不只输出汇总中位数。

### 阶段门禁

| Gate | 放行条件 |
| --- | --- |
| Gate 0 | RFC、目标平台、软件版本、四套合法 checkpoint 和 License 已冻结 |
| Gate 1 | CUDA 参考四 suite smoke 和跨 task 切换通过 |
| Gate 2 | NPU 四 suite smoke、health/reset 和 device 审计通过 |
| Gate 3 | 连续 20 episode、异常恢复、内存、性能基线和优化对比完成 |
| Gate 4 | 2,000 条唯一记录、四项 SR 和 `Avg >= 98.0%` 完成 |
| Gate 5 | README、优化报告、CI、Atlas 实机验收和 PR 完成 |

任一 Gate 未通过，均不能将任务标记为完成。四 suite smoke 只能证明链路可运行，不能替代 Gate 4。

### 主要风险

| 风险 | 影响 | 处置 |
| --- | --- | --- |
| Spatial/Object/Goal 无合法公开权重 | 无法完成四 suite 正式评测 | Gate 0 前确定来源或可复现 post-training 配方 |
| 上游 checkpoint/config 漂移 | 结果不可复现 | 固定 commit、revision、文件 hash 和配置 hash |
| action norm stats 不一致 | 动作明显退化、成功率不可用 | 每 suite 绑定并校验 norm stats |
| 核心计算回落 CPU | 时延和验收不达标 | device 审计、失败即退和日志证据 |
| reset 不完整 | 跨 episode/task 状态污染 | 同 server 多 task 连续测试和 cache 生命周期检查 |
| 2,000 episode 耗时过长 | 资源和周期不可控 | 先做代表性计时和预检，再申请资源并冻结正式运行 |
| 失败样本被误删 | 指标失真 | 唯一键、完整率校验和失败类型汇总 |

## 交付物

1. `manipulation/lingbot-va/infer_with_torch/` NPU 推理代码和 patch。
2. 模型与仿真环境依赖、启动配置和四 suite checkpoint 映射。
3. server 启动、health、reset、client 和正式评测脚本。
4. 结果 schema、校验、汇总和断点续跑工具。
5. README。
6. NPU 迁移和优化过程文档。
7. 四个 suite smoke、跨 task 切换和连续 20 episode 稳定性结果。
8. 恰好 2,000 条 episode 结构化明细或外部制品链接。
9. 四项 SR、Avg、失败类型和失败样例索引。
10. 首 action/action chunk 性能、吞吐、Host/NPU 内存和 profiling 数据。
11. LingBot-VA/LIBERO commit、平台、软件版本、checkpoint/config hash 清单。
12. License、已知限制、复现命令和 CI/实机验收结果。

权重、LIBERO 资产、仿真镜像、凭据和 rollout 视频不进入仓库。
