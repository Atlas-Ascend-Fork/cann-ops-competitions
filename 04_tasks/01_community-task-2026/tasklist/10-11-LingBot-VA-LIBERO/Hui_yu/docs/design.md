# LingBot-VA 模型接入与 LIBERO 评测设计

## 1. 需求背景

### 1.1 需求来源

- 社区任务：10-11，LingBot-VA 模型接入与 LIBERO 评测，贡献者 Hui_yu。
- [任务与官方澄清](https://gitcode.com/cann/cann-recipes-embodied-ai/discussions/10)。
- [模型接入 RFC #158](https://gitcode.com/cann/cann-recipes-embodied-ai/issues/158)。
- [实现候选 19ddbf6a4ab0da3cb6e076aa94fdbd71bf3315b3](https://gitcode.com/Hui_yu/cann-recipes-embodied-ai/commit/19ddbf6a4ab0da3cb6e076aa94fdbd71bf3315b3)，目录 `manipulation/lingbot-va/infer_with_torch/`。

2026-10-26 按官方“更新进展中的设计文档链接须指向 cann-ops-competitions”回复，在本活动仓补交个人目录的设计文档。原 [Recipes PR #225](https://gitcode.com/cann/cann-recipes-embodied-ai/pull/225) 是历史设计入口，尚未合入；实现仍按任务书归属 Recipes 仓。本次仅申请设计评审，不将已有开发、自验或 CI 结果表述为设计已批准或官方验收通过。

### 1.2 背景介绍

LingBot-VA 通过交错的视频与动作序列进行机器人控制。迁移涉及视频/动作扩散、双相机观测、文本条件、KV Cache 与已执行动作回灌，不能以独立张量推理替代 LIBERO 闭环评测。

方案复用锁定上游，以薄适配和 monkey patch 实现 NPU 执行，隔离模型服务与 CPU 仿真客户端。模型权重、数据集、仿真资产及 rollout 视频不提交 Git 仓库，只记录合法来源、固定 revision、配置及校验值，保留原 License/NOTICE。

## 2. 需求分析

### 2.1 验收范围与口径

| 要求 | 方案与判定 |
| --- | --- |
| 平台与核心计算 | Atlas A3；主干、VAE、文本编码、action head、KV 关键计算在 NPU 执行，CPU 仅承担明确的仿真与前后处理，不静默回退核心模型 |
| 四套件覆盖 | Spatial、Object、Goal、Long 各 10 个任务，每任务 50 个 episode，共 2000 个唯一 `(suite, task, episode)` |
| 成功判定 | 由 LIBERO 环境 `done`/官方 evaluator 决定，不能由模型自报 |
| 成功率 | 每套件 SR 单独公开，算术 `Avg=(SR_spatial+SR_object+SR_goal+SR_long)/4 ≥ 96%`；任务书 98%/96% 冲突依本题官方澄清处理 |
| 并行与失败 | 依官方澄清允许多实例；单模型服务内串行，任务唯一分配，失败与中断保留分母，不换 seed 重试取优 |
| 性能 | 无强制绝对时延门槛；优化必须保存相同调用范围、条件可比的优化前后原值及基线波动 |
| 稳定性 | 记录健康检查、reset、请求超时、构造/动作异常、跨任务切换、状态清理、内存与中断续跑；失败和修复分别披露 |

### 2.2 固定模型与评测配置

| 套件 | 原权重与固定 revision |
| --- | --- |
| Spatial | `Wjjjh/lingbot-va-libero-spatial@933ab666f8cd21ff71d8765dea9fdd110512034f` |
| Object | `Wjjjh/lingbot-va-libero-object@489806feb4074eee71f97180948e8b85e33ebeda` |
| Goal | `Wjjjh/lingbot-va-libero-goal@42455fd87bc5b9c4613cc13b03be2f2d80605efe` |
| Long | `robbyant/lingbot-va-posttrain-libero-long@0e89d1e753019988aba484e8da2dc0810e264d9f` |

checkpoint、norm stats、共享组件和配置均逐文件 SHA 锁定；不训练或替换原权重。LIBERO 固定提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`；LingBot-VA 与 Recipes 基线、完整文件清单见实现候选的 `reproduction/` 恢复索引。

实际评测仿真为 **MuJoCo 3.3.2 / robosuite 1.4.0**；最大 800 环境步、等待 5 步、128×128 双相机及垂直翻转、7 维动作、视频扩散 20 步、动作扩散 50 步、`frame_chunk=4`、`attn_window=30`、`action_snr_shift=0.05`。固定 init-state、episode index 和 seed；实际控制频率 20 Hz、control timestep 0.05、model timestep 0.002。配置中保留的 null 由锁定 LIBERO 默认值解析，不伪造原配置显式设值。

## 3. 详细设计

### 3.1 模块与数据流

```text
LIBERO CLIENT_PY（CPU 仿真、双相机、prompt）
  → reset / 观测与已执行动作 → 有界客户端协议
  → MODEL_PY NPU server → 模型主干 / VAE / 文本编码 / action / KV
  → 动作 chunk → LIBERO step → 官方成功判定
  → episode 明细、失败类型、延迟、状态与内存原件
```

`npu_runtime.py`、`server.py`、`upstream_compat.py` 复用上游并限定设备、attention/RoPE 与 KV 适配；`client.py`、`protocol.py` 保留 prompt、双相机、reset、动作 chunk 和原 action 回灌语义。模型私有适配不散落到公共框架；不新增 Ascend C 自定义算子，因此本设计不另设算子 Tiling/Kernel，算子执行由兼容的 torch_npu/CANN 路径完成。

服务启动先检查权重/源码/环境、设备和端口，健康检查完成后才接收请求。每 episode 独立 reset 并在结束时释放观测、动作与 KV 状态；非有限值、协议错误、仿真异常均显式记录。退出按 PID/start_ticks/session 身份回收自有进程，不使用宽泛 kill。

### 3.2 独立环境与兼容性

模型与仿真使用两个解释器：

- `MODEL_PY`：模型依赖环境；原实测使用 Python 3.10.13、torch 2.9.0、torch_npu 2.9.0.post6、CANN 9.1.0 兼容环境。该环境的 **MuJoCo 2.3.7 是历史模型环境依赖，不参与 LIBERO 仿真**，不构成第二种允许的评测版本。
- `CLIENT_PY`：安装 **MuJoCo 3.3.2 / robosuite 1.4.0**，实际执行仿真创建、reset 和 step。
- 有界监督与清理使用支持 Linux pidfd 的独立 CPU Python；不假设模型 Python 支持 pidfd。

驱动、固件、依赖、设备、命令和实际解释器按每台原环境清单绑定，不将一台机器的环境推及其他主机。地址、端口、权重与输出路径可配置，不提交生产地址或凭据。保留第三方协议、上游固定源码来源及适用的公共回归范围。

### 3.3 原 R7 正式执行与恢复

原 R7 在三台 A3 上使用六个独立单 die 实例，每个 server 独占一个 die。已执行的 durable 前缀封口后，其余任务由全局唯一 grant 池分配；已领取、未知或失败 grant 都占唯一任务，不能只按成功结果重新派发。通常一个任务由同一个新 server 配合两个各 25 条客户端完成，清理后再领取任务。

Spatial task4/episode0 原中断保留为失败，从 episode1–49 真实续跑；不能把该任务描述为一个 server 连续运行 50 条。结果以 `(suite, task, episode)` 去重，重复、缺项或来源不符使汇总拒绝；未知的中断 steps/requests/latency/init-state SHA 保留 null，不填造数值。

### 3.4 新机器复现入口

已发布候选中的 `reproduction/reviewer.py` 与配套指南负责固定源码恢复、独立环境映射、计划预览、有界监督、单机恢复和 B1/B2/O 编排。恢复 143 项冻结运行输入并核对 91 项 guard；补充材料依赖及命令来源列入固定 support 索引。

执行顺序为：四套 smoke 全部完成 → 跨任务切换预检完成 → 正式评测。连接、请求与外层进程均按入口的有界策略处理；失败保留且清理自有进程。单机恢复读取本次计划及已完成唯一键，不冒充旧三机 `round.json`/grant。优化使用新机器可配置的同 die、独立角色 server，不依赖历史主机地址。

**2000 条结果来自原 R7 冻结运行源码；后来新增的 r8 复现入口未参与这 2000 条。** 新入口已经真实 A3 的 entry-only 核验：源码/143 恢复/91 guard、四套 preflight 预览及 B1/B2/O 预览、pidfd 清理和 NPU 空闲，计数为 **0 episode / 0 forward / 0 server**。完整新机器工作流尚未实跑，不宣称已完成完整官方复现。

### 3.5 性能优化方案

保持模型、权重、精度和动作语义，优化已锁定的 KV gather adapter，减少 KV 布局整理与收集开销。比较顺序固定 B1→B2→O，各组独立新 server，同 Long 原权重、物理 die、环境、case、init-state、seed、视频 20/动作 50，关闭 profiler。每组 1 完整预热 + 5 完整测量；B1/B2 仅禁用该 adapter，O 使用冻结实现。

对每条 episode 保存完整 action 客户端请求耗时的中位数，令 `W=max_i(abs(B1_i-B2_i))`、`M=min(median(B1_i),median(B2_i))`。仅当 `median(O)<M` 且 `M-median(O)>W` 才认定超过基线波动；保留失败和不可比 pair，不把该口径当作 kernel 时间或全部 40 任务收益。

## 4. 可维可测分析

### 4.1 已有原 R7 结果与验证范围

正式轮次 SHA：`19128ac5b9f662ba52c7cb582aae87f6f1fd26872e43dc11fb63130cf02b3593`。2000 唯一结果 = 1999 durable + 原 Spatial 中断 1；1941 成功、59 失败，40 任务及 40 真实末内存点齐全。

| 套件 | 成功 / 总数 | 失败 | SR |
| --- | --- | --- | --- |
| Spatial | 479 / 500 | 21 | 95.8% |
| Object | 499 / 500 | 1 | 99.8% |
| Goal | 476 / 500 | 24 | 95.2% |
| Long | 487 / 500 | 13 | 97.4% |
| 算术 Avg | — | 共 59 | **97.05%** |

59 失败包含 58 次策略 800 步 timeout 和 1 次真实中断，全部保留分母，不重跑取优。5 份正式 Long profile 共 184 原请求（38/34/32/36/44）完成语义解析；延迟、Host/NPU 内存按 29 个 server 生命周期解释，不跨机器 monotonic 时钟或跨重启拼增长曲线。

四套同输入复放完成 80 个原 request、80 个 BF16 噪声，包含 reset 与 CUDA 原 action 的 KV 更新；shape/finite 一致，但非逐位一致。官方未给出数值容差，不自行签发任意 numeric PASS。

B1/B2/O 共 18 条成功（各 1 预热 + 5 测量）。`M=10889.016155 ms`、`W=96.781525 ms`、`median(O)=6071.147725 ms`，差额 `4817.868430 ms>W`，同 5 pair 完整 action 客户端请求中位耗时下降约 **44.245%**。补证独立保存，不进入正式 2000 分母。

### 4.2 异常、清理与已知限制

Long task2/3/4、Spatial task4/5/6 六段出现 `server close timeout`，涉及 299 条 durable。根因是旧启动链继承忽略 SIGINT 的状态。原 failure、abort、ledger/ACK、进程身份、cleanup 与事后封口保留，分类为 `ABNORMAL_CLOSE_POSTMORTEM`；不能伪装成原生正常关闭或宣称整轮无异常。

修复仅复位子进程信号再 exec 原 Python，模型、权重、评测配置及原账本不变。修复后 Spatial task7 的 50 条有原生正常关闭及 cleanup 原件。结果完整性、异常闭合与完整稳定性分别说明。

必要短测覆盖服务未就绪、真实请求超时、仿真构造失败与单 episode/action 异常，保存预期故障、状态清理及 3 次真实恢复成功。800 步策略 timeout 不替代请求超时；实际 Spatial 中断续跑不冒充预先故障注入。四套各首连续 20 条清理、reset/switch 与已核 CUDA 输入按适用范围复用。

### 4.3 交付、评审与后续顺序

实现候选包含源码、恢复清单、环境/命令及复现说明；完整小型原件和失败索引在 [Hui_yu 任务私仓](https://gitcode.com/Hui_yu/lingbot-va-ascend-20261018) 的 `codex/lingbot-r7-evidence-r9p1` 分支、`task-delivery/lingbot-va-libero/r9p1/frozen-r9p1` 中定位。已发送 Ascend-CANN 开发者邀请；邀请发出不等于评审实际访问已经证实，访问按官方评审反馈跟踪。

当前验收附件 r9p1 为 3,900,847 字节、94 文件，SHA256 `b066b65d64175125390b77cd6c0ba199841697028dd77554fd3289b4eccc4bd2`。大尺寸 native 原字节曾用于独立审计，随后按用户清理指令删除本机副本；保留逐文件 SHA、派生语义报告及审计范围记录，不把历史本机路径称为当前可下载原件地址。

2026-10-24 提交记录现已被平台退回，原因是“请在设计文档PR合入后再提交验收”。本次按正确活动目录申请设计审核并更新平台进展；**待维护者评审合入后再重交验收**。维护者如要求修改设计或候选，按实际影响补证并重新绑定材料，不迁移旧结论。外部审计、设计批准、平台验收、实现 PR 合入与激励分别记录；本文不代表任何官方放行。目标在任务截止 2026-11-15 前按实际审核结果完成后续流程。
