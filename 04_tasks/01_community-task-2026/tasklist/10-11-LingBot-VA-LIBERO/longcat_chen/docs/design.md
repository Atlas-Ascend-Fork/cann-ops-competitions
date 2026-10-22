# 需求背景

## 需求来源

本设计对应 10 月社区任务「新增 LingBot-VA 模型并完成 LIBERO 四套件评测」，所属 Recipes SIG。任务目录为 `10-11-LingBot-VA-LIBERO`。代码交付目标仓为 [cann/cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai)，本阶段提交/更新设计文档；实现与复测材料在 RFC 锁定后另行交付。任务类型为视频—动作世界模型接入、仿真闭环与基准评测，难度 L3，建议周期 4 周。

## 背景介绍

### LingBot-VA 模型与上游现状

LingBot-VA 是基于自回归视频—动作世界建模（Video-Action World Modeling）的机器人控制模型，将视觉动态预测与动作推断组织在交错序列中，并支持异步执行。上游仓库 [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) 提供独立推理及服务端—客户端架构，LIBERO 客户端与模型环境相互隔离，便于分别管理仿真依赖和模型依赖。

上游公开结果显示，LIBERO Spatial、Object、Goal、Long 的成功率分别为 98.5%、99.6%、97.2% 和 98.5%，算术平均值为 98.5%。本任务要求在昇腾目标平台上复现可追溯评测。**验收门槛以任务书第 3 章验收表为准：四项成功率算术平均值 Avg ≥ 96.0%**（概述若仍写 98% 以表为准）。

### 仓内现状与缺口

`cann-recipes-embodied-ai` 需按仓内「样例目录结构参考」提供 `manipulation/lingbot-va/infer_with_torch/`，通过 overlay/patch 或清晰依赖接入上游，而不是复制整套第三方源码；模型私有逻辑以 monkey patch 使能，不得散落到公共 `module/` 框架中。

### 评测闭环背景

官方评测依赖 LIBERO 四套件 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`（Long），每个 suite 含 10 个任务。默认每任务 50 个 episode，每套件 500、总计 2000 次 rollout。成功仅由 LIBERO 环境 `done` 或官方 evaluator 判定，不允许模型自报成功或从分母剔除失败样本。

### 任务书权重更新（2026-10-20）

| 套件 | 权重来源（锁定） |
| --- | --- |
| 共享底座 | ModelScope / HF `Robbyant/lingbot-va-base`（VAE、tokenizer、text encoder） |
| Long (`libero_10`) | ModelScope / HF `Robbyant/lingbot-va-posttrain-libero-long` |
| Spatial | HuggingFace `Wjjjh/lingbot-va-libero-spatial`（任务书建议版本） |
| Object | HuggingFace `Wjjjh/lingbot-va-libero-object` |
| Goal | HuggingFace `Wjjjh/lingbot-va-libero-goal` |

自训对照路径不作为主验收依据；主路径以建议社区权重 + 官方 Long 为准。下载完成后须将关键 `config.json` 的 sha256 写入样例侧 `configs/checkpoints.yaml`。

# 需求分析

## 需求描述

在 RFC 锁定的 Atlas 平台（本设计默认提案为 **Atlas A3**；正式推理以单卡进程为单位，`ASCEND_RT_VISIBLE_DEVICES` 指定设备）上，完成 LingBot-VA 的 NPU 推理接入与 LIBERO 四套件闭环评测，满足：

1. 模型主干、VAE、文本编码、action head、KV Cache 关键计算在目标 NPU 执行，无 CUDA 残留与核心计算静默 CPU 回退。
2. 保持上游 server-client 协议：健康检查、reset、双相机观测、prompt、动作 chunk、已执行步骤回灌、KV Cache 更新、超时与退出。
3. 四套件全覆盖，恰好 2000 条结构化 episode 明细可复算；`Avg = (SR_spatial + SR_object + SR_goal + SR_long) / 4 >= 96.0%`。
4. 目录、配置、文档、License、错误处理符合 `cann-recipes-embodied-ai` 贡献指南；公共代码改动须回归全部直接调用样例。

## 需求拆解

| 编号 | 拆解项 | 交付要点 |
| --- | --- | --- |
| R1 | RFC 锁定 | 平台/卡数、Recipes 与上游提交、CANN/PyTorch/torch_npu/Python、LIBERO 提交、四套件 checkpoint 与哈希、评测协议；未锁定不得进入最终验收 |
| R2 | NPU 模型接入 | 新增 `manipulation/lingbot-va/infer_with_torch/`；overlay/patch；monkey patch；NPU server 启动与协议 |
| R3 | 权重与配置 | 四套件 checkpoint 来源/revision/哈希；`attn_mode` 推理禁止 `flex`；`action_snr_shift=0.05`；**按套件 action quantile** |
| R4 | LIBERO 评测 | 40 任务 × 50 episode；固定 init states/seed/相机/反归一化；失败保留；断点续跑键 `(suite, task, episode)` |
| R5 | 可复现与稳定性 | 独立环境说明；超时/异常可诊断；内存不泄漏；中断可续跑 |
| R6 | 优化与文档 | 图模式/融合/预取等优化过程；README；结构化结果与性能基线 |
| R7 | 工程质量 | pre-commit、CI、公共回归；权重/资产/凭据/视频不入库 |

# 详细设计

## 算子分析

本任务主体是 **模型样例接入与评测闭环**，而非单一 aclnn 算子。推理图中的关键算子由 `torch_npu` / 图模式 / 融合算子承载；仅当某算子无法直接替换时，才走自定义 Ascend C 路径，且须先按贡献指南提交至算子仓，禁止在 recipes 仓内偷偷实现核心算子。

### 数学公式

视频—动作交错自回归扩散（概念层）：

1. 观测编码：`(I_agent, I_wrist) → z_t = VAE_encode(I)`，输入分辨率固定 `128×128`。
2. 条件与历史：文本条件 `c`、历史 latent/action 在 `attn_window=30` 内构成上下文；`frame_chunk_size=4`。
3. 视频扩散：对 latent 噪声执行 `T_v = 20` 步去噪，得下一帧 chunk 预测。
4. 动作扩散：对动作噪声执行 `T_a = 50` 步去噪，在 `used_action_channel_ids=[0..6]` 上输出动作；`action_snr_shift=0.05`。
5. 执行与回灌：环境执行动作 chunk 后回灌已执行步骤，更新 KV Cache；episode `reset` 时清空缓存。

单套件与最终指标：

- `SR_suite = 成功 episode 数 / 500 × 100%`
- `Avg = (SR_spatial + SR_object + SR_goal + SR_long) / 4`，要求 **`Avg >= 96.0%`**（任务书第 3 章表）

### 支持数据类型

| 模块 | 推理常用 dtype | 说明 |
| --- | --- | --- |
| 图像预处理 | uint8 → float | Host/边界 CPU 允许；进入模型前上 NPU |
| VAE / Transformer / 文本编码 | float16 / bfloat16 / float32 | 以 RFC 锁定精度为准；关键 matmul/attention 在 NPU |
| 动作输出 | float32 | 按套件 quantile 反归一化后下发仿真 |
| KV Cache | 与主干一致 | 必须在 NPU 管理，禁止静默落到 Host |

### 支持形状

| 张量/结构 | 形状约束 | 备注 |
| --- | --- | --- |
| 双相机图像 | `2 × 3 × 128 × 128`（每观测） | 相机翻转与套件配置绑定 |
| frame chunk | `frame_chunk_size=4` | Host 按 chunk 调度 |
| attention 窗口 | `attn_window=30` | KV Cache 滚动窗口 |
| 动作通道 | 7 维，`[0..6]` | 与 checkpoint / 套件 quantile 绑定 |
| 评测批量 | 4×10×50 = 2000 episodes | 不得隐式缩短分母 |

### 接口定义（样例侧，非 REG_OP）

| 接口/入口 | 输入 | 输出 | 参数 |
| --- | --- | --- | --- |
| NPU Server 启动 | 模型配置、runtime checkpoint、设备号、端口 | 健康检查就绪 | `attn_mode∈{torch, flashattn, npu}`，禁止 `flex` |
| Runtime 准备 | 原始 HF/MS 权重树 | 可推理 runtime 目录 | `prepare_runtime_ckpt`：改写 `attn_mode`，对齐 base VAE |
| `reset` | suite/task/episode、init state、seed、prompt | KV Cache 清空确认 | 跨任务切换必须全清 |
| 观测请求 | 双相机图像、已执行步、prompt | ack | 协议不得丢字段 |
| 动作生成 | 当前观测 + cache | action chunk、延迟统计 | 视频 20 步 / 动作 50 步 |
| Client 评测 | `--libero-benchmark`、`--test-num 50`、`--task-range start end` | `*_True.mp4` / `*_False.mp4` | 支持 resume；端口可配置 |

上游参考：`evaluation/libero/launch_server.sh`、`launch_client.sh`、`wan_va/configs/va_libero_cfg.py`。固定上游提交：`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`。

## 算子实现

### 实现方案

交付路径：

```text
manipulation/lingbot-va/
├── infer_with_torch/          # bootstrap、locked_eval、checkpoints、serve_npu、split_eval、下载与 runtime 准备
└── train/                     # 可选对照；非本任务主验收路径
```

私有逻辑通过 monkey patch 挂到公共入口；公共 `module/` 若改动，须列出全部直接调用样例并附回归结果。

#### 推理与评测进程模型

```text
serve_npu.py (NPU, 1 进程/卡)  ◄──WebSocket──►  split_eval_client.py (LIBERO sim)
        │                                              │
   visualization_*                              eval_<suite>/.../*_{True,False}.mp4
```

- **Resume 键**：`(suite, task, episode)`；已有结果文件则跳过。  
- **动作归一化**：`action_norm_method=quantiles`，**禁止跨套件混用** quantile。  
- **并行加速（不改变协议）**：同一套件可按互斥 `task-range` 拆到多张卡 / 多机；产物合并到同一 out-dir。快机可帮慢机尾部 task（独立临时目录再同步）。协议仍要求每套件恰好 500 条、同一权重与超参。

#### Host / Tiling 设计

本任务的 Host 侧对应 **模型 server 进程与评测调度**；Tiling 对应 **序列/扩散步/缓存切分**，自定义 Ascend C 算子时再落到核内 UB 切分。

1. **Host 进程模型**  
   - 模型 server 与 LIBERO client 分环境部署；端口、地址、权重路径全部配置化。  
   - Host 负责协议编排：健康检查 → reset → 观测入队 → 调度扩散步 → 回写动作 → 异常超时诊断。  
   - workspace：当前 chunk latent、双相机观测缓存、动作反归一化缓冲、请求延迟环形缓冲；不把权重路径或凭据写入仓库。

2. **分核策略（自定义算子路径）**  
   - 样例主路径使用 PyTorch + `torch_npu`，由运行时自动分核。  
   - 若某算子必须自定义：按核数对 token/元素均分；能均分则无大小核，不能均分则余块分给前若干核（满核优先）。  
   - 分核参数写入 tiling 结构并下发 kernel。

3. **分块 / UB / 切分**  
   - 序列切分：`frame_chunk_size=4`，`attn_window=30`；KV Cache 按窗口滚动，episode reset 与跨任务切换时整段释放。  
   - 扩散切分：视频 `T_v=20`、动作 `T_a=50`，Host 逐步调度，避免一次性占满 NPU 显存。  
   - 自定义算子：按 UB 容量与是否 double buffer 计算 tile 大小；尾块并入最后一次搬运，避免碎片。

4. **tilingkey 规划**  
   - key0：四套件之一的配置哈希（Spatial/Object/Goal/Long），禁止跨套件隐式改参。  
   - key1：attention 实现分支（`torch_npu` / 融合 attention / 自定义 kernel）。  
   - key2：是否启用量化或图模式优化。  
   Host 侧根据 RFC 锁定配置选择 tilingkey，保证可复现。

5. **workspace**  
   - 常驻：模型权重、KV Cache、套件 quantile。  
   - 临时：单次请求的观测、latent chunk、动作噪声轨迹。  
   - 评测侧 workspace：2000 条明细落盘目录（外部制品），仓库只保留汇总与哈希索引。

#### Kernel 设计

Kernel 侧描述 NPU 上的数据流；对 `torch_npu` 路径，等价于图中各算子在 Device 上的 CopyIn → Compute → CopyOut。自定义 Ascend C 核函数时按同样三阶段实现。

1. **CopyIn**  
   - 将双相机 `128×128` 观测与文本 token 从 Host/协议缓冲搬入 Device。  
   - VAE encode 输入对齐到 NPU 可接受布局；禁止在核心路径保留 CUDA API。

2. **Compute（核函数 / 图计算）**  
   - VAE encode → 交错视频—动作 Transformer（含 attention 与 KV Cache 更新）→ action head。  
   - 文本编码与主干同设备执行。  
   - 扩散循环在 Device 上完成 20/50 步；Host 仅做步调度与协议，不把 matmul/attention 静默回落 CPU。  
   - 自定义 Ascend C 算子：实现 Init/Process，Process 内 CopyIn → Compute → CopyOut；dtype 分支与 tilingkey 一致。

3. **CopyOut**  
   - 输出 7 维动作 chunk；Host 侧按**该套件** quantile 反归一化后经协议返回 client。  
   - 同步写出首 action / action chunk 延迟采样，供 profiling。

4. **状态与异常**  
   - `compute_kv_cache` 与 episode 生命周期绑定；超时、无效 action、仿真创建失败均记失败并保留明细。  
   - 连续 episode 验证无状态污染；profiler `profile_memory` 检查峰值。

# 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas A3 | √ | 本设计 RFC 提案默认目标平台；正式推理按单卡进程验收 |
| Atlas 800I/T A2 | √ | 任务书允许 A2/A3；若改锁 A2，须重跑完整评测与文档修订 |
| Atlas A5 / 950 |  | 可用于训练对照，**非**本任务默认推理验收平台 |

## 算子约束限制

| 约束项 | 限制说明 |
| --- | --- |
| 平台 | 目标平台未 RFC 锁定前不得最终验收；本设计提案为 A3 |
| 权重 | Long 用官方 post-train；S/O/G 用任务书建议的 Wjjjh HF；须登记 sha256；禁止用不合法/未锁定权重冒充验收 |
| 精度模式 | 推理 `attn_mode` 仅为 `torch` / `flashattn` / `npu`；禁止 `flex`；缺 VAE 的残缺目录不得直接 serve |
| 设备边界 | 主干/VAE/文本编码/action head/KV Cache 禁止 CUDA 残留与核心静默 CPU 回退；CPU 仅限明确前后处理 |
| 评测协议 | 不得改成功判定、跳过失败任务、缩短分母、用测试数据再训练凑门槛 |
| 入库 | 不提交权重、LIBERO 资产、凭据、2000 条 rollout 视频；只记录来源、版本、哈希与脱敏配置 |
| 自定义算子 | 新增/修改须先提交算子仓，不得在 recipes 样例中隐藏实现 |

### RFC 锁定摘要（执行用）

| 项 | 锁定值 |
| --- | --- |
| 上游 | Robbyant/lingbot-va `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` |
| CANN | 9.1.0-beta.3 |
| Python | 3.11.6 |
| torch / torch_npu | 2.12.0+cpu / 2.12.0 |
| 协议文件 | `configs/locked_eval.yaml` |
| 门槛 | Avg ≥ 96.0%，恰好 2000 条 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | `Avg = (SR_spatial+SR_object+SR_goal+SR_long)/4 >= 96.0%`；四项 SR 分别公开；恰好 2000 条唯一 `(suite,task,episode)`；成功只认 LIBERO `done`/官方 evaluator；失败不从分母删除 | 任务书第 3 章验收表 |
| Smoke | 正式评测前每套件至少 1 个单 episode smoke（共 4 条）及至少 1 次跨任务切换全部成功 | 自验 README |
| 配置一致性 | checkpoint 与 `action_snr_shift`、动作通道、**套件 quantile**、相机、推理步数绑定并可追溯哈希；四套件无手工隐式改参 | 任务书 |
| 性能标准 | 无强制绝对时延门槛；须记录首 action 与 action chunk 的均值/P50/P95、吞吐；优化前后同 suite/task/init/seed 至少各 5 episode，优化后中位延迟低于优化前且超过基线波动 | 任务书 / 自验 |
| 存储/稳定性 | `profile_memory`；40 个任务各采样一次 NPU/Host 内存；扣除框架常驻后不得持续单调增长；完整评测无 OOM、死锁、结果损坏 | 自验 README |
| 公共回归 | 公共代码默认行为不变；直接调用样例功能与精度测试通过 | 任务书 |

## 兼容性分析

- 新增目录 `manipulation/lingbot-va/`，默认不影响既有 manipulation 样例。  
- 若必须改公共 `module/`：列出全部直接调用方，附回归结果后再合入。  
- 上游提交固定 `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`；升级上游须新开 RFC 并重跑 2000。  
- 依赖 License（代码、权重、LIBERO 数据）在 RFC 与 README 中列明；权重/数据不进仓。

## 测试与验收方案

### 环境与基线

1. 安装独立模型环境与独立 LIBERO 仿真环境；`npu_bootstrap` 打印 `npu_bootstrap_ok`。  
2. 下载 base + Long + Wjjjh 三套件；`prepare_runtime_ckpt` 后确认 `attn_mode ≠ flex` 且 VAE 齐全。  
3. 为四套件分别固定 checkpoint、来源、revision、哈希与配置哈希；登记套件 quantile。  
4. 每套件 1 episode smoke 通过后再开 `TEST_NUM=50` 正式评测。

### 正式评测命令骨架

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# runtime 准备（示例）
bash manipulation/lingbot-va/infer_with_torch/prepare_runtime_ckpt.sh libero_10

# server（NPU）
bash manipulation/lingbot-va/infer_with_torch/launch_server.sh

# client：单套件正式评测（可按 task-range 拆分并行）
python manipulation/lingbot-va/infer_with_torch/split_eval_client.py \
  --libero-benchmark libero_spatial \
  --task-range 0 10 \
  --test-num 50 \
  --port <configurable> \
  --out-dir outputs/libero_spatial

# 汇总
python manipulation/lingbot-va/infer_with_torch/eval_aggregate.py --out-dir outputs/
```

### 交付检查清单

| 检查项 | 通过条件 |
| --- | --- |
| 模型接入 | 目标 Atlas 可加载锁定权重并稳定输出有效动作 |
| 协议闭环 | 健康检查/reset/双相机/prompt/chunk/回灌/跨 episode 清理正确 |
| 评测完整性 | 40 任务全覆盖；2000 条可复算；续跑键唯一 |
| 成功率 | **Avg≥96.0%** 且四项 SR 公开 |
| 文档与 PR | README、优化过程、结构化结果索引、关联 RFC/Issue；pre-commit + CI + 实机验收 |

### 已知风险与对策

| 风险 | 对策 |
| --- | --- |
| `attn_mode=flex` / 缺 VAE | 强制 `prepare_runtime_ckpt`；启动前校验 |
| 跨套件 quantile 错配 | server 按 suite 加载对应 quantile；禁止混用 Long 统计 |
| 多路并行写冲突 | `task-range` 互斥；帮工使用独立临时目录再合并 |
| 长评测中断 | `(suite,task,episode)` 续跑；配置哈希不一致则重跑该 suite |
| 内存增长 | 40 点采样 + reset 后 cache 释放验证 |

### 正式评测结果（2026-10-21）

| 套件 | True / False | SR |
| --- | ---: | ---: |
| Spatial | 473 / 27 | 94.6% |
| Object | 497 / 3 | 99.4% |
| Goal | 482 / 18 | 96.4% |
| Long | 487 / 13 | 97.4% |
| **Avg** | **2000 / 2000** | **96.95%** ≥ 96.0% **PASS** |

相关链接：

- 公开样例 PR：https://gitcode.com/cann/cann-recipes-embodied-ai/merge_requests/231  
- 私仓（已邀 Ascend-CANN）：https://gitcode.com/longcat_chen/cann-recipes-embodied-ai-LingBot-VA  
- 验收包：私仓 `task_submission/acceptance_pack.zip`

---

**文档阶段说明**：本文件用于设计评审与任务系统「设计文档」链接。权重、视频、凭据不入库。代码样例见 `cann-recipes-embodied-ai` PR；自验证材料见私仓 `task_submission/`。
