# 需求背景（required）

## 需求来源

CANN 社区任务 2026，任务编号 **10-11**，任务名称 **LingBot-VA-LIBERO**。

- 任务书：CANN 社区任务 2026 / 具身智能模型样例接入
- 提交仓库：`cann/cann-ops-competitions` → `04_tasks/01_community-task-2026/tasklist/10-11-LingBot-VA-LIBERO/`
- 参考流程：`04_tasks/01_community-task-2026/README.md`、`resources/design_template.md`

## 背景介绍

### LingBot-VA 模型简介

LingBot-VA 是面向机器人操作的**自回归视频—动作世界模型**：以视频动态预测为中间表征，
在同一套交错序列里推断未来动作。

| 组件 | 实现 | 说明 |
|---|---|---|
| 视频 + 动作主干 | Wan2.2 DiT（30 层 / 24 头 × 128 / ffn 14336） | 上游 `wan_va/modules/model.py`，非 diffusers 版本 |
| 动作专家 | MoT（mixture-of-transformers），与主干共享 30 个 block | 按 `action_mode` 路由 |
| VAE | `AutoencoderKLWan`（Wan2.2 版，z_dim = 48，16× 空间压缩） | 流式编码，带时序缓存 |
| 文本编码器 | UMT5 | |
| KV Cache | 2160 slot/block，FIFO 淘汰 | LIBERO 配置下约 1.6 GB |

单次推理要做 **20 个视频步 + 50 个动作步 = 72 次 DiT 前向**，每次前向仅 128 个 token。

### 上游现状分析

上游（<https://github.com/Robbyant/lingbot-va> @ `7c6ffa9b`）只提供 CUDA 路径，直接搬到昇腾会在
8 个位置阻断运行。逐一实测记录如下：

| # | 阻塞点 | 现象 |
|---|---|---|
| 1 | `flash_attn` 顶层无条件 import | `ModuleNotFoundError: No module named 'flash_attn'` |
| 2 | `flex_attention` 同上 | 同上 |
| 3 | `VA_Server.__init__` 硬编码 `torch.device(f"cuda:{rank}")` | `AssertionError: Torch not compiled with CUDA enabled` |
| 4 | 用 `torch.distributed.run` 启动 | 单卡下无谓的进程组初始化 |
| 5 | FSDP 包装 | 单卡不需要，且与 HCCL 语义不一致 |
| 6 | `sever_utils.py` 三处 `device='cuda' if torch.cuda.is_available() else 'cpu'` | 昇腾上产出 CPU 张量再进集合通信，硬失败 |
| 7 | RoPE 用 `torch.polar` + `view_as_complex` + **fp64** | fp64 Matmul 不支持（`aclnnMatmul` 161002），且 torch_npu 把 double **静默转 float** |
| 8 | `save_async` 判断 `obj.is_cuda` | NPU 张量暴露的是 `is_npu`，会走错分支 |

### 昇腾适配现状

截至本任务开始，`cann-recipes-embodied-ai` 已接入 Pi0 / Pi0.5 / ACT / OpenVLA / SmolVLA /
DiffusionPolicy / Isaac-GR00T / PUMA / Spirit v1.5 等样例，但**没有覆盖"视频—动作世界模型"这一类**。
该类负载的特征（小 batch、长序列、自回归、去噪迭代多步）与已有样例差异明显，值得单独沉淀。

# 需求分析（required）

## 需求描述

在昇腾 NPU 上接入 LingBot-VA，使其能够：

1. 加载官方权重、生成有效动作，不依赖任何 CUDA 运行时，且无核心算子静默回退到 CPU；
2. 完整复现上游"服务端—客户端"协议（`reset` / `infer` / `compute_kv_cache`）；
3. 提供 LIBERO 四套件评测工具，输出可复算的逐 episode 明细与汇总，评测协议固定、结果可复现。

## 需求拆解

| # | 需求 | 验收方式 |
|---|---|---|
| 1 | 8 个 CUDA 阻塞点全部处理，无静默跳过 | 补丁报告逐项 `installed: true` |
| 2 | RoPE 复数运算在 fp64 不可用的前提下数值正确 | 独立单元测试（不依赖 torch），`max_abs_err <= 1e-15` |
| 3 | 峰内存可单卡承载（不靠 offload 掩盖） | `npu-smi` 实测峰值 HBM |
| 4 | 协议三动词闭环，KV Cache 回灌时序正确 | 单 episode 端到端不崩 |
| 5 | 兼容 LIBERO 四套件（Spatial / Object / Goal / Long） | 四套件 task 0 各跑通一次 |
| 6 | 评测可复现：相同 seed 下结果稳定 | 环境确定性自测 + 同 seed 重复运行 |
| 7 | 输出逐 episode 结构化明细与 SR 汇总 | `results/*.md` + 汇总脚本 |
| 8 | 提供性能基线，并给出优化结论 | 消融实验 + 无效优化清单 |

# 详细设计（required）

## 模型分析

### 架构与数据流

沿用上游的**进程分离架构**：模型侧与仿真侧不能共存于同一 Python 解释器
（`torch_npu` 与 `robosuite`/MuJoCo 依赖互相破坏），因此通过 websocket + msgpack-numpy 通信。

```
┌──────────────────────────────┐        ┌───────────────────────────────┐
│ 模型进程（Python 3.11）       │        │ 仿真进程（Python 3.10）        │
│ Ascend NPU + torch_npu       │◄──────►│ robosuite + MuJoCo + LIBERO   │
│ wan_va + lingbot_va_npu_patch│ msgpack│ run_eval_parallel.py          │
└──────────────────────────────┘        └───────────────────────────────┘
```

### 输入输出与形状

| 项 | 形状 / 类型 | 说明 |
|---|---|---|
| 观测输入 | 2 × `(128,128,3)` uint8 | agentview + eye_in_hand，垂直翻转 |
| prompt | string | LIBERO task 的 language 指令 |
| 动作输出 | **`(7, 4, 4)` float32** | (action_channels, frame_chunk_size, steps_per_frame)，共 16 个环境步 |
| KV 回灌 | 16 帧 × 2 相机 | 必须 16 帧，见 §时序契约 |

### 关键约束

1. **动作通道顺序**固定为 `[dx, dy, dz, dax, day, daz, gripper]`（`used_action_channel_ids=[0..6]`）；
2. **反归一化**用 checkpoint 内 `norm_stat` 的 q01/q99 quantile；
3. **VAE 必须用 Wan2.2 TI2V-5B 版本**（16× 空间压缩、z_dim = 48）。Wan2.1 或 Wan2.2-I2V-A14B
   是 8× / z_dim = 16，通道数对不上（判据是 `config.json` 的 `in_channels=48`）；
4. `config.json` 的 `attn_mode` 需按上游 README 改为 `torch`（原件与改动后哈希均已存档）。

## 实现方案

### 服务端设计（host 侧）

新增独立补丁层 `lingbot_va_npu_patch.py`，**不改动上游任何源文件**，全部以"导入后替换"介入，
上游升级时冲突面最小。

| # | 阻塞点 | 方案 |
|---|---|---|
| 1, 2 | flash_attn / flex_attention | 惰性桩模块，**必须挂真实 `ModuleSpec`** —— 否则 diffusers 的 `find_spec` 抛 `flash_attn.__spec__ is None` |
| 3 | `torch.device("cuda:N")` | patch `torch.device` 做 `cuda→npu` 重映射，**保留 `__instancecheck__`** 以免下游 isinstance 失效 |
| 4, 5 | 分布式启动 / FSDP | 单卡直接构造 `VA_Server`，跳过 `torch.distributed.run` 与 FSDP |
| 6 | `sever_utils` 三处 device 字面量 | 替换三处实现（仅 world_size > 1 可达） |
| 7 | RoPE fp64 复数 | complex-pair shim 展开 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`，把平台的静默降级改为**显式、可审计、可单测**的 fp32 路径 |
| 8 | `save_async` 的 `is_cuda` | 补 `is_npu` 判断（仅 i2va 模式可达） |

**分核与内存策略**（对应模板中 host 侧 tiling 一节）：

- 单卡单进程，不做算子级 tiling；模型侧的并行由框架负责；
- HBM 预算：transformer 10.18 GB + VAE/UMT5 14.2 GB，实测峰值 **23.1 GiB / 64 GiB**，
  因此**不需要** CPU offload（上游默认 `enable_offload=True`，本样例显式设为 `False`，
  省掉每次请求的同步 H2D）；
- 单卡 64 GB 时余量充足；若目标机为 32 GB HBM 的 A2，需打开 offload 并把 UMT5 与 VAE 放回 CPU。

### NPU 适配层设计（device 侧）

对应模板中 kernel 侧设计的三阶段（搬入 / 计算 / 搬出），本样例的实际对应是：

| 阶段 | 实现 |
|---|---|
| 搬入 | 客户端把 16 帧观测编码后送入；服务端 VAE 流式编码，`feat_cache` 维持时序上下文 |
| 计算 | 72 次 DiT 前向（20 视频步 + 50 动作步），走 `torch_npu` 下发的 SDPA 与 matmul；RoPE 走 fp32 complex-pair shim |
| 搬出 | 动作 `(7,4,4)` 反归一化后返回客户端；`compute_kv_cache` 阶段把 16 帧写入 KV Cache |

attention 后端提供两档，默认保守、可切换：

- `LINGBOT_ATTN_BACKEND=sdpa`（默认）：沿用上游 `custom_sdpa`，torch_npu 自动降到融合算子；
- `LINGBOT_ATTN_BACKEND=fusion`：显式走 `torch_npu.npu_fusion_attention`，
  配置为对全部 gathered slot 做稠密注意力，与上游语义等价。

### 时序契约设计（本样例最关键的约束）

协议要求严格按：

```
reset → infer → 执行 16 个环境步 → compute_kv_cache(16 帧) → infer → ...
```

`compute_kv_cache` **不是可选的性能优化**，它同时维持 Wan VAE 流式编码器的时序缓存。
跳过它直接连续 `infer`，第二次必然崩在 `WanCausalConv3d`：

```
RuntimeError: Calculated padded input size per channel: (2 x 16 x 16).
              Kernel size: (3 x 1 x 1).
```

回灌帧数必须为 **16**：Wan VAE 时序压缩 `T_lat = 1 + floor((T-1)/4)`，只有 16 帧得到
`T_lat = 4`，恰好一个 `frame_chunk_size`，KV Cache 的 128 token/chunk 才成立。

另一条架构约束：**一个 server 只能服务一个 client**。`streaming_vae.feat_cache` 与 attention
KV Cache 都是进程级全局状态，多客户端并发会互相踩 VAE 时序缓存（实测 4 客户端并发时
2/4 立即崩溃）。并行必须靠多卡多 server。

### 可复现性设计

排查中发现两类随机性，缺一不可固定：

| 来源 | 表现 | 处置 |
|---|---|---|
| LIBERO `env.reset()` 用 numpy 全局 RNG 采样物体摆放 | 同一 task / init state 的初始画面不同 | 客户端每次 reset 前 `random.seed` + `np.random.seed`，种子 = `base + task_idx*1000 + episode_idx` |
| 上游 server 用 `torch.randn` 初始化 latents/actions，全仓无 `manual_seed` | 每次采样路径不同 | server 每次 reset 前 `torch.manual_seed` + `torch.npu.manual_seed_all` |

未固定时，同一 episode 的初始画面差异达 **2966 个像素 / 最大通道差 191**，
每个 episode 实际上是不同的任务实例，成功率没有意义。

### 评测工具设计

| 设计点 | 说明 |
|---|---|
| 逐 episode 结构化明细 | 一条 JSON 含 suite/task/episode/seed/观测指纹/步数/请求数/时延序列/终止原因 |
| 断点续跑 | 唯一键 `suite\|task\|episode`，中断后只补未完成的 |
| 双成功口径 | 同时记录 `env.check_success()`（LIBERO 官方）与 robosuite `done`（上游 client 口径），口径可后切换而不必重跑 |
| 无文件锁 | 每个 worker 写自己的 JSONL，并发安全 |
| 覆盖率校验 | 汇总时校验"10 任务 × N episode 是否齐全"，不完整直接报错而非静默平均 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实测情况 |
|---|---|---|
| Atlas A3（Ascend910_9382，2 die × 64 GB HBM） | √ | 已实测：峰值 23.1 GiB，单请求 7.2~11 s |
| Atlas A2（Ascend910B，32 GB HBM） | ○ | 未实测。按 HBM 预算推算需要打开 `--offload`（UMT5 11.36 GB + VAE 放 CPU） |

软件栈实测版本：CANN 9.0.0（`V100R001C10SPC001B250`）、torch 2.10.0+cpu、
torch_npu 2.10.0、Python 3.11.15、驱动 26.1.1。

## 约束限制

1. **单 server 单 client**：KV Cache 与 VAE `feat_cache` 为进程级全局状态，不支持并发客户端；
2. **`compute_kv_cache` 不可省略**，且回灌帧数必须为 16；
3. **两侧 RNG 必须固定**，否则结果不可复现；
4. **图模式不可用**（当前环境缺 `triton` 与 `torchair`），需在组件镜像中启用 Triton-Ascend；
5. `/healthz` 在推理期间不可用 —— handler 内 `policy.infer()` 阻塞事件循环约 11 s；
   判断服务存活应看 `npu-smi` 的 HBM 占用（加载后约 28 GB）；
6. **Spatial / Object / Goal 三个套件官方无 checkpoint**，无法评测，详见 §可维可测分析。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 | 实测结果 |
|---|---|---|---|
| 功能标准 | NPU 上加载锁定权重、生成非退化动作，无 CUDA 依赖、无算子静默回退 | 任务书 3.1 / 3.2 | ✅ 达成：23.1 GiB HBM，动作 std 0.34~0.39 |
| 精度标准 | LIBERO 四套件 `Avg = (SR_s+SR_o+SR_g+SR_l)/4 >= 98.0%` | 任务书 3.7 | ❌ **未达成：Avg = 19.38%**（Long 77.50%，其余三套件无权重记为 0%） |
| 性能标准 | 单请求推理时延、单 episode 端到端耗时 | 自定义基线 | 7.2 s（KV Cache 小）→ 约 11 s（填满）；单 episode 148.9~550.9 s，均值 354.7 s |
| 内存标准 | 单卡可承载，不靠 offload 掩盖 | 任务书 3.5 | ✅ 23.1 GiB / 64 GiB |
| 复现性标准 | 相同 seed 下结果稳定、逐条可查 | 任务书 3.6 | ✅ 40 集逐条明细含 seed 与观测指纹 |

### 精度实测与偏差说明

| 套件 | 样本 | SR | 说明 |
|---|---|---|---|
| `libero_10`（Long） | 40（4 seed × 10 任务） | **77.50%**（95% Wilson CI 62.5~87.7%） | 9 个失败全部为 `max_steps_reached`，零异常 |
| Spatial / Object / Goal | — | 无数据 | 官方仅发布 Long 一个 posttrain checkpoint |

**已排除的移植侧原因**（均有实测记录）：权重哈希完整性、`attn_mode` 配置、
动作反归一化与通道顺序、`max_steps` 口径（520 vs 800 对照）、RoPE 数值、
平台算子确定性（NPU randn / bf16 matmul / SDPA / layernorm / softmax 均 bit-exact）、
两侧随机种子。剩余差距位于推理实现语义或上游评测口径层面，需参照实现才能进一步定位。

### 性能实测与优化结论

单次 `infer` = 2160 次 block 执行，实测 **108.4 ms/前向 = 3.61 ms/层**，
按单 block FLOPs 换算理论仅需约 0.13 ms，**有效率约 4%** —— 属**延迟受限**
（每次前向仅 128 token、batch = 2，矩阵乘单元严重欠占用）。

消融实验（同进程、模型只加载一次、4 次取中位数）显示：同配置重跑噪声基线为 **−6.5%**，
而单项消融收益分别为 −7.4% / −13.5% / −20.6% / −9.0% —— **没有单点热点**，
开销平摊在约 3.2 万次 kernel 启动上。

已实测且**无收益**的优化（7 项）：融合 attention、mask 替代 gather、文本 K/V 缓存、
索引缓存、显式 fp32 RoPE、CPU offload、减少去噪步数。后续唯一未试方向
是启用 Triton-Ascend 打通 `torch.compile`。

## 兼容性分析

| 项 | 说明 |
|---|---|
| 上游代码兼容 | **零改动**。适配全部通过运行时替换完成，上游升级时只需复核 8 个阻塞点是否仍存在 |
| 配置兼容 | 仅需按上游 README 把 `attn_mode` 由 `flex` 改为 `torch`（原件与改动后 sha256 均存档） |
| 权重兼容 | transformer 逐文件 sha256 校验；VAE / UMT5 / tokenizer 复用共享盘 Wan2.2 TI2V-5B，11 个文件逐一对齐 |
| 已有样例兼容 | 新增独立目录 `manipulation/lingbot-va/`，不修改任何现有样例与公共代码 |
| 环境兼容 | 仿真环境（Python 3.10 + robosuite）与模型环境（Python 3.11 + torch_npu）完全隔离 |
| 已知不兼容 | `torch 2.6+` 将 `torch.load` 的 `weights_only` 默认改为 `True`，与 LIBERO 旧写法冲突；评测脚本在 import LIBERO 前显式关闭（仓内 init_files 为受信任资产） |

## 可维护性分析

| 维度 | 措施 |
|---|---|
| 可观测 | 每条 episode 记录 `seed`、首帧观测 sha1、逐次请求时延、终止原因，问题可回溯 |
| 可诊断 | 提供带 IO 指纹的调试 server（`_selftest/debug_server_io_fingerprint.py`），可快速区分分歧出在客户端还是模型侧 |
| 可自测 | 三层自测：RoPE shim 数值单测（秒级，不依赖 torch）、环境确定性自测（约 30 秒，不依赖模型）、模型冒烟 |
| 可复算 | 明细 JSON + 汇总脚本，SR 与置信区间可独立重算 |
| 可扩展 | 评测工具与模型侧解耦，新增套件只需改配置 |

---

## 附录：AI 参与情况

本设计文档与对应样例的开发过程中使用了 AI 辅助工具，用于源码梳理、方案设计、
实验设计与数据分析、文档撰写与格式检查。文中所有实测数据均来自本环境真实运行记录
（时间、步数、请求数、耗时逐条可查于样例仓库的 `results/long_40ep_summary.md`），
未做推断性填充。
