# LingBot-VA 昇腾接入与 LIBERO 四套件评测设计方案（任务 10-11）

# 需求背景（required）

## 需求来源

CANN 社区任务 2026，任务编号 **10-11**，任务名称「LingBot-VA 模型接入与 LIBERO 四套件评测」。

- 任务书：10月社区任务《新增 LingBot-VA 模型并完成 LIBERO 四套件评测》
- 提交仓库：`cann/cann-ops-competitions` → `04_tasks/01_community-task-2026/tasklist/10-11-LingBot-VA-LIBERO/u011078141/`
- 目标平台（RFC 锁定提案）：**Atlas A3（Ascend 910_9382，2 die × 64 GB HBM，单卡）**；全部核心计算与验收在 A3 执行，昇腾 950PR（128 GB HBM）作为**附加实证平台**（移植层真机验证已完成）

## 背景介绍

### LingBot-VA 模型简介

LingBot-VA（[RSS 2026](https://github.com/robbyant/lingbot-va)，@`7c6ffa9`）是视频生成式 VLA 世界模型：以自回归"未来视频预测"为中间表征，在交错序列中联合推断机器人动作。

| 组件 | 实现 | 规模 |
|---|---|---|
| 视频+动作主干 | Wan2.2 DiT（30 层 / 24 头×128 / ffn 14336），MoT 动作专家与主干共享 block | 5.09B |
| 文本编码器 | UMT5 | 5.68B |
| VAE | `AutoencoderKLWan`（z_dim=48，16× 空间压缩），流式编码带时序缓存 | 0.70B |
| KV Cache | 2160 slot/block，FIFO 淘汰 | LIBERO 配置 ~1.6 GB |

单次推理为 **20 视频去噪步 + 50 动作去噪步**（CFG 双 batch，128 token/前向），自回归推进。

### 上游现状分析

上游仅提供 CUDA 路径。我们在 **Ascend950PR 真机逐项实测**的阻塞点：

| # | 阻塞点 | 现象（真机实测） |
|---|---|---|
| 1 | `flash_attn`/`flex_attention` 顶层无条件 import | `ModuleNotFoundError`，import 链断裂 |
| 2 | `VA_Server.__init__` 硬编码 `torch.device("cuda:0")` | NPU 上 assert 失败 |
| 3 | `torch.distributed.run` + FSDP `fully_shard` 启动链 | 单卡无需且与 HCCL 语义不一致 |
| 4 | `sever_utils.py` 三处 `device='cuda'` 字面量 | 集合通信路径硬失败 |
| 5 | 5 处 `torch.cuda.empty_cache()` | NPU 无对应语义 |
| 6 | `save_async` 判断 `obj.is_cuda` | NPU 张量走错分支 |
| 7 | `wan_va_server.py` `sys.path.append` 导致**双模块身份** | patch 打在 server 不绑定的那份模块上（隐蔽：表面 applied 实际失效） |
| 8 | RoPE `torch.polar`+`view_as_complex` 依赖 fp64 | torch_npu 静默 double→float 降级，数值语义待对拍定量 |

### 昇腾适配现状

`cann-recipes-embodied-ai` 已有 Pi0/Pi0.5/ACT/OpenVLA/SmolVLA/DiffusionPolicy/Isaac-GR00T/PUMA/Spirit 等样例，均为"单帧编码 + 动作头"形态；**视频—动作世界模型类（小 batch、长自回归、多步去噪、流式 VAE 时序缓存）尚无覆盖**，迁移点与既有样例不重叠，本任务填补该空类。

# 需求分析（required）

## 需求描述

1. 在 Atlas 950PR 上接入 LingBot-VA：主干/VAE/文本编码/action head/KV Cache 关键计算全部在 NPU 执行，无核心算子静默回落 CPU；
2. 完整保留上游服务端—客户端协议（`reset` / `infer` / `compute_kv_cache` / 健康检查 / 退出），不丢失 prompt、双相机图像或 reset 状态；
3. 提供 LIBERO 四套件评测工具：协议固定、逐 episode 明细可复算、断点续跑、双成功口径；
4. 数值对拍：同 seed 噪声注入下 NPU 与 CUDA action chunk 逐位/容差比对，证明 patch 不改公共行为；
5. 交付性能基线（首 action 延迟、chunk 延迟、HBM 峰值、单 episode 端到端）与优化前后对比。

## 需求拆解

| # | 需求 | 验收方式 |
|---|---|---|
| 1 | 8 个阻塞点全部处理，无静默跳过 | patch 装配报告逐项 `applied: true` + 零 diff 探针（不装 patch 时 import 链零变化） |
| 2 | 真机端到端：server 拉起 + 真实 LIBERO episode | 已实证（见可维可测），最终以四套件全量为准 |
| 3 | 同 seed 可复现 | 噪声 CPU generator 统一入口，同 seed 逐位一致（已实证） |
| 4 | 显存可承载 | npu-smi 实测峰值 HBM + 周期采样无单调增长 |
| 5 | 评测协议固定 | 50 eps × 10 task × 4 套件，mp4 文件名唯一判定源，零人工改判 |
| 6 | 断点续跑 | supervisor + client_resume wrapper，中断后仅补缺失 episode |
| 7 | 对拍 | NPU vs CUDA action chunk（同 seed 噪声）分组件容差报告 |
| 8 | 性能基线与优化记录 | 基线消融 + 优化对比 + 无效优化清单 |

# 详细设计（required）

## 模型分析

### 架构与数据流

沿用上游**进程分离**架构：模型侧（torch_npu）与仿真侧（robosuite/MuJoCo）不能共存于同一解释器，通过 websocket+msgpack 通信。

```
client(reset) → server: reset(prompt) → T5 编码 + KV cache 建立
client        → server: infer(obs=[cam1,cam2], prompt) → action chunk (7,4,4)
client 执行 16 环境步 → server: compute_kv_cache(16 帧) → infer → …
```

### 关键约束

- `compute_kv_cache` **不可省略**且回灌帧数必须为 16：Wan VAE 时序压缩 `T_lat = 1+⌊(T-1)/4⌋`，16 帧恰得整数 `frame_chunk_size`，128 token/chunk 才成立；跳过则第二次 `infer` 崩在 `WanCausalConv3d`；
- **一个 server 只服务一个 client**：KV Cache 与 VAE `feat_cache` 为进程级全局状态，多 client 并发互踩（已实测验证），吞吐并行靠多实例而非 client 复用；
- 双侧 RNG 必须固定：客户端 reset 前 `random.seed`+`np.random.seed`，服务端每次 reset 前 `manual_seed`；未固定时同一 episode 初始画面差异可达数千像素，SR 无意义。

## 实现方案

### NPU 适配层设计（npu_patches，8 项可装配补丁，上游零改动）

| 补丁项 | 设计 |
|---|---|
| flash_attn stub | `flash_attn_varlen_func` 等映射 torch SDPA 等价实现，解除 import 断裂 |
| 顶层别名（startup） | 消除 `wan_va_server` 双模块身份：server 导入**前**预置顶层名→同一模块对象别名，保证 patch 打在 server 实际绑定的身份上 |
| device 重定向 | `torch.device` 元类工厂替身：`__instancecheck__` 转发真类型（保持 diffusers 等第三方 isinstance 兼容），`__call__` 做 cuda→npu 重定向 |
| 分布式单卡化 | 跳过 `torch.distributed.run`/FSDP，单卡直启 `VA_Server` |
| empty_cache / save_async | 前者转发 NPU 等价/空操作；后者同步替换所有持有原函数对象的命名空间（含包级再导出与顶层别名） |
| remote-infer 桩 | 服务模式专用依赖缺失时的受控桩（依赖补齐后退场，no-op） |
| noise 统一 | NPU `randn` 拦截走 CPU generator 后搬运——CUDA/NPU 输入 **bit 级一致**（对拍前提） |

全部补丁可完整卸载（uninstall 后 torch 指针/模块属性/stub 全还原），配零 diff 探针证明不启用时公共行为零变化；CPU 语义单测 34 例在本地 torch 与容器 torch_npu 双环境全绿。

### 数值对拍方法学（分层）

噪声 CPU generator 统一 → ①输入 bit 级一致 → ②组件激活级对拍（VAE/UMT5/主干/action head 分器容差）→ ③episode 级配对统计（同 init state，McNemar）。统计功效依据：50 eps/suite 的 se≈0.63pp，0.3pp 级差异 episode 级无法分辨，故组件级对拍为主。

### 评测工具设计

| 设计点 | 说明 |
|---|---|
| 唯一判定源 | mp4 文件名 `_True/_False`（环境 done），零人工改判；`env.check_success` 官方口径并行记录备切换 |
| 逐 episode 明细 | suite/task/episode/seed/步数/耗时/chunk 时延序列/终止原因（JSONL） |
| 断点续跑 | 唯一键 `suite\|task\|episode`；supervisor 周期巡检 + client_resume wrapper，进程死亡自动重启只补缺口（已历生产中断验证） |
| 并发安全 | 每 worker 独立 JSONL，无文件锁 |
| 覆盖率校验 | 汇总时校验 10 任务×N episode 齐全，缺失报错而非静默平均 |

### 权重与四套件配置

| 套件 | checkpoint | License |
|---|---|---|
| Long | `robbyant/lingbot-va-posttrain-libero-long`（官方） | 上游仓许可 |
| Object/Goal/Spatial | `Wjjjh/lingbot-va-libero-{object,goal,spatial}`（社区 post-train，base=官方 base） | Apache-2.0，revision+SHA256 留痕 |

官方仅发布 Long 权重。三套件社区权重的"合法来源"认定与自训备选（官方配方 + 社区 LeRobot 数据，血缘完整披露）已列入 RFC 前置询问。**提分预算事实**（社区 3-run 实测口径）：即使完美复现参考数字 Avg≈97.4%，spatial/goal 需自训提分逼近任务书 98%。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实测情况 |
| --- | --- | --- |
| Atlas A3（Ascend 910_9382，2 die × 64 GB HBM） | **√（目标平台，验收环境）** | RFC 锁定后部署验证（hidevlab 算力）；HBM 预算：权重常驻 23.8 GB bf16 + 运行时 ≈ 29 GB，64 GB 余量充足；offload 作为小显存备选开关 |
| Atlas 950PR（128 GB HBM） | ○（附加实证平台） | 真机实证：L1-L5 阶梯全 PASS、server 三配置 60/60 chunk、真实 LIBERO episode、推理峰值 29.3 GB/131 GB——移植层（npu_patches）平台无关，950PR 结论可迁移性由 A3 复核确认 |
| A2（Ascend 910B，32 GB HBM） | ○ | 未实测；按 HBM 预算需打开 offload（UMT5/VAE 放 CPU），非验收目标 |

软件栈实测（950PR 容器）：torch 2.13.0+cpu / torch_npu 2.13.0.rc1 / diffusers 0.41.0（补 accelerate 1.15.0）；A3 环境按 RFC 锁定版本部署后补记。CUDA 参照侧 torch 2.9.0+cu128 / transformers 4.55.2（按上游 requirements 锁定）。

## 约束限制

1. 单 server 单 client（KV Cache/VAE `feat_cache` 进程级全局状态）；
2. `compute_kv_cache` 不可省略，回灌帧数必须为 16；
3. 双侧 RNG 必须固定，否则结果不可复现；
4. RoPE fp64→fp32 静默降级影响待组件级对拍定量确认（当前无阻塞，登记已知项）；
5. `argsort` 走 AiCpu 路径（性能非阻塞，登记）；
6. 图模式（torchair/Triton-Ascend）组件镜像未启用，列为性能优化阶段项。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 来源 | 当前状态 |
|---|---|---|---|
| 功能标准 | NPU 加载官方 Long 权重、生成非退化动作，无 CUDA 依赖 | 任务书 3.1/3.2 | ✅ 真机实证：L1-L5 全 PASS，首 action chunk (7,4,4) 合法 |
| 协议标准 | server 三配置 60/60 chunk，KV 语义（增长/reset 清空/同输入逐位一致） | 任务书 3.4 | ✅ 已实证 |
| 端到端标准 | 真实 LIBERO episode 闭环 | 任务书 3.5 | ✅ NPU 首个真实 episode 已通 |
| 精度标准 | 四套件 Avg ≥ 98% | 任务书 3.7 | 进行中（见下） |
| 性能标准 | 首 action 延迟 / chunk 延迟 / HBM 峰值 / 单 ep 耗时 | 任务书 3.9 | 已有基线（见下） |
| 稳定性 | 无状态污染、内存无持续增长、断点续跑 | 任务书 3.10 | ✅ supervisor 真续跑经生产中断验证 |

### 精度实测与偏差说明

| 套件 | 样本 | SR | 说明 |
|---|---|---|---|
| Long（CUDA 参照侧） | 500（10 任务×50） | **94.8%** | 官方权重；对照官方 98.5%/社区复现 97.2%（上游 #112，同权重同规模） |
| Object（CUDA 参照侧） | 500 | **93.0%** | 社区权重；对照 Wjjjh 3-run 区间 [98.6,99.4] |
| Spatial / Goal | — | 排队中 | 基线批按队列执行 |
| 四套件 NPU 侧 | — | 待全量 | 依赖对拍先行 |

差距归因排查已做两轮（渲染后端/物理引擎单因素均为噪声，预注册门槛未达成），当前最高嫌疑为上游 #108 推理 bug（我们栈内实锤存在，修复开关已实现），对照实验进行中。**诚实预期**：现有权重路线复现上限 Avg≈97.4%，98% 达标依赖 bug 修复回收 + spatial/goal 自训提分。

### 性能实测与优化结论

- NPU 常驻路径 **P50 2.97 s/chunk**（三配置对照，offload on/off/常驻全 PASS）；
- HBM 峰值 29.3 GB/131 GB，释放验证回落基线（29.3→6.3 GB）；
- CUDA 参照侧消融：attn 后端 flashattn 84.7 s/ep（5/5）vs torch ~120-184 s/ep；关 offload 在 24G 卡 OOM（记录退出，不重试）；
- 已实测无收益/不可行项均记录在案（环境单因素扰动两类），避免重复投入。

## 兼容性分析

| 项 | 说明 |
|---|---|
| 上游代码兼容 | **零改动**。适配全部运行时替换；上游升级只需复核 8 个阻塞点是否仍在 |
| 配置兼容 | 仅按上游 README 将 `attn_mode` 由 `flex`（训练存档值）改 `torch`（原件与改动后 sha256 均留痕） |
| 权重兼容 | transformer 逐文件 SHA256 与 HF tree API 双向核对（8/8 抽样全对）；VAE/UMT5/tokenizer 与 base 共享件交叉验证 |
| 已有样例兼容 | 新增独立目录 `manipulation/lingbot-va/infer_with_torch/`，不触碰现有样例与公共代码 |
| 环境兼容 | 仿真环境（py3.10 + robosuite）与模型环境（torch_npu 容器）完全隔离；容器离线依赖经 wheel 预打包补齐 |
| Python 兼容 | `torch 2.6+` `weights_only` 默认翻转与 LIBERO 旧写法冲突：评测脚本在 import LIBERO 前显式处理 |

## 可维护性分析

| 维度 | 措施 |
|---|---|
| 可观测 | 逐 episode 记录 seed/耗时/chunk 时延序列/终止原因；GPU/HBM 周期采样入档 |
| 可诊断 | patch 装配报告（逐项 applied/uninstalled 状态）+ 真机七步验证脚本，分层定位 client/server/patch |
| 可自测 | 三层：patch 层 CPU 单测 34 例（双环境）→ 真机 verify 脚本 7 步 → server 三配置冒烟 |
| 可复算 | 明细 JSONL + 汇总脚本，SR/CI 可独立重算；覆盖率校验拒绝静默缺失 |
| 可扩展 | 评测工具与模型侧解耦，新增套件仅改配置 |

---

## 附录：AI 参与情况

本设计与对应开发过程使用 AI 辅助工具，用于源码梳理、方案设计、实验设计与数据分析、文档撰写与格式检查。文中全部实测数据（成功率、耗时、显存、chunk 时延）均来自真机/远程 GPU 环境真实运行记录，明细日志与逐 episode JSONL 在工作区归档可查，未做推断性填充。
