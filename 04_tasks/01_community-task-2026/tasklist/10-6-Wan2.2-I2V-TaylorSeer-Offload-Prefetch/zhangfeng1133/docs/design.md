# 需求背景（required）

## 需求来源

社区任务 10-6「10月社区任务-Wan2.2-I2V 使能 TaylorSeer、Offload 与 Prefetch」，在 Atlas A3 上基于 [cann/cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer)，为 Wan2.2-I2V 的 dit cache 适配 TaylorSeer 方法，并支持 Offload 与 Prefetch。流程依据[社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)与目标仓 `CONTRIBUTION.md`；本设计为设计评审稿，验收以任务书为准。

## 背景介绍

### 仓内现状（工作量边界）

| 模块 | 现状 | 本任务动作 |
| --- | --- | --- |
| `module/dit_cache/cache_method.py` | **TaylorSeer 类已存在**（同文件含 CacheManager/BaseCache/FBCache/TeaCache/NoCache） | 不重写方法本体；打通 Wan2.2-I2V 接入与配置 |
| `models/wan2.2-i2v/wan/cache/cache_block.py` | 模型侧 cache 挂接点 | 扩展接入 TaylorSeer 分支与调用契约 |
| `models/wan2.2-i2v/wan/modules/model.py` | DiT block 前向 | 增加 cache 钩子（跳算/复用）与 offload 感知 |
| Offload/Prefetch | 仓内无 DiT 权重/激活 CPU offload 通用实现 | 新增 offload 管理器 + 流水预取 |

TaylorSeer 语义：以固定间隔执行完整 DiT 计算（fresh step），中间步用缓存历史输出做泰勒外推（差分预测 + 每层残差校正），跳过的 block 直接复用预测输出，从而省算；Offload 解决 TaylorSeer 跳算后"省算不省显存"的问题，把冷权重/激活搬出 HBM，Prefetch 用 NPU 计算/NPU-CPU 拷贝流水掩盖搬运时延。

# 需求分析（required）

## 需求描述

在 Wan2.2-I2V 上提供三能力一体的推理路径：TaylorSeer 缓存跳算、DiT 权重 CPU offload、访问前异步预取；开关可独立启停、可组合；输出与全量计算路径在任务书质量门槛内一致；A3 实测端到端时延与峰值内存双指标达标。

## 需求拆解

1. TaylorSeer 接入：`cache_block.py` 挂接 CacheManager(TaylorSeer)；配置贯通（间隔、外推阶数、fresh 阈值）；跳算调度确定性。
2. Offload：DiT transformer block 级权重 offload（跳算窗口外权重驻 CPU pinned 内存），offload 粒度/水位可配。
3. Prefetch：按步调度预取下一批所需 block 权重（独立拷贝流 + event 依赖），双缓冲掩盖；与跳算调度器共享步号避免取错批。
4. 质量：视频 PSNR/SSIM 与全量 BF16 基线对照过任务书门槛；固定 seed 可复现。
5. 性能：端到端时延、峰值 HBM 占用对照基线（无 cache / 仅 TaylorSeer / TaylorSeer+Offload+Prefetch 三档），门槛以任务书为准。
6. 兼容：不开开关时行为零变化；与 matmul-int8（10-5）正交组合验证。

# 详细设计（required）

## 功能分析

### 调度与数据面

- **调度**：去噪共 T 步，TaylorSeer 以 interval 步为周期：fresh step 全量计算并写入各层缓存（保留最近 K 步输出做差分），中间步逐层用 `Ô_t = Σ_k Δ^k·t^k/k!` 形式的外推值替代该 block 前向。调度表（步号→fresh/复用）在 step 0 一次性生成，全链路共享，保证逐位确定。
- **Offload**：以 transformer block 为粒度。fresh/即将访问的 block 常驻 HBM；窗口外 block 权重 offload 至 CPU pinned。水位策略：HBM 余量 < 阈值时按 LRU（跳算步号距离）驱逐。
- **Prefetch**：对"下一步将 fresh 的 block 集合"提前发起 CPU→NPU 拷贝（专用 stream + pinned 双缓冲），用 event 串联"拷贝完成 → 该 block 可用"；fresh 步号由调度表静态可知，prefetch 窗口深度可配（默认 2 批）。

| 项 | 契约 |
| --- | --- |
| offload 粒度 | transformer block（attn+ffn 权重整块），BF16 原样搬运不转精度 |
| prefetch 一致性 | 同一块不被计算流与拷贝流并发读写：event 依赖 + 步号版本号 |
| 降级 | offload 关闭时 Prefetch 自动无效；TaylorSeer 关闭时 Offload/Prefetch 仍可独立用于显存不足场景 |
| 失败语义 | 拷贝失败/OOM 逐层回退常驻并记审计日志，不静默换结果 |

## 接入与实现设计

### host 侧（工程接入）

1. **配置**：`config/taylorseer_offload.yaml`（`model_args: dit-cache: taylorseer, taylorseer-interval: K, taylorseer-order: M, offload: true, prefetch-depth: 2`）；`infer.sh` 入口。
2. **接入点**：`cache_block.py` 构造 CacheManager(TaylorSeer) 并注入 `model.py` block 前向；复用判定在 block 入口完成（跳算时不进入 attention/mlp 子模块）。
3. **OffloadManager**（新增，`models/wan2.2-i2v/wan/cache/offload.py`）：注册 block → 按水位 offload/recall → 提供 `ensure(step)` 接口供 Prefetch 调度；CPU 侧 pinned buffer 池按 block 字节数预分配。
4. **PrefetchScheduler**（同文件）：依据调度表生成每步 prefetch 计划，与 OffloadManager 共享版本号；拷贝流与计算流用 record/wait event 同步。
5. **审计日志**：命中/跳算统计、offload 次数与字节、prefetch 等待时长（profiling 口径）；正式运行不逐调用打印。

### 评测设计

| 层 | 方法 | 门槛 |
| --- | --- | --- |
| 调度正确性 | 单测：调度表生成/外推数值对拍参考实现（CPU float32） | 逐位/容差内一致 |
| 质量 | 同 seed 三组 prompt，PSNR/SSIM 对照全量基线 | 任务书门限 |
| 性能 | 三档（无 cache / 仅 TS / TS+Offload+Prefetch）×(≥1 预热 + ≥3 正式) | 端到端时延、峰值 HBM 对照表，门槛以任务书为准 |
| 流水有效性 | profiling：prefetch 等待事件时长占端到端比例 | 报告并给出收敛迭代方向 |
| 兼容 | 全关时同 seed 输出与上游 master 逐位一致；与 int8 组合用例 | 零回归 |

### 风险与对策

| 风险 | 对策 |
| --- | --- |
| TaylorSeer 高阶外推在交叉注意力层发散 | 分层配置（文本交叉层降阶/缩短间隔）；质量门槛卡点 |
| prefetch 取错批（步号错位） | 调度表为唯一事实源，版本号断言，错配直接报错不静默 |
| pinned 内存不足（权重 20GB+） | 水位自适应收缩 offload 窗口；启动时校验可分配 pinned 上限 |
| offload 与 CFG 并行分支并发访问 | 两个 CFG 分支共享同一份步号调度，prefetch 按"分支集合"整体 ensure |
