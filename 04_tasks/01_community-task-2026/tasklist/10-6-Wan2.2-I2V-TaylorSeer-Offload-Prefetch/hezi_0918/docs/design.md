# Wan2.2-I2V TaylorSeer、Offload 与 Prefetch（A3）设计文档

- 提交者：hezi_0918
- 对应任务：10月社区任务 - Wan2.2-I2V 使能 TaylorSeer、Offload 与 Prefetch（A3）
- 关联 RFC（含评审讨论与口径澄清）：https://gitcode.com/cann/cann-recipes-infer/issues/457
- 主实现 PR（模型代码，提交至 cann-recipes-infer）：https://gitcode.com/cann/cann-recipes-infer/merge_requests/1040
- 自验材料（四档消融/盲测/profiling/内存）：https://gitcode.com/hezi_0918/cann-recipes-infer/tree/task_submission/task_submission

> 性能结论（Atlas A3 实测，每档预热1+正式3）：基线 364.75s → 最终组合（TaylorSeer+Offload+Prefetch）167.59s，**-54.0%**；二轮盲测（2 名验收人）结论已回收并披露映射。

---

## 一、需求背景

### 1.1 任务来源

本设计对应昇腾社区「10月社区任务 - Wan2.2-I2V 使能 TaylorSeer、Offload 与 Prefetch（A3，L3）」任务书要求：在 Atlas A3 上基于 cann-recipes-infer 开源仓，为 Wan2.2-I2V 图生视频模型的 DiT（Diffusion Transformer）适配 TaylorSeer 特征缓存方法，并支持专家模型 Offload（下放）与 Prefetch（预取），在基线固定口径下实现端到端推理耗时降低 ≥50%，同时满足精度盲测、内存与稳定性等验收要求。实现与自验材料随主 PR 提交。

### 1.2 模型与业务背景

Wan2.2-I2V-A14B 是 Wan 系列 140 亿参数的图生视频模型，采用 **high_noise / low_noise 双专家 DiT 架构**（dim=5120、40 层单流 block），单卡 Atlas A3（64GB HBM）上双专家权重常驻约 2×27GB。视频生成需 40 步去噪、每步执行 cond/uncond 两次 DiT 前向（CFG 分离），端到端一次生成约 6 分钟（基线 364.75s）。推理成本高企的核心原因是**去噪过程存在大量冗余计算**：相邻去噪步之间，DiT 各 block 子模块（self-attention / cross-attention / FFN）的输出特征随时间步平滑变化，具备强时序相关性——这是特征缓存类加速方法（FBCache、TeaCache、TaylorSeer）的物理基础。

### 1.3 TaylorSeer 技术背景

TaylorSeer（arXiv:2503.06923）利用 DiT 特征的时序平滑性，将部分去噪步中子模块输出以**低阶泰勒展开外推**重建，跳过该步子模块的完整计算，仅需维护特征各阶导数。相比 FBCache（特征直拷贝）与 TeaCache（模长差阈值跳步），TaylorSeer 在同档跳步率下精度损失更小、对运动类内容的时序一致性保持更好，且与 CFG 双分支、多专家切换天然正交。本任务将其首次适配到 Wan2.2-I2V 的双专家 + CFG 分离执行结构上，并与专家 Offload/Prefetch 组合以同时降低计算量与显存峰值。

### 1.4 仓内现状与差距

cann-recipes-infer 仓现行状态：① dit_cache 框架已有 NoCache / FBCache / TeaCache 三种方法，无 TaylorSeer；② Wan2.2 的 `_prepare_model_for_timestep` 专家下放判断为 `device.type == 'cuda'`，在 NPU 上不触发，双专家常驻 NPU（HBM 峰值 62.7GB），大分辨率/长时序场景受限；③ 无请求级缓存隔离与 Host/NPU 分层驻留机制。差距 = 缺一套 NPU-correct 的 TaylorSeer 四桶缓存 + 专家生命周期管理（下放/预取）组合方案，且不得影响既有基线路径的逐比特行为。

## 二、需求分析

### 2.1 需求描述

在 cann-recipes-infer 仓为 Wan2.2-I2V 实现：①**TaylorSeer dit-cache**（四桶隔离的缓存状态机 + 一阶泰勒外推）；②**专家 Offload**（非活跃专家与缓存因子下放 Host，降低 HBM 峰值）；③**异步 Prefetch**（boundary 切换前预取下一专家、首专家加载与 VAE 编码重叠）。三能力可独立启停，形成四档消融（baseline / +TaylorSeer / +offload / +offload+prefetch）。

### 2.2 功能需求清单

| 编号 | 需求 | 说明 |
|---|---|---|
| FR1 | 缓存桶隔离 | key={expert,cond/uncond} 四桶（high_cond/high_uncond/low_cond/low_uncond），导数与步进状态互不串扰 |
| FR2 | 步进控制 | warmup（桶内前 N 步强制 full）/ cutoff（全局末尾强制 full）/ skip_interval（周期性 full） |
| FR3 | 外推重建 | predict 步以一阶泰勒公式重建 self_attn/cross_attn/ffn 三子模块输出，残差流照常累加 |
| FR4 | int8 因子压缩 | 缓存因子仅存储压缩（absmax 一次性校准、逐层逐通道 scale），权重与计算全程 BF16 |
| FR5 | 专家生命周期 | NPU-correct 的专家下沉/预取：boundary 步前一拍切换、首专家 H2D 与 VAE encode 重叠、请求级 release_all 隔离 |
| FR6 | 回归不变 | NoCache / FBCache / TeaCache 与 README 基线路径逐比特不变 |
| FR7 | 参数校验 | 非法参数（7 类）启动即报错退出（exitcode 1），参数可配置/可校验/可回退 |
| FR8 | 资源清理 | 生成结束与 SIGINT 中断后 HBM/Host 内存全回收，连续多次生成无累积泄漏 |

### 2.3 性能需求

- 判定口径：同机同配置，预热 1 次丢弃 + 正式 3 次平均端到端耗时，相对基线降低 **≥50%**（任务书硬门槛）。
- 四档消融矩阵：baseline / +TaylorSeer / +offload / +offload+prefetch，逐档归因（计算减少 / 迁移开销 / 重叠收益）。
- 性能不得以牺牲精度换取：与盲测、精度扫描同步验收。

### 2.4 精度需求

- 5 组固定样本（人物/动物/快速运动/镜头变化/复杂背景）× {优化, 基线} A/B 盲测：由 2 名独立验收人对比主体/语义/动作/时序/伪影五维，结论留痕。
- 全日志 NaN/Inf 精确词匹配扫描 = 0；输出视频 A/B md5 两两不同（随机化防猜）。

### 2.5 稳定性与内存需求

- HBM 峰值：优化档 ≤ 基线（prefetch 不得增加峰值内存）。
- 连续 5 次生成 Host 内存增量稳定（无泄漏）；SIGINT 中断后 HBM 回落、进程退出干净。

---

## 第一部分：设计说明书（含需求背景 / 需求分析 / 算子分析 / Host 设计 / Kernel 设计 / 硬件约束 / 可维可测）

本 Issue 为 RFC，承载「10月社区任务 - Wan2.2-I2V TaylorSeer、Offload 与 Prefetch 任务书（A3，L3）」的基线固定、缓存单元语义、offload/prefetch 设计与验收判定标准；对应实现见 PR #1040（分支 `feat/wan22-i2v-taylorseer`，head 8ff16aa）。

四档消融、盲测、内存与 profiling 自验材料随 PR #1040 一并提供。

---

# RFC: Wan2.2-I2V TaylorSeer dit-cache 适配（含 Offload 与 Prefetch）

> 任务: 10月社区任务 - A3上基于cann-recipes-infer仓，为Wan2.2-I2V的dit cache适配TaylorSeer方法，支持Offload与prefetch
> 目标仓: https://gitcode.com/cann/cann-recipes-infer
> 状态: 已完成（实施与四档实测均以修复后代码收口，实测见 §7）
> 关联实现: PR #1040（分支 feat/wan22-i2v-taylorseer，head 8ff16aa）
> 基线锚点: master bbd278f

## 六、硬件与约束（1）：基线固定

| 项 | 值 |
|---|---|
| 模型 | Wan2.2-I2V-A14B-BF16 (high_noise + low_noise 双专家, dim=5120, 40层单流block) |
| 权重 | ModelScope `Wan-AI/Wan2.2-I2V-A14B-BF16`, revision 见自验日志 |
| 输入图 | `examples/i2v_input.JPG` (仓内置) |
| prompt | `Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.` |
| 规格 | i2v-A14B, 640*360, 61帧(4n+1), sample_steps=40, unipc, shift=5.0, seed=100 |
| 并行 | 单卡 (world_size=1, cfg/ulysses/ring=1) |
| 硬件 | Atlas A3, 1×Ascend910 64GB HBM, npu-smi 25.5.5 |
| 软件 | CANN 9.1.0, PyTorch 2.7.1+cpu, torch_npu 2.7.1.post4, Python 3.11.4 |
| 基线配置 | `config/14b_single.yaml`, `dit_cache.method: NoCache`, `convert_model_dtype: true` |
| 计时口径 | generate.py 内建: 预热 1 次(2步) 后, `Generating video ...` 起至 `stream.synchronize()` 止, 含 T5、VAE encode、40步去噪、VAE decode |
| 性能判定 | 同机同配置 3 次正式运行平均端到端耗时, 优化版相对基线降低 ≥50% |

基线行为说明: 仓库现行 `_prepare_model_for_timestep` 的设备判断为 `device.type == 'cuda'`,
在 NPU 上不触发专家模型下放, 即双专家模型常驻 NPU (2×~27GB)。本任务**不修改**该基线路径
(NoCache/FBCache/TeaCache 与 README 基线保持逐比特一致), TaylorSeer 使用独立的
NPU-correct 模型管理路径。

## 三、算子分析（2）：Wan2.2 denoising 结构梳理与缓存单元定义

### 2.1 执行结构

- 每 denoise step 顺序执行两次 DiT forward: cond(正向) → uncond(负向), 即 CFG 两分支分离
  (`enable_separate_cfg: true`)。
- `t >= boundary(=0.9*1000)` 时使用 high_noise_model, 其余使用 low_noise_model。
  unipc + shift=5 下 40 步中约前 14 步为 high 专家, 后 26 步为 low 专家 (以日志实测为准)。
- WanModel.forward: patch embed → 40 × WanAttentionBlock → head → unpatchify。
- WanAttentionBlock (单流): `y1=self_attn(norm1(x,e)); x+=y1*e2; y2=cross_attn(norm3(x),ctx); x+=y2; y3=ffn(norm2(x,e)); x+=y3*e5`。

### 2.2 缓存单元

| 概念 | 定义 |
|---|---|
| cache unit | 每个 block 内 3 个子模块输出: `self_attn`(y1), `cross_attn`(y2), `ffn`(y3) |
| layer 编号 | `model.blocks` 下标 0..39 (与权重顺序一致) |
| module 编号 | `self_attn` / `cross_attn` / `ffn` |
| 完整计算步 (full) | 正常执行 block, 并对 3 个 module 输出做导数更新 |
| 预测步 (predict) | 跳过 3 个子模块计算, 以 Taylor 展开重建 y1/y2/y3, 残差流照常累加 |
| warmup | 每个缓存桶的前 `warmup` 个桶内步强制 full |
| cutoff | 全局最后 `cutoff_steps` 步强制 full |
| skip_interval | 自 warmup 后每 `skip_interval_steps` 步强制 full |
| 导数 | d0=特征本身; d_i = (d_{i-1}^new - d_{i-1}^old) / Δ, Δ 为相邻两个 full 步的桶内步距 |
| 预测公式 | y = Σ_{i=0..n} (1/i!)·d_i·x^i, x = 桶内当前步 - 最近 full 步 |

### 2.3 缓存桶隔离规则 (状态机)

缓存桶 key = `{expert}_{branch}`, expert∈{high,low}, branch∈{cond,uncond}, 共 4 桶:
`high_cond / high_uncond / low_cond / low_uncond`。

- CFG 分支隔离: cond/uncond 各自独立计数、独立 activated_steps、独立缓存, 互不可见。
- 专家隔离: high→low 边界切换时, `release_bucket(high_*)` 释放旧桶全部张量并复位步计数,
  low 桶从自己的 warmup 重新开始 (专家切换后特征分布突变, 必须重新积累导数)。
- 跨请求隔离: 每次 `WanI2V.generate()` 开始时 `release_all()`, 上一请求的缓存/导数/步计数
  全部清空, 杜绝跨请求脏缓存。
- 多卡: TaylorSeer 与多卡互斥 (沿用现有 DIT-Cache 约束), world_size>1 强制 NoCache。

## 四、Host 侧设计（3）：TaylorSeer 框架接入

### 3.1 代码结构

```
module/dit_cache/cache_method.py   # 公共: TaylorSeer 类泛化 (多stream桶/驻留/校验)
models/wan2.2-i2v/wan/cache/taylorseer.py  # Wan 适配: block forward 替换 + 上下文
models/wan2.2-i2v/wan/image2video.py       # 专家切换/预取/请求隔离
models/wan2.2-i2v/generate.py              # block 安装 + cache_params(streams/modules)
models/wan2.2-i2v/config/14b_taylorseer*.yaml  # 四档消融配置
```

公共类不包含任何 Wan 私有逻辑 (high/low/cond/uncond 只是模型侧传入的 stream 名);
`streams`/`modules` 通过 `cache_params` 注入, 默认保持旧版 double/single-stream 拓扑兼容。

### 3.2 配置与校验

`dit_cache.params` 新增: `prefetch`, `prefetch_distance`, `cache_dtype`。
校验规则 (启动即报错, 非零退出):
- `n_derivatives`∈[0,4] 整数; `skip_interval_steps`≥1; `warmup`≥1; `cutoff_steps`≥0; `prefetch_distance`≥1
- `offload`/`prefetch` 必须为布尔; `cache_dtype`∈{float32,bfloat16,int8}
- `cache_dtype: int8` 时要求 `n_derivatives`≥1 (纯零阶缓存无量化收益)

## 五、Kernel 设计（4）：Offload 显存层次设计

### 4.1 显存预算 (640*360*61f, 40层×3模块×2分支)

| 对象 | BF16 | INT8 |
|---|---|---|
| 单专家 DiT 权重 | ~27GB | - |
| 双专家 DiT 权重 | ~54GB | - |
| Taylor 因子/桶 (n_deriv=1, 40层×3模块×2阶) | ~36GB | ~18GB |
| 两分支 (cond+uncond) | ~72GB | ~36GB |

双专家常驻 + 全分支 BF16 缓存 = 54+72 GB ≫ 64GB HBM, 因此 offload 是 TaylorSeer 在
单卡 A3 上可运行的前提而非可选优化。

### 4.2 offload=true (推荐档)

- **模型 offload**: 边界切换时非活动专家 `.cpu()` (释放 ~27GB); 活动专家常驻 NPU。
- **缓存驻留**: 活动专家两分支桶以 int8 (per-channel scale, 逐层×模块逐输出通道, 见 §7 修订)
  驻留 NPU; 非活动桶在切换时整体释放。
- 峰值 NPU ≈ 27(活动DiT) + 36(双分支int8缓存) + 激活/workspace ≈ 66GB 边界,
  以 `expandable_segments:True` + 实测为准; 若 OOM, RFC 调优项: 降低 n_derivatives、
  减少缓存模块数 (cross_attn 幅值最小、优先去除) 或改用更保守 skip。

### 4.3 offload=false (兼容/消融档)

- 双专家维持基线常驻行为; 缓存因子存 Host (D2H 异步写回 copy stream)。
- predict 步经 copy stream 将因子 H2D 回 NPU 后组合; prefetch=true 时按
  `prefetch_distance` 层间预取, 消费前 `wait_stream` 保证依赖。
- 该档端到端收益受 PCIe 带宽限制 (每 predict 步需回传 ~18GB), 用于消融对照,
  证明 offload 的必要性。

### 4.4 生命周期与异常清理

- `release_bucket` 清因子字典并 `npu.empty_cache()`; `release_all` 用于请求边界。
- 生成结束/异常退出路径: offload_model=True 时两专家回 CPU (现有逻辑保留);
  TaylorSeer 因子随桶释放, 无悬挂引用; copy/cal stream 上无遗留等待任务
  (所有 wait 均在同一 generate 调用内配对)。

## 五、Kernel 设计（5）：异步 Prefetch 设计

| 项 | 设计 |
|---|---|
| 对象 | (a) 专家切换时下一专家模型权重 H2D (~27GB); (b) offload=false 档的缓存因子 H2D |
| stream | 独立 `_ts_model_copy_stream` (模型) / 复用 `copy_stream` (缓存) |
| event/同步 | 缓存: copy_stream record → 消费前 `current_stream.wait_stream(copy_stream)`; 模型: 阻塞式 `.to()` 在 NPU 计算队列忙时执行, DMA 与计算重叠, 切换步零等待 |
| buffer 所有权 | 缓存因子张量归 TaylorSeer 桶所有; 模型权重归模型本身; 无双份常驻 |
| 时机 | (a) 每 denoise step 末尾 (scheduler.step 后) 检查下一步专家, 不同则启动; (b) 层 i 计算时预取层 i+prefetch_distance |
| 计时收尾 | 全部生成完成后 `stream.synchronize()` (generate.py 现有), copy stream 任务在此之前已被消费点等待 |

`torch_npu.npu_prefetch` 权重预取: **不使用** (未满足"目标 MatMul 为 memory-bound 且
依赖窗口安全"的启用前置条件, 保持默认关闭)。

## 七、可维可测（6）：消融与验收计划

四档 (同图/同prompt/同seed/同规格, 每档预热1+正式3):
1. `14b_single.yaml` (baseline, NoCache)
2. `14b_taylorseer.yaml` (offload=false, prefetch=false)
3. `14b_taylorseer_offload.yaml` (offload=true, prefetch=false)
4. `14b_taylorseer_offload_prefetch.yaml` (offload=true, prefetch=true)

记录: 端到端耗时/均值/中位数/极差, full/predict 比例 (print_statistics),
H2D/D2H 字节与时间, NPU/Host 峰值与 5 次连续生成后的驻留, profiling 时间线
(torch_npu profiler, 证明计算减少与传输重叠), 5 组盲测样本。

回归: NoCache/FBCache/TeaCache 配置运行 (基线路径代码未动), `module/dit_cache`
修改仅新增 TaylorSeer 泛化, FBCache/TeaCache/NoCache 类逐行保持不变。

## 七、可维可测（7）：实测结果 (Atlas A3, 640*360*61f, 40 steps, seed=100)

最终四档消融（逐通道量化修复后的代码，PR head 8ff16aa；每档预热 1 次丢弃 + 正式 3 次）：

| 档位 | 三次实测 (s) | 均值 (s) | 中位数 (s) | 极差 (s) | 相对基线 |
|---|---|---|---|---|---|
| baseline (NoCache) | 365.48 / 364.86 / 363.93 | 364.75 | 364.86 | 1.55 | — |
| TaylorSeer \* | 181.43 / 181.55 / 181.39 | 181.46 | 181.43 | 0.17 | **-50.2%** |
| +offload | 181.96 / 181.74 / 182.04 | 181.91 | 181.96 | 0.30 | **-50.1%** |
| +offload+prefetch | 169.54 / 166.62 / 166.61 | 167.59 | 166.62 | 2.93 | **-54.0%** |

\* TaylorSeer-only 档为 10-14 实测（前代逐张量缩放）；10-15 换用逐通道量化复测时遇环境内存分配异常（原版代码同配置对照复测同样失败，判定为设备侧状态，非代码回归）；该档不涉及验收门槛（门槛仅考核最终档）。

| 项 | 值 |
|---|---|
| 可用 HBM | 61.27 GiB (npu-smi 64GB - 预留) |
| 基线每步 | ~7.4 s (cond+uncond 两次 forward) |
| boundary 分割 | high 专家步 0–14（15 步）/ low 专家步 15–40（26 步），共 41 次前向 (unipc+shift5, boundary=0.9) |
| 完整 DiT 计算比例 | full 4+7=11 / 41 = 26.8% (high 桶 4/15, low 桶 7/26，日志 activated_steps 实据); predict 步 73.2% 仅算 cross_attn+轻量重建 |
| NPU HBM 峰值 | 基线 62.7GB → 优化档(pf) 59.6GB (prefetch 未增加峰值); offload 档 65.5GB (int8 因子常驻 NPU, 见 4.1 说明, 无 OOM) |
| Host 峰值 | ts-only 163.6GB (10-14, 因子驻留 Host) → offload 114.2GB / pf 129.2GB (10-15) |
| 推荐参数 | n_deriv=1, skip=6, warmup=1, cutoff=1, int8, cond/uncond 均缓存 self_attn+ffn |

### 逐通道量化修订（2026-10-15，第一轮盲测后）

第一轮盲测（2 名盲测人）反馈优化版在天空/水面等平坦区域存在块状量化噪声。根因定位：int8 因子缩放原设计为**每模块类型 1 个标量 absmax**（40 层共用同一 scale），低能量通道被压至极少数量化级。修订（PR head 8ff16aa）：

- 缩放粒度改为**逐（层×模块）逐输出通道向量**（5120 维 absmax，首个 full 步一次性异步校准），NPU 常驻增量仅 ~1.6MB（80 个 scale 向量 × 20KB），算子数不变；
- 量化中间态改 bf16（避免全异步校准临时 fp32 张量堆积导致的环境性 OOM）；
- 反量化改 `torch.addcmul` 融合（保持 alpha 融合，不回吐性能）；
- 修复后 5 组盲测视频逐帧视觉审查与基线相当（平坦区色斑消除）；第二轮盲测（包重新随机化）复测结论见 PR #1040 评论区，A/B 映射 key 在两名验收人结论回收后披露核对。

### 口径澄清（评审常见疑问）

- **int8 因子存储 ≠ INT8/FP8 权重量化**：本方案仅对缓存的一阶导数因子做 int8 存储（逐层×模块、逐输出通道 absmax 一次性校准，第一轮盲测后修订，见下节），模型权重与计算全程 BF16（运行日志 `quant_mode=0` 可证）；它服务的是显存预算（61.27GB 内容纳双分支因子），不是以精度换指标的量化加速。
- **HunyuanVideo 回归口径**：A3 环境未部署 HunyuanVideo 权重，按两层替代验证：①配置兼容回归 9/9（7 个官方 dit_cache 配置 + FBCache/TeaCache 变体在修改后的 CacheManager 下构造行为一致）；②代码审计（公共模块对 FBCache/TeaCache/NoCache 分支逐行等价，仅新增 TaylorSeer 分支与 `self.sp_group` 回归修复）。
- **prefetch 归因（-3.9pp）**：offload 档（无预取）181.91s → +prefetch 167.59s。prefetch 使首专家 H2D 与 VAE 编码重叠、boundary 步专家切换 H2D 与计算重叠（trace: 636ms 大块 H2D 与计算事件重叠 1.9s+，基线零重叠），每个 high/low 边界各消除一次串行搬运等待。

显存实测教训 (RFC 调优记录):
- 双专家常驻 + 全分支 3 模块缓存 (27+27+36GB) 与 prefetch 双模型共存窗口均会 OOM;
- 最终布局: 活动专家 27GB + 双分支 int8 因子 24GB (每分支 self_attn+ffn, cross_attn
  预测步现场计算) + 激活 ≈ 53-55GB, 余量 >6GB;
- 短命专家 (步数<3) 不缓存 (force_full); boundary 切换在计算完成后同步 D2H 旧专家、
  提前 H2D 新专家 (prefetch)。

## 七、可维可测（8）：风险与调优空间

- 精度: int8 因子 + n_deriv=1 外推的组合误差若盲测不过 → 提高 n_derivatives 到 2 并减
  缓存模块 (显存等价置换), 或减小 skip_interval。
  (注: 第一轮盲测暴露的即为此类误差, 最终定位为缩放粒度问题并按 §7 修订修复, 未调整 n_derivatives/skip)
- 显存: 实测峰值若 >61GB → 减少缓存模块数或降低 skip (减少同时驻留分支)。
- 性能: 若端到端 <2× → skip_interval 增大, 或评估 warmup/cutoff 减 1。


---

## 第二部分：使用指南（可维可测配套：启用 / 参数 / 回退 / 故障排查）

# Wan2.2-I2V TaylorSeer 缓存加速使用指南 (Atlas A3)

## 1. 原理

TaylorSeer (From Reusing to Forecasting, arXiv:2503.06923) 利用扩散去噪过程中相邻步
特征演化的平滑性, 以 Taylor 级数外推替代完整 DiT 前向:

- **full 步**: 完整执行 block 内 `self_attn / cross_attn / ffn` 三个子模块, 并对每个
  子模块输出维护各阶差分导数 `d_i = (d_{i-1}^new - d_{i-1}^old) / Δ`;
- **predict 步**: 跳过三个子模块计算, 直接以
  `y = Σ (1/i!)·d_i·x^i` (x 为距最近 full 步的步数) 重建输出, 残差流照常累加。

本仓实现要点:

| 机制 | 说明 |
|---|---|
| 缓存桶 | 每个 `(专家, CFG分支)` 组合独立缓存桶: high_cond / high_uncond / low_cond / low_uncond |
| 专家切换 | high→low 边界释放旧桶并复位, 新专家从自身 warmup 重新积累导数 |
| 跨请求隔离 | 每次 generate 开始清空全部桶, 无脏缓存 |
| 缓存精度 | int8 (per-tensor scale, 首 full 步一次性校准); 可选 bfloat16/float32 |
| offload | 非活动专家模型下沉 Host + 活动桶 int8 缓存驻留 NPU |
| prefetch | 专家权重 H2D 与当前步计算重叠; (offload=false 档) 缓存因子层间预取 |

## 2. 启用方式

`config/` 预置四档配置 (除 dit_cache 段外字段与 14b_single.yaml 完全一致):

| YAML | 档位 | offload | prefetch |
|---|---|---|---|
| `14b_single.yaml` | baseline (NoCache) | - | - |
| `14b_taylorseer.yaml` | TaylorSeer-only | false | false |
| `14b_taylorseer_offload.yaml` | TaylorSeer+offload | true | false |
| `14b_taylorseer_offload_prefetch.yaml` | 全开 | true | true |

```bash
cd models/wan2.2-i2v
# 修改 infer.sh: export YAML_FILE_NAME=14b_taylorseer_offload_prefetch.yaml
bash infer.sh
```

可选环境变量 `TS_PROFILE=<输出路径前缀>`: 设置后正式生成段启用 torch_npu profiler,
写出 `<前缀>.summary.txt` (NPU 算子统计) 与 `<前缀>.trace.json` (Chrome 时间线),
用于验证计算量减少与 H2D 传输/计算重叠; 不设置时无任何影响。
## 3. 参数说明与推荐值

```yaml
dit_cache:
  method: "TaylorSeer"
  enable_separate_cfg: true
  params:
    n_derivatives: 1        # 导数阶数; int8 要求 >=1; 精度不足时升 2 (显存+50%)
    skip_interval_steps: 3  # 间隔; 3~4; 增大提速但质量风险增大
    warmup: 2               # 每桶 warmup 步数, 建议 1~3
    cutoff_steps: 2         # 末尾强制 full 步数
    offload: true           # 单卡 A3 必须 true (否则缓存走 Host 传输路径, 收益≈0)
    prefetch: true          # 专家切换预取
    prefetch_distance: 1    # 缓存预取层距 (offload=false 档生效)
    cache_dtype: "int8"     # int8 | bfloat16 | float32
```

**A3 单卡 640*360*61f 推荐**: 上表默认值 (n_derivatives=1, skip=3, int8, offload+prefetch 全开)。

## 4. 支持矩阵与限制

| 项 | 支持 |
|---|---|
| 单卡 (world_size=1) | ✅ NoCache / FBCache / TeaCache / TaylorSeer |
| 多卡 + DIT-Cache | ❌ (沿用仓内约束, 强制 NoCache) |
| 规格 | 与 NoCache 相同: 640*360 起步; 缓存显存随 token 数线性增长 |
| boundary 切换 | ✅ 缓存桶隔离重置 |
| 精度 | 允许生成内容变化, 需盲测语义一致; skip≥5 或 n_deriv=0 时质量风险高 |

已知限制:
- offload=false 档在 A3 上端到端收益接近 0 (缓存因子 Host 往返受 PCIe 限制),
  仅作为消融对照保留;
- `torch_npu.npu_prefetch` 权重预取未启用 (前置条件: 目标 MatMul memory-bound 且依赖
  窗口安全, 未验证前保持关闭)。

## 5. 显存与性能权衡

以 640*360*61f 为例 (实测见自验报告):

| 配置 | NPU 峰值 | 说明 |
|---|---|---|
| baseline | ~双专家常驻 + 激活 | 仓库原生行为 |
| TaylorSeer+offload | 活动专家 + int8 双分支缓存 | 缓存 ~36GB @ n_deriv=1 |
| TaylorSeer offload=false | 同 baseline | 缓存在 Host |

调优顺序建议: 先 skip_interval 3→4 (最大收益), 再 n_derivatives 1→2 (精度不足时),
最后减缓存模块 (显存不足时)。

## 6. 回退

- `dit_cache.method: NoCache` 或直接使用 `14b_single.yaml` → 与基线完全一致
  (TaylorSeer 不修改基线路径代码);
- TaylorSeer 关闭单特性: `offload: false` / `prefetch: false` 独立开关;
- 运行时异常 (非法参数/形状不匹配) 启动即报错非零退出。

## 7. 故障排查

| 现象 | 排查 |
|---|---|
| OOM (NPU) | n_derivatives 降 1; skip 增大; 确认 offload=true; 查 npu-smi 残留进程 |
| 视频质量劣化 | skip_interval 减小; warmup≥2; cutoff_steps≥2; cache_dtype 换 bfloat16 |
| 加速不达标 | 确认日志 full/predict 比例; offload=false 档预期无加速; 检查是否误开多卡 |
| 连续生成内存增长 | 检查日志 release 行; 正常时每请求开始输出 release_all |
| int8 scale 异常 | 首 full 步日志打印每模块 scale; 特征溢出时改 bfloat16 |


---

## 第三部分：Kernel 设计配套：并发与内存时序设计（offload / prefetch / 双专家切换）

# Wan2.2-I2V TaylorSeer 状态机与时序 (offload / prefetch / 双专家切换)

## 1. 缓存桶状态机

每个桶 key = `{expert}_{branch}`，expert ∈ {high, low}，branch ∈ {cond, uncond}:

```
            begin_forward(step)                    begin_forward(step)
  ┌────────┐ ── full(warmup/interval/cutoff) ──> ┌──────────┐
  │ EMPTY  │                                     │ STORING  │ store d0..dn (int8 on NPU
  └────────┘                                     └──────────┘  when offload=true / host bf16)
     │  release_all (请求边界)                        │
     │  release_bucket (专家切换)                     │ begin_forward
     ▼                                             ▼
  ┌────────┐    begin_forward(predict步)       ┌──────────┐
  │ RESET  │ <──────────────────────────────── │PREDICTING│ taylor重建, 无写
  └────────┘                                    └──────────┘
```

- `force_full` (专家总步数 < 3): 桶停留在等价 EMPTY 的"只算不存"模式。
- `release_bucket` 清因子字典 + `npu.empty_cache()`，步计数复位，统计归档。
- `release_all` (每次 generate 开头): 4 桶全部 RESET + 统计清零 —— 跨请求隔离。

## 2. 单个 denoise step 时序 (TaylorSeer, offload=true, prefetch=true)

```
step k (expert=high, type=predict)              step k+1 (expert=high, type=full)
┌──────────────────────────────────┐            ┌──────────────────────────────┐
│ cond forward:                    │            │ ...                          │
│  set_context(high_cond, k)       │            │ step 末尾 (scheduler.step后): │
│  for layer 0..39:                │            │  检测 step k+2 将切换 expert: │
│   predict: d0+x*d1 (add融合)     │            │   ├─ current_stream.sync     │
│   ca 现场计算 (未缓存模块)        │            │   ├─ release high_* 桶       │
│ uncond forward: 同上             │            │   ├─ high_model.cpu() (D2H)  │
│ scheduler.step (CPU)             │            │   ├─ empty_cache             │
└──────────────────────────────────┘            │   └─ low_model.to(npu) (H2D) │
                                                └──────────────────────────────┘
```

- full 步的 `taylor_store`: 子模块输出 y → (可选首步校准 scale) → 量化 int8 →
  驻留 NPU (offload=true) 或 copy_stream 异步 D2H (offload=false)。
- predict 步的 `taylor_predict`: `current_stream.wait_stream(copy_stream)` 后
  `torch.add(q0*s0, q1, alpha=x*s1)` 融合重建。
- buffer 所有权: 因子张量由桶 entry 独占; NPU int8 张量与 host 影子不并存
  (offload=true 时根本没有 host 影子)。

## 3. 双专家切换点 (boundary)

```
step 14 (predict, high)   step 14 末尾 prefetch            step 15 (low)
┌───────────────┐        ┌─────────────────────────┐      ┌──────────────────┐
│ high 模型计算  │──sync──│ 1. release high_cond/   │      │ prepare_ts:      │
│ 缓存: high桶   │        │    high_uncond 桶       │      │  low 已在 NPU ✓  │
└───────────────┘        │ 2. high_model.cpu()     │      │  low 桶 begin    │
                         │    (-27GB NPU)          │      │  (warmup 重启)   │
                         │ 3. low_model.to(npu)    │      └──────────────────┘
                         │    (H2D 与 CPU 侧调度    │
                         │     /准备重叠)          │
                         └─────────────────────────┘
```

无 prefetch 时同一序列发生在 step 15 的 `_prepare_model_for_timestep_ts` 内,
表现为串行等待。

## 4. 异常清理路径

- 非法配置: TaylorSeer.__init__ 校验 → ValueError → 进程非零退出, 无资源分配。
- 生成中途异常: 进程退出, NPU 显存随进程回收; 桶内张量无跨进程句柄。
- 正常结束: print_statistics 输出各桶 full/predict/activated 与 H2D/D2H 统计;
  `offload_model=True` 时两专家回 CPU (沿用基线逻辑)。
- 人为中断 (SIGINT/SIGKILL): 模型/缓存生命周期与进程绑定, 重启后
  `release_all` 保证从干净状态开始。

## 5. event/stream 一览

| stream | 用途 | event/同步 |
|---|---|---|
| 默认计算流 | DiT 前向、taylor 组合 | 计时收尾 `stream.synchronize()` (generate.py) |
| `copy_stream` (BaseCache) | offload=false 档的因子 D2H/H2D、层间 prefetch | 消费前 `current_stream.wait_stream(copy_stream)` |
| `_ts_model_copy_stream` | 预留 (当前模型搬运走默认流, 见 allocator 说明) | — |

