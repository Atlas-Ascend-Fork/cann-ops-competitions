# 【CANN社区任务】daVinci-MagiHuman RoPE 优化（A3）设计

| 项目 | 内容 |
|---|---|
| 任务 | 10 月社区任务 — daVinci-MagiHuman RoPE 优化（A3） |
| 开发者 | dy_dessert |
| 文档版本 | v1.0，2026-10-18 |
| 提交目录 | `04_tasks/01_community-task-2026/tasklist/10-daVinci-MagiHuman-RoPE-A3/dy_dessert/docs/design.md` |
| RFC | https://gitcode.com/cann/cann-recipes-infer/issues/474 |
| 设计文档 PR | https://gitcode.com/cann/cann-ops-competitions/pull/2286 |
| 目标仓 | https://gitcode.com/cann/cann-recipes-infer |
| 代码基线 | `93527538f1567a4cf42db58883231e4e00f8b213` |
| 模型固定上游 | `209209b7086eba2020c5439265221495a8357322`（GAIR-NLP/daVinci-MagiHuman） |
| 私仓与材料 | https://gitcode.com/dy_dessert/cann-recipes-infer |
| 待验收代码提交 | `39d5231cef52a7f3f8c987c19ed4dbfc5137e54e` |

## 1. 需求背景

daVinci-MagiHuman 把文本、视频与音频 token 放进同一条联合序列做去噪，RoPE 位于 Q/K 归一化之后、Attention 之前。任务书的前提是“仓内实现虽已使用 `torch_npu.npu_rotary_mul`，但仍保留运行时 `cat/unsqueeze/to` 等冗余转换与非融合回退路径”，要求基于 Atlas A3 profiling 收敛到可复用的融合路径，并保留不满足算子约束时的显式回退；改动限于 RoPE 数据准备与调用边界，不重写 Attention、线性层、并行策略或公共模块。

**实测结论先摆在这里**：目标 base 模式在原版代码里**本来就已命中融合算子**（每步 160 次 `RotaryPositionEmbedding`，80 次非融合 0 次）。因此本任务不存在“把小算子路径换成融合算子”的收益空间，实际可做的是**准备阶段收敛**：去掉每步重复的合并/拆分往返，把展开后的 cos/sin 表在请求内复用，并把融合调用条件写成显式判定、非法输入改为可读异常。本设计如实按此定位，不把“已存在的融合”计为新增成果。

## 2. 需求分析与范围

### 2.1 功能范围

- 修改限定于 `models/davinci-magihuman/`，实际只动 `inference/model/dit/dit_module.py` 与 `inference/pipeline/video_generate.py`，105 增 25 删。
- 复用既有 `torch_npu.npu_rotary_mul`，不新增独立 kernel、aclnn 接口或 tiling 工程。
- 支持路径默认直接使能，不新增用户配置开关；基线与候选分别使用改动前后的代码。
- 不支持的合法输入（CPU、float32、interleaved、非连续布局、非 4 维）保留显式数学回退；非法维度、cos/sin 形状不匹配、跨设备输入给出可读异常，不以宽泛捕获掩盖设备错误。
- 不修改 FlashAttention、线性层、并行策略、权重格式与公共 profiler。

### 2.2 基线与源码身份

- 代码基线 `9352753…`；模型固定上游 `209209b7…`（`git ls-remote` 核实为该仓 HEAD）。
- 基线树与候选树是两棵独立源码树（`/mnt/workspace/projects/cann-recipes-infer` 与 `…-optimized`），各自 `git status` 只保留预期改动：基线树目标文件为 `de226ae70df3233cced4167de88867a5097c28baa8989f9ef62fa9d2c45ed6de`、`53d3039757770af152d0867ace422da9784248ac33da4ac2f32f02f396782cc5`（未改动）。
- 候选源码：A3 上被测文件为 Windows 换行，`dit_module.py` = `3ba8bd3f…`、`video_generate.py` = `24e4d540…`；交付到私仓按仓内规范归档 LF，`8a159d07…`、`e2ed7352…`。两者只差换行，LF 版本已回 A3 复跑一次完整 base 生成，输出仍为 `4476754342…`。
- 权重按固定 ModelScope revision 下载并逐文件校验：GAIR `48986f36…`、T5Gemma `3df33de4…`、Stable Audio `1389ee4a…`、Wan2.2 VAE `d85a629f…`；base 与基础 TI2V 共 34 个文件，蒸馏与两档超分各补 14 个文件。
- 环境：Atlas A3 单卡（取 `ASCEND_RT_VISIBLE_DEVICES=0`，单 die HBM 64 GiB），Python 3.12.15、CANN 9.0.0、PyTorch 2.10.0+cpu、torch_npu 2.10.0（产品版本 v26.0.0）、驱动 25.5.5；固件版本工具报 `NA`，未以推测值填充。

### 2.3 输入与数值语义

- 主链记录：BF16、B=1、Q heads=40、K heads=8、`head_dim=128`、`rotary_dim=96`（partial rotary），`num_layers=40`、`hidden_size=5120`；序列长度 4025（文生视频）与 4102（图生视频）。
- cos/sin 由 `ElementWiseFourierEmbed` 生成 `[L, 48]` 两分量（`L = 3 × dim/8`），`cat([d, d])` 后为 `[L, 96]`，再 `unsqueeze(0).unsqueeze(2)` 扩为 `(1, L, 1, 96)`；`rotary_mode='half'`，配对 `(i, i+48)`，末尾 32 维原样拼回。
- 每步有 2 次独立 DiT 前向（`video_generate.py` 对 cond 与 uncond 各调用一次），基线因此**每步各建一次表**（一步 4 个 Sin/Cos 核）；这也是本任务唯一可压缩的确定性开销。
- partial rotary 的 160 次 Slice 与回拼是 `npu_rotary_mul` 在 `rotary_dim < head_dim` 下的结构代价；消除它需要改动 Attention 的输入布局，超出任务书“不重写 FlashAttention、不改公共模块”的范围，本设计明确不做。

## 3. 详细设计

### 3.1 调用链

```text
pipeline 请求作用域（predict_noise 计算缓存键）
  → DiTModel.forward：命中则复用缓存表，未命中则构建
  → ElementWiseFourierEmbed 直接返回 (sin, cos) 两分量（不再合并后拆分）
  → 按 x.dtype 转 BF16 → cat([d, d]) → unsqueeze → contiguous 得到 (1, L, 1, 96)
  → 各 Attention 层：约束判定 → npu_rotary_mul 或显式数学回退
  → 原 Attention 与后续生成链
```

### 3.2 表复用与缓存键

- 键由“调用方提供的位置无关键 + 坐标形状/dtype/device + 频带张量标识与版本”构成。调用方键取自 `eval_input` 的形状、音频/文本长度、设备、dtype、patch 尺寸、插值开关、音频偏移、文本偏移与坐标风格；**不含 prompt 内容**，因为 RoPE 表只由位置与尺寸决定。
- 最多保留 2 组，覆盖 CFG 两条分支；键变化自动重算，训练态禁用缓存。
- 版本保护：`inference_mode` 下构造的频带张量没有版本计数器，读取 `_version` 会抛 `RuntimeError`；这种情况不建缓存键、按原路径重算。该缺陷由 1080p 超分模式实测暴露（`RuntimeError: Inference tensors do not track version counter`），属本轮修复项。
- 长度取自 `eval_input` 的 list 字段；若出现非 int 元素（动态长度场景）则不提供键，退化为每次都重算，避免为建键触发 device→host 同步。

### 3.3 融合派发与显式回退

`apply_rotary_emb_npu` 改为先判定、再分派：

| 条件 | 处理 |
|---|---|
| NPU、4 维、非交错、BF16/FP16、cos/sin 与输入同设备同 dtype、形状可广播、`rotary_dim` 为偶数且不超过头维 | `torch_npu.npu_rotary_mul(rotary_mode='half')` |
| 其它合法输入（CPU、float32、interleaved、3 维、非连续 BSND） | 等价数学回退 `x*cos + rotate_half(x)*sin`（interleaved 用对应配对） |
| 非法维度、cos/sin 形状不匹配、跨设备、不可广播 | 抛出可读异常 |

`apply_rotary_emb_torch` 的非 tuple 分支同样收敛到上述入口，并把原来的 `assert` 换成带信息的 `ValueError`。

### 3.4 文件与影响

| 文件 | 职责 |
|---|---|
| `inference/model/dit/dit_module.py` | 表生成返回两分量、请求内表复用、融合派发与显式回退、异常信息 |
| `inference/pipeline/video_generate.py` | `predict_noise` 计算并传入位置无关缓存键 |

不新增文件，不引入新依赖，不改配置结构。

## 4. 可维可测分析

### 4.1 测试协议

- 固定 base：单卡 A3、BF16、448×256、5 秒、32 步、CFG=2、seed=42、`dit_cache.method: NoCache`，同权重、同 prompt、同环境；基线与候选分别用改动前后代码。
- RoPE 数值：`verify_rotary_npu.py` 覆盖 `head_dim` 64/128、`rotary_dim` 64/96/128、seqlen 1/17/256/4025/4102、heads 4/8/40，含 CPU、float32、interleaved、非连续 BSND 回退与非法维度报错；判据 BF16 `atol=rtol=0.02`，无 NaN/Inf，失败返回非零。
- 数据准备等价性：`verify_rope_components.py` 以固定基线修订 `9352753` 为参照，要求送入 `npu_rotary_mul` 的扩维张量**逐位相同**（6 组 dim × token 组合）。
- 性能：两档各预热 1 次、正式 3 次，计时覆盖 `run_offline`（含编码与音画合成），停止计时前 `torch.npu.synchronize()`；剖面用 `ENABLE_PROFILER=1` 采 `kernel_details.csv`，两档在同一批次内**交替**采样。
- 内容一致性：3 组固定 prompt + 基础 TI2V + 蒸馏/两档超分各两种模式，逐对比较 MP4 SHA256 并做 `ffmpeg` 全片解码。

### 4.2 实测记录

| 项目 | 结果 |
|---|---|
| 融合路径 | 每步 160 次 `RotaryPositionEmbedding`（40 层 × q/k × 2 次 CFG 前向），非融合 0 次；原版已融合 |
| 每步 kernel 数 | 3,965 → 3,885（**−80**，4 组采样完全一致）；Sin/Cos 4 → 0，表准备链路（Slice/ConcatD/Equal/RealDiv/NonZero 等）整体消失 |
| 选定 RoPE 路径（融合核 + 紧邻切片 + Sin/Cos） | 4 组完整交替采样中位 14,892.08 → 14,540.78 µs（**−351.30 µs，2.4%**）；配对逐轮 −327.66/−377.80/−275.94/−385.30 µs，**4/4 轮为负**；基线自身极差 57.72 µs |
| 端到端完整生成 | 中位 45.054261 → 44.952472 s（**−0.101789 s，0.23%**），基线极差 0.033641 s，改善超过波动范围 |
| 全步所有 kernel 合计 | 中位 −1,220.89 µs，但小于基线极差 1,631.98 µs 且 4 轮中 1 轮为正，**不判为已证实的全步下降** |
| 内容一致性 | base 三组 prompt 与基础 TI2V 逐字节相同（`4476754342…`、`1327f146…`、`ec6d5c36…`、`8f970084…`）；蒸馏、540p、1080p 共六种模式逐对相同且全片可解码 |
| 内存 | 加载后 47.855 GiB；五次生成后 48.148/48.152 GiB；峰值 49.826/49.830 GiB（差 0.004 GiB）；Host RSS 采样峰值 3.853/3.833 GiB |
| 异常中断 | 60 秒中断触发（退出码 124），随后芯片健康 `OK`、无残留进程；中断后 5 秒 HBM 仍显示 16,467 MB，稍后回到常驻水平，未进一步归因 |
| 代码检查 | 两份目标文件 `pre-commit`、`py_compile`、`git diff --check` 通过；全仓 `pre-commit --all-files` 在**基点干净检出**上即失败（自动改写 490 个基线文件、脚本执行位告警），未通过改其它模型文件规避 |

**同 kernel 耗时差的归因边界**：同一 `RotaryPositionEmbedding` 核在候选中位 7,488 µs、基线 7,892 µs（核数、Block Num、Core Type 全同），且代码路径未改动的 `AddRmsNorm`、`MatMulV2` 也出现 0.5~1% 的系统性漂移，更像缓冲区落位/环境效应；因此本设计**只把“每步少 80 个 kernel”和“Sin/Cos 4→0”作为可确定性归因的收益**，不把“融合核本身变快”计入。

### 4.3 交付索引

- 私仓 `dy_dessert/cann-recipes-infer`（private）`main`：① 基线逐字节导入 29 个模型文件（两份目标文件哈希与基线树一致）；② 代码提交 `39d5231cef52a7f3f8c987c19ed4dbfc5137e54e`；③ `task_submission/` 七件 + `MANIFEST.sha256.txt`。
- 自验材料七件：`1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`、`3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`、`4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`；匿名盲测包 `blind_samples_20261018.zip`（六视频，标签单独保存）。
- 权重、数据集、生成视频、完整 profiling 产物不进入 Git，凭获取位置与 SHA256 索引。

## 5. 待维护者与验收人确认的事项

1. 测试说明里 baseline 标为“RoPE fusion 关闭”，但优化前代码**已启用融合**；本设计按“baseline = 改动前代码”执行，未人为禁用融合构造更慢基线，请确认该口径。
2. 任务书 §3.3 “profiling 可观测到总耗时减少”若指**全步所有 kernel 合计**，本轮未证明（差值小于基线极差）；若指**RoPE 路径累计耗时**，已满足。请确认判定口径。
3. 两名指定验收人的人工双盲尚未进行，机器哈希相同只证明固定输入下输出一致。
4. 仓库 CI 由 PR 触发，当前未运行；全仓 `pre-commit --all-files` 在基点即失败，属既有问题。

## 6. 参考与 AI 辅助披露

- 任务书与 `test_cases/README.md`（任务包 SHA256 `81ae18903dee6f6166c2d36ac9e9c117b2cedf72462480b7ca23072857a5d260`）。
- 目标仓：https://gitcode.com/cann/cann-recipes-infer
- RFC：https://gitcode.com/cann/cann-recipes-infer/issues/474
- 设计文档 PR：https://gitcode.com/cann/cann-ops-competitions/pull/2286
- 待验收代码提交：https://gitcode.com/dy_dessert/cann-recipes-infer/commit/39d5231cef52a7f3f8c987c19ed4dbfc5137e54e?ref=main
- 上游模型：https://github.com/GAIR-NLP/daVinci-MagiHuman（固定提交 `209209b7…`）
- 私仓：https://gitcode.com/dy_dessert/cann-recipes-infer

本设计的整理与提交由 AI 编码助手（DeepSeek Harness）辅助完成；所有数值来自本机 A3 实测记录，未生成或虚构人工验收、环境接受或社区批准结论。
