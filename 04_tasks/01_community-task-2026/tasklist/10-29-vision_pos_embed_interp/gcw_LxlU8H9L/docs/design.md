# 需求背景（required）

## 需求来源

2026-10 社区任务《视觉位置编码双线性插值守卫算子(vision_pos_embed_interp)Ascend 适配》。

任务书原始 ZIP：`15808416dace40f2adc20b32d54d9862.zip`
ZIP SHA256：`f5f2b3dbe1a3606076dfa6b780c195dfa4638871627036b7e818831d833c62a7`
任务书正文 SHA256：`2e7a79f783a5dbc980b3e3a2fc8925ab06e7795ff43d26bf8e2dd9ceab6c0a07`

目标代码仓库：[mojo_opset](https://github.com/XPU-Forces/mojo_opset)，基于 triton-ascend 编程语言实现；验收平台为 Ascend A2 / A3 / 950。

## 背景介绍

上游 [vllm](https://github.com/vllm-project/vllm) 在 Qwen3-VL 系列视觉编码器中提供了 Triton 融合算子 `triton_pos_embed_interpolate`：将 `G×G`（默认 48×48）可学习位置表双线性插值到 `(h, w)` 特征网格，4 邻域 gather 加权，spatial-merge（2×2）重排编进地址算术，时间维 `t` 以重计算代替物化，单次 launch 完成；native 路径 `pos_embed_interpolate_native` 需约 20 次 dispatch、5 次全量访存。

当前 NPU 上的状况：

- vllm-ascend 在 `vllm_ascend/patch/worker/patch_qwen3vl.py:72-89` 以 monkey-patch 强制 Qwen3-VL 的 `fast_pos_embed_interpolate` 走 native 路径，即 Triton kernel 在 NPU 上不可用；
- 上游测试 `tests/kernels/core/test_vit_bilinear_pos_embed.py` 为 CUDA-only，NPU 无实机用例。

本任务目标：基于 triton-ascend 使 `_bilinear_pos_embed_kernel` 在昇腾 NPU 上原生运行，摘除 vllm-ascend 的 native 强制 patch，并将多图循环 + `cat` 合并为单次 launch。

### 标杆算子现状分析

标杆为上游 `pos_embed_interpolate_native`（`vllm/model_executor/models/qwen3_vl.py:301-368`）：`torch.linspace` 采样 + 4 邻域 gather 加权 + `permute` 重排，与 HF 插值语义同源，FP32 可跑。该实现为本任务全部精度用例的对拍基准。

| 项目 | 内容 |
| --- | --- |
| 标杆来源 | vllm `pos_embed_interpolate_native`（`main`，`qwen3_vl.py:301-368`） |
| 语义 | `linspace(0, G-1, size)` 采样 → floor/ceil + clamp → 双线性 4 邻域加权 → spatial-merge permute |
| 数据类型 | FP32 / BF16 |
| 时间维 | `expand(t,-1,-1).reshape(-1, hidden)`，逐帧重复同一空间模式 |

### 算子功能分析

双线性插值权重计算：

```text
h_frac = row * h_scale,    h_scale = (G-1)/(h-1)   (h > 1; h == 1 时 h_scale = 0)
hf = floor(h_frac),  hc = min(hf + 1, G-1)
dh = h_frac - hf
w11 = dh*dw,  w10 = dh - w11,  w01 = dw - w11,  w00 = 1 - dh - w01
out[row, :] = w00*E[hf,wf] + w01*E[hf,wc] + w10*E[hc,wf] + w11*E[hc,wc]
```

输出顺序为 spatial-merge 顺序：`combined.reshape(h/m, m, w/m, m, hidden).permute(0,2,1,3,4)`，即同一 `(m×m)` 邻域在输出行上连续。

# 需求分析（required）

## 需求描述

基于 triton-ascend 3.6.0 实现 `_bilinear_pos_embed_kernel` 的昇腾亲和适配，保证算子功能与上游实现一致；输出为 spatial-merge 顺序的 `[t*h*w, hidden_dim]`；实现多图合并单 launch 版本；提供 vllm-ascend 接入改造代码。

## 需求拆解

1. **kernel 本体适配**：`_bilinear_pos_embed_kernel` 在 Ascend A2/A3/950 上编译通过、结果与标杆一致。
2. **输出语义**：块内 `(m_size×m_size)` 邻域聚在一起（`br/bc/lr/lc` 地址算术），与 native 的 `permute` 语义一致。
3. **权重精度**：双线性权重以输出 dtype 计算，FP32 与 BF16 的舍入路径与上游一致；`h_scale/w_scale` 由 wrapper 预计算传入，`h==1` 或 `w==1` 时 `scale=0` 的边界保持。
4. **多图合并**：`fast_pos_embed_interpolate` 对 `grid_thw` 列表逐图调用后 `torch.cat`；须实现合并单 launch 版本，输出与逐图版逐位一致。
5. **vllm-ascend 接入**：提供摘除 `patch_qwen3vl.py` native 强制 patch、改按平台能力分派的改造代码，附接入前后对比验证。
6. **泛化能力**：`G`、`m_size`、`hidden_dim` 不得针对 Qwen3-VL 单一配置硬编码。

# 详细设计（required）

## 算子分析

### 数学公式

见「算子功能分析」。核心为 4 邻域双线性加权 + spatial-merge 地址重排 + 时间维广播。

### 接口与参数

| 参数名 | 输入/输出/属性 | 数据类型 | 维度 |
| --- | --- | --- | --- |
| embed_weight | 输入 | FP32/BF16 | `[G*G, hidden_dim]`（Qwen3-VL: `[2304, 1152]`） |
| t | 属性 | int | - |
| h, w | 属性 | int（可整除 `m_size`） | - |
| num_grid_per_side | 属性 | int（Qwen3-VL=48） | - |
| m_size | 属性 | int（Qwen3-VL=2） | - |
| dtype | 属性 | FP32/BF16 | - |
| out | 输出 | 与 dtype 一致 | `[t*h*w, hidden_dim]` |

### 支持数据类型与形状

- 数据类型：FP32、BF16
- 形状：`h`、`w` 必须被 `m_size` 整除；`G` 任意正整数；`hidden_dim` 任意（含非 2 的幂）
- `t >= 1`；多图场景为 `grid_thw` 列表

## 算子实现

### Host 侧设计

1. **参数校验与预计算**：校验 `h % m_size == 0 and w % m_size == 0`；按上游同式预计算 `h_scale = (G-1)/(h-1)`（`h>1`，否则 0）、`w_scale` 同理；`hidden_dim` 取自 `embed_weight.shape[1]`。
2. **输出分配**：按 `[t*h*w, hidden_dim]`、目标 dtype 分配输出。
3. **grid 规划**：按输出行数量与平台 Vector Core 数规划 grid，使逻辑任务数不超过平台核心维度上限，避免依赖超大 grid。
4. **多图合并**：将 `grid_thw` 列表解析为设备侧标量张量（每图的 `h/w/scale/块数/输出行偏移`），一次 launch 覆盖全部图，替代逐图 launch + `cat`。
5. **分派**：函数层按 `implementation` 选择 `triton` / `torch` 实现，默认由 config 决定；A2 与 A5 分别登记各自的 triton provider。

### Kernel 侧设计

1. **索引解码**：由线性 program id 解出 `(row, col)`，spatial-merge 重排通过地址算术直接产出，无需单独 permute 步骤。块内 `m_size×m_size` 邻域在输出行上连续，可减少重复的整除/取模运算。
2. **双线性权重**：`h_frac = row * h_scale`，`hf = floor(h_frac)`；因 `h_frac` 非负，`floor` 可用 int32 截断等价实现。`hc = min(hf+1, G-1)` 做上界 clamp。4 个权重按上游同式顺序计算，不做代数化简，以保持浮点运算顺序一致。
3. **权重类型**：权重在 FP32 下算完，再 cast 到输出 dtype 后参与乘累加，与 native 的 `weights.to(dtype)` 路径一致。
4. **访存**：每次迭代读 4 个邻域行、写 1 个输出行；输出按 mask 保护尾块，不得越界写。
5. **时间维**：输出行号对 `t*h*w` 解码，时间维通过对 `h*w` 取模自然重复，实现"重计算代替物化"，保证 `t>1` 与 `t=1` 结果逐位一致。
6. **多图**：由第二维 program id 选择图像，每图的标量参数从设备张量读取，使合并 launch 与逐图调用具有相同的逐行指令序列。
7. **尾块**：`hidden_dim` 不是 2 的幂时，按块宽循环并对最后一段加 mask，保证不写出越界。

### 调优计划

1. **写入效率**：本算子为写主导，输出写入带宽是主要瓶颈，调优以提升有效写带宽为目标，并实测不同块宽下的耗时。
2. **块宽选择**：在 UB 容量约束下比较不同 `block_d` 的耗时，兼顾掩码 lane 浪费与循环轮数。
3. **编译选项**：仅在 Vector-only 前提下评估适用的编译选项（如流水搬运），不引入与 Cube 相关的配置。
4. **数据依据**：调优以固定 case 集的实测耗时为准，不依据 case 编号或输入数据内容分支。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列 | √ |
| Atlas A3 系列 | √ |
| Ascend 950PR/950DT 系列 | √ |

配套软件：triton-ascend 3.6.0、torch_npu 2.10.0.post7、CANN 9.2.0。

> 说明：A2/A3 共用同一编译后端（`linalg_to_bin_enable_npu_compile_A2_A3`），A5 使用独立的 `..._910_95` 后端且关闭 FFTS，故三平台需分别实测，不以单平台结果外推。

## 算子约束限制

- 公共 API 签名与上游 `triton_pos_embed_interpolate` 一致，不变更。
- `h`、`w` 必须被 `m_size` 整除，保持上游断言。
- 权重表达式的运算顺序与上游保持一致，不做代数化简，避免改变浮点舍入路径。
- 输出必须为 spatial-merge 顺序，且 `t>1` 与 `t=1` 的 `repeat` 逐位一致。
- `hidden_dim` 尾块不得越界写。
- `+ patch` 加法（`qwen3_vl.py:881-882`）不在本任务范围。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 精度套件全量用例 0 failed / 0 error；不修改断言与容差 | 任务书「精度测试套件」 |
| 精度（容差） | FP32：`rtol=1e-5, atol=5e-5`；BF16：`rtol=1e-2, atol=1e-2` | 上游 `test_vit_bilinear_pos_embed.py` |
| 精度（位精确） | `t>1` 与 `t=1.repeat(t,1)` 逐位相等（`atol=0/rtol=0`）；多图合并与逐图 `cat` 逐位相等 | 任务书边界用例 1 |
| 性能 | MBU：常规 ≥60%，尾块 ≥50%，小图 `(1,8,16)` 不劣于 native | 任务书「性能测试套件」 |
| 稳定性 | 无 507014/507034 超时及编译器崩溃，视频大图不超时 | 任务书「任务标准」第 3 条 |

**性能测量口径**：主指标为 `torch_npu.profiler` 采集的 NPU kernel `Duration(us)` 之和 ÷ 30（热身 5 + 有效 30）；MBU = 读写字节数 ÷ 主指标 ÷ 本机标定带宽，读写字节数按任务书固定算式自动计算，标定带宽取 `fill_` 纯写探针为分母（本算子为写主导）。

**用例覆盖**：

| 维度 | 取值 |
| --- | --- |
| dtype | FP32、BF16 |
| grid（单图） | `(1,4,4)`、`(1,16,16)`、`(1,32,32)`、`(1,48,48)`、`(1,8,16)`、`(1,14,20)`、`(1,32,48)`、`(1,60,80)` |
| grid（时间维） | `t=3, h=w=16` |
| grid（大网格） | `(16,100,168)`，输出 268800 行 |
| num_grid_per_side | 48（主）、14（泛化抽测） |
| 多图列表 | `[(1,8,16), (1,32,48), (3,16,16)]` |
| 尾块 | `hidden=1000`；grid `(1,16,16)`、`(1,32,48)` |

## 兼容性分析

- 新算子，不涉及已有算子行为变更。
- 公共 API 与上游 `triton_pos_embed_interpolate` 保持签名一致，可作为 drop-in 替换。
- vllm-ascend 接入为**可选增强**：摘除 native 强制 patch 后按平台能力分派，失败时回退 native，不破坏现有服务链路。
- A2/A3/A5 三平台分别登记 provider 与 config，互不影响；未登记平台保持原有行为。

## 实现与验证状态

| 阶段 | 状态 |
| --- | --- |
| 上游源码与标杆定位 | 已完成（行号与任务书索引一致） |
| 环境方案（版本配套、三平台差异、获取路径） | 已完成 |
| kernel 骨架与函数/模块封装 | 待实施 |
| 精度套件 | 待实施 |
| 性能测量 | 待实施 |
| 三平台验证 | 待实施（A2/A3/A5 设备已就绪） |

> 说明：任务书随附的 `testsuites/` 精度与性能套件**未包含在任务包 ZIP 内**（ZIP 内仅有任务书正文一个文件）。当前计划以上游 `tests/kernels/core/test_vit_bilinear_pos_embed.py` 为母本移植 NPU 版本，并在提交材料中明确标注「自行移植，非随附原件」；同时持续向任务方申请原始套件。若原件到位，将以原件为准重新执行全部用例。
