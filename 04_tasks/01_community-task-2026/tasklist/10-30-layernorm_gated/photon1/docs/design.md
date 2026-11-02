# 需求背景（required）

## 需求来源

本设计对应社区任务《【社区任务】FLA 门控归一化融合算子(layernorm_gated)Ascend 适配》及任务书《任务书3-layernorm_gated-Ascend适配.md》。参考实现为 SGLang v0.5.22 中的
[`layernorm_gated.py`](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/kernels/ops/attention/fla/layernorm_gated.py)，
上游来源为
[`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)。

本任务在昇腾 NPU 上基于 Triton-Ascend 实现等价的 `_layer_norm_fwd_1pass_kernel` 前向，并在验收通过后贡献到 `mojo_opset`。

## 背景介绍

### 算子用途

`layernorm_gated` 把 LayerNorm/RMSNorm、门控激活、可选仿射变换和 FP8 e4m3 逐头量化融合到一个前向 kernel。该算子用于 GDN 线性注意力、Mixer2RMSNormGated 和 QK-Norm 等路径，典型输入具有以下形态：

- 2D 连续张量 `[M, N]`；
- 3D token-major 跨步门控张量 `[T, H, D]`，其中 `T*H=M`、`D=N`；
- GroupNorm 分组归一化；
- Ascend 950 上的 `quant_heads>0` 融合量化。

### 标杆现状分析

SGLang CUDA 实现使用 Triton kernel，将门控和归一化融合，并在量化模式下把舍入、amax、scale 和 FP8 转换放进 epilogue。仅使用 torch eager 组合会拆成多个 kernel，增加中间张量读写和 launch 开销。

本实现以 FP32 独立参考公式为数值标杆；性能对照使用任务书定义的 torch eager 多算子组合。Ascend A2、A3 运行非量化路径，950 运行完整路径和 FP8 量化路径。

### 功能分析

| 能力 | 特化参数 | 本设计支持范围 |
| --- | --- | --- |
| 归一化类型 | `IS_RMS_NORM` | LayerNorm 与 RMSNorm |
| 门控 | `HAS_Z`、`Z_IS_3D`、`Z_HEADS` | 无门控、2D 门控、3D strided 门控 |
| 门控顺序 | `NORM_BEFORE_GATE` | 先门控后归一化、先归一化后门控 |
| 激活 | `ACTIVATION` | `swish`/`silu`、`sigmoid` |
| 仿射 | `HAS_BIAS` | 权重必选，偏置可选 |
| 分组 | `ngroups` | GroupNorm，要求 `N % group_size == 0` |
| 量化 | `QUANT`、`quant_heads` | FP8 e4m3 与 FP32 scale，仅 950 验收 |
| 精度 | FP32 统计量 | FP16/BF16 输入，`mean`/`rstd` 为 FP32 |

# 需求分析（required）

## 外部组件依赖

- Python 3.10；
- PyTorch 2.10.0；
- torch_npu 2.10.0；
- Triton-Ascend 3.6.0；
- CANN 9.2.0；
- SGLang v0.5.22 作为接口和语义参考。

## 内部适配模块

| 模块 | 作用 |
| --- | --- |
| `mojo_opset/functions/layernorm_gated.py` | 保持 `_layer_norm_fwd` 参数和返回结构，执行 provider 分派 |
| `mojo_opset/kernels/_layernorm_gated_common.py` | 共享 Triton-Ascend kernel、tile 选择和 launch 参数 |
| `mojo_opset/kernels/npu_a2_triton/` | A2 provider；A3 通过 arch22 兼容映射复用 |
| `mojo_opset/kernels/npu_a5_triton/` | 950 provider，启用三轴 3D 门控 tile |
| `mojo_opset/modules/layernorm_gated.py` | 带参数的模块封装 |
| `tests/functions/test_layernorm_gated.py` | 功能组合与位精确测试 |
| `tests/performance/functions/test_layernorm_gated.py` | Shape 矩阵、profiler 与 MBU 测试 |

## 需求描述

在 Triton-Ascend 中实现 `layernorm_gated` 前向，保持公开 `_layer_norm_fwd` API 不变，覆盖任务书规定的归一化、门控、仿射、GroupNorm 和 950 FP8 量化组合；A2、A3、950 均可运行，精度、性能和交付材料满足任务书对应门槛。

## 需求拆解

1. 实现 LayerNorm/RMSNorm 的 FP32 一行归约，并输出正确的 `out`、`mean`、`rstd`。
2. 实现 2D/3D 门控、两种门控顺序和两种激活，3D strided 与连续化输入逐位一致。
3. 实现 GroupNorm 和按 group 偏移广播的权重/偏置。
4. 在 950 实现融合 FP8 e4m3 epilogue，与非融合量化逐位一致。
5. 依据 AI Vector Core 数选取 row worker，使用多行 tile、持久循环和 3D 聚合搬运降低 launch 与访存开销。
6. 保持 batch-invariant 模式的固定行块，保证不同 batch 下共享行的数值稳定。
7. 按任务书 Shape 矩阵提供精度、性能和内存证据。

# 详细设计（required）

## 算子分析

### 数学公式

输入二维视图记为 \(X\in\mathbb{R}^{M\times N}\)，第 \(g\) 组的归一化宽度为 \(D\)，组数 \(G=N/D\)。3D 门控张量记为 \(Z\in\mathbb{R}^{T\times H\times D}\)，满足 \(T\cdot H=M\)。

门控激活函数为：

\[
f(z)=
\begin{cases}
z\,\mathrm{sigmoid}(z), & \text{swish/silu},\\
\mathrm{sigmoid}(z), & \text{sigmoid}.
\end{cases}
\]

当 `NORM_BEFORE_GATE=False` 时：

\[
X'_{m,g,d}=X_{m,g,d}\,f(Z_{m,g,d}).
\]

LayerNorm 的统计量为：

\[
\mu_{m,g}=\frac{1}{D}\sum_{d=0}^{D-1}X'_{m,g,d},
\qquad
\sigma_{m,g}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}(X'_{m,g,d}-\mu_{m,g})^2+\epsilon}.
\]

RMSNorm 不计算均值：

\[
\sigma_{m,g}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}(X'_{m,g,d})^2+\epsilon}.
\]

仿射输出为：

\[
Y_{m,g,d}=\widehat X_{m,g,d}W_{g,d}+B_{g,d},
\qquad
\widehat X_{m,g,d}=\frac{X'_{m,g,d}-\mu_{m,g}}{\sigma_{m,g}}.
\]

RMSNorm 使用 \(\widehat X=X'/\sigma\)。当 `NORM_BEFORE_GATE=True` 时，在仿射后计算：

\[
Y'_{m,g,d}=Y_{m,g,d}\,f(Z_{m,g,d}).
\]

FP8 量化先按输出激活 dtype 舍入，再计算 `amax`：

\[
a=\max_{d}|Y_d|,\qquad
s=\frac{\max(a,10^{-10})}{448},\qquad
q=\mathrm{clamp}(Y/s,-448,448).
\]

`q` 写为 float8 e4m3，`s` 写为 FP32。

### 支持数据类型

| 张量 | 数据类型 |
| --- | --- |
| `x`、`weight`、`bias`、`z`、非量化 `out` | FP16 或 BF16，且 dtype 一致 |
| `mean`、`rstd` | FP32 |
| 量化 `q` | FP8 e4m3 |
| 量化 `s` | FP32 |

归一化统计和中间计算使用 FP32。A2/A3 的正式范围为 FP16/BF16 非量化路径；950 同时覆盖 FP8 量化路径。

### 支持形状

- `x`、`out`：二维 `[M, N]`，最后一维连续；
- `weight`：`[N]`；
- `bias`：可选 `[N]`；
- 2D `z`：`[M, N]`，最后一维连续；
- 3D `z`：`[T, H, D]`，满足 `T*H=M`、`D=N`，允许 token-major 跨步；
- `mean`、`rstd`：`[G*M]`；
- 量化 `q`：`[M/quant_heads, quant_heads*N]`；
- 量化 `s`：`[M/quant_heads, 1]`。

GroupNorm 要求 `N % group_size == 0`，且 `group_size <= 65536 // element_size`。

### 参数与接口

Kernel 的关键入参如下：

| 参数 | 方向 | 数据类型 | 含义 |
| --- | --- | --- | --- |
| `X` | 输入 | FP16/BF16 | 二维输入视图 |
| `Y` | 输出 | FP16/BF16 | 归一化输出 |
| `W`、`B` | 输入 | FP16/BF16 | 仿射权重和可选偏置 |
| `Z` | 输入 | FP16/BF16 | 2D 或 3D 门控 |
| `Mean`、`Rstd` | 输出 | FP32 | 逐组逐行统计量 |
| `Q`、`S` | 输出 | FP8 e4m3、FP32 | 融合量化结果与 scale |
| `stride_x_row`、`stride_y_row` | 属性 | INT64 | 输入输出行步长 |
| `stride_z_row` | 属性 | INT64 | 2D 门控行步长 |
| `stride_z_token`、`stride_z_head` | 属性 | INT64 | 3D 门控 token/head 步长 |
| `M` | 属性 | INT | 二维行数 |
| `N` | 属性 | INT constexpr | 每组归一化宽度 |
| `eps` | 属性 | FLOAT | 方差修正项 |

公开 wrapper `_layer_norm_fwd(x, weight, bias, eps, z, out, group_size, norm_before_gate, is_rms_norm, activation, quant_heads)` 保持 SGLang 接口。`quant_heads>0` 返回 `(q, s)`，否则返回 `out`；函数同时返回 `mean`、`rstd`。

## 算子实现

### 实现方案

`mojo_opset/functions/layernorm_gated.py` 保持公开接口和返回结构，根据 device provider 分派到 `npu_a2_triton` 或 `npu_a5_triton`。A3 设备使用 arch22 同源代码族，通过 `MOJO_ASCEND_ARCH=a2` 映射到 A2 provider；950 使用 A5 provider。

#### 3.2.1 Host 侧设计

##### 1. 输入检查与布局

- 从 `x` 读取 \(M,N\)，二维预分配输出时直接沿用；
- `group_size=None` 时取 \(D=N\)，否则计算 \(G=N/D\)；
- 校验 `x/z/weight/bias/out` 最后一维连续；
- 3D `z` 保留原视图，由 `stride_z_token`、`stride_z_head` 直接寻址；
- 将归一化类型、门控类型、3D、偏置和激活映射为 Triton constexpr；
- 量化模式校验 `G=1`、`M % quant_heads == 0`，分配 `q`、`s`。

##### 2. 分核策略

每个归一化组使用独立的 `program_id(1)`。行方向使用持久 worker：

\[
\text{rowWorkers}=\min\left(\text{AI Vector Core 数},
\left\lceil\frac{M}{\text{ROWS\_PER\_BLOCK}}\right\rceil\right).
\]

每个 worker 以固定步长消费后续 row-task，避免为每个 row-task 重复 launch。量化 block 较短，grid 的上限提高到 `2 * Vector Core`，使 epilogue 与 store 流水重叠。核数在运行时通过设备属性读取，不硬编码。

##### 3. 数据分块和内存优化策略

- tile 形状为 `[ROWS_PER_BLOCK, BLOCK_N]`；
- `BLOCK_N=min(65536//element_size, next_power_of_2(N))`；
- `group_size<=256` 且不是二次幂时，`BLOCK_N=group_size`，例如 192 列尾块直接生成有效 192 列 tile；
- 行尾和列尾用同一 mask 处理，避免越界写；
- `M` 可被行块整除时设置 `EVEN_ROWS=True`，编译期消除行 mask；
- 普通 BF16 路径按 \(D\) 选择行块：

\[
\begin{array}{c|ccccc}
D & \le 128 & \le 256 & \le 512 & \le 1024 & >1024 \\
\hline
ROWS\_PER\_BLOCK & 64/32 & 32 & 8 & 4 & 2
\end{array}
\]

\(D\le128\) 且 \(M\cdot N\ge32\text{M}\) 元素时使用 32 行。大流式张量原先使用 64 行 tile 时，A2 profiler 的 AIV vector ratio 为 68.4%、MTE2 ratio 为 72.6%；改用 32 行后分别达到 91.9%、97.5%，单次耗时约由 614 us 降至 498 us。

##### 4. 批不变策略

环境变量 `LN_GATED_BATCH_INVARIANT=1` 时，`ROWS_PER_BLOCK` 固定为 `MAX_ROWS_PER_BLOCK=4`。该模式不根据 \(M\)、\(N\) 或硬件核数改变行归约树，共享行在不同 batch 下使用相同计算路径。

##### 5. Tiling key 规划

使用 Triton constexpr 特化，不使用独立 tiling key：

- `BLOCK_N`、`ROWS_PER_BLOCK`；
- `HAS_BIAS`、`HAS_Z`、`Z_IS_3D`、`Z_HEADS`；
- `Z_3D_ALIGNED`、`NORM_BEFORE_GATE`；
- `IS_RMS_NORM`、`ACTIVATION`、`QUANT`、`EVEN_ROWS`。

#### 3.2.2 Kernel 侧设计

Kernel 每轮处理一个归一化组和 `ROWS_PER_BLOCK` 行：

1. 计算二维或三维门控地址，按需读取 \(Z\)。
2. 对 `norm_before_gate=False` 先执行门控乘法。
3. 将输入转为 FP32，按行求 LayerNorm 均值/中心化方差或 RMSNorm 平方均值。
4. 写出 `mean`、`rstd`，执行权重、偏置和门控。
5. 非量化路径直接写 `out`；量化路径先按输出 dtype 舍入，再求 head 维 `amax`、scale、clamp 并写 `q`、`s`。

3D 门控首先按二维行号分解为 token/head：

\[
t=\lfloor m/H\rfloor,\qquad h=m\bmod H,
\qquad
\text{offset}=t\cdot stride_{z,token}+h\cdot stride_{z,head}.
\]

950 的长序列 3D 路径在 `rows_per_block` 能被 head 数整除且 \(M\cdot N\ge32\text{M}\) 时，把 tile 显式组织为 `[tokens, heads, BLOCK_N]`，随后无拷贝 reshape 为二维 tile。该布局把同一 token 的各 head 聚合搬运；A2/A3 保留通用二维行偏移路径。

A5 上，相同数学内容的长序列 3D contiguous 路径使用二维偏移时约 526 us，启用三轴 tile 后约 315 us；二维连续门控对照约 284 us。

## 支持硬件

| 芯片平台 | 适配路径 | 正式能力 |
| --- | --- | --- |
| Ascend A2（Ascend910B3） | `npu_a2_triton` | FP16/BF16 全功能非量化路径 |
| Ascend A3（Ascend910_9382） | `MOJO_ASCEND_ARCH=a2`，arch22 同源代码 | FP16/BF16 全功能非量化路径 |
| Ascend 950（Ascend950PR） | `npu_a5_triton` | FP16/BF16 路径、3D 三轴 tile、FP8 e4m3 量化 |

A3 的共享 kernel 路径与 A2 一致，设备差异通过运行时设备属性和 arch22 provider 映射处理。

## 算子约束限制

- `x`、`weight`、`bias`、`z`、`out` 的最后一维必须连续；
- 3D `z` 仅允许 token-major 跨步视图，要求 `T*H=M`、`D=N`；
- GroupNorm 要求 `N % group_size == 0`；
- `group_size` 不超过 64 KiB 对应的元素数；
- FP8 e4m3 量化只在 950 的正式验收范围内；
- batch-invariant 模式下行块固定为 4；
- 不修改 SGLang 输入输出语义，不在测试中放宽容差或跳过正式用例。

# 特性交叉分析

| 交叉能力 | 处理方式 |
| --- | --- |
| RMSNorm × 无 bias | 允许；`mean=None`，直接平方均值 |
| LayerNorm × bias | 允许；归一化后加偏置 |
| 2D/3D z × 两种门控顺序 | 地址计算一致，只改变乘法位置 |
| 3D strided × 950 长序列 | 使用三轴 tile 聚合搬运，短序列走通用路径 |
| GroupNorm × 2D z | 每组独立行统计，权重按 group 偏移 |
| FP8 × z | 在单 token 的 `quant_heads` 个 head 上计算全局 amax |
| batch invariant × 分组 | 固定 4 行，保持共享行归约一致 |
| A2/A3 × quant | 平台范围外，不进入正式精度分母 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 正式标准 | 来源 |
| --- | --- | --- |
| Pytest 精度 | 正式用例全部通过，`0 failed、0 error` | 任务书“测试标准—精度测试” |
| LayerNorm/RMSNorm | 对 FP32 独立参考实现满足测试容差 | 任务书“标杆选型” |
| 3D strided z | 与 contiguous 结果 `torch.equal` 级一致 | 任务书约束 3 与位精确边界 1 |
| FP8 | 与非融合量化结果逐位一致 | 任务书约束 2 与位精确边界 2 |
| batch invariant | 共享行跨 batch 逐位一致 | 任务书约束 5 与位精确边界 3 |
| MBU | 单次字节数不小于 16 MB 的正式配置达到矩阵规定线，长序列与预填为 `>=60%`，尾块为 `>=50%` | 任务书 Shape 矩阵 |
| 小 Shape 相对线 | 32×8、320×8、GroupNorm 1024 三行的主指标和端到端耗时均不劣于 torch eager | 任务书“小 shape 相对线” |
| 采样口径 | 5 次 warmup，30 次有效调用；每轮现场采集 `fill_`、`copy_`，以 `copy_` 为 MBU 分母 | 任务书“术语与测量口径” |
| 交付件 | 设计文档、代码、精度报告、性能报告、内存报告和日志 | 任务书“验收标准”和交付模板 |

性能字节数固定使用任务书公式：

- 门控路径：\((3MN+2N)\cdot es\)；
- 无门控路径：\((2MN+2N)\cdot es\)；
- 融合量化路径：\(2MN\cdot es+MN+4M/quant\_heads+2N\cdot es\)。

## 当前自验结果

当前代码树在三个平台的实机结果为：

| 平台 | 精度 | 性能适用行 | 长序列 MBU | 尾块 MBU | `copy_` 标定 |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 | 30 passed，4 个平台范围外 FP8 skipped，0 failed/error | 10/10 | 63.30% | 71.27% | 1257.91 GB/s |
| A3 | 30 passed，4 个平台范围外 FP8 skipped，0 failed/error | 10/10 | 68.74% | 80.16% | 1265.95 GB/s |
| 950 | 34 passed，0 failed/error | 11/11 | 89.13% | 124.76% | 1437.93 GB/s |

950 融合 FP8 配置的 MBU 为 60.34%。三个小 Shape 配置的主指标和端到端耗时均优于 torch eager。最终自验值不替代社区验收，设计评审通过后再按流程提交验收材料。

## 兼容性分析

- 本算子为新增能力，不改变已有算子的输入输出契约；
- 公开 wrapper 保持 SGLang `_layer_norm_fwd` 的参数顺序和返回结构；
- A2/A3 共享 arch22 provider，A3 通过环境映射使用同一代码族；
- 950 通过 A5 provider 启用三轴门控 tile 和 FP8 epilogue；
- 非量化路径保持 FP16/BF16 契约，量化仅在 950 正式启用；
- 连续化 3D z 与原始 strided z 使用同一计算顺序，目标为零容差一致。

# 测试与验证方案

## 精度测试

执行：

```bash
pytest tests/functions/test_layernorm_gated.py
```

覆盖：

- LayerNorm/RMSNorm；
- 无门控、2D 门控、3D strided 门控；
- `swish/silu`、`sigmoid`；
- `norm_before_gate=True/False`；
- FP16/BF16；
- 有/无 bias；
- GroupNorm `N=256, group_size=64`；
- 行尾 `M=129`、列尾 `N=192`；
- 950 FP8 `quant_heads=4`；
- 3D strided 零容差、FP8 bit-exact、batch invariant。

正式结果要求 A2/A3 为 `30 passed, 4 skipped`，950 为 `34 passed`，三者均为 `0 failed、0 error`。

## 性能测试

执行：

```bash
pytest tests/performance/functions/test_layernorm_gated.py
```

测量流程：

1. 使用 Triton-Ascend `do_bench_npu_profiler`；
2. 5 次热身，30 次有效调用；
3. 保存 profiler CSV 和 `npu-smi` 信息；
4. 同进程执行连续 BF16 `fill_`/`copy_` 标定；
5. 按固定公式计算每行字节数和 MBU；
6. 小 Shape 行同时记录 profiler 主指标和端到端耗时。

## 内存验证

在同一进程记录 `npu-smi`、进程内存及 HBM 使用，覆盖完整精度和性能调用。任务书未规定内存数值门槛，因此不新增硬性内存通过线；验证目标是全部运行完成、无 OOM、无异常增长。

## 优化前后对比

交付材料按正式配置提供优化前、优化后和变化比例，重点记录：

- A2/A3 的长序列、尾块和小 Shape；
- 950 的完整 11 行矩阵；
- 3D 二维偏移与三轴 tile 的长序列对照；
- quant 单块与双排队 block 对照；
- row tile 64 与 32 的 profiler ratio 和耗时对照。
