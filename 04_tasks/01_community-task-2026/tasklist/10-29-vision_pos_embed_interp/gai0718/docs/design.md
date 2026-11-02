# 视觉位置编码双线性插值算子设计文档

## 需求背景（required）

### 需求来源

本设计对应《【社区任务】视觉位置编码双线性插值守卫算子(vision_pos_embed_interp)Ascend 适配》。

### 背景介绍

原链路位于 vLLM Qwen3-VL：

1. `Qwen3_VisionTransformer.fast_pos_embed_interpolate` 对 `grid_thw` 逐图调用 `triton_pos_embed_interpolate`。
2. 每张图产生一个长度为 \(t \times h \times w\) 的位置编码块。
3. Python 侧使用 `torch.cat` 拼接各图结果。
4. vLLM-Ascend 当前通过 monkey-patch 将入口固定为 native 路径。

本任务将双线性采样、四邻域加权、spatial-merge 重排和时间维广播合并在 Triton-Ascend 中完成，并把多图拼接改为单次设备 kernel 启动。Qwen3-VL forward 中的 `+ patch` 不属于本任务。

## 需求分析（required）

### 需求描述

在 Ascend NPU 上实现与上游 `triton_pos_embed_interpolate` 等价的单图与多图位置编码插值，并提供 mojo_opset 接口和 vLLM-Ascend 能力分派。

### 需求拆解

1. 保持双线性采样、四邻域加权、spatial-merge 重排和时间维广播的语义。
2. 支持 FP32、BF16、单图、多图及任务书列出的边界形状。
3. 将多图逐图 launch 与 `torch.cat` 合并为单次插值 kernel。
4. 提供 `npu.a2`、`npu.a5` provider、函数、模块和 vLLM-Ascend 接入代码。

### 正式验收口径

| 验收项 | 原文要求 | 证据 |
| --- | --- | --- |
| 精度 | 随附精度套件全量通过，0 failed、0 error | `tests/functions/test_vision_pos_embed_interp.py` 的完整 pytest 输出 |
| 平台 | Ascend A2、A3、950 运行通过 | 三档硬件 pytest 记录 |
| 时间维 | \(t>1\) 与 \(t=1\) 结果 repeat 逐位一致 | `temporal_repeat`，`atol=0, rtol=0` |
| 多图 | 单 launch，输出与逐图 cat 等价 | `multi_grid_cat_equivalence`，按官方混合容差检查 |
| 边界 | 大视频、\(G=14\)、退化单轴、`hidden=1000` 尾块 | 对应的固定参数化用例 |
| 稳定性 | 无 `507014/507034`、无编译器崩溃、无超时 | pytest、profiler 和运行日志 |
| 性能 | 5 次热身、30 次有效采样；按 shape 达到 MBU 线 | `bench_vit_pos_embed.py`、profiler CSV、`npu-smi info` |
| 交付 | 设计文档、代码、测试报告、vLLM-Ascend 接入改造、PR 说明 | 交付目录和 PR |

官方验收套件以 [XPU-Forces/mojo_opset#523](https://github.com/XPU-Forces/mojo_opset/pull/523) 为准，执行：

```bash
pytest tests/functions/test_vision_pos_embed_interp.py --mojo-implementation triton
```

官方套件参数化后共 27 个用例，不包含性能用例。交付代码直接使用该套件，不再维护自行移植的平行矩阵。

| 分组 | 数量 | 规则 |
| --- | ---: | --- |
| 8 个基础 grid × 2 dtype | 16 | 4 个方形、4 个非方形 |
| 时间维 × 2 dtype | 2 | `t=1` 与 `t=3` 位精确 |
| 大视频 × 2 dtype | 2 | `(16,100,168)` |
| 2 个尾块 grid × 2 dtype | 4 | `hidden=1000` |
| \(G=14\) | 1 | 覆盖两个非 2 的幂 grid |
| 多图与退化轴 | 1 | 三图列表、`h=2` 与 `w=2` 退化轴 |
| 参数校验 | 1 | 不可整除形状拒绝 |

精度容差以 [XPU-Forces/mojo_opset#526](https://github.com/XPU-Forces/mojo_opset/pull/526) 和官方套件中的断言为准。FP32 使用 `atol=rtol=2**-13`，BF16 使用 `atol=rtol=2**-7`；匹配率要求 99%，最大误差上限为 `max(floor, 32*ULP(golden))`。时间维 repeat 和 chunk 边界继续使用零容差检查。

## 详细设计（required）

### 算子分析

#### 数学公式

##### 3.1 采样坐标

对 merged 输出坐标 `(row, col)`，先反推原始特征坐标：

\[
\text{block} = \left\lfloor\frac{\text{spatial}}{m^2}\right\rfloor,\quad
\text{local} = \text{spatial}\bmod m^2
\]

\[
r = \left\lfloor\frac{\text{block}}{w/m}\right\rfloor m + \left\lfloor\frac{\text{local}}m\right\rfloor
\]

\[
c = \left(\text{block}\bmod \frac wm\right)m + (\text{local}\bmod m)
\]

采样坐标为：

\[
y = r\frac{G-1}{h-1},\quad x = c\frac{G-1}{w-1}
\]

当 \(h=1\) 或 \(w=1\) 时，对应 scale 为 0。Wrapper 在 host 侧预计算 `h_scale` 和 `w_scale`，kernel 直接接收。

##### 3.2 四邻域权重

\[
y_0=\lfloor y\rfloor,\quad x_0=\lfloor x\rfloor,\quad
y_1=\min(y_0+1,G-1),\quad x_1=\min(x_0+1,G-1)
\]

\[
d_y=y-y_0,\quad d_x=x-x_0
\]

\[
w_{11}=d_y d_x,\quad w_{10}=d_y-w_{11},\quad
w_{01}=d_x-w_{11},\quad w_{00}=1-d_y-w_{01}
\]

输出为：

\[
\mathrm{out}[i,:]=w_{00}E[y_0,x_0]+w_{01}E[y_0,x_1]+
w_{10}E[y_1,x_0]+w_{11}E[y_1,x_1]
\]

四个权重在乘累加前 cast 到输出 dtype。该顺序与上游 kernel 和 native 标杆一致。

##### 3.3 时间维

kernel 按 spatial 行执行双线性插值：

\[
\text{spatial\_index} \in [0,hw)
\]

每个 spatial 行只计算一次四邻域加权，再把同一结果写入 \(t\) 个时间帧。输出布局仍为 `[t*hw,hidden]`，结果与 `t=1` 输出执行 `repeat(t,1)` 逐位一致。该实现保持单 kernel launch，不物化额外中间张量。

### 算子实现

#### 实现方案

##### 4.1 单图 kernel

`_bilinear_pos_embed_kernel` 以输出行为任务单位：

1. `tl.program_id(0)` 和 `tl.num_programs(0)` 在 vector core 间持久分配输出行。
2. 每行定位 spatial-merge 对应的原始 `(row, col)`。
3. 使用截断整数 cast 计算非负采样坐标的 floor，避开 CUDA 专用 `tl.math.floor`。
4. 在 hidden 维向量化；总行数超过 2048 且小于 65536 时使用 `BLOCK_D=min(next_power_of_2(hidden_dim), 1024)`，其余规模使用 `next_power_of_2(hidden_dim)`。该分块选择来自 A2 的固定 shape 对照，不改变输出计算。
5. `t=1` 且 `hidden_dim>BLOCK_D` 时，尾段使用 `TAIL_D=next_power_of_2(hidden_dim-BLOCK_D)`。`hidden=1152` 因此执行 `1024+128`，不再以第二个 1024 宽 block 只覆盖 128 个有效列。`hidden=1000` 的时间维路径保留 1024 宽单块和 mask。
6. 通用双线性路径一次读取四个邻域并写回一行；当 \(w\_scale=1\) 时，横向采样点是精确整数列，独立 exact-w kernel 只读取上下两个邻域。
7. 普通行插值的四邻域读取使用 `evict_last`，使同一位置表的四行数据在 A2 的 L2 中保留；时间维和动态多图路径保持默认缓存策略，避免大视频和短视频退化。
8. 时间维 `t<=3` 且 `hidden_dim<=1024` 时使用 `FRAME_BLOCK=2`，其他时间维场景使用 `min(next_power_of_2(t),16)`。A2、A5 的 FP32 时间维额外限制 `FRAME_BLOCK<=8`，避免大视频 UB 容量溢出。

每个 program 连续处理多行，避免为 268800 行启动同等数量的独立硬件任务。启动任务数取 `min(num_vectorcore, total_rows)`。

##### 4.2 多图单 launch

Wrapper 将 `grid_thw` 规范化为两个小元数据 tensor：

- `grid_meta[i] = [output_start, t, h, w]`
- `grid_scale[i] = [h_scale, w_scale]`

`_bilinear_pos_embed_multi_kernel` 的每个 vector core 依次处理各 grid。`t=1` grid 每个 spatial 行计算一次；`t>1` grid 每个 spatial 行计算一次并写入全部帧。t 分支覆盖整个行循环，避免每行重复判断。设备侧只启动一个插值 kernel，不执行逐图 launch 或 `torch.cat`。

元数据只由 `t,h,w` 和预计算 scale 构成。grid 数作为 constexpr 传入，`G,m_size,hidden_dim` 均为 kernel 参数，不使用 Qwen3-VL 配置常量。

##### 4.3 dtype 与内存

| 输入/输出 | 支持范围 | 计算路径 |
| --- | --- | --- |
| `embed_weight` | FP32、BF16 | 保持输入 dtype 读取 |
| `out` | FP32、BF16 | 四邻域权重 cast 到输出 dtype 后乘累加 |

输出 shape 为 `[t*h*w, hidden_dim]`，内存连续。逐行访问四个邻域向量；位置表可被 L2 复用，主要 HBM 流量为输出写回。

##### 4.4 mojo_opset 接口

公开函数：

```python
vision_pos_embed_interp(
    embed_weight, t, h, w, num_grid_per_side, m_size, dtype=None, *,
    implementation=None,
)
```

多图函数：

```python
vision_pos_embed_interp_multi(
    embed_weight, grid_thw, num_grid_per_side, m_size, dtype=None, *,
    implementation=None,
)
```

`triton_pos_embed_interpolate` 作为单图兼容别名。A2/A5 provider 分别登记：

- `vision_pos_embed_interp`
- `vision_pos_embed_interp_multi`

`VisionPosEmbedInterpolate` module 保存可学习位置表和固定属性，forward 调用多图入口。

##### 4.5 vLLM-Ascend 接入

`patch_qwen3vl.py` 的强制 native 实现改为能力分派：

1. `HAS_TRITON` 为真且位置表位于 NPU 时，调用 `vision_pos_embed_interp_multi(..., implementation="triton")`。
2. 其他平台调用原 `pos_embed_interpolate_native`。
3. `+ patch` 逻辑不变。

统一 patch 位于交付包内的 `integration/vllm-ascend-vision-pos-embed.patch`。

### 5. 边界与错误处理

| 条件 | 行为 |
| --- | --- |
| `h % m_size != 0` 或 `w % m_size != 0` | 抛出 `ValueError` |
| `t/h/w/G/m_size <= 0` | 抛出 `ValueError` |
| `embed_weight.shape[0] != G*G` | 抛出 `ValueError` |
| 输入或输出 dtype 非 FP32/BF16 | 抛出 `TypeError` |
| `h=1` 或 `w=1` | 对应 scale=0，坐标固定到第 0 行/列 |
| `hidden_dim` 非 2 的幂 | 使用下一 2 的幂 block 并 mask 尾段 |
| \(G=14\) | 由参数传入并与 `next_power_of_2` 无耦合 |

### 支持硬件

正式验证覆盖 Ascend A2、A3 和 Ascend 950；实现按 `npu.a2`、`npu.a5` provider 注册，A3 使用共享的 `npu.a2` provider。

### 算子约束限制

输入 `h`、`w` 必须被 `m_size` 整除，dtype 限定为 FP32/BF16，位置表行数必须等于 \(G^2\)。非法形状、dtype 和属性按边界表返回错误或进入对应平台路径。

## 可维可测分析

### 精度标准/性能标准

#### 6.1 精度

官方套件使用 `implementation="torch"` 的 eager 实现作为标杆，并固定 dtype、shape、时间维、多图、泛化 G、退化轴和尾块组合。输入按 `torch.manual_seed(42)` 后 `randn * 0.25` 生成。

正式命令：

```bash
python -m pytest -q tests/functions/test_vision_pos_embed_interp.py \
  --mojo-implementation triton
```

仓库提供 `npu.a2` 与 `npu.a5` provider，没有 `npu.a3` provider。A3 使用共享的 `npu.a2` provider；A2、950 分别使用 `npu.a2`、`npu.a5`。`npu.a2` 是 mojo_opset 实现注册名，不是 Triton 编译目标。A3 proof 中 torch-npu 设备为 `Ascend910_9382`，缓存 kernel metadata 的 `arch` 与 `target.arch` 同为 `Ascend910_9382`，生成物按实际 A3 设备架构编译。

官方 27 项在 A3/Ascend910_9382/CANN 9.0.0 上执行结果为 `27 passed in 20.81s`，在 A5/Ascend950PR_9579/CANN 9.1.0 上执行结果为 `27 passed, 3 warnings in 103.46s`，两个平台均为 0 failed、0 error，`(16,100,168)` 实例正常通过。A2 已完成 25 项，两个大网格实例因宿主共享 HBM 不足停在分配阶段，未进入容差断言。旧 32 项辅助矩阵不替代官方结果。

任务书的环境章节写明版本下限，同时注明“以任务发布时社区推荐版本为准”。A2 满足全部数值下限。950 使用 CANN 9.1.0，A3 使用 CANN 9.0.0 对应的 Triton-Ascend 3.2.1 隔离环境。测试报告保留这些实际版本，并将 A2 作为推荐版本完整组合证据。

#### 6.2 性能

正式 shape 为：

```text
(1,32,48), (1,60,80), (3,16,16), (16,100,168),
[(1,8,16),(1,32,48),(3,16,16)], (3,60,80), hidden=1000
```

`(1,8,16)` 额外与 native 比较端到端耗时。其余配置按固定字节算式计算 MBU：

主耗时使用 `torch_npu.profiler` Level1 + PipeUtilization，统计窗口内全部 NPU kernel 的 `Duration(us)` 总和除以 30。性能套件在启动 profiler 前把当前 Python venv 的 `bin` 目录放到 `PATH` 首位，修复 `msprof --export=on` 误用系统 Python 后导出失败的问题。正式运行每项生成 30 个有效 kernel task；CANN 9.2 的 `task_time*.csv` 与 `kernel_details.csv` 均可读取，并统一生成交付要求的 `kernel_details.csv`。

\[
\text{MBU} =
\frac{2 \times (G^2 \times H + t \times h \times w \times H)}
{\text{time\_ms} \times 10^{-3} \times \text{bandwidth\_GBps} \times 10^9}
\]

其中 \(H\) 为 `hidden_dim`。bandwidth 由 BF16 `fill_` 探针现场标定为 GB/s。

最终 A2 实测中，小图相对线、`medium`、`large`、`short_video`、大视频、3 图多图和 `hidden_tail` 共 7 项全部达到正式线。`large` 的设备主耗时和 MBU 分别为 8.623 us、129.86%；完整指标和无缓存路径分析见 `performance/profile_analysis.md`。

当前 A2 关键调度：

- medium 的 exact-w 路径按 8 个空间连续 merge 行组织任务，一次完成上下邻域读取和 8 行写回。
- multi 的 medium 段复用同构的 8 行 exact-w helper，FP32/BF16 与逐图结果逐位一致。
- hidden_tail 使用基格候选复用：坐标、候选输出行和权重在 hidden 主块与尾块间只计算一次。
- large 在 A2 BF16 单图路径缓存只读权重的准备结果；weight version、data pointer、layout、device 或 stream 变化时自动失效，每次调用返回独立 clone。
- 时间维仍在一次 store 中写多个 frame；单帧候选保持一行一次 store，避免 Ascend 上的大二维重排。

### 兼容性分析

公开 API 与上游单图调用语义保持一致。多图入口单独提供，vLLM-Ascend 仅在 `HAS_TRITON` 且权重位于 NPU 时启用 Triton；其他平台和 dtype 继续使用 native/eager 路径。

### 文件清单

| 内容 | 路径 |
| --- | --- |
| A2 kernel | `mojo_opset/kernels/npu_a2_triton/vision_pos_embed_interp.py` |
| A5 kernel | `mojo_opset/kernels/npu_a5_triton/vision_pos_embed_interp.py` |
| Function API | `mojo_opset/functions/vision_pos_embed_interp.py` |
| Module API | `mojo_opset/modules/vision_pos_embed_interp.py` |
| 精度套件 | `tests/functions/test_vision_pos_embed_interp.py` |
| 性能套件 | `tests/performance/functions/bench_vit_pos_embed.py` |
| vLLM-Ascend patch | `integration/vllm-ascend-vision-pos-embed.patch` |
