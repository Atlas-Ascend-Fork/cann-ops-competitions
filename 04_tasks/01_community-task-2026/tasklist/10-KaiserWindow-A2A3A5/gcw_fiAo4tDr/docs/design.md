# KaiserWindow 算子设计（A2 / A3 / A5）

- 作者：gcw_fiAo4tDr
- 实现方式：SiP Ascend C kernel 直调 + sip_pta PyTorch 扩展
- 对应源码提交：`8aa05011c1e04a466e1587ebceac132e55bcf2d6`
- SiP 上游基线：`d1cf7c821e28b206288fe4bd7383c72a334e464f`

# 需求背景（required）

## 需求来源

社区任务《torch_npu 支持 kaiser_window 接口（A2/A3/A5）》要求在 SiP 中实现 KaiserWindow，并通过 `torch_sip.asd_kaiser_window` 提供 NPU 工厂接口。本设计按社区任务的 [design_template.md](../../../../resources/design_template.md) 编写。

- 算子及 Torch 封装提交目标：[cann/sip](https://gitcode.com/cann/sip)。
- 任务讨论与环境说明：[SiP discussion #4](https://gitcode.com/cann/sip/discussions/4)。任务包原文列 CANN 9.3.0，本次按讨论确认的 CANN 9.2 开发，三台实测设备均使用 CANN 9.2.0。
- 数学与接口参考：PyTorch 2.7.1 的 `torch.kaiser_window`、`ATen/native/Math.h`。

## 背景介绍

Kaiser 窗用于频谱分析、滤波器设计等场景。窗口值仅由长度、周期性和形状参数决定，没有输入 Tensor。将公式拆成多个逐元素算子会增加 kernel 下发和中间张量的开销；本设计在单个计算 kernel 内完成索引生成、贝塞尔函数近似、指数和输出转换。

A2/A3 使用 arch22 向量指令路径，A5 使用 arch35 RegBase/MicroAPI 路径。两条路径共享接口、数学语义和分块原则，分别组织设备代码及 tiling 结构。

# 需求分析（required）

## 需求描述

生成长度为 `window_length` 的一维连续 NPU Tensor，支持 FLOAT16、BFLOAT16、FLOAT32；实现周期窗、对称窗及任务规定的边界行为。Kernel 和 Torch 封装位于同一 SiP 源码仓库。

### Python 接口

```python
torch_sip.asd_kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> torch.Tensor
```

| 参数 | 语义与约束 |
| --- | --- |
| `window_length` | host int64，范围 `[0, INT32_MAX]`，输出形状 `[window_length]` |
| `periodic` | 默认 True；True 为周期窗，False 为对称窗 |
| `beta` | host double，默认 12.0；不限制取值，负值及非有限值按下述边界规则处理 |
| `dtype` | 支持 `torch.float16`、`torch.bfloat16`、`torch.float32`；None 固定按 float32 处理 |
| `layout` | `torch.strided` 或 None；输出为一维连续 strided Tensor |
| `device` | 仅 NPU；None 使用当前 NPU，指定索引时由 NPUGuard 绑定 |
| `requires_grad` | 默认 False；True 时返回需要梯度的叶子 Tensor，无针对 host 标量参数的反向计算 |

底层注册入口为 `torch.ops.torch_sip.asd_kaiser_window`，完整默认值、layout 和 requires_grad 由 Python 封装处理。

### SiP 直调接口

```cpp
extern "C" {
AsdSip::AspbStatus asdKaiserWindowGetWorkspaceSize(size_t *workspaceSize);
AsdSip::AspbStatus asdKaiserWindow(aclTensor *outTensor,
                                  int64_t windowLength,
                                  bool periodic,
                                  double beta,
                                  void *stream,
                                  void *workspace);
}
```

`GetWorkspaceSize` 检查指针后返回 0。调用方提供输出 Tensor 和当前 stream，SiP 不创建输出。输出须为 ND 格式、单维 `[windowLength]`、连续 view、非负 storage offset；非空输出需要有效 stream 和足够的有效设备存储。非零 offset 的输出地址按 storage 地址加 view offset 绑定。

参数或元数据错误返回 `ACL_ERROR_INVALID_PARAM`，不支持的 dtype 返回 `ACL_ERROR_UNSUPPORTED_DATA_TYPE`，底层执行失败映射为 `ACL_ERROR_INTERNAL_ERROR`。PTA 使用 `TORCH_CHECK` 检查设备、长度和 dtype，并检查 SiP 返回状态。

## 需求拆解

| 子项 | 设计方案 |
| --- | --- |
| 功能与数值 | 使用缩放贝塞尔函数 i0e 形式，FP32 中间计算；独立处理空窗、单点窗、零 beta 和非有限 beta |
| 平台适配 | 公共 Operation 按平台选择 arch22 或 arch35 Kernel |
| 性能 | 单计算 kernel、32 B 对齐分核、4096 元素 tile、host 计划复用；无与输出等长的中间 GM 张量 |
| Torch 接入 | TORCH_LIBRARY 注册、BackendSelect/PrivateUse1 实现、当前 NPU stream 下发 |
| 生命周期 | 异步回调按值持有输出与标量，转换描述符在回调内创建并释放；计划不长期保存输出地址 |
| 验证 | 原始精度 1000 条、接口 23 条、性能 200 条及补充边界、异步、偏移、内存测试 |

# 详细设计（required）

## 算子分析

### 数学公式

令 `L = window_length`。当 `L >= 2` 且 beta 有限时：

```text
N = L                  periodic=True
N = L - 1              periodic=False
b = abs(beta)
r[i] = (2*i - N) / N
t[i] = sqrt(abs(1 - r[i]*r[i]))
w[i] = I0(b*t[i]) / I0(b)                    0 <= i < L
```

周期窗相当于长度 `L+1` 的对称窗丢弃最后一点，实际只计算并写出 `L` 个元素。

使用 `i0e(x) = exp(-abs(x))*I0(x)` 改写：

```text
w[i] = i0e(b*t[i]) * invI0e * exp(b*(t[i]-1))
invI0e = 1 / i0e(b)
```

Host 在计划初始化时用 double 计算 `invI0e`，转为 float 传入 kernel。核内用 FP32 计算，最终转换到输出 dtype。该形式避免在核内直接形成大数 `I0(b)`。

`2*i-N` 先在整数域构造。实现按 tile 起点使用 `start-(N-start)`，再叠加局部偶数索引，避免直接形成可能超出 int32 的全局 `2*i`，并减少中心附近的浮点相减误差。整数分子为零的真实中心点直接输出 1。

### 边界行为

| 场景 | 处理 |
| --- | --- |
| `L=0` | 返回空 Tensor，不启动计算 kernel，workspace 为 0 |
| `L=1` | 输出 `[1]`，优先于 beta 的所有特殊值处理 |
| `L>=2, beta=0` | 输出全 1 |
| `L>=2, beta<0` 且有限 | 与 `abs(beta)` 相同 |
| `L>=2, beta=NaN/+inf/-inf` | 输出全 NaN |
| `L=2, periodic=False` | 两点均为 `1/I0(beta)`，按有限 beta 的正常数值路径计算 |
| `L=2, periodic=True` | 有限 beta 时为 `[1/I0(beta), 1]` |
| 开方参数 | 先取绝对值再 sqrt，有限 beta 的端点不因微小负舍入量产生 NaN |

### 支持数据类型与形状

- 无输入 Tensor，无广播；三个 host 标量共同确定输出。
- 输出 ND 格式、形状 `[L]`、FLOAT16/BFLOAT16/FLOAT32。
- FP16/BF16 从 FP32 以 round-to-nearest-even 转换；不以低精度 dtype 执行贝塞尔近似。
- FLOAT64、整数、BOOL、复数不在支持范围。

## 算子实现

### 实现方案

```text
torch_sip.asd_kaiser_window
  -> typed dispatcher / Torch mode 分派
  -> TORCH_LIBRARY 的 BackendSelect / PrivateUse1 实现
  -> NPUGuard + at::empty 创建输出
  -> RunOpApiV2 回调
  -> asdKaiserWindow + OpDesc
  -> RunAsdOpsV2 + MKI Operation/Kernel
     -> ASCEND_910B: KaiserWindowKernel / kaiser_window
     -> ASCEND_950:  KaiserWindowArch35Kernel / kaiser_window_arch35
  -> 在当前 NPU stream 写出结果
```

主要文件按当前实现组织如下，路径相对于 SiP 根目录：

| 层次 | 文件 |
| --- | --- |
| C API 与声明 | `core/base/kaiser_window.cpp`、`include/base_api.h` |
| 参数与注册 | `ops/include/params/kaiser_window.h`、`configs/op_list.yaml`、`ops/base/CMakeLists.txt` |
| Operation 与构建 | `ops/base/kaiser_window/kaiser_operation.cpp`、`ops/base/kaiser_window/CMakeLists.txt` |
| arch22 host / tiling / device | `ops/base/kaiser_window/kaiser_kernel.cpp`、`kaiser_tiling.h`、`kaiser_window.cpp` |
| arch35 host / tiling / device | `ops/base/kaiser_window/kaiser_kernel_arch35.cpp`、`kaiser_tiling_arch35.h`、`kaiser_window_arch35.cpp` |
| 公共执行上下文 | `core/utils/include/utils/ops_base.h`、`core/utils/ops_base.cpp` |
| PTA 与 Python | `sip_pta/csrc/base/asd_kaiser_window.cpp`、`sip_pta/torch_sip/__init__.py` |
| 单算子测试 | `sip_pta/test/base/kaiser_window.py` |

#### host 侧设计

##### 1. 分核策略

按输出字节宽度，将输出划分为 32 B 块。FLOAT32 每块 8 元素，FLOAT16/BFLOAT16 每块 16 元素。设每块元素数为 `e`：

```text
B = ceil(L/e)
C = min(B, available_vector_cores)     # arch22 另将可用核数上限设为 48
q = B // C
r = B % C
start_block(k) = k*q + min(k,r)
capacity(k) = (q + (k<r)) * e
start(k) = start_block(k) * e
count(k) = min(capacity(k), L-start(k))
```

前 `r` 个核多处理一块，核间相差最多一个 32 B 块。核起点对齐，只有整体最后一块可能不足 32 B。每个核独占不重叠的输出区间，无核间通信或同步。

A3 按当前单芯片可用 AIV 数计算 blockDim；两逻辑芯片的核数不能相加作为一次 kernel 的核数。

##### 2. 分块与内存规划

每核顺序处理自己的连续区间，单 tile 最多 4096 元素。最后一个 tile 按真实元素数计算和搬出。

| 项目 | arch22 | arch35 |
| --- | --- | --- |
| tiling | `KaiserTiling`：长度、分母、48 个 start/count、beta、invI0e、dtype、special；当前字段布局 800 B，构建以 sizeof 为准 | `KaiserWindowTilingArch35`：长度、分母、q/r、核数、e、beta、invI0e、dtype、special；static_assert 为 56 B |
| 数学临时数据 | 六个 4096×4 B UB buffer，共 98304 B | VF 寄存器 |
| 掩码 UB | `4096/8+256 = 768 B` | 使用寄存器掩码 |
| 输出队列 | 单 buffer，4096×4 = 16384 B | 单 buffer，4096×4 = 16384 B |
| 显式 UB buffer 合计 | 115456 B | 16384 B |
| 业务 workspace | 0 | 0 |

上述 UB 数值为源码显式申请量。运行时的 tiling 存储、kernel 参数及管理性分配单独存在，workspace=0 不代表整个进程只有输出一次分配。

##### 3. 路由和 tiling key

Operation 根据 `PlatformType::ASCEND_910B` / `ASCEND_950` 选择两个具名 Kernel，未另设 GE/ACLNN 式 tilingKey 编码。tiling 中 `dtype=0/1/2` 分别选择 FLOAT32/FLOAT16/BFLOAT16；`special=0/1/2` 分别表示普通计算、全 1、全 NaN。`L=0` 在 host 提前返回。

##### 4. host 计划复用与地址安全

重复工厂调用没有输入 Tensor，计划复用用于减少 Operation 查找、元数据和 tiling 初始化开销：

1. 保留原五参数 `RunAsdOpsV2`，增加显式上下文重载，由调用方选择使用。
2. Kaiser 使用 thread-local 单项缓存，键为 device、dtype、L、periodic 和 beta 的完整 double 位模式。键变化时 Reset，避免把 NaN、正负零等不同位模式错误复用。
3. 公共上下文检查 device、opName、参数类型及输出元数据；标量变化由 Kaiser 调用方负责 Reset。
4. 仅复用无输入、单个 dense ND 输出，且无 scratch、const Tensor、hwsync、memset、特殊 kernel args 的工厂计划。不满足条件时在提交前走原调用路径。
5. 每次重新绑定当前输出的 storage+view offset 和当前 stream；调用结束清除借用的 data、hostData 和输出参数地址。计划只保留可复用的元数据及 kernel 状态。
6. 缓存使用中发生重入时采用独立上下文；Reset 对使用中的计划延迟生效。已提交执行返回失败后不自动再次提交。

##### 5. PTA 异步生命周期

PTA 仅以 `at::empty` 创建输出。`RunOpApiV2` 回调按值捕获输出、标量和 stream，回调内执行 `ConvertTypes`，并以 RAII 在执行结束后 `ReleaseConvertTypes`。这样不会在回调执行前销毁临时 aclTensor 描述符，输出存储也由被捕获的 Tensor 持有。

#### kernel 侧设计

##### 1. 公共计算流程

每 tile 执行：整数索引生成 → FP32 归一化 → `sqrt(abs(1-r*r))` → i0e 分段近似 → 指数缩放 → 中心点修正 → 输出 dtype 转换 → GM 写出。无输入 CopyIn，也不创建与输出等长的中间 GM Tensor。

i0e 以 `x=8` 分段，小值段使用 12 个系数，大值段使用 4 个系数，按 Horner 形式求值；大值段含 `1/sqrt(x)` 缩放。系数来源为 PyTorch 2.7.1 / Cephes 的近似展开，源码保留来源及 `PYTORCH_LICENSE.txt`。分段边界、端点和中心点纳入精度测试。

##### 2. A2/A3：arch22

- 用 TBuf 保存 FP32 临时值及整数索引，向量运算按依赖插入 `PipeBarrier<PIPE_V>`。
- baseIndex 在 tile 间复用，叠加当前 tile 的整数起点；beta 不大于 8 时省略大值段。
- 中心点用标量 SetValue 写成 1，前后使用 V_S / S_V 事件保护标量与向量依赖。
- 输出通过 `AllocTensor → Cast/拷贝 → EnQue → DeQue → DataCopyPad → FreeTensor` 流转。DMA 长度为 `n*sizeof(T)`，最后一块不扩大 GM 写出长度。

##### 3. A5：arch35

- 使用 `__simd_vf__` 和 Reg/MicroAPI，按 64 个 FP32 lane 计算，tile 内 `UpdateMask` 使用同一个递减 remaining，保护最后不足 64 lane 的数据。
- 以整数分子比较生成中心点掩码；小、大两段均计算后按 x 选择，大值段分母使用 `max(x,8)`。
- FLOAT16/BFLOAT16 使用 `NO_SAT`、`CAST_RINT` 转换和 packed store；FP32 直接 StoreAlign。
- 数学临时值保存在寄存器，显式 UB 仅保留输出队列；队列依赖和精确字节 DataCopyPad 负责计算到搬出的衔接。
- arch35 构建指定 AIV_ONLY，设备入口设置 `KERNEL_TYPE_AIV_ONLY`，CCE 目标为 `dav-c310-vec`。该选项仅用于此 A5 向量 kernel 的构建路径。

## 支持硬件

| 平台 | 编译/实现路径 | 当前实测设备 | 验证范围 |
| --- | --- | --- | --- |
| A2 | ascend910b / arch22 | Ascend910B3，NpuArch2201，40 AIV | 完整同版构建及上板测试 |
| A3 | ascend910b / arch22 | Ascend910_9382 V1，NpuArch2201，每芯片 48 AIV，2 逻辑芯片 | 完整同版构建及上板测试，含双设备补充验证 |
| A5 | ascend950 / arch35 | Ascend950PR_9579，NpuArch3510，56 AIV / 28 AIC | 完整同版构建及上板测试；950DT 未实机验证 |

三平台使用同一份 27 文件源码快照。环境为 CANN 9.2.0、torch 2.7.1+cpu、torch_npu 2.7.1.post11、C++11 ABI=1；SiP 与 torch_sip 从同源构建。A5 的单设备验证不构成 A5 多设备验证。

## 算子约束限制

1. 长度接口上限为 INT32_MAX；实际分配受设备内存限制。最大长度的索引安全按源码分析，未实测该长度的完整分配与运行。
2. 正常数值精度验收域按任务限定 beta 为 `[0,88]`；负 beta 作偶函数语义补充测试。非有限值单独验证全 NaN 或单点窗行为，不计作数值精度样本。
3. Host 对 beta>500 有渐近计算分支，但极大有限 beta 的全域精度及与 CPU 溢出行为的完全等价性未覆盖，不能从当前结果外推。
4. 直调调用方负责输出实际存储容量、设备/stream 一致性和异步执行期间的存储生命周期。
5. 仅提供所述三种输出 dtype、一维连续输出及 NPU 后端，无广播、非连续输出和 CPU 静默回退。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| 精度 golden | CPU `torch.kaiser_window(dtype=torch.float64)` 在线生成，再收窄到被测输出 dtype | 任务包精度脚本及任务书 §3.2 |
| 混合容差 | `abs(actual-golden) <= atol+rtol*abs(golden)`；FP16 的 atol/rtol=2^-10，BF16=2^-7，FP32=2^-13；matched_ratio≥0.99 | 任务书及生态算子精度标准 |
| 最大误差 | 有限比对点 max_abs_error 不超过 `max(兜底值,32*ULP(g_low))`，g_low 为最大误差点的同 dtype golden；FP16/BF16 兜底 0.1，FP32 兜底 0.01 | 任务包原始实现 |
| 非有限及退化边界 | 非有限 beta 全 NaN、L=0 空 Tensor、L=1 全 1，分别按语义判定 | 任务书边界表 |
| A2/A3 性能 | 每条任务提供的 A100 时间 / NPU 时间≥0.8 | 任务包 A100 CSV，200 条 |
| A5 性能 | 每条任务提供的 H100 时间 / NPU 时间≥0.4 | 任务包 H100 CSV，200 条 |
| 内存 | workspace=0、无中间 GM 数据张量；源码审查及运行时观察输出生命周期、重复调用保留量 | 任务书 §3.4 及补充测试 |

性能严格沿用任务原脚本：每例 warmup 10 次，NPU Event 区间包住 60 次调用，elapsed/60 为单次平均耗时；保留全部 200 条，不替换成中位数或拼接各轮最优结果。GPU 基线来自任务包，未现场重采。原始 10 个测试/数据文件在回归前后按 SHA256 核对保持不变。

### 当前最终同版验证结果

| 项目 | A2 | A3 | A5 |
| --- | --- | --- | --- |
| 精度语义用例 | 1000/1000 | 1000/1000 | 1000/1000 |
| API 契约 | 23/23 | 23/23 | 23/23 |
| 完整性能两轮 | 200/200、200/200 | 200/200、200/200 | 200/200、200/200 |
| 两轮最小 GPU/NPU 比值 | A100：0.8685 / 0.8680 | A100：1.0054 / 0.9930 | H100：0.6729 / 0.7050 |
| 核数/tile 边界 | 336/336 | 336/336 | 264/264 |
| 异步等补充场景 | 16/16 | 18/18，含双设备 | 16/16 |
| 内存场景及重复调用 | 15 项、128 次通过 | 15 项、128 次通过 | 15 项、128 次通过 |
| 非零 offset | 3 种 dtype 通过 | 3 种 dtype 通过 | 3 种 dtype 通过 |
| 调用模式 | 6/6 | 6/6 | 6/6 |

每组 1000 条由 867 条数值精度、118 条非有限边界、3 条空张量、12 条长度 1 组成，非有限边界不作为数值精度样本。上述数据描述最终正式测试覆盖范围，不保证所有负载和后续测量无波动。

### 可维护性与复现

- 数学/接口语义集中在公共入口和两个明确命名的 Kernel，架构差异限定在 kernel、tiling 和平台构建配置中。
- 原始测试包与补充测试按平台配置运行，分别留存精度、性能、内存报告及日志和实际环境/终端截图。
- 通过源码快照 SHA、原测试文件 SHA、构建/安装库身份及证据 MANIFEST 关联源码和报告；原始日志保存逐例结果与退出状态。
- 重现时先核对 CANN、torch/torch_npu、ABI 及 SoC，再按 SiP 和 sip_pta 的构建说明同源安装，最后运行原测试与相应设备的补充验证。

## 兼容性分析

- 新增 `asdKaiserWindow`、`asdKaiserWindowGetWorkspaceSize` 和 `torch_sip.asd_kaiser_window`，不修改既有算子的接口定义。
- 公共执行器原有五参数入口保留；显式上下文重载限定工厂算子的适用条件，其他场景沿原执行路径。Kaiser 自行负责标量缓存键，公共执行器不假定 Any 中参数值相等。
- Python 封装保留 Torch function mode 分派能力；默认 dtype 固定为任务指定 float32，默认 device 跟随当前 NPU。
- A2/A3 及 A5 分别编译匹配架构的设备代码；CANN 9.2.0 为当前验证版本，其他 CANN/torch_npu 组合需重新构建及回归，不能仅凭源码支持推定二进制兼容。
