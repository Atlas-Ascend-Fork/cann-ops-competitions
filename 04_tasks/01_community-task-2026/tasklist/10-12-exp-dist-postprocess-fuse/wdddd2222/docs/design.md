# FusedExponential 算子设计

# 需求背景

## 需求来源

本设计对应十月份社区任务编号 12「指数分布后处理融合算子开发」，实现目标为
`cann/ops-math` 的 `experimental/math/FusedExponential`。
需求依据为[官方任务书与自验包](https://www.hiascend.com/p/resource/202610/9853119d782f4b139ba97206b28a458e.zip)。

## 背景介绍

[op-plugin 26.1.0 的指数分布实现](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)
在均匀采样后依次执行 neg、add、比较、mask_fill、log 和 mul。
比较操作会生成 bool 中间张量，多次 kernel 调用还会重复读写全局内存（GM）。
本算子融合采样之后的变换，直接接收调用方提供的均匀随机样本，不负责随机数生成。

# 需求分析

## 需求描述

算子提供 `aclnnFusedExponentialGetWorkspaceSize` 和 `aclnnFusedExponential`
两段式接口，使用 Ascend C 实现逐元素指数变换。连续输入、输出时不产生 GM 中间张量，
尤其不生成 bool mask，返回的 workspaceSize 为 0。

## 需求拆解

1. 输入、输出支持 FLOAT16、BFLOAT16 和 FLOAT32，二者具有相同 shape、dtype 和 ND 格式。
2. 支持秩 1–8 的张量，并处理任务书要求的零维标量和空张量。
3. 支持非连续输入、非连续输出及输入输出完全原地复用；非连续布局由框架完成连续化和结果回写。
4. Host 侧接收 DOUBLE 类型的率参数，检查参数和张量元数据；Device 侧不扫描输入值域。
5. 在 A2/A3 上实现确定性的异步计算，以 910B3 完成任务书规定的精度、性能和内存自验。

# 详细设计

## 算子分析

### 数学公式

对每个元素计算：

$$
y_i=-\log\left(\min(1-u_i,1-\varepsilon)\right)/\lambda,
\qquad \varepsilon=\mathrm{finfo}(\mathrm{dtype}).\mathrm{eps}/2.
$$

输入由调用方保证满足 `0 <= u < 1`。上限截断只作用于 `u <= epsilon` 一侧，
不截断 `u` 接近 1 时的指数分布长尾。
FLOAT16、BFLOAT16、FLOAT32 的 epsilon 分别为 `2^-11`、`2^-8`、`2^-24`。
BFLOAT16 采用任务书公式和自验 CPU 标杆的 `finfo(dtype).eps/2`，即 `2^-8`；
任务书参数表中的 BFLOAT16 近似指数不参与常量计算。

### 接口与参数

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor *u, const aclScalar *lambd, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFusedExponential(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

| 参数 | 设计约束 |
| --- | --- |
| `u` | Device 张量，FLOAT16、BFLOAT16 或 FLOAT32，ND 格式，秩不超过 8。 |
| `lambd` | Host DOUBLE 标量，要求 `lambda > 0`；非正数和 NaN 返回参数错误，正无穷对应全零输出。 |
| `out` | Device 张量，与 `u` 的逻辑 shape、dtype 相同，支持独立输出和完全原地复用。 |
| `workspaceSize`、`executor` | 第一阶段返回执行所需的工作区大小和执行器。 |
| `workspace`、`stream` | 第二阶段按返回大小提供工作区，在调用方指定的 stream 上异步执行。 |

必要的指针为空时返回 `ACLNN_ERR_PARAM_NULLPTR`；dtype、shape、格式、秩或率参数不合法时
返回 `ACLNN_ERR_PARAM_INVALID`。成功返回 `ACLNN_SUCCESS`，框架内部执行错误向上传递。
Device 结果读回之前，调用方需要同步 stream。

### 支持数据类型和形状

输入、输出的 dtype 一致，不提供类型提升或广播。张量按逻辑元素数量展开为一维计算，
因此不需要为每个维度单独设计 kernel。零维标量按一个元素计算；空张量返回空执行器，
不启动融合 kernel。

## 算子实现

### 实现方案

#### Host 侧设计

接口层依次完成元数据校验、连续化处理、率参数转换、kernel 登记和必要的输出回写。
连续输入复用原存储；连续输出通过 `CreateView` 保留逻辑 shape 和 storage offset，
不申请额外 Device 张量。非连续输入调用框架 `Contiguous`；非连续输出申请连续临时张量，
执行后用 `ViewCopy` 写回原视图。这些布局转换可能需要工作区，区别于连续融合路径的中间计算存储。

Host 计算 `upper = 1 - epsilon` 和 FP32 缩放属性。普通率参数使用 `scale = -1/lambda`。
为防止先转为 FP32 的倒数过早溢出或下溢，极端率参数分两步缩放：

| 条件 | 第一次乘法系数 | 第二次乘法系数 |
| --- | --- | --- |
| `lambda < 2^-126` | `-2^-64/lambda` | `2^64` |
| `2^-126 <= lambda <= 2^126` | `-1/lambda` | `1` |
| `lambda > 2^126` | `-2^64/lambda` | `2^-64` |

正无穷对应零系数，kernel 直接写零。超过输出 dtype 表示范围的合法结果允许成为正无穷。

设元素总数为 `N`，平台提供的 AI Vector Core 数量为 `C`，使用核数为
`max(1, min(C, ceil(N/512)))`。每核处理元素数由 `ceil(N/核数)` 向上对齐至 32 字节，
FLOAT32 对齐 8 个元素，FLOAT16/BFLOAT16 对齐 16 个元素。
最后一个有效区间按实际剩余元素裁剪，多余核不执行数据处理。

tiling 数据包括总元素数、每核分段长度、第一次缩放系数、截断上限和第二次缩放系数。
使用一个 tiling key，数据类型由编译模板区分，不引入形状专用分支。
kernel 工作区大小设为 0，同时保留生成的 kernel ABI 中 workspace 参数的位置。
接口层以空地址的不可变描述符传递该参数，不创建零长度 GM 工作区张量。

#### Kernel 侧设计

每个核以最多 4096 个元素为一块执行搬入、计算和搬出。输入队列、输出队列及 FP32
计算缓冲区位于片上统一缓冲区（UB）。使用 `DataCopyPad` 按实际字节数处理非整块尾部，
避免越界读写。队列管理搬运与计算的依赖，向量计算之间按依赖插入同步。

FLOAT16、BFLOAT16 先转换为 FLOAT32，随后在 UB 中完成取负、加一、上限截断、对数和缩放，
最后以 round-to-nearest-even 舍入回原 dtype。FLOAT32 使用相同融合计算路径。
上限截断使用向量最小值指令，不构造 bool mask，GM 仅保存输入和最终输出。

FLOAT32 在第一次缩放系数小于 `-500` 时采用减法误差补偿，避免小 `lambda`
放大 `1-u` 的舍入误差。令 `v=max(u,epsilon)`、`t=fl(1-v)`，计算
`log(t) * v/(1-t)`，再执行上述缩放。`v>=epsilon` 保证合法输入下分母不为零，
额外的补偿缓冲区仍位于 UB。其他条件使用直接计算路径。

完全原地复用时，每块先读入 UB，再写回对应区间，各核处理互不相交的区间。
同一输入和率参数的运算顺序固定，不涉及原子更新或跨核归约，因此重复调用结果确定。

## 支持硬件

| 系列 | 编译目标 | 验证安排 |
| --- | --- | --- |
| A2 | `ascend910b` | 以任务指定的 910B3 完成全量自验。 |
| A3 | `ascend910_93` | 验证相同接口、融合路径及任务自验用例。 |

要求 CANN 9.1.0 及以上，PyTorch 和 torch_npu 2.7.1 及以上。

## 算子约束限制

1. 调用方提供均匀随机输入，本算子不负责采样、随机种子或随机状态管理。
2. 不支持广播，输入输出必须同 shape、同 dtype；不要求编译期未知 shape 的支持。
3. 接口不检查 `u` 的元素值域。`u>1` 时对数输入为负，输出 NaN；`u<0` 超出采样语义。
4. 输出受 dtype 表示范围限制，数学上的正值可能舍入为零或上溢为正无穷。
5. 完全原地复用受支持；部分重叠的不同视图不作为本次任务的支持承诺。
6. 连续输入、连续输出路径返回零工作区；非连续布局转换允许申请必要的临时存储。

# 可维可测分析

## 精度标准、性能标准和内存标准

| 验收项目 | 判据和方法 | 来源 |
| --- | --- | --- |
| 精度 | 使用任务自验包的 CPU torch 五算子链，以 FLOAT64 计算后按输出 dtype 收窄；逐元素混合容差匹配比例不低于 0.99，并同时检查最大绝对误差。 | 任务书 §3.2、§3.5。 |
| 性能 | 同设备、同 shape、同 dtype，NPU 拼接链平均耗时除以融合调用 Device 平均耗时不小于 1。 | 任务书 §3.3。 |
| 内存 | 记录输入、输出大小及 GetWorkspaceSize 返回值；连续路径 workspaceSize 为 0，且无 GM 算术中间张量或 bool mask。 | 任务书 §3.4。 |

| dtype | rtol | atol | 最大绝对误差上限 |
| --- | --- | --- | --- |
| FLOAT16 | `2^-10` | `2^-10` | `1e-1` 或 `32*ULP`。 |
| BFLOAT16 | `2^-7` | `2^-7` | `1e-1` 或 `32*ULP`。 |
| FLOAT32 | `2^-13` | `2^-13` | `1e-2` 或 `32*ULP`。 |

按任务包运行 1000 条精度用例，并启用比例为 0.4 的非连续输入测试。覆盖三种 dtype、
不同规模、零维标量、空张量、率参数边界、`u=0` 及接近 1 的输入。
对于标杆中存在 Inf 的用例，额外检查有限元素误差、非有限值位置及无穷的符号，
避免工具提前判定掩盖有限元素误差。任务包已登记的非连续空张量归一化失败需要在报告中注明，
保留原始失败记录，并确认融合算子未被调用，不将工具失败计为精度通过。

性能使用任务包选定的 200 条用例。拼接基线实际包含 neg、add、ge、mask_fill、log、mul
共六个 kernel，每条预热 10 次，Profiler 窗口内采集 60 次；将 `op_statistic.csv`
中各算子 Total Time 求和后除以采集次数。融合耗时由 ATK `performance_device` 采集。
该比较是算子 Device 耗时比较，不作为模型端到端加速结论。
性能分析使用官方 `msopprof`，仅对实际成功导出的设备指标作结论。

接口测试补充检查非连续输出、完全原地复用、尾块哨兵、参数错误及重复调用的确定性。
内存探针直接调用两段式接口记录工作区，不以编译时 tiling 声明代替运行时观测。

## 兼容性分析

本算子是新增接口，不修改现有指数分布接口或随机数状态。
构建遵循 `ops-math` 实验算子的标准入口，例如
`bash build.sh --pkg --experimental --soc=ascend910b --ops=FusedExponential --vendor_name=fused_exp`，
安装生成的自定义算子包后，通过导出的两段式接口验证。
对外验收材料按任务书另行提交，设计文档的提交和评审不代表代码已合入或任务已经验收。
