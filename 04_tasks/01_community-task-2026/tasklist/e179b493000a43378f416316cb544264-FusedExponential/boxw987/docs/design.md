# 需求背景

## 需求来源

本设计依据[10月社区任务—指数分布后处理融合算子开发](https://www.hiascend.com/activities/task-center/details/e179b493000a43378f416316cb544264)及其任务书，面向 A2/A3 系列产品设计 `aclnnFusedExponential`。实现代码拟合入 [ops-math](https://gitcode.com/cann/ops-math) 的 `experimental/math/FusedExponential` 目录。

本文说明接口、融合计算、数据搬运与验证方案。验证章节为实现后的测试计划。

## 背景介绍

指数分布采样可以拆为均匀随机数生成和逐元素后处理。对于已经生成的均匀样本 `u`，后处理依次执行 `neg`、`add`、边界处理、`log` 和 `mul`，得到率参数为 `lambda` 的指数分布样本。

[op-plugin 26.1.0 参考实现](https://gitcode.com/Ascend/op-plugin/blob/75532ae4300ea7e221cb3f89fdd503c635e4bdd8/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)在 A2/A3 路径中先生成 `u`，随后通过多个原地小算子完成变换。其中比较操作生成与输入等长的 bool mask，`masked_fill_` 再读取该 mask 处理边界。

本任务将后处理融合为一次逐元素计算。输入仍由调用方提供，均匀采样及随机数状态管理沿用调用方已有实现。

## 现状分析

参考链的逻辑步骤如下：

```text
u → neg → add(1) → ge(1-eps) → masked_fill → log → mul(-1/lambda)
```

比较与填充共完成 `min(1-u, 1-eps)`。融合后可在 UB 中直接执行标量上限截断，消除 bool mask 的 GM 分配、写入与读取，同时减少中间结果搬运和 Kernel 下发次数。

工程组织参考 `ops-math` 提交 `b30dd75310759813aafb6fe44bc265126e994c05` 中的 `experimental/math/log`：公开 ACLNN 接口、L0 调用、算子定义、tiling 和 Kernel 分层组织，非连续张量使用公共层的连续化与视图写回能力。

# 需求分析

## 需求描述

实现 `FusedExponential` 的 ACLNN 两段式接口，支持 FLOAT16、BFLOAT16、FLOAT32，输入输出使用 ND 格式且具有相同 shape 和 dtype。计算通过传入的 stream 异步执行，连续路径不产生中间 GM 张量，workspace 目标为 0。

## 需求拆解

| 项目 | 设计要求 |
| --- | --- |
| 数学语义 | `out = -log(min(1-u, 1-eps))/lambda`，`eps = finfo(dtype).eps/2` |
| 参数 | `u`、`out` 为 Device Tensor；`lambd` 为 Host DOUBLE Scalar |
| 数据类型 | FLOAT16、BFLOAT16、FLOAT32；输入输出类型一致 |
| shape | 同 shape，无广播；普通 Tensor、空 Tensor 与标量分别处理 |
| 布局 | 支持连续输入输出和非连续视图 |
| 内存 | 连续路径直接读 `u`、写 `out`；不落盘 bool mask |
| 执行 | 使用调用方 stream；固定输入和参数下重复执行逐位一致 |
| 精度 | 按官方 CPU golden 和逐 dtype 混合容差判断 |
| 性能 | 官方 200 条性能用例中，拼接链耗时/融合耗时不小于 1.0 |
| 环境 | A2/A3；自验证使用 910B3、CANN 9.1.0 及以上 |

# 详细设计

## 算子分析

### 数学公式

对每个有效元素：

$$
t_i=\min(1-u_i,1-\varepsilon),\qquad
\mathrm{out}_i=-\frac{\ln(t_i)}{\lambda}.
$$

`eps` 由输入 dtype 决定，计算过程中保持该定义：

| dtype | `finfo(dtype).eps` | `eps` | `1-eps` |
| --- | --- | --- | --- |
| FLOAT16 | `2^-10` | `2^-11` | `1-2^-11` |
| BFLOAT16 | `2^-7` | `2^-8` | `1-2^-8` |
| FLOAT32 | `2^-23` | `2^-24` | `1-2^-24` |

三个类型的 `1-eps` 均可表示。任务书中 BFLOAT16 的近似数值与 `dtype.epsilon()/2` 存在出入，本设计使用公式及官方 golden 的 `torch.finfo` 定义。

截断作用于 `u<=eps` 一侧：`u=0` 时使用 `1-eps`，得到小正数。`u` 接近 1 时保留 `1-u`，不增加重尾一侧的截断。有效域为 `[0,1)`，由调用方保证，ACLNN 接口不扫描 Device 输入的值域。

### 计算类型与舍入

官方自验包的 `function_FusedExponential.py` 将 `u` 和 `lambda` 转为 FP64，计算数学表达式后收窄到输出 dtype。参考 op-plugin 的低精度原地链则可能在每个小算子后舍入，两者不能据此宣称逐位相同。

融合 Kernel 拟采用 FP32 工作类型：FLOAT16/BFLOAT16 搬入 UB 后转为 FP32，完成减法、截断、自然对数和缩放，再按 round-to-nearest-even 转回原 dtype。FLOAT32 使用相同计算顺序直接输出。该选择减少低精度 `1-u` 的逐级舍入，精度结论以官方 golden 的容差测试为准。

`lambda` 保留 Host DOUBLE 校验。常规范围可用 FP32 缩放系数；当其倒数无法以有限非零 FP32 表示时，采用尾数与指数形式 `lambda=s*2^e`，计算 `(-log(t)/s)*2^-e`。指数缩放结合结果范围处理，避免先将倒数转为 0 或 Inf 而丢失本可表示的输出。实现后补充该范围的边界验证。

### 公开接口

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor *u, const aclScalar *lambd, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFusedExponential(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

第一阶段检查参数并创建执行计划，返回所需 workspace 字节数及 executor；第二阶段使用公共 `CommonOpExecutorRun` 在调用方 stream 上下发计划。

### shape 与内存别名

拟支持 rank 0–8，元素总数由 shape 计算。rank 0 是含一个元素的标量；存在零长度维度的 Tensor 才是空 Tensor。空 Tensor 返回 0 workspace 和空执行计划。

独立输出保持 `u` 内容不变。对 `u` 与 `out` 指向完全相同视图的场景，拟支持原地计算：各核先搬入自己的输入 tile，再写回同一区间。不同 tile 的有效范围互不重叠。部分重叠且视图不相同的输入输出不进入直接写出路径，其支持口径与任务书的别名要求一并评审。

## 算子实现

### Host 与接口层设计

参数检查包含指针、dtype、格式、相同 shape、rank 以及 Host Scalar 类型。非法 dtype、shape、格式或 `lambda` 返回 `ACLNN_ERR_PARAM_INVALID`；空指针使用仓库现有空指针错误码。

`lambda>0` 为合法条件，因此 0、负数和 NaN 均拒绝。`lambda=+inf` 建立置零执行计划，输出有效区域为 0。Torch 适配层沿用已有 `TORCH_CHECK(lambd>0)`；ACLNN 直接调用的校验责任见“待确认事项”。

连续输入输出直接将用户的输出 Tensor 注册为融合 Kernel 输出，L0 层不分配同 shape 的临时结果，也不添加一次多余的 `ViewCopy`。FP32 转换及运算临时量均位于 UB。

非连续路径使用公共 `l0op::Contiguous` 将输入连续化，融合计算完成后通过 `l0op::ViewCopy` 写入输出视图。连续输入但非连续输出时只准备必要的连续输出缓冲。workspace 查询包含这些缓冲，不将其计入连续路径的 0 workspace 要求。

原地非连续视图先完成输入连续化，再计算并写回原视图，避免写回覆盖尚未读取的元素。

### Tiling 设计

Host 将连续有效区域展平为长度 `N` 的一维向量，查询平台的 AIV 数量、UB 容量和搬运对齐要求。

tiling 保存 `N`、每核有效区间、tile 长度、尾块有效长度、dtype、上限常量和率参数缩放信息。工作量按完整搬运块分配，剩余元素归入尾块，核间写出边界遵守对齐要求。

tile 长度由 UB 预算决定。设 `T` 为 tile 元素数、`d` 为输入元素字节数、`Q` 为队列深度，预算包括：

```text
输入队列：Q*T*d
输出队列：Q*T*d
FP32 工作区及可同时存活的临时区
Ln/Cast API 所需临时空间及对齐开销
```

按 API 的实际临时空间需求确定 `T`，不使用固定的芯片 UB 容量。输入较小时减少启动核数和队列深度；长向量在预算允许时使用双缓冲。两种路径保持相同逐元素运算顺序。

### Kernel 设计

每个核心循环执行 `CopyIn → Compute → CopyOut`：

1. 按本核有效区间搬入 `u`，FLOAT16/BFLOAT16 转为 FP32。
2. 计算 `t=1-u`。
3. 使用 `Mins(t, 1-eps)` 完成上限截断，结果保留在 UB。
4. 使用 `Ln` 计算自然对数，并按 Host 准备的率参数信息完成负号和缩放。
5. 收窄到输出类型，写回 `out` 的对应有效区间。

`lambda=+inf` 使用置零分支，不执行对数计算。尾块按有效元素数设置向量长度，搬运采用目标产品支持的非对齐接口；补齐元素不参与有效结果，也不写到用户 Tensor 之外。

融合不产生跨核归约。每个输出元素只有一个写入者，固定输入下不同分核方式不会改变该元素的运算顺序。

### 融合依据与 workspace

`min` 等价替代比较与填充的数值结果，且无需生成 GM mask；`log` 和缩放都只依赖本元素，可接续在同一 UB tile 内完成。

连续路径的主要 GM 数据量为一次输入读取和一次输出写入，即 `2*N*d` 字节；完全原地复用时输入输出共享同一存储。该数据量说明搬运方案，不代表实测耗时。

不申请 Kernel scratch。若执行框架在目标版本返回固定系统 workspace，按实际返回值记录用途与大小，并按任务书的例外说明核对；不能把系统开销记作严格的 0 workspace 通过结果。

### 工程组织

按 `ops-math/experimental/math/FusedExponential` 的算子工程组织：

| 目录 | 内容 |
| --- | --- |
| `op_api/` | ACLNN 两段式接口与 L0 调用 |
| `op_host/` | 算子定义、shape/type 推导、tiling |
| `op_kernel/` | 融合 Kernel 与 tiling 数据定义 |
| `docs/`、`examples/` | 接口说明与两段式调用示例 |
| `tests/` | 仓库规范要求的接口及 Kernel 测试 |

Torch 接入沿用 op-plugin 的均匀采样和随机数管理入口，在 A2/A3 对应类型分支调用融合后处理。Double 与 Ascend 950 的现有分支不属于本任务适配范围。

## 支持硬件

目标为 A2/A3 系列产品。按任务书，自验证覆盖 910B3；环境使用 CANN 9.1.0 及以上，配套 op-plugin 26.1.0、torch/torch_npu 2.7.1 及以上。产品配置、向量指令及非对齐搬运支持按对应 CANN 版本核对。

## 算子约束限制

支持三种浮点类型、ND、相同 shape 和 dtype，不支持广播。调用方提供合法均匀样本并负责随机源。执行完成前保持输入、输出及 workspace 的有效性，读回结果前同步调用方 stream。

## 待确认事项

| 任务书表述差异 | 本设计采用的方案 | 评审确认点 |
| --- | --- | --- |
| 参数表将 `u` 定为只读、`out` 定为独立输出；实现约束又要求原地复用 | 同时覆盖独立输出与完全相同视图原地复用 | 是否要求其他部分重叠视图 |
| 参数表写 rank 1–8，约束与官方用例允许 0 维 | 支持 rank 0–8；标量计算一个元素，空 Tensor 不计算 | 统一 rank 与标量描述 |
| 参数表要求非法 `lambda` 报错，约束与 `facts.json` 强调 Torch 层校验 | Torch 校验保留，ACLNN 第一阶段也校验 Host DOUBLE | 是否认可直接 ACLNN 的错误码约定 |

# 可维可测分析

## 精度标准与验证

以官方自验包的 `function_FusedExponential.py` 为 golden，保留其输入值、FP64 计算和最终 dtype 收窄流程。

| dtype | rtol、atol | 匹配比例 | 最大绝对误差要求 |
| --- | --- | --- | --- |
| FLOAT16 | `2^-10` | `>=0.99` | 任务书 `1e-1` 或 `32*ULP` 口径 |
| BFLOAT16 | `2^-7` | `>=0.99` | 任务书 `1e-1` 或 `32*ULP` 口径 |
| FLOAT32 | `2^-13` | `>=0.99` | 任务书 `1e-2` 或 `32*ULP` 口径 |

逐元素判断 `abs(actual-golden)<=atol+rtol*abs(golden)`，并核对最大误差。非有限值及 ULP 判定沿用官方工具和生态精度标准。

运行官方精度全量 1000 条，非连续输入按自验指导使用 0.4 比例切片。补充接口级验证覆盖 `u=0`、`eps` 相邻值、接近 1 的可表示值、全部 dtype、`lambda=+inf` 和非法率参数、完全原地复用、非连续输出、标量、空 Tensor、非默认 stream 与重复执行确定性。

官方测试生成的重尾边界限定为 `u<=1-2^-10`，与接口 `[0,1)` 的完整值域分别记录。`known_tool_defects.json` 登记了 29 条空 Tensor 用例；仅在日志证明算子尚未调用且与已登记缺陷一致时标记工具缺陷，空 Tensor 接口行为仍单独验证。

## 性能与内存验证

运行官方 200 条性能用例，按 dtype 和规模与同型号 NPU 拼接链比较：

```text
ratio = T_chain / T_fused >= 1.0
```

拼接基线包含 `neg/add/ge/masked_fill/log/mul` 六个 Kernel；按官方脚本预热 10 次、采集 60 次，累计全部 Kernel 时间后求平均。融合路径按 ATK `performance_device` 采集一次完整调用的 Device 时间。

内存验证记录输入大小、输出大小和 `GetWorkspaceSize` 返回值，并检查执行图中是否存在中间 GM 张量。连续独立输出、连续原地和非连续视图分别记录，核对连续路径 0 workspace 目标。

## 兼容性分析

新增 ACLNN 接口，使用现有 executor、连续化、视图写回与 stream 执行能力。低精度计算与参考原地链可能存在舍入差异，按官方容差验证。Torch 接入保留 `exponential_` 的原地返回语义及 generator 行为。

## 参考资料

- [官方任务页](https://www.hiascend.com/activities/task-center/details/e179b493000a43378f416316cb544264)及其 `FusedExponential_task_doc.md`、自验包。
- [op-plugin ExponentialKernelNpuOpApi.cpp](https://gitcode.com/Ascend/op-plugin/blob/75532ae4300ea7e221cb3f89fdd503c635e4bdd8/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)。
- [ops-math Log ACLNN 实现](https://gitcode.com/cann/ops-math/blob/b30dd75310759813aafb6fe44bc265126e994c05/experimental/math/log/op_api/aclnn_log.cpp)。
- [CANN 生态精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
