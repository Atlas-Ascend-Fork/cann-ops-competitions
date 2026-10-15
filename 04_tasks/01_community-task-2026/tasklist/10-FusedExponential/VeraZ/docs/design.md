# 指数分布后处理融合算子开发设计文档

开发者：VeraZ。算子：FusedExponential。接口：aclnnFusedExponential。

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务“指数分布后处理融合算子开发”，使用 Ascend C 实现均匀分布样本到指数分布样本的融合变换。

参考 [op-plugin 26.1.0 ExponentialKernelNpuOpApi.cpp](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp) 中 `neg_ → add_ → ge → masked_fill_ → log_ → mul_` 的后处理过程。输入均匀随机数由调用方提供，本算子不产生随机数，也不管理随机种子。

## 背景介绍

拼接实现需要多次 kernel 启动、多次读取和写入 GM，并产生 bool mask 中间张量。融合后在一次 Ascend C kernel 内完成逐元素计算，将比较结果保留在 UB 中，消除 GM mask 及各阶段中间结果。连续输入、连续输出场景不申请计算 workspace；非连续视图使用框架连续化及回写能力保持语义。

# 需求分析（required）

## 需求描述

使用 Ascend C 将均匀样本的 neg、add、比较、masked_fill、log、mul 后处理融合为逐元素算子，提供标准两段式 aclnn 接口。覆盖 FLOAT16、BFLOAT16、FLOAT32，支持标量、空张量、原地及非连续视图；连续输入与连续输出场景不产生 GM 中间张量，workspace 为 0。在 A2/A3 上保持公式及边界语义，性能不低于同设备的 NPU 拼接链。

## 需求拆解

| 需求 | 实现措施 | 验证方式 |
| --- | --- | --- |
| 数值及边界一致 | dtype 对应 eps、FLOAT32 内部计算、小输入多项式、DOUBLE lambd 拆分 | 三种 dtype 的 CPU golden、极端输入及非法参数检查 |
| 消除 GM mask | 比较和选择在 UB 完成，逐 tile 读写输入输出 | 源码核对、真实 workspace 和内存测量 |
| 标量和空张量 | rank 0 进入正常计算；含零长度轴走空执行器 | 标量、全部 29 个空 shape 和真实接口调用检查 |
| 视图和原地 | Contiguous 归一化、连续输出直写、非连续 ViewCopy | 非连续输入/输出、连续及非连续原地测试 |
| 多核及尾部安全 | 256 元素核边界、平台 UB 约束 tile、DataCopyPad | 非对齐长度、多核区间及确定性检查 |
| 性能不劣化 | 单 kernel 融合、双缓冲、按规模选择核数 | 200 条性能用例逐项 ratio>=1 |

# 详细设计（required）

## 算子分析

### 数学公式

对每个输入元素执行：

```text
eps = finfo(dtype).eps / 2
out[i] = -log(min(1 - u[i], 1 - eps)) / lambd
```

等价计算形式为 `x = max(u, eps)`、`out = -log(1-x)/lambd`。保护项只作用于小输入侧；接近 1 的重尾侧不设上限。

### 支持数据类型

| dtype | eps | 内部计算类型 |
| --- | --- | --- |
| FLOAT16 | 2^-11 | FLOAT32 |
| BFLOAT16 | 2^-8 | FLOAT32 |
| FLOAT32 | 2^-24 | FLOAT32 |

BFLOAT16 的 `finfo.eps` 为 2^-7，因而本设计使用 2^-8，与任务书中 `finfo(dtype).eps/2` 的公式及 golden 一致。

### 接口定义与参数

```cpp
aclnnStatus aclnnFusedExponentialGetWorkspaceSize(
    const aclTensor *u, const aclScalar *lambd, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFusedExponential(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

| 参数 | 约束及语义 |
| --- | --- |
| u | Device tensor；FLOAT16/BFLOAT16/FLOAT32；ND；秩 0～8。调用方保证采样域为 [0,1)。 |
| lambd | Host DOUBLE scalar；有限正数或 +inf；<=0、NaN、非法标量类型返回参数错误。 |
| out | 与 u 的 shape、dtype 一致；支持独立输出、连续原地、非连续输入及输出视图。 |
| workspaceSize / executor | 第一段返回所需空间及执行计划；空张量返回零 workspace 执行计划。 |
| stream | 第二段在调用方 stream 上异步执行，读取输出前由调用方同步。 |

### 支持形状

输入、输出均为 ND，shape 相同，秩 0～8。不支持广播；不要求编译期未知 shape。0 维 tensor 是含一个元素的标量，正常计算；任一维长度为 0 才是空 tensor，输出保留原 shape。逐元素计算在连续化后只依赖元素总数 N，无需在 kernel 内重新解析各维索引。

### 边界行为

接口层不扫描 `u` 的值域。`u<0` 时按公式落到小输入侧保护值，`u==1` 时有限正 lambd 对应 +inf，`u>1` 时对应 NaN；NaN 沿浮点计算传播。有效输入的结果超出输出 dtype 表示上限时允许 +inf。`lambd==+inf` 使用全零模式，跳过输入读取及对数计算，保证输出为 0。

## 算子实现

### 实现方案与工程分层

| 层次 | 职责 |
| --- | --- |
| L2 op_api | 参数校验、DOUBLE lambd 拆分、执行器创建、空张量处理、连续化及输出回写。 |
| L0 launcher | 注册融合算子，传递输入、输出及 scale1/scale2/mode 属性。 |
| op_host | shape/dtype 推导、平台 AIV 核数和 UB 查询、分核及 tile 参数生成。 |
| op_kernel | DataCopyPad 搬运，UB 内转换、比较、对数与缩放，输出转换并写回。 |

实现位于 `experimental/math/FusedExponential/`，使用标准两段式 aclnn 工程。Tiling key 按输入 dtype 选择模板，缩放模式由 tiling 数据传入。

### Host/Tiling 设计（host侧设计）

#### 参数校验和 Tiling 数据

L2 首先检查 tensor/scalar、workspaceSize/executor 指针、输入输出类型及 shape、最大秩、ND 格式和 DOUBLE lambd 的正数约束。空张量在创建执行器后直接返回，避免启动 kernel。Host 获取平台信息与输入元素数，将 `totalElements`、`blockElements`、`tileElements`、`scale1`、`scale2`、`mode` 写入 TilingData；kernel 根据这些字段计算本核起点、有效区间和缩放方式。

#### 分核策略

算子没有归约或广播，将连续化后的输入视为 N 个元素的一维向量。读取平台 AIV 核数 C，核数选择为 `min(C, max(1, ceil(N/1024)))`。每核区间长度为 `ceil(ceil(N/blockDim)/256)*256`，核间区间不重叠；实际有效元素数受 N 限制，多出的空区间不访问数据。

#### 数据分块和 UB 内存策略

每次最多处理 4096 个元素，tile 为 256 元素整数倍，且受平台 UB 容量约束。输入、输出各使用两级队列，另分配三个 FLOAT32 临时向量和 UB 比较缓冲区。每元素按 `4*sizeof(dtype)+3*sizeof(float)+1` 字节保守预算，预留 8 KiB，再向下对齐确定 tile。比较缓冲区实际按位存放，预算按每元素一字节留出余量。

#### TilingKey 规划

TilingKey 根据输入 dtype 选择 FLOAT16/BFLOAT16/FLOAT32 模板实例，三种类型共用分核算法。`mode` 作为 tiling 字段选择普通双因子缩放、+inf lambd 全零、极端有限 lambd 溢出或下溢路径；连续化与输出回写由 L2 执行器处理，不增加 kernel 的 stride 分支。

### Kernel 设计（kernel侧设计）

Init 阶段根据 TilingData 确定本核 GM 区间并初始化 TPipe、输入输出双缓冲队列及临时 TBuf。Process 阶段以 tile 为单位循环搬运、计算和回写；末块使用实际有效长度，零有效长度的核直接返回。

每核独立执行 `CopyIn → Compute → CopyOut`：

1. 按有效元素数使用 DataCopyPad 将输入搬到 UB，尾部按实际字节数搬运。
2. FLOAT16/BFLOAT16 转为 FLOAT32，FLOAT32 保持原计算类型。
3. 以对应 dtype 的 eps 计算 `x=max(u,eps)`，在 UB 内完成比较及选择。
4. 计算 `-log(1-x)`。FLOAT32 小输入区间使用下述多项式，其余使用向量 Ln。
5. 按 host 给出的模式及两个缩放因子计算除以 lambd 的结果。
6. 转换为输出 dtype，按有效字节数写回 GM。

向量计算长度在已分配 UB tile 内对齐至 64 元素，无效尾部不写回 GM。不同核无数据依赖，无需核间同步。没有随机数、原子操作或并行归约，相同输入和运行环境下结果确定。

### 数值稳定性

FLOAT16/BFLOAT16 先提升到 FLOAT32，避免逐级低精度舍入，最终仅收窄一次。精度标杆在 CPU FLOAT64 中执行公式并最终转换输出 dtype。

FLOAT32 在 `x<=1/64` 的区域采用：

```text
-log(1-x) ≈ x * (1 + x * (1/2 + x * (1/3 + x/4)))
```

避免 `1-x` 消减误差。级数截断相对误差上界小于 `x^4/(5*(1-x))`，在该区间约为 1.3e-8；该界不包含 FLOAT32 运算舍入，实际精度以逐元素测试判定。其他输入使用向量 Ln，小输入路径和 Ln 路径均在 UB 内完成选择。

DOUBLE lambd 在 host 校验，避免先转 FLOAT32 将合法有限正数变成 0 或 +inf。通过 `frexp` 得到尾数及二进制指数，再拆成两个有限正常 FLOAT32 缩放因子，kernel 分两次乘法完成缩放。极端指数走溢出/下溢模式，并保留 `u==1` 的无穷及 `u>1` 的 NaN；+inf lambd 使用独立全零模式。

### 连续化、原地及 workspace

连续输入直接使用原 tensor；非连续输入使用框架 Contiguous 归一化。连续输出直接作为 launcher 输出，非连续输出分配连续临时结果，再通过 ViewCopy 写入原视图。非连续原地输入先完成读取及连续化，再执行和回写；连续原地每个 tile 在覆盖自身输入前完成搬入。

连续输入且连续输出时，融合 kernel 不申请 GM 中间张量，不产生 GM bool mask，workspace 返回 0。非连续路径允许 Contiguous/ViewCopy 所需的框架临时空间，应与融合计算 workspace 区分并实际测量。

Tiling 使用 `GetWorkspaceSizes(0)` 声明零个 workspace 槽，避免“一个大小为 0 的槽”触发最小对齐块分配。CANN 生成 wrapper 保留 `u,out,workspace,tiling` 四地址 ABI，L0 通过 OP_WORKSPACE 传入零元素 host 描述符作为未访问的地址占位，并设置 `SetWorkspaceOffset(UINT64_MAX)`。描述符由执行器管理，无 GM 存储；kernel 不访问该地址。该兼容处理不修改生成代码，已在 A2/A3 上验证真实 workspace 返回值。

### 融合依据

原链的算术、比较及选择均为逐元素操作，可以在同一个 tile 中完成，无跨元素依赖。融合消除阶段之间的 GM 读写及 mask，不引入归约或额外 GM 暂存。保留框架视图转换，使连续快速路径和非连续语义均可实现。

## 支持硬件

| 支持的芯片系列 | 适配范围 | 已验证型号 |
| --- | --- | --- |
| Atlas A2 | Ascend C AIV、ND 三种浮点类型 | 910B3 |
| Atlas A3 | Ascend C AIV、ND 三种浮点类型 | Ascend910_9392 |

目标 CANN 9.1.0 及以上；A2 实测 CANN 9.1.0，A3 实测 CANN 9.2.0-beta.2。Host 动态查询核数和 UB，不能把某一型号容量硬编码到 Tiling 中。

## 算子约束限制

仅支持 FLOAT16/BFLOAT16/FLOAT32 的 ND tensor，输入输出 shape 和 dtype 必须一致，秩不超过 8，不支持广播。lambd 必须为 Host DOUBLE 且大于 0，允许 +inf。输入合法采样域由调用方保证，接口不进行 Device 值域扫描。异步执行的输出读回必须由调用方同步 stream。

零 workspace 要求适用于连续输入、连续输出路径，非连续归一化及回写可以使用框架临时空间。CANN wrapper 与执行器相关接口需随目标版本复核，不能仅凭 kernel 未访问 workspace 就推断整条调用链零分配。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判据 | 标准来源 |
| --- | --- | --- |
| 精度 | 各 dtype 的 rtol/atol、匹配率和最大误差限制同时满足，Inf/NaN 一致 | 任务书 §3.2 及生态精度标准 |
| 性能 | 同设备、同规模、同 dtype 的拼接/融合耗时 ratio>=1，耗时有限且大于 0 | 任务书 §3.3 |
| 内存 | 连续场景无中间 GM 张量，实际 workspaceSize=0 | 任务书 §3.4 |
| 功能 | shape/dtype、标量、空张量、视图、原地、非法参数及异步执行符合接口定义 | 任务书 §2.3～§2.5 |

### 精度与功能测试

冻结任务自验包 1000 个用例的 shape、dtype、lambd、ID 及种子，非连续输入比例为 0.4。CPU golden 在对应 torch 环境生成，使用 FLOAT64 五算子链并最终收窄。按 [生态精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) 及任务书逐 dtype 阈值判断：

| dtype | rtol / atol | required_matched_ratio | 最大绝对误差限制 |
| --- | --- | --- | --- |
| FLOAT16 | 2^-10 / 2^-10 | >=0.99 | 0.1 或 32 ULP |
| BFLOAT16 | 2^-7 / 2^-7 | >=0.99 | 0.1 或 32 ULP |
| FLOAT32 | 2^-13 / 2^-13 | >=0.99 | 0.01 或 32 ULP |

逐元素检查 `abs(actual-golden)<=atol+rtol*abs(golden)`，并核对最大误差限制及 Inf/NaN。补充标量、空张量、非对齐尾部、原地、非连续输入/输出、无效参数、极端 lambd、确定性及实际 workspace 测试。

### 性能与内存测试

性能覆盖任务 200 条非空非标量子集，逐例要求 `ratio=拼接链耗时/融合耗时>=1`，耗时必须为有限正数。基线在同型号 NPU 使用 neg/add/ge/masked_fill/log/mul，共 6 个 kernel；Profiler Level1 窗口外预热 10 次，窗口内采集 60 次，整张 `op_statistic.csv` 的 Total Time(us) 求和后除以 60。融合侧使用 ATK performance_device 的 Device 耗时。

内存测量输入、输出和第一段返回的 workspace，重点检查连续场景 workspace=0，并结合源码确认无 GM mask 或其他中间张量。非连续路径独立记录框架临时空间。

### 已完成验证

验证对象为 [算子源码 4a7bdbc](https://gitcode.com/VeraZ/ops-math/commit/4a7bdbcd3e32a490328c478d0077a34e414fd640)。A2 环境为 910B3、CANN 9.1.0、torch 2.7.1；A3 环境为 Ascend910_9392、CANN 9.2.0-beta.2、torch 2.9.0。

| 项目 | 实测结果 |
| --- | --- |
| A2/A3 直接两段式 ACLNN 功能 | 各 95/95 通过。 |
| A3 完整严格精度 | 1000/1000 通过，含全部 29 个空张量；2000 次实际两段式 API 返回码均为 0。 |
| A2 性能 | 200/200 通过；拼接/融合倍率最小 1.807800、中位数 4.558008、最大 8.441278。 |
| A2 内存 | 12/12 通过，连续 6 项实际 workspace 均为 0。 |

初轮 A2 中 4 个空张量在 ATK 切片构造阶段失败，原因是零长度轴被错误构造成长度 1。修复保留零长度切片并避免空数据 copy 后，A3 上原四项、72 项定向 NPU 回归和完整 1000 项均通过，CPU 相关回归 173/173 通过。修复见 [ATK PR !49](https://gitcode.com/Ascend/ATK/merge_requests/49)，尚待上游合并。A2 未进行修复后完整精度补跑；表中 A2 性能、内存保留原始实测，算子源码及二进制未因工具修复而修改。

## 兼容性分析

新增独立接口，不改变已有接口。类型选择、分核、缩放模式及视图处理分别落在对应工程层；测试同时检查数值、布局和真实接口返回值，便于定位框架适配、工具造数与 kernel 计算问题。
