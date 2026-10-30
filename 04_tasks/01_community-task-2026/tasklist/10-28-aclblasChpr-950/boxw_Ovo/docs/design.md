# 需求背景（required）

## 需求来源

[10月社区任务-aclblasChpr算子开发（950）](https://www.hiascend.com/activities/task-center/details/f783b3800ad744ef8af7f50ba237de04)，以随任务发布的《aclblasChpr 算子开发任务书》及 `test_cases` 为功能、测试和交付依据。

本文按 [CANN 社区任务设计模板](https://gitcode.com/cann/cann-competitions/blob/e1ab79ca0d495afce93dfb1e85f95f546b74d2cf/04_tasks/01_community-task-2026/resources/design_template.md)编写，描述 Ascend 950PR、CANN 9.1.0 下的实现与验证方案。

## 背景介绍

`aclblasChpr` 对以 packed 格式存储的 Hermitian 矩阵执行单精度复数秩一更新。packed 格式仅保存一个三角区，将矩阵存储量由 `n*n` 个复数降为 `n*(n+1)/2` 个复数。

ops-blas 已提供公共 `aclblasChpr` 声明和其他产品线实现。本任务沿用同一 handle、stream 和接口，在 `blas/hpr/arch35/` 补齐 Ascend 950PR 实现，对应测试放在 `test/hpr/chpr/arch35/`。

# 需求分析（required）

## 需求描述

使用 Ascend C kernel 直调方式实现：

```cpp
aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* ap);
```

接口复用 `include/cann_ops_blas.h` 中的现有声明，参数顺序与 cuBLAS `cublasChpr` 对应。`alpha` 为 Host 端 FP32 实数标量，`x` 和 `ap` 为 Device 端 complex64 数组；结果原地写回 `ap`。

## 需求拆解

| 项目 | 设计要求 |
| --- | --- |
| 功能 | `A := alpha*x*x^H + A`；仅引用和更新 `uplo` 指定的 packed 三角区 |
| 类型 | `alpha` 为实数 FLOAT32；`x/ap` 为 `aclblasComplex`，实部、虚部各 FLOAT32 |
| 形状与步长 | `n >= 0`；`incx != 0 && incx != INT_MIN`；支持正、负及非单位步长 |
| Hermitian 语义 | 非零更新后对角虚部置零；`n=0` 和通过参数校验的 `alpha=0` 保持 `ap` 不变 |
| 错误语义 | 空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法 `uplo/n/incx` 或必需指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 工程 | 公共 handle 绑定 stream；复用 Hpr 算子族及仓库构建、GTest、CSV 和 CBLAS 框架 |
| 精度 | 完整 packed 输出的实部、虚部分别按任务书 FLOAT32 标准判定，golden 为 Netlib BLAS 3.10.0 `cblas_chpr` |
| 性能 | 覆盖随包 200 条性能用例；NPU 平均单次耗时不超过 H100 对应基线的 2.5 倍 |
| 验证与交付 | 覆盖随包全部 1200 条用例，提供测试步骤、自测报告及精度、性能原始结果；设计通过评审并合入指定设计仓 |

# 详细设计（required）

## 算子分析

### 数学公式

对于被存储的元素 `A(i,j)`：

```text
A(i,j) := A(i,j) + alpha*x(i)*conj(x(j))
```

令 `x(i)=r_i+i*s_i`、`x(j)=r_j+i*s_j`，并令 `a=alpha*r_j`、`b=alpha*s_j`：

```text
delta_real = r_i*a + s_i*b
delta_imag = s_i*a - r_i*b
ap.real   += delta_real
ap.imag   += delta_imag
```

各复数乘积分量先完成合并，再与旧 `ap` 相加，以保留乘积相互抵消时的原值。计算使用 FP32 中间结果；对角线按 Hermitian 语义将输出虚部置为零。

### 支持数据类型

| 参数 | 数据类型 | 存放位置 | 说明 |
| --- | --- | --- | --- |
| `handle` | `aclblasHandle_t` | Host | 已创建的上下文句柄，携带 stream |
| `uplo` | `aclblasFillMode_t` | Host | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| `n`、`incx` | `int` | Host | 阶数和复数元素步长 |
| `alpha` | `const float*` | Host | 指向 FP32 实数标量 |
| `x` | `const aclblasComplex*` | Device | 只读 complex64 向量 |
| `ap` | `aclblasComplex*` | Device | complex64 packed 矩阵，原地更新 |

复数布局使用 `include/cann_ops_blas_common.h` 定义的 `aclblasComplex`。计算过程中保留 FP32 精度。

### 支持形状

`x` 的逻辑长度为 `n`，物理存储长度至少为 `1+(n-1)*abs(incx)`；`ap` 长度为 `P=n*(n+1)/2`。`n=0` 为合法 quick return。

采用从零开始的列主序 packed 索引：

| `uplo` | 行范围 | packed 索引 |
| --- | --- | --- |
| `ACLBLAS_UPPER` | `0 <= i <= j < n` | `i+j*(j+1)/2` |
| `ACLBLAS_LOWER` | `0 <= j <= i < n` | `i+(2*n-j-1)*j/2` |

正步长的逻辑元素 `x(k)` 位于 `k*incx`；负步长时位于 `(n-1-k)*abs(incx)`，与 Netlib 从数组物理起点解释负步长的约定一致。地址乘积和 packed 偏移使用 64 位整数计算。

随包精度用例覆盖 `n=0..2048` 的基础、非对齐及边界形状，并包含负维度等非法参数用例；200 条性能用例的 `n=1..4096`，均为 `incx=1`。这些范围用于说明已给出的测试集合，接口仍按任务书支持合法的运行时 `n/incx`。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

Host 完成参数校验、核数选择、tiling 参数装配及 kernel 下发。

参数处理顺序如下：

1. 检查 handle；为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n<0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n=0` 直接返回成功，不再校验 `uplo/incx/alpha/x/ap`。
3. `n>0` 时，依次检查 `uplo`、`incx`、`alpha/x/ap`。`uplo` 仅接受 UPPER/LOWER；`incx` 排除 0 和 INT_MIN；三个指针均须非空。任一失败返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 完成上述校验后读取 `alpha`；值为零时直接返回成功，保持 `ap` 原值。
5. 获取公共平台接口提供的 AIV 核数，装配 tiling，通过 handle 绑定的 stream 下发 kernel。

##### 1. 分核策略：

按列配对：列 `j` 与列 `n-1-j` 组成一对。上下三角的每个完整列对均包含 `n+1` 个 packed 元素，可减小三角区列长差异造成的负载不均。

令 `pairs=ceil(n/2)`，使用核数 `C=min(pairs,AIV核数)`。核 `b` 处理 `b,b+C,b+2C,...` 对列。奇数阶的中间列只处理一次。各核写入互不相交的列区间，无跨核归约和原子更新。

##### 2. 数据分块和内存优化策略：

列内按 1024 个复数划分 tile，末块按实际元素数搬运和写回。单 tile 有效数据为 8192 字节，分别设置 X、AP 输入和 AP 输出队列，每队列两份 buffer，常规 UB payload 为 48 KiB。

在 `abs(incx)=1`、`n<=4096` 且 `n/2>=C` 时，每核预载一次完整 X，随后复用行向量数据；列标量继续从 GM 读取并在整列内复用。该路径的 UB payload 为 `2*(8*n+16384)` 字节，`n=4096` 时为 96 KiB，实际分配还需计入 SDK 对齐及队列管理开销。

其他步长按 tile 搬运，通过公开 Compact DMA 将间隔存储的复数压紧到 UB。负步长先按物理地址递增搬入，再按逻辑行序取数。

##### 3. tilingkey规划策略：

该算子采用公共 BLAS 接口直调，不引入额外的算子注册入口或 tiling key。Host 向 kernel 传递 `n`、`useCoreNum`、`uplo`、`alpha` 和 64 位 `incx`，由这些字段选择三角区、步长方向及 X 复用路径。Host/Kernel 共用 tiling 定义并检查结构布局。

#### 3.2.2 kernel侧设计：

kernel 分为 Init 和 Process。Init 绑定 X/AP 全局地址并初始化队列；Process 按列对执行各列的 CopyIn、Compute、CopyOut。

| 阶段 | 处理方式 |
| --- | --- |
| CopyIn | 根据 packed 列起点读取 AP；X 使用完整行缓存或当前 tile，负步长保持复数实虚部配对 |
| Compute | 使用 Ascend C SIMD 解交织 FP32 实虚部，复用列标量计算复数乘积，再更新 AP；对角虚部置零 |
| CopyOut | 按当前 tile 的有效字节数回写原 packed 区间，尾块通过 mask 控制有效元素 |

任务书中的零向量、Inf/NaN 和 `alpha=0` 语义分别验证。对 `x(j)=0` 的列，遵循 Netlib 对该列非对角元素保持原值、对角虚部清零的处理，避免零乘非有限值改变无需更新的元素。`alpha=0` 在 Host 返回，整个 AP 保持不变。

队列管理数据搬运与向量计算之间的依赖。数据块和末块共享同一计算流程，使用公共 Ascend C API 完成搬运、Gather、复数分量更新及写回。

#### 3.2.3 性能成本与优化方案：

令 `P=n*(n+1)/2`。AP 读写的有效请求量为 `16P` 字节；不缓存 X 时，行数据为 `8P` 字节，列标量为 `8n` 字节，总有效请求量约 `24P+8n`。缓存行 X 后，总量约为 `16P+8nC+8n`。有效计算量约为 `8P+2n` FLOP，因而重点是连续搬运、输入复用和核间负载分配。

优化按以下顺序验证：

1. 保持 AP packed 连续读写和列标量复用，减少重复访问与尾块开销。
2. 对性能用例的单位步长路径验证完整 X 复用收益，结合实际 UB 占用选择分块参数。
3. 统计各核元素数、tile 数、向量尾部及列控制开销，针对存在明确差距的官方用例调整核数或分配策略。

上述字节数是有效请求量，实际 HBM 流量还受缓存命中影响。性能结论以正式用例的 msprof 结果为依据，同时保留相同输入下的精度回归。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | CANN 版本 |
| --- | --- | --- |
| Ascend 950PR（arch35） | √ | 9.1.0 |

## 算子约束限制

| 约束 | 说明 |
| --- | --- |
| 输入有效性 | 调用方提供有效 handle，以及满足长度要求的 Device X/AP 存储 |
| 标量位置 | `alpha` 位于 Host，类型为 FP32 实数 |
| 存储 | AP 为列主序 packed 存储，无 `lda`；仅指定三角区参与更新 |
| 步长 | `incx` 以复数元素为单位，允许正负非零值，INT_MIN 非法 |
| 原地更新 | AP 原地更新；X 为只读输入 |
| 空输入 | `n=0` 直接成功；`n>0 && alpha=0` 校验后成功，均不修改 AP |
| 异步执行 | 使用 handle 绑定的 stream；读取 Device 结果前同步该 stream |
| 其他能力 | 不涉及广播或超出 `incx` 语义的非连续 Tensor；不要求确定性计算 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| golden | Netlib BLAS 3.10.0 `cblas_chpr`，全量比较 AP 实部、虚部 | 任务书 §3.1、§3.2 |
| 有限值精度 | `atol=rtol=2^-13`；逐分量误差不超过 `atol+rtol*abs(golden)` 的比例至少 0.99；并检查 `1e-2` 或 `32*ULP` 的绝对误差上限 | 任务书 §3.2及其引用的生态精度标准 |
| 非有限值精度 | NaN/NaN、同号 Inf/Inf、NaN/任意符号 Inf 等效；有限值与非有限值、异号 Inf 判为不通过；单独统计，全部满足特殊值规则 | 任务书 §3.2 |
| 性能 | `NPU平均耗时(us) <= gpu_ms*1000/0.4`，逐条比对全部 200 条 H100 基线 | 任务书 §3.3、随包 `gpu_baseline.csv` |
| 采样边界 | msprof op 采集本算子的 `Task Duration(us)`；预热和有效样本分开，采用算术平均 | 任务书 §3.3及随包测试指导 |

精度在算子测试范围内应用任务规定的阈值及特殊值规则。实部、虚部分别统计有限值元素数、匹配比例、最大绝对误差和非有限值通过数；非有限值不进入有限值统计。没有有限值时，仅由全部非有限值的判定决定该分量结果。CSV 中 MERE/MARE 列保留为测试框架配置，与固定的生态精度指标分别报告。

绝对误差上限按任务书引用的[生态精度标准 §2.1.2](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)计算。实部、虚部分别在双方均为有限值的比对点中选取最大绝对误差点 `i_max`，将该点 golden 按实际输出 dtype 收窄为 `g_low=FP32(golden[i_max])`，再检查：

```text
max_abs_error <= max(1e-2, 32*ULP(g_low))
```

因此，ULP 上限锚定于最大绝对误差点的 golden 值域；非有限值则采用本任务书明确规定的等效规则单独判定。

### 测试覆盖

通过仓库 GTest+CSV 工程加载随包全部 1200 条参数，明确划分 1000 条精度用例和 200 条性能用例。测试工程适配随包的 `x_fill/ap_fill` 列、标量 `null`、张量 `NULLPTR` 及非法枚举值，按原始 case ID 核对条数和执行结果。

基础和精度用例覆盖上下三角、尺寸扫描、非对齐尾块、正负步长、非单位步长、alpha 特殊值、Inf/NaN 和参数错误。正常 AP 输入按 Hermitian 约束生成，对角虚部为零。另验证 `alpha=0` 保留 AP 原值，以及非零更新清除对角虚部的语义。

随包参数中的固定 alpha、特殊值和负向用例按原参数执行。对于随机类精度用例，相同参数分别生成均匀、正态两组 X/AP 数据，两种分布各占 50%，实部、虚部独立采样。均匀分布为 `[-5,5]`；正态分布的 `mu` 取自 `[-5,5]`、`sigma` 取自 `[0.1,2]`。

另外通过上下三角、`incx=±1/±2/±3` 与两种分布的组合增加 24 条用例，覆盖随机 alpha 与 X/AP 的组合：其中均匀和正态各 12 条，alpha 独立采样。保留独立的空 handle、复数抵消和对角清零回归。报告分别列出 CSV 参数条数与实际数据批次，记录生成器、seed 和输入摘要，区分官方提供的参数与据此生成的输入数据。

随包 `RANDOM_NORM_5_5` 标记的说明值域为 `[-5,5]`，测试适配按该值域生成均匀分布；正态覆盖作为补充用例提供，原始官方资产保持不变。

### 性能验证

200 条性能用例均为 `incx=1`。其中四组典型形状的 H100 基线及换算门槛如下：

| `uplo` | `n` | H100 `gpu_ms` | NPU 平均耗时上限（us） |
| --- | ---: | ---: | ---: |
| UPPER | 512 | 0.011909 | 29.7725 |
| LOWER | 1024 | 0.030999 | 77.4975 |
| UPPER | 2048 | 0.096491 | 241.2275 |
| LOWER | 4096 | 0.570764 | 1426.9100 |

逐条核对性能参数与基线对应关系；相同形状的不同 case 仍独立保留。每次测量前恢复初始 AP，分配、初始化、传输、golden 和结果比对均在算子 task duration 之外。只有接口返回 SUCCESS 的调用计入有效样本。

任务书 §3.3 指定预热 5 次后取 10 个有效样本，§7 写为有效采样大于 10 次，随包 README 写为大于 50 次。为保留各口径，测试方案采集 51 个有效样本，同时报告前 10 次均值和全部 51 次均值；最终验收采用官方确认的采样口径。报告保留全部原始样本、参数、环境版本及异常说明。

### 可维护性与交付

Host 参数、tiling、kernel 和测试职责分别组织，复用仓库构建入口与统一测试主程序。正式自测检查全部预期 case 的发现、执行和结果，报告缺失、跳过及失败项。

按任务书提交设计评审，提供算子 README、测试代码及步骤、自测报告、实部/虚部精度和性能截图与日志。实现文件位于 `blas/hpr/arch35/`，测试位于 `test/hpr/chpr/arch35/`；Hpr README 的产品支持表标注 Ascend 950PR。

## 兼容性分析

继续使用现有 `aclblasChpr` 公共声明、复数结构、状态码和 handle/stream 生命周期。新增 arch35 实现与对应测试，保持其他产品线实现、公共框架和已有接口兼容。任务针对 Ascend 950PR，其他产品的支持情况沿用各自实现与验证结论。

本文为设计方案；功能、精度与性能结果在 CANN 9.1.0、Ascend 950PR 的完整自测后单独提供。
