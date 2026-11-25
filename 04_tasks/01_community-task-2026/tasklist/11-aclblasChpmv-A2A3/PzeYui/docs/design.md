# 需求背景

## 需求来源

2026 年 11 月社区任务「aclblasChpmv 算子开发（A2/A3）」，团队为 PzeYui。以任务书 `aclblasChpmv_AtlasA2_A3_task_doc.md` 和随附的测试用例、GPU 基线为准。[昇腾任务广场](https://www.hiascend.com/activities/task-center/details/c596a05d90b9471683e41fcf4d026ae5?menu=tasks)

## 背景介绍

CHPMV 是 BLAS Level 2 的单精度复数 Hermitian 压缩矩阵向量乘。矩阵只存上三角或下三角，调用方可通过压缩表示减少矩阵存储和传输。算子复用 `ops-blas` 的句柄式接口，在调用方绑定的 stream 上直接执行 Ascend C kernel。

在调研基线 `55ef791b77f3677b45da3468b0a8b17941424f84` 中，公共头已声明该接口，`arch35` 和 `arch22` 均已有实现。现有 `arch22` 使用 AIV 分列累积和跨核归约，对照本任务仍需修正空阶结构校验、支持设备标量，并验证短向量启动开销及长向量归约误差。[ops-blas 调研基线](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84)

# 需求分析

## 需求描述

面向 Atlas A2/A3 的 `arch22`，调用形式与 `cublasChpmv` 的参数顺序一致。复用 `include/cann_ops_blas.h` 中现有声明及库内状态码。[cuBLAS CHPMV 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublast-hpmv)

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

| 参数 | 类型与位置 | 约束 |
| --- | --- | --- |
| `handle` | `aclblasHandle_t`，Host | 有效句柄，使用其绑定的 stream |
| `uplo`、`n` | 枚举、整数，Host | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，`n ≥ 0` |
| `alpha`、`beta` | `const aclblasComplex*`，Host 或 Device | 均为非空标量指针，分别判断位置 |
| `AP` | `const aclblasComplex*`，Device | 顺序存放 `n(n+1)/2` 个复数 |
| `x`、`y` | `aclblasComplex` 向量，Device | 逻辑长度为 `n`，`x` 只读，`y` 原地更新 |
| `incx`、`incy` | 有符号整数，Host | 以复数元素为单位，允许任意非零步长 |

## 需求拆解

| 任务书条款 | 设计和验证要点 |
| --- | --- |
| §2.1–2.3 | 保留公共 ABI，覆盖上下三角、共轭补齐、实数对角和任意非零步长 |
| §2.4–2.5 | 按顺序校验非法参数，处理空阶和标量退化。在绑定 stream 上支持 Host 与 Device 标量 |
| §3.1–3.2、§3.5 | 在 Atlas 800T A2（910B3）用 Netlib CBLAS 3.10.0 对照 1000 条精度用例，按实部和虚部分量分别判断 |
| §3.3、§3.5 | 用 `msprof op` 对 200 条性能用例按 kernel task duration 与同参数 GPU 基线逐条比对 |
| §3.4 | 内存验收不涉及，验证实现中的访存边界及 workspace 使用仍属于工程自测 |

# 详细设计

## 算子分析

### 数学公式和压缩布局

对逻辑行 `i` 计算 `y_i ← α·Σ_j A_ij·x_j + β·y_i`。以 0 为起点，压缩矩阵按列存储：

| 压缩模式 | 存储下标 | 共轭规则 |
| --- | --- | --- |
| `UPPER` | `i + j(j+1)/2`，`i ≤ j` | 当 `i > j` 时，读取已存的 `A_ji` 并取共轭 |
| `LOWER` | `i + j(2n-j-1)/2`，`i ≥ j` | 当 `i < j` 时，读取已存的 `A_ji` 并取共轭 |

对角元素的虚部不参与计算，即使其为 NaN 或 Inf。复数使用仓库定义的 `aclblasComplex`，实部、虚部各为 FP32。计算以压缩索引访问矩阵，保留压缩存储的访存规模。调用方传入向量分配区首地址。逻辑索引 `k` 在正步长时访问 `k·inc`，负步长时访问 `(n-1-k)·(-inc)`，向量至少包含 `1+(n-1)·|inc|` 个复数。压缩索引和步长乘积用 64 位计算，先扩展步长再取负，涵盖 `INT_MIN`。

## 算子实现

### Host 侧设计

1. 依次检查 handle、负 `n`、`uplo`、零步长、空 `alpha/beta`。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余结构错误返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. 结构校验通过后，`n=0` 返回成功，无需检查 `AP/x/y`，不读取标量值或发射 kernel。正阶再检查 `AP/x/y` 是否为空。
3. Host 标量在下发前复制进 tiling。Device 标量指针随 kernel 下发，在绑定 stream 上读取，保留它们与同 stream 上先前写入的顺序。两类标量可以混用。
4. Host 已知 `α=0, β=1` 时直接返回。Device 标量参与判断时在 kernel 入口处理同一条件。若只有 `α=0`，只执行 `βy`，跳过 `AP/x`。所有路径的 `β=0` 分支直接产生输出，跳过旧 `y` 的读取。
5. tiling 携带 `n`、对齐后的向量长度、三角模式、两路步长、Host 标量和工作核数。通过平台信息取得 AIV 数量。二维 DMA 的步长字段使用前检查能否表示，超出时逐元素搬运。将实际 kernel 下发错误映射为执行失败状态。

### Kernel 侧设计

针对启动耗时、吞吐和长归约舍入差异设计三条计算路径。分派使用阶数及标量退化条件。在 910B3 上用精度、性能回归确定最终切换点，并覆盖切换点两侧的尺寸。

| 路径 | 分核与计算 | 存储与同步 |
| --- | --- | --- |
| 小规模单 kernel | 每个 AIV 处理一行或若干行，按 packed 索引读取矩阵并在本核完成复数点积、标量乘加 | 小矩阵和连续向量可搬入 UB。各核独占输出元素，单次下发，无跨核归约 |
| 常规规模两阶段 | 第一阶段沿 packed 列交错分核，以平衡上下三角的列长度。按行段搬入列片段，一次读取同时产生直接侧的复数 AXPY 和镜像侧的共轭点积。第二阶段按输出行段分核，合并各核部分向量并完成标量乘加 | 单核部分向量能放入 UB 时先累积至 UB，再批量写回。更大的向量按行段使用 workspace。两个 kernel 在同一 stream 内保持先写后读 |
| 长归约有序路径 | 按不相交的输出行块分给 AIV。对每行分别累计直接侧和共轭侧，在各分量内按 Netlib 列序进行 FP32 更新，保持乘积、加法和标量作用的位置 | 行块的左右矩形用 SIMD 同时更新多行。直接侧沿 packed 列搬运，共轭侧沿每个输出行对应的 packed 列搬运，再在 UB 重排。对角块按行处理并只取对角实部，最后将结果写回 |

长归约采用与 Netlib `chpmv` 一致的累加结构。以上三角为例，每行先形成低列号一侧的共轭点积，在对角更新处乘 `α` 加入输出，再顺序累积高列号一侧的 `(αx_j)A_ij`。下三角先累计低列号的直接项并处理对角，再合入高列号共轭点积。初始输出按 `β` 处理。直接侧与共轭侧各自推进，以减少改变 FP32 归约顺序造成的误差。[Netlib CHPMV 源码](https://www.netlib.org/blas/chpmv.f)

连续向量直接搬运，负单位步长按连续物理区间读写后调整顺序，常见非单位步长用二维 DMA 搬运有效元素。写回只更新逻辑输出位置，保持步长间隙不变。UB 内按对齐粒度为 planes、临时乘积、索引及搬运缓冲预留空间，使用实际有效长度处理尾块。搬运、向量和标量流水线间用 Ascend C 事件配对同步。两阶段路径通过仓内有效 workspace API 取地址和容量，每个累积核的两个 FP32 平面使用约 `8 × align_up(n, 8)` 字节，核数随容量约束。由句柄管理 workspace，其内容在当前 stream 的归约完成之前保持有效。

性能重点是减少小规模调用的 kernel 启动数、长步长的逐元素 DMA 开销和大规模路径的 HBM 往返。有序路径关注 packed 搬运的发起次数、UB 布局以及对角块的标量占比。对每个路径分别验证精度与时延后再收敛分派及分块参数。

## 支持硬件

| 产品 | 支持与验证 |
| --- | --- |
| Atlas A2 系列 | 支持。Atlas 800T A2（910B3）为自验和正式性能测试设备 |
| Atlas A3 系列 | 支持 `arch22`。任务书允许自验选一种款型，计划在 910B3 执行 |

任务书 §3.1 指定 CANN 9.1.0。开发环境使用 CANN 9.2.1。测试记录注明实际工具链版本，正式环境按社区验收时确认的版本执行。

## 算子约束限制

采用一个矩阵和一对向量，无广播、批处理或前导维度参数。除步长表达的跨距外，调用方提供相应的连续分配区。调用方保证 `y` 的可写区域与 `AP/x` 的可读区域不重叠，并在同步绑定的 stream 后读取 Device 结果。浮点归约顺序可随分派改变，验收使用规定的逐分量容差。

# 可维可测分析

## 精度标准和性能标准

| 验收项 | 方法与通过条件 | 来源 |
| --- | --- | --- |
| 精度 | 跑全量 1000 条精度用例。Netlib BLAS 3.10.0 的 `cblas_chpmv`（`CblasColMajor`）生成 golden。实部、虚部分别使用 FP32 混合容差，`atol = rtol = 2^-13`，匹配率 ≥ 99%，逐元素绝对误差上限为 `max(1e-2, 32 × ULP)` | 任务书 §3.1、§3.2、§3.5 |
| 性能 | 跑全量 200 条性能用例。预热 5 次，收集后续 10 次成功调用。单次调用有多个 kernel 时对其 task duration 求和。对每个用例的 10 次 kernel 耗时取均值，要求 ≤ 对应 GPU kernel-only 耗时的 1.25 倍 | 任务书 §3.3、§3.5，`gpu_baseline.csv` |
| 内存 | 无单独验收指标。补充验证 y 步长间隙及输出缓冲边界 | 任务书 §3.4 |

测试代码接入 `test/spmv/chpmv/arch22/` 的 GTest 框架，使用随任务提供的 CSV。覆盖上下三角、复数标量、正负和大步长、对角虚部干扰、NaN/Inf、空阶叠加非法结构参数、空句柄和其它负向用例。另测 Host/Device 混合标量、长归约路径、分派边界和非对齐尾块。日志记录逐用例通过、失败或缺失的样本。

性能对照按 `uplo,n,incx,incy` 对齐 200 条 GPU 数据，并将 `Task Duration(us)` 和 `gpu_ms` 换成相同单位。GTest 内的 `aclrtEvent` 采样供回归排查，正式性能取 `msprof op` 的 kernel task duration。

## 兼容性分析

沿用现有公共 API、复数 ABI、句柄和构建规则。实现范围位于 `blas/spmv/arch22/`，测试接入 `test/spmv/chpmv/arch22/`。上游 `arch35` 的设计和工程结构作为参考。Atlas A2/A3 的功能细节和返回状态以本任务书约定为准。
