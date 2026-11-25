# aclblasSgemm、aclblasCgemm、aclblasHgemm 算子设计文档

# 需求背景（required）

## 需求来源

本设计对应 11 月社区任务“gemm 类 3 个算子开发（A2/A3）”，面向 Atlas A2、A3 提供单精度实数、单精度复数和半精度实数通用矩阵乘接口。

## 背景介绍

ops-blas 的 arch22 已提供 Sgemm 入口及通用 GEMM 测试工程。本次在同一工程中完善 Sgemm 的验证，补齐 Cgemm、Hgemm 的 arch22 实现，并新增 Hgemm 公共接口声明。调用方通过同一套 BLAS 参数描述矩阵布局、转置方式和缩放系数。

# 需求分析（required）

## 需求描述

三个接口均计算一次矩阵乘加，将结果写回调用方提供的矩阵 C。矩阵存放在 Device，标量 `alpha`、`beta` 由 Host 指针传入。矩阵元素和标量采用同一接口对应的数据类型。

| 接口 | 矩阵及标量类型 | 转置语义 |
| --- | --- | --- |
| `aclblasSgemm` | FLOAT32 | N 为原矩阵，T 和 C 均为转置 |
| `aclblasCgemm` | COMPLEX64，实部和虚部各为 FLOAT32 | N 为原矩阵，T 为转置，C 为共轭转置 |
| `aclblasHgemm` | FLOAT16，按 IEEE FP16 表示 | N 为原矩阵，T 和 C 均为转置 |

## 需求拆解

1. 统一三个接口的参数校验、列主序寻址、空矩阵返回和标量缩放规则。
2. 分别支持 FP32 乘加、直接复数乘加和 FP16 输入输出，覆盖 A、B 两个转置参数的九种组合。
3. 接入 arch22 构建和通用 GTest 工程，对三个接口分别验证精度、边界行为及性能。

# 详细设计（required）

## 算子分析

### 数学公式

`C ← alpha × op(A) × op(B) + beta × C`，其中 `op(A)` 为 m×k，`op(B)` 为 k×n，输出 C 为 m×n。Cgemm 使用直接复数乘法：两个复数的乘积为 `(ar×br − ai×bi) + i×(ar×bi + ai×br)`。操作数指定共轭转置时，交换行列并对虚部取反。

### 支持形状与存储布局

矩阵按列主序存储，前导维度以元素为单位，表示相邻两列起点之间的距离。三个尺寸 m、n、k 均为非负整数。

| 矩阵 | 存储列数 | 前导维度下限 |
| --- | --- | --- |
| A | N 时为 k，T/C 时为 m | N 时 `lda ≥ max(1,m)`，T/C 时 `lda ≥ max(1,k)` |
| B | N 时为 n，T/C 时为 k | N 时 `ldb ≥ max(1,k)`，T/C 时 `ldb ≥ max(1,n)` |
| C | n | `ldc ≥ max(1,m)` |

## 算子实现

### Host 侧设计

按句柄、转置枚举、矩阵尺寸、前导维度、指针的顺序校验参数。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法转置枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法尺寸、前导维度或必需指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。结构参数合法后，按以下顺序确定执行路径。

| 条件 | 行为 |
| --- | --- |
| m=0 或 n=0 | 返回成功，允许 A、B、C 为空，不启动 Kernel。 |
| 输出非空，且 k=0 或 alpha=0 | 只计算 `C ← beta × C`，A、B 可为空。beta=0 时直接置零，beta=1 时保留原值。 |
| 其余情况 | 根据类型、矩阵规模及转置方式组织分块，在句柄绑定的 stream 上异步执行。beta=0 时直接写乘积的缩放结果，不读取 C 旧值。 |

### Kernel 侧设计

按 C 的有效行列划分输出块，依据转置方式和前导维度读取 A、B，沿 k 维累加。Sgemm 执行单精度实数乘加；Cgemm 分别累加实部与虚部，并执行复数标量缩放；Hgemm 采用 FP32 累加，最终按 FP16 写回。各输出块只写有效元素，尾块和每列末尾的填充区分别处理。

## 支持硬件

目标平台为 Atlas A2、Atlas A3 系列产品，软件环境为 CANN 9.1.0，性能验证设备为 Atlas 800T A2（910B3）。

## 算子约束限制

接口处理单组列主序矩阵，支持由前导维度描述的填充布局。调用方保证 C 的写入区域与 A、B 的读取区域互不重叠，并在读取结果前同步句柄绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验证项 | 设计与通过条件 |
| --- | --- |
| 参考结果 | Sgemm、Cgemm 按 Netlib BLAS 3.10.0 的 SGEMM、CGEMM 语义生成 CPU 参考值。Hgemm 将 FP16 输入及标量转为 FP64 累加，再依次收窄至 FP32、FP16，最终按 FP16 输出比对。 |
| 精度 | 有限值按混合容差逐元素比对。Sgemm 和 Cgemm 的 rtol、atol 均为 2⁻¹³，Cgemm 的实部、虚部分别判定；Hgemm 均为 2⁻¹⁰。每个用例要求匹配比例至少 0.99，同时满足最大绝对误差限值：FP32 档为 10⁻² 或 32 ULP，FP16 档为 10⁻¹ 或 32 ULP。 |
| 性能 | 每个用例预热 5 次后，用 `msprof op` 采集连续 10 次有效 Kernel Task Duration，取平均并统一换算为毫秒。每条用例要求 NPU 平均耗时 ≤ 同参数 GPU 基线耗时的 1.25 倍。另用 `aclrtEvent` 记录 GTest 的 `[PERF] avg_us` 供交叉验证。 |

共用 GTest 工程，按接口分别加载用例、执行精度与性能测试，并以“接口名 + 用例名”归档结果。随包用例按 Sgemm、Cgemm、Hgemm 分别覆盖 334、333、333 条精度用例和 67、67、66 条性能用例。

边界验证使用非对称复数矩阵区分转置与共轭转置；在 alpha=0 时给 A、B 填入 NaN，在 beta=0 时给 C 填入 NaN，检查被跳过的数据是否影响输出。在 C 每列填充区设置哨兵，检查尾块写回边界。FP16 用例另覆盖次正规数、接近表示上限的输入以及 Inf、NaN。

## 兼容性分析

Sgemm、Cgemm 沿用 ops-blas 公共声明。Hgemm 沿用相同参数顺序及句柄、转置枚举、状态码约定，矩阵和标量使用 `aclblasHalf`。三个接口均通过 arch22 构建接入 Atlas A2/A3。
