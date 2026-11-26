# 需求背景（required）

## 需求来源

昇腾社区 [11 月社区任务-aclblasChpr2 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/7a1275ee6b654459957fc27c8dde32b0?menu=tasks)。以任务书第 2 至 4 节为需求和验收依据。

## 背景介绍

`aclblasChpr2` 执行单精度复数 Hermitian 矩阵的压缩存储秩 2 更新。代码目标仓为 `cann/ops-blas`，参考基线为 `master` 的 `12e37111`。该基线已包含 `arch22` 的 SIMD 实现及 GTest，部分边界行为与本次任务书有差异。本方案复用仓内实现，并补齐任务书规定的接口语义与验证覆盖。

# 需求分析（required）

## 需求描述

沿用 `include/cann_ops_blas.h` 的公共声明，通过句柄绑定的 AscendCL stream 直接下发 Ascend C kernel。对 `AP` 原地更新，支持列主序 UPPER 和 LOWER、Host 或 Device 复数标量及正负非零步长。调用方保证 AP 的写入区域与 x、y 的读取区域互不重叠。

## 需求拆解

| 项目 | 设计要求 |
| --- | --- |
| 功能 | 两种压缩布局、复数标量、正负步长、对角虚部清零 |
| API | 校验顺序和返回码按任务书 §2.4–2.5，保留空输入和零标量的语义 |
| 异步 | Device 标量在同一 stream 内由 kernel 读取，保留调用方的 stream 顺序 |
| 验证 | 基于 Netlib CBLAS 进行实部和虚部比较，在 910B3 上采集 kernel 耗时 |

# 详细设计（required）

## 算子分析

### 数学公式

令 `A` 为由 `AP` 表示的 Hermitian 矩阵，更新为 `A + α·x·yᴴ + conj(α)·y·xᴴ`。第 j 列的系数为 `t1 = α·conj(y[j])`、`t2 = conj(α·x[j])`，元素增量为 `x[i]·t1 + y[i]·t2`。正常更新的对角项只保留实部，写入虚部 `+0.0f`。

### 支持数据类型及形状

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| handle | aclblasHandle_t | 有效句柄及其已绑定的 stream |
| uplo | aclblasFillMode_t | ACLBLAS_UPPER 或 ACLBLAS_LOWER |
| n | int | n ≥ 0 |
| alpha | const aclblasComplex* | 一个 complex64，Host 或 Device 地址 |
| x、y | const aclblasComplex* | 逻辑长度 n，n > 0 时物理长度至少为 1 + (n−1) × abs(相应步长) |
| incx、incy | int | 非零且不等于 INT_MIN |
| AP | aclblasComplex* | n(n+1)/2 个 complex64，原地读写 |

实部和虚部为 float32，按交错布局在 GM 中存放。以零为起始索引，UPPER 列 j 的起点为 `j(j+1)/2`，访问行 0 至 j。LOWER 的列起点为 `j(2n−j+1)/2`，访问行 j 至 n−1。正步长的物理下标为 `i × inc`，负步长为 `(n−1−i) × abs(inc)`。偏移及占用计算采用 64 位。

## 算子实现

### Host 侧设计

按顺序校验 handle、负 n、alpha 空指针、uplo、步长。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余值错误返回 `ACLBLAS_STATUS_INVALID_VALUE`。校验通过且 n 为零时直接成功返回。n 大于零时检查 x、y、AP 及字节跨度，之后才处理 alpha 的值。

复用仓内 `CheckPtrLocation`。Host alpha 在调用时读入 tiling，零值直接返回成功。Device alpha 指针随 kernel 参数传入，kernel 在加载向量和 AP 前读取该标量，零值时结束该核。launch 失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

### Kernel 侧设计

核间按列分配，`B = min(n, AIV 核数)`。每个批次含 B 列，偶数批次按核号正序分配，奇数批次反序分配，从而平衡三角列长度。每列唯一归属一个核。核内按最多 1024 行切分，将行片和当前列区间求交，仅更新交集。AP 写回长度按交集计算。tiling 记录 n、uplo、核数、有符号步长及 Host alpha 两个分量，Device alpha 使用独立指针参数区分。

| UB 缓冲 | 个数 | 单个字节数 | 用途 |
| --- | ---: | ---: | --- |
| Xraw、Xsprime、Yraw、Ysprime | 4 | 8448 | 交错向量与旋转后的向量分量 |
| AP、D1、D2、Work | 4 | 8448 | 压缩矩阵片段、两项复乘、非对齐运算 |
| ShiftIdx、PackIdx | 2 | 8448 | Gather 索引 |
| SplitIdx | 1 | 4096 | 分量拆分索引 |
| Scalar | 1 | 256 | 复数标量搬运及对角写回 |
| Win | 1 | 32768 | 非单位步长路径的连续读取窗口 |

常驻 UB 为 88,832 字节，带窗口为 121,600 字节。以 1024 行计，去掉 256 字节标量缓冲的开销分别为每行 86.5 和 118.5 字节，包含对齐冗余及索引。无需新增 GM workspace。窗口覆盖不了的步长按物理索引逐元素读取。

普通路径先用 `DataCopyPad` 载入向量片段，借助 `Gather` 和 `Muls` 建立旋转分量。复乘用 `Muls` 与 `Axpy`，两项贡献用 `Add` 累加到 AP。用实际元素数执行 `DataCopyPad` 写回，搬运、标量写入和向量计算间沿用事件同步。若列系数或输入出现非有限值，应单独验证复乘的 Inf/NaN 传播，与 Netlib CBLAS 对齐后再纳入验收。两个列向量分量均为复零时沿用 Netlib 跳过列更新的行为，仍清零该列对角虚部。

## 支持硬件

| 硬件 | 用途 |
| --- | --- |
| Atlas A2 系列 | arch22 实现，正式性能自测使用 Atlas 800T A2（910B3） |
| Atlas A3 系列 | arch22 实现 |

## 算子约束限制

输入须遵守向量步长和 AP 压缩布局。n 或 alpha 为零且校验通过时，AP 保持原有内容。Device alpha 指针及向量和 AP 的设备内存需保持有效，直到相应 stream 工作完成。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验收标准 | 标准来源 |
| --- | --- | --- |
| 精度 | 1000 条精度用例，实部和虚部分开与 Netlib CBLAS 对比。rtol 和 atol 为 2⁻¹³，匹配率至少 99%，最大绝对误差满足 1e−2 或 32 ULP 上限 | 任务书 §3.2、§3.5 |
| 性能 | 200 条性能用例，预热 5 次，取后续 10 次有效 kernel Task Duration 的算术平均。逐例与同一用例的 GPU kernel 标杆比较，NPU 耗时不超过标杆的 1.25 倍 | 任务书 §3.3、§3.5 |
| 边界 | 覆盖非法 uplo、n 为零时的参数校验、零 alpha、Host/Device alpha、负步长、对角虚部及非有限值 | 任务书 §2.4–2.5、§3.5 |

通过仓内 GTest 加载任务 CSV，精度参考 `cblas_chpr2(CblasColMajor, ...)`。性能由 `msprof op` 采集并按用例与随包 `gpu_baseline.csv` 匹配，日志记录异常样本和有效样本数。事件计时用于快速回归。

## 兼容性分析

沿用现有公共 ABI。参数校验顺序及非法 uplo 的返回码按任务书更新 arch22 行为。正常更新和快速返回遵循同一公开接口。基线上的首次开发测试已运行 1003 项，996 项通过。5 项失败涉及仓内用例的旧参数预期，2 项涉及 Inf 传播差异。修订后在 910B3、CANN 9.1.0 上通过任务包 1,000 条精度用例及 2 条补充接口用例。正式性能采集仍待取得有效样本。

## 参考资料

- [Netlib CHPR2](https://www.netlib.org/blas/chpr2.f)
- [NVIDIA cuBLAS CHPR2](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpr2)
- [ops-blas](https://gitcode.com/cann/ops-blas)
