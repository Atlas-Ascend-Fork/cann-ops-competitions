# 需求背景（required）

## 需求来源

10月社区任务-aclblasChpr2算子开发（950）。在 ops-blas 中增加 Atlas 950PR 的复数 Hermitian packed 秩2更新能力，接口与 cuBLAS `cublasChpr2` 对应，计算语义参考 [Netlib CHPR2](https://www.netlib.org/blas/chpr2.f)。

## 背景介绍

BLAS 的 Hermitian 矩阵只需保存上三角或下三角。packed 按列连续保存该三角的 n(n+1)/2 个复数，避免分配完整 n×n 矩阵。本任务使用 Ascend C kernel 直调、handle 绑定 stream，实现原地更新；复数的实部与虚部均为 float32，表示方式以仓内 `aclblasComplex` 定义为准。

# 需求分析（required）

## 需求描述

新增公共接口 `aclblasChpr2`，在 Atlas 950PR、CANN 9.1.0 上对 packed 矩阵执行 `A ← A + αxyᴴ + conj(α)yxᴴ`。只更新指定的三角形，非 quick return 时对角虚部置零。支持正负非零向量步长，标量 alpha 从 Host 读取，向量和矩阵位于 Device。

## 需求拆解

1. 新增公共头文件声明，复用仓内 handle、stream 和状态码，不引入 950PR 私有 API。
2. 实现上下三角 packed 地址映射、复数计算和对角线处理；支持任意合法正负步长。
3. 严格遵循任务书的参数检查顺序与 quick return，不改变无操作场景的 ap。
4. 基于随任务提供的 CSV、自测指导和 Netlib cblas golden 完成精度、异常、性能验证。

# 详细设计（required）

## 算子分析

### 数学公式

对于存储的元素 `(i,j)`，`A'ij = Aij + α xi conj(yj) + conj(α) yi conj(xj)`。以列为单位可复用 `t1 = α conj(yj)`、`t2 = conj(α xj)`，再计算 `A'ij = Aij + xi t1 + yi t2`。复数乘法 `(a+bi)(c+di)=(ac-bd)+(ad+bc)i`，使用 float32 计算。对角位置只取旧实部与更新项的实部，输出虚部为 0；不读取旧对角虚部参与计算。按参考实现，当该列的 xj、yj 均为复数零时，跳过非对角更新，仅将对角虚部置零，以保证含非有限值输入时的参考语义。

### 支持数据类型与形状

| 参数 | 位置及类型 | 形状或约束 |
| --- | --- | --- |
| handle | Host 句柄 | 已创建的有效句柄 |
| uplo | Host 枚举 | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | Host int | 非负矩阵阶数 |
| alpha | Host `const aclblasComplex*` | 单个 complex64 标量 |
| x、y | Device `const aclblasComplex*` | 各至少 `1+(n-1)*abs(inc)` 个复数元素 |
| incx、incy | Host int | 非零且不为 `INT_MIN`，单位为复数元素 |
| ap | Device `aclblasComplex*` | 至少 `n*(n+1)/2` 个复数，原地输入输出 |

不支持广播、超出步长描述范围的非连续张量或 Device 标量指针。`n` 为运行时参数，不要求 dynamic shape 能力。

### 存储与步长

下标均为 0-based。上三角 `0≤i≤j<n`：`p=i+j*(j+1)/2`；下三角 `0≤j≤i<n`：`p=i+(2*n-j-1)*j/2`。未存储的另一半不引用。步长为正时 `start=0`，为负时 `start=(1-n)*inc`；逻辑元素 k 对应 `start+k*inc`。所有长度、偏移和中间乘法先提升为 64 位并检查可表示性，避免 int 中间溢出；调用方负责提供足够的有效 Device 内存。

## 算子实现

### 接口

在 `include/cann_ops_blas.h` 增加与其他产品线共用的声明：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy,
    aclblasComplex* ap);
```

实现位于 `blas/spr2/arch35/`，接入同族构建、导出和测试框架；复用仓内公共头文件中的复数类型和状态定义。

### 实现方案

#### Host 侧设计

按固定顺序检查并返回：

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0 返回 `ACLBLAS_STATUS_SUCCESS`，不访问其余参数。
2. alpha 为空、uplo 非法、incx 或 incy 为 0 或 `INT_MIN`，依次返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. x、y、ap 任一为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；之后读取 Host alpha，若实虚部均为零，返回成功，不访问 x/y/ap 内容，也不修改 ap。
4. 计算 64 位 packed 长度、向量起点、分核与 tile 参数；基于平台查询可用核数和 UB 大小，传递 n、uplo、步长、起点、alpha 和切分参数。通过 handle 绑定的 stream 异步启动 Ascend C kernel，沿用仓内启动错误处理和状态码规范。

分核按 packed 连续区间而非简单按列等分，以平衡不同列长度：`P=n*(n+1)/2`，对齐边界的连续区间分配给各核，保证每个 packed 元素仅一个核写入。小规模减少核数以降低启动和分核开销。区间起点通过单调列起始偏移的整数二分确定列号，随后按列递增遍历，避免浮点反解导致边界错误；上下三角分别使用 `j*(j+1)/2` 和 `j*(2*n-j+1)/2` 作为列起点。相邻核不共享写入元素；非对齐边界的搬出采用硬件支持的安全尾部写入方式，避免对邻核元素的覆盖。具体对齐粒度和最大 tile 根据实际平台 API 与 UB 约束确定并以性能测试调优。

#### Kernel 侧设计

各核按所属 packed 区间逐列、逐 tile 执行 CopyIn、Compute、CopyOut。每列先读取 xj、yj 并生成 t1/t2；连续装载 ap 和逻辑 x、y 对应区段。步长为 1 时采用连续搬运；其他步长以支持的 gather 或安全分段搬运装入 UB，负步长由 Host 计算起点后使用有符号 64 位地址。将复数实虚分量拆分为 float32 向量，完成复数乘加后重新交错存回。根据 UB 容量保留输入、输出和临时缓冲区，可采用双缓冲重叠搬运与计算；若长度不足对齐粒度，使用边界安全的搬运/掩码，禁止越界访问。

每个 tile 检测对角下标，仅写实部并把虚部设为零；列中 xj、yj 均为零时保留非对角原值、仅清理对角虚部。非零 alpha 场景逐列处理所有对角；`n=0` 和零 alpha 在 Host 返回，不启动 kernel，因此 ap 完全不变。x、y 只读、ap 原地更新，核间不需要归约或原子操作。读取结果前由调用方同步 stream。

### 支持硬件

| 支持的芯片版本 | 支持情况 |
| --- | --- |
| Atlas 950PR | 支持 |

### 算子约束限制

alpha 为 Host 指针；x、y、ap 是 Device 指针。只访问指定三角的 packed 存储；不存在 lda。调用方需保证缓冲区长度和地址有效，避免 ap 与输入发生有害重叠。计算不保证 bit-exact；零 alpha 时对角虚部也保持原样。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 使用 Netlib BLAS 3.10.0 `cblas_chpr2` 生成 packed golden，对实虚部分别比较；FLOAT32 rtol、atol 均为 2^-13，匹配率至少 0.99，最大绝对误差上限为 1e-2 或 32 ULP，按任务精度标准执行。 | 任务书及 [生态算子精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 特殊值 | NaN 与 NaN、同号 Inf，或 NaN 与任意 Inf 视为等效；有限值与非有限值不等效。非有限值单独统计并全部通过；有限值按匹配率与最大误差判定。 | 任务书 |
| 性能 | 950PR 使用 msprof op，5 次 warmup 后取 10 次有效单算子 task duration 平均值；每个性能用例耗时不超过 GPU 标杆的 2.5 倍。参考随附 `test_cases/gpu_baseline.csv` 及全部性能用例。 | 任务书 |

精度用例覆盖上下三角、n=0/1、小质数、2 的幂及邻近值、大规模和非对齐尾块，正负步长组合、零/特殊 alpha、Inf/NaN 和不同输入分布。全量检查 packed 输出和对角虚部；零 alpha、n=0 单独检查 ap 未变化。负向用例覆盖空 handle、负 n、非法 uplo、空 alpha、零及 `INT_MIN` 步长、空 x/y/ap，以及 n=0 与非法参数组合、零 alpha 与空设备指针组合，以验证检查优先级。测试代码基于仓内 GTest 加载随任务提供的完整 CSV；记录参数、结果、精度与性能日志和截图，提供可复现步骤及自测报告。性能目标为设计目标，实际通过情况以 950PR 实测结果为准。

## 兼容性分析

新增公共 API，不改变已有接口签名；实现仅针对 arch35/950PR，其他产品线继续使用同一公共声明并遵守各自已有的支持策略。调用依赖 handle 所绑定的 stream，保持仓内异步行为。
