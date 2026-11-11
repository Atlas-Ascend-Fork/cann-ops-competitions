# 需求背景（required）

## 需求来源

11月社区任务《aclblasChbmv算子开发（A2A3）》要求补齐 ops-blas 的单精度复数 Hermitian 带状矩阵向量乘能力。接口语义对齐 cuBLAS `cublasChbmv`，参考 Netlib BLAS `chbmv`。

## 背景介绍

Hermitian 带状矩阵仅保存一个带形三角可减少存储与访存，另一半由共轭对称性恢复。本设计面向 Atlas A2/A3，使用 Ascend C kernel 直调，在 ops-blas 句柄与 stream 框架中实现，结果原地写回 y。

# 需求分析（required）

## 需求描述

新增公共接口 `aclblasChbmv`，支持 complex64 列主序带状矩阵、上下三角、合法正负向量步长及列间填充，实现 `y = alpha * A * x + beta * y`。按照任务书校验并异步提交到 handle 绑定的 stream。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 新增公共声明，复数布局复用 `include/cann_ops_blas_common.h`；代码位于 `blas/sbmv/arch22/`，测试位于 `test/sbmv/chbmv/arch22/`。
2. 正确处理上下三角带、对角虚部、padding、正负步长及 `k > n-1`；只读有效数据，`beta=0` 不读 y 旧值。
3. 按要求校验、快速返回并避免溢出；按绑定 stream 异步执行。
4. 覆盖随包用例并补充正态分布；在 A2/A3 一款上自验精度，在 910B3 上按任务书采集性能并对照 GPU 标杆。

# 详细设计（required）

## 算子分析

### 数学公式

对 `0 ≤ i < n`，`y_i = alpha * Σ(j=max(0,i-k)..min(n-1,i+k)) A_ij * x_j + beta * y_i`。`A_ji = conj(A_ij)`，对角线只取实部。复数乘法 `(a+bi)(c+di)=(ac-bd)+(ad+bc)i`，实虚部以 float32 计算和累加。每个输出由唯一执行单元拥有，避免写冲突。

零基下标、复数元素为单位：UPPER 的 `i≤j` 元素存于 `j*lda+k+i-j`；LOWER 的 `i≥j` 元素存于 `j*lda+i-j`。未存储侧交换 i、j 查找并对虚部取反。只访问行列均合法且 `|i-j|≤min(k,n-1)` 的元素，禁止读取无效带区和列 padding。`k>n-1` 时 UPPER 对角存储行仍为原始 k。

### 支持数据类型

alpha、beta、A、x、y 为 `aclblasComplex`，实部和虚部各 float32；n、k、lda、incx、incy 为 int。累加和输出均为 float32 分量，不支持广播或其他数据类型。

### 支持形状

A 逻辑 n×n，物理列主序 `lda×n`，`lda≥k+1`；x/y 逻辑长度 n，物理长度至少 `1+(n-1)*abs(inc)`。负步长逻辑起点偏移 `(1-n)*inc`，逻辑元素地址为起点加 `index*inc`。`n=0` 不访问数据；不支持超出 lda 和步长表达的其他视图。

## 算子实现

### 实现方案

#### Host/Tiling 设计

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

1. 校验顺序：handle 非空（否则 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）、n≥0（否则 `ACLBLAS_STATUS_INVALID_VALUE`）、n=0 立即成功；然后检查 uplo、k、lda、incx/incy、alpha/beta/A/x/y。uplo 仅 UPPER/LOWER，k≥0，lda≥k+1，步长非 0 且非 INT32_MIN，各指针非空，非法值统一返回 `ACLBLAS_STATUS_INVALID_VALUE`。n=0 不检查其余参数，不启动 kernel；beta=0 时 y 仍须非空。
2. 用 int64 计算 `k+1`、`(n-1)*abs(inc)`，跨度超过 INT32_MAX 返回非法；地址和字节偏移采用宽整数，避免 int32 溢出。Host 读取 Host 标量并传入 kernel，A/x/y 为 Device 地址。
3. 根据 n、有效带宽和平台可用核心数选择 `P≤min(n,可用核数)`；连续输出行均分，前 `n%P` 核各多一行。小规模减少核数；阈值经设备测试确定。每核独占行范围，不发生写冲突。
4. 按平台 UB 容量、对齐要求及临时缓冲预算确定行块和带宽块；双缓冲计入两份缓冲和同步空间。向 kernel 传 n、原 k、lda、步长、逻辑起点、分核范围、块大小、uplo、标量和 beta 零标记。按 uplo、单位步长及 beta 零值可选择特化分支，通用分支覆盖其余情形；具体参数需实测调优。
5. 依据 handle 获取 stream，遵循仓库既有方式直调 kernel 并返回下发状态；读取 Device 结果前调用方须同步 stream。x/y 重叠情况遵循上游接口约定；若允许重叠，需先在同一 stream 上快照完整逻辑 x，再启动计算并保证快照生命周期，不能依赖并行原地读取。

#### Kernel 设计

1. Init 建立输出行区间与 GM/UB 缓冲；Process 分批处理输出行并计算有效 j 范围，尾块采用有效长度或掩码。只读取指定三角的有效位置；镜像侧虚部取反，对角虚部强制为零。
2. 分带宽块搬运 A、x 至 UB；尽量合并有效连续片段，非单位步长和不对齐尾块走安全路径，禁止越界读取。累加实部 `ar*xr-ai*xi`、虚部 `ar*xi+ai*xr`；每行块依序累加，不跨核归约。元素偏移换算字节使用宽整数。
3. 完成行后计算 `alpha*sum`；beta 为 `(0,0)` 时直接输出，绝不读旧 y；否则读取对应 y 并加 `beta*y`，仅写本行。`alpha=0` 不保证跳过 A/x，含 Inf/NaN 的 x 不与 alpha=0 组合。
4. 按需使用 CopyIn/Compute/CopyOut 流水；双缓冲须用事件保证缓冲复用与写回顺序。各核拥有独立输出行，尾块仅搬运有效范围。

## 支持硬件

| 芯片版本 | 支持 | 验证计划 |
| --- | --- | --- |
| Atlas A2 系列（含 Atlas 800T A2，910B3） | √ | A2/A3 至少一款精度自验；910B3 性能验收 |
| Atlas A3 系列（含 Atlas 800T A3） | √ | 可作为精度自验设备 |

软件环境为 CANN 9.1.0；arch22 路径的可用核心数、UB 容量与 API 对齐要求以运行平台为准。

## 算子约束限制

仅支持 complex64、列主序 Hermitian 带状格式；无广播、额外非连续视图和转置参数。n/k 为运行时参数，不要求动态 shape 图编译；y 原地覆写。不要求 bit-exact 或确定性。n=0 快速返回，alpha=0 且 beta=1 不要求短路。对角虚部不参与计算，未引用带区不得访问；beta=0 时 y 旧值无需有效但指针仍须非空。Device 数据异步使用期间由调用方保证生命周期。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 按任务书和随包测试框架使用 Netlib CBLAS 列主序 Hermitian 带状计算，输入转 float64 计算高精度真值并收窄为 float32；全量 y 实、虚部分别比较。有限元素 `atol=rtol=2^-13`，匹配率≥0.99 且最大绝对误差≤1e-2 或最大误差点的 32 ULP。非有限值单独判定：同为 NaN、同号 Inf、或一方 NaN 一方任意 Inf 等效；有限/非有限不等效，全部非有限元素须通过。 | 任务书 §3.2 与生态算子精度标准 |
| 性能标准 | 910B3 上 msprof op 的单算子 task duration，预热 5 次后至少 10 次有效采样均值；逐用例平均耗时≤随包 GPU 标杆的 1.25 倍，异常样本及环境记入报告。 | 任务书 §3.3 与 `test_cases/gpu_baseline.csv` |

测试覆盖 UPPER/LOWER、n=0/1/小质数/非对齐/大规模、k=0/1/小带宽/n-1/>n-1、lda 最小值及 padding、步长 ±1/±2/±3、零/一/纯虚数与大值标量、对角非零虚部、无效带区及 beta=0 旧 y 特殊值；覆盖负 n/k、非法 uplo/lda、零及 INT32_MIN 步长、跨度溢出、空指针、n=0 叠加非法参数与空 handle 优先级。含 Inf/NaN 按任务书规则验证。采用随包 1000 条精度及 200 条性能 CSV，补充均匀/正态各 50% 的分布用例并记录生成方式和数量。负向用例用单元素占位，避免按非法跨度分配。自测报告记录参数、实虚部精度、性能、设备版本、日志和复现步骤；结果以设备实测为准。

## 兼容性分析

新增公共 API 不改变已有接口签名；头文件、导出、构建与调用约定遵循 ops-blas 规范。非法 uplo 按任务要求返回 `ACLBLAS_STATUS_INVALID_VALUE`，即使部分已有同族接口采用不同状态码。与 cuBLAS 对齐参数顺序和核心数值语义，设备、句柄及状态码依 ops-blas 定义。
