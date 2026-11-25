# 需求背景

`aclblasChpmv` 计算单精度复数 Hermitian 矩阵与向量的乘加。调用方只存上三角或下三角，矩阵占用 `n(n+1)/2` 个复数元素。Atlas A2/A3 的实现和测试已在 `ops-blas` 的 `blas/spmv/arch22` 与 `test/spmv/chpmv/arch22` 中。本次沿用公开接口和已有分核算法，完善结构参数检查、设备标量的读取与当期用例验证。[任务书和测试包](https://www.hiascend.com/p/resource/202611/c07419f0566d426dacdcc1e3de6885c8.zip) · [arch22 源码基线](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/spmv/arch22)

# 需求分析

按 `y ← alpha × A × x + beta × y` 原地更新逻辑向量 y。AP 按列打包，以零为起点。上三角的 `A[i,j]` 在 `AP[j*(j+1)/2+i]`，下三角的 `A[i,j]` 在 `AP[j*(2*n-j-1)/2+i]`。每个非对角元素 `a = A[i,j]` 分别贡献 `a*x[j]` 和 `conj(a)*x[i]` 给第 i、j 项，对角只贡献 `real(A[j,j])*x[j]`。例如上三角存储 `[2+9i, 1+2i, 3-7i]`，取 `x=[1,i]`、`alpha=1`、`beta=0`，得到 `y=[i,1+i]`。

# 详细设计

## 算子原型

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* AP,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

| 参数 | 方向和类型 | 约束 |
| --- | --- | --- |
| handle、uplo、n | 句柄、枚举、int | 使用句柄绑定的 stream。uplo 取 ACLBLAS_UPPER 或 ACLBLAS_LOWER。阶数 n ≥ 0。 |
| alpha、beta | 输入，各一个 complex64 | 标量指针非空，支持 Host 和 Device 地址。 |
| AP | 输入，complex64 | 长度为 n(n+1)/2。每个元素由两个 float32 表示，按实部、虚部排列。 |
| x、incx | 输入向量、以元素为单位的步长 | x 的逻辑长度为 n，incx 为任意非零 int。n > 0 时存储至少容纳 1+(n−1)·abs(incx) 个元素。 |
| y、incy | 输入输出向量、以元素为单位的步长 | 与 x 使用同样的长度和步长约定。只写回逻辑元素。可写区与 AP、x 的可读区互不重叠。 |

Host 依次检查 handle、负阶数、uplo、零步长及 alpha/beta 空指针。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。通过这些检查后，n=0 立即返回成功，无需访问 AP、x、y 或启动内核。n>0 时先检查三个张量指针，再按标量选择计算路径。

## 实现方案

Host 标量取值随启动参数传入。Device 标量地址传给同一 stream 上的内核，由设备按流顺序读取，支持此前在该 stream 上更新的标量。alpha=0 时跳过 AP 和 x。此时 beta=1 保留 y，其他 beta 仅缩放 y，beta=0 直接写零。在一般乘加中 beta=0 同样跳过 y 的旧值，避免 NaN 污染结果。

性能基线沿用两阶段向量核。第一阶段将压缩列分配给不同核心，每核连续分段读取自己负责的列，把两侧贡献累加到独立的实部、虚部工作区。第二阶段按输出行分块，归约各核部分和，再进行复数标量乘加和写回。核心数量受可用工作区和矩阵阶数约束。两阶段在同一 stream 上依次下发，避免多个核心同时写同一输出元素。单位步长连续搬运，非单位步长仅访问有效元素。对负步长 s，第 k 个逻辑元素的物理偏移为 `(n-1-k)*(-int64_t(s))`，计算起点和压缩偏移时使用 64 位整数。

当不同列分到多个核心时，每个核心先累加部分和，跨核归约会改变 FP32 的舍入顺序。在正负项相消、输出接近零的行，这一差异可能放大相对误差。为验证误差来源，另用每行独立累加的实现对照同一份输入。上三角从缩放后的 y 开始，依次加入对角项、当前压缩列的共轭点积和后续列对本行的贡献。下三角先加入前序列的贡献，再加入对角项和当前列的共轭点积。复数乘积分别计算实、虚部，在加入行累加器前完成该乘积的舍入。

对 Host 标量 alpha=1、beta=0 且工作区充足的输入，将非对角块的压缩列片段搬入统一缓冲区（UB）。每块读取一次矩阵元素，同时计算它对块行的直接贡献和对块列的共轭贡献。对角块只使用对角元素实部。各核将块的部分和写入独立工作区，核间同步后按逻辑行合并。非单位步长的 x 可先按逻辑顺序收集；写回 y 时按 incy 定位，仅覆盖每个复数元素的两个分量。

大矩阵的精度修正先估计各分块入口处的 FP32 前缀和，再以该前缀为起点计算块对行累加器的增量。下三角每行的共轭点积从对角块独立起算。若行末某一分量相对中间部分和很小，则按 Netlib 的复数乘积和加法次序重算该行。通过小规模、正负步长和大矩阵用例共同选择分块参数，并用同一源码版本完成全量精度和逐例性能自验。

# 可维可测分析

| 验证项 | 方法 | 通过条件 |
| --- | --- | --- |
| 数值精度 | 在 CANN 9.1.0 的 A2 或 A3 上，通过公开接口运行附件中 1,000 条精度用例。Netlib BLAS 3.10.0 的 `cblas_chpmv(CblasColMajor, …)` 提供参考结果。实、虚部分别比较，记录匹配率、最大误差以及附件要求的 MERE、MARE。 | 两个分量均满足 float32 的混合容差。atol、rtol 均为 2⁻¹³，匹配比例至少 0.99，最大绝对误差按 10⁻² 或 32 ULP 判定。附件另要求两个分量分别满足 MERE < 2⁻¹³、MARE 符合 CSV 中 10 倍阈值的离群限制。特殊值分别检查 NaN、Inf 及无穷符号。 |
| 边界与异步语义 | 覆盖 n=0 叠加非法枚举、零步长及空标量，n=1 取 INT_MIN 步长，正负非单位步长，对角虚部非零，beta=0 且 y 含 NaN。补测 Host/Device 标量组合及同一 stream 先更新标量再调用接口。 | 返回码符合检查顺序。检查输出分量、步长间隙和缓冲区保护区。退化路径保留或覆盖相应数据。 |
| 正式性能 | Atlas 800T A2（910B3），CANN 9.1.0。按 uplo、n、incx、incy 匹配 200 个性能用例。每例预热 5 次后，用 `msprof op` 采集连续 10 次成功调用的 `Task Duration(us)`。一次调用下发多个内核时，汇总该次调用的各段耗时再取平均。 | 每例平均微秒数除以 1,000 后须 ≤ 1.25 × 对应的 `gpu_ms`。采集报告保留各段明细。事件计时只用于快速回归。 |

本题未设置独立的内存性能指标。测试原地写回时检查 AP、x 的内容和 y 的非逻辑元素，覆盖正反向跨度及分块尾部。[官方用例与验收标准](https://www.hiascend.com/activities/task-center/details/c596a05d90b9471683e41fcf4d026ae5?menu=tasks)
