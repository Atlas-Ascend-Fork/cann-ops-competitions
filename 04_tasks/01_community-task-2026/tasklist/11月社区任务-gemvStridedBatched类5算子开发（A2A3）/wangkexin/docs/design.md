# 需求背景

批量矩阵向量乘为多组独立的矩阵和向量执行相同的线性变换。当各批次的首地址间隔固定时，调用方只需传入首批基址和三种批间距。此次开发对齐五个 strided batched GEMV 接口在 arch22 上的参数校验、混合精度计算和测试覆盖，沿用 ops-blas 句柄及异步 stream 接口。[社区任务书](https://www.hiascend.com/p/resource/202611/a828ff4efc46426bac03f2bbeff728bd.zip)

# 需求分析

对于批次 b，三个首地址分别为 `A + b*strideA`、`x + b*stridex` 和 `y + b*stridey`，偏移单位均为元素。每个批次原地更新 `y = alpha * op(A) * x + beta * y`。A 始终列主序，物理元素 `A(i,j) = A[i + j*lda]`。N 时 x 长 n、y 长 m。T 和 C 对实数执行同一转置路径，x 长 m、y 长 n。所有变体都用 FP32 标量和 FP32 运算，最终向 y 的元素类型舍入一次。

| 接口变体 | A 和 x | y |
| --- | --- | --- |
| S | FP32 | FP32 |
| HSH | FP16 | FP16 |
| HSS | FP16 | FP32 |
| TST | BF16 | BF16 |
| TSS | BF16 | FP32 |

例如 A 的两列分别为 (1, 2, 3) 和 (4, 5, 6)，取 `lda=4` 时每列末尾留一个填充元素。T 模式取 x=(1, 0, −1)、alpha=1、beta=0，输出为 (−2, −2)。每批 A 至少占 8 个元素，转置不改变其批间距下界。对负增量 inc，长度为 L 的向量第 k 个逻辑元素位于 `(1-L)*inc + k*inc`。

# 详细设计

## 接口与参数

公开声明位于 `include/cann_ops_blas.h`。FP16 和 BF16 输入/输出以 `uint16_t` 承载位模式。每个入口共享结构参数校验，再按输入、输出类型分派。[上游接口](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/include/cann_ops_blas.h)

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

## Host 侧

1. 校验句柄、trans、m/n、lda、incx/incy、三种批间距、批次数和标量指针。空句柄、非法 trans、非法取值分别返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、`ACLBLAS_STATUS_INVALID_ENUM`、`ACLBLAS_STATUS_INVALID_VALUE`。对于非空计算，检查 A/x/y 基址。任务书 §2.4 把增量写为任意非零 int，附件 TC_ED_212/213 与上游文档明确把 INT_MIN 列为非法，按这两个具名用例返回 INVALID_VALUE。
2. 无论转置与否，`lda >= max(1,m)`，三种批间距均为正。非空且存在相邻批次时，矩阵跨度为 `int64_t(lda)*n`，向量跨度为 `1+(len-1)*abs(int64_t(inc))`。检查步距覆盖对应跨度，并确保最后一批加上批内偏移的元素索引可表示。不同操作数之间的别名由调用方按契约避免。
3. 当 m、n、batchCount 中任一为 0，结构与标量合法后直接返回成功，允许 A/x/y 为空。非空输入中 alpha 为零仍检查基址，只对 y 执行缩放。beta 为零时跳过旧 y 的读取。
4. 从句柄取得 stream，根据可用 AIV 核数、批次数和输出长度确定分块。Host 向 Kernel 传入形状、批次范围、增量、FP32 标量、类型分派和输出区间。初始路径按 batch 分核。针对 batch 少、单矩阵大的形状，拟按批内输出区间继续分核，按实际设备访存粒度划分写回边界。对负步长、非连续输出及可能共享写回单元的边界，使用同核处理相邻输出。比较同卡同例结果后选择分核策略。

## Kernel 侧

N 路径沿行划分输出块，在统一缓冲区（UB）保存该块的 FP32 累加值，逐列加载连续 A 片段，把 A 元素和 x 元素转换为 FP32 后执行向量乘加。T/C 路径利用每列连续存储的特点分块读 A 与 x，逐列计算 FP32 点积，在归约完成后写输出。输出块大小按输入类型、FP32 累加器和归约临时区的 UB 占用确定。单位增量使用连续搬运，非单位及负增量按逻辑地址取值。尾块按实际元素数搬运，写回仅覆盖 y 的逻辑位置。批间填充和向量步长产生的间隙保持原值。使用 handle 绑定的 stream 异步下发，调用方在读回前同步。

## 接口约束与硬件

适配目标为 Atlas A2/A3（arch22）。m、n、batchCount 为非负整数，A 的物理列数恒为 n。三个批间距使用有符号 64 位元素偏移。alpha 和 beta 是 Host 端 FP32 标量，Device 端矩阵与向量位于调用方分配的缓冲区。各批次独立计算，输出空间由调用方提供。

# 可维可测分析

| 验证项 | 实施与判定 |
| --- | --- |
| 精度 | CANN 9.1.0，按任务书在 A2/A3 选一种款型自验。各变体 1,000 个精度用例，以 Netlib BLAS 3.10.0 逐批语义获得参考，经对应输出 dtype 量化。报告注明 FP64 累加后收窄或 FP32 CBLAS 的参考路径。逐元素混合容差 atol=rtol，FP32 为 2⁻¹³、FP16 为 2⁻¹⁰、BF16 为 2⁻⁷。每例匹配率至少 0.99，同时最大误差满足 §3.2 的绝对上限（FP32 为 0.01，FP16/BF16 为 0.1）或对应 32 ULP 标准。 |
| 功能边界 | 用真实 API 验证空矩阵/空批次及与非法参数的组合、正负步长、列填充、相邻批次重叠、N/T/C。按状态码逐例核对。以 NaN 检验 alpha=0、beta=0 的不读取语义，检查 y 间隙和批间填充。分别覆盖五个变体的正向与负向。 |
| 性能 | Atlas 800T A2（910B3），各变体 200 个性能用例，对齐 A100 cuBLAS 同配置基线。预热 5 次，取随后 10 次成功执行的 `msprof op` kernel Task Duration 平均值，换算为 ms 后逐例 ≤1.25 × GPU kernel 耗时。另以 `aclrtEvent` 在预置缓冲区计算端到端 `avg_us`。记录 T/C 和非默认标量所用 GPU 标杆的版本，以同一参数与已确认标杆完成正式比较。 |
| 兼容和复现 | 采用 ops-blas 现有五个声明和句柄接口，arch22 代码与测试分别进入官方指定目录。用例读取显式区分变体、填充模式和非法枚举，逐例记录名称、参数、预期与实际状态。正式测试使用 GTest/ATK，正式内存评价按 §3.4 记为不涉及。 |
