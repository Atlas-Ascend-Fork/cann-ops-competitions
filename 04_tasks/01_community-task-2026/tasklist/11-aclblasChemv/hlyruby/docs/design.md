# 需求背景

## 需求来源

11 月社区任务《aclblasChemv 算子开发（A2A3）》要求为 ops-blas 增加 Ascend C 实现，覆盖 Atlas A2/A3，语义对齐 cuBLAS `cublasChemv` 和 Netlib `chemv`。本设计是开发方案，性能与精度结论以实际自测结果为准。

## 背景介绍

复数 Hermitian 矩阵向量乘是复数线性代数的基础运算，在求解线性方程组等场景使用。只引用矩阵的上三角或下三角，可避免读取未定义的另一半数据；对角元素虚部不参与计算。ops-blas 当前需补充面向 A2/A3 的同名共享接口及实现。

# 需求分析

## 需求描述

提供句柄式接口，使用句柄绑定的 stream 直调 Ascend C kernel，原地计算 `y = alpha * A * x + beta * y`。A 是按列主序存储的 `n×n` complex64 Hermitian 矩阵；alpha、beta、x、y 同为 complex64。`uplo` 选择唯一允许读取的三角，未存储三角通过共轭对称关系推导，对角只取实部。`lda` 支持列间填充，`incx`、`incy` 支持正负非零步长。接口在 `include/cann_ops_blas.h` 新增，签名与任务书一致：

```cpp
aclblasStatus_t aclblasChemv(aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

复数内存布局以 `include/cann_ops_blas_common.h` 的 `aclblasComplex` 定义为准。alpha、beta 支持 Host 和 Device 指针；A、x、y 为 Device 指针。输出覆盖原 y，函数返回状态码；计算完成前读取结果须同步对应 stream。

## 需求拆解

| 项目 | 要求 |
| --- | --- |
| 功能 | UPPER/LOWER 均支持，支持任意合法 `n`、`lda` 和正负步长；不读取未指定三角、对角虚部不参与运算 |
| 参数与边界 | `n=0` 成功；`alpha=0,beta=1` 成功且不访问 A/x/y；`alpha=0` 不访问 A/x；`beta=0` 不读取旧 y，但输出地址须有效 |
| 工程 | `blas/symv/arch22/` 实现，`test/symv/chemv/arch22/` 测试，`test/symv/chemv/CMakeLists.txt` 的 arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)` |
| 验收 | 使用 Netlib CBLAS 全向量比对实部和虚部；910B3 上以 msprof op 按任务书采集并与 GPU 基线逐 case 比较 |

# 详细设计

## 算子分析

### 数学公式

将逻辑矩阵记为 H，列主序存储地址为 `A[i+j*lda]`（0 基）。UPPER 时，`i<j` 取 `A[i+j*lda]`，`i>j` 取 `conj(A[j+i*lda])`；LOWER 时，`i>j` 取 `A[i+j*lda]`，`i<j` 取 `conj(A[j+i*lda])`。`i=j` 取 `real(A[i+i*lda])+0i`。因此 `y_i=alpha*sum_j(H_ij*x_j)+beta*y_i`。复数乘法为 `(a+bi)(c+di)=(ac-bd)+(ad+bc)i`，实虚分量以 float32 计算和累加；顺序不同可能产生正常的浮点误差。

### 支持数据类型

输入输出复数均为 complex64，实部、虚部各为 float32；累加使用 float32。`n/lda/incx/incy` 为 int，模式为 `aclblasFillMode_t`。不支持其他输入类型。

### 支持形状

A 的逻辑形状为 `[n,n]`，物理列主序空间为 `lda*n` 个复数；`lda>=max(1,n)`。x/y 逻辑长度为 n，其物理存储至少 `1+(n-1)*abs(inc)` 个复数。n 运行时给出，既不广播，也不支持除 lda 与向量步长外的任意视图步幅。步长为负时逻辑第 0 个元素偏移为 `(1-n)*inc`，逻辑第 k 个元素偏移再加 `k*inc`。

## 算子实现

### 实现方案

#### Host 侧设计

1. 接口取得 handle 对应 stream，依次检查 handle 非空（返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）、`n<0`（返回 `ACLBLAS_STATUS_INVALID_VALUE`）；`n=0` 立即返回成功，不访问其他参数。随后检查 uplo（非法返回 `ACLBLAS_STATUS_INVALID_ENUM`）、lda、步长和 alpha/beta 指针（非法返回 `ACLBLAS_STATUS_INVALID_VALUE`）。拒绝 0 和 `INT32_MIN` 步长；以 int64 计算 `(n-1)*abs(inc)`，并按仓内 arch22 地址偏移可表示范围检查，溢出时返回错误。矩阵、workspace 大小和字节偏移也用宽整数检查溢出。
2. 参考仓内 csymv arch35 `ComplexScalarSource` 检测 alpha、beta 指针属性。Host 标量直接读取；Device 标量在绑定 stream 上有序复制至 Host 并等待复制完成，再读取两个分量。这样能依据实际值同步返回条件性参数错误，并尊重该 stream 上先前对标量的写入；Device 标量路径会产生必要的 stream 同步开销，标量读取后 kernel 的执行仍异步。具体内存类型检测、拷贝及错误映射沿用仓内公共机制，不直接解引用 Device 指针。
3. `alpha=(0,0), beta=(1,0)` 直接成功，A/x/y 允许为空；否则 y 必须非空，alpha 非零时 A、x 必须非空。alpha 为零走单独的 y 缩放 kernel，不读取 A/x；beta 为零直接生成结果，不读旧 y。传入 kernel 的标量为已读取的值，避免设备指针生命周期及路径分歧。
4. 根据设备核数、n、UB 可用容量与地址安全上限选择 tiling；申请所需临时 workspace，并保证其生命周期覆盖所有异步 kernel。资源分配、kernel 下发失败映射为仓内已有状态码，异常路径释放资源。复用 ops-blas 的 stream 安全工作区管理机制；如现有管理不支持异步释放，实现时须补齐等效生命周期管理。

#### Tiling 策略、分核与数据分块

把输出行分为长度 `R` 的行块，把归约维 j 分为长度 `C` 的列块。以 `(行块,列块)` 为独立任务，分配给可用核，核心数不超过设备可用核数及任务数。每任务计算该行块对该列块的复数部分和，写入独占 workspace `partial[column_block][row]`；第二个 kernel 按输出行块合并所有列块部分和，并融合 alpha、beta 和 y 写回。这样跨核无需原子加，不同核不同时写同一 y；第二阶段与第一阶段在同一 stream 顺序执行。小 n 或只有一个列块时可采用单 kernel 完成对应行块的全部归约，避免 workspace 与二次启动。alpha=0 仅按输出行分核缩放 y。

R、C 根据 UB 容量选择：同时计入矩阵片段、x 片段、实虚累加器、输出和必要的双缓冲；预留对齐与指令临时空间，不能超过实际可用 UB。尾块使用有效行数和有效列数，不能用越界读取填充。`ceil(n/R)`、`ceil(n/C)` 及 workspace 字节数用宽整数计算并检查；对超出 workspace 或地址限制的规模，以分批列块并在同一 stream 上顺序累加的方式控制内存，若实际平台资源仍不足则返回对应资源错误。参数、块大小和批次边界经性能实测调优，不预设达到验收目标。

#### TilingKey

按计算路径区分：通用 Hermitian 路径、alpha=0 缩放路径；通用路径再依据 UPPER/LOWER、连续或非连续向量、单列块或多列块组合选择实现。beta=0 作为写回分支或 key，确保不加载旧 y。key 的具体编号与 tiling 数据结构遵循 ops-blas 的注册及 kernel 直调约定；不为 n 的每种取值创建新 key。

#### Kernel 侧设计

通用 kernel 对 `(i,j)` 根据 uplo 和相对位置选择合法的存储坐标；矩阵块跨对角线时按区域拆分，直接区、共轭区、对角分别处理，禁止以整块读取方式触及未引用三角。对共轭区读取转置位置并对虚部取负；对角仅使用实部。加载按实际有效边界执行，对齐或非连续访问使用安全的标量/分段搬运，不以读过界换取对齐。x 根据 `start_x+j*incx` 索引；在 UB 内按实虚分量计算复乘并归约，部分和写入各任务独占的 workspace。第二阶段以固定顺序合并各列块结果、乘 alpha；beta 非零时读取 `start_y+i*incy` 并复乘加，beta 为零时不读取，最后按步长原地写回。alpha=0 的缩放路径同样保证 beta=0 时不读旧值。

按列主序对可连续区域优先合并搬运，跨对角线和转置区域使用合法分段；在 UB 允许时复用 x 并流水搬运和计算。多核并行、分块以及额外启动的选择需以真实硬件 profiling 决定。该方案不承诺 bit-exact，也不要求跨不同分核配置位级确定性。

## 支持硬件

支持 Atlas A2、Atlas A3 系列，目标架构 arch22；CANN 9.1.0。自验证至少覆盖一种款型，性能验收设备为 Atlas 800T A2（910B3）。具体核数、UB 大小和对齐能力运行时或按平台接口获取，不假设不同型号完全一致。

## 算子约束限制

- `lda>=max(1,n)`，`incx/incy` 非零且非 `INT32_MIN`；超出实现可表示地址范围的跨度报 `INVALID_VALUE`。只支持任务书指定的列主序和复数类型。
- 除 y 原地更新外，不承诺 A、x、y 之间重叠时的结果；无广播、额外视图语义或确定性保证。
- 标量允许 Host/Device；Device 标量为满足同步参数判错可能等待 stream，实际 kernel 结果仍须在读取前同步。计算过程不能访问未指定三角，不能依赖对角虚部。
- 大尺寸计算有 workspace 和设备资源约束；实际分块与资源不足处理须经平台验证。

# 可维可测分析

## 精度标准/性能标准

**精度**：使用 Netlib BLAS 3.10.0 的 `cblas_chemv`、`CblasColMajor` 生成 golden，全向量按实部、虚部分别比对。有限值逐元素 `abs(actual-golden)<=2^-13+2^-13*abs(golden)`；有限值匹配率至少 0.99，最大绝对误差满足任务书 `1e-2 or 32*ULP` 上限。非有限值单独统计：同为 NaN、同号 Inf、或一方 NaN 另一方任意 Inf 均视为通过；有限与非有限不匹配，所有非有限元素须通过。测试同时保留框架的 MERE/MARE 统计，不以其替代任务书标准。

**覆盖**：运行随任务提供的 1000 个精度用例和 200 个性能用例，补充 `GAUSS_NORM_5_2` 正态输入并记录数量与覆盖；覆盖 UPPER/LOWER、0/1、质数与 2 的幂及其邻值、尾块、lda padding、步长 ±1/±2/±3、标量 0/1/纯虚数、NaN/Inf、空指针与非法参数。以未引用三角和对角虚部设置特殊值验证不会影响输出；beta=0 时 y 旧值填 NaN；负向 case 用单元素 dummy 或空指针，避免错误参数导致测试本身分配溢出。单测检查状态码与校验顺序，包括 n=0 覆盖非法参数、INT32_MIN 和跨度溢出；测试 Host/Device 标量及 stream 顺序。

**性能**：在 910B3 对 `gpu_baseline.csv` 每个 case，使用 `msprof op` 获取对应算子 task duration（us），5 次 warmup 后采集不少于 10 次成功调用的有效样本并取算术平均；异常样本剔除并记录。`gpu_ms` 乘 1000 转换为 us，NPU 平均耗时须不高于对应基线的 1.25 倍。报告逐 case 输入参数、样本数、平均值、基线和通过情况，必要时结合 profiling 调优。测试包 README 中 event 测时口径与任务书不一致，本设计以任务书指定的 msprof 验收口径为准，event 值供辅助分析。

## 兼容性分析

共享 API 加入现有头文件，不建立私有平行接口，签名与已有产品线共同使用；实现及测试按 arch22 目录、CMake 和句柄/stream 约定接入，不改变其他架构的行为。计算对标 cuBLAS/Netlib 的列主序、负步长、三角存储与标量语义；精度以任务书容差判断，不要求跨设备位级一致。按仓库规范提供算子 README、复现步骤、精度和性能自测报告及日志；实际硬件验证后才能声明验收通过。
