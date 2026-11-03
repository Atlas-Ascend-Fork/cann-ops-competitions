# 需求背景（required）

## 需求来源

社区任务：aclblasChbmv 算子开发（950）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，交付形态为 Ascend C kernel 直调，运行平台 Ascend 950PR / CANN 9.1.0，芯片架构 arch35。

## 背景介绍

### aclblasChbmv 算子实现路径

本算子为 BLAS 接口新增实现，直接以 Ascend C SIMT 方式开发，接口语义对齐 cuBLAS 的 `cublasChbmv`。

实现路径与相关 API 路径：

- 接口声明：`include/cann_ops_blas.h`（仓内唯一公共声明）
- Host 实现：`blas/sbmv/arch35/chbmv_host.cpp`
- Tiling 结构：`blas/sbmv/arch35/chbmv_tiling_data.h`
- Kernel 实现：`blas/sbmv/arch35/chbmv_kernel.cpp`
- SIMT 运行时接口：`simt_api/asc_simt.h`（`asc_vf_call`、`__simt_vf__`、`LAUNCH_BOUND`）
- 公共 SIMT 常量：`blas/common/helper/kernel_constant.h`（`SIMT_MIN_THREAD_NUM=128`、`SIMT_MAX_THREAD_NUM=2048`）
- 测试与对照：`test/sbmv/chbmv/arch35/`（GTest 与测试用例 CSV）、`test/sbmv/chbmv/`（参数读取、CBLAS 对照、GPU 基线、精度验证与 msprof 采集脚本）

同族实现可参考：实数带状乘 `Ssbmv` 位于 `blas/sbmv`，复数 Hermitian 秩二更新 `Cher2` 位于 `blas/syr2`。Chbmv 同时叠加"带状存储"与"共轭镜像"两种约束，因此复用二者的句柄、stream、复数类型与构建框架，但**不复用**其数值内核。

### aclblasChbmv 参考实现现状分析

现状基线取两条权威参考：

- 接口语义：cuBLAS `cublasChbmv`
- 数值 Golden：Netlib BLAS 3.10.0 `cblas_chbmv(CblasColMajor, ...)`

参考实现的参数能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 算子句柄 | 结构体指针 | aclblasHandle_t | 非空 | - |
| uplo | 三角标识 | 枚举 | ACLBLAS_UPPER / ACLBLAS_LOWER | 仅这两个取值 | - |
| n | 矩阵阶数 | int | int32 | n >= 0；n == 0 直接返回成功 | 标量 |
| k | 上/下带宽 | int | int32 | k >= 0；k 可大于 n-1 | 标量 |
| alpha | 复数标量 | complex64 | float32 实虚部 | Host 指针，非空 | 标量 |
| A | 带状矩阵 | complex64 | float32 实虚部 | lda >= k+1；按列主序 `lda x n` 存储 | lda * n |
| lda | A 的主维 | int | int32 | >= k+1；lda * n 不得溢出 size_t | 标量 |
| x | 输入向量 | complex64 | float32 实虚部 | 非空；步长非零 | 1 + (n-1)*abs(incx) |
| incx | x 步长 | int | int32 | 非零，可为负 | 标量 |
| beta | 复数标量 | complex64 | float32 实虚部 | Host 指针，非空 | 标量 |
| y | 输入输出向量 | complex64 | float32 实虚部 | 非空；原地更新 | 1 + (n-1)*abs(incy) |
| incy | y 步长 | int | int32 | 非零，可为负 | 标量 |

计算公式：`y := alpha * A * x + beta * y`，A 为 Hermitian 矩阵，仅访问 uplo 指定三角与带状范围内的元素。

### aclblasChbmv 算子功能分析

功能：完成单精度复数 Hermitian 带状矩阵与向量的乘积并累加到 y。

输入：uplo、n、k、alpha、A、lda、x、incx、beta、incy。

输出：y（原地更新，不返回视图）。

支持数据类型：complex64（实部、虚部各为 float32）。

支持广播：不支持。矩阵与向量维度已由 n 唯一确定，不存在广播语义。

特殊语义（由参考实现决定，非可选优化）：

1. `beta == (0,0)` 时不读取旧 y，旧值可以是 NaN/Inf。
2. `beta == (1,0)` 时参考实现不执行缩放（`chbmv.f` 用 `IF (BETA.NE.ONE)` 包住缩放循环），因此非有限的 y 分量必须原样保留。
3. `alpha == (0,0)` 时参考实现短路返回 `beta * y`，不求值 `0 * x`，因此 x 含 NaN/Inf 时输出仍是有限值。
4. 对角元素只使用存储实部，存储虚部不参与运算。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言（SIMT 直调路径）实现 aclblasChbmv 算子，支持 complex64 数据类型，支持上三角/下三角、正/负步长、任意合法带宽，数值与 Netlib `cblas_chbmv` 对齐，性能不低于同规模 GPU 基线的 0.4 倍。

公开接口唯一声明于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

`aclblasComplex` 沿用公共头文件的 `{float real; float imag;}`。alpha/beta 位于 Host，A/x/y 位于 Device。原地更新 y，不引入 950PR 私有接口。

## 需求拆解

| 项目 | 实现要求 |
|---|---|
| 数据类型 | 全链路 complex64，中间乘加保持 FLOAT32，不降精度 |
| 校验顺序 | n>=0 → handle 非空 → n==0 短路返回 SUCCESS → uplo/k/lda/incx/incy/alpha/beta/A/x/y |
| 非法 uplo | 返回 INVALID_VALUE（有意背离 aclblasCsymv 的 INVALID_ENUM，与同目录 aclblasSsbmv 一致） |
| 带状结构 | k>=0 且 lda>=k+1；k 大于 n-1 合法，此时仅 min(k,n-1) 条对角线有元素 |
| 向量步长 | 非零正负步长；负步长按 Netlib 物理基址约定反向取元素 |
| 三角与 padding | 仅指定三角与带内参与读写；其余存储逐字节保持 |
| 对角 | 只使用存储实部，虚部不参与运算 |
| beta 为零 | `beta=(0,0)` 时不读取旧 y，直接写结果 |
| beta 为一 | `beta=(1,0)` 时不做缩放，y 原样参与累加；与 `alpha=(0,0)` 同时出现时整次调用是恒等，Host 直接返回 SUCCESS |
| 对齐 | A/x/y 8 字节对齐时走 float2 访问，否则自动降级为 4 字节分量访问，不因对齐拒绝调用 |
| 异步 | 使用 handle 当前 stream，API 不同步设备，无 CPU fallback、无结果读回、无额外 workspace |
| 自验 | 1000 条精度 + 200 条性能，均匀/正态各 50%，经真实 C++ 接口执行 |

随包 `TC_ED_246..257` 同时含多项错误，其期望结果确定了校验优先级：`n=-1` 与 `uplo=999` 并发时先判 n 返回 INVALID_VALUE；`n=0` 与非法 uplo、空指针并发时短路返回 SUCCESS。

# 详细设计（required）

## 算子分析

### 数学公式

`y := alpha * A * x + beta * y`

其中 A 为 n 阶 Hermitian 矩阵，`A(j,i) = conj(A(i,j))`，仅带宽内元素非零。实现按行组织计算，输出行 r 由三段组成，且三段的**累加顺序**与参考实现的列扫描逐一对应：

- 存储侧：UPPER 为 `c > r` 的 `A(r,c)`（列 c 第 `k+r-c` 行），LOWER 为 `c < r` 的 `A(r,c)`（列 c 第 `r-c` 行），按列升序逐项累加；
- 对角：只取 `real(A(r,r))`，存储虚部被忽略；
- 镜像侧：UPPER 为 `c < r` 的 `conj(A(c,r))`（列 r 第 `k+c-r` 行），LOWER 为 `c > r` 的 `conj(A(c,r))`（列 r 第 `c-r` 行），单独聚合成自己的 `TEMP2` 后整体乘 alpha 加回一次。

参考实现按列扫描：行 r 在列 r 被处理时写入"对角 + alpha\*TEMP2"，在列 r+1..r+k（UPPER）或列 r-k..r-1（LOWER）被处理时逐项写入存储侧。因此 **UPPER 的顺序是 对角→镜像→存储侧，LOWER 是 存储侧→对角→镜像**。先 `y = beta*y`（`beta=(0,0)` 时不读旧 y），再进入上述三段。

这个顺序不是风格问题：若把三段并进同一个 FP32 累加器、最后统一乘 alpha，精确值不变，但几百项的舍入路径与参考实现不同，误差会随数据抵消程度放大。复刻参考的分组后，本实现的误差不再由数据的抵消程度决定，而稳定在参考自身的舍入量级。

### 支持数据类型

alpha、beta、x、y、A 统一 complex64（实虚部各 float32）。中间乘加使用 FLOAT32，不降精度。文件级 `#pragma clang fp contract(off)` 关闭乘加融合，使每一项单独舍入，与 Netlib 的逐步累加口径一致。

**对齐处理**：kernel 优先把每个复数按一次 8 字节 `float2` 访问（`reinterpret_cast<const __gm__ float2*>`），这要求 A、x、y 三个设备指针 8 字节对齐，而 `aclblasComplex` 自身只保证 4 字节对齐（`alignof(aclblasComplex) == 4`）。因此 Host 在 tiling 时探测三个基址并置 `aligned8` 标志：三者同为 8 字节对齐时 kernel 走 float2 路径，否则降级为按实部、虚部两次 4 字节访问。两条路径结果逐位相同，对齐只影响访存指令数，不构成接口契约，也不产生错误码。

绝大多数调用会命中 float2 路径：`aclrtMalloc` 返回的缓冲区对齐远大于 8 字节，`sizeof(aclblasComplex) == 8` 也使按元素偏移得到的视图保持 8 字节对齐。只有把 `float` 数组按奇数下标强转成复数视图这类做法才会落到 4 字节路径，此时结果依然正确，只是元素级 GM 请求数翻倍。

### 支持形状

A 物理跨度为 `lda*n`，元素地址 `A[i+j*lda]`；x/y 至少具有 `1+(n-1)*abs(inc)` 个复数元素。Host 检查 `lda*n` 的字节跨度不溢出 `size_t`。运行时 n 支持 0、1 及一般尺寸；k 无上界约束，k 大于 n-1 时自动退化为仅有 min(k,n-1) 条对角线。不支持广播。

## 算子实现

### 实现方案

#### host 侧设计

**tiling 策略：**

1. 按固定顺序校验参数：n>=0、handle 非空、n==0 短路返回、uplo、k、lda、incx、incy、alpha/beta 非空、A、x、y 非空。
2. 探测 A/x/y 三个设备指针的 8 字节对齐情况，写入 `aligned8` 标志，供 kernel 选择 float2 或 4 字节分量访问；不因未对齐拒绝调用。
3. 检查 `lda*n` 不溢出 `size_t`。
4. 查询平台 AIV 核数（为 0 返回 EXECUTION_FAILED），取得 handle 的当前 stream。
5. 计算 numBlocks / numThreads，填充 `ChbmvTilingData`，包含 n、k、lda、uplo、alpha/beta 实虚部、64 位 incx/incy，以及 `betaZero`、`betaOne`、`alphaZero`、`windowed`、`aligned8` 五个路径标志。
6. 一次非空调用发射一个 Kernel；`alphaZero && betaOne` 同时成立时整次调用是恒等（`y = 1*y + 0*A*x`），Host 在发射前直接返回 SUCCESS。
7. 无 CPU fallback、无 Device 结果读回、无额外 workspace。

##### 1. 分核策略

采用"一线程一行 + 按行数压块"的策略，不使用 UB 切块，因此不涉及大核/小核的 tile 均分：

- `numBlocks = min(ceil(n / SIMT_MIN_THREAD_NUM), AIV 核数)`，即块数不超过核数，且每块至少有 `SIMT_MIN_THREAD_NUM` 行可分配；
- `numThreads = min(align_up(ceil(n / numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)`，向上对齐到 128 的整数倍，上限 2048；
- 网格跨步：kernel 内以 `blockDim * gridDim` 为步长遍历行，保证任意 `numBlocks * numThreads` 与 n 的组合都正确，不需要"余数块"特判；
- 小 n 时刻意走"单块 + 最少线程"：此时主导成本是发射而非运算，多开块只会增加固定开销。

##### 2. 数据分块和访存优化策略

本算子是访存受限的 SIMT 直调实现，数据直接从 GM 读取，不经 UB 搬运，因此优化点落在"减少 GM 往返次数"与"减少每元素指令数"：

- **批宽 4 的独立载入**：同一批的 4 次 GM 载入互不依赖，先全部发射再按原顺序依次累加，把"每个元素等一次 GM 往返"的串行等待变成 4 路并发。累加顺序未变，舍入结果不受影响。批宽取 4 而非更大：8 路与 4 路在大用例上的倍率差小于 0.001（大用例已是带宽瓶颈），但 k < 8 的用例会整个落进标量尾循环、失去重叠，而 k = 4 正是小规模用例的带宽。
- **单次 8 字节复数访问**：三址同为 8 字节对齐时，每个复数按一次 `float2` 读取（仓内 `scasum` 同款写法），元素级 GM 请求数减半；未对齐时自动改用 4 字节分量访问。
- **指针步进代替下标乘加**：存储侧每前进一列移动 `lda-1` 个复数，镜像侧移动 1 个复数，内层循环不再出现 64 位乘加。
- **小规模寄存器窗口**：`n <= SIMT_MIN_THREAD_NUM` 时块数必为 1、线程数恰为 128，一行由固定线程独占；若该行带宽 `2k+1 <= CHBMV_WINDOW_MAX`（=12，覆盖 k<=5），整行连同对角线只有个位数个复数，可一次性取进寄存器再累加，把三段"三次串行 GM 往返"合并成一批互不依赖的载入。窗口宽度只为寄存器压力而设：两个窗口共 `2 * 12 * 2` 个 float，须在 128 线程块的每线程寄存器内不溢出，故该路径的 launch bound 用 `SIMT_MIN_THREAD_NUM` 而非 `SIMT_MAX_THREAD_NUM`；窗口下标全部来自编译期常量上界的展开循环，避免动态下标把整组寄存器挤进局部内存。
- **地址计算全部 64 位**：`inc>0` 取 `i*inc`，`inc<0` 取 `(n-1-i)*(-inc)`，与 Netlib 的 `KX = 1-(n-1)*abs(inc)` 一致；A 的列主序地址按 `rowIndex + lda*col` 计算，`lda` 为 32 位但参与 64 位乘。

##### 3. tilingkey 规划策略

本算子不使用数值型 tilingkey，而是用 **5 个布尔标志 + 2 个模板参数** 完成分支规划，目的是让"host 侧已知的语义信息"既不退化成 kernel 内的逐项判断，也不膨胀成 tilingkey 组合爆炸。

需要分支的原因：`uplo`、`alpha`、`beta` 的取值会改变被访问的元素集合与舍入路径（见"数学公式"与"算子功能分析"的 4 条特殊语义），若全部留到运行时判断，热循环每项都要多一次比较。

分派方式分两层：

| 条件 | 分派层 | kernel 入口 |
| --- | --- | --- |
| `alpha == (0,0)` | 运行时（标志 `alphaZero`） | `ChbmvScaleOnly`，只做 `y = beta*y`，A/x 不寻址 |
| `windowed == 1`（单块 且 2k+1<=12） | 运行时（标志 `windowed`） | `ChbmvWindow<UPPER, ALPHA_ONE>` |
| 其余 | 运行时 | `ChbmvGm<UPPER, ALPHA_ONE>` |
| `uplo == UPPER / LOWER` | 编译期（模板参数 `UPPER`） | 消除循环内三角判断，并决定三段先后 |
| `alpha == (1,0)` | 编译期（模板参数 `ALPHA_ONE`） | 跳过 `TEMP1 = alpha*x` 的逐项复乘 |

`betaZero` / `betaOne` 作为运行参数传入：前者决定是否读旧 y，后者决定是否做缩放拷贝。二者都只影响每次调用开头与结尾的一小段代码，不影响内层循环，因此不必进模板参数，避免组合数翻倍。

规模组合共 2（uplo）× 2（alphaOne）× 2（windowed）× 1（alphaZero 单独入口）= 8 个 kernel 入口，均为编译期确定，无运行时字符串或查表开销。

#### kernel 侧设计

SIMT 直调路径没有 Ascend C 的 Init / CopyIn / Compute / CopyOut 四段式，kernel 一次调用即完成全部计算，流程为：

1. `chbmv_kernel` 入口先判 `alphaZero`：成立则调用 `ChbmvScaleOnly` 后直接返回，半个带子不寻址。
2. 再判 `windowed`：成立走 `ChbmvWindow`，否则走 `ChbmvGm`；两条路径都按 `uplo` 与 `alphaOne` 在编译期选择模板实例。
3. 每个线程处理一行（网格跨步）。行 r 的列范围 `[max(0,r-k), min(n,r+k+1))` 由"存储侧""对角""镜像侧"三段覆盖，各自是连续区间，循环体内不携带三角判断。
4. 三段拆成 `AccumulateStored` 与 `AccumulateDiagAndMirror` 两个 `__simt_callee__ inline` 函数，由 `ChbmvGm<UPPER, ALPHA_ONE>` 按 uplo 决定调用顺序，避免同一段循环在两个分支里各写一遍。镜像侧在函数内用独立的 `TEMP2` 累加器，最后整体乘 alpha 加回，与参考实现的分组一致。
5. `beta` 处理在行计算之前：`betaZero` 时不读旧 y；`betaOne` 时直接拷贝旧 y（保留非有限分量）；否则做复数缩放。
6. 结果以 `make_float2(yReal, yImag)` 一次 8 字节写回。

`alpha == (0,0)` 走独立入口不是性能取舍而是语义对齐：参考实现在该情形下直接短路、不求值 `0*X(J)`，因此 x 含 NaN/Inf 时返回的是精确的 `beta*y` 而非 NaN；若照代数计算，`0*NaN` 会得到 NaN，与 golden 的有限输出不符，非有限一致性判据会判失败。该结论由 `TC_FL_213_NORMAL`（alpha 抽到 (0,0)、x 全 NaN）实测得到：修复前 actual=NaN、golden=0，修复后两侧同为 0。

寄存器窗口路径只改数据搬运：累加顺序、镜像侧独立聚合后整体乘 alpha 的分组、beta 先乘的次序都与流式路径逐字一致，因此结果逐位相同（另以离线校验程序对 222 组配置同时校验"窗口落点等于逻辑元素"与"窗口结果 == 参考 == 流式路径"）。两条路径由 host 的 `windowed` 标志选择，判据是结构性的（单块 + 窄带），不针对任何具体用例：门控覆盖 n=1/2/4/8/16 这一档，200 条性能用例中命中 5 条，其余 195 条走流式路径。

每行只有一个写者，行间无共享状态，不需要原子操作或同步。A/x 只读，y 只写自身步长覆盖的元素。

### 工程组织

| 位置 | 内容 |
|---|---|
| `include/cann_ops_blas.h` | 新增唯一公共声明 |
| `blas/sbmv/arch35/chbmv_host.cpp` | 参数合同、tiling、异步调度 |
| `blas/sbmv/arch35/chbmv_tiling_data.h` | Tiling 结构及 launch 约定 |
| `blas/sbmv/arch35/chbmv_kernel.cpp` | SIMT 实现 |
| `test/sbmv/chbmv/arch35/` | GTest 与测试用例 CSV |
| `test/sbmv/chbmv/` | 参数读取、CBLAS 对照、GPU 基线、README、精度验证与 msprof 采集脚本 |

构建复用仓内按文件名前缀与架构筛选机制，测试通过 `ops_blas_add_gtest_tests` 注册。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR | √ |

## 算子约束限制

仅支持任务约定的 complex64、列主序与 incx/incy/lda 布局；不支持广播。A/x/y 只需满足 `aclblasComplex` 的 4 字节自然对齐，三址同为 8 字节对齐时算子内部走 float2 快路径，否则自动降级，两者结果一致。alpha/beta 为 Host 指针，不做设备指针探测或 D2H 标量读取。调用方保证操作数存储与异步生命周期。k 无上界，但 lda>=k+1 且 `lda*n` 不得溢出。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| Golden | Netlib BLAS 3.10.0 `cblas_chbmv`，CblasColMajor | 任务书 §3.1/§3.2 |
| 普通精度 | 实部、虚部分别按 FLOAT32 判定，逐元素 error <= 2^-13 * (1 + \|golden\|)，matched_ratio >= 0.99 | 任务书 §3.2 |
| 绝对误差上限 | max_abs <= max(1e-2, 32*ULP) | 任务书 §3.2 |
| 非有限值 | NaN/NaN、同号 Inf 及 NaN/Inf 等效；有限/非有限不等效 | 任务书 §3.2 |
| 完整性 | 未引用三角、padding、带外单元的 NaN 载荷哨兵逐字节检查；A/x 只读检查 | 任务书 §3.2/§3.5 |
| 对角 | 存储对角虚部填 0.5 哨兵，实现若用完整复数对角立即发散 | 任务书 §2.3 |
| 性能 | 200 条逐例 GPU/NPU 平均耗时 >= 0.4；使用 msprof kernel task duration | 任务书 §3.3 |
| 采样 | 每例 5 次预热后 10 次有效采样取算术平均 | 任务书 §3.3 |

测试侧关键设计：原始 1000 条精度用例保持 CSV 参数不变，另以 `normal=true` 配对执行一遍正态分布输入，使均匀/正态各覆盖 50% 精度工作量；200 条性能用例不复制。CSV 装载时固定校验总计 1200 行、精度 1000 行、性能 200 行及名称唯一性，防止 CSV 悄悄缩水。

`InitializeMatrix` 把带宽外、padding、未存储三角全部写成 NaN 哨兵；对结构非法的负向用例（k<0、lda<k+1）不做对角哨兵写入并做边界保护，避免测试自身越界。

### 实测结果

Ascend 950PR / CANN 9.1.0 单卡实测：

- 精度 2004/2004 通过：1000 条用例 × 均匀/正态两遍 + 4 条 `TEST_F`（含 `beta=(1,0)` 非有限 y 回归与 4 字节对齐降级用例），2 个测试套件均通过；3922 条实部/虚部判定行（1961 个用例各产生实部、虚部两行；2000 次执行中有 39 条负向用例只校验返回状态码、不产生逐元素判定，故 2000-39=1961），最小 `matched_ratio` 为 **0.998043**（门限 0.99），所有 `max_abs` 均在逐例上限 `cap = max(1e-2, 32*ULP)` 内。
- 性能 200/200 达到 `0.4 × GPU`：两轮独立 msprof 采集，合并最小倍率 **0.4629**（`TC_PF_1143`，n=1258、k=876），中位 **0.7005**，最大 **1.5121**（`TC_PF_1075`，n=2048、k=0），倍率 ≥ 1 共 3 条，低于 0.4 的为 0 条。
- 小规模档（n<=128）由寄存器窗口路径覆盖，n=16、k=4 用例倍率约 0.80，n=1 用例约 0.72，均高于门限。

### 复现方式

构建与验证脚本、原始日志、逐例数据见交付材料：自验证步骤说明、精度日志、性能日志、msprof 原始 `op_summary` CSV。性能结果由 msprof 原生 op task duration 解析，排除 GTest 整例墙钟耗时与输入恢复开销。

参考：[Netlib Chbmv](https://www.netlib.org/blas/chbmv.f)、[ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)、[社区任务流程](https://gitcode.com/org/cann/discussions/39)。

## 兼容性分析

新增函数声明与 arch35 实现，复用既有公共复数类型，不修改 Ssbmv 与 Csymv 的接口及实现。其他产品线共用本次公开声明，其实现按各自任务独立接入。本任务仅声明 Ascend 950PR 支持。

需要调用方注意的一处有意差异：非法 `uplo` 在本算子返回 `ACLBLAS_STATUS_INVALID_VALUE`，而复数族 `aclblasCsymv` 返回 `INVALID_ENUM`。该差异已写在公共声明的注释上，属于任务书 §2.4/§7.5 "每个非法参数统一返回 INVALID_VALUE"的要求，不构成对既有接口的破坏。
