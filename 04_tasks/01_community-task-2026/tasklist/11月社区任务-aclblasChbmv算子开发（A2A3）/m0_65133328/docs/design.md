# 需求背景（required）

## 需求来源

社区任务：`aclblasChbmv` 算子开发（A2A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，运行平台为 Atlas A2/A3（910B3，arch22），软件环境为 CANN 9.1.0。

## 背景介绍

### aclblasChbmv 算子实现

`aclblasChbmv` 是 BLAS Level-2 的单精度复数 Hermitian 带状矩阵-向量乘法接口，计算

\[
y=\alpha A x+\beta y
\]

其中 \(A\) 为 \(n\times n\) Hermitian 带状矩阵，具有 \(k\) 条次对角线和 \(k\) 条超对角线；\(x,y\) 为长度 \(n\) 的复数向量；\(\alpha,\beta\) 为 complex64 标量。

接口语义与 cuBLAS `cublasChbmv` 对齐。数值参考采用 Netlib BLAS 3.10.0 的双精度复数接口 `cblas_zhbmv(CblasColMajor, ...)`：complex64 输入提升到 complex128 后计算，结果收窄到 float32。

### 参考实现现状

ops-blas 当前已有实数对称带状矩阵-向量乘 `aclblasSsbmv`，但仓内没有 `aclblasChbmv` 的公共声明和 arch22 实现。Chbmv 不能通过调用 Ssbmv 或 Csymv 得到正确结果：

- Ssbmv 只处理实数，没有虚部和共轭镜像。
- Csymv 处理复数对称矩阵，未存储侧不做共轭，且对角虚部参与计算。
- Chbmv 的未存储侧满足 \(A(i,j)=\overline{A(j,i)}\)，对角元素只使用实部。

本任务复用 Ssbmv 的目录、句柄、stream、构建和 CSV/GTest 体系，新增独立的复数带状数值内核。

### 功能分析

| 参数 | 参数含义 | 数据类型 | 约束 |
|---|---|---|---|
| handle | ops-blas 上下文 | `aclblasHandle_t` | 非空 |
| uplo | 存储三角 | `aclblasFillMode_t` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | 方阵阶数和向量逻辑长度 | int | \(n\ge 0\) |
| k | 半带宽 | int | \(k\ge 0\) |
| alpha | 复数缩放系数 | complex64，Host 指针 | 非空 |
| A | 列主序带状矩阵 | complex64，Device 指针 | `lda >= k+1` |
| lda | A 的前导维 | int | 列间跨度为 `lda` 个复数 |
| x | 输入向量 | complex64，Device 指针 | 非空，`incx != 0` |
| incx | x 步长 | int | 可正可负，拒绝 `INT32_MIN` |
| beta | y 缩放系数 | complex64，Host 指针 | 非空 |
| y | 输入输出向量 | complex64，Device 指针 | 非空，原地更新 |
| incy | y 步长 | int | 可正可负，拒绝 `INT32_MIN` |

返回值为 `aclblasStatus_t`。`n=0` 在 handle 判空后直接返回 `ACLBLAS_STATUS_SUCCESS`。`n>0` 时非法 uplo、`k<0`、`lda<k+1`、零步长、`INT32_MIN` 步长、步长跨度超出 int32 表示范围以及非 handle 空指针统一返回 `ACLBLAS_STATUS_INVALID_VALUE`。

# 需求分析（required）

## 需求描述

在 ops-blas 中新增唯一公共接口：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

实现使用 Ascend C kernel 直调方式，经 handle 绑定 stream 异步下发。代码位于 `blas/sbmv/arch22/`，测试位于 `test/sbmv/chbmv/arch22/`。

## 需求拆解

| 项目 | 实现要求 |
|---|---|
| 数据类型 | A、x、y、alpha、beta 均为 complex64；中间乘加使用 float32 |
| 存储语义 | A 按列主序带状格式存储，只引用 uplo 指定三角 |
| Hermitian 镜像 | 未存储侧使用共轭元素，虚部取反 |
| 对角 | 只读取存储对角元素的实部 |
| 带宽 | \(k>n-1\) 时按有效带宽 \(\min(k,n-1)\) 计算 |
| 步长 | 支持正负 incx/incy，物理下标按 int64 计算 |
| beta 为零 | 不读取旧 y |
| alpha 为零 | 执行 \(y=\beta y\)，允许读取 A/x 但结果不受其影响 |
| 异步 | 使用 handle 当前 stream，不在 Host 同步或读回结果 |
| 验证 | 随任务提供的 1000 条精度用例和 200 条性能用例 |

# 详细设计（required）

## 算子分析

### 数学公式与带状索引

UPPER 存储：

\[
A(i,j)=A_{\mathrm{band}}[k+i-j+j\cdot lda],\quad
\max(0,j-k)\le i\le j
\]

LOWER 存储：

\[
A(i,j)=A_{\mathrm{band}}[i-j+j\cdot lda],\quad
j\le i\le \min(n-1,j+k)
\]

对于未存储的 \(A(i,j)\)，实现读取另一侧的 \(A(j,i)\) 并取共轭。对角位置只使用实部。

输出行 \(r\) 的有效列为

\[
\max(0,r-k)\le j\le\min(n-1,r+k)
\]

实现按偏移 \(e=j-r\in[-k,k]\) 遍历。每条偏移对角线在 A 的带状存储中具有固定列内起点，并以 `lda` 为跨列步长；x 侧是一段连续窗口加固定错位。该性质用于向量化 Gather 和批量 FMA。

### 支持数据类型

所有输入输出均为 complex64，实部、虚部各为 float32。实现将复数交错存储拆成两个 float32 平面后计算：

\[
z_r=a_r x_r-a_i x_i,\qquad
z_i=a_r x_i+a_i x_r
\]

未存储侧先令 \(a_i=-a_i\)，再执行同一复乘。

### 支持形状

A 物理容量为 `lda*n` 个复数；x、y 的物理容量分别至少为

\[
1+(n-1)|\mathrm{incx}|,\qquad
1+(n-1)|\mathrm{incy}|
\]

算子不使用广播，也不要求编译期 dynamic shape。常规调用不申请额外 workspace；`x` 与 `y` 为同一指针且 `alpha != 0` 时，Host 通过 handle 自有 workspace 保存 x 快照，保证原地调用读取的是调用前 x。所需快照为 `1+(n-1)|incx|` 个 complex64，workspace 不足时由 handle 默认 workspace 机制扩容。

## 算子实现

### Host 侧设计

Host 实现位于 `blas/sbmv/arch22/chbmv_host.cpp`，流程如下：

1. 先检查 handle 是否为空，再检查 \(n\ge 0\)。
2. \(n=0\) 时直接返回成功，不读取 uplo、k、lda、指针或步长。
3. 依次检查 uplo、k、lda、incx、incy、int32 步长跨度、alpha/beta/A/x/y 非空。
4. 将 \(k\) 截断为有效带宽 \(\min(k,n-1)\)。
5. 计算 A/x/y 的物理跨度和 handle workspace 大小。
6. 计算动态分块、边界任务和分核数量，选择向量路径或通用路径。
7. 填写 `ChbmvTilingData` 并异步下发。

`lda<k+1` 使用 int64 比较，避免 \(k=INT32_MAX\) 时 `k+1` 溢出。负步长绝对值在 int64 域计算，显式拒绝 `INT32_MIN`。

#### 分核策略

向量主体按 64 行 chunk 执行；大规模问题使用 128 行宽任务，在一次偏移遍历中计算两个 64 行窗口。Host 根据问题规模选择 `blockRows`：

| 条件 | `blockRows` |
|---|---:|
| \(n<1024\) | 64 |
| \(1024\le n<4096\) | 64 |
| \(n\ge4096\) | 128 |

LOWER 的首段为不包含完整两侧带形的边界区。Host 取

\[
r_{\mathrm{boundary}}=\min(n,k_{\mathrm{eff}}),
\]

将 `[0,r_boundary)` 按 1 行一个任务交给边界路径；其余行从第 \(k_{\mathrm{eff}}\) 行开始执行无前缀的向量主体。LOWER 边界行分别计算已存储的下三角部分和由 A 下三角共轭得到的镜像部分，使用 DMA、Gather、复数乘和 `ReduceSum` 完成单行点积；UPPER 边界行使用标量点积。边界任务和向量任务都按 block index 网格跨步领取。一个输出行只由一个 block 写回，不需要原子操作或跨核同步。

\[
\text{taskCount}=r_{\mathrm{boundary}}+
\left\lceil(n-r_{\mathrm{boundary}})/\text{blockRows}\right\rceil,
\]

\[
\text{blockCount}=\min(\text{taskCount},\text{AIV core count}).
\]

#### 路径选择

| 条件 | 路径 |
|---|---|
| `incx==1 && incy==1 && beta==(0,0)`、`n>=64` 且 UB 预算满足上限 | 向量主体；LOWER 的边界行使用向量点积，UPPER 的边界行使用标量点积 |
| 其他情况或 UB 预算超限 | 逐行通用路径 |
| `alpha==(0,0)` | 跳过 A/x 计算，只执行 beta 缩放 |

正式 200 条性能用例均为 `incx=incy=1`、`beta=(0,0)`，满足向量路径入口条件；其中 LOWER 大带宽用例同时包含向量边界行和宽任务主体。非连续步长或非零 beta 用例由通用路径保证功能正确性。

### Kernel 侧设计

Kernel 实现位于 `blas/sbmv/arch22/chbmv_kernel.cpp`。

#### 向量路径

每个 64 行或 128 行向量任务执行以下步骤：

1. 将本块覆盖的 x 窗口搬入 UB。窗口起点按 8 个 float 向下对齐，范围为

   \[
   [\max(0,\operatorname{alignDown}(r_0-k_{\mathrm{eff}}-8,8)),\
    \min(n,r_1+k_{\mathrm{eff}})).
   \]

2. 使用 Gather 把交错存储的 complex64 拆成实部、虚部两个平面。
3. 一次性搬入本任务所需的 A 连续 span。LOWER 的跨度按 `blockRows + 2*k_eff` 列预留，覆盖两侧带形引用；超出 UB 预算时使用标量路径。
4. 对偏移 \(e=-k_{\mathrm{eff}},\ldots,k_{\mathrm{eff}}\)：
   - 从 A span 使用 Gather 取出一条错位对角线；
   - 无效前缀在 UB 内补零并保持 32B 对齐，再进入向量运算；
   - 使用 Gather 拆成 A 实部和 A 虚部；
   - 对未存储侧取虚部相反数；
   - 从 x 窗口取得对应连续切片；
   - 执行复数乘加并累加到本行的实部、虚部累加器。
5. 对角偏移 \(e=0\) 只使用 A 实部。
6. 将实部、虚部交错重排，一次性写回 y。

UB 主要缓冲包括 x 窗口、A span、A 实虚平面、x 切片、输出累加器、临时复数乘加缓冲和重排索引。Host 用与 Kernel 一致的预算公式判断路径可用性。

#### 通用路径

对非单位步长、非零 beta 或 UB 放不下的形状，每个核以行网格跨步处理输出。每行直接按 \(e=-k_{\mathrm{eff}},\ldots,k_{\mathrm{eff}}\) 读取 A/x，计算复数乘加，再用 beta 更新 y。LOWER 向量路径的标量边界任务也使用短行段输出、UB 重排和 `DataCopyPad` 写回，避免多个核直接写相邻 y 区域后执行全 cache 清理造成相互覆盖。

#### 复数运算

交错 complex64 在向量侧转换为 planar float32。每次复乘使用四次乘法、一次减法和一次加法。A 的镜像侧只对虚部执行一次 `Muls(-1)`。非对角项按 A/x 原值计算，不提前转为双精度。

#### 非有限值

实现不通过 `0*Inf` 构造结果：

- 越界 lane 不进入 Gather 的有效长度，不参与结果累加。
- `beta=(0,0)` 时没有 y 旧值载入和乘法。
- `alpha=(0,0)` 时没有 A/x 乘加。
- `alpha=(0,0), beta=(1,0)` 时 Host 在完成全部参数校验后直接返回成功。

### 工程组织

| 位置 | 内容 |
|---|---|
| `include/cann_ops_blas.h` | 新增唯一公共接口声明 |
| `blas/sbmv/arch22/chbmv_host.cpp` | 参数校验、tiling、stream 下发 |
| `blas/sbmv/arch22/chbmv_tiling_data.h` | 共享 tiling 数据 |
| `blas/sbmv/arch22/chbmv_kernel.h` | kernel launch 声明 |
| `blas/sbmv/arch22/chbmv_kernel.cpp` | 向量路径与通用路径 |
| `test/sbmv/chbmv/arch22/chbmv_test.cpp` | CSV 驱动 GTest、边界用例、golden 比对 |
| `test/sbmv/chbmv/arch22/chbmv_test.csv` | 随任务提供的 1200 条用例 |
| `test/sbmv/chbmv/arch22/README.md` | 构建、运行和复现说明 |
| `blas/sbmv/README.md` | Chbmv 接口、参数和产品支持说明 |

构建沿用 ops-blas 的 `--soc=ascend910b3 --ops=chbmv` 单算子流程。测试通过 `ops_blas_add_gtest_tests` 注册。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas A2 训练系列产品 | √ |
| Atlas A3 训练系列产品 | √ |

验证硬件使用 Atlas 800T A2 (910B3)，CANN 9.1.0。

## 算子约束限制

- 数据类型仅 complex64。
- A 使用列主序 Hermitian 带状格式，`lda>=k+1`。
- `incx`、`incy` 非零，拒绝 `INT32_MIN`，且 \((n-1)|\mathrm{inc}|\) 不超过 int32 表示范围。
- 不支持广播、batch 或转置参数。
- alpha、beta 为 Host 标量指针；A、x、y 为 Device 指针。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_zhbmv`，CblasColMajor；complex64 输入提升到 complex128，结果收窄到 float32 | 任务书 §3.1、§3.2 |
| 逐元素容差 | FLOAT32：\(rtol=2^{-13}\)、\(atol=2^{-13}\)，实部、虚部分别判定 | 任务书 §3.2 |
| matched_ratio | 不低于 0.99 | 任务书 §3.2 |
| max_abs_error | 不大于 `max(1e-2, 32*ULP)` | 任务书 §3.2 |
| 非有限值 | NaN/NaN、同号 Inf、NaN 与 Inf 视为等效；有限与非有限判失败 | 任务书 §3.2 |
| 性能 | 200 条性能用例逐例平均 kernel 时间不超过 GPU 基线的 1.25 倍 | 任务书 §3.3 |
| 采样 | msprof op，warmup 5 次，有效样本不少于 10 次取平均 | 任务书 §3.3 |

### 测试设计

CSV 固定为 1200 条：

- 精度：排除 `TC_PF_*` 后 1000 条。
- 性能：`TC_PF_*` 200 条。
- 精度与性能分进程执行、分别判定。

测试代码按 uplo、n、k、lda、incx、incy 原位构造 A/x/y。A 的带内元素按 Hermitian 约束生成，对角虚部置零；未存储侧不参与 golden。输出按 incy 提取逻辑 \(n\) 个位置，实部、虚部分别统计 matched ratio、最大绝对误差和非有限值。

CSV 加载在测试启动时校验总行数 1200、精度 1000、性能 200、必需列齐全，以及 case name 非空且唯一。1000 条精度用例在运行时各生成一条 `_NORM` 正态镜像，连同 3 条手写校验共执行 2003 条正式精度测试。

补充用例覆盖：

- 空 handle；
- uplo/n=0 校验顺序；
- `INT32_MIN` 与跨度溢出；
- `alpha=0`、`beta=0` 特殊语义；
- 正态分布输入；
- 前导维 padding 与正负步长；
- 未引用三角、padding 区域的毒值检查；
- A2/A3 产品支持声明。

### 复现方式

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DSOC_VERSION=ascend910b3 -DBUILD_TEST=ON -DTEST_NAMES=chbmv
cmake --build build -j32
./build/test/sbmv/chbmv/chbmv_test --gtest_filter='*-*TC_PF*'
```

性能使用 msprof op 采集 `Task Duration(us)`，并与 `gpu_baseline.csv` 逐 case 对照。

## 兼容性分析

新算子只新增 `aclblasChbmv` 公共声明和 arch22 实现，不修改 `aclblasSsbmv`、`aclblasCsymv` 等既有 ABI。非法 uplo 按本任务书返回 `ACLBLAS_STATUS_INVALID_VALUE`，不改动其他复数接口的既有返回口径。
