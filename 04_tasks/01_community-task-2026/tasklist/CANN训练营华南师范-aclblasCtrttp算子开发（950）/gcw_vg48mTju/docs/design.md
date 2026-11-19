# 需求背景（required）

## 需求来源

[11月社区任务：aclblasCtrttp 算子开发（950）](https://www.hiascend.com/activities/task-center/details/ca73d7664ff84e8eb889157b964b5c4e?menu=guide)。基于 [cann/ops-blas](https://gitcode.com/cann/ops-blas) 工程，在 Ascend 950PR 上使用 Ascend C 实现单精度复数三角矩阵压缩，接口与 cuBLAS `cublasCtrttp` 对齐。

## 背景介绍

三角矩阵的标准存储为列主序二维矩阵，包含有效三角、另一半三角及前导维 padding；packed 存储按列连续保存有效三角，只需 `n*(n+1)/2` 个元素。

ops-blas 的 `blas/trttp/arch35/` 已有同族实数算子 `aclblasStrttp`。本任务在该目录补齐复数版本 `aclblasCtrttp`，复用公共复数类型、句柄、stream 和构建框架。

算子只改变数据布局。每个复数的实部和虚部均按原始位模式搬运，计算过程由三角区域定位与 GM—UB—GM 数据搬运组成。

# 需求分析（required）

## 需求描述

将列主序三角矩阵 `A` 中由 `uplo` 指定的区域，按列压缩到连续输出 `AP`。公开接口声明于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasCtrttp(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* A, const int lda, aclblasComplex* AP);
```

| 参数 | 位置与方向 | 含义与约束 |
| --- | --- | --- |
| handle | Host，输入 | ops-blas 有效句柄，提供执行 stream |
| uplo | Host，属性 | `ACLBLAS_UPPER` 取上三角；`ACLBLAS_LOWER` 取下三角 |
| n | Host，属性 | 矩阵阶数，`n>=0` |
| A | Device，只读输入 | 列主序 complex64 矩阵，存储跨度为 `lda*n` 个复数 |
| lda | Host，属性 | 前导维，`lda>=max(1,n)` |
| AP | Device，独立输出 | 连续 packed 数组，长度为 `n*(n+1)/2` 个复数，与 A 不重叠 |

`aclblasComplex` 使用 `include/cann_ops_blas_common.h` 的公共定义，实部和虚部各为 FLOAT32，每个元素共 8 字节。

## 需求拆解

| 项目 | 设计要求 |
| --- | --- |
| 存储语义 | 按列依次输出指定三角，分别计算 UPPER、LOWER 的 packed 偏移 |
| 输入引用 | 只访问指定三角，跳过另一三角和 `lda` padding |
| 数据精度 | complex64 逐位搬运，保留 NaN 载荷、Inf 符号及正负零 |
| 索引计算 | 元素总数、列起点和设备地址偏移使用 64 位整数 |
| 空矩阵 | 形状校验通过后，`n=0` 返回 SUCCESS，不访问 A、AP，不启动 kernel |
| 异步执行 | 使用 handle 绑定的 stream 发射 kernel，调用方在读回结果前同步 stream |
| 工程接入 | 公共 API 与其他产品线共用，arch35 实现接入同族目录 |
| 运行环境 | Ascend 950PR，CANN 9.1.0 |

# 详细设计（required）

## 算子分析

### 数学公式

以下行号 `i`、列号 `j` 均从 0 开始。输入元素地址为：

```text
A(i,j) = A[j*lda+i]
P = n*(n+1)/2
```

以 `S(j)` 表示第 j 列在 packed 数组中的起点：

| 模式 | 有效行范围 | 列起点 S(j) | 输出映射 |
| --- | --- | --- | --- |
| UPPER | `0<=i<=j` | `j*(j+1)/2` | `AP[S(j)+i] = A[j*lda+i]` |
| LOWER | `j<=i<n` | `j*(2*n-j+1)/2` | `AP[S(j)+i-j] = A[j*lda+i]` |

两种模式均满足 `S(0)=0`、`S(n)=P`。每列的有效元素在输入中连续，在输出中与下一列有效元素首尾相接。

以 `n=3` 为例，设 `aij` 表示第 i 行、第 j 列元素：

```text
UPPER: AP = [a00, a01, a11, a02, a12, a22]
LOWER: AP = [a00, a10, a20, a11, a21, a22]
```

### 支持数据类型

输入和输出均为 complex64。Kernel 将一个复数视为两个相邻的 `uint32_t`，直接搬运原始位模式，不进行浮点计算、类型转换或共轭操作。

### 支持形状

`n`、`lda` 为运行时参数，输入按 `[lda,n]` 的列主序布局访问，输出为 `[P]` 的连续数组。覆盖连续矩阵、带前导维 padding 的矩阵及空矩阵；任务用例的 n 覆盖 0 至 8192。

## 算子实现

### 实现方案

使用 Ascend C kernel 直调方式，由 Host 完成参数校验和 tiling，发射一个 AIV kernel。各核处理互不重叠的 packed 输出区间，核内沿列顺序搬运。数据通路为 `A(GM) → UB → AP(GM)`。

#### 3.2.1 host侧设计

**参数处理顺序**

1. 检查 handle；空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 依次检查 `n<0`、uplo 枚举、`lda<max(1,n)`；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n=0` 时返回 `ACLBLAS_STATUS_SUCCESS`。此时允许 A、AP 为空，但仍要求 `lda>=1`。
4. 非空矩阵检查 A、AP 非空且存储区间不重叠；不满足时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. 获取平台 AIV 核数并生成 tiling，通过 handle 的 stream 调用 kernel。

Host 不扫描矩阵元素，不分配额外 GM 工作区，不在 API 内同步 stream。

**分核策略**

三角矩阵每列长度不同，搬运成本同时包含有效字节数和每列 DMA 的固定开销。采用加权成本划分，使输出区间的划分同时考虑元素数量与列数量。

设平台可用 AIV 核数为 `C_hw`，启动核数为 `C=min(C_hw,n)`；每列固定成本权重为 `W=2048`，以复数元素数为计量单位。定义前 j 列的累计成本：

```text
F(j) = S(j) + W*j
Q = ceil((P + W*n) / C)
```

Host 将 `Q` 作为 `costPerCore` 传给 kernel。第 k 个核边界由以下规则确定：

```text
J(k) = 满足 F(j)>=k*Q 的最小 j；查找范围为 [0,n]
B(k) = min(ceil(S(J(k))/4)*4, P)
```

当目标成本超过 `F(n)` 时，边界取 `P`。核 k 处理半开区间 `[B(k),B(k+1))`；空区间直接结束。

边界以 4 个复数为单位向上对齐，对应 32 字节。`B(k)` 单调不减，且 `B(0)=0`、`B(C)=P`，因此各核覆盖完整输出且没有重叠。边界对齐可能切分列头或列尾，由核内列定位逻辑处理。

**Tiling 数据**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| packedElements | uint64_t | packed 输出元素数 P |
| costPerCore | uint64_t | 每核目标成本 Q |
| n | uint32_t | 矩阵阶数 |
| lda | uint32_t | 输入前导维 |
| upper | uint32_t | 1 表示 UPPER，0 表示 LOWER |

两种三角模式共用一个 kernel，通过 `upper` 选择列起点公式，不设置额外 tiling key。

#### 3.2.2 kernel侧设计

**初始化与列定位**

Kernel 类型设置为 `KERNEL_TYPE_AIV_ONLY`。每核读取 tiling，绑定输入、输出 GM 张量，申请 UB 队列，并计算自身输出区间。

设区间起点为 `offset`，通过二分查找确定首列 j，使：

```text
S(j) <= offset < S(j+1)
```

该核在首列内的偏移为 `offset-S(j)`。后续按列递增处理，避免对每个元素单独求解三角索引。

**分块搬运**

单 tile 容量为 8192 个复数，即 64 KiB。每次实际搬运元素数取三个约束的最小值：

```text
count = min(S(j+1)-offset, coreEnd-offset, 8192)
source = j*lda + (upper ? 0 : j) + offset-S(j)
```

执行以下步骤：

1. 从 UB 队列申请张量。
2. 使用 `DataCopyPad` 从 `A[source]` 搬入 `count*8` 字节。
3. 张量入队、出队，完成搬入与搬出的依赖同步。
4. 使用 `DataCopyPad` 向 `AP[offset]` 写回 `count*8` 字节。
5. 释放张量，推进 offset；到达 `S(j+1)` 时切换到下一列。

每次输入请求都限制在当前列的有效三角片段内。UB 尾部补齐至 32 字节，补齐数据只存在于 UB；GM 写回长度始终为实际有效字节数。

**缓冲与同步**

使用深度为 2 的 `TQueBind<VECIN,VECOUT>`，单缓冲 64 KiB，共使用 128 KiB UB 数据空间。队列管理缓冲复用及数据依赖，各核独立处理自己的输出区间，不使用原子操作和跨核同步。

**性能设计**

有效输入和输出各为 `8*P` 字节，总有效流量为 `16*P` 字节。设计重点为：

- 按加权成本分核，兼顾不同列长度及列级搬运开销。
- 在每核连续输出区间内顺序遍历列，减少标量索引计算。
- 以 tile 方式搬运较长列，使用精确长度 DMA 处理短列与尾块。
- 使用整数位模式搬运，省去复数拆分、重组和浮点运算。

### 工程组织

| 路径 | 内容 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增 aclblasCtrttp 公共声明 |
| `blas/trttp/arch35/ctrttp_host.cpp` | 参数校验、tiling 和 stream 调度 |
| `blas/trttp/arch35/ctrttp_tiling_data.h` | Tiling 结构与分块常量 |
| `blas/trttp/arch35/ctrttp_kernel.cpp` | AIV kernel、列定位和数据搬运 |
| `blas/trttp/README.md` | 算子功能、接口约束及产品支持信息 |
| `test/trttp/ctrttp/arch35/ctrttp_test.cpp` | CSV 驱动 GTest 与边界用例 |
| `test/trttp/ctrttp/arch35/ctrttp_golden.h` | 独立 CPU 参考实现 |
| `test/trttp/ctrttp/arch35/ctrttp_test.csv` | 任务配套功能、精度及性能用例 |
| `test/trttp/ctrttp/README.md` | 构建、自验及性能采集步骤 |

源文件沿用仓库按算子文件名前缀及架构筛选的构建方式，测试接入现有 GTest 框架。

## 支持硬件

| 芯片版本 | 支持情况 | CANN 版本 |
| --- | --- | --- |
| Ascend 950PR（arch35） | 支持 | 9.1.0 |

## 算子约束限制

输入和输出使用 complex64，A 按列主序及 lda 访问，AP 连续存储；不涉及广播、任意 stride 或原地转换。调用方保证输入输出存储容量，并在 stream 执行完成前保持内存有效。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判定方法 | 标准来源 |
| --- | --- | --- |
| Golden | 按列、按有效行依次复制的独立 CPU 实现，与 kernel 的分核和二分定位逻辑独立 | 任务书 §3.1、§3.2 |
| 精度 | AP 前 P 个复数的实部、虚部逐位相等，包含 NaN 载荷、Inf 符号及正负零 | 任务书 §3.2 |
| 写入范围 | AP 尾部预置哨兵，执行后逐位检查保持不变 | 任务书 §3.2、§3.5 |
| 性能采集 | Ascend 950PR、CANN 9.1.0，使用 msprof op 采集 kernel task duration；预热 5 次后，每例至少 10 个有效样本取算术平均 | 任务书 §3.3 |
| 单例性能 | 每个性能用例的 GPU 标杆耗时 / NPU 平均耗时 >=0.4 | 任务书 §3.3 |
| 整体性能 | 全部 200 个性能用例的倍率算术平均 >=0.5 | 任务书 §3.3 |
| 内存 | 任务书不设独立内存指标；设计使用固定 UB 缓冲和独立 AP 输出，无额外 GM 工作区 | 任务书 §3.4 |

### 用例覆盖

采用随任务提供的 1000 条功能与精度用例、200 条性能用例，通过 CSV 加载到 GTest。输入数据及种子按附件生成规则处理，实部、虚部分别生成；均匀分布和 `GAUSS_NORM_5_2` 正态填充沿用配套用例定义。

功能验证覆盖 UPPER、LOWER，`n=0`、1、小质数、2 的幂及其相邻值、非对齐规模，`lda=n` 与带 padding 的布局，以及非法枚举、负 n、无效 lda、空句柄、空指针和输入输出重叠。

针对 `n=1、2、3、5` 穷举两种三角的 packed 顺序。额外检查特殊浮点位模式、空矩阵不修改输出、相邻输入输出缓冲区，以及另一三角与 padding 填充 NaN 后的输出一致性。重复调用同一输入验证确定性。

输出有效区预先填入与期望值不同的位模式，结合尾部哨兵检查漏写和越界写；参考实现使用 `memcpy` 保留复数位模式。

### 性能数据处理

按 `(n,uplo,lda)` 关联 `gpu_baseline.csv`，将 `gpu_ms` 乘以 1000 转换为微秒。NPU 单例耗时取设备 kernel task duration 的算术平均，按 `GPU_us/NPU_us` 计算倍率。采集过程记录命令、环境、用例标识和原始 profiler 数据，核对 200 个用例及每例有效样本数量。

## 兼容性分析

新增公共 `aclblasCtrttp` 声明及 arch35 实现，沿用既有 `aclblasComplex`、句柄、stream 和状态码。保持同族实数 `aclblasStrttp` 接口及实现不变，其他产品线复用同一公共 API 声明。

## 参考资料

1. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)。
2. [cuBLAS cublasCtrttp 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trttp)。
3. [Netlib LAPACK ctrttp 参考实现](https://www.netlib.org/lapack/lapack_routine/ctrttp.f)。
4. 社区任务附件《aclblasCtrttp 算子开发任务书》及配套测试用例。
