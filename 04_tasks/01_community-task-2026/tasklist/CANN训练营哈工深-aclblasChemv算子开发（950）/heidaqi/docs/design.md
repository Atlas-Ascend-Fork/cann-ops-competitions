# 需求背景（required）

## 需求来源

本设计对应“CANN训练营哈工深-aclblasChemv算子开发（950）”任务，在 ops-blas 中实现 Ascend 950PR 平台的单精度复数 Hermitian 矩阵向量乘接口 `aclblasChemv`。

实现目录为 `blas/symv/arch35/`，测试目录为 `test/symv/chemv/arch35/`，开发环境为 CANN 9.1.0。

## 背景介绍

### aclblasChemv算子实现优化

Hermitian 矩阵满足共轭对称关系，仅需存储上三角或下三角。计算时根据 `uplo` 读取指定三角，由共轭关系恢复另一侧元素，并将对角元素的虚部视为零。

本实现使用 Ascend C Kernel 直调。连续向量和常用系数组合采用 SIMD 向量计算，通用系数与步长采用 SIMT，实现矩阵向量乘及输出向量的原地更新。

### aclblasChemv实现现状分析

| 参数 | 含义 | 类型与位置 | 约束 |
|---|---|---|---|
| handle | BLAS上下文及执行流 | Host句柄 | 由aclblasCreate创建 |
| uplo | 矩阵存储三角 | Host枚举 | ACLBLAS_UPPER或ACLBLAS_LOWER |
| n | 矩阵阶数 | Host int | n≥0 |
| alpha、beta | 复数系数 | Host complex64指针 | n>0时有效 |
| A | Hermitian矩阵 | Device complex64指针 | 列主序，物理形状lda×n |
| lda | 前导维 | Host int | n>0时lda≥n |
| x | 输入向量 | Device complex64指针 | 逻辑长度n |
| incx、incy | 向量元素步长 | Host int | 非零且不为INT_MIN |
| y | 输入及原地输出向量 | Device complex64指针 | 逻辑长度n |

实现包含两条执行路径：

- `alpha=(1,0)`、`beta=(0,0)`、`incx=incy=1`：RegBase SIMD 路径，采用自适应行块和三缓冲搬运流水。
- 其他系数及步长组合：SIMT 路径，每个线程独立计算一个输出元素。

### aclblasChemv算子功能分析

计算 `y = alpha × A × x + beta × y`。输入、输出及标量均为 complex64，实部和虚部分别使用 float32。输出写回 y 的逻辑元素位置，支持 lda padding 和正负向量步长。

# 需求分析（required）

## 需求描述

在 Ascend 950PR / CANN 9.1.0 上实现 `aclblasChemv`，支持 Hermitian 矩阵语义、上下三角选择、带符号步长及异步执行流。

使用 Netlib BLAS/CBLAS 3.10.0 生成参考结果。性能以任务提供的 cuBLAS GPU 基线为参照，全部200项性能用例分别满足 `GPU耗时 / NPU耗时 ≥ 0.4`。

## 需求拆解

1. 在公共头文件 `include/cann_ops_blas.h` 声明接口。
2. 实现列主序寻址、共轭补全、对角虚部忽略和正负步长。
3. 按接口顺序完成句柄、维度、枚举、步长及指针校验。
4. 处理n=0、alpha=0和beta=0/1等特殊分支，减少无关数据访问。
5. 使用句柄绑定的stream异步提交Kernel。
6. 结合多核切分、三缓冲和向量化提高计算效率。
7. 完成原始1000项精度用例、200项性能用例及补充测试。

# 详细设计（required）

## 算子分析

### 数学公式

第r个输出元素为：

`y[r] = alpha × Σ(j=0…n−1) H(r,j) × x[j] + beta × y[r]`

其中 `H(r,r)=Re(A[r+r×lda])`。当 `uplo=UPPER` 时：

- r<j：`H(r,j)=A[r+j×lda]`。
- r>j：`H(r,j)=conj(A[j+r×lda])`。

当 `uplo=LOWER` 时，直接读取和共轭读取的方向相反。

向量逻辑元素i的物理偏移为：

`index(i,n,inc) = (inc<0 ? (1−n)×inc : 0) + i×inc`

调用方传入物理数组首地址。地址计算先提升为int64，再进行乘法和加法。

### 支持数据类型

使用 `aclblasComplex` 表示complex64，成员real和imag均为float32。alpha和beta使用Host标量指针，A、x和y使用Device指针。

### 支持形状

- A：逻辑形状n×n，物理存储形状lda×n，列主序。
- x、y：逻辑长度n；n>0时物理存储长度至少为 `1+(n−1)×abs(inc)`。
- n：运行时指定，n=0时直接返回。
- incx、incy：支持正负非零步长，取值不为INT_MIN。

## 算子实现

### 实现方案

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

Host、Kernel和切分参数分别位于 `chemv_host.cpp`、`chemv_kernel.cpp` 和 `chemv_tiling_data.h`。Host通过 `ChemvTilingData` 传递n、lda、uplo、行块大小、线程数、向量步长及四个标量分量。

#### 3.2.1 host侧设计：

参数处理顺序如下：

1. handle为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0时返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0时返回成功。
3. 检查uplo、lda、步长及alpha/beta指针。
4. 读取标量；alpha=0且beta=1时返回成功。
5. 检查y；alpha非零时检查A和x。
6. 获取可用AIV核数，构造切分参数，在句柄绑定的stream上启动Kernel。

非法uplo返回 `ACLBLAS_STATUS_INVALID_ENUM`；其他非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。可用AIV核数为零时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

##### 1. 分核策略：

设可用AIV核数为C。SIMD路径根据 `ceil(n/C)` 向上选择16、32或64行作为行块大小R，最大为64行。
行组数为 `G=ceil(n/R)`，启动核数为 `B=min(G,C)`。SIMT路径使用32行切分和32线程配置。

SIMD路径按行组分配任务。令 `q=G/B`、`r=G%B`：UPPER将余数组分配给前r个核，LOWER将余数组分配给后r个核，以平衡两侧不同的镜像访问量。

当q=1且r>1时，对双组核进行首尾配对：

- UPPER：核c<r处理组c和 `2r−1−c`。
- LOWER：令 `j=c−(B−r)`，双组核处理组 `G−2r+j` 和 `G−1−j`。

其他情况采用连续组分配。每个输出行由一个核独立计算并写回。

##### 2. 数据分块和内存优化策略：

列块大小为64，最大行块为64。UB矩阵行跨度为136个float，即544字节，兼顾32字节对齐及bank映射。

A和x使用深度为3的TQue。先预取两个实际列块，每轮消费当前块前预取下一块，通过FIFO和队列事件衔接搬运与计算。计算完成后释放对应槽位。

完整位于存储三角内的矩形使用带步长的 `DataCopyPad` 批量搬运；跨对角块按有效三角范围搬运。对角路径每个元素搬运一个实部float，并跳过GM中的虚部。

后缀区间在对齐后列块数不增加时，将首块结束位置对齐到64列边界。预取游标和消费游标分别按实际列数推进，覆盖短首块和尾块，同时保持列访问顺序。

最大行块的UB预算如下：

| Buffer | 单槽字节 | 槽数 | 合计字节 |
|---|---:|---:|---:|
| A列块，64×136个float | 34816 | 3 | 104448 |
| x原值及预乘值 | 1024 | 3 | 3072 |
| 对角实部及补齐 | 2048 | 1 | 2048 |
| 当前行x、输出、state、mirror | 512 | 各1，共4 | 2048 |
| 合计 | — | — | 111616 |

VECIN队列共占8个物理槽位：A×3、x×3、diag×1、xRow×1。输出使用独立方向的队列。每个核保留自己的累加状态和镜像部分，输出行之间独立。

##### 3. tilingkey规划策略：

本算子采用Kernel直调，通过 `ChemvTilingData` 中的uplo、行块大小和标量参数选择执行分支：

- 连续向量及常用系数组合选择SIMD路径。
- 通用参数选择 `ChemvCompute<true/false>` SIMT路径。
- n=0、alpha=0且beta=1的快速返回由Host完成。

#### 3.2.2 kernel侧设计：

Kernel类型为AIV_ONLY。SIMD路径通过TPipe管理搬入、计算和写回，使用Gather和DeInterleave拆分复数实虚部，在镜像路径中取共轭。

各输出元素保持参考实现的列序及计算分组：LOWER依次处理直接贡献、对角和镜像部分；UPPER依次处理对角、镜像和直接贡献。相位结束时使用PIPE_V屏障完成向量计算同步。

SIMT路径通过 `asc_vf_call` 启动32线程VF，每个线程计算一个输出行。alpha=0时跳过A/x读取，beta=0时跳过旧y读取。复数乘法使用 `fmaf(a,b,0.0f)` 分别计算乘积，再进行加减，保留float32的舍入顺序。

Kernel编译使用 `-ffp-contract=off` 和 `-fno-fast-math`。

### 性能优化与实测结果

主要优化包括自适应行块、双组核首尾配对、矩阵批量搬运、后缀列块对齐和三缓冲流水。这些优化分别改善核间负载、搬运次数与计算重叠，同时保持每个输出元素的累加顺序。

在Ascend 950PR / CANN 9.1.0上，全部200项性能用例通过。部分结果如下：

| 用例 | GPU耗时 μs | NPU前10次均值 μs | NPU全60次均值 μs | 上限 μs |
|---|---:|---:|---:|---:|
| TC_PF_1001 | 9.771000 | 7.450900 | 7.446550 | 24.427500 |
| TC_PF_1002 | 14.071000 | 13.230600 | 13.211350 | 35.177500 |
| TC_PF_1003 | 18.284000 | 36.606300 | 36.693167 | 45.710000 |
| TC_PF_1004 | 40.722000 | 87.735799 | 87.730884 | 101.805000 |
| TC_PF_1183 | 40.890000 | 100.690500 | 100.187617 | 102.225000 |

## 支持硬件

| 产品 | 是否支持 | 软件环境 |
|---|---|---|
| Ascend 950PR | 支持 | CANN 9.1.0 |

## 算子约束限制

- n≥0；n>0时lda≥n，incx/incy非零且不为INT_MIN。
- alpha/beta使用Host标量指针，A/x/y使用Device缓冲区。
- 调用方为x和y提供独立存储，缓冲区在stream完成前保持有效。
- n=0时直接返回；alpha=0、beta=1时允许A/x/y为空。
- alpha=0时允许A/x为空；beta=0时y仍需提供可写输出空间。
- 结果读取前同步句柄绑定的stream。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 标准 | 测试结果 |
|---|---|---|
| 精度 | 实部、虚部分别校验；atol=rtol=2^-13；有限值匹配比例≥0.99，最大绝对误差≤0.01；同时执行MERE/MARE及特殊值检查 | 1002/1002项通过 |
| 性能 | 全部200项用例分别满足GPU/NPU≥0.4；每例60次采样，报告前10次及全部60次算术均值 | 200/200项通过 |
| 标准构建 | `bash build.sh --ops=chemv --soc=ascend950 --run` | 构建成功，1202/1202项正确性测试通过 |
| 独立内存指标 | 任务书§3.4不涉及 | 不涉及 |

精度测试包含原始1000项用例、句柄校验和随机标量测试。随机标量测试内部执行128组计算，覆盖均匀及正态分布。测试覆盖上下三角、lda padding、正负步长、快速返回、NaN/Inf和步长空洞。

性能使用 `msprof op` 的 `Task Duration(us)`。应用执行5次预热后进行60次有效采样；工具对每次采样另执行5次replay预热，每例共300次工具预热。

构建与复现方法见 `test/symv/chemv/README.md`，测试汇总见 `test/symv/chemv/FINAL_VALIDATION.md`。

## 兼容性分析

公共头新增 `aclblasChemv` 声明，沿用已有BLAS句柄、执行流、复数类型及状态码。现有Ssymv/Csymv接口保持原有签名，arch35实现通过仓库构建框架注册。测试使用Netlib BLAS/CBLAS 3.10.0作为golden，并接入现有GTest及合并测试入口。
