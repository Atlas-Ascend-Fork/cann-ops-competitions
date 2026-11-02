# aclblasChbmv 算子设计文档

## 一、需求背景

### 1.1 需求来源

本需求来源于“CANN训练营哈工深-aclblasChbmv算子开发（950）”。开发者为 qunzhongjia，目标是在 Ascend 950PR、CANN 9.1.0 环境下，基于 ops-blas 工程新增单精度复数 Hermitian 带状矩阵与向量乘法接口 `aclblasChbmv`。

### 1.2 背景介绍

Chbmv 属于 BLAS Level-2，计算 `y = alpha*A*x + beta*y`。A 为 n×n Hermitian 带状矩阵，主对角线上下各有 k 条带；x、y 为长度 n 的复数向量；alpha、beta 为复数标量。A 仅保存上带或下带，另一侧通过共轭对称关系获得，结果原地写入 y。

仓库已有 `aclblasSsbmv` 的 arch35 实现，可参考其带状寻址、参数校验、句柄和 stream 调用方式；`aclblasCsymv` 提供复数分量运算和片上缓冲的实现参考。本算子增加 Hermitian 共轭处理、实对角计算及复数带状归约，复用仓库公共类型、构建和测试框架。

## 二、需求分析

### 2.1 接口定义

公共声明加入 `include/cann_ops_blas.h`，参数顺序与任务书及 cuBLAS `cublasChbmv` 一致：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

| 参数 | 位置与含义 | 约束 |
| --- | --- | --- |
| handle | Host，携带执行 stream 的句柄 | 有效句柄 |
| uplo | Host，保存的带形三角 | ACLBLAS_UPPER 或 ACLBLAS_LOWER |
| n | Host，矩阵阶数及向量逻辑长度 | n≥0 |
| k | Host，次对角线或超对角线条数 | k≥0 |
| alpha、beta | Host，复数标量指针 | n>0 时非空 |
| A | Device，只读带状矩阵，物理形状 lda×n | complex64，列主序，n>0 时非空 |
| lda | Host，相邻存储列的复数元素间距 | lda≥k+1 |
| x | Device，只读输入向量 | complex64，n>0 时非空 |
| y | Device，原地更新的输出向量 | complex64，n>0 时非空 |
| incx、incy | Host，向量逻辑步长 | 非零，支持正负值 |

复数类型使用仓库 `aclblasComplex`，实部、虚部各为 float32。向量容量至少为 `1+(n-1)*abs(inc)` 个复数元素。Device 数据在 stream 完成计算前保持有效，调用方同步后读取结果。

### 2.2 功能要求

1. 支持 UPPER、LOWER 两种列主序带状存储，镜像侧按共轭关系读取。
2. 对角仅加载实部，带状数组中不对应矩阵元素的区域及 lda padding 不参与计算。
3. 支持正负步长，保持 A、x 和 y 的步长间隙不变。
4. 支持 n=0 的快速返回、alpha=0 的向量缩放及 beta=0 时跳过旧 y 读取。
5. 使用 handle 绑定的 stream 异步下发，输出 y 的每个逻辑元素由唯一线程写回。
6. 通过任务附件全部精度和性能用例，提供测试代码、复现步骤、报告及原始日志。

### 2.3 支持范围

本次适配 Ascend 950PR（arch35），软件环境为 CANN 9.1.0。支持 complex64、列主序及 lda/incx/incy 描述的存储间距。n、k 为运行时参数，有效带宽为 `b=min(k,n-1)`；k>n-1 时保留原始存储布局。

任务不涉及广播、批处理和额外张量 stride。调用方提供满足容量要求的有效设备存储，y 输出区域与只读 A、x 分离。精度按任务规定的 FLOAT32 混合容差判断。

## 三、详细设计

### 3.1 算子分析

输出行 r 的参与列范围为 `[max(0,r-b),min(n-1,r+b)]`。以下偏移均采用从 0 开始的下标，以复数元素为单位：

| 存储方式 | 直接侧 | 镜像侧 |
| --- | --- | --- |
| UPPER | r<c：A[c×lda+k+r-c] | c<r：conj(A[r×lda+k+c-r]) |
| LOWER | r>c：A[c×lda+r-c] | c>r：conj(A[r×lda+c-r]) |

UPPER 对角偏移为 `r*lda+k`，LOWER 为 `r*lda`，均只加载实部分量。k>n-1 时，b 控制参与计算的列范围，UPPER 存储偏移继续使用原始 k。

负步长向量的逻辑起点为 `-(int64(n)-1)*int64(inc)`，后续按 inc 递增；正步长起点为 0。地址乘法和步长计算使用 64 位中间量。

总有效元素数为 `n*(2*b+1)-b*(b+1)`，计算复杂度为 O(n×(b+1))。矩阵保持带状存储，计算采用输出行并行，无额外 GM 工作区。

### 3.2 Host 侧设计

#### 3.2.1 参数校验

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | n<0 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 2 | handle 为空 | 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 3 | n=0 | 返回 SUCCESS，不下发 Kernel |
| 4 | uplo、k、lda、步长或必要指针非法 | 返回 INVALID_VALUE |
| 5 | alpha=0 且 beta=1 | 返回 SUCCESS |
| 6 | AIV 核数查询结果为 0 | 返回 EXECUTION_FAILED |
| 7 | 其余合法输入 | 构造 Tiling 并下发 Kernel |

`lda≥k+1` 在 int64 下比较。n>0 时，即使 alpha=0 或 beta=0，仍检查 alpha、beta、A、x、y 的非空条件。Host 读取标量并按值传入 Kernel，不访问 Device 数据，不执行隐式 stream 同步。

#### 3.2.2 分核与模板选择

通过 `GetAivCoreCount()` 查询可用核数。每个 block 使用 128 个 SIMT 线程，线程组宽度 W 按以下条件选择：

| 条件 | W | 处理方式 |
| --- | ---: | --- |
| alpha=0 | 1 | 向量缩放 |
| alpha≠0 且 b≤4 | 1 | 单线程负责一行 |
| alpha≠0、b>4，且（b≤32 或 n≤2560） | 8 | 8 个线程共同处理一行 |
| alpha≠0、b>32 且 n>2560 | 1 | 增加并行输出行数，省去组内协作同步 |

`numBlocks=min(ceil(n*W/128),aivCoreNum)`。每个 block 同时处理 `128/W` 行，线程组按 `numBlocks*128/W` 的跨度领取后续输出行；组内 lane 0 写回。

UPPER/LOWER、W=1/8、连续步长和 beta=0 通过 Host 启动层选择模板。连续步长条件为 `incx==1 && incy==1`，其余步长使用通用索引路径。Tiling 保存 n、原始 k、b、lda、64 位步长与起点、标量分量、启动参数及模板标志。

### 3.3 Kernel 侧设计

Kernel 使用 AIV-only 任务类型，通过 `asc_vf_call` 执行 SIMT 计算。每个输出行独立完成左右带段、对角和 beta*y 的合并，无跨核结果依赖。

#### 3.3.1 运算顺序

直接侧先计算 `alpha*x[c]`，再乘对应矩阵元素；镜像侧先按列序累计 `conj(A)*x`，之后乘 alpha。各阶段按 Netlib 对该输出元素的更新顺序组织：

| 阶段 | UPPER | LOWER |
| --- | --- | --- |
| 初始化 | beta*y，beta=0 时直接置零 | 同左 |
| 左侧 | 累计镜像侧临时和 | 将直接侧贡献逐项加入结果 |
| 对角 | 加实对角项，再加 alpha×左侧临时和 | 加实对角项 |
| 右侧 | 将直接侧贡献逐项加入结果 | 累计镜像侧临时和，再乘 alpha 加入结果 |

实部、虚部使用 FP32，复乘按分量展开。编译关闭乘加收缩，保留乘法和加减的舍入边界。

#### 3.3.2 两种计算路径

W=1 时，单线程读取带内 A/x，在寄存器中顺序累加并写回，无共享乘积缓冲和线程间同步。

W=8 时，组内线程每轮并行计算最多 8 个复数乘积，写入每 block 共 1 KiB 的 UB 缓冲。lane 0 按列序消费本组乘积。每轮在写入后、缓冲复用前各执行一次块内同步。左右带段使用统一的带宽循环上界，尾部无效组仍参与同步，只屏蔽数据访问与写回。

beta=0 模板不加载旧 y；beta=1 直接使用旧 y。alpha=0 使用独立缩放入口，alpha=0、beta=0 时直接写零。k=0 只计算实对角项。所有地址先受逻辑行列范围约束，输出仅写 y 的逻辑位置。

### 3.4 性能设计

附件 200 条性能用例中，n 为 1～4096，k 为 0～2047，其中 165 条 k>128。小规模重点控制启动与同步成本，中大规模平衡行并行度和行内乘积并行度。

矩阵向量乘和 beta 缩放在单 Kernel 内完成。窄带采用 W=1；中小规模采用 W=8 并行读取和计算乘积；大规模宽带采用 W=1，提高同时处理的输出行数。三角、步长和零 beta 分支在 Host 分派，减少内层判断。

W=8 保留顺序累加，其收益需覆盖 UB 交换和同步成本。分派边界以同条件逐例测量为依据，补充测试覆盖 b=4/5、32/33 和 n=2559/2560/2561。调整线程组、访存或分块方式时，重新执行精度检查和全部 200 条性能用例。

### 3.5 工程集成

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 公共接口声明 |
| `blas/sbmv/arch35/chbmv_host.cpp` | 参数校验、Tiling 和分派 |
| `chbmv_kernel.cpp`、`chbmv_compute.h` | 启动、行调度、复数计算及寻址 |
| `chbmv_kernel.h`、`chbmv_tiling_data.h` | 启动声明与共享参数 |
| `test/sbmv/chbmv/` | 参数解析、数据生成、Netlib Golden 与测试注册 |
| `test/sbmv/chbmv/arch35/` | GTest、任务 CSV 和 GPU 基线 |
| `blas/sbmv/README.md` | 产品支持、接口参数、约束及调用说明 |

构建沿用仓库源码收集规则，选择 ascend950、Release 和 chbmv 测试目标，通过 `ops_blas_add_gtest_tests` 接入测试框架。Kernel 与测试使用 `-O2 -ffp-contract=off`。源文件保留仓库要求的许可头，执行格式、头文件和 OAT 检查。

## 四、可维可测分析

### 4.1 精度验证

采用 CSV 驱动的 C++ GTest，通过公共接口执行；Golden 使用 Netlib BLAS 3.10.0 `cblas_chbmv`，设置 CblasColMajor。执行原始 1000 条精度及边界用例，并对 200 条性能形状验证精度。

实部、虚部分别判定：rtol=atol=2^-13，有限值 matched_ratio≥0.99，最大绝对误差≤`max(1e-2,32*ULP(g_low))`，g_low 为最大绝对误差点的 FLOAT32 Golden 值。非有限值独立统计：NaN/NaN、同号 Inf/Inf、NaN/Inf 配对通过；有限值与非有限值、异号 Inf 配对失败。

随机输入使用固定种子，实部、虚部独立生成，均匀和正态分布按任务要求配置。覆盖 UPPER/LOWER、零维、不同带宽、lda padding、正负步长、零标量、Inf/NaN 和非法参数。

补充 126 组输入覆盖分派边界、线程组尾部及 k>n-1。检查对角虚部和无效带区填 NaN、beta=0 时旧 y 含 NaN、A/x 不变、y 间隙不变及参数校验优先级。报告保存每例参数、返回状态、实部/虚部误差与特殊值统计。

### 4.2 性能验证

在 Ascend 950PR、CANN 9.1.0 上执行全部 200 条性能用例，使用 `msprof op` 后端采集 BasicInfo 的 `Task Duration(us)`。设置工具 `warm-up=5`、kernel replay，预热不计入有效样本。

任务书 §3.3 规定每例 10 个有效样本；§7.4 要求大于 10 次，附件 README 要求大于 50 次。分别执行每例 10 次和 60 次的两轮完整采集，各自取算术平均并保存原始数据。

调用前恢复原始 y，检查 API 返回 SUCCESS 和 stream 同步状态。采样区间覆盖被测 Kernel，输入准备、数据恢复、CPU Golden 和比较位于区间外。容器设备映射通过独立构建的未修改上游 msopcom 组件处理，保存工具版本、源码版本及库哈希。

GPU 基线按 uplo/n/k/incx/incy/lda 关联，gpu_ms 转换为 μs。逐例要求 `NPU平均耗时≤GPU基线×2.5`。核对调用次序、有效样本数量及原始时间记录，缺失或异常采集单独记录。两轮结果分别提交，包含逐例耗时、倍率、环境和构建版本。

### 4.3 兼容性与维护

新增公共接口，复用现有复数类型、枚举、状态码和句柄。实现限定在 arch35，测试按架构注册，Ssbmv、Csymv 独立维护。

寻址修改回归两种三角、负步长和 padding；调度修改回归分派边界和尾组；浮点运算修改执行完整精度验证。代码、文档、测试结果对应同一交付版本。

### 4.4 交付内容

提交设计文档、算子源码、测试代码及 CSV、算子 README、自验证步骤、精度和性能报告与原始日志。个人代码仓提供分支、提交和算子目录，按任务流程开放验收访问。内存项按任务书 §3.4 填写“不涉及”。

## 五、参考资料

1. 任务附件：`aclblasChbmv_Atlas950PR_task_doc.md`、`chbmv_test.csv`、`gpu_baseline.csv` 和测试 README。
2. [社区设计模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
3. [ops-blas](https://gitcode.com/cann/ops-blas)：Ssbmv、Csymv、公共接口与测试框架。
4. [Netlib chbmv](https://www.netlib.org/blas/chbmv.f)：存储、共轭、步长和运算次序。
5. [cuBLAS hbmv](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hbmv)：接口语义。
6. [FLOAT32 混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
