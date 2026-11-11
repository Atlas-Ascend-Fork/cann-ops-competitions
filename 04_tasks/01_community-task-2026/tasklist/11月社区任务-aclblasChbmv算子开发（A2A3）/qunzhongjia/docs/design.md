# aclblasChbmv 算子设计文档

## 一、需求背景

### 1.1 需求来源

本需求来源于“11月社区任务-aclblasChbmv算子开发（A2A3）”。开发者为 qunzhongjia。在 Atlas A2/A3、CANN 9.1.0 环境下，基于 ops-blas 工程新增单精度复数 Hermitian 带状矩阵与向量乘接口 `aclblasChbmv`。

### 1.2 背景介绍

算子计算 `y = alpha × A × x + beta × y`。A 为 n 阶 Hermitian 带状矩阵，半带宽为 k，只存储指定三角内的带状元素；另一侧通过共轭关系得到。输入、输出及标量均为 complex64，y 原地更新。

仓内 `blas/sbmv/arch22/ssbmv_host.cpp` 和 `ssbmv_kernel.cpp` 提供同族实数算子的 Host 校验、UB 分块及带状访问方式；`blas/gbmv/arch22/cgbmv_kernel.cpp` 提供复数解交织、向量乘加和归约方式。本设计参考这些实现，增加 Hermitian 共轭、实数对角及复数标量处理，并接入仓库公共接口和 CSV 驱动测试框架。

## 二、需求分析

### 2.1 接口定义

在 `include/cann_ops_blas.h` 新增公共声明：

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

| 参数 | 位置与含义 | 约束 |
| --- | --- | --- |
| handle | Host，上下文及 stream | 非空 |
| uplo | Host，存储三角 | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n、k | Host，矩阵阶数与半带宽 | n ≥ 0，k ≥ 0 |
| alpha、beta | Host，复数标量指针 | n > 0 时非空 |
| A、lda | Device，列主序带状数组及前导维度 | n > 0 时 A 非空，lda ≥ k+1，容量为 lda×n 个复数 |
| x、incx | Device，只读向量及逻辑步长 | n > 0 时 x 非空；步长非 0、非 INT32_MIN，跨度不超过 INT32_MAX |
| y、incy | Device，原地更新向量及逻辑步长 | n > 0 时 y 非空；步长约束同 x |

复数布局采用 `include/cann_ops_blas_common.h` 的 `aclblasComplex` 定义，实部与虚部各为 float32。x、y 分别至少容纳 `1+(n-1)×abs(inc)` 个复数。调用通过 handle 绑定的 stream 异步执行；输入及输出缓冲保持有效，直至 stream 执行完成。

### 2.2 功能要求

1. 对 UPPER、LOWER 分别支持紧凑带状存储及 lda padding，支持正负步长。
2. 对角元素仅使用实部；带外、未使用三角区域及 padding 不参与计算。
3. 有效半带宽为 `b=min(k,n-1)`。k 大于 n−1 时，UPPER 的存储偏移仍使用原始 k。
4. 按任务书顺序检查 handle 和 n；n=0 返回成功。alpha=0 执行向量缩放；beta=0 不读取 y 的旧值。
5. 通过附件全部用例，并补齐正态输入、标量随机分布、校验顺序及内存访问检查。

### 2.3 支持范围

目标架构为 arch22，适配 Atlas A2/A3。自验证使用 CANN 9.1.0，性能设备采用 Atlas 800T A2（910B3）。精度参考库为 Netlib BLAS 3.10.0。功能范围按接口整数约束及可用内存确定，测试集最大尺寸不作为接口尺寸上限。

## 三、详细设计

### 3.1 算子分析

以下下标从 0 开始，A 的地址以复数元素为单位。输出第 i 行参与计算的列范围为 `[max(0,i-b), min(n-1,i+b)]`。

| 存储方式 | 条件 | 逻辑矩阵元素 M(i,j) |
| --- | --- | --- |
| UPPER | i < j | `A[j×lda+k+i-j]` |
| UPPER | i > j | `conj(A[i×lda+k+j-i])` |
| UPPER | i = j | `real(A[i×lda+k])` |
| LOWER | i > j | `A[j×lda+i-j]` |
| LOWER | i < j | `conj(A[i×lda+j-i])` |
| LOWER | i = j | `real(A[i×lda])` |

负步长起点为 `start=-(n-1)×inc`，正步长起点为 0；逻辑元素 i 的地址为 `start+i×inc`。地址乘加使用 64 位整数。UB Gather 使用字节偏移，复数元素偏移乘以 8，实部与虚部相差 4 字节。

计算量随 `n×(2b+1)-b×(b+1)` 增长。对一个已存储的非对角元素 M(i,j)，分别计算其对输出 i 的直接贡献和对输出 j 的共轭贡献。输出按行块分配，每个输出元素由一个核完成全部累加并写回。

### 3.2 Host 侧设计

#### 3.2.1 参数校验

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | handle 为空 | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | n < 0 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | n = 0 | 返回 `ACLBLAS_STATUS_SUCCESS` |
| 4 | uplo 非法、k < 0 或 lda < k+1 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | incx/incy 为 0 或 INT32_MIN | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | 任一向量的 `(n-1)×abs(inc)` > INT32_MAX | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | alpha、beta、A、x、y 任一为空 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | 参数合法 | 读取 Host 标量，生成 Tiling，向 handle 的 stream 下发计算 |

`k+1`、绝对步长及向量跨度均在 64 位域计算。alpha=0 和 beta=0 的分派位于非空检查之后。

#### 3.2.2 分核与模板选择

| 条件 | 计算路径 | 处理方式 |
| --- | --- | --- |
| alpha=0、beta=1 | 快速返回 | 保留 y |
| alpha=0 | 向量缩放 | beta=0 直接写零；其余按复数 beta 缩放 |
| b=0 | 对角乘 | 只读取实数对角，计算逐元素复数乘加 |
| 连续向量、小规模 | 小规模行块 | 减少核数和缓冲配置，合并一次下发 |
| 连续向量、窄带 | 带状行块 | 多列成批搬运，复用 x 窗口与输出累加器 |
| 连续向量、宽带 | 分段带状行块 | 沿带宽分段加载，分段归约后合并结果 |
| 非单位步长 | 通用步长 | 按逻辑顺序装载 x，按目标步长写回 y |

UPPER/LOWER、beta 是否为零、步长是否连续由 Host 选择模板；Kernel 的内层计算不重复判断这些条件。分派只依赖接口入参和设备能力。

通过仓库平台接口查询 AIV 核数及 UB 容量。连续输出按行块分配，块数由 n、带宽和每核工作量确定，上限为可用核数。Tiling 按值传递，包含 n、k、b、lda、步长及起点、复数标量、行块长度、带宽分段长度和批列数。

UB 预算逐项计入 A 的交织缓冲、实虚分量、x 窗口、累加器、乘法及归约临时区、Gather 索引、输出缓冲和对齐空间。双缓冲按两份计入。Host 与 Kernel 使用同一套容量计算；预算不足时缩小分块。

小规模与窄带的切换点通过代表形状测量确定：覆盖 n、b、lda、尾块，以及候选边界两侧。以全量精度通过和逐例耗时为选择依据。

### 3.3 Kernel 侧设计

采用 Ascend C AIV SIMD，使用 UB 缓冲、`DataCopyPad`、`Gather`、向量乘加与归约。

1. **输出初始化**：为当前行块建立实部、虚部累加器。beta 非零时装入 y 旧值并在结果合并阶段缩放；beta 为零时直接生成输出。
2. **直接贡献**：固定存储列，计算其有效非对角行段与输出行块的交集。将多个连续列段分批搬入 UB，拆分实部、虚部。每列的 x 标量在该列段复用，以 SIMD 更新行块累加器。
3. **共轭贡献**：对属于当前输出行块的存储列，装载该列的有效非对角带段及对应 x 窗口。使用 `conj(a)×x` 的实虚公式做向量乘法和分段归约，累加到对应输出。带宽较大时沿有效段循环分块。
4. **对角贡献**：单独读取对角实部，与复数 x 相乘；直接和共轭路径均排除对角，每行对角贡献只计算一次。
5. **结果合并**：将累加值乘以 alpha，与 beta×y 合并，重新交织后写回目标向量。

普通复数乘法使用 `(ar×xr-ai×xi, ar×xi+ai×xr)`，共轭乘法使用 `(ar×xr+ai×xi, ar×xi-ai×xr)`。有限值累加采用分段归约；长带累加的误差通过全量 FP64 参考及抵消场景检查，必要时增加补偿累加。特殊值路径按有效元素计算，对角采用实数乘法，避免无效虚部参与运算。

每个行块装载所需的 x 窗口；窗口超过 UB 容量时与带段同步分片。负步长按起点和有符号偏移恢复逻辑顺序。非单位输出步长采用单核写回，每个逻辑元素按精确字节数进行 DMA 搬运，保留向量间隙。

所有 GM 搬运限制在有效列段与分配范围内。边界掩码在算术运算前生效，带外元素不以乘零方式参与计算。UB 对齐补位显式初始化，尾块仅写有效元素。各核输出区间互不重叠，普通计算路径不使用跨核原子累加或 GM 中间归约区。

MTE2 装载完成后才允许向量读取；向量计算完成后才允许 MTE3 写回或 MTE2 覆盖对应缓冲。批次切换和缓冲复用使用成对事件同步，避免逐元素同步。双缓冲是否启用由 UB 预算和测量结果共同确定。

### 3.4 性能设计

附件有 200 条性能用例，n 为 1～4096，k 为 0～2047，UPPER 95 条、LOWER 105 条。全部使用 `incx=incy=1`、`lda=k+1`、alpha=(1,0)、beta=(0,0)。带宽分布如下：

| k 范围 | 用例数 | 优化重点 |
| --- | --- | --- |
| 0 | 3 | 对角路径、启动开销 |
| 1～4 | 7 | 小规模分核、搬运批次 |
| 5～32 | 9 | x 窗口复用、连续列段 |
| 33～128 | 16 | 批列处理、搬运与计算重叠 |
| 大于 128 | 165 | 宽带分段、向量归约与负载分配 |

前四条基线及允许耗时如下，单位为 μs，表中数值保留六位小数，判定使用原始数据：

| uplo | n | k | A100 基线 | NPU 耗时上限 |
| --- | --- | --- | --- | --- |
| UPPER | 512 | 8 | 9.881600 | 12.352000 |
| UPPER | 1024 | 16 | 15.872000 | 19.840000 |
| LOWER | 2048 | 32 | 28.006400 | 35.008000 |
| LOWER | 4096 | 64 | 53.026133 | 66.282666 |

主要优化项为减少小 DMA 和流水事件次数、重复使用 x 的 UB 数据、控制复数解交织成本、避免逐元素 GM 访问，并使上下三角的边界块工作量接近。使用 msprof 定位搬运、向量计算和调度成本；每项优化均比较同形状的完整算子耗时，并回归对应路径的精度。

### 3.5 工程集成

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增公共接口声明 |
| `blas/sbmv/arch22/chbmv_host.cpp` | 参数检查、分块及模板选择 |
| `blas/sbmv/arch22/chbmv_tiling_data.h` | Host 与 Kernel 共用的分块参数 |
| `blas/sbmv/arch22/chbmv_kernel.h`、`chbmv_kernel.cpp` | 核函数入口、搬运、计算及写回 |
| `blas/sbmv/README.md` | 接口、产品支持、存储及调用约束 |
| `test/sbmv/chbmv/CMakeLists.txt` | GTest 构建注册 |
| `test/sbmv/chbmv/chbmv_param.h`、`chbmv_golden.h` | CSV 参数解析与 Netlib 参考计算 |
| `test/sbmv/chbmv/arch22/` | NPU wrapper、GTest 与原始及补充 CSV |
| `test/sbmv/chbmv/README.md` | 编译、执行、采集和判定步骤 |

算子源码接入现有 arch22 构建筛选，测试沿用同族 `ops_blas_add_gtest_tests` 注册方式和仓库测试入口。校验单算子构建、公共接口链接及测试枚举，避免测试被架构筛选跳过。源码遵循仓库许可头和格式规则，使用命名表达逻辑，不添加解释性或开发过程注释。

## 四、可维可测分析

### 4.1 精度验证

保留附件原始 1200 条 CSV，包括 1000 条精度及负向用例、200 条性能形状。GTest 调用公共接口执行，通过 Netlib BLAS 3.10.0 的 `cblas_zhbmv` 计算参考：complex64 输入提升为双精度复数，采用 `CblasColMajor`，输出收窄至 float32 后比较。该方式对应附件 README 的 FP64 golden 说明。

对逻辑 y 的全部 n 个元素分别统计实部和虚部，按任务书 §3.2 判定：

- `abs(actual-golden) ≤ 2^-13 + 2^-13×abs(golden)` 的有限值元素比例至少为 0.99。
- 最大绝对误差不超过 `max(0.01,32×ULP(golden))`，ULP 使用最大误差点的 float32 golden。
- 非有限值独立计数：NaN 对 NaN、同号 Inf 对 Inf、NaN 对任意符号 Inf 通过；有限值对非有限值、异号 Inf 不通过。所有非有限值比较均须通过。
- 无有限值元素时仅应用非有限值判据；零维用例核对返回码和输出未被修改。

CSV 的 MERE/MARE 保留列按附件说明保留，算子精度判定使用上述混合容差与非有限值规则。参考库须确认实际链接到 Netlib BLAS 3.10.0。

对原始均匀随机输入增加同参数的正态输入组，并按固定种子成对补充随机复数标量组合，使随机验证输入中均匀与正态各占 50%。均匀区间为 [-5,5]；正态参数 μ∈[-5,5]、σ∈[0.1,2]，使用 `GAUSS_NORM_5_2`。A、x、y、alpha、beta 的实部和虚部独立生成；标量特殊值及 Inf/NaN 用例另列。原始用例、补充用例、输入组及各参数的分布数量分别记录，完整执行。

接口及存储检查覆盖 handle 与零维的校验顺序、k 超宽、k+1 溢出、INT32_MIN 步长、向量跨度溢出、负步长、padding、对角虚部、beta=0 的无效 y 旧值、分块边界与尾块。负向用例使用单元素占位缓冲，避免测试框架按非法参数申请巨量内存。

`CsvCaseCount` 校验原始总数 1200、性能分类数 200 和名称唯一性；补充集合独立核对。执行自检比较预期名称集合与 GTest XML，要求无缺项、无重复、无失败、无跳过，并核对测试进程退出状态。精度判断模块使用人工构造的通过及失败数据检查阈值边界。

### 4.2 性能验证

在 Atlas 800T A2（910B3）、CANN 9.1.0 上执行全部 200 条性能用例。任务书 §3.3、§7.4 指定 msprof op；附件 README 的 `verify_performance.py` 读取 Event 结果。两组结果分别采集、保存和判定。

**任务书采集**：使用官方 `msprof op`，按单条用例过滤，保存原始采集目录与 `OpBasicInfo.csv`。工具执行 5 次预热，预热记录排除；获取至少 10 次有效执行的 `Task Duration(us)`，取算术平均。若一次 API 含多个 kernel，按一次调用的完整序列汇总再求平均。采集小样本先核对应用下发、工具重放、预热及有效样本的对应关系，防止重复计数或遗漏。接口返回异常、设备执行异常或样本不足的用例判定采集失败，并保留记录。

**附件脚本采集**：在 GTest 中预置 Device 输入，预热 5 次后，使用同一 stream 的 `aclrtEvent` 包围连续 10 次接口调用，将总耗时除以 10，输出 `[PERF] case=... avg_us=...`，由原始 `verify_performance.py` 读取。Host 造数、内存分配、H2D/D2H 及 golden 比较位于计时区间之外。Event 测试独立于 profiler 执行。

本任务 200 条性能用例的 beta 均为零，y 旧值不参与计算，可连续复用同一输出缓冲；A、x 和标量保持不变。补充 beta 非零的测量时，在区间外预置每次调用的独立 y 输入，并核对 profiler 重放对输入状态的处理。

基线采用附件 `gpu_baseline.csv`，环境为 NVIDIA A100-SXM4-80GB、CUDA 12.8、cuBLAS 12.8.4。统一 uplo 名称后按 `(uplo,n,k,incx,incy,lda)` 对应，要求 200 条完整且唯一。先将 `gpu_ms` 乘以 1000 转为 μs，再计算 `NPU_us/GPU_us`。两组测量均逐例要求倍率 ≤1.25，保留有效样本数、均值、基线、倍率及判定。

### 4.3 兼容性与维护

公共接口使用仓库统一类型，实现集中于 arch22。测试通过仓库构建及公共入口运行，README 产品支持表标注 Atlas A2/A3。寻址修改回归两种三角、padding、负步长与尾块；调度修改回归核数和分块边界；计算修改回归随机输入、抵消、极端值及非有限值。

任务书 §3.4 未设内存性能指标。补充使用官方 msSanitizer 的 memcheck 检查有效带访问、步长、UB 分块和输出边界，保留实际工具输出。检查程序要求检测启动及结束记录完整配对、执行数为正、工具错误为零；不执行 kernel 的接口负向用例单独核对返回码。UB 复用的同步检查按官方工具支持范围执行。

### 4.4 交付内容

设计文档按社区目录提交至 cann-competitions。待验收个人代码仓包含实现、完整测试工程、原始 CSV、GPU 基线、测试入口及 README，并邀请 Ascend-CANN 为开发者。验收说明填写对应分支的代码提交链接及算子目录。

按任务书 §4，在个人仓根目录 `task_submission/` 提交以下七份文件，同步用于任务平台验收：

1. `1 自验证步骤说明.md`
2. `2.1 精度自验证报告.xlsx`
3. `2.2 精度自验证日志.log`
4. `3.1 性能自验证报告.xlsx`
5. `3.2 性能自验证日志.log`
6. `4.1 内存自验证报告.xlsx`
7. `4.2 内存自验证日志.log`

报告填写环境、命令、用例参数、判据及结果，嵌入真实终端与采集结果截图。日志保留官方工具原始输出，复现所需采集和统计脚本随测试工程交付。算子代码 PR 按 ops-blas 目录规范提交实现、测试和文档，验收材料保存在个人仓指定目录。

## 五、参考资料

1. [社区任务及原始附件](https://www.hiascend.com/activities/task-center/details/ea8efcc9f98c4d54930f8b8e94a59366)：`aclblasChbmv_AtlasA2_A3_task_doc.md`、`test_cases/README.md`、`chbmv_test.csv`、`gpu_baseline.csv`。
2. [社区任务设计模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
3. [ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)。
4. [Ssbmv A2/A3 实现](https://gitcode.com/cann/ops-blas/tree/master/blas/sbmv/arch22)及[Cgbmv A2/A3 实现](https://gitcode.com/cann/ops-blas/tree/master/blas/gbmv/arch22)。
5. [已合入 Ssbmv A2/A3 设计 PR](https://gitcode.com/cann/cann-ops-competitions/merge_requests/1724)及[已合入 Cgbmv A2/A3 设计 PR](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2029)。
6. [Netlib CHBMV](https://www.netlib.org/blas/chbmv.f)、[Netlib ZHBMV](https://www.netlib.org/blas/zhbmv.f)及[cuBLAS HBMV 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hbmv)。
7. [生态算子混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
