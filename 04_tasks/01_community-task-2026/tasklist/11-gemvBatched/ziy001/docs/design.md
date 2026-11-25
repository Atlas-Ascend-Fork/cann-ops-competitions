# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务《aclblasSgemvBatched 类算子开发任务书》要求在 Atlas A2/A3 上实现五个实数批量矩阵向量乘接口。本文采用任务书 2026-11-24 修订的矩阵布局、异常行为和验收标准。实现使用 CANN 9.1.0、Ascend C 和 ops-blas 的句柄式接口。

## 背景介绍

一次批量调用可将多组形状相同、地址独立的矩阵向量乘提交到同一条 stream。对于小矩阵，逐批启动 kernel 会放大下发开销。对于单个大矩阵，按批次分核又可能只用到一个核。设计同时利用批次和输出维度的并行性。

任务书编制时将 arch35 的五个实数变体和 arch22 的复数实现列为参考。核对 ops-blas 提交 `6df9aa04` 时，arch22 已有五个实数入口，Host 按批次分核，校验只接受 N/T，并将零维、零批作为非法值处理。实施时从该版本的公共入口、类型分派和 launch 封装入手，按任务书补齐行为。参见 [arch22 Host 实现](https://gitcode.com/cann/ops-blas/blob/6df9aa04a1d37288966e1b97a20ae8033ca2ce3c/blas/gemv_batched/arch22/gemv_batched_host.cpp)。

# 需求分析（required）

## 需求描述

五个接口沿用 `include/cann_ops_blas.h` 中的声明，返回 `aclblasStatus_t`。参数顺序统一为 `handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount`。调用者在 Host 提供两个 FP32 标量，在 Device 提供三个指针数组。调用成功后，乘加结果异步写回各批次原来的 y。

## 需求拆解

| 开发项 | 可观察的结果 |
| --- | --- |
| 五种精度组合 | 按公开 API 解释输入位模式，以 FP32 乘加，一次转换到指定输出类型 |
| 布局和寻址 | N/T/C 均支持同一份列主序矩阵，支持带 padding 的 lda、任意非零有符号向量步长、非连续 batch |
| 异常与退化运算 | 多项入参错误时按规定顺序返回；合法空任务无 launch；零标量路径只访问数学结果需要的操作数 |
| 实现与集成 | 在 `blas/gemv_batched/arch22/` 实现，沿用 handle 绑定的 stream；在 `test/gemv_batched/arch22/` 接入 GTest 和上级 CMake，按验收语义更新算子 README |
| 交付验证 | 五套 CSV 驱动精度和状态码测试；在 910B3 逐用例比较 NPU 与 GPU 的 kernel 耗时 |

# 详细设计（required）

## 算子分析

### 数学公式

对每个批次 `b ∈ [0, batchCount)`，计算 `y_b ← alpha · op(A_b) · x_b + beta · y_b`。下表中的下标是逻辑下标，物理偏移见“支持形状”。

| trans | 输出长度 O | 归约长度 K | 输出 j 的点积 |
| --- | ---: | ---: | --- |
| ACLBLAS_OP_N | m | n | 对 k 从 0 到 n−1 累加 A[j,k] × x[k] |
| ACLBLAS_OP_T、ACLBLAS_OP_C | n | m | 对 k 从 0 到 m−1 累加 A[k,j] × x[k]；实数共轭转置等同于转置 |

### 支持数据类型

| 公开接口 | A、x | y 读写类型 | alpha、beta、内部累加 |
| --- | --- | --- | --- |
| aclblasSgemvBatched | FP32 | FP32 | FP32 |
| aclblasHSHgemvBatched | FP16 | FP16 | FP32 |
| aclblasHSSgemvBatched | FP16 | FP32 | FP32 |
| aclblasTSTgemvBatched | BF16 | BF16 | FP32 |
| aclblasTSSgemvBatched | BF16 | FP32 | FP32 |

混合精度 API 使用 `uint16_t` 承载 FP16/BF16 的位模式。按入口确定编码，将输入提升到 FP32，完成点积与缩放后仅在写 HSH/TST 输出时按相应浮点格式就近舍入。输入中的 Inf/NaN 在实际参与计算的路径上传播。

### 支持形状

- A 的逻辑形状为 m×n，物理形状恒为 lda×n，元素偏移为 `c * int64(lda) + r`。所有模式下 `lda >= max(1,m)`，列尾填充不参与计算。
- 对逻辑长度 L>0、步长 s 的向量，最小存储跨度为 `1+(L−1)*abs(int64(s))`。第 j 个逻辑元素的偏移是 `start+j*int64(s)`，其中负步长取 `start=(1−int64(L))*s`，正步长取 0。x 使用 L=K、s=incx，y 使用 L=O、s=incy。
- 每张指针表含 batchCount 个 Device 地址，各批形状一致，可分别分配内存，不做广播。例：m=2、n=3、lda=4、trans=T、incx=−2、incy=2 时，输出依次取 A 的三个列段 [0,1]、[4,5]、[8,9]，均乘 x 的偏移 [2,0]，仅写 y 的 [0,2,4]。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

五个入口复用一个参数校验和调度过程。Host 只验证可见的结构参数及顶层地址。以下各行按先后顺序执行，首个失败项决定返回值。

| 顺序 | 判定 | 处理 |
| --- | --- | --- |
| 1 | handle 为空 | 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 2 | trans 超出 N/T/C | 返回 ACLBLAS_STATUS_INVALID_ENUM |
| 3–6 | 依次检查 m 或 n < 0、lda < max(1,m)、incx 或 incy 为 0、batchCount < 0 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 7 | alpha 或 beta 为空，包括空任务 | 返回 ACLBLAS_STATUS_INVALID_VALUE |
| 8 | m、n、batchCount 全部大于 0，任一顶层 Aarray/xarray/yarray 为空 | 返回 ACLBLAS_STATUS_INVALID_VALUE；alpha=0 仍执行顶层校验 |
| 9 | m、n 或 batchCount 为 0 | 返回 ACLBLAS_STATUS_SUCCESS，允许三表为空，不解引用张量、不启动 kernel |

通过校验后读取 Host 标量，以值传给 kernel，C 归入 T。结构合法且 alpha=0、beta=1 时直接成功返回；alpha=0 的其余情况只调度 y 缩放或清零。其余调用在 `handle->stream` 异步下发；沿用仓中对平台查询失败、launch 失败的错误映射。调用方在读回 y 或释放缓冲区前同步该 stream。

##### 1. 分核策略

用 (b, 输出块) 标识任务，输出块含至多 R 个相邻逻辑输出。令 `G=ceil(O/R)`、`W=int64(batchCount)*G`，启动核数为 `min(W, 平台可用 AIV 核数)`。核 h 处理序号 h、h+blockDim、… 小于 W 的任务，分别以整除 G 和对 G 取余还原批次及块号。小输出可由一核完成一批，大输出可在批内分核；每个输出的完整 K 维归约只由一个核完成。切块边界兼顾连续 y 的搬运粒度，非对齐地址和跨步路径使用仅写有效元素的回写方式。

##### 2. 数据分块和内存优化策略

用 R 控制输出块、Q 控制归约分段。按平台提供的 UB 可用容量和变体字节数选择二者。一个保守预算由 A 暂存 `R*Q*sizeof(input)`、x 暂存 `Q*sizeof(input)`、FP32 转换及计算缓冲、R 个 FP32 累加器、输出转换缓冲、归约临时区组成。逐块按搬运对齐向上取整，双缓冲区按两个槽计入。优先复用已消费的乘积与转换缓冲；超预算时先缩 Q，再缩 R，必要时使用单缓冲。N 可按列流入 R 个元素，T/C 可按列流入 Q 个元素，均可用小于整片 R×Q 的暂存。

Host 传入变体、模式、m/n/lda、带符号步长、batchCount、R/Q、G、blockDim 以及 alpha/beta。局部块的有效长度用剩余尺寸截断。矩阵、向量、任务数和指针表的全局偏移先提升至 64 位，单次搬运的长度和指令计数字段按硬件上限切段。现有实现带有 32 位跨度检查；放宽该限制须同步验证所有设备侧地址计算、最后一块和搬运参数。

##### 3. tilingkey规划策略

五个入口确定输入和输出类型。内部按 `scale / normal / transpose` 三种计算路径分派，另传 `beta==0` 标志以选择是否读 y。连续向量使用批量搬运，非单位步长走逻辑寻址路径；这些内部信息随直调参数传递。小矩阵优先减少分块与队列切换，大矩阵依据 msprof 的核利用率和访存耗时调 R/Q。

#### 3.2.2 kernel侧设计

| 阶段 | N 路径 | T/C 路径 |
| --- | --- | --- |
| Init / CopyIn | 由 Device 表取本批基址，初始化 R 个 FP32 累加器。每次沿一列搬入连续的有效行段，按 incx 取相应列的 x | 读取待计算的一组列。按行段搬入各列连续的 Q 个元素，同一段 x 可供这些列复用 |
| Compute | 将列段提升为 FP32，乘以本列的 x 标量并累加到各行的输出，遍历全部列 | 将 A 列段和 x 段提升为 FP32 后逐元素乘，沿有效 Q 长度归约，将部分和计入该列的累加器 |
| CopyOut | 两条路径都在完整点积上乘 alpha；beta 非零才读取并提升旧 y，加上 beta×y。按目标类型一次转换并写回 | 同 N，按列号得到输出下标，用 incy 定位目标 |

缩放路径只解引用 y 表。alpha=beta=0 时直接写零，避免将未使用输入中的 NaN 带入结果。一般乘加中 beta=0 同样跳过旧 y 的加载。尾段在 UB 中补零或以掩码约束计算，只从 GM 搬入有效字节；跨步输出按有效地址写回，不覆盖孔洞或相邻工作的输出。流水搬运用独立队列槽衔接 CopyIn、Compute、CopyOut，槽复用前等待上一阶段消费完成。该分工无需跨核原子累加和 GM 中间结果。

## 支持硬件

| 产品 | 软件和验证安排 |
| --- | --- |
| Atlas A2 系列，arch22 | CANN 9.1.0、Ascend C；在 Atlas 800T A2（910B3）采集正式性能，可在同机完成精度自验 |
| Atlas A3 系列，arch22 | 同一接口与数值规则，按目标 SoC 核数、UB 容量建立 tiling；按任务书自验可选 A2/A3 中一种款型 |

A2 编译入口为 `bash build.sh --ops=sgemv_batched --soc=ascend910b3`，A3 按目标产品使用仓库支持的 SoC 配置。参考 arch35 的变体组织时，仅复用 arch22 和 CANN 9.1.0 支持的指令。

## 算子约束限制

尺寸和 batchCount 使用公开 ABI 中的 int。各批次 y 可写区域彼此独立，且与所有参与计算的 A/x 读区分离。该内存约束由调用者保证，Host 不扫描 Device 表项。标量取有限 FP32 值；向量支持反向遍历。运行时按调用参数划分工作，浮点归约顺序允许变化，验收采用数值容差。任务书未设独立内存性能门槛。

# 可维可测分析

## 精度标准/性能标准

以任务书 §3 和随包五组测试为依据，在 ops-blas 测试工程中按变体加载 CSV，固定随机种子。逐批以实际输入类型量化的 A、x 和初始 y 生成参考结果。语义对照 Netlib BLAS 3.10.0 的 `cblas_sgemv`；冻结 golden 可用 FP64 累加后按 y dtype 收窄。报告记录参考实现、累加精度和最终舍入方式。参见 [Netlib SGEMV](https://www.netlib.org/blas/sgemv.f) 和 [CANN 混合误差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。

| 按输出类型验收 | atol = rtol | 最低命中率 | 最大绝对误差限（任务书同时列出 32 ULP 口径） |
| --- | ---: | ---: | ---: |
| S / HSS / TSS，FP32 | 2^-13 | 0.99 | 1e-2 或 32 ULP |
| HSH，FP16 | 2^-10 | 0.99 | 1e-1 或 32 ULP |
| TST，BF16 | 2^-7 | 0.99 | 1e-1 或 32 ULP |

逐批按步长提取逻辑 y，有限值以 `abs(actual−golden) <= atol+rtol*abs(golden)` 计一次命中。每例同时检查命中率及最大绝差，记录使用的上限口径。特殊值验证 NaN 位置和 Inf 符号。状态码用例单独断言返回值；空任务同时检查零下发、原缓冲区未改写。

| 覆盖项 | 执行和定位方法 |
| --- | --- |
| 全量精度和边界 | 每变体 1000 条，共 5000 条。覆盖均匀分布 [-5,5] 与标准正态各约半数、N/T/C、非方阵、1 和对齐边界两侧、最大测试边长 4096、batchCount 从 0 到 256。按变体记录失败用例、种子、首个失配的批次和下标 |
| 布局与读写 | 测 lda=m、m+1、m+8、m+64，正负单位及大步长，验证 x/y 首地址、y 跨步孔洞与保护区哨兵。用 beta=0 且旧 y 为 NaN、alpha=0 且 x 为 NaN 验证短路；用不可连续推算基址的批次验证三张设备表 |
| 异常及优先级 | 分别测试空句柄、非法枚举、负尺寸、过小 lda、零步长、负批次、空标量/顶层表，以及空维度叠加非法结构参数。用测试夹具识别重叠输出，避免把调用方违约误判为一个可依赖的 API 状态码 |
| 待核对的附件差异 | 任务书 §2.4 定义步长为任意非零 int，附件 TC_L6_021/022 预期 INT_MIN 非法。设计按任务书做 64 位寻址；将两例单列，测试长度 1 的无溢出情况，并在验收前确认预期 |
| 性能 | 每变体 200 条，共 1000 条。用变体和 trans/m/n/lda/incx/incy/batchCount/alpha/beta 关联同配置的 cuBLAS kernel 基线，覆盖默认和非默认标量 |

正式性能在 Atlas 800T A2（910B3）以 `msprof op` 的 kernel Task Duration(us) 衡量。每例先 warmup 5 次，然后连续取 10 次成功执行的有效样本，取算术平均。逐例要求 `mean_us <= 1.25 × gpu_ms × 1000`。失败样本记录原因并补齐有效样本。11 月 24 日修订后，T/C 对应的 lda 键已变，须按修订配置重采 GPU 数据，非默认标量也须核对原始测量配置。

辅助计时在预置 Device 缓冲上用 `aclrtEvent` 包围连续 10 次下发，同样预热 5 次。GTest 打印 `[PERF] case=<case_name> ... avg_us=<数值>` 供脚本解析，排除 H2D/D2H、造数及 golden 比对。报告保留 CANN/驱动/SoC、代码版本、两套计时、失败记录和原始 profiler 样本。性能结论以 kernel-to-kernel 判据为准。

## 兼容性分析

保留公共头文件的五个函数符号、参数顺序、16 位存储 ABI 和 handle/stream 生命周期。arch22 的变化集中在实数路径；回归应同时覆盖原有 N/T 输入、实数 C 与 T 同输入等价、复数入口和 arch35 的独立构建。非法 trans 使用 ops-blas 的 ACLBLAS_STATUS_INVALID_ENUM。评审后，按同一套布局、空任务及步长语义同步 arch22 测试和 `blas/gemv_batched/README.md`，使 API 说明与运行行为可逐项对照。
