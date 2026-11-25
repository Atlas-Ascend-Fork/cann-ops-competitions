# 需求背景

## 需求来源

11 月 CANN 社区任务《11月社区任务-gemm类3个算子开发（A2A3）》要求在 Atlas A2/A3 上，基于 ops-blas 以 Ascend C 实现并验收单矩阵通用乘法 `aclblasSgemm`、`aclblasCgemm`、`aclblasHgemm`。需求和验收以任务书 §2–§3 为准，接口逐参数对标 cuBLAS 的同名 S/C/H GEMM，设计章节遵循 [社区任务设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

## 背景介绍

任务书将已有 arch22 Sgemm 和 arch35 Cgemm、HgemmStridedBatched 列为实现参考。设计时核对的 ops-blas 公共基线 `55ef791b` 已有 arch22 Sgemm 和 Cgemm，其中 GEMM README 将 Cgemm 的 A2 标为已支持、A3 标为不支持。现有 Cgemm 使用 Device 标量读取，且在判定 `alpha=0` 前要求 `k>0` 时 A/B 非空。实施时复用已有符号和计算路径，补齐任务的 Host 标量、退化分支和 A3 验证。Hgemm 补齐单矩阵公开入口及 arch22 实现。 [arch22 实现基线](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch22) · [GEMM README](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/README.md)

# 需求分析

## 需求描述

三个入口均使用 `aclblasHandle_t` 和 `aclblasSetStream` 绑定的 stream 直调 NPU kernel。标量由 Host 指针传入，矩阵由 Device 指针和列主序前导维描述，C 原地更新。调用返回状态码，涉及 Device 工作的路径在绑定 stream 上有序提交，调用方同步后读取结果。

## 需求拆解

| 项目 | 设计交付 |
| --- | --- |
| 接口 | 保留 Sgemm 和 Cgemm 公开签名，在 `include/cann_ops_blas.h` 增加单矩阵 Hgemm 声明。实现遵循库的参数校验、状态码和句柄约定。 |
| 实现 | 在 `blas/gemm/arch22/` 复用 Sgemm 与 Cgemm，增加 Hgemm。覆盖九种 N/T/C 组合、任意合法前导维、空输出与零乘积、标量缩放和 C padding 保护。 |
| 测试与说明 | 扩展 `test/gemm/arch22/gemm_test.cpp` 和 CSV 测试入口，接入三种类型的自验用例。更新 `blas/gemm/README.md` 的产品支持与约束。 |
| 验收 | 完成 Atlas A2/A3 适配和自验，在 910B3 使用同参数的 GPU kernel 基线核验性能。 |

# 详细设计

## 算子分析

### 数学公式

令 `P(i,j)=Σ(p=0…k−1) op(A)(i,p)·op(B)(p,j)`。对 `0≤i<m`、`0≤j<n` 写入 `C[i+j·ldc]=alpha·P(i,j)+beta·C_old[i+j·ldc]`。在 `k=0` 或 `alpha=0` 时直接执行 `beta·C_old`。在 `beta=0` 时跳过旧 C 的读取，零乘积路径直接写零。

| 操作 | A 的元素偏移 | B 的元素偏移 |
| --- | --- | --- |
| `N` | `i+p·lda` | `p+j·ldb` |
| `T` 或 `C` | `p+i·lda` | `j+p·ldb` |

对 Cgemm，`C` 模式取所读元素的共轭，沿用上述地址映射。把应用共轭后的操作数分为 `Ar,Ai,Br,Bi`，实部为 `Ar·Br−Ai·Bi`，虚部为 `Ar·Bi+Ai·Br`。alpha 和 beta 同样按复数乘法展开。Sgemm 和 Hgemm 的 `C` 模式与 `T` 等价。

### 支持数据类型

| 算子 | 标量及矩阵类型 | 计算与输出 |
| --- | --- | --- |
| `aclblasSgemm` | `float`（FLOAT32） | FP32 乘加和缩放，输出 FP32 |
| `aclblasCgemm` | `aclblasComplex`（COMPLEX64） | 四个 FP32 实数乘积，实部和虚部分别以 FP32 累加、缩放和写回 |
| `aclblasHgemm` | `aclblasHalf`（IEEE 754 binary16 位模式） | FP16 输入，FP32 累加和缩放，最终一次舍入至 FP16 |

`aclblasHalf` 以 16 位载体存放，标量按 FP16 位模式解码。复数布局使用 `cann_ops_blas_common.h` 定义的两个 FP32 分量。

### 支持形状

| 矩阵 | 逻辑形状 | 设备端物理形状及前导维下界 |
| --- | --- | --- |
| `A` | `op(A): m×k` | N 时 `lda×k`、`lda≥max(1,m)`。T/C 时 `lda×m`、`lda≥max(1,k)`。 |
| `B` | `op(B): k×n` | N 时 `ldb×n`、`ldb≥max(1,k)`。T/C 时 `ldb×k`、`ldb≥max(1,n)`。 |
| `C` | `m×n` | `ldc×n`、`ldc≥max(1,m)`。每列只访问逻辑行。 |

`m,n,k` 为非负 `int`。尺寸、前导维和转置参数运行时传入。尺寸池含 `4095³`、`4096³`，计算限额由实际地址范围、硬件能力和工作区决定。

## 算子实现

### 实现方案

三个接口使用同一参数序列和状态码类型。下面的 `T` 分别对应 `float`、`aclblasComplex`、`aclblasHalf`。

```cpp
aclblasStatus_t operator_entry(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const T* alpha, const T* A, int lda, const T* B, int ldb,
    const T* beta, T* C, int ldc);
```

#### 3.2.1 host侧设计

入口按 handle → transa/transb → m/n/k → lda/ldb/ldc → 标量和矩阵指针的顺序校验。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法维度、非法前导维或必需指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。当 `m,n>0` 时读取 Host 标量并按值写入 launch 参数，按 `needProduct=(k>0 && alpha!=0)` 决定 A/B 是否必需。复数零和一分别为 `(0,0)`、`(1,0)`，FP16 按解码后的值比较，正负零等价。Sgemm 将前导维校验移到空输出判断之前。Cgemm 的任务路径从 Host 取标量，并将 A/B 校验放到 `needProduct` 判定之后。

| 已通过结构校验的分支 | 必需矩阵指针 | 提交动作 |
| --- | --- | --- |
| `m=0` 或 `n=0` | 无 | 返回 `ACLBLAS_STATUS_SUCCESS`。允许 A/B/C 为空，alpha/beta 指针仍必须非空。 |
| 零乘积，`beta=1` | C | 直接返回成功。 |
| 零乘积，`beta=0` | C | 单次 Vector launch 写逻辑零。 |
| 零乘积，其他 beta | C | 单次 Vector launch 原地缩放整个逻辑 C。 |
| 有乘积 | A/B/C | 依据分块参数提交计算与写回。`beta=0` 时选专用写回分支。 |

Host 对实际访问矩阵的最大元素索引和字节数、向上取整、分块计数以及工作区偏移用带溢出检查的 64 位运算，再转换到 SDK 要求的字段宽度。所有分块规划与工作区容量检查在首次 launch 前完成。平台查询、分配和下发错误按现有 ops-blas 状态码映射。

##### 1. 分核策略

将 C 逻辑域切为 `ceil_div(m,BM)×ceil_div(n,BN)` 个输出块，优先沿 M/N 分配。执行核数取可用 AIC 或 AIV 核数与可执行块数的较小值，核 `b` 处理编号 `b,b+blockDim,…` 的块，同一输出块每次归约只由一个核写入。规则、可直接写回的 Sgemm 复用已有全网格和二维分块方案。瘦长或小 K 形状优先缩小输出块，避免打包与 dispatch 开销盖过计算。基线方案中输出块跨完整 K，工作区需要分段时按 K 顺序在同一 stream 累加中间块。alpha/beta 只在最后一段完成后应用一次。

##### 2. 数据分块和内存优化策略

- 依据目标 SKU 的 AIC/AIV 核数及 L1/L0A/L0B/L0C、UB 容量，从 SDK 支持粒度内选择 `BM/BN/BK`。每次搬入的数据量按硬件布局对齐后计费，A/B 面板分别约为 `BM·BK·sizeof(input)` 和 `BK·BN·sizeof(input)`，FP32 累加块约为 `BM·BN·4` 字节。复数分量、FP32 临时块、旧 C 读取、写回和 SDK 预留量按其实际同时存活阶段预算，双缓冲将对应面板预算乘二。容量不足时先关闭双缓冲，再缩小分块。
- 紧凑、形状和布局满足 Matmul 直读条件时，使用既有列主序到行主序映射 `Cᵀ=op(B)ᵀ·op(A)ᵀ`，交换 A/B 与 M/N 的解释。通用路径按实际前导维从 GM 搬入，Vector 完成有界打包、必要的转置和尾块填零。尾部剩余的有效范围按各维度取最小值，填充留在内部缓冲。
- 工作区划分为按对齐取整的打包区、乘积区和累加区，优先复用句柄的 `EnsureDefaultWorkspace`、`GetEffectiveWorkspace`。Cgemm 复用实虚分面和四次实乘工作区，单次实乘临时区按 stream 次序复用。整形状打包超出可用空间时改用分段工作区，同一 stream 完成全部消费者后再复用该区域。用户提供的缓冲按句柄所有权规则处理。若缩小分块后仍无法获得必要工作区且无直接路径，在提交前返回错误。
- 同一调用的各阶段使用同一 stream，避免跨 stream 重用尚未消费的临时数据。同一 handle 的工作区或 stream 配置变更沿用库的同步规则。调用方在计算完成前保持 Device 缓冲有效。

##### 3. tilingkey规划策略

以 Host 枚举或对应 kernel 入口表达调度路径。设备参数包含 dtype、左右转置与共轭标志、逻辑尺寸和前导维、块尺寸和尾块信息、核数、标量、是否读 C、工作区偏移。启动路径由形状、布局、标量和平台能力选择。

| 调度路径 | 选择条件与动作 |
| --- | --- |
| 置零／缩放 | 无有效乘积且 beta 非一。仅启用 AIV，按逻辑 C 列或有界段调度。 |
| Cube 直读及直接写回 | S/H 满足紧凑和对齐条件。直接写回还要求 `alpha=1,beta=0`、单次完成 K，并且输出类型转换与尾块写入满足对应 SDK 路径。 |
| Cube 乘积及 Vector 写回 | 有效乘积的通用路径。按需打包 A/B，随后处理缩放、复数合成和逻辑域散写。 |
| Vector 小块或边界路径 | 短 K 或简小形状采用经实测选择的低开销路径。不能安全使用块搬运的边界以有界访问处理。 |

#### 3.2.2 kernel侧设计

1. **Init / CopyIn**：解析调度参数，确定有效 `M/N/K` 范围、输入步幅和输出块所有权。按列主序偏移搬运 A/B，给内部对齐区域补零。Cgemm 的交错复数拆为实部、虚部分面，在装载或合成符号时应用 `OP_C`。读取旧 C 的条件为 `beta!=0`，读取范围限定在逻辑域。
2. **Compute**：Sgemm 复用 arch22 FP32 Cube Matmul 并显式禁用 HF32。Cgemm 复用四个 FP32 实乘 `ArBr`、`AiBi`、`ArBi`、`AiBr`，按共轭标志组合成 FP32 实部和虚部。保留已有 Cgemm 的长 K 分段防护，核验 `512×512×1536`、K=1024 边界以及 K=4095/4096，确认当前 Sgemm 路径安全后再评估放宽分段。Hgemm 使用 FP16 输入及 FP32 累加，缩放与合成保持 FP32。小块路径执行同样的数学分支。 [Sgemm 长 K 问题记录](https://gitcode.com/cann/ops-blas/issues/1077)
3. **Epilogue / CopyOut**：最后一段 K 完成后应用 alpha 和必要的 beta 乘加。Hgemm 最终按 IEEE round-to-nearest-even 输出 FP16。写回以 `i+j·ldc` 为准，禁止跨到 padding 或逻辑矩阵之外。非对齐尾块优先用目标 SDK 验证过的有界写回，要求多个核的实际写入粒度不重叠。若写入粒度跨相邻列，合并相关输出写入的所有权，尾边界使用精确元素访问；同一物理块由单个写入者处理。
4. **流水与观测**：有足够片上空间且 SDK 支持时交叠面板搬入与 Cube 计算，跨 AIC/AIV 的阶段通过同 stream 顺序或已验证的事件机制交接。用 profiling 分离打包、乘积、累加与写回，定位带宽、并行度和 launch 次数瓶颈。

## 支持硬件

| 支持产品 | 编译和验证 |
| --- | --- |
| Atlas A2（含 Atlas 800T A2） | arch22，CANN 9.1.0，正式性能采集使用 Atlas 800T A2 910B3。 |
| Atlas A3（含 Atlas 800T A3） | arch22，CANN 9.1.0，使用与实机匹配的 `ascend910_93` 系列目标。 |

A2 构建使用 `bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`。A3 使用相同入口并替换 `--soc`。任务自验至少覆盖一种指定款型，记录实际 SKU、软件版本与提交；A3 适配需校验其构建和执行路径。

## 算子约束限制

| 项目 | 约束 |
| --- | --- |
| 布局 | 按前导维描述带 padding 的列主序 ND 矩阵。无广播、任意 stride、0 维 Tensor 或 batch 语义，无 AutoContiguous/ViewCopy。 |
| 原地输出 | C 的可写区域与 A/B 的可读区域不得重叠，由调用方保证。仅写逻辑 `m×n` 区域。Host 不扫描 Device 数据。 |
| 标量与异常 | alpha/beta 指向 Host，A/B/C 指向 Device。对有访问的数据按算术规则传播 Inf/NaN。零乘积分支不访问 A/B，beta 为零时不访问旧 C。 |
| 精度与可重复性 | 归约顺序可随形状及硬件调整，按混合容差验收。Cgemm 使用四实乘语义，3M 与 batched 入口独立。 |
| 异步 | 遵循 handle 绑定 stream 的执行顺序和缓冲生命周期。可共用的公开 ABI 位于 ops-blas 头文件。 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 通过条件 | 标准来源 |
| --- | --- | --- |
| Sgemm 精度 | FP32 档 `rtol=atol=2^-13`，匹配比例至少 0.99，绝对误差硬上限以 `1e-2` 为兜底。 | 任务书 §3.2、生态算子开源精度标准 |
| Cgemm 精度 | 实、虚部分别按 FP32 档判定，两部分均通过。 | 同上 |
| Hgemm 精度 | FP16 档 `rtol=atol=2^-10`，匹配比例至少 0.99，绝对误差硬上限以 `1e-1` 为兜底。 | 同上 |
| 正式性能 | 910B3 预热 5 次，取随后 10 次有效调用的 `msprof op` kernel Task Duration 算术平均。换算为 ms 后须 `avg_ms≤1.25×gpu_ms`。记录并排除异常样本。 | 任务书 §3.3、同参数 cuBLAS GPU kernel 基线 |
| 辅助性能 | 预置 Device 缓冲，预热 5 次，用 `aclrtEvent` 对 10 次下发计时并输出 `[PERF] case=<case_name> ... avg_us=<数值>`。脚本缺日志时判失败，用作交叉验证。 | 任务书 §3.3、附件测试脚本 |
| 内存 | 记录工作区峰值和回退路径，任务书 §3.4 无单独内存验收项。 | 任务书 §3.4 |

有限值满足 `abs(actual−golden)≤atol+rtol·abs(golden)` 时计为匹配。每个用例同时检查匹配比例与最大有限绝对误差，硬上限为 `max(兜底值,32×ULP(g_low))`，`g_low` 为最大有限误差点按输出 dtype 收窄后的 golden。NaN 同为 NaN、Inf 同号分别判定。 [混合容差精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)

- **Golden**：S/C 以 Netlib BLAS 3.10.0 的 `cblas_sgemm` / `cblas_cgemm` 为数学参考，测试可用冻结的 float64 累加后收窄到 FP32 的数据，报告记录选择。H 按 float64 累加、收窄到 FP32 再舍入至 FP16 生成 golden。Cgemm 分量单独记录误差。 [Netlib BLAS](https://www.netlib.org/blas/) · [cuBLAS GEMM](https://docs.nvidia.com/cuda/cublas/index.html)
- **功能覆盖**：ATK/GTest 加载 CSV。附件 S/C/H 分别有 334/333/333 条非性能用例（含显式负向）和 67/67/66 条性能用例。验证全部九种转置组合、非对称复数的 OP_C、实数 OP_C=OP_T、紧凑与 padding、整块及奇数尾部、4095³/4096³、负维度、非法枚举、非法前导维、空指针及校验顺序。用 guard 和哨兵检查 C padding，分别用空 A/B 和含 NaN 的 A/B 验证零乘积，用含 NaN 的旧 C 验证 beta=0。覆盖 FP16 次正规、接近有限上限、Inf/NaN 及舍入边界。保留均匀与正态输入及任务要求的标量取值。
- **性能覆盖**：200 条用例按算子和最大单矩阵逻辑大小分层（≤64 KiB、≤2 MiB、>2 MiB），剔除 m/n/k 为零的性能样本，包含 24 条非默认标量样本。基线按 transa/transb/m/n/k/lda/ldb/ldc/alpha/beta 逐项匹配并复核采集来源，补测带前导维填充的组合。对应一次调用存在多段 kernel 时按调用归并并报告各段 Task Duration，正式比较覆盖完整 Device 计算。端到端事件计时单独保存，排除 H2D、D2H、Host 造数及 golden 比对。
- **可维护性与故障定位**：复用校验、尺寸运算、工作区和索引逻辑。测试失败记录 dtype、矩阵尺寸、前导维、转置、标量、调度路径、首个偏差坐标及阶段耗时，以相同 CSV 单例复现。交付时提供测试用例包、精度和性能报告及日志、可追溯的代码提交和复现步骤。

当前文档给出设计与验收计划，算子真机编译、精度及性能结果待实现后采集。

## 兼容性分析

Sgemm 和 Cgemm 保留现有符号、签名、类型和状态码。Sgemm 对合法输入保留列主序数学语义，空输出分支调整为按任务顺序先校验前导维。现有 arch22 Cgemm 的 Device 标量路径与任务要求的 Host 标量语义存在兼容性差异。实现前在 ops-blas 核对调用点和公开约定，明确需要迁移的调用和产品范围。若仓库已有显式指针模式，则按其约定区分，禁止通过解引用尝试判断指针所在内存。Hgemm 通过新增声明导出，复用 arch22 构建目标和同族测试框架。arch35 作为工程参考，须使用各自 SDK 支持的能力。
