# 需求背景

## 需求来源

11 月 CANN 社区任务 `aclblasCsyrk3mEx 算子开发（950）`，以任务包中的 `aclblasCsyrk3mEx_Atlas950PR_task_doc.md` 为规格和验收依据。在 Ascend 950PR 上使用 Ascend C，基于 `cann/ops-blas` 的句柄式 API 新增单精度复数对称矩阵秩-k 更新接口，对标 cuBLAS `cublasCsyrk3mEx`。[ops-blas](https://gitcode.com/cann/ops-blas) · [cuBLAS 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublascsyrk3mex)

## 背景介绍

### 现状分析

`ops-blas` 的 arch35 `aclblasCsyrk` 提供单精度复数对称秩-k 更新，可复用枚举检查、前导维度处理、对角块裁剪和复数缩放的设计。`cgemm3m` 提供复数乘积分解及混合核的搬运与同步参考。`aclblasCsyrk3mEx` 独立提供 Ex 类型分派，两个输入面共用三次实数乘积的对称更新流程。11 月同批 `aclblasCsyrkEx` 为独立接口，无构建或运行依赖。[非 Ex CSYRK](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/blas/syrk/arch35/csyrk_host.cpp) · [CGEMM 混合核](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch35/cgemm_onchip_kernel.h)

### 算子功能分析

算子对 `C` 的一个三角执行 `C ← α·op(A)·op(A)ᵀ + β·C`。矩阵沿用 BLAS 列主序布局，`C` 经原指针返回，显式计算对角元素的实部和虚部。上三角和下三角是独立存储选择，调用方可把另一侧用于放置任意数据。[Netlib CSYRK](https://netlib.org/blas/csyrk.f)

# 需求分析

## 需求描述

在 `include/cann_ops_blas.h` 新增以下声明，复用 `cann_ops_blas_common.h` 中的句柄、状态码、枚举及 `aclblasComplex` 定义。通过 `aclblasSetStream` 绑定的 stream 下发 Ascend C kernel。

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

## 需求拆解

| 项目 | 开发要求 |
| --- | --- |
| 参数与数据类型 | 支持 `UPPER/LOWER`、`OP_N/OP_T/OP_C` 及两种 A/C 类型组合，沿用 ops-blas 的接口错误码。 |
| 形状与数据访问 | 支持运行时 `n/k`、任意合法 `lda/ldc` 和矩阵边缘，按复元素寻址。只读写指定三角，矩阵列填充保留。 |
| 计算 | 同一次 kernel launch 内执行三路实数乘积及复数组合，输出 `α·X·Xᵀ + β·C`。 |
| 退化与异步语义 | 结构检查完成后分派空操作、只缩放和完整更新。支持 Host、Device 标量位置，使用句柄绑定的 stream。 |
| 工程与验收 | 在 ops-blas 注册编译及测试目标，按任务书验证 FLOAT32 分量精度与 kernel-to-kernel 性能。 |

# 详细设计

## 算子分析

### 数学公式

令 `X = op(A) = R + iI`，逻辑形状为 `n × k`。对于 `UPPER` 的 `i ≤ j` 或 `LOWER` 的 `i ≥ j`，定义下列实数秩-k 乘积。`i`、`j` 分别为行号和列号，均从 0 开始。

```text
P0 = R Rᵀ
P1 = I Iᵀ
P2 = (R + I)(R + I)ᵀ
Qr = P0 − P1
Qi = (P2 − P0) − P1

Cr' = αr·Qr − αi·Qi + βr·Cr − βi·Ci
Ci' = αr·Qi + αi·Qr + βr·Ci + βi·Cr
```

两侧面板均来自同一个 `X`，右面板作普通转置。`OP_C` 与 `OP_T` 使用同一寻址，`Qi` 包含两项交叉乘积之和。例如 `n=k=1`、`A=1+2i`、`α=1`、`β=0` 时输出 `−3+4i`，对角虚部参与计算。`β=0` 时直接写秩-k 项，旧 `C` 可含 NaN 且无需读取。

### 支持数据类型

| 参数 | 数据类型及存储 |
| --- | --- |
| `alpha`、`beta` | `aclblasComplex`，实部和虚部各为 FP32，支持 Host 或 Device 标量。 |
| `A`、`Atype` | `ACL_COMPLEX64` 为交错 FP32 对，每个复元素 8 B。`ACL_INT8` 为交错有符号 INT8 对，每个复元素 2 B，对应 `CUDA_C_8I`。 |
| `C`、`Ctype` | `ACL_COMPLEX64`，每个复元素 8 B，原地更新。 |

INT8 的两个分量分别符号扩展并精确转换为 FP32，再生成 `R+I`，其取值范围为 `[-256,254]`。乘积、归约及输出默认采用 FP32 精度，INT8 长 K 的分段处理见 Kernel 设计。

### 支持形状

| `trans` | A 的物理矩阵形状 | `X(i,t)` 的复元素偏移 | `lda` 下界 |
| --- | --- | --- | --- |
| `ACLBLAS_OP_N` | `n × k` | `i + t·lda` | `max(1,n)` |
| `ACLBLAS_OP_T`、`ACLBLAS_OP_C` | `k × n` | `t + i·lda` | `max(1,k)` |

`C` 的逻辑形状为 `n × n`，存储首维为 `ldc ≥ max(1,n)`，偏移为 `i + j·ldc`。`n,k` 为非负 `int`。列填充通过 `lda/ldc` 表达。偏移和字节跨度在乘法前提升到 64 位，Host 检查最大可达偏移、字节换算和 tile 数是否溢出。任务包的尺寸覆盖用于选择和验证 tile，API 边界以参数类型、地址空间及合法缓冲为准。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计（含 Tiling）

参数按任务书 §2.5 的顺序校验，同一调用返回最先检出的错误。

| 顺序 | 检查与状态 |
| --- | --- |
| 1 | `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。 |
| 2 | `uplo/trans` 超出支持枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`。校验后将 `OP_C` 归一为 `OP_T`。 |
| 3 | `n/k` 为负、`lda/ldc` 低于下界或 `n>0` 时 C 的跨度无法安全表示，返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| 4 | `alpha/beta` 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。`Atype/Ctype` 组合超出支持集返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。`n>0` 时 `C` 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| 5 | `n=0` 时结构校验完成后成功返回。其余场景中，仅 `k>0` 且 `alpha≠(0,0)` 时检查 `A` 非空及活跃地址跨度，失败返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |

Host 标量取值后按值传给 kernel。Device 标量沿用同一调用的两个标量位于同一侧的 ops-blas 约定。通过现有指针位置查询识别两种模式。对需要读取的 Device 标量，在绑定 stream 上排队小尺寸 D2H 并等待该批拷贝一次，再判定空 `A`、无下发返回及缩放分支。`k=0` 时只读取 `beta`。该等待纳入 Device 标量模式的端到端耗时，后续 kernel 仍在该 stream 上异步下发。若拷贝部分提交后失败，按运行时约定回收或等待已在途拷贝，保持 Host 目标缓冲存活。查询、拷贝或 launch 失败时沿用 ops-blas 的状态映射并记录底层错误码。

| 已验证条件 | 分派 |
| --- | --- |
| `n=0` | 成功返回，无 kernel 下发，无矩阵访问。 |
| `k=0` 或 `α=0`，且 `β=1+0i` | 成功返回，无 kernel 下发，`C` 保持原值。 |
| `k=0` 或 `α=0`，且 `β≠1+0i` | 下发 AIV 三角缩放，`β=0` 时按元素写零，整个路径跳过 `A`。 |
| 其余 | 根据形状和硬件资源选 AIV 或 AIC/AIV 混合 3M kernel，单次下发。`β=0` 关闭 `C` 输入搬运，`β=1` 直接加旧值。 |

##### 1. 分核策略

将 C 的两个坐标轴按相同边长 `B` 划为 `b = ceil(n/B)` 个区间。只枚举 `T = b·(b+1)/2` 个三角 tile。上三角用 `t = c·(c+1)/2+r` 映射 `0≤r≤c<b`，下三角交换行列。用整数前缀查找逆映射，最后一个区间裁到 `n`。设 `G` 为平台支持且工作量可用的逻辑核组数，组 `g` 处理 `g, g+G, …`。每组独占自己 tile 的完整 K 区间，含对角 tile 的裁剪后输出，无跨组归约或原子写。AIV 路径按有效三角列段分派，避免小矩阵为保持空闲配对而增加启动开销。

##### 2. 数据分块和内存优化策略

混合 FP32 候选 tile 从 `B=64, BK=64` 起步，Host 按平台 AIC/AIV 核数和 UB、L1、L0A/B/C 可用容量筛选。单个 FP32 面板为 16 KiB，三路结果 tile 共 48 KiB。L1 输入双缓冲粗算 `2×3×2×B×BK×4 = 192 KiB`。L0A/B 以每侧一套工作面板顺序复用，L0C 保存三个独立累加块。按 COMPLEX64 估算，单 AIV 的原始交错输入双缓冲为 64 KiB，三路打包暂存为 48 KiB，单槽三路输出半块加复数读写暂存为 40 KiB。该配置合计 152 KiB/AIV，另计对齐、风险标记和接口开销。满足硬件对齐与容量检查后才开启双缓冲。

`OP_N` 用同一列中的连续行段搬运，`OP_T/C` 用同一列中的连续 K 段搬运，再在片上统一面板布局。对角 tile 可复用同一输入面板的解交织结果，T/C 与 N 的源布局各自选择适用的连续搬运和重排。通用方案以内核片上缓冲工作，无全局 `n×n` 中间矩阵。候选 tile 与阈值由编译资源和 950PR 实测确认。tile 缩小仍不合适时使用单 launch AIV 路径，按列段分工并保持同样的 3M 原语。

##### 3. Tiling key 规划策略

分派键包含执行方式（缩放、AIV 3M、混合 3M）、A 类型及 N/T 布局。参数区携带 `n/k/lda/ldc`、`uplo`、`B/BK`、有效核组数、各维 tile 数及尾块长度、双缓冲配置和标量分量。用于启停旧 `C` 搬运的 `beta=0/1` 标记随参数传递。根据形状、布局和片上资源分派，运行策略与测试编号和随机种子无关。

#### 3.2.2 Kernel 侧设计

| 阶段 | 混合 3M kernel 的处理 |
| --- | --- |
| Init | 一个逻辑组配对一个 AIC 和两个 AIV，按相同 tile 序列运行。初始化 UB/L1/L0 分区和输入、输出槽的 ready/free 事件。Cube 显式禁用 HF32，确保 FP32 面板和累加路径。 |
| CopyIn | 两个 AIV 分别准备左右面板，通过安全的定长或带尾搬运读有效 A 段，解交织出 `R/I` 并生成 `R+I`。将按 Cube 布局打包的数据由 UB 写入 L1。行、列和 K 对齐扩展只在片上补零。 |
| Compute | AIC 等待两侧输入就绪，经 L1 搬至 L0A/L0B。对逻辑 tile `C[行块,列块]`，以列块面板乘行块面板的转置，生成按 `C` 列连续的内部结果。每个 K 面板顺序执行 `R×Rᵀ`、`I×Iᵀ`、`(R+I)×(R+I)ᵀ`，第一片初始化各自 L0C，后续片续加。同一个 AIC/AIV kernel launch 包含三路计算、合成和写回。 |
| Epilogue | AIC 用 Fixpipe 将三块 FP32 结果搬至配对 AIV 的 UB，内部行优先对应 `C` 的连续列段，两个 AIV 分持不同逻辑列。AIV 按 `Qr/Qi` 合成复数结果，加权并与可选的旧 `C` 合并。只为有效三角区间读取旧 `C`。 |
| CopyOut | 对每列取 tile 行区间、矩阵边界和 `uplo` 的交集。`UPPER` 行上界为 `j+1`，`LOWER` 行下界为 `j`。批量写有效连续段，短尾按实际字节数搬运或逐复元素写回。每列填充和未选三角的字节均保留。 |

输入事件在 AIV 写 L1 完成及 AIC 搬取完成后分别发布 ready/free。输出事件在 Fixpipe 完成和 AIV 消费完成后分别发布 ready/free，空半块也参与对应握手。固定硬件对齐的 Cube tile 仅用于片上缓冲，L0C 双目的按 M 切分要求偶数 M，尾块先补齐再按逻辑范围写 GM。各路径使用一致的事件次数、成对槽号及末尾排空顺序，避免覆盖在途数据。[CGEMM 片上协作](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch35/cgemm_onchip_kernel.h) · [L0C 至 UB 双目的搬运](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_key_features/l0c_buffer_to_ub_dual_target_mode.md)

小尺寸 AIV 路径按输出元素分配三个 FP32 累加器，沿 K 用相同 3M 式计算并复用三角写回。缩放路径复用列段裁剪和复数输出，只执行 `β·C` 或填零。所有输出路径均按复数对读写，内核回写前的对齐零填充局限于片上内存。

**数值处理。** 默认采用原生 FP32 MMAD。对消风险集中在 `R+I`、`P0−P1`、`P2−P0−P1`，以 3M 专项和大 K 样本评估分段累加。INT8 需要分段时取每段 `K≤256`，每路乘积和的绝对值上界为 `256×256² = 2²⁴`，FP32 能精确表示该范围内的整数。分段结果分别转入足够宽的整数或经验证的补偿累加后完成重构，最后进入 FP32 epilogue，分段归约仍在同一次下发内完成。对参与计算的非有限分量及 FP32 `R+I` 溢出，使用 CSV 复现主路径与 CBLAS 的结果分类。若 3M 展开导致分类差异，依据输入分量、面板风险标记和 CBLAS 顺序在同 launch 的 AIV epilogue 修正受影响的元素，保留其余元素的 3M 路径。`α=0`、`β=0` 和空操作从入口保证对应输入无需访问。[FP32/HF32 模式](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute_aux_config/SetHF32Mode.md)

### 工程接入

| 位置 | 变更 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增与任务书参数次序一致的公共声明。 |
| `blas/syrk/arch35/` | 组织 `csyrk3m_ex_host.cpp`、`csyrk3m_ex_kernel.cpp` 与其 Tiling 定义，按仓库约定注册 arch35 编译和下发。 |
| `blas/syrk/README.md` | 新增 API、数据类型、三角语义、合法输入及产品支持说明。 |
| `test/syrk/csyrk3m_ex/arch35/` | 增加 GTest、ATK 用例装配、比较逻辑及构建目标。 |

## 支持硬件

| 芯片 | 支持 | 软件及实现约束 |
| --- | --- | --- |
| Ascend 950PR（arch35） | √ | CANN 9.1.0 及以上，构建选项 `--soc=ascend950`。Cube 负责实数乘积，Vector 负责解交织、复数合成和三角读写。实现和自验均以该款型为目标。 |

## 算子约束限制

- 支持列主序、由 `lda/ldc` 描述的带填充内存。矩阵需位于有效 Device 存储，`C` 的可写区域与 `A` 的可读区域须互不重叠。接口无广播、任意 stride 视图或 0 维矩阵张量语义。
- `n/k` 为本次调用的运行时尺寸，Host 据此切分与下发。Host 无需扫描 Device 张量元素或验证另一三角。三角访问遵守本接口语义，调用方负责缓冲大小和生命周期。
- `C` 原地更新，返回成功意味着 Host 检查和所需的下发成功。调用方在绑定 stream 完成后读结果。3M 舍入和归约顺序由执行路径决定，精度按混合容差验收。
- 任务书 §3.4 将独立内存验收项标为不涉及。实现仍检查片上缓冲预算，并用哨兵和尾块用例验证 GM 访问范围。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 设计与判据 | 标准来源 |
| --- | --- | --- |
| 精度 | CPU Netlib BLAS 3.10.0 `cblas_csyrk` 为 golden，`T/C` 均映射为 `CblasTrans`，INT8 按实部、虚部精确提升，需要 FP64 复核时用 `cblas_zsyrk`。对有效三角的 FP32 实、虚分量分别比较。`abs(error) ≤ 2⁻¹³ + 2⁻¹³·abs(golden)` 为匹配，用例 `matched_ratio ≥ 0.99`，且最大绝对误差不超过 `max(1e−2, 32×ULP(golden32))`，ULP 取最大绝对误差分量对应的 golden。NaN 对 NaN、Inf 比较同号，各自记录，并从有限值最大误差统计中排除。 | 任务书 §3.2，[混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能 | 950PR 上逐用例通过 `msprof op` 采样 `OpBasicInfo.csv` 中目标 kernel 的 `Task Duration(us)`。预热 5 次，随后 10 个有效样本取均值并换算 ms。逐例 `gpu_ms/npu_ms ≥ 0.4`，全部倍率算术平均值 `≥ 0.5`，同时满足。`gpu_ms` 来自相同参数的 cuBLAS kernel-only 基线，重复键按用例标识保留。INT8 另采样并配对同参数的 COMPLEX64 基线。 | 任务书 §3.3，`gpu_baseline.csv` |
| 内存 | 记录候选 tile 的片上预算、输入形状及填充，哨兵检查未选三角和列填充保持原值。 | 任务书 §2.5、§3.4 |

### 自验与问题定位

1. 使用 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 构建。用 ops-blas ATK/GTest 加载任务包中的 `csyrk3m_ex_test.csv`，记录 1000 条精度用例的实际执行数（含 4 条 `TC_3M_*`），基于两个 A 类型覆盖有效及无效组合。数据涵盖均匀、正态、有符号 INT8 极值、NaN/Inf、对消和非零对角虚部。脚本 `verify_accuracy.py` 用于核对用例通过和漏跑情况。
2. 补充正向校验 `OP_T/C` 等价、`n=1` 的复对角、`n=0` 的空 A/C、`k=0`、空 A 与零 `alpha` 的组合、`beta=0` 且旧 C 为 NaN、三角掩码及 `lda/ldc` 的 `+1/+8/+64` 填充。验证两种标量模式与 stream 依赖，并为错误优先级、空 handle、非法枚举、前导维度及非法 dtype 增加断言。保持未选三角及 padding 的字节级哨兵。用 profiling 核验空操作没有 kernel、完整 3M 只有一次下发。
3. 逐条运行官方 200 条 COMPLEX64 性能用例，按用例过滤采集，例如 `msprof op --application="./build/test/syrk/csyrk3m_ex/csyrk3m_ex_test --gtest_filter=*TC_PF*" --output=./prof_csyrk3m_ex`。原始 `Task Duration(us)`、排除的无效样本、`gpu_ms`、比值及汇总写入自测报告。`verify_performance.py` 和 GTest `[PERF] avg_us` 用于回归趋势和 Host 下发排查，正式报告使用 `msprof op` 的 kernel 耗时。
4. 保留用例 ID、seed、标量模式、类型、`uplo/trans/n/k/lda/ldc`、实际分派与环境版本。精度失败定位到实或虚分量、矩阵坐标、golden 和实际值。性能失败记录 tile 策略、核组数及对齐方式，分析 Cube 利用率、UB/L1 搬运与尾块开销。

## 兼容性分析

本任务新增 arch35 API，沿用已有公共句柄、stream、`aclblasComplex` 布局和状态码。既有 `aclblasCsyrk` 的调用方式及实现入口保持原样。Ex 类型组合采用任务书约定。非法 `uplo/trans` 按 ops-blas 返回 `ACLBLAS_STATUS_INVALID_ENUM`，cuBLAS 对应枚举错误映射为 `CUBLAS_STATUS_INVALID_VALUE`。算子开发完成后在 950PR 执行上述自验，记录精度、性能及构建结果。
