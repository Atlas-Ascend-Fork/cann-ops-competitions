# 需求背景（required）

## 需求来源

[昇腾社区 11 月算子开发任务](https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c?menu=tasks)要求为 Atlas A2/A3 实现 `aclblasSgemmBatched` 和 `aclblasHgemmBatched`。接口遵循 [cuBLAS Level 3](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmbatched) 的指针数组批量 GEMM 语义和 [ops-blas](https://gitcode.com/cann/ops-blas) 的句柄、错误码和异步执行约定。

## 背景介绍

两个接口已在 `include/cann_ops_blas.h` 声明，arch35 已有 S/H 实现，arch22 有 FP32 类型的 `aclblasGemmBatchedEx`。本任务补齐 arch22 下独立的 S/H 接口，在现有句柄绑定的 stream 上直调 Ascend C kernel。批量矩阵的地址由 Device 指针数组给出，各批之间允许分开分配。实现需要兼顾小矩阵的调度开销、大矩阵的 Cube 利用率及带 padding 的列主序读写。

# 需求分析（required）

## 需求描述

对每个批次原地计算 `C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`。S 的矩阵和标量为 FP32，H 的矩阵和标量为 IEEE FP16，参数 `aclblasHalf` 按位解释。实数 `OP_C` 等价于 `OP_T`，两个转置参数覆盖 N/T/C 的九种组合。Host 校验完成后异步下发，调用方在消费输出前同步 stream。

| 参数 | 布局与约束 |
| --- | --- |
| `handle` | 有效 ops-blas 句柄，携带 stream |
| `transa`、`transb` | `ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` |
| `m`、`n`、`k`、`batchCount` | 非负整数，所有批次共享维度和标量 |
| `Aarray`、`lda` | Device 指针数组，A 按列主序存储。N 下为 m×k，`lda ≥ max(1,m)`。T/C 下为 k×m，`lda ≥ max(1,k)` |
| `Barray`、`ldb` | Device 指针数组，B 按列主序存储。N 下为 k×n，`ldb ≥ max(1,k)`。T/C 下为 n×k，`ldb ≥ max(1,n)` |
| `Carray`、`ldc` | Device 指针数组，C 为列主序 m×n，`ldc ≥ max(1,m)`。仅更新逻辑区域 |
| `alpha`、`beta` | Host 标量指针，类型随 S/H 接口 |

## 需求拆解

1. 补齐 arch22 的两个公开入口，共用参数校验、退化分支、列主序地址计算和用例入口。
2. 根据矩阵规模、前导维度及 batch 数分配工作。确保 `beta=0` 时跳过旧 C 读取，`k=0` 或 `alpha=0` 时跳过 A/B 读取。
3. 按任务书覆盖精度、参数异常、特殊值和性能场景，分别采集正式的 kernel 耗时与辅助的端到端耗时。

# 详细设计（required）

## 算子分析

### 数学公式

对批次 `b`、行 `r`、列 `c`，计算 `sum(q=0..k-1, op(A[b])[r,q] * op(B[b])[q,c])`，在有效时乘 `alpha` 并加 `beta * C[b][r,c]`。A 的元素偏移在 N 下为 `q*lda+r`，T/C 下为 `r*lda+q`。B 的元素偏移在 N 下为 `c*ldb+q`，T/C 下为 `q*ldb+c`。C 元素偏移为 `c*ldc+r`，地址计算提升为 64 位。

### 支持数据类型

| 接口 | A/B/C、alpha/beta | 计算精度 |
| --- | --- | --- |
| `aclblasSgemmBatched` | FP32 | FP32 乘加，Cube 路径沿用 arch22 的 FP32 计算配置 |
| `aclblasHgemmBatched` | FP16 | FP16 输入，FP32 累加与缩放，写回时转 FP16 |

### 支持形状

`m/n/k`、前导维度和批次数为运行时参数。按独立批次访问矩阵，通过逻辑行列范围处理奇数尺寸、尾块和带填充布局。批次共享形状与转置类型。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

1. 按句柄、转置枚举、m/n/k、前导维度、batchCount、标量和数组指针的顺序校验。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。`m=0`、`n=0` 或 `batchCount=0` 的合法调用直接返回。
2. 非空输出要求 Carray 有效。仅矩阵乘积有效时要求 Aarray/Barray 有效。将 `OP_C` 归一为 `OP_T`，把 alpha/beta 转成计算标量传入 kernel。退化分支按 beta 选择清零、缩放或直接返回。
3. 以输出矩阵分块数与 batchCount 计算可并行任务数。小矩阵优先把多个 batch 合并到每次下发，大矩阵同时沿 batch、M、N 分配，必要时沿 K 分段并在写回前归约。核数、块形状和切换阈值由核类型、输入维度、对齐、存储容量和真机 profile 确定。

#### 3.2.2 kernel侧设计

1. 从 Device 指针数组取当前 batch 的地址。参考 arch22 `gemm_batched_ex` 的批次与输出块映射和 Cube tiling，将列主序的 `C=A×B` 等价为行主序 `Cᵀ=Bᵀ×Aᵀ`，保留用户传入的前导维度。H 使用原生 FP16 Cube 输入及 FP32 累加配置。
2. 对支持直接写回的 `alpha=1,beta=0` 形状，优先让 Cube 写回逻辑 C 块。需要融合标量或合并 K 分段时，在写回前用 Vector 完成 FP32 归约、缩放和类型转换。其他尺寸使用适用的 Vector 路径处理尾块、缩放和小规模乘积。选择路径依据算子参数与平台容量。
3. 缩放与写回覆盖实际 m×n 区域。仅 `beta≠0` 路径读取旧 C，且退化缩放路径只访问 Carray。必要时使用掩码或精确长度搬运处理列尾。使用 profiling 验证小矩阵启动成本、批量地址读取、Cube 利用率和写回开销。

### 工程接入

在 `blas/gemm_batched/arch22/` 增加 S/H 实现，在 `test/gemm_batched/sgemm_batched/arch22/` 注册共享 GTest，按 `op` 识别算子，补充 `blas/gemm_batched/README.md` 的平台、构建和调用说明。参数及测试组织参考仓内 arch35 实现。优化参考 [CANN Samples](https://gitcode.com/cann/cann-samples/tree/master/Samples) 的 Matmul 及数据搬运用例。

## 支持硬件

| 支持的产品 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2 | √ |
| Atlas 800T A3 | √ |

## 算子约束限制

A/B/C 为 Device 指针数组及 Device 矩阵地址，alpha/beta 为 Host 标量。调用方保证数组元素有效，C 批次之间及 C 与 A/B 的读写区域互不重叠。接口支持由前导维度描述的列主序 padding。每个批次使用相同维度、前导维度和转置标志。支持的布局、输入约束和测试构建入口与任务书一致。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | S、H 各 500 条。S 用 Netlib BLAS 3.10.0 `cblas_sgemm` 或任务书允许的双精度累加参考，H 用 FP64 累加后依次收窄为 FP32、FP16。S 的 atol/rtol 为 2⁻¹³，H 为 2⁻¹⁰，匹配比例均至少 99%。同步检查任务书规定的绝对误差上限、特殊值和 padding | 任务书 §3.2、§3.5 |
| 性能标准 | S、H 各 100 条。warmup 5 次后采集 10 次成功调用的 `msprof op` kernel Task Duration，平均耗时不超过同参 GPU 基线的 1.25 倍。GTest 用 ACL event 输出独立的端到端 `avg_us`，排除数据准备和传输 | 任务书 §3.3、§3.5 |
| 内存标准 | 任务书 §3.4 无独立内存指标。复核边界尺寸、padding 哨兵和工作区生命周期 | 任务书 §3.4 |

精度用例覆盖 N/T/C 全组合、合法空操作、非法状态码、alpha=0 时 A/B 为 NaN、beta=0 时 C 为 NaN、非连续批次地址、FP16 次正规数与 Inf/NaN、前导维度 padding、4095/4096 大矩阵及最多 65536 个小矩阵批次。归档测试用例、输入、golden、逐用例精度结果、两套性能数据和复现步骤。测试平台按报告记录实际型号和 CANN 版本。

## 兼容性分析

复用公开声明与现有 ACL BLAS 句柄，arch22 新增入口，arch35 构建仍选择各自的实现。参数、返回码和 stream 契约以现有 ops-blas 规范及任务书为准。
