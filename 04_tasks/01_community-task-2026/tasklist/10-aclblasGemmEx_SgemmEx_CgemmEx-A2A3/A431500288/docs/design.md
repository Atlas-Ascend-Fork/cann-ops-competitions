# aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 算子设计（A2/A3）

# 需求背景（required）

## 需求来源

本设计依据《10月社区任务-aclblasGemmEx_SgemmEx_CgemmEx算子开发（A2A3）》任务书，面向 Atlas A2/A3 系列产品，在 CANN 9.1.0 环境下使用 Ascend C 实现三个扩展矩阵乘法接口。

## 背景介绍

扩展矩阵乘法接口通过数据类型参数描述矩阵类型，并支持转置、标量缩放和结果原地更新。本任务需要补齐 A2/A3 实现，支持单精度实数和单精度复数计算。

按照任务书，三个接口均采用句柄式 kernel 直调方式，使用 handle 绑定的 stream 执行。实现代码放在 `blas/gemm_ex/arch22/`，测试代码放在 `test/gemm_ex/`，公共声明统一放在 `include/cann_ops_blas.h`。

# 需求分析（required）

## 需求描述

实现以下计算：

```text
C = alpha * op(A) * op(B) + beta * C
```

A、B、C 使用列主序存储，op(A) 的形状为 m×k，op(B) 的形状为 k×n，C 的形状为 m×n。A、B 为只读输入，C 为原地更新的输入输出。

| 接口 | 主路径数据类型 | 接口要求 |
| --- | --- | --- |
| aclblasGemmEx | FLOAT32 | 19 参，包含 computeType、algo |
| aclblasSgemmEx | FLOAT32 | 17 参，不包含 computeType、algo |
| aclblasCgemmEx | COMPLEX64 | 17 参，alpha/beta 为复数，不包含 computeType、algo |

## 需求拆解

1. 按任务书提供公共接口，统一参数校验和错误码处理。
2. 支持 N、T、C 三种操作；实数 C 与 T 等价，复数 C 表示共轭转置。
3. 支持合法前导维及 padding，正确处理非对齐尺寸和尾块。
4. 实现零维、k=0、alpha=0、beta=0/1 和空指针等边界行为。
5. 设计 Host 分核与分块策略、实数和复数 Kernel 计算流程。
6. 按任务书完成精度验证与性能测试，主路径性能达到 GPU 标杆的 0.8 倍以上。

# 详细设计（required）

## 算子分析

### 数学公式

实数路径：

```text
P(i,j) = sum[p=0..k-1] op(A)(i,p) * op(B)(p,j)
C_new(i,j) = alpha * P(i,j) + beta * C_old(i,j)
```

复数路径设运算后的矩阵为 `Ar+iAi`、`Br+iBi`，则：

```text
Pr = Ar*Br - Ai*Bi
Pi = Ar*Bi + Ai*Br
Cr_new = alpha_r*Pr - alpha_i*Pi + beta_r*Cr_old - beta_i*Ci_old
Ci_new = alpha_r*Pi + alpha_i*Pr + beta_r*Ci_old + beta_i*Cr_old
```

复数矩阵以实部、虚部交错存储。共轭转置在交换行列索引的同时对虚部取负。

### 支持数据类型

GemmEx、SgemmEx 的主路径 Atype/Btype/Ctype 为 `ACL_FLOAT`，alpha/beta 为 FP32；GemmEx 的主路径计算类型为 `ACLBLAS_COMPUTE_32F`，算法为 `ACLBLAS_GEMM_DEFAULT`。

CgemmEx 的主路径 Atype/Btype/Ctype 为 `ACL_COMPLEX64`，alpha/beta 使用公共类型 `aclblasComplex`。

数据类型组合通过 Host 支持表进行分发。FP16/BF16/FP8 等扩展组合按照任务书的支持范围要求补充对应计算分支和分 dtype 测试；支持集以外的组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

### 支持形状与布局

m、n、k 均为非负运行时参数，前导维满足：

| 参数 | 约束 |
| --- | --- |
| lda | transA=N 时不小于 max(1,m)，否则不小于 max(1,k) |
| ldb | transB=N 时不小于 max(1,k)，否则不小于 max(1,n) |
| ldc | 不小于 max(1,m) |

列主序下，零基索引的地址映射为：

```text
A_N(i,p)   = A[i + p*lda]
A_T/C(i,p) = A[p + i*lda]
B_N(p,j)   = B[p + j*ldb]
B_T/C(p,j) = B[j + p*ldb]
C(i,j)     = C[i + j*ldc]
```

ld 以对应数据类型的元素为单位。计算只访问有效矩阵区域，padding 不参与计算，输出不改写 padding。

## 算子实现

### 接口设计

三接口使用同一套参数语义，声明如下：

```cpp
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const void* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb, const void* beta,
    void* C, aclDataType Ctype, int ldc,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb, const float* beta,
    void* C, aclDataType Ctype, int ldc);

aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb, const aclblasComplex* beta,
    void* C, aclDataType Ctype, int ldc);
```

### 3.2.1 Host侧设计

#### 参数校验

按照任务书规定的顺序进行校验：handle → m/n/k → lda/ldb/ldc → alpha/beta → dtype 支持集 → transA/transB → A/B/C 指针 → GemmEx 的 computeType/algo → no-op 短路。

| 异常 | 返回值 |
| --- | --- |
| handle 为空 | ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| 维度、前导维或必需指针不合法 | ACLBLAS_STATUS_INVALID_VALUE |
| transA/transB 枚举不合法 | ACLBLAS_STATUS_INVALID_ENUM |
| 不支持的 dtype 组合、computeType 或 algo | ACLBLAS_STATUS_NOT_SUPPORTED |

alpha/beta 判空后读取标量值。k>0 且 alpha 非零时检查 A/B 非空，beta 非零时检查 C 非空。复数标量的零值要求实部和虚部均为零。

#### 边界处理

完成参数校验后，按照以下规则选择执行路径：

| 场景 | 处理方式 |
| --- | --- |
| m=0 或 n=0 | 返回 SUCCESS，不执行计算 |
| beta=0 且 C 为空 | 按任务书要求返回 SUCCESS，全程不读写 C |
| k=0 或 alpha=0，beta=1 | C 保持不变，不访问 A/B |
| k=0 或 alpha=0，beta=0，C 非空 | 将 C 有效区域置零，不读旧 C |
| k=0 或 alpha=0，其他 beta | 执行 C=beta*C，不访问 A/B |
| 其他 | 执行矩阵乘及 alpha/beta 合成 |

#### 分核与分块策略

将输出矩阵沿 M、N 方向划分为不重叠的二维块，分配给多个核心。各核心负责对应输出块的 K 方向累加及结果写回，避免多个核心同时更新同一输出元素。

K 方向分段搬入 A/B 数据，使计算所需的数据块适配片上存储。Host 根据输入形状、数据类型、设备核数和可用存储容量选择分块大小；小矩阵减少活跃核心，大矩阵通过增加并行输出块提高计算资源利用率。具体分块参数在开发阶段结合精度与性能测试确定。

Tiling 参数包含矩阵尺寸、前导维、数据类型、转置与共轭标志、标量值、分块大小及尾块有效尺寸。地址跨度和工作区大小使用足够位宽计算并检查溢出。

#### 工作区与执行流

按需设置输入整理和中间结果工作区，用于转置、复数拆分及部分结果保存。工作区容量根据分块策略计算，避免无条件复制完整矩阵。

各计算阶段使用 handle 绑定的同一 stream 按依赖顺序下发。工作区在相关计算完成前保持有效，调用者在读取 Device 输出前同步 stream。

### 3.2.2 Kernel侧设计

#### 实数计算流程

1. 数据搬入：按转置标志和前导维读取 A/B 的有效块，必要时进行布局整理。
2. 矩阵乘：拟采用 Ascend C 矩阵乘计算接口完成分块乘加，主路径使用满足任务精度要求的 FP32 计算配置。
3. 结果合成：计算 alpha*P+beta*C；beta=0 时跳过旧 C 的读取。
4. 结果搬出：按 ldc 将有效输出块写回列主序 C。

沿 K 方向的中间结果先累加，完成全部 K 计算后再执行一次 alpha/beta 合成，避免重复缩放旧 C。

#### 复数计算流程

拟将复数乘法拆为四组实数矩阵乘：ArBr、AiBi、ArBi、AiBr。数据搬入时拆分实部和虚部，并按 N/T/C 标志处理索引和共轭符号。

四组乘积完成后，根据数学公式形成 Pr/Pi，再与复数 alpha/beta 合成。读取旧 C 时先取得同一元素的实部和虚部，再写回结果，保证原地更新的正确性。输出恢复为实虚部交错布局。

#### 缩放、尾块与同步

缩放分支使用向量计算处理 C=beta*C。beta=1 时保留 C 原值，beta=0 时直接置零，其他 beta 执行实数或复数缩放；alpha=0 场景按任务书要求进行位精确验证。

尾块仅搬运和写回有效元素；参与矩阵乘的临时补齐区域填零。通过队列或事件管理搬入、计算、搬出之间的依赖，缓冲区在使用完成后复用。

### 性能优化思路

在正确性满足要求的基础上，重点考虑以下优化：

- 根据方阵、长条矩阵和小矩阵的形状特点调整分块与核数。
- 复用 A/B 分块数据，减少重复搬运；结合片上容量考虑双缓冲。
- 对可直接表达的紧凑布局减少中间数据整理。
- 合并缩放、复数结果合成与写回，减少中间结果搬运和 Kernel 下发。

优化配置通过目标硬件测试确定，不在设计阶段预设收益数值。

## 支持硬件

| 支持产品 | 说明 |
| --- | --- |
| Atlas A2 系列，含 Atlas 800T A2 | 功能适配；910B3 用于任务要求的性能验证 |
| Atlas A3 系列，含 Atlas 800T A3 | 功能适配；自验证按任务书要求至少覆盖一种款型 |

## 算子约束限制

矩阵按列主序及合法前导维存储，不支持其他任意非连续步长，不涉及广播。C 原地覆写，不返回视图。GEMM 不要求位精确；alpha=0 的缩放场景按任务书执行 EXACT 校验。异步执行依赖 handle 绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 要求 | 来源 |
| --- | --- | --- |
| FLOAT32 | rtol=atol=2^-13，matched_ratio≥0.99，最大绝对误差上限为 1e-2 或 32×ULP | 任务书 §3.2 |
| COMPLEX64 | 实部、虚部分别按 FLOAT32 阈值进行全矩阵验证 | 任务书 §3.2 |
| 扩展数据类型 | FP16 的 rtol/atol 为 2^-10，BF16 为 2^-7，按各 dtype 标准判定 | 任务书 §3.2 |
| alpha=0 | C=beta*C，位精确校验 | 任务书 §3.2 |
| 性能 | NPU 平均耗时不高于对应 GPU 标杆的 1.25 倍 | 任务书 §3.3 |

精度 Golden 使用 Netlib BLAS 3.10.0：实数接口对比 `cblas_sgemm`，复数接口对比 `cblas_cgemm`。Inf/NaN 按任务书规定单独比较，不纳入有限值误差统计。

## 测试设计

使用任务配套 CSV，并补充未覆盖场景。测试内容包括：

- 小尺寸、非对齐尺寸、长条矩阵及大矩阵。
- N/T/C 全组合，复数共轭与普通转置的差异。
- 前导维 padding、对齐偏移、尾块及输出区域边界。
- alpha/beta 特殊取值、零维、k=0、空指针及参数校验顺序。
- 不支持的数据类型组合、非法枚举、负维度和非法前导维。
- 均匀/正态输入、Inf/NaN 及任务规定的其他特殊值。

负向测试检查返回值，正向测试检查 C 的全矩阵结果，复数分别检查实部和虚部。缩放分支增加 EXACT 校验。

性能在 Atlas 800T A2 (910B3)、CANN 9.1.0 上使用 msprof op 采集。预热 5 次后采集有效样本，按照任务书要求统计平均耗时；单次接口调用涉及多个 Kernel 时，应统计完整计算流程。逐 case 与对应 gpu_baseline.csv 比较，并在自测报告中保留采样次数、原始耗时及异常说明。

## 兼容性分析

公共接口与 950PR 共用，不定义产品私有平行 API。按任务书保留 GemmEx 的 19 参接口，将 SgemmEx 修正为 17 参，新增 CgemmEx 的 17 参接口。

SgemmEx 签名调整时同步修改相关实现、调用样例和测试，避免公共声明与各架构实现不一致。完成开发后更新 GemmEx README，补充接口说明、约束、使用方式及 A2/A3 产品支持信息。

## 参考资料

1. 《aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx 任务书》及配套测试材料。
2. [社区任务设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
3. [ops-blas 开源仓](https://gitcode.com/cann/ops-blas)。
4. [cuBLAS 接口文档](https://docs.nvidia.com/cuda/cublas/index.html)。
5. [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
