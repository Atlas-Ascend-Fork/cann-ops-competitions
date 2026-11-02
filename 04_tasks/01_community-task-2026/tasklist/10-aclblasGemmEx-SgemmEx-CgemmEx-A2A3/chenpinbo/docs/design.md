# aclblasGemmEx、aclblasSgemmEx、aclblasCgemmEx（A2/A3）算子设计

## 1 需求背景

### 1.1 需求来源

[10月社区任务：3个GemmEx类算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/2b4e298cbf6e4ed89e7fbcfaea1a5b02)及其配套任务书、自测用例和 GPU 基线。实现提交至 [ops-blas](https://gitcode.com/cann/ops-blas)。接口参数分别对齐 cuBLAS 的 `cublasGemmEx`、`cublasSgemmEx` 和 `cublasCgemmEx`，任务特有的校验顺序、空指针和精度要求按任务书执行。

### 1.2 背景介绍

在现有句柄式 BLAS 框架中补齐 Atlas A2/A3 的扩展矩阵乘法能力，采用 Ascend C kernel 直调方式，通过 handle 绑定的 stream 异步执行。现有 `arch35` 实现作为工程与接口兼容性参考；本任务的设备实现放在 `blas/gemm_ex/arch22/`，测试放在 `test/gemm_ex/`，公共接口统一声明于 `include/cann_ops_blas.h`。

## 2 需求分析

### 2.1 数学语义与数据布局

三个接口均计算：

\[
C_{ij}\leftarrow\alpha\sum_{p=0}^{k-1}\operatorname{op}(A)_{ip}\operatorname{op}(B)_{pj}+\beta C_{ij},\quad 0\le i<m,\ 0\le j<n.
\]

`op(A)` 的逻辑维度为 `m×k`，`op(B)` 为 `k×n`，输出 `C` 为 `m×n`。矩阵按列主序存储，元素地址为 `base + row + column * ld`，其中 `ld` 以元素为单位；复数元素为实部、虚部交错的 `aclblasComplex`。只写 C 的有效矩阵区域，保留前导维填充区。

| 运算枚举 | 实数路径 | 复数路径 |
| --- | --- | --- |
| `ACLBLAS_OP_N` | 原矩阵 | 原矩阵 |
| `ACLBLAS_OP_T` | 转置 | 转置 |
| `ACLBLAS_OP_C` | 与转置相同 | 共轭转置，虚部变号 |

### 2.2 接口原型

```cpp
aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const void* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);

aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const float* beta, void* C, aclDataType Ctype, int ldc);

aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const void* B, aclDataType Btype, int ldb,
    const aclblasComplex* beta, void* C, aclDataType Ctype, int ldc);
```

`GemmEx` 为 19 参数，`SgemmEx` 和 `CgemmEx` 均为 17 参数，后两者无 `computeType` 和 `algo`。修正现有 `SgemmEx` 的 15 参数声明及相应调用方，新增 `CgemmEx` 声明，不引入产品私有接口。

### 2.3 参数与支持范围

| 参数 | 约定 |
| --- | --- |
| handle、stream | handle 在 Host，使用已创建句柄和 `aclblasSetStream` 绑定的 stream。 |
| alpha、beta | Host 标量指针。GemmEx 指向 FP32，SgemmEx 指向 `float`，CgemmEx 指向 `aclblasComplex`。 |
| A、B、C | Device 内存。A/B 只读，C 原地覆写；三矩阵无广播。 |
| Atype/Btype/Ctype | 本次 arch22 主路径：GemmEx/SgemmEx 均为 `ACL_FLOAT`，CgemmEx 均为 `ACL_COMPLEX64`。配套用例未规定扩展 dtype 正向支持，其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。 |
| computeType、algo | 仅 GemmEx：支持 `ACLBLAS_COMPUTE_32F`、`ACLBLAS_GEMM_DEFAULT`；其他取值返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。 |
| m、n、k | 非负运行时整数。 |
| lda | `transA=N` 时不小于 `max(1,m)`，否则不小于 `max(1,k)`。 |
| ldb | `transB=N` 时不小于 `max(1,k)`，否则不小于 `max(1,n)`。 |
| ldc | 不小于 `max(1,m)`。 |

前导维允许 padding，不支持除此之外的任意步长。数值合法域为相应 dtype 的值域，随机输入的 `[-5,5]` 仅用于测试生成，不限制接口输入。

### 2.4 参数校验与边界

按任务书次序校验：handle → m/n/k → lda/ldb/ldc → alpha/beta → dtype 支持集 → 转置枚举 → A/B/C 指针 → GemmEx 的 computeType/algo → no-op。

- handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法转置枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；不支持的 dtype、computeType 或 algo 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；其余非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。多项同时非法时保留上述优先级。
- alpha、beta 不可为空。`k>0` 且 alpha 非零时 A/B 不可为空；beta 非零时 C 不可为空。这些校验在 `m=0` 或 `n=0` 的短路之前执行。
- 全部校验通过后，`m=0` 或 `n=0` 直接成功返回，不启动 kernel。
- `beta=0` 且 `C=nullptr` 时直接成功返回，不读写 C，`k=0` 同样如此。C 非空且 beta 为零时不读取 C 旧值。
- `k=0` 或 alpha 为零时不引用 A/B，只处理 `C=beta*C`。beta 为零时写零；beta 为一时保留 C；其他值执行缩放。alpha 为零的缩放按任务书 EXACT 要求验证。

## 3 详细设计

### 3.1 Host 侧

三接口共享参数校验与分派逻辑；GemmEx/SgemmEx 复用 FP32 计算路径，CgemmEx 保留复数标量和独立的共轭标志。Host 读取标量，获取平台核数、计算 tiling，并在 handle 的 stream 上下发任务。

按一般输入属性选择缩放、小矩阵和分块矩阵乘法路径。tiling 携带逻辑维度、前导维、转置/共轭标志和标量，不依据用例编号或输入地址分派。地址和工作区容量计算采用足够宽的整数并检查溢出。需要临时工作区时沿用仓内分配及 stream 生命周期管理方式，保证异步访问期间有效，不在正常调用中同步回读矩阵。

### 3.2 Kernel 侧

实数路径将矩阵乘法映射到 Cube，缩放和输出合并映射到 Vector。利用 `C^T=op(B)^T*op(A)^T` 对应列主序和行主序，按实际布局直接读取或按块整理输入。M/N 方向划分输出块，每个输出块由一个核负责；K 方向分段累加，处理非对齐尾块。FP32 路径以精度要求为前提选择矩阵乘法模式，避免未经精度验证的降精度乘法。

复数路径在应用转置和共轭后记为 `A=Ar+iAi`、`B=Br+iBi`，使用实数矩阵乘法形成：

\[
P_r=A_rB_r-A_iB_i,\qquad P_i=A_rB_i+A_iB_r.
\]

Vector 完成实虚部拆分、共轭变号及复数 alpha/beta 合并：

\[
C'_r=\alpha_rP_r-\alpha_iP_i+\beta_rC_r-\beta_iC_i,
\]
\[
C'_i=\alpha_rP_i+\alpha_iP_r+\beta_rC_i+\beta_iC_r.
\]

按块复用输入和临时缓冲，流水重叠搬运与计算，减少重复拆分和中间结果的全矩阵读写。小矩阵可采用直接向量归约以减少启动、整理和工作区成本；实际分派边界由目标设备验证确定。所有路径均覆盖有效区域和尾块，beta 为零时在加载旧 C 前分支，避免无效输入传播 NaN。

### 3.3 工程与兼容性

实现遵循 ops-blas 的 `arch22` 自动源码收集及 CSV/GTest 测试机制，依赖 CANN 9.1.0。CPU golden 仅在测试端依赖 Netlib BLAS 3.10.0，不作为设备实现的计算后端。同步更新算子 README 的接口、约束和 A2/A3 支持表。

公共 `SgemmEx` 签名修正须同步适配 `arch35` 实现与相关调用，保留该产品线原有计算能力；按产品分别构建并回归受影响接口。新 CgemmEx 在本任务覆盖 A2/A3，不宣称新增其他产品的设备支持。

| 目标产品 | 支持范围 |
| --- | --- |
| Atlas A2 系列产品 | 本任务三个接口，arch22。 |
| Atlas A3 系列产品 | 本任务三个接口，arch22。 |
| Ascend 950PR/950DT | 公共头文件共用，回归受签名变更影响的已有实现。 |

## 4 可维可测分析

### 4.1 正确性与精度

采用 `test/gemm_ex/` 的 CSV/GTest 框架，分别调用三个公开接口。GemmEx/SgemmEx 使用 Netlib 3.10.0 `cblas_sgemm` golden，CgemmEx 使用 `cblas_cgemm`，对有效矩阵的实部、虚部分别完整比对。

FP32 及 COMPLEX64 各分量均使用 `atol=rtol=2^-13`、匹配比例至少 0.99，同时满足任务书的最大绝对误差限制 `1e-2` 或 `32*ULP`。alpha 为零的缩放分支另做 EXACT 比对；beta 为一时检查原位保留，beta 为零时验证不读取旧 C。Inf/NaN 按任务书单独检查：同为 NaN、同号 Inf，或 NaN 与 Inf 配对视为等效；有限与非有限不等效。非有限元素单独统计并全部通过，有限元素仍按混合容差和最大误差判定。

覆盖附件三组 CSV 的全部用例及 GPU 基线对应性能用例，补充未覆盖的校验优先级、全部 N/T/C 组合、复数非零虚部标量、padding 保护、非对齐地址与尾块、零维、k=0、alpha=0、空指针和非有限值。附件每接口提供 1000 条精度及 200 条性能用例，按实际文件完整回放，不按任务书概述的“200条”截断。扩展 dtype 不支持组合检查返回码，不伪报正向覆盖。

### 4.2 性能

正式性能设备使用 Atlas 800T A2（910B3），环境为 CANN 9.1.0；A3 可用于开发与精度自测。通过官方 `msprof op` 采集 task duration，按任务书 §3.3 使用 5 次 warmup 后的 10 次有效样本算术平均，逐 case 验证 `T_NPU<=1.25*T_GPU`，其中 GPU 耗时取该接口附件 `gpu_baseline.csv`，注意毫秒和微秒换算。

计入一次公开调用触发的全部计算阶段，不只选取内部矩阵乘 kernel。自测报告另保留满足交付章节“有效采样>10次”的完整采样统计，注明样本数与计时范围。异常调用不计为有效样本并说明原因。需要分析流水、访存或核间负载时使用官方 `msopprof`。

### 4.3 复现与交付

按 ops-blas 标准入口构建和执行 `gemm_ex` 测试，在测试 README 中给出 CSV 位置、环境、命令、golden 依赖及结果文件。交付三接口实现、公共头文件和兼容性适配、完整测试与自测报告；报告分别说明功能、精度、逐例性能及实际覆盖产品。设计评审与代码验收分别完成。
