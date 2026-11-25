# 需求背景

## 需求来源

[11月社区任务-aclblasCherk3mEx算子开发（950）](https://www.hiascend.com/activities/task-center/details/5f0045789d554794aca26d146e93d6b7)要求在 `cann/ops-blas` 的句柄式 BLAS 框架中新增复数 Hermitian 秩-k 更新接口。设计依据为任务页面发布的[任务书和测试附件](https://www.hiascend.com/p/resource/202611/81ebf8cdaf4f42ea838d68821eb60e9b.zip)。任务发布资料未标识对应 Git commit。

## 背景介绍

HERK 计算复数 Gram 矩阵及其累积结果，可用于协方差矩阵与子空间求解。仓内 `aclblasCherk` 提供三角更新与 stream 调用的工程参考。扩展接口支持交错复整数输入，通过 Hermitian 专用 3M 分解减少实数矩阵乘法次数，接口对标 [cuBLAS Cherk3mEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasherk3mex)，基础语义参考 [Netlib CHERK](https://www.netlib.org/blas/cherk.f)。

# 需求分析

## 需求描述

在 `include/cann_ops_blas.h` 声明接口，在 `blas/herk/arch35/` 实现 Ascend C 入口和 device kernel，在 `test/herk/cherk3m_ex/arch35/` 增加 CSV 驱动自验。以 `bash build.sh --ops=cherk3m_ex --soc=ascend950` 选择构建。`blas/herk/README.md` 说明公开接口及使用条件。

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const void* A, aclDataType Atype,
    int lda, const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

## 需求拆解

| 项目 | 接口约定 |
| --- | --- |
| A | 列主序。`N` 的存储形状为 `n×k`，`lda ≥ max(1,n)`。`C` 的存储形状为 `k×n`，`lda ≥ max(1,k)`。`ACL_COMPLEX64` 使用两个 float，`ACL_INT8` 使用两个有符号 int8，每个元素实部在前。 |
| C | `ACL_COMPLEX64`，列主序 `n×n`，`ldc ≥ max(1,n)`。原址更新 `uplo` 选中的含对角三角，实际更新时对角虚部写正零。非活动三角和 `ldc` 填充保持原始字节。 |
| 标量与执行 | `alpha` 和 `beta` 为实数 FLOAT32，支持 Host 和 Device 指针。Device 标量按 handle stream 顺序读取。计算在该 stream 上提交，Device 数据须持续有效至 stream 完成。 |
| 空域与缩放 | `n=0` 返回成功，允许 A/C 为空。`k=0` 或 `alpha=0` 允许 A 为空，`beta=1` 时 C 字节保持不变，其余情况仅缩放选中三角。`beta=0` 时跳过旧 C 的数值读取。`n>0` 需要 C 指针。 |
| 参数错误 | 空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法 `uplo` 或 `trans`（含 `OP_T`）返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法尺寸、跨度或必需指针返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法 dtype 组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。 |

# 详细设计

## 算子分析

令 `X=op(A)`。对于输出下标 `(i,j)`，有 `X(i,t)=A[t*lda+i]`（`N`），或 `X(i,t)=conj(A[i*lda+t])`（`C`）。按实部 `R` 和虚部 `I` 分面，计算以下三个实数乘积。

```text
P0 = R × Rᵀ
P1 = I × Iᵀ
P2 = (R + I) × (R − I)ᵀ
C_real(i,j) = alpha × (P0 + P1)(i,j) + beta × C_real_old(i,j)
C_imag(i,j) = alpha × (P2 − P0 + P1)(i,j) + beta × C_imag_old(i,j)
```

展开 `P2` 后，虚部为 `I Rᵀ − R Iᵀ`。对角只采用实部。实数乘法工作量随 `n²k` 增长，按三角枚举块可去掉非活动半平面的计算。3M 的虚部包含相消，按输入类型、秩跨度和部分积幅值确定累积方式。`int8` 数据在取共轭、和与差之前先扩大表示，以保留 `-128` 的相反数。大秩下将较短 K 段的三项先复原再归约，可降低大幅值实数分量相减造成的误差。

## 算子实现

### Host/Tiling 设计

1. 依次验证 handle、枚举、尺寸和前导维、标量指针与 dtype、由计算路径决定的 A/C 指针。跨度及工作区用 64 位无符号量检查乘加溢出。`n=0` 在结构校验后返回。
2. Host 标量按值传递。Device 标量优先由 kernel 读取，调度不依赖其值。若 A 为空且 `n,k` 非零，为了同步返回准确状态，需在 handle stream 上读 alpha 后判定。退化路径用向量 kernel 处理，Host 已知 `beta=1` 时直接返回。
3. Host 按实际 AIV/AIC 核数和 `n,k,Atype` 选择路线。小矩阵、低秩及要求精确边界顺序的计算以 AIV 处理，较大的常规矩阵以 AIV 准备面板、AIC 计算、AIV 写回。目标为单次 AIC launch 完成全部三次实数矩阵乘，准备与写回由相同 stream 排序，必要时将 AIC 和写回合并。调度仅依据接口参数、硬件资源与可读取的输入特征。
4. Cube 初始使用 `64×64` 的 C 块。按块的列坐标和三角方向筛选活动块，分配给可用核心。一个块在完整 K 秩上由一个写回方负责，避免多个核心竞争更新 C。小尺寸的 AIV 路径按列内有效区间切分，避免把活跃输出都放在单核。Tiling 传递尺寸、stride、三角方向、标量位置和面板跨度。

### Kernel 设计

1. 准备阶段以有效面板边界读取 A，按原始列主序跨距去交错，产生 `R,I,R+I,R−I`。`OP_C` 的负号作用于逻辑 `I`。未满的 Cube K 段使用零填充。`alpha=0` 或 `k=0` 时所有阶段跳过 A 的访问。
2. Cube 路径参考 `blas/syrk/arch35/csyrk_gemm_fused.h` 的三角块枚举和流水，以及 `blas/gemm/arch35/cgemm3m_host.cpp` 的精度方案。AIC 从 GM 把左右面板装入 L1，经 L0A/L0B 向三个 L0C 累加器分别提交矩阵乘。每个 `64×64` FP32 累加器占 16 KiB，三个分量共 48 KiB。需要双缓冲时两组共 96 KiB，L1/L0A/L0B 上限及尾块形状按工程的 arch35 常量与接口校验。Cube 模式显式设置浮点模式，精度敏感形状使用完整 FP32 或经验证的补偿路径。
3. AIV 小规模路径在单次 launch 内按活跃元素计算三项。调整分块与分段方式时，用实部和虚部的独立精度结果检验相消误差。对非有限值或中间溢出引起的分类差异，按参考运算顺序补充校核，避免将 3M 中间量的伪 NaN 当作输入约定。
4. 写回按列裁切边缘块和对角块，只搬运选中三角的有效行。`beta=0` 直接生成结果。对角丢弃旧虚部并写正零。用 MTE 和计算事件保护片上缓冲区复用，核间使用同 stream 的阶段顺序或显式生产者与消费者同步。输入与输出地址计算使用逻辑下标，并保留 `lda` 和 `ldc` 各自跨度。

## 支持硬件

| 芯片 | CANN | 验证范围 |
| --- | --- | --- |
| Ascend 950PR（arch35） | 9.1.0 或以上 | 任务要求的自验设备 |

## 算子约束限制

按前导维访问列主序数据，输入 A 与 C 的活跃存储不重叠。只支持规定的两个输入类型与一个输出类型。采用一个公开接口并复用仓库已有 stream、工作区、类型和错误码设施。工作区取相同 stream 下可复用的存储，分配范围在 Host 检查。性能路径选用 AIC 和 AIV 资源时以实际核数为准。任务书 §3.4 未设独立内存性能指标，仍验证越界、非活动三角和填充区写保护。

# 可维可测分析

## 精度标准/性能标准

| 验证 | 计划与判据 |
| --- | --- |
| 功能与精度 | 执行附件的 1000 条精度用例及额外契约用例。使用 Netlib BLAS 3.10.0 `cblas_cherk`，必要时使用 `cblas_zherk` 并收窄结果，int8 精确提升后调用。实部和虚部分别统计，`atol=rtol=2^-13`，匹配率至少 99%，最大误差检查 `1e-2` 及任务书允许的 `32 ULP` 口径，保留两类判定字段。精度测试工程的 MERE/MARE 指标同时记录。 |
| 契约 | 检查 `OP_T`、非法枚举、空 handle、前导维下界、dtype、Host/Device 标量、stream 写在前的标量可见性、特化缩放、对角虚部、非活动区和填充哨兵。手算复数小矩阵覆盖两种转置下虚部的相反符号。 |
| 正式性能 | 逐 case 采集 `msprof op` 的 `Task Duration(us)`，预热 5 次后保留 10 次成功样本。保留本算子各阶段的计时，并给出计算核与整条 kernel 链各自的耗时。GPU 采用附件的实测 kernel-only CSV，依据用例顺序及形状核对重复 key。阈值为每条倍率至少 `0.4`，全部倍率算术平均至少 `0.5`。补采 int8 路径，与匹配的 complex64 GPU 基线比较。ACL event 计时作为补充数据。 |
| 复现 | 从完整代码 commit 和验收附件重建、执行 CSV 自验、重做逐条 profile 并生成报告。随材料记录实际参数、版本、原始输出与判定，报告截图由执行过程产生。 |

## 兼容性分析

新增 `aclblasCherk3mEx` 符号与 arch35 路径。与同批 `aclblasCherkEx` 各自独立。公用 handle、stream、枚举和 `aclblasComplex` 定义沿用 ops-blas。`OP_T` 按任务书返回 `INVALID_ENUM`。复数 3M 与非 3M 的结合顺序不同，精度以任务书的分量阈值评估。
