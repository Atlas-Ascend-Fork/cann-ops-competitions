# GEMV Strided Batched 算子设计

## 需求背景

### 需求来源

11 月 CANN 社区任务「gemvStridedBatched 算子开发（Atlas A2/A3）」。基于 `cann/ops-blas` 的公开句柄式 BLAS API，为 S、HSH、HSS、TST、TSS 五个变体实现并验证 arch22 路径。选用的上游基线 `55ef791b77f3677b45da3468b0a8b17941424f84` 已包含五个 arch22 入口和初版 kernel，本次补齐任务书 2026-11-24 修订要求。

### 背景介绍

接口语义与对应 cuBLAS GEMV Strided Batched 对齐，运算和负步长寻址依据 Netlib SGEMV。调用方在 handle 上绑定 stream，算子以 Ascend C kernel 异步执行。 [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html) · [Netlib SGEMV](https://www.netlib.org/blas/sgemv.f)

## 需求分析

### 需求描述

计算 `y_b ← alpha · op(A_b) · x_b + beta · y_b`，其中 `b ∈ [0, batchCount)`，每个操作数按各自元素单位的 batch stride 定位。A 的物理存储恒为列主序 `lda × n`。N 路径的 x、y 逻辑长度分别为 n、m，T/C 路径分别为 m、n。

### 需求拆解

1. 五个接口共享 Host 校验、tiling 与 kernel。保留公开接口、参数顺序和 ABI。
2. 覆盖 N/T/C、带 padding 的 lda、正负元素步长、batch 间隙、空尺寸、alpha/beta 省读与 NaN/Inf 传播。
3. 在 910B3 上使用 CANN 9.1.0 验证精度与 kernel 耗时，再按社区流程提交验收。

## 详细设计

### 算子分析

| 变体 | A / x | y | alpha / beta |
| --- | --- | --- | --- |
| S | FP32 | FP32 | Host FP32 |
| HSH | FP16 | FP16 | Host FP32 |
| HSS | FP16 | FP32 | Host FP32 |
| TST | BF16 | BF16 | Host FP32 |
| TSS | BF16 | FP32 | Host FP32 |

五个签名以 `include/cann_ops_blas.h` 为准。16 位张量的公开类型为 `uint16_t`，其位模式由对应变体解释。向量在每个 batch 内从地址最低端传入，负步长的逻辑起点为 `(len-1)·|inc|`。y 原地覆写，向量间隙和 batch 间隙保持原值。

### 公开接口

S 变体的参数顺序如下。其余四个入口为 `aclblasHSHgemvStridedBatched`、`aclblasHSSgemvStridedBatched`、`aclblasTSTgemvStridedBatched` 和 `aclblasTSSgemvStridedBatched`，各张量的指针类型依据上表。

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex, const float* beta,
    float* y, int incy, int64_t stridey, int batchCount);
```

| 参数 | 约束 |
| --- | --- |
| handle | 已创建的 BLAS 句柄，kernel 在该句柄绑定的 stream 上下发 |
| trans | N/T/C，实数的 C 与 T 等价 |
| m、n、lda | m 和 n 非负，lda ≥ max(1, m)，与转置无关 |
| alpha、beta | 非空 Host FP32 指针，有效非空操作中标量值有限 |
| A、x、y | Device 缓冲区。A、x 为输入，y 为原地输入输出 |
| incx、incy | 非零，有符号，以元素为单位 |
| strideA、stridex、stridey | 正的 64 位元素偏移，非空运算中要覆盖各 batch 的物理占用 |
| batchCount | 非负，零值为合法空操作 |

空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，无效转置值返回 `ACLBLAS_STATUS_INVALID_ENUM`，其他无效参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。

### 算子实现

#### Host

先校验 handle、trans、维数、lda、非零 inc、正 stride、batchCount 和标量指针。空操作返回成功，允许 A/x/y 为空。有效非空操作要求 A/x/y 有效，alpha/beta 有限。矩阵占用 `lda·n` 元素，向量占用 `1+(len-1)·|inc|` 元素。64 位检查确保 batch 跨步覆盖占用、最后一个 batch 的字节范围可表示。合法的一元素向量接受任意非零 int 步长。`alpha=0 && beta=1` 直接返回。kernel 下发失败映射为 `ACLBLAS_STATUS_EXECUTION_FAILED`。

以 `(batch, 输出 tile)` 为任务单位。batch 数不足时增加输出分块，N 路径按 32 元素粒度，T/C 路径按 16 元素粒度。每块最多 2,048 个输出，各 AIV 核按核数步长分领任务。单个 y 逻辑元素由一个核负责。

#### Kernel

- N 路径将同一输出范围对应的一组矩阵列搬到 UB，每组最多 16 列。向量 Axpy 计算每 32 列的 FP32 部分和，再加到输出累加器。
- T/C 路径一次搬入一个 x 行 tile，并在该 tile 对应的多个 A 列间复用。计算 FP32 乘积与向量规约，再累加不同行 tile 的部分和。
- 矩阵搬运以 32 B 对齐 UB 列首，DMA 每段仅从 GM 读取有效行。连续 x/y 用精确字节数搬运，跨步 x/y 按逻辑索引寻址。y 只写逻辑元素。
- 对绝对值至少为 1,024 的缩放因子，且 alpha 非零时，使用 FP32 双分量累加保留乘积和求和的残差，降低小差值被大系数放大后超出绝对误差上限的概率。普通路径先完成点积，再应用 alpha 和 beta。beta 为零省掉 y 的旧值读取，alpha 为零省掉 A/x 读取。输出按目标类型转换。
- 每核使用最多 96.5 KiB UB，全局 workspace 为 0 字节。

### 支持硬件

Atlas A2 和 Atlas A3（arch22）。精度验证可使用任一款型，正式性能使用 Atlas 800T A2（910B3）。目标工具链为 CANN 9.1.0。

### 算子约束限制

相邻 batch 区域独立，y 的有效写区域与 A/x 的读区域由调用方保证互不重叠。A/x/y 地址由调用方在正确设备分配，调用方同步 stream 后读取结果。输出容差基于目标 dtype。

## 可维可测分析

| 项目 | 判据 |
| --- | --- |
| 精度 | 5,000 条原始精度用例，加接口边界用例。按 Netlib 公式逐 batch 用 FP64 累加参考值，将结果按目标 dtype 收窄。FP32、FP16、BF16 的 rtol/atol 分别为 2⁻¹³、2⁻¹⁰、2⁻⁷，匹配比例 ≥ 99%，有限值逐元素误差上限为固定阈值与 32 ULP 的较大值。非有限值要求同为 NaN 或同符号 Inf。 |
| 性能 | 1,000 条用例，预热 5 次，`msprof op` 取后 10 次 kernel `Task Duration(us)` 平均值。经确认的同参数 GPU kernel 耗时乘 1.25 为上限。ACL event 耗时单独记录，单位 µs。 |
| 内存 | 任务书 §3.4 无独立内存测试项。 |

本地 CPU 模拟已通过 5,000 条精度与异常用例、1,000 条性能形状的功能比对和 3 条补充接口用例。Atlas A2/A3 的设备编译与精度、910B3 的正式性能采集待实测。任务书修订涉及 T/C 基线复测，正式性能判定前会核对基线的采集参数与当前 CSV。

构建使用 `bash build.sh --soc=ascend910b3 --ops=gemv_strided_batched --run`。GTest 框架和验收复现说明见下方链接。 [上游 GTest 目录](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/test/gemv_strided_batched) · [验收仓复现说明](https://gitcode.com/gcw_HuYbgqHW/ops-blas-gemv-strided-batched-a2-a3/tree/main/task_submission)

### 兼容性分析

五个公开 API 沿用上游原型。arch22 的空操作校验遵循任务书约定，旧测试中空尺寸使用的零跨步更正为正值。arch35 由现有构建分支提供。
