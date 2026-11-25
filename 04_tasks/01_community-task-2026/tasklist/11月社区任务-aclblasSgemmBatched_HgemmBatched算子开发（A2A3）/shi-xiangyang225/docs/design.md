# aclblasSgemmBatched 与 aclblasHgemmBatched 算子设计（Atlas A2/A3）

## 需求背景

两个接口分别对标 cuBLAS 的 `cublasSgemmBatched` 和 `cublasHgemmBatched`。训练和推理中，一批矩阵可以具有相同的形状和前导维，却分配在互不相邻的地址。调用者一次提交三张 Device 指针表，算子在句柄绑定的 stream 上完成整批运算。[社区任务](https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c)

ops-blas 已公开两个接口，arch35 有 S/H 实现，arch22 有复数批量 GEMM 和实数跨步批量 GEMM。本任务在 `blas/gemm_batched/arch22/` 实现实数指针数组版本，沿用公开 ABI，并在算子 README 中补充 A2/A3 的产品支持和约束。[公共头文件](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/include/cann_ops_blas.h) · [GEMM Batched 源码](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm_batched)

## 需求分析

对每个 `i ∈ [0, batchCount)`，计算 `C[i] ← alpha × op(A[i]) × op(B[i]) + beta × C[i]`。变换后的 A、B 和输出 C 的逻辑形状分别为 `m×k`、`k×n` 和 `m×n`。各 batch 共用尺寸、标量和两个转置参数。实数域中 `OP_C` 与 `OP_T` 采用相同索引。

| 接口 | A、B、C 及 Host 标量 | 乘积与输出 |
| --- | --- | --- |
| `aclblasSgemmBatched` | FLOAT32 | FP32 输入与累加，写回 FP32 |
| `aclblasHgemmBatched` | IEEE binary16，以 `aclblasHalf` 表示 | 以 FP32 累加及缩放，最终一次舍入为 FP16 |

矩阵按列主序存储。下表用 `r`、`j`、`q` 分别表示输出行、输出列和归约下标，偏移量以元素计。

| 访问 | N | T 或 C |
| --- | --- | --- |
| `op(A)[r,q]` | `r + q*lda`，`lda ≥ max(1,m)` | `q + r*lda`，`lda ≥ max(1,k)` |
| `op(B)[q,j]` | `q + j*ldb`，`ldb ≥ max(1,k)` | `j + q*ldb`，`ldb ≥ max(1,n)` |
| `C[r,j]` | `r + j*ldc`，`ldc ≥ max(1,m)` | 同左 |

### 接口与参数契约

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* const Aarray[], int lda,
    const float* const Barray[], int ldb, const float* beta,
    float* const Carray[], int ldc, int batchCount);

aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha, const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb, const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc, int batchCount);
```

## 详细设计

### Host 校验与下发

入口依次检查句柄、转置枚举、`m/n/k`、前导维、`batchCount`、两个 Host 标量指针和本次运算所需的指针表。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法转置返回 `ACLBLAS_STATUS_INVALID_ENUM`，负维度、无效前导维或缺少必需指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。S/H 共用校验和调度，仅在读取标量、装载元素及输出转换处区分类型。

| 结构参数合法后的情形 | 需要有效的 Device 指针表 | 设备动作 |
| --- | --- | --- |
| `m==0`、`n==0` 或 `batchCount==0` | 无 | Host 返回 SUCCESS，kernel 下发数为 0 |
| `k==0` 或 `alpha==0`，且 `beta==1` | C | 保留 C，直接返回 |
| `k==0` 或 `alpha==0`，且 `beta==0` | C | 将每列前 m 个元素写零，只写 C |
| `k==0` 或 `alpha==0`，其他 beta | C | 对 C 的逻辑区域执行缩放 |
| 常规乘积 | A、B、C | `beta==0` 时只写 C，其他 beta 读取旧 C 参与更新 |

调用者保证表中实际使用的地址有效，C 的可写区域与任意 A/B 的读区及其他 C 的写区互不重叠。Host 仅检查表指针本体。各核在 Device 侧按 batch 编号取出真实矩阵地址。偏移和工作区字节数在扩大到 64 位后计算，Host 在乘法可能超出可寻址范围时返回错误。

### Kernel 分工与数据流

1. **常规矩阵乘。** 以一个 batch 内的输出子矩阵为调度单位，将 `(batch, 输出列块, 输出行块)` 展为线性任务。每个输出块由一个 Cube 核完成全部 K 段的归约。Cube 按行主序的转置等式 `Cᵀ = op(B)ᵀ × op(A)ᵀ` 装载 B、A。实际基址取自对应指针表，地址计算始终使用调用者的 lda/ldb。M、N、K 尾块显式传入有效长度。
2. **计算精度与写回。** S 路径使用完整 FP32 输入，关闭 HF32。H 路径装载 FP16 并以 FP32 累加。当 `alpha==1 && beta==0` 且目的地址、前导维和尾块满足直接写回条件时，Cube 将结果写入 C。其余路径先保留 FP32 乘积，再由向量核按逻辑行数执行标量组合。分支先判断 beta，再决定是否装载旧 C，因而旧 C 中的 NaN 不会污染 `beta==0` 的结果。
3. **小矩阵与大 batch。** 当一个小矩阵无法充分占用 Cube 时，每个核循环处理若干 batch 或若干完整输出列，摊薄取指针和下发开销。以运行时形状、数据类型、片上容量和可用核数决定分块。大 batch 沿指针表范围切分，保持调用者给出的映射。
4. **填充与临时空间。** C 的每列只有前 m 行可以写。遇到非对齐首址、奇数行长或带 padding 的目标，选用支持有效长度的搬运或经对齐的临时块再逐列拷回。一般标量分支按可用工作区分块保存 FP32 乘积，当前块完成缩放后复用缓冲。Host 校验工作区容量及分配状态，全部计算和缓冲复用均按同一 stream 的顺序下发。

### 硬件、约束与兼容

目标为 Atlas A2 和 Atlas A3 的 arch22 产品，工具链为 CANN 9.1.0。两接口使用已有 handle 及其绑定的 stream，异步返回；主机读取结果前由调用者同步。矩阵布局支持紧凑存储和指定前导维的列填充，张量广播、任意视图和额外的 Python 接口不在任务范围内。arch35 继续使用其既有实现。任务书 §3.4 将内存验收列为 N/A，临时空间仍须受上文的容量与生命周期约束。

## 可维可测分析

### 精度与边界

在 `test/gemm_batched/sgemm_batched/arch22/` 注册共用的 CSV 驱动 GTest。分别读取两个接口的原始 CSV，以“接口、case_name”区分两表同名行。按原 batch、随机种子、填充与阈值复现。S 的参考结果采用 Netlib BLAS 3.10.0 的 `cblas_sgemm` 逐 batch 计算；采用任务书允许的 FP64 累加后收窄方案时，在报告注明参考域。H 将量化后的输入与标量转为 FP64 累加，再依次收窄为 FP32、FP16。参考值取调用前的 C 快照。

| 口径 | S | H |
| --- | --- | --- |
| 精度用例 | 500 条 | 500 条 |
| 逐元素 `atol = rtol` | `2^-13` | `2^-10` |
| 用例通过条件 | 匹配比例 ≥ 0.99，且最大绝对误差满足 0.01 或 32 ULP 界限 | 匹配比例 ≥ 0.99，且最大绝对误差满足 0.1 或 32 ULP 界限 |

逐元素计算 `abs(actual − golden) ≤ atol + rtol × abs(golden)`。只统计逻辑输出，单独逐列核对 padding 哨兵。正向用例记录匹配数、分母、最大误差及位置，特殊值分别核对 NaN 传播和无穷符号。负向用例核对返回码。抽样覆盖九种转置组合、零维度、零 batch、空指针表、退化标量、独立且非等距的矩阵地址、FP16 次正规/接近上限及带填充布局。全量包括 4095³/4096³、1×1×1 大 batch，并保留一般输入中均匀与正态两类分布。[生态精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)

### 性能与证据

每个接口 100 条性能用例。按转置、m/n/k、batchCount、alpha/beta 和三个前导维与随包 A100 基线逐行匹配。任务书 §3.3 的正式环境是 Atlas 800T A2（910B3）。其他款型的结果分别标注实际型号。

| 测量项 | 采样 | 指标 |
| --- | --- | --- |
| 正式 kernel 计时 | `msprof op`，预热 5 次，随后连续 10 次成功调用 | 对每次调用涉及的算子 kernel task duration 归集后取算术平均；逐例要求 `mean_npu_us ≤ 1.25 × gpu_ms × 1000` |
| 端到端辅助计时 | 缓冲先就绪，同一 stream 用 `aclrtEvent` 包住连续 10 次下发，之前预热 5 次 | 打印 `[PERF] case=<case_name> ... avg_us=<数值>`，记录事件均值 |

两套计时均排除输入生成、H2D/D2H 和 CPU golden；保留每个 case 的原始采样、算子返回码和 profiler 记录。报告列出未舍入倍率及可比用例的平均倍率。精度和性能抽样通过后，按接口分别进行全量精度、全量性能。交付记录关联源码提交、库哈希、设备、CANN 版本以及用例和基线版本。
