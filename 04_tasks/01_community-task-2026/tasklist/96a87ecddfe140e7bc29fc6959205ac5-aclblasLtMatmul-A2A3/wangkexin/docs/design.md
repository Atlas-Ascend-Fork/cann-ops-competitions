# aclblasLtMatmul（Atlas A2/A3）算子设计

# 需求背景（required）

## 需求来源

[10 月社区任务：aclblasLtMatmul 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/96a87ecddfe140e7bc29fc6959205ac5)。任务书及随附 CSV、测试指导和基线以任务页提供的附件为依据。

## 背景介绍

在 ops-blas 中补齐 Atlas A2/A3 的 FLOAT32 轻量级矩阵乘法能力，复用已有 Lt 句柄、计算描述符、矩阵布局描述符、算法及工作空间查询接口。实现使用 Ascend C kernel 直调，由显式 stream 异步执行，不新增产品专用公共接口。

# 需求分析（required）

## 需求描述

计算单 batch 矩阵乘加：

\[
D_{ij}=\alpha\sum_{p=0}^{k-1}\operatorname{op}(A)_{ip}\operatorname{op}(B)_{pj}+\beta C_{ij}.
\]

op(A)、op(B)、C、D 的逻辑形状分别为 m×k、k×n、m×n、m×n，m、n、k 非负。A、B、C、D 均为 FLOAT32，computeType 为 `ACLBLAS_COMPUTE_32F`，scaleType 为 `ACL_FLOAT`，epilogue 为 `ACLBLASLT_EPILOGUE_DEFAULT`。四个布局均为 `ACLBLASLT_ORDER_ROW`，batchCount=1，stridedBatchOffset=0。

## 需求拆解

1. 支持 transA、transB 为 N、T、C 的全部组合，实数路径中 C 与 T 等价。
2. 支持紧凑及带行间 padding 的布局，保留 padding 区域；不提供行列广播及前导维语义以外的任意 stride。
3. 支持 C 与 D 异地及 C==D 原地计算。C/D 的数据类型、逻辑行列数、memory order 一致，原地时布局一致，Cdesc 必须提供。
4. 实现零维、零归约及零乘数的短路行为，并保留规定的校验顺序。
5. 支持 `algo == nullptr` 的默认选择，以及已支持算法的显式传入。所需 workspace 由调用方提供，并与 heuristic 返回的容量一致。
6. 以任务附件的全部用例为验收覆盖，补充附件无法表达的指针对齐等检查。

# 详细设计（required）

## 算子分析

### 接口原型

沿用 `include/cann_ops_blasLt.h` 的声明，16 个参数与 cuBLASLt 对应：

```cpp
aclblasStatus_t aclblasLtMatmul(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, aclblasLtMatrixLayout_t Adesc,
    const void* B, aclblasLtMatrixLayout_t Bdesc,
    const void* beta, const void* C, aclblasLtMatrixLayout_t Cdesc,
    void* D, aclblasLtMatrixLayout_t Ddesc,
    const aclblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, aclrtStream stream);
```

lightHandle、computeDesc、Adesc/Bdesc/Cdesc/Ddesc、algo 为 Host 对象；矩阵及 workspace 位于 Device。标量按描述符的 pointerMode 解释，HOST 模式在 Host 读取，DEVICE 标量模式在 Device 读取；不将 Device 指针直接在 Host 解引用。不支持的 pointerMode 返回 `ACLBLAS_STATUS_INVALID_VALUE`。主路径自验覆盖任务规定的 HOST 模式。

### 布局与地址

| 操作数 | N 时物理形状及地址 | T/C 时物理形状及地址 | 最小前导维 |
| --- | --- | --- | --- |
| A | m×k，A[i×lda+p] | k×m，A[p×lda+i] | N：max(1,k)；T/C：max(1,m) |
| B | k×n，B[p×ldb+j] | n×k，B[j×ldb+p] | N：max(1,n)；T/C：max(1,k) |
| C | m×n，C[i×ldc+j] | 不适用 | max(1,n) |
| D | m×n，D[i×ldd+j] | 不适用 | max(1,n) |

ld 为零时归一为对应物理矩阵的行宽，零宽布局按最小前导维规则处理。计算仅写 D 的逻辑 m×n 区域。Host 对维度、乘积、地址范围及 workspace 大小进行溢出检查，避免截断后下发。

### 边界与错误处理

| 条件 | 行为 |
| --- | --- |
| m=0 或 n=0 | 返回 SUCCESS，不下发 kernel，不访问矩阵；仍执行前置的句柄、描述符、标量指针与 pointerMode 校验。 |
| k=0 或 alpha=0 | 不读取 A/B；beta=0 时写零且不读取 C，beta=1 时复制 C 至 D（原地则无需写回），其余情况计算 beta×C。alpha=0 路径按 EXACT 校验。 |
| beta=0 且矩阵乘有效 | 不读取 C，允许 C 为空，不让 C 中 NaN/Inf 污染结果。 |
| C==D | 在覆盖前读取所需 C 值，保留所有原始输入依赖。 |
| handle 为空 | NOT_INITIALIZED。 |
| 描述符或 alpha/beta 为空、维度/ld/batch/stride 非法 | INVALID_VALUE。 |
| 布局 order 非 ROW | NOT_SUPPORTED。 |
| 转置等枚举非法 | INVALID_ENUM。 |
| 不支持的数据类型、计算模式、epilogue 或算法配置 | 返回对应不支持状态，禁止静默替换为其他语义。 |
| workspace 非空且未 16B 对齐，或容量不足 | INVALID_VALUE；需要 workspace 时必须提供有效设备地址。 |
| kernel 下发失败 | EXECUTION_FAILED；异步执行错误由 stream 同步确认。 |

校验依次为：handle → 描述符判空 → alpha/beta 判空 → pointerMode → m/n 零维短路 → 布局合法性（含 order、batch）→ 当前计算实际需要的 A/B/C/D → workspace。零维短路不访问布局数据区及矩阵地址。beta 非零且 C 为空按安全参数校验拒绝，不下发对空地址的访问；任务书列为待研发确认的该项单独记录，不将未定义行为视为精度通过。

## 算子实现

### Host 侧设计

复用 `blasLt/api/aclblaslt_matmul.cpp` 及公共描述符，按目标架构接入 `blas/lt_matmul/arch22/`，不修改 950 路径的接口声明。Host 从描述符提取实际物理形状、转置、ld、标量模式、矩阵地址及 stream，完成校验和归一化后生成 tiling。

tiling 包括 m/n/k、四个前导维、转置标志、输出块范围、归约分块、标量及 workspace 偏移。根据目标设备核心数、片上容量和矩阵形状划分输出，保证每个输出元素由确定的块负责。默认算法与显式算法共用容量计算，算法编码必须与可执行配置匹配。内部不为每次矩阵乘申请隐式 Device 临时内存。

### Kernel 侧设计

矩阵乘使用 Cube 执行分块乘加，Vector 负责必要的布局适配、尾块以及 alpha/beta 后处理。转置与行间 padding 优先通过描述符和块内搬运表达；需要整理的数据计入 workspace 与全部 kernel 时间。输入搬运与计算采用片上分块复用及流水，针对小矩阵减少启动和同步开销，针对大矩阵减少重复搬运。

FLOAT32 运算路径以任务精度为前提，不直接将输入截成低精度后当作单精度结果。主路径保持 FP32 累加；候选计算模式必须通过完整输出及特殊值检查后才能使用。alpha=0、k=0 和 beta=0 使用独立条件分支，避免数学上为零但硬件实际仍执行 0×NaN/Inf。

原地计算由输出块独立持有相应 C 数据，或使用已计入预算的临时乘积缓冲。异地 beta=0 路径可直接写 D。所有路径遵守显式 stream，不为读取结果加入隐式全设备同步。

### 工作空间

调用方通过 `aclblasLtMatmulAlgoGetHeuristic` 获取容量后分配 workspace，并传入本接口。容量覆盖该算法所需的布局整理、临时乘积及归约区域，分区按要求对齐；无需额外空间时允许 nullptr。自测逐性能 case 记录实际峰值，并核对任务规定的相对预算。外部 workspace 生命周期至少覆盖本次 stream 中相关 kernel 的完成。

## 支持硬件与约束

| 产品及软件 | 支持范围 |
| --- | --- |
| Atlas A2 / Atlas A3 系列 | 支持；目标 CANN 版本为 9.1.0。 |
| 精度自验 | 按任务书选用 A2 或 A3 一种款型完成。 |
| 性能自验 | 优先使用任务指定的 Atlas 800T A2（910B3），跨款型数据分别注明。 |

仅单 batch、FLOAT32、行主序、默认 epilogue。不包含 Bias/ReLU/GELU、混合数据类型、任意 stride、广播或确定性保证。

# 可维可测分析

## 精度标准与测试方法

在 ops-blas 的 CSV 驱动 GTest 框架接入 arch22 测试，golden 使用 Netlib BLAS 3.10.0 的 `cblas_sgemm`。保留附件原始 CSV，并适配解析器覆盖其全部 1200 条记录（1000 条精度及 200 条性能），不只截取任务正文概称的 200 条。补充 workspace 未对齐、所需容量不足、显式算法、独立 C/D 前导维与 DEVICE 标量等路径。

有限值逐元素判断 `abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)`，matched_ratio≥0.99，且满足任务规定的 max_abs_error 上限（1e-2 或 32×ULP）。非有限值单独统计：双方 NaN、同号 Inf、以及 NaN 与任一符号 Inf 等效；有限与非有限不等效，不将非有限值混入有限值误差统计。alpha=0 的缩放路径按任务要求执行 EXACT 检查。负向用例比较返回状态码；零维用例检查无矩阵访问，带 padding 用例检查有效区与保护区。

覆盖 N/T/C 组合、零维及零归约、alpha/beta 特值、奇数与尾块、行间 padding、原地/异地、空指针、非法描述符/枚举及 Inf/NaN。正式报告分别列出精度、状态码和性能覆盖。

## 性能与内存标准

以附件 `gpu_baseline.csv` 对应的 transA/transB/m/n/k 为逐例比较键，将 ms 统一换算为 us。使用 CANN 官方 `msopprof`（兼容环境使用官方 `msprof op`）采集 task duration；一次接口包含多个 kernel 时计入完整路径，不以 GTest 总运行时间或单个最快 kernel 代替算子耗时。

依据任务书 §3.3，正式采集预热 5 次，随后取 10 个有效样本的算术平均，逐例要求 NPU Avg time≤GPU time/0.8。工作空间峰值按 §3.4 不超过输入输出总字节数的 1.2 倍。附件 README、任务书交付章节及末尾注意事项对基线设备、采样次数、性能倍率和内存比较分母另有不一致描述，提交验收前以任务方确认的统一口径出具报告，并保留原始采样和字节数据以便复算。

## 兼容性分析

保持既有 Lt 公共 ABI、描述符及显式 workspace/stream，A2/A3 按架构分派。公共校验的修改仅在本任务 FLOAT32 范围内生效，并对受影响的 950 路径补充回归检查；不以拒绝实数 OP_C 或忽略 ld、C/D 别名的方式复用现有实现。

## 参考资料

- [任务官方附件](https://www.hiascend.com/p/resource/202610/d2ee7535fd9e4e448a89fd954a07a346.zip)。
- [ops-blas 源码与贡献指南](https://gitcode.com/cann/ops-blas)。
- [cuBLASLt Matmul](https://docs.nvidia.com/cuda/cublas/index.html#cublas-LtMatmul)。
- [Netlib SGEMM](https://www.netlib.org/blas/sgemm.f)。
- [生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
