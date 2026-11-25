# 需求背景（required）

## 需求来源

11 月 CANN 社区任务《aclblasCherk3mEx 算子开发（950）》要求在 ops-blas 中提供昇腾 950PR 的 Hermitian 秩-k 更新接口。接口对标 cuBLAS 的 cublasCherk3mEx，采用 Ascend C 实现。[社区任务入口](https://www.hiascend.com/developer/activities/cann-community-task#cann-task-all)

## 背景介绍

Hermitian 秩-k 更新只需存储和更新结果的一个三角。任务书给出的 arch35 参考实现中，cherk 提供三角访问与句柄调度逻辑，cgemm3m 提供复数乘法的三实乘分解。本设计复用这两类思路，在一次 kernel 调用内完成三实乘，并支持交错存储的复数 int8 输入。[ops-blas](https://gitcode.com/cann/ops-blas)

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 增加以下入口。令 X = op(A)，在 uplo 指定的半区原地更新 C = αXXᴴ + βC。上三角包含 i ≤ j，下三角包含 i ≥ j。

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const void* A, aclDataType Atype,
    int lda, const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 提供 stream、标量指针模式 | 句柄 | aclblasHandle_t | 已创建 | 标量 |
| uplo / trans | 更新区域、A 的运算方式 | 枚举 | aclblasFillMode_t / aclblasOperation_t | UPPER、LOWER / OP_N、OP_C | 标量 |
| n / k | C 的阶数、归约长度 | 整数 | int | 均 ≥ 0 | 标量 |
| alpha / beta | 实数系数 | 标量指针 | FLOAT32 | 均非空，支持 Host 和 Device 模式 | 各 1 个 |
| A / Atype | 列主序输入及其类型 | 复数矩阵 / 枚举 | ACL_COMPLEX64、ACL_INT8 | 每个复元素分别占 8、2 字节 | N 为 n×k，C 转置模式为 k×n |
| C / Ctype | 列主序原地输入输出及其类型 | 复数矩阵 / 枚举 | ACL_COMPLEX64 | n > 0 时可写且非空 | 逻辑 n×n |
| lda / ldc | 相邻列间隔 | 整数 | int | 按复元素计，至少为对应逻辑行数与 1 的较大值 | 标量 |

## 需求拆解

Host 负责校验、退化分支和分块。Kernel 负责解交错、共轭、3M 累加与三角写回。实现归入 `blas/herk/arch35/`，在 `blas/herk/README.md` 增加产品支持说明，在 `test/herk/cherk3m_ex/arch35/` 注册 CSV 驱动的 GTest。构建入口为 `bash build.sh --ops=cherk3m_ex --soc=ascend950`。

# 详细设计（required）

## 算子分析

令 X = R + iI。对输出行块 a、列块 b，在相同 K 区间取两组行面板，累计以下三个实数乘积：

| 累加器 | 左面板 | 右面板（转置后参与乘法） |
| --- | --- | --- |
| P0 | R[a] | R[b] |
| P1 | I[a] | I[b] |
| P2 | R[a] + I[a] | R[b] − I[b] |

结果实部为 P0 + P1，虚部为 P2 − P0 + P1。展开虚部可得 I[a]R[b]ᵀ − R[a]I[b]ᵀ，共轭带来的减号由右面板承担。所有乘加以 FP32 精度设计。INT8 的实、虚分量先分别有符号扩展并精确转换为 FP32，再取负、求和或求差，以覆盖 −128 的共轭和两分量相加的范围。

## 算子实现

### 实现方案

**Host 校验与调度。** 按下列顺序检查，结构校验完成后才选择计算路径。

1. 空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。非法 uplo、trans（含 OP_T）返回 `ACLBLAS_STATUS_INVALID_ENUM`。
2. 检查 n、k 非负，N 模式 lda ≥ max(1,n)，C 模式 lda ≥ max(1,k)，ldc ≥ max(1,n)。违反约束返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. alpha、beta 空指针或 n > 0 且 C 为空返回 INVALID_VALUE。类型组合超出参数表返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。仅在 n、k 均正且 alpha 非零时要求 A 非空，否则允许 A 为空。Device 标量由设备端按 stream 顺序读取；当 A 为空且其合法性取决于 alpha 时，需在入口取得该 stream 上有序的 alpha 值再返回相应状态。正常路径无需为标量引入主机同步。

| 条件（自上而下匹配） | 行为 |
| --- | --- |
| n = 0 | 返回 SUCCESS，无需访问 A、C 或启动 kernel |
| (k = 0 或 α = 0) 且 β = 1 | 保留 C 全部字节。Device 模式在设备侧判断标量后直接退出 |
| k = 0 或 α = 0 | 只缩放指定三角，对角虚部写零。β = 0 时直接清零 |
| 其余 | 执行 3M 更新。β = 0 时跳过旧 C 的读取 |

**分核与片上存储。** 将 n×n 划分为方形输出块，仅编号指定三角的块。按有效块序号轮转分核，每个块由一核完成整个 K 归约，避免跨核归并和重复写。Host 按 n、k、可用核数与片上容量选择输出块边长和 K 分段长度，容量预算包含两侧面板、和差暂存及三个 FP32 累加块；不足时先缩短 K 面板，再缩小输出块。分派依据为形状、布局和输入类型。

**面板搬运与写回。** 对逻辑 X(i,t)，N 模式读取 A[i+t·lda]，C 模式读取 A[t+i·lda] 并共轭。偏移使用宽整数，按输入元素字节数寻址。分段加载时将交错实虚部拆开，尾块只搬运有效范围，计算面板余位补零。三次实乘共享这些面板，在同一次 launch 内按 K 段累加，最后一次性组合 P0、P1、P2 并施加 α、β。对角块逐列裁剪有效行区间后访问 C[i+j·ldc]；对角只使用旧实部，并将输出虚部显式写零。另一三角和列尾 padding 均不参与搬运。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| 昇腾 950PR（arch35），CANN 9.1.0 及以上 | √ |

## 算子约束限制

调用方保证 A 的可读区与 C 的可写区不重叠。矩阵布局由列主序与前导维描述，无广播和任意视图步长。Host 按运行时尺寸选择分块，无需扫描矩阵内容。计算在 handle 绑定的 stream 上执行，调用方读回结果前同步该 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 用 Netlib BLAS 3.10.0 的 cblas_cherk 生成 golden，INT8 输入先精确提升。指定三角内实、虚分量分别比对，绝对误差 ≤ 2⁻¹³ × (1 + abs(golden)) 计为匹配。每例要求 matched_ratio ≥ 0.99，且 max_abs_error ≤ 1e−2（或 32 × ULP）。 | 任务书 §3.1、§3.2 |
| 性能 | 每条性能用例用 msprof op 在 950PR 上预热 5 次，取随后 10 次有效 kernel Task Duration(us) 的均值除以 1000 得 npu_ms。每例 gpu_ms / npu_ms ≥ 0.4，且全部用例倍率算术平均 ≥ 0.5。逐例保留基线 ID 与采样结果；GTest avg_us 用于快速回归。 | 任务书 §3.3、gpu_baseline.csv |

附件 CSV 含 1000 条精度及 200 条性能用例。精度验证覆盖两种转置、上下三角、带填充前导维、非对齐尾块、K 长短差异、3M 虚部抵消及 int8 极值。为未引用三角与 padding 设置哨兵，验证逐字节保持；以非零对角虚部验证正常写回与 quick return 的区别。以 A 含 NaN 且 α=0、旧 C 含 NaN 且 β=0 检查无效操作数的隔离。对非法输入核对具体状态码，空 handle 在 CSV 装配层单独构造。开发回归另记录附件的 MERE、MARE 结果。

当前 CSV 的 200 条 TC_PF 均为 ACL_COMPLEX64。按任务书 §3.3 另采 INT8 性能，选择与复数浮点基线相同的形状、转置和三角，显式标注借用的 COMPLEX64 基线及其 ID。补充结果单列，以便复核正式全集的平均倍率。

## 兼容性分析

新增接口使用 ops-blas 公共句柄、状态码和 aclblasComplex 定义，可与非 Ex 的 cherk 独立调用。OP_T 按本任务的 Hermitian 定义域返回 INVALID_ENUM。数值验收以分量误差为准，允许 3M 中间舍入产生与直接复数乘法不同的末位结果。本接口独立构建，无需依赖同期开发的 aclblasCherkEx。
