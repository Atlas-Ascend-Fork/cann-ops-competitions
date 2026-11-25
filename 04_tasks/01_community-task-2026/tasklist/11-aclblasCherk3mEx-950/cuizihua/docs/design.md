# 需求背景（required）

昇腾 950PR 上的复数 Hermitian 秩-k 更新用于计算复数样本的 Gram 矩阵。调用方只存取结果的一个三角区。本题在 ops-blas 的句柄式接口中新增 `aclblasCherk3mEx`，支持 FP32 复数和交错 int8 复数输入，用三次实矩阵乘完成一次复数矩阵乘。[官方任务书](https://www.hiascend.com/p/resource/202611/81ebf8cdaf4f42ea838d68821eb60e9b.zip)

# 需求分析（required）

设 `X = op(A)`，输出三角区原地更新为 `C ← α X Xᴴ + β C`。例如 `trans=N`、`n=2`、`k=1`、`α=1`、`β=0` 时，若 A 的一列为 `(1+i, 2−i)`，则 C 的上三角应写入两个实对角元 2、5 和非对角元 `1+3i`。改为下三角时应写入 `1−3i`。未选三角和列末填充均应保留原值。此例可区分普通转置与共轭转置，也能查出虚部符号翻转。

# 详细设计（required）

## 算子原型

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const void* A, aclDataType Atype, int lda,
    const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

| 参数 | 约束及语义 |
| --- | --- |
| handle | 有效的 ops-blas 句柄。使用其绑定的 stream。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。 |
| uplo、trans | uplo 取 `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`。trans 取 `ACLBLAS_OP_N` 或 `ACLBLAS_OP_C`。普通转置和其他非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`。 |
| n、k、lda、ldc | n、k 非负。A 为列主序。N 时物理形状 n×k 且 lda ≥ max(1,n)，C 时为 k×n 且 lda ≥ max(1,k)。输出矩阵的 ldc ≥ max(1,n)。维度或前导维不合法返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| alpha、beta | 均为以指针传入的实数 float，支持 Host 和 Device 标量。空标量指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |
| A、Atype | `ACL_COMPLEX64` 每个复数占两个 float；`ACL_INT8` 每个复数占两个有符号字节，依次为实部和虚部，两者先精确提升到 FP32。 |
| C、Ctype | `ACL_COMPLEX64`，列主序原地输入输出。A 的可读区与 C 的可写区不重叠。其他类型组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。矩阵指针缺失且本次调用要求其有效时返回 `ACLBLAS_STATUS_INVALID_VALUE`。 |

## Host 侧设计

按句柄、枚举、维度及前导维、标量与类型和矩阵指针的顺序检查，结构合法后再选执行路径。偏移与按字节计的长度用 64 位运算。标量指针位置沿用仓内查询工具，Host 标量按值传给 kernel。Device 标量须遵守句柄 stream 上的先后关系；需在 Host 判定返回码或退化分支时，只回读所需标量。正常矩阵计算异步提交，调用方在读回 C 前同步该 stream。

| 条件 | 对矩阵的访问 |
| --- | --- |
| n = 0 | A、C 可为空，返回成功，无 kernel 启动。 |
| n > 0，且 k = 0 或 α = 0 | C 必须有效，A 可为空。β = 1 时直接返回，保留对角虚部在内的全部原值。其他 β 只更新选定三角，β = 0 直接写零。 |
| n > 0，k > 0 且 α ≠ 0 | A、C 必须有效。β = 0 时跳过旧 C 的读取。更新后对角虚部显式写正零。 |

## Kernel 侧设计

Cube 路径将输出按二维块划分，仅调度与所选三角相交的块。每个块由一个 AIC 与配合的 AIV 协同完成整个 K 方向归约。AIV 对输入分面板解交错、按需共轭，并将交错 int8 精确提升为 FP32。输入面板通过 L1 双缓冲供 AIC 使用。AIV 在显式声明的统一缓冲区（UB）中分区存放打包数据、三个实乘积及分段归约结果，通过同步标志协调缓冲区复用，最后仅写回 uplo 指定的三角元素。小矩阵及不适合 Cube 分块的布局采用 SIMT 路径，按行组和列分派线程，在读取输入和写回输出前过滤未选三角与尾部越界元素，保留另一半三角和列末填充。

N 路径使用 `X[i,t]=A[i+t·lda]`，C 路径使用 `X[i,t]=conj(A[t+i·lda])`。记 `X=R+iI`。直接归约路径在**同一次 kernel 启动**内使用如下三个实乘积：

| 累加量 | 左操作数 | 右操作数 |
| --- | --- | --- |
| P₀ | R | Rᵀ |
| P₁ | I | Iᵀ |
| P₂ | R + I | (R − I)ᵀ |

实部取 `P₀ + P₁`，虚部取 `P₂ − P₀ + P₁`。Cube 采用等价的融合形式。右因子是已经共轭转置的 `B = Xᴴ = Bᵣ + iBᵢ`，其中 `Bᵣ = Rᵀ`、`Bᵢ = −Iᵀ`。三个实矩阵积为 `Q₀ = (R + I) Bᵣ`、`Q₁ = R (Bᵢ − Bᵣ)`、`Q₂ = I (Bᵣ + Bᵢ)`。实部取 `Q₀ − Q₂`，虚部取 `Q₀ + Q₁`，两式分别展开为 `RRᵀ + IIᵀ` 和 `IRᵀ − RIᵀ`。

常规路径使用 FP32 累加。有限 complex64 输入可能使三个实乘积出现中间溢出，或使虚部在重组时发生显著抵消。对检测到风险的输出元素，在同一次 kernel 启动内依据分量量级选择等价的三乘求值方式，保留扩展的中间动态范围，在应用 α 和 β 后舍入写回。更新对角元时只需求旧实部。以 profiler 轨迹核对整个非退化调用的设备工作，以独立参考检验各分量的误差。

公开声明放入 `include/cann_ops_blas.h`，实现放入 `blas/herk/arch35/`，在 `blas/herk/README.md` 登记产品支持范围。测试入口为 `test/herk/cherk3m_ex/arch35/`，按工程已有规则注册 `cherk3m_ex`。现有非 Ex 的 `aclblasCherk` 和新接口分别保留各自的参数语义。[ops-blas 基线实现](https://gitcode.com/cann/ops-blas/blob/55ef791b77f3677b45da3468b0a8b17941424f84/blas/herk/arch35/cherk_host.cpp)

# 可维可测分析

目标环境为昇腾 950PR（arch35）、CANN 9.1.0 或更高版本。用 `bash build.sh --ops=cherk3m_ex --soc=ascend950` 构建，GTest 从原始 CSV 载入用例。任务书 §3.4 未设单独内存门限。

| 验证项 | 方法和通过条件 |
| --- | --- |
| 精度 | 运行 1,000 条精度用例。按输入类型解包，使用 Netlib BLAS 3.10.0 的 `cblas_cherk` 计算独立参考，需要双精度参考时使用 `cblas_zherk`。在选定三角内分别比较实、虚分量。误差 ≤ 2⁻¹³ × (1 + abs(golden)) 的分量占比须 ≥ 99%，并同时满足最大误差规则 0.01 或 32 ULP。保留误差最大位置、实际值、参考值和使用的判据。 |
| 访问范围及边界 | 交叉覆盖 N/C、上/下三角、两种 A 类型、非紧凑 lda/ldc。用哨兵逐字节核对未选三角、填充及输入 A。补查无效句柄、OP_T、非法类型和维度，以及 n=0、k=0、α=0、β=0/1。分别放入空 A、带 NaN 的 A、带 NaN 的旧 C 和非零对角虚部，检查应保留或应覆盖的位置。 |
| 3M 数值与类型 | 用有解析解的非实矩阵验证共轭方向。覆盖极值 int8、交替符号虚部抵消和大 K。对更新结果检查对角虚部为零。Host 与 Device 两种标量传入方式分别覆盖。 |
| 性能 | 200 条正式性能用例逐条用 `msprof op` 采集，在 5 次预热后取 10 次有效的 `Task Duration(us)` 均值，以微秒除以 1000 换为 ms。倍率为 GPU 基线 ms / NPU 核耗时 ms；每条 ≥ 0.4，全部倍率的算术平均 ≥ 0.5。按用例 ID 留存原始采样与 GPU 基线行，保留重复形状的独立测量。另补交错 int8 路径，对照相同形状的 complex64 GPU 基线。 |
