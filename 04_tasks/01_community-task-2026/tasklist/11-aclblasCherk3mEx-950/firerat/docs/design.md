# 需求背景（required）

## 需求来源

基于 [ops-blas](https://gitcode.com/cann/ops-blas) 开发，对标 cuBLAS `cublasCherk3mEx`。

## 背景介绍

在昇腾 950PR（arch35）上使用 Ascend C 实现复数 Hermitian 矩阵秩-k 更新扩展接口。接口采用 ops-blas 句柄式 BLAS 工程模式，通过 handle 绑定 stream 异步直调 kernel。A 支持 complex64 和交错 complex-int8，C 为 complex64；使用 Hermitian 3M 分解在一次 kernel launch 内完成三次实数矩阵乘。

# 需求分析（required）

## 需求描述

计算 `C = alpha · op(A) · op(A)^H + beta · C`，alpha、beta 为 float32 实数。矩阵列主序，仅读写 uplo 指定的 C 三角，对角虚部计算后置零；另一三角不写入。`trans=N` 时 A 为 n×k，`trans=C` 时 A 为 k×n 并取共轭转置。`trans=T` 返回非法枚举，这是相对 cuBLAS 的有意差异。

## 需求拆解

1. 新增公开接口和 arch35 host、kernel 实现，支持两种 A 类型及上、下三角和 N、C 两种运算。
2. 按任务书校验参数、处理空矩阵及零系数，遵守原地更新与异步 stream 语义。
3. 实现单次 launch 的 Hermitian 3M 计算，满足任务书精度与性能标准。
4. 注册构建和测试目标，编写算子说明、自验代码及报告。

# 详细设计（required）

## 算子分析

### 数学公式

令 `X = op(A) = Xr + iXi`，则 `C = alpha · X · X^H + beta · C`。3M 分解为：

- `P0 = Xr · Xr^T`，`P1 = Xi · Xi^T`；
- `P2 = (Xr + Xi) · (Xr - Xi)^T`；
- 乘积实部为 `P0 + P1`，虚部为 `P2 - P0 + P1`。

第二操作数的共轭不可省略。3M 的舍入与直接复数乘法不同，不要求逐位一致。

### 支持数据类型和形状

| 参数 | 类型及布局 | 约束 |
| --- | --- | --- |
| alpha、beta | float32 实数标量指针 | 支持 Host 和 Device 指针 |
| A | `ACL_COMPLEX64` 或 `ACL_INT8`，列主序 | N：n×k；C：k×n。INT8 每个复元素以实部、虚部交错两字节存储，分别精确提升为 float |
| C | `ACL_COMPLEX64`，列主序，原地更新 | 逻辑形状 n×n，物理跨度由 ldc 指定 |

仅支持上述两种 A/C 类型组合；n、k 非负，N 时 `lda ≥ max(1,n)`，C 时 `lda ≥ max(1,k)`，`ldc ≥ max(1,n)`。不支持广播或超出前导维度语义的任意非连续布局。

## 算子实现

### 接口设计

在 `include/cann_ops_blas.h` 新增：

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const void* A, aclDataType Atype, int lda,
    const float* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

`aclblasComplex` 以仓内公共头文件定义为准。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；uplo/trans 非法（包括 OP_T）返回 `ACLBLAS_STATUS_INVALID_ENUM`；维度、前导维度或必需指针非法返回 `ACLBLAS_STATUS_INVALID_VALUE`；类型组合不支持返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。枚举错误码与 cuBLAS 不同，是遵循 ops-blas 的有意差异。

### Host 侧设计

1. 依次校验 handle、uplo/trans、n/k 和 lda/ldc，随后校验标量、类型和矩阵指针等结构参数，再执行 quick return。n=0 允许 A、C 为空；n>0 时 C 必须非空；k=0 或 alpha=0 时 A 可为空。标量支持 Host 和 Device 指针，Device 标量的值相关分支需在设备侧处理，不能在 Host 直接解引用。
2. n=0 不启动 kernel；k=0 或 alpha=0 且 beta=1 时 C 不变；否则仅对指定三角执行 beta 缩放。beta=0 时不读取旧 C，避免 NaN 污染；对角输入虚部忽略，实际执行更新时输出虚部置零。
3. 根据 n、k、uplo、trans、数据类型及硬件片上容量选择分块和分核配置。将输出划分为矩形 tile，跳过指定三角之外的 tile；对角相交 tile 在写回时逐元素掩码。K 维分块归约，尾块按有效范围处理，避免越界。具体 tile 大小和缓冲配置需通过 950PR 实测调优。
4. 将尺寸、前导维度、类型和分块信息传给 kernel，在 handle 所绑定 stream 上下发；不依赖 `aclblasCherkEx`，不以多次 GEMM 调用拼接。

### Kernel 侧设计

每个工作单元负责不重叠的 C tile，按列主序及 lda 读取 A，按 trans 构造 X 的实虚分量；INT8 分量先提升为 float。分块执行上述 P0、P1、P2 并在同一次 launch 内完成 K 归约和复数组合。根据片上容量安排搬入、计算、搬出及必要的缓冲复用；仅在 beta 非零时读取 C 指定三角。合并 alpha、beta 后写回有效三角，输出对角虚部强制为零，保留另一三角及 padding。零贡献路径不访问 A，quick return 保持 C 原样。

### 工程组织

实现位于 `blas/herk/arch35/`，参考非 Ex `cherk` 的校验和三角处理以及 `cgemm3m` 的 3M 实现；测试位于 `test/herk/cherk3m_ex/arch35/` 并注册构建目标；在 `blas/herk/README.md` 增加接口说明。构建入口：`bash build.sh --ops=cherk3m_ex --soc=ascend950`。

## 支持硬件

昇腾 950PR（arch35），CANN 9.1.0 及以上。

## 算子约束限制

C 的可写区域不得与 A 的可读区域重叠；调用方保证输入满足 Hermitian 语义。接口不扫描设备张量元素。运行时 n、k 驱动 tiling，不要求动态 shape 机制；结果读取前调用方须同步绑定 stream。相同输入不保证逐位确定性。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 来源 |
| --- | --- | --- |
| 精度 | CPU Netlib BLAS 3.10.0 `cblas_cherk` 生成 golden；INT8 分量精确提升后计算。指定三角实、虚分量分别比较，`atol=rtol=2^-13`，满足混合误差的比例至少 0.99，最大绝对误差不超过 1e-2 或 32 ULP。 | 任务书 §3.2 |
| 性能 | 950PR 使用 `msprof op` 采集 kernel `Task Duration`，预热 5 次后有效采样 10 次取平均，与 GPU kernel-only `gpu_ms` 比较：每 case 加速比至少 0.4，全部 case 算术平均至少 0.5。INT8 沿用 complex64 GPU 基线；事件计时仅用于快速回归。 | 任务书 §3.3 |
| 内存 | 无专项指标。 | 任务书 §3.4 |

采用 ops-blas 测试框架和任务自验包：精度全量 1000 条（含 4 条 3M 专项），性能子集 200 条，并覆盖类型、上下三角、转置、边界尺寸、前导维度 padding、非法参数、零系数、NaN/Inf、INT8 极值及虚部抵消。重点验证未选三角和 padding 不变、beta=0 不读取旧 C、零贡献不读取 A、对角虚部规则。按任务书记录逐 case 精度、kernel 耗时和自验报告；上述为设计验收目标，实际结果以自验为准。

## 兼容性分析

这是新增公开接口，不依赖尚未合入的 `aclblasCherkEx`。参数顺序与对标接口对应；OP_T 和非法枚举返回码的差异如上所述。支持的产品和类型范围以本设计及任务书为准。
