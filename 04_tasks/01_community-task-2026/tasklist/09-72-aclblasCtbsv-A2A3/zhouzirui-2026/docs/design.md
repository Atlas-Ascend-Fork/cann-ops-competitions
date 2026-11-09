# 需求背景（required）

## 需求来源

本设计对应“9月社区任务-aclblasCtbsv算子开发（A2/A3）”。任务要求在 Atlas A2/A3 系列产品上，使用 Ascend C kernel 直调方式实现单精度复数三角带状求解接口 `aclblasCtbsv`，功能和参数语义对齐 cuBLAS `cublasCtbsv`，代码归档到 ops-blas 的 `blas/tbsv/arch22/`。

参考接口：

```cpp
aclblasStatus_t aclblasCtbsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    int k,
    const aclblasComplex* A,
    int lda,
    aclblasComplex* x,
    int incx);
```

## 背景介绍

### aclblasCtbsv 算子能力

`aclblasCtbsv` 求解：

```text
op(A) * x = b
```

其中 A 为 n×n 的 COMPLEX64 三角带状矩阵，b 在调用前存放于 x，计算完成后 x 被原地覆盖为解。矩阵采用列主序带状存储，实际数组为 `lda × n`，有效带宽为 `k + 1` 行。

`op(A)` 由 trans 决定：

- `ACLBLAS_OP_N`：A；
- `ACLBLAS_OP_T`：A 的转置；
- `ACLBLAS_OP_C`：A 的共轭转置。

uplo 决定 A 使用上三角带或下三角带；diag 决定主对角是否从 A 读取。当 `diag == ACLBLAS_UNIT` 时，主对角视为 1，不能读取 A 中对应位置。

### 主仓现状

开发开始时，ops-blas 主仓 `tbsv` 算子族仅包含 arch35 的 `aclblasStbsv`，未包含面向 arch22 的 COMPLEX64 `aclblasCtbsv`。因此需要新增 Host 参数校验、arch22 Ascend C kernel、公开接口声明、CBLAS golden、CSV 驱动 GTest、性能测试及 README。

### 参数能力

| 参数 | 含义 | 数据类型 | 约束 |
| --- | --- | --- | --- |
| handle | ops-blas 句柄 | aclblasHandle_t | 不得为空 |
| uplo | 上/下三角带 | aclblasFillMode_t | UPPER / LOWER |
| trans | 矩阵操作 | aclblasOperation_t | N / T / C |
| diag | 主对角类型 | aclblasDiagType_t | NON_UNIT / UNIT |
| n | 矩阵阶数 | int | n ≥ 0 |
| k | 上/下带宽 | int | k ≥ 0 |
| A | 三角带状矩阵 | COMPLEX64 | n>0 时非空，Device 内存 |
| lda | A 的 leading dimension | int | lda ≥ k+1 |
| x | 右端和原地输出 | COMPLEX64 | n>0 时非空，Device 内存 |
| incx | x 的步长 | int | 非 0，支持负值；配套测试将 INT_MIN 作为非法值 |

# 需求分析（required）

## 需求描述

基于 Ascend C 为 Atlas A2/A3（arch22）实现 `aclblasCtbsv`。实现需覆盖上下三角、N/T/C、单位/非单位对角、正负 incx、lda padding、n=0、k=0、非法参数以及 Inf/NaN 等场景；接口通过 handle 绑定的 stream 异步下发。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 增加标准公开接口声明。
2. Host 侧完成全部枚举、维度、步长、句柄和指针校验，并实现 quick return。
3. Kernel 正确实现四种求解方向和 COMPLEX64 乘减、除法、共轭。
4. 支持正负 incx，且不访问 stride hole。
5. 使用 Netlib CBLAS `cblas_ctbsv` 生成 golden，按任务书 Mixed Tolerance 验证实部和虚部。
6. 覆盖任务配套的 1000 条精度/边界用例和 200 条性能/内存用例。
7. 910B3 五个性能基准点的 kernel 平均耗时不超过任务书上限。
8. 不申请额外 Device workspace，保持 Host 异步语义。

# 详细设计（required）

## 算子分析

### 数学公式

三角带状系统为：

```text
op(A) * x = b
```

复数乘法：

```text
(ar + i*ai) * (xr + i*xi)
= (ar*xr - ai*xi) + i*(ar*xi + ai*xr)
```

复数除法采用 Smith 形式，避免直接计算 `dr² + di²` 时过早溢出。共轭转置时，将参与计算的 A 元素虚部取反，并使用共轭后的主对角进行除法。

### 带状存储索引

列主序带状数组中：

```text
UPPER: bandRow = k + row - col
LOWER: bandRow = row - col
offset = bandRow + col * lda
```

只有落在 uplo 指定带宽内的元素会被访问。A 的每个 COMPLEX64 元素在 Device 上按连续两个 float 存储。

### 求解方向

| uplo | trans | op(A) 的有效三角 | 求解方向 |
| --- | --- | --- | --- |
| LOWER | N | 下三角 | j=0→n-1，前代 |
| UPPER | N | 上三角 | j=n-1→0，回代 |
| LOWER | T/C | 上三角 | j=n-1→0，回代 |
| UPPER | T/C | 下三角 | j=0→n-1，前代 |

行/列之间存在严格的数据依赖，因此 kernel 使用 1 个 AIV block；并行优化放在单列带内的向量计算，求解步骤之间保持串行依赖。

## 算子实现

### Host 侧设计

Host 侧执行顺序：

1. 首先检查 handle；为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 校验 uplo、trans、diag；非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. 校验 `n >= 0`、`k >= 0`、`lda > k`、`incx != 0` 且 `incx != INT_MIN`。
4. `n == 0` 时返回成功，不要求 A/x 非空。
5. n>0 时检查 A/x 非空。
6. `k == 0 && diag == ACLBLAS_UNIT` 时返回成功，x 不变。
7. 填充 `CtbsvTilingData`，其中保存 A/x 地址、n、k、uplo、trans、diag、incx、lda 和大规格 GM 兜底标志。
8. 通过 `handle->stream` 异步下发对应的模板特化 kernel，不在 Host 侧同步。

Tiling 数据按值传给 kernel，不申请 Host 临时 Device 内存，也不申请 workspace。

### Kernel 侧设计

#### 模板特化

对 `uplo × trans × diag` 的 2×3×2 共 12 种组合进行编译期模板特化，消除求解内层的枚举分支。Host 根据入参选择对应 kernel。

#### x 缓存与负步长

逻辑下标 i 到物理 x 地址的映射为：

```text
incx > 0: i * incx
incx < 0: (n - 1 - i) * abs(incx)
```

快速路径将 x 的实部和虚部分别缓存到 UB。`incx == 1` 时批量搬运打包数据；其他步长逐元素装载，计算完成后按相同映射写回，stride hole 不被修改。

#### A 列双缓冲

A 以列为求解单元搬入 UB，使用 ping/pong 两块列缓冲：当前列计算时预取下一求解列，隐藏部分 GM→UB 搬运延迟。只搬运当前列可能被访问的有效带区间，尾块使用 DataCopyPad 处理非 32 字节对齐长度。

#### N 路径

N 路径按列完成 BLAS 标准的 column update：

1. 读取 x[j]；NON_UNIT 时除以 A[j,j]。
2. 将解写回 UB。
3. 更新带内后继元素 `x[i] -= x[j] * A[i,j]`。

短带使用标量复数乘减。长带先处理最多 7 个未对齐前缀，使 x 的 UB 地址满足 32 字节对齐；剩余区间用 Gather 拆出 A 的实部/虚部，并使用 Axpy 完成向量化复数乘减。8～16 个元素的 UNIT 更新采用向量计算、标量写回的混合路径。满带 UNIT 系统容易出现 FP32 溢出边界，为保持与 CBLAS 相同的逐列求值顺序，满带场景使用标量 column update。

#### T/C 路径

T/C 路径需要计算当前列对应的带内点积。NON_UNIT 的极短区间使用标量累加；其余区间使用 Gather 得到 A、x 的连续实部/虚部向量，通过 MulAddDst 形成复数乘积后分别 ReduceSum。C 路径在乘加公式中直接处理共轭符号。UNIT 路径不读取主对角，因此不建立 MTE2→Scalar 同步事件。

#### 大规格 GM 兜底

快速路径需要在 UB 中缓存完整 x 及单列工作区。为覆盖任务书未限定上界的合法输入，当 `n > 4096` 或 `k > 2048` 时，Host 设置 `useGmFallback`，kernel 切换到标量 GM 求解路径。GM 路径保持相同的 12 种模板特化、求解方向、共轭和正负 incx 语义，不申请 workspace。任务性能规格均位于快速路径范围内。

#### 特殊值

复数乘法包含与 libgcc complex multiply 一致的 Inf-Inf 恢复路径。复数除法使用 Smith 形式，并处理零对角、Inf 和 NaN 的传播。算子不检查奇异性；NON_UNIT 测试数据通过对角增强保证非零。

### UB 与内存

主要 UB 区域包括：

- x 实部、x 虚部及打包缓冲；
- A 列 ping/pong 缓冲；
- 实部/虚部分离缓冲；
- 向量乘法、规约和 Gather offset 缓冲。

任务精度最大 n=2048，性能最大 n=4096、k=128，所需 UB 在 arch22 可用范围内。更大的合法 n/k 由 GM 兜底路径处理。算子没有额外 Device workspace；最大性能 case 的 A+x 为 4,259,840 字节，约 4.063 MiB。

### TilingKey 规划

本算子不需要 TilingKey。uplo、trans、diag 由 Host 选择不同模板 kernel；n、k、lda、incx 作为运行时 tiling 参数传入。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3 | √ |

## 算子约束限制

- 仅支持 COMPLEX64。
- A 使用 BLAS 列主序三角带状存储，`lda >= k+1`。
- x 原地读写，A 与 x 不允许内存重叠。
- incx 支持正负值，不支持 0；为避免 `abs(INT_MIN)` 无法由 int 表示，并与配套负向用例一致，当前实现将 INT_MIN 返回为非法值。
- UNIT 时主对角不读取。
- 不检查矩阵奇异性。
- 不支持超出 lda/incx 语义的任意非连续 Tensor 或 broadcast。
- n=0 为合法 no-op。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32 Mixed Tolerance：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤max(1e-2, 32×ULP) | 任务书与生态算子开源精度标准 |
| 性能标准 | 910B3 五个任务书 case 的 NPU kernel 平均耗时不高于 GPU 基线/0.8 | 任务书 3.3 节 |
| 内存目标 | 不申请 workspace | 本方案实现目标 |

实测环境为 Ascend 910B3（A2）和 Ascend910_9382（A3），CANN 版本均为 9.1.0。两端分别完成 1000 条 CSV 精度/边界用例、NullHandle、大 n GM 兜底和大 k GM 兜底测试，GTest 结果均为 1003/1003 通过。

测试保留任务附件 1200 条用例的全部原始字段和参数，并为 972 条有效随机精度用例增加分布标记：486 条使用 `[-5,5]` 均匀分布，486 条使用 `μ=0、σ=0.1` 正态分布，满足任务书规定的 50%/50% 比例及 `μ∈[-5,5]、σ∈[0.1,2]` 范围。为避免 UNIT 三角系统在正常精度测试中因条件数失控而掩盖实现误差，482 条 UNIT 随机用例进入正态组，另取 4 条 NON_UNIT 用例补足正态组；其余 486 条 NON_UNIT 用例进入均匀组。

当前精度验证以 Netlib CBLAS `cblas_ctbsv` 为 golden，按任务书 Mixed Tolerance 分别比较实部和虚部，阈值见上表。普通有限值精度用例超阈即判失败，不使用 FP64 参考解或残差复核将失败结果转为通过，也不以 MERE/MARE 替代 Mixed Tolerance。

1000 条 CSV 用例中，994 条为常规精度及边界用例，6 条为显式极值、Inf、NaN 用例。常规数值精度按上述 Mixed Tolerance 判定；负向用例精确比较接口返回码，`n=0` 用例检查成功返回，`k=0 && diag=UNIT` 用例检查成功返回及 x 保持不变。显式特殊值用例按特殊值传播语义验证，对齐溢出分类后仍执行 Mixed Tolerance，未参与溢出分类对齐的有限分量不豁免精度检查。另有 3 条独立 GTest 验证空句柄和两类 GM 兜底路径。

早期版本曾对 14 条未直接通过 Mixed Tolerance 的有限值用例增加 FP64 前向误差与归一化残差复核。后续实现改进后已取消这项二次放行，本段同步当前 arch22 测试的验证方式，任务书的精度阈值保持不变。对应代码评审：https://gitcode.com/cann/ops-blas/pull/612 。

性能程序为每次 kernel 调用准备独立 x 副本，避免原地重复求解改变后续工作量。20 次 warmup 后统计 100 次有效调用。配套 200 条性能用例在 A2、A3 均为 200/200 通过，最小 `gpu_us / npu_us` 为 0.804。

A2 五个任务书基准点 `msprof op` 最终结果：

| case | n | k | uplo | trans | diag | NPU kernel(us) | 上限(us) | 结果 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | 256 | 8 | UPPER | N | NON_UNIT | 162.380 | 254.5 | 通过 |
| 2 | 512 | 32 | LOWER | N | NON_UNIT | 239.900 | 493.8 | 通过 |
| 3 | 1024 | 16 | UPPER | T | UNIT | 428.000 | 695.5 | 通过 |
| 4 | 2048 | 64 | LOWER | C | NON_UNIT | 957.320 | 2265.0 | 通过 |
| 5 | 4096 | 128 | UPPER | N | UNIT | 2146.380 | 2468.0 | 通过 |

## 测试覆盖

- `uplo × trans × diag` 全组合；
- n=0、1、质数、2 的幂及相邻值、非对齐值；
- k=0、1、小带宽、半带宽、满带宽；
- lda padding；
- incx=±1、±2、±3 和其他非对齐步长；
- 空句柄、空 A/x、非法枚举、负 n/k、非法 lda、incx=0/INT_MIN；
- UNIT 对角不读、Inf/NaN 和极值场景；
- 5 个任务书性能 case 和全部 200 条性能/内存参数；
- `n=4097,k=0` 和 `n=2050,k=2049` 两类 GM 兜底，后者覆盖 UPPER/C/UNIT/负 incx。

## 兼容性分析

新增 `aclblasCtbsv` 标准接口，不修改已有 `aclblasStbsv` 行为。接口声明位于公共头 `include/cann_ops_blas.h`，Host 使用现有 aclblasHandle stream 语义，ABI 中不包含新增全局状态。代码仅在 arch22 目录参与 A2/A3 构建，其他架构保持原有支持范围。
