# aclblasCherk3mEx 算子设计文档

# 1 需求背景

## 1.1 需求来源

完成“11月社区任务-aclblasCherk3mEx算子开发（950）”，基于昇腾 950PR（arch35）使用 Ascend C 在 [ops-blas](https://gitcode.com/cann/ops-blas) 仓中实现单精度复数 Hermitian 矩阵秩-k 更新（扩展版）算子 `aclblasCherk3mEx`。

算子实现位于 `blas/herk/arch35/`，公开声明新增于 `include/cann_ops_blas.h`，测试代码位于 `test/herk/cherk3m_ex/arch35/`，算子说明补充于 `blas/herk/README.md`，编译与测试入口为 `bash build.sh --ops=cherk3m_ex --soc=ascend950`。

## 1.2 背景介绍

`aclblasCherk3mEx` 对标 cuBLAS `cublasCherk3mEx`，采用 3M 复数乘法分解完成单精度复数 Hermitian 矩阵秩-k 更新，支持 `COMPLEX64` 与交错 `COMPLEX8I`（`ACL_INT8`）两类输入矩阵 `A`。设计重点在于实现 Hermitian 专用的 3M 分解、在单次 kernel launch 内完成全部三次实数乘与融合后处理、严格保持非目标三角与前导维 padding 不被改写，并在 Atlas 950PR 上兼顾小规模低延迟与中大规模片上融合吞吐。

# 2 需求分析

## 2.1 需求描述

在 `include/cann_ops_blas.h` 新增公共接口声明（与 `cublasCherk3mEx` 参数顺序一致，不定义私有平行 API）：

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

| 参数 | 类型与方向 | 语义与约束 |
| --- | --- | --- |
| `handle` | 输入，`aclblasHandle_t` | ops-blas 句柄，携带绑定 stream；为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 属性，`aclblasFillMode_t` | `C` 的存储三角：`ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `trans` | 属性，`aclblasOperation_t` | `op(A)` 运算类型：`ACLBLAS_OP_N` 或 `ACLBLAS_OP_C`；`ACLBLAS_OP_T` 及其他值返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | 属性，`int` | Hermitian 矩阵 `C` 阶数，`n >= 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，`n = 0` 为合法 no-op |
| `k` | 属性，`int` | 秩-k 更新的内维，`k >= 0`；`k < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | 输入，`const float*` | 实数 `FLOAT32` 标量指针，支持 Host 与 Device 两种位置；为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 输入，`const void*` | 列主序输入矩阵，`OP_N` 时为 `n×k`，`OP_C` 时为 `k×n`；需引用 `A` 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Atype` | 属性，`aclDataType` | 支持 `ACL_COMPLEX64` 与 `ACL_INT8`（交错 2 字节 `(int8 real, int8 imag)`）；不支持返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `lda` | 属性，`int` | `A` 的前导维，满足 `lda >= max(1, trans == ACLBLAS_OP_N ? n : k)`；越界返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | 输入，`const float*` | 实数 `FLOAT32` 标量指针，支持 Host 与 Device 两种位置；为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `C` | 原地输出，`aclblasComplex*` | 列主序 `n×n` Hermitian 复数矩阵，仅读写 `uplo` 三角；`n > 0` 且为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Ctype` | 属性，`aclDataType` | 仅支持 `ACL_COMPLEX64`；不支持返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `ldc` | 属性，`int` | `C` 的前导维，满足 `ldc >= max(1, n)`；越界返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 2.2 需求拆解

1. 严格按约定顺序校验句柄、枚举、维度、前导维、标量指针、数据类型组合与必选矩阵指针，结构校验先于 quick return。
2. 支持 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)` 两种类型组合、`uplo ∈ {UPPER, LOWER}`、`trans ∈ {OP_N, OP_C}`，以及 Host/Device 标量指针。
3. 严格执行 Hermitian 语义：仅读写 `uplo` 指定三角，另一半三角与前导维 padding 保持原值；对角元素虚部计算后强制置 0（忽略输入对角虚部）。
4. 完整覆盖边界条件：`n = 0` no-op、`k = 0` 或 `alpha = 0` 时不引用 `A`（`beta = 1` quick return、`beta != 1` 仅对 `uplo` 三角做 `C = beta * C` 并置对角虚部为 0）、`beta = 0` 不读取旧 `C`。
5. 在单次 kernel launch 内完成 Hermitian 专用 3M 分解与乘加融合，不得通过三次独立 GEMM 拼接，亦不依赖未合入的 `aclblasCherkEx`。

# 3 详细设计

## 3.1 算子分析

### 数学公式与 Hermitian 专用 3M 分解

算子计算如下 Hermitian 秩-k 更新：

```text
C = alpha * op(A) * op(A)^H + beta * C
```

其中 `op(A) = A`（`trans = ACLBLAS_OP_N`，形状 `n×k`）或 `op(A) = A^H`（`trans = ACLBLAS_OP_C`，由 `k×n` 存储的 `A` 共轭转置得到 `n×k`）；`alpha`、`beta` 为实数 `FLOAT32`。

令 `X = op(A) = Xr + i·Xi`（当 `Atype = ACL_INT8` 时，每个复元素按交错 `(int8 real, int8 imag)` 读取并在片上无损提升为 `FP32`）。本算子的右操作数为共轭转置 `B = X^H = Xr^T - i·Xi^T`（即 `Br = Xr^T`、`Bi = -Xi^T`），因此采用 Hermitian 专用 3M 分解：

```text
P0 = Xr * Xr^T
P1 = Xi * Xi^T
P2 = (Xr + Xi) * (Xr - Xi)^T

Re(X * X^H) = P0 + P1
Im(X * X^H) = P2 - P0 + P1    // 等价于 Xi * Xr^T - Xr * Xi^T
```

三次实数矩阵乘（`P0`、`P1`、`P2`）在同一次 kernel launch 内完成，不拆分为三次独立 GEMM 调用；同时严格区分于对称乘积 `X * X^T` 的分解（后者虚部为 `Xi * Xr^T + Xr * Xi^T`，不适用于 Hermitian 秩-k 更新）。对角元素 `C(i, i)` 的虚部在完成 `alpha` 乘积与 `beta * C` 累加后强制置 0。

### 支持数据类型与布局

- `(Atype, Ctype)` 仅支持 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)`。
- `ACL_COMPLEX64` 为交错 `(float real, float imag)` 8 字节复数；`ACL_INT8` 为交错 `(int8 real, int8 imag)` 2 字节复数，在片上无损转为 `FP32`。
- `A`、`C` 均为列主序带前导维（`lda`、`ldc`）布局，不支持广播、0 维 Tensor 或超出前导维语义的非连续布局。

## 3.2 算子实现

### Host 侧设计

**1. 参数校验顺序与边界处理**

Host 入口按如下固定顺序执行校验，结构合法后再进入 quick return 或路径分派：

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `uplo` 不在 `{ACLBLAS_UPPER, ACLBLAS_LOWER}` 或 `trans` 不在 `{ACLBLAS_OP_N, ACLBLAS_OP_C}`（含 `ACLBLAS_OP_T`）返回 `ACLBLAS_STATUS_INVALID_ENUM`；
3. `n < 0`、`k < 0`、`lda < max(1, trans == ACLBLAS_OP_N ? n : k)` 或 `ldc < max(1, n)` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `alpha == nullptr` 或 `beta == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`(Atype, Ctype)` 不在支持集返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；
5. `n == 0` 视为合法 no-op（允许 `A`、`C` 为空），直接返回 `ACLBLAS_STATUS_SUCCESS`；
6. `n > 0` 时若 `C == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；当 `k > 0` 且需引用 `A`（未处于已知 `alpha == 0` 路径）时若 `A == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。

对 `alpha`/`beta` 指针自动识别 Host 与 Device 位置：Host 指针可直接读取用于路径分派与 quick return；Device 指针不在 Host 侧同步回读，交由 kernel 在 handle 绑定的 stream 上直接读取并在设备侧判断 `alpha == 0`、`beta == 0/1`。

**2. 多路径选路与 Tiling 规划（单次 kernel launch）**

Host 依据 `n`、`k`、`Atype`、标量位置及 950PR 片上容量，在四条路径中选择其一（不按用例编号或随机种子分派）：

| 路径 | 适用条件 | 启动方式与职责 |
| --- | --- | --- |
| 1. Quick return | `n == 0`，或 Host 标量满足 `(k == 0 \|\| alpha == 0) && beta == 1` | 不启动 kernel，直接返回 `ACLBLAS_STATUS_SUCCESS` |
| 2. AIV-only 纯 beta 缩放 | `(k == 0 \|\| alpha == 0) && beta != 1`（或 `k == 0` 且 Device 标量） | 单次 AIV kernel，仅对 `uplo` 三角执行 `C = beta * C` 并置对角虚部为 0，不引用 `A`；`beta == 0` 时直接写零而不读旧 `C` |
| 3. 小规模 / 低秩 AIV-only 直接 3M | 小 `n` 或短 `k` 等启动开销敏感形态 | 单次 AIV kernel，直接用向量/SIMT 完成 `uplo` 三角内 3M 计算与 `alpha/beta` 合并，规避 MIX 协同开销 |
| 4. 中大规模片上融合 3M MIX | 中大 `n` 且有效 `k` 的常规与大尺寸形态 | 单次 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC + 2 AIV），仅调度 `uplo` 三角 tile，零 GM 中间工作区 |

在路径 4 中，Host 按输出块尺寸仅统计并编号 `uplo` 三角有效 tile（含对角 tile 与严格上/下三角 tile），计算各核组任务区间与 K 方向分块参数，打包 Tiling 结构体提交至 handle 绑定的 stream 异步执行。

### Kernel 侧设计

**1. 纯 beta 缩放路径与小规模 AIV-only 3M 路径**

- 纯 beta 缩放路径按 `uplo` 有效三角元素切分给 AIV 核：`beta == 1`（Device 标量分支）直接返回；`beta == 0` 跳过旧 `C` 读取并将 `uplo` 三角写零；一般 `beta` 读取旧 `C` 乘以 `beta` 并将对角虚部置 0 后写回，不触碰另一半三角与 padding。
- 小规模 / 低秩 AIV-only 直接 3M 路径将 `A` 的实虚部加载至 UB（`ACL_INT8` 先无损提升为 `FP32`），对 `uplo` 有效位置直接以向量方式完成 `P0`/`P1`/`P2` 累加、合成 `Re/Im`、乘 `alpha`、条件累加 `beta * C_old` 及对角虚部置 0，最后按有效边界掩码写回 `C`。

**2. 中大规模片上融合 3M MIX 路径（`KERNEL_TYPE_MIX_AIC_1_2`）**

单次 kernel launch 内由 1 个 AIC 与 2 个 AIV 通过片上缓冲双缓冲流水协同，无需 GM 中间工作区：

- **仅调度 `uplo` 三角 tile**：核组按三角块线性索引解码得到行列块坐标 `(tileRow, tileCol)`，仅处理满足 `uplo` 约束的块（上三角 `tileRow <= tileCol`，下三角 `tileRow >= tileCol`），避免对冗余半区执行计算与访存。
- **AIV 输入搬运、提升与 3M NZ 面板构造**：AIV 从 GM 读取当前 K 块对应的 `A` 切片至 UB；对 `ACL_COMPLEX64` 解交织得到 `Xr`、`Xi`，对 `ACL_INT8` 解交织并无损转换为 `FP32`；结合 `trans`（`OP_N` 直接取行块，`OP_C` 按转置加载并对虚部取反）构造 `Xr`、`Xi`、`Xr + Xi`、`Xr - Xi` 所需的 NZ 面板，通过双缓冲直写至配对 AIC 的 L1。
- **AIC 关闭 HF32 执行三次 FP32 Cube MMAD**：AIC 显式关闭 HF32 开关，使用纯 FP32 Cube 累加沿 K 维依次计算 `P0 = Xr * Xr^T`、`P1 = Xi * Xi^T`、`P2 = (Xr + Xi) * (Xr - Xi)^T` 至 L0C，K 维归约完成后通过 Fixpipe 双目标路径将三个结果块直写回两个配对 AIV 的 UB（零 GM 落盘）。
- **AIV 片上合成、标量融合与三角掩码写回**：AIV 在 UB 内按 `Re = P0 + P1`、`Im = P2 - P0 + P1` 合成复数结果，乘以实数 `alpha`；当 `beta != 0` 时搬入旧 `C` 并计算 `alpha * prod + beta * C_old`（`beta == 0` 跳过旧 `C` 读取）；对角 tile 强制将对角元素虚部置 0；按 `uplo` 元素级掩码与 `n` 尾块边界将结果写回 GM，保证另一半三角与 `ldc` padding 不被改写。
- **异常数值修复回退**：由于 3M 中间项 `Xr ± Xi` 在极端大值或 Inf/NaN 场景下可能引入虚假溢出或异常抵消，AIV 在 UB 合成阶段检测 tile 内非有限（Inf/NaN）或异常抵消条件；一旦触发，在同一次 kernel 内对该 tile 有效元素回退为逐元素 `FP32` 直接复数乘 `sum_p (X_ip * conj(X_jp))` 重新计算并写回，兼顾常规路径吞吐与极端数值正确性。

## 3.3 支持硬件

| 产品 | 架构与支持范围 |
| --- | --- |
| Atlas 950PR | arch35；`(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)` |

软件基线为 CANN 9.1.0 及以上。

## 3.4 算子约束限制

1. `C` 为原地更新，仅读写 `uplo` 指定三角；调用方须保证 `C` 可写区域与 `A` 可读区域不重叠，并在 Host 读取结果前同步 `handle` 绑定的 stream。
2. 不支持广播、0 维 Tensor 及超出 `lda`/`ldc` 前导维语义的任意非连续布局，不使用 AutoContiguous/ViewCopy。
3. 不支持 `ACLBLAS_OP_T`（返回 `ACLBLAS_STATUS_INVALID_ENUM`），不支持指定组合以外的 `Atype`/`Ctype`（返回 `ACLBLAS_STATUS_NOT_SUPPORTED`）。
4. 3M 路径与直接复数乘舍入路径不同，不要求与非 3M 实现或重复执行逐位一致，验收按混合容差精度标准判定。
5. 本算子与 `aclblasCherkEx` 为独立接口，互不依赖。

# 4 可维可测分析

## 4.1 精度标准、性能标准与内存要求

| 验收项 | 验证方法与标准 |
| --- | --- |
| 精度 Golden | CPU 侧采用 Netlib BLAS 3.10.0 非 3M `cblas_cherk`（float64 参考可用 `cblas_zherk` 后收窄）；`Atype = ACL_INT8` 先将交错 `int8` 实部/虚部无损提升为 `float` 后调用 |
| 精度阈值 | 按生态算子混合容差标准对实部、虚部分别比对：`FLOAT32` 分量 `atol = rtol = 2^-13`，`matched_ratio >= 0.99`，`max_abs_error <= 1e-2`（或 `32 * ULP`） |
| 写保护与特殊场景 | 在非目标三角与 `ldc` padding 写入哨兵值做逐位校验；覆盖对角虚部非零输入（输出置 0）、`alpha = 0` 且 `A` 含 NaN、`beta = 0` 且 `C` 含 NaN、`k = 0` 且 `beta != 1`、`ACL_INT8` 极值、Inf/NaN、`TC_3M_*` 3M 专项用例、Host/Device 标量及全部负向参数用例（全量 1000 条精度用例口径） |
| 性能采集与判据 | 在 Atlas 950PR 上使用 `msprof op` 采集单算子 kernel `Task Duration(us)`，warmup 5 次后取后续 10 次有效采样算术平均并转为 ms，与 GPU cuBLAS `cublasCherk3mEx` 的 kernel-only `gpu_ms` 按 kernel-to-kernel 口径对比：单 case `>= 0.4x`（`npu_ms <= gpu_ms / 0.4`），全部性能用例加速比算术平均 `>= 0.5x`（200 条性能子集口径；另补充 `ACL_INT8` 性能采样并对标 `ACL_COMPLEX64` `gpu_ms` 基线）；GTest `[PERF] avg_us` 仅用于快速趋势回归，不替代正式判据 |
| 内存要求 | 按任务书 §3.4 标注不涉及；MIX 主路径零 GM 中间工作区，并可辅以内存检测工具核查越界与同步安全 |

实测精度报告、性能采样数据、原始日志与截图单独保存在交付验收材料中，不写入本设计文档。

## 4.2 兼容性分析

在 `include/cann_ops_blas.h` 新增公开声明 `aclblasCherk3mEx`，与 `cublasCherk3mEx` 参数序列一致，可被各产品线共用。本次实现与测试限定于 `blas/herk/arch35/` 与 `test/herk/cherk3m_ex/arch35/`，不影响现有非 Ex `aclblasCherk` 及其他架构实现，不引入私有平行接口或独立编译工程。
