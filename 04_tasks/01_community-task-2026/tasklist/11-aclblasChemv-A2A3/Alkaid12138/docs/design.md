# aclblasChemv 算子设计文档

> 任务：aclblasChemv（Atlas A2/A3，社区任务）
> 目标仓库：`https://gitcode.com/cann/ops-blas`
> 目标目录：`blas/symv/arch22/`（实现）、`test/symv/chemv/arch22/`（测试）
> 参考模板：cann-competitions `04_tasks/01_community-task-2026/resources/design_template.md`

# 需求背景（required）

## 需求来源

- 任务来源：CANN 社区任务「11月社区任务-aclblasChemv算子开发（A2/A3）」。
- 对标接口：cuBLAS `cublasChemv`（参数序列与语义），数学参考 Netlib `chemv.f`。
- 上游仓库：`gitcode.com/cann/ops-blas`，BLAS Level-2（矩阵-向量乘）能力补齐。

## 背景介绍

### 对标实现与现状

BLAS 的 `hemv` 计算 `y = alpha*A*x + beta*y`，其中 `A` 为 `n×n` Hermitian 矩阵
（`A = A^H`），按列主序存放在 `lda×n` 数组中，只引用 `uplo` 指定的上/下三角，
未引用三角由共轭对称隐含；对角元素虚部假定为 0，不参与计算。

现状（`ops-blas` master，commit `5bd9514`）：

| 能力 | 位置 | 架构 | 与本任务关系 |
|---|---|---|---|
| `aclblasSsymv`（实数对称 matvec） | `blas/symv/arch22/ssymv_*` | A2/A3 | 同族算法骨架（列扫 + 原子累加、双 phase） |
| `aclblasCsymv`（复数对称 matvec） | `blas/symv/arch35/csymv_*` | Ascend 950 | Host 校验编排、Host/Device 标量、tile 部分和 + reduce |
| `aclblasChpmv`（Hermitian packed matvec） | `blas/spmv/arch35/chpmv_*` | Ascend 950 | Hermitian 语义：镜像共轭、对角只读实部 |
| `chemm`（Hermitian 矩阵乘） | `blas/hemm/arch22/chemm_*` | A2/A3 | arch22 Hermitian 三角 mask、共轭镜像、对角虚部置零 |
| `aclblasChemv` | **不存在** | — | 本任务新增 |

### 现状分析

- `include/cann_ops_blas.h` 无 `aclblasChemv` 声明，A2/A3 无 Hermitian 复数 matvec；
  950 的 SIMT/RegBase 实现不能在 arch22 复用（指令与并行模型不同）。
- 需在 arch22 上以 Ascend C 向量 API 重新设计：复数按实部/虚部平面拆解、镜像共轭、
  只读单三角、正负步长、Host/Device 标量、原地 `y`。
- 本算子为 O(n²) 访存主导的 Level-2 算子，性能关键在于「三角只读一次」与
  「x 复用、搬运与计算重叠」，因此设计上给出多档路径并按 n 选择。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 上实现：

```text
y = alpha * A * x + beta * y
```

- `alpha`、`beta`、`A`、`x`、`y` 均为 `aclblasComplex`（COMPLEX64，实/虚各 float32）。
- `A` 为 Hermitian 矩阵，列主序存储于 `lda×n`，仅 `uplo` 指定三角被引用；
  对角元素只取实部参与计算。
- `x`、`y` 为长度 `n` 的向量，支持正、负步长（`incx/incy != 0`）。
- 接口与原 `aclblasSsymv`/`aclblasCsymv` 同族，声明放入 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

## 需求拆解

| 编号 | 需求 | 设计落点 | 验证方式 |
|---|---|---|---|
| R1 | Hermitian 语义：镜像共轭、对角只取实部、未引用三角不读 | kernel 索引与掩码；`chemv_golden.h` 按 `cblas_chemv` 语义；测试对未引用三角填 NaN | 精度用例 TC_L0/SQ/EX、Hermitian 专项 |
| R2 | 参数校验顺序（7 步）与错误码 | host `Validate…` 顺序实现 | TC_ED 负向用例 |
| R3 | `n=0` / `alpha=(0,0)&beta=(1,0)` / `beta=0` 语义 | host quick return + scale-only 路径 | TC_ED_191~203 |
| R4 | 正负步长、`lda` padding、`INT32_MIN`、跨度溢出 | generic 路径 + host int64 守卫 | TC_INC / TC_LD / TC_ED |
| R5 | 精度：FLOAT32 分量档（2^-13, ratio 0.99, max_abs 1e-2/32ULP） | 测试 `VerifyComponent`，实/虚分别判定 | 随包 1000 条精度用例 + GAUSS 补充 |
| R6 | 性能：≤1.25× A100（≥0.8×） | owner/fused 路径、x 复用、双缓冲 | TC_PF 200 条 + msprof |
| R7 | 接口与产品支持表 | `include/cann_ops_blas.h`、`blas/symv/README.md` | 编译与文档评审 |

# 详细设计（required）

## 算子分析

### 数学公式

记 `A(i,j)` 为第 i 行第 j 列元素，列主序存储 `a[j*lda + i]`。Hermitian 性质：

```text
A(j,i) = conj(A(i,j)),   A(i,i) 为实数（对角虚部不参与计算）
```

从而

```text
y[i] = alpha * ( Σ_{j} A(i,j) * x[j] ) + beta * y[i]
```

对 UPPER（只存 `i <= j`）：

```text
y[i] = alpha * ( Σ_{j>=i} A(i,j)x[j] + Σ_{j<i} conj(A(j,i))x[j] + real(A(i,i))x[i] ) + beta*y[i]
```

对 LOWER（只存 `i >= j`），把上式的三角方向取反。

只读集合：UPPER 读 `{a[j*lda+i] | i <= j}`；LOWER 读 `{a[j*lda+i] | i >= j}`。
未引用三角的存储值一律不读（测试用 NaN 填充以拦截误读）。

复数乘法分解：

```text
(ar + i·ai)(xr + i·xi) = (ar·xr - ai·xi) + i·(ar·xi + ai·xr)
conj(a)·x            = (ar·xr + ai·xi) + i·(ar·xi - ai·xr)
```

### 支持数据类型

| 参数 | 类型 | 说明 |
|---|---|---|
| `alpha`、`beta` | `aclblasComplex`（COMPLEX64） | Host 或 Device 标量（运行时探测指针位置） |
| `A`、`x`、`y` | `aclblasComplex`（COMPLEX64） | Device 内存；`y` 原地覆写 |

### 支持形状/布局

- `A`：`lda × n` 列主序，`lda >= max(1, n)`；padding 列不参与计算。
- `x`、`y`：逻辑长度 `n`，物理长度至少 `1 + (n-1)·|inc|`；负步长时逻辑第 i 个元素位于
  `(n-1-i)·|inc|`。
- `n = 0` 合法；`n` 为运行时入参，不做 dynamic shape 特化。
- 性能用例上限 `n = 4096`（单用例 host 侧内存 ≤ 512MB 的用例约束）。

## 算子实现

### 总体方案

复数按「实部平面 / 虚部平面」计算：搬入时以 AoS（实虚交错）连续搬运，UB 内用
`Gather` 按字节索引拆成 `re/im` 两个 float32 平面；复数乘法用 4 次乘加在平面上完成。
按 n 与步长选择三档路径：

| 路径 | 适用条件 | 归约/原子 | workspace | 说明 |
|---|---|---|---|---|
| `OWNER` | `incx==1 && incy==1` 且 `n < FUSED_MIN_N` | 无 | 无 | 每个 AIV 独占一段连续输出行，核内完成全部累加 |
| `FUSED` | `incx==1 && incy==1` 且 `n >= FUSED_MIN_N` | workspace 部分和 + reduce kernel | `~n²/R·8B` | 三角块只读一次，双方向部分和，二段归约 |
| `GENERIC` | 其余（任意正负步长、超大 lda） | 无 | 无 | 按输出行标量实现，正确性优先 |
| `SCALE` | `alpha == 0` 且 `beta != 1` | 无 | 无 | `y = beta*y`，不读 A/x |

`FUSED_MIN_N` 初始取 2048（按 profiling 调整）。基线实现先落地 `OWNER/GENERIC/SCALE`，
`FUSED` 作为性能优化项按 A/B 数据启用（见「性能方案」）。

### host 侧设计

#### 参数校验与 quick return

严格按任务书 §2.5 的顺序（顺序错误会被 TC_ED 用例拦截）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `n == 0` → `ACLBLAS_STATUS_SUCCESS`（**不访问任何指针/枚举**）
4. `uplo` 非 `ACLBLAS_UPPER/ACLBLAS_LOWER` → `ACLBLAS_STATUS_INVALID_ENUM`
5. `lda < max(1, n)` → `ACLBLAS_STATUS_INVALID_VALUE`
6. `incx == 0 || incy == 0` → `INVALID_VALUE`；`incx/incy == INT32_MIN` → `INVALID_VALUE`；
   `(n-1)·|inc|` 以 int64 计算，超过 `INT32_MAX` → `INVALID_VALUE`（对齐 ssymv arch22 issue #807 守卫）
7. `alpha == nullptr || beta == nullptr` → `INVALID_VALUE`
8. 按指针属性读取 `alpha/beta`（`ReadScalarFromHostOrDevice`：host 直接解引用；
   device 走 `aclrtMemcpyAsync + aclrtSynchronizeStream`）
9. `alpha == (0,0) && beta == (1,0)` → `ACLBLAS_STATUS_SUCCESS`（不读 A/x/y）
10. `alpha != (0,0) && (A == nullptr || x == nullptr)` → `INVALID_VALUE`
11. 非第 9 步且 `y == nullptr` → `INVALID_VALUE`（`beta == 0` 不豁免非空）

特殊值处理：`beta == 0` 时不读 `y` 旧值（只写）；`beta == 1` 时直接加旧值；
`alpha == 0` 时走 `SCALE` 路径，不读 `A/x`。

#### tiling 与路径选择

Host 读取 `GetAivCoreCount()` 与 handle workspace 容量，按下表构造 tiling：

| 条件 | `path` | 参数 |
|---|---|---|
| `alpha == 0 && beta == 1` | 无 kernel | quick return |
| `alpha == 0` | `SCALE` | `useCoreNum=1`，按 `incy` 处理 |
| `incx != 1 \|\| incy != 1` | `GENERIC` | `useCoreNum` 按行分配 |
| `incx == 1 && incy == 1 && n >= FUSED_MIN_N && workspace 足够` | `FUSED` | `tileRows`、`colChunk`、`coreNum` |
| 其余 `incx == 1 && incy == 1` | `OWNER` | `blockRows`、`coreNum` |

`OWNER` 切分：`coreNum = min(useCoreNum, ceil(n / 16))`，`blockRows = roundUp(ceil(n/coreNum), 8)`，
第 c 个核处理行 `[c·blockRows, min(n, (c+1)·blockRows))`；不足一行的核直接返回。

`FUSED` 切分：`R = tileRows`（初始 128，`n < 512` 时 64），`K = ceil(n/R)`，
三角任务数 `T = K(K+1)/2`（UPPER）/ 同（LOWER），任务按 `task % coreNum` 轮转分配；
部分和 workspace 大小 `K² · R · 2 · sizeof(float)`（UPPER/LOWER 均按满阵索引，便于 reduce 统一）。

tiling 数据结构（`chemv_tiling_data.h`）：`n/lda/uplo/path/useCoreNum/blockRows/tileRows/colChunk/
alphaReal/alphaImag/betaReal/betaImag/incx/incy`，字段 8B 对齐。

#### 校验与下发流程

```text
ValidateCall (7 步) → 读标量 → quick return 判定 → 选路径 → 构造 tiling
  → chemv_kernel_do(..., numBlocks, stream)  // 同一 stream，异步
```

### kernel 侧设计

#### 数据搬运（CopyIn）

- `x`：`OWNER/FUSED` 路径要求 `incx == 1`，每核一次性搬入整个 `x`（`2n` floats，n≤4096 时 32KB），
  UB 内 `Gather` 拆成 `xRe/xIm` 两个平面，供本核所有列复用（对应性能项 P5「x 复用」）。
- `A`：按列块 `DataCopyPad` 搬入，`DataCopyExtParams{blockCount=列数, blockLen=rows·8B, srcStride=(lda-rows)·8B, dstStride}`；
  `dstStride` 把每列对齐到 32B，便于后续 `Gather` 取平面。
- `y` 旧值：仅 `beta != 0` 时搬入，`incy == 1` 时连续搬运；`beta == 0` 完全不读。
- 尾块：`DataCopyPadExtParams` 补 0，向量操作按有效长度（count）裁剪，避免越界读。

#### 计算（Compute）

复数统一拆为 `re/im` 平面，乘加用 `Axpy`/`Muls`/`Add`/`Sub`：

- 直接项（`A(i,j)·x[j]`，j 固定、i 为向量）：

```text
accRe[i] += ar[i]*xr[j] - ai[i]*xi[j]
accIm[i] += ar[i]*xi[j] + ai[i]*xr[j]      // 4×Axpy
```

- 镜像点积（`conj(A(i,j))·x[i]` 求和，输出到 `y[j]`）：

```text
pRe += ar*xr + ai*xi
pIm += ar*xi - ai*xr                       // 4×Muls + 2×Add/Sub + ReduceSum
```

- 对角元素：搬入后把该列 UB 中的 `ai[diag]` 置 0（`SetValue`），使后续运算等价于
  `real(A(i,i))·x[i]`，且**从不读取对角虚部**（避免 `0*Inf` 污染）。
- 三角边界：`OWNER` 路径把每个 chunk 限制在三角内：
  - UPPER：第 1 阶段处理列 `[r1, n)` 的行 `[r0, r1)`；第 2 阶段处理列 `i ∈ [r0, r1)` 的行 `[0, i]`；
  - LOWER：第 1 阶段处理列 `[0, r0)` 的行 `[r0, r1)`；第 2 阶段处理列 `i ∈ [r0, r1)` 的行 `[i, n)`。
  所有搬入与向量长度均落在三角内，**不读未引用三角**。

`FUSED`：把三角划分为 `R×R` 块（对角块与上/下三角块）：
- 非对角块 `(I,J)`（UPPER `I<J`）：一次搬入同时计算 `rowPartial[I][J]`（直接项，向量累加）
  与 `colPartial[J][I]`（镜像项，`WholeReduceSum` 按行归约），写入 workspace；
- 对角块 `(I,I)`：块内三角直接项 + 镜像点积写同一部分和；
- reduce kernel：每个输出行块读 `K` 个部分和，`Add` 求和 → 乘 `alpha` → 加 `beta·y_old` → 写回 `y`。

#### 写回（CopyOut）

`OWNER`：核内 `out = alpha*acc + beta*y_old` 一次写回（`y` 每元素只写一次，无原子操作）：

```text
outRe = alphaRe*accRe - alphaIm*accIm + betaRe*yRe - betaIm*yIm
outIm = alphaRe*accIm + alphaIm*accRe + betaRe*yIm + betaIm*yRe
```

`beta == 0` 时省略后两项；`beta == 1` 时退化为加法。`incy == 1` 用 `DataCopyPad` 写回；
`GENERIC` 路径按物理位置标量写回。

#### 同步与流水

- `OWNER`：MTE2（搬 A 列块）与 V（乘加）用 `TQue`/双缓冲重叠；`MTE3`（写 y）在最后一次性发出，
  避免每列同步。
- `FUSED`：部分和写 workspace 后由 reduce kernel 读取，依赖由 stream 顺序保证；
  同一 kernel 内不同 tile 无数据依赖，不需要跨核 barrier。
- 仅在真实依赖处使用 `SetFlag/WaitFlag` 或 `PipeBarrier<PIPE_V>`；不在内层循环加 barrier。

#### 入口

```cpp
__global__ __aicore__ void chemv_kernel(...)       // 按 tiling.path 分派
void chemv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workspace,
                     const ChemvTilingData& tiling, uint32_t numBlocks, void* stream);
```

`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；每次调用由 host 按 `numBlocks` 下发。

## 支持硬件

| 芯片/产品 | 支持 | 说明 |
|---|---|---|
| Atlas 800T A2（910B3） | √ | arch22，CANN 9.1.0；性能基准设备 |
| Atlas 800T A3 / Atlas A3 训练推理系列（ascend910_93） | √ | arch22，CANN 9.1.0 |

## 算子约束限制

- `A/x/y` 位于 Device；`alpha/beta` 可位于 Host 或 Device。
- 仅支持 `lda/incx/incy` 语义内的非连续访问，不支持任意 stride 的 tensor 视图。
- 不支持的输入组合返回 `ACLBLAS_STATUS_INVALID_VALUE` / `INVALID_ENUM`（见校验表）。
- `n` 很大时受 host 侧工作区与用例内存预算约束（性能用例上限 4096）。
- 对角虚部不读：即使设备内存对角虚部为 NaN/Inf，也不得影响结果（专项用例覆盖）。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源 |
|---|---|---|
| 精度 | golden 为 `cblas_chemv(CblasColMajor)`；实部、虚部分别按 FLOAT32 判定：`|actual-golden| <= atol + rtol*|golden|`，`atol=rtol=2^-13`；`matched_ratio >= 0.99`；`max_abs_error <= max(1e-2, 32*ULP)` | 任务书 §3.2 |
| 非有限值 | golden/actual 同为 NaN、同为同号 Inf、一方 NaN 一方 ±Inf 视为等效；有限 vs 非有限判失败；非有限元素不计入 ratio/max_abs，单独统计 | 任务书 §3.2 |
| 性能 | `msprof op` 单算子 task duration（us）；5 次 warmup 不计入，≥10 次有效采样取算术平均；判据 `Avg <= GPU/0.8` | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

性能基线（`gpu_baseline.csv`，NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，
warmup 5 后有效采样均值）与 1.25× 上限：

| case | uplo | n | gpu_ms | 上限 us（×1.25） |
|---|---|---:|---:|---:|
| chemv-base-000 | UPPER | 512 | 0.0146432 | 18.30 |
| chemv-base-001 | UPPER | 1024 | 0.0211115 | 26.39 |
| chemv-base-002 | LOWER | 2048 | 0.0300885 | 37.61 |
| chemv-base-003 | LOWER | 4096 | 0.0652971 | 81.62 |

参考：同族 09-52 aclblasCsymv-A2A3 设计文档记录 A2 实测 512=6.7us、2048≈26.4/27.1us、
4096≈79.3/79.0us（其性能上限更严）。本算子多出镜像共轭与对角处理，需在实现中以
「三角只读一次 + x 复用 + 双缓冲」抵消。

## 测试方案

测试工程：`test/symv/chemv/`（CSV 驱动 GTest，arch22 注册）。

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)` |
| `chemv_param.h` | CSV 参数解析（`a_fill/x_fill/y_fill`、`GAUSS_<mu>_<sigma>`、`NULLPTR`） |
| `chemv_golden.h` | `cblas_chemv` CPU golden（Hermitian：对角只取实部） |
| `arch22/chemv_test.cpp` | GTest 主体：设备缓冲、调用、分量比对、`[PERF]` 输出 |
| `arch22/chemv_test.csv` | 随包 1200 条用例 + 补充 GAUSS 用例 |
| `arch22/gpu_baseline.csv` | 随包 200 条 A100 基线 |
| `README.md` | 编译、运行、msprof 复现步骤 |

覆盖：

- 随包 1200 条：TC_L0（基础）、TC_SQ（尺寸扫描）、TC_AB（标量组合）、TC_LD（lda padding）、
  TC_INC（±1/±2/±3 步长全组合）、TC_FL（均匀/零/交替/极端/Inf/NaN）、TC_CV、TC_ED（边界负向）、
  TC_EX（扩展）、TC_PF（性能）。
- 补充正态分布用例：`GAUSS_μ_σ`（μ∈[-5,5]、σ∈[0.1,2]）覆盖 A/x/y 与 alpha/beta，
  条数与覆盖在自测报告说明。
- 专项：对角虚部置 NaN（验证不读）、未引用三角置 NaN（验证不读）、`y` 步长空隙保持不变。
- 负向用例缓冲区：单元素 dummy 或 `NULLPTR`，不按 `(n-1)|inc|` 预分配。

复现命令：

```bash
bash build.sh --soc=ascend910b3 --ops=chemv --run --device=0
./build/test/symv/chemv/chemv_test --gtest_filter=-*TC_PF*
msprof op --application="./build/test/symv/chemv/chemv_test --gtest_filter=*TC_PF*" --output=./prof_chemv
```

## 兼容性分析

- 新增公开接口 `aclblasChemv`，不改动既有接口签名与行为；与 `aclblasSsymv`/`aclblasCsymv`
  参数序列一致，便于其他产品线复用。
- arch22 与 arch35 按 `SOC_ARCH_DIRS` 分别构建，互不影响。
- 新增测试目录不改变既有测试注册（`test/symv/ssymv`、`test/symv/csymv` 保持原状）。

# 性能方案

## 基线、profiling 与 bound 判断

1. 固定环境：A3 单卡、CANN 9.1.0、Release、同一输入（`gpu_baseline.csv` 的 TC_PF 参数）。
2. 基线：`OWNER` 路径（首个可用实现），记录 `msprof op` task duration；同时报 avg/median/min。
3. bound 判断：
   - 4096 半三角 ≈ 67MB，上限 81.6us → 约 0.82TB/s；若 `OWNER` 2× 读（≈134MB）则先撞 MTE2/HBM bound。
   - 对比 MTE2/MTE3、V、Scalar 占比，确定是搬运主导还是归约主导。
4. 每轮只改一个维度，保留 A/B 与失败方案。

## 优化候选（按算法/切分/搬运/计算/同步/host 分类）

| # | 类别 | 候选 | 预期收益 | 风险/代价 |
|---|---|---|---|---|
| P1 | 算法 | **FUSED 三角块单次读 + 双方向部分和 + reduce**（大 n） | 矩阵读量减半（4096: ~134MB → 67MB） | workspace ~1MB + reduce kernel；任务映射复杂 |
| P2 | 算法 | **OWNER 行块**（中小 n，无原子/无归约） | 单遍写回、实现简单、无跨核同步 | 大 n 双读放大 |
| P3 | 算法 | 列扫 + `SetAtomicAdd`（Netlib 直译，ssymv 式两 phase） | 每元素只读一次、实现最短 | 原子争用；仅作对照基线 |
| P4 | 切分 | 行矩形 R×K（R=4/8/16/32/48）与 64/128 三角 tile；尾块特化；小 n 单核 | 满载、少空转 | 需 A/B 定阈值 |
| P5 | 搬运 | x 全量常驻 UB；A 列块 `DataCopyPad`（lda stride）；双缓冲；32B/512B 对齐；列块分块命中 L2 | MTE2 与 V 重叠、降 HBM 流量 | UB 预算与流水复杂度 |
| P6 | 计算 | 实/虚平面化；复杂乘加 4 乘 4 加；`WholeReduceSum` 批量归约；去 `GetValue/SetValue` | V 周期下降、归约指令数下降 | 需保持 FP32 舍入口径 |
| P7 | 同步/host | 单 kernel 融合 beta（不做 ssymv 式 phase-0 额外 y 读写）；tiling 缓存；workspace 复用；避免多余 D2H 同步 | 少一次 y 读写与一次 launch | 分支处理 beta=0/1 |
| P8 | 特值/步长 | `alpha=0` → SCALE；`beta=0/1`；`alpha=1`；`incx=1` 连续快路径；负步长只改索引 | 常见 case 直接收益 | 用例需覆盖各分支 |

## 迭代计划（A/B 与门禁）

1. M1：`OWNER + GENERIC + SCALE` 落地 → 全量精度 + 基线性能；确认 512/1024/2048 达标。
2. M2：`FUSED`（P1）落地 → 4096 与大 n case A/B；精度回归。
3. M3：P4/P5/P6 微调（tile 尺寸、列块、归约方式）→ 每轮同 case 对比。
4. 每轮记录：commit、编译选项、设备、CANN、case、warmup/采样数、avg/median/min、相对基线倍数。
5. 精度不回退是进入下一轮的前提；失败方案保留在 `work/` 并说明原因。

# 风险与备选方案

## 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| `OWNER` 大 n 双读放大 | 4096 可能超出 1.25× 上限 | 启用 `FUSED`（P1）；阈值按实测调整 |
| `FUSED` 归约开销 | reduce kernel 增加 2~4us + workspace 读写 | 同驻留 reduce（`SyncAll`）或增大 R 降低部分和数量 |
| 复数平面化搬运成本 | `Gather` 拆平面占 V 周期 | 用 `TransDataTo5HD` 批量转置/拆平面（P6） |
| 负步长/非 1 步长标量路径 | 精度用例耗时长 | 标量路径仅覆盖 TC_INC/TC_EX 中的非 1 步长；性能用例均为 1 |
| A3 与 910B3 差异 | 性能数据口径 | A3 实测并在报告注明，910B3 待补测 |
| 对角虚部/未引用三角误读 | 精度失败或 NaN 污染 | 搬入后对角虚部置 0、只用三角内长度、测试用 NaN 填充拦截 |

## 备选实现

1. **列扫 + 原子累加**（P3）：最贴近 Netlib，代码量最小；缺点是原子争用与 y 读写竞争，
   需两 phase（先 `y=beta*y` 再累加），在 n≥2048 时预计落后 `OWNER`/`FUSED`。
2. **按行 owner（P2）**：无归约、无原子，适合 n≤2048；大 n 需切 `FUSED`。
3. **FUSED 的两种归约**：独立 reduce kernel（简单）vs 同 kernel `SyncAll`（少一次 launch）；
   先用独立 kernel，按 profiling 决定是否合并。
4. **tile 尺寸**：R=64（UB 压力小、任务多）vs R=128（部分和少、归约轻）；按 n 分档。
