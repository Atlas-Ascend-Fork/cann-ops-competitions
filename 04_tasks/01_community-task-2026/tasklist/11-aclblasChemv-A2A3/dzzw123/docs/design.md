# aclblasChemv A2/A3 算子设计文档

| 项目 | 内容 |
|---|---|
| 任务 | 11月社区任务-aclblasChemv算子开发（A2/A3） |
| 提交人 | `dzzw123`（GitCode 账号，报名与提交一致） |
| 目标硬件 | Atlas A2 系列（性能/精度主验证：Atlas 800T A2 / 910B3）、Atlas A3 系列 |
| CANN 版本 | CANN 9.1.0 |
| 代码合入位置 | `ops-blas`：`blas/symv/arch22/`（实现）、`test/symv/chemv/arch22/`（测试）、`include/cann_ops_blas.h`（接口声明） |
| 设计依据 | 本任务书 `aclblasChemv_AtlasA2_A3_task_doc.md`、`test_cases/README.md`；`ops-blas` master（拉取于 2026-11-10，HEAD `5bd9514`） |

---

# 需求背景

## 需求来源

本设计对应「11月社区任务-aclblasChemv算子开发（A2/A3）」。任务要求在 Atlas A2/A3 上使用 Ascend C 实现单精度复数（complex64）Hermitian 矩阵-向量乘 `aclblasChemv`，补齐 ops-blas 仓 Blas 能力，验收通过后合入 `https://gitcode.com/cann/ops-blas`。

## 背景介绍

### 算子功能

Chemv 为 BLAS Level-2 算子，计算

```text
y = alpha * A * x + beta * y
```

其中 `alpha`、`beta`、`A`、`x`、`y` 均为 complex64；`A` 为 n×n **Hermitian（厄米特）** 矩阵，满足 `A = A^H`（共轭转置）。列主序地址为 `A[i + j*lda]`，仅 `uplo` 指定三角被引用，未引用三角由共轭对称隐含、**不得读取**；对角元素虚部假定为 0、不参与计算。

### 与仓内同族算子的关系

| 已有实现 | 语义 | 与本任务的关系 |
|---|---|---|
| `blas/symv/arch22/ssymv_*` | 实数对称 `y = alpha*A*x + beta*y`，`A = A^T` | **结构基线**：arch22 上同族算子的入口/校验/两阶段下发/分核方式，按本任务补齐 arch22 复数能力 |
| `blas/symv/arch35/csymv_*` | 复数对称，镜像不取共轭、对角虚部参与运算 | 复数标量（Host/Device）探测与本任务一致，可直接复用口径；**数学语义不同，不可改名复用** |
| `blas/dot/arch22/cdot_kernel.cpp` | 复数点积（conj 可选） | **复数算术基线**：arch22 上用 `GatherMask` 将交错实虚布局拆为 SoA 再做 FP32 复乘/归约的既有写法 |
| `blas/dgmm/arch22/cdgmm_kernel.cpp` | 复数逐元素缩放 | 同上，验证 `GatherMask` 拆分口径在 arch22 可用 |

`blas/symv/arch22/` 当前只有实数 `ssymv`（含 `ssymv_kernel.h`、`ssymv_tiling_data.h`），arch22 上尚无复数 symv 族实现；`test/symv/` 下 `ssymv`、`csymv` 的 `CMakeLists.txt` 都只在 `arch35` 分支注册 GTest，A2/A3 构建会退回非 GTest runner（输出整数毫秒）。本任务新增 arch22 复数实现，并按任务书要求补齐 `test/symv/chemv/CMakeLists.txt` 的 **arch22 GTest 注册**。

### 设计要点

Chemv 与 Ssymv 的差别只有两处，但都必须显式体现：

1. **共轭**：存储三角中除对角外的元素，在"沿列归约（点积）"方向需取共轭，在"沿列散射"方向按原值使用；Ssymv 两个方向都不取共轭。
2. **实数对角**：对角元素只取实部，虚部不参与乘法（不是"先参与乘法再丢弃"）。

复数算术在 arch22 上没有复数向量指令，必须按 FP32 实数运算展开，因此核心工程问题是如何在 UB 中低成本得到实部/虚部数组（SoA），并控制矩阵只被读取一遍。

---

# 需求分析

## 需求描述

新增共享公开接口（声明置于 `include/cann_ops_blas.h`，与其他产品线共用，禁止私有平行接口）：

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

| 参数 | 位置/类型 | 语义与约束 |
|---|---|---|
| handle | Host 句柄 | 携带已绑定 stream；空句柄返回 `HANDLE_IS_NULLPTR` |
| uplo | Host 枚举 | `ACLBLAS_UPPER` / `ACLBLAS_LOWER`；非法返回 `INVALID_ENUM` |
| n | Host int | `n >= 0`；`n = 0` 合法 no-op |
| alpha | Host/Device complex64，可空判定见 §2.5 | n>0 时不可为 nullptr |
| A | Device complex64，`lda×n` 列主序，只读 | 仅 uplo 三角被引用；对角虚部忽略 |
| lda | Host int | `lda >= max(1, n)` |
| x | Device complex64 | 逻辑长度 n，物理长度 `>= 1+(n-1)*|incx|` |
| incx | Host int | `!= 0`，支持负步长，拒绝 `INT32_MIN` |
| beta | Host/Device complex64 | n>0 时不可为 nullptr；`beta=(0,0)` 时 y 旧值不必有效 |
| y | Device complex64，原地覆写 | 逻辑长度 n；仅写逻辑位置，不写步长空隙 |
| incy | Host int | 同 incx |

## 需求拆解

1. 新增 `include/cann_ops_blas.h` 接口声明，Host 入口放入 `blas/symv/arch22/chemv_host.cpp`。
2. 实现按任务书 §2.5 顺序的参数校验、`n = 0` 与 `alpha=(0,0)&&beta=(1,0)` 短路、`alpha=(0,0)` 退化路径。
3. 实现 arch22 复数 kernel：Hermitian 共轭、实数对角、complex64 乘加、原地写回，支持 `lda` padding、正负步长、尾部非对齐与任意 n。
4. 控制矩阵读取量：存储三角每个元素 **只读一遍**，同时完成"该列的点积"与"镜像行的散射"两项贡献。
5. 建立 `test/symv/chemv/` 测试工程：CSV 驱动 GTest、`cblas_chemv`（Netlib CBLAS 3.10.0）golden、arch22 GTest 注册、精度与性能用例全集。
6. 精度按任务书 §3.2（实部/虚部分别 FLOAT32 档位 2^-13）；性能按 §3.3（`msprof op` 采样，≤1.25× GPU 标杆）。
7. 按仓库规范提供算子 README（产品支持表标注 A2/A3 支持）。

---

# 详细设计

## 算子分析

### 数学公式

记 `A[i + j*lda]` 为存储元素，`A_s[i,j]` 为 Hermitian 逻辑矩阵：

```text
A_s[i,j] = Re(A[i+i*lda])                    , i == j        （虚部忽略）
A_s[i,j] = A[i+j*lda]                        , (i<=j 且 UPPER) 或 (i>=j 且 LOWER)
A_s[i,j] = conj(A[j+i*lda])                  , 其余（镜像，仅由共轭对称隐含，不读取）

y[i] = alpha * sum_{j} A_s[i,j] * x[j] + beta * y[i]
```

复数乘法（FP32 实数运算展开）：

```text
(a+bi)(c+di) = (ac - bd) + (ad + bc)i
```

按存储三角第 `j` 列（UPPER 的有效行 `i ∈ [0, j]`，LOWER 为 `i ∈ [j, n)`）组织计算，一个存储元素 `A(i,j)` 同时产生两条贡献：

| 贡献方向 | 表达式 | 是否取共轭 | 目标 |
|---|---|---|---|
| 沿列归约（点积） | `conj(A(i,j)) * x[i]` | **是** | `y[j]` |
| 沿列散射（镜像） | `A(i,j) * x[j]` | 否 | `y[i]`，`i != j` |
| 对角 | `Re(A(j,j)) * x[j]` | 实数乘复数 | `y[j]` |

共轭方向由 Hermitian 定义决定：`y[j]` 需要 `A_s[j,i] = conj(A(i,j))`（`i<j` 时该位置未存储），而 `y[i]` 需要 `A_s[i,j] = A(i,j)`（`i<j` 时该位置即存储值）。以 2×2 校验：`A = [[2, 3+4i],[3-4i, 5]]`、UPPER 存储 `{2, 3+4i, 5}`，`y[0] = 2x0 + (3+4i)x1`（散射不取共轭）、`y[1] = (3-4i)x0 + 5x1`（点积取共轭）。

### 支持数据类型

仅 complex64：`aclblasComplex`（`real`、`imag` 各 float32，内存中实虚交错）。计算全程 FP32，不做低精度转换。alpha、beta 为复数标量，与 cuBLAS `cublasChemv` 语义一致。

### 支持形状

- `A` 为 `lda×n`，逻辑 `n×n`，列主序；`lda >= max(1, n)`，`lda` padding 列内尾部不参与计算。
- `x`、`y` 逻辑长度 n，物理长度 `1+(n-1)*|inc|`；负步长时逻辑首元素位于物理高地址端。
- 无 batch、无广播、无 view 语义；n 为运行时入参（不设 API 上限，测试性能最大 n=4096）。

---

## 算子实现

### 实现方案

沿用 arch22 同族句柄式直调结构：Host 侧在同一 `handle->stream` 上顺序下发少量 kernel，Kernel 侧用 AIV（`KERNEL_TYPE_AIV_ONLY`）完成计算。

复数在 AIV 上无原生指令，故在 UB 内把交错的复数列块用 `GatherMask`（偏移 1 取实部、偏移 2 取虚部）拆成 SoA 的 `Ar/Ai`、`Xr/Xi`，再用 FP32 向量指令完成复乘与归约——这与仓内 arch22 的 `cdot`/`cdgmm` 复数实现口径一致，不引入新的重排机制。

两条实现路径 + 一个退化路径：

| 路径 | 触发条件 | 计算组织 | workspace |
|---|---|---|---|
| **P1 列扫描 + UB 部分和**（主性能路径） | `incx == 1 && incy == 1` 且 `2*4*n` 字节的 UB 部分和放得下 | 每核按列轮转，存储三角只读一遍；核内维护私有 SoA 部分和 `ypR/ypI`（长度 n），末尾一次性原子并入 y | 无 |
| **P2 行 owner**（通用/兜底） | 其余情况（任意 incx/incy，或 n 超出 UB 部分和预算） | 每核独占若干输出行，按行块 `R×L` 计算完整输出，单写者、无原子 | 无 |
| **P3 仅缩放**（退化） | `alpha == (0,0)`（Host 读标量后判定） | 只做 `y = beta*y`；`beta == 1` 返回成功不下发，`beta == 0` 走清零 | 无 |

P1 的关键收益：写入 y 的流量为 `O(useCoreNum * n)` 而不是 `O(n²)`。若采用 Ssymv 的"两阶段 + 全量原子累加"，每个非对角元素都要对 y 做一次原子读改写（n=4096 时约 130 MB 额外读写，且为 8B 粒度的离散原子），在本算子上不可接受，因此 P1 用核内部分和把散射累加留在 UB。

#### Host 侧设计

**参数校验与短路顺序**（严格按任务书 §2.5）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n == 0` → `ACLBLAS_STATUS_SUCCESS`，**此后不再校验 uplo/lda/incx/incy/alpha/beta，不访问 A/x/y**。
4. `uplo` 非 UPPER/LOWER → `INVALID_ENUM`；`lda < max(1,n)`、`incx == 0`、`incy == 0`、`incx/incy == INT32_MIN`、`alpha == nullptr`、`beta == nullptr` → `INVALID_VALUE`。
5. 读取复数标量（见下），若 `alpha == (0,0) && beta == (1,0)` → `SUCCESS`（可不下发任何 kernel，A/x/y 允许全空）。
6. `alpha != (0,0)` 时要求 `A != nullptr && x != nullptr`；非第 5 步短路时要求 `y != nullptr`（`beta == (0,0)` 只免去读旧值，不豁免非空）。非法 → `INVALID_VALUE`。
7. 跨度守卫：以 `int64` 计算 `(n-1)*|incx|`、`(n-1)*|incy|`、`lda*n`，超过 `UINT32_MAX`（kernel 索引域）返回 `INVALID_VALUE`（对齐同族 `ssymv` arch22 的 issue #807 口径与 `INT32_MIN` 拒绝口径 #780）。

**复数标量读取**：复用 `blas/common/helper/host_utils.h` 的模板 `ReadScalarFromHostOrDevice<aclblasComplex>`（`CheckPtrLocation` 探测 Host/Device；Device 指针异步 D2H 后同步，失败返回 `INTERNAL_ERROR`）。任务书要求"读到标量后判定 alpha=(0,0)且beta=(1,0)"，因此 Host 在任何路径上都需要两个标量的数值，不存在"不读标量直接下发"的分支；这一点与 `csymv` arch35 仅在缺指针时才 D2H 的做法不同，此处按本任务书语义执行，并在注释中说明。

**分核**：`aivNum = GetAivCoreCount()`（查询失败返回 `EXECUTION_FAILED`）。P1 取

```text
useCoreNum = min(aivNum, max(1, n / kMinColPerCore))      // kMinColPerCore 初值 8，按实测调整
```

列按 `col = blockIdx; col < n; col += useCoreNum` 轮转分配：单列工作量正比于列内有效长度（UPPER 为 `j+1`，LOWER 为 `n-j`），轮转后各核总工作量偏差小于一列的代价，天然均衡，无需前缀和划分。

**下发序列（P1）**：

1. `beta != (1,0)` 时：`beta == (0,0)` 用 `aclrtMemsetAsync(y, 0, n*sizeof(aclblasComplex), stream)` 清零（省一次 kernel 启动）；否则下发 `chemv_scale_kernel`（多核按 chunk 做 `y = beta*y`，复数标量由 tiling 传入）。`beta == (1,0)` 跳过此步。
2. 下发 `chemv_kernel`（列扫描 + 原子并入）。

两次下发位于同一 stream，天然有序；即使 `beta == 0` 也保证清零先于原子累加，避免未初始化 y 参与累加。P2 为单次下发（在核内先算完整行再写，无需清零前置）。

**Tiling 字段**（`ChemvTilingData`）：

| 字段 | 说明 |
|---|---|
| `n`、`lda` | 逻辑阶数、前导维（uint32，已过守卫） |
| `uplo`、`path` | 上/下三角；P1/P2/P3 路径标识 |
| `alphaReal/alphaImag`、`betaReal/betaImag` | Host 读到的复数标量（P1 只用 alpha） |
| `useCoreNum`、`chunkLen` | 参与核数、单次搬运的复元素数 |
| `incx`、`incy` | `int64`，负步长下地址换算在 64 位域完成 |

#### Kernel 侧设计

**内存布局与拆分**。A 的列内元素在列主序下连续：第 `j` 列有效行 `[k0, k1)` 对应 GM 偏移 `(k0 + j*lda) * sizeof(aclblasComplex)`，一次 `DataCopyPad` 搬入 `fillLen = ceil((k1-k0)/32)*32` 个复元素（32 复元素 = 256B = 一个 GatherMask repeat；不足部分由 pad 补 0、且只按逻辑长度参与写回）。随后两次 `GatherMask` 拆出 `Ar`、`Ai`（`srcOffset = 1 / 2`，`repeatTimes = fillLen/32`），x 块同样拆出 `Xr`、`Xi`。

**点积贡献（取共轭，目标 `ypR[j]/ypI[j]`）**：对块内 `k`：

```text
re_chunk = ReduceSum( Ar*Xr + Ai*Xi )     // conj(A)*x 的实部
im_chunk = ReduceSum( Ai*Xr - Ar*Xi )     // conj(A)*x 的虚部
yp[j] += alpha * (re_chunk, im_chunk)     // 标量域复数乘，仅 1 个元素
```

`ReduceSum` 为树形归约；块间在标量域 FP32 累加。误差量级约 `(log2(chunkLen) + 块数) * eps * |y|`，与 `max(1e-2, 32*ULP)` 上限相比留有余量；若实测边界吃紧，块间累加改为 double（标量域，代价可忽略），不作为默认。

**散射贡献（不取共轭，目标 `ypR[k0..k1)`）**：令 `c = alpha * x[j]`（标量域复数，每列只算一次）：

```text
ypR[k] += Ar[k]*c.re - Ai[k]*c.im
ypI[k] += Ar[k]*c.im + Ai[k]*c.re
```

两条向量 `Mul` + `Add/Sub`，按有效长度用带 mask 的 `Add` 累加，**不得写到 `k1` 之外**（`ypR/ypI` 在 k>k1 处存放其它列的有效贡献）。块内 x 已经从 L2 命中读取，x 总长仅 `8n` 字节（n=4096 为 32 KB），重复读取不产生 HBM 流量。

**对角**：`k = j` 从向量块中排除，单独处理——读 `Re(A[j*lda+j])` 与 `x[j]`，`yp[j] += (alpha*x[j]) * Re(Ajj)`。避免把内存中的对角虚部先乘进来再丢弃。

**P1 主循环伪代码**（`blockIdx = b`，`S = useCoreNum`，UPPER 为例）：

```cpp
Duplicate(ypR, 0, n);  Duplicate(ypI, 0, n);            // 核内私有部分和
for (uint32_t j = b; j < n; j += S) {                   // 列轮转
    float cR = alphaR * xR[j] - alphaI * xI[j];         // c = alpha * x[j]
    float cI = alphaR * xI[j] + alphaI * xR[j];
    // 对角：只取实部
    float ajj = af[2 * (j * lda + j)];                  // af 为 A 的 float* 视图；imag 不读入计算
    ypR[j] += cR * ajj;  ypI[j] += cI * ajj;
    for (uint32_t k0 = 0; k0 < j; k0 += L) {            // UPPER 有效行 [0, j)
        uint32_t len = min(L, j - k0);
        CopyInA(a + (k0 + j * lda), len);               // DataCopyPad，pad=0
        CopyInX(k0, len);                               // x 块，L2 命中
        GatherMask(Ar, Ai, aUb);  GatherMask(Xr, Xi, xUb);
        // 点积：conj(A) * x，归约到 yp[j]
        dotR = ReduceSum(Ar * Xr + Ai * Xi);
        dotI = ReduceSum(Ai * Xr - Ar * Xi);
        ypR[j] += alphaR * dotR - alphaI * dotI;
        ypI[j] += alphaR * dotI + alphaI * dotR;
        // 散射：A * x[j]，结果按 len 掩码累加到 yp[k0..k1)（tail 用带 mask 的 Add）
        Add(ypR + k0, ypR + k0, Ar * cR - Ai * cI, mask = len);
        Add(ypI + k0, ypI + k0, Ar * cI + Ai * cR, mask = len);
    }
}
SetAtomicAdd<float>();
CopyOutInterleaved(y, ypR, ypI, n);   // 两条 4B 步长原子写，dstStride = 8
SetAtomicNone();
```

`LOWER` 仅把有效行区间换成 `(j, n)`、`k0` 从 `j+1` 起，其余一致。

**pad 与 `0*Inf` 防护**：pad 区域 A、x 均为 0（`DataCopyPadExtParams` 的 padding value 取 0），乘法结果为 0，`ReduceSum` 求和不受影响；散射方向靠 mask 限制写入范围。不允许对整块做"先算后屏蔽"的越界读写。



**写回 y（原子并入）**：`SetAtomicAdd<float>()` 后把 `ypR`、`ypI` 以 4B 步长写入交错 y（`DataCopyPad`，`blockCount = n`、`blockLen = 4`、`dstStride = 8`，两条），随后 `SetAtomicNone()`。写入量 `2 * useCoreNum * n` 个 float，n=4096、40 核约 1.3 MB，相对 67 MB 的矩阵读取可忽略。因多核累加顺序不定，结果不保证位级确定性（任务书 §2.5：确定性不作要求）。若该 4B 步长原子写回实测开销偏高，备选方案为每核把部分和写入 handle workspace（`useCoreNum * n` complex），再由一次归约 kernel 合并；该备选不改变精度口径，仅在 P1 内部替换最后一步。

**P2 行 owner**：每核持有 `R` 行输出累加器（R 由 UB 预算确定，初值 16），对行 `i` 的两部分输入分别读取——直接部分取存储三角中行 `i` 的列 `j ∈ [i, n)`（UPPER），按列块组织：一次搬入 `L` 个列段、每段长度 `R`（覆盖整个行块，段内连续），使行方向的跨列访问被摊薄到 R 行的复用上；镜像部分取列 `i` 的段 `j ∈ [0, i)`（连续一次搬完），按 `A(i,j) = conj(A(j,i))` 取共轭。每行的完整和先在 UB/标量域算完再写 y，**单写者、无原子、无跨核归约**，因此对任意 n 成立；代价是存储三角被读两遍（直接行 + 镜像列），只作为不通向 P1 的输入的兜底。`|inc| > 1` 时列内 x/y 物理位置按 `int64` 偏移逐元素访问（对齐 `ssymv` arch22 的通用路径写法）。

**流水与同步**：P1 用 `TQue` 双缓冲搬运 A 块与 x 块，`MTE2→V`、`V→MTE3` 用 `SetFlag/WaitFlag` 显式配对，1 个 compute 单元、无跨核同步。

**chunkLen 的选取**：Host 侧按 `UB 容量 ≥ ypR+ypI (8n B) + A/x 双缓冲 (2*8L + 2*8L B) + SoA 与临时 (6*4L B)` 反解，取 32 的倍数、上界 2048。n=4096 时（910B3 UB 192 KB）：`yp` 共 32 KB（实虚各 16 KB），余量可容纳 `L = 1024` 的双缓冲与临时（约 56 KB），仍有富余，可上调 `L`；n 过大导致 `8n` 超预算时由 Host 判定改走 P2。

**Kernel 函数与启动**：

```cpp
void chemv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace,
                     const ChemvTilingData& tiling, uint32_t numBlocks, void* stream);
void chemv_scale_kernel_do(uint8_t* y, const ChemvTilingData& tiling,
                           uint32_t numBlocks, void* stream);
```

`numBlocks = useCoreNum`，入口 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；`workSpace` 仅在备选归约方案下使用。

### 工程文件规划

| 位置 | 改动 |
|---|---|
| `include/cann_ops_blas.h` | 新增 `aclblasChemv` 声明（置于 `aclblasSsymv`/`aclblasCsymv` 邻近位置） |
| `blas/symv/arch22/chemv_host.cpp` | 入口、校验、标量读取、路径选择、tiling、两次下发 |
| `blas/symv/arch22/chemv_kernel.cpp` | P1 列扫描 kernel、P2 行 owner kernel、scale kernel |
| `blas/symv/arch22/chemv_kernel.h` | kernel 启动声明 |
| `blas/symv/arch22/chemv_tiling_data.h` | `ChemvTilingData`（Host/Kernel 共享） |
| `blas/symv/README.md` | 追加 `aclblasChemv` 接口说明（产品支持表标注 A2/A3：支持） |
| `test/symv/chemv/chemv_param.h` | CSV 列解析（`alpha_real/alpha_imag/beta_real/beta_imag`、`null` 指针标记） |
| `test/symv/chemv/chemv_golden.h` | 接入 `cblas_chemv` 生成 golden（不复用 `csymv_golden.h` 的对称算法） |
| `test/symv/chemv/arch22/chemv_test.cpp` | GTest 用例驱动、精度判定、`[PERF]` 打印 |
| `test/symv/chemv/arch22/chemv_npu_wrapper.h` | Device 缓冲与调用封装（含负向用例 dummy 缓冲） |
| `test/symv/chemv/arch22/chemv_test.csv` | 随包 1200 条用例 + 补充正态分布/边界用例 |
| `test/symv/chemv/CMakeLists.txt` | `if("arch22" IN_LIST SOC_ARCH_DIRS) ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test) else() ops_blas_add_tests(${OPS_BLAS}) endif()` |

`blas/CMakeLists.txt` 已按 `arch*` 目录自动收集源码、`test` 由 `ops_blas_add_gtest_tests` 自动发现 `*_test.cpp`，新增算子不需要改构建主干；需用 `bash build.sh --soc=ascend910b --ops=chemv` 实测确认（单算子过滤依赖 `TEST_NAMES`，本算子不复用 `gemm_kernel_do`，无需登记 `_GEMM_KERNEL_DEP_OPS`）。

### 支持硬件

| 产品 | 支持 | 验证环境 |
|---|---|---|
| Atlas A2 训练/推理系列（`--soc=ascend910b`） | 支持 | Atlas 800T A2（910B3），CANN 9.1.0：精度全量 + 性能 |
| Atlas A3 训练/推理系列（`--soc=ascend910_93`） | 支持 | Atlas A3，CANN 9.1.0：精度全量（性能以 A2 为准） |
| Ascend 950PR/950DT（arch35） | 不支持（本任务范围外） | 接口声明共享，arch35 无实现时按仓内既有符号策略处理 |

### 算子约束限制

- 仅 complex64；不提供 batch、广播、视图语义；A/x 只读，y 原地覆写。
- 只读 uplo 指定三角；对角只取实部。A/x/y 之间允许重叠不作承诺（任务书未要求）。
- `n = 0` 与 `alpha=(0,0)&&beta=(1,0)` 直接返回成功，不访问 A/x/y；`alpha=(0,0)` 时不读 A/x。
- 不保证位级确定性；异步执行，调用方读回结果前需同步 stream。
- 返回值：

| 场景 | 返回码 |
|---|---|
| handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| n/lda/inc 非法、跨度溢出、必要指针为空、标量地址查询失败 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 标量 D2H / 同步失败 | `ACLBLAS_STATUS_INTERNAL_ERROR` |
| 获取 AIV 核数失败、kernel 启动失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 合法调用完成短路或提交计算 | `ACLBLAS_STATUS_SUCCESS` |

---

# 可维可测分析

## 精度标准/性能标准

### 精度口径

| 项 | 内容 |
|---|---|
| golden | `cblas_chemv`（Netlib CBLAS 3.10.0，`CblasColMajor`），随测试工程链接 refblas；对角仅取实部、镜像按共轭补全 |
| 比对范围 | 输出 y 全向量，实部/虚部分别统计（不合并稀释） |
| 阈值 | `rtol = atol = 2^-13`（FLOAT32 档位），`required_matched_ratio = 0.99`，`max_abs_error ≤ max(1e-2, 32×ULP)` |
| Inf/NaN | 同为 NaN、同为同号 Inf、NaN 与 ±Inf 视为等效；有限 vs 非有限判失败；非有限元素不计入 ratio/最大误差，单独统计且须全部通过 |
| 统计精度 | 误差与 ULP 用 FP64 计算；实虚分别建掩码，逐用例记录 matched_ratio、max_abs_error、非有限计数 |

用例生成输入全部按 Hermitian 约定构造：未引用三角填**独立随机值（含 NaN 污染专项）**，用于拦截"误读未引用三角"；对角虚部填非零/NaN，用于拦截"对角虚部参与运算"。

### 测试工程与用例

`test/symv/chemv/arch22/chemv_test.csv` 直接加载随包 1200 条（1000 精度 + 200 性能，`TC_PF` 前缀），并按任务书 §3.5 补齐：

| 补充项 | 做法 |
|---|---|
| 正态分布用例（`GAUSS_NORM_5_2`） | 随包生成器只产 `RANDOM_NORM_5_5`；本任务在 CSV 中按 `a/x/y` 分别替换为 `GAUSS_NORM_5_2` 补充用例（`fill.h` 已支持该 token，μ∈[-5,5]、σ∈[0.1,2]），覆盖 UPPER/LOWER × 小中大尺寸，报告写明条数与补齐方式 |
| 校验顺序专项 | 空 handle、`n=0` 叠加全非法参数（期望 SUCCESS）、`alpha=(0,0)&&beta=(1,0)` 且 A/x/y 全空、`alpha=(0,0)` 且 A/x 为空、`beta=(0,0)` 且 y 为 NaN、`beta=(0,0)` 且 y 为空（期望 INVALID_VALUE） |
| 负向用例缓冲 | 期望非 SUCCESS 的用例（含 `incx/incy = 0`、`INT32_MIN`、非法 lda/uplo/负 n）一律用单元素 dummy 缓冲或 `NULLPTR`，**不按 `(n-1)*|inc|` 预分配**，由接口在读数前返回错误码判定通过 |
| 语义专项 | 未引用三角 NaN 污染、对角虚部 NaN、`incx=-1`（物理逆序）、`lda = n+1/n+8/n+16`、y 步长空隙哨兵（只写逻辑位置） |
| 负步长 | `incx/incy ∈ {±1,±2,±3}` 由随包 `TC_INC` 覆盖（72 条），走 P2 路径 |

### 性能口径

- 采样：`msprof op`，自带 5 次 warmup，之后连续 ≥10 次有效样本取算术平均（`OpBasicInfo.csv` 的 `Task Duration(us)`）；API 返回 SUCCESS 的样本才计入。
- 同时 GTest 内用 `aclrtEvent` 夹连续 10~100 次下发打印 `[PERF] case=<name> ... avg_us=<数值>`，供 `verify_performance.py` 与 `gpu_baseline.csv` 关联判定（该口径含下发开销、不含 H2D/D2H、造数与 golden 比对）。两套数据同时保留：脚本口径用于判定，`msprof` 用于交叉验证与瓶颈定位。
- 判据：单 case 平均耗时 ≤ 1.25 × GPU 标杆。四条典型用例预算（由 `gpu_baseline.csv` 换算，仅预算、非实测）：

| id | n / uplo | GPU（us） | 上限（us） | 存储三角读取量 |
|---|---|---:|---:|---:|
| chemv-base-000 | 512 UPPER | 14.64 | 18.30 | 1.05 MB |
| chemv-base-001 | 1024 UPPER | 21.11 | 26.39 | 4.20 MB |
| chemv-base-002 | 2048 LOWER | 30.09 | 37.61 | 16.8 MB |
| chemv-base-003 | 4096 LOWER | 65.30 | 81.62 | 67.1 MB |

- 设计预算判断：P1 的 HBM 读取量即为存储三角体量（n=4096 为 67.1 MB），在 A2 有效带宽下约 45~60 us，叠加两次下发与 1.3 MB 原子写回仍在 81.6 us 内；n=512 时读取几乎为 0，主要由两次 kernel 下发开销决定（`beta=(0,0)` 走 memset 省一次启动），需实测确认落在 18.3 us 内。小尺寸性能用例（如 n=1）以启动开销为瓶颈，将单独记录并说明。

### 验收命令

```bash
# 构建 + 跑 GTest（A2）
bash build.sh --soc=ascend910b --ops=chemv --run --device=0
# 仅精度（排除性能用例）；A3 用 --soc=ascend910_93
build/test/symv/chemv/chemv_test --gtest_filter='-*TC_PF*'
# 性能采集
msprof op --application="./build/test/symv/chemv/chemv_test --gtest_filter=*TC_PF*" --output=./prof_chemv
```

自测报告按任务书 §4 模板（`task_submission` 目录 7 件套）整理，包含精度日志（实部/虚部截图）、性能日志与 `msprof` 原始结果、环境与 commit 号。

## 兼容性分析

- 只新增接口与新增文件，不改动 `aclblasSsymv`、`aclblasCsymv` 等既有实现与公共类型布局；`aclblasComplex` 沿用 `include/cann_ops_blas_common.h` 定义。
- 共享头文件新增声明不等于所有架构已实现：README 产品支持表明确 A2/A3 支持、其他产品线按仓内既有"未实现架构"策略处理；构建只在 `arch22` 目录收集源码，不影响 arch35 构建。
- 新增 `test/symv/chemv/CMakeLists.txt` 的 arch22 GTest 注册是**本任务附带修复**：同族目录当前仅 arch35 分支注册 GTest，A2/A3 会退回非 GTest runner（输出整数毫秒），与本算子的 CSV 用例 + 性能脚本口径不兼容；该修改只影响 chemv 自己的测试目录，不改变 `ssymv`/`csymv` 现有行为。

## 风险与待确认事项

| 编号 | 风险/问题 | 应对 |
|---|---|---|
| R1 | `GatherMask` 拆分吞吐在 n=4096 时可能成为瓶颈（每 chunk 2 次 V 指令） | 已测项：`cdot`/`cdgmm` arch22 使用同一写法；若实测受限，改用"交错 + mask + `PairReduceSum`"等价写法（不重排，指令数相近），两版都保留在 kernel 内以编译期开关切换，按实测择优 |
| R2 | 末尾 4B 步长原子写回的开销 | 备选：部分和写入 handle workspace + 归约 kernel，接口语义不变 |
| R3 | 大 n 时 UB 容纳不下部分和 | Host 判定改走 P2（无 UB 部分和依赖，单写者） |
| R4 | 精度边界（n≥2048 抵消场景）贴近 `32×ULP` | 块间累加可切 double（标量域）；先按 FP32 实测，失败样本单独分析 |
| R5 | 任务书 §3.3（`msprof`，≥10 样本）与 `test_cases/README.md`（`[PERF]` 脚本口径）采样方式不同 | 两套数据同时提交：脚本口径用于判定，`msprof` 用于交叉验证；若审核方要求以 `msprof` 为唯一判据，按 `msprof` 重算均值 |

## 参考

1. 任务书：`aclblasChemv_AtlasA2_A3_task_doc.md`；测试材料：`test_cases/README.md`、`chemv_test.csv`、`gpu_baseline.csv`、`gen_csv.py`。
2. ops-blas（HEAD `5bd9514`）：`blas/symv/arch22/ssymv_{host,kernel}.cpp`、`blas/symv/arch35/csymv_{host,kernel}.cpp`、`blas/dot/arch22/cdot_kernel.cpp`、`blas/dgmm/arch22/cdgmm_kernel.cpp`、`blas/common/helper/{host_utils.h,aclblas_handle_internal.h}`、`test/frame/{fill.h,csv_loader.h,mixed_tolerance.h}`、`cmake/test.cmake`。
3. [Netlib CHEMV 参考实现](https://www.netlib.org/blas/chemv.f)、[cuBLAS HEMV 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hemv)。
4. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
