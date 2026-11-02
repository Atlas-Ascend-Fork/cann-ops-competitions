# aclblasCsyrkx 算子设计文档（Atlas 950PR）

# 需求背景（required）

## 需求来源

- 任务：**10 月社区任务-aclblasCsyrkx 算子开发（950）**，任务书 `aclblasCsyrkx_Atlas950PR_task_doc.md`（社区任务 IT 系统发布）。
- 目标仓与合入目录：[cann/ops-blas](https://gitcode.com/cann/ops-blas) `blas/syrkx/arch35/`（算子代码）与 `test/syrkx/`（测试代码），任务书 §5。
- 验收流程：设计文档以 MR 形式提交至 [cann/cann-ops-competitions](https://gitcode.com/cann/cann-ops-competitions) `04_tasks/01_community-task-2026/tasklist/`，评审合入后进入开发验收（任务书 §4）。
- 软硬件基线：Atlas 950PR（arch35 / DAV_3510），CANN 9.1.0；精度 golden 由 Netlib BLAS 3.10.0 复数实现生成；性能标杆为 cuBLAS `cublasCsyrkx`（NVIDIA 驱动 550.104.05 / CUDA 12.4，P102 实测数据随任务书提供 `test_cases/gpu_baseline.csv`）。

## 背景介绍

### 算子功能定位

`aclblasCsyrkx` 实现单精度复数（COMPLEX64）**扩展对称秩 k 更新**：以两个可不同的输入矩阵 A、B 计算乘积并按 uplo 指定三角更新 n×n 对称矩阵 C：

```
C = alpha * op(A) * op(B)^T + beta * C
```

当 A = B 时退化为 syrk；当 B = A·D（D 对角）时结果仍保持对称。A ≠ B 的一般情形下 API 契约仅保证写回 uplo 指定三角（与 cuBLAS `cublasCsyrkx` 一致）。本算子补齐 ops-blas 仓 syrkx 族的复数能力：仓内已有 `aclblasSsyrkx`（实数，`blas/syrkx/arch35/ssyrkx_*`）与复数 Hermitian 变体 `aclblasCherkx`（`blas/herkx/arch35/cherkx_*`），缺对标 cuBLAS `cublasCsyrkx` 的复数对称（非共轭）版本。

### 业界对标

| 项 | cuBLAS `cublasCsyrkx`（legacy） | 本算子 aclblasCsyrkx |
| --- | --- | --- |
| 参数序列 | handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc | 完全一致（任务书 §2.3） |
| trans 语义 | OP_N / OP_T / OP_C，其中 **OP_C 按 OP_T 执行**（结果矩阵为对称矩阵，共轭转置会破坏对称性；本任务已在 P102 + CUDA 12.5 实测 trans=T 与 trans=C 输出逐位相同） | 同口径：OP_C ≡ OP_T，不取共轭 |
| 标量 | alpha/beta 均为 cuComplex 指针（device） | 均为 `aclblasComplex` 指针（支持 Host 或 Device，二者须同模式） |
| 对角虚部 | 不做 Hermitian 强制（对称非 Hermitian） | 同口径：对角虚部保留计算值，不清零 |
| 返回码 | cublasStatus_t | `aclblasStatus_t`（`cann_ops_blas_common.h` 语义） |

与 Netlib BLAS 对标：Netlib 无 csyrkx 标准 entry；golden 按任务书 §3.2 由 `cblas_cgemm` 组合生成（trans=N：`NoTrans×Trans`；trans=T/C：`Trans×NoTrans`），再将非 uplo 三角恢复为调用前原值。

### 工程形态

- **Kernel 直调句柄式 BLAS 接口**：Host C API + Ascend C kernel 直调，经 `aclblasHandle_t` 绑定 stream 下发，无 aclnn 两段式、无图模式（任务书 §2.2）。
- 代码落位：`blas/syrkx/arch35/` 四件套（`csyrkx_host.cpp` / `csyrkx_kernel.cpp` / `csyrkx_kernel.h` / `csyrkx_tiling_data.h`），并入既有 syrkx 目录，不新建族目录。
- **仓内近亲与复用方式**（已现场核验 master）：
  - 主近亲 `blas/herkx/arch35/cherkx` 五件套：复数双输入三段式（AIV 解交错 → 4× 实 GEMM → AIV 合并），本算子沿用其总体架构、tiling 骨架与同步护栏，仅改组合符号、复数 beta 与去掉 Hermitian 对角清零；
  - 次近亲 `blas/syrkx/arch35/ssyrkx`：同族实数版，沿用参数校验顺序与 no-op 口径（校验先于 quick return）；
  - 复数 beta 交叉缩放公式参照已合入 `blas/syr2k/arch35/csyr2k`；
  - Phase 1 直接复用 `blas/gemm/arch35/gemm_kernel.cpp` 的 `gemm_kernel_do` + `GemmTilingData`，不新写 GEMM kernel。

### 接口规格

ops-blas 统一头 `include/cann_ops_blas.h` 新增声明（任务书 §2.3/§7.3 明确要求新增，禁止 950PR 私有平行 API）：

```cpp
aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,     // ACLBLAS_UPPER / ACLBLAS_LOWER
    aclblasOperation_t trans,   // ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C（OP_C ≡ OP_T）
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

参数规格（对齐任务书 §2.4，含非法值行为）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 算子句柄，绑定 stream 直调 NPU kernel | aclblasHandle_t | - | 非空；空指针返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR | - |
| uplo | C 的三角引用方式 | aclblasFillMode_t | - | 仅 ACLBLAS_UPPER / ACLBLAS_LOWER；其他值返回 ACLBLAS_STATUS_INVALID_VALUE | - |
| trans | A、B 运算方式 | aclblasOperation_t | - | 仅 OP_N / OP_T / OP_C；OP_C 与 OP_T 完全等价（不取共轭）；其他值返回 ACLBLAS_STATUS_INVALID_VALUE | - |
| n | C 的阶数，亦为 op(A) 行数 | int | - | ≥ 0；负值返回 ACLBLAS_STATUS_INVALID_VALUE | 标量 |
| k | 缩并维 | int | - | ≥ 0；负值返回 ACLBLAS_STATUS_INVALID_VALUE | 标量 |
| alpha | 复数标量乘子（指针） | const aclblasComplex* | COMPLEX64 | 非空，空指针返回 ACLBLAS_STATUS_INVALID_VALUE；Host/Device 均可（与 beta 同模式） | 标量 |
| A | 输入矩阵（列主序） | const aclblasComplex* | COMPLEX64 | n>0 且 k>0 时非空；lda ≥ max(1, trans=N ? n : k) | trans=N：n×k；trans=T/C：k×n |
| lda | A 前导维 | int | - | ≥ max(1, trans=N ? n : k)，否则 ACLBLAS_STATUS_INVALID_VALUE | 标量 |
| B | 输入矩阵（列主序） | const aclblasComplex* | COMPLEX64 | n>0 且 k>0 时非空；ldb ≥ max(1, trans=N ? n : k) | 同 A |
| ldb | B 前导维 | int | - | ≥ max(1, trans=N ? n : k)，否则 ACLBLAS_STATUS_INVALID_VALUE | 标量 |
| beta | 复数标量乘子（指针），作用于原 C | const aclblasComplex* | COMPLEX64 | 非空，空指针返回 ACLBLAS_STATUS_INVALID_VALUE；与 alpha 同指针模式 | 标量 |
| C | 输入/输出对称矩阵（列主序），仅写 uplo 三角 | aclblasComplex* | COMPLEX64 | n>0 时非空；ldc ≥ max(1, n)，否则 ACLBLAS_STATUS_INVALID_VALUE | n×n |
| ldc | C 前导维 | int | - | ≥ max(1, n) | 标量 |

### 数学公式

设 `A = Ar + i·Ai`、`B = Br + i·Bi`（Ar 等为实部/虚部实矩阵），alpha/beta 为复数标量，矩阵均列主序：

```
trans = OP_N：C = alpha · A · B^T  + beta · C      （A、B 为 n×k）
trans = OP_T：C = alpha · A^T · B  + beta · C      （A、B 为 k×n）
trans = OP_C：与 OP_T 逐位相同（转置、不取共轭）
仅对 uplo 指定三角读写；另一三角不引用、不写回
```

复数乘积的 4 实矩阵分解（kernel 设计基础，两种 trans 取值下符号一致）：

```
op(A)·op(B)^T = P_r + i·P_i
  P_r = t1 − t2        t1 = Ar·Br^T（或 Ar^T·Br）   t2 = Ai·Bi^T（或 Ai^T·Bi）
  P_i = t3 + t4        t3 = Ai·Br^T（或 Ar^T·Bi）   t4 = Ar·Bi^T（或 Ai^T·Br）
```

与 Hermitian 变体 cherkx（P_r = t1+t2、P_i = t3−t4，因共轭）的差异即在此两处符号；t1..t4 的四次实 GEMM 调用与 cherkx 完全相同，仅 Phase 2 合并符号不同。

alpha、beta 均为复数时的最终合并式（COMPLEX64 按 FLOAT32 分量展开）：

```
C_r = αr·P_r − αi·P_i + βr·C_or − βi·C_oi
C_i = αr·P_i + αi·P_r + βr·C_oi + βi·C_or        （C_or/C_oi 为 C 的旧值分量）
```

# 需求分析（required）

## 需求描述

在 Atlas 950PR（arch35 / DAV_3510，CANN 9.1.0）上以 **Ascend C kernel 直调**实现 `aclblasCsyrkx`：COMPLEX64 输入、uplo(2) × trans(3, 其中 C≡T) 全组合、n/k ≥ 0 含全部退化分支；精度满足生态开源混合容差标准（FLOAT32 分量 rtol=atol=2^-13、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32·ULP）；性能达到单 case ≥ 0.4× GPU、全部 case 平均 ≥ 0.5× GPU（cuBLAS `cublasCsyrkx` 标杆）；同输入多次执行逐位一致（确定性）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
| --- | --- | --- |
| 1 | 数学定义与可白嫖的代数量 | 对称（非 Hermitian）秩 k 扩展更新；**4M 实 GEMM 分解** P_r=t1−t2、P_i=t3+t4，四次实 GEMM 与已合入 cherkx 的调用序列完全相同（cherkx 因共轭取 t1+t2 / t3−t4，仅合并符号不同）；alpha/beta 双复数交叉缩放公式同 csyr2k |
| 2 | 接口原型 | 任务书 §2.3 逐字；`include/cann_ops_blas.h` 新增声明（该头现仅有 aclblasSsyrkx，无 Csyrkx，已现场核验） |
| 3 | 参数规格 | §2.4 表逐行落地；非法枚举/负维/非法前导维/空指针 → ACLBLAS_STATUS_INVALID_VALUE（handle 空 → HANDLE_IS_NULLPTR） |
| 4 | 支持数据类型 | 仅 COMPLEX64（A/B/C/alpha/beta），kernel 内部拆分为 FLOAT32 实平面计算 |
| 5 | 目标芯片与 arch 映射 | Atlas 950PR → arch35 / SOC=ascend950（DAV_3510）；A2/A3（arch22）不在本任务范围 |
| 6 | 精度要求（可溯源） | 任务书 §3.2：FLOAT32 分量 rtol=atol=1.2e-4、ratio 0.99、max_abs 1e-2 或 32·ULP；Inf/NaN 按 §3.2-4 等效规则；COMPLEX64 实/虚部分别比对 |
| 7 | 性能预期（可溯源） | 任务书 §3.3：msprof op 采样、warmup 5 + 有效 10 次均值；单 case ≤ 2.5×GPU 且平均倍率 ≥ 0.5×，双指标同时满足 |
| 8 | 代码架构选型 + 可行性 | 三段式：AIV 解交错（×2）→ AIC 4× 实 GEMM（复用 gemm_kernel_do）→ AIV 合并（复数 alpha/beta + uplo 三角写回）。**可行性已验证**：同构 cherkx 已合入 arch35 且本机单算子构建通过（`build.sh --ops=cherkx --soc=ascend950` 成功产出 cherkx_test）；GEMM 主导阶段与 cherkx 计算量一致。已排除的死路：直接复数 Cube MAC（arch35 FP32 Cube 不支持复数操作数，仓内所有复数 L3 均走实平面分解）；ssyrkx 的单 GEMM+scale 两段式（依赖 fixpipe 转置直读，无复数共轭/符号合并的落点） |
| 9 | 评测来源 | 精度：Netlib BLAS 3.10.0 `cblas_cgemm` golden（任务书随附测试工程）；性能：gpu_baseline.csv（P102 + CUDA 12.4 实测） |

# 详细设计（required）

## 算子分析

### 数学公式

见「接口规格 → 数学公式」：`C = alpha·op(A)·op(B)^T + beta·C`，4M 分解 `P_r = t1 − t2`、`P_i = t3 + t4`，合并式含复数 alpha 与复数 beta 的完整交叉缩放。

退化行为（任务书 §2.1-4 / §2.5）：

- `n = 0`：quick return SUCCESS，不写 C（参数校验已先行完成）；
- `k = 0`：跳过解交错与 GEMM，仅对 uplo 三角执行 `C = beta × C`（beta 为复数零时 uplo 三角置零；beta=1 时数值不变），不引用 A/B，另一三角不动。

### 支持数据类型

| 数据 | 类型 |
| --- | --- |
| A / B / C | aclblasComplex（COMPLEX64 = float 实部 + float 虚部，交错存储，实部在前） |
| alpha / beta | aclblasComplex 标量（指针传入，Host 或 Device，二者须同为一种模式） |
| kernel 内部计算 | FLOAT32 实平面（Ar/Ai/Br/Bi 与 t1..t4） |

### 支持形状

| trans | A、B 形状（列主序） | lda/ldb 下限 | C 形状 |
| --- | --- | --- | --- |
| OP_N | n×k | max(1, n) | n×n 对称 |
| OP_T / OP_C | k×n | max(1, k) | n×n 对称 |

n、k 为编译期已知静态值（不要求动态 shape）；n=0 / k=0 为合法退化；ldc ≥ max(1, n) 支持前导维 padding（测试覆盖 ld > min 的非对齐值）。

## 算子实现

### 实现方案

#### 总体架构与数据流

沿用仓内复数 L3 已验证的「实平面分解」三段式（cherkx 同构），单 stream 上按序下发 7 个 kernel：

```
            ┌─────────────────────────── Phase 0：解交错（AIV ×2） ───────────────────────────┐
 GM: A(复) ──▶ csyrkx_deinterleave_kernel ──▶ GM: Ar, Ai        （列主序复矩阵 → 两个紧致实矩阵）
 GM: B(复) ──▶ csyrkx_deinterleave_kernel ──▶ GM: Br, Bi

            ┌─────────────────────────── Phase 1：4× 实 GEMM（AIC ×4） ───────────────────────┐
 GM: Br,Ar ──▶ gemm_kernel_do ──▶ GM: t1 = Ar·Br^T   (trans=N) / Ar^T·Br   (trans=T/C)
 GM: Bi,Ai ──▶ gemm_kernel_do ──▶ GM: t2 = Ai·Bi^T   (trans=N) / Ai^T·Bi   (trans=T/C)
 GM: Br,Ai ──▶ gemm_kernel_do ──▶ GM: t3 = Ai·Br^T   (trans=N) / Ar^T·Bi   (trans=T/C)
 GM: Bi,Ar ──▶ gemm_kernel_do ──▶ GM: t4 = Ar·Bi^T   (trans=N) / Ai^T·Br   (trans=T/C)
              （复用 blas/gemm/arch35/gemm_kernel.cpp，GemmTilingData 驱动，多核分块）

            ┌─────────────────────────── Phase 2：合并缩放（AIV ×1） ─────────────────────────┐
 GM: t1..t4 ─▶ csyrkx_combine_kernel ─▶ GM: C（仅 uplo 三角）
    P_r = t1 − t2；P_i = t3 + t4
    C_r = αr·P_r − αi·P_i + βr·C_or − βi·C_oi
    C_i = αr·P_i + αi·P_r + βr·C_oi + βi·C_or
    对角虚部保留（对称非 Hermitian，不做 Hermitian 清零）
```

kernel 变体划分（无独立变体文件，按 tiling 标量分支）：

| 分支 | 触发条件 | 职责 |
| --- | --- | --- |
| 完整路径 | alpha≠0 且 k>0 | Phase 0+1+2 全执行 |
| skipTemp | alpha==0（复数零）或 k==0 | 跳过 Phase 0/1，Phase 2 直接 C = beta×C（uplo 三角） |
| beta==0 | βr==0 且 βi==0（复数零） | Phase 2 不读 C 旧值缩放；uplo 三角直接写新值（含全零） |
| skipTemp && beta==0 | 上述叠加 | uplo 三角写 (0,0)；非 uplo 段按 C 旧值恒等回写（逐位不变） |

数据流（单核内，Phase 0/2 为 AIV，Phase 1 为 AIC）：GM →(MTE2, te::Copy 2D strided)→ UB（64×64 块，双缓冲）→(V 管, Transform/DeInterleave/Interleave)→ UB →(MTE3, te::Copy)→ GM。Phase 1 内部为 gemm_kernel 既有 GM→L1→L0A/L0B→L0C(fixpipe)→GM 流水，本算子不改动。

#### host 侧设计

**校验顺序与异常码**（对齐任务书 §2.4；「校验先于 no-op 判定」§2.5）：

```
1. handle == nullptr            → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. uplo ∉ {UPPER, LOWER}        → ACLBLAS_STATUS_INVALID_VALUE
3. trans ∉ {OP_N, OP_T, OP_C}   → ACLBLAS_STATUS_INVALID_VALUE
4. n < 0 或 k < 0               → ACLBLAS_STATUS_INVALID_VALUE
5. lda/ldb < max(1, trans==N ? n : k)、ldc < max(1, n) → ACLBLAS_STATUS_INVALID_VALUE
6. alpha == nullptr / beta == nullptr → ACLBLAS_STATUS_INVALID_VALUE
7. A/B 空指针 且 n>0 且 k>0     → ACLBLAS_STATUS_INVALID_VALUE
8. C 空指针 且 n>0              → ACLBLAS_STATUS_INVALID_VALUE
（全部通过后）9. n == 0 → 直接 SUCCESS（quick return，不写 C）
```

与已合入 cherkx 的口径差异（有意，以任务书为准）：非法枚举返回 INVALID_VALUE（cherkx 返回 INVALID_ENUM）；全量校验先于 n==0 quick return（cherkx 在 n==0 时先返回）；A/B/C 空指针为无条件口径（不依赖 alpha/beta 真值分支）。该口径与同族实数版 ssyrkx 一致，属于族内已文档化的差异。

**标量解析**（三步式，沿用仓内模式）：`CheckPtrLocation` 判定 alpha/beta 指针位置 → 二者须同为 Host 或同为 Device（混合返回 INVALID_VALUE）→ Host 直接解引用；Device 走两次 `aclrtMemcpyAsync` D2H + 一次 `aclrtSynchronizeStream` 读回真值。alpha 与 beta 均为 aclblasComplex（8B），读回尺寸一致。复数零判定：`isAlphaZero = (αr==0 && αi==0)`、`isBetaZero = (βr==0 && βi==0)`（纯虚数不算零）。

**trans 归一**：`trans == OP_C → OP_T`（不取共轭，任务书 §2.1-2 实测口径），后续仅按 N / T 两分支。

**分核策略**：
- Phase 0（AIV）：按矩阵物理行均分 `rowsPerCore = CeilDiv(rows, aivCoreNum)`，各核独立处理自己行段内的 64×64 块，写不冲突；
- Phase 1（AIC）：复用 cherkx 的 `CalcMultiCorePartition`——枚举 mBlocks×nBlocks 组合取利用率最大的划分（≤ cubeCoreNum），输出 usedCoreNum；
- Phase 2（AIV）：按 C 的行均分 `rowsPerCore = CeilDiv(n, usedAivCoreNum)`，核内只遍历 uplo 三角覆盖的 64×64 块（UPPER 从对角起、LOWER 到对角止），完全落在三角外的块直接跳过。

**数据分块与内存优化**：Phase 0/2 统一 64×64 块（SCALE_BLOCK=64，32B 对齐 ELEMENTS_PER_BLOCK=8）；UB 内 t1..t4 各 64×64 float、cIn/cOut 各 64×64 复数（交错），队列深度 2 双缓冲使 MTE 与 V 管重叠。Phase 1 tiling 完全由 GemmTilingData 承载（baseM=32/baseN=16/baseK=8、tileKChunk=64、GEMM_FRACTAL=16 对齐 tempLdc）。

**workspace 规划**（经 `EnsureDefaultWorkspace` 申请，512B 对齐）：

| 区域 | 尺寸 | 说明 |
| --- | --- | --- |
| t1 / t2 / t3 / t4 | tempLdc × n × 4B × 4 | tempLdc = CeilAlign(n, 16)；4 次实 GEMM 结果 |
| Ar / Ai / Br / Bi | physRows × physCols × 4B × 4 | physRows = trans==N ? n : k；解交错输出，紧致存放（stride = physRows） |
| 超限防护 | 总需求 > 2 GiB（ACLBLAS_MAX_WORKSPACE_SIZE） | 单区预乘护栏（任一区 > 2GiB/4 即拒绝），返回 ACLBLAS_STATUS_ALLOC_FAILED |
| skipTemp 时 | 不申请/不使用 | 指针保持 null，kernel 侧对应路径不读 |

**tilingkey 规划**（tiling 标量分支表，kernel 以 by-value 结构体接收）：

| tiling 字段 | 取值 | kernel 分支 |
| --- | --- | --- |
| isAlphaZero / isKZero | 0/1 | skipTemp：跳过 t1..t4 读取，P 项按 0 处理 |
| isBetaZero | 0/1 | beta 缩放跳过读 C 旧值（乘法路径退化） |
| alphaImag / betaImag | 0 / ≠0 | 实数标量快路径（指令序列与实数版逐位一致）/ 复数交叉缩放全路径 |
| uploMode | UPPER/LOWER | 三角写回掩码（列截断 + LOWER 对角列恢复） |
| isTransN | 0/1 | Phase 0 物理形状与 Phase 1 GEMM transa/transb 组合 |

#### kernel 侧设计

**Phase 0 `csyrkx_deinterleave_kernel`（AIV，与 cherkx 同构）**：按核行段循环 64×64 块：`te::Copy(GM→UB)` 交错复块（ubCplxStride = 2×对齐行数）→ `DeInterleave` 拆实/虚（V 管）→ `te::Copy(UB→GM)` 双平面紧致存放。同步护栏沿用 cherkx 实测修复：V→MTE2（load 复用 aInBuf 前等待上一块 V 读完）、V→MTE3（store 读取 ar/ai 输出前等待本块 V 写完）、MTE3→V（下一块 V 重写输出 buf 前等待 store 读完）三对 SetFlag/WaitFlag 有向握手。

**Phase 1（AIC，直接调用 `gemm_kernel_do`）**：四次调用入参为 (Br,Ar→t1)、(Bi,Ai→t2)、(Br,Ai→t3)、(Bi,Ar→t4)（trans=N）；trans=T/C 时 t3/t4 换为 (Bi,Ar→t3)、(Br,Ai→t4)。GemmTilingData 固定 alpha=1/beta=0（无后处理），列主序交换（ApplyColMajorSwap：m↔n、lda↔ldb、isTransA↔isTransB）后多核分块。四次 GEMM 与 cherkx 完全一致——cherkx 的共轭差异全部消化在 Phase 2 符号里。

**Phase 2 `csyrkx_combine_kernel`（AIV，cherkx 改造型，核心差异所在）**：按核行段遍历 uplo 三角 64×64 块：

```
LoadGemmResults: t1..t4 四平面 GM→UB（2D strided te::Copy）      [skipTemp 时跳过]
ComputeP:        t1 = t1 − t2 (Sub)  → P_r
                 t3 = t3 + t4 (Add)  → P_i                         ← 与 cherkx 符号相反的两处
ApplyAlpha:      αi==0 快路径：t1 *= αr；t3 *= αr（指令序列与实数 alpha 逐位一致）
                 否则交叉缩放：Cr = αr·P_r − αi·P_i，Ci = αr·P_i + αi·P_r
                 （t2/t4 让位作 scratch：零额外 UB）
LoadCOld:        C 旧值交错复块 GM→UB（beta≠0 时用于缩放；恒用于 LOWER 对角列恢复）
ApplyBeta:       DeInterleave C_old → C_or(t2), C_oi(t4)
                 βi==0 快路径：Axpy(t1, t2, βr)、Axpy(t3, t4, βr)（skipTemp 时 MulScalar）
                 复数 beta：Cr += βr·C_or − βi·C_oi；Ci += βr·C_oi + βi·C_or
                 （交叉项借 cInUb 空闲段作 scratch，公式同 csyr2k 复数 beta） ← 本算子新增
（无 ZeroDiagonal —— 对角虚部保留，对称非 Hermitian）              ← 与 cherkx 的第三处差异
InterleaveAndStore: Interleave(Cr,Ci) → cOut；uplo 感知写回：
                 非对角块（整块在三角内）：单次 2D te::Copy 直写
                 对角块：逐列 uploCnt 截断（UPPER）；LOWER 对角列先在 UB 内恢复
                 非 uplo 段（C_old 原值恒等回写，语义上不修改、逐位不变）后整列单次写出
                 （两遍式：先全列 UB 恢复 + 单次栅栏，再统一 MTE3，规避同址双写乱序）
同步护栏：MTE2→V（结果可见）、V→MTE3（Interleave 后写出）、MTE3→V / MTE3→MTE2（块间 buffer 复用）
```

确定性：全部运算为固定顺序向量指令（无原子、无跨核归约、无 split-k 动态调度），同一输入任意多次执行逐位一致（任务书 §2.5 确定性要求）。

NaN/Inf 语义：βi==0 与 αi==0 快路径不引入 0×Inf 交叉项（沿用 cherkx 已验证的实数快路径指令序列）；skipTemp && beta==0 路径 uplo 三角写确定值 (0,0)，C 旧值中的 NaN/Inf 不扩散。

#### 性能优化要点（终态）

| 优化项 | 状态 | 说明 |
| --- | --- | --- |
| 4× 实 GEMM 复用 gemm_kernel_do 多核分块 | 设计即得 | 与 cherkk/cherkx 同款 AIC 利用率最大划分；GEMM 阶段占主导，计算量与 cherkx 完全一致 |
| Phase 0/2 双缓冲 + 2D strided te::Copy 块搬运 | 设计即得 | MTE 与 V 管重叠；对角块两遍式单栅栏写出（cherkx K3 修复成果直接继承） |
| 实/复数标量快路径 | 设计即得 | αi/βi 为 0 时走与实数版逐位一致的短指令序列，避免多余向量乘 |
| 实测数据回填 | S4/S5 | 部署后按 msprof op 口径实测，差距表与优化迭代记录进自测报告（本 MR 阶段不填未经实测的数字） |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR / Ascend 950DT | √（本任务目标，arch35） |
| Atlas A3 训练系列 / A3 推理系列 | 不支持（同族 ssyrkx/cherkx 口径） |
| Atlas A2 训练系列 / A2 推理系列 | 不支持 |

> 依赖 CANN asc-devkit ≥ 9.1（tensor_api），与 cherkx 相同经 `cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` 条目门控：低版本环境自动跳过编译、测试跳过。单算子构建口径：`bash build.sh --ops=csyrkx --soc=ascend950`。

## 算子约束限制

对齐任务书 §2.5 逐条：

- 非连续 Tensor：不支持（A/B/C 按列主序前导维连续存放，kernel 直调不处理非连续步长）；
- broadcast：不支持（维度按 n/k 固定对齐）；
- dynamic shape：不要求（n/k 编译期已知静态值）；
- 原地与视图语义：C 为输入/输出；beta 非零（复数零判定）时读取 uplo 三角旧值参与累加；仅写回 uplo 三角，另一三角不引用不修改，不返回视图；
- 确定性计算：满足（固定顺序向量指令 + 固定 tiling，无原子/动态归约），同输入多次执行逐位一致；
- 空 Tensor 与 0 维：合法（n=0 quick return；k=0 仅 uplo 三角 beta 缩放）；参数校验先于 no-op 判定。

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 按 FLOAT32 分量分别比对：逐元素通过条件 `&#124;actual − golden&#124; ≤ atol + rtol×&#124;golden&#124;`，rtol = atol = 2^-13 (1.2e-4)，required_matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32·ULP（逐元素 max(1e-2, 32·ULP(&#124;golden&#124;))）；双指标同时满足。Inf/NaN 特殊值按等效规则单独判定（同为 NaN / 同号 Inf / NaN↔±Inf 视为通过，有限值对非有限值不通过；非有限值不计入双指标统计）。golden 由 Netlib BLAS 3.10.0 `cblas_cgemm` 生成后恢复非 uplo 三角原值；比对范围为 uplo 三角，非 uplo 三角须与调用前逐位一致 | 任务书 §3.2；生态算子开源精度标准 mixed_tolerance_standard（opbase） |
| 性能标准 | Atlas 950PR，msprof op 模式单算子 task duration（us），warmup 5 次 + 有效采样 10 次算术平均（仅 SUCCESS 计入）；判据双指标同时满足：① 每 case 平均耗时 ≤ 2.5× GPU 标杆（≥0.4× 性能）；② 全部性能 case 倍率（GPU/NPU）算术平均 ≥ 0.5×。标杆 cuBLAS `cublasCsyrkx`（P102 / 驱动 550.104.05 / CUDA 12.4），数据 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（任务书 §3.4） | 任务书 §3.4 |

## 测试概览

- 框架：ops-blas `test/` 目录 CSV 驱动 C++ GTest 工程，落位 `test/syrkx/csyrkx/arch35/`（`csyrkx_test.cpp` + `csyrkx_test.csv` + golden/param 头 + npu_wrapper），构建接入仓内 test CMake（与 test/syrkx/ssyrkx、test/herkx/cherkx 同构）；README 写明复现步骤，保证验收人可复跑。
- 用例分类（对齐任务书 §3.5-4）：小尺寸基础（n/k ∈ {0,1,小质数,2 的幂,±1 邻域,非对齐,大规格} 尺寸扫描）；uplo(2) × trans(3，含 C≡T 等价验证) 全组合；标量特殊值（alpha/beta ∈ {(0,0), (±1,0), 纯虚数, 大值, 均匀[-5,5], 正态}，A/B/C 数据同规则、复数实虚独立采样）；前导维 padding（lda/ldb/ldc > min）；边界/负向（n=0 期望 SUCCESS 且 C 不变、k=0 期望 SUCCESS 且 C=beta×C 仅 uplo 三角、空指针/非法枚举/非法前导维/负维期望对应错误码）；Inf/NaN（按 §3.2-4 规则判定）；性能用例（对齐 gpu_baseline.csv 的 case 集）。
- 判定实现：uplo 三角逐元素混合容差 + 非 uplo 三角逐位一致性双检查；Inf/NaN 分通道统计。

## 兼容性分析

- 新增算子，无存量调用方，无兼容性破坏。
- 公共文件改动面（任务书 §2.3/§7.3 授权范围内，逐项最小化）：
  1. `include/cann_ops_blas.h`：新增 `aclblasCsyrkx` 声明（置于 aclblasSsyrkx/aclblasCherkx 等族声明邻域）；
  2. `cmake/asc_devkit_version.cmake`：`TENSOR_API_OPS` 追加 `CSYRKX|/syrkx/arch35/csyrkx|csyrkx` 条目（低版本 devkit 门控，cherkx 同款机制）；
  3. `blas/CMakeLists.txt`：`_GEMM_KERNEL_DEP_OPS` 追加 `csyrkx`（host 直链 gemm_kernel_do 的单算子构建登记表，仓内既定流程：新增复用 gemm_kernel_do 的算子须同步登记）；
  4. `blas/syrkx/README.md`：增补 aclblasCsyrkx 接口节（沿用族 README 结构与产品支持表口径）。
- 不改动 include/ 既有声明、common/ 框架与其他算子；对其他 arch（arch22）与产品线零影响。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| 与 cherkx 的三处语义差异（合并符号、复数 beta、无对角清零）引入实现错误 | 差异点全部收敛在 Phase 2 combine（单文件单函数簇）；用例含 A=B 退化（可对照 csyrk 口径）与 trans=C≡T 逐位一致性专项 |
| LOWER 对角列非 uplo 段的恒等回写造成写放大或逐位不一致 | 继承 cherkx 已修复的两遍式 UB 恢复 + 单栅栏整列写出；测试含非 uplo 三角逐位一致性强校验 |
| 大规格 workspace 超 2 GiB | 单区预乘护栏提前拒绝（ALLOC_FAILED），Ssyrkx/cherkx 同款；性能 case 最大规格按 gpu_baseline 核对余量 |
| 复数 beta 交叉缩放引入 0×Inf 项改变 NaN 形态 | αi/βi==0 快路径保持实数指令序列（不引入 0·P 交叉项）；Inf/NaN 用例按 §3.2-4 等效规则判定 |
