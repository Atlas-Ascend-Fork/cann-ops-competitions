# aclblasCsyrkx 算子开发设计文档

## 需求背景（required）

### 需求来源

昇腾社区 2026 年 10 月社区任务：基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ），
使用 Ascend C 编程语言实现单精度复数（complex64）对称秩-k 扩展更新算子 `aclblasCsyrkx`，
适配 Atlas 950PR（arch35 / DAV_3510）。验收通过后合入昇腾算子开源仓
`blas/syrkx/arch35`。

### 背景介绍

#### aclblasCsyrkx 算子功能

`aclblasCsyrkx` 计算对称秩-k 扩展更新（syrk eXtended）：

```
C = alpha * op(A) * op(B)^T + beta * C
```

- `trans = ACLBLAS_OP_N` 时：op(A) = A、op(B) = B（A、B 均为 n×k），`C = alpha * A * B^T + beta * C`；
- `trans = ACLBLAS_OP_T` 时：op(A) = A^T、op(B) = B^T（A、B 均为 k×n），`C = alpha * A^T * B + beta * C`；
- `trans = ACLBLAS_OP_C` 时：按 ACLBLAS_OP_T 等价处理（**不共轭**）——syrkx 的结果矩阵是对称
  （非厄米特）矩阵，共轭转置会破坏对称性；cuBLAS `cublasCsyrkx` 对 `CUBLAS_OP_C` 同样按
  `CUBLAS_OP_T` 执行（P102 + CUDA 12.5 实测 T 与 C 输出逐位相同）；
- C 为 n×n **对称**复数矩阵（`C = C^T`，非厄米特，对角元素虚部不假定任何值）；
  仅 `uplo` 指定的上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER）被引用并更新，
  另一三角不被访问、不被写回；
- alpha、beta、A、B、C 均为单精度复数（complex64，实部/虚部各 float32），列主序存储。

对标 cuBLAS `cublasCsyrkx`。与 `aclblasCsyrk` 的唯一区别是第二个因子独立为 B
（`A*B^T` 而非 `A*A^T`），因此 A、B 各有自己的前导维 lda/ldb，乘积矩阵不再是 Gram 矩阵。

#### 现状分析

ops-blas 仓已有同族实数算子 `aclblasSsyrkx`（arch35）与复数算子 `aclblasCsyrk`（arch35），
但**尚无 `aclblasCsyrkx` 接口声明与实现**，`include/cann_ops_blas.h` 需新增声明。

与 csyrk 相比的实现差异：

1. **两个独立输入**：4M 分解从 `Ar*Ar^T` 变为 `Ar*Br^T`，A、B 可以有不同的 leading dim；
2. **deinterleave 处理两个矩阵**：A、B 在同一 kernel 内按列分段一次拆分，避免两次启动；
3. **对角线直算公式不同**：`Σ(ar*br - ai*bi)` / `Σ(ar*bi + ai*br)`，需同时读 A、B；
4. **小 shape 单 kernel 路径**：syrkx 的两个操作数使三阶段 pipeline 的固定开销占比更高，
   小 n 全部走 SIMT 单 kernel，且该路径的求值顺序与 Netlib cgemm 参考实现逐位一致。

### 算子功能分析

| 参数 | 参数含义 | 数据类型 | 支持类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | BLAS 句柄 | aclblasHandle_t | - | 非空，绑定 stream | - |
| uplo | 三角方向 | aclblasFillMode_t | UPPER/LOWER | 非空 | - |
| trans | 转置 | aclblasOperation_t | N/T/C | C 等价 T | - |
| n | C 阶数 | int | - | ≥0 | C: n×n |
| k | 内维 | int | - | ≥0 | op(A), op(B): n×k |
| alpha | 标量 | const aclblasComplex* | complex64 | 非空 | - |
| A | 输入 | const aclblasComplex* | complex64 | 非空（n,k>0） | n×k 或 k×n |
| lda | A 前导维 | int | - | ≥max(1,n)（N）/ ≥max(1,k)（T/C） | - |
| B | 输入 | const aclblasComplex* | complex64 | 非空（n,k>0） | n×k 或 k×n |
| ldb | B 前导维 | int | - | ≥max(1,n)（N）/ ≥max(1,k)（T/C） | - |
| beta | 标量 | const aclblasComplex* | complex64 | 非空 | - |
| C | 输入/输出 | aclblasComplex* | complex64 | 非空 | n×n |
| ldc | C 前导维 | int | - | ≥max(1,n) | - |

### 边界/异常行为

- 参数校验先于 no-op 判定（\|n=0 且 alpha 为空指针仍返回 INVALID_VALUE\|）；
- `n = 0`：合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不写 C；
- `k = 0`：仅对 uplo 三角执行 `C = beta*C`（beta=(0,0) 置零、beta=(1,0) 不变），不引用 A/B，
  另一三角不得改写；
- `(alpha=(0,0) 或 k=0) 且 beta=(1,0)`：C 不变，直接返回；
- 空指针 alpha/beta、非法 uplo/trans、前导维不足、负 n/k：返回
  `ACLBLAS_STATUS_INVALID_VALUE`；handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- A/B 仅在「n>0 且 k>0」时报空指针错误，C 仅在「n>0」时报错。

## 需求分析（required）

### 需求描述

使用 Ascend C 语言在 Ascend 950PR 上实现 `aclblasCsyrkx`，功能/精度/性能与 cuBLAS 对标：

1. 支持 uplo×trans 全枚举（UPPER/LOWER × N/T/C）与 complex64；
2. 精度满足生态算子开源精度标准（complex64 按 FLOAT32 分量判定：rtol=2⁻¹³、atol=2⁻¹³、
   matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP），非 uplo 三角须与调用前逐位一致；
3. 性能验收：单 case NPU 耗时 ≤ 2.5× H100 GPU 基线（即 ≥0.4× 性能），
   且全部性能 case 的倍率算术平均 ≥ 0.5×；
4. 1200 条自测用例（1000 精度 + 200 性能）全量通过。

### 需求拆解

1. host 侧：句柄式 BLAS 接口、参数校验、路径选择、workspace 规划、三阶段 tiling 计算、kernel 调度；
2. 小 shape：单 kernel SIMT 直算（顺序累加，与参考实现逐位一致）；
3. 中/大 shape Phase 0（AIV）：复数 A、B 解交错为实矩阵 Ar/Ai、Br/Bi；
4. 中/大 shape Phase 1（AIC）：4 个实数 GEMM（`Ar*Brᵀ, Ai*Biᵀ, Ar*Biᵀ, Ai*Brᵀ`）融合并行；
5. 中/大 shape Phase 2（AIV）：三角区域 combine（Q0-Q1 / Q2+Q3 → 复数 C），仅写 uplo 三角；
6. 精度保障：大 k 对角直算（useDirectDiag）、HF32x3 补偿 cube（大 shape）、单 kernel 顺序累加。

## 详细设计（required）

### 算子分析

#### 数学公式

复数对称 rank-k 扩展更新的 4M 实数分解（op 省略，trans 只改变 A/B 的读法）：

```
C = alpha * op(A) * op(B)^T + beta * C,   op(A) = Ar + i*Ai,  op(B) = Br + i*Bi
Q0 = Ar * Br^T      (实·实)
Q1 = Ai * Bi^T      (虚·虚)
Q2 = Ar * Bi^T      (实·虚)
Q3 = Ai * Br^T      (虚·实)
Cr = Q0 - Q1        (C 实部)
Ci = Q2 + Q3        (C 虚部)
C  = alpha*(Cr + i*Ci) + beta*C_old
```

4 个 GEMM 均为实数乘累加，复用仓内成熟的 `SyrkGemmKernelImpl`
（`common/helper/syrk_gemm_arch35.h`）与 fused-4M 驱动
（`common/helper/syrk_gemm_fused_arch35.h`，与 csyrk 共用）。
等效计算量 4·n²k，比朴素复数 GEMM（6·n²k）节省 1/3。

#### 支持数据类型

complex64（float32 实部/虚部）。

#### 支持形状

n∈[1,4096]，k∈[1,4096]，方阵/宽矩形/窄矩形（trans=N 时 k≫n、trans=T 时 n≫k），
含奇数、非 32B 对齐、边界值。

### 算子实现

#### 实现方案总览

两条路径，由 host 侧 `CsyrkxPreferDirect` 按 shape 选择：

```
小 shape（n<=64，或 n<=160 且 n²k<=8M）：
  csyrkx_direct_kernel (AIV SIMT, 1 次启动)
  └─ C(uplo) = alpha*op(A)*op(B)^T + beta*C，逐元素顺序累加

中/大 shape：
A, B (complex, col-major)
  │  Phase 0: csyrkx_deinterleave_kernel (AIV)
  ▼
Ar, Ai, Br, Bi (实矩阵, aLdc/bLdc × cols)
  │  Phase 0b（n>256 或 k>512）: csyrkx_hf32resid_kernel (SIMT)
  ▼
ArLow, AiLow, BrLow, BiLow（HF32 低残差）
  │  Phase 1: csyrkx_cube_kernel (AIC, fused 4-quad)
  ▼
temp[4n×tempLdc] = [Q0|Q1|Q2|Q3]
  │  Phase 2: csyrkx_combine_kernel (AIV)
  ▼
C (complex, 仅 uplo 三角)
```

#### host 侧设计

入口 `aclblasCsyrkx` → `LaunchCsyrkxKernel`，流程：

1. **参数校验**（`ValidateCsyrkxParams`）：uplo/trans 枚举 → n/k/lda/ldb/ldc 约束 →
   alpha/beta/A/B/C 指针非空；全部校验完成后才判定 `n == 0`；
2. **小 shape 分流**（`CsyrkxPreferDirect`）：`n<=64` 或 `n<=160 且 n²k<=8M` 时直接启动
   direct kernel。该分流**先于** alpha/beta 的 D2H 读回，因此小 shape 无任何 host↔device
   标量往返（launch-bound 场景的关键）；
3. **标量准备**：alpha/beta 为 Device 内存，两次 `aclrtMemcpyAsync` + 一次
   `aclrtSynchronizeStream` 读回 host，得到 `isAlphaZero/isBetaZero/isKZero`；
4. **beta-only 分支**（alpha=(0,0) 或 k=0）：只启动 combine kernel 的 beta 分支，
   不分配 workspace、不解交错、不做 GEMM；
5. **workspace 规划**（`CsyrkxEnsureWorkspace`）：
   - 布局 `[Ar | Ai | (ArLow | AiLow) | Br | Bi | (BrLow | BiLow) | temp]`；
   - `aLdc = ceil_align(nRows, 8)`（trans=N 时 nRows=n，否则 nRows=k），`bLdc = aLdc`；
     `tempLdc = ceil_align(n, 8)`（workspace 预算用；cube 实际列距为 n）；
   - `arBytes = aLdc * aCols * 4`、`tempBytes = 4 * n * tempLdc * 4`；
   - **溢出检查**：所有字节数用 `CsyrkxMulU64/CsyrkxAddU64` 计算，防止 uint32 维数
     相乘回绕后绕过 2 GiB 上限；
6. **核数规划**：Phase 0/2 用 AIV 核数（`GetAivCoreCount()`），Phase 1 用
   `GetUsedAicCoreNum(n, 16)`；combine 在小 n 时按 `ceil(n/16)` 限核以避免每核行数过少；
7. **三阶段 tiling 计算**并依次 `<<<numBlocks, nullptr, stream>>>` 启动。

##### 分核策略

- **Phase 0（deinterleave）**：按 A/B 的列均分——`colStart = blockIdx * ceil(cols/核数)`，
  每核处理连续列段；A、B 共用同一段列、同一个 kernel，
  单次启动完成两个矩阵的拆分；
- **Phase 0b（HF32 残差）**：SIMT，线程沿列内行连续访问，block 切分列段（与 Phase 0 相同划分）；
- **Phase 1（cube）**：64×64 单属主 tile，`tileIdx % blockNum == blockIdx` 分核，
  `triangleMode` 跳过 uplo 侧的整块 tile（UPPER 跳过 `mStart > nStart+63`）；
  可用 AIC 核 <4 时回退为单核串行 4 个 quad（`SyrkGemmKernelImpl` 通用驱动）；
- **Phase 2（combine）**：64×64 tile 按索引 stride 分核（三角 tile 数 ≥ 核数），
  否则按行带分核（小 n）；对角 tile 逐行处理，行间无依赖。

##### tiling 计算

三个 TilingData 结构（`csyrkx_tiling_data.h`）：

- `CsyrkxDeinterleaveTilingData`：nRows/nCols/lda/ldb/aLdc/bLdc/colsPerCore；
  A、B 逻辑形状相同（列数、行数一致），前导维可不同；
- `CsyrkxCubeTilingData`：n/k/aLdc/bLdc/tempLdc/usedCoreNum/singleCoreM/singleCoreN/
  tileM/tileN/tileKChunk/isTransN/triangleMode/hf32；
  - tileM/tileN 默认 128，K chunk 默认 256，受
    `l1Budget / (L1_BUF_NUM * 4 * (alignedM + alignedN))` 反推的上限约束；
  - `hf32 = !(n<=256 && k<=512)`：大 shape 走 HF32x3 补偿 cube；
- `CsyrkxCombineTilingData`：n/k/ldc/tempLdc/aLdc/bLdc/rowsPerCore/
  alphaReal/alphaImag/betaReal/betaImag/uploMode/isTransN/
  isAlphaZero/isKZero/isBetaZero/useDirectDiag；
  - `useDirectDiag = (k > 512)`：大 k 时对角线从 Ar/Ai/Br/Bi 直接重算（精度兜底）。

#### kernel 侧设计

四个 kernel 均通过 `csyrkx_*_kernel_do` 以 `<<<numBlocks, nullptr, stream>>>` 启动。

##### csyrkx_direct_kernel（AIV，小 shape）

- 每个线程负责一个 uplo 元素，`index → (col, row)`（行是快轴，保证 A/B/C 访问合并）；
- 求值顺序**严格复刻 Netlib cgemm.f 的内层循环**：
  `C = beta*C_old`，然后按 k 升序 `C += (alpha * op(B)(j,k)) * op(A)(i,k)`；
- 中间量用 `volatile` 强制逐次舍入：设备后端不保证 `#pragma clang fp contract(off)`，
  FMA 收缩会让结果与 golden 产生 ~1e-4 的差异（在相消元素上超出容差）；
- 进入累加前对 alpha/beta 与每个 k 项的操作数做特殊值扫描（`CsyrkxUnsafeValue`），
  命中 Inf/NaN/超范围值时该元素改用 `CsyrkxReferenceElement`（C99 `__mulsc3` 复数乘法
  语义），保证 Inf/NaN 的位置与 golden 一致。

##### Phase 0：csyrkx_deinterleave_kernel（AIV）

- 输入：complex A、complex B（列主序，每列 `[re0,im0,re1,im1,...]` 交错）；
- 输出：Ar、Ai、Br、Bi 实矩阵（`aLdc/bLdc × nCols`）；
- 每核处理连续列段；列内按 `DEINT_MAX_ELEMS = 2040` 个元素批量：
  一次 2D `DataCopyPad`（`PaddingMode::Compact`，每列 `rows` 个 8B block，列间 stride = `lda/ldb`）
  读入 `nb` 列 → 向量 `DeInterleave` 拆分实/虚 → 一次 2D `DataCopyPad` 写回（列间
  dstStride = `outLdc - rows`）；
- 行数超过批量上限时按列分块处理；A、B 在同一核上顺序处理，共用 UB 缓冲。

##### Phase 0b：csyrkx_hf32resid_kernel（SIMT）

- 读已拆分好的 Ar/Ai、Br/Bi，写低残差 `x - HF32(x)`（HF32 = 10 位尾数、就近舍入）；
- 目的：Phase 1 的 HF32x3 路径用三次 mmad（`xh*yh + xl*yh + xh*yl`）恢复 fp32 精度，
  同时保留 HF32 mmad 的吞吐。

##### Phase 1：csyrkx_cube_kernel（AIC）

- 输入：Ar/Ai（+ 低残差）、Br/Bi（+ 低残差）；输出：temp[4n×tempLdc] = [Q0|Q1|Q2|Q3]；
- 复用 `SyrkGemmQuadFusedTileKernelImpl64` / `...Hf32x3`
  （`common/helper/syrk_gemm_fused_arch35.h`）：
  每个 64×64 C tile 的 4 个 quad 背靠背计算，K chunk 从 GM 读一次 4 个 panel
  （Ar/Ai/Br/Bi），L1 双缓冲、L0 mmad 循环、L0C 双 acc-set ping-pong；
- quad 选择：`SyrkQuadUsesLeftReal(q) = (q==0||q==2)`，`SyrkQuadUsesRightReal(q) = (q==0||q==3)`；
- trans=N：`left=A(n×k col-major, DNExt(aLdc,k))`、`right=B^T(k×n, NDExt(k,bLdc))`；
  trans=T/C：`left=A^T(n×k, NDExt(n,aLdc))`、`right=B(k×n, DNExt(bLdc,n))`；
- temp 布局 `DNExtLayoutPtn(n, tempLdc)`，quad 间距 `n*tempLdc`。

##### Phase 2：csyrkx_combine_kernel（AIV）

`CsyrkxKernel` 类，Init + Process 两阶段：

- **Init**：UB 规划（约 232 KB）。quad 缓冲 q0..q3（各 64×64×4B）、C_old 交错缓冲 cIn
  （2×64×64×4B，DeInterleave 后复用为 Interleave 输出）、cRe/cIm、outRe/outIm、
  对角直算的 ar/ai/br/bi（各 1024×4B）与 accRe/accIm、缩放临时 t1/t2、ReduceSum work；
- **Process**：三角 tile 遍历（tile-stride 或 row-band），每 tile 分三类：
  - `CombineBlock`（非对角块，整块在三角内）：8 行对齐部分一次读 4 个 quad 的
    rows×cols 块 → `Sub(Cr = Q0-Q1)` + `Add(Ci = Q2+Q3)` → 复数 alpha/beta 缩放
    （4 次 `Muls` + `Sub/Add`）→ `Interleave` 交错 → 一次 `DataCopyPad` 写回；
    尾部 0..7 行退化为逐行 strip；
  - `CombineDiagBlock`（对角 tile）：uplo 边界切过每一行，逐行处理，每行长度为
    `len = cols - r`（UPPER）或 `r + 1`（LOWER），走 `CombineRow`：
    4 条 strided `DataCopyPad`（Compact，每元素 4B）读 quad strip → Sub/Add →
    （可选）对角覆盖 → 缩放 → 2 条 strided `DataCopyPad` 写复数实部/虚部；
  - `beta-only`（alpha=(0,0) 或 k=0）：不读 quad，仅读 C_old、`C = beta*C_old` 写回；
    `beta=(1,0)` 时直接返回不改 C；
- **`DirectDiag`**：`useDirectDiag` 开启时，对角线元素从 Ar/Ai/Br/Bi 直算
  `Σ(ar*br - ai*bi)` / `Σ(ar*bi + ai*br)`（每项先乘后减再累加，按 1024 元素分块 + `ReduceSum`），
  避免 `Q0[i,i] - Q1[i,i]` 两个大正数相减导致的相对精度损失；
- **流水同步**：向量缩放结果到 MTE3 写出之间使用 `PipeBarrier<PIPE_ALL>()`。
  仅 `PIPE_V` 无法保证 MTE3 看到向量结果，会读到旧 UB 数据（联调中实测到该缺陷：
  部分元素随机取到上一 tile 的残留值）。

#### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（arch35 / DAV_3510） | √ |

#### 算子约束限制

1. 仅支持 complex64（单精度复数）；
2. C 为对称（非厄米特）矩阵，对角元素虚部不置零（与 cherk 区别）；
3. OP_C 按 OP_T 等价处理（不共轭）；
4. 仅更新 uplo 指定三角，另一三角不被引用、不被写回；
5. 不支持非连续 Tensor（A/B/C 只按列主序前导维寻址）、不支持广播、不要求 dynamic shape。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 按 FLOAT32 分量判定：rtol=2⁻¹³、atol=2⁻¹³、matched_ratio≥0.99、max_abs_error≤1e-2 或 32×ULP；非 uplo 三角与调用前逐位一致 | 生态算子开源精度标准（任务书 §3.2） |
| 性能标准 | 单 case NPU 耗时 ≤ 2.5× H100 基线（≥0.4×），且全部性能 case 的倍率平均值 ≥ 0.5× | 任务书 §3.3 |

### 自测设计

- **精度**：1000 条精度用例（TC_L0/TC_SQ/TC_MN/TC_AB/TC_LD/TC_FL/TC_CV/TC_ED/TC_EX），
  ops-blas C++ GTest（`build/test/syrkx/csyrkx/csyrkx_test`）加载 `csyrkx_test.csv`；
  golden 为 Netlib cgemm.f 参考循环（`test/syrkx/csyrkx/csyrkx_golden.h`，按输出列并行，
  逐列结果与串行一致），比对 uplo 三角实部/虚部 + 非 uplo 三角逐位一致；
- **性能**：200 条 TC_PF 性能用例（与任务书 `gpu_baseline.csv` 逐条对应），
  `msprof op` 采集单算子 task duration，10 次 warmup 后 50 次有效采样取均值；
- **内存**：单用例 host 侧 ≤512MB。

### 兼容性分析

新算子，不涉及兼容性分析。接口声明放入 `include/cann_ops_blas.h`，与其他产品线共用
同一 `aclblasCsyrkx` API，无 950PR 私有平行接口。
