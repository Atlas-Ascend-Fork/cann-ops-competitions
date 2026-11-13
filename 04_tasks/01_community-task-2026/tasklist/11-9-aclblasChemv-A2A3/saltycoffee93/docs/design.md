# aclblasChemv 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-ops-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasChemv 算子开发任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/symv/arch22/`（ssymv 四件套）、`blas/symv/arch35/csymv_*`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasChemv 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/symv/arch22/`（实现，与实数 ssymv 同族目录）与 `test/symv/chemv/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数 Hermitian 矩阵-向量乘能力（BLAS-2），
对标 cuBLAS `cublasChemv`（语义参考 Netlib `chemv.f`）。

## 背景介绍

### 算子功能定位

BLAS-2 Hermitian 矩阵-向量乘加：`y = α·A·x + β·y`，A 为 n×n Hermitian 复数矩阵（列主序全存储 lda×n），
仅 uplo 指定三角被引用，另一侧由共轭对称性隐含；对角元素虚部假定为 0、不参与计算。

数学公式：

```
y[i] = α × Σ_j A[i,j] × x[j] + β × y[i]，i = 1..n
A = A^H：仅 uplo 三角读取；对角元素只取实部
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasChemv | cublasChemv | 单精度复数 HEMV：单三角引用 + 镜像共轭读、复数 α/β、正负步长、原地输出 |

### 业界对标

- 对标接口：cuBLAS `cublasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)`。
- Golden 语义来源：Netlib BLAS `chemv.f`（https://www.netlib.org/blas/chemv.f），cblas_chemv（CblasColMajor）。
- 语义差异点（相对仓内既有实现）：
  1. **对角虚部为 0 假设**：实现只取对角元素实部参与计算（golden 同口径，任务书 §2.1-2/§3.2）；
  2. **α/β 双读取口径（issue #869）**：alpha/beta 为 Host **或** Device 内存指针，运行时按指针属性探测，
     对齐仓内 `blas/symv/arch35/csymv_host.cpp` 的 `ComplexScalarSource` 双读取口径（本任务现场核验该文件在仓）；
  3. **quick return 双条件**：n=0 直接成功；`alpha=(0,0)` 且 `beta=(1,0)` 亦直接成功
     （此时 A/x/y 可均为 nullptr，任务书 §2.5）；
  4. **beta=(0,0) 不读 y 旧值**，但不豁免 y 非空要求（§2.4 y 行）；
  5. **跨度守卫（issue #807/#780）**：`(n−1)×|inc|` int64 计算，INT32_MIN 拒绝——对齐同目录
     ssymv arch22 既有大跨度守卫实现（本任务现场核验 ssymv 四件套在 `blas/symv/arch22/`）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/symv/arch22/`（该目录**已存在**，含实数 ssymv 完整四件套 → **并列加复数文件**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同目录 arch22 实数实现（主复用源） | `blas/symv/arch22/ssymv_{host,kernel,kernel.h,tiling_data}.*` | arch22 工程骨架、单三角读取 + 镜像访问的 tiling/kernel 范式、校验序、大跨度守卫（issue #807）直接平移 |
| 同算子 arch35 复数实现 | `blas/symv/arch35/csymv_{host,kernel,tiling_data}.*` | 复数语义、ComplexScalarSource（issue #869）双读取口径、共轭镜像访问平移；kernel 不平移（SIMT） |
| 交错复数处理范式 | `blas/asum/arch35/scasum_*` 及同批次 Scasum/ICAMIN 设计 | 交错存储按 2n float 处理、复数乘加的实/虚分解 |

### 接口规格

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

- **本次任务需在 `include/cann_ops_blas.h` 新增该声明**（任务书 §2.3/§7.3 明确授权：仓内当前无
  aclblasChemv 声明，现场核验确认；签名与同族 aclblasSsymv（头文件 L181）风格一致，int 维度参数）。
  新增位置与同族声明相邻，不改其他行 → 红线 3 授权例外，登记例外清单（连带文件仅 `include/cann_ops_blas.h`）。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| uplo | A 存储模式，Host 内存 | aclblasFillMode_t | — | ∈ {ACLBLAS_UPPER, ACLBLAS_LOWER}；不在枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` | — |
| n | A 的阶数，Host 内存 | int | int32 | n≥0；n<0 → INVALID_VALUE；n=0 合法 no-op | — |
| alpha | 复数标量 α 指针，Host/Device 内存（运行时探测，issue #869） | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；值不校验 | 标量 |
| A | Hermitian 矩阵，Device 内存，只读；仅 uplo 三角被引用，对角虚部假定 0 | const aclblasComplex* | COMPLEX64 | α≠(0,0) 且 n>0 时 nullptr → INVALID_VALUE；未引用三角不得读取 | lda×n，列主序 |
| lda | A 前导维度，Host 内存 | int | int32 | lda ≥ max(1,n)，否则 INVALID_VALUE | — |
| x | 复数向量，Device 内存，只读 | const aclblasComplex* | COMPLEX64 | α≠(0,0) 且 n>0 时 nullptr → INVALID_VALUE | 一维 [n]，物理长度 `1+(n−1)*|incx|` |
| incx | x 步长，Host 内存 | int | int32 | ≠0 且非 INT32_MIN；0/INT32_MIN 或 `(n−1)*\|incx\|` 溢出（int64，issue #807/#780）→ INVALID_VALUE | — |
| beta | 复数标量 β 指针，Host/Device 内存（同 α）；β=(0,0) 时 y 旧值不必有效 | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE | 标量 |
| y | 输入兼输出（原地覆写），Device 内存 | aclblasComplex* | COMPLEX64 | n>0 且非（α=(0,0) 且 β=(1,0)）时 nullptr → INVALID_VALUE；β=(0,0) 不豁免非空 | 一维 [n]，物理长度 `1+(n−1)*|incy|` |
| incy | y 步长，Host 内存 | int | int32 | 同 incx | — |

### 数学公式

```
y ← α·A·x + β·y，A = A^H（仅 uplo 三角读取，对角只取实部）
```

与 §3.2 golden（cblas_chemv，CblasColMajor；对角虚部置 0、另一三角共轭补全）同一形态。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-2 单精度复数 HEMV：COMPLEX64 列主序 Hermitian 矩阵、
单三角引用 + 共轭镜像读、复数 α/β 双位置探测、正负步长、原地输出；
精度实部/虚部分别按 FLOAT32 档判定（rtol=atol=2^-13，ratio≥0.99）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `y ← α·A·x + β·y`（Hermitian，单三角）。golden 白嫖 cblas_chemv（Netlib 复数单精度，CblasColMajor）；golden 侧按 Hermitian 约束补全（对角虚部置 0、未存三角共轭对称），与实现口径一致 |
| 2 | 接口原型 | 11 参签名（上节）；**声明须新增**至 `include/cann_ops_blas.h`（任务书 §2.3 授权，现场核验无既有声明），风格对齐同族 aclblasSsymv |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | 输入/输出全 **COMPLEX64**（α/β/A/x/y，任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 已有 arch35 csymv，不受影响 |
| 6 | 精度要求 | 实部/虚部分别按 FLOAT32 档：rtol=atol=2^-13、ratio 0.99、max_abs_error ≤ 1e-2 or 32×ULP；Inf/NaN 按 §3.2 等效规则（mixed_tolerance_standard，opbase commit 28a1ce52） |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定，uplo 双态考核。典型 case（实测 CSV 读数）：UPPER n=512 → ≤18.30us；UPPER n=1024 → ≤26.39us；LOWER n=2048 → ≤37.61us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase：ssymv arch22 骨架 + 复数化**（单三角读取、镜像共轭、对角实数化三处改造）。死路①：arch35 csymv kernel 平移（SIMT）；死路②：全矩阵显式补全另一三角（存储与带宽翻倍，违反"未引用三角不得读取"） |
| 9 | 评测来源 | 精度 golden：Netlib `cblas_chemv`；性能标杆：NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4（`test_cases/gpu_baseline.csv`） |

**校验顺序（V1..V9，由任务书 §2.5 校验顺序条款逐字反推，顺序不可调换）**：

| 序 | 条件 | 返回码 / 行为 |
|----|------|--------------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（合法 no-op，不执行计算） |
| V4 | `uplo ∉ {UPPER, LOWER}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| V5 | `lda < max(1, n)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `incx == 0 \|\| incy == 0 \|\| incx == INT32_MIN \|\| incy == INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V7 | `n>0 && (alpha == nullptr \|\| beta == nullptr)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V8 | 读到 α/β 标量后：`α==(0,0) && β==(1,0)` | `ACLBLAS_STATUS_SUCCESS`（quick return，A/x/y 可均 nullptr） |
| V9 | `α≠(0,0) && (A==nullptr \|\| x==nullptr)`；`n>0 && y==nullptr`（β 任意） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V10 | `(n−1)*\|incx\| 或 (n−1)*\|incy\| 超 int32 范围（int64，issue #807 双边 + #780 INT32_MIN 已并入 V6）` | `ACLBLAS_STATUS_INVALID_VALUE` |

> V8 位于指针判空之前（任务书 §2.5"读到标量后判定"）：α=(0,0) 时 A/x 可为 nullptr（V9 条件含 α≠0）；
> α=(0,0) 时按 y=βy 计算，实现不保证跳过 A/x 读取的说法仅适用于 Chbmv——Chemv 任务书 §2.4 A/x 行
> 明确 α≠(0,0) 才要求非空，故 α=(0,0) 时 A/x nullptr 合法，计算路径须按 y=βy 短路（不读 A/x）。

---

# 详细设计（required）

## 算子分析

### 数学公式

`y[i] = α·Σ_j A[i,j]·x[j] + β·y[i]`；仅 uplo 三角读取，镜像元素 `A[i,j] = conj(A[j,i])`（i<j 对 UPPER）；
对角元素只取实部（虚部不参与）。归约累加顺序实现自定，按阈值判定。

### 支持数据类型

全 COMPLEX64（分量 FLOAT32，交错存储）；中间乘加保持 FLOAT32，复数乘按
`(ac−bd) + (ad+bc)i` 分解，共轭乘为 `(ac+bd) + (bc−ad)i`。
特殊值按 IEEE 传播（含 0×Inf 不定式），按 §3.2 等效规则判定。

### 支持形状

| 场景 | 约定 |
|------|------|
| 矩阵 A | 列主序 lda×n 全存储，仅 uplo 三角被引用；lda=max(1,n) 紧凑与 padding 均支持 |
| 向量 | x/y 均为 n 元素复数向量；正负步长均生效（负步长反向遍历，同 Netlib chemv） |
| n=0 | 合法 no-op：成功、不访问任何指针 |
| α=(0,0) 且 β=(1,0) | quick return：成功，A/x/y 可均 nullptr（不启动 kernel） |
| β=(0,0) | 不读 y 旧值；y 指针仍须非空 |
| 广播 / 0 维 | 不涉及（BLAS-2） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（uplo 双态为 tiling 字段，镜像读方向在 kernel 内分支——实数 ssymv 同族形态）：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_CONTIG** | `incx==1 && incy==1` | 连续读 x/y，按 uplo 三角带列块乘累加 | 是（基线 3 条 base case 均连续、双 uplo） |
| **V_STRIDED** | 其余（含负步长） | 步长 DMA 读 x/y，负步长物理基址映射 | 否（精度覆盖） |

数据流（V_CONTIG，按 y 行分块；UPPER 为例，LOWER 镜像对称）：

```
   for each y 行块 [i0, i0+cnt)（按核均分）:
      acc(复数, 2×cnt float) ← β·y[i0..]（β=(0,0) 置零不读 GM）
      // 段一：j < i0（下三角区域）→ 读 A[j,i]=conj(A[i,j])（镜像共轭）
      for each 列块 j:
         GM A[uplo 带内, j] ──DataCopy──▶ UB aCol
         UB 内 conj 镜像换算（或按列块方向组织避免逐元素换算）
         acc += α·aCol镜像·x[j]
      // 段二：j ∈ [i0, i0+cnt)（本行块的对角与超对角区）
      acc += α·A[i,j]·x[j]（对角取实部；块内三角形态处理）
      GM y[i0..] ◀──DataCopy── acc
```

#### host 侧设计

**分核策略**：
- 按 y 长度 n 分核：每核负责连续行块 `[myOffset, myOffset+myCount)`，kernel 侧自算
  `myOffset = blockIdx*perCoreN + min(blockIdx, remainder)`（R4，同目录 ssymv arch22 方案）；
- y 按核不重叠 → 无跨核归约，workspace 仅 α/β Device 探测后的中转（见下）。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；
- 复数路径 UB 占用约为实数 ssymv 2 倍：aCol tile 取 2048 复数元素量级（16 KiB float×2），acc 每核 2048 复数；
- 段一（镜像区）的组织：按列块读取 uplo 带内连续段，在 UB 内做实/虚分解与共轭换算
  （`conj(a+bi) = a−bi`：虚部分量取反，纯向量 Muls(−1) 指令）；换算后的"等价下三角列"直接参与乘累加；
- 对角块：单独小循环处理（长度 cnt≤行块大小），对角元素仅取实部（虚部分量置 0 后参与）；
- β=(0,0)：acc 直接置零，不发起 GM y 读；α=(0,0)：kernel 整体短路 y=βy（不读 A/x）。

**tilingkey 规划策略**：tiling 字段：uplo、n、lda、perCoreN、remainder、tileJ、incx、incy、
α/β 位置标志（host/device，issue #869）、α、β 值（host 时直填）；字段命名对齐同目录 ssymv_tiling_data.h。

**α/β 位置探测（issue #869）**：host 侧 `aclrtPointerGetAttributes` 判 Host/Device：
- Host → 值读出直填 tiling（kernel 零额外读）；
- Device → 指针经 tiling 下发，kernel 开头一次性读入 UB 标量缓冲。

**数据检测**：按 V1..V10 顺序执行（见需求拆解节）：
- V3（n=0）先于后续判空（任务书 §2.5 顺序条款）；V8（αβ 短路）在读到标量后、A/x/y 判空前；
- V10 int64 双边守卫（对齐同目录 ssymv arch22 issue #807 既有实现）。

**其他 host 侧约定**：
- workspace：α/β 为 Device 指针时 kernel 直接经 GM 读，无需额外 workspace；本任务不申请（传 0/nullptr）；
- 同步：异步下发，host 不同步（任务书 §2.5）；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策与 α/β 位置探测结果。

#### kernel 侧设计

**Init/Process 两阶段**（V_CONTIG）：

```
Init:   解析 tiling → myOffset/myCount → SetGlobalBuffer → InitBuffer(aCol/xBuf/yAcc/conjScratch)
Process（每核，UPPER 视角；LOWER 为镜像分支）:
   1) α/β 取值（tiling 或 GM 一次性读入 UB）
   2) acc 初始化：β=(0,0) → Duplicate 0；否则 DataCopy GM y + Cmul β
   3) 段一（镜像区，j ∈ [0, i0)）:
        MTE2: DataCopy aCol ← GM A 带内列段（UPPER: 行 1..j 的第 j 列）
        V:    虚部取反（conj）→ 与 x[j] 复乘 α → Axpy 累加进 acc
   4) 段二（对角+带内区，j ∈ [i0, i0+cnt)）:
        逐列读 A[i,j]（UPPER: 行 1+k..i）, 对角列仅取实部
        acc += α·A[i,j]·x[j]
   5) 写回 GM y[i0..]（incy==1 DataCopy / 步长 DMA）
```

**复数运算分解**（交错存储 2n float 视角）：
- 复乘 α·(a+bi)：(αr·a − αi·b) + (αr·b + αi·a)i —— 4 Mul + 2 Add 纯向量指令；
- 共轭：(a, −b) —— 虚部通道 Muls(−1)；
- 对角实数化：虚部通道置 0。

**步长路径**：`asc_copy_gm2ub_align(src_gap=(|inc|−1)*8)`（复数 8B）；负步长物理基址
`(n−1−i)*|inc|`（同族 caxpy arch22 PhysicalIndex 范式）；x/y 基址随 tile 起点换算。

**workspace 布局**：无 workspace（y 按核不重叠，α/β 经 tiling/GM 直读）。

**同步事件配对**：MTE2→V、V→MTE3/MTE2 显式 SetFlag/WaitFlag；段一/段二之间 UB 复用需栅栏。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | y 按核不重叠，无跨核归约与第二次 launch |
| 镜像共轭向量化 | conj 仅虚部取反（Muls −1），换算全程 UB 内向量指令，无逐元素标量 |
| 带内连续 DMA | uplo 带内列段连续读，不触带外与未引用三角（满足"不得读取"约束） |
| 双缓冲流水 | aCol 2 槽，MTE2 与 V 重叠 |
| β=0 / α=0 短路 | 免读 GM y / 免读 A+x，提前退出 |
| host 零同步 | tiling const 引用；α/β Host 态零额外 H2D |

**未采用的路线及原因**：
- **arch35 kernel 平移**：SIMT——死路；
- **全矩阵补全另一三角后按普通 gemv 计算**：需写临时 GM 缓冲（带宽/存储翻倍）且违反"未引用三角不得读取"——不用；
- **逐元素标量共轭换算**：破坏向量流水——改为 UB 内通道级取反。

---

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | **支持**（本任务目标） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | **支持**（本任务目标） |
| Ascend 950PR / Ascend 950DT | 支持（由 arch35 csymv 既有实现提供，本次不改） |

---

## 算子约束限制

逐条对齐任务书 §2.5：

| 约束项 | 结论 |
|--------|------|
| 参数合法性 | V1..V10 全序执行：n≥0（n<0 INVALID_VALUE）、n=0 成功 no-op、非法 uplo → INVALID_ENUM、lda≥max(1,n)、incx/incy ≠0 且非 INT32_MIN、跨度 int64 守卫、α/β 非空；α≠(0,0) 时 A/x 非空；y 非空（β=(0,0) 不豁免）；α=(0,0) 且 β=(1,0) → 成功（A/x/y 可 nullptr） |
| 三角引用 | 仅引用 uplo 指定三角，未引用三角不读取；对角虚部为 0（仅实部参与）；Hermitian 镜像由实现按 conj 推断 |
| 非连续 Tensor 支持 | 不涉及额外 padding 场景：lda/incx/incy 语义内支持；负步长反向遍历 |
| broadcast 规则 | 不涉及（BLAS-2） |
| dynamic shape 要求 | 不要求：n 运行时入参 |
| 原地与视图语义 | y 原地覆写，不返回视图 |
| 确定性计算要求 | 不要求 |
| 空 Tensor 与 0 维处理 | n=0 合法 no-op；α=(0,0)∧β=(1,0) 直接成功 |
| 异步执行 | 依赖 aclblasSetStream；读回 Device 结果前须同步 stream |

**已知限制（如实标注）**：
1. 对角元素虚部不参与计算（Hermitian 假设）——输入对角含非零虚部时结果由该假设定义（golden 同口径）；
2. α=(0,0) 时 A/x 不读取（短路 y=βy）——与任务书 §2.4 判空条件一致；
3. 步长路径不参与性能判定；
4. 本次新增公共头声明（任务书授权例外），波及面仅 `include/cann_ops_blas.h` 追加行，无既有接口改动。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | `cblas_chemv`（Netlib 复数单精度，CblasColMajor）；对角仅取实部、未存三角共轭补全 | 任务书 §3.2 |
| 比对方式 | y 全向量，实部/虚部分别按 FLOAT32 档判定 | 任务书 §3.2 |
| 普通精度（rtol/atol） | 2^-13 ≈ 1.2e-4 | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP | 同上 |
| 判定式 | 逐元素 `\|actual − golden\| ≤ atol + rtol*|golden|`（实/虚分别） | 任务书 §3.2 |
| Inf/NaN 特殊值 | 同为 NaN / 同为同号 Inf / 一方 NaN 另一方 ±Inf → 等效判过；一方有限另一方非有限 → 判不过；非有限值不计入统计、单独记录；NaN/Inf 形态取决于累加顺序，双方均表示无界/未定义 | 任务书 §3.2 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），COMPLEX64 场景 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | NVIDIA A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4，见 `test_cases/gpu_baseline.csv` | 任务书 §3.3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{chemv_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/symv/chemv/arch22/`，并在 `test/symv/chemv/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)`，任务书 §2.2）；
- 用例覆盖（§3.5）：uplo 全覆盖 × 尺寸扫描（0/1/小质数/2 的幂及 ±1/非对齐/大规模）；α/β 特殊值 (0,0)/(1,0)/纯虚数/大值；lda 紧凑与多档 padding；incx/incy ±1/±2/±3；y 含 NaN 的 β=(0,0) 用例；Inf/NaN 特殊值；负向含 n=0（SUCCESS）、空指针、非法枚举、非法前导维、负维度、零步长；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]，实/虚独立采样）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则（§3.5）：期望非 SUCCESS 的用例不得按 `(n−1)*|inc|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **公共头新增声明（任务书 §2.3/§7.3 授权）**：`include/cann_ops_blas.h` 追加 `aclblasChemv` 声明一行
  （与同族 aclblasSsymv/aclblasCsymv 相邻），不修改任何既有行——其他产品线共用，无 ABI 破坏；
  红线 3 例外登记（连带文件清单：仅该头文件）；
- **对 arch35 影响**：无。`blas/symv/arch35/csymv_*` 保持不变，按 SOC 选目录编译互不覆盖；
- **对 arch22 同目录 ssymv 影响**：无。并列复数新文件，接口独立、无共享可变状态；
- **测试构建影响**：`test/symv/chemv/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支不变；
- **文档影响**：`blas/symv/README.md` 产品支持表新增 aclblasChemv「Atlas A2/A3：支持」，并同步约束说明
  （uplo 枚举、α/β 双位置探测、quick return 双条件、对角虚部假设）。
