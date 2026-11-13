# aclblasChbmv 算子设计文档（Atlas A2/A3，arch22）

> 模板来源：官方 cann-ops-competitions `resources/design_template.md` 六段结构（套件 AGENTS.md §F0 模板纪律）。
> 数据来源：任务书《aclblasChbmv 算子开发任务书》逐条溯源 + 对 ops-blas master@947f448 现场核验
> （`blas/sbmv/arch22/`（ssbmv 四件套）、`blas/sbmv/arch35/`、`include/cann_ops_blas.h`）。
> 文中所有数值均标注出处；无未经测试的性能数字（S5 实测数据在交付件 03 中回填）。

---

# 需求背景（required）

## 需求来源

11月社区任务：aclblasChbmv 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，
开发分支 master，合入路径 `blas/sbmv/arch22/`（实现，与实数 ssbmv 同族目录）与 `test/sbmv/chbmv/arch22/`（测试）。
在 Atlas A2/A3（arch22）、CANN 9.1.0 上以 Ascend C kernel 直调补齐单精度复数 Hermitian 带状矩阵-向量乘能力（BLAS-2），
对标 cuBLAS `cublasChbmv`（语义参考 Netlib `chbmv.f`）。

## 背景介绍

### 算子功能定位

BLAS-2 Hermitian 带状矩阵-向量乘加：`y = α·A·x + β·y`，A 为 n×n Hermitian 带状复数矩阵
（k 条次对角线 + k 条超对角线），按列主序带状格式存储于 (k+1)×n 数组，仅 uplo 带形三角被引用。

数学公式：

```
y[i] = α × Σ_{j: |i−j| ≤ k} A[i,j] × x[j] + β × y[i]，i = 1..n
UPPER：A(i,j) 存于 A(1+k+i−j, j)，j∈[max(1,j−k), j]；主对角在第 k+1 行
LOWER：A(i,j) 存于 A(1+i−j, j)，j∈[j, min(n,j+k)]；主对角在第 1 行
对角元素虚部假定为 0（仅实部参与）
```

| 接口名 | 对标接口 | 作用 |
|--------|----------|------|
| aclblasChbmv | cublasChbmv | 单精度复数 HBMV：带状 packed 存储 + Hermitian 镜像共轭 + 复数 α/β、正负步长、原地输出 |

### 业界对标

- 对标接口：cuBLAS `cublasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)`。
- Golden 语义来源：Netlib BLAS `chbmv.f`（https://www.netlib.org/blas/chbmv.f），cblas_chbmv
  （CblasColMajor；输入转 float64 真值计算、按输出 FLOAT32 收窄落盘，issue #434）。
- 语义差异点（相对仓内既有实现）：
  1. **packed 带状下标映射（第一坑）**：A(i,j) → A(1+k+i−j, j)（UPPER）/ A(1+i−j, j)（LOWER），
     带内越界区（UPPER 左上 k×k、LOWER 右下 k×k）不被引用；对照 Netlib chbmv.f 逐字核对；
  2. **k > n−1 超界带宽**：按有效带宽 min(k, n−1) 参与计算（任务书 §2.4 k 行）；
  3. **对角虚部为 0**：仅取实部参与计算（Hermitian 假设，golden 同口径）；
  4. **lda 守卫在 int64 域**：`lda ≥ k+1` 的 k+1 计算与比较在 int64 进行（k=INT32_MAX 时 int32 回绕防穿透，任务书 §2.4 lda 行）；
  5. **α=(0,0) 不短路 A/x 读取**：按 y=βy 计算但实现不保证跳过 A/x 读 → 含 Inf/NaN 的 x 不得与 α=(0,0) 组合设用例（任务书 §2.1-4）；
  6. **非法 uplo 返回 INVALID_VALUE**（非 INVALID_ENUM——与仓内 csymv 的 INVALID_ENUM 不同，任务书 §7.5 已确认差异由研发后续统一）。

### 工程形态

- 工程模式：**Host C API + Kernel 直调**。
- 代码落位：`blas/sbmv/arch22/`（该目录**已存在**，含实数 ssbmv 完整四件套——现场核验；知识库旧记录"sbmv/arch22 空"已过期 → **并列加复数文件**）。
- 仓内近亲与复用方式：

| 近亲 | 位置 | 复用方式 |
|------|------|----------|
| 同目录 arch22 实数实现（主复用源） | `blas/sbmv/arch22/ssbmv_{host,kernel,kernel.h,tiling_data}.*` | arch22 工程骨架、**带状 packed 窗口访问与列块 tiling 范式**、校验序、int64 跨度守卫直接平移；复数化三改造：交错复数读、共轭镜像、对角实数化 |
| 同算子 arch35 | `blas/sbmv/arch35/ssbmv_*`（无复数版） | 带状语义参考；kernel 不平移（SIMT） |
| 带状窗口访问（复数现成参考） | 知识库近亲表：gbmv/arch22 有 cgbmv+sgbmv 双 dtype 全套（带状窗口访问） | cgbmv 的复数带状窗口 DMA 范式可交叉参考 |
| 镜像共轭换算 | 同批次 Chemv 设计 | UB 内虚部取反（Muls −1）通道级共轭 |

### 接口规格

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

- **本次任务需在 `include/cann_ops_blas.h` 新增该声明**（任务书 §2.3/§7.3 明确授权：仓内当前无
  aclblasChbmv 声明，现场核验确认；签名按 cuBLAS cublasChbmv 参数序列，int 维度参数，
  与同族 aclblasSsbmv（头文件 L189）口径一致）。红线 3 授权例外，登记例外清单
  （连带文件仅 `include/cann_ops_blas.h` 追加行）。

参数规格表：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|----------|----------|--------------|------|------|
| handle | 库上下文句柄（携带 stream），Host 内存 | aclblasHandle_t | — | 非空句柄；nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | — |
| uplo | 带形三角存储模式，Host 内存 | aclblasFillMode_t | — | ∈ {ACLBLAS_UPPER, ACLBLAS_LOWER}；不在枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书 §7.5，非 INVALID_ENUM） | — |
| n | A 的阶数，Host 内存 | int | int32 | n≥0；n<0 → INVALID_VALUE；n=0 合法 quick return | — |
| k | 带宽（次/超对角线条数），Host 内存 | int | int32 | k≥0；k<0 → INVALID_VALUE；k>n−1 按有效带宽 min(k,n−1) 计算 | — |
| alpha | 复数标量 α 指针，Host 内存 | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；值不校验；α=(0,0) 不保证跳过 A/x 读 | 标量 |
| A | Hermitian 带状矩阵，Device 内存，只读；列主序带状 packed | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；对角虚部假定 0；带内越界区不被引用 | (k+1)×n（lda×n 数组） |
| lda | 带状数组前导维度，Host 内存 | int | int32 | lda ≥ k+1（int64 域计算比较），否则 INVALID_VALUE | — |
| x | 输入向量，Device 内存，只读 | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE | 一维 [n]，物理长度 `1+(n−1)*|incx|` |
| incx | x 步长（可正可负），Host 内存 | int | int32 | ≠0 且非 INT32_MIN；0/INT32_MIN 或 `(n−1)*\|incx\|` 溢出（int64，issue #780/#866）→ INVALID_VALUE | — |
| beta | 复数标量 β 指针，Host 内存；β=(0,0) 时 y 旧值不必有效 | const aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE | 标量 |
| y | 输入兼输出（原地覆写），Device 内存 | aclblasComplex* | COMPLEX64 | n>0 时 nullptr → INVALID_VALUE；β=(0,0) 不豁免非空 | 一维 [n]，物理长度 `1+(n−1)*|incy|` |
| incy | y 步长（可正可负），Host 内存 | int | int32 | 同 incx | — |

### 数学公式

```
y ← α·A·x + β·y（A：Hermitian 带状，仅 uplo 带形三角被引用，对角只取实部）
```

与 §3.2 golden（cblas_chbmv，fp64 真值收窄；对角虚部 0、带内共轭对称、带外为 0）同一形态。

---

# 需求分析（required）

## 需求描述

用 Ascend C SIMD（arch22 Membase）实现 BLAS-2 单精度复数 HBMV：COMPLEX64 带状 packed 矩阵、
uplo 双态、k 带宽扫描、复数 α/β、正负步长、原地输出；实部/虚部分别按 FLOAT32 档判定
（rtol=atol=2^-13，ratio≥0.99，32×ULP 按 golden 值动态计算——issue #434 修订口径）；
性能在 Atlas 800T A2 (910B3) 上单 case 平均耗时 ≤ A100 标杆 / 0.8。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|----------|-----------|
| 1 | 数学定义与可白嫖的代数量 | `y ← α·A·x + β·y`（Hermitian 带状）。golden 白嫖 cblas_chbmv（fp64 真值收窄）；packed 下标映射可对照 Netlib chbmv.f 逐字核对 |
| 2 | 接口原型 | 12 参签名（上节）；**声明须新增**至 `include/cann_ops_blas.h`（任务书 §2.3 授权，现场核验无既有声明） |
| 3 | 参数规格 | 见上表；异常行为逐条对应任务书 §2.4 |
| 4 | 支持数据类型 | 全 **COMPLEX64**（α/β/A/x/y，任务书 §2.4 dtype 面个数=1） |
| 5 | 目标芯片与 arch 映射 | Atlas A2/A3 → **arch22**（`SOC_VERSION=ascend910b`，自验证覆盖 910B3）；950 侧 sbmv arch35 亦仅实数，不受影响 |
| 6 | 精度要求 | 实/虚分别 FLOAT32 档：rtol=atol=2^-13、ratio 0.99、max_abs_error ≤ 1e-2 or 32×ULP（按最大误差点 golden 值动态计算，issue #434）；Inf/NaN 按 §3.2 等效规则 |
| 7 | 性能预期 | 门槛 = `gpu_baseline.csv` 的 `gpu_ms / 0.8`，单 case 判定。典型 case（实测 CSV 读数）：UPPER n=512 k=8 → ≤12.35us；UPPER n=1024 k=16 → ≤19.84us；LOWER n=2048 k=32 → ≤35.01us |
| 8 | 代码架构选型 + 可行性（含已排除的死路） | **SIMD Membase：ssbmv arch22 带状骨架 + 复数化**（packed 下标映射、列带 DMA、镜像共轭、对角实数化）。死路①：arch35 kernel 平移（SIMT + 无复数版）；死路②：全矩阵展开 packed→dense 再算（存储/带宽 ×n，违反"未引用区域不读"精神且性能不可达标）；死路③：逐元素标量下标换算（破坏向量流水） |
| 9 | 评测来源 | 精度 golden：Netlib `cblas_chbmv`（fp64 真值收窄）；性能标杆：`gpu_baseline.csv`（含 GPU 型号与 CUDA/驱动版本信息）或自测报告记录并经确认（任务书 §3.3-3） |

**校验顺序（V1..V8，由任务书 §2.5 校验顺序条款逐字反推，顺序不可调换）**：

| 序 | 条件 | 返回码 |
|----|------|--------|
| V1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| V2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V3 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（quick return：不执行计算） |
| V4 | `k < 0 \|\| uplo ∉ {UPPER, LOWER}` | `ACLBLAS_STATUS_INVALID_VALUE`（uplo 非法也是 INVALID_VALUE，§7.5） |
| V5 | `lda < k+1`（**int64 域**计算 k+1 与比较） | `ACLBLAS_STATUS_INVALID_VALUE` |
| V6 | `incx/incy == 0 或 INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V7 | `n>0 && (alpha/beta/A/x/y 任一 nullptr)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| V8 | `(n−1)*\|incx\| 或 (n−1)*\|incy\| 超 int32 范围（int64）` | `ACLBLAS_STATUS_INVALID_VALUE` |

> V1 判空最优先、V3 quick return 不得先于判空（#765/#861 维护者拍板口径——注意本算子 quick return
> 只豁免"执行计算"，n=0 时 alpha 等指针判空按任务书 §2.5 顺序列在 n=0 之后，故 n=0 无须判空）。

---

# 详细设计（required）

## 算子分析

### 数学公式

`y[i] = α·Σ_{|i−j|≤k} A[i,j]·x[j] + β·y[i]`；仅 uplo 带内读取，
镜像元素 `A[i,j] = conj(A[j,i])`（|i−j|≤k 的对侧带）；对角只取实部。
每行累加项数 ≤ 2k+1——计算量 O(n·k)，带外零开销是本算子性能设计核心。

### 支持数据类型

全 COMPLEX64（分量 FLOAT32，交错存储）；中间乘加保持 FLOAT32，复数乘按
`(ac−bd) + (ad+bc)i` 分解，共轭乘 `(ac+bd) + (bc−ad)i`。
特殊值按 IEEE 传播；α=(0,0) 与 x 含 Inf/NaN 的组合不设精度用例（§2.1-4）。

### 支持形状

| 场景 | 约定 |
|------|------|
| 带状 A | packed (k+1)×n；UPPER 主对角在第 k+1 行、LOWER 在第 1 行；带内越界区不读取 |
| k=0 | 纯对角阵：y[i] = α·diag(i)·x[i] + β·y[i]（退化路径，边界用例） |
| k = n−1 | 满带退化（等价全三角 HEMV，带宽扫描边界） |
| k > n−1 | 按有效带宽 min(k, n−1) 计算 |
| n=0 | 合法 quick return：成功、不访问任何指针 |
| 步长 | 正负步长均生效（负步长反向遍历）；0/INT32_MIN 拒绝 |
| 广播 / 0 维 | 不涉及（BLAS-2） |

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（uplo/k 均为 tiling 字段，镜像方向 kernel 内分支）：

| 变体 | 触发条件 | 职责 | 是否参与性能判定 |
|------|----------|------|------------------|
| **V_CONTIG** | `incx==1 && incy==1` | 连续读 x/y，按带状列块乘累加 | 是（基线 3 条均连续、双 uplo） |
| **V_STRIDED** | 其余（含负步长） | 步长 DMA 读 x/y | 否（精度覆盖） |

数据流（V_CONTIG，按 y 行块分核；UPPER 为例）：

```
   for each y 行块 [i0, i0+cnt)（按核均分）:
      acc(复数) ← β·y[i0..]（β=(0,0) 置零不读 GM）
      // 段一：j < i0−k 之外的镜像下带 → 对每个 j ∈ [max(0,i0−k), i0)：
      //   读 A(j,i)=conj(A(i,j))：列 j 的带内行 1+k+i−j（i∈[i0,i0+cnt)∩[j,j+k]）
      for each 相关列 j（窗口 [i0−k, i0)）:
         GM A 带内段 ──DataCopy──▶ UB aSeg（列内连续！packed 按列存）
         UB 虚部取反（conj）→ 复乘 α·x[j] → Axpy 累加 acc
      // 段二：对角+上带，j ∈ [i0, min(n, i0+cnt+k))
      //   列 j 贡献行 i∈[i0, min(i0+cnt, j+k)]：对角元素（第 k+1 行）取实部
      for each 列 j:
         GM A 列段 ──DataCopy──▶ UB aSeg
         对角元素虚部置 0 → 复乘 α·x[j] → 累加 acc（带内行对齐 offset 处理）
      GM y[i0..] ◀──DataCopy── acc
```

> 关键观察：packed 按列存储，**列内连续**——每列对 y 行块的贡献是该列的一段连续行，
> 与行块的交集可整体 DMA；下标映射 `row_packed = 1+k+i−j`（UPPER）在 host 侧算出每列段的
> packed 起始行与长度，kernel 按（列号 → 段基址/段长）表搬运，避免逐元素换算。

#### host 侧设计

**分核策略**：
- 按 y 长度 n 分核（每核一段连续行块），kernel 侧自算 offset/count（R4，同目录 ssbmv 方案）；
- 每核需读的 A 列窗口 = [i0−k, min(n, i0+cnt+k))——窗口大小随 k 增大；
  k 很大（满带）时窗口≈n，此时 ssbmv 的列块 tiling 退化为近似 HEMV 形态（tileJ 控制 UB 占用）。

**数据分块和内存优化策略**：
- UB 预算 248 KiB（`blas/common/helper/kernel_constant.h:16`）；复数路径 tile 取
  aSeg 2048 复数（32 KiB）量级 + acc 2048 复数 + xBuf 窗口驻留；
- x 驻留策略：窗口列数 ≤ tileJ 时整窗驻留；大 k 分段驻留、外层循环（与 Chemv 同款决策）；
- β=(0,0)：acc 置零不读 GM y；α=(0,0)：按 y=βy 计算（不保证跳过 A/x 读——实现选择仍读，
  与任务书声明一致，注释标注）；
- 尾块/列边界：段长按窗口交集精确计算，无补零污染（对角元素虚部置 0 在段内处理）。

**tilingkey 规划策略**：tiling 字段：uplo、n、kEff（=min(k,n−1)）、lda、perCoreN、remainder、
tileJ、incx、incy、α、β 值（Host 直填）；命名对齐同目录 ssbmv_tiling_data.h。

**数据检测**：按 V1..V8 顺序执行（见需求拆解节）：
- V5 的 lda 比较必须 int64（k=INT32_MAX 时 k+1 int32 回绕为负，int32 比较会穿透——任务书 §2.4 显式要求）；
- V3（n=0）在 V4..V8 之前（§2.5 顺序条款）。

**其他 host 侧约定**：
- workspace：无（y 按核不重叠，α/β Host 值直填 tiling）；
- 同步：异步下发，host 不同步（任务书 §2.5）；
- 日志：`OP_LOGE` 各校验失败分支，`OP_LOGD` tiling 决策（kEff、列窗口范围）。

#### kernel 侧设计

**Init/Process 两阶段**（V_CONTIG，UPPER 视角；LOWER 镜像分支）：

```
Init:   解析 tiling → myOffset/myCount → 列窗口 [jLo, jHi) → InitBuffer(aSeg/xBuf/acc)
Process（每核）:
   1) acc 初始化：β=(0,0) → Duplicate 0；否则 DataCopy GM y + Cmul β
   2) 段一（镜像下带，j ∈ [jLo, i0)）:
        MTE2: DataCopy aSeg ← GM A[packed 行段, j]（列内连续）
        V:    虚部取反（conj）→ Cmul(α·x[j]) → Axpy(acc, …)
   3) 段二（对角+上带，j ∈ [i0, jHi)）:
        MTE2: DataCopy aSeg ← GM A[packed 行段, j]
        V:    对角元素虚部置 0 → Cmul(α·x[j]) → 累加（含行块内偏移对齐）
   4) 写回 GM y[i0..]（incy==1 DataCopy / 步长 DMA）
```

**packed 下标映射（host 预计算，kernel 只按表搬）**：
- UPPER：列 j 的带内矩阵行集 [max(0,j−k), j] → packed 行 [1+k+i−j]，即 packed 段
  起始行 `1 + k + max(0,j−k) − j`、长度 `j − max(0,j−k) + 1`；
- LOWER：列 j 行集 [j, min(n−1, j+k)] → packed 行 `1 + i − j` 起始行 1、长度相应；
- 与 y 行块交集后得到段基址/段长二元组，随 tiling 下发或 kernel 侧按公式自算（首选自算，公式封闭）。

**复数运算分解**：复乘 4 Mul + 2 Add；conj = 虚部通道 Muls(−1)；对角实数化 = 虚部通道置 0——
全部 UB 内向量指令，无逐元素标量循环。

**步长路径**：x/y 负步长物理基址 `(n−1−i)*|inc|`（同族 caxpy arch22 PhysicalIndex 范式）；
`asc_copy_gm2ub_align(src_gap=(|inc|−1)*8)`（复数 8B burst）。

**workspace 布局**：无 workspace。

**同步事件配对**：MTE2→V、V→MTE3/MTE2 显式 SetFlag/WaitFlag；段一/段二 UB 复用加栅栏。

#### 性能优化要点（终态）

| 优化项 | 说明 |
|--------|------|
| 单 kernel + 满核 | y 按核不重叠，无跨核归约 |
| 带外零开销 | 只读带内区域，计算量 O(n·kEff)；k 小（基线 case k=8/16/32）时接近纯带宽 |
| 列内连续 DMA | packed 按列存 → 每列贡献为连续段，整段 DMA，无跨步访存 |
| 镜像共轭向量化 | conj 仅虚部取反（通道 Muls −1），UB 内完成 |
| 双缓冲流水 | aSeg 2 槽，MTE2 与 V 重叠 |
| β=0 免读 | 不读 GM y 旧值 |
| host 零同步 | tiling const 引用传参，无 H2D、无 aclrtMalloc |

**未采用的路线及原因**：
- **packed→dense 展开**：临时缓冲 O(n²)，带宽与存储不可接受，且带外元素读取无意义——死路；
- **逐元素标量下标换算**：热路径标量化，破坏向量流水——改为 host/公式预计算段表；
- **arch35 平移**：SIMT 且无复数版——死路；
- **cgbmv 窗口整段拷贝 + Gather**（知识库备选）：保留为列段 DMA 不可用时的降级方案（gbmv/arch22 带状窗口范式）。

---

## 支持硬件

| 芯片版本 | 支持情况 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | **支持**（本任务目标） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | **支持**（本任务目标） |
| Ascend 950PR / Ascend 950DT | 不涉及（sbmv arch35 当前仅实数 ssbmv，本次不改） |

---

## 算子约束限制

逐条对齐任务书 §2.5：

| 约束项 | 结论 |
|--------|------|
| 参数合法性 | V1..V8 全序：n≥0（n<0 INVALID_VALUE）、n=0 quick return 成功、k≥0（k<0 INVALID_VALUE）、非法 uplo → INVALID_VALUE（§7.5）、lda≥k+1（int64 域）、incx/incy ≠0 且非 INT32_MIN、跨度 int64 守卫、α/β/A/x/y 非空（n>0） |
| 三角引用 | 仅引用 uplo 带形三角；未引用侧共轭对称隐含；对角虚部为 0（仅实部参与）；UPPER 左上 / LOWER 右下 k×k 不读取 |
| 非连续 Tensor 支持 | 不涉及额外 padding 场景：lda/incx/incy 语义内支持；负步长反向遍历 |
| broadcast 规则 | 不涉及（BLAS-2） |
| dynamic shape 要求 | 不要求：n/k 运行时入参 |
| 原地与视图语义 | y 原地覆写，不返回视图 |
| 确定性计算要求 | 不要求 |
| 空 Tensor 与 0 维处理 | n=0 合法 quick return（成功、不执行计算）；α=(0,0)∧β=(1,0) 组合不要求额外短路 |
| 异步执行 | 依赖 aclblasSetStream；读回 Device 结果前须同步 stream |

**已知限制（如实标注）**：
1. 对角元素虚部不参与计算（Hermitian 假设，golden 同口径）；
2. α=(0,0) 时实现不保证跳过 A/x 读取——含 Inf/NaN 的 x 不得与 α=(0,0) 组合设用例（任务书 §2.1-4 已声明）；
3. 非法 uplo 返回 INVALID_VALUE，与仓内 csymv 的 INVALID_ENUM 不同（任务书 §7.5 研发确认的差异，注释标注）；
4. 步长路径不参与性能判定；
5. 本次新增公共头声明（任务书授权例外），波及面仅 `include/cann_ops_blas.h` 追加行。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | `cblas_chbmv`（Netlib 复数实现，CblasColMajor；输入转 float64 真值计算、按输出 FLOAT32 收窄落盘，issue #434）；对角取实部、带内共轭对称补全、带外为 0 | 任务书 §3.2 |
| 比对方式 | y 全向量，实部/虚部分别按 FLOAT32 档判定 | 任务书 §3.2 |
| 普通精度（rtol/atol） | 2^-13 ≈ 1.2e-4 | 任务书 §3.2，mixed_tolerance_standard（opbase commit 28a1ce52） |
| required_matched_ratio | 0.99 | 同上 |
| max_abs_error_limit | 1e-2 or 32 × ULP（**32×ULP 按最大误差点 golden 值动态计算**，issue #434 修订口径） | 任务书 §3.2 |
| 判定式 | 逐元素 `\|actual − golden\| ≤ atol + rtol*|golden|`（实/虚分别） | 任务书 §3.2 |
| Inf/NaN 特殊值 | 同为 NaN / 同为同号 Inf / 一方 NaN 另一方 ±Inf → 等效判过；一方有限另一方非有限 → 判不过；非有限值不计入统计、单独记录 | 任务书 §3.2 |
| α=(0,0) 特例 | 实现不保证跳过 A/x 读——含 Inf/NaN 的 x 不与 α=(0,0) 组合设用例 | 任务书 §2.1-4 |
| 性能判据 | 平均单次耗时 ≤ A100 标杆 / 0.8（单 case 判定），COMPLEX64 场景 | 任务书 §3.3 |
| 性能采集口径 | msprof op 采集 task duration (us)；5 次 warmup 不计样本；有效样本 ≥10 取算术平均；仅 SUCCESS 计入 | 任务书 §3.3 |
| 性能标杆 | `gpu_baseline.csv`（含 GPU 型号与版本信息）；若包内未提供则在自测报告记录并经确认 | 任务书 §3.3-3 |
| 内存标准 | 不涉及（§3.4） | 任务书 §3.4 |

## 测试概览

- 用例包：`test_cases/{chbmv_test.csv, gpu_baseline.csv, verify_accuracy.py, verify_performance.py, README.md}`；
- 复现入口：交付件 03「自验证步骤说明」（A2 `--soc ascend910b`，A3 `--soc ascend910_93`）；
- 测试代码落位 `test/sbmv/chbmv/arch22/`（CSV 驱动 GTest 工程，参照 arch35 模式新建），并在 `test/sbmv/chbmv/CMakeLists.txt` 补齐 arch22 GTest 注册（`ops_blas_add_gtest_tests(${OPS_BLAS} chbmv_test)`）；
- 用例覆盖（§3.5-4）：uplo 全组合 × 尺寸扫描（0/1/小质数/2 的幂及 ±1/非对齐/大规模）；带宽扫描 k ∈ {0, 1, 小带宽, k=n−1 满带, 典型大带宽} 及 k>n−1 有效带宽截断；α/β 特殊值 (0,0)/(1,0)/纯虚数/大值；lda 紧凑（k+1）与多档 padding；incx/incy ±1/±2/±3 及零步长负向；A 按 Hermitian 带状约束填充（对角虚部 0、带内共轭对称、带外 0）；Inf/NaN；负向含空指针、非法 uplo、非法前导维、负维度、零步长、INT32_MIN 步长（期望码按 §3.5-4 清单）；
- 正态分布补齐：gen_csv.py 默认仅均匀分布，§3.5 要求的 `GAUSS_NORM_5_2` 正态用例（μ∈[-5,5]、σ∈[0.1,2]，实/虚独立采样）需扩展生成器补充并纳入精度自验证，自测报告说明条数；
- 负向用例缓冲区分配规则：期望非 SUCCESS 的用例不得按 `(n−1)*|inc|` 预分配，以单元素 dummy 或 NULLPTR 占位；quick return 用例同样不按步长分配。

## 兼容性分析

- **公共头新增声明（任务书 §2.3/§7.3 授权）**：`include/cann_ops_blas.h` 追加 `aclblasChbmv` 声明
  （与同族 aclblasSsbmv 相邻），不修改既有行——无 ABI 破坏；红线 3 例外登记（连带文件：仅该头文件）；
- **对 arch35 影响**：无。`blas/sbmv/arch35/ssbmv_*` 保持不变；
- **对 arch22 同目录 ssbmv 影响**：无。并列复数新文件，接口独立、无共享可变状态；
- **测试构建影响**：`test/sbmv/chbmv/CMakeLists.txt` 补 arch22 GTest 注册；arch35 分支不变；
- **文档影响**：`blas/sbmv/README.md` 产品支持表新增 aclblasChbmv「Atlas A2/A3：支持」，并同步约束说明
  （packed 下标映射、k>n−1 有效带宽、lda int64 守卫、非法 uplo=INVALID_VALUE 与 csymv 的差异）。
