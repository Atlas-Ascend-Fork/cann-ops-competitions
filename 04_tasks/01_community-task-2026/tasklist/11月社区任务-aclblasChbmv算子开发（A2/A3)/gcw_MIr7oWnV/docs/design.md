# aclblasChbmv 算子设计文档（Atlas A2/A3）

> 状态：**设计稿 + 首版实现已落地**。`aclblasChbmv` 的库与测试可执行文件（`chbmv_test` / `chbmv_perf`）已在 Atlas 800 A2（8×910B3，CANN 9.2.0）上编译通过；精度与性能实测进行中，本文所有性能数字仍为任务目标或估算，不是实测结果。
>
> 本文的算法、索引与 tiling 公式已用四个独立模型对照验证（均为本机可复现）：索引语义 33000 组、kernel 级寻址（含 NaN 毒值哨兵）34304 组、golden 1680 组、随包 1200 条用例的 tiling/UB 预算；日志见 `work/model/` 与 `logs/`。
>
> 依据：本任务随包任务书 `work/input/task_doc.md`；随包用例 `work/input/test_cases/`（`chbmv_test.csv` 1200 条、`gpu_baseline.csv` 200 条）；`cann/ops-blas` 源码基线 `5bd9514`（本地 `repo/ops-blas`）；设计模板 `04_tasks/01_community-task-2026/resources/design_template.md`；性能清单 `shared/ops-blas-perf-playbook.md`。
>
> 提交目录：`04_tasks/01_community-task-2026/tasklist/11月社区任务-aclblasChbmv算子开发（A2/A3)/gcw_MIr7oWnV/docs/design.md`

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务要求在 Atlas A2/A3 的 `arch22` 目录新增单精度复数（complex64）Hermitian 带状矩阵-向量乘 `aclblasChbmv`，接口与语义对齐 cuBLAS `cublasChbmv`（语义参考 Netlib `chbmv.f`），使用 Ascend C kernel 直调方式与 ops-blas 的 handle/stream 体系。目标是补齐 ops-blas 在复数带状 BLAS-2 上的能力空白。任务截止 2026-11-26；本任务只做 A2/A3（`arch22`），不做 950（`arch35`）。

## 背景介绍

### 对标实现与仓内现状

| 对象 | 现状 | 与本任务的关系 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 有 `aclblasSsbmv`（实数对称带状），**无** `aclblasChbmv` | 需新增声明，与其他产品线共用，禁止私有平行接口 |
| `blas/sbmv/arch22/ssbmv_*` | 实数对称带状 MV 的完整 A2/A3 实现（host 713 行 / kernel 1908 行） | 结构参考：行块切分、整带 span 常驻 UB、`2k+1` 对角 pass、UB 预算公式、核数成本模型、宽带列形式 |
| `blas/gbmv/arch22/cgbmv_*` | complex64 一般带状 MV（kernel 1525 行） | 复数实现参考：AoS→re/im 拆分 `Gather`、`DataCopyPad` 复数列搬运、`BuildPackTable` 交织写回 |
| `blas/hpr/arch22/chpr_*` | complex64 Hermitian 打包 rank-1 更新 | Hermitian 语义与状态码参考：非法 `uplo` → `INVALID_VALUE`、`incx == INT_MIN` 校验顺序 |
| `blas/symv/arch22/ssymv_*` | 实数对称 MV | 复数同族 `csymv` 只有 arch35，arch22 无复数对称/带状实现 |

结论：`aclblasChbmv` 是**新增**接口与新增实现；仓内没有可复用的 arch22 复数带状内核，必须新写，但可复用上述结构、复数拆分技巧与 tiling 方法论。

### 目标能力与输入输出

| 对象 | 类型/位置 | 形状与约束 | 作用 |
| --- | --- | --- | --- |
| `handle` | Host 指针 | 非空，携带 stream | 绑定执行流；为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | Host 属性 | `ACLBLAS_UPPER`(121) / `ACLBLAS_LOWER`(122) | 选择引用哪一半带 |
| `n` | Host 标量 | `n >= 0`；`n = 0` 合法 quick return | 方阵阶数、x/y 逻辑长度 |
| `k` | Host 标量 | `k >= 0`；`k > n-1` 时按有效带宽 `min(k, n-1)` 参与计算 | 次/超对角线条数 |
| `alpha`, `beta` | Host 上的 `aclblasComplex*` | 非空；实部/虚部取 FLOAT32 全集 | 标量乘数 |
| `A` | Device `complex64` | 列主序带存储 `lda × n`，`lda >= k+1` | 只读 Hermitian 带状矩阵 |
| `x` | Device `complex64` | 物理容量 ≥ `1+(n-1)|incx|` | 只读向量 |
| `y` | Device `complex64` | 原地输出，物理容量 ≥ `1+(n-1)|incy|` | 输入旧值、原地覆写 |

`aclblasComplex` 为 `include/cann_ops_blas_common.h` 中相邻两个 `float`（`real`, `imag`），即 8 字节 AoS 元素。

# 需求分析（required）

## 需求描述

接口签名（与 cuBLAS `cublasChbmv` 参数序列一致，维数参数为 `int`）：

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

数学语义：`y ← alpha · A · x + beta · y`，其中 `A` 为 `n×n` Hermitian 带状矩阵（含 `k` 条次对角线与 `k` 条超对角线），满足 `A^H = A`；`alpha`、`beta`、`A`、`x`、`y` 均为 complex64。

## 需求拆解与验证对照

| 编号 | 要求及依据 | 设计落点 | 验证方式 |
| --- | --- | --- | --- |
| R1 | complex64、Hermitian、列主序带存储、`uplo` 两分支；§2.1/§2.4 | 3.1 索引推导、3.4 对角 pass 表 | Python 索引模型（33000 组）+ 随包 1000 条精度用例 + `cblas_zhbmv` golden |
| R2 | 只引用 uplo 指定半带；未引用 `k×k` 区域不读；对角虚部按 0；未存储侧共轭补全；§2.1.2 | 3.1.3 下标不等式证明、3.4.2 对角特化 | 模型 + NaN 哨兵填充未引用区（越界读即暴露） |
| R3 | 校验顺序：handle → n≥0 → n=0 quick return → 其余；§2.5 | 3.3.1 | 16 条负向 CSV + `handle` 单测 + n0 叠加非法参数用例 |
| R4 | `lda >= k+1`、`k+1` 在 int64 域；非法 `uplo` → `INVALID_VALUE`；`inc = 0/INT32_MIN`、`(n-1)|inc| > INT32_MAX` → `INVALID_VALUE`；§2.4/§2.5 | 3.3.1 int64 校验 | `TC_ED_251/252/256/257/260~263` 等 16 条负向用例 |
| R5 | `beta=(0,0)` 不读 y 旧值；`alpha=(0,0)` 不保证跳过 A/x 读取；§2.1.4 | 3.3.2 `betaZero/alphaZero` 分支 | `TC_ED_242~245`、`beta0_y_nan` |
| R6 | 精度：COMPLEX64 按 FLOAT32（rtol=atol=2^-13、matched_ratio ≥ 0.99、max_abs ≤ 1e-2 或 32ULP）、实/虚分别判；§3.2 | 4.1 | 随包 1000 条精度 + 正态分布补充集 |
| R7 | 性能：Avg ≤ GPU 标杆 / 0.8；msprof warmup 5 + ≥10 采样；§3.3 | 5.1–5.4 | `[PERF] avg_us` GTest + msprof 交叉验证 |
| R8 | 异步执行、y 原地覆写、不支持任意视图/batch/广播；§2.5 | 3.3.3 | 只同步 handle 绑定的 stream |
| R9 | A2/A3 支持，CANN 9.1.0；§3.1 | 3.5 | 至少一种款型真机自验 |
| R10 | 交付件：设计文档、测试代码+README、自测报告、个人仓+README、`task_submission/`；§4 | 4.3 / 交付说明 | 逐项核对 |

## 边界与错误码

| 输入 | 期望行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（最优先，quick return 不得先于判空） |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE`（先于 uplo 判定：`neg_n_invalid_uplo` 亦返回该值） |
| `n == 0` | `ACLBLAS_STATUS_SUCCESS`，不下发 kernel；**叠加非法 uplo/k/lda/incx/空指针仍为 SUCCESS**（`TC_ED_232~238`） |
| `uplo` 非 UPPER/LOWER | `ACLBLAS_STATUS_INVALID_VALUE`（**不是** `INVALID_ENUM`，与 `aclblasCsymv` 的现状刻意不同，按任务书 §7.5） |
| `k < 0` | `INVALID_VALUE` |
| `lda < k+1`（int64 比较） | `INVALID_VALUE` |
| `incx/incy == 0` 或 `== INT32_MIN` | `INVALID_VALUE` |
| `(n-1)·|inc| > INT32_MAX`（int64 计算） | `INVALID_VALUE`（注意：仓内 `ssbmv` 用 `UINT32_MAX` 判据，本算子按任务书用 int32 判据） |
| `alpha/beta/A/x/y == nullptr` 且 `n > 0` | `INVALID_VALUE`；`alpha = 0` **不豁免** A/x 非空（与 `ssbmv` 的现有豁免不同，按任务书 §2.4） |
| `beta == 0` | 不读 y 旧值，但 `y` 仍须非空 |
| `alpha == 0` | 允许短路为 `y = beta·y`；golden（Netlib）同样短路，故行为一致 |

# 详细设计（required）

## 算子分析

### 数学公式

以下均用 **0 起始**下标。Hermitian：`A(i,j) = conj(A(j,i))`，对角 `A(i,i)` 为实数（实现只取存储值的实部）。

带存储（列主序，`lda × n` 个 complex64，仅 `uplo` 指定的半带被引用）：

```text
UPPER: A(i,j) 存于 A_storage(k + i - j, j)，  0 <= i <= j, j - i <= k
       对角（i == j）位于第 k 行（0 起算），主对角在第 k+1 行（1 起算）
LOWER: A(i,j) 存于 A_storage(i - j, j)，      j <= i, i - j <= k
       对角位于第 0 行
```

线性下标（complex 元素单位）：`UPPER: j*lda + (k+i-j)`，`LOWER: j*lda + (i-j)`。

未被引用的区域：

```text
UPPER: 列 j < k 且 t < k-j 的左上 k×k 三角（存储有槽位但不对应任何矩阵元素）
LOWER: 列 j > n-1-k 且 t > n-1-j 的右下三角
```

有效带宽 `kEff = min(k, n-1)`：`k > n-1` 时超出矩阵的带没有元素，**但存储布局仍按声明的 k 定位对角**，因此下标公式用 `k`、循环上限用 `kEff`。

### 逐输出行的展开

UPPER，输出行 `r`：

```text
y[r] = sum_{d=0..kEff, r+d<n}  A(r, r+d) · x[r+d]        (d>0: 存于列 r+d, t = k-d)
     + sum_{d=1..kEff, r-d>=0} conj(A(r-d, r)) · x[r-d]   (存于列 r,   t = k-d)
```

LOWER，输出行 `r`：

```text
y[r] = sum_{d=0..kEff, r-d>=0}  A(r, r-d) · x[r-d]        (存于列 r-d, t = d)
     + sum_{d=1..kEff, r+d<n} conj(A(r+d, r)) · x[r+d]    (存于列 r,   t = d)
```

两种 uplo 的每一项都写成 `y[r] += a_storage(r*lda + cE) · x[r+e]`，其中 `e` 与 `cE` 只依赖 `d`、`k`、`lda`：

| uplo | 项 | e（x 偏移） | cE（下标减 r·lda） | 是否取共轭 | 对角？ |
| --- | --- | --- | --- | --- | --- |
| UPPER | d = 0 | 0 | k | 否（只取实部） | 是 |
| UPPER | own | +d | d·(lda-1) + k | 否 | 否 |
| UPPER | mirror | -d | k - d | **是** | 否 |
| LOWER | d = 0 | 0 | 0 | 否（只取实部） | 是 |
| LOWER | own | -d | -d·(lda-1) | 否 | 否 |
| LOWER | mirror | +d | d | **是** | 否 |

即：**两条 uplo 分支只差一张符号表**，kernel 因此可以共用同一套 pass 循环（与 `ssbmv` 的 `DiagPass` 同构）。

### 下标安全性（为什么不会读到未引用区）

- UPPER own：列 `j = r+d`，`t = k-d`；该列有效 `t >= max(0, k-j) = max(0, k-r-d)`，而 `k-d >= k-r-d` 对 `r >= 0` 恒成立 ⇒ 落在被引用区间。
- UPPER mirror：列 `r`，`t = k-d`；有效需 `t >= max(0, k-r)` ⇔ `r >= d`，由行区间 `lo = max(c0, d)` 保证。
- LOWER own：列 `r-d`，`t = d`；有效需 `0 <= d <= min(k, n-1-(r-d))`，第一个不等式由 `d>=1`、第二个等价于 `r <= n-1`，恒成立。
- LOWER mirror：列 `r`，`t = d`；有效需 `d <= min(k, n-1-r)` ⇔ `r <= n-1-d`，由行区间 `hi = min(c1, n-d)` 保证。

因此 kernel 的 `2kEff+1` 条对角访问**只**落在被引用的带内/带内对角上；不存在"读进来再乘 0"的路径。

### 支持数据类型与形状

仅 COMPLEX64（实部/虚部各 FLOAT32）。`A` 容量至少 `lda*n` 个 complex；`x/y` 物理容量 ≥ `1+(n-1)|inc|`。支持紧凑与 padded `lda`、正负步长、`k = 0`、`k = n-1`（满带退化）、`k > n-1`。不支持任意非连续视图、batch、广播；`y` 与 `A`/`x` 重叠不在支持范围（BLAS 惯例）。

## 实现方案与文件落点

| 文件 | 内容 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增 `aclblasChbmv` 声明（紧随 `aclblasSsbmv`） |
| `blas/sbmv/arch22/chbmv_tiling_data.h` | `ChbmvTilingData`（host/kernel 共享）与常量 |
| `blas/sbmv/arch22/chbmv_kernel.h` | `chbmv_kernel_do` 签名 |
| `blas/sbmv/arch22/chbmv_kernel.cpp` | 设备侧：`chbmv_kernel` + 两个计算形态 |
| `blas/sbmv/arch22/chbmv_host.cpp` | Host：`ValidateChbmvArgs` / `LaunchChbmvKernel` / `aclblasChbmv` |
| `blas/sbmv/README.md` | 增加 Chbmv 章节（产品支持表标注 A2/A3 支持） |
| `docs/zh/api_list.md`、根 `README.md` | 接口清单/公告增量 |
| `test/sbmv/chbmv/**` | CMake、参数解析、`cblas_zhbmv` golden、NPU wrapper、arch22 GTest+CSV、perf、README |

`blas/CMakeLists.txt` 以 GLOB 收集 `blas/*/*/arch22/*.cpp`，新增实现文件无需改 CMake；测试侧新建 `test/sbmv/chbmv/CMakeLists.txt` 调用 `ops_blas_add_gtest_tests(${OPS_BLAS} chbmv_test)`（`test/CMakeLists.txt` 按家族子目录自动发现）。

## Host 侧设计

### 校验顺序（冻结）

```text
1  handle == nullptr                    -> HANDLE_IS_NULLPTR
2  n < 0                                -> INVALID_VALUE
3  n == 0                               -> SUCCESS（不看 k/lda/指针）
4  uplo ∉ {UPPER, LOWER}                -> INVALID_VALUE
5  k < 0                                -> INVALID_VALUE
6  (int64)lda < (int64)k + 1            -> INVALID_VALUE      // k+1 在 int64 域，k=INT_MAX 不回绕
7  incx == 0 || incx == INT32_MIN       -> INVALID_VALUE
8  incy == 0 || incy == INT32_MIN       -> INVALID_VALUE
9  (int64)(n-1) * |incx| > INT32_MAX    -> INVALID_VALUE      // issue #780/#866 口径
10 同理 incy
11 alpha == nullptr                     -> INVALID_VALUE
12 beta  == nullptr                     -> INVALID_VALUE
13 A/x/y == nullptr                     -> INVALID_VALUE      // alpha == 0 不豁免
```

`uplo` 非法返回 `INVALID_VALUE`，与仓内 `aclblasCsymv`（`INVALID_ENUM`）不同，按任务书 §7.5 执行并在文档中显式记录该差异。

短路：`alpha == 0 && beta == 1` → 直接 SUCCESS（不下发 kernel，等价于恒等）。`alpha == 0` 其余情况走 `betaScale` 形态（`y = beta·y`，不读 A/x）。

### 形态选择与 tiling

| 形态 | 条件 | 结构 |
| --- | --- | --- |
| `BETA_SCALE` | `alphaR == 0 && alphaI == 0` | 单 pass 对 y 做复数 `beta` 缩放；`beta==0` 时直接写 0 不读 y |
| `DIAG` | 整带 span `lda·(chunkRows+kEff)+kEff+1`（complex）连同其余缓冲能装进 UB | 行块 + 整带 span 常驻 UB + `2kEff+1` 条对角 pass（主路径） |
| `COLUMN` | `DIAG` 装不进 UB（宽带 `k ~ n`、`k+1 >> 行块`） | 行块 + 逐列扫描：列段 own 累加 + 列内 mirror 归约（`ssbmv` bandForm 同构） |

核数与分块：

1. `cap = min(GetAivCoreCount(), max(1, ceil(n/8)))`。
2. `DIAG` 形态用成本模型（step 1 初值，后续在 A3 上 A/B 校准）：
   `cost(c) = 0.40·c + 0.134·(2kEff+1)·ceil(ceil(n/c)/64) + bytes/bw(c)`，
   `bytes = 8·n·lda + 16·n`，`bw(c) = min(74e9·c, 2367e9)`；取 cost 最小者。
3. `COLUMN` 形态取 `min(cap, ceiling(n/64))`（每列固定开销大，按行均分即可）。
4. `blockRows = CeilAlign8(ceil(n/cores))`，每核 `r ∈ [blockIdx·blockRows, min(n, ...))`。
5. `chunkRows`：`DIAG` 从 `min(blockRows, CAP=2048)` 二分求出能装进 UB 的最大值（谓词随行数单调）；`COLUMN` 同样二分。**至少 1 行**，因此任何合法 shape 都有可下发路径，正常路径不会返回 `NOT_SUPPORTED`。

UB 预算（float 数量，host 与 kernel 逐项一致）：

实现里 host 把每一项的**浮点个数**算好后直接写进 tiling（`accRows/xWinLen/spanFloats/tblLen/ioLen/colBuf/yStageFloats/ubFloats`），
kernel 只按这些数切片，两侧不存在公式漂移。全部用 64 位计算后再与预算比较（否则超大 `k`/`lda`
的候选会回绕成小值骗过准入）。

```text
公共（两种形态都有）:
  8·accRows        // accRe/accIm/aRe/aI/xRe/xIm 六个行平面 + pack 交织表（2·accRows）
  + 2·tblLen       // tblA/tblX 两张 Gather 字节偏移表，tblLen = max(accRows, xWinLen, colBuf+8)
  + 2·ioLen        // yIo：AoS 暂存（x 窗口/ y 读入/ y 写出共用），ioLen = max(accRows, xWinLen)
DIAG 追加:
  + 2·xWinLen      // x 窗口 re/im 平面，xWinLen = chunkRows + 2·kEff + 8
  + spanFloats     // 2·(8·lda + Align8(lda·(chunkRows+kEff) + k + 1)) + 16
COLUMN 追加:
  + 9·(colBuf+8)   // colRaw(2) + colRe/colIm/colProdRe/colProdIm(4) + xRaw(2) + redWork(1)
  + 40             // redRe/redIm/redAccRe/redAccIm + off8
yStage（仅 incy != 1）: 2·(Align8((yStageRows-1)·|incy|+1) + 16)
```

`accRows = Align8(chunkRows) + 16`，`colBuf = 1024`。host 与设备 UB
（`PlatformAscendCManager::GetCoreMemSize(UB)`）比较时留 4KB 安全余量；`chunkRows` 由二分求最大可行值，
1 行一定可行，因此正常路径不会 `NOT_SUPPORTED`（该返回码只作为结构性兜底）。

### 下发与资源

- 单 kernel 下发：`chbmv_kernel<<<numBlocks, nullptr, stream>>>`，tiling 按值传递（`ChbmvTilingData`），指针作为 kernel 参数。
- 无 workspace、无 `aclrtMalloc`/`aclrtMemcpy`、无 stream 同步；结果由调用方同步后读回（任务书 §2.5 异步语义）。
- Host 侧每次调用只做 O(1) 计算；UB 容量与 AIV 核数在首次调用后缓存（`platform_ascendc` 单例），小 shape 不重复查询。
- `alpha`/`beta` 在 Host 读值后放进 tiling，不把 Host 地址传给 Device。

## Kernel 侧设计

### 总体流程

```text
chbmv_kernel(A, x, y, tiling):
  TPipe pipe;                       // R3：TPipe 在入口创建，指针传入 Kernel 对象
  ChbmvKernel op; op.Init(A, x, y, &pipe, tiling); op.Process();
```

`Init`：取 tiling 字段、算 `kEff`、按形态 `InitBuffer`（一块 `TBuf` 手工切分以减少 `InitBuffer` 次数）、`FetchEventID`、建立偏移表（`CreateVecIndex` + `Muls`）；`r0 = blockIdx·blockRows`。

`Process`（DIAG）：

```text
for chunk [c0, c1) in block:
    装载 x 窗口 [c0-kEff, c1+kEff)（AoS，越界不搬）→ 2 条 Gather 拆成 re/im 平面
    装载整带 span（UPPER 列 [c0, min(n, c1+kEff)) / LOWER 列 [max(0,c0-kEff), c1)，AoS 连续）
    一次 MTE2→V 同步
    d = 0: acc = Re(A_diag) ⊗ x            （2 条 Mul，无需清零累加器；对角只取实部）
    for d = 1..kEff: pass(+d); pass(-d)    （每条 4 Gather + 1 Muls(共轭符号) + 5 MulAdd/Muls）
    收尾：× alpha → + beta·y → pack 交织 → 写回 y
```

### 行块与对角 pass

单条 pass 的地址（`row0 = c0 + loL`，`loL = (loRaw-c0) & ~7`，保证累加器 32B 对齐）：

```text
A 侧: Gather(aRe, span, tblA, (spanOff + cE + row0·lda - spanCol0·lda)·8B, cnt)
      Gather(aI, span, tblA, 同上 + 4B, cnt)                    // 同 base，偏移 +4B 取虚部
x 侧: Gather(xRe, xWinRe, tblX, (row0 + e - winLo)·4B, cnt)     // 平面内步长 1 float
      Gather(xIm, xWinIm, tblX, 同上, cnt)
计算: aI' = s·aI（s = +1 own / -1 conj，1 条 Muls）
      accIm += aRe·xIm ; accIm += aI'·xRe
      xIm ← -xIm（1 条 Muls，就地取负）                          // 于是不需要第二份虚部平面
      accRe += aRe·xRe ; accRe += aI'·xIm
```

有效行区间（`e >= 0: [c0, min(c1, n-e))`；`e < 0: [max(c0, -e), c1)`）保证每条 pass 只覆盖真实存在的元素。`loRaw` 到 `loL` 之间被对齐吃掉的至多 7 lane，其 `a` 与 `x` 都在 UB 内被显式 `Duplicate(0)`，于是乘积恰为 `0⊗0 = 0`（**不是** `0·Inf`），不会污染累加器。

`d = 0` 的对角 pass 只取存储值的实部：只 Gather `aRe`，用 `Mul`/`MulAddDst` 两条指令完成 `accRe += aR·xR`、`accIm += aR·xI`，既不读对角虚部（对未初始化/NaN 虚部免疫），也比通用 pass 少一半指令。

### 复数运算与精度

- 全部按 FLOAT32 实/虚分量计算，不降精度；复数乘加 `(aR·xR - aI·xI) + i(aR·xI + aI·xR)` 由 4 条 `MulAddDst` 完成（融合乘加，减少 UB 往返）。
- 累加顺序：同一输出行的 `2kEff+1` 项按 `d = 0, +1, -1, +2, -2, ...` 顺序累加，与 Netlib 的逐列顺序不同，属允许的非 bit-exact 差异（任务书 §3.2 补充说明）。
- `alpha`/`beta` 在 chunk 收尾统一施加一次（不在每条 pass 内乘），减少重复乘法并保持与 golden 同量级的舍入。
- 特殊值：`beta == 0` 完全不读 y（避免把 y 的 Inf/NaN 带进结果）；`alpha == 0` 走 `BETA_SCALE`；`d = 0` 不读对角虚部。三处都直接对应任务书的语义要求。

### x / y 搬入搬出

- `incx == 1`：x 窗口按 `DataCopyPad` 连续搬入，越界前后由 `leftPadding/rightPadding` 补 0；随后 2 条 `Gather` 拆 re/im 平面。
- `|incx| > 1`（含负）：按 `DataCopyPad{blockCount=count, blockLen=8B, srcStride=(|inc|-1)·8B}` 整段搬入物理顺序；负步长先搬入再 `Gather` 反序，不做逐元素 `GetValue`（R1）。
- `incy == 1`：`alpha`/`beta` 合成后一次 `DataCopyPad` 写出。
- `|incy| > 1`（含负）：整段读入 `yStage`（保留非本 chunk 元素）→ 用 `Gather` 取本 chunk 的 re/im → 合成 → `Gather` 交织回 → 整段写回；负步长用反序索引表。
- 所有步长路径共用同一份 `XPhysicalPos/YPhysicalPos` 约定：`inc>0 → i·inc`，`inc<0 → (n-1-i)·|inc|`。

### 宽带退化（COLUMN）形态

每个核拥有行块 `[c0,c1)`，对列 `j ∈ [c0, min(n,c1+kEff))`（UPPER）或 `[max(0,c0-kEff), c1)`（LOWER）：

1. 取该列**被引用的整段**（UPPER `t ∈ [max(0,kEff-j), k]`，LOWER `t ∈ [0, min(kEff, n-1-j)]`），按 `colTile` 分块 `DataCopyPad` 搬入 AoS 段；own 与 mirror 都要用这一段，因此每列只搬一次。
2. own（与行块相交的连续子段）：`Gather` 拆 re/im → 4 条 `Muls`/`Axpy` 得到 `A段·x[j]` → **显式清零对齐头部 lane 后再 `Add` 进累加器**（顺序很重要：`0·Inf = NaN`，所以必须在乘法之后清零，而不是像 `ssbmv` 那样在乘法之前清源）。
3. mirror（仅 `j ∈ [c0,c1)`，作用到 `y[j]`）：`Gather` 拆 re/im + 共轭 → `Mul` → `ReduceSum`（re/im 各一次）→ 标量加进 `acc[j-c0]`。
4. 收尾同 DIAG（`alpha`/`beta`/交织写回）。

`k = 0` 时 mirror 为空，只走 own。

### 同步与流水

| 位置 | 事件 | 说明 |
| --- | --- | --- |
| chunk 装载后 | `MTE2→V` | span 与 x 窗口两条搬运合并成一次等待 |
| 收尾写回前 | `V→MTE3` | 累加器 → GM |
| 多 chunk 复用缓冲 | `V→MTE2`、`MTE3→V`、`S→V`/`V→S`（仅用到时） | 每核通常只有 1 个 chunk，该组 flag 不进快路径 |
| pass 之间 | **无同步** | 同一累加器、同一向量流水，天然有序 |

`DIAG` 形态每条 pass 无任何跨流水等待；整 chunk 只有两次等待。双缓冲（MTE2 与上一 chunk 计算重叠）作为后续优化项（见 §5.3 E2），首版不做，避免首次上板把"正确性"和"流水"两个变量混在一起。

## 支持硬件

| 产品 | 设计目标 | 验证状态 |
| --- | --- | --- |
| Atlas A2 训练/推理系列（Atlas 800 A2 / 910B3） | `arch22` 支持；CANN 9.1.0（实测环境为 9.2.0） | **已编译通过**（`build.sh --ops=chbmv --soc=ascend910b`，`chbmv_test` / `chbmv_perf` 链接成功）；精度与性能实测进行中 |
| Atlas A3 训练/推理系列 | 同一 `arch22` 接口与内核 | 未实测；同一份设备代码 |

## 算子约束限制

- `y` 与 `A`/`x` 不得重叠（BLAS 语义）；不支持任意非连续视图、batch、广播、确定性逐位复现。
- 标量指针在 Host 内存，矩阵/向量在 Device 内存；只允许在 `aclblasSetStream` 绑定的 stream 上异步执行。
- 非法参数必须在 Host 阶段返回，不允许为通过负向用例而让 Device 越界。
- `lda*n`、`(n-1)|inc|`、`k+1` 等全部在 64 位域计算或校验，避免 int32 回绕。

# 可维可测分析

## 精度标准

| 项 | 口径 | 来源 |
| --- | --- | --- |
| golden | **fp64 真值**（Netlib `chbmv` 的列扫描语义：对角只取实部、未存储侧共轭补全），double 累加后按 FLOAT32 收窄落盘；实现于测试工程 `aclblasChbmv_cpu`，不依赖 cblas 复数 BLAS-2 入口（与仓内 cdotu 等复数算子一致），需要时可直接换调 `cblas_zhbmv`，口径不变 | 任务书 §3.2/§3.5 |
| dtype 档 | FLOAT32：`rtol = atol = 2^-13 (1.2207e-4)`，`required_matched_ratio = 0.99`，`max_abs_error_limit = 1e-2` 或最大误差点的 `32×ULP(golden)` | 任务书 §3.2；`test/frame/mixed_tolerance.h` 的 `ACL_FLOAT` 实测为同一组值 |
| 逐元素 | `|actual - golden| <= atol + rtol·|golden|`，实部、虚部分别判定 | 任务书 §3.2 |
| 非有限值 | 同 NaN / 同号 Inf / 一方 NaN 一方 ±Inf 视为等效；有限 vs 非有限判失败；非有限元素不计入 matched_ratio 与 max_abs_error，单独统计 | 任务书 §3.2 |
| 边界语义 | `beta = 0` 时 golden 不读 y 旧值；`n = 0` 不执行计算；`alpha = 0 && beta = 1` 不要求短路 | 测试包 README |

判定器复用仓内 `test/frame/verify.h` 的 `MixedTolerance`（`applyMixedTolerance(cfg, ACL_FLOAT, ...)`），CSV 的 `mere_threshold`/`mare_multiplier` 仅解析、不参与判定（测试包 README 明确保留列）。

## 性能标准

| 项 | 口径 |
| --- | --- |
| 设备 | Atlas 800 A2（8×910B3 64GB，多人共享），与任务书要求的性能设备一致；实测时记录 CANN 9.2.0、驱动/固件版本与单卡独占情况 |
| 采集 | `msprof op` 单算子 `Task Duration(us)`；自带 5 次 warmup 不计入；随后 ≥10 次有效采样取算术平均；仅 `SUCCESS` 样本计入 |
| 判据 | Avg ≤ `gpu_baseline.csv` 的 `gpu_ms / 0.8`（单 case 同判据） |
| GTest 侧 | GTest 打印 `[PERF] case=<case_name> ... avg_us=<数值>`（aclrtEvent 计时、warmup 后连续下发取均值、含下发开销、不含 H2D/golden）；`verify_performance.py` 只解析该行，未解析到即判 FAIL |

典型 case 门槛（`gpu_ms / 0.8`）：

| case | uplo | n | k | lda | gpu_ms | 门槛 us |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chbmv-base-000 | UPPER | 512 | 8 | 9 | 0.0098816 | 12.35 |
| chbmv-base-001 | UPPER | 1024 | 16 | 17 | 0.0158720 | 19.84 |
| chbmv-base-002 | LOWER | 2048 | 32 | 33 | 0.0280064 | 35.01 |
| chbmv-base-003 | LOWER | 4096 | 64 | 65 | 0.0530261 | 66.28 |

最小 case（n=1）门槛 7.87us —— 小尺寸完全由 launch 开销决定，是"能否过线"的主要风险点。

## 测试方案

随包 `chbmv_test.csv` 共 1200 条：1184 条 `SUCCESS` + 16 条 `INVALID_VALUE`；`TC_PF` 200 条（UPPER 95 / LOWER 105），其余 1000 条为精度/负向。

| 类别 | 前缀 | 覆盖内容 |
| --- | --- | --- |
| L0 基础 | `TC_L0` | uplo × 小尺寸 (4/8/12)，k=1 |
| L1 尺寸/带宽 | `TC_SQ` | 23 种尺寸 × uplo、20 组 (n,k) 扫描、`k > n-1` 退化 |
| L2 标量 | `TC_AB` | alpha/beta ∈ {0, 1, -1, 纯虚, 大值, 复数} × uplo |
| L4 前导维/步长 | `TC_LD`/`TC_INC` | lda = k+1+{1,4,8,16}；incx×incy ∈ ±1/±2/±3 全组合 |
| L5 填充 | `TC_FL` | 均匀/全零/交替/极端值/Inf/NaN（A、x、y 分别） |
| L5b | `TC_CV` | 中等尺寸 10~400，k=n/2 |
| L6 边界 | `TC_ED` | 零维 quick return（含叠加非法 uplo/k/lda/incx/空指针）、空指针、非法 uplo/lda、负 n/负 k、零步长、INT32_MIN、跨度溢出、特殊标量 |
| EX/PF | `TC_EX`/`TC_PF` | 确定性采样扩展 / 200 条性能用例（含 n=2048 带宽扫描至 k=2047） |

补充要求与做法：

1. **正态分布**：随包 `gen_csv.py` 默认只生成均匀分布。**保留 `chbmv_test.csv` 原样**（其 1000/200 与 `gpu_baseline.csv` 逐条对应），另生成 `chbmv_test_gauss.csv`（`GAUSS_NORM_5_2`，μ∈[-5,5]、σ∈[0.1,2]）作为补充精度集，在自测报告中说明条数与覆盖。
2. **handle 空指针**：CSV 列序不含 handle，无法表达，由 GTest 用例单独覆盖（期望 `HANDLE_IS_NULLPTR`）。
3. **负向缓冲区**：期望非 `SUCCESS` 的用例一律用单元素 dummy 或 `NULLPTR`，不按 `(n-1)|inc|` 预分配（避免测试工程自身分配失败误判）。
4. **越界哨兵**：未引用区（UPPER 左上 `k×k`、LOWER 右下、`lda` padding）填充 NaN 哨兵，若 kernel 误读即出现 NaN，直接暴露。
5. **内存**：复用 `run_memcheck.py` 思路做越界检查，产出 `4.1/4.2` 内存自验证材料（任务书 §3.4 无数值门槛但 §4 要求报告与日志）。
6. **可复现**：测试 README 写明 `build.sh` 编译、`--soc ascend910b`/`ascend910_93`、`--gtest_filter`、CSV 安装位置与 golden 生成步骤。

## 兼容性分析

新增 `aclblasChbmv`，不改任何既有接口；公共头只增一行声明。`arch22` 实现不复用 `arch35` 设备代码，避免架构不匹配。`blas/sbmv/README.md`、`docs/zh/api_list.md`、根 `README.md` 为增量修改。与同族 `aclblasSsbmv` 的三处刻意差异（非法 uplo 状态码、步长跨度判据、`alpha=0` 不豁免空指针）均在本文 §2.4 与 README 中记录。

# 性能优化方案

## 基线与放量

1. 固定环境：同一 A3 单卡、同一 CANN 9.1.0、同一编译选项、同一输入。
2. 基线：本文首版实现（`DIAG`+`COLUMN`），记录 `msprof op` 的 `Task Duration(us)` 与 GTest `[PERF] avg_us`。
3. 每候选 ≥5 次 warmup + ≥10 次有效采样，报告 Avg/Median/Min；任务书以 Avg 判定。
4. 每轮只改一个维度，保留 A/B 两组数据；失败方案写入 `work/PERF.md` 并说明原因。
5. 每轮结束跑完整精度/负向用例（1000 条 + 16 条负向 + 补充集），精度不通过不进下一轮。

## bound 预判（估算，待实测校准）

| case | A 字节 | x/y 字节 | 复乘次数 | 估算主导项 |
| --- | ---: | ---: | ---: | --- |
| 512/k8 | 36.9KB | 8KB | 8.7K | launch（~4–6us） |
| 1024/k16 | 139KB | 16KB | 34K | launch + 少量访存 |
| 2048/k32 | 541KB | 32KB | 134K | 访存（~0.4us @1.5TB/s）+ launch |
| 4096/k64 | 2.13MB | 64KB | 528K | 访存（~1.4us）+ 向量（~1.2us/核）+ launch |

结论：4 条典型 case 的门槛（12.35/19.84/35.01/66.28us）相对估算成本有 5–20 倍余量，**性能风险集中在 n ≤ 64 的 launch 主导区间**（门槛 7.87us）与宽带长尾（`k ≈ n`，靠 `COLUMN` 形态）。因此优化优先级为：① 小尺寸固定开销 → ② 宽带访存放大 → ③ 大 n 的向量效率。

## 候选优化清单

| 类 | # | 候选 | 预期收益 | 验证 |
| --- | --- | --- | --- | --- |
| 算法 | A1 | 只遍历 `2kEff+1` 条对角、每元素只读一次；`k > n-1` 用 `kEff` 截断 | 访存量取下界 | msprof 访存/带宽指标 |
| 算法 | A2 | `d=0` 只取实部（2 条指令）；`alpha=0 → BETA_SCALE`、`beta=0` 不读 y、`alpha/beta==1` 省乘 | 小 k 与退化 case 直接收益 | 对应 case A/B |
| 算法 | A3 | `TINY` 快路径：`n ≤ 64` 或整块不足一个向量宽度时单 block，省多核派发 | 打 7.87us 门槛 | `TC_PF_1005~1013` |
| 切分 | B1 | 行块切分（y 每元素只写一次，无 atomics/workspace/跨核归约）+ 成本模型选核数 | 主路径基线 | 对比"固定 64 行/核" |
| 切分 | B2 | 负载均衡：`blockRows` 8 对齐、余数前置、尾核不空转、小 n 核数 `ceil(n/64)` | 大 n 多核效率 | 各核 `Task Duration` 方差 |
| 切分 | B3 | 宽带 `COLUMN`：逐列 own 累加 + 列内 mirror 归约，`colTile` 分块使 UB 与 k 解耦 | 覆盖 `k ≈ n` 长尾 | n=2048 带宽扫描 |
| 搬运 | C1 | 整带 span 一次连续 DMA 常驻 UB，后续 pass 全在 UB 内 | 减少 GM 往返 | 主路径 |
| 搬运 | C2 | 复数 re/im 平面化：窗口只拆一次，pass 内 4 Gather 取代 AoS 掩码运算 | 指令数减半 | A/B |
| 搬运 | C3 | x 相位副本（complex64 的 32B 对齐只需 4 相位，而非 float32 的 8 相位）→ 省掉每 pass 的 2 次 x Gather | 每 pass 少 2 条 Gather | A/B（k 大时更明显） |
| 搬运 | C3b | 每 pass 用就地取负的 xIm 代替额外虚部平面（首版已采用），减少一条 buffer 与一次 chunk 级 Muls | 省 1 个平面 + 每 chunk 1 条指令 | 已实现 |
| 搬运 | C4 | A 侧抽取方式 A/B：vector `Gather` vs `DataCopyPad(blockLen=8B, srcStride=(lda-1)·8B)` 走 MTE 与 V 重叠 | 抽取是最大变量 | 同 case A/B |
| 搬运 | C5 | 步长路径：`|inc|=1` 连续；`>1` 整段 stride DMA；负步长一次反序 | 覆盖 `TC_INC` 全组合 | 步长 A/B |
| 计算 | D1 | 融合乘加（`MulAddDst`）完成复数乘加；`conj` 折叠成 `aI` 的符号（1 条 `Muls`）；虚部取负就地完成（1 条 `Muls`） | V 指令数 | 指令计数 |
| 计算 | D2 | `alpha/beta` 只在 chunk 收尾合成一次 | 减少重复乘 | A/B |
| 同步 | E1 | 单 kernel、无 workspace、无 host 同步；pass 间无 flag | 固定开销 | 入口空跑档（`mode=PROBE`）对比 |
| 同步 | E2 | chunk 级双缓冲：span 搬入与上一 chunk 计算重叠、y 写回与下一 chunk 计算重叠 | 大 n 主收益 | A/B |
| 同步 | E3 | `InitBuffer` 合并为单块 `TBuf` 手工切分；`FetchEventID` 按形态裁剪 | 每核入口开销 | 小 n A/B |
| Host | F1 | tiling O(1)、无分配、UB/核数首调缓存；tiling 结构体按值传参、指针独立传参 | 连续下发时 host 不饿死流 | GTest 10 次连续下发 |
| Host | F2 | 小 shape 跳过 platform 查询（缓存） | n ≤ 64 固定开销 | A/B |

## 借鉴与差异

| 参考 | 借鉴 | 与本任务的差异 |
| --- | --- | --- |
| `blas/sbmv/arch22/ssbmv_*` | 行块切分、整带 span、`2k+1` 对角 pass、UB 预算逐项公式、核数成本模型、宽带列形式、`XPhysicalPos` 约定 | 实数→complex64（AoS 拆分、4 相位而非 8、Gather 偏移 ×2）；对称→共轭（符号表）；对角只取实部；非法 uplo/步长判据/`alpha=0` 空指针三处语义不同；**不沿用**其"乘法前清零源 lane"的写法（`0·Inf = NaN`，本设计改为乘法后清零乘积） |
| Netlib `chbmv.f` | own/mirror 双重用途、`j-k..j` / `j..j+k` 边界、`alpha=0`/`beta=0` 语义 | 1-based 标量实现→列块向量化；对角全复数→只取实部（任务书 §2.1.2 与 golden 生成规则） |
| `blas/gbmv/arch22/cgbmv_*` | AoS→re/im 的 `Gather` 拆分、`BuildPackTable` 交织写回、`DataCopyPad` 复数列搬运、成本打分 | 一般带（kl/ku 不对称、无共轭）→ Hermitian 共轭符号；无 `trans` |
| `blas/hpr/arch22/chpr_*` | Hermitian 语义、校验顺序、`incx == INT_MIN`、复数掩码技巧 | 打包 rank-1（有写回）→ 带状 MV（只读 A）；无带索引 |
| cuBLAS `cublasChbmv` | 接口序列、语义、性能对标 | 内部实现不可见，只做行为与门槛对齐 |

## 优化流程与证据

`work/PERF.md` 按轮次登记：commit、编译选项、设备、CANN、case、warmup/采样数、Avg/Median/Min、相对 baseline 倍数、精度回归结论。失败方案同样保留（预期/实际/原因/证据）。最终交付：最佳实现 + 次优方案对比 + 失败方案列表 + 复现命令。

# 风险与备选方案

## 风险

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 小尺寸 launch 主导（门槛 7.87us） | 性能 case FAIL | `TINY`/`BETA_SCALE` 快路径；`InitBuffer` 合并；小 n 不查 platform；实测校准核数 |
| 宽带 `k ≈ n` 访存放大 | 长尾 case 超时 | `COLUMN` 形态每列只搬一次；必要时按 `colTile` 分块与 `DataCopyPad` 整段搬运 |
| 复数对齐/去交错的 0×Inf 污染 | 精度 FAIL（NaN） | 对齐 lane 在乘法**之后**清零乘积；`beta=0` 不读 y；`d=0` 不读对角虚部 |
| 索引写反（UPPER/LOWER、own/mirror） | 全量精度 FAIL | 已用独立模型 33000 组组合验证；上板先跑 `TC_L0`/`TC_SQ` 小集 |
| `INT32_MIN` / 跨度溢出判据 | 负向用例 FAIL | int64 校验；`ssbmv` 的 `UINT32_MAX` 判据不照抄 |
| 本机无 CANN，无法编译 | 首版可能编译错误 | 逐行对照仓内已编译的 `ssbmv`/`cgbmv`/`chpr` API 用法；上板第一轮只做编译+小集精度 |
| A3 与 910B3 差异 | 性能数据口径 | 报告注明设备；如需 910B3 补测 |

## 备选实现

1. **列切 + workspace 归约**（不做行块）：每核拥有列块，own 部分写 workspace，二次 kernel 归约。优点：列访问完全连续；缺点：需 workspace、两次下发、跨核归约。当前作为 `COLUMN` 形态之外的备选，仅在行块方案在宽带 case 实测不达标时启用。
2. **AoS 掩码直算**：不做 re/im 拆分，用偶数/奇数 lane 掩码 + 交换副本直接算复数乘加。优点：省拆分 Gather；缺点：每 pass 多 2–3 条掩码指令且易引入 `0·Inf`，暂不采用。
3. **按行单核标量**：不作为正式路径，只在诊断时用于定位（R1 禁止进主路径）。

# 待确认项与阶段状态

1. **设计文档 PR**：提交到 `04_tasks/01_community-task-2026/tasklist/11月社区任务-aclblasChbmv算子开发（A2/A3)/gcw_MIr7oWnV/docs/design.md`（目录名按任务 IT 系统原文，含全角/半角括号混排）。
2. **NPU 环境**：Atlas 800 A2（8×910B3 64GB，多人共享，CANN 9.2.0）。编译已通过；1200 条精度、性能采集与内存检查需按「只用 npu-smi 空闲卡」的规则排队执行。
3. **CANN 版本差异**：任务书要求 CANN 9.1.0，实测环境为 9.2.0（该机唯一可用版本），需在验收材料中注明。
4. **设计取舍**（已在本文固定，若评审有不同意见再改）：主路径 `DIAG` + 宽带 `COLUMN`；对角只取实部；`alpha=0` 短路为 `y=beta·y`；精度按实/虚分别判；正态用例以补充 CSV 提供。
5. **交付件**：`task_submission/` 目录（1/2.1/2.2/3.1/3.2/4.1/4.2）在真机验证完成后生成。
