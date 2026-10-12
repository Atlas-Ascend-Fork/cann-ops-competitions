# aclblasCtbsv 算子设计文档（Atlas A2/A3 / arch22）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。
> 参考样例：cann-ops-competitions PR #1318（aclsparseDenseToSparse 设计文档）。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclblasCtbsv 算子开发(A2A3)」。在 `ops-blas` 仓为 Atlas
A2/A3（`arch22`，覆盖 910B3/910B4 及 A3 系列）基于 Ascend C 全新实现单精度复数
（COMPLEX64）三角带状线性方程组求解算子 `aclblasCtbsv`：求解 `op(A) * x = b`（b
存于 x，解原地写回 x），对齐 cuBLAS `cublasCtbsv` 的功能与参数语义。

代码交付至 `https://gitcode.com/cann/ops-blas`：实现放 `blas/tbsv/arch22/`，测试放
`test/tbsv/ctbsv/arch22/`；接口声明入 `include/cann_ops_blas.h`。设计文档交付至
`https://gitcode.com/cann/cann-ops-competitions`。

## 背景介绍

### 算子定位：带串行依赖的 Level-2 求解类算子

`ctbsv` 是 BLAS Level-2 三角带状求解：A 为 n×n 复数三角带状矩阵（带宽 k），求解
`op(A)·x = b`。与前代/回代过程的本质是**严格串行链**：`x_j` 依赖
`x_{j-k} … x_{j-1}`（下三角）或 `x_{j+1} … x_{j+k}`（上三角），依赖距离为 k+1。
这条性质决定本算子的设计重心：

1. **并行性来自带宽受限的依赖距离**：任意两个下标距离 > k 的未知量相互独立，
   可按「块波前」（block wavefront）调度到多核；
2. **关键路径是 n 次串行的复数除法 + k 点规约**，不可被核数摊薄，是性能建模的
   第一要素；
3. **数据量小、访存支配**：最大性能 case（n=4096, k=128）的 A 带状存储仅
   (k+1)×n×16B ≈ 4.2 MB，x 仅 64 KB——远小于 HBM/一级缓存敏感规模，性能取决于
   串行链延迟与核间同步开销，而非带宽或算力。

### 同族算子与工程模式参照

| 参照 | 关系 | 可借鉴点 |
| --- | --- | --- |
| ops-sparse `sparse/gtsv2`（三对角求解，arch35） | k=1 的带状特例 | 求解类多核调度、tiling_data/host/kernel 四件套工程结构 |
| ops-sparse `sparse/spsv`（三角稀疏求解，arch35） | 三角求解的稀疏版 | 前代/回代方向处理、边界用例 |
| ops-blas 仓既有 blas 算子（arch22） | 同仓工程规范 | kernel 直调模式、handle/stream 绑定、CSV 驱动 GTest 测试框架、README 规范 |

本算子在 ops-blas 仓为**全新算子**：host 校验/分发、Ascend C kernel、tiling、
构建接入、测试、README 全套从零交付，无存量代码可复用，也无兼容性包袱。

### 任务书关键约束的解读

1. **性能预算口径**：达标耗时 = GPU 标杆实测 ÷ 0.8（即 NPU 平均单次 kernel 耗时
   不得超过 GPU 的 1.25 倍）。GPU 侧 `cublasCtbsv` 本身为开销主导（n=256, k=8 的
   GPU 耗时 203.6 us，远超该规模的纯计算需求），预算整体宽松，但最小预算
   254.5 us 仍约束 kernel 数量与同步次数。
2. **原地语义**：b 存于 x、解写回 x；A 与 x 不允许内存重叠。读写同一向量的交替
   扫描要求 kernel 内对 x 的读-改-写顺序严格受控。
3. **六个枚举组合的统一**：uplo×trans = 2×3 共 6 种组合，加上 diag 的 2 种对角
   语义。带状列主序存储下，`trans=T/C` 时 op(A) 的「列」对应 A 原存储中的
   「对角条纹」，访存不连续——需要规范化预处理（详见 §3.3.3）。
4. **负步长 incx**：x 物理布局 `1+(n-1)·|incx|`，精度用例覆盖 ±1/±2/±3；性能
   用例全部 incx=1（紧凑连续），设计上「紧凑快路径 + 通用 strided 路径」分流。
5. **n=0 合法 no-op、k=0 纯对角**：均为快速返回/特化路径。
6. **确定性不要求**：浮点求和顺序不保证逐位一致（任务书 §2.5），但同一 case 内
   计算图固定，混合容差判定不受影响。

### 功能缺口

ops-blas 仓当前无任何 tbsv 实现（实数/复数、任何 arch 均无）。缺口为整体性缺失：
接口声明、host 实现、kernel、tiling、构建接入、测试用例与 README 均需新建。

---

# 需求分析（required）

## 需求描述

在 `blas/tbsv/arch22/` 以 Ascend C 实现 `aclblasCtbsv` 句柄式 BLAS 接口，通过
handle 绑定 stream 直调 NPU kernel，能力达到：COMPLEX64 单精度复数 ×
uplo{UPPER,LOWER} × trans{N,T,C} × diag{UNIT,NON_UNIT} 全组合（12 组）×
n ∈ [0, 大规模] × k ∈ [0, n-1] × lda ≥ max(1, k+1)（含 padding）×
incx ≠ 0（含负步长），b 存于 x 原地求解；精度满足生态算子开源精度标准
COMPLEX64 档（混合容差 rtol=atol=2⁻¹³，matched_ratio ≥ 0.99，
max_abs_error ≤ 1e-2 或 32 ULP）；任务书 §3.3 五个性能 case 达标
（NPU 平均单次 kernel 耗时 ≤ GPU/0.8）。

## 需求拆解

| # | 需求 | 来源 | 性质 |
| --- | --- | --- | --- |
| R1 | arch22 全新实现：Host 校验/分发 + Ascend C Kernel + tiling + 构建接入 + README | §1、§2.2、§5 | 新增（主体） |
| R2 | 12 组 uplo×trans×diag 全组合语义正确，前/回代方向、共轭随组合正确翻转 | §2.1、§7.3 | 新增 |
| R3 | 带状列主序（banded column-major）寻址与 cuBLAS 对齐：LOWER 对角在带内第 1 行、UPPER 对角在第 k+1 行 | §2.1、§7.3 | 新增 |
| R4 | x 原地读改写；A 与 x 不重叠前提下的交替扫描正确性 | §2.5 | 需保证 |
| R5 | incx 正/负/非 1 步长通用支持；incx=1 紧凑快路径 | §2.4 | 新增 |
| R6 | 参数校验全覆盖：handle 空、非法枚举、n<0、k<0、incx=0、lda 非法、n>0 时空指针，返回码对齐 §2.4 异常行为表 | §2.4、§2.5 | 新增 |
| R7 | n=0 no-op、k=0 纯对角、UNIT 对角不读 A 对角位置等边界语义 | §2.1、§3.5 | 新增 |
| R8 | Inf/NaN 传播行为对齐 cuBLAS（不做奇异性检查，除零行为未定义但传播路径一致） | §3.5 | 需保证 |
| R9 | 五个性能 case 达标（预算 = GPU/0.8：254.5/493.8/695.5/2265/2468 us） | §3.3 | 约束 |
| R10 | 接入 ops-blas CSV/GTest 测试框架，覆盖随任务提供的 1200 条用例并按需补充 | §3.5 | 新增 |

---

# 详细设计（required）

## 算子分析

### 数学公式

设 `op(A)` 由 trans 决定：N → A；T → Aᵀ；C → Aᴴ（共轭转置）。求解：

```
op(A) · x = b        （b 初值存于 x，解写回 x）
```

记有效矩阵 `M = op(A)`（n×n 三角带状，带宽 k）。分量为：

- M 下三角（前代，j = 0 → n-1）：

```
x_j = ( b_j − Σ_{i=max(0,j−k)}^{j−1} M(i,j)·x_i ) / M(j,j)
```

- M 上三角（回代，j = n-1 → 0）：

```
x_j = ( b_j − Σ_{i=j+1}^{min(n−1,j+k)} M(i,j)·x_i ) / M(j,j)
```

`diag=UNIT` 时 `M(j,j) ≡ 1` 且不读取 A 的对角存储位置；`NON_UNIT` 时正常读取。
本设计采用**列扫描（axpy）等价形式**（见 §3.3.4），其浮点求和顺序与上式点积
形式不同但数学等价——任务书不要求确定性，两者均在混合容差内。

### 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| A / x 元素 | COMPLEX64（8 字节 = 2×FP32） | 唯一支持档；kernel 内实/虚部成对处理 |
| 对角倒数 | 复数除法 `x/d = x·conj(d)/|d|²` | |d|² 用 FP32 乘加；|d|=0 时行为未定义（对齐 cuBLAS，不做奇异性检查） |
| 属性参数 | int 枚举 / 标量 | Host 内存，launch 前完成校验并固化进 TilingData |

### 支持形状

| 参数 | 范围 | 说明 |
| --- | --- | --- |
| n | ≥ 0 | n=0 no-op；n=1 单元素 |
| k | ≥ 0 | 有效带宽取 `kEff = min(k, n−1)`（k ≥ n 时超出部分无意义，内部钳制，不报错） |
| A | lda×n 带状数组 | lda ≥ max(1, k+1)；每列存 k+1 行有效带（LOWER 对角在带内行 0，UPPER 在行 k，0 基） |
| x | 逻辑长 n，物理长 1+(n−1)·\|incx\| | incx ≠ 0，可为负 |

### 六组合规范化（本设计的关键降维）

对任意 (uplo, trans)，M 的三角性、求解方向、元素来源、是否共轭、是否需要 pack
如下表（0 基寻址，`AB[p]` 为 A 的带状存储）：

| uplo | trans | M 三角性 | 扫描方向 | M(i,j) 来源 | 共轭 | pack |
| --- | --- | --- | --- | --- | --- | --- |
| LOWER | N | 下三角 | 前代 | `AB[j·lda + (i−j)]` | 否 | 不需要 |
| UPPER | N | 上三角 | 回代 | `AB[j·lda + (k+i−j)]` | 否 | 不需要 |
| LOWER | T | 上三角 | 回代 | `AB[i·lda + (j−i)]` | 否 | pack → 紧凑上带 |
| LOWER | C | 上三角 | 回代 | `AB[i·lda + (j−i)]` | 是 | pack(conj) → 紧凑上带 |
| UPPER | T | 下三角 | 前代 | `AB[i·lda + (k+j−i)]` | 否 | pack → 紧凑下带 |
| UPPER | C | 下三角 | 前代 | `AB[i·lda + (k+j−i)]` | 是 | pack(conj) → 紧凑下带 |

要点：`trans=N` 时 M 的列恰好对应 AB 的一个连续段（列内带），扫描访存天然连续；
`trans=T/C` 时 M 的列对应 AB 的**跨列对角条纹**（每列取 1 个元素、跨 k+1 列），
无法连续向量访问。因此 T/C 组合先做一次 O(n·k) 的带状重排（pack），把条纹聚成
紧凑带（lda' = k+1，对角位置与原生布局约定一致：下带行 0 / 上带行 k），共轭在
pack 时一并完成。pack 后所有组合统一为「对紧凑带数组的连续列扫描」，kernel 只
需实现前代/回代两个方向。

## 算子实现

### 总体结构

```text
include/cann_ops_blas.h              公开接口声明（仓库公共头，随本算子新增）
blas/tbsv/arch22/
  ctbsv_host.cpp                     参数校验、规范化、workspace、tiling、launch
  ctbsv_tiling_data.h                TilingData 结构（host/kernel 共享）
  ctbsv_kernel.h                     kernel 启动函数声明
  ctbsv_kernel.cpp                   Ascend C kernel（pack/scan/gather 融合单 kernel）
blas/tbsv/README.md                  支持矩阵与接口说明（产品支持表标注 A2/A3）
test/tbsv/ctbsv/arch22/              CSV 驱动 GTest（沿用仓内 blas 测试框架）
```

构建：ops-blas 根 CMake 按 SOC 将 `arch22` 目录纳入构建（`ascend910b*`、
A3 系列 SOC 均映射 arch22），无需改构建脚本。

### Host 侧设计

#### Host 调用流程

```text
┌───────────────────────── Host: aclblasCtbsv ─────────────────────────┐
│                                                                       │
│ ① 参数校验（顺序对齐 §2.4 异常行为表）                                 │
│    handle == nullptr        → ACLBLAS_STATUS_HANDLE_IS_NULLPTR       │
│    uplo/trans/diag 非法枚举  → ACLBLAS_STATUS_INVALID_VALUE           │
│    n < 0 / k < 0            → ACLBLAS_STATUS_INVALID_VALUE           │
│    incx == 0                → ACLBLAS_STATUS_INVALID_VALUE           │
│    lda < max(1, k+1)        → ACLBLAS_STATUS_INVALID_VALUE           │
│    n == 0                   → ACLBLAS_STATUS_SUCCESS（no-op，返回）    │
│    n > 0 且 A/x == nullptr  → ACLBLAS_STATUS_INVALID_VALUE           │
│                                                                       │
│ ② 规范化：查 §3.3.1 六组合表 → {方向, pack 标记, conj 标记}            │
│    kEff = min(k, n−1)                                                │
│                                                                       │
│ ③ workspace 规划（按需，aclrtMallocAsync 挂 stream，调用后异步释放）    │
│    pack 缓冲   ：trans ∈ {T,C} 时 n×(kEff+1) complex（16B 对齐）      │
│    x 紧凑缓冲  ：incx ≠ 1 时 n complex                                │
│    均不需要时 workspace = 0，零分配                                    │
│                                                                       │
│ ④ tiling：NB、块数 P=⌈n/NB⌉、核分配、阶段偏移 → TilingData            │
│                                                                       │
│ ⑤ launch（handle 绑定的 stream）：单 kernel                          │
│    [phase0 并行 pack/gather] → SyncAll → [phaseW 波前求解]            │
│    → SyncAll → [phaseE scatter]                                       │
└───────────────────────────────────────────────────────────────────────┘
```

#### tiling 策略

1. **分块粒度 NB**：波前分块大小取 `NB = align8(kEff+1)`（向上对齐到 8 个
   complex，128B）。k+1 为依赖距离上界——块内列对后续行的更新至多延伸 k 行，
   NB ≥ k+1 保证「块 w 的行只接收来自块 ≤ w 的列的更新」，波前边界干净。
   上限钳制 `NB ≤ 4096`（UB 容量约束，见 kernel 侧）。
2. **k=0 特化**：纯对角，`x_i = UNIT ? x_i : x_i/d_i`，全并行逐元素，无波前。
3. **单块特化**：n ≤ NB 时单核顺序求解（无 SyncAll），其余核空转。
4. **核数**：`blocks = min(GetAivCoreCount(), 需要的并行度)`；phase0/phaseE 的
   pack/gather 按元素均分满核。
5. **tilingkey**：`direction(2) × hasPack(2) × unitDiag(2) × compactX(2)`
   编入 tilingkey，kernel 侧按 key 走对应分支，避免运行时判断开销。

### Kernel 侧设计

单 kernel 三阶段（AIV，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`）：

#### 阶段 0：并行预处理（pack / gather）

- **pack**（仅 trans ∈ {T,C}）：把 AB 的跨列条纹重排为紧凑带。按列均分到各核：
  核 c 处理列区间 `[c·⌈n/blocks⌉, …)`，每列读 kEff+1 个元素（跨列各 1 个，
  标量读或 gather），顺序写入 pack 缓冲对应列（连续写）。conj 在此完成。
  数据量 ≤ 4.2 MB，满核带宽下 < 20 us。
- **gather**（仅 incx ≠ 1）：按元素均分，`wx[i] = x[i·incx]`（incx 为负时
  地址递减，符号运算天然覆盖）。
- `SyncAll()`。

#### 阶段 W：波前分块前代/回代（核心）

以「前代 + 紧凑下带 L（lda'=kEff+1，对角行 0）」为例（回代为镜像，扫描逆序、
更新区间反向）：

```text
for wave w = 0 … P−1:
  ┌─ Stage A（owner 核串行）────────────────────────────────────┐
  │ 块 Bw = [w·NB, min(n, (w+1)·NB))                             │
  │ for j in Bw:                                                 │
  │   dj  = UNIT ? 1 : L(0, j)            # 对角元素             │
  │   xj  = x[j] / dj                     # 复数除法             │
  │   for i = j+1 … min(n−1, j+kEff):  x[i] −= L(i−j, j)·xj     │
  │ （x 的活跃窗口 [w·NB, w·NB+NB+kEff) 常驻 UB，读写不落 GM）      │
  └──────────────────────────────────────────────────────────────┘
  SyncAll()
  ┌─ Stage B（全核并行）────────────────────────────────────────┐
  │ 块 w 的列对块 > w 的行的跨界更新：                              │
  │ for j in Bw:  for i = (w+1)·NB … min(n−1, j+kEff):           │
  │   x[i] −= L(i−j, j)·xj                                       │
  │ （行列矩形 ≤ kEff×NB，按 i 均分到各核；GM 读改写或 UB 窗口）     │
  └──────────────────────────────────────────────────────────────┘
  SyncAll()
```

正确性依据：块 w 的行 `[w·NB, w·NB+kEff)` 只接收块 w−1 及更早列的更新（NB ≥
kEff+1 保证），故 Stage A 前置一次 SyncAll 即安全；Stage B 与后续块的 Stage A 间
以 SyncAll 划界。每波 2 次 `AscendC::SyncAll()`，P = ⌈n/NB⌉ ≤ 32（最大 case），
同步总开销 ≤ 64 次，预算内。

并行度分析：串行关键路径 = n ×（k 点复数点积规约 + 1 次复数除法）。规约在
owner 核 UB 内以向量乘加 + 树状规约完成（k ≤ 128 时深度 ≤ 7 级），单步延迟估
150–400 ns；n=4096 时关键路径估 0.6–1.6 ms < 2265/2468 us 预算；Stage B 的跨界
更新总量 n·kEff 个 MAC，满核分摊后可忽略。**若上板实测关键路径超预算**，回退
方案为分区仿射法（各分区对边界 k 个未知量做参数化求解，再解小规模耦合系统），
见风险 2。

#### 阶段 E：scatter（仅 incx ≠ 1）

`x[i·incx] = wx[i]`，按元素均分并行；`SyncAll` 后结束。

#### UB 占用（arch22 约 192 KB）

- x 活跃窗口：`NB + kEff` 个 complex ≤ (4096+128)×16B ≈ 64 KB（上限钳制保证）；
- 带列缓冲：kEff+1 个 complex ≤ 2 KB + 规约/除法临时 < 4 KB；
- pack/gather 阶段复用同一 UB 区域（阶段间生命周期不重叠）。

#### 精度考量

- 每 x_j 的点积在 UB 内以 FP32 乘加规约，求和长度 ≤ k+1 ≤ 129，误差累积有界，
  满足 rtol=atol=2⁻¹³、matched_ratio ≥ 0.99 的 COMPLEX64 档（实/虚部分别按
  FLOAT32 判定）；
- 复数除法用 `x·conj(d)/|d|²`，与 Netlib 参考实现（golden）公式一致，仅舍入
  顺序差异；大数规约场景 `max_abs_error_limit` 已有 1e-2/32 ULP 裕量；
- Inf/NaN：逐元素传播，无特殊分支（对齐 cuBLAS 不做检查的语义）。

### 支持硬件

| 芯片 | 支持情况 |
| --- | --- |
| Atlas 800I/T A2（910B3 / 910B4） | 支持（arch22），CANN 9.1.0+ |
| Atlas A3 系列（910_93 等） | 支持（arch22） |

自验证以 910B3 为主；验收覆盖 A2/A3 两个系列（任务书 §3.1、§7.4）。

### 算子约束限制

1. 仅支持 COMPLEX64；不支持其他 dtype（调用方传入即为本接口，无 dtype 参数）。
2. A 与 x 不允许内存重叠（任务书 §2.5，调用方保证）。
3. 不支持超出 lda/incx 语义的非连续访问（§2.5）。
4. 求解类对奇异矩阵无定义：对角为零时除零结果未定义，不报错、不检查（对齐
   cuBLAS）；测试工程以对角强化偏移规避（README 已约定，golden 同矩阵）。
5. k ≥ n 时按 kEff = n−1 钳制，超出部分忽略，不报错。
6. n=0 时返回 SUCCESS，不访问 A/x（允许空指针）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 混合容差（COMPLEX64 档）：逐元素 `|actual−golden| ≤ atol + rtol·|golden|`，atol=rtol=2⁻¹³；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP；golden 由 Netlib BLAS ctbsv 生成，对输出 x 全元素（实/虚部分别）比对 | 任务书 §3.2、opbase 混合容差标准 |
| 性能标准 | 910B3 上 5 个性能 case 平均单次 kernel 耗时 ≤ GPU 标杆 ÷ 0.8（msprof `OpBasicInfo.csv` 的 Task Duration 均值），预算：254.5 / 493.8 / 695.5 / 2265 / 2468 us | 任务书 §3.3、test_cases README |
| 内存标准 | 不涉及（任务书 §3.4）；workspace 峰值（pack ≤ n·(k+1)·16B + 紧凑 x ≤ n·16B）在自测报告中说明 | 任务书 §3.4 |

## 测试策略

对接随任务提供的测试资产（`test_cases/`：`tbsv_test.csv` 1200 条 + 
`gpu_baseline.csv` 200 条 + `gen_csv.py`/`verify_accuracy.py`）：

| 层次 | 内容 |
| --- | --- |
| 框架 | ops-blas 仓 CSV 驱动 C++ GTest；`verify_accuracy.py` 编译-加载-执行-解析 PASS/FAIL 一键化 |
| 精度覆盖 | TC_L0（12 组枚举 × n=8,k=2）/ TC_SQ（尺寸扫描 1→2048，含质数、2 幂±1、非对齐）/ TC_BD（k=0/1/小值/半带/满带 n−1）/ TC_DG（UNIT 对角不读验证）/ TC_LD+TC_INC（lda padding、incx ±1/±2/±3）/ TC_FL（A 均匀/交替/极端；x 含 Inf/NaN；UNIT 时对角 NaN/Inf 不读）/ TC_CV / TC_ED（零维、k=0&&UNIT、空指针、非法前导维、负维度、负带宽、零步长、INT_MIN 步长、非法枚举 999、handle 空）/ TC_EX（864 条扩展采样） |
| 补充用例 | 按任务书 §3.5 第 4 条自查补齐：大 n（>2048）大 k 采样、k ≥ n 钳制行为、A 与 x 不重叠断言（guard 区哨兵）、重复调用幂等性 |
| 性能 | GTest 跑 TC_PF（200 条，全连续访存）；msprof op 采集 `--gtest_filter` 收敛到单 case，`OpBasicInfo.csv` Task Duration 均值对比 `gpu_baseline.csv`（阈值 gpu_ms/0.8）；前 5 条 `pf_doc_*` 与任务书 §3.3 逐参数对齐 |
| 报告 | task_submission 目录：自验证步骤说明、精度/性能/内存报告（xlsx+log），精度截图含逐 case PASS 明细与 matched_ratio/max_abs_error 统计 |

## 兼容性分析

- **新算子，无存量行为影响**：`blas/tbsv/`、`test/tbsv/` 为新目录；唯一触碰的
  公共文件是 `include/cann_ops_blas.h`（新增接口声明，纯增量，供其他产品线共用，
  任务书 §4 交付件 4 要求）。
- **构建**：arch22 目录经仓内既有 SOC→arch 映射机制自动纳入，不改公共构建脚本。
- **同名冲突**：社区同期存在 Ctbsv-950（A5/arch35）任务，代码目录与 arch 不同，
  接口为同一定义；若双方先后合入，以先合入方为准做 rebase 对齐（接口声明按
  任务书 §2.3 签名逐字一致，无语义分歧空间）。
- **CANN 版本**：9.1.0 及后续配套；不使用实验性 API。

---

# 风险点

| # | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| 1 | 串行关键路径（n 次复数除法 + 规约延迟）实测超预算 | 大 n case 性能不达标 | 规约用向量乘加+树状规约降低单步延迟；先 msprof 归因；回退：分区仿射法（依赖距离 k 的参数化分区求解，把链长从 n 降到 n/P + P·k 量级） |
| 2 | SyncAll 次数 × 单次开销在大 n 小 k（块多）场景累积 | case 3（n=1024, k=16）块数 64×2 次同步 | NB 下限钳制（≥64）控制块数；单块/少块路径免同步；实测超限时合并波前（一次 SyncAll 处理 2 个块） |
| 3 | pack 路径（T/C）条纹跨列读效率 | case 3/4（T/C 组合）性能 | pack 数据量 ≤ n·(k+1)·16B（≤ 4.2 MB）满核分摊；标量读改为按对角线分组的批量 DataCopy（每核认领连续列段，条纹段内连续 16B 步长可用 strided copy） |
| 4 | 带状寻址/方向翻转的正确性（6 组合 × 边界） | 精度误报 | L0 12 组合小例手算对拍；TC_BD 满带/半带用例放大寻址错误；kernel 内寻址宏集中定义，单一出处 |
| 5 | incx 负步长/零步长边界 | 负向用例失败 | 负向返回码用例先行（TC_ED 全覆盖）；正步长 gather/scatter 与求解解耦，符号运算统一处理 |
| 6 | A3 环境到位晚 | 双系列验收进度 | arch22 同源，先 910B3 全量自验，A3 到位后补跑功能/精度/性能三件套 |

# 交付物

1. **算子设计文档**：本文件，PR 至 `cann-ops-competitions`
   `04_tasks/01_community-task-2026/tasklist/09-aclblasCtbsv-A2A3/<gitcode_id>/docs/design.md`
   （目录编号若与仓内 `09-*` 惯例冲突，按评审意见调整）。
2. **代码**（PR 至 `ops-blas`，任务书 §5 结构）：`include/cann_ops_blas.h` 新增
   `aclblasCtbsv` 声明；`blas/tbsv/arch22/` host/tiling/kernel；`blas/tbsv/README.md`
   （产品支持表标注 Atlas A2/A3 系列：支持）。
3. **自测用例与测试代码**：`test/tbsv/ctbsv/arch22/`（CSV + GTest，结构对齐主仓
   blas 算子）；复现 README（环境/编译/步骤）。
4. **自测报告**：task_submission 目录八件套（步骤说明、精度/性能/内存报告与
   日志），精度含 golden 对比与截图，性能含 >10 次有效采样均值与 msprof 证据。
