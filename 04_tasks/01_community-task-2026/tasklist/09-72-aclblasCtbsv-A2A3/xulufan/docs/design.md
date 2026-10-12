# aclblasCtbsv 算子设计文档（Atlas A2/A3 / arch22）

> 交付件①。按 `resources/design_template.md` 组织，四个必需一级章节齐全。

---

# 需求背景（required）

## 需求来源

CANN 生态 9 月社区任务「aclblasCtbsv 算子开发(A2A3)」。在 `ops-blas` 仓
（https://gitcode.com/cann/ops-blas）为 Atlas A2/A3 系列（DAV-2201，`arch22`，覆盖
910B3/910B4/910_93 等）实现单精度复数（COMPLEX64）三角带状线性方程组求解算子
`aclblasCtbsv`：求解 `op(A) * x = b`，b 存于 x、解原地写回 x，kernel 直调（Ascend C）。

代码交付 `blas/tbsv/arch22/`，测试交付 `test/tbsv/ctbsv/arch22/`；设计文档交付
`cann-ops-competitions` 仓 `04_tasks/01_community-task-2026/tasklist/09-72-aclblasCtbsv-A2A3/`
（目录已存在）。

## 背景介绍

### 算子语义：带依赖链的原地回代

A 为 n×n 三角带状复矩阵（带宽 k，列主序束带存储于 lda×n 数组），op(A) 由 trans 决定
（N / T / C），uplo 取上/下三角带，diag 决定对角是否读写，x 为长 n、步长 incx 的复向量
（incx 可负）。求解为逐分量前代/回代：

```text
x_i = ( b_i − Σ_{j∈band(i)} op(A)(i,j)·x_j ) / op(A)(i,i)     (diag=NON_UNIT)
x_i =   b_i − Σ_{j∈band(i)} op(A)(i,j)·x_j                     (diag=UNIT)
```

**关键性质（决定整个设计）**：k ≥ 1 时 x_i 依赖 x_{i−1}，递推依赖距离为 1，临界路径
长度为 n，不存在天然的行间波并行；可用并行度只有两个来源——(a) 行内 ≤ k+1 宽的复数
点积；(b) 面板级把「已知解部分」的乘加批量向量化。GPU 侧同样受此支配：任务书 5 条
性能 case 的 cuBLAS 实测为 482~883 ns/行（见 §3.5），是串行核行为，这给了 NPU 单核
面板算法足够的预算空间。精度风险集中在复数除法与求和顺序（混合容差判定，见可维可测）。

### ops-blas 现状分析

`blas/tbsv/` 当前仅有 **arch35（Ascend 950PR/DT，DAV-3510）实现**，基于 `master` 源码核对：

| 文件 | 职责 |
|---|---|
| `arch35/ctbsv_host.cpp` | 参数校验、SIMT 线程数选择、tiling 填充、`ctbsv_kernel_do` 启动 |
| `arch35/ctbsv_kernel.cpp` | AIV 向量 gather 路径 kernel |
| `arch35/ctbsv_kernel_simt.cpp` | SIMT kernel（per-thread 列分配 / 窄带 T/C 单 warp） |
| `arch35/ctbsv_complex_arith.h` | 复数标量算术 |
| `arch35/ctbsv_tiling_data.h` | `CtbsvTilingData` |

`blas/tbsv/README.md` 已登记 `aclblasCtbsv` 接口与 COMPLEX64 语义，产品支持表中
Atlas A2/A3 系列标注「不支持」——**本任务即把该行翻转为「支持」**。

arch22 侧的仓库先例（工程模式参照）：

- `blas/tbmv/arch22/`：`stbmv_host.cpp / stbmv_kernel.cpp / stbmv_kernel.h /
  stbmv_tiling_data.h` 四件套；`*_kernel_do(指针…, tiling, numBlocks, stream)` 启动
  约定、tiling 内携带 `useCoreNum`；
- `blas/trsv/arch22/`、`blas/tpmv/arch22/`：同为「arch35 已有 + arch22 新增」的三角类
  level-2 算子，host 结构与 tbmv 一致；
- 公共设施：`common/helper/aclblas_handle_internal.h`（handle→stream）、
  `common/helper/kernel_constant.h`、`common/helper/host_utils.h`；
- 构建分发：仓内 CMake 按 SOC 将 `ascend910b*`、`ascend910_93*` 映射到 `arch22`
  目录——新增 `blas/tbsv/arch22/` 即自动参与构建，无需改构建脚本。

arch35 host 的既有语义（源码核对，arch22 按等价移植 + 差异对齐，见兼容性分析）：

1. 校验顺序：handle → uplo/trans/diag 枚举 → n ≥ 0、k ≥ 0 → lda > k（即 ≥ k+1，
   不与 n 比较）→ incx ≠ 0 且 ≠ INT_MIN → n==0 quick return → A/x 非空；
2. `k == 0 && diag == UNIT` 直接返回 SUCCESS（x 不变，恒等求解）；
3. SIMT 路径按 n 与有效带宽 `min(k, n−1)+1` 选线程数；窄带 T/C 用单 warp（32 线程），
   宽带 T/C 与宽带 N 走 AIV 向量 gather。

### 功能缺口

**缺口一：arch22 整体缺失。** 无 host、无 kernel、无测试、README 支持矩阵为「不支持」。

**缺口二：SIMT → 标准 Ascend C 移植。** arch35 的 SIMT kernel（`blockIdx/threadIdx`、
warp 级同步）依赖 DAV-3510 特性，arch22 不可用；全部路径必须重写为 AIV 向量 kernel
（`TPipe`、`DataCopy`、向量指令）。Host 侧校验/tiling/启动逻辑与架构无关，可按函数
等价移植。

**缺口三：错误码口径差异（需对齐任务书）。** arch35 对非法枚举返回
`ACLBLAS_STATUS_INVALID_ENUM`；任务书 §2.4 要求 `ACLBLAS_STATUS_INVALID_VALUE`
（配套测试 TC_ED 负向用例亦按 INVALID_VALUE 期望）。arch22 按任务书口径实现，
差异在 README 中注明（见兼容性分析）。

### 功能分析：uplo × trans 的方向/寻址归一化

8 种 (uplo, trans) 组合可归一为「求解方向 × 带寻址模式 × 共轭」三个正交量，且只有
两种寻址模式（详见 §3.3 寻址表）：**N 模式跨列取带**（方程 i 的带元素分布在 k+1 个
存储列中各一个，等差跨步），**T/C 模式列内连续**（方程 i 的带元素是存储列 i 的一段
连续 r 区间，天然向量友好）。这一归一化把 kernel 分派收敛为 2 种数据通路 × 2 个方向
× 共轭开关，是移植正确性的第一道结构保障。

---

# 需求分析（required）

## 需求描述

在 `arch22` 上实现 `aclblasCtbsv`：COMPLEX64 三角带状方程组求解，uplo×trans×diag
（2×3×2）全枚举组合语义对齐 cuBLAS `cublasCtbsv`；b 存于 x、解原地写回；支持
incx 正/负/非 1 步长；n=0 与 k=0&&UNIT quick-return；Inf/NaN 传播语义对齐；异步
执行（handle 绑定 stream，Host 无隐式同步）；5 条典型性能 case 达标（GPU 实测 ÷ 0.8，
msprof kernel 口径），全量精度用例满足 COMPLEX64 混合容差。

## 需求拆解

| # | 需求 | 来源 | 性质 |
|---|---|---|---|
| R1 | arch22 全新实现：Host 分发 + Ascend C kernel + 构建 + 测试 + README | §1、§2.2、§5 | 新增（主体） |
| R2 | uplo×trans×diag 12 组枚举全组合，语义对齐 cublasCtbsv（含 T/C 共轭、UNIT 对角不读写） | §2.1 | 新增 |
| R3 | 原地求解：b 存于 x、解写回 x；A 与 x 不允许内存重叠；不返回视图 | §2.1、§2.5 | 规定 |
| R4 | incx ≠ 0 且 ≠ INT_MIN，正/负/非 1 步长均支持；n=0 与 k=0&&UNIT quick-return | §2.4、§2.5 | 规定 |
| R5 | 参数校验与错误码对齐任务书（HANDLE_IS_NULLPTR / INVALID_VALUE） | §2.4 | 规定（与 arch35 现行为有差异） |
| R6 | Inf/NaN 按浮点运算自然传播；不做奇异性检查（除零行为未定义，对角强化由测试工程完成） | §2.4、§3.5 | 需保证并补测 |
| R7 | 性能：5 条典型 case 平均单次 kernel 耗时 ≤ GPU 实测 ÷ 0.8（msprof 口径，10 次有效采样） | §3.3 | 约束 |
| R8 | 精度：COMPLEX64 混合容差（rtol=atol=2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP） | §3.2 | 约束 |
| R9 | 异步语义：`aclblasSetStream` 绑定 stream，Host 侧不做流同步 | §2.5 | 规定 |
| R10 | 交付：`blas/tbsv/arch22/` 代码、README 支持矩阵翻转、`test/tbsv/ctbsv/arch22/` 测试工程、自测报告 | §4、§5 | 交付 |

---

# 详细设计（required）

## 算子分析

### 数学公式

设带状存储数组 AB（lda×n，列主序，0-based 存储行 r、列 j，元素偏移 = j·lda + r）：

```text
LOWER:  A(i,j) = AB[r = i−j,   j]，j ≤ i ≤ min(n−1, j+k)   （对角 r=0）
UPPER:  A(i,j) = AB[r = k+i−j, j]，max(0, j−k) ≤ i ≤ j      （对角 r=k）
```

求解方向与方程 i 的带元素位置（8 组合归一化表，kernel 分派的唯一依据）：

| uplo | trans | op(A) 三角 | 求解方向（i） | 方程 i 的带元素 | 共轭 |
|---|---|---|---|---|---|
| LOWER | N | 下 | 0 → n−1（前代） | 跨列：存储列 j ∈ [i−k, i]，行 r = i−j | 否 |
| UPPER | N | 上 | n−1 → 0（回代） | 跨列：存储列 j ∈ [i, i+k]，行 r = k+i−j | 否 |
| LOWER | T | 上 | n−1 → 0 | 列内：存储列 i，行 r = j−i ∈ [0, k]（x_j，j ∈ [i, i+k]） | 否 |
| LOWER | C | 上 | n−1 → 0 | 同上 | 是 |
| UPPER | T | 下 | 0 → n−1 | 列内：存储列 i，行 r = k+j−i（x_j，j ∈ [i−k, i]） | 否 |
| UPPER | C | 下 | 0 → n−1 | 同上 | 是 |

复数除法（diag=NON_UNIT 时每行一次）：

```text
(s_r + i·s_i) / (d_r + i·d_i) = ((s_r·d_r + s_i·d_i) + i·(s_i·d_r − s_r·d_i)) / (d_r² + d_i²)
```

测试值域（|A|、|b| ≤ ~10，对角经测试工程 boost 后 |d| ≥ 5）下 |d|² 无上溢风险，
直接公式精度 ~1–2 ULP，满足 2⁻¹³ 容差；Smith 缩放法作为数值兜底（见风险点）。

### 支持数据类型

| 数据 | 类型 | 说明 |
|---|---|---|
| A / x / b | COMPLEX64（aclblasComplex） | 实/虚部均为 FP32 |
| n / k / lda / incx | int（Host） | — |
| uplo / trans / diag | 枚举（Host） | 归一化为方向/寻址/共轭/单位对角四个开关 |

### 支持形状

n ∈ [0, INT32_MAX]（n=0 quick-return）；k ∈ [0, n−1] 为有效带宽（k ≥ n 时按
min(k, n−1) 截断，校验只要求 k ≥ 0）；lda ≥ k+1（与 n 无关）；x 物理占位
1+(n−1)·|incx|，incx ≠ 0 且 ≠ INT_MIN；A、x 不重叠。

## 算子实现

### 3.1 总体结构

```text
include/cann_ops_blas.h            aclblasCtbsv 声明（arch35 已登记，确认即可）
blas/tbsv/arch22/                  本任务新增：
  ctbsv_host.cpp                     校验 + 归一化 + tiling + 启动
  ctbsv_tiling_data.h                arch22 tiling 结构（含归一化模式位）
  ctbsv_kernel.h                     ctbsv_kernel_do 声明
  ctbsv_kernel.cpp                   AIV 面板式求解 kernel
  ctbsv_complex_arith.h              复数向量/标量算术（乘、点积、除法）
blas/tbsv/README.md                支持矩阵：Atlas A2/A3 系列 →「支持」
test/tbsv/ctbsv/arch22/            csv 驱动 GTest（镜像仓内 blas 测试结构）
```

### 3.2 Host 侧设计

#### Host 调用流程图

```text
┌────────────────────────── Host ──────────────────────────┐
│  aclblasCtbsv(handle, uplo, trans, diag, n, k, A, lda,   │
│               x, incx)                                   │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ handle/  │──→│ n/k/lda/ │──→│ quick    │──→│ 归一化  │ │
│  │ 枚举校验  │   │ incx 校验 │   │ return   │   │ 模式位  │ │
│  │ §2.4 口径 │   │          │   │ 判定      │   │ (§3.3) │ │
│  └──────────┘   └──────────┘   └──────────┘   └───┬────┘ │
│                                                    │      │
│              ┌─────────────────────────────────────▼────┐ │
│              │ CalTiling：n/k/lda/incx/模式位/           │ │
│              │ 面板宽度 W（按 k 自适应）/ useCoreNum      │ │
│              └─────────────────────────────────────┬────┘ │
│                                                    │      │
│              ┌─────────────────────────────────────▼────┐ │
│              │ ctbsv_kernel_do(tiling, handle->stream)   │ │
│              │ （无 workspace、无 Host 同步）             │ │
│              └──────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

要点（与 arch35 逐函数等价移植 + 差异修正）：

1. **校验顺序**（全 Host 完成，launch 前返回）：handle 非空（否则
   `HANDLE_IS_NULLPTR`）→ uplo/trans/diag 枚举合法（否则 `INVALID_VALUE`，任务书
   口径）→ n ≥ 0、k ≥ 0 → lda > k → incx ≠ 0 且 ≠ INT_MIN → n == 0 返回 SUCCESS
   → A、x 非空。
2. **quick-return**：`n == 0` 或 `k == 0 && diag == UNIT`（恒等求解，x 不变）直接
   返回 SUCCESS，不 launch。
3. **归一化**：uplo×trans 查 §3.3 表得 {方向, 寻址模式(N-跨列 / TC-列内), 共轭,
   有效三角侧}，与 diag 一起作为 tiling 模式位——kernel 内零枚举分支。
4. **tiling**：`CtbsvTilingData`（arch22 版）携带 GM 指针、n、k、lda、incx、模式位、
   面板宽度 W、KC（大 k 时的已知段分块粒度）、useCoreNum（初版恒 1，见多核扩展）。
   W 自适应：`W = clamp(8, 32, UB预算 / ((k+1)·8B))`，k 巨大（≥ 1024）时收缩到 8 并
   启用 KC 分块。
5. **启动**：沿用仓内 arch22 先例的 `*_kernel_do` 约定，从 handle 取 stream 直调；
   无 workspace 分配（原地算子，内存要求「不涉及」天然满足）。

### 3.3 Kernel 侧设计：面板式两遍结构（本任务技术主体）

arch35 的并行来自 SIMT 线程模型；arch22 上以「**面板（panel）+ 两遍拆分**」重建
并行度：把 n 行切成宽 W 的面板，每个面板内：

- **已知段（向量批处理）**：方程 i 中所有 x_j 在面板开始前已解的部分（依赖距离
  ≥ W 的项），对一个 W×(≤k+1) 的带窗口做批量带状 matvec，一次向量 pass 完成；
- **未知段（面板内短递推）**：仅剩 j 与 i 同面板的 ≤ W−1 项，串行 W 步，每步 =
  短点积（≤ W 项）+ 复数除法 + 写回。

串行临界路径从「n 步 × k 宽点积」压缩为「n 步 × W 宽短点积 + 1 次除法」，
W=16~32 时每步 ~100–300 ns，是全部性能 case 达标的核心（§3.5）。

#### Kernel 流程图（统一骨架，方向/寻址由模式位参数化）

```text
┌──────────────── Device (AIV × 1 core, 单 kernel) ───────────────┐
│                                                                  │
│  for panel p = 0 … n−1 step W（沿求解方向）:                     │
│                                                                  │
│    ┌────────────┐   ┌────────────┐   ┌───────────────────────┐  │
│    │ ① 带窗口   │──→│ ② x 窗口   │──→│ ③ 已知段向量批处理     │  │
│    │ 装载 UB    │   │ 装载(滚动) │   │ partial[i] = Σ_known   │  │
│    │ (§寻址)    │   │ ±1 直拷 /  │   │ band[i,j]·x[j]         │  │
│    │            │   │ |incx|>1   │   │ (大 k: KC 分块循环)    │  │
│    └────────────┘   │ gather     │   └───────────┬───────────┘  │
│                     └────────────┘               │              │
│    ┌─────────────────────────────────────────────▼───────────┐  │
│    │ ④ 面板内串行递推: for i in panel（方向序）                │  │
│    │    s = b[i] − partial[i] − Σ_intra band[i,j]·x[j] (≤W项) │  │
│    │    x[i] = s / diag[i]   (UNIT: x[i] = s)   ← 复数除法    │  │
│    │    x[i] 同时写入 UB 滚动窗口（供后续方程消费）            │  │
│    └─────────────────────────────────────────────┬───────────┘  │
│                                                  │              │
│    ┌─────────────────────────────────────────────▼───────────┐  │
│    │ ⑤ 面板 x 写回 GM（incx 步长散射；±1 连续写）             │  │
│    └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**① 带窗口装载（两种寻址模式）**：

| 模式 | 装载方式 | 面板窗口大小 |
|---|---|---|
| TC-列内（LOWER/UPPER × T/C） | 方程 i 读存储列 i 的连续 r 切片 → 每列一次连续 DataCopy | W×(k+1) 复数 |
| N-跨列（× N） | 方程 i 在每存储列 j 取单个元素（等差位置）→ 平行四边形切片（每列一段连续 r）→ 2D DataCopy 后 UB 内按行视图消费；或直接 vgather（索引 = base + i·lda 等差） | W×(W+k) 复数 |

N 模式的 vgather 与转置两条通路列为**上板首日探针项**，任一达标即定版；兜底为
KC 分块 + 标量子块循环（仅小规模用）。

**② x 窗口**：|incx| = 1 时按方向连续装载/滚动（负步长 = 基址翻转到尾端 + 正向
处理，语义等价）；|incx| > 1 时面板级 gather 进 UB、面板尾散射回 GM（精度路径，
性能 case 均为 incx=1，不做性能承诺）。

**③ 已知段批处理**：带窗口内对每行 i 的已知 span 做复数点积。复数乘拆实虚：
`(a·x)_r = a_r·x_r − a_i·x_i`，`(a·x)_i = a_r·x_i + a_i·x_r`；实虚解交织到独立
FP32 向量（装载时双通路搬运或 UB 内向量 Shuffle/Select），点积用向量乘加 +
`ReduceSum` 水平归约，每行得到一个复数 partial。大 k（≥ 1024）按 KC=512 分块
循环累加，UB 峰值受控。

**④ 短递推 + 复数除法**：≤ W 项的点积在 UB 标量/短向量完成；除法按 §数学公式
直接法，`diag=UNIT` 跳过；除数即对角元素（LOWER r=0 / UPPER r=k），T/C 模式下
对角位置随寻址表取自存储列 i，C 模式取共轭。

**UB 预算**（k=128、W=32 最紧性能 case）：带窗口 33–41 KB + x 滚动窗口 ~1.3 KB +
partial/b ~0.5 KB + 解交织/归约临时 < 8 KB，合计 < 60 KB，远低于 arch22 可用 UB
（~192 KB），余量支撑 W 上调的调参空间。

**确定性**：单核串行递推天然逐位确定（任务书不要求，但免费获得，便于复测）。

#### 多核扩展（phase-2，可选）

单核面板算法已满足全部性能预算（§3.5 余量 1.7×–7×）；若后续 msprof 归因显示
串行链仍是瓶颈，启用分块两阶段算法：按行分 P 块（B ≥ k），阶段 A 各核并行计算
本块 `u_p = D_p⁻¹·b_p` 与耦合算子 `W_p = D_p⁻¹·C_p`（B×k），阶段 B 串行扫尾
`x_p = u_p − W_p·x_{p−1,tail}`（P 步短 matvec）。触发条件、B 取值与数值影响
（求和顺序变化仍在混合容差内）留待实测数据支撑，本版不实现、仅在 tiling 预留
useCoreNum 字段。

### 3.4 特殊值与边界

| 场景 | 行为 |
|---|---|
| n = 0 | SUCCESS，不 launch |
| k = 0 且 diag = UNIT | SUCCESS，x 不变 |
| k = 0 且 NON_UNIT | 纯对角求解：x_i = b_i / A_ii |
| k ≥ n | 有效带宽按 min(k, n−1) 截断，越界带元素不读 |
| Inf/NaN（x、A 带内） | 按浮点运算自然传播（NaN 经乘加/除法扩散），与 Netlib golden 一致 |
| diag = UNIT 时 A 对角为 Inf/NaN | 对角不读取，值不影响结果（TC_DG/TC_FL 用例覆盖） |
| 奇异矩阵（对角为 0） | 不检查，除零 → Inf/NaN（BLAS 未定义行为）；测试工程对 NON_UNIT 对角加 boost 保证非零，kernel 与 golden 用同一强化矩阵 |
| A 与 x 重叠 | 任务书禁止，不检测（对齐 cuBLAS） |

### 3.5 性能设计

任务书口径：NPU 平均单次 kernel 耗时（msprof `OpBasicInfo.csv` 的 `Task Duration`
按 kernel 名平均）≤ `gpu_baseline.csv` 的 `gpu_ms ÷ 0.8`；10 次有效采样、msprof 自带
5 次 warmup。

| case | n | k | uplo | trans | diag | GPU 实测 (us) | 达标预算 (us) | A 数据量 | 每行预算 (ns) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 8 | UPPER | N | NON_UNIT | 203.6 | 254.5 | 18 KB | 994 |
| 2 | 512 | 32 | LOWER | N | NON_UNIT | 395.1 | 493.8 | 135 KB | 964 |
| 3 | 1024 | 16 | UPPER | T | UNIT | 556.4 | 695.5 | 139 KB | 679 |
| 4 | 2048 | 64 | LOWER | C | NON_UNIT | 1812.4 | 2265 | 1.06 MB | 1106 |
| 5 | 4096 | 128 | UPPER | N | UNIT | 1974.0 | 2468 | 4.22 MB | 602 |

**瓶颈结论**：GPU 基线自身为串行核（482–883 ns/行），数据量极小（≤ 4.22 MB，
单核 GM 读一遍 ~200 us 量级），**预算由串行临界路径支配，不考验带宽**。

**单核面板算法的每行成本**：串行段 = ≤ W 项短点积（2–4 条向量指令）+ 复数除法
（~20 条标量/向量指令）+ UB 写回，估 100–300 ns/行；已知段批处理与装载按面板摊薄
< 50 ns/行；合计 150–350 ns/行。对 5 条 case 的余量：

| case | 每行预算/估算成本 | 余量 |
|---|---|---|
| 1 (n=256) | 994 / ~250 | ~4× |
| 2 (n=512) | 964 / ~250 | ~3.9× |
| 3 (n=1024, TC 列内) | 679 / ~180 | ~3.8× |
| 4 (n=2048, TC 列内) | 1106 / ~200 | ~5.5× |
| 5 (n=4096, N 跨列) | 602 / ~300 | ~2× |

最紧的 case 5（n=4096, k=128, N 模式）估算 ~1.2 ms vs 预算 2.47 ms，仍有 ~2×
余量；若 N 模式 vgather/转置通路实测不及预期，退到 KC 分块标量子块兜底（估算
~2 ms，仍压线达标）——两条通路都设计在 kernel 内，探针后定版。

**归因方法**：第一步即用 `msprof op --task-time=on` 拿单 kernel 耗时，不靠端到端
计时反推；已知候选优化：W 上调（UB 余量充足）、已知段 KC 合并、TC 模式列切片
2D 合并搬运。

### 3.6 workspace 与内存

原地算子，**零 workspace、零额外 Device 内存**（任务书 §3.4 内存要求不涉及，
天然满足）。多核扩展启用时需 P×k 复数的小 workspace（≤ 数 MB），届时按
`aclrtMalloc` 惰性分配并在 README 注明。

## 支持硬件

| 芯片 | 支持情况 |
|---|---|
| Atlas 800I/T A2（910B3 / 910B4 等） | 支持（arch22 / DAV-2201），CANN 9.1.0+ |
| Atlas A3 训练/推理系列（910_93 等） | 支持（arch22 / DAV-2201） |

自验证优先 910B3（精度与性能）；交付含 910B3 与 A3 两个系列的功能/精度/性能报告
（任务书 §7：A2 与 A3 均需完成验收）。

## 算子约束限制

1. 仅 COMPLEX64；不支持其他 dtype（对齐任务书，不做泛化）。
2. uplo ∈ {UPPER, LOWER}、trans ∈ {N, T, C}、diag ∈ {NON_UNIT, UNIT}，非法枚举
   返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书口径；arch35 返回 INVALID_ENUM，
   两架构行为差异在 README 注明）。
3. n ≥ 0、k ≥ 0、lda ≥ k+1（与 n 无关）、incx ≠ 0 且 ≠ INT_MIN；n>0 时 A、x 非空。
4. 不支持超出 lda/incx 语义的非连续内存；不支持 A 与 x 内存重叠；不返回视图。
5. x 原地读写（b 存于 x、解写回 x）；写回严格按 incx 步长散射，不触碰步长间隙。
6. 不做奇异性检查；diag=NON_UNIT 对角为零时结果为 Inf/NaN（未定义行为）。
7. 异步执行：依赖 `aclblasSetStream`；Host 侧无隐式同步，读回结果前须同步 stream。
8. 确定性：单核串行实现天然逐位确定（超出任务书「不要求」的下限）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | COMPLEX64 混合容差：逐元素 `|actual−golden| ≤ atol + rtol×|golden|`，atol = rtol = 2⁻¹³ (1.22e-4)（实/虚部分别按 FLOAT32 判定）；matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32 ULP（大数规约误报时可放宽 2 ULP） | 任务书 §3.2 / opbase mixed_tolerance_standard |
| 性能标准 | 5 条典型 case（及 TC_PF 全 200 条关联比对）：NPU 平均单次 kernel 耗时 ≤ gpu_ms ÷ 0.8，msprof 口径，10 次有效采样 | 任务书 §3.3 / 测试指导 |
| golden | Netlib BLAS `ctbsv` 单标杆生成，对输出 x 全元素验证 | 任务书 §3.1 |

## 测试策略

任务包提供 1200 条 csv 用例（`gen_csv.py` 固定种子 20260823，逐字节可复现）：
1000 精度（TC_L0 12 全枚举 / TC_SQ 23 尺寸 / TC_BD 22 带宽含 k=0、半带、满带 /
TC_DG 12 对角语义 / TC_LD+TC_INC 18 前导维与 ±步长 / TC_FL 11 填充含 Inf/NaN /
TC_CV 24 / TC_ED 14 边界负向 / TC_EX 864 扩展）+ 200 性能（TC_PF，均纯连续访存）。
自补重点（任务包未覆盖或需强化处）：

| 补充点 | 说明 |
|---|---|
| 归一化 8 组合 × 2 方向交叉 | uplo×trans 全组合在大尺寸（n ≥ 1024）下的方向/寻址正确性（任务包 EX 已抽样，自补系统化对拍表） |
| 满带大尺寸 | k = n−1、n ≥ 512 的 KC 分块路径（精度，无性能要求） |
| 负步长大规模 | incx < 0、|incx| > 1 与 ±1 结果一致性（同一数学输入） |
| 对角 boost 一致性 | 复核 golden 与 kernel 使用同一强化矩阵（测试工程行为，非 kernel 逻辑） |
| guard 区 | x 缓冲前后哨兵，检测步长散射越界写 |

复现步骤（写入测试 README，验收人可复现）：

```bash
python gen_csv.py                                   # 生成 tbsv_test.csv / gpu_baseline.csv
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --csv ./tbsv_test.csv
msprof op --application="./build/test/tbsv/ctbsv/ctbsv_test --gtest_filter=*TC_PF*" \
         --output=./prof_ctbsv                      # OpBasicInfo.csv 按 kernel 名平均
```

## 兼容性分析

- **arch22 为全新目录**，不触碰 arch35 任何行为；公共头 `include/cann_ops_blas.h`
  的 `aclblasCtbsv` 声明已存在（arch35 已登记），本任务仅需确认签名一致，无接口变更。
- **README 支持矩阵**：`blas/tbsv/README.md` 中 aclblasCtbsv 的 Atlas A2/A3 系列
  由「不支持」翻转为「支持」，并注明错误码口径差异（非法枚举：arch22 按任务书返回
  INVALID_VALUE，arch35 返回 INVALID_ENUM）。
- **与 950（arch35 增强）任务的关系**：`09-54-aclblasCtbsv-950` 为 A5 侧任务，
  两者代码目录（arch22 vs arch35）与测试目录（test/tbsv/ctbsv/arch22 vs arch35）
  互不重叠，无公共文件冲突；语义以任务书与本设计 §3.3 归一化表为共同基准。
- **SOC 网关**：测试框架按 SOC 版本网关跳过不适型号（沿用仓内 `skipped_tests.list`
  机制），arch22 目标 SOC 为 `ascend910b*` 与 `ascend910_93*`。
- **版本兼容**：CANN 9.1.0 及后续配套。

---

# 风险点

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | 8 组合方向/寻址归一化的移植正确性（LOWER/UPPER × N/T/C 的存储行 r 公式、方向翻转、共轭） | 错一组即精度全红 | §3.3 归一化表作为唯一分派依据；TC_L0 12 组合先行；Host 侧单测直接对拍寻址表 |
| 2 | 复数除法数值（直接法在极端幅值下溢出） | 个别用例超容差 | 测试值域内无风险；Smith 缩放法作为兜底开关（编译期二选一，探针后定版） |
| 3 | N 模式跨列通路（vgather / UB 转置）的 arch22 实测效率 | case 5 性能余量收窄 | 两条通路 + KC 标量子块三级兜底；上板首日探针定版 |
| 4 | 面板参数（W/KC）与 UB 峰值 | 性能或 UB 溢出 | W 自适应公式 + UB 预算表（< 60 KB / 192 KB）；msprof 归因后调参 |
| 5 | 每行串行成本估算（100–300 ns）偏差 | case 5 压线 | 估算对预算有 ~2× 余量；必要时启用 §3.3 多核扩展（tiling 已预留） |
| 6 | A3（910_93）环境晚到位 | 双系列验收进度 | 架构同源（arch22），先 910B3 全量自验，A3 到位后补跑 |
| 7 | 错误码口径与任务书/测试期望不一致（INVALID_ENUM vs INVALID_VALUE） | 负向用例失败 | arch22 严格按任务书 INVALID_VALUE；TC_ED 负向用例先行验证 |

# 交付物

1. **算子设计文档**：本文件，PR 至 `cann-ops-competitions`
   `04_tasks/01_community-task-2026/tasklist/09-72-aclblasCtbsv-A2A3/<gitcode账号>/docs/design.md`。
2. **代码**（PR 至 `ops-blas` `master`，任务书 §5 结构）：`blas/tbsv/arch22/`
   host/kernel/tiling 四件套 + 复数算术头；`blas/tbsv/README.md` 支持矩阵更新；
   `include/cann_ops_blas.h` 确认声明。
3. **自测用例与测试代码**：`test/tbsv/ctbsv/arch22/`（csv 驱动 GTest，镜像仓内
   blas 测试结构）+ 任务包 1200 条 csv + 复现 README（环境/编译/步骤）。
4. **自测报告**：用例参数、精度对比结果及截图（1000 条精度全量 + 自补用例）、
   性能数据（5 条典型 case + TC_PF 200 条 msprof kernel 口径，10 次有效采样平均）
   及截图、内存说明（零额外内存）、910B3 与 A3 双系列结果。
