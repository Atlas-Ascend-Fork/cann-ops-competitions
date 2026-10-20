# 需求背景（required）

## 需求来源

社区任务「单精度实数 QR 分解与最小二乘（950）」；任务书
`Atlas950_Sgeqrf_Sorgqr_Sormqr_Xlarft_SSgels_task_doc.md`；目标仓 `cann/ops-solver`；
入仓形态按仓内既有算子规范（算子目录 + 架构子目录 `arch35/`）。

## 背景介绍

### 算子功能定位

四个接口构成 BLAS/LAPACK 的实数单精度 QR 与最小二乘完整链路：

| 接口 | LAPACK 对应 | 作用 |
|---|---|---|
| `aclsolverSgeqrf` | `sgeqrf` | 列主序 A(m×n) 的 QR 分解，输出打包反射子与 `tau` |
| `aclsolverSorgqr` | `sorgqr` | 由 geqrf 的打包反射子生成前 n 列正交 Q(m×n) |
| `aclsolverSormqr` | `sormqr` | 用 Q 或 Qᵀ 左乘/右乘一般矩阵 C |
| `aclsolverSSgels` | `sgels` | 最小二乘：min‖B − A·X‖₂，A(m×n) 且 m ≥ n |

链路上 `Sgeqrf` 是基础，`Sorgqr`/`Sormqr` 消费它的 Householder 表示，`SSgels` 复用它 + 上三角回代。

### 业界对标

对标 CUDA cuSolver DN 的 `cusolverDnSgeqrf/Sorgqr/Sormqr/Sgels`：同为列主序、Device 指针、
真实写 `info`、支持 `lda ≥ max(1,m)` 的 padding。差异：本实现按 Ascend 950 的 Cube(AIC)+Vector(AIV)
异构核重排算法（分块 Householder + WY 表示 + Cube 板 GEMM），并额外要求**确定性**（同输入同 stream
反复执行结果逐位一致）。

### 工程形态

`ops-solver` 仓内四算子目录，各含 `arch35/`（`*_tiling.h` / `*_kernel.cpp` / `*_host.cpp`）；
公共入口 `include/cann_ops_solver.h` + `include/cann_ops_solver_common.h`。
`arch35` 源按 `ARCH_SRC_FILES` 通配，仅 `--soc=ascend950` 收取。

### 接口规格

按仓内 ACLSOLVER 风格：`aclsolverHandle_t`（内含 stream）在前，其后「维度 → 数组 → 步长 → 工作区 → info」。

```c
aclsolverStatus_t aclsolverSgeqrf        (handle, int m, int n,     float *A, int lda, float *TAU,
                                          float *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverSorgqr        (handle, int m, int n, int k, float *A, int lda,
                                          const float *TAU, float *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverSormqr        (handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
                                          int m, int n, int k, const float *A, int lda, const float *TAU,
                                          float *C, int ldc, float *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverSSgels        (handle, int m, int n, int nrhs, float *dA, int ldda,
                                          float *dB, int lddb, float *dX, int lddx, void *dWorkspace,
                                          size_t lwork_bytes, int *niter, int *dinfo);
```

另有各自 `*_bufferSize`。枚举（`cann_ops_solver_common.h`）：`aclsolverSideMode_t{LEFT=141,RIGHT=142}`、
`aclsolverOperation_t{N=111,T=112,C=113}`，取值与任务书枚举表一致。

### 数学公式

- **geqrf**：`A = Q·R`，`Q = H₁H₂…H_k`，`H_i = I − τᵢ·vᵢ·vᵢᵀ`（`vᵢ` 首元为 1，`k = min(m,n)`）；
  打包形式：`vᵢ` 的第 `i+1..m` 个分量存于 `A(i:m, i)` 下三角，`τᵢ` 存 `TAU(i)`，`R` 为 `A` 的上三角。
- **orgqr**：由 `(vᵢ, τᵢ)` 显式构造 `Q` 的前 `n` 列。
- **ormqr**：`C ← op(Q)·C`（LEFT）或 `C ← C·op(Q)`（RIGHT），`op(Q) = Q` 或 `Qᵀ`。
- **gels**：`min_X ‖B − A X‖₂`，`X` 为 `n × nrhs` 独立输出（不与 `B` 共用存储）。

---

# 需求分析（required）

## 需求描述

四个接口在 950PR 上实现并达到任务书全量精度判据与全量性能门限；允许 lda padding；
`info` 契约（QR 族为 bad_param → `-1`）与数值精度各出独立结论。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义 + 哪些量可由代数性质免费获得 | 见上文公式。**免费的量**：`R` 直接是打包输出的上三角；`Q` 的显式矩阵无需构造即可用于 `ormqr`（用反射子作用，比先构造 Q 再乘快 O(m) 倍）；`ssgels` 的 `X₁:n` 与 `B₁:n` 可原地复用、`X_{n+1:m}` 恒为残差不必算。 |
| 2 | 接口原型 | 见「接口规格」。 |
| 3 | 参数规格（取值与非法值） | `m,n ≥ 0`；`geqrf` 允许 m<n（k=min(m,n)）；`orgqr` 要求 `0 ≤ k ≤ n ≤ m`；`ormqr` 要求 `k ≤ (LEFT? m : n)`；`gels` 要求 `m ≥ n`；`lda ≥ max(1,m)`、`ldc ≥ max(1,m)`、`ldx ≥ max(1,n)`；`lwork ≥ _bufferSize`；空指针非法。非法参数**经 `info` 上报并正常返回**（LAPACK 语义，见「兼容性分析」）。 |
| 4 | 支持数据类型 | `float32`（本次任务仅单精度实数）。 |
| 5 | 目标芯片与架构 | **Ascend 950PR → arch35 / dav-3510**（与任务书 §3.3 一致）；CANN 9.1.0。 |
| 6 | 精度要求（可溯源） | 任务书 §3.2.2：`geqrf` SQRT01 `max(5·ratio_cpu, 0.1)`；`orgqr` SQRT02 `‖I−QᵀQ‖₁/(m·ε)`；`ormqr` SQRT03 反射子形式 vs 显式 Q 形式之差；`gels` SQRT16/17（方阵/超定）；`ε = 2⁻²⁴`。 |
| 7 | 性能预期（可溯源） | 任务书 §3.3：`T_NPU ≤ avg_ms / 0.35`，`avg_ms` 取 `bench_result.json` 中 CUDA cuSolver 实测均值。四个基准档门限实测为 512→4.57ms、1024→10.84ms、2048→25.09ms、4096→72.71ms（**注意门限本身含 1/0.35≈2.857 的倍率**）。 |
| 8 | 代码架构选型 + 可行性 | **Cube(AIC) + SIMT(AIV) 混合**：Householder 面板用 SIMT（天然串行、数据量小），尾随更新用 Cube tensor_api GEMM（占绝对算力）。可行性已实测排除两条死路：ops-solver 既有 MemBase 内核在 950PR 上 `aclrtGetHardwareSyncAddr` 返回 **207000 FEATURE_NOT_SUPPORT**、绕过后挂死；经典高层 Matmul API (`matmul_intf.h`) 报 507015 后挂死。 |
| 9 | 评测来源 | 任务自带评测输入（`package/canonical_cases.json` + 包内冻结 `ratio_cpu`）+ 包自带 `verify_accuracy.judge`；不使用自建 golden 作结论。 |

---

# 详细设计（required）

## 算子分析

### 数学公式

分块 QR（WY 表示）：块 `k0..k0+kb-1` 的反射子聚成 `Q_b = I − V·T·Vᵀ`，于是尾随更新
`A22 ← A22 − V·(Tᵀ·(Vᵀ·A22))`（**注意是 Tᵀ 不是 T**：QR 尾随更新需要 `Q_bᵀ`）。
`V` 为 `(m−k0)×kb` 的单位下梯形（对角 1、严格上三角 0），`T` 为 `kb×kb` 上三角。
`T` 由 Larft 递推得到：`T(i,j) = −τⱼ·( G(i,j) + Σ_{p<i} T(p,i)·G(p,j) )`，
其中 `G = VᵀV`（Gram 矩阵）——把 Gram 交给 Cube GEMM 是本设计的关键（见下）。

### 支持数据类型

`float32`（A/B/C/X/TAU/workspace/info；`info` 为 `int32`）。

### 支持形状

| 接口 | 约束 |
|---|---|
| geqrf | `m ≥ 0, n ≥ 0`，允许 `m<n`（`k=min(m,n)`） |
| orgqr | `0 ≤ k ≤ n ≤ m` |
| ormqr | `k ≤ (LEFT ? m : n)`；LEFT 时需 `m ≥ k`，RIGHT 时需 `n ≥ k` |
| gels | `m ≥ n ≥ 0`，`nrhs ≥ 0` |

全覆盖：方阵/高矩阵/宽矩阵（geqrf）、部分块（`k % kb ≠ 0`）、`k<n`、四种 side/trans 组合。

## 算子实现

### 实现方案

**代码架构（分层）**

```
include/cann_ops_solver.h            公开入口（4 算子 + bufferSize + 枚举）
src/utils/arch35/gemm_cube_tiling.h  Cube FP32 GEMM 组件（tiling 结构 + 帧约定）
src/utils/arch35/gemm_cube_kernel.cpp tensor_api 四段搬运 + Mmad + Fixpipe
src/sgeqrf/arch35/{_tiling.h,_kernel.cpp,_host.cpp}   面板 SIMT + Larft SIMT + host 编排
src/sorgqr|sormqr|ssgels/arch35/…                      复用同一 GEMM 与 Larft
```

**GEMM 组件**（精简移植 `ops-blas blas/gemm/arch35`，去掉 tiling API 依赖以免 optiling 未定义符号连坐）：
导出 `gemm_cube_do(coreNum, stream, a, b, c, t)`；`t` 描述 `m',n',k'`+三个前导维+两个转置标志+
是否原子累加。帧约定（实测钉死）：**ND** `X[i][j]=buf[i*ld+j]`（列主序操作数、开启转置时用）、
**DN** `X[i][j]=buf[i+j*ld]`。分块常量：`BASE_M=128`、`BASE_N=16`、`BASE_K=32`、`L1=128KB`。

**sgeqrf 数据流（每块 kb 列，共 ⌈k/kb⌉ 块）**

1. **面板**（SIMT，`<<<1,1024>>>`）：逐列 Householder，写打包形式到 `A`，同时把显式 `V` 写入
   工作区 `Vw`（对角 1、严格上三角 0、下方放反射子）。分 4 阶段、每反射子 4 个块栅栏：
   ① 全线程求 `Σ A(r,i)²`（4 条累加链 + warp 蝶形）② warp0 折叠并发布 `alpha/tau/1/(x1−alpha)`
   ③ 归一化列 i 并把 `Vw[:,j]` 与 UB 的「反射子列 i」缓存一次写好 ④ **每目标列两个 warp**
   各占一半行块做点积与更新（半和经 scratch 合并）。
2. **Gram**：Cube GEMM，`A=B=Vw`、`isTransB=1`、`m'=n'=kb`、`k'=rows` → `G`（行主序，`ldc=kb`）。
3. **Larft**（SIMT）：`G` 载入 UB，按列递推 `T`；输出**取负**、列主序、**前导维 = 调用方 ldb**；
   `T` 的列快照必须与 `g_tfac` 同数组（见「风险与对策」）。
4. **尾随更新**：三次 Cube GEMM —— `Z=Vᵀ·A22`（`isTransA=1`）、`Y=Tnegᵀ·Z`、`A22 += V·Y`（GM 原子累加）。
   `n2 = n−(k0+kb) = 0` 时跳过 2~4 步（空块）。

**sorgqr**：自 `k` 向 0 **降序**过块；每块从 `A` 取反射子建 `Vw`（`sorgqr_build_v_do`）→ Gram →
Larft → 同样三次 GEMM 作用在单位阵工作区 `W`（`nk×(m−k0)`）→ 末尾拷回 `A`。

**sormqr**：按 `side/trans` 四个分支（顺序 `descending = ((side==LEFT)==(trans==N))`、
块形式 `useTtrans = (trans!=N)`）；每块建 `Vw` → Gram → Larft → 三次 GEMM 把反射子作用到 `C`，
累加（原子）或覆盖按分支决定。

**ssgels**：`geqrf(A)` → 用同一套反射子机制把 `Qᵀ` 作用到 `B` → 上三角回代得 `X`
（`X` 独立缓冲，不回写 `B`）；`m<n` 或 `nrhs=0` 走快速返回。

**Buffer / 工作区规划**（`*_bufferSize` 返回浮点元素数，除 `SSgels` 为字节数）：

| 区域 | 规模 | 说明 |
|---|---|---|
| `Vw` | `m × kb` | 显式单位下梯形 V |
| `Z` / `Y` | `n × kb` | ①③ 的中间量 |
| `T` | `kb × kb` | WY 的 T 因子（列主序，ld=调用方 ldb） |
| `panelScratch` | 128 | 面板跨 warp 折叠（32 partial + alpha/tau/inv + 2-warp 半和） |

SIMT 侧 UB 占用：`g_colI[4096]`(16KB) + `g_gram[1024]`(4KB) + `g_tfac[2048]`(8KB) = 28KB ≤ UB。

**Tiling 策略**：`kb = GEQRF_BLOCK_SIZE = 16`（实测最优，见下）；块序列 `k0 = 0, kb, 2kb, …`，
末块 `kb' = k − k0`（可为 1..15，即「部分块」，是 bug 高发区）。多核：面板单核（串行语义），
GEMM 用 `min(24, 核数)` 核（`GEMM_TILE_N=128` 切 n，`m'` 按 `BASE_M=128` 切）。

**实测定界与优化（全部为 950PR 实测，non-推算）**

| 优化项 | 效果（4096 档） |
|---|---|
| 面板：块级栅栏 → 每反射子 4 栅栏 + warp 蝶形 | 145.7ms→8.89ms（512 档） |
| 面板：单线程折叠 32 个 GM partial → warp0 并行 | 3.75→2.99ms（512 档） |
| 面板：stage4 每目标列 2 个 warp（关键路径 rows/32→rows/64） | 84.5→79.7ms |
| 面板：反射子列缓存进 UB（模板分派） | 87.2→84.5ms |
| GEMM `BASE_K` 8→32（Mmad 32×16×32） | 34.3→33.6ms 全档受益 |
| GEMM `BASE_M` 32→64 + L1 32→64KB | 109.8→88.2ms |
| **GEMM `BASE_M` 64→128 + L1 64→128KB** | **79.7→69.9ms（一举过门限）** |
| `kb` 32→16 | 各档全面更优（面板冗余读 ∝ kb） |

**最终实测性能**：512→2.64ms、1024→6.34ms、2048→18.29ms、4096→69.92ms
（门限 4.57 / 10.84 / 25.09 / 72.71），512 档全程提速 55×。小尺寸（2×2，门限 40.6µs）实测 38.4µs；
该档为**单 kernel**（`n2=0` 时只跑面板），~38µs 是 launch/sync 固定开销，官方 ΣTask Duration 口径下余量更大。

**Bring-up 阶梯（M1→M4）**：M1 单核面板 32×32 与 scipy 逐项对齐 → M2 接入 Cube GEMM 组件并跑
probe（12/12）→ M3 打通 64² 全链路（逐阶段 dump 比对 ①Z/②T/③Y/④A22）→ M4 铺规模与形状面
（512/1024/2048/4096 + 部分块 + m<n + 四 side/trans）。

**落仓与交付形态**（对齐任务书 §7 的交付结构）：

```
src/sgeqrf/arch35/{sgeqrf_tiling.h, sgeqrf_panel_kernel.cpp, sgeqrf_host.cpp}   # 四算子各一份
src/sorgqr/arch35/   src/sormqr/arch35/   src/ssgels/arch35/
src/utils/arch35/{gemm_cube_tiling.h, gemm_cube_kernel.cpp}      # 共享 Cube FP32 GEMM 组件
include/{cann_ops_solver.h, cann_ops_solver_common.h}           # 八个公开接口（4 算子 + 4 bufferSize）
test/sgeqrf/  test/solver_case_runner/  test/gemm_cube/          # 测试工程（见下）
docs/zh/{sgeqrf,sorgqr,sormqr,ssgels}.md   docs/api_list.md      # 算子文档与接口列表（§7 要求）
```

> **测试结构说明（有意偏离 §7 的 `test/<op>/` 四目录形式）**：四个算子的用例驱动是**同一份**
> （`test/sgeqrf/run_cases.py` 按 `--op` 参数化 + `test/solver_case_runner` 按 `meta.txt` 承载各算子的
> 维度/枚举），各算子之间的差异只在入参，做成四份会直接违反 §7 另一条「不得包含明显冗余内容」；
> 因此按「一个共用驱动 + 一个 geqrf 专用 runner（其参数布局与其余三者不同）」组织，并在
> `02-测试代码/README.md` 写清对应关系。

## 支持硬件

Ascend 950PR（Atlas 950），arch35 / dav-3510，CANN 9.1.0。依赖：Cube（AIC）+ Vector（AIV）
异构核、`tensor_api` 的 L1/L0A/L0B/L0C 与 Fixpipe、SIMT（`__simt_vf__` / `asc_shfl_xor` / `asc_syncthreads`）、
GM 原子累加（`SetAtomicAdd`）。

## 算子约束限制

- 仅 `float32`；仅列主序、Device 指针（无 row-major 枚举，ORDER 固定 102）。
- `geqrf` 允许 `m<n`；`orgqr` 必须 `n ≤ m`；`gels` 必须 `m ≥ n`（超定）。
- `lda/ldc/ldx` 允许 padding（`> max(1,m/n)`），代码按步长寻址；本次任务包用例恒为最小合法值。
- 工作区必须由调用方按 `*_bufferSize` 分配；`lwork` 不足 → `info=-7`。
- 确定性：同输入同 stream 反复执行 `tau`/`info` 逐位一致（面板 γ 用除法定式、GEMM 累加序固定）。

---

# 可维可测分析

## 精度标准/性能标准

**精度（口径溯源：任务书 §3.2.2 与各包 `verify_accuracy.py`）**

| 接口 | 判据 | 阈值 | 出处 |
|---|---|---|---|
| geqrf | SQRT01 `ratio = ‖R − QᵀA‖₁ / (m·‖A‖₁·ε)`，`R=triu(out32)`，`Q=orgqr(v,τ)` | `max(5·ratio_cpu, 0.1)` | §3.2.2.1 |
| orgqr | SQRT02 `ratio = ‖I − QᵀQ‖₁ / (m·ε)` | `max(5·ratio_cpu, 0.1)` | §3.2.2.2 |
| ormqr | SQRT03 反射子形式结果 vs 显式 Q 形式结果之差 | `max(5·ratio_cpu, 0.1)` | §3.2.2.3 |
| gels | DQRT16（方阵）/ DQRT17（超定，`R=B−A·X`） | `max(5·ratio_cpu, 0.1)`；超定取 `ratio_opt_cpu` | §3.2.2.4 |

其中 `ε = 2⁻²⁴`；**参考值 `ratio_cpu` 只读包内冻结 index，不现场重算替代**（任务包 README 原文）。
info 契约：QR 族 `info_probe: bad_param` → 期望 `info == -1`，只比 info、不做残差、不进精度结论。

**性能（口径溯源：任务书 §3.3）**

`T_NPU ≤ avg_ms / 0.35`，`avg_ms` 为 `bench_result.json` 中 CUDA cuSolver 实测均值；
四档基准门限（本次实测）：512→4.57ms、1024→10.84ms、2048→25.09ms、4096→72.71ms。
**注**：任务书「若 case 无 prof 数据则为非自验必跑项」——本包 694 条逐条都有 `perf.avg_ms`，故全量必跑。
⚠ 本机 `msprof`/`msopprof` 无法采集任何 kernel（连 `ops-blas` 参考构建也 `0 success`、`Task Duration=0`），
属环境限制；本文性能数字为 `wall-clock + aclrtSynchronizeStream` 代理口径（**含 kernel 间 launch 间隙，偏保守**），
官方 ΣTask Duration 口径需在验收环境补采。

## 测试概览

**精度：全量 T2 = 706/706 通过**（判定用各包自带 `verify_accuracy.judge` + 包内冻结参考值）

| 算子 | 数值 | info 契约 | 合计 |
|---|---|---|---|
| sgeqrf | 174/174 | 3/3 | **177/177** |
| sorgqr | 165/165 | 3/3 | **168/168** |
| sormqr | 189/189 | 3/3 | **192/192** |
| sgels | 166/166 | 3/3 | **169/169** |

分级执行：T0 冒烟（各算子前 5~8 条）→ T1 定向（前 40 条，含部分块与 `k<n`）→ **T2 全量一次**。

覆盖维度因子：形状（方阵 / 高矩阵 `m>n` / 宽矩阵 `m<n`）、块尾（`k % kb ≠ 0`，含 `kb=1`）、
`orgqr` 的 `k<n`、`ormqr` 四种 `side×trans`、`gels` 的 `nrhs = 1..8`、`info` 契约每算子 3 例。

**性能**：按任务书 §3.3 的**逐条口径**（对每一个 case 判 `T_NPU ≤ perf.avg_ms / 0.35`），全量 694 条实测为 **PASS 415 / FAIL 279 ⟹ 尚未达标**（sgeqrf 的四个基准档本身是过的：512 2.62ms≤4.57、1024 6.27≤10.84、2048 18.26≤25.09、4096 71.71≤72.71；小尺寸档最小 2×2，门限 40.6µs，实测 38.4µs）。瓶颈定位为**逐块 host 侧发射结构**（每个反射块约 6 次 kernel 发射，k=4096 时约 1500 次发射 ≈ 10~15ms，而门限仅 1~2ms）叠加 **K=kb=16 偏窄、Cube 深度只用一半**的 GEMM；块宽受 `GEQRF_LARFT_KB_MAX=32` 封顶（实测调到 32 零收益）。修复方向为把逐块发射收敛进核内循环（融合 kernel）并扩 grouped GEMM。

**补充自检（任务包用例未覆盖、本任务已实测通过）**：`test/sgeqrf/extra_checks.py`，判定用任务书
§3.2.2.1 官方残差式自判（ε=2⁻²⁴）：

| 检查 | 用例 | 结果 |
|---|---|---|
| **ld padding**（四算子全覆盖；含 **padding 区不得被写坏**） | geqrf 64×64/lda=71、128×100/lda=140、65×33/lda=68；orgqr 64×40/lda=75；ormqr 64×48/lda=69,ldc=73；gels 70×36/ldda=73,ldb=74,ldx=42 | **7/7 PASS**：geqrf ratio 8.6e-2 / 4.7e-2 / 7.7e-2；orgqr SQRT02 0.669 ≤ 5×参考 1.611；ormqr ratio 8.9e-2；gels 解与 numpy lstsq 同残差（1.153e7 双侧一致）；**geqrf/orgqr/ormqr 的 padding 行逐字节原样** |
| **空问题** `m`/`n` = 0 | 0×0、0×5、5×0 | 3/3 PASS：`info=0`，不崩（runner 已为零尺寸分配加下限保护） |
| **确定性** | 96×96 同输入同 stream 重复执行 | PASS：`out32` 与 `tau` **逐位一致** |
| 非默认 stream | — | 已天然覆盖：驱动自建 `aclrtCreateStream`，算子全程跑在非默认 stream 上 |

**按任务书 §3.5「功能自验至少覆盖」补齐的项（全部 PASS）**：阈值改用**官方口径** `max(5·ratio_cpu, 0.1)`
（`ratio_cpu` 取 CPU 参考实现自身的残差比，非固定 0.1）：

| §3.5 项 | 覆盖情况 | 结果 |
|---|---|---|
| 形状：2 的幂**与 2 的幂-1** | `31 / 63 / 127` 方阵（部分块最吃紧的形状） | 6/6 PASS（ratio 5.6e-2 ~ 1.87e-1 ≤ 官方阈值） |
| 入参分布 | `U[-5,5]` 与 `正态(μ∈[-5,5], σ∈[0.1,2])` 各半 | 均已覆盖（上表 6 例即两种分布 × 三种形状） |
| padding：最小合法值 及 **+8 / +32** | `geqrf lda = m+0 / m+8 / m+32` | 3/3 PASS，且 padding 行逐字节原样 |
| side/trans：LEFT/RIGHT、**N/T/C** | ormqr LEFT 的 N/T/C 与 RIGHT 的 N/T | 5/5 PASS（ratio ≤ 1.16e-1 ≤ 官方阈值） |
| hyper定 SSgels：**m<n 必须报错**、nrhs=0/多右端、m=n 与 m>n | `m<n` 被拒（bufferSize 报错）；`nrhs = 0 / 8 / 32 / 128` | 5/5 PASS（nrhs=0 成功返回；nrhs≥8 与 numpy 解 max\|Δ\| ≤ 1.6e-5） |
| 空问题：`k=0` | `orgqr k=0` | PASS（**修出真 bug**：原实现直接早退、A 未被写；按 LAPACK 语义改为写 `Q=I` 的前 n 列，现 `max\|Δ\| = 0`） |

> 判定用**官方 1-范数口径**（矩阵 1-范数 = 最大列和，与包内 `_norm1` 一致；ε=2⁻²⁴）。
> 注：`gels` 的 `X` 用独立缓冲、`ldx` 的 padding 行是设备未初始化内存，故只断言解的正确性，不断言其 padding 未被写。
> 自检脚本：`test/sgeqrf/extra_checks.py`（可直接重跑复现上表全部结论）。

**仍未覆盖（如实列出）**：多 handle 并发调用。

## 兼容性分析

| 维度 | 行为 |
|---|---|
| dtype | 仅 `float32`（本任务范围）；`info` 为 `int32` |
| 形状越界 | 不满足约束（如 `gels` 的 `m<n`、`orgqr` 的 `n>m`）**经 `info` 上报负值并正常返回**，不抛异常、不部分写内存 |
| 枚举取值 | `side ∈ {LEFT=141, RIGHT=142}`、`trans ∈ {N=111, T=112, C=113}`（与任务书枚举表一致）；非法枚举 → `info=-1` |
| 异常入参 | 空指针 / 负维度 / `lda < max(1,m)` / `lwork < bufferSize` → 对应 `info=-1..-7`，LAPACK 参数编号（handle 不计入） |
| 空问题 | `k = min(m,n) = 0` 或 `nrhs = 0` → 直接返回 `info=0`，不启动 kernel |
| 降级 | 无低精度/近似通路；同输入同 stream 结果确定 |

**参数错误语义（重要，与直觉不同）**：参数非法时**不返回错误状态码**，而是按 LAPACK 约定
「调用正常返回 + `info` 指认非法参数」。这是任务包 info 契约的硬要求（judge 要求 `status=="ok"` 且
`info==k_expected`），四个 host 的参数校验块统一为 `return WriteInfo(devInfo, badParam, stream);`。

**info 编号与参数顺序（已核验符合 §2.1.3「−i 指第 i 个参数、handle 不计」）**：本实现严格按 §2.4.1 的顺序编号——
`m→-1`、`n→-2`、`A==nullptr 且 m,n>0→-3`、`lda<max(1,m)→-4`、`TAU==nullptr 且 min(m,n)>0→-5`、
`Workspace==nullptr→-6`、`Lwork` 不足→`-7`（`orgqr` 多一个 `k` 参数，其后编号顺移）；
与 §2.4.1 表格里「空指针且 m,n>0 时报错」「空指针且 min(m,n)>0 时报错」的措辞逐条对应。

**无 Host 同步（已核验符合 §2.1.4）**：`grep -rn "SynchronizeStream" src/` 命中的全部是仓内**既有**算子，
本任务四个算子的实现内**一处都没有**——计算全程提交到调用方 stream（`aclsolverSetStream` 设定），
不做任何不必要的 Host 同步。


