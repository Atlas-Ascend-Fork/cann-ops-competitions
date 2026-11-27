# aclblasCgemm3m 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasCgemm3m_AtlasA2_A3_task_doc.md》（2026-11-25 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemm/arch22/`，测试 `test/gemm/cgemm3m/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasCgemm3m算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数三乘法通用矩阵乘 `aclblasCgemm3m`
（对标 cuBLAS `cublasCgemm3m`，语义参考 Netlib `cgemm`）：

```text
C = alpha · op(A) · op(B) + beta · C        （complex64，列主序）
transa/transb ∈ {N, T, C}；复数算子中 OP_C 为共轭转置、OP_T 为（非共轭）转置，二者结果不同
3M 分解：复数乘法拆为 3 次实数矩阵乘（Gauss 3M），而非朴素 4 次
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasCgemm3m_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（cgemm3m_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasCgemm3m` 声明**已存在**（L584），与 `aclblasCgemm` 参数序列完全一致 | 无需新增/修改声明，直接按声明实现 arch22 版本（任务书 §2.3） |
| `blas/gemm/arch22/` | 实数 `aclblasSgemm` 已实现（`gemm_host.cpp` + `gemm_kernel.cpp/.h`，Ascend C Matmul API，INNER_PRODUCT、ND、双缓冲） | 本任务实现落点（与 Sgemm 同目录）；Cube 管线/tiling 传递/校验框架直接沿用 |
| `blas/gemm/arch35/cgemm3m_host.cpp` + `cgemm3m_rounding_kernel.cpp` | 950PR 版 3M 路径选择与缩放逻辑 | 3M 分解公式、非融合/融合路径切换、缩放与舍入处理的实现参照（不得直接复制 SDK 不兼容代码） |
| `test/gemm/cgemm3m/` | 仅有 arch35 目标（CMakeLists 需注册 arch22） | 测试工程复用 `cgemm3m_golden/param/verify` 公共件，新建 arch22 分支 |
| `blas/gemm3m/` | 实数 `aclblasSgemm3m`（3 个实矩阵沿 K 拼接）实现目录 | **禁止落位**：与复数 `aclblasCgemm3m` 不是同一接口族（任务书 §2.2 明确） |

### 功能缺口与本任务差异点

**arch22 Cgemm3m 整体缺失**。相对既有 Sgemm arch22 的差异即主要工作量：

1. **3M 复数拆解**：把 complex64 输入 deinterleave 为实/虚平面，按 3M 公式组织 3 次实数
   Matmul（同一次 launch 内完成，禁止三次独立 GEMM 调用拼接）；
2. **OP_C ≠ OP_T**（复数共轭转置）：非对称复数矩阵下两者结果不同，pack 阶段按 transa/transb
   独立翻转虚部符号；
3. **两条 3M 路径**（任务书 §2.1 给定）：非融合（P0=ArBr、P1=AiBi、P2=(Ar+Ai)(Br+Bi)）与
   融合含纠正重算（P0=(Ar+Ai)Br、P1=Ar(Bi−Br)、P2=Ai(Br+Bi)）——代数等价、舍入不同，
   按形状与片上容量选择，误差分析不得交叉套用；
4. **复数标量 epilogue**：alpha/beta 为 complex64（Host 内存），交叉缩放
   `Cr=αr·Pr−αi·Pi+βr·Cr−βi·Ci`、`Ci=αr·Pi+αi·Pr+βr·Ci+βi·Cr`；
5. **精度域**：COMPLEX64 按 FLOAT32 分量判定（2⁻¹³ 档），Cube FP16 Mmad 必须叠加 hi/lo
   补偿平面（或走 Matmul API FP32 路径）才能守住阈值。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemm/arch22/` 实现 `aclblasCgemm3m`：transa/transb 全 9 组合（N/T/C）× m/n/k 覆盖
0/1/2/3/5/7/16…4095/4096（含 4095³/4096³ 立方档）× lda/ldb/ldc（紧凑 + +1/+8/+64 padding）×
alpha/beta（特殊值 (0,0)、(1,0)、(0,1)、(-1,0)、±1e4 + 随机值）× Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准 FLOAT32 档（实部/虚部分别判定：rtol = atol = 2⁻¹³，
matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP）；golden 为 Netlib CBLAS 3.10.0
`cblas_cgemm`（非 3M 直接复数乘参考；3M 与其舍入过程不同，不逐位比对，按阈值判定）。
性能 msprof op 模式 task duration，5 次 warmup + 10 次有效采样均值，
逐 case ≤ H100 gpu_ms × **1.25**（即 ≥ 0.8× GPU kernel 标杆）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `C = α·op(A)·op(B) + β·C`，3M 三实积单 launch | §2.1 | 3M Cube 管线（§3.2.2） |
| R02 | OP_C 共轭转置（≠OP_T）；非法枚举 → `INVALID_ENUM`；校验顺序 handle→trans→m/n/k→ld→标量/指针 | §2.4/§2.5 | Host 校验链 + pack 符号翻转（§3.2.1） |
| R03 | m=0 或 n=0 no-op；k=0 或 α=0 只做 `C=β·C`；β=0 不读 C | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R04 | alpha/beta 为 **Host 内存** complex64 标量（与 cuBLAS 默认 host pointer mode 一致） | §2.4 | Host 直读标量（§3.2.1） |
| R05 | 仅写 C 逻辑 m×n 区域，ldc padding 列不修改 | §2.1 | epilogue 裁剪写回（§3.2.2） |
| R06 | 3M 两路径（融合/非融合）按形状与片上容量选择 | §2.1 | 路径选择模型（§3.2.2） |
| R07 | 精度 FLOAT32 档 + Inf/NaN 等效 | §3.2 | hi/lo 补偿 + 测试工程（§4.2） |
| R08 | 性能逐 case ≤ 1.25×H100（msprof op task duration，kernel-to-kernel） | §3.3 | 性能反算与预算（§3.2.5） |
| R09 | arch22 GTest 注册（`test/gemm/cgemm3m/CMakeLists.txt`） | §2.2 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式（3M 两路径）

记 Ar、Ai、Br、Bi 为 A、B 的实/虚平面（列主序），op 按列主序视图装载（T/C 翻转装载方向，
C 额外对虚部平面取负）：

- **非融合路径**：P0 = op(Ar)·op(Br)，P1 = op(Ai)·op(Bi)，P2 = op(Ar+Ai)·op(Br+Bi)，
  `op(A)·op(B) = (P0 − P1) + i·(P2 − P0 − P1)`；
- **融合路径（含纠正重算）**：P0 = op(Ar+Ai)·op(Br)，P1 = op(Ar)·op(Bi−Br)，
  P2 = op(Ai)·op(Br+Bi)，`op(A)·op(B) = (P0 − P2) + i·(P0 + P1)`。

两条分解代数等价，中间项 FP32 舍入不同（P2−P0−P1 与 P0+P1 的抵消结构不同）；
实现按形状与片上容量选择路径（大 k 用非融合——P2 的输入和 (Ar+Ai)、(Br+Bi) 不会溢出
FP16 表示域时的补偿平面更规整；接近值域边界或小形状用融合——少一次平面加法）。

3M 相对朴素 4 实积节省 25% 实数乘法量：arch22 Cube FP16 Mmad 吞吐下，
3M 把复数 GEMM 的有效算力从 1/4 提升到 1/3 峰值实数乘。

### 3.1.2 支持数据类型

COMPLEX64（interleaved float32 re/im，`aclblasComplex`），A/B/C/alpha/beta 同型；
输出 C 原地覆写。A：lda×k（N）或 lda×m（T/C）；B：ldb×n（N）或 ldb×k（T/C）；
C：ldc×n。m/n/k/lda/ldb/ldc 与 transa/transb 均为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemm/arch22/
├── gemm_host.cpp            # 已有 Sgemm 入口；新增 aclblasCgemm3m 入口（校验/3M tiling/发射）
├── gemm_kernel.cpp/.h       # 已有实数 Cube 管线；新增 3M 复数 kernel 路径（亦可按仓内规范拆分
│                            #   cgemm3m_kernel.cpp/.h，文件个数对齐仓内同类算子）
└── （tiling 结构共用 gemm 族 tiling 头，扩展 3M 平面参数）
test/gemm/cgemm3m/arch22/
├── cgemm3m_test.cpp         # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── cgemm3m_test.csv         # 官方 1000 精度 + 200 性能
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（L584），无需修改
test/gemm/cgemm3m/CMakeLists.txt  # 注册 arch22 目标 ops_blas_add_gtest_tests
blas/gemm/README.md          # aclblasCgemm3m 章节补充 Atlas A2/A3 支持情况
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：
   handle 判空（`HANDLE_IS_NULLPTR`）→ transa/transb 非法（`INVALID_ENUM`）→
   m/n/k < 0（`INVALID_VALUE`）→ lda/ldb/ldc 低于下界（`INVALID_VALUE`）→
   alpha/beta 指针判空 → **结构合法后 quick return**：m=0 或 n=0 → SUCCESS（A/B/C 可为
   nullptr、不启动 kernel）；k=0 或 α=(0,0) → 只做 `C=β·C`（A/B 可 nullptr）→
   A/B 指针判空（m>0 且 n>0 且 k>0 且 α≠0 时）→ C 指针判空（m>0 且 n>0 时）。
2. **标量读取**：alpha/beta 为 Host 内存 complex64（任务书 §2.4 说明），Host 直读实/虚部；
   α=(0,0)/β=(0,0) 短路标志下传 tiling。
3. **路径选择**：按 m/n/k/片上容量在融合/非融合两 3M 路径间选择（§3.1.1），
   选择规则随 tiling 下发，不按测试编号或随机种子分派（§2.5 确定性条款）。
4. **tiling 组装**：复用 Sgemm arch22 的 Cube tiling 框架，扩展 3M 平面参数
   （三积共用 base tile 布局 + 平面 stride）；K 维尾块、M/N 尾块裁剪参数；
   经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（3M Cube 管线，本任务技术主体）

Ascend C **Matmul API**（对齐仓内 Sgemm arch22：`MatmulImpl` INNER_PRODUCT、ND 格式、
双缓冲 ping-pong）承载 3 次实数乘；AIC+AIV 协同：

- **pack/装载**（AIV，或 Matmul ND 直读）：interleaved complex64 按 Gather 索引
  （byte offset 8m/8m+4）deinterleave 为 Ar/Ai、Br/Bi 平面；同时生成
  (Ar+Ai)、(Br+Bi)（非融合）或 (Bi−Br)、(Br+Bi)（融合）派生平面；
  transa/transb=T 时按转置视图装载（M/N 语义交换），=C 时对应虚部平面符号翻转
  （共轭）。OP_C 与 OP_T 在 pack 符号位区分，结果不同（R02）。
- **三积调度（单 launch）**：P0/P1/P2 共享同一 tiling 与 base tile 循环——
  A tile（Ar、Ar+Ai 两平面）与 B tile（Br、Br+Bi 两平面）各装载一次，
  在同一 K 轮次内连续完成 3 次 Mmad 累加（不同目的累加器），
  摊薄 tile 装载成本并保证三积在同一次 kernel launch 内（任务书红线）。
- **精度平面**：Cube FP16 Mmad + **hi/lo 补偿平面**（x = xh + xl，xh 为 FP16 舍入主部、
  xl 为残差）时每积 3 次 Mmad（xh·yh + xl·yh + xh·yl）保 FLOAT32 分量阈值；
  或按仓内 Sgemm 的 Matmul FP32 路径直接承载（CONFIG_NORM、`DenseFp32` ND 形态）——
  两条精度路线在 spike 阶段以 4096³/小 k 两锚点对拍择优。
- **epilogue（AIV combine）**：按路径公式合成
  `Pr=P0−P1、Pi=P2−P0−P1`（非融合）或 `Pr=P0−P2、Pi=P0+P1`（融合）；
  复数 α/β 交叉缩放（§1.2 第 4 条全套代数）；仅写 C 逻辑 m×n 区域
  （ldc padding 列哨兵不写，R05）；Interleave 写回。
- **K=0/α=0 短路**：跳过三积，仅 epilogue 的 β 缩放（β=(0,0) 不读 C 旧值）。
- **小形状路径**：m 或 n ≤ 16 时 AIV 单发射 kernel（deinterleave + 向量 FMA +
  树归约），避免 Cube 启动开销；判据分布中 small 档占比按
  gpu_baseline.csv 规模分层数据核算。
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶；
  Set/Wait 严格相邻配对；id 7 硬件不可用；等待对不足处补 PipeBarrier。

### 3.2.3 调用与执行流程

```text
aclblasCgemm3m ─校验链→ 标量直读 → quick-return 判定 → 路径/精度平面选择
  → 3M tiling 组装 → cgemm3m_kernel<<<grids, stream>>>
       ├─ AIV: deinterleave/派生平面装载（ping-pong）
       ├─ AIC+Matmul: 单 launch 三积（共享 tile 轮次，hi/lo 补偿）
       └─ AIV: epilogue（路径合成 + α/β 交叉缩放 + 裁剪写回）
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| FP16 Mmad 直接承载 C64 分量精度不足（2⁻¹³ 档） | hi/lo 补偿平面（每积 3 Mmad）或 Matmul FP32 路径，spike 双路线对拍 |
| (Ar+Ai)/(Br+Bi) 平面值域放大（接近 FP16 上限） | 融合路径切换（P1=Ar·(Bi−Br) 差值平面）+ 值域护栏动态选路 |
| 三积与三平面 UB 占用超 190KB 预算 | tile 位宽按 UB 自适应（64/128 二选）；派生平面在 tile 内即时生成不整驻留 |
| 4096³ 大档 Cube 效率必须逼近 Sgemm（见 §3.2.5 反算） | 共享 tile 轮次摊薄装载；三积 Mmad 占比 ≥ 75% 时总效率 ≥ 0.9×Sgemm |
| OP_C/OP_T 符号翻转遗漏（非对称复数用例暴露） | pack 符号位单点维护 + `TC_C_N`/`TC_T_N` 专项对拍用例（golden cblas_cgemm） |

### 3.2.5 性能设计反算（R08）

判据 = H100 gpu_ms × 1.25（任务书 §3.3，kernel-to-kernel）。官方 gpu_baseline.csv 锚点：

| case | transa/transb | m×n×k | H100 gpu_ms | 达标线(ms) |
| --- | --- | --- | --- | --- |
| cgemm3m-base-003 | C/N | 4096³ | 22.039 | 27.55 |
| cgemm3m-perf-014 | N/N | 4096³ | 21.971 | 27.46 |

参照系：同仓 Sgemm 4096³ H100 gpu_ms ≈ 8.82ms（H100 复数 3M≈2.5× 实数耗时）。
arch22 上 3M 三积 ≈ 3× Sgemm Cube 工作量 → 目标是把 3M 管线的有效吞吐做到
Sgemm 的 ≥ 0.9×（共享 tile 轮次 + 三积连续 Mmad 摊薄装载），
4096³ 预算 ≈ 3×(8.82/0.9)/0.8×效率余量 ≈ 26~27ms 量级，与达标线匹配，
spike 阶段以 4096³ 实测校准成本模型后回填；小 k/小形状档以 Cube 启动开销
（≤ 30us）为下限约束。采集口径：msprof op task duration，5 warmup + 10 采样均值。

### 3.2.6 UB 与调度预算

AIC Cube grid × AIV pack/combine 协同（对齐 Sgemm arch22 组织）；每 tile UB 占用：
A 两平面 + B 两平面 + 三积累加器（hi/lo 时平面数翻倍）→ tile 位宽 64/128 自适应，
UB 预算 ≤ 190KB；无 GM 中转 workspace（除 Matmul API 内部）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

- COMPLEX64 专用（Atype/Ctype 组合不支持返回语义不涉及——本接口无 dtype 参数）；
  A/B/C 按 lda/ldb/ldc 列主序带填充访问，不支持任意非连续布局，不使用 AutoContiguous/ViewCopy；
- C 原地覆写，调用方保证 C 可写区与 A/B 可读区不重叠；仅写逻辑 m×n 区域；
- alpha/beta 指向 Host 内存；异步执行依赖 handle 绑定 stream；
- 3M 与直接复数乘舍入过程不同，与 `aclblasCgemm`/`cublasCgemm` 不保证逐位一致（按 §3.2 阈值判定）；
- 确定性：融合/非融合路径与 K 归约顺序由形状与片上容量选择（不按测试编号/种子分派），
  同一输入两次执行不保证逐位一致（任务书 §2.5）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP），
  实部/虚部分别判定；golden = Netlib CBLAS 3.10.0 `cblas_cgemm`
  （float64 累加后收窄 complex64 落盘口径见任务书 §3.2 第 4 条，差异在自测报告说明）；
  Inf/NaN 等效规则；3M 与非 3M golden 不逐位比对。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us`（aclrtEvent 夹连续 10 次下发）仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 cgemm3m_test.csv 全量接入（1000 精度 + 200 性能；transa/transb 9 组合、
   m/n/k 全覆盖含 4096³/4096³ 立方档、lda/ldb/ldc 紧凑与 +1/+8/+64 padding、
   alpha/beta 特殊值 (0,0)/(1,0)/(0,1)/(-1,0)/±1e4、复数输入覆盖全零/纯实/纯虚/一般四类、
   Inf/NaN 特殊值、负向用例含非法枚举/负阶数/低前导维/空指针/α=0 叠 A/B=NaN/β=0 叠 C=NaN、
   k=0 且 β≠1、ldc padding 列哨兵写保护）；
2. **工程**：`test/gemm/cgemm3m/arch22/cgemm3m_test.cpp` CSV 驱动 GTest；
   `test/gemm/cgemm3m/CMakeLists.txt` 注册 arch22 目标
   （`bash build.sh --ops=cgemm3m --soc=ascend910b3`，A3 用对应 --soc 取值）；
3. **精度回归**：1000 条全量 VerifyMixed（实/虚部分别统计 matched_ratio/max_abs_error）；
4. **性能回归**：200 条 PERF 用例逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建命令、gtest filter、msprof 命令、golden 构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 接口与其他产品线共用（声明已存在，不改既有签名，无私有平行 API）；
- arch22 与 Sgemm 同目录共存，互不影响（Sgemm 路径零改动，3M 路径独立 kernel 入口）；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）；
- 与 `blas/gemm3m/`（实数 Sgemm3m）目录族严格隔离（任务书 §2.2 禁令）。

# 5. 参考链接

1. 任务书：aclblasCgemm3m_AtlasA2_A3_task_doc.md（2026-11-25）
2. Netlib cgemm：https://www.netlib.org/blas/cgemm.f
3. cuBLAS cublasCgemm3m：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
