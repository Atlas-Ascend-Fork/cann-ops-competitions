# aclblasChemv 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasChemv_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/symv/arch22/`，测试 `test/symv/chemv/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasChemv算子开发(A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数 Hermitian 矩阵-向量乘 `aclblasChemv`
（对标 cuBLAS `cublasChemv`，语义参考 Netlib `chemv`）：

```text
y = alpha · A · x + beta · y
A：n×n Hermitian 复数矩阵（complex64，列主序，lda×n，仅 uplo 三角被引用）
x/y：n 元素复数向量（支持正/负步长 incx/incy ≠ 0）
对角元素虚部假定为 0、不参与计算
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasChemv_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（chemv_test.csv 1200 条、gpu_baseline.csv 200 条、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasChemv` 声明缺失 | 需新增声明（§2.3 签名），供各产品线共用，禁止私有平行 API |
| `blas/symv/arch22/` | 仅实数参考算子 aclblasSsymv（host/kernel/tiling 三件套） | 同族目录即本任务实现落点；host 校验框架与 tiling 传递模式直接沿用 |
| `blas/symv/arch35/` | ssymv 950 版（含 ComplexScalarSource 双读取口径，issue #869） | alpha/beta Host/Device 指针属性探测的实现参照 |
| `blas/dot/arch22/`、`blas/herk/arch22` 系 | 复数 kernel 先例：Gather 索引 deinterleave、跨核同步 | 复数向量原语与多核协同模式参照 |
| `test/symv/chemv/` | 目录不存在；同族 CMakeLists 仅注册 arch35 GTest | 需新建测试工程并补 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)` 的 arch22 注册（任务书 §2.2） |

### 功能缺口与本任务差异点

**arch22 chemv 整体缺失**。相对实数 ssymv（arch22）的差异即主要工作量：

1. **复数语义**：三角侧 scatter 项无共轭、点积项带共轭（Netlib chemv 精确语义），
   对角仅取实部；interleaved complex64 内存需 deinterleave 为 re/im 平面；
2. **alpha/beta Host/Device 双读取**（issue #869）：运行时按指针属性探测，Device 指针
   走同步小拷贝读值，对齐 csymv arch35 口径；
3. **步长大跨度守卫**（issue #807）：(n−1)×|inc| 以 host 侧 int64 计算并拒绝溢出；
   INT32_MIN 步长直接拒绝（#780）；
4. **负步长**：逻辑 i ↔ 物理 (n−1−i)×|inc|（Netlib KX 口径）；
5. **多核并行 + 复数原子累加**：三角矩阵-向量乘访存受限（无 Cube 收益），需按
   列均衡切分 + 向量引擎平面化乘累加 + MTE3 原子加回写 y。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/symv/arch22/` 实现 `aclblasChemv`：uplo ∈ {UPPER, LOWER} × n 覆盖 0/1/质数/2 的幂
及其 ±1/非对齐值直至 4096 大规模 × lda（紧凑与多档 padding）× incx/incy ∈ {±1, ±2, ±3}
× alpha/beta（均匀 [-5,5] 与正态 N(μ∈[-5,5], σ∈[0.1,2]) 各半，含 (0,0)、(1,0)、纯虚数、
大值）× Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准 FLOAT32 档（实部/虚部分别判定：rtol = atol = 2⁻¹³，
matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP，Inf/NaN 等效规则）；
golden 为 Netlib CBLAS 3.10.0 `cblas_chemv`（CblasColMajor）单标杆。
性能 msprof op 模式 task duration，5 次 warmup + ≥10 次有效采样均值，
逐 case ≤ H100 gpu_ms × **1.25**（即 ≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `y = alpha·A·x + beta·y`（Hermitian 三角引用） | §2.1 | 窗口主序组批 kernel（§3.2.2） |
| R02 | uplo 非法 → `INVALID_ENUM`；校验顺序 handle→n→uplo/lda/inc/标量→A/x/y | §2.5 | Host 校验链（§3.2.1） |
| R03 | n=0 no-op；alpha=(0,0)&&beta=(1,0) quick return；alpha=(0,0) 不读 A/x；beta=(0,0) 不读 y | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R04 | alpha/beta Host/Device 双读取 | §2.4/#869 | ComplexScalarSource 指针属性探测 + Device 小拷贝（§3.2.1） |
| R05 | inc ∈ {0, INT32_MIN} 拒绝；(n−1)×\|inc\| int64 溢出拒绝 | §2.4/#807/#780 | Host 步长守卫（§3.2.1） |
| R06 | 负步长语义（逻辑↔物理反转） | §2.1 | X/YPhysPos 映射（§3.2.2） |
| R07 | 对角虚部不读；未引用三角不读 | §2.5 | 对角标量捕获 + 装载裁剪（§3.2.2） |
| R08 | 精度 FLOAT32 标准 + Inf/NaN 等效 | §3.2 | 测试工程 VerifyMixed（§4.2） |
| R09 | 性能逐 case ≤ 1.25×H100（msprof op task duration） | §3.3 | 性能反算与预算（§3.2.5） |
| R10 | arch22 GTest 注册 + 正态分布用例补齐（GAUSS_NORM_5_2） | §2.2/§3.5 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

0 基列主序，UPPER（存 A(i,j), i ≤ j）逐列 j（Netlib chemv 精确语义）：

```text
temp1 = alpha · x(j)
i < j:  y(i) += temp1 · A(i,j)          # scatter 项，无共轭
        temp2 += conj(A(i,j)) · x(i)    # dot 项，带共轭
y(j) += temp1 · real(A(j,j)) + alpha · temp2
```

LOWER（存 A(i,j), i ≥ j）对称地按 i > j 展开。等价的平面化形式（本实现）：

```text
dot(j)  = Σ_{i<j} conj(A(i,j))·x(i)                 # 列点积（带共轭）
t1(i)   = Σ_{j>i, 同组} A(i,j)·x(j)                  # 行散射（无共轭）
y(j)    = alpha·(x(j)·real(A(j,j)) + dot(j)) + alpha·Σ_windows t1(j) + beta·y(j)
```

### 3.1.2 支持数据类型

COMPLEX64（interleaved float32 re/im，`aclblasComplex`），alpha/beta 同型；
输出原地覆写 y。

### 3.1.3 支持形状

n ≥ 0（运行时入参，非动态 shape）；A 为 lda×n（lda ≥ max(1,n)）；
x/y 物理元素数 ≥ 1+(n−1)×|inc|。性能覆盖 n ≤ 4096。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/symv/arch22/
├── chemv_host.cpp         # aclblasChemv 入口：校验链/标量读取/tiling/发射
├── chemv_kernel.cpp       # Ascend C AIV kernel（窗口主序组批引擎）
├── chemv_kernel.h
└── chemv_tiling_data.h    # tiling 结构 + 常量（B/R/E/窗口）+ CumChunks 闭式
test/symv/chemv/arch22/
├── chemv_test.cpp         # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── chemv_test.csv         # 官方 1200 条 + 正态分布补充
├── gpu_baseline.csv       # 官方 H100 基线 200 条
└── README.md              # 复现步骤
include/cann_ops_blas.h    # 新增 aclblasChemv 声明（共用 API）
test/symv/chemv/CMakeLists.txt  # arch22 分支注册 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：
   handle 判空（`HANDLE_IS_NULLPTR`）→ n<0（`INVALID_VALUE`）→ n=0 SUCCESS →
   uplo 非法（`INVALID_ENUM`）→ lda<max(1,n) / incx=0 / incy=0 / INT32_MIN /
   (n−1)×|inc| int64 溢出（`INVALID_VALUE`）→ alpha/beta 指针判空 →
   读标量后 quick-return 判定（alpha=(0,0)&&beta=(1,0) → SUCCESS）→
   A/x（alpha≠(0,0) 且 n>0）/ y（n>0 且非 quick-return）判空。
2. **标量双读取**（#869）：探测 alpha/beta 指针属性——Device 指针经
   `aclrtMemcpy` 同步 8B 小拷贝读回；Host 指针直读。对齐 csymv arch35
   `ComplexScalarSource`。
3. **快速分支**：alpha=(0,0) 时跳过 A/x 装载，仅 beta 重写 y；
   beta=(1,0) 且 alpha=(0,0) 直接 SUCCESS。
4. **tiling 组装**：n/lda/uplo/incx/incy/alpha/beta 及其 0/1 标志 +
   coreNum（ChooseCoreNum：strided 或 beta∉{0,1} 复数重写 → K=1 单核；
   列组数 ≤ 32 → K=组数；否则按 chunk 均衡 K）。经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（窗口主序组批引擎，本任务技术主体）

AIV 向量引擎（无 Cube 路径——Level-2 BLAS 访存受限，Cube 无收益）。
按「行窗口 × 列组」两级组织：

- **组批参数**：B=8 列/组、R=512 行/窗、E=B×R=4096 元素平面；
  每组 chunk 的 A 三角按列连续读入 UB ping-pong 暂存（inter×2=64KB）。
- **deinterleave**：interleaved complex64 经 Gather 索引（byte offset 8m / 8m+4）
  一次拆分为 re/im 平面（planes 32KB），后续乘累加全为平坦 E 元素向量 pass，
  调用次数与列数解耦（O(1)/组，替代逐列 O(B)）。
- **列点积**：`d = conj(A)·x窗`（4 Mul + 2 Add/Sub 平面 pass）→
  ReduceSum 按列求和（经典实现会在 dst 槽间隙写垃圾——仅压缩槽头 +
  紧凑 stage 布局规避）→ 累加进 per-group stage 槽（跨窗口持久）。
- **行散射**：x 列广播（mini 暂存 + Gather 广播）× A 平面 →
  8 槽树状归约（3 层 Add）→ 累加进 per-window win 累加器。
- **对角**：每列在其边沿窗口内一次标量捕获 `real(A(j,j))` 后置零平面，
  fold 阶段补 `x(j)·d(j)` 项；LOWER 对齐装载的 straggler 行标量清零。
- **回写**：每窗一次 α 折叠 + interleave + **MTE3 原子加** flush 到 y
  （win 累加器）；每组一次 `alpha·(x·d + dot)` 折叠原子加到 y[c0..c0+8)。
  多核 β 重写必须走 MTE3 向量路径（标量 GM 写对原子加不可见）。
- **chunk 均衡 slab**：CumChunks(m) 闭式（UPPER/LOWER 各自）+ 二分，
  各核分得近似等量 (窗口,组) chunk，等时化墙钟。
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V 按 ping-pong
  缓冲奇偶；Set/Wait 严格相邻配对；id 7 硬件不可用；MTE2_V/V_MTE3
  等待对不足处补 PipeBarrier 硬排空。
- **单核路径**（strided/强制 K=1）：每 chunk 点积直接标量 RMW 折进 y，
  β 基底先铺；x/y 步长访问用标量 gather/scatter，A 读取仍向量化。
- **UB 预算**：~165KB / 174KB 实测上限
  （inter 64K + planes 32K + xrow 32K + xwin/win/stage/idx 44K）。

### 3.2.3 调用与执行流程

```text
aclblasChemv ─校验链→ 标量读取(Host/Device) → quick-return 判定
  → ChooseCoreNum → tiling 组装 → chemv_kernel<<<K, stream>>>
       └─ AIV: Init(idx 构建/清零) → [β 重写(MTE3)+SyncAll] → 窗口循环
            { x 窗装载驻留 → 组循环{ping-pong 载入→组批计算} → win flush 原子加 }
          → FoldGroups(α·(x·d+dot) 逐组原子加) → 完成
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| n=4096 带宽天花板（三角 67MB 流读，HBM 1.44TB/s@40 核 → 底线 ~46.5us vs 判据 50.9us） | A 每元素仅读一次的列连续流式装载；原子回写/同步开销压到 ≤3us |
| 小 n 判据 ~3.6-5.6us 与固定开销矛盾 | 判据分布 153/200 条 ≥15us；小 n 走轻量路径并压 init 链 |
| 复数原子加多核可见性 | β 重写走 MTE3 + 全核屏障；win/stage 双累加器解耦跨窗/跨组序 |
| 经典 ReduceSum dst 槽间隙垃圾 | 紧凑 stage + 槽头 Gather 压缩 |
| strided/负步长性能 | 门控 K=1 保精度（性能用例全为 inc=1/lda 紧凑，无代价） |

### 3.2.5 性能设计反算（R09）

判据 = H100 gpu_ms × 1.25（任务书 §3.3）。典型锚点（官方 gpu_baseline.csv）：

| case | uplo | n | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- |
| chemv-base-000 | UPPER | 512 | 0.0146 | 18.3 |
| chemv-base-001 | UPPER | 1024 | 0.0211 | 26.4 |
| chemv-base-002 | LOWER | 2048 | ~0.028 | ~35 |
| chemv-base-003 | LOWER | 4096 | ~0.055 | ~69 |

多核路径实测核耗时与 n 弱相关（固定开销 + 带宽两段）；目标固定开销 ≤8us、
n=4096 流读 ≤46.5us，全档留 ≥20% 余量。采集口径：msprof op task duration，
5 warmup + 10 采样均值。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立 slab（列区间 + chunk 均衡）；UB 165KB 布局见 §3.2.2；
无 workspace/GM 中转（除原子回写）。

## 3.3 支持硬件

Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)）与 Atlas A3 系列产品；
本设计在 Atlas A3 (910B4) 开发验证（自验证覆盖一种款型即可，任务书 §3.1）。

## 3.4 算子约束限制

n ≥ 0；lda ≥ max(1,n)；incx/incy ≠ 0 且非 INT32_MIN 且 (n−1)×|inc| 不溢出；
COMPLEX64 专用；y 原地覆写；异步执行依赖 handle 绑定 stream。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP），
  实部/虚部分别判定；Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf 等效）；
  golden = Netlib CBLAS 3.10.0 `cblas_chemv` 单标杆。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100。

## 4.2 测试方案

1. **用例**：官方 chemv_test.csv 1200 条全量替换接入（L0 4 / SQ 46 / AB 16 / LD 18 /
   INC 72 / FL 18 / CV 16 / ED 30 / EX 780 / PF 200）+ 正态分布补充用例
   （GAUSS_NORM_5_2，μ∈[-5,5]、σ∈[0.1,2]，扩展 CSV 生成器补齐并在自测报告说明）；
2. **工程**：`test/symv/chemv/arch22/chemv_test.cpp` CSV 驱动 GTest；
   CMakeLists arch22 分支注册 `ops_blas_add_gtest_tests`；负向用例按任务书 §3.5
   以单元素 dummy/NULLPTR 占位（不得按步长规模预分配）；
3. **精度回归**：全量 1000 条精度 + 200 条性能内嵌正确性校验；
4. **性能回归**：200 条 PERF 用例逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建命令、gtest filter、msprof 命令、golden 构建说明）。

## 4.3 兼容性分析

- 接口与其他产品线共用（仅新增声明，不改既有签名）；
- arch22 与 ssymv 同目录共存，互不影响；测试目录按 chemv 专用；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasChemv_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib chemv：https://www.netlib.org/blas/chemv.f
3. cuBLAS cublasChemv：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hemv
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
