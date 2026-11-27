# aclblasChbmv 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasChbmv_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/sbmv/arch22/`，测试 `test/sbmv/chbmv/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasChbmv算子开发（A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数 Hermitian 带状矩阵-向量乘 `aclblasChbmv`
（对标 cuBLAS `cublasChbmv`，语义参考 Netlib `chbmv`）：

```text
y = alpha · A · x + beta · y        （complex64）
A：n×n Hermitian 带状矩阵，含 k 条次对角线与 k 条超对角线，按列主序带状格式存储于
(k+1)×n 数组（lda×n，lda ≥ k+1），仅 uplo 指定带形三角被引用，另一侧由共轭对称性隐含补全：
  UPPER：A(i,j) 存于 A(1+k+i-j, j)（1 基，i ∈ [max(1,j-k), j]）；左上 k×k 三角不被引用
  LOWER：A(i,j) 存于 A(1+i-j, j)（i ∈ [j, min(n,j+k)]）；右下 k×k 三角不被引用
对角元素虚部假定为 0（实现与 golden 均只取对角实部）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasChbmv_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（chbmv_test.csv、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasChbmv` 声明**已存在**（L202，上游 master；任务书发布时标注"需新增"，窗口期内已合入） | 按现有声明实现 arch22 版本（签名与任务书 §2.3 一致），禁止私有平行接口 |
| `blas/sbmv/arch22/` | 实数同族 `aclblasSsbmv`（host/kernel/tiling 三件套）；上游 master 已合入 chbmv arch22 扩展文件 | 实现落点（与实数同族同目录，任务书 §2.2）；实数族的 host 校验框架与 tiling 传递模式沿用 |
| `blas/sbmv/arch35/`、`blas/chbmv` 族 950 实现 | arch35 已有 chbmv（带状寻址、uplo 分支、对角虚部 0） | 带状下标映射与 Hermitian 语义的实现参照（不得直接复制 SDK 不兼容代码） |
| `blas/symv/arch22` 系（chemv 同族 BLAS-2 先例） | 复数三角矩阵-向量 kernel 先例：列点积带共轭 + 行散射无共轭 + MTE3 原子加 | 复数三角引用的两类贡献组织模式参照 |
| `test/sbmv/` | 既有 ssbmv 测试组织（chbmv 子目录需新建 arch22 分支） | 测试工程新建 `test/sbmv/chbmv/arch22/`（任务书 §5） |

### 功能缺口与本任务差异点

**arch22 Chbmv 为本任务交付主体**。相对实数 ssbmv（arch22）的差异即主要工作量：

1. **带状 Hermitian 复数语义**：带内非对角元素一侧存储、另一侧共轭补全——
   逐列 j 的贡献为「上/下带 scatter 项（无共轭）+ 对侧带 dot 项（带共轭）」
   （Netlib chbmv 精确语义，与 chemv 同构但受带宽 k 截断）；
2. **带状下标映射**：UPPER `A(1+k+i−j, j)` / LOWER `A(1+i−j, j)` 的非线性
   行索引——多核列段切分需「对角线索引起偏移表」（等差闭式）规避逐列乘除；
3. **k 的退化与守卫**：k > n−1 按有效带宽 min(k, n−1) 参与计算；k = n−1 满带
   退化为全矩阵（≈ chemv）；k = 0 退化为纯对角（Ddot 形态）；
   `lda ≥ k+1` 校验在 int64 域进行（k 取 INT32_MAX 时 int32 回绕会使校验穿透，
   任务书 §2.4 明确）；
4. **alpha=(0,0) 不保证跳过 A/x 读取**（任务书 §2.1 第 4 条）——含 Inf/NaN 的 x
   用例不得与 alpha=(0,0) 组合，实现不以此为短路依据（与 chemv 族差异点）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/sbmv/arch22/` 实现 `aclblasChbmv`：uplo ∈ {UPPER, LOWER} × n 覆盖
0/1/小质数/2 的幂及其 ±1/非对齐值直至大规模 × k 覆盖 0/1/小带宽/k=n−1（满带退化）/
典型大带宽 × lda（紧凑 k+1 与多档 padding）× incx/incy（±1/±2/±3，零步长负向）×
alpha/beta（均匀 [-5,5] 50% + 正态 50% + 特殊值 (0,0)/(1,0)/纯虚数/大值）×
Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准 FLOAT32 档（实部/虚部分别判定：rtol = atol = 2⁻¹³，
ratio ≥ 0.99，max_abs ≤ 1e-2 或 32·ULP 按最大误差点 golden 动态计算）；
golden 为 `cblas_chbmv`（Netlib CBLAS 复数实现，CblasColMajor；float64 真值计算
收窄 FLOAT32 落盘，issue #434 高精度 golden 口径）。性能 msprof op 模式
task duration，5 次 warmup + 有效采样 ≥10 次均值，逐 case ≤ H100 gpu_ms × **1.25**
（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `y = α·A·x + β·y`（带状 Hermitian，uplo 带形三角引用） | §2.1 | 带窗组批 kernel（§3.2.2） |
| R02 | uplo 非法 → `INVALID_VALUE`（对齐 cuBLAS，本批次口径）；校验顺序 handle→n≥0→**n=0 quick return**→uplo/k/lda/inc/标量/指针 | §2.4/§2.5 | Host 校验链（§3.2.1） |
| R03 | k<0 → `INVALID_VALUE`；k>n−1 按有效带宽 min(k,n−1)；lda ≥ k+1 int64 域校验 | §2.4 | Host 带宽守卫（§3.2.1） |
| R04 | n=0 quick return（判空后）；β=(0,0) 不读 y；α=(0,0) 不作短路依据 | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R05 | 带状下标映射（UPPER/LOWER 行索引闭式）+ 未引用三角不读 | §2.1/§2.5 | 对角线索引起偏移表（§3.2.2） |
| R06 | 对角虚部假定为 0（只取实部） | §2.1 | 对角实部标量捕获（§3.2.2） |
| R07 | incx/incy = 0/INT32_MIN/跨度溢出 → `INVALID_VALUE`（int64 守卫，#780/#866） | §2.4 | Host 步长守卫（§3.2.1） |
| R08 | 精度 FLOAT32 档 + Inf/NaN 等效 + golden fp64 域 | §3.2 | 测试工程（§4.2） |
| R09 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

0 基列主序带状存储。UPPER（存 A(i,j), i ∈ [max(0,j−k), j]，物理行 1+k+i−j−1 = k+i−j）
逐列 j（Netlib chbmv 精确语义）：

```text
temp1 = alpha · x(j)
i < j（带内）：y(i) += temp1 · A(i,j)              # 上带 scatter 项，无共轭
              temp2 += conj(A(i,j)) · x(i)          # 对侧（下带）dot 项，带共轭
y(j) += temp1 · real(A(j,j)) + alpha · temp2
```

LOWER（存 A(i,j), i ∈ [j, min(n,j+k)]，物理行 i−j）对称地按 i > j 展开。
等价的平面化形式（本实现，带窗组织）：

```text
dot(j)  = Σ_{i∈带(j) 除对角} conj(A(i,j))·x(i)     # 列点积（带共轭，带宽截断）
t1(i)   = Σ_{j∈带(i) 除对角} A(i,j)·x(j)            # 行散射（无共轭，带宽截断）
y(j)    = alpha·(x(j)·real(A(j,j)) + dot(j)) + alpha·t1(j) + beta·y(j)
```

带物理 (k+1)×n：UPPER 列 j 的带段物理偏移 `[j×lda + (k−min(k,j)), j×lda + k]`；
LOWER 列 j 带段 `[j×lda, j×lda + min(k, n−1−j)]`。有效带宽 kb = min(k, n−1)。

### 3.1.2 支持数据类型

COMPLEX64（interleaved float32 re/im，`aclblasComplex`），alpha/beta 同型；
输出原地覆写 y。A 为 lda×n 带状数组（lda ≥ k+1）；x/y 物理元素数 ≥ 1+(n−1)×|inc|。
n、k 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/sbmv/arch22/
├── chbmv_host.cpp           # aclblasChbmv 入口：校验链/标量读取/tiling/发射
├── chbmv_kernel.cpp         # Ascend C AIV kernel（带窗组批引擎）
├── chbmv_kernel.h
└── chbmv_tiling_data.h      # tiling 结构 + 带窗参数 + 索引起偏移表闭式
test/sbmv/chbmv/arch22/
├── chbmv_test.cpp           # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── chbmv_test.csv           # 官方精度 + 性能用例（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（上游 master L202），无需修改
test/sbmv/chbmv/CMakeLists.txt  # 注册 arch22 目标 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：
   handle 判空（`HANDLE_IS_NULLPTR`）→ n<0（`INVALID_VALUE`）→ **n=0 直接返回
   SUCCESS**（不执行计算；quick return 不得先于 handle 判空，#765/#861 口径）→
   uplo 非法（`INVALID_VALUE`，本批次对齐 cuBLAS 口径）→ k<0（`INVALID_VALUE`）→
   lda < k+1（`INVALID_VALUE`，**int64 域计算比较**，k=INT32_MAX 防回绕）→
   incx/incy = 0 或 INT32_MIN（`INVALID_VALUE`）→ (n−1)×|inc| int64 溢出守卫
   （`INVALID_VALUE`，#780/#866）→ alpha/beta 指针判空（n>0 时，`INVALID_VALUE`）→
   A/x/y 指针判空（n>0 时，`INVALID_VALUE`）。
2. **有效带宽归一**：kb = min(k, n−1)（k > n−1 时按 kb 参与计算，任务书 §2.4）。
3. **标量读取**：alpha/beta 为 Host 内存 complex64 标量，直读实/虚部（值不校验；
   β=(0,0) 标志下传——不读 y 旧值；**α=(0,0) 不作短路依据**，照常执行，
   任务书 §2.1 第 4 条）。
4. **tiling 组装**：uplo × kb × n/incx/incy/alpha/beta + 带窗参数（§3.2.2）+
   coreNum（ChooseCoreNum：strided → K=1 单核；列段数 ≤ 32 → K=段数；
   否则按带窗 chunk 均衡 K ≤ 40）；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（带窗组批引擎，本任务技术主体）

AIV 向量引擎（Level-2 带状访存受限，无 Cube 收益）：

- **对角线索引起偏移表**：Host 侧等差闭式预生成——每列带段物理起偏移
  `seg(j) = j×lda + (UPPER: k − min(k,j)；LOWER: 0)`、带段长
  `len(j) = min(kb, j) + 1`（UPPER）/ `min(kb, n−1−j) + 1`（LOWER），
  kernel 查表装载，规避带状非线性行索引逐列乘除；
- **带窗组批**：B=8 列/组 × 带窗（对角线族聚集：同一条对角线在带数组中
  物理连续——UPPER 主对角在第 k 行、第 d 条超对角在第 k−d 行，行内跨列
  按 lda 步进）→ 带窗按「对角线族 × 列组」组织平面装载（inter ping-pong 64KB）；
- **deinterleave**：带段 complex64 经 Gather 索引（byte offset 8m/8m+4）一次拆
  re/im 平面，后续乘累加平坦向量 pass；
- **列点积（带共轭）**：`d = conj(A)·x带窗`（4 Mul + 2 Add/Sub）→ ReduceSum
  按列求和（紧凑 stage 槽）→ per-group stage 槽累加；
- **行散射（无共轭）**：x 列广播 × A 平面 → 8 槽树状归约 → per-window win
  累加器；
- **对角**：每列对角元素（UPPER 带段末 / LOWER 带段首）一次标量捕获
  `real(A(j,j))` 后置零平面，fold 阶段补 `x(j)·d(j)`；对角虚部不读（带装载
  裁剪）；
- **回写**：每窗一次 α 折叠 + interleave + **MTE3 原子加** flush 到 y；每组一次
  `alpha·(x·d + dot)` 折叠原子加；β 重写走 MTE3 向量路径 + 全核屏障
  （标量 GM 写对原子加不可见）；
- **退化路径**：k=0 → 纯对角 Ddot 形态（A 对角行单遍流读 + x 点乘）；
  kb=n−1 满带 → 等价 chemv 形态（全三角贡献）；
- **strided/单核路径**（K=1 门控）：标量 gather/scatter 步长访问，带读取仍向量化；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、
  Set/Wait 相邻配对、id 7 不可用、等待对不足补 PipeBarrier；
- **UB 预算**：~165KB（inter 64K + planes 32K + xwin 32K + win/stage/idx/表 44K）。

### 3.2.3 调用与执行流程

```text
aclblasChbmv ─校验链（n=0 在 uplo 前）→ kb 归一 → 标量直读
  → ChooseCoreNum → 偏移表 tiling 组装 → chbmv_kernel<<<K, stream>>>
       └─ AIV: Init(偏移表/清零) → [β 重写(MTE3)+SyncAll] → 带窗循环
            { x 带窗装载 → 组循环{对角线族 ping-pong 载入 → 列点积/行散射}
            → win flush 原子加 } → FoldGroups → 完成
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 带状非线性下标多核寻址 | 对角线索引起偏移表（等差闭式）+ kernel 查表 |
| 满带 kb=n−1 大档带宽天花板 | 带数组 (k+1)×n 单遍流读（≈ n²×8B/2 实流量）+ 每元素仅读一次 |
| 复数原子加多核可见性 | β 重写 MTE3 + 全核屏障；win/stage 双累加器解耦 |
| k=INT32_MAX 时 lda≥k+1 的 int32 回绕 | int64 域校验（任务书 §2.4 明确） |
| UP/LOW 带段端点差异（UPPER 左上 k×k、LOWER 右下 k×k 不引用） | 偏移表闭式裁剪单点维护 + 专项用例 |

### 3.2.5 性能设计反算（R09）

判据 = H100 gpu_ms × 1.25（msprof op task duration，kernel-to-kernel）。
官方 gpu_baseline.csv 锚点：

| case | uplo | n | k | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- | --- |
| chbmv-perf-081 | LOWER | 2048 | 2047 | 0.4430 | 553.8 |

满带大档（n=2048, k=2047）：带数组 (k+1)×n×8B ≈ 33.5MB 流读（Hermitian 补全
使逻辑流量翻倍但物理读单遍）→ HBM 1.44TB/s 底线 ~23us + x/y 与组批开销 ≤ 8us
→ 预算 ~31us，对 553.8us 达标线余量 ~17×（H100 cuBLAS chbmv 满带档 0.44ms
远超其带宽底线——GPU 带状实现非带宽最优）；小带宽档（k≪n）流量 ~(k+1)×n×8B
按实际带窗核算；小 n 档以固定开销 ≤ 8us 为下限。采集口径：msprof op 自带
5 warmup + ≥10 采样均值；spike 以满带大档 + 小带宽档双锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立带窗 slab（chunk 均衡）；UB ~165KB 布局见 §3.2.2；
无 workspace/GM 中转（除原子回写）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

n ≥ 0；k ≥ 0（有效带宽 min(k, n−1)）；lda ≥ k+1（int64 域）；incx/incy ≠ 0 且
非 INT32_MIN 且跨度不溢出；COMPLEX64 专用；仅 uplo 带形三角被引用、对角虚部
假定为 0；y 原地覆写；α=(0,0) 不作短路依据（含 Inf/NaN 的 x 用例不得与
α=(0,0) 组合，任务书 §2.1）；异步执行依赖 handle 绑定 stream；
n=0 quick return 在结构校验后（handle 判空最优先）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32·ULP
  ——按最大误差点 golden 动态计算，issue #434 修订口径）；实部/虚部分别判定；
  Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf 等效；非有限值单独统计）；
  golden = `cblas_chbmv`（Netlib CBLAS 复数实现，CblasColMajor，float64 真值
  计算收窄 FLOAT32 落盘；按 Hermitian 约束解释 A：对角只取实部、未存储带
  共轭补全）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 chbmv_test.csv 全量接入（精度全量 + 性能子集；uplo 全组合、
   n 全档、k 扫描 0/1/小带宽/满带 k=n−1/典型大带宽、lda 紧凑 k+1 与 padding、
   incx/incy ±1/±2/±3、alpha/beta 均匀+正态+特殊值、A 按 Hermitian 带状约束生成
   （对角虚部 0、带内共轭对称、带外 0）、Inf/NaN 特殊值、负向用例（nullptr handle/
   非法 uplo（`INVALID_VALUE`）/负 n/**负 k**/**k=INT32_MAX 叠低 lda 回绕**/
   lda<k+1/零步长/INT32_MIN/跨度溢出/空标量/空指针/n=0 期望 SUCCESS））+
   **GAUSS_NORM_5_2 正态分布补充**（μ∈[-5,5]、σ∈[0.1,2]，任务书 §3.5-正态补充要求，
   自测报告说明补齐方式）；
2. **工程**：`test/sbmv/chbmv/arch22/chbmv_test.cpp` CSV 驱动 GTest；
   `test/sbmv/chbmv/CMakeLists.txt` 注册 arch22 目标
   （编译入口对齐仓内 `--ops=chbmv --soc=ascend910b3`，A3 对应 --soc）；
   负向用例按单元素 dummy/NULLPTR 占位（不按步长规模预分配）；
3. **精度回归**：全量精度条数（实/虚部分别统计 matched_ratio/max_abs_error）；
4. **性能回归**：性能子集逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、golden 构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 声明已存在（上游 master L202）与其他产品线共用，禁止私有平行 API；
- arch22 与 ssbmv 同目录共存互不影响；arch35 参考实现只读引用；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasChbmv_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib chbmv：https://www.netlib.org/blas/chbmv.f
3. cuBLAS cublasChbmv：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
