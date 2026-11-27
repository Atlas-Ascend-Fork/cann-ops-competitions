# aclblasSgemv 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSgemv_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemv/arch22/`，测试 `test/gemv/sgemv/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasSgemv算子开发（A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现单精度实数矩阵-向量乘 `aclblasSgemv`
（对标 cuBLAS `cublasSgemv`，语义参考 Netlib `sgemv`）：

```text
y = alpha · op(A) · x + beta · y        （FLOAT32，列主序）
op(A) = A（N）/ Aᵀ（T）/ Aᴴ（C，实数等价 Aᵀ）
N：op(A) m×n，x 长 n，y 长 m；T/C：op(A) n×m，x 长 m，y 长 n
β=0 时不读 y 旧值；校验通过后 m=0 或 n=0 quick return；实现侧 α=0 短路 y=βy（不读 A/x）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSgemv_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（sgemv_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasSgemv` 声明**已存在**（L552，与 950PR 共用） | 无需新增声明，禁止私有平行接口（任务书 §2.3） |
| `blas/gemv/arch22/` | 复数 `aclblasCgemv` 已有（cgemv_host + do/no_trans 双 kernel + utils）；**实数 Sgemv arch22 为本任务缺口**；上游 master 已合入同任务实现（sgemv_host/kernel/tiling 三件套） | 实现落点与同族工程模式（host 入口 + tiling + device kernel，任务书 §2.2）；复数兄弟算子为 fork 底座：N/T 双路径划分、列核/点积组织、步长与标量处理直接参照 |
| `blas/gemv/arch35/` | sgemv 950 版（slab/ordered/scale 多 kernel） | 参数校验（ValidateSgemvParams L31-107）、quick return 与标量口径的实现参照 |
| `test/gemv/sgemv/` | 通用 `sgemv_golden.h`/`sgemv_param.h` 存在；CMakeLists 仅注册 arch35 | 新建 arch22 测试工程并补 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)` arch22 注册（任务书 §2.2——A2/A3 构建否则退回非 GTest runner） |

### 功能缺口与本任务差异点

**arch22 实数 Sgemv 为本任务交付主体**。相对兄弟算子 cgemv（arch22）的差异即主要工作量：

1. **实数路径**：无 deinterleave/共轭语义，直接 FP32 平面乘累加——吞吐与带宽均优于复数路径，
   设计重点转为**带宽压榨**（A 矩阵占绝对流量）与双路径均衡；
2. **α=0 短路**（任务书 §2.1）：`y=βy` 不读 A/x（A/x 含 Inf/NaN 时同样正常——短路在数值
   读取之前）；β=1 且 α=0 时 y 不变；
3. **跨度溢出守卫**：`(len−1)×|inc| > INT32_MAX` 返回 `INVALID_VALUE`（host 侧 int64 计算，
   issue #866/#860 口径）——**arch35 参考实现未提供，arch22 必须补齐**（任务书 §2.5 明确）；
4. **GTest 注册**：arch22 分支注册补齐（A2/A3 当前退回整数毫秒 runner）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemv/arch22/` 实现 `aclblasSgemv`：trans ∈ {N, T, C}（实数 C≡T）×
m/n 覆盖 {0,1,3,64,512,2048} 与非方阵组合（m=2048×n=512 等，含 4097×4097 大档）×
lda ∈ {max(1,m), m+8}（列间空洞）× incx/incy ∈ {1,2,7,−1,−3}（正负步长）×
alpha ∈ {0.0,1.0,2.5,−1.5,0.1,10.0} × beta ∈ {0.0,1.0,0.6,−0.5} ×
Inf/NaN 特殊值与负向用例（含跨度溢出 n=4×incx=INT_MAX、n=2³⁰×inc=8）。

精度按生态开源精度标准 FLOAT32 档（rtol = atol = 2⁻¹³，matched_ratio ≥ 0.99，
max_abs ≤ 1e-2 或 32·ULP，32×ULP 按最大误差点 golden 动态计算）；golden 为 Netlib
BLAS `sgemv` CPU 参考实现（float64 真值计算收窄 FLOAT32 落盘，任务书 §3.2）。
性能 msprof op 模式 task duration，自带 5 次 warmup + 有效采样 ≥10 次均值，
逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `y = α·op(A)·x + β·y`（N/T/C 三路径） | §2.1 | N 列核 / T 行 dot 双路径（§3.2.2） |
| R02 | trans 非法 → `INVALID_VALUE`（对齐 cuBLAS，本批次口径）；校验链对齐 arch35 ValidateSgemvParams | §2.4/§2.5 | Host 校验链（§3.2.1） |
| R03 | m=0/n=0 quick return（校验后）；α=0 短路 y=βy；β=0 不读 y | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R04 | incx/incy = 0 或 INT32_MIN → `INVALID_VALUE`；(len−1)×\|inc\| 溢出守卫（arch35 缺失，arch22 补齐） | §2.4/§2.5 | Host 步长守卫（§3.2.1） |
| R05 | 负步长反向遍历（物理 span 尾端起算） | §2.4 | X/YPhysPos 映射（§3.2.2） |
| R06 | 精度 FLOAT32 档 + Inf/NaN 等效规则 | §3.2 | 测试工程 VerifyMixed（§4.2） |
| R07 | 性能逐 case ≤ 1.25×H100（msprof op task duration） | §3.3 | 性能反算与预算（§3.2.5） |
| R08 | arch22 GTest 注册 + 正态分布用例补充（GAUSS_NORM_5_2） | §2.2/§3.5 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

0 基列主序（A(i,j) 物理偏移 i + j×lda）：

```text
N：y(r) = α·Σ_j A(r,j)·x(j) + β·y(r)          r ∈ [0,m)
T/C：y(c) = α·Σ_r A(r,c)·x(r) + β·y(c)        c ∈ [0,n)
x/y 逻辑索引 i ↔ 物理位置 i×|inc|（负步长自 span 尾端 (len−1)×|inc| 反向）
```

N 路径每输出元素为一行点积（行段读）；T 路径每输出元素为一列点积（列连续读，
对列主序带宽友好）。实数 OP_C 与 OP_T 等价。

### 3.1.2 支持数据类型

A/x/y/alpha/beta = FLOAT32；y 原地覆写。A：lda×n（lda ≥ max(1,m)）；
x/y 物理元素数 ≥ 1+(len−1)×|inc|。m/n/lda/incx/incy 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemv/arch22/
├── sgemv_host.cpp          # aclblasSgemv 入口：校验链/标量读取/tiling/发射
├── sgemv_kernel.cpp        # Ascend C AIV kernel（N 列核 / T 行 dot 双路径）
├── sgemv_kernel.h
└── sgemv_tiling_data.h     # tiling 结构 + slab/块参数
test/gemv/sgemv/arch22/
├── sgemv_test.cpp          # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── sgemv_test.csv          # 官方 1000 精度 + 200 性能（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv        # 官方 H100 基线 200 条
└── README.md               # 复现步骤
include/cann_ops_blas.h     # 声明已存在（L552），无需修改
test/gemv/sgemv/CMakeLists.txt  # arch22 分支注册 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（对齐 arch35 `ValidateSgemvParams`，按任务书 §2.4/§2.5 补齐守卫）：
   handle 判空（`HANDLE_IS_NULLPTR`）→ trans 非法 / m<0 / n<0 / lda<max(1,m)
   （`INVALID_VALUE`）→ incx/incy = 0 或 INT32_MIN（`INVALID_VALUE`）→
   (len−1)×|inc| 以 int64 计算超 INT32_MAX（`INVALID_VALUE`，**arch35 缺失项，
   本任务补齐**）→ alpha/beta 指针判空（`INVALID_VALUE`）→ 结构合法后
   m=0 或 n=0 quick return SUCCESS（不访问 A/x/y、不启动 kernel）→
   A（m>0 且 n>0）/ x、y（逻辑长度非零）指针判空（`INVALID_VALUE`）。
2. **标量读取**：alpha/beta 为 Host 内存 float 指针，Host 侧同步解引用后随 tiling
   下发（值不校验；α=0/β=0/β=1 短路标志）。
3. **快速分支**：α=0 → 只执行 `y=βy`（β=0 直接覆写、β=1 y 不变、不读 A/x——
   A/x 含 Inf/NaN 亦正常，短路先于数值读取）；m=0 或 n=0 → SUCCESS。
4. **tiling 组装**：trans 路径（N/T）× slab 均衡参数（N：按行 slab；T：按列 slab，
   列主序连续）× 尾块/空洞参数（lda>m 列间空洞、尾段标量）× 步长映射参数
   （正/负）；经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（N/T 双路径带宽引擎，本任务技术主体）

AIV 向量引擎（BLAS-2 访存受限，无 Cube 收益；fork arch22 cgemv 骨架，去复数平面）：

- **N 路径（列核式行 dot）**：x 全量驻留 UB（n ≤ 4097：16KB FP32）；A 按行 slab
  流式装载（行内连续 GM burst、跨列按 lda 步进），每行段 Mul + Add 树归约 →
  行累加器 α 折叠 + β 折叠 → y 按步长 scatter 写回；
- **T 路径（列 dot）**：A 按列 slab 流式装载（列主序天然连续 burst，含 lda 空洞
  跳段），x 广播驻留；逐列 Mul + 树归约 → y(c) 折叠写回（incy=1 时 MTE3 连续
  burst，负步长自 span 尾端反向 scatter）；
- **多核满并发大块 DMA**：A 流量占绝对主导（4097² 大档 67MB）——slab 均衡切分
  至 K ≤ 40 核，每核独立 (MTE2→V) ping-pong 双缓冲，A 每元素仅读一次；
- **尾块与空洞**：行/列尾段（非 32 对齐）走标量段；lda>m 列间空洞在列 slab
  装载时按段跳过（不读空洞）；
- **α=0/β 短路 kernel 路径**：单独轻量 kernel 只做 y=βy 重写（不装载 A/x）；
- **小 n 单核路径**：n ≤ 32 时 K=1 单核（发射开销主导，避免多核 init 链）；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、
  Set/Wait 严格相邻配对、id 7 不可用、等待对不足处补 PipeBarrier；
- **host 侧零分配**：性能口径含下发开销（aclrtEvent 交叉验证路径）——每调用
  零 malloc/零 H2D memcpy（tiling 走栈上结构 + kernel 参数区）。

### 3.2.3 调用与执行流程

```text
aclblasSgemv ─校验链（含溢出守卫）→ 标量解引用 → quick-return/短路判定
  → 路径×slab tiling 组装 → sgemv_kernel<<<K, stream>>>
       └─ AIV: [短路路径：y=βy 重写]
          ├─ N：x 驻留 → 行 slab 流读(ping-pong) → 行 dot 树归约 → α/β 折叠 scatter
          └─ T：x 广播驻留 → 列 slab 流读 → 列 dot 树归约 → α/β 折叠写回
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 4097² 大档 A 67MB 流读逼近带宽天花板（见 §3.2.5 反算） | A 每元素仅读一次 + 多核满并发大块 DMA + ping-pong 双缓冲 |
| 事件口径（GTest avg_us 含下发开销）与 kernel-only 判据错位 | host 零 malloc/零 memcpy；正式判据走 msprof op task duration |
| 跨度溢出守卫遗漏（arch35 参考无此检查） | host int64 闭式校验单点维护 + 溢出专项负向用例 |
| 非对齐 4097/空洞 lda 带宽损失 | 段切分（连续 burst + 尾标量段）+ 空洞跳段 |

### 3.2.5 性能设计反算（R07）

判据 = H100 gpu_ms × 1.25（msprof op task duration，kernel-to-kernel）。
官方 gpu_baseline.csv 锚点：

| case | trans | m×n | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- |
| sgemv-perf-021 | N | 4097×4097 | 0.0601 | 75.1 |

N 大档：A 67.1MB@HBM 1.44TB/s → 流读底线 ~46.6us + x/y 装载回写（~1MB，<1us）+
归约/折叠开销 ≤ 8us → 预算 ~55us，对 75.1us 达标线留 ≥ 25% 余量；
T 大档流量同量级（A 同为单遍读）。小档（m/n ≤ 64）以固定开销
（init 链 + 发射 ≤ 8us）为下限。采集口径：msprof op 自带 5 warmup + ≥10 采样均值；
spike 以 4097² N/T 双锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核 UB：x 驻留（≤ 16KB）+ A slab ping-pong（双缓冲 ≤ 190KB 总预算）+
y 段暂存；无 workspace/GM 中转。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证仅需覆盖 910B3（任务书 §3.1）。

## 3.4 算子约束限制

m/n ≥ 0；lda ≥ max(1,m)；incx/incy ≠ 0 且非 INT32_MIN 且跨度不溢出（int64 校验）；
FLOAT32 专用；y 原地覆写（β=0 不读旧值）；异步执行依赖 handle 绑定 stream；
归约顺序不保证跨次一致（无确定性要求，任务书 §2.5）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32·ULP
  ——按最大误差点 golden 动态计算）；Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf
  等效；非有限值单独统计）；golden = Netlib `sgemv` CPU 参考实现
  （float64 真值计算收窄 FLOAT32 落盘，任务书 §3.2）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us`（aclrtEvent 夹连续下发）仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 sgemv_test.csv 全量接入（1000 精度 + 200 性能；trans N/T/C、
   m/n 全档含非方阵、lda 紧凑与 m+8 空洞、incx/incy 正负步长、
   alpha/beta 特殊值含 α=0/β=0、Inf/NaN 传播；负向用例：nullptr handle/
   非法 trans/负 m、n/低 lda/零步长/INT32_MIN/**跨度溢出（n=4×incx=INT_MAX、
   n=2³⁰×inc=8）**/空标量/空指针）+ **GAUSS_NORM_5_2 正态分布补充**（μ∈[-5,5]、
   σ∈[0.1,2]，扩展 gen_csv.py 补齐并在自测报告说明，任务书 §3.5-正态补充要求）；
2. **工程**：`test/gemv/sgemv/arch22/sgemv_test.cpp` CSV 驱动 GTest；
   `test/gemv/sgemv/CMakeLists.txt` 补 arch22 分支
   `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)` 注册（任务书 §2.2）；
   编译入口 `bash build.sh --ops=sgemv --soc=ascend910b3`（A3 对应 --soc）；
   **负向用例缓冲区分配规则**：期望非 SUCCESS 的用例以单元素 dummy/NULLPTR 占位，
   不按 (n−1)×|inc| 预分配（任务书 §3.5 规则）；
3. **精度回归**：1000 条全量 VerifyMixed（含 Inf/NaN 等效统计）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、golden 构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 声明已存在（L552）与 950PR 共用，不改既有签名；
- arch22 与 cgemv/上游 sgemv 实现同目录共存（互不影响，文件独立三件套）；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasSgemv_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib sgemv：https://netlib.org/blas/sgemv.f
3. cuBLAS cublasSgemv：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
