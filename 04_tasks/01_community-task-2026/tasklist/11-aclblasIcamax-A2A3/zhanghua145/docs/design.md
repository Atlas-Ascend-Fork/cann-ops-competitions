# aclblasIcamax 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasIcamax_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/iamax/arch22/`，测试 `test/iamax/icamax/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasIcamax算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数向量最大模元素索引 `aclblasIcamax`
（对标 cuBLAS `cublasIcamax`，语义参考 Netlib `icamax.f` 与仓内同族实数接口
`aclblasIsamax`）：

```text
result = argmax(i=1..n) ( |Re(x[k])| + |Im(x[k])| )，k = 1+(i−1)×incx，1-based
复数"模"为 1-范数口径（|Re|+|Im|，Netlib SCABS1 语义，非欧氏模）
n=0 或 incx≤0（含 0 与负步长）：不触发归约，直接写 result = 0 并返回成功
（本算子不支持负步长反向遍历——Netlib 1993 年起 incx ≤ 0 直接返回 0）
n<0 → ACLBLAS_STATUS_INVALID_VALUE（对齐仓内同族 aclblasIsamax 约束）
特殊值语义：比较采用"严格大于才更新当前最大值"（对齐 Netlib）；含 Inf 元素时
首个 Inf 所在元素胜出；NaN 元素在严格大于比较下不会胜出（首元素含 NaN 时
Netlib 语义下结果停留为索引 1）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasIcamax_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（icamax_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasIcamax` 声明**已存在**（L262，与 950PR 共用；签名逐参数对齐同族实数 `aclblasIsamax`：int n / int incx / int* result） | 无需新增声明，禁止私有平行接口（任务书 §2.3） |
| `blas/iamax/arch22/` | 实数 `aclblasIsamax` 已有（host/kernel/impl 三件套）；**复数 Icamax arch22 为本任务缺口**（上游 master 的 arch22 目录目前仍无 icamax 文件——arch35 已有 icamax 三件套） | 实现落点（与 isamax 同目录，任务书 §5）；实数兄弟的归约骨架与校验框架沿用；本任务为该目录补齐复数路径 |
| `blas/iamax/arch35/` | icamax 950 版 | 参数校验、quick return 与 SCABS1 口径的实现参照 |
| `test/iamax/` | 通用测试组织存在（icamax 子目录需新建 arch22 分支） | 测试工程新建 `test/iamax/icamax/arch22/`（任务书 §5） |

### 功能缺口与本任务差异点

**arch22 Icamax 为本任务交付主体（上游 arch22 目录当前仍缺复数路径）**。
相对实数兄弟（isamax）的差异即主要工作量：

1. **复数 1-范数模**：`v(i) = |Re| + |Im|`（SCABS1 口径）——interleaved
   complex64 deinterleave 后双平面 Abs + Add；
2. **argmax 严格大于更新语义**：`v > best` 才更新（并列保持最先出现）；
   Inf 元素首个胜出（严格大于链上 Inf 只被更早的 Inf 前置）；NaN 不胜出
   （NaN 比较恒 false）——首元素 NaN → 结果停留 1；
3. **incx ≤ 0 quick return**（含 0 与负步长，本算子不支持负步长反向遍历）；
   **n<0 → `INVALID_VALUE`**（对齐 isamax 家族约束）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/iamax/arch22/` 实现 `aclblasIcamax`：n 覆盖大档（至 4194304）×
incx 覆盖正步长（incx≤0 quick return 负向验证）× 全零（并列最小索引语义）/
常量（全部元素同模，并列取索引 1）/正负交替/极端值/Inf/NaN 特殊值用例 ×
负向用例（含跨度溢出，#745 口径）。

精度判定：`result`（INT32 索引）与 golden **精确相等（bit-exact）**，不做浮点
容差比对；特殊值用例以 cblas golden 精确一致为判定。性能 msprof op 模式
task duration，5 次 warmup + 有效采样 ≥10 次均值，逐 case ≤ H100 gpu_ms × **1.25**
（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | argmax ( \|Re\|+\|Im\| )，1-based，严格大于更新（并列取最先） | §2.1 | (值,索引) 对两段归约 + 复数平面模（§3.2.2） |
| R02 | n=0 或 incx≤0 → result 写 0 并 SUCCESS（不触发归约；不支持负步长） | §2.1/§2.4 | Host quick return（写 result）（§3.2.1） |
| R03 | n<0 → `INVALID_VALUE`（对齐 isamax 家族）；result nullptr（任意 n）→ `INVALID_VALUE` | §2.4/§2.5 | Host 校验链（§3.2.1） |
| R04 | Inf 首个胜出；NaN 不胜出（首 NaN → 停留 1） | §2.1/§3.2 | 归约比较语义（§3.2.2） |
| R05 | (n−1)×incx 溢出守卫（int64，#745 口径） | §2.4 | Host 步长守卫（§3.2.1） |
| R06 | 精度 bit-exact 索引比对（特殊值以 cblas golden 判定） | §3.2 | 测试工程（§4.2） |
| R07 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
result = 1 + argmax_{i ∈ [0,n)} v(i)，v(i) = |Re(x[i×incx])| + |Im(x[i×incx])|
比较规则：v(i) > best → 更新（严格大于，对齐 Netlib）；v(i) == best → 保持最小
idx（并列取最先出现）；Inf：首个 Inf 严格大于此前一切有限值 → 胜出后仅更早
的 Inf 可并列；NaN：参与比较恒 false → 不胜出；首元素 NaN → best=NaN 停留 →
后续（含 Inf）均不严格大于 NaN？——Inf > NaN 为 false（IEEE 无序），
NaN 停留即最终停留 → result = 1（对齐任务书 §2.1 Netlib 语义）
```

归约结果为离散量（IEEE 语义下确定），**归约划分顺序不影响结果**。

### 3.1.2 支持数据类型

x = COMPLEX64（interleaved float32 re/im，`aclblasComplex`）；result = INT32
（单元素，Device 内存，独立输出）。x 物理元素数 ≥ 1+(n−1)×|incx|。
n、incx 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/iamax/arch22/
├── icamax_host.cpp          # aclblasIcamax 入口：校验链/quick return/tiling/发射
├── icamax_kernel.cpp        # Ascend C AIV kernel（两段 (值,索引) 归约 + 复数模）
├── icamax_kernel.h
└── icamax_tiling_data.h     # tiling 结构 + slab/中间表参数
test/iamax/icamax/arch22/
├── icamax_test.cpp          # CSV 驱动 GTest（精度精确比对 + 性能 PERF 路径）
├── icamax_test.csv          # 官方 1000 精度 + 200 性能（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（L262），无需修改
test/iamax/icamax/CMakeLists.txt  # 注册 arch22 目标 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（对齐同族 isamax 家族约束）：handle 判空（`HANDLE_IS_NULLPTR`）→
   result 指针判空（**任意 n，含快速返回路径**，`INVALID_VALUE`）→
   n<0（`INVALID_VALUE`，对齐 isamax）→ **n=0 或 incx≤0 → quick return**：
   result 写 0 + SUCCESS（不触发归约 kernel）→ (n−1)×incx 以 int64 计算超
   INT32_MAX（`INVALID_VALUE`，#745 口径）→ x 指针判空（n>0 且 incx≥1 时，
   `INVALID_VALUE`）。
2. **tiling 组装**：n×incx → slab 均衡 K ≤ 40；两段归约中间表参数
   （K 槽 (值,索引) 对）；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（两段 (值,索引) 归约 + 复数模，本任务技术主体）

AIV 向量引擎（BLAS-1 带宽型归约，无 Cube 收益）：

- **Stage 1（多核分段归约）**：x 复数 slab 流式装载（步长 incx 元素跳过，
  incx=1 连续 burst；非 32 对齐尾段标量）→ Gather 索引（byte offset 8m/8m+4）
  deinterleave 为 re/im 平面 → 向量模平面 pass `v = |re| + |im|`
  （Abs×2 + Add）→ 段内 (值,索引) 对树归约：pair 比较
  `v > b`（**严格大于**，对齐 Netlib）更新、相等保持最小 idx → 每核产出
  1 个 (best, idx) 写入 GM 中间表（K 槽，固定分配无原子）；
- **NaN/Inf 语义**：NaN 参与的严格大于比较恒 false → 不胜出；首元素 NaN →
  段基准停留 NaN（后续 Inf > NaN 亦 false）→ 段产出 (NaN, 段首)；Inf 在
  严格大于链上首个胜出；跨段合并同规则 → 全 NaN/首 NaN → result=1
  （对齐任务书 §2.1 Netlib 语义）；分段归约与整体语义一致性由 CPU 参考
  golden 专项用例验证；
- **Stage 2（合并，K ≤ 40 槽）**：同 pair 规则合并 → result = idx+1（1-based），
  以尾块 block 与 Stage 1 同 launch（SyncAll + 中间表 GM 落盘）；
- **小 n 单核路径**：n ≤ 4096 时 K=1 直出；
- **UB 预算**：x slab ping-pong（≤ 190KB）+ re/im 平面暂存 + pair 累加器。

### 3.2.3 调用与执行流程

```text
aclblasIcamax ─校验链（result 判空任意 n）→ n=0/incx≤0 → result=0 + SUCCESS
  → 溢出守卫 → slab tiling → icamax_kernel<<<K, stream>>>
       └─ AIV: Stage1 各核{deinterleave → |re|+|im| 平面 → 段内 pair 树归约
                → 中间表槽} → SyncAll → 尾核 Stage2{K 槽合并 → result = idx+1}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 严格大于语义（并列取最先、Inf 首个胜出、NaN 停留） | pair 比较 `v > b` 单点维护 + Inf/NaN 分布专项对拍 cblas golden |
| 模口径（1-范数 SCABS1） | `|Re|+|Im|` 平面 pass 单点承载 + 全零/常量/正负交替专项 |
| 段基准 NaN 与跨段合并一致性 | NaN 停留语义 + 专项用例（NaN 段 × Inf 段混合） |
| result 判空含快速返回路径 | Host 校验链单点维护（任务书 §2.4 明确） |

### 3.2.5 性能设计反算（R07）

判据 = H100 gpu_ms × 1.25（msprof op task duration）。官方 gpu_baseline.csv 锚点：

| case | n | incx | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- |
| icamax-base-002 | 4194304 | 1 | 0.0276 | 34.5 |

大档 n=4.19M：x 33.5MB@HBM 1.44TB/s → 流读底线 ~23.3us + deinterleave/模
平面/树归约/SyncAll 开销 ≤ 5us → 预算 ~28us，对 34.5us 达标线留 ~19% 余量；
小档以固定开销（init 链 + 发射 ≤ 6us）为下限。采集口径：msprof op 自带
5 warmup + ≥10 采样均值；spike 以 4.19M 连续档锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立 slab；中间表 K×8B（GM，固定分配无原子）；
x slab ping-pong + 平面暂存（UB ≤ 190KB 总预算）；无额外 workspace。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

n ≥ 0（n<0 → `INVALID_VALUE`，对齐 isamax 家族）；n=0 或 incx≤0 → result 写 0
并 SUCCESS（不支持负步长反向遍历，Netlib 口径）；(n−1)×incx 跨度不溢出
（int64 校验，#745 口径）；COMPLEX64 输入（SCABS1 1-范数模）/ INT32 输出
（独立输出）；result 判空任意 n；异步执行依赖 handle 绑定 stream；
归约划分顺序不影响结果（整数索引天然确定）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：`result` 与 golden **精确相等（bit-exact）**，不做浮点容差比对；
  特殊值用例以 cblas golden 精确一致为判定（任务书 §3.2/§2.1）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 icamax_test.csv 全量接入（1000 精度 + 200 性能；n 全档、
   incx 正步长（incx≤0 quick return 负向验证）、全零（并列最小索引语义）、
   常量（全部同模并列取 1）、正负交替、极端值、Inf/NaN 特殊值（首 NaN 停留 1/
   首个 Inf 胜出/中段 NaN 跳过/全 NaN → 1）、负向用例（nullptr handle/
   result nullptr 含 n=0 路径/n<0/n>0 且 incx≥1 时 x nullptr/步长溢出））+
   GAUSS_NORM_5_2 正态分布补充（μ∈[-5,5]、σ∈[0.1,2]，任务书 §3.5 要求）；
   负向用例以单元素 dummy/NULLPTR 占位；
2. **工程**：`test/iamax/icamax/arch22/icamax_test.cpp` CSV 驱动 GTest；
   `test/iamax/icamax/CMakeLists.txt` 注册 arch22 目标
   （编译入口对齐仓内 `--ops=icamax --soc=ascend910b3`，A3 对应 --soc）；
3. **精度回归**：1000 条全量精确比对（Inf/NaN 语义专项）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、golden 构建说明）。

## 4.3 兼容性分析

- 声明已存在（L262）与 950PR 共用（签名对齐同族 isamax：int n/incx/result），
  不改既有签名；
- arch22 与 isamax 同目录共存互不影响（isamax 路径零改动，icamax 独立三件套）；
  arch35 参考实现只读引用；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasIcamax_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib icamax：https://www.netlib.org/blas/icamax.f
3. cuBLAS cublasIcamax：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
