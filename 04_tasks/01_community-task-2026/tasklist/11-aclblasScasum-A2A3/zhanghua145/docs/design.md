# aclblasScasum 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasScasum_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/asum/arch22/`，测试 `test/asum/scasum/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasScasum算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现复数向量元素绝对值之和 `aclblasScasum`
（对标 cuBLAS `cublasScasum`，语义参考 Netlib `scasum`）：

```text
result = Σ(i=1..n) ( |Re(x[k])| + |Im(x[k])| )，k = 1+(i−1)×|incx|
quick return：n ≤ 0 或 incx ≤ 0（含 0 与负步长）时不触发 kernel，
直接写 result = 0.0f 并返回成功（golden 固定 0.0f，无需调用 cblas）
累加为 |Re|+|Im| 逐项非负累加——大 n × 极端值下部分和可溢出到 Inf（自然 IEEE 行为）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasScasum_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（scasum_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasScasum` 声明**已存在**（L259，与 950PR 共用，与仓内同族 `aclblasSasum` 同构：float* result） | 无需新增声明，禁止私有平行接口（任务书 §2.3） |
| `blas/asum/arch22/` | 实数 `aclblasSasum` 已有（host/kernel 三件套）；**复数 Scasum arch22 为本任务缺口**（上游 master 已合入同任务实现 scasum 三件套） | 实现落点（与 sasum 同目录，任务书 §5）；实数兄弟的归约骨架与校验框架沿用 |
| `blas/asum/arch35/` | scasum 950 版 | 参数校验、quick return 与标量输出口径的实现参照 |
| `test/asum/` | 通用测试组织存在（scasum 子目录需新建 arch22 分支） | 测试工程新建 `test/asum/scasum/arch22/`（任务书 §5） |

### 功能缺口与本任务差异点

**arch22 Scasum 为本任务交付主体**。相对实数兄弟（sasum）的差异即主要工作量：

1. **复数 1-范数逐项**：`|Re| + |Im|`（interleaved complex64 deinterleave 后
   双平面 Abs + Add）再求和——非欧氏模；
2. **标量输出比对**：输出为单个 FLOAT32 实数标量，精度验证为**标量比对**
   （单值 `|actual − golden| ≤ atol + rtol·|golden|`，matched_ratio 退化为
   单值判定），非逐元素比对；
3. **quick return 口径**：n ≤ 0 **或** incx ≤ 0（含负维度 n<0 亦为 quick
   return——任务书 §2.4 与实数族 n<0 报错口径不同，本任务书明确 n≤0 含
   负维度一律 quick return 写 0.0f）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/asum/arch22/` 实现 `aclblasScasum`：n 覆盖大档（至 4194304）×
incx 覆盖正步长（incx≤0 quick return 负向验证）× 全零/正负交替/极端值/
Inf/NaN 特殊值用例 × 大 n × 极端值覆盖累加溢出场景（结果 Inf）。

精度判定：标量比对（FLOAT32 档：rtol = atol = 2⁻¹³，max_abs ≤ 1e-2 或
32·ULP；标量场景 matched_ratio 退化为单值判定）；golden 由 cblas
（Netlib `scasum`）单标杆生成；quick return 用例 golden 固定 0.0f；
Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf 等效——大 n×FLT_MAX 累加溢出
到 Inf 的结果按 IEEE 语义与 golden 等效判定）。性能 msprof op 模式
task duration，5 次 warmup + 有效采样 ≥10 次均值，逐 case ≤ H100 gpu_ms ×
**1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `result = Σ(|Re|+|Im|)`（SCABS1 逐项） | §2.1 | 复数平面模 + 两段求和归约（§3.2.2） |
| R02 | n≤0 或 incx≤0 → result 写 0.0f + SUCCESS（含负维度，不触发 kernel） | §2.1/§2.4 | Host quick return（写 result）（§3.2.1） |
| R03 | result 为 nullptr → `INVALID_VALUE`；x nullptr（n>0 且 incx>0）→ `INVALID_VALUE` | §2.4 | Host 校验链（§3.2.1） |
| R04 | (n−1)×\|incx\| 溢出守卫（int64，#805 口径） | §2.4 | Host 步长守卫（§3.2.1） |
| R05 | 累加溢出到 Inf 的 IEEE 自然行为 | §2.1 | FP32 部分和 + 等效判定（§4.1） |
| R06 | 精度标量比对 + Inf/NaN 等效 | §3.2 | 测试工程（§4.2） |
| R07 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
result = Σ_{i ∈ [0,n)} ( |Re(x[i×incx])| + |Im(x[i×incx])| )
逐项非负累加；元素接近 FLT_MAX 且 n 大时部分和可溢出到 Inf（IEEE 自然行为，
golden 同口径）；归约累加顺序不要求与参考实现逐位一致（任务书 §2.1/§3.2——
浮点和对顺序敏感，按阈值判定）
```

注意：**归约顺序影响浮点结果**（与索引类归约不同）——分段求和与 golden
（顺序累加）存在舍入差异，按 §3.2 阈值判定（2⁻¹³ 档，n≤4.19M 非负累加
的分段树归约误差远低于阈值）。

### 3.1.2 支持数据类型

x = COMPLEX64（interleaved float32 re/im，`aclblasComplex`）；result = FLOAT32
（单元素，Device 内存，独立输出标量）。x 物理元素数 ≥ 1+(n−1)×|incx|。
n、incx 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/asum/arch22/
├── scasum_host.cpp          # aclblasScasum 入口：校验链/quick return/tiling/发射
├── scasum_kernel.cpp        # Ascend C AIV kernel（复数平面模 + 两段求和归约）
├── scasum_kernel.h
└── scasum_tiling_data.h     # tiling 结构 + slab/中间表参数
test/asum/scasum/arch22/
├── scasum_test.cpp          # CSV 驱动 GTest（精度标量比对 + 性能 PERF 路径）
├── scasum_test.csv          # 官方 1000 精度 + 200 性能（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（L259），无需修改
test/asum/scasum/CMakeLists.txt  # 注册 arch22 目标 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**：handle 判空（`HANDLE_IS_NULLPTR`）→ result 指针判空
   （`INVALID_VALUE`）→ **n≤0 或 incx≤0 → quick return**：result 写 0.0f +
   SUCCESS（含负维度 n<0——任务书 §2.4 口径，与实数族 n<0 报错不同；
   不触发 kernel）→ (n−1)×|incx| 以 int64 计算超 INT32_MAX（`INVALID_VALUE`，
   #805 口径）→ x 指针判空（n>0 且 incx>0 时，`INVALID_VALUE`）。
2. **tiling 组装**：n×incx → slab 均衡 K ≤ 40；两段求和中间表参数
   （K 槽部分和）；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（复数平面模 + 两段求和，本任务技术主体）

AIV 向量引擎（BLAS-1 带宽型归约，无 Cube 收益）：

- **Stage 1（多核分段求和）**：x 复数 slab 流式装载（步长 incx 元素跳过，
  incx=1 连续 burst；非 32 对齐尾段标量）→ Gather 索引（byte offset 8m/8m+4）
  deinterleave 为 re/im 平面 → 向量平面 pass `|re| + |im|`（Abs×2 + Add）→
  段内树归约（ReduceSum）→ 每核产出 1 个 FP32 部分和写入 GM 中间表
  （K 槽，固定分配无原子）；
- **Stage 2（合并，K ≤ 40 槽）**：K 槽部分和顺序累加（固定顺序：按核号
  升序——确定性归约口径）→ result 写回；以尾块 block 与 Stage 1 同 launch
  （SyncAll + 中间表 GM 落盘）；
- **溢出自然行为**：FP32 部分和累加不设饱和截断——大 n × FLT_MAX 元素
  自然溢出到 Inf（与 golden IEEE 同口径，Inf/NaN 等效判定）；
- **小 n 单核路径**：n ≤ 4096 时 K=1 直出（单段树归约）；
- **UB 预算**：x slab ping-pong（≤ 190KB）+ re/im 平面暂存 + 段和累加器。

### 3.2.3 调用与执行流程

```text
aclblasScasum ─校验链 → n≤0/incx≤0 → result=0.0f + SUCCESS
  → 溢出守卫 → slab tiling → scasum_kernel<<<K, stream>>>
       └─ AIV: Stage1 各核{deinterleave → |re|+|im| 平面 → 段内树归约
                → 中间表槽} → SyncAll → 尾核 Stage2{K 槽顺序累加 → result}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 分段求和与 golden 顺序累加的舍入差异 | 非负项部分和 + 树归约误差远低于 2⁻¹³ 档阈值；自测报告说明归约结构 |
| 累加溢出到 Inf 的判定 | FP32 不截断（IEEE 自然）+ Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf） |
| 模口径（1-范数逐项） | `|Re|+|Im|` 平面 pass 单点承载 + 正负交替专项 |
| result 判空与 quick return 顺序 | Host 校验链单点维护（任务书 §2.4 明确） |

### 3.2.5 性能设计反算（R07）

判据 = H100 gpu_ms × 1.25（msprof op task duration）。官方 gpu_baseline.csv 锚点：

| case | n | incx | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- |
| scasum-base-002 | 4194304 | 1 | 0.0336 | 41.9 |

大档 n=4.19M：x 33.5MB@HBM 1.44TB/s → 流读底线 ~23.3us + deinterleave/模
平面/树归约/SyncAll 开销 ≤ 5us → 预算 ~28us，对 41.9us 达标线留 ~33% 余量；
小档以固定开销（init 链 + 发射 ≤ 6us）为下限。采集口径：msprof op 自带
5 warmup + ≥10 采样均值；spike 以 4.19M 连续档锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立 slab；中间表 K×4B（GM，固定分配无原子）；
x slab ping-pong + 平面暂存（UB ≤ 190KB 总预算）；无额外 workspace。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

n≤0 或 incx≤0 → result 写 0.0f 并 SUCCESS（含负维度，不触发 kernel）；
(n−1)×|incx| 跨度不溢出（int64 校验，#805 口径）；COMPLEX64 输入（|Re|+|Im|
逐项）/ FLOAT32 标量输出（独立输出）；累加不截断（IEEE 自然溢出）；
归约顺序不要求与 golden 逐位一致（按阈值判定）；异步执行依赖 handle 绑定 stream。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档标量比对（`|actual − golden| ≤ atol + rtol·|golden|`，
  rtol = atol = 2⁻¹³，max_abs ≤ 1e-2 或 32·ULP；matched_ratio 退化为单值
  判定）；golden = cblas（Netlib `scasum`）单标杆；quick return golden
  固定 0.0f；Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf 等效；一方有限
  一方 Inf/NaN 不通过）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 scasum_test.csv 全量接入（1000 精度 + 200 性能；n 全档、
   incx 正步长（incx≤0 quick return 负向验证）、全零/正负交替/极端值/
   Inf/NaN 特殊值、大 n × 极端值累加溢出场景（结果 Inf 等效判定）、
   负向用例（nullptr handle/result nullptr/x nullptr（n>0 且 incx>0）/
   步长溢出））+ GAUSS_NORM_5_2 正态分布补充（μ∈[-5,5]、σ∈[0.1,2]，
   任务书 §3.5 要求）；负向用例以单元素 dummy/NULLPTR 占位；
2. **工程**：`test/asum/scasum/arch22/scasum_test.cpp` CSV 驱动 GTest；
   `test/asum/scasum/CMakeLists.txt` 注册 arch22 目标
   （编译入口对齐仓内 `--ops=scasum --soc=ascend910b3`，A3 对应 --soc）；
3. **精度回归**：1000 条全量标量比对（溢出/Inf/NaN 等效专项）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、golden 构建说明）。

## 4.3 兼容性分析

- 声明已存在（L259）与 950PR 共用（与同族 sasum 同构），不改既有签名；
- arch22 与 sasum 同目录共存互不影响（sasum 路径零改动，scasum 独立三件套）；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasScasum_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib scasum：https://netlib.org/blas/scasum.f
3. cuBLAS cublasScasum：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
