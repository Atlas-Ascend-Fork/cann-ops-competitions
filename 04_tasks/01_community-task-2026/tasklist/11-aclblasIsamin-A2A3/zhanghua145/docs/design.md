# aclblasIsamin 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasIsamin_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/iamin/arch22/`，测试 `test/isamin/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasIsamin算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现 BLAS-1 绝对值最小元素索引 `aclblasIsamin`
（对标 cuBLAS `cublasIsamin`；Netlib 参考库无 isamin.f，golden 按 cuBLAS 文档语义
编写 CPU 参考实现）：

```text
result = argmin(i=1..n) |x[j]|，并列取最先出现的 i（1-based）
j = 1 + (i−1)×incx；正常计算时 result ∈ [1, n]
n=0 或 incx<1：result 写 0 并返回成功（不触发归约计算）
NaN 语义：首元素为 NaN 时基准置 FLT_MAX，后续 NaN 跳过（严格小于比较天然跳过）；
         全部元素为 NaN 时结果为 1
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasIsamin_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（isamin_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasIsamin` 声明**已存在**（L263，与 950PR 共用） | 无需新增声明，禁止私有平行接口（任务书 §2.3） |
| `blas/iamin/arch22/` | **本任务缺口**（上游 master 已合入同任务实现 isamin/icamin 三件套） | 实现落点与同族工程模式（任务书 §2.2） |
| `blas/iamin/arch35/` | isamin 950 版（校验入口 L29-53） | 参数校验、quick return 与 NaN 口径的实现参照 |
| `test/isamin/` | CMakeLists 仅注册 arch35 | 新建 arch22 测试工程并补 `ops_blas_add_gtest_tests(${OPS_BLAS} isamin_test)` 注册（任务书 §2.2） |

### 功能缺口与本任务差异点

**arch22 Isamin 为本任务交付主体**。设计要点集中在归约正确性语义：

1. **(值, 索引) 对归约**：`result` 为 1-based 索引、并列取最先出现——
   归约比较须携带索引参与（`v < best || (v == best && idx < bestIdx)`）；
2. **NaN 基准语义**（任务书 §2.1）：首元素 NaN 时基准置 **FLT_MAX**（区别于
   Icamin 的"基准停留 NaN"口径）——后续 NaN 在严格小于比较下天然跳过、
   全 NaN → result=1；
3. **quick return 语义**：n=0 或 incx<1 → **result 写 0** 并返回成功
   （写 result 不可跳过）；result 指针判空在**任意 n（含快速返回路径）**执行。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/iamin/arch22/` 实现 `aclblasIsamin`：n 覆盖 {0,1,2,3,1024,16384,262144,1048576} ×
incx ∈ {1,2,7,16}（incx<1 走快速返回）× 并列最小值（含重复元素与 ±0）×
NaN 样本 × 负向用例（含跨度溢出：n=4×incx=INT_MAX、n=2³⁰×步长 8）。

精度判定：`result`（INT32 索引输出）与 golden **精确一致**（`actual == golden`，
matched_ratio 1.0、max_abs_error 0——整数索引 bit-exact，不适用浮点容差）；
golden 由测试工程内按 cuBLAS 文档语义编写的 CPU 参考实现直接计算 1-based 索引。
性能 msprof op 模式 task duration，5 次 warmup + 有效采样 ≥10 次均值，
逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | argmin \|x\|，1-based，并列取最先出现 | §2.1 | (值,索引) 对两段归约（§3.2.2） |
| R02 | n=0 或 incx<1 → result 写 0 并 SUCCESS | §2.1/§2.4 | Host quick return（写 result）（§3.2.1） |
| R03 | n<0 → `INVALID_VALUE`；result 为 nullptr（任意 n，含快速返回路径）→ `INVALID_VALUE` | §2.4/§2.5 | Host 校验链（§3.2.1） |
| R04 | NaN：首元素基准置 FLT_MAX、后续 NaN 跳过、全 NaN → 1 | §2.1 | 归约比较语义（§3.2.2） |
| R05 | (n−1)×incx 溢出守卫（int64） | §2.4 | Host 步长守卫（§3.2.1） |
| R06 | 精度 bit-exact 索引比对 | §3.2 | 测试工程（§4.2） |
| R07 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
result = 1 + argmin_{i ∈ [0,n)} v(i)，v(i) = |x[i×incx]|
比较规则：v(i) < best → 更新 (best, idx)；v(i) == best → 保持最小 idx（最先出现）
NaN 语义：首元素 NaN → best 初值置 FLT_MAX（idx=0 基准保留）；NaN 与任何值的
严格小于比较均为 false → 后续 NaN 天然跳过；全 NaN → 无更新 → result = 1
```

归约结果为离散量（比较在 IEEE 语义下确定）——**归约划分顺序不影响结果**，
多核两段归约与单核结果 bit-exact 一致（天然确定性）。

### 3.1.2 支持数据类型

x = FLOAT32；result = INT32（单元素，Device 内存，独立输出）。
x 物理元素数 ≥ 1+(n−1)×incx（incx ≥ 1 时）。n、incx 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/iamin/arch22/
├── isamin_host.cpp          # aclblasIsamin 入口：校验链/quick return/tiling/发射
├── isamin_kernel.cpp        # Ascend C AIV kernel（两段 (值,索引) 归约）
├── isamin_kernel.h
└── isamin_tiling_data.h     # tiling 结构 + slab/中间表参数
test/isamin/arch22/
├── isamin_test.cpp          # CSV 驱动 GTest（精度精确比对 + 性能 PERF 路径）
├── isamin_test.csv          # 官方 1000 精度 + 200 性能（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（L263），无需修改
test/isamin/CMakeLists.txt   # arch22 分支注册 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（对齐 arch35 校验入口 L29-53）：handle 判空（`HANDLE_IS_NULLPTR`）→
   result 指针判空（**任意 n，含快速返回路径**，`INVALID_VALUE`）→
   n<0（`INVALID_VALUE`）→ **n=0 或 incx<1 → quick return**：result 写 0 +
   SUCCESS（Host 侧经 stream 异步小写入或轻量 kernel 写 0，不触发归约 kernel）→
   (n−1)×incx 以 int64 计算超 INT32_MAX（`INVALID_VALUE`）→
   x 指针判空（n>0 且 incx≥1 时，`INVALID_VALUE`）。
2. **tiling 组装**：n×incx → slab 均衡 K ≤ 40（段数）；两段归约中间表参数
   （K 槽 (值,索引) 对）；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（两段 (值,索引) 归约，本任务技术主体）

AIV 向量引擎（BLAS-1 带宽型归约，无 Cube 收益）：

- **Stage 1（多核分段归约）**：x 按 slab 流式装载（步长 incx 元素跳过，
  incx=1 连续 burst；非 32 对齐尾段标量）→ 向量 `|x|` 平面 pass（Abs）→
  段内 (值, 索引) 对树归约：pair 比较规则 `v < b || (v == b && idx < bIdx)`
  （并列取最先出现；NaN 在严格小于下天然跳过——比较结果 false 即不更新）→
  每核产出 1 个 (best, idx) 写入 GM 中间表（K 槽，固定分配无原子）；
- **Stage 2（合并 kernel，K ≤ 40 槽单核）**：按同规则合并 K 槽 →
  result = idx+1（1-based）；NaN 基准语义（首元素 NaN → best 置 FLT_MAX、
  全 NaN → 1）由 Stage 1 段内基准与 Stage 2 合并规则共同承载：段首元素
  NaN 时段基准置 FLT_MAX（保留 idx=段首，任务书 §2.1 口径），全 NaN 段
  保持 (FLT_MAX, 段首) → 合并后无有效更新 → result=1；
- **±0 并列**：|±0|=0 相等 → 取最小索引（pair 规则天然覆盖）；
- **两段合并为一发射**：K ≤ 40 槽合并量极小——Stage 2 以尾块 block 形式
  与 Stage 1 同 launch（尾核执行，跨核同步用 SyncAll + 中间表 GM 落盘）；
- **小 n 单核路径**：n ≤ 4096 时 K=1（单段归约直出 result）；
- **UB 预算**：x slab ping-pong（≤ 190KB）+ pair 累加器（寄存器/UB 槽）。

### 3.2.3 调用与执行流程

```text
aclblasIsamin ─校验链（result 判空任意 n）→ n=0/incx<1 → result=0 + SUCCESS
  → 溢出守卫 → slab tiling → isamin_kernel<<<K, stream>>>
       └─ AIV: Stage1 各核{|x| 平面 → 段内 pair 树归约 → 中间表槽}
          → SyncAll → 尾核 Stage2{K 槽 pair 合并 → result = idx+1}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 并列取最先出现（1-based tie-breaking） | (值,索引) pair 比较 `v<b \|\| (v==b && idx<bIdx)` 单点维护 |
| 首元素 NaN 基准口径（FLT_MAX vs 停留 NaN） | 按任务书 §2.1：段首 NaN → 段基准置 FLT_MAX（保留段首 idx）；专项用例对拍 |
| 全 NaN → result=1 | Stage1 无有效更新保持 (FLT_MAX, 首段首) → Stage2 直出 1 |
| result 判空含快速返回路径 | Host 校验链单点维护（任务书 §2.4 明确） |

### 3.2.5 性能设计反算（R07）

判据 = H100 gpu_ms × 1.25（msprof op task duration）。官方 gpu_baseline.csv 锚点：

| case | n | incx | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- |
| isamin-base-002 | 4194304 | 1 | 0.0212 | 26.5 |

大档 n=4.19M：x 16.8MB@HBM 1.44TB/s → 流读底线 ~11.6us + Abs/树归约/SyncAll
开销 ≤ 5us → 预算 ~17us，对 26.5us 达标线留 ~35% 余量；小档以固定开销
（init 链 + 发射 ≤ 6us）为下限。采集口径：msprof op 自带 5 warmup + ≥10 采样均值；
spike 以 4.19M 连续档锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立 slab；中间表 K×8B（GM，固定分配无原子）；
x slab ping-pong（UB ≤ 190KB 总预算）；无额外 workspace。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证仅需覆盖 910B3（任务书 §3.1）。

## 3.4 算子约束限制

n ≥ 0（n<0 → `INVALID_VALUE`）；n=0 或 incx<1 → result 写 0 并 SUCCESS；
incx ≥ 1 时跨度不溢出（int64 校验）；FLOAT32 输入 / INT32 输出（独立输出，
不复用输入内存）；result 判空任意 n；异步执行依赖 handle 绑定 stream；
归约划分顺序不影响结果（整数索引天然确定）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：`result` 与 golden **精确一致**（`actual == golden`，matched_ratio 1.0、
  max_abs_error 0——整数索引 bit-exact；x 的 FLOAT32 值仅为生成域不参与比对）；
  golden 由测试工程内按 cuBLAS 文档语义编写的 CPU 参考实现直接计算
  （float32 输入，不涉及浮点收窄，任务书 §3.2）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 isamin_test.csv 全量接入（1000 精度 + 200 性能；n 全档、
   incx ∈ {1,2,7,16}、并列最小值（重复元素与 ±0）、NaN 样本（首 NaN/中段
   NaN/全 NaN → 1）、负向用例（nullptr handle/result nullptr 含 n=0 路径/
   n<0/n>0 且 incx≥1 时 x nullptr/步长溢出 n=4×incx=INT_MAX、n=2³⁰×inc=8））+
   GAUSS_NORM_5_2 正态分布补充（μ∈[-5,5]、σ∈[0.1,2]，任务书 §3.5 要求）；
   负向用例以单元素 dummy/NULLPTR 占位（不按步长规模预分配）；
2. **工程**：`test/isamin/arch22/isamin_test.cpp` CSV 驱动 GTest；
   `test/isamin/CMakeLists.txt` 补 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS}
   isamin_test)` 注册（任务书 §2.2）；编译入口 `bash build.sh --ops=isamin
   --soc=ascend910b3`（A3 对应 --soc）；
3. **精度回归**：1000 条全量精确比对；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、CPU 参考实现说明）。

## 4.3 兼容性分析

- 声明已存在（L263）与 950PR 共用，不改既有签名；
- arch22 与 icamin 同目录共存互不影响；arch35 参考实现只读引用；
- 不引入仓外依赖（golden 为测试工程内置 CPU 参考实现——Netlib 无 isamin）。

# 5. 参考链接

1. 任务书：aclblasIsamin_AtlasA2_A3_task_doc.md（2026-11-11）
2. cuBLAS cublasIsamin：https://docs.nvidia.com/cuda/cublas/index.html
3. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
4. ops-blas 仓：https://gitcode.com/cann/ops-blas
