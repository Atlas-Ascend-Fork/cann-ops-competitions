# aclblasSrot 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSrot_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/rot/arch22/`，测试 `test/rot/srot/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasSrot算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现 BLAS-1 平面向量 Givens 旋转 `aclblasSrot`
（对标 cuBLAS `cublasSrot`，语义参考 Netlib `srot.f`）：

```text
x[i] ← c·x[i] + s·y[i]
y[i] ← −s·x[i] + c·y[i]        （右端 x[i] 为更新前值；G = [[c, s], [−s, c]]）
物理位置 1+(i−1)×|inc|（1 基；负步长自 span 尾端）
n=0：x、y 不修改（no-op）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSrot_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（srot_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasSrot` 声明**已存在**（L543，与 950PR 共用） | 无需新增声明，禁止私有平行接口（任务书 §2.3） |
| `blas/rot/arch22/` | 复数族 `aclblasCsrot` 已有（host/kernel 三件套）；**实数 Srot arch22 为本任务缺口**（上游 master 已合入同任务实现 srot_host/kernel/tiling） | 实现落点与同族工程模式（任务书 §2.2）；csrot 的双流读改写与步长分派框架沿用 |
| `blas/rot/arch35/` | srot 950 版 | 参数校验、no-op 口径的实现参照 |
| `test/rot/srot/` | 通用测试组织存在；CMakeLists 仅注册 arch35 | 补 `ops_blas_add_gtest_tests(${OPS_BLAS} srot_test)` arch22 注册（任务书 §2.2——A2/A3 构建否则退回非 GTest runner） |

### 功能缺口与本任务差异点

**arch22 实数 Srot 为本任务交付主体**。相对兄弟算子 csrot（arch22）的差异即主要工作量：

1. **实数 Givens 旋转**：2×2 旋转作用到 (x[i], y[i]) 元素对——x、y **成对读改写**，
   `y` 更新须用**旋转前的 x[i]**（单遍内联完成，双缓冲下无 RAW 冒险）；
2. **c/s 为 Host 内存实数标量**（float 指针），Host 侧同步解引用后随 tiling 下发；
3. **no-op 口径**：n=0 时 x、y 不修改；步长守卫（0/INT32_MIN/跨度溢出）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/rot/arch22/` 实现 `aclblasSrot`：n 覆盖大档（至 4194304）×
incx/incy 正负步长组合 × c/s 覆盖典型旋转对（如 c=0.6、s=0.8 等）×
Inf/NaN 特殊值与负向用例（含跨度溢出）。

精度按生态开源精度标准 FLOAT32 档（rtol = atol = 2⁻¹³，matched_ratio ≥ 0.99，
max_abs ≤ 1e-2 或 32·ULP）；golden 为 Netlib BLAS `srot` CPU 参考实现
（float64 真值计算收窄 FLOAT32 落盘）。性能 msprof op 模式 task duration，
5 次 warmup + 有效采样 ≥10 次均值，逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | Givens 旋转成对更新（y 用旋转前 x） | §2.1 | 成对 slab 读改写 kernel（§3.2.2） |
| R02 | n=0 no-op（x/y 不修改） | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R03 | incx/incy = 0 或 INT32_MIN → `INVALID_VALUE`；(n−1)×\|inc\| 溢出守卫（int64，#866 口径） | §2.4/§2.5 | Host 步长守卫（§3.2.1） |
| R04 | 负步长反向遍历 | §2.4 | X/YPhysPos 映射（§3.2.2） |
| R05 | 精度 FLOAT32 档 + Inf/NaN 等效 | §3.2 | 测试工程（§4.2） |
| R06 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |
| R07 | arch22 GTest 注册 + GAUSS_NORM_5_2 补充 | §2.2/§3.5 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
x'[i] =  c·x[i] + s·y[i]
y'[i] = −s·x[i] + c·y[i]        （右端 x[i]、y[i] 均为更新前值；G = [[c, s], [−s, c]]）
i = 1..n（1 基）；负步长自物理 span 尾端起算沿负方向步进
```

逐元素成对运算（无归约）：x、y 各读一次、各写一次（4×n×4B 流量）；
单次执行内确定、同输入同步长跨次 bit 级一致。

### 3.1.2 支持数据类型

x/y/c/s = FLOAT32（c、s 为 Host 内存标量指针）；x、y 均原地覆写。
x/y 物理元素数 ≥ 1+(n−1)×|inc|。n、incx、incy 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/rot/arch22/
├── srot_host.cpp            # aclblasSrot 入口：校验链/标量解引用/tiling/发射
├── srot_kernel.cpp          # Ascend C AIV kernel（成对 slab 读改写）
├── srot_kernel.h
└── srot_tiling_data.h       # tiling 结构 + slab/尾块/步长映射参数
test/rot/srot/arch22/
├── srot_test.cpp            # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── srot_test.csv            # 官方 1000 精度 + 200 性能（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（L543），无需修改
test/rot/srot/CMakeLists.txt # arch22 分支注册 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（对齐 arch35 srot 校验框架）：handle 判空（`HANDLE_IS_NULLPTR`）→
   **n=0 → SUCCESS（x、y 不修改）** → n<0（`INVALID_VALUE`，对齐仓内 arch22
   家族契约）→ c/s/x/y 指针判空（`INVALID_VALUE`）→ incx/incy = 0 或
   INT32_MIN（`INVALID_VALUE`）→ (n−1)×|inc| 以 int64 计算超 INT32_MAX
   （`INVALID_VALUE`，#866 口径）。
2. **标量读取**：c、s 为 Host 内存 float 指针，Host 侧同步解引用（值不校验）；
   随 tiling 以标量参数下发（+1.0/−1.0 特殊对可选快速乘加简化分支）。
3. **tiling 组装**：步长分派（对齐 csrot dispatch 风格）——incx=incy=1 连续路径 /
   单侧步长 / 双侧步长；slab 均衡 K ≤ 40；尾块标量段；经 kernel 直调发射。

### 3.2.2 Kernel 侧设计（成对 slab 读改写，本任务技术主体）

AIV 向量引擎（BLAS-1 纯带宽型，无 Cube 收益）：

- **连续路径（incx=incy=1）**：x、y 按 slab **双流对读**（MTE2 ping-pong ×2）→
  向量 pass 同时产出 `x' = c·x + s·y`、`y' = −s·x + c·y`（右端均为装载的
  原始 x/y——单遍内联，无 RAW 冒险）→ MTE3 **双流写回** x'、y'——
  每 slab 成对读改写一次完成（4×n×4B 流量）；
- **单侧/双侧步长路径**：步长侧 Gather 索引平面化；负步长 X/YPhysPos 反向映射
  （物理 span 尾端起算）；性能用例以 inc=1 为主；
- **尾块**：非 32 对齐尾段标量成对处理；
- **多核均分**：slab 均衡 K ≤ 40 核，各核独立 ping-pong，无跨核依赖
  （无原子、无同步）；
- **UB 预算**：x/y slab 双流 ping-pong（双缓冲 ≤ 190KB 总预算）。

### 3.2.3 调用与执行流程

```text
aclblasSrot ─校验链→ 标量解引用 → n=0 no-op
  → 步长分派 tiling 组装 → srot_kernel<<<K, stream>>>
       └─ AIV: slab 循环{x/y 双流 ping-pong 对读 → 旋转平面 pass(x',y' 同产出)
                → MTE3 双流写回}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| y' 需旋转前 x（双写 RAW 感知） | 成对单遍内联（右端用装载原始值），无中间回写 |
| 大档 4×n×4B 流量带宽压榨 | 双流对读 + 双流写 + ping-pong；每元素恰读改写一次 |
| 跨度溢出守卫 | host int64 闭式校验 + 溢出专项负向用例 |

### 3.2.5 性能设计反算（R06）

判据 = H100 gpu_ms × 1.25（msprof op task duration）。官方 gpu_baseline.csv 锚点：

| case | n | incx/incy | c/s | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- | --- |
| srot-base-002 | 4194304 | 1/1 | 0.6/0.8 | 0.0488 | 61.0 |

大档 n=4.19M：流量 4×16.8MB=67.1MB@HBM 1.44TB/s → 底线 ~46.6us + 发射/init
开销 ≤ 4us → 预算 ~51us，对 61.0us 达标线留 ~16% 余量；小档以固定开销 ≤ 6us 为下限。
采集口径：msprof op 自带 5 warmup + ≥10 采样均值；spike 以 4.19M 连续档锚点
校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立 slab；x/y 双流 ping-pong 双缓冲（UB ≤ 190KB 总预算）；
无 workspace/GM 中转。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证仅需覆盖 910B3（任务书 §3.1）。

## 3.4 算子约束限制

n=0 no-op（x/y 不修改）；n<0 → `INVALID_VALUE`（arch22 家族契约）；incx/incy ≠ 0
且非 INT32_MIN 且跨度不溢出；FLOAT32 专用；x、y 均原地覆写（y 用旋转前 x）；
c/s Host 标量同步解引用；异步执行依赖 handle 绑定 stream；
同输入同步长跨次 bit 级一致。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32·ULP）；
  Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf 等效；非有限值单独统计）；
  golden = Netlib `srot` CPU 参考实现（float64 真值计算收窄 FLOAT32 落盘）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 srot_test.csv 全量接入（1000 精度 + 200 性能；n 全档、
   incx/incy 正负步长、c/s 典型旋转对、Inf/NaN 传播（含 0×Inf 不定式传播）、
   负向用例（nullptr handle/n<0/空 c、s/x、y/零步长/INT32_MIN/跨度溢出））+
   GAUSS_NORM_5_2 正态分布补充（μ∈[-5,5]、σ∈[0.1,2]，任务书 §3.5 要求）；
   负向用例以单元素 dummy/NULLPTR 占位（不按步长规模预分配）；
2. **工程**：`test/rot/srot/arch22/srot_test.cpp` CSV 驱动 GTest；
   `test/rot/srot/CMakeLists.txt` 补 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS}
   srot_test)` 注册（任务书 §2.2）；编译入口 `bash build.sh --ops=srot
   --soc=ascend910b3`（A3 对应 --soc）；
3. **精度回归**：1000 条全量 VerifyMixed（x、y 分别比对）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、golden 构建说明）。

## 4.3 兼容性分析

- 声明已存在（L543）与 950PR 共用，不改既有签名；
- arch22 与 csrot/上游 srot 实现同目录共存（文件独立三件套）；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasSrot_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib srot：https://netlib.org/blas/srot.f
3. cuBLAS cublasSrot：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
