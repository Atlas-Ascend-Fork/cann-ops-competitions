# aclblasSaxpy 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSaxpy_AtlasA2_A3_task_doc.md》（2026-11-11 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/axpy/arch22/`，测试 `test/saxpy/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasSaxpy算子开发（A2/A3)」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）实现 BLAS-1 向量数乘累加 `aclblasSaxpy`
（对标 cuBLAS `cublasSaxpy`，语义参考 Netlib `saxpy.f`）：

```text
y[i] = α × x[i] + y[i]，i = 1..n（1 基；物理位置 1+(i−1)×|inc|，负步长自 span 尾端）
n ≤ 0：合法 no-op（不访问 alpha、x、y，对齐 Netlib IF (N.LE.0) RETURN）
alpha = 0.0：按 Netlib IF (DA.EQ.ZERO) RETURN 直接返回（y 不变且不读取 x）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSaxpy_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（saxpy_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasSaxpy` 声明**已存在**（L556，与 950PR 共用） | 无需新增声明，禁止私有平行接口（任务书 §2.3） |
| `blas/axpy/arch22/` | 复数 `aclblasCaxpy` 已有（host/kernel/tiling 三件套 + dispatch_policy）；**实数 Saxpy arch22 为本任务缺口**（上游 master 已合入同任务实现） | 实现落点与同族工程模式（任务书 §2.2）；复数兄弟的 dispatch 策略（步长/规模分派）与 tiling 框架沿用 |
| `blas/axpy/arch35/` | saxpy 950 版（ValidateSaxpyParams L20-160） | 参数校验、no-op 与 α=0 短路口径的实现参照 |
| `test/saxpy/` | CMakeLists 仅注册 arch35 | 新建 arch22 测试工程并补 `ops_blas_add_gtest_tests(${OPS_BLAS} saxpy_test)` 注册（任务书 §2.2） |

### 功能缺口与本任务差异点

**arch22 实数 Saxpy 为本任务交付主体**。相对兄弟算子 caxpy（arch22）的差异即主要工作量：

1. **实数路径**：无 deinterleave，FP32 平面 `y = α·x + y` 单遍流式读改写——
   设计重点为**带宽压榨**（x+y 读 + y 写三倍流量）与步长分派；
2. **BLAS-1 no-op/短路口径**：n ≤ 0 不访问 alpha、x、y（对齐 Netlib）；
   α=0 直接返回（y 不变、不读 x）——**注意与 11.25 批次「α=0 只做 y=βy」口径不同**，
   本任务为无 β 参数的纯 saxpy 语义；
3. **跨度溢出守卫**：(n−1)×|inc| > INT32_MAX 返回 `INVALID_VALUE`（host int64 计算，
   #780/#866 口径）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/axpy/arch22/` 实现 `aclblasSaxpy`：n 覆盖 {0,1,2,3,1024,16384,262144,1048576} ×
incx/incy ∈ {1,2,7,−1,−3}（正负步长）× alpha ∈ {0.0,1.0,2.5,−1.5,0.1,10.0} ×
Inf/NaN 特殊值与负向用例（含跨度溢出：n=3×incx=INT_MAX、n=2³⁰×inc=8）。

精度按生态开源精度标准 FLOAT32 档（rtol = atol = 2⁻¹³，matched_ratio ≥ 0.99，
max_abs ≤ 1e-2 或 32·ULP）；golden 为 Netlib BLAS `saxpy` CPU 参考实现
（float64 真值计算收窄 FLOAT32 落盘）。性能 msprof op 模式 task duration，
5 次 warmup + 有效采样 ≥10 次均值，逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `y = α·x + y`（原地覆写） | §2.1 | 平面流式读改写 kernel（§3.2.2） |
| R02 | n≤0 no-op（不访问 alpha/x/y）；α=0 直接返回（y 不变、不读 x，Netlib IF(DA.EQ.ZERO) RETURN） | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R03 | incx/incy = 0 或 INT32_MIN（n>0 时）→ `INVALID_VALUE`；(n−1)×\|inc\| 溢出守卫 | §2.4 | Host 步长守卫（§3.2.1） |
| R04 | 负步长反向遍历（物理 span 尾端起算） | §2.4 | X/YPhysPos 映射（§3.2.2） |
| R05 | 精度 FLOAT32 档 + Inf/NaN 等效 | §3.2 | 测试工程（§4.2） |
| R06 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |
| R07 | arch22 GTest 注册 + GAUSS_NORM_5_2 补充 | §2.2/§3.5 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
y[i] = α · x[i] + y[i]，i = 1..n（1 基）
x[i] 物理位置 1+(i−1)×|incx|（负步长自 span 尾端 (n−1)×|incx| 反向），y 同理
```

逐元素运算（无归约）：输出与输入一一对应，单次执行内确定、同输入同步长跨次
bit 级一致（任务书 §2.5 确定性口径——逐元素算子天然逐位确定）。

### 3.1.2 支持数据类型

x/y/alpha = FLOAT32；y 原地覆写；x 只读。
x/y 物理元素数 ≥ 1+(n−1)×|inc|。n、incx、incy 为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/axpy/arch22/
├── saxpy_host.cpp           # aclblasSaxpy 入口：校验链/标量解引用/tiling/发射
├── saxpy_kernel.cpp         # Ascend C AIV kernel（平面流式读改写）
├── saxpy_kernel.h
└── saxpy_tiling_data.h      # tiling 结构 + slab/尾块/步长映射参数
test/saxpy/arch22/
├── saxpy_test.cpp           # CSV 驱动 GTest（精度 VerifyMixed + 性能 PERF 路径）
├── saxpy_test.csv           # 官方 1000 精度 + 200 性能（+ GAUSS_NORM_5_2 补充）
├── gpu_baseline.csv         # 官方 H100 基线 200 条
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 声明已存在（L556），无需修改
test/saxpy/CMakeLists.txt    # arch22 分支注册 ops_blas_add_gtest_tests
```

### 3.2.1 Host 侧设计

1. **校验链**（对齐 arch35 `ValidateSaxpyParams`）：handle 判空
   （`HANDLE_IS_NULLPTR`）→ **n ≤ 0 → SUCCESS（不访问 alpha、x、y）** →
   alpha/x/y 指针判空（`INVALID_VALUE`）→ incx/incy = 0 或 INT32_MIN
   （`INVALID_VALUE`）→ (n−1)×|inc| 以 int64 计算超 INT32_MAX（`INVALID_VALUE`）。
2. **标量读取**：alpha 为 Host 内存 float 指针，Host 侧同步解引用（值不校验）。
3. **快速分支**：α=0.0 → 直接 SUCCESS（y 不变、不读 x，Netlib `IF (DA.EQ.ZERO) RETURN`）。
4. **tiling 组装**：步长分派（对齐仓内 dispatch_policy 风格）——
   incx=incy=1 连续路径 / 单侧步长路径 / 双侧步长路径；slab 均衡 K ≤ 40；
   尾块标量段参数；经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（平面流式读改写，本任务技术主体）

AIV 向量引擎（BLAS-1 纯带宽型，无 Cube 收益）：

- **连续路径（incx=incy=1）**：x/y 按 slab 流式对读（MTE2 双流 ping-pong）→
  向量 `y' = α·x + y`（Mul + Add 平面 pass）→ MTE3 连续写回——
  每 slab 读改写一次完成，x/y 各读一次、y 写一次（3×n×4B 流量）；
- **单侧步长路径**：步长侧按 Gather 索引（stride 元素跳过）平面化，连续侧
  保持 burst；负步长按 X/YPhysPos 反向映射（物理 span 尾端起算）；
- **双侧步长路径**：双 Gather 索引 + 平面 pass + Scatter 写回（性能用例以
  inc=1 为主，本路径为正确性保障路径）；
- **尾块**：非 32 对齐尾段标量处理；
- **多核均分**：slab 均衡切分 K ≤ 40 核，各核独立 ping-pong，无跨核依赖
  （无原子、无同步——逐元素算子天然免锁）；
- **UB 预算**：x/y slab ping-pong（双缓冲 ≤ 190KB 总预算）。

### 3.2.3 调用与执行流程

```text
aclblasSaxpy ─校验链→ 标量解引用 → n≤0/α=0 快速分支
  → 步长分派 tiling 组装 → saxpy_kernel<<<K, stream>>>
       └─ AIV: slab 循环{x/y ping-pong 对读 → α·x+y 平面 pass → MTE3 写回}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 大档 3×n×4B 流量逼近带宽（见 §3.2.5 反算） | 双流 ping-pong 对读 + MTE3 连续写；每元素恰读改写一次 |
| 步长路径带宽退化 | 分派策略（连续/单侧/双侧三路径）+ Gather 索引平面化；性能用例以 inc=1 为主 |
| 跨度溢出守卫 | host int64 闭式校验 + 溢出专项负向用例（n=3×incx=INT_MAX） |

### 3.2.5 性能设计反算（R06）

判据 = H100 gpu_ms × 1.25（msprof op task duration）。官方 gpu_baseline.csv 锚点：

| case | n | incx/incy | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- |
| saxpy-base-002 | 4194304 | 1/1 | 0.0348 | 43.5 |

大档 n=4.19M：流量 3×16.8MB=50.3MB@HBM 1.44TB/s → 底线 ~35us + 发射/init
开销 ≤ 4us → 预算 ~39us，对 43.5us 达标线留 ~10% 余量——大档为带宽贴线 case，
靠双流 ping-pong 与连续 MTE3 写压满带宽；小档（n ≤ 16K）以固定开销 ≤ 6us 为下限。
采集口径：msprof op 自带 5 warmup + ≥10 采样均值；spike 以 4.19M 连续档锚点
校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核独立 slab；x/y slab ping-pong 双缓冲（UB ≤ 190KB 总预算）；
无 workspace/GM 中转。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证仅需覆盖 910B3（任务书 §3.1）。

## 3.4 算子约束限制

n ≤ 0 为合法 no-op（不访问 alpha/x/y）；incx/incy ≠ 0 且非 INT32_MIN 且跨度
不溢出（n>0 时校验）；FLOAT32 专用；y 原地覆写（x 只读）；α=0 直接返回；
异步执行依赖 handle 绑定 stream（alpha Host 侧同步解引用后下发）；
同输入同步长跨次 bit 级一致（任务书 §2.5）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32·ULP）；
  Inf/NaN 等效规则（同 NaN/同号 Inf/NaN~±Inf 等效；非有限值单独统计）；
  golden = Netlib `saxpy` CPU 参考实现（float64 真值计算收窄 FLOAT32 落盘）。
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 saxpy_test.csv 全量接入（1000 精度 + 200 性能；n 全档、
   incx/incy 正负步长、alpha 特殊值（α=0 验证 y 不变不读 x）、Inf/NaN 传播；
   负向用例：nullptr handle/n>0 时空 alpha/x/y/空指针/零步长/INT32_MIN/
   **跨度溢出（n=3×incx=INT_MAX、n=2³⁰×inc=8）**）+ GAUSS_NORM_5_2 正态分布补充
   （μ∈[-5,5]、σ∈[0.1,2]，任务书 §3.5-正态补充要求）；
   **负向用例缓冲区分配规则**：期望非 SUCCESS 的用例以单元素 dummy/NULLPTR
   占位，不按 (n−1)×|inc| 预分配（任务书 §3.5 规则）；
2. **工程**：`test/saxpy/arch22/saxpy_test.cpp` CSV 驱动 GTest；
   `test/saxpy/CMakeLists.txt` 补 arch22 分支 `ops_blas_add_gtest_tests(${OPS_BLAS}
   saxpy_test)` 注册（任务书 §2.2）；编译入口 `bash build.sh --ops=saxpy
   --soc=ascend910b3`（A3 对应 --soc）；
3. **精度回归**：1000 条全量 VerifyMixed；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof 命令、golden 构建说明）。

## 4.3 兼容性分析

- 声明已存在（L556）与 950PR 共用，不改既有签名；
- arch22 与 caxpy/上游 saxpy 实现同目录共存（文件独立三件套）；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasSaxpy_AtlasA2_A3_task_doc.md（2026-11-11）
2. Netlib saxpy：https://netlib.org/blas/saxpy.f
3. cuBLAS cublasSaxpy：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
