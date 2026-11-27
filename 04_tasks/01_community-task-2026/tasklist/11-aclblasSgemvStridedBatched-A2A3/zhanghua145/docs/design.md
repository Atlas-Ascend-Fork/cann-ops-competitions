# aclblasSgemvStridedBatched 类算子（S/HSH/HSS/TST/TSS 五变体）设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md》（2026-11-25 发布，含 2026-11-24 修订：A 恒为 lda×n、lda ≥ max(1,m)，与 trans 无关）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemv_strided_batched/arch22/`，测试 `test/gemv_strided_batched/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「gemvStridedBatched类5个算子开发（A2/A3）」。在 ops-blas 仓为
Atlas A2/A3（arch22，性能测试设备 Atlas 800T A2 (910B3)）实现跨步批量矩阵-向量乘实数五变体：

```text
y + i*stridey = alpha · op(A + i*strideA) · (x + i*stridex) + beta · (y + i*stridey)
i = 0..batchCount-1；偏移以元素为单位（strideA/stridex/stridey 为 int64_t）
S   ：A/x/y FP32        （对标 cublasSgemvStridedBatched）
HSH ：A/x/y FP16        （对标 cublasHSHgemvStridedBatched）
HSS ：A/x FP16, y FP32  （对标 cublasHSSgemvStridedBatched）
TST ：A/x/y BF16        （对标 cublasTSTgemvStridedBatched）
TSS ：A/x BF16, y FP32  （对标 cublasTSSgemvStridedBatched）
alpha/beta 恒为 FLOAT32 Host 标量；计算在 FP32 完成后按输出 y 的 dtype 收窄
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md` | 接口签名（§2.3）、参数/校验语义（§2.4–§2.5，含 stride 合法性）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5）、修订记录（§9） |
| `test_cases/`（5 变体各 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 5 个变体声明**均已存在**（约 L311-337；FP16/BF16 以 `uint16_t` 承载；stride 用 `int64_t`） | 无需新增声明，直接按声明实现 arch22 版本（任务书 §2.3） |
| `blas/gemv_strided_batched/` | 仅 `arch35/`（950PR 版 5 变体已实现：`gemv_strided_batched_host.cpp` 等） | 本任务实现落点：**arch22 目录需新建** host/kernel（任务书 §2.2）；arch35 作 stride 偏移、变体分派与缩放逻辑参照（不得直接复制 SDK 不兼容代码） |
| `test/gemv_strided_batched/` | arch35 通用测试已有 | 复用/扩展通用 `gemv_strided_batched_test` 工程，注册 arch22 目标（任务书 §2.2） |
| `blas/gemv_batched/arch22/`（cgemv_batched）与 `blas/gemv/arch22/`（sgemv） | 同族 arch22 先例 | 指针/基址寻址、N/T 双路径与 tiling 框架参照 |

### 功能缺口与本任务差异点

**arch22 目录整体缺失（需新建）**。相对 arch35 版与 gemv_batched（指针数组版）的差异即主要工作量：

1. **stride 寻址**：batch i 的 A/x/y 由基址 + i×stride 定位（`int64_t` 元素偏移）——
   相比指针数组版，地址在 kernel 内闭式计算（无需读 Device 指针数组），发射更轻；
   但 stride 任意（带间隙）→ 逐 batch 地址不保证对齐，装载按对齐降级处理；
2. **stride 合法性校验**：`strideA/stridex/stridey > 0` 且相邻 batch 区域不重叠
   （`strideA ≥ lda×n` 最小间距由调用方保证，stride ≤ 0 或检测到重叠返回 `INVALID_VALUE`）；
3. **五变体 dtype 分派**：与 gemvBatched 任务同一约定（load/convert/store 三点分派、
   计算统一 FP32、按 y dtype 收窄）；
4. **A 布局口径（§9 修订）**：A[i] 恒为 `lda×n`、`lda ≥ max(1,m)`（与 trans 无关）；
   相邻 batch 的 A 区域最小间距为 `strideA ≥ lda·n`。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemv_strided_batched/arch22/` 新建 5 个变体入口：trans ∈ {N, T, C}（实数 C≡T）×
m/n 覆盖 0/1/…/4095/4096 × batchCount 覆盖 0/1/2/3/8/16/64/256（性能子集 1/8/64）×
lda（紧凑 + +1/+8/+64 padding）× incx/incy（±1/±2/±3 + 大步长 {7,8,16,31,64,127} 抽样）×
strideA/stridex/stridey（紧凑 + 带间隙；负向含 stride ≤ 0 与相邻 batch 重叠）×
alpha/beta 特殊值 × Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准：S/HSS/TSS 输出按 FLOAT32 档（2⁻¹³），HSH/TST 按 FLOAT16/BFLOAT16 档；
golden 由 Netlib `cblas_sgemv` 逐 batch（`op(A + i·strideA)`、`x + i·stridex`、`y + i·stridey`）
FP32 计算后按输出 y dtype 收窄。性能 msprof op 模式 task duration，
5 次 warmup + 10 次有效采样均值，逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | 五变体共享计算核、dtype 三点分派 | §2.1/§2.4 | 变体模板参数化 kernel（§3.2.2） |
| R02 | stride 寻址（int64 元素偏移、闭式计算） | §2.1/§2.3 | 基址 + i×stride 寻址（§3.2.2） |
| R03 | stride > 0 且不重叠；校验顺序 handle→trans→m/n→lda→incx/incy→stride→batchCount→标量→基址→quick return | §2.4/§2.5 | Host 校验链（§3.2.1） |
| R04 | batchCount=0/m=0/n=0 no-op；α=0 只做 `y=β·y`；β=0 不读 y | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R05 | A 恒 lda×n（lda ≥ max(1,m)，与 trans 无关） | §2.1/§9 | N 行 dot / T 列 dot 双路径（§3.2.2） |
| R06 | 负步长反向遍历语义 | §2.5 | X/YPhysPos 映射（§3.2.2） |
| R07 | 精度按输出 dtype 三档；golden 逐 batch cblas_sgemv | §3.2 | 测试工程（§4.2） |
| R08 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
y_i = alpha · op(A_i) · x_i + beta · y_i,   A_i = A + i·strideA, x_i = x + i·stridex, y_i = y + i·stridey
trans=N：op(A_i) = A_i（m×n），x_i 长 n，y_i 长 m：y_i(r) = α·Σ_j A_i(r,j)·x_i(j) + β·y_i(r)
trans=T/C：op(A_i) = A_i^T（n×m），x_i 长 m，y_i 长 n：y_i(c) = α·Σ_r A_i(r,c)·x_i(r) + β·y_i(c)
A_i 恒为 lda×n 列主序（lda ≥ max(1,m)）；相邻 batch 的 A 区域最小间距 strideA ≥ lda·n
```

实数 OP_C 与 OP_T 等价（不共轭）。batch 间由 stride 偏移定位、无数据依赖。

### 3.1.2 支持数据类型与形状

| 变体 | A | x | y | alpha/beta |
| --- | --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 | FLOAT32 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 |
| TST | BF16 | BF16 | BF16 | FLOAT32 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 |

`uint16_t` 仅承载 FP16/BF16 位模式；计算统一 FP32。
m/n/lda/incx/incy/strideA/stridex/stridey/batchCount 为运行时入参（stride 为 int64）。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemv_strided_batched/arch22/     # 新建目录（任务书 §2.2）
├── gemv_strided_batched_host.cpp     # 5 变体入口（校验链/标量/变体分派 tiling/发射）
├── gemv_strided_batched_kernel.cpp/.h  # 五变体 kernel（变体模板 + N/T 双路径 + stride 寻址）
└── （tiling 结构对齐 arch35 版与仓内族组织风格）
test/gemv_strided_batched/arch22/
├── gemv_strided_batched_test.cpp     # 复用/扩展通用 GTest（变体 × trans × stride 全覆盖）
├── （5 变体 csv：官方各 1000 精度 + 200 性能）
├── gpu_baseline.csv                   # 官方 H100 基线（5 变体各 200 条）
└── README.md                          # 复现步骤
include/cann_ops_blas.h                # 5 声明已存在（约 L311-337），无需修改
test/gemv_strided_batched/CMakeLists.txt  # 注册 arch22 目标
blas/gemv_strided_batched/README.md   # 补充 Atlas A2/A3 支持情况
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：handle 判空（`HANDLE_IS_NULLPTR`）→
   trans 非法（`INVALID_ENUM`）→ m/n < 0（`INVALID_VALUE`）→ lda < max(1,m)
   （`INVALID_VALUE`，与 trans 无关）→ incx/incy = 0（`INVALID_VALUE`）→
   strideA/stridex/stridey ≤ 0（`INVALID_VALUE`）→ 相邻 batch 区域重叠检测
   （strideA < lda·n 等最小间距 → `INVALID_VALUE`，§2.5 stride 合法性条款）→
   batchCount < 0（`INVALID_VALUE`）→ alpha/beta 指针判空 → 结构合法后 quick return：
   batchCount=0 或 m=0 或 n=0 → SUCCESS（A/x/y 可 nullptr、不启动 kernel）→
   A/x/y 基址判空（batchCount>0 且逻辑长度>0）。
2. **标量读取**：alpha/beta 为 FLOAT32 Host 标量，直读；α=0/β=0 短路标志下传 tiling。
3. **变体分派 tiling**：五变体 → 模板参数（load dtype / store dtype / 收窄方式）×
   trans 路径（N/T）× batch slab 与块参数；**stride 溢出防护**：末 batch 地址 =
   base + (batchCount−1)×stride + 跨度以 int64 计算，超 device 可寻址范围拒绝
   （`INVALID_VALUE`）；经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（变体模板 + N/T 双路径 + stride 寻址）

AIV 向量引擎（无 Cube 路径——批量 BLAS-2 访存受限）：

- **batch 维调度**：每核认领 batch slab（round-robin），batch i 的 A/x/y 地址
   **闭式计算**（基址 + i×stride，无 Device 指针数组读取，发射开销低于指针数组版）；
   逐 batch 地址对齐不保证（任意 stride）→ 装载按 32B 对齐段切分 + 非对齐首尾标量段；
- **N 路径（行 dot）**：x_i 全量驻留 UB（stridex 内段 gather），A_i 按行段块流读、
   向量 Mul + 树归约 per 行段、跨段累加器折叠；y_i 按步长 scatter 写回；
- **T 路径（列 dot）**：A_i 按列块流读（列主序天然连续），x_i 广播驻留，
   逐列 Mul + 树归约；y_i 段内连续写（incy=1 时 MTE3 burst）；
- **变体三点分派**：load（FP32 直入 / FP16·BF16 位模式 → FP32 转换平面）、
   compute（统一 FP32 FMA + 树归约）、store（FP32 直出 / FP16·BF16
   IEEE round-to-nearest 收窄；Inf/NaN 按 dtype 语义传播）；
- **负步长**：逻辑 i ↔ 物理 (len−1−i)×|inc| 反向映射（X/YPhysPos）；
   性能用例以单位步长为主，strided 走门控标量路径；
- **α=0 短路**：跳过 A/x 装载，仅 `y=β·y` 重写（β=0 不读 y 旧值；y 段仅逐 batch
   步进不读 A）；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、
   Set/Wait 相邻配对、id 7 不可用。

### 3.2.3 调用与执行流程

```text
aclblas{S,HSH,HSS,TST,TSS}gemvStridedBatched ─校验链（含 stride 合法性）→ 标量直读
  → quick-return 判定 → 变体×路径 tiling → kernel<<<grids, stream>>>
       └─ AIV: batch slab 循环{闭式 stride 寻址（对齐段切分）
                → N 行 dot / T 列 dot（ping-pong 流读）→ α/β 折叠 → 按 y dtype 收窄写回}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 任意 stride 导致地址非对齐装载降级 | 32B 对齐段切分 + 首尾标量段；紧凑 stride（=lda·n）为主档天然对齐 |
| int64 stride 溢出（batchCount × stride 超 2³¹） | Host 侧 int64 闭式校验末地址，超界拒绝（`INVALID_VALUE`） |
| stride 带间隙浪费带宽（性能子集紧凑为主） | 判定键以 stride 列显式保留（任务书 §3.5），间隙档以实际流读量核算预算 |
| FP16/BF16 收窄精度（HSH/TST 档） | FP32 统一计算 + 单点收窄；专项次正规/Inf/NaN 用例对拍 |
| 负步长/大步长性能退化 | 门控标量路径（性能子集以 inc=1 为主，无主档代价） |

### 3.2.5 性能设计反算（R08）

判据 = H100 gpu_ms × 1.25（kernel-to-kernel）。官方 gpu_baseline.csv 大档锚点：

| case | trans | m×n | batch | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- | --- |
| sgemv_strided_batched-base-003（S 族大档） | N | 4096² | 1 | ≈0.055 | ≈68.7 |
| hsh_gemv_strided_batched-base-003 | N | 4096² | 1 | 0.0301 | 37.6 |
| hss_gemv_strided_batched-base-003 | N | 4096² | 1 | 0.0295 | 36.9 |

- S 大档：A 67MB@HBM 1.44TB/s ≈ 46.5us 流读下限 + 固定开销 ≤ 8us → 预算 ~55us，
  对 68.7us 达标线留 ≥ 20% 余量（与 gemvBatched 同量级，且无指针数组读取开销）；
- FP16/BF16 变体：A 33.5MB → ~23us 下限，对 37us 达标线余量充足；
- 批量档（batch=8/64 小矩阵）stride 闭式寻址 + 单发射固定开销 ≤ 30us 目标；
- 采集口径：msprof op task duration，5 warmup + 10 采样均值；spike 以
  batch=1 大档 + batch=64 小档双锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核 UB：x 驻留（n ≤ 4096：FP32 16KB / FP16·BF16 8KB）+ A 段块
（ping-pong ≤ 190KB 总预算）+ y 段暂存 + 对齐段切分头；无 GM 中转 workspace。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

- strideA/stridex/stridey > 0 且相邻 batch 区域不重叠（最小间距 `strideA ≥ lda·n`
  等由调用方保证，接口对可判定项校验）；A/x/y 基址 Device 内存；
  x/y 支持非零跨步（含负步长）；A 按 lda 带填充访问（恒 lda×n），
  不使用 AutoContiguous/ViewCopy；
- y 原地覆写（各 batch y 区域互不重叠，调用方保证）；alpha/beta 恒为 FLOAT32 Host 标量；
  异步执行依赖 handle 绑定 stream；
- 确定性：浮点累加顺序不保证逐位一致（任务书 §2.5），按 §3.2 阈值判定。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：S/HSS/TSS 输出按 FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，
  max_abs ≤ 1e-2 或 32 ULP）；HSH/TST 输出分别按 FLOAT16/BFLOAT16 档
  （2⁻¹⁰ / 2⁻⁷，1e-1 或 32 ULP）；golden = Netlib `cblas_sgemv` 逐 batch
  （stride 定位 op(A_i)/x_i/y_i）FP32 计算后按输出 y dtype 收窄
  （float64 累加口径见任务书 §3.2 第 4 条，差异在自测报告说明）；
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 5000 精度（5 变体各 1000）+ 1000 性能（各 200）全量接入
   （变体全覆盖、trans N/T/C、m/n 全档、batchCount 全档（性能 1/8/64）、
   lda 紧凑与 padding、incx/incy ±1/±2/±3 + 大步长抽样、
   stride 紧凑 + 带间隙、alpha/beta 特殊值、FP16/BF16 含 Inf/NaN、
   负向用例（nullptr handle/非法枚举/负阶数/零步长/**stride ≤ 0**/**相邻 batch 重叠**/
   负 batchCount/空标量/空基址/α=0 叠 x=NaN/β=0 叠 y=NaN））；
2. **工程**：复用/扩展 `test/gemv_strided_batched/arch35/` 通用工程，
   在 `test/gemv_strided_batched/CMakeLists.txt` 注册 arch22 目标；编译入口
   `bash build.sh --ops=sgemv_strided_batched --soc=ascend910b3`（A3 对应 --soc）；
3. **精度回归**：5000 条全量（逐 batch 逐元素、按输出 dtype 分档统计）；
4. **性能回归**：1000 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof、golden 生成说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 5 声明已存在不改（约 L311-337），与其他产品线共用，禁止私有平行接口；
- arch22 目录新建与 arch35 版并存（互不影响，只读引用参考）；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasSgemvStridedBatched_AtlasA2_A3_task_doc.md（2026-11-25，含 §9 修订记录）
2. Netlib sgemv：https://www.netlib.org/blas/sgemv.f
3. cuBLAS cublasSgemvStridedBatched 等五变体：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
