# aclblasSgemvBatched 类算子（S/HSH/HSS/TST/TSS 五变体）设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSgemvBatched_AtlasA2_A3_task_doc.md》（2026-11-25 发布，含 2026-11-24 修订：A 恒为 lda×n、lda ≥ max(1,m)，与 trans 无关）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemv_batched/arch22/`，测试 `test/gemv_batched/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「gemvBatched类5个算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3
（arch22，性能测试设备 Atlas 800T A2 (910B3)）实现批量矩阵-向量乘实数五变体：

```text
y_i = alpha · op(A_i) · x_i + beta · y_i,  i = 0..batchCount-1
S   ：A/x/y FP32          （对标 cublasSgemvBatched）
HSH ：A/x/y FP16          （对标 cublasHSHgemvBatched）
HSS ：A/x FP16, y FP32    （对标 cublasHSSgemvBatched）
TST ：A/x/y BF16          （对标 cublasTSTgemvBatched）
TSS ：A/x BF16, y FP32    （对标 cublasTSSgemvBatched）
alpha/beta 恒为 FLOAT32 Host 标量；计算在 FP32 完成后按输出 y 的 dtype 收窄
Aarray/xarray/yarray 为 Device 侧指针数组（长度 batchCount）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSgemvBatched_AtlasA2_A3_task_doc.md` | 接口签名（§2.3）、参数/校验语义（§2.4–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5）、修订记录（§9 布局口径） |
| `test_cases/`（5 变体各 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 5 个变体声明**均已存在**（约 L287-308；FP16/BF16 以 `uint16_t` 承载） | 无需新增声明，直接按声明实现 arch22 版本（任务书 §2.3） |
| `blas/gemv_batched/arch22/` | 仅 `cgemv_batched_*`（复数实现：host/kernel/plan，含 `CalMaxMatNum` 多矩阵驻留 UB 的组批技术） | 本任务实现落点：新增 5 个实数变体入口与 kernel（任务书 §2.2）；指针数组处理与多矩阵组批先例 |
| `blas/gemv_batched/arch35/gemv_batched_host.cpp` | 950PR 版已实现 5 个变体 | 变体分派、tiling 与缩放逻辑参照（不得直接复制 SDK 不兼容代码） |
| `test/gemv_batched/` | arch35 通用测试 + `cgemv_batched/arch22/` 复数版组织 | 测试工程复用/扩展通用 `gemv_batched_test`，注册 arch22 目标（任务书 §2.2） |

### 功能缺口与本任务差异点

**arch22 实数五变体缺失**。相对 cgemv_batched（arch22）的差异即主要工作量：

1. **五变体 dtype 分派**：S（全 FP32）与 HSH/HSS（FP16）、TST/TSS（BF16）输入平面
   装载后统一 FP32 计算、按 y dtype 收窄（HSH/TST 收窄、HSS/TSS 直出 FP32）——
   变体差异集中在 load/convert/store 三点，计算核共享；
2. **A 布局口径（§9 修订）**：A[i] 恒为 `lda×n`、`lda ≥ max(1,m)`（与 trans 无关），
   trans 只作用于 op(A[i])；转置读按行方向 dot 组织；
3. **bandwidth-bound 纯 AIV 设计**：BLAS-2 批量形态访存受限，无 Cube 收益；
   指针数组 + 每矩阵 m×n 流读 → 多矩阵驻留 UB 组批（沿 `CalMaxMatNum` 思路）摊薄发射开销；
4. **步长语义**：incx/incy 非零跨步（含负步长反向遍历），零步长负向。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemv_batched/arch22/` 实现 5 个变体入口：trans ∈ {N, T, C}（实数 C≡T）×
m/n 覆盖 0/1/…/4095/4096 × batchCount 覆盖 0/1/2/3/8/16/64/256（性能子集覆盖 1/8/64）×
lda（紧凑 + +1/+8/+64 padding）× incx/incy（±1/±2/±3 + 大步长 {7,8,16,31,64,127} 抽样；
性能子集以单位步长为主）× alpha/beta 特殊值 0/1/-1/±1e4 + 随机值 × Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准：S/HSS/TSS 输出按 FLOAT32 档（2⁻¹³，1e-2 或 32 ULP），
HSH/TST 输出分别按 FLOAT16/BFLOAT16 档（2⁻¹⁰ / 2⁻⁷）；golden 由 Netlib `cblas_sgemv`
逐 batch FP32 计算后按输出 y dtype 收窄。性能 msprof op 模式 task duration，
5 次 warmup + 10 次有效采样均值，逐 case ≤ H100 gpu_ms × **1.25**（≥ 0.8× GPU）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | 五变体共享计算核、dtype 三点分派（load/convert/store） | §2.1/§2.4 | 变体模板参数化 kernel（§3.2.2） |
| R02 | A 恒为 lda×n（lda ≥ max(1,m)，与 trans 无关）；op 转置只作用于逻辑 | §2.1/§9 | N 列核 / T 行 dot 双路径（§3.2.2） |
| R03 | 校验顺序 handle→trans→m/n→lda→incx/incy→batchCount→标量→数组→quick return；零步长 → `INVALID_VALUE` | §2.5 | Host 校验链（§3.2.1） |
| R04 | batchCount=0/m=0/n=0 no-op；α=0 只做 `y=β·y`；β=0 不读 y | §2.1/§2.5 | Host 快速分支（§3.2.1） |
| R05 | 负步长反向遍历语义 | §2.1/§2.5 | X/YPhysPos 映射（§3.2.2） |
| R06 | 指针数组 Device 内存，Host 只判数组本体非空 | §2.4/§2.5 | kernel 侧数组寻址（§3.2.2） |
| R07 | 精度按输出 dtype 三档；golden 逐 batch cblas_sgemm | §3.2 | 测试工程（§4.2） |
| R08 | 性能逐 case ≤ 1.25×H100（msprof op） | §3.3 | 性能反算与预算（§3.2.5） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
y_i = alpha · op(A_i) · x_i + beta · y_i
trans=N：op(A_i) = A_i（m×n），x_i 长 n，y_i 长 m：y(r) = α·Σ_j A(r,j)·x(j) + β·y(r)
trans=T/C：op(A_i) = A_i^T（n×m），x_i 长 m，y_i 长 n：y(c) = α·Σ_r A(r,c)·x(r) + β·y(c)
A_i 恒为 lda×n 列主序（lda ≥ max(1,m)，与 trans 无关——任务书 §9 修订口径）
```

N 路径按行组织（每输出元素一行 dot），T 路径按列组织（每输出元素一列 dot，
列连续读对带宽友好）；实数 OP_C 与 OP_T 等价。

### 3.1.2 支持数据类型与形状

| 变体 | A | x | y | alpha/beta |
| --- | --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 | FLOAT32 |
| HSH | FP16 | FP16 | FP16 | FLOAT32 |
| HSS | FP16 | FP16 | FP32 | FLOAT32 |
| TST | BF16 | BF16 | BF16 | FLOAT32 |
| TSS | BF16 | BF16 | FP32 | FLOAT32 |

`uint16_t` 仅承载 FP16/BF16 位模式；计算统一 FP32。
m/n/lda/incx/incy/batchCount 为运行时入参；x/y 物理元素数 ≥ 1+(len−1)·|inc|。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemv_batched/arch22/
├── gemv_batched_host.cpp    # 新增 5 变体入口（校验链/标量/变体分派 tiling/发射）
├── gemv_batched_kernel.cpp/.h  # 新增实数五变体 kernel（变体模板 + N/T 双路径 + 组批）
└── （plan/tiling 结构对齐 cgemv_batched_plan.h 组织风格扩展）
test/gemv_batched/arch22/
├── gemv_batched_test.cpp    # 复用/扩展通用 GTest（变体 × trans × batch 全覆盖）
├── （5 变体 csv：官方各 1000 精度 + 200 性能）
├── gpu_baseline.csv         # 官方 H100 基线（5 变体各 200 条）
└── README.md                # 复现步骤
include/cann_ops_blas.h      # 5 声明已存在（约 L287-308），无需修改
test/gemv_batched/CMakeLists.txt  # 注册 arch22 目标
blas/gemv_batched/README.md  # 补充 Atlas A2/A3 支持情况
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：handle 判空（`HANDLE_IS_NULLPTR`）→
   trans 非法（`INVALID_ENUM`）→ m/n < 0（`INVALID_VALUE`）→ lda < max(1,m)
   （`INVALID_VALUE`，与 trans 无关）→ incx/incy = 0（`INVALID_VALUE`）→
   batchCount < 0（`INVALID_VALUE`）→ alpha/beta 指针判空 → 结构合法后 quick return：
   batchCount=0 或 m=0 或 n=0 → SUCCESS（三数组可 nullptr、不启动 kernel）→
   Aarray/xarray/yarray 本体判空（batchCount>0 且逻辑长度>0）。
   Host 不读取/校验数组元素（Device 内存，R06）。
2. **标量读取**：alpha/beta 为 FLOAT32 Host 标量（五变体一致），直读；
   α=0/β=0 短路标志下传 tiling。
3. **变体分派 tiling**：五变体 → 模板参数（load dtype / store dtype / 收窄方式）×
   trans 路径（N/T）× 组批参数（按 `CalMaxMatNum` 思路计算单核可驻留 UB 的矩阵数，
   FP16/BF16 变体密度加倍）；经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（变体模板 + N/T 双路径 + 多矩阵组批，技术主体）

AIV 向量引擎（无 Cube 路径——批量 BLAS-2 访存受限，Cube 无收益）：

- **batch 维调度**：每核认领 batch slab（round-robin），核内按组批块推进——
   每块从本核数组段 Gather `A[i]/x[i]/y[i]` 指针（指针表 UB 驻留 ≤ 4KB），
   块内多矩阵驻留 UB（沿 `CalMaxMatNum` 思路，一次装载多个小矩阵摊薄发射开销）；
- **N 路径（行 dot）**：y(r) = α·Σ_j A(r,j)·x(j) + β·y(r)——x 全量驻留 UB（n ≤ 4096 时
  FP32 16KB），A 按行段块流读（行连续 GM burst），向量 Mul + 树归约 per 行段，
  跨段累加器折叠；y 按步长 scatter 写回；
- **T 路径（列 dot）**：y(c) = α·Σ_r A(r,c)·x(r) + β·y(c)——A 按列块流读
  （列主序天然连续），x 广播驻留，逐列 Mul + 树归约；y 按列连续写回
  （incy=1 时 MTE3 连续 burst）；
- **变体三点分派**：load（FP32 直入 / FP16·BF16 位模式 → FP32 转换平面），
  compute（统一 FP32 FMA + 树归约），store（FP32 直出 / FP16·BF16
  IEEE round-to-nearest 收窄；Inf/NaN 按 dtype 语义传播）；
- **负步长**：逻辑 i ↔ 物理 (len−1−i)×|inc| 反向映射（X/YPhysPos）；
  性能用例以单位步长为主（任务书 §3.5），strided 走标量 gather/scatter 门控路径不影响主档；
- **α=0 短路**：跳过 A/x 装载，仅 `y=β·y` 重写（β=0 不读 y 旧值）；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、
  Set/Wait 相邻配对、id 7 不可用。

### 3.2.3 调用与执行流程

```text
aclblas{S,HSH,HSS,TST,TSS}gemvBatched ─校验链→ 标量直读 → quick-return 判定
  → 变体×路径×组批 tiling → kernel<<<grids, stream>>>
       └─ AIV: 指针段 Gather → 组批块循环{多矩阵驻留装载(ping-pong)
                → N 行 dot / T 列 dot → α/β 折叠 → 按 y dtype 收窄写回}
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 小矩阵 × 大 batch 发射开销占比高 | 多矩阵驻留 UB 组批（CalMaxMatNum 思路，FP16/BF16 密度加倍） |
| N 路径行读不连续（lda padding 行段） | 行段块流读 + 行内连续 burst；padding 档（+1/+8/+64）实测校准 |
| FP16/BF16 收窄精度（HSH/TST 2⁻¹⁰/2⁻⁷ 档） | FP32 统一计算 + 单点收窄；专项次正规/Inf/NaN 用例对拍 |
| 负步长/大步长性能退化 | 门控标量路径（性能子集以 inc=1 为主，无主档代价） |
| 修订口径（A 恒 lda×n）与旧 golden 错位 | 按任务书 §9：测试包已按新口径重新生成，T/C 行 `lda` 判定键以新口径对齐 |

### 3.2.5 性能设计反算（R08）

判据 = H100 gpu_ms × 1.25（kernel-to-kernel）。官方 gpu_baseline.csv 大档锚点：

| case | trans | m×n | batch | H100 gpu_ms | 达标线(us) |
| --- | --- | --- | --- | --- | --- |
| sgemv_batched-base-003 | N | 4096² | 1 | 0.0550 | 68.7 |
| hsh_gemv_batched-base-003 | N | 4096² | 1 | 0.0330 | 41.2 |
| tst_gemv_batched-base-003 | N | 4096² | 1 | 0.0329 | 41.1 |

- S 大档：A 67MB@HBM 1.44TB/s ≈ 46.5us 流读下限，加组批/折叠开销 ≤ 8us →
  预算 ~55us 对 68.7us 达标线留 ≥ 20% 余量；
- FP16/BF16 变体：A 33.5MB → ~23us 下限，对 41us 达标线余量充足；
- 批量档（batch=8/64 小矩阵）按 H100 批量 kernel 同样受发射/寻址开销限制
  （GPU 小档耗时主要非带宽），NPU 侧组批驻留 + 单发射的固定开销 ≤ 30us 目标；
- 采集口径：msprof op task duration，5 warmup + 10 采样均值；spike 以
  batch=1 大档 + batch=64 小档双锚点校准后回填自测报告。

### 3.2.6 UB 与调度预算

K ≤ 40 AIV；每核 UB：指针表（≤ 4KB）+ x 驻留（n ≤ 4096：FP32 16KB / FP16·BF16 8KB）+
组批矩阵块（CalMaxMatNum 自适应，ping-pong ≤ 190KB 总预算）+ y 段暂存；
无 GM 中转 workspace。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

- 三数组本体及元素均 Device 内存；各 batch 的 y[i] 之间不重叠（调用方保证）；
  x/y 支持非零跨步（含负步长反向遍历）；A 按 lda 带填充访问（A 恒 lda×n），
  不使用 AutoContiguous/ViewCopy；
- y[i] 原地覆写；alpha/beta 恒为 FLOAT32 Host 标量；异步执行依赖 handle 绑定 stream；
- 确定性：浮点累加顺序不保证逐位一致（任务书 §2.5），按 §3.2 阈值判定。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：S/HSS/TSS 输出按 FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，
  max_abs ≤ 1e-2 或 32 ULP）；HSH/TST 输出分别按 FLOAT16/BFLOAT16 档
  （2⁻¹⁰ / 2⁻⁷，1e-1 或 32 ULP）；golden = Netlib `cblas_sgemv` 逐 batch
  FP32 计算后按输出 y dtype 收窄（float64 累加口径见任务书 §3.2 第 4 条，
  差异在自测报告说明）；
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 5000 精度（5 变体各 1000）+ 1000 性能（各 200）全量接入
   （变体 S/HSH/HSS/TST/TSS 全覆盖、trans N/T/C、m/n 全档、batchCount 全档
   （性能子集 1/8/64）、lda 紧凑与 padding、incx/incy ±1/±2/±3 + 大步长抽样
   （性能以单位步长为主）、alpha/beta 特殊值、FP16/BF16 输入按 dtype 生成含
   Inf/NaN 特殊值、负向用例（nullptr handle/非法枚举/负阶数/零步长/负 batchCount/
   空标量/空数组本体/α=0 叠 x=NaN/β=0 叠 y=NaN/y[i] 重叠负向）；
2. **工程**：复用/扩展 `test/gemv_batched/arch35/` 通用 `gemv_batched_test` 工程，
   在 `test/gemv_batched/CMakeLists.txt` 注册 arch22 目标；编译入口
   `bash build.sh --ops=sgemv_batched --soc=ascend910b3`（A3 对应 --soc）；
3. **精度回归**：5000 条全量（逐 batch 逐元素、按输出 dtype 分档统计）；
4. **性能回归**：1000 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof、golden 生成说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 5 声明已存在不改（约 L287-308），与其他产品线共用，禁止私有平行接口；
- arch22 与 cgemv_batched 同目录共存互不影响；arch35 参考实现只读引用；
- 不引入仓外依赖（golden Netlib CBLAS 随测试工程提供）。

# 5. 参考链接

1. 任务书：aclblasSgemvBatched_AtlasA2_A3_task_doc.md（2026-11-25，含 §9 修订记录）
2. Netlib sgemv：https://www.netlib.org/blas/sgemv.f
3. cuBLAS cublasSgemvBatched 等五变体：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
