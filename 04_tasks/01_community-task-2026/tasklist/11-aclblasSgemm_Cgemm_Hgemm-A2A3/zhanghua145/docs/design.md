# aclblasSgemm / aclblasCgemm / aclblasHgemm 类算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md》（2026-11-25 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemm/arch22/`，测试 `test/gemm/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「gemm类3个算子开发（A2/A3）」。在 ops-blas 仓为 Atlas A2/A3（arch22，
性能测试设备 Atlas 800T A2 (910B3)）补齐通用矩阵乘三 dtype 入口：

```text
C = alpha · op(A) · op(B) + beta · C        （列主序）
aclblasSgemm ：FLOAT32     （对标 cublasSgemm，语义参考 Netlib sgemm）
aclblasCgemm  ：COMPLEX64  （对标 cublasCgemm，语义参考 Netlib cgemm；直接复数乘加，不用 3M）
aclblasHgemm  ：FLOAT16    （对标 cublasHgemm；Netlib 无 half 参考实现）
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（gemm_test.csv 1000 精度 + 200 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例（3 算子分摊约 333/333/334）、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `blas/gemm/arch22/gemm_host.cpp` | **仅导出 `aclblasSgemm`**（Ascend C Matmul API，INNER_PRODUCT、ND、双缓冲） | 本任务实现落点：在同目录同 host 增加 `aclblasCgemm`、`aclblasHgemm` 入口（任务书 §2.2） |
| `include/cann_ops_blas.h` | `aclblasSgemm`（L575）、`aclblasCgemm`（L579）**已存在**；`aclblasHgemm`（非 batch）**不存在** | S/C 无需改声明；H 需按同族风格新增声明（建议紧随 Cgemm 之后，任务书 §2.3） |
| `blas/gemm/arch35/` | cgemm_host.cpp（Cgemm）、gemm_host.cpp（Sgemm）、hgemmStridedBatched_host/kernel（FP16 计算路径） | dtype 分派、复数/FP16 计算路径与缩放逻辑的实现参照（不得直接复制 SDK 不兼容代码） |
| `test/gemm/` | 通用 `gemm_test.cpp`/`gemm_test.csv`（arch35 已注册）；`test/gemm/cgemm/arch35/` | 本任务在既有通用测试工程上**扩展**用例（不新建 test/gemm/sgemm/arch22，任务书 §2.2 明确） |

### 功能缺口与本任务差异点

**arch22 的 Cgemm/Hgemm 入口缺失**（Sgemm 已有）。差异即主要工作量：

1. **Cgemm 直接复数路径**：`(a+bi)(c+di)=(ac−bd)+(ad+bc)i` 直接计算，**不采用 3M 分解**
   （与 `aclblasCgemm3m` 不是同一实现路径，禁止把 Cgemm 实现混入 3M 文件，任务书 §2.2）；
   OP_C 为共轭转置（虚部符号翻转）、OP_T 为非共轭转置，二者结果不同；
2. **Hgemm FP16 全链路**：alpha/beta 亦为 `aclblasHalf`，Matmul FP16 Mmad + FP32 累加，
   epilogue 按 FP16 收窄（含 ±65504 上限、次正规数、Inf/NaN 处理）；
3. **声明新增**：`aclblasHgemm` 头文件声明（共用 API，禁止私有平行接口）；
4. **测试工程扩展**：既有 gemm_test.cpp 按 dtype 列分派扩展，golden 三口径
   （S/C 走 cblas_sgemm/cblas_cgemm，H 走 FP32 float64 累加收窄 FP16）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemm/arch22/` 实现 `aclblasCgemm`、`aclblasHgemm` 两个新入口（`aclblasSgemm` 已有、
零改动）：transa/transb 全 9 组合（N/T/C）× m/n/k 覆盖 0/1/2/3/5/7/16…4095/4096（含立方档）×
lda/ldb/ldc（紧凑 + +1/+8/+64 padding）× alpha/beta（S float32 特殊值；C 复数特殊值
(0,0)/(1,0)/(0,1)/(-1,0)/±1e4；H 可表示 FP16 值含 ±65504）× Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准：S/C 按 FLOAT32 档（C 实部/虚部分量判定），H 按 FLOAT16 档
（rtol = atol = 2⁻¹⁰，matched_ratio ≥ 0.99，max_abs ≤ 1e-1 或 32 ULP）。
性能 msprof op 模式 task duration，5 次 warmup + 10 次有效采样均值，
逐 case ≤ H100 gpu_ms × **1.25**（即 ≥ 0.8× GPU kernel 标杆）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | `C = α·op(A)·op(B) + β·C` 三 dtype（S FP32 / C C64 / H FP16） | §2.1 | dtype 分派 Cube 管线（§3.2.2） |
| R02 | Cgemm 直接复数乘加（非 3M）；OP_C 共轭 ≠ OP_T；非法枚举 → `INVALID_ENUM` | §2.1/§2.3 | 四实积平面组织 + pack 符号翻转（§3.2.2） |
| R03 | Hgemm 新增声明（`aclblasHalf` 位模式类型） | §2.3 | `include/cann_ops_blas.h` 新增（§3.2.0） |
| R04 | m=0 或 n=0 no-op；k=0 或 α=0 只做 `C=β·C`；β=0 不读 C；仅写逻辑 m×n | §2.1/§2.5 | Host 快速分支 + epilogue 裁剪（§3.2.1） |
| R05 | alpha/beta 为 Host 内存（S/C/H 同 dtype 标量） | §2.4 | Host 直读（§3.2.1） |
| R06 | 精度：S/C FLOAT32 档、H FLOAT16 档；golden 三口径 | §3.2 | 测试工程（§4.2） |
| R07 | 性能逐 case ≤ 1.25×H100（msprof op，kernel-to-kernel） | §3.3 | 性能反算与预算（§3.2.5） |
| R08 | 复用/扩展通用 gemm_test 工程（不新建 sgemm 子目录） | §2.2 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

S/H（实数）：`C = α·op(A)·op(B) + β·C`，op(A) 为 m×k、op(B) 为 k×n；OP_C ≡ OP_T（不共轭）。

C（复数直接乘加）：

```text
T0 = op(Ar)·op(Br)   T1 = op(Ai)·op(Bi)   T2 = op(Ar)·op(Bi)   T3 = op(Ai)·op(Br)
Re(op(A)·op(B)) = T0 − T1        Im(op(A)·op(B)) = T2 + T3
Cr = αr·Re − αi·Im + βr·Cr_old − βi·Ci_old
Ci = αr·Im + αi·Re + βr·Ci_old + βi·Cr_old
```

op(A)=A^T（T）按转置视图装载；op(A)=A^H（C）装载时对 Ai 平面符号翻转（共轭）；
Br/Bi 同理按 transb 处理。四实积组织为两对共享 tile 的 Mmad 组（Ar/Ai tile 各装载一次，
配 Br/Bi tile 各一次 → 4 次 Mmad），**非 3M**（无 (Ar+Ai) 派生平面、无路径二选）。

### 3.1.2 支持数据类型

S：A/B/C/alpha/beta = FLOAT32；C：COMPLEX64（`aclblasComplex`）；H：FLOAT16（`aclblasHalf`）。
A：lda×k（N）或 lda×m（T/C）；B：ldb×n（N）或 ldb×k（T/C）；C：ldc×n。
m/n/k/lda/ldb/ldc 与 transa/transb 均为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemm/arch22/
├── gemm_host.cpp            # 已有 Sgemm 入口；新增 aclblasCgemm、aclblasHgemm 入口
│                            #   （校验链/标量读取/dtype 分派 tiling/发射）
├── gemm_kernel.cpp/.h       # 已有实数 Cube 管线；新增复数四实积与 FP16 kernel 路径
│                            #   （可按仓内规范拆分文件，个数对齐仓内同类算子）
include/cann_ops_blas.h      # 新增 aclblasHgemm 声明（紧随 aclblasCgemm，同族风格）
test/gemm/arch22/
├── gemm_test.cpp            # 既有通用 GTest 扩展（dtype 列分派 S/C/H）
├── gemm_test.csv            # 1000 精度（3 算子约 333/333/334）+ 200 性能（3 算子分摊）
├── gpu_baseline.csv         # 官方 H100 基线（sgemm/cgemm/hgemm 各档）
└── README.md                # 复现步骤
blas/gemm/README.md          # 三算子章节补充 Atlas A2/A3 支持情况
```

### 3.2.1 Host 侧设计

1. **校验链**（三入口共用，严格按任务书 §2.5 顺序）：
   handle 判空（`HANDLE_IS_NULLPTR`）→ transa/transb 非法（`INVALID_ENUM`）→
   m/n/k < 0（`INVALID_VALUE`）→ lda/ldb/ldc 低于下界（`INVALID_VALUE`）→
   alpha/beta 指针判空 → 结构合法后 quick return：m=0 或 n=0 → SUCCESS（A/B/C 可 nullptr）；
   k=0 或 α=0 → 只做 `C=β·C`（A/B 可 nullptr）→ A/B 指针判空（m>0 且 n>0 且 k>0 且 α≠0）→
   C 指针判空（m>0 且 n>0）。
2. **标量读取**：三算子 alpha/beta 均为 Host 内存（任务书 §2.4），按 dtype 直读
   （C 读实/虚部；H 读 `aclblasHalf` 位模式并转 float 参与 epilogue）；
   α=0/β=0/β=1 短路标志下传 tiling。
3. **dtype 分派 tiling**：复用 Sgemm arch22 Cube tiling 框架；C 扩展四实积平面参数
   （Ar/Ai/Br/Bi 平面 stride 与尾块裁剪），H 扩展 FP16 tile 参数（Mmad FP16→FP32 累加）。
   经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（dtype 分派 Cube 管线）

Ascend C **Matmul API**（对齐仓内 Sgemm arch22：INNER_PRODUCT、ND、双缓冲）：

- **Sgemm（已有）**：FP32 ND 路径（`DenseFp32`/CONFIG_NORM）或 FP16 hi/lo 补偿平面，
  零改动复用；
- **Cgemm（新）**：AIV pack 把 interleaved complex64 经 Gather 索引（byte offset 8m/8m+4）
  deinterleave 为 Ar/Ai、Br/Bi 平面（trans=T 转置视图装载、trans=C 虚部平面符号翻转）→
  AIC 四实积：Ar/Ai tile 各装载一次配 Br/Bi tile，同 K 轮次内连续 4 次 Mmad
  （两对共享 tile，摊薄装载）→ AIV epilogue：`Re=T0−T1、Im=T2+T3` 合成 +
  复数 α/β 交叉缩放（§3.1.1 全套代数）+ Interleave 写回（仅逻辑 m×n，ldc padding 不写）；
  精度平面：FP16 Mmad 叠 hi/lo 补偿（每积 3 Mmad）或 Matmul FP32 路径，spike 双路线对拍择优
  （FLOAT32 分量 2⁻¹³ 档为硬约束）；
- **Hgemm（新）**：FP16 tile 直入 Mmad（原生 dtype，无补偿需求——阈值 FLOAT16 档 2⁻¹⁰）、
  FP32 累加器；epilogue：α/β（位模式转 float）FP32 缩放后按 IEEE round-to-nearest 收窄 FP16
  写回；±65504 上限自然饱和、次正规数由 Mmad/收窄路径天然承载、Inf/NaN 按 FP16 语义传播
  （`α=0 且 A/B=NaN` 不读 A/B、`β=0 且 C=NaN` 不读 C 由短路保证）；
- **小形状路径**：m 或 n ≤ 16 时 AIV 单发射（向量 FMA + 树归约；C 为平面化复数 FMA），
  避免 Cube 启动开销；
- **K=0/α=0 短路**：跳过乘积，仅 epilogue β 缩放（β=0 不读 C 旧值）；
- **管线纪律**（910B 实测事件定律）：事件 id 0/1 = MTE2→V ping-pong 奇偶、Set/Wait 相邻配对、
  id 7 不可用、等待对不足补 PipeBarrier。

### 3.2.3 调用与执行流程

```text
aclblas{Cgemm,Hgemm} ─校验链→ 标量直读 → quick-return 判定 → dtype tiling 组装
  → kernel<<<grids, stream>>>
       ├─ AIV: pack（C：deinterleave+符号翻转；H：FP16 tile 装载）
       ├─ AIC+Matmul: 乘积（C：四实积两对共享 tile；H：FP16 Mmad FP32 累加）
       └─ AIV: epilogue（C：合成+交叉缩放+Interleave；H：FP32 缩放收窄 FP16；均裁剪写回）
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| FP16 Mmad 承载 C64 分量精度不足（2⁻¹³ 档） | hi/lo 补偿平面或 Matmul FP32 路径，spike 对拍 |
| Cgemm 四实积 = 4×Sgemm 工作量 vs H100 cgpu 基线（4096³ cgemm 29.1ms ≈ 3.3× sgemm 8.82ms） | H100 cgemm 自身非 4×（cuBLAS 复数路径亦平面化）；NPU 四实积两对共享 tile + Cube 效率 ≥ 0.85×Sgemm，4096³ 预算 ~4×8.82/0.85 ≈ 41ms > 达标线 36.4ms → spike 校准后必要时 K 维 split-K 与 tile 位宽 128 提密度 |
| Hgemm ±65504 大值与次正规数 | epilogue 收窄单点承载；专项用例（接近上限/次正规/Inf/NaN）对拍 FP32 golden |
| OP_C/OP_T 符号翻转遗漏 | pack 符号位单点维护 + 非对称复数矩阵专项（OP_C 结果 ≠ OP_T，golden 对拍） |
| H 声明位置/风格不合仓规范 | 紧随 `aclblasCgemm`（L579）同族签名；`aclblasHalf` 引用 `cann_ops_blas_common.h` |

### 3.2.5 性能设计反算（R07）

判据 = H100 gpu_ms × 1.25（kernel-to-kernel）。官方 gpu_baseline.csv 锚点：

| case | transA/B | m×n×k | H100 gpu_ms | 达标线(ms) |
| --- | --- | --- | --- | --- |
| sgemm-perf-023 | N/N | 4096³ | 8.820 | 11.03 |
| cgemm-perf-023 | N/N | 4096³ | 29.127 | 36.41 |
| hgemm-perf-023 | N/N | 4096³ | 0.617 | 0.77 |

- S：已有 Sgemm arch22 实现直接复用，4096³ 预算已在既有管线覆盖；
- C：四实积 Cube 效率是唯一杠杆（共享 tile/提密度/split-K），spike 以 4096³ 实测校准
  成本模型后回填；
- H：FP16 Mmad 原生吞吐（相对 FP32 ~2×），4096³ 预算 < 1ms 量级可行（0.77ms 上限），
  注意 4095³ 非对齐档（H100 反而更慢 1.68ms → 达标线 2.1ms，余量更宽）。
采集口径：msprof op task duration，5 warmup + 10 采样均值。

### 3.2.6 UB 与调度预算

AIC Cube grid × AIV pack/combine 协同；C 路径每 tile UB：Ar/Ai + Br/Bi 平面 +
四累加器（hi/lo 翻倍 → tile 位宽 64/128 自适应，UB ≤ 190KB）；H 路径 FP16 tile
密度加倍；无 GM 中转 workspace（除 Matmul API 内部）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

- 三 dtype 专用（本任务无 batchCount、无 dtype 枚举参数）；A/B/C 按 lda/ldb/ldc 列主序
  带填充访问，不支持任意非连续布局，不使用 AutoContiguous/ViewCopy；
- C 原地覆写，调用方保证 C 可写区与 A/B 可读区不重叠；仅写逻辑 m×n 区域；
- alpha/beta 指向 Host 内存；异步执行依赖 handle 绑定 stream；
- Cgemm 与 Cgemm3m 实现路径严格隔离（非 3M，禁止混入 3M 文件，任务书 §2.2 补充说明）；
- 确定性：多核切分与 K 归约顺序由形状与片上容量选择（不按测试编号/种子分派），
  同一输入两次执行不保证逐位一致（任务书 §2.5）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：S/C 按 FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP；
  C 实部/虚部分量分别判定）；H 按 FLOAT16 档（rtol = atol = 2⁻¹⁰，max_abs ≤ 1e-1 或 32 ULP）；
- golden：S = `cblas_sgemm`；C = `cblas_cgemm`（实/虚部分别比对）；H 由测试工程 FP32
  （float64 累加）后收窄 FP16（任务书 §3.2 第 1/4 条口径，差异在自测报告说明）；
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 gemm_test.csv 全量接入（1000 精度 3 算子约 333/333/334 + 200 性能 3 算子分摊；
   transa/transb 9 组合、m/n/k 全覆盖含立方档、lda/ldb/ldc 紧凑与 padding、
   alpha/beta 特殊值（C 复数 (0,0)/(1,0)/(0,1)/(-1,0)/±1e4；H 含 ±65504）、
   复数输入全零/纯实/纯虚/一般四类、Inf/NaN 特殊值、负向用例（非法枚举/负阶数/低前导维/
   空指针/α=0 叠 A/B=NaN/β=0 叠 C=NaN/k=0 且 β≠1/ldc padding 哨兵写保护/
   Cgemm OP_C 非对称矩阵 ≠ OP_T 专项）；
2. **工程**：`test/gemm/arch22/gemm_test.cpp` 既有通用工程按 dtype 列扩展（不新建
   test/gemm/sgemm/arch22，任务书 §2.2）；编译入口
   `bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`（A3 对应 --soc）；
3. **精度回归**：1000 条全量（C 实/虚部分别统计；H 逐元素 FP16 档）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof、golden 三口径构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- S/C 声明已存在不改；H 新增声明与其他产品线共用（禁止私有平行 API）；
- arch22 三入口与既有 Sgemm 实现同目录共存（Sgemm 路径零改动）；
- 与 `blas/gemm/arch35/` 参考实现互不依赖；不引入仓外依赖
  （golden Netlib CBLAS 随测试工程提供，H golden 由测试工程生成）。

# 5. 参考链接

1. 任务书：aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md（2026-11-25）
2. Netlib sgemm/cgemm：https://www.netlib.org/blas/sgemm.f 、https://www.netlib.org/blas/cgemm.f
3. cuBLAS cublasSgemm/Cgemm/Hgemm：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
