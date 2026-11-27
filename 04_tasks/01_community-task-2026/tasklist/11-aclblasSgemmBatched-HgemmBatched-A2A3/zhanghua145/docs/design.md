# aclblasSgemmBatched / aclblasHgemmBatched 算子设计文档（Atlas A2/A3，arch22）

> 依据《aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md》（2026-11-25 发布）编写，按官方设计模板组织。
> 交付目标仓：https://gitcode.com/cann/ops-blas （实现 `blas/gemm_batched/arch22/`，测试 `test/gemm_batched/sgemm_batched/arch22/`）。

---

# 1. 需求背景（required）

## 1.1 需求来源

CANN 生态 11 月社区任务「aclblasSgemmBatched/aclblasHgemmBatched算子开发（A2/A3）」。在 ops-blas 仓
为 Atlas A2/A3（arch22，性能测试设备 Atlas 800T A2 (910B3)）实现批量通用矩阵乘双 dtype 入口：

```text
C[i] = alpha · op(A[i]) · op(B[i]) + beta · C[i],  i = 0..batchCount-1
aclblasSgemmBatched ：FLOAT32（对标 cublasSgemmBatched，语义参考 Netlib sgemm）
aclblasHgemmBatched ：FLOAT16/aclblasHalf（对标 cublasHgemmBatched）
Aarray/Barray/Carray 为 Device 侧指针数组（长度 batchCount）；各 batch 共享
transa/transb、m/n/k、lda/ldb/ldc、alpha、beta，数据与地址相互独立
```

| 材料 | 用途 |
| --- | --- |
| 任务书 `aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md` | 接口签名、参数/校验语义（§2.3–§2.5）、精度/性能门槛（§3.2–§3.5）、交付件要求（§4–§5） |
| `test_cases/`（sgemm_batched/hgemm_batched 各 500 精度 + 100 性能、gpu_baseline.csv、gen_csv.py、verify_*.py、README） | 官方用例、H100 基线、msprof op 采样口径 |
| ops-blas 仓 master | 工程框架与同族模板 |

## 1.2 背景介绍

### ops-blas 现状分析

| 对象 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `aclblasSgemmBatched`（L606）、`aclblasHgemmBatched`（L616）**声明均已存在** | 无需新增声明，直接按声明实现 arch22 版本（任务书 §2.3） |
| `blas/gemm_batched/arch22/` | 仅 `cgemm_batched_*`（复数实现：host/kernel/plan） | 本任务实现落点：与 cgemm_batched 同目录新增 S/H host 与 kernel（任务书 §2.2）；指针数组 staging 与 batch plan 的先例 |
| `blas/gemm_batched/arch35/` | `gemm_batched_host.cpp`（S）、`hgemmBatched_host.cpp`（H）已实现 | batch 调度、指针数组处理与 FP16 缩放路径的实现参照（不得直接复制 SDK 不兼容代码） |
| `blas/gemm/arch22/` | 实数 `aclblasSgemm`（Matmul API Cube 管线） | 单矩阵 Cube 管线为大矩阵 batch 复用底座 |
| `test/gemm_batched/sgemm_batched/` | 仅 arch35 注册；arch35 独立 H 测试见 `test/gemm/hgemmBatched/arch35/` | 测试工程复用通用 GTest，S/H 用例同一用例集按 `op` 列分派（任务书 §1/§2.2） |

### 功能缺口与本任务差异点

**arch22 S/H 双入口缺失**。相对 cgemm_batched（arch22）与 Sgemm（arch22）的差异即主要工作量：

1. **指针数组语义**：Aarray/Barray/Carray 本体及其元素均为 Device 内存——Host 侧只校验数组
   指针非空，不读取/不校验数组元素，各 batch 地址有效性与 C[i] 不重叠由调用方保证；
   kernel 侧按数组元素寻址（grid 组织直接以 batch 为主维度）；
2. **batch 维调度**：batch 间无数据依赖（共享形状与标量）→ 三级路径（巨型矩阵 batch /
   常规 batch / 微型 batch 1×1×1×65536，任务书 §3.5 batchCount 覆盖到 65536）；
3. **H 全 FP16 链路**：alpha/beta 为 `aclblasHalf`（与 S 的 float 不同）、Mmad FP16→FP32 累加、
   epilogue 收窄；FP16 次正规/接近上限/Inf/NaN 专项；
4. **实数 OP_C ≡ OP_T**（不共轭，两算子均为实数）。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/gemm_batched/arch22/` 实现 `aclblasSgemmBatched`、`aclblasHgemmBatched`：
transa/transb 全 9 组合 × m/n/k 覆盖 0/1/…/4095/4096（含立方档）× batchCount 覆盖
0/1/2/3/8/16/64/256/1024/4096/16384/65536（后 4 档配 1×1×1 小矩阵）×
lda/ldb/ldc（紧凑 + +1/+8/+64 padding）× alpha/beta 特殊值 0/1/-1/0.5/±1e4（H 用可表示 FP16 值）
× Inf/NaN 特殊值与负向用例。

精度按生态开源精度标准：S 按 FLOAT32 档、H 按 FLOAT16 档（2⁻¹⁰，max_abs ≤ 1e-1 或 32 ULP）；
S golden = Netlib `cblas_sgemm` 逐 batch；H golden = 测试工程 FP32（float64 累加）收窄 FP16。
性能 msprof op 模式 task duration，5 次 warmup + 10 次有效采样均值，
逐 case ≤ H100 gpu_ms × **1.25**（即 ≥ 0.8× GPU kernel 标杆）。

## 2.2 需求拆解

| # | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | 每 batch 独立 GEMM，共享形状/标量 | §2.1 | batch 维调度三级路径（§3.2.2） |
| R02 | 非法枚举 → `INVALID_ENUM`；校验顺序 handle→trans→m/n/k→ld→batchCount→标量→数组本体 | §2.4/§2.5 | Host 校验链（§3.2.1） |
| R03 | m=0/n=0/batchCount=0 no-op；k=0 或 α=0 只做 `C=β·C`；β=0 不读 C；仅写逻辑 m×n | §2.1/§2.5 | Host 快速分支 + epilogue 裁剪（§3.2.1） |
| R04 | 指针数组为 Device 内存，Host 不扫描元素 | §2.4/§2.5 | kernel 侧数组寻址（§3.2.2） |
| R05 | alpha/beta Host 内存（S=FLOAT32、H=FLOAT16） | §2.4 | Host 直读（§3.2.1） |
| R06 | 精度 S FLOAT32 / H FLOAT16 档；golden 双口径 | §3.2 | 测试工程（§4.2） |
| R07 | 性能逐 case ≤ 1.25×H100（msprof op，kernel-to-kernel） | §3.3 | 性能反算与预算（§3.2.5） |
| R08 | S/H 用例复用同一 GTest 按 op 列分派；CMakeLists 注册 arch22 | §1/§2.2 | 测试工程（§4.2） |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```text
C[i] = alpha · op(A[i]) · op(B[i]) + beta · C[i]
op(A[i])：transa=N 为 m×k（A[i] 为 lda×k），T/C 为 k×m（A[i] 为 lda×m）；实数 OP_C ≡ OP_T
op(B[i])：transb=N 为 k×n（B[i] 为 ldb×n），T/C 为 n×k（B[i] 为 ldb×k）
C[i]：ldc×n（逻辑 m×n）
```

每 batch 独立执行一次标准 GEMM；batch 间共享 tiling 形状 → tiling 计算一次、
按 batch 实例化发射。

### 3.1.2 支持数据类型

S：A/B/C/alpha/beta = FLOAT32；H：均为 FLOAT16（`aclblasHalf`，IEEE FP16 位模式）。
m/n/k/lda/ldb/ldc/batchCount 与 transa/transb 均为运行时入参。

## 3.2 算子实现

### 3.2.0 文件布局（对齐仓内族组织）

```text
blas/gemm_batched/arch22/
├── gemm_batched_host.cpp   # 新增 S/H 入口（校验链/标量读取/batch tiling/三级路径分派/发射）
│                            #  （亦可按仓内规范与 cgemm_batched_host.cpp 并列拆分文件）
├── gemm_batched_kernel.cpp/.h  # 新增 S/H kernel（batch 维调度 + Cube/AIV 路径）
└── （plan/tiling 结构对齐 cgemm_batched_plan.h 组织风格扩展）
test/gemm_batched/sgemm_batched/arch22/
├── sgemm_batched_test.cpp  # S/H 共用 GTest（op 列分派 S/H）
├── sgemm_batched_test.csv  # 1000 精度（S/H 各 500）+ 200 性能（各 100）
├── gpu_baseline.csv        # 官方 H100 基线（两算子各 100 条）
└── README.md               # 复现步骤
include/cann_ops_blas.h     # 声明已存在（L606/L616），无需修改
test/gemm_batched/sgemm_batched/CMakeLists.txt  # 注册 arch22 目标
blas/gemm_batched/README.md  # S/H 章节补充 Atlas A2/A3 支持情况
```

### 3.2.1 Host 侧设计

1. **校验链**（严格按任务书 §2.5 顺序）：handle 判空（`HANDLE_IS_NULLPTR`）→
   transa/transb 非法（`INVALID_ENUM`）→ m/n/k < 0（`INVALID_VALUE`）→
   lda/ldb/ldc 低于下界（`INVALID_VALUE`）→ batchCount < 0（`INVALID_VALUE`）→
   alpha/beta 指针判空 → 结构合法后 quick return：m=0 或 n=0 或 batchCount=0 →
   SUCCESS（三数组可为 nullptr、不启动 kernel）；k=0 或 α=0 → 只做 `C=β·C`
   （Aarray/Barray 可 nullptr，Carray 必须非空）→ Aarray/Barray 本体判空
   （batchCount>0 且 m>0 且 n>0 且 k>0 且 α≠0）→ Carray 本体判空（batchCount>0 且 m>0 且 n>0）。
   **Host 不读取、不校验数组元素**（Device 内存，R04）。
2. **标量读取**：alpha/beta 为 Host 内存（S float / H `aclblasHalf`），直读；
   α=0/β=0/β=1 短路标志下传 tiling。
3. **batch tiling 组装**：形状 tiling 计算一次（共享 m/n/k/ld/trans），扩展 batch 实例化
   参数（batch 序号 → 数组元素地址由 kernel 计算）；三级路径分派（§3.2.2）；
   经 kernel 直调发射（handle 绑定 stream）。

### 3.2.2 Kernel 侧设计（batch 维调度三级路径）

Ascend C Matmul API Cube 管线（对齐仓内 Sgemm arch22）+ 指针数组寻址：

- **大矩阵路径**（单 batch 矩阵 ≥ 512×512 量级，典型 batchCount ≤ 8）：
  逐 batch 复用单矩阵 Cube 管线（tile 网格 × K 轮次），batch 间顺序发射于同一 launch 的
  grid 外层循环；batch=1 时与 Sgemm 等价（性能锚点即单 GEMM H100 基线）；
- **常规 batch 路径**（中小矩阵 × 中大 batchCount）：batch 为主维度切分到多核——
  每 AIC 核认领整数个 batch 的完整 tile 网格（形状共享 → tiling 常量一次下发），
  batch 均衡按 `ceil(batchCount / coreGroups)` slab 划分；
  指针数组按核分段预取（AIV 辅助 block 把本核 batch 段的 A/B/C 指针 Gather 到 UB，
  避免 AIC 侧逐元素随机 GM 读）；
- **微型 batch 路径**（m=n=k=1 或 m·n·k ≤ 16 微形状 × batchCount ≥ 1024，
  覆盖 65536 档）：AIV 单发射 flat kernel——每工作项按数组元素取
  `A[i]/B[i]/C[i]` 标量/微 tile，`C[i] = α·A[i]·B[i] + β·C[i]` 元素级完成
  （1×1×1 即 65536 个独立标量乘加），无 Cube 参与；
- **epilogue**：α/β 缩放（H：`aclblasHalf` 位模式转 float 的 FP32 缩放后
  IEEE round-to-nearest 收窄 FP16；±65504 饱和/次正规/Inf/NaN 按 FP16 语义）、
  仅写各 C[i] 逻辑 m×n（ldc padding 列哨兵不写）；
- **K=0/α=0 短路**：全部 batch 跳过乘积，仅 epilogue β 缩放（β=0 不读 C 旧值）；
- **小形状路径**：m 或 n ≤ 16 的常规 batch 也走 AIV 单发射（向量 FMA）；
- **管线纪律**：事件 id 0/1 = MTE2→V ping-pong 奇偶、Set/Wait 相邻配对、
  id 7 不可用（910B 实测事件定律）。

### 3.2.3 调用与执行流程

```text
aclblas{SgemmBatched,HgemmBatched} ─校验链→ 标量直读 → quick-return 判定
  → 形状 tiling（一次）+ 三级路径分派 → kernel<<<grids, stream>>>
       ├─ 大矩阵：grid 外层 batch 循环 × Cube tile 网格
       ├─ 常规：核间 batch slab + 指针分段预取（AIV）+ Cube（AIC）
       └─ 微型：AIV flat（数组元素级乘加）
            └─ epilogue：α/β 缩放（H 收窄）+ 裁剪写回 C[i]
```

### 3.2.4 关键技术点与风险

| 风险 | 缓解 |
| --- | --- |
| 常规 batch 指针数组随机 GM 读放大 | AIV 辅助 block 分段 Gather 指针到 UB，AIC 只读 UB 段 |
| batch 不均（尾核空闲） | batch slab 均衡 + tile 级二次切分（尾核吃 tile 碎片） |
| 微型 65536 档 Host 侧数组预取不可行（Device 数组） | flat kernel 工作项直接 load 指针元素（每工作项 1 次 8B GM 读，65536×24B ≈ 1.5MB 总量，带宽可忽略） |
| H ±65504/次正规/Inf/NaN | epilogue 收窄单点承载 + 专项用例对拍 FP32 golden |
| 大矩阵 batch=1 与 Sgemm 重复实现 | 直接复用 gemm/arch22 的 Matmul 管线常量与 tiling 头（只读引用，不复制） |

### 3.2.5 性能设计反算（R07）

判据 = H100 gpu_ms × 1.25（kernel-to-kernel）。官方 gpu_baseline.csv 锚点：

| case | transA/B | m×n×k | batch | H100 gpu_ms | 达标线(ms) |
| --- | --- | --- | --- | --- | --- |
| sgemmBatched-perf-016 | N/N | 4096³ | 1 | 8.808 | 11.01 |
| hgemmBatched-perf-016 | N/N | 4096³ | 1 | 0.617 | 0.77 |

- batch=1 大档与单 GEMM 基线等价：S 复用 Sgemm 管线（4096³ H100 8.82ms 档已在
  既有实现覆盖）；H FP16 Mmad 原生吞吐（预算 < 1ms 量级）；
- 常规 batch 档按「batch 数 × 单 batch tile 网格」核间铺满（40 AIC/AIV 协同），
  指针预取与乘积计算重叠（ping-pong），H100 批量 kernel 对指针数组的串行寻址
  同样受限——NPU 侧 batch 主维天然并行，余量主要看小形状档 Cube 启动开销（≤ 30us）；
- 微型档 65536×1×1×1：总数据 ~1.5MB 指针 + 12KB 矩阵，AIV flat 一次发射 < 50us；
采集口径：msprof op task duration，5 warmup + 10 采样均值；
spike 阶段以 batchCount=1/64/65536 三锚点实测校准后回填自测报告。

### 3.2.6 UB 与调度预算

大矩阵/常规路径：AIC Cube grid × AIV pack/combine（对齐 Sgemm arch22 组织）；
常规路径每核 UB：本核 batch 段指针表（≤ 4KB）+ tile 平面（S FP32 或 H FP16，
双缓冲 ping-pong ≤ 190KB）；微型路径 AIV flat（每工作项寄存器级标量）；
无 GM 中转 workspace（除 Matmul API 内部）。

## 3.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（性能测试设备 Atlas 800T A2 (910B3)） | √ |
| Atlas A3 系列产品 | √ |

自验证只需覆盖一种款型（精度建议 910B3 或 A3，性能建议 910B3，任务书 §3.1）。

## 3.4 算子约束限制

- 三数组本体及元素均 Device 内存；各 batch 的 A/B/C 地址可独立，C[i] 之间及 C[i] 与
  A[j]/B[j] 不得重叠（调用方保证）；A/B/C 按 ld 带填充访问，不使用 AutoContiguous/ViewCopy；
- C[i] 原地覆写；仅写逻辑 m×n 区域；alpha/beta 指向 Host 内存；
  异步执行依赖 handle 绑定 stream；
- 确定性：batch 调度与 K 归约顺序由形状/片上容量决定（不按测试编号/种子分派），
  同一输入两次执行不保证逐位一致（任务书 §2.5）。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

- 精度：S 按 FLOAT32 档（rtol = atol = 2⁻¹³，ratio ≥ 0.99，max_abs ≤ 1e-2 或 32 ULP）；
  H 按 FLOAT16 档（2⁻¹⁰，1e-1 或 32 ULP）；
- golden：S = `cblas_sgemm` 逐 batch；H = 测试工程 FP32（float64 累加）后收窄 FP16
  （任务书 §3.2 口径，差异在自测报告说明）；
- 性能：msprof op task duration，5 warmup + ≥10 有效样本均值，逐 case ≤ 1.25×H100；
  GTest `[PERF] avg_us` 仅作交叉验证。

## 4.2 测试方案

1. **用例**：官方 1000 精度（S/H 各 500）+ 200 性能（各 100）全量接入
   （transa/transb 9 组合、m/n/k 全覆盖含立方档、batchCount 全档含 1024/4096/16384/65536
   小矩阵、lda/ldb/ldc 紧凑与 padding、alpha/beta 特殊值 0/1/-1/0.5/±1e4、
   H 输入接近 FP16 上限/次正规/Inf/NaN、负向用例（非法枚举/负阶数/负 batchCount/
   低前导维/空标量/空数组本体/α=0 叠 A/B=NaN/β=0 叠 C=NaN/k=0 且 β≠1/
   batchCount=0 不引用数组/ldc padding 哨兵写保护）；
2. **工程**：`test/gemm_batched/sgemm_batched/arch22/`（S/H 共用 GTest 按 op 列分派，
   任务书 §1）；`test/gemm_batched/sgemm_batched/CMakeLists.txt` 注册 arch22 目标；
   编译入口 `bash build.sh --ops=sgemm_batched,hgemm_batched --soc=ascend910b3`
   （A3 对应 --soc）；
3. **精度回归**：1000 条全量（逐 batch 逐元素统计）；
4. **性能回归**：200 条 PERF 逐条 msprof op 采集，输出 vs 判据余量表；
5. **README**：复现步骤（构建、gtest filter、msprof、golden 双口径构建说明、
   verify_accuracy.py / verify_performance.py 用法）。

## 4.3 兼容性分析

- 声明已存在不改（L606/L616），与其他产品线共用；
- arch22 与 cgemm_batched 同目录共存互不影响；arch35 参考实现只读引用；
- 不引入仓外依赖（S golden Netlib CBLAS 随测试工程提供，H golden 测试工程生成）。

# 5. 参考链接

1. 任务书：aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md（2026-11-25）
2. Netlib sgemm：https://www.netlib.org/blas/sgemm.f
3. cuBLAS cublasSgemmBatched / cublasHgemmBatched：https://docs.nvidia.com/cuda/cublas/index.html
4. 生态算子开源精度标准：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
5. ops-blas 仓：https://gitcode.com/cann/ops-blas
