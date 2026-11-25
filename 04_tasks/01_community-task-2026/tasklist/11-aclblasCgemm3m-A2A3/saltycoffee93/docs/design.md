# aclblasCgemm3m 算子设计文档

> 目标芯片：Atlas A2/A3（arch22，自验证 SOC=ascend910b3）｜CANN 9.1.0｜合入位置 `ops-blas/blas/gemm/arch22/`
> 任务书：`aclblasCgemm3m_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— aclblasCgemm3m 算子开发（A2A3）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/gemm/arch22/`；测试 `test/gemm/cgemm3m/arch22/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

`aclblasCgemm3m` 实现单精度复数（complex64）通用矩阵乘 `C = alpha * op(A) * op(B) + beta * C`（列主序），用 Gauss 3M 算法把复数矩阵乘分解为 3 次实数矩阵乘。

| 接口 | 对标 | 作用 |
|---|---|---|
| aclblasCgemm3m | cuBLAS `cublasCgemm3m`｜Netlib `cgemm.f`（数学语义） | 3M 复数通用矩阵乘 |

与本仓另外两个同族接口的边界：与 `aclblasCgemm`（直接复数乘加，等价 4 次实数乘积）参数序列相同但实现语义不同，3M 少一次实数 GEMM、舍入过程不同，与 `cblas_cgemm` 不保证逐位一致（§2.1、§3.2）；与 `blas/gemm3m/` 的实数 `aclblasSgemm3m`（3 个实数矩阵沿 K 拼接的接口）不是同一接口族，**禁止把实现放入 `blas/gemm3m/`**（§2.2）。

### 业界对标

| 项 | cuBLAS cublasCgemm3m | 本算子 |
|---|---|---|
| 参数序列 | 与 `cublasCgemm` 相同的 13 参数 | 与 `aclblasCgemm` 逐参数一致（§2.3、§3.5） |
| trans 取值 | N / T / C，C 为共轭转置、与 T 结果不同 | 同 |
| alpha/beta 位置 | 随 pointer mode 可为 Host 或 Device | 约定指向 Host 内存（§2.4） |
| 非法 trans 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |
| 数值口径 | 3M 舍入 | 同为 3M 舍入，按 §3.2 阈值判定，不与 4M 逐位对齐 |

### 工程形态

- 工程模式：Kernel 直调——Host C API 完成参数校验、分派与 tiling，经 handle 绑定的 stream 直调 NPU kernel（§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 内容 | 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| arch22 实现 | 不存在（`blas/gemm/arch22/` 仅有 `gemm_*`（Sgemm）与 `cgemm_*`（Cgemm）） | 目录清单核验 | **新增** `cgemm3m_host.cpp` 等 |
| arch35 参考 | 已有整套 | `blas/gemm/arch35/cgemm3m_host.cpp`（24 行薄壳，转调 `LaunchComplexGemm(..., forceThreeProducts=true)`）；`cgemm3m_rounding_kernel.cpp`（499 行，SIMT 舍入风险筛查与纠正） | 参考 3M 路径选择与舍入处理思想 |
| 公开声明 | 已存在，无需修改 | `include/cann_ops_blas.h` L584 | 零公共头改动 |
| 测试工程 | `test/gemm/cgemm3m/` 仅 arch35（CMakeLists 明确 arch35 才注册目标） | `test/gemm/cgemm3m/CMakeLists.txt` | 新增 arch22 目录并注册 arch22 目标 |
| README | `aclblasCgemm3m` 章节写"支持 Atlas 950PR / arch35" | `blas/gemm/README.md` L956-958 | 补充 Atlas A2/A3 支持情况 |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemm/arch22/cgemm_*`（已合入的 arch22 Cgemm） | ★ 主模板 | 三阶段骨架：AIV 拆实虚（`cgemm_split_kernel_do`）→ 实数 GEMM 委托 `aclblasSgemm`（子句柄 + K≤1024 分块）→ AIV 复数组装（`cgemm_combine_kernel_do`）；平面累加 kernel `cgemm_accum_kernel_do`（支持 `dst = scale*src` 与 `dst += scale*src`）；标量读回 `CBlas3ReadScalarsFromDevice` | "4 次实数乘积"换成"3M 的 3 次实数乘积 + 平面预加"，乘积组装次序不同 |
| `blas/gemm/arch22/gemm_*`（arch22 Sgemm） | ★（被委托） | 3 次实数 GEMM 的执行体 | 不改动（其 K>1024 缺陷见 host 侧设计） |
| `blas/gemm/arch35/cgemm3m_*` | ◇ 参考 | 3M 路径选择、舍入风险筛查/纠正思想（`cgemm3m_rounding_kernel.cpp` 的 variance 筛选启发式） | arch35 为共享发射器 + SIMT 纠正 kernel；arch22 只借鉴思想，不复制代码 |
| `common/helper/complex_blas3_host_utils.h` | 共享基建 | 标量读回、workspace 字节数估算 | 直接调用 |

### 接口规格

声明已存在于 `include/cann_ops_blas.h` L584（与 `aclblasCgemm` 参数序列一致），本任务**无需新增或修改任何声明**（§2.3）。签名：

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

`aclblasComplex` 为实部/虚部各 float32 的结构体（`include/cann_ops_blas_common.h` L77）。

参数规格（逐行对齐 §2.4）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| transa | op(A) 运算类型 | attr | int | N / T / C；其它 → `ACLBLAS_STATUS_INVALID_ENUM` | — |
| transb | op(B) 运算类型 | attr | int | 同上 | — |
| m | op(A) 与 C 的行数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |
| n | op(B) 与 C 的列数 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |
| k | 缩并维长度 | attr | int | ≥0；负 → `INVALID_VALUE`；0 只做 `C = beta*C` | — |
| alpha | 复数标量乘数（Host 内存） | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE` | 标量 |
| A | 输入矩阵 A（列主序，Device） | tensor | COMPLEX64 | m、n、k>0 且 alpha≠0 时非空，空 → `INVALID_VALUE` | 物理 lda×k（N）或 lda×m（T/C） |
| lda | A 前导维 | attr | int | N：≥max(1,m)；T/C：≥max(1,k)；否则 `INVALID_VALUE` | 标量 |
| B | 输入矩阵 B（列主序，Device） | tensor | COMPLEX64 | 非空条件同 A | 物理 ldb×n（N）或 ldb×k（T/C） |
| ldb | B 前导维 | attr | int | N：≥max(1,k)；T/C：≥max(1,n)；否则 `INVALID_VALUE` | 标量 |
| beta | 复数标量乘数（Host 内存） | scalar | COMPLEX64 | 非空；空 → `INVALID_VALUE`；=0 时不读 C 旧值 | 标量 |
| C | 输出矩阵 C（列主序，Device，原地覆写） | tensor | COMPLEX64 | m>0 且 n>0 时非空，空 → `INVALID_VALUE` | 逻辑 m×n；物理 ldc×n |
| ldc | C 前导维 | attr | int | ≥max(1,m)；否则 `INVALID_VALUE`；padding 列不被修改 | 标量 |

### 数学公式

任务书 §2.1 给出的两条 3M 分解（记 Ar、Ai、Br、Bi 为 A、B 的实部/虚部平面，op 为 N/T/C 取向）：

```text
非融合路径： P0 = op(Ar)·op(Br)        P1 = op(Ai)·op(Bi)        P2 = op(Ar+Ai)·op(Br+Bi)
             op(A)·op(B) = (P0 − P1) + i·(P2 − P0 − P1)

融合路径（含纠正重算）：
             P0 = op(Ar+Ai)·op(Br)     P1 = op(Ar)·op(Bi−Br)     P2 = op(Ai)·op(Br+Bi)
             op(A)·op(B) = (P0 − P2) + i·(P0 + P1)

随后：Cr = αr·Re − αi·Im + βr·Cr_old − βi·Ci_old
      Ci = αr·Im + αi·Re + βr·Ci_old + βi·Cr_old
```

两条分解代数等价，但中间项的 FP32 舍入不同，不能把一种路径的误差分析套用到另一种（§2.1 原文要求）；实现按形状与片上容量选择路径。

# �求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现句柄式 BLAS 接口 `aclblasCgemm3m`：complex64、trans 3×3 全组合、任意合法前导维与运行时形状（用例覆盖 m/n/k ≤ 4096），精度达生态算子开源混合容差标准 FLOAT32 分量档，性能达 GPU 标杆（cuBLAS `cublasCgemm3m`）的 0.8 倍（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 3M：3 次实数 GEMM。arch22 已合入的 Cgemm 已把"复数拆平面 → 委托 `aclblasSgemm` → 组装"整条链跑通，本算子把"4 乘积 + 组装"换成"3 乘积 + 平面预加 + 组装"，拆分/累加/组装三个 kernel 直接复用 |
| 2 | 接口原型 | §2.3 签名（13 参数），返回 `aclblasStatus_t`；声明已存在（L584），零公共头改动 |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | A/B/C/alpha/beta 全 COMPLEX64（实部/虚部各 FP32） |
| 5 | 目标芯片与 arch 映射 | A2（`ascend910b*`）与 A3（`ascend910_93*`）→ arch22（`build.sh` 的 SOC→arch 映射，L164）；950（arch35）已有实现，不在本任务范围 |
| 6 | 精度要求 | §3.2：FLOAT32 分量档 rtol=atol=2^-13（1.2e-4）、matched_ratio≥0.99、max_abs_error≤1e-2 或 32ULP；golden=`cblas_cgemm`（float64 累加），3M 舍入不保证逐位一致 |
| 7 | 性能预期 | §3.3：单 case NPU kernel 平均耗时 ≤ `gpu_ms`/0.8；口径 `msprof op` Task Duration，warmup 5 次 + 10 次有效样本取均值 |
| 8 | 代码架构选型 + 可行性 | 复用 arch22 Cgemm 三阶段骨架 + 3M 乘积表（见"详细设计"）。**已排除**：把实现放进 `blas/gemm3m/`（§2.2 明令禁止）；委托链沿用 K≤1024 分块以绕开未修复的 ops-blas#1077。NPU 侧达标性**待实测** |
| 9 | 评测来源 | 任务包 `test_cases/`：精度 1000 条 + 性能 200 条（`gpu_baseline.csv` 200 条全回填），CSV 安装到 `test/gemm/cgemm3m/<arch>/` 由 `cgemm3m_test` 加载 |

# 详细设计

## 算子分析

### 数学公式

实现采用**非融合路径**为主：

```text
预加（AIV，复用 cgemm_accum_kernel_do）：
    S_A = Ar + Ai          （m×k 实数平面）
    S_B = Br + Bi          （k×n 实数平面）
三次实数 GEMM（AIC，委托 aclblasSgemm，子句柄 + K≤1024 分块）：
    P0 = op(Ar)·op(Br)
    P1 = op(Ai)·op(Bi)
    P2 = op(S_A)·op(S_B)
组装（AIV，复用 cgemm_accum_kernel_do + cgemm_combine_kernel_do）：
    Re = P0 − P1
    Im = P2 − P0 − P1
    (Cr, Ci) = 复数 alpha/beta 组装（CgemmCombineTilingData 语义，写全 m×n）
```

共轭转置的处理：`OP_C` 对复数整体取共轭转置，等价于"转置 + 虚部平面取负"。拆分后对虚部平面做一次 `Ai' = s_a·Ai`、`Bi' = s_b·Bi`（s = −1 表示共轭，复用累加 kernel 的 `dst = scale*src` 模式），预加在取负之后进行，随后三个 GEMM 统一按 T 取向发射——与已合入 Cgemm 的"`OP_C` 归并为 `OP_T`"做法一致。

融合路径（含纠正重算）作为备选：`P0 = op(Ar+Ai)·op(Br)`、`P1 = op(Ar)·op(Bi−Br)`、`P2 = op(Ai)·op(Br+Bi)`，`Re = P0−P2`、`Im = P0+P1`。它需要三个预加平面（`Ar+Ai`、`Br+Bi`、`Bi−Br`）但组装少一次平面减法；两条路径的 FP32 舍入不同（§2.1），路径切换的取舍**待 S3 精度实测裁决**。

### 支持数据类型

| 项 | 类型 | 说明 |
|---|---|---|
| A/B/C | COMPLEX64（实部/虚部各 float32） | 列主序，元素 8 B |
| alpha/beta | COMPLEX64，Host 内存 | 标量读回后参与组装 |
| 计算精度 | FP32 | 3 次实数 GEMM 在 FP32 Cube 上执行（HF32 关闭，沿用 arch22 Sgemm 的做法） |

### 支持形状

| 维度 | 约束 | 退化行为 |
|---|---|---|
| m、n | ≥0 | m=0 或 n=0：**结构校验通过后** quick return SUCCESS，不引用 A/B/C、不启动 kernel |
| k | ≥0 | k=0 或 alpha=0：只做 `C = beta*C`，不引用 A/B（允许为空）；beta=0 时 C 可为未初始化 |
| lda/ldb/ldc | 见参数表下界 | 允许 padding；低于下界 → INVALID_VALUE |
| OP_C | 复数 | 与 OP_T 结果不同（虚部平面取负），非对称复数矩阵用例验证 |

仅写 C 的逻辑 m×n 区域，`ldc` 的 padding 列保持不变；C 与 A/B 存储不得重叠（调用方保证，§2.1）。

## 算子实现

### 实现方案

#### 总体架构与数据流

```text
GM: A(complex)──┐
GM: B(complex)──┤ [AIV cgemm_split_kernel] 拆实虚 ──> GM: Ar, Ai, Br, Bi（对齐实数平面）
                │                    │
                │                    └ [AIV 累加 kernel] S_A = Ar+Ai（含共轭取负）；S_B = Br+Bi
                │
                └ [AIC] 3 次实数 GEMM（委托 aclblasSgemm，子句柄）：
                     P0 = op(Ar)·op(Br)   → re 平面（覆写）
                     P1 = op(Ai)·op(Bi)   → scratch 平面（覆写）
                     P2 = op(S_A)·op(S_B) → im 平面（覆写）
                     组装：im += −1·re（im = P2−P0）；re += −1·scratch（Re = P0−P1）；
                           im += −1·scratch（Im = P2−P0−P1）
GM: C_old ──[AIV cgemm_combine_kernel] C = alpha·(Re + i·Im) + beta·C_old ──> GM: C
```

工作区布局（float 平面，各段 32B 对齐，自低址向高址）：

| 区段 | 大小 | 说明 |
|---|---|---|
| Ar、Ai | 各 packedLd_A × k | A 的实部/虚部平面 |
| Br、Bi | 各 packedLd_B × n | B 的实部/虚部平面 |
| S_A | 同 Ar | Ar+Ai 预加平面 |
| S_B | 同 Br | Br+Bi 预加平面 |
| re、im、scratch | 各 CeilAlign(n,16) × CeilAlign(m,16) | 乘积平面（内部 C^T 行主序）与组装 scratch |
| 委托 scratch | 4 MiB 余量 | 委托 `aclblasSgemm` 的工作区（与已合入 Cgemm 同款余量） |

总量约 `3·(m·k + k·n) + 3·m·n` 个 float（Cgemm 为 `2·(m·k + k·n + 2·m·n)`），超出 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

#### host 侧设计

**分核策略**：本算子自身只做平面级操作（拆分/预加/组装），分核沿用被复用 kernel 的既有策略——AIV 侧按元素数与 AIV 核数取小（`ClampBlocks` 风格）；GEMM 分核完全由被委托的 arch22 `aclblasSgemm` 决定（AIC 满核、行属主/整问题分派）。

**数据分块和内存优化策略**：
- 预加平面 S_A、S_B 与拆分平面同布局，预加逐列连续访问。
- K 向分块：每次委托调用的 K ≤ 1024（`CGEMM_K_CHUNK` 同款常量），块间把部分和经累加 kernel 并入同一 re/im 平面——fp32 加法精确，精度不低于单次长归约。
- 标量读回用共享的 `CBlas3ReadScalarsFromDevice`（兼容 Host 指针约定）。

**tilingkey 规划策略**：仓内 arch22 gemm 族不使用 tilingkey 宏分支，host 侧显式选择 kernel。本算子的分派字段：

| 条件 | 分支 |
|---|---|
| alpha=(0,0) 或 k=0 | 跳过全部 GEMM 与预加，仅组装（`hasTemp=false`）；alpha=0 且 beta=(1,0) 时直接返回不发射 kernel |
| beta=(0,0) | 组装不读 C 旧值 |
| transa/transb = C | 虚部平面取负（s = −1）后统一按 T 发射 |
| K ≤ 1024 | 单块直出；K > 1024 | 分块累加 |

**数据检测**（校验顺序逐条对齐 §2.4/§2.5）：

1. handle 非空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. transa、transb ∈ {N, T, C} → 否则 `ACLBLAS_STATUS_INVALID_ENUM`
3. m、n、k ≥ 0 → 否则 `INVALID_VALUE`
4. lda ≥ max(1, transa=N ? m : k)、ldb ≥ max(1, transb=N ? k : n)、ldc ≥ max(1, m) → 否则 `INVALID_VALUE`
5. alpha、beta 非空 → 否则 `INVALID_VALUE`
6. 结构全部合法后判 quick return：m=0 或 n=0 → SUCCESS（A/B/C 可为空）
7. m>0 且 n>0 时 C 非空 → 否则 `INVALID_VALUE`
8. k>0 且 alpha≠0 时 A、B 非空 → 否则 `INVALID_VALUE`（alpha 值在标量读回后判定）
9. k=0 或 alpha=0 → 仅 `C = beta*C`；beta=0 不读 C 旧值

（顺序按任务书 §2.5"结构合法后再执行 quick return"设计；已合入 Cgemm 的 `ValidateCgemmPointers` 在 alpha=0 且 A/B 为空时会误返 `INVALID_VALUE`，本算子为全新实现，直接按任务书顺序实现，不受该偏差影响。）

**已知缺陷 ops-blas#1077 的关联**：委托的 arch22 实数入口在 K>1024 且 M、N 均低于整问题门限时结果错误（`blas/gemm/README.md` L356）。本算子的委托调用沿用已合入 Cgemm 的 K≤1024 分块绕行；#1077 修复后可移除（与 Cgemm 同步）。

#### kernel 侧设计

```text
[AIV] Init    ：解析平面 tiling → 求本核元素区间 → 队列初始化
       Process：拆分（A/B → Ar/Ai/Br/Bi）→（共轭取负）→ 预加（S_A、S_B）
[AIC]  由被委托的 aclblasSgemm 执行 3 次实数 GEMM（每次调用是独立的 kernel 发射）
[AIV]  Init    ：解析组装 tiling → 求本核元素区间
       Process：CopyIn（re/im/scratch + C_old）→ Compute（复数 alpha/beta 交叉缩放）
                → CopyOut（写逻辑 m×n，ldc padding 列不写）
```

文件落位（复用优先，新增文件数最少）：

| 文件 | 内容 |
|---|---|
| `blas/gemm/arch22/cgemm3m_host.cpp` | 公开入口 `aclblasCgemm3m`：校验、3M 乘积表编排、工作区、委托发射（新增） |
| `blas/gemm/arch22/cgemm_kernel.cpp/.h` | `cgemm_split_kernel_do`、`cgemm_accum_kernel_do`、`cgemm_combine_kernel_do` **复用不改动** |
| `test/gemm/cgemm3m/arch22/` | `cgemm3m_test.cpp`（复用 `test/gemm/cgemm3m/` 顶层的 golden/param/profile/verify 头）+ CSV（新增） |
| `test/gemm/cgemm3m/CMakeLists.txt` | 注册 arch22 目标（现为 arch35-only 分支，需扩展） |

#### 性能优化要点

本文未上板，以下为设计依据与规划，实测数据在性能阶段以 `msprof op` 回填。由 `gpu_baseline.csv` 算出的 NPU 允许耗时（`gpu_ms`/0.8，括号内为 `gpu_ms`；该任务包性能子集无 4096³ 立方用例，最大为 1722×1352×1916）：

| 代表 case（任务包 `gpu_baseline.csv`） | 转置 | m×n×k | 允许耗时 |
|---|---|---|---|
| cgemm3m-perf-194 | N,T | 1722×1352×1916 | 2.32 ms（1.86） |
| cgemm3m-perf-198 | N,T | 1162×1773×782 | 0.86 ms（0.69） |
| cgemm3m-perf-196 | C,C | 1025×816×1610（alpha=2，beta=0.5） | 0.91 ms（0.73） |
| cgemm3m-perf-197 | C,C | 1823×610×685 | 0.46 ms（0.37） |
| cgemm3m-perf-195 | N,T | 1161×67×1754 | 0.14 ms（0.11） |

| 优化项 | 依据 | 状态 |
|---|---|---|
| 3 次实数 GEMM（相对 Cgemm 的 4 次少 25% GEMM 工作量） | GEMM 主导的规模下耗时近似随次数线性；代价是 2 次预加平面的额外带宽 | 设计定案（3M 的意义所在） |
| 预加与拆分合并 | 拆分写平面时顺带写出 S_A/S_B 可省 2 次平面读 | 待实测后再决定是否需要专用融合 kernel（先用现成累加 kernel 保正确性） |
| K 分块粒度 | 委托调用 K≤1024（#1077 绕行）；块数增多会增加发射次数 | 依赖 #1077 修复 |
| 小形状 kernel 链 | perf-195 预算仅 0.14 ms，链上有拆分/预加/3×GEMM/组装多次发射 | 待 `msprof` 归因 |
| 未采用：融合路径作主路径 | 预加平面多一个（`Bi−Br`），组装少一次减法；两者 FP32 舍入不同 | 备选，S3 精度实测后裁决 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（`ascend910b*`，arch22） | ✓ 本任务目标；自验证设备 Atlas 800T A2（910B3） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（`ascend910_93*`，arch22） | ✓ 本任务目标（§3.1：自验证覆盖一种款型即可） |
| Ascend 950PR / 950DT（arch35） | 已有实现（`blas/gemm/arch35/cgemm3m_host.cpp`），不在本任务范围 |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：host 入口按上表顺序校验并返回状态码，不对 Device 张量做元素级扫描。
- 非连续 Tensor：不支持，仅支持 lda/ldb/ldc 描述的列主序带填充布局；不使用 AutoContiguous/ViewCopy。
- broadcast：不支持。
- dynamic shape：不要求，运行时入参由 host 按实际取值 tiling 与发射。
- 原地与视图语义：C 原地覆写，不返回视图；C 与 A/B 存储不得重叠。
- 确定性计算：不要求；3M 路径（融合/非融合）与 K 向归约顺序由实现按形状选择。
- 空 Tensor：m=0 或 n=0 合法 no-op；k=0 或 alpha=0 只做 `C = beta*C`。
- 0 维 Tensor：不涉及。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 的实部/虚部分量记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。按 FLOAT32 分量档：rtol=atol=2^-13（1.2e-4），max_abs_error 限值 1e-2 或 32 ULP | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | `cblas_cgemm`（Netlib `cgemm`），float64 累加后收窄到 complex64；实部、虚部分别比对。3M 舍入与 4M 不同，不保证逐位一致，按阈值判定 | 任务书 §3.1、§3.2 第 1、4 条 |
| 性能 | 单 case：NPU kernel 平均耗时 ≤ `gpu_ms` / 0.8 | 任务书 §3.3 |
| 性能口径 | `msprof op` 的 kernel task duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均；GTest `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

用例来自任务包 `test_cases/cgemm3m_test.csv`（固定随机种子，逐条可复现），精度 1000 条 + 性能 200 条：

| 类别 | 前缀 | 条数 | 覆盖 |
|---|---|---|---|
| L0 基础 | TC_L0 | 18 | trans 3×3 全组合 × 小尺寸 |
| 方阵尺寸 | TC_SQ | 110 | 尺寸池方阵（含 4095³/4096³） |
| 矩形 | TC_MN | 91 | 非方阵形状 |
| 标量 | TC_AB | 88 | 复数 alpha/beta 特殊值 (0,0)/(1,0)/(0,1)/(−1,0)/±1e4 与随机值 |
| 前导维 | TC_LD | 16 | 紧凑与 padding |
| 填充值域 | TC_FL | 21 | 均匀/正态/全零/交替/极端/Inf/NaN |
| 边界负向 | TC_ED | 38 | m/n/k=0、alpha=0、beta=0+C NaN、空指针、非法枚举、负维度、OP_C 对照 |
| 扩展精度 | TC_EX | 586 | 尺寸 × 转置 × 标量 × 前导维采样 |
| 性能 | TC_PF | 200 | 含 9 种转置组合与非默认标量（约 12%），baseline 判定键含 alpha/beta |

- 测试工程：新增 `test/gemm/cgemm3m/arch22/`，复用 `test/gemm/cgemm3m/` 顶层的 `cgemm3m_golden.h`（`cblas_cgemm` 口径 golden）、`cgemm3m_param.h`、`cgemm3m_profile.h`、`cgemm3m_verify.h`；在 `test/gemm/cgemm3m/CMakeLists.txt` 注册 arch22 目标（现文件为 arch35-only 分支）。
- 复现入口：`bash build.sh --ops=cgemm3m --soc=ascend910b3` → `build/test/gemm/cgemm3m/arch22/cgemm3m_test`；任务包 `verify_accuracy.py` / `verify_performance.py` 把 CSV 安装到 `test/gemm/cgemm3m/<arch>/cgemm3m_test.csv` 后加载执行。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h` **零改动**：`aclblasCgemm3m` 声明已存在（L584），本任务不新增、不修改任何声明。
- 新增文件仅 `blas/gemm/arch22/cgemm3m_host.cpp` 与 `test/gemm/cgemm3m/arch22/`；复用的 `cgemm_kernel.cpp/.h`、`gemm_host.cpp`（Sgemm）不做任何改动，已合入算子零回归风险。
- `test/gemm/cgemm3m/CMakeLists.txt` 的注册方式保持 arch35 行为不变（在 else 分支上扩展 arch22），不影响其他 SOC 的构建。
- `blas/gemm/README.md` 的 `aclblasCgemm3m` 章节补充 Atlas A2/A3 支持情况（现 L958 仅写 950PR/arch35）。
- 与 `blas/gemm3m/`（实数 `aclblasSgemm3m`）无代码交叉；委托链沿用 K≤1024 分块绕行 ops-blas#1077，该缺陷修复后与 Cgemm 一起移除绕行。


