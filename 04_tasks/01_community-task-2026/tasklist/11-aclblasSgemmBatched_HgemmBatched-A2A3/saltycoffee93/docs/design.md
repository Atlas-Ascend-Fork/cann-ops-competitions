# aclblasSgemmBatched / aclblasHgemmBatched 算子设计文档

> 目标芯片：Atlas A2/A3（arch22，自验证 SOC=ascend910b3）｜CANN 9.1.0｜合入位置 `ops-blas/blas/gemm_batched/arch22/`
> 任务书：`aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md`（下文 §n 均指该任务书章节）
> 仓内现状基线：ops-blas master `55ef791`（2026-11-25 源码核验）。本文撰写时未上板，凡需 NPU 实测的结论均标注"待实测"，不含未经测试的性能数字。

# 需求背景

## 需求来源

- 任务名称/批次：11 月社区任务 —— aclblasSgemmBatched / aclblasHgemmBatched 算子开发（A2A3）。
- 合入目标仓与目录：`https://gitcode.com/cann/ops-blas` → 实现 `blas/gemm_batched/arch22/`；测试 `test/gemm_batched/sgemm_batched/arch22/`（任务书 §2.2、§5）。
- 验收流程：设计文档以 PR 形式提交至 cann-competitions 仓 tasklist，评审合入后进入开发与验收（任务书 §4）。

## 背景介绍

### 算子功能定位

两个批量通用矩阵乘算子：`C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`（i = 0..batchCount−1）。各 batch 共享 transa/transb、m/n/k、lda/ldb/ldc、alpha、beta，A/B/C 数据与地址经 Device 指针数组独立。

| 接口 | 对标 | dtype | 作用 |
|---|---|---|---|
| aclblasSgemmBatched | cuBLAS `cublasSgemmBatched`｜Netlib `sgemm`（数学语义） | FLOAT32 | FP32 批量 GEMM |
| aclblasHgemmBatched | cuBLAS `cublasHgemmBatched` | FLOAT16（`aclblasHalf`） | FP16 批量 GEMM |

### 业界对标

| 项 | cuBLAS 同名接口 | 本算子 |
|---|---|---|
| 参数序列 | 13 参数（含三个指针数组） | 逐参数一致（§2.3） |
| trans | N/T/C，实数算子 C≡T | 同（§2.1） |
| alpha/beta 位置 | pointer mode 可 Host/Device | 约定 Host 内存（与 cuBLAS 默认一致，§2.4 说明） |
| 指针数组 | Aarray/Barray/Carray 为 Device 指针数组 | 同；数组本体 Device 内存，元素有效性由调用方保证 |
| 非法 trans 返回码 | `CUBLAS_STATUS_INVALID_VALUE` | `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 约定，有意差异，§2.3） |

### 工程形态

- 工程模式：Kernel 直调（Host C API 校验/分派/tiling，经 handle 绑定的 stream 直调 NPU kernel，§2.2）。
- 仓内现状（master `55ef791` 源码核验）：

| 内容 | 现状 | 证据 | 本任务工作 |
|---|---|---|---|
| 公开声明 | 均已存在 | `include/cann_ops_blas.h` 约 L616（S）/L626（H） | 零公共头改动 |
| arch22 实现 | 不存在 batched S/H（仅有同目录 `cgemm_batched_*` 复数批量） | `blas/gemm_batched/arch22/` 目录清单 | **新增** S/H 的 host 与 kernel |
| arch35 参考 | 已有 S/H 两套 | `blas/gemm_batched/arch35/gemm_batched_host.cpp`（S）、`hgemmBatched_host.cpp`（H），合 5863 行 | 参考 batch 调度、指针数组处理与 FP16 缩放路径 |
| 测试工程 | 仅 arch35 | `test/gemm_batched/sgemm_batched/arch35/`、`test/gemm/hgemmBatched/arch35/` | 新建 `test/gemm_batched/sgemm_batched/arch22/` 并注册目标 |
| README | S/H 章节已有 | `blas/gemm_batched/README.md` L25/L362 | 补充 A2/A3 支持情况 |

- 仓内近亲与复用方式：

| 近亲 | 复用等级 | 复用内容 | 差异处置 |
|---|---|---|---|
| `blas/gemm_batched/arch22/cgemm_batched_*`（已合入复数批量） | ★ 主模板 | arch22 上的批量调度骨架：Device 指针数组读取、batch×分核二维切分、原地 beta 组装 | 数据平面从复数换成实数 FP32/FP16；S/H 共享一套调度 |
| `blas/gemm/arch22/gemm_*`、`hgemm_*`（已合入单算子） | ★ | 单 batch GEMM 的 Cube tiling 与 K 归约结构 | host 侧 batch 维扩展 |
| `blas/gemm_batched/arch35/gemm_batched_host.cpp`、`hgemmBatched_host.cpp` | ◇ 参考 | batch 调度、指针数组处理、FP16 缩放路径 | arch35 专用，只借鉴思想 |

### 接口规格

声明均已存在于 `include/cann_ops_blas.h`，**无需新增或修改**（§2.3）。签名（S 为 FLOAT32 版，H 将 float 换为 `aclblasHalf`）：

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount);
```

参数规格（逐行对齐 §2.4；S/H 按 dtype 分列）：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|---|---|---|---|---|---|
| handle | 句柄，携带 stream | scalar | — | 非空；空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 标量 |
| transa/transb | op(A)/op(B) 运算类型 | attr | int | {N, T, C}；C≡T（实数不共轭）；其它 → `INVALID_ENUM` | — |
| m / n / k | GEMM 维度 | attr | int | ≥0；负 → `INVALID_VALUE`；m=0 或 n=0 合法 no-op；k=0 只做 `C=beta*C` | — |
| alpha / beta | 标量乘数（Host 内存） | scalar | S=FLOAT32；H=FLOAT16 | 非空；beta=0 不读 C 旧值 | 标量 |
| Aarray / Barray | Device 指针数组（长 batchCount） | tensor | 同上 | batchCount>0 且 m,n,k>0 且 alpha≠0 时非空 | 每元素指向 lda×k（N）或 lda×m（T/C）矩阵 |
| lda / ldb | A/B 前导维 | attr | int | trans=N：lda≥max(1,m)、ldb≥max(1,k)；T/C：lda≥max(1,k)、ldb≥max(1,n)；否则 `INVALID_VALUE` | 标量 |
| Carray | 输出指针数组（原地覆写） | tensor | 同上 | m>0 且 n>0 时非空；C[i] 互不重叠且不与 A[j]/B[j] 重叠 | 每元素指向 ldc×n 矩阵 |
| ldc | C 前导维 | attr | int | ≥max(1,m)；否则 `INVALID_VALUE`；padding 列不修改 | 标量 |
| batchCount | batch 数量 | attr | int | ≥0；负 → `INVALID_VALUE`；0 为合法 no-op | — |

### 数学公式

```text
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]，i = 0..batchCount−1
op(X)：N 直读；T/C 转置（实数，C 与 T 等价）
各 batch 独立计算，batch 间无数据依赖。
```

# 需求分析

## 需求描述

用 Ascend C 在 Atlas A2/A3（arch22）上实现句柄式批量 GEMM 接口 `aclblasSgemmBatched`（FP32）与 `aclblasHgemmBatched`（FP16）：Device 指针数组批量语义、trans 3×3 全组合、batchCount 覆盖 0..65536；S 精度达 FLOAT32 分量档、H 达 FLOAT16 分量档，性能达 GPU 标杆 0.8 倍（kernel-to-kernel）。

## 需求拆解

| # | 必须回答 | 本算子结论 |
|---|---|---|
| 1 | 数学定义与可白嫖的代数量 | 逐 batch 独立标准 GEMM。同目录已合入的 `cgemm_batched_*`（arch22 复数批量）提供"Device 指针数组读取 + batch×分核二维切分 + 原地 beta 组装"的现成调度骨架，S/H 把数据平面换成实数 FP32/FP16 即可；单 batch GEMM 结构复用已合入的 `gemm_*`/`hgemm_*` |
| 2 | 接口原型 | §2.3 签名（13 参数），返回 `aclblasStatus_t`；两个声明均已存在，零公共头改动 |
| 3 | 参数规格 | 见"接口规格"参数表；异常行为逐条对齐 §2.4 |
| 4 | 支持数据类型 | S：全 FLOAT32；H：全 FLOAT16（FP16 计算域，golden 按 FP32 精度域收窄） |
| 5 | 目标芯片与 arch 映射 | A2（`ascend910b*`）与 A3（`ascend910_93*`）→ arch22（`build.sh` 的 SOC→arch 映射，L164）；arch35 已有实现，不在本任务范围 |
| 6 | 精度要求 | §3.2：S 按 FLOAT32 档（rtol=atol=2^-13、ratio≥0.99、1e-2 或 32ULP），H 按 FLOAT16 档（2^-10、ratio≥0.99、1e-1 或 32ULP）；golden：S=`cblas_sgemm`（float64 累加收窄），H=测试工程 float64 累加收窄 FP16 |
| 7 | 性能预期 | §3.3：单 case NPU kernel 平均耗时 ≤ `gpu_ms`/0.8；口径 `msprof op` Task Duration，warmup 5 + 10 样本；S/H 各 100 条性能子集 |
| 8 | 代码架构选型 + 可行性 | 新增 `blas/gemm_batched/arch22/` 下 S/H host 与共享 kernel：复用 `cgemm_batched_*` 的批量调度骨架（batch×tile 二维分核、指针数组 Device 侧读取），单 batch 计算复用已合入 `gemm_*`/`hgemm_*` 的 Cube 结构。**已排除**：host 侧逐 batch 循环发射（batchCount 达 65536 时发射次数爆炸） |
| 9 | 评测来源 | 任务包 `test_cases/`：S 600 条（500 精度 + 100 性能）+ H 600 条（500 精度 + 100 性能），`gpu_baseline.csv` 各 100 条全回填；S/H 共用同一 GTest 套件按 op 列分派 |

# 详细设计

## 算子分析

### 数学公式

见"接口规格 · 数学公式"节。每个 batch 是标准 GEMM：单 batch 计算结构复用已合入 arch22 `gemm_*`/`hgemm_*` 的 Cube tiling 与 K 归约；batch 维新增的工作仅是"Device 指针数组 → (batch, tile) 二维工作项"的分派。

### 支持数据类型

| 项 | S | H | 说明 |
|---|---|---|---|
| A/B/C | FLOAT32 | FLOAT16（`aclblasHalf`） | 列主序带填充 |
| alpha/beta | FLOAT32 | FLOAT16 | Host 内存标量 |
| 计算精度 | FP32 | FP16 计算（FP32 累加口径与 golden 一致） | H 的 golden 按 float64 累加收窄 FP16，kernel 用 FP32 累加满足 FLOAT16 档阈值 |

### 支持形状

| 项 | 约束 | 退化行为 |
|---|---|---|
| m/n/k | ≥0 | m=0 或 n=0：结构校验通过后 SUCCESS，三数组可空；k=0 或 alpha=0：只做 `C=beta*C`（Aarray/Barray 可空） |
| batchCount | ≥0 | 0 为合法 no-op，不引用任何数组元素；覆盖 0..65536（后 4 档大 batch 用 1×1×1 小矩阵） |
| lda/ldb/ldc | 见参数表下界 | 允许 padding；低于下界 → INVALID_VALUE |
| transa/transb | N/T/C（C≡T） | 非法枚举 → INVALID_ENUM |

仅写各 C[i] 的逻辑 m×n 区域，ldc padding 列不修改；C[i] 互不重叠且不与 A[j]/B[j] 重叠（调用方保证）。

## 算子实现

### 实现方案

#### 总体架构与数据流

kernel 变体划分（变体/触发条件/职责）：

| 变体 | 触发条件 | 职责 | kernel 数 |
|---|---|---|---|
| quick return | batchCount=0；或 m=0/n=0；或 alpha=0 且 beta=1 | host 直接返回 | 0 |
| beta 缩放 kernel | k=0 或 alpha=0 且 beta≠1 | 各 C[i] 的 m×n 区域 `C = beta*C`（beta=0 清零） | 1（按 dtype 实例化） |
| 批量 GEMM 主 kernel | 其余 | 单 launch：Device 指针数组读取 + batch×tile 二维分核完成全部 batch 的乘加 | 1（按 dtype 实例化） |

数据流（主 kernel）：

```text
GM: Aarray/Barray/Carray（Device 指针数组）──[kernel 内读取]──> 第 wid 号工作项 = (batch i, m-tile, n-tile)
GM: A[i], B[i]（按 trans/lda/ldb 地址映射）──[Cube 乘加，K 归约]──> C[i] 的 m×n tile
GM: C[i] 旧值 ─[beta 组装，beta=0 不读]──> 写回（ldc padding 列不写）
```

workspace：无额外平面（各 batch 就地读写）；若单 batch 大矩阵需要 K 分块中间区，按已合入单算子 gemm 的 workspace 口径。

#### host 侧设计

**分核策略**：工作项总数 = batchCount × ceil(m/tileM) × ceil(n/tileN)，按 AIC 核数线性分派（每核取连续工作项，同 batch 的 tile 相邻以提升 A[i]/B[i] 复用）；batchCount 极大而矩阵极小（1×1×1×65536）时每工作项退化为单元素乘加，天然负载均衡。

**数据分块和内存优化策略**：指针数组在 kernel 内按需读取（每工作项 3 次数组访问，可缓存于 UB）；S 走 FP32 Cube，H 走 FP16 Cube（累加 FP32）。

**tilingkey 规划策略**：无 tilingkey 宏分支，host 经 tiling 结构传参；变体由 host 显式选择。S/H 通过同一 tiling 结构 + dtype 模板实例化共享调度代码。

**数据检测**（校验顺序逐条对齐 §2.5 校验顺序说明）：

1. handle 非空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. transa/transb ∈ {N, T, C} → 否则 `INVALID_ENUM`
3. m/n/k ≥ 0 → 否则 `INVALID_VALUE`
4. lda/ldb/ldc ≥ 对应下界（按 trans 取值）→ 否则 `INVALID_VALUE`
5. batchCount ≥ 0 → 否则 `INVALID_VALUE`
6. alpha/beta 非空 → 否则 `INVALID_VALUE`
7. **quick return**：batchCount=0，或 m=0/n=0 → SUCCESS（三数组可空）
8. m>0 且 n>0 时 Carray 非空 → 否则 `INVALID_VALUE`；k>0 且 alpha≠0 时 Aarray/Barray 非空 → 否则 `INVALID_VALUE`
9. alpha=0 且 beta=1 → SUCCESS；beta=0 不读 C 旧值

#### kernel 侧设计

```text
[批量 GEMM 主 kernel（S/H 按 dtype 实例化）]
 Init   ：读取 tiling（m,n,k,lda,ldb,ldc,batchCount,trans, alpha/beta 内联）→ 求本核工作项区间
 Process：逐工作项取 Aarray[i]/Barray[i]/Carray[i] → Cube 乘加（K 归约）→ beta 组装写回
[beta 缩放 kernel]
 Process：逐 C[i] 的 m×n 区域 C = beta*C
```

文件落位：

| 文件 | 内容 |
|---|---|
| `blas/gemm_batched/arch22/sgemm_batched_host.cpp` | S 入口：校验、分派、发射（新增） |
| `blas/gemm_batched/arch22/hgemm_batched_host.cpp` | H 入口：同上，FP16 路径（新增） |
| `blas/gemm_batched/arch22/gemm_batched_kernel.cpp/.h`、`gemm_batched_tiling_data.h` | S/H 共享的批量调度 kernel（dtype 模板实例化，新增） |
| `test/gemm_batched/sgemm_batched/arch22/` | S/H 共用 GTest 套件 + CSV（按 op 列分派），注册 CMake arch22 目标 |

#### 性能优化要点

本文未上板，以下为设计依据与规划，实测数据在性能阶段以 `msprof op` 回填。由 `gpu_baseline.csv` 算出的 NPU 允许耗时（`gpu_ms`/0.8，括号内为 `gpu_ms`）：

| 代表 case（S） | trans | m=n=k×batch | 允许耗时 |
|---|---|---|---|
| sgemmBatched-perf-099 | T,N | 400³×8 | 0.25 ms（0.20） |
| sgemmBatched-perf-098 | N,N | 200³×16 | 0.07 ms（0.06） |
| sgemmBatched-perf-097 | C,C | 100³×64 | 0.05 ms（0.04） |
| sgemmBatched-perf-096 | N,T | 96³×128 | 0.07 ms（0.06） |
| sgemmBatched-perf-095 | T,N | 48³×256 | 0.06 ms（0.05） |

（H 同形状基线约为 S 的 1/4：perf-099 允许 0.035 ms、perf-098 允许 0.023 ms、perf-097 允许 0.025 ms。）

| 优化项 | 依据 | 状态 |
|---|---|---|
| 单 launch 全 batch 分派 | host 逐 batch 循环发射在 batchCount=65536 时不可接受；单 launch 内 (batch, tile) 二维分核 | 设计定案 |
| 小矩阵负载均衡 | 性能子集以 48³..400³ 的小矩阵为主（单 batch 计算量小），batch 维提供了充足的并行工作项 | 设计定案 |
| 指针数组 UB 缓存 | 每核工作项连续时同 batch 多 tile 复用数组元素 | 首版实现 |
| FP16 Cube + FP32 累加 | H 的 FLOAT16 档阈值（2^-10）与 cuBLAS 语义一致 | 首版实现 |
| 未采用：batch 维展平为单大 GEMM | 各 batch 地址独立（指针数组），不保证可展平 | 排除 |

## 支持硬件

| 支持的芯片版本 | 涉及 |
|---|---|
| Atlas A2 训练/推理系列产品（`ascend910b*`，arch22） | ✓ 本任务目标；自验证设备 Atlas 800T A2（910B3） |
| Atlas A3 训练/推理系列产品（`ascend910_93*`，arch22） | ✓ 本任务目标（§3.1：自验证覆盖一种款型即可） |
| 昇腾 950PR / 950DT（arch35） | 已有实现（`blas/gemm_batched/arch35/`），不在本任务范围 |

## 算子约束限制

逐条对齐 §2.5：

- 参数合法性与校验层级：host 入口按上表顺序校验；指针数组元素有效性、C[i] 重叠由调用方保证；不对 Device 指针数组元素或 Device 张量做元素级扫描。
- 非连续 Tensor：各 batch 仅支持 lda/ldb/ldc 描述的列主序带填充布局；不使用 AutoContiguous/ViewCopy。
- broadcast：不支持。
- dynamic shape：不要求，m/n/k/lda/ldb/ldc/batchCount 为运行时入参。
- 原地与视图语义：C[i] 原地覆写，不返回视图；C[i] 可写区域与任意 A[j]/B[j] 可读区域不重叠。
- 确定性计算：不要求；batch 调度与 K 归约顺序由实现按形状选择。
- 空 Tensor：batchCount=0、m=0/n=0 合法 no-op（三数组可空）；k=0 或 alpha=0 时 Aarray/Barray 可空。
- 0 维 Tensor：不涉及；标量以 Host 标量指针传入。
- 异步执行：kernel 在 handle 绑定的 stream 上异步执行。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度逐元素与用例判定 | `\|actual − golden\| ≤ atol + rtol·\|golden\|` 记为 matched；用例须同时满足 matched_ratio ≥ 0.99 与 max_abs_error ≤ 限值。S 按 FLOAT32 档（2^-13、1e-2 或 32ULP），H 按 FLOAT16 档（2^-10、1e-1 或 32ULP） | 任务书 §3.2；生态算子开源精度标准 `mixed_tolerance_standard.md`（pin `28a1ce52`） |
| 精度 golden | S：`cblas_sgemm`（float64 累加收窄 float32，逐 batch）；H：测试工程 float64 累加 → 收窄 FP16（Netlib 无 half 参考实现） | 任务书 §3.1、§3.2 |
| 性能 | 单 case：NPU kernel 平均耗时 ≤ `gpu_ms` / 0.8（S/H 各自基线） | 任务书 §3.3.2 |
| 性能口径 | `msprof op` 的 kernel task duration（正式判据）：warmup 5 次不计入，之后 10 次有效样本取算术平均；GTest `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3.1、§3.3.3 |
| 内存 | 不涉及 | 任务书 §3.4 |

## 测试概览

任务包两套 CSV（S/H 各 600 条 = 500 精度 + 100 性能），S/H 共用同一 GTest 套件按 op 列分派：

| 维度 | 覆盖 |
|---|---|
| transa×transb | 9 种全组合 + 非法枚举负向 |
| m/n/k | 尺寸池 0→4096（含 4095³/4096³ 立方），负值负向 |
| batchCount | 0、1、2、3、8、16、64、256、1024、4096、16384、65536（后 4 档 1×1×1），负值负向 |
| alpha/beta | 特殊值 0/1/−1/0.5/±1e4 与随机值；alpha=0+A NaN、beta=0+C NaN |
| lda/ldb/ldc | 紧凑与 +1/+8/+64 padding，低于下界负向 |
| H 特殊 | FP16 上限、次正规数、Inf/NaN |
| 性能子集 | 各 100 条，small/medium/large 分层；显式保留 alpha/beta 列（约 12% 非默认标量），baseline 判定键含 alpha/beta |

- 测试工程：新建 `test/gemm_batched/sgemm_batched/arch22/`，在 `test/gemm_batched/sgemm_batched/CMakeLists.txt` 注册 arch22 目标（参考 arch35 与 `test/gemm/hgemmBatched/arch35/`）。
- 复现入口：`bash build.sh --ops=sgemm_batched,hgemm_batched --soc=ascend910b3`；任务包 `verify_accuracy.py` / `verify_performance.py` 将 CSV 安装到 `test/gemm_batched/<op>/<arch>/` 后加载执行。
- 负向与专项重点：batchCount=0（不引用指针数组）、实数算子 OP_C 与 OP_T 结果一致、ldc padding 列哨兵写保护、FP16 溢出/次正规处理。

## 兼容性分析

- 公共头 `include/cann_ops_blas.h` **零改动**：两个声明均已存在（约 L616/L626）。
- 新增文件仅 `blas/gemm_batched/arch22/` 下 S/H 的 host/kernel/tiling 与 `test/gemm_batched/sgemm_batched/arch22/`；同目录已合入的 `cgemm_batched_*` 与 `blas/gemm/arch22/` 单算子实现零改动、零回归风险。
- `test/gemm_batched/sgemm_batched/CMakeLists.txt` 注册 arch22 目标保持 arch35 行为不变。
- `blas/gemm_batched/README.md` 的 `aclblasSgemmBatched`（L25）/`aclblasHgemmBatched`（L362）章节补充 Atlas A2/A3 支持情况。

