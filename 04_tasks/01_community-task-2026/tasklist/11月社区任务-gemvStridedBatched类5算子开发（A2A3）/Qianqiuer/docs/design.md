# gemvStridedBatched 类 5 算子设计文档（Atlas A2/A3）

团队及开发者：Qianqiuer。设计日期：2026-11-25。

# 需求背景（required）

## 需求来源

本设计对应《11 月社区任务 aclblasSgemvStridedBatched 类算子开发任务书（Atlas A2/A3）》（含 2026-11-24 修订记录）及随附 5 变体自验材料（`test_cases/{sgemv,hsh_gemv,hss_gemv,tst_gemv,tss_gemv}_strided_batched/`）。任务目标是在昇腾 Atlas A2 / Atlas A3 系列产品（`arch22` / `dav-2201`）上，基于 CANN 9.1.0 与 Ascend C 编程语言完成单精度与半精度实数跨步批量矩阵-向量乘算子族 5 个句柄式 BLAS 接口的设计、开发、自验与合入，目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。

本文档为算子开发阶段提交的技术设计方案，提交至竞赛仓 `04_tasks/01_community-task-2026/tasklist/11月社区任务-gemvStridedBatched类5算子开发（A2A3）/Qianqiuer/docs/design.md` 用于设计评审，不代表已完成全部真机验收或代码合入。

## 背景介绍

### 算子应用场景与价值

跨步批量矩阵-向量乘（Strided Batched GEMV）在单次 API 调用中对 `batchCount` 组按等距跨步（`strideA`、`stridex`、`stridey`）连续排布的独立矩阵-向量对执行 `y[i] = alpha * op(A[i]) * x[i] + beta * y[i]`，广泛应用于多头注意力解码阶段、批量线性滤波、科学计算批量求解以及中小规模批量推荐推理等场景。相较于循环调用单矩阵 GEMV 或指针数组式 `gemvBatched`，StridedBatched 模式免除了 Host 侧多次下发开销与 Device 侧二级指针数组访存延迟，更适合在 NPU 上通过单次 Kernel 发射实现高带宽利用率。

### ops-blas 代码仓现状与基线分析

本任务基于 `ops-blas` 开源仓（核验基线提交 `55ef791`）的句柄式 BLAS 工程模式开发，不涉及历史 TBE 算子迁移。仓内现状核查如下：

1. **公共接口声明**：`include/cann_ops_blas.h`（L311-L334）已存在全部 5 个接口声明（`aclblasSgemvStridedBatched`、`aclblasHSHgemvStridedBatched`、`aclblasHSSgemvStridedBatched`、`aclblasTSTgemvStridedBatched`、`aclblasTSSgemvStridedBatched`），状态码与枚举定义在 `include/cann_ops_blas_common.h`，本任务直接复用，无需修改公共头或新增私有平行接口。
2. **arch35 参考实现**：`blas/gemv_strided_batched/arch35/` 已实现 Ascend 950PR/950DT 版本，采用 `simt_api/asc_simt.h`（SIMT 编程模型）及 `trans=T/C` 下的 RegBase SIMD 路径。Atlas A2/A3（`arch22`）不支持 950 专属 SIMT 与 RegBase 接口，仅可参考其接口分派与跨步地址语义，不得直接复制不兼容代码。
3. **arch22 既有基础实现瓶颈**：当前上游 `blas/gemv_strided_batched/arch22/` 已存在一套初始基础实现（`gemv_strided_batched_host.cpp`、`gemv_strided_batched_kernel.cpp`、`gemv_strided_batched_kernel.h`、`gemv_strided_batched_tiling_data.h`），但逐项核对任务书与实测需求存在以下关键缺口，须在本任务中系统重构与优化：
   - **Host 校验顺序与跨步重叠检查缺失**：现有实现先在 `m == 0 || n == 0 || batchCount == 0` 时提前返回 `SUCCESS`，绕过了 `trans`、`lda`、`incx/incy`、`strideA/stridex/stridey`、`alpha/beta` 等结构参数校验，不符合任务书 §2.5 校验顺序；同时缺少相邻 batch 区域重叠校验（如 `strideA < lda * n`、`stridex < 1 + (L_x - 1)*|incx|`、`stridey < 1 + (L_y - 1)*|incy|`），无法通过 `TC_ED_220`~`TC_ED_222` 负向用例。
   - **分核仅沿 batch 维度，严重欠并行**：现有实现限定 `usedCoreNum <= batchCount`，导致性能用例中 `batchCount=1`（含 `4096×4096`、`4095×4095`、`1×4096` 等）仅使用 1 个 AIV 核、`batchCount=2/3/8` 仅使用 2~8 个核，远未利用 Atlas 800T A2（910B3）满核带宽。
   - **访存与向量流水串行、缺少数据复用**：现有 `Normal` 路径逐列循环执行 1D `DataCopyPad` + 标量取 `x` + 同步等待 + `Muls`/`Add`，无多列 2D 批量搬运与双缓冲；`Transpose` 路径对输出每一列都从 GM 重复加载整个向量 `x` 并逐列同步标量写回，访存冗余严重。
   - **非单位步长路径开销过大**：随附 200 条性能用例中 158 条包含非单位或负步长（`incx`/`incy` 覆盖 `±1/±2/±3` 及 `{7,8,16,31,64,127}`），现有 `StoreStridedVec` 逐元素执行 `S_MTE3`/`MTE3_S` 单元素写回，`LoadStridedVec` 使用标量循环 Gather，难以达到 GPU 0.8× 标杆。
   - **数值乘加顺序风险**：现有 `Normal` 路径先计算 `alpha * x[c]` 再与矩阵列相乘，在 `alpha = ±1e4` 等大标量场景下存在额外中间溢出与舍入放大风险；更稳妥的做法是在 FP32 内完成 `op(A) * x` 归约后再统一执行 `alpha * acc + beta * y`。
   - **测试工程适配缺口**：现有 `test/gemv_strided_batched/arch22/gemv_strided_batched_test.cpp` 与任务书 5 变体自验包在 CSV 列格式、FP16/BF16 RNE 量化、FP64 累加 golden、空指针解析、`[PERF]` 计时输出及 `--ops` 变体构建映射方面存在差异，需同步完善。

# 需求分析（required）

## 需求描述

在 Atlas A2 / Atlas A3 系列 NPU（CANN 9.1.0）上，使用 Ascend C kernel 直调模式重构并优化 `blas/gemv_strided_batched/arch22/` 下的 5 个实数 StridedBatched GEMV 算子接口，并完善 `test/gemv_strided_batched/arch22/` 测试工程与 `blas/gemv_strided_batched/README.md` 文档：

- 完整支持 `S`（FP32→FP32）、`HSH`（FP16→FP16）、`HSS`（FP16→FP32）、`TST`（BF16→BF16）、`TSS`（BF16→FP32）5 个精度变体，内部统一在 FP32 精度域完成乘加与标量缩放，最终按输出 dtype 收窄；
- 数学语义、列主序物理布局（A 恒为 `lda × n`、`lda >= max(1, m)`）、16 参数签名与 cuBLAS 对应接口完全对齐（`trans` 非法枚举按 `ops-blas` 约定返回 `ACLBLAS_STATUS_INVALID_ENUM`）；
- 支持任意运行时非负维度 `m`/`n`/`batchCount`、带填充的 `lda`、正负非零步长 `incx`/`incy`、带间隙的正跨步 `strideA`/`stridex`/`stridey`、`alpha=0`/`beta=0` 未读语义及异步 stream 执行；
- 在 Atlas 800T A2（910B3）上完成 5 变体全量精度（5000 条）与性能（1000 条）自验，精度满足生态算子混合精度标准，性能经 `msprof op` 采集 kernel `Task Duration(us)` 达到 A100 cuBLAS kernel 标杆 0.8× 以上（即 `NPU_avg_us <= gpu_ms * 1000 / 0.8`）。

## 需求拆解

| 维度 | 拆解项 | 具体设计目标 |
| --- | --- | --- |
| 接口与精度变体 | 5 个句柄式 API | 复用 `include/cann_ops_blas.h` 声明，覆盖 `S / HSH / HSS / TST / TSS`；`alpha/beta` 恒为 Host FLOAT32 标量指针；FP16/BF16 以 `uint16_t` 承载 |
| 参数与安全校验 | 严格 10 步 Host 校验 | 按 `handle → trans → m/n → lda → incx/incy → strideA/stridex/stridey → batchCount → alpha/beta → A/x/y → quick return` 执行；64 位校验相邻 batch 重叠与寻址防溢出 |
| 特殊语义 | Quick Return 与零标量 | `batchCount=0` / `m=0` / `n=0` 校验通过后合法 quick return 且不启核；`alpha=0` 不读 A/x；`beta=0` 不读旧 y（防 NaN 污染） |
| 高性能并行 | 多维度动态分核 | 结合 `batchCount`、输出维度 `outDim` 与归约维度 `dotDim` 设计三级并行（Batch 并行、2D `Batch × OutTile` 并行、Split-K 两阶段并行），消除小 batch 欠并行 |
| 向量化与访存 | AIV 流水 + 跨步优化 | `N` 模式采用 2D 多列双缓冲搬运 + `Brcb/Mul/Tree-Add` 融合；`T/C` 模式采用 `x` 复用 + 多列批量 2D 搬运 + `WholeReduceSum` 批量归约；非单位/负步长采用 UB 窗口 Gather 与 2D 跨步 MTE3 写回 |
| 精度保障 | FP32 累加与安全舍入 | 16-bit 输入无损 `CAST_NONE` 转 FP32，树形/分组归约降低舍入误差，输出收窄使用 `CAST_RINT`（IEEE RNE），大 `alpha` 延后缩放避免中间溢出 |
| 工程与验证 | 测试、文档与交付 | 适配 5 变体 CSV / FP64 累加 golden / `msprof op` 与 `verify_performance.py` 交叉验证；同步更新 `README.md`；按模板组织 `task_submission` 交付件 |

# 详细设计（required）

## 算子分析

### 数学公式与布局约定

对每个批次 `b ∈ {0, 1, ..., batchCount - 1}`，记第 `b` 个 batch 的基地址（以元素偏移表示）为：

```text
A_b = A + b * strideA
x_b = x + b * stridex
y_b = y + b * stridey
```

运算定义为：

```text
y_b = alpha * op(A_b) * x_b + beta * y_b
```

矩阵 `A_b` 恒按**列主序（column-major）**存储，物理维度**恒为 `lda × n`**，且与 `trans` 无关（遵循 2026-11-24 任务书修订及 cuBLAS 13.5 / Netlib `sgemv` 语义）：`A_b(i, j) = A_b[i + j * lda]`（`0 <= i < m`，`0 <= j < n`，`lda >= max(1, m)`）。`trans` 仅决定逻辑运算形态：

| `trans` 取值 | `op(A_b)` 逻辑形状 | `x_b` 逻辑长度 `L_x` | `y_b` 逻辑长度 `L_y` | 归约长度 `K` | 逐元素公式（零基逻辑下标） |
| --- | --- | --- | --- | --- | --- |
| `ACLBLAS_OP_N` | `m × n` | `n` | `m` | `n` | `y_b[i] = alpha * sum_{j=0}^{n-1} A_b[i + j*lda] * x_b[j] + beta * y_b[i]` |
| `ACLBLAS_OP_T` | `n × m` | `m` | `n` | `m` | `y_b[j] = alpha * sum_{i=0}^{m-1} A_b[i + j*lda] * x_b[i] + beta * y_b[j]` |
| `ACLBLAS_OP_C` | `n × m`（实数等价 `OP_T`） | `m` | `n` | `m` | 同 `ACLBLAS_OP_T`（不执行共轭） |

### 5 变体接口与数据类型映射

5 个公开接口均声明于 `include/cann_ops_blas.h`（L311-L334）：

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);
aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t* A, int lda, int64_t strideA,
    const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

各变体数据类型与精度路径对照：

| 变体 | 内部 `dtype` 编码 | `A` / `x` 存储类型 | `y` 存储类型 | `alpha` / `beta` | 内部累加域 | 输出收窄方式 |
| --- | --- | --- | --- | --- | --- | --- |
| `S` | `1` (`FP32`) | `float`（FP32） | `float`（FP32） | Host `float` | FP32 | 无（直写 FP32） |
| `HSH` | `0` (`FP16_FP16`) | `uint16_t`（FP16） | `uint16_t`（FP16） | Host `float` | FP32 | `Cast(CAST_RINT)` 至 `half` |
| `HSS` | `2` (`FP16_FP32`) | `uint16_t`（FP16） | `float`（FP32） | Host `float` | FP32 | 无（直写 FP32） |
| `TST` | `3` (`BF16_BF16`) | `uint16_t`（BF16） | `uint16_t`（BF16） | Host `float` | FP32 | `Cast(CAST_RINT)` 至 `bfloat16_t` |
| `TSS` | `4` (`BF16_FP32`) | `uint16_t`（BF16） | `float`（FP32） | Host `float` | FP32 | 无（直写 FP32） |

### 零基寻址、负步长与跨步重叠分析

1. **正/负步长物理映射**：对逻辑长度 `L > 0`、步长 `inc != 0` 且 `inc != INT32_MIN` 的向量，先将 `inc` 提升为 `int64_t` 再取绝对值 `absInc = |int64(inc)|`，避免 `INT32_MIN` 取反溢出。单 batch 内逻辑元素 `t ∈ [0, L - 1]` 相对 `batchBase` 的物理偏移为：
   - 当 `inc > 0`：`physOff(t) = t * int64(inc)`；
   - 当 `inc < 0`（BLAS 反向遍历）：`startOff = (1 - int64(L)) * int64(inc) = (int64(L) - 1) * absInc`，`physOff(t) = startOff + t * int64(inc) = (int64(L) - 1 - t) * absInc`。
   - 单 batch 占用最小跨度（元素数）：`minSpanVec(L, inc) = (L == 0) ? 0 : (1 + (int64(L) - 1) * absInc)`。
2. **矩阵与相邻 batch 最小非重叠间距**：
   - 当 `m > 0 && n > 0` 时，单 batch 矩阵物理占用为 `minSpanA = int64(lda) * int64(n)`；当 `m == 0 || n == 0` 时，矩阵逻辑为空，`minSpanA = 0`。
   - 当 `batchCount > 1` 且对应张量逻辑非空时，为保证相邻 batch 区域不重叠，必须满足：
     - `strideA >= minSpanA`（`m > 0 && n > 0`）
     - `stridex >= minSpanVec(L_x, incx)`（`L_x > 0`）
     - `stridey >= minSpanVec(L_y, incy)`（`L_y > 0`）
   - 无论 `batchCount` 与 `m/n` 取何值，结构上始终要求 `strideA > 0`、`stridex > 0`、`stridey > 0`。
   - 对 `m=0, n=16, lda=1, strideA=1, batchCount=2` 这类合法 quick return 用例（对应 `TC_ED_191`），因 `m=0` 时矩阵无有效元素（`minSpanA = 0`），满足 `strideA > 0` 且无重叠，应合法返回 `SUCCESS` 而非误判重叠。
3. **64 位寻址防溢出**：由于 `strideA`/`stridex`/`stridey` 为 `int64_t`，在大规模高 batch 场景（如 `4096×4096` 多 batch）总元素偏移可超过 `INT32_MAX`，Host 校验与 Kernel 侧所有 GM 地址偏移及字节跨度均采用 64 位整数（`int64_t` / `uint64_t`）计算，防止 32 位截断回绕。
4. **特殊标量与未读保护语义**：
   - `alpha == 0.0f`：禁止读取 `A` 与 `x`（防止 `A`/`x` 含 `NaN`/`Inf` 时按 `0 * NaN = NaN` 污染结果），仅执行 `y_b = beta * y_b`；若同时 `beta == 1.0f`，`y` 逻辑保持不变，可不读 `A/x` 直接返回或跳过计算；若同时 `beta == 0.0f`，则不读 `A/x/y` 直接将 `y` 有效逻辑位置置 `0`。
   - `beta == 0.0f`：禁止读取 `y` 旧值（防止 `y` 旧值含 `NaN`/`Inf` 污染结果），直接写回 `alpha * op(A_b) * x_b`（或 `0`）。
   - `lda > m` 的列尾 padding、`stride` 间隙以及 `|incy| > 1` 时的向量间隙元素均不得参与计算，也不得被 Kernel 写坏。

## 算子实现

### Host 与 Tiling 侧设计

#### 1. 严格参数校验与执行顺序

Host 入口模板 `GemvStridedBatchedImpl<TIn, TOut>` 严格按照任务书 §2.5 规定的顺序逐项校验，确保任何结构参数非法均先于 quick return 被拦截：

| 步骤 | 校验条件 | 失败返回状态码 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C` | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `m < 0 \|\| n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `lda < std::max(1, m)`（与 `trans` 无关） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx == 0 \|\| incx == INT32_MIN \|\| incy == 0 \|\| incy == INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `strideA <= 0 \|\| stridex <= 0 \|\| stridey <= 0`；及（在第 7 步确认 `batchCount > 1` 后）相邻 batch 重叠检查与 64 位地址溢出检查 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `batchCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `alpha == nullptr \|\| beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | 当 `batchCount > 0 && m > 0 && n > 0` 时检查 `A == nullptr \|\| x == nullptr \|\| y == nullptr`（空维度时按 §2.5 允许 `A/x/y` 为空） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | 若 `batchCount == 0 \|\| m == 0 \|\| n == 0`，执行合法 quick return，不读设备数据、不启动 Kernel | `ACLBLAS_STATUS_SUCCESS` |

补充说明：

- Host 侧仅在校验通过且非 quick return 后解引用 Host 标量指针 `*alpha`、`*beta`，不对 Device 张量做元素级扫描，不引入额外 `aclrtSynchronizeStream`。
- 对 `batchCount > 1` 的跨步总偏移，在 Host 侧使用 64 位安全乘法检查 `(batchCount - 1) * stride + minSpan` 不发生 `int64_t` / 字节数溢出，超限时在发射前返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 通过 `PlatformAscendCManager::GetInstance()` 动态查询当前设备 AIV 核数（`GetAivCoreCount()`）与单核 UB 容量（`GetCoreMemSize(CoreMemType::UB)`），不硬编码特定型号核心数；查询失败返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。
- Kernel 发射前调用 `(void)aclrtGetLastError(ACL_RT_THREAD_LEVEL)` 清理线程遗留错误，发射后检查 `aclrtGetLastError(ACL_RT_THREAD_LEVEL)`，失败时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，避免假成功。

#### 2. 多级分核策略

记 `outDim = (trans == N) ? m : n`，`dotDim = (trans == N) ? n : m`，可用 AIV 核数为 `coreNum`（Atlas 800T A2 / 910B3 为 40 核，Atlas A3 / 910B1 等动态获取）。为同时兼顾高 batch 小矩阵（如 `batchCount=64/256`）、低 batch 大矩阵（如 `batchCount=1, 4096×4096`）和极窄输出长归约（如 `m=1, n=4096, batchCount=1`），设计三级分核策略：

1. **模式 0 — Batch 维均衡分核（`batchCount >= coreNum`）**：
   - 当 `batchCount` 足以填满全部 AIV 核（如 `batchCount=64, 256`）时，仅沿 `batch` 维切分，启用 `usedCoreNum = coreNum` 个核。
   - 采用大小核均衡分配（`base = batchCount / usedCoreNum`，`rem = batchCount % usedCoreNum`，前 `rem` 核处理 `base + 1` 个 batch，其余核处理 `base` 个 batch），替代旧实现 `ceil` + 偏小尾核的切分方式，显著降低尾核负载不均。
   - 单 Kernel 发射，零跨核通信与零 workspace 开销。
2. **模式 1 — 2D `(Batch × OutTile)` 联合分核（`batchCount < coreNum` 且常规输出维）**：
   - 将每个 batch 的输出维度 `outDim` 按对齐块大小 `outBlock`（`N` 模式按 32B/256B 对齐如 `64/128/256`；`T/C` 模式按 `8/16/32` 列组对齐）划分为 `outBlocks = ceil(outDim / outBlock)` 个独立输出任务。
   - 总独立任务数 `totalTasks = batchCount * outBlocks`，实际启用核数 `usedCoreNum = min(coreNum, totalTasks)`；小规模矩阵限制 `usedCoreNum` 避免空转核启动开销。
   - 核 `blockIdx` 通过循环或连续区间领取 `taskId ∈ [taskBegin, taskEnd)`，由 `batchIdx = taskId / outBlocks`、`outTileIdx = taskId % outBlocks` 解码任务。每个输出元素 `y[batchIdx, outIdx]` 仅由唯一核负责写入，**零原子加、零跨核同步、单 Kernel 完成**。
   - 示例：`batchCount=1, 4096×4096` 划分为 40 个输出块（每块 104 行或 104 列），直接跑满 40 个 AIV 核，充分吃满 HBM 带宽。
3. **模式 2 — Split-K 两阶段分核（`batchCount * outBlocks < coreNum` 且 `dotDim >= SPLIT_K_MIN_DOT`）**：
   - 针对 `m=1, n=4096, batchCount=1`（`TC_PF` 极窄输出长归约）或 `m=3, n=2049, batchCount=16`、`m=17, n=1024, batchCount=2` 等输出任务数远少于 `coreNum` 但归约维 `dotDim` 很长的场景，沿 `dotDim` 进一步切分为 `kSplits` 段（`2 <= kSplits <= min(MAX_K_SPLITS, coreNum / totalTasks)`，每段长度不小于最小向量块阈值如 `256`）。
   - 使用 `ops-blas` 句柄自带的 workspace 机制（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace`，禁止在热路径调用 `aclrtMalloc`/`aclrtFree`）申请 `batchCount * kSplits * outDim * sizeof(float)` 字节的 FP32 部分和缓冲区。
   - **Stage 1**：`usedCoreNum = totalTasks * kSplits` 个 AIV 核并行计算各 `(batchIdx, outTileIdx, kSliceIdx)` 的 FP32 部分和写入 workspace；
   - **Stage 2**：固定顺序 `s = 0 .. kSplits - 1` 累加部分和（保证确定性顺序与高数值精度），再一次性完成 `alpha * sum + beta * y_old` 与输出类型收窄并写回 `y`。
   - 空间不足或 `workspace` 申请失败时自动无错回退至模式 1（无额外 workspace 依赖）。
   - 实测阶段将同时评估“单 Kernel 内 `SyncAll` 融合两阶段（免除二次 Kernel 下发延迟）”与“两阶段独立 Kernel”的 `msprof op` 耗时；若采用双 Kernel，正式性能统计严格对同一次 API 调用的全部 Kernel 耗时求和。

#### 3. 数据分块与 UB 内存预算策略

Atlas A2 / Atlas A3（`arch22`）单 AIV 核 UB 容量为 192 KB（196,608 字节，运行时通过 `GetCoreMemSize(CoreMemType::UB)` 获取），要求所有 UB 缓冲区按 32B/256B 对齐且总和严格小于可用 UB 容量（预留系统栈与对齐余量，上限控制在 `<= 160 KB`）。

以双缓冲（`BUFFER_NUM = 2`）及最坏 16-bit 输入需要额外 FP32 Cast 缓冲为例，典型分块参数与 UB 预算核算如下：

| 路径 | 典型 Tile 参数（动态受 UB 约束） | 主要 UB 缓冲区明细（字节） | 单位步长合计 | 非单位步长合计（含 Gather/偏移表） |
| --- | --- | --- | --- | --- |
| `Trans=N` 向量主路径 | `mTile = 128`，`kBatch = 32` | `inA`(双缓冲): 32KB(FP32)/16KB(16b)；`a32`(仅16b): 16KB；`work`(广播/树归约): 16KB；`inX+x32`: 16KB；`acc+yIn+yOut`: 4KB | 约 68 KB | 约 92 KB（+24 KB 窗口与偏移表） |
| `Trans=T/C` 多列主路径 | `mTile = 512`，`cBatch = 16` | `inA`(双缓冲): 64KB(FP32)/32KB(16b)；`a32/prod`: 32KB；`inX+x32`: 4KB；`sums+acc+yIn+yOut`: 8KB | 约 108 KB | 约 132 KB（+24 KB 窗口与偏移表） |
| `Trans=T/C` 单趟常驻 `x` | `m <= 2048` 时 `mTile = AlignUp(m, 64)`，动态缩减 `cBatch` | `x32` 常驻 8KB；`inA`(双缓冲) + `prod` 动态匹配剩余 UB（`<= 128 KB`） | `<= 148 KB` | `<= 160 KB` |
| `alpha == 0` 缩放路径 | `yTile = 2048`（双缓冲） | `yIn`(双缓冲): 16KB；`y32`: 8KB；`yOut`(双缓冲): 16KB；`gatherWin`: 24KB | 约 40 KB | 约 64 KB |

Host 侧在 `CalTilingData` 中根据实际 `dtype`、`trans`、`m`、`n`、`incx/incy` 与平台 UB 容量自动求取合法的 `(outTile, dotTile, batchCols)`，保证在任何输入维度下均不发生 UB 越界。

#### 4. TilingData 与路径分派（TilingKey）规划

`GemvStridedBatchedTilingData` 通过值传递下发至设备核函数，字段全部采用显式定宽整数与 64 位步长/跨度：

```cpp
enum class GemvSbPath : uint32_t {
    ALPHA_ZERO_SCALE = 0U, // alpha == 0：仅执行 y = beta * y（或写 0），绝不读 A/x
    NORMAL_N_FAST    = 1U, // trans == N，incx == 1 && incy == 1，多列 2D DMA + Brcb/Tree-Add
    NORMAL_N_GENERAL = 2U, // trans == N，通用正负非单位步长（复用 2D A 搬运 + UB Gather/2D 写回）
    TRANS_T_FAST     = 3U, // trans == T/C，incx == 1 && incy == 1，x 复用 + 多列批量 WholeReduceSum
    TRANS_T_GENERAL  = 4U, // trans == T/C，通用正负非单位步长（x 复用 + UB Gather/2D 写回）
    SPLIT_K_STAGE1   = 5U, // 小 batch×outDim + 长 dotDim：Split-K 部分和路径
};
```

分派严格仅依据结构与标量属性（`dtype`、`trans`、`alpha == 0`、`beta == 0`、`m`、`n`、`lda`、`incx`、`incy`、`batchCount`），严禁依赖测试用例编号、随机种子或数据值分布。

### Kernel 侧设计

#### 1. 整体执行流水与 5 变体统一精度域

Kernel 采用标准 Ascend C `Init` + `Process` 架构，`TPipe` 在核函数入口作用域内创建。单核内遵循 `CopyIn (MTE2) → Compute (Vector) → CopyOut (MTE3)` 三段式流水，并以 `TQue<QuePosition::VECIN, 2>` / 精确 `HardEvent`（`MTE2_V`、`V_MTE2`、`V_MTE3`、`MTE3_V`、`MTE3_MTE2`）同步替代粗粒度 `PIPE_ALL`：

- **输入升精度**：`HSH`/`HSS`（`half`）与 `TST`/`TSS`（`bfloat16_t`）搬入 UB 后，通过 `Cast(dstFp32, src16, RoundMode::CAST_NONE, count)` 无损转为 FP32；`S` 变体直接以 FP32 参与向量运算，避免冗余 `Muls(..., 1.0f)` 拷贝。
- **FP32 乘加归约**：全部 5 个变体在 FP32 累加器 `acc` 中完成 `op(A) * x` 的点积/列加权累加；**先完成整个归约维度 `K` 的累加，最后在写回前统一乘 `alpha` 并合并 `beta * y_old`**，既规避 `alpha = ±1e4` 预乘导致的溢出风险，也符合 `alpha * (op(A) * x) + beta * y` 数值语义。
- **输出收窄**：`HSH`（→`half`）与 `TST`（→`bfloat16_t`）仅在最终写回前执行一次 `Cast(yOut, acc, RoundMode::CAST_RINT, count)`（IEEE 754 Round-to-Nearest-Even）；`S`/`HSS`/`TSS` 直接写回 FP32。

#### 2. `trans == ACLBLAS_OP_N` 向量化多列融合设计

列主序矩阵 `A` 的每列在 GM 中沿 `m`（输出维）连续。针对每个核领取的 `(batchIdx, mBegin..mEnd)` 任务，按 `mTile`（行块，按 8/16 元素对齐）循环处理：

1. 将当前 `mTile` 的 FP32 累加缓冲 `acc` 清零（`Duplicate(acc, 0.0f, mTileAligned)`）。
2. 沿列维 `n` 以 `kBatch`（如 16/32/64 列）为步长循环：
   - **2D 批量搬运 `A`**：使用单次 2D `DataCopy`（满足 32B 对齐且 `srcStride` 合法时）或 `DataCopyPad`（通用列长与任意 `lda`），一次将 `kBatch` 列、每列 `mCnt` 个有效行元素搬入双缓冲 UB 槽位，`srcStride = (lda - mCnt) * sizeof(TIn)`，UB 内每列按 32B 对齐补零（`rightPadding = mCntAligned - mCnt`，填充值 `0.0f`），确保 `lda` 列尾 padding 绝不参与计算。
   - **批量加载 `x`**：一次加载 `kBatch` 个逻辑 `x` 元素（单位步长直接 1D `DataCopyPad`，非单位步长经 UB Gather 打包）并转为 FP32。
   - **双缓冲重叠**：在 Vector 单元计算当前 `kBatch` 时，MTE2 异步预取下一批 `kBatch` 列的 `A` 与 `x`。
   - **纯向量融合乘加（参考仓内 `blas/gemv/arch22/sgemv_kernel.cpp` 已验证模式）**：
     - 当 `kBatch` 为 8 的倍数且 `kBatch >= 8` 时，通过 `Brcb` 将 `xTile` 中 `kBatch` 个标量广播展开为 `kBatch × mCntAligned` 矩阵，执行整块 `Mul(a32, a32, xBrcb, kBatch * mCntAligned)`，随后进行 `log2(kBatch)` 轮**二叉树归约 `Add`** 并累加至 `acc`——彻底消除逐列标量 `GetValue` 循环，同时将长链舍入误差由 `O(K)` 降至 `O(log2(K))`。
     - 当尾块 `kBatch` 较小（`< 8` 或非 8 倍数）时，使用 `Axpy(acc, a32[c * mCntAligned], xVal, mCntAligned)` 逐列快速收尾。
3. **统一 Epilogue**：全部列归约完成后执行 `Muls(acc, acc, alpha, mCntAligned)`；若 `beta != 0.0f`，加载旧 `y`（必要时升至 FP32）并执行 `Axpy` / `Muls + Add`；若 `beta == 0.0f`，严禁读取旧 `y`。最后按 `TOut` 收窄并写回有效 `mCnt` 元素。

#### 3. `trans == ACLBLAS_OP_T / ACLBLAS_OP_C` 多列复用与批量归约设计

列主序下 `A` 的每列沿 `m`（归约维）连续，天然契合转置点积访存。针对每个核领取的 `(batchIdx, cBegin..cEnd)` 输出列区间，按 `cBatch`（同时处理的输出列数，如 8/16/32 列）分组：

1. **外层归约分块 + `x` 高度复用**：
   - 若 `m` 可放入单次 UB（如 `m <= 2048`），将当前 batch 的逻辑向量 `x[0..m-1]` 仅从 GM 加载**一次**并转为 FP32 常驻 UB，复用于该核负责的所有输出列，彻底消除既有基线对 `x` 的 `n` 倍重复读取。
   - 若 `m` 较大（如 `m = 4095, 4096` 或更大），按 `mTile`（64 元素对齐，如 512/1024）沿 `m` 维切分，在每个 `mTile` 内加载一次 `x[mOff..mOff+mCnt-1]`，复用至当前 `cBatch` 全部列。
2. **2D 批量双缓冲搬运 `A`**：单次 2D `DataCopy` / `DataCopyPad` 搬入 `cBatch` 列的当前 `mCnt` 个连续元素，尾块超出 `mCnt` 的对齐填充位显式置 `0.0f`（同步将 `xTile` 对应尾部置 `0.0f`，防止 `0 * NaN` 需注意：有效长度外仅置零且不使用未初始化 UB 内存）。
3. **批量向量点积与 `WholeReduceSum` 归约**：
   - 将 `cBatch` 列 `a32[col * mTileStride]` 与 `x32` 逐列执行向量 `Mul`；
   - 使用两级 `WholeReduceSum`（每 repeat 处理 64 个 FP32 = 256B，单次指令 repeat 上限按硬件约束分段 `<= 248`）将 `cBatch` 列的归约结果一次性产出到连续 UB 缓冲 `colPartial[0..cBatch-1]`，彻底消除逐列 `ReduceSum` + `V_S` 标量停顿。
   - 若沿 `m` 维分多块，将各 `mTile` 的 `colPartial` 向量加至 `acc[0..cBatch-1]`（针对 FP32 长归约 `m >= 2048` 评估分块成对累加或 Kahan 补偿以对齐 FP64 golden 精度）。
4. **统一 Epilogue**：所有 `mTile` 归约完成后，在向量单元对 `cBatch` 个输出一次性执行 `alpha * acc + beta * y_old`、输出类型收窄与写回。

#### 4. `alpha == 0` 专用缩放路径

当 `alpha == 0.0f` 时进入 `ALPHA_ZERO_SCALE` 路径：

- **严禁访问 `A` 与 `x` 指针或发起任何 `A`/`x` DMA 搬运**（直接保障 `TC_ED_198`、`TC_ED_199` 中 `A`/`x` 为 `NaN` 时的语义正确性）。
- 若 `beta == 0.0f`：同时不读旧 `y`，按 `yTile` 直接向 `y` 的有效逻辑位置写入 `0`（同样不覆写 stride 间隙）。
- 若 `beta == 1.0f`：逻辑上 `y` 保持不变，Host 侧在确认参数合法后可直接返回（或按变体保持位模式不写回，保障 `TC_ED_194` 零耗时）。
- 若 `beta != 0.0f && beta != 1.0f`：沿 `(batch, outTile)` 满核并行搬入 `y` →（必要时升 FP32）→ `Muls(y32, y32, beta)` →（必要时 `CAST_RINT` 收窄）→ 写回 `y`。

#### 5. 非单位步长、负步长与非对齐尾块安全搬运

性能测试 200 条用例中约 79% 含非单位或负步长，本设计对 `LoadStridedVec` / `StoreStridedVec` 做分层向量化优化，并确保**绝不越界访问或覆盖 stride 空洞**：

| 步长类别 | `LoadStridedVec` 策略（GM → 连续 UB） | `StoreStridedVec` 策略（连续 UB → GM） |
| --- | --- | --- |
| `inc == 1` | 单次连续 1D `DataCopyPad` | 单次连续 1D `DataCopyPad`（按精确字节数 `count * sizeof(T)` 写入） |
| `inc == -1` | 单次连续 1D `DataCopyPad` 搬入反转窗口 + UB 预生成逆序字节偏移表 `Gather` 反转（零标量循环） | UB `Gather` 反转至连续缓冲 + 单次连续 1D `DataCopyPad` 写回 |
| 小跨步 `\|inc\| > 1` 且窗口 `span <= WIN_MAX` | 单次突发 `DataCopyPad` 搬入 GM 连续窗口至 `bufWin`，以向量化 `Adds` 生成字节偏移表后 `Gather` 打包为连续向量 | 若 `inc < 0` 先在 UB 内反转顺序，再通过 2D 跨步 `DataCopyPad`（`blockCount = count`, `blockLen = sizeof(T)`, `dstStride = (\|inc\| - 1) * sizeof(T)`）单条指令精确写回有效元素，严格保护间隙 |
| 超大跨步 `span > WIN_MAX` | 按 `DataCopyExtParams` 2D 跨步 `DataCopyPad` 分批搬入（`blockLen = sizeof(T)`，`srcStride = (\|inc\| - 1) * sizeof(T)`，`srcStride` 超 32 位上限时降级分块单元素搬运）；`inc < 0` 时 UB 内反转 | 同上，优先 2D 跨步 `DataCopyPad` 分批精确写回，`dstStride` 超 `uint32_t` 上限时降级安全单元素写回 |

对于所有非对齐尾块：GM→UB 使用 `DataCopyPad` 精确 `blockLen = validCount * sizeof(T)` 配合显式 `rightPadding` 清零，UB→GM 严禁扩大 `blockLen` 到 32B 对齐，防止写坏相邻列、相邻 batch 或 `y` 间隙保护哨兵。

#### 6. AIC Cube 混合路径评估边界

对于 `batchCount` 较小且矩阵较大（如 `batchCount ∈ {1, 2, 8}`、`m, n >= 256`、`incx == 1 && incy == 1`）的 `trans=N` 场景，参考 `ops-blas` 仓内 `blas/gemv/arch22/sgemv_kernel.cpp` 的经验，可在开发调优阶段评估基于 Ascend C `MatmulImpl`（将 `x^T` 视为 `1×K`、`A^T` 视为 `K×M`）的 AIC Cube 候选路径（FP16/BF16 原生 `Mmad` 内部 FP32 累加与 HSH/HSS/TST/TSS 语义天然一致）。但本设计坚持：

- **AIV 向量主路径作为全部 5 变体的通用基线与保底实现**，不假设 Cube 路径对所有输入都优于 AIV（GEMV 本质为内存带宽受限算子，满核 AIV 流水往往已可逼近 HBM 带宽上限）；
- 任何 Cube 候选路径仅在通过全量 5000 条精度测试（含极值、Inf/NaN 与 FP64 golden 严格校验）且经 `msprof op` 实证优于 AIV 路径时才纳入分派，绝不以降低计算精度换取速度。

### 工程组织与构建适配

严格遵循 `ops-blas` 仓目录约定与最小变更原则（不修改无关公共构建策略、不格式化无关公共头）：

| 内容 | 目标路径 | 规划说明 |
| --- | --- | --- |
| 公开接口声明 | `include/cann_ops_blas.h`（L311-L334） | 5 个接口均已存在，**无需修改** |
| arch22 Host 实现 | `blas/gemv_strided_batched/arch22/gemv_strided_batched_host.cpp` | 重构严格 10 步参数校验、64 位重叠/溢出保护、三级分核与 UB 动态 Tiling、5 个公开 API 入口 |
| arch22 Kernel 实现 | `blas/gemv_strided_batched/arch22/gemv_strided_batched_kernel.cpp` | 重构 AIV 多列双缓冲 N/T 向量化主路径、`alpha=0` 缩放路径、跨步 Gather/2D 写回、Split-K 路径及 `gemv_strided_batched_kernel_do` |
| arch22 内部头 | `blas/gemv_strided_batched/arch22/gemv_strided_batched_tiling_data.h`<br>`blas/gemv_strided_batched/arch22/gemv_strided_batched_kernel.h` | Tiling 结构与 launch 声明分离，头文件仅放声明/结构定义，不含全局 `using namespace` |
| 测试工程 | `test/gemv_strided_batched/arch22/`<br>`test/gemv_strided_batched/CMakeLists.txt` | 扩展/升级 arch22 测试工程：支持 5 变体 CSV 解析（含 `alpha/beta=null`、`NULLPTR`、`GAUSS_NORM_0_1`、非法 `trans=999`）、IEEE RNE 量化、FP64 累加 golden、输出 dtype 混合精度判定、`[PERF]` 事件计时输出及显式设备选择 |
| 算子文档 | `blas/gemv_strided_batched/README.md` | 核对并更新 Atlas A2 / Atlas A3 产品支持表、参数约束（含 2026-11-24 `lda >= max(1, m)` 与相邻 batch 非重叠语义）及 arch22 路径说明 |

构建与单算子选择核查：`ops-blas` 顶层 `blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动收集 `blas/*/arch22/*.cpp`，`test/gemv_strided_batched/CMakeLists.txt` 已调用 `ops_blas_add_gtest_tests`。开发阶段将重点核查 `bash build.sh --ops=gemv_strided_batched --soc=ascend910b3` 以及任务脚本使用的 `--ops={sgemv,hsh_gemv,hss_gemv,tst_gemv,tss_gemv}_strided_batched` 映射行为，在保持全仓构建与 `arch35` 编译兼容的前提下最小化适配构建规则。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas A2 训练系列产品（含 Atlas 800T A2 / 910B3） | √ | `arch22` 实现；计划作为本任务精度与性能主自验平台 |
| Atlas A2 推理系列产品 | √ | 共享 `arch22` 实现，动态查询核数与 UB 容量 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ | 共享 `arch22` 实现，动态查询核数与 UB 容量（任务书仅要求覆盖 A2/A3 之一自验） |
| Ascend 950PR / Ascend 950DT | √（既有 `arch35` 保持不变） | 仓内 `arch35` 已有实现，本任务不改动其逻辑并保持兼容 |
| Atlas 200I/500 A2 及历史训练/推理系列（如 310P） | × | 不在本任务适配范围 |

## 算子约束限制

逐项对照任务书 §2.5 实现约束：

| 约束项 | 本设计处理方式 |
| --- | --- |
| 参数合法性与校验层级 | Host 入口严格按 10 步顺序完成全部结构参数、标量指针与基址校验，不对 Device 张量做元素级扫描 |
| stride 合法性与重叠 | 严格要求 `strideA > 0`、`stridex > 0`、`stridey > 0`；当 `batchCount > 1` 且逻辑非空时，64 位校验 `strideA >= lda * n`、`stridex >= 1 + (L_x - 1)*\|incx\|`、`stridey >= 1 + (L_y - 1)*\|incy\|`，不满足返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 非连续 Tensor 支持 | 支持 `lda >= max(1, m)` 列主序填充、任意非零且非 `INT32_MIN` 的正负 `incx/incy`、带间隙正 `stride`，严格保护所有填充与间隙区域不被修改 |
| broadcast 规则 | 不支持广播，每个 batch 为独立操作数 |
| dynamic shape 要求 | 不涉及图模式动态 shape，所有尺寸、步长与 batchCount 均为运行时标量入参，Host 侧动态完成 Tiling |
| 原地与视图语义 | `y` 原地覆写，不返回视图；调用方保证 `y` 可写区与 `A`/`x` 可读区不重叠，且各 batch 的 `y` 区域互不重叠 |
| 确定性计算要求 | 不要求逐位确定性；主路径与 Split-K 规约均采用固定顺序无原子加设计，在满足吞吐的同时提升复现稳定性，验收以 §3.2 精度阈值为准 |
| 空 Tensor 处理 | 结构参数校验通过后，`batchCount=0`、`m=0` 或 `n=0` 直接返回 `ACLBLAS_STATUS_SUCCESS`，允许 `A/x/y` 为 `nullptr`，不访问 Device 数据、不启核 |
| 0 维 Tensor 处理 | 不涉及，矩阵与向量均由 Device 基址配合维度/步长/跨步描述，`alpha/beta` 为 Host 标量指针 |
| 异步执行 | 严格使用 `handle` 绑定的 `aclrtStream` 异步下发 Kernel，不在算子内部插入强制流同步；Host 读回结果前由调用方同步该 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 具体要求与口径 | 标准来源 |
| --- | --- | --- |
| 精度 Golden | CPU 侧按 Netlib `cblas_sgemv` 逐 batch 数学语义实现：将（经对应输入 dtype 量化后的）`A`/`x`/`y` 及 `alpha`/`beta` 提升为 `float64` 累加计算，最后一次性按输出 `y` 的 dtype（FP32 / FP16 / BF16）收窄；同时保留 `cblas_sgemv` FP32 参考对照，若两者存在差异在自测报告中说明 | 任务书 §3.2 |
| 精度阈值（`S`/`HSS`/`TSS`，输出 FLOAT32） | `rtol = atol = 2^-13`（约 `1.2207e-4`），`required_matched_ratio >= 0.99`，`max_abs_error <= 1e-2 or 32 * ULP`；`Inf/NaN` 非有限值单独校验且全部一致 | 任务书 §3.2、生态算子混合精度标准 |
| 精度阈值（`HSH`，输出 FLOAT16） | `rtol = atol = 2^-10`（约 `9.7656e-4`），`required_matched_ratio >= 0.99`，`max_abs_error <= 1e-1 or 32 * ULP`；`Inf/NaN` 传播校验通过 | 任务书 §3.2、生态算子混合精度标准 |
| 精度阈值（`TST`，输出 BFLOAT16） | `rtol = atol = 2^-7`（约 `7.8125e-3`），`required_matched_ratio >= 0.99`，`max_abs_error <= 1e-1 or 32 * ULP`；`Inf/NaN` 传播校验通过 | 任务书 §3.2、生态算子混合精度标准 |
| 正式性能判据（Kernel-to-Kernel） | 测试设备 Atlas 800T A2（910B3），使用 `msprof op` 采集单算子 kernel `Task Duration(us)`；warmup 5 次不计入，后续连续 10 次有效样本取算术平均（异常调用不计入并单独说明）；逐 case 满足 `NPU_avg_us <= gpu_ms * 1000 / 0.8`（即达到 A100 cuBLAS `gpu_ms` 的 0.8× 以上性能），且全量平均倍率 >= 0.8；多 Kernel 路径须按单次 API 调用对全部 Kernel 耗时求和 | 任务书 §3.3、§7 |
| 辅助性能交叉验证 | GTest 在预置 Device 缓冲上 warmup 5 次 + `aclrtEvent` 连续 10 次下发取平均，打印 `[PERF] case=<case_name> ... avg_us=<数值>`，供 `verify_performance.py` 快速回归与交叉验证；该口径含下发开销，**不得替代 `msprof op` 正式判据** | 任务书 §3.3、随包 README |
| 内存验收要求 | 不涉及（仍通过哨兵测试验证无越界写、无 stride 空洞覆写、workspace 有界释放） | 任务书 §3.4 |

## 测试方案与用例覆盖

### 官方自验包全量覆盖（5000 精度 + 1000 性能）

保留 5 个变体原始 CSV、200 行 `gpu_baseline.csv` 与验证脚本原件不变（记录 SHA-256 哈希），全量执行不得删减或筛选：

- **规模分布**：实核 5 变体 CSV 原件，每变体均含 1000 条精度用例（`TC_L0` 6 条、`TC_L1` 62 条、`TC_L2` 69 条、`TC_L3` 28 条、`TC_L4` 14 条、`TC_L5` 10 条、`TC_ED` 39 条含 25 条负向、`TC_EX` 772 条）+ 200 条性能用例（`TC_PF`，含 25 条非默认 `alpha/beta` 标量、167 条含奇数 `m`/`n`、158 条非单位/负步长、`batchCount ∈ {1,2,3,8,16,64,256}`、覆盖至 `4095/4096` 大尺寸）；5 变体合计 **5000 条精度 + 1000 条性能**。
- **数据分布**：均匀分布 `RANDOM_NORM_5_5`（`[-5, 5]`）约 50% + 正态分布 `GAUSS_NORM_0_1`（`N(0, 1)`）约 50%，叠加零值、正负交替、极值、`Inf` 传播及未读 `NaN` 场景。
- **关键负向与边界**：覆盖 `handle=nullptr`、`trans=999`、`m/n/batchCount < 0`、`lda=0` 及 `trans=N/T` 下 `lda < m`、`incx/incy = 0` 与 `INT32_MIN`、`strideA/stridex/stridey <= 0`、相邻 batch 重叠（`TC_ED_220`~`TC_ED_222`）、`alpha/beta/A/x/y` 空指针，以及 `batchCount=0` / `m=0` / `n=0` quick return、`alpha=0`（A/x 为 NaN）、`beta=0`（y 为 NaN）。

### 补充契约与健壮性单测

在官方 CSV 之外补充独立单元测试，重点覆盖：

1. **校验优先级交叉用例**：`m=0` / `n=0` / `batchCount=0` 与非法 `trans=999`、非法 `lda`、非法 `incx=0`、非法 `stride <= 0`、`alpha=nullptr`、`handle=nullptr` 的组合，验证结构非法一律优先报错而不会误走 quick return。
2. **保护区与越界哨兵测试**：对 `lda > m` 的列尾 padding、`strideA/stridex/stridey` 的 batch 间隙、`|incy| > 1` 的元素间隙写入特定哨兵值（含 `NaN` 毒化可读 padding），验证计算结果不受 padding 影响且 `y` 的所有间隙字节保持原样不变。
3. **非对齐与大跨步尾块回归**：针对 `m/n` 覆盖 `1, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 255, 257, 4095` 及 `incx/incy ∈ {±1, ±2, ±3, ±7, ±8, ±16, ±31, ±64, ±127}` 交叉验证，核查 2D DMA 与跨步写回无越界。
4. **Split-K 边界与设备选择**：覆盖 `m=1, n=4096`、`m=3, n=2049` 等 Split-K 触发与非触发临界尺寸；测试入口支持通过环境变量显式指定 NPU 设备（`ASCEND_DEVICE_ID` / `ASCEND_RT_VISIBLE_DEVICES`），在设备分配前完成设置，避免默认抢占 0 号卡。

## 任务材料差异与处理原则

前期逐文件核查任务书、随附 5 变体 `test_cases/` 与 `ops-blas` 基线（`55ef791`）发现以下材料差异，开发中按如下原则透明处理：

1. **`ops-blas` 既有 `arch22` 现状差异**：任务书 §2.2 称当前仅有 `arch35/` 需新建 `arch22/`，而上游 `55ef791` 已存在初版 `blas/gemv_strided_batched/arch22/` 与 `test/gemv_strided_batched/arch22/`。本任务在其目录原位升级重构，消除上述校验顺序、重叠检测、分核欠并行与跨步性能缺口，保持文件组织精简规范。
2. **测试构建 `--ops` 映射与可执行名差异**：随包 5 份 `verify_accuracy.py` / `verify_performance.py` 分别使用 `--ops={sgemv,hsh_gemv,hss_gemv,tst_gemv,tss_gemv}_strided_batched` 及对应 `<op>_test` 二进制名，而当前 `ops-blas` 仓仅在 `test/gemv_strided_batched/CMakeLists.txt` 注册了统一的 `gemv_strided_batched_test`（`test/CMakeLists.txt` 去首字母匹配仅能识别单字母前缀 `sgemv_strided_batched`，无法直接识别 `hsh_`/`hss_`/`tst_`/`tss_`）。开发时通过最小范围适配测试目录/目标命名（或提供 5 变体入口复用公共测试模板），在不改动全仓编译策略的前提下保证仓内默认测试与官方验证脚本均可直接运行。
3. **CSV 列名与分布 token 差异**：任务书 §3.5 表格写作 `RANDOM_GAUSS_0_1`，实际 CSV 及测试 README 使用 `GAUSS_NORM_0_1`；现有仓内 `gemv_strided_batched_param.h` 读取 `A/x/y` + `dtype` 列，而官方 5 变体 CSV 使用 `aarray_fill/xarray_fill/yarray_fill` 且按文件区分变体、不含 `dtype` 列，并支持 `alpha/beta=null`。测试解析器需同时兼容仓内格式与官方 5 变体 CSV 格式，不擅自修改官方 CSV 原件。
4. **`gpu_baseline.csv` 备注与实测内容差异**：随包 README 说明“新增非默认标量用例待复测”，任务书 §9（2026-11-24）指出 T/C 行 `lda` 修订后原实测值需核对；实查 5 份 `gpu_baseline.csv` 当前均为 200/200 行非空（含 25 条非默认标量行）。开发与验收阶段严格保留随包官方 `gpu_baseline.csv` 原件及哈希，若任务方后续发布更新基线则按更新版复核，绝不私自修改基线数值。
5. **交付目录格式差异**：任务书 §4 表格提及 `cases.json` / `facts.json`，实际下发自验包为 CSV + `gpu_baseline.csv` + Python 脚本；自验与交付按实际随包结构与 `task_submission` 7 文件模板执行，并在 `1 自验证步骤说明.md` 中完整记录版本、哈希与命令以保证可复现。

## 兼容性分析

- **ABI 与跨架构兼容**：完全沿用 `include/cann_ops_blas.h` 既有 5 个函数签名及 `include/cann_ops_blas_common.h` 枚举与状态码，不引入私有平行接口；不修改 `blas/gemv_strided_batched/arch35/` 与 `test/gemv_strided_batched/arch35/`，确保 Ascend 950PR/950DT 编译与功能不受影响。
- **向前/向后兼容**：严格校验顺序与相邻 batch 重叠检查对齐任务书 §2.5 及 2026-11-24 修订口径；`lda >= max(1, m)` 规则与 `arch35` 及 `gemv_strided_batched_golden.h` 保持完全一致。
- **可维护性**：5 个精度变体共享同一套 Host 校验、Tiling 规划与 C++ 模板化 Kernel 主干，通过模板特化隔离 FP32 / FP16 / BF16 存取与 Cast 差异，避免复制 5 份近似代码引发 CodeCheck 重复告警与维护分裂。

## 风险分析与交付计划

| 潜在风险 | 影响 | 应对与控制措施 |
| --- | --- | --- |
| 小尺寸用例（如 `1×1`、`15×15`）Kernel 启动与标量准备开销接近 A100 `8~10 us` 基线 | 导致小矩阵性能达标困难 | 优先单 Kernel 零 `SyncAll` 发射，精简 Tiling 解码与分支，避免小形状过度分核及不必要 UB 初始化 |
| 性能子集中 79% 为非单位/负步长用例 | 逐元素标量 Gather/Scatter 会造成性能断崖式下降 | 采用突发窗口 + UB `Gather` 与 2D 跨步 `DataCopyPad` 写回，并按 `msprof op` 热点验证优化收益 |
| FP32 长归约（`4096`）或极值/大 `alpha` 场景误差放大 | 超过 `rtol=atol=2^-13` 或 `max_abs_error` 限值 | 采用树形归约降低累加深度，归约后统一 `alpha` 缩放；必要时对 FP32 长归约评估分段补偿累加，严格以 FP64 golden 检验 |
| `DataCopyPad` 在非对齐尾块或大 `lda` 跨度下的重载单位误用 | 引发地址越界或精度错误 | 逐方向核对 CANN 9.1.0 `arch22` 实际接口字节/块单位，配合哨兵测试在 910B3 真机验证 |

**开发与交付计划**：

1. **阶段一（设计评审）**：提交本设计文档 PR 至 `cann/cann-ops-competitions` 并完成 CLA、CI 与评审合入。
2. **阶段二（核心实现与功能自验）**：在 Atlas 800T A2（910B3，CANN 9.1.0）完成 `blas/gemv_strided_batched/arch22/` 重构与 `test/gemv_strided_batched/arch22/` 适配，通过 5 变体全部 5000 条精度用例及补充边界/哨兵单测。
3. **阶段三（性能剖析与调优）**：基于 `msprof op` 逐条采集 5 变体 1000 条 `TC_PF` 用例的 kernel `Task Duration(us)`，定位热点迭代优化至全部达标，并运行 `verify_performance.py` 完成辅助交叉验证。
4. **阶段四（验收交付与源码 PR）**：在个人私仓根目录按官方模板整理 `task_submission/` 7 项交付件（`1 自验证步骤说明.md`、`2.1 精度自验证报告.xlsx`、`2.2 精度自验证日志.log`、`3.1 性能自验证报告.xlsx`、`3.2 性能自验证日志.log`、`4.1 内存自验证报告.xlsx`、`4.2 内存自验证日志.log`，其中内存两项明确标注“不涉及”不伪造数据），邀请 `Ascend-CANN` 为开发者并提交平台验收；验收通过后向 `cann/ops-blas` 提交仅含源码、测试与长期文档的洁净 PR（不混入一次性验收报告或临时脚本）。

## 参考资料

1. [CANN 社区任务设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
2. [CANN 社区任务 2026 参与说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
3. [ops-blas 开源仓及贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)
4. [生态算子开源精度标准（Mixed Tolerance Standard）](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
5. [NVIDIA cuBLAS 文档（gemvStridedBatched）](https://docs.nvidia.com/cuda/cublas/index.html)
6. [Netlib BLAS `sgemv` 参考实现](https://www.netlib.org/blas/sgemv.f)
