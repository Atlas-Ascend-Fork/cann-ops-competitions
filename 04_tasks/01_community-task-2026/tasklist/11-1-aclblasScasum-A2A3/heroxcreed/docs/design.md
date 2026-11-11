# 需求背景（required）

## 需求来源

- 任务来源：2026 年 11 月 CANN 社区任务「aclblasScasum 算子开发（A2/A3）」，HiAscend 任务广场认领，任务截止 2026/11/19。
- 合入目标：昇腾算子开源仓 `https://gitcode.com/cann/ops-blas`，算子目录 `blas/asum/arch22/`，测试目录 `test/asum/scasum/arch22/`。
- 任务书：`aclblasScasum_AtlasA2_A3_task_doc.md`（本设计文档的唯一验收依据来源）。

## 背景介绍

### aclblasScasum 算子实现优化

> **说明（回应模板「原实现分析」）**：本算子在仓内**无前序 TBE 实现**，也不属于"已有算子优化"；因此下文以「同族实数实现 aclblasSasum 的现状分析 + 复用决策」替代传统旧版剖析，不虚构不存在的 TBE 版本。

ops-blas 仓 `blas/asum/` 目前只有实数版向量绝对值求和 `aclblasSasum`（`arch22` 直调实现 + `arch35` 版本），单精度复数版 `aclblasScasum` 缺失。本任务基于仓内既有 `arch22` 直调框架与同族实数实现，使用 Ascend C 编程语言补齐 `aclblasScasum`，与 cuBLAS `cublasScasum` / Netlib BLAS `scasum` 语义对齐。

**接口现状（已确认，不新增平行接口）**：`include/cann_ops_blas.h` 第 239 行已声明

```cpp
aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);
```

该声明与 950PR 共用，本任务直接复用，不得定义私有平行接口。复数类型 `aclblasComplex` 以 `include/cann_ops_blas_common.h` 定义为准（实部/虚部各 `float32`，交错存储，n 个复数元素对应 2n 个 float）。

**参考实现路径**：

- 对标基线：cuBLAS `cublasScasum`（语义参考 Netlib `scasum.f`，`https://www.netlib.org/blas/scasum.f`）。
- 精度 golden：cblas（Netlib BLAS 3.10.0）`cblas_scasum`，随任务测试工程提供，不得用自写 FP64 golden 替代。

### aclblasSasum（同族实数实现）现状分析

`blas/asum/arch22/sasum_host.cpp` 与 `blas/asum/arch22/sasum_kernel.cpp` 是实现本算子的直接参照，其现状能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄（携带 stream） | scalar | - | 非空，空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | - |
| n | 向量元素个数 | scalar | int | 无（void 实现未做 quick return 校验） | - |
| x | 输入向量（Device 内存，只读） | tensor | FLOAT32 | 物理长度 1+(n-1)\|incx\| | [n] |
| incx | 元素步长 | scalar | int | 无 | - |
| result | 输出标量（Device 内存） | scalar | FLOAT32 | 单值 | - |

实现要点（复用基础）：

1. **接口层**：`aclblasSasum` 校验 handle 非空 → 取 `handle->stream` → 计算 tiling → `aclrtMalloc` 一块 tiling Device 内存并 `aclrtMemcpy`（H2D）→ 调 `sasum_kernel_do(...)` 以 `<<<numBlocks, nullptr, stream>>>` 直调 kernel → `aclrtSynchronizeStream` → `aclrtFree`。
2. **tiling 下发方式**：`arch22` 走「tiling 结构体放 Device，GM 指针下发」，kernel 侧用 `ParseTilingData(GM_ADDR tilingGm)` 按固定偏移手工解析；`arch35` 走「tiling 按值传入」。本任务沿用 `arch22` 方式。
3. **tiling 结构（arch22 实际定义在 host.cpp 内）**：`AsumTilingData { uint32_t n; int32_t incx; uint32_t useCoreNum; uint32_t startOffset[40]; uint32_t calNum[40]; }`，以 32B（=8 个 float 元素）为一块做核间均分，余数块与尾元素归到最后一个核心。
4. **kernel 结构**：`Init → Process(CopyIn → Compute → CopyOut)`；`Compute` 为 `Abs` + `ReduceSum`；`Process` 首尾分别 `SetAtomicAdd<float>()` / `SetAtomicNone()`，多核通过原子加汇总到单一 `outGM`；`Init` 内先把 `outGM` 置 0 并 `SyncAll()`。
5. **尾块处理**：非 8 元素对齐的尾块用 `DataCopyPad`（补 0）或先对齐到 8 再计算；`incx > 1` 时退化用 `SetValue/GetValue` 逐元素搬运。
6. **现状缺口（本任务要补）**：`aclblasSasum` 未做 `n <= 0` / `incx <= 0` / `x` 空指针 / `result` 空指针等校验；`test/asum/scasum/CMakeLists.txt` 当前只在 `arch35` 分支注册 GTest（`ops_blas_add_gtest_tests`），A2/A3 构建会退回非 GTest runner（输出整数毫秒）。

### aclblasScasum 算子功能分析

- 计算公式：`result = Σ (|real(x[k])| + |imag(x[k])|)`，其中 `i = 1..n`，`k = 1 + (i-1) * incx`（1-based 索引，兼容 Fortran）。
- 语义要点：**取的是分量模（|Re| + |Im|），不是复数模长 `sqrt(Re² + Im²)`**。
- 输入：单精度复数向量 `x`（complex64，Device 内存，只读），物理长度 `1+(n-1)*|incx|` 个复数元素。
- 输出：单个 FLOAT32 实数标量（Device 内存）。
- 累加说明：`|Re| + |Im|` 逐项非负累加；元素接近 `FLT_MAX` 且 n 较大时结果可溢出到 `Inf`，属允许行为。
- quick return：`n <= 0` 或 `incx <= 0`（含 0 与负步长）时不遍历向量、不触发 kernel，直接写 `result = 0.0f` 并返回成功。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言、以 Ascend C kernel 直调方式在 `ops-blas` 仓 `arch22`（Atlas A2/A3）路径实现 `aclblasScasum` 句柄式 BLAS 接口：通过 handle 绑定 stream 直调 NPU kernel，完成单精度复数向量「分量模之和」归约到 FLOAT32 标量的计算；补齐 `arch22` 测试工程与 GTest 注册；精度、性能、用例覆盖满足任务书 §3 要求。

## 需求拆解

1. **接口层**：复用 `include/cann_ops_blas.h` 第 239 行声明；实现放在 `blas/asum/arch22/`（参考同目录 `sasum_host.cpp` 扩展复数路径）。
2. **参数校验与 quick return（严格按任务书 §2.4/§2.5 顺序）**：
   1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   2. `result == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`（任意 n，**含 quick return 路径**）；
   3. `n > 0 && incx > 0` 且 `x == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
   4. `n > 0 && incx > 0` 且 `(int64_t)(n-1) * |incx| > 2147483647` → `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算）；
   5. `n <= 0` 或 `incx <= 0` → 不触发 kernel、不遍历 `x`，写 `result = 0.0f`，返回 `ACLBLAS_STATUS_SUCCESS`；
   6. 元素值不做校验（`Inf/NaN` 合法输入）。
3. **计算实现**：分量模求和，归约累加顺序不要求与参考实现逐位一致；不得遗漏 `incx` 步长语义。
4. **精度**：按任务书 §3.2，FLOAT32 标量比对，`atol = rtol = 2^-13`，且 `max_abs_error ≤ 1e-2 或 32×ULP`；标量场景 `matched_ratio` 退化为单值判定；`Inf/NaN` 按等效规则判定（同为 NaN、同为同号 Inf，或一方 NaN 一方 ±Inf 视为等效；一方有限另一方 Inf/NaN 判失败）。
5. **性能**：按任务书 §3.3，性能设备 Atlas 800T A2（910B3），平均单次耗时不高于 GPU 标杆（`gpu_baseline.csv`，A100 实测）的 1.25 倍；采样口径见下方「性能测试方法」。
6. **测试工程**：用例与测试代码放 `test/asum/scasum/arch22/`，CSV 列格式对齐 `test/asum/sasum/sasum_param.h`（`n` / `incx` / `x` 填充模式 / `result` 空指针标记）；并在 `test/asum/scasum/CMakeLists.txt` 的 **arch22 分支**补 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)`。
7. **用例覆盖**：任务书 §3.5 要求的尺寸扫描（0/1/小质数/2 的幂±1/≥2^20）、步长、填充模式、边界与负向、`Inf/NaN`、性能与内存用例；随任务提供的 `RANDOM_NORM_5_5` 是均匀分布，**正态分布（`GAUSS_NORM_5_2`）用例须自行补充**。
8. **交付件**：设计文档 PR（cann-competitions）、自测用例与测试代码（含复现 readme）、自测报告、待验收代码地址（个人仓 + 邀请 `Ascend-CANN` 为开发者 + 算子 README 产品支持表补 `aclblasScasum`，标注 Atlas A2/A3 支持）、私仓 `task_submission/` 七个规定命名文件。

# 详细设计（required）

## 算子分析

### 数学公式

```
result = Σ_{i=1..n} ( |real(x[1+(i-1)*incx])| + |imag(x[1+(i-1)*incx])| )
```

复数按 `float32` 实部/虚部交错存储，故 n 个复数元素对应内存中 2n 个 float；`incx = 1` 时 2n 个 float 连续，`incx > 1` 时第 i 个复数元素的两个分量位于 `2*(i-1)*incx` 与 `2*(i-1)*incx + 1`。

### 支持数据类型

| 输入 x | 输出 result | 计算类型 |
| --- | --- | --- |
| COMPLEX64（实/虚各 FLOAT32，交错） | FLOAT32 标量 | FLOAT32 |

### 支持形状

- 逻辑一维 `[n]`，物理长度 `1 + (n-1) * |incx|` 个复数元素。
- 不要求非连续 Tensor 支持（非连续由 `incx` 表达）；不涉及 broadcast；不要求 dynamic shape（n 为运行时入参）。

## 算子实现

### 实现方案

数据流（流程图）：

```
                host                                   device
┌──────────────────────────────┐        ┌──────────────────────────────────────┐
│ aclblasScasum(handle,n,x,incx│        │                                      │
│              ,result)        │        │  kernel<<<numBlocks, nullptr, stream>>> │
│  1 参数校验 / quick return    │        │   ┌────────── Init ──────────┐        │
│  2 计算 tiling（按 n 切分核） │  tiling│   │ ParseTilingData(GM)      │        │
│  3 aclrtMalloc + H2D tiling  │───────►│   │ outGM 置 0 + SyncAll     │        │
│  4 直调 kernel               │        │   └──────────────────────────┘        │
│  5 aclrtSynchronizeStream     │        │   ┌──── 每个 AIV 核循环 ─────┐        │
│  6 aclrtFree tiling          │        │   │ CopyIn  GM→UB (分块)     │        │
└──────────────────────────────┘        │   │ Compute Abs → ReduceSum  │        │
                                        │   │ CopyOut UB→outGM 原子加  │        │
                                        │   └──────────────────────────┘        │
                                        └──────────────────────────────────────┘
```

#### host 侧设计（`blas/asum/arch22/scasum_host.cpp`）

1. **参数校验顺序**（严格按任务书）：
   ```cpp
   if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
   if (result == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;           // 任意 n，含 quick return
   const bool positive = (n > 0 && incx > 0);
   if (positive && x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
   if (positive && (int64_t)(n - 1) * std::abs((int64_t)incx) > 2147483647LL)
       return ACLBLAS_STATUS_INVALID_VALUE;                              // host 侧 int64
   if (!positive) { /* 走 stream 写 0.0f，返回 SUCCESS */ }
   ```
2. **quick return 路径**：不启 kernel、不遍历 `x`；在 handle 绑定的 stream 上以异步方式把 `0.0f` 写到 `result`（4 字节，不写相邻内存），返回 `ACLBLAS_STATUS_SUCCESS`。调用方负责在读取 `result` 前同步 stream。
3. **tiling 策略**：以复数元素个数 `n` 为切分长度，沿用 `arch22` 的「32B 为一块」粒度（32B = 8 个 float = 4 个 complex 元素）。
   - `totalBlockNum = n / 4`，`remainNum = n % 4`；
   - `totalBlockNum == 0`：单核处理 `remainNum`；
   - `totalBlockNum <= vecCoreNum`：每核一块，余数并入最后一核；
   - 否则按 `blockNumEachCore = totalBlockNum / vecCoreNum` 均分，前 `remainBlock` 个核多分一块，`remainNum` 并入最后一核；
   - 核数上限与 `startOffset[]` / `calNum[]` 数组长度沿用 arch22 同族实现的 40（实际使用核数取 `min(核数上限, GetBlockNum())`）。
4. **tiling 下发**：`ScasumTilingData { uint32_t n; int32_t incx; uint32_t useCoreNum; uint32_t startOffset[40]; uint32_t calNum[40]; }` 放 Device 内存，按字节偏移传给 kernel（与 `sasum_host.cpp` 一致）。
5. **stream 与内存**：计算走 `handle->stream`；tiling Device 内存申请后释放；`result` 严格 4 字节；调用方同步后读回。无额外 workspace 需求（跨核汇总用原子加，`workSpace` 传 `nullptr`）。

#### kernel 侧设计（`blas/asum/arch22/scasum_kernel.cpp`）

Init → Process（CopyIn → Compute → CopyOut），`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，双缓冲 `BUFFER_NUM = 2`：

1. **Init**：`ParseTilingData`（按固定偏移从 GM 读 n / incx / useCoreNum / 本核 startOffset / calNum）；`inGM` 按物理长度 `1+(n-1)*|incx|` 复数元素（即 2 倍 float 长度）建 `GlobalTensor`；`outGM` 长度 1；把 `outGM` 置 0 后 `SyncAll()`，保证原子累加前输出已清零；初始化 `workBuf`（`ReduceSum` 所需临时空间）与 `inQueue`/`outQueue`（按 UB 容量的分块大小）。
2. **CopyIn（分块）**：
   - `incx == 1`（连续快路径，性能用例全部命中）：把输入视作 `float` 的 `2n` 长连续数组，用 `DataCopy`/`DataCopyPad` 一次搬入 UB；单调块大小取 UB 可容纳值（沿用同族实现 80KB 量级），尾块补 0 对齐到 8 元素，补 0 不影响非负累加结果。
   - `incx > 1`（功能用例）：逐复数元素 gather `(2*i*incx, 2*i*incx+1)` 两个分量写入 UB（`SetValue/GetValue`），再走同一 `Abs + ReduceSum` 流程；尾元素同样补 0 对齐。
3. **Compute**：`Abs(in, in, count)` + `pipe_barrier(PIPE_V)` + `ReduceSum(out, in, work, count)`；对 `float` 视图而言 `|Re|`、`|Im|` 都在同一 `Abs` 内完成，等价于逐项 `|Re|+|Im|` 累加，无需额外复数拆分指令。
4. **CopyOut**：每核只写一标量，`Process` 内 `SetAtomicAdd<float>()` / `SetAtomicNone()` 包裹，使多核结果原子累加到 `outGM`（4 字节），不写相邻内存。
5. **分核负载**：核间不均衡由 tiling 的 Former/Tail 均分消除；小 shape（`n < 4`）单核完成。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800T A2（910B3，性能基准设备） | √ |
| Atlas 800T A3 | √ |
| Atlas 800I A2（同族） | √（同 arch22 路径，随 A2 一起构建） |

### 算子约束限制

（以下逐条取自任务书 §2.5，不新增、不删减）

| 约束项 | 内容 |
| --- | --- |
| 参数合法性 | `result` 为 nullptr（任意 n，含 quick return 路径）返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n > 0` 且 `incx > 0` 时 `x` 为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE`；校验通过后 `n <= 0` 或 `incx <= 0` 走 quick return（`result` 置 `0.0f`，返回成功）；元素值不校验 |
| 非连续 Tensor 支持 | 不要求（向量非连续由 `incx` 表达，但 `incx <= 0` 时按 quick return 处理不遍历） |
| broadcast 规则 | 不涉及（单向量归约到标量） |
| dynamic shape 要求 | 不要求（`n` 为运行时入参） |
| 原地与视图语义 | 不涉及（`result` 为独立输出标量） |
| 确定性计算要求 | 不要求（归约累加顺序不保证逐位一致，按精度阈值判定） |
| 空 Tensor 与 0 维处理 | `n <= 0` 为合法 quick return，返回成功且 `result` 写 `0.0f` |
| 异步执行 | 依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出 FLOAT32 标量，标量比对：`\|actual - golden\| ≤ atol + rtol × \|golden\|`，`atol = rtol = 2^-13 (≈1.2207e-4)`；`matched_ratio ≥ 0.99`（标量场景退化为单值判定）；且 `max_abs_error ≤ 1e-2` 或 `32 × ULP` | 任务书 §3.2；生态算子开源精度标准（mixed_tolerance_standard.md，FLOAT32 列） |
| 特殊值标准 | 输入含 `Inf/NaN` 时：golden 与 actual 同为 NaN、同为同号 Inf、或一方 NaN 一方 ±Inf，判通过；一方有限另一方 Inf/NaN 判失败 | 任务书 §3.2「Inf/NaN 特殊值判定规则」 |
| 性能标准 | 平均单次耗时（us）≤ GPU 标杆耗时 × 1.25；有效采样 ≥10 次取算术平均；warmup 5 次不计入 | 任务书 §3.3；标杆见 `test_cases/gpu_baseline.csv`（A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4 实测回填） |
| 内存标准 | 单用例 host 侧 ≤ 4GB（设计选择，非硬件限制）；本算子无额外 workspace 要求 | 任务书 §3.4「不涉及」；测试指导 README「内存预算」 |

**精度测试方法**：golden 由 cblas（Netlib BLAS 3.10.0）`cblas_scasum` 生成，随任务测试工程提供；CSV 驱动 GTest 逐条执行，逐条判定并输出 PASS/FAIL；quick return 用例 golden 固定为 `0.0f`，无需调用 cblas。

**性能测试方法（口径冲突已显式登记）**：任务书 §3.3 规定的**正式口径**是 `msprof op` 模式采集单算子 `Task Duration(us)`——自带 5 次 warmup 不计入，warmup 后连续执行 ≥10 次且**执行返回 SUCCESS 方计入有效样本**，取算术平均作为 `Avg time`，逐 case 判据 `mean_target_us ≤ 1.25 × gpu_ms × 1000`。

随任务测试指导（`assignment/test_cases/README.md` 第 99 行）另行说明：`verify_performance.py` 只解析 GTest 打印的 `[PERF] case=<case_name> ... avg_us=<数值>`（含下发开销、不含 H2D/D2H），并注明 msprof 仅供交叉验证 —— **该说明与任务书 §3.3 存在口径冲突**。

**处置**：自测报告同时给出两套数据；`msprof op` 结果列为**任务书判定口径**，`avg_us` 列为可自动化回归的辅助口径与交叉验证。两者并列展示并标注来源，不以后者替代前者，最终以任务书口径判通过/不通过。GTest 仍需实现 `[PERF] ... avg_us=` 打印契约，否则 `verify_performance.py` 无法产出辅助数据。

## 兼容性分析

`aclblasScasum` 为仓内新增实现，接口声明已在公共头文件 `include/cann_ops_blas.h` 提供并与其它产品线（950PR）共用，本任务只补实现与测试，不修改既有接口签名、不改变 `aclblasSasum` 等既有算子行为，不涉及兼容性改造。

## 待实测决策（decisions pending measurement）

以下选择由「先精度、后性能」的实测证据确定，本设计不预先定稿：

| 编号 | 待定项 | 定稿方式 |
| --- | --- | --- |
| D2 | 归约实现形态：单核单段 `Abs + ReduceSum` 直算（正确性锚点/回退路径） vs 分核两阶段归约 | 先以单段实现跑通全量精度；若正式性能测量出现逐条不达标，再启用分核两阶段并复测 |
| D3 | 分块参数（单次搬运块大小、`BUFFER_NUM` 流水深度）、尾块处理方式、`incx > 1` 的 gather 实现 | 由实测性能（`avg_us`）与精度结果共同选定，记录到 `status.md` |
| D4 | `incx == 1` 连续快路径阈值的具体取值 | 由小/中/大 n 的实测对比确定 |

## 测试计划（摘要）

1. **用例来源**：随任务的 `test_cases/scasum_test.csv`（1200 条，精度 1000 / 性能 200）与 `gpu_baseline.csv`（200 条基线）冻结使用，生成器不得覆盖原文件。
2. **补充用例**：补 `GAUSS_NORM_5_2` 正态分布用例（μ∈[-5,5]，σ∈[0.1,2]，实虚独立采样）及任务书 §3.5-4 指出的其他缺失覆盖；补充数量与占比在自测报告中说明。
3. **负向用例缓冲区规则**：期望非 `ACLBLAS_STATUS_SUCCESS` 的用例不得按 `(n-1)*|incx|` 预分配缓冲（INT32_MIN 步长会溢出），一律用单元素 dummy 缓冲或 `NULLPTR` 占位；`n=0` / `incx≤0` 的 quick return 用例同样无需按步长分配。
4. **测试注册**：`test/asum/scasum/CMakeLists.txt` 的 arch22 分支补 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)`，使 A2/A3 构建走 GTest runner 并输出 `avg_us`。
5. **A3 验证**：A3 要求能构建并做代表性执行验证（任务书允许自验证只覆盖一种款型；A2 为正式自验证主平台）。

---

# 附录（模板无对应章的补充材料）

## 附录 A：调用方式

```cpp
aclblasHandle_t handle = nullptr;
aclblasCreate(&handle);
aclblasSetStream(handle, stream);           // 绑定 stream

int n = 1024, incx = 1;
aclblasComplex* dx = nullptr;               // Device 内存，长度 1+(n-1)*|incx|
float* dResult = nullptr;                   // Device 内存，4 字节
aclrtMalloc((void**)&dx, sizeof(aclblasComplex) * n, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dx, sizeof(aclblasComplex) * n, hx, sizeof(aclblasComplex) * n, ACL_MEMCPY_HOST_TO_DEVICE);

aclblasStatus_t st = aclblasScasum(handle, n, dx, incx, dResult);
aclrtSynchronizeStream(stream);             // 读回前同步
float hResult = 0.0f;
aclrtMemcpy(&hResult, sizeof(float), dResult, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
```

## 附录 B：测试设计

**功能矩阵**（来源：任务书 §3.5-3 + 随任务 `scasum_test.csv` 分类）

| 维度 | 覆盖 |
| --- | --- |
| n | 0、1、小质数 2/3/7、2 的幂与 ±1、≥2^20；负 n 为 quick return 边界 |
| incx | 1/2/3 正常路径；0、-1、-2、-3 为 quick return |
| x 填充 | 均匀 [-5,5] 50%、正态（μ∈[-5,5]，σ∈[0.1,2]）50%、全零、正负交替、极端值、Inf/NaN |
| result | 空指针负向；n>0 且 incx>0 时 x 空指针负向 |
| 规模 | 精度 ≤2^20；性能 1…4194304（连续访存） |

**负向与边界**：`n=0`/负 n、`incx=0/-1/-2/-3`、`x=nullptr`、`result=nullptr`、quick return 组合空指针；缓冲按单元素 dummy 或 `NULLPTR` 占位。

**性能用例**：200 条 `TC_PF`，前 3 条与任务书 §3.3 典型 case 对齐（n=1048576/2097152/4194304，incx=1），全部连续访存。

**A3 计划**：A3 构建 + 代表性执行（小/中/大 n 各若干条 + quick return 各分支），不重复全套正式采样。

## 附录 C：开发计划（对应执行工单）

| 阶段 | 内容 | 工单 |
| --- | --- | --- |
| 设计 | 本设计文档 + 评审 | WO-02 |
| 测试工程 | golden/比较器/用例/注册 | WO-03 |
| 首版实现 | 校验 + quick return + 正步长路径 | WO-04 |
| 代表验证 | 精度与性能代表点 | WO-05 |
| 边界补齐 | 特殊值与边界 | WO-06 |
| 调优 | 仅对实测不达标项 | WO-07 |
| A3 | 构建 + 代表性验证 | WO-08 |
| 正式数据 | 全量精度 + 200 条性能 + 内存 | WO-09 |
| 交付 | 七件材料 + README | WO-10 |
| 提交 | PR + 验收 | WO-11 |

## 附录 D：风险与待确认项

| 编号 | 风险/待确认 | 影响 | 处置 |
| --- | --- | --- | --- |
| R1 | 本机 A2 为 **910B4**，任务书性能基准为 910B3 | 性能数据不可直接等同基准设备 | 自测报告标注实际设备；性能判据仍按任务书 `gpu_baseline.csv` |
| R2 | 归约形态若单段不达标，需两阶段 | 工期 | D2 决策点，仅在实测不达标时启用 |
| R3 | `outGM` 清零与原子加的时序 | 结果错 | `Init` 清零后 `SyncAll`，`Process` 才 `SetAtomicAdd` |
| R4 | 大 n × 极端值溢出到 Inf | 精度判定 | 按任务书「与 golden 同为 Inf 判通过」，报告单独标注 |
| R5 | `incx>1` 走标量 gather 性能差 | 仅功能用例 | 性能用例全为 incx=1，不影响判据 |

## 附录 E：交付件清单（任务书 §4）

1. 本设计文档（cann-competitions PR，评审合入）
2. 自测用例及测试代码（含 readme 复现步骤）
3. 自测报告（精度/性能/内存）
4. 待验收代码地址（个人仓 + 分支 + `blas/asum/arch22/`，邀请 `Ascend-CANN` 为开发者，README 支持表补 `aclblasScasum`）
5. 私仓 `task_submission/` 七个规定命名文件

## 附录 F：参考资料

1. Ascend C 算子开发文档（任务书 §6-1）
2. 算子接口文档（任务书 §6-2）
3. ops-blas 仓：`https://gitcode.com/cann/ops-blas`
4. 生态算子开源精度标准：`mixed_tolerance_standard.md`
5. cuBLAS `cublasScasum` 文档；Netlib `scasum.f`
## 附录 G：实现细节补充（文件清单 / UB 推导 / 伪代码）

### G.1 新增与改动文件清单（预期）

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 新增 | `blas/asum/arch22/scasum_host.cpp` | `aclblasScasum` 接口实现：校验、quick return、tiling 计算与下发、kernel 直调、stream 同步 |
| 新增 | `blas/asum/arch22/scasum_kernel.cpp` | `ScasumAIV` kernel：Init / CopyIn / Compute / CopyOut、原子加汇总 |
| 新增 | `test/asum/scasum/arch22/scasum_param.h` | CSV 参数定义（`n` / `incx` / `x` 填充 / `result` 空指针标记） |
| 新增 | `test/asum/scasum/arch22/scasum_test.cpp` | GTest：精度比对、边界负向、性能采样与 `[PERF] ... avg_us=` 打印 |
| 新增 | `test/asum/scasum/arch22/*.csv` | 补充用例（正态 `GAUSS_NORM_5_2`、对齐偏移等），原始 CSV 保持冻结 |
| 改动 | `test/asum/scasum/CMakeLists.txt` | arch22 分支增加 `ops_blas_add_gtest_tests(${OPS_BLAS} scasum_test)` |
| 不改 | `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h` | 声明与类型已存在，复用 |
| 不改 | `blas/asum/arch22/sasum_*.cpp`、`arch35/*` | 既有算子行为不变 |

### G.2 UB 容量与分块推导（可核验的算术）

- 单复数元素 = 2 个 float32 = **8 字节**；一个 32B block = 8 个 float = **4 个复数元素**。
- 输入分块：两块缓冲（`BUFFER_NUM = 2`）。取单调块 `tileElems` 个复数元素，则 UB 占用 ≈ `2 × tileElems × 8B`（输入）+ 输出/临时 `workBuf`。
- 沿用同族实现量级：单块 80KB → `tileElems = 80KB / 8B = 10240` 个复数元素；双缓冲输入占 160KB，A2 的 UB 可容纳，留足 `ReduceSum` 的 `workBuf`（按 `repeat` 上界预留，同 `sasum_kernel.cpp` 的算法）。
- 尾块：`n % tileElems` 剩余元素补 0 到 8 元素（32B）对齐；补 0 对非负累加无影响（`|0| = 0`）。
- 由此得到逐核工作量：`calNum[i]` 为复数元素个数，核内按 `tileElems` 次迭代，最后一次走 padded 路径。

### G.3 Kernel 伪代码（评审用）

```cpp
__aicore__ inline void ScasumAIV::Init(GM_ADDR xGM, GM_ADDR outGM, GM_ADDR tilingGm) {
    blockNum = GetBlockNum(); vecIdx = GetBlockIdx();
    ParseTilingData(tilingGm);                       // n, incx, useCoreNum, startOffset[vecIdx], calNum[vecIdx]
    const uint64_t physLen = (n > 0 && incx > 0) ? (uint64_t)(n - 1) * incx + 1 : 0;  // 复数元素个数
    inGM.SetGlobalBuffer((__gm__ float*)xGM, physLen * 2);   // 视作 float 视图，长度 2*physLen
    outGMF.SetGlobalBuffer((__gm__ float*)outGM, 1);
    InitBuffers();                                    // inQueue/outQueue/ workBuf
    WriteZeroToOut(); SyncAll();                      // 原子累加前清零
}

__aicore__ inline void ScasumAIV::Process() {
    SetAtomicAdd<float>();
    uint32_t done = 0;
    while (done < calNum) {
        uint32_t cnt  = min(tileElems, calNum - done);
        CopyIn(startOffset + done, cnt);               // incx==1: DataCopy/Pad 连续 float
                                                       // incx>1 : 逐元素 gather 2 个分量
        Compute(cnt);                                  // Abs(f32) + ReduceSum -> 单 float
        CopyOut();                                     // DataCopyPad 4B 到 outGM（原子加）
        done += cnt;
    }
    SetAtomicNone();
}
```

### G.4 `incx == 1` 与 `incx > 1` 的访存形态

| 场景 | 访存 | UB 内计算 | 用于 |
| --- | --- | --- | --- |
| `incx == 1` | `2n` 个 float 连续，`DataCopy` / `DataCopyPad` | 一次 `Abs` 覆盖 `\|Re\|` 与 `\|Im\|`，再 `ReduceSum` | 全部性能用例 + 多数精度用例 |
| `incx > 1` | 第 i 个复数取 `(2i·incx, 2i·incx+1)`，`GetValue` gather | 同上（补齐到 8 元素） | 功能用例（`incx=2/3`） |
| `incx <= 0` / `n <= 0` | 不访存 | 无 kernel | quick return 路径 |

## 附录 H：特性交叉分析

| 交叉项 | 影响 | 结论 |
| --- | --- | --- |
| 与 `aclblasSasum`（实数版） | 仅复用其工程范式（tiling 下发布局、atomic 汇总、尾块补齐），不共用同一函数实现 | 不改动 `sasum` 任何行为 |
| 与 950PR（arch35） | 声明共用（`cann_ops_blas.h`），实现目录不同（本任务只做 arch22） | 950PR 路径不受影响 |
| 与公共头文件/ABI | 无新增声明、无结构体变更 | 无 ABI 破坏 |
| 与误差/确定性特性 | 归约顺序不保证逐位一致（任务书允许） | 不需要额外确定性开关 |
| 与 stream/异步语义 | 使用 handle 绑定 stream；读回前需同步 | 与仓内其它 BLAS 接口一致 |
| 与浮点特殊值 | `Abs` 对 `Inf/NaN` 有确定传播行为，按任务书等效谓词判定 | 无需额外分支 |

## 附录 I：用例规模与分母（自测报告口径）

| 类别 | 条数 | 来源 | 说明 |
| --- | --- | --- | --- |
| 原始精度用例 | 以 `scasum_test.csv` 实际精度行数为准（设计上限 1000） | 随任务提供，**冻结** | 逐条 PASS/FAIL，不做跨用例比率 |
| 原始性能用例 | 200 | `scasum_test.csv` 的 `TC_PF` 段 | 与 `gpu_baseline.csv` 逐条 ID 映射 |
| 补充：正态 | 待定（生成后写入报告） | 自建，`GAUSS_NORM_5_2` | μ∈[-5,5]、σ∈[0.1,2] |
| 补充：对齐偏移 | ≥4 | 自建 | 偏移 0/4/8/16 字节 |
| 补充：边界负向 | 覆盖 `n=0/负 n`、`incx=0/-1/-2/-3`、`x/result` 空指针 | 自建/原始 CSV | 与原始用例去重后计数 |

> 分母口径：报告里同时给「原始用例数」「补充用例数」「合计」，避免 1000 / 1200 的口径歧义。

## 修订记录

| 版本 | 日期 | 说明 |
| --- | --- | --- |
| v0.1 | 2026-11-11 | 初稿：按官方模板四大章 + 附录 A–F；含待实测决策 D2/D3/D4 |
