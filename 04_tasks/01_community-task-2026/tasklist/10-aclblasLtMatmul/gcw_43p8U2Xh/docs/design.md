# aclblasLtMatmul 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务 2026，任务书《aclblasLtMatmul 算子开发任务书》。目标代码仓为昇腾算子开源仓 `ops-blas`（https://gitcode.com/cann/ops-blas ），目标目录为 `blas/lt_matmul/` 下 Atlas A2/A3 对应架构目录 `arch22`，测试代码合入 `test/lt_matmul/`。

本任务为补齐 Blas 仓能力：在 Atlas A2/A3 系列产品上使用 Ascend C 编程语言实现单精度实数 cuBLASLt 风格矩阵乘算子 `aclblasLtMatmul`，接口签名以 `include/cann_ops_blasLt.h:909-924` 既有声明为准，与现网及 950PR 共用同一声明，禁止定义产品私有平行接口。

### 背景介绍

`aclblasLtMatmul` 对齐 cuBLAS `cublasLtMatmul` 的核心功能与参数语义，计算：

```text
D = alpha * op(A) * op(B) + beta * C
```

其中 alpha、beta 为标量，A、B、C、D 为矩阵；D 可与 C 原地（C == D）或异地（C != D，两者数据类型、行列数、memory order 须一致）计算。本任务仅支持单 batch：各 layout 的 `batchCount = 1` 且 `stridedBatchOffset = 0`。

ops-blas 仓已有该接口声明，但 A2/A3 实现缺失。本任务采用 Ascend C kernel 直调方式，基于 ops-blas 工程框架实现句柄式 BLAS 接口：通过 handle 绑定 stream 直调 NPU kernel，workspace 由调用方分配并通过 `workspaceSizeInBytes` 传入，不得省略。

主路径数据类型为单精度实数 FLOAT32（`Atype/Btype/Ctype` 取 `ACL_FLOAT`，`computeType`/`scaleType` 取对应 FLOAT32 计算类型）。混合精度经 `computeType`/`scaleType` 描述，不在本任务范围。cuBLASLt 的 epilogue（Bias/ReLU/GELU）等能力不在主路径强制范围，主路径对齐 `EPILOGUE_DEFAULT`。

## 需求分析（required）

### 需求描述

在 Atlas A2/A3 上使用 Ascend C 实现 `aclblasLtMatmul`，完成算子设计、开发、测试全流程。算子需与 cuBLAS `cublasLtMatmul` 核心功能、参数语义完全对齐，支持 FLOAT32 单 batch、N/T/C 转置组合、C == D 原地与 C != D 异地计算、alpha/beta 标量缩放，并满足任务书规定的边界行为、精度标准、性能标准、内存要求与自验要求。验收通过后合入 `ops-blas` 仓 `blas/lt_matmul/arch22`。

### 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 接口签名与 `include/cann_ops_blasLt.h:909-924` 一致，与 950PR 共用，不定义私有平行接口 | 代码审查；接口声明比对 |
| R2 | 实现 `D = alpha * op(A) * op(B) + beta * C`，单 batch，FLOAT32 主路径 | 功能用例；golden 由 cblas `cblas_sgemm` 生成 |
| R3 | 支持 transA/transB ∈ {N, T, C}，实数路径下 C 与 T 等价 | 转置组合用例 |
| R4 | 支持 C == D 原地与 C != D 异地；Cdesc 必须非空，原地时传入与 Ddesc 相同布局描述符 | 原地/异地用例 |
| R5 | 边界行为对齐 cuBLAS/Netlib quick return：m=0 或 n=0 no-op；k=0 只做 beta 缩放；alpha=0 按 beta 缩放且不引用 A/B | 边界用例；`alpha=0` 且 `beta` 缩放结果位精确 |
| R6 | 参数校验与返回码符合任务书 §2.4/§2.5，校验顺序明确 | 负向用例；状态码断言 |
| R7 | workspace 由调用方分配，非空须 16B 对齐，容量满足 `algo->max_workspace_bytes` | workspace 负向用例；峰值统计 |
| R8 | 精度满足 FLOAT32 生态算子开源精度标准，含 Inf/NaN 特殊值判定规则 | 200 条自测用例；自测报告 |
| R9 | 性能达到 GPU 标杆 0.8 倍以上（平均单次耗时 ≤ GPU 标杆 1.25 倍） | msprof op 采样；自测报告 |
| R10 | workspace 峰值不超过输入输出数据总量的 1.2 倍 | 自测报告 workspace 数据 |
| R11 | 自验覆盖任务书 200 条用例，覆盖尺寸扫描、转置、原地/异地、填充、边界负向、INF/NAN、性能 | 测试代码 + 自测报告 |
| R12 | 交付设计文档 PR、自测用例及测试代码、自测报告、待验收代码地址；README 产品支持表标注 Atlas A2/A3 支持 | 验收交付件审查 |

## 详细设计（required）

### 算子分析

#### 数学公式

```text
D = alpha * op(A) * op(B) + beta * C
```

- `op(A)` 为 m×k；`op(B)` 为 k×n；C、D 为 m×n。
- alpha、beta 为标量，类型对应 `computeType` 与 `scaleType`。
- 主路径 FLOAT32；`alpha = 0` 且 alpha 指针非空时跳过矩阵乘，`D = beta * C` 结果应位精确。
- `k = 0` 时只做 beta 缩放，不引用 A 与 B。
- `m = 0` 或 `n = 0` 时为合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不执行计算。

#### 接口与参数

接口签名以 `include/cann_ops_blasLt.h:909-924` 为准：

```cpp
aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t lightHandle,
                                aclblasLtMatmulDesc_t computeDesc,
                                const void* alpha,
                                const void* A,
                                aclblasLtMatrixLayout_t Adesc,
                                const void* B,
                                aclblasLtMatrixLayout_t Bdesc,
                                const void* beta,
                                const void* C,
                                aclblasLtMatrixLayout_t Cdesc,
                                void* D,
                                aclblasLtMatrixLayout_t Ddesc,
                                const aclblasLtMatmulAlgo_t* algo,
                                void* workspace,
                                size_t workspaceSizeInBytes,
                                aclrtStream stream);
```

句柄类型差异（`aclblasLtHandle_t` ↔ `cublasLtHandle_t`、`aclrtStream` ↔ `cudaStream_t`）属平台类型映射，参数顺序与语义一一对应。workspace 由调用方分配并通过 `workspaceSizeInBytes` 传入，不得省略。

关键参数语义：

| 参数 | 设计要点 |
| --- | --- |
| `lightHandle` | 已创建的有效句柄；nullptr 返回 `ACLBLAS_STATUS_NOT_INITIALIZED` |
| `computeDesc` | 描述 transA/transB、computeType、scaleType、pointerMode 等；nullptr、非法描述或 pointerMode 不受支持返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` / `beta` | 不可为 nullptr；类型对应 computeType/scaleType；支持 Host 或 Device 内存 |
| `A` / `B` | Device 内存只读；需执行计算时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Adesc` / `Bdesc` / `Ddesc` | order 必须为 `ACLBLASLT_ORDER_ROW`；`batchCount = 1`、`stridedBatchOffset = 0` |
| `C` / `Cdesc` | Cdesc 必须非空；原地时传入与 Ddesc 相同布局的描述符；实现需补充 C 指针校验（beta ≠ 0 且 C 为 nullptr 属未定义行为，待研发确认） |
| `D` | m > 0 且 n > 0 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `algo` | NULL 时执行隐式 heuristics 搜索默认算法；实现当前仅按 `workspaceSizeInBytes < algo->max_workspace_bytes` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，算法支持性校验须在实现中补充 |
| `workspace` | 无额外空间需求可传 nullptr；非空须 16B 对齐，未对齐返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `workspaceSizeInBytes` | algo 非空时须 ≥ `algo->max_workspace_bytes`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `stream` | 已创建的 stream 或 nullptr（默认 stream）；非法导致下发失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED` |

#### 数据类型与形状

- 主路径：FLOAT32（`Atype/Btype/Ctype = ACL_FLOAT`，computeType/scaleType 对应 FLOAT32）。
- m、n、k ≥ 0；四个 layout 描述符固定 `order = ACLBLASLT_ORDER_ROW`，前导维按行主序解释为“一行元素数（行宽）”：
  - `transA = ACLBLAS_OP_N` 时 `lda ≥ max(1,k)`，否则 `lda ≥ max(1,m)`；
  - `transB = ACLBLAS_OP_N` 时 `ldb ≥ max(1,n)`，否则 `ldb ≥ max(1,k)`；
  - `ldc ≥ max(1,n)`、`ldd ≥ max(1,n)`。
- `ld == 0` 时按行宽归一处理（等价紧凑布局），不提供行/列广播语义。
- 单 batch：`batchCount = 1`、`stridedBatchOffset = 0`。

#### 转置语义

依据 cuBLAS 官方文档：

- `transA = ACLBLAS_OP_N`：op(A) = A；
- `transA = ACLBLAS_OP_T`：op(A) = A^T；
- `transA = ACLBLAS_OP_C`：op(A) = A^H（实数路径下 C 与 T 等价）；
- transB 同理。

#### 边界行为与校验顺序

边界行为对齐 cuBLAS/Netlib `sgemm` quick return 语义：

- `m = 0` 或 `n = 0`：合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，不执行计算；
- `k = 0`：只做 beta 缩放，不引用 A 与 B；
- `alpha = 0`：若 `beta = 1` 则 C 不变，若 `beta = 0` 则 D 各元素置零，其余 beta 值逐元素缩放，不引用 A 与 B。

校验顺序按任务书 §2.5：

```text
handle
→ 描述符判空
→ alpha/beta 判空
→ pointerMode
→ m == 0 / n == 0 no-op
→ 布局合法性（含 order 与 batch）
→ A/B/D 判空
→ workspace
```

因此 `m = 0`/`n = 0` 时即使布局非法或 A/B/D 为空也返回 `ACLBLAS_STATUS_SUCCESS`（描述符与 alpha/beta 仍须非空）。非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`；order 非 ROW 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；handle 为空返回 `ACLBLAS_STATUS_NOT_INITIALIZED`；stream 非法导致下发失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

### 算子实现

#### Host 侧设计

Host 侧负责描述符解析、参数校验、tiling 计算、算法/workspace 检查和 kernel 下发。

1. **描述符解析**：从 `computeDesc` 读取 transA/transB、computeType、scaleType、pointerMode；从四个 layout 描述符读取 dtype、order、ld、batchCount、stridedBatchOffset。主路径仅接受 `ACL_FLOAT`、`ACLBLASLT_ORDER_ROW`、`batchCount = 1`、`stridedBatchOffset = 0`；pointerMode 主路径支持 HOST，DEVICE 不支持时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. **参数校验**：严格按上述校验顺序实现。优先处理 `m == 0`/`n == 0` no-op，保证即使布局非法或 A/B/D 为空也返回 SUCCESS；随后校验 order、batch、ld 约束、A/B/D 指针、workspace 对齐与容量。`k == 0`、`alpha == 0` 可在 host 侧转为轻量 kernel 或直接按 beta 缩放路径下发，避免无效矩阵乘。
3. **Tiling 计算**：根据 m、n、k、lda/ldb/ldc/ldd、转置组合、是否原地、alpha/beta 特殊值生成 tilingkey；按 A2/A3 的核数、UB/L1/L0 容量、double buffer 需求计算分核与分块参数。分核优先满核均分，不能均分时将余块分配到前几个核；单核内按 UB 容量切 tile，处理尾块。
4. **算法与 workspace**：`algo == NULL` 时执行隐式 heuristics 搜索默认算法；`algo` 非空时检查 `workspaceSizeInBytes ≥ algo->max_workspace_bytes`。非空 workspace 必须 16B 对齐；workspace 用于算法临时缓冲，峰值须控制在输入输出数据总量 1.2 倍以内。算法支持性校验需在实现中补充。
5. **Kernel 下发**：通过 handle 绑定 stream，使用 Ascend C kernel 直调方式下发；异步执行，读回 Device 结果前须同步该 stream。下发失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。

#### Kernel 侧设计

Kernel 侧按 Ascend C 矩阵乘编程范式实现，采用 Init + Process 两阶段，Process 包括 CopyIn、Compute、CopyOut。

1. **模板与 tilingkey**：按 transA/transB 组合、是否原地、alpha/beta 特殊值、k 是否为 0、ld 是否紧凑等生成 tilingkey，不同分支走不同 kernel 模板，避免运行期冗余判断。
2. **数据搬入**：A、B 按分块从 Global Memory 搬入 L1/L0 或 UB；支持行主序前导维，转置通过 Matmul API 的 transpose 参数或索引映射实现；实数路径下 `OP_C` 与 `OP_T` 等价。C 在 beta ≠ 0 时搬入用于 epilogue；C == D 原地时须先读 C 再写 D，避免覆盖。
3. **矩阵乘计算**：使用 Ascend C Matmul API 或等价分块矩阵乘实现 `op(A) * op(B)`，FLOAT32 输入输出，按 k 方向累加；合理使用 L1/L0/UB，开启 double buffer，批量搬运减少冗余开销。
4. **Epilogue**：按 `D = alpha * (A*B) + beta * C` 计算。`k == 0` 时跳过矩阵乘，仅做 `D = beta * C`；`alpha == 0` 时跳过矩阵乘，按 beta 缩放 C 写入 D，结果位精确；`beta == 0` 时 C 不参与计算。C != D 异地时 D 独立写出；C == D 原地时保证读写顺序正确。
5. **分核与尾块**：优先满核均分；不能均分时将余块分配到前几个核；单核内按 tile 切分，尾块单独处理，避免数据碎片。
6. **异步与同步**：kernel 由显式 stream 下发，异步执行；host 侧读回前同步 stream。

#### 原地/异地与 workspace

- C == D 原地：`Cdesc` 必须非空，且传入与 `Ddesc` 相同布局的描述符；kernel 需保证先读 C 后写 D。
- C != D 异地：C、D 数据类型、行列数、memory order 须一致；D 为独立输出，覆写新值。
- workspace：由调用方分配；无额外空间需求时可传 nullptr；非空须 16B 对齐；容量由 `aclblasLtMatmulAlgoGetHeuristic` 返回，`workspaceSizeInBytes` 小于 `algo->max_workspace_bytes` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

#### 性能与内存设计

- 性能目标：Atlas 800T A2 (910B3) 上 FLOAT32 平均单次耗时 ≤ GPU 标杆 1.25 倍（达到标杆 0.8 倍以上）。采用 msprof op 模式采集单算子 task duration，自带 5 次 warmup，warmup 后连续 10 次有效样本取算术平均；异常样本不计入并说明。
- 优化手段：满核均分、double buffer、批量搬运、合理 tile 切分、尾块合并、减少冗余同步；针对小尺寸 m/n/k 避免过度分核，针对大规模保证 L1/L0 复用。
- 内存目标：workspace 峰值不超过输入输出数据总量的 1.2 倍；自测报告给出各性能 case 的 workspace 峰值。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列产品（含 Atlas 800T A2 / 910B3） | √ |
| Atlas A3 系列产品（含 Atlas 800T A3） | √ |

自验证只需覆盖一种款型；精度建议 910B3 或 A3，性能建议 910B3。实现放在 `blas/lt_matmul/arch22`，README 产品支持表标注 Atlas A2/A3 系列产品：支持。

### 算子约束限制

- 仅支持单 batch：`batchCount = 1`、`stridedBatchOffset = 0`。
- 四个 layout 描述符固定 `order = ACLBLASLT_ORDER_ROW`；`ld == 0` 按行宽归一，不提供行/列广播语义。
- 主路径仅 FLOAT32；混合精度不在本任务范围。
- 不支持超出 lda/ldb/ldc 语义的非连续内存访问；不涉及广播，A/B/C/D 为独立操作数。
- 不要求 dynamic shape；m/n/k 为运行时入参，由 host tiling 支持。
- 不要求确定性计算。
- 支持 C == D 原地计算，`Cdesc` 须传入与 `Ddesc` 相同布局的描述符，不得省略；不返回视图。
- `alpha`、`beta`、`computeDesc`、`Adesc`、`Bdesc`、`Cdesc`、`Ddesc` 不可为 nullptr；`m > 0` 且 `n > 0` 时 D 不可为 nullptr。
- workspace 非空须 16B 对齐；容量不足返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- 主路径对齐 `EPILOGUE_DEFAULT`；Bias/ReLU/GELU 等 epilogue 不在强制范围。
- 禁止定义产品私有平行接口；接口与 950PR 共用。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib BLAS 3.10.0 `cblas_sgemm`）生成，输出矩阵 D 全矩阵验证；FLOAT32 按 `rtol = atol = 2^-13`、`required_matched_ratio = 0.99`、`max_abs_error_limit = 1e-2 或 32 * ULP` 判定 | 任务书 §3.2；生态算子开源精度标准 |
| Inf/NaN 规则 | golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf，视为等效通过；一方有限另一方 Inf/NaN 不通过；非有限值不计入 matched_ratio 与 max_abs_error，单独记录总数及通过数；有限值满足 matched_ratio ≥ 0.99 且 max_abs_error ≤ 限值，且非有限值全部通过，用例通过 | 任务书 §3.2 |
| 位精确特例 | `alpha = 0` 且 alpha 指针非空时跳过矩阵乘，`D = beta * C` 结果位精确（EXACT 校验） | 任务书 §3.2 |
| 性能标准 | Atlas 800T A2 (910B3)，FLOAT32 平均单次耗时 ≤ GPU 标杆 1.25 倍（达到 0.8 倍以上）；msprof op 采集 task duration，5 次 warmup，10 次有效样本平均；单 case ≥ 0.8× GPU | 任务书 §3.3 |
| 内存标准 | workspace 峰值不超过输入输出数据总量的 1.2 倍；非空 workspace 16B 对齐 | 任务书 §3.4 |

自验要求：使用 ops-blas 仓 test 目录测试框架，CSV 描述用例，python 脚本调用 C++ GTest 加载 CSV 调用 `aclblasLtMatmul`，golden 由 cblas 生成；覆盖任务书 200 条用例，并补充尺寸扫描（0、1、小质数、2 的幂及 ±1、非对齐、大规模）、转置组合、原地/异地、填充、边界负向、INF/NAN、性能用例；输出自测报告，含用例参数、精度对比结果及截图、性能数据及截图、workspace 数据。

### 兼容性分析

- 本算子为已有接口的 A2/A3 实现补齐，接口声明与 950PR 共用，不定义产品私有平行 API，不改变现有声明语义。
- 新实现放在 `blas/lt_matmul/arch22`，不修改公共接口和其他产品线实现；测试代码放在 `test/lt_matmul/`。
- 不涉及旧版本算子行为兼容；README 产品支持表标注 Atlas A2/A3 系列产品：支持。
- 模型/框架侧调用方通过 `aclblasLtHandle_t`、layout 描述符、workspace、stream 使用本算子，参数顺序与语义对齐 cuBLAS `cublasLtMatmul`，平台类型映射明确。
- 公共代码改动须遵循 `ops-blas` 仓 `CONTRIBUTING.md`，不得包含冗余文件或无用代码；PR 关联设计文档与任务书，附自验结果。

## 验收交付件

1. 本设计文档：按模板提交至 `cann-competitions` 仓库 `04_tasks/01_community-task-2026/tasklist` 对应任务目录，以 PR 形式评审合入。
2. 自测用例及测试代码：覆盖任务书 200 条用例，清晰列出精度 case 与性能 case；README 说明测试步骤，保证可复现。
3. 自测报告：按模板包含用例参数、精度对比结果及截图、性能数据（有效采样 >10 次平均值）及截图、workspace 相对 GPU 数据 ≤1.2×。
4. 待验收代码地址：个人代码仓链接、分支、算子目录；邀请账号 `Ascend-CANN` 作为开发者；README 按 ops-blas 规范提供，产品支持表标注 Atlas A2/A3 支持；接口已在 `include/cann_ops_blasLt.h` 声明，不重复定义。
5. PR 合入：测试通过后向 `ops-blas` 提交 PR，算子合入 `blas/lt_matmul/arch22`，测试代码合入 `test/lt_matmul/`，遵循代码仓规范与贡献指南。