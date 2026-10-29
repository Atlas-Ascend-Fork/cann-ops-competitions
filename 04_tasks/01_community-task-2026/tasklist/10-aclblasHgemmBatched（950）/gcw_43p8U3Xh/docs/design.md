# aclblasHgemmBatched 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务 2026，任务书《aclblasHgemmBatched 算子开发任务书》。目标代码仓为昇腾算子开源仓 `ops-blas`（https://gitcode.com/cann/ops-blas ），实现目录为 `blas/gemm/arch35/`，测试代码合入 `test/gemm/hgemmBatched/arch35/`。

本任务为补齐 Blas 仓能力：在 Ascend 950PR 上使用 Ascend C 编程语言开发半精度实数批量通用矩阵乘算子 `aclblasHgemmBatched`，对标 cuBLAS `cublasHgemmBatched`。接口声明需新增到 `include/cann_ops_blas.h`，与其他产品线共用同一 API，禁止定义 950PR 私有平行接口。

### 背景介绍

`aclblasHgemmBatched` 属于 Level-3 GEMM 族，计算每个 batch 的半精度矩阵乘：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i,  i ∈ [0, batchCount)
```

其中 `A_i`、`B_i`、`C_i` 均为 FLOAT16 矩阵，`alpha`、`beta` 为 FLOAT16 标量指针；`op()` 支持 N/T/C 转置，FLOAT16 实数路径下 T 与 C 等价；矩阵按列主序存储；`Aarray`、`Barray`、`Carray` 为指向 batchCount 个矩阵首地址的指针数组。

ops-blas 仓当前在 `include/cann_ops_blas.h` 中尚无 `aclblasHgemmBatched` 声明，arch35 目录也缺少对应实现。本任务需新增接口声明，基于仓内同族 GEMM batch kernel 直调方式复用 Gemm 模板，完成 950PR 实现、精度验证和性能达标。

## 需求分析（required）

### 需求描述

在 Ascend 950PR 上使用 Ascend C 实现 `aclblasHgemmBatched`，完成算子设计、开发、测试全流程。算子需与 cuBLAS `cublasHgemmBatched` 参数序列和核心语义对齐，支持 FLOAT16 批量矩阵乘、N/T/C 转置组合、alpha/beta 标量缩放、列主序前导维、batchCount 批量指针数组，并满足任务书规定的边界行为、精度标准、性能标准与自验要求。验收通过后合入 `ops-blas` 仓 `blas/gemm/arch35`。

### 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 在 `include/cann_ops_blas.h` 新增 `aclblasHgemmBatched` 声明，签名与 cuBLAS 参数序列一致，与其它产品线共用，不定义 950PR 私有平行接口 | 代码审查；头文件比对 |
| R2 | 实现 `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`，逐 batch 计算，FLOAT16 输入输出 | 功能用例；golden 由 FLOAT32 `cblas_sgemm` 逐 batch 计算后舍入 FLOAT16 |
| R3 | 支持 transa/transb ∈ {N, T, C}，FLOAT16 实数下 T 与 C 等价 | 转置组合用例 |
| R4 | 支持 `Aarray/Barray/Carray` 指针数组，各矩阵按列主序、前导维 lda/ldb/ldc | 多 batch、padding 用例 |
| R5 | 边界行为：m=0、n=0 或 batchCount=0 为合法 no-op，不访问数组；k=0 时逐 batch 仅 `C_i = beta × C_i`，beta=0 置零，不引用 A/B | 边界与负向用例；状态码断言 |
| R6 | alpha、beta 不可为 nullptr；batchCount>0 时 Aarray/Barray/Carray 不可为 nullptr；校验顺序符合任务书 §2.5 | 负向用例 |
| R7 | 精度满足 FLOAT16 混合容差：rtol=atol=2^-10、matched_ratio≥0.99、max_abs_error≤1e-1 或 32 ULP；含 Inf/NaN 特殊值规则 | 自测报告；200 条/配套用例 |
| R8 | 性能：单 case ≥ 0.4× GPU，全部性能 case 倍率算术平均 ≥ 0.5× GPU；msprof op 5 次 warmup、10 次有效采样平均 | 自测报告性能截图 |
| R9 | 复用 ops-blas 同族 GEMM batch kernel 模板，实现放在 `blas/gemm/arch35/`，不复制整套第三方源码 | 代码审查；目录结构 |
| R10 | 自验覆盖转置全组合、尺寸扫描、batchCount 扫描、标量特殊值、padding、边界负向、Inf/NaN、性能用例 | 测试代码 + 自测报告 |
| R11 | 交付设计文档 PR、自测用例及测试代码、自测报告、待验收代码地址；README 产品支持表标注 Ascend 950PR：支持 | 验收交付件审查 |

## 详细设计（required）

### 算子分析

#### 数学公式

对每个 batch `i ∈ [0, batchCount)`：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i
```

逐元素：

```text
C_i[r][c] = alpha * Σ_{p=0}^{k-1} op(A_i)[r][p] * op(B_i)[p][c] + beta * C_i[r][c]
r ∈ [0, m), c ∈ [0, n)
```

#### 接口定义

在 `include/cann_ops_blas.h` 新增：

```cpp
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const* Aarray, int lda,
    const aclblasHalf* const* Barray, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const* Carray, int ldc,
    int batchCount);
```

参数序列与 cuBLAS `cublasHgemmBatched` 一一对应；`aclblasHalf` 为 FLOAT16；`Aarray/Barray/Carray` 为设备指针数组；`lda/ldb/ldc` 为列主序前导维；`batchCount` 为批量数。

#### 数据类型与形状

- 主路径：FLOAT16（`aclblasHalf`），alpha/beta 为 FLOAT16 标量指针。
- `m/n/k ≥ 0`，`batchCount ≥ 0`。
- `op(A_i)` 为 m×k（transa=N）或 k×m（transa∈{T,C}）；`op(B_i)` 为 k×n（transb=N）或 n×k（transb∈{T,C}）；`C_i` 为 m×n。
- 前导维：`lda ≥ max(1, A_i 行数)`，`ldb ≥ max(1, B_i 行数)`，`ldc ≥ max(1, m)`。
- 列主序，各 batch 矩阵连续存储，仅支持前导维语义，不支持超出 lda/ldb/ldc 的非连续 stride。

#### 转置语义

- `transa/transb = ACLBLAS_OP_N`：op(A_i)=A_i，op(B_i)=B_i；
- `transa/transb = ACLBLAS_OP_T`：op(A_i)=A_i^T，op(B_i)=B_i^T；
- `transa/transb = ACLBLAS_OP_C`：FLOAT16 实数路径下与 T 等价。

#### 边界行为与校验顺序

- `m=0`、`n=0` 或 `batchCount=0`：合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS`，不访问 `Aarray/Barray/Carray`。
- `k=0`：逐 batch 仅执行 `C_i = beta × C_i`；`beta=0` 时置零；不引用 A/B，返回 SUCCESS。
- 校验顺序按任务书 §2.5：alpha/beta 判空 → `Aarray/Barray/Carray` 判空（仅 batchCount>0 时）→ m/n/batchCount 短路 → 枚举与维度合法性 → 前导维约束。因此 `m=0` 或 `n=0` 且 `batchCount>0` 时数组指针仍不可为 nullptr；`batchCount=0` 时不校验数组指针。
- 非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

### 算子实现

#### Host 侧设计

Host 侧负责接口校验、batch 指针数组处理、tiling 计算和 kernel 下发。

1. **参数校验**：严格按上述顺序实现；alpha/beta 判空优先于 no-op 短路；Aarray/Barray/Carray 仅在 `batchCount > 0` 时判空；枚举、负维度、前导维约束逐项校验并返回对应状态码。
2. **Batch 指针数组**：`Aarray/Barray/Carray` 为 Device 内存中的指针数组，Host 侧不逐元素解引用。Kernel 通过 `batchCount` 和数组首地址在 Device 侧按 batch 索引读取对应矩阵首地址。Host 侧只校验数组指针非空和 batchCount 合法性。
3. **Tiling 计算**：按 m、n、k、lda/ldb/ldc、transa/transb、batchCount 生成 tilingkey；按 950PR 的核数、UB/L1/L0 容量、double buffer 需求计算分核与分块参数。batch 维度可在 kernel 内串行或按核/子核并行，由 batchCount 与单 batch 规模决定；小 batchCount 大矩阵优先矩阵内并行，大 batchCount 小矩阵优先 batch 间并行。
4. **Kernel 下发**：通过 handle 绑定 stream，Ascend C kernel 直调方式下发；异步执行，读回 Device 结果前调用方同步 stream。
5. **k=0 / no-op 路径**：`m=0`、`n=0`、`batchCount=0` 直接返回 SUCCESS，不下发 kernel；`k=0` 下发轻量 beta 缩放 kernel，逐 batch 对 C_i 的 m×n 操作区按 `beta` 缩放，`beta=0` 置零，不访问 A/B。

#### Kernel 侧设计

Kernel 侧基于 ops-blas 同族 GEMM batch kernel 直调，复用 Gemm 模板，按 Init + Process 组织。

1. **Batch 索引**：从 `Aarray/Barray/Carray` 指针数组按 batch 索引读取第 i 个矩阵首地址；batch 循环可映射到 block 维或核内循环，取决于 tiling 策略。
2. **数据搬入**：按转置语义从 `A_i/B_i` 搬入对应 tile；N 路径按列主序直接读，T/C 路径通过转置索引或 Matmul API transpose 参数实现；FLOAT16 输入，累加可在 FLOAT32 进行后转回 FLOAT16 输出，以降低累加误差。
3. **矩阵乘**：使用 Ascend C Matmul API 或同族 GEMM batch 模板完成 `op(A_i) * op(B_i)`；按 k 方向累加；合理使用 L1/L0/UB，开启 double buffer，批量搬运减少冗余开销。
4. **Epilogue**：按 `C_i = alpha * (A_i*B_i) + beta * C_i` 计算；`beta=0` 时 C_i 不参与读取（允许 Carray 为 NaN）；`k=0` 时跳过矩阵乘，仅做 beta 缩放；alpha/beta 为 FLOAT16 标量，Kernel 内按标量广播。
5. **边界与尾块**：m/n/k 非对齐时按 tile 切分并处理尾块；lda/ldb/ldc 大于逻辑行数时，只读写逻辑区域，padding 不参与计算；batchCount 非 2 的幂时按实际数量循环，不越界读取指针数组。
6. **异步与同步**：kernel 由显式 stream 下发，异步执行；Host 侧不做流同步。

#### 复用同族 GEMM batch kernel

任务书要求基于 ops-blas 仓同族 GEMM batch kernel 直调，复用 Gemm 模板，实现代码放在 `blas/gemm/arch35/`。设计上：

- 优先复用现有 batch GEMM 的 tiling、数据搬入、Matmul 调用和 epilogue 框架；
- 新增 `aclblasHgemmBatched` 的 Host 接口封装、参数校验、指针数组处理和 FLOAT16 标量路径；
- 不复制整套第三方源码，不新增与同族 GEMM 重复的 kernel 主体；
- 若现有模板缺少 FLOAT16 alpha/beta 或 T/C 等价路径，在 arch35 内做最小扩展，保持接口和公共模板默认行为不变。

#### 性能与内存设计

- 性能目标：单 case 平均单次耗时 ≤ GPU 标杆 2.5 倍（≥0.4× GPU）；全部性能 case 倍率算术平均 ≥0.5× GPU。采用 msprof op 采集 task duration，自带 5 次 warmup，warmup 后 10 次有效样本取平均；异常样本不计入并说明。
- 优化手段：batch 间并行与矩阵内并行按规模自适应；满核均分；double buffer；批量搬运；FLOAT16 输入、FLOAT32 累加、FLOAT16 输出；小矩阵避免过度分核；大 batchCount 优先 batch 并行。
- 内存：本任务不设额外内存门禁；不新增与问题规模无关的持久 Device 缓存；指针数组本身由调用方分配，Kernel 只读。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

实现目录为 `blas/gemm/arch35/`；README 产品支持表标注 Ascend 950PR：支持；接口声明放入 `include/cann_ops_blas.h`，供其他产品线共用。自验证与性能测试均在 Ascend 950PR、CANN 9.1.0 环境完成。

### 算子约束限制

- 仅支持 FLOAT16 主路径；alpha/beta 为 FLOAT16 标量指针，不可为 nullptr。
- 仅支持列主序；各 batch 矩阵连续存储，仅支持 lda/ldb/ldc 前导维语义，不支持超出前导维的非连续 stride。
- 不支持广播；各 batch 的 A_i/B_i/C_i 为独立矩阵。
- 不要求 dynamic shape；m/n/k/batchCount 为运行时入参，编译期 shape 未知。
- 不要求确定性计算。
- `Carray` 各 batch 原地覆写，不返回视图。
- `m=0`、`n=0`、`batchCount=0` 为合法 no-op；`k=0` 仅做 beta 缩放。
- `batchCount>0` 时 `Aarray/Barray/Carray` 不可为 nullptr，即使 `m=0` 或 `n=0`。
- 禁止定义 950PR 私有平行 API；接口与其它产品线共用。
- 合入代码遵循 `ops-blas` 仓 `CONTRIBUTING.md`，不得包含冗余文件或无用代码；PR 关联设计文档与任务书，附自验结果。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 FLOAT32 参考实现（Netlib `cblas_sgemm` 逐 batch 计算）得到后按 IEEE 754 就近舍入为 FLOAT16，逐元素比对各 batch C_i 全矩阵。FLOAT16 按 `rtol=atol=2^-10`、`required_matched_ratio=0.99`、`max_abs_error_limit=1e-1 或 32 ULP` 判定：`|actual-golden| ≤ atol + rtol×|golden|`；用例同时满足 matched_ratio 与 max_abs_error 限值则通过 | 任务书 §3.2；混合容差精度标准 |
| Inf/NaN 规则 | golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf，视为等效通过；一方有限另一方 Inf/NaN 不通过；非有限值不计入 matched_ratio 与 max_abs_error，单独记录总数及通过数；有限值满足 matched_ratio≥0.99 且 max_abs_error≤限值，且非有限值全部通过，用例通过 | 任务书 §3.2.5 |
| k 过大误差举证 | 若 FLOAT16 因 k 过大不满足阈值，须在自测报告给出 CPU golden 单标杆对比、最大误差随 k 增长曲线、累加精度档位说明，不得放宽阈值代替举证 | 任务书 §3.2.4 |
| 性能标准 | Ascend 950PR，FLOAT16 平均单次耗时 ≤ GPU 标杆 2.5 倍（单 case ≥0.4× GPU）；全部性能 case 倍率算术平均 ≥0.5× GPU；msprof op 5 次 warmup、10 次有效样本平均；异常样本不计入并说明 | 任务书 §3.3 |
| 内存标准 | 不涉及额外内存门禁；不新增与问题规模无关的持久 Device 缓存 | 任务书 §3.4 |

自验要求：使用 ops-blas 仓 test 目录测试框架，CSV 描述用例，C++ GTest 加载 CSV 调用 `aclblasHgemmBatched`；golden 由 FLOAT32 `cblas_sgemm` 逐 batch 计算后舍入 FLOAT16。覆盖 transa/transb 全组合、m/n/k 尺寸扫描（0、1、小质数、2 的幂及 ±1、非对齐、大规模至 16384）、batchCount 扫描（0、1、2、小质数、2 的幂、非对齐至 65536）、alpha/beta 特殊值（0、±1、大值）、前导维 padding、边界负向（m=0/n=0/batchCount=0 no-op、k=0 beta 缩放、空指针、非法枚举、负维度、非法前导维）、Inf/NaN、性能用例；输出自测报告，含用例参数、精度对比结果及截图、性能数据及截图。

### 兼容性分析

- 本算子为新增接口与新增实现：在 `include/cann_ops_blas.h` 新增 `aclblasHgemmBatched` 声明，与 cuBLAS 参数序列一致，供其他产品线共用，不定义 950PR 私有平行 API。
- 实现放在 `blas/gemm/arch35/`，复用同族 GEMM batch kernel 模板；不修改公共模板默认行为；若需最小扩展，须回归同族 GEMM 直接调用样例。
- 测试代码放在 `test/gemm/hgemmBatched/arch35/`，文件结构参考主仓 blas 算子测试代码结构，包含 CSV。
- 不涉及旧版本算子行为兼容；README 产品支持表标注 Ascend 950PR：支持。
- 遵循 `ops-blas` 仓 `CONTRIBUTING.md`，PR 不得包含冗余文件或无用代码；PR 关联设计文档与任务书，附精度、性能自验结果。

## 验收交付件

1. 本设计文档：按模板提交至 `cann-competitions` 仓库 `04_tasks/01_community-task-2026/tasklist` 对应任务目录，以 PR 形式评审合入。
2. 自测用例及测试代码：覆盖任务书配套全部自测用例，清晰列出精度 case 与性能 case；README 说明测试步骤，保证可复现。
3. 自测报告：按模板包含用例参数、精度对比结果及截图、性能数据（有效采样 >10 次平均值）及截图。
4. 待验收代码地址：个人代码仓链接、分支、算子目录；邀请账号 `Ascend-CANN` 作为开发者；README 按 ops-blas 规范提供，产品支持表标注 Ascend 950PR：支持；接口声明放入 `include/cann_ops_blas.h`，供其他产品线共用。
5. PR 合入：测试通过后向 `ops-blas` 提交 PR，算子合入 `blas/gemm/arch35`，测试代码合入 `test/gemm/hgemmBatched/arch35`，遵循代码仓规范与贡献指南。