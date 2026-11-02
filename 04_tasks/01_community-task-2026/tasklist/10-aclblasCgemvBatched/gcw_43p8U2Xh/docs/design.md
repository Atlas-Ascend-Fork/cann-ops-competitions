# aclblasCgemvBatched 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务 2026，任务书《aclblasCgemvBatched 算子开发任务书》。目标代码仓为昇腾算子开源仓 `ops-blas`（https://gitcode.com/cann/ops-blas ），实现目录为 `blas/gemv_batched/arch35/`，测试代码合入 `test/gemv_batched/cgemv_batched/arch35/`。

本任务为补齐 Blas 仓能力：在 Ascend 950PR 上使用 Ascend C 编程语言开发单精度复数（complex64）批量矩阵-向量乘算子 `aclblasCgemvBatched`，对标 cuBLAS `cublasCgemvBatched`（legacy）。同时须修正 `include/cann_ops_blas.h:234-237` 现有的单指针声明为指针数组形态，与同族 `aclblasSgemvBatched`（`include/cann_ops_blas.h:243-246`）及 cuBLAS 保持一致。

### 背景介绍

`aclblasCgemvBatched` 属于 Level-2 GEMV 族的批量变体，对每个 batch `i ∈ [0, batchCount)` 计算：

```text
y_i = alpha * op(A_i) * x_i + beta * y_i
```

其中 `A_i` 为 m×n 列主序复数矩阵，`x_i`、`y_i` 为复数向量，alpha、beta 为复数标量；`op()` 取 N（不转置）、T（转置）、C（共轭转置）；`Aarray/xarray/yarray` 为指向 batchCount 个矩阵/向量首地址的指针数组。

现有仓内 `include/cann_ops_blas.h:234-237` 的 `aclblasCgemvBatched` 声明为单指针形态（`const aclblasComplex* A` / `*x` / `*y`），与 cuBLAS `cublasCgemvBatched` 的指针数组形态不一致，也与同文件 `aclblasSgemvBatched` 的指针数组形态不一致，属接口缺陷。本任务须同步修正该声明与 `blas/gemv_batched/arch22/cgemv_batched_host.cpp` 的入参与批量循环，禁止定义 950PR 私有平行接口或保留单指针重载。

## 需求分析（required）

### 需求描述

在 Ascend 950PR 上使用 Ascend C 实现 `aclblasCgemvBatched`，完成算子设计、开发、测试全流程。算子需与 cuBLAS `cublasCgemvBatched` 参数序列和核心语义对齐，支持 COMPLEX64 批量矩阵-向量乘、N/T/C 转置组合、alpha/beta 复数标量、指针数组、列主序前导维、正负步长、确定性计算，并满足任务书规定的参数校验顺序、精度标准、性能标准与自验要求。验收通过后合入 `ops-blas` 仓 `blas/gemv_batched/arch35`。

### 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 修正 `include/cann_ops_blas.h:234-237` 声明为指针数组形态，同步 `blas/gemv_batched/arch22/cgemv_batched_host.cpp` 入参与批量循环；不定义 950PR 私有平行接口，不保留单指针重载 | 代码审查；头文件比对；arch22 host 同步修改 |
| R2 | 实现 `y_i = alpha * op(A_i) * x_i + beta * y_i`，逐 batch 计算，COMPLEX64 输入输出 | 功能用例；golden 由 cblas 复数实现生成 |
| R3 | 支持 trans ∈ {N, T, C}；N 时 x 长 n、y 长 m；T/C 时 x 长 m、y 长 n；A 始终按 m×n 列主序存储，`lda ≥ max(1,m)` 与 trans 无关 | 转置组合用例；非法枚举负向用例 |
| R4 | 支持 `Aarray/xarray/yarray` 指针数组，逐 batch 独立计算；数组指针本身及任一元素为 nullptr 均返回 `ACLBLAS_STATUS_INVALID_VALUE` | 多 batch 用例；空指针负向用例 |
| R5 | 支持 incx/incy 正负步长；`incx=0`、`incy=0`、`incx=INT_MIN`、`incy=INT_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE` | 步长组合用例；负向用例 |
| R6 | 参数校验顺序符合任务书 §2.5：handle → m/n → lda → incx/incy/batchCount → 数组指针 → alpha/beta → trans；任一失败不启动 kernel | 负向用例；状态码断言 |
| R7 | 零维/零批不是 no-op：`m≤0`、`n≤0`、`batchCount≤0` 均返回 `ACLBLAS_STATUS_INVALID_VALUE`，不启动 kernel、不写 yarray | 边界负向用例 |
| R8 | 确定性计算：相同输入两次执行结果逐位一致 | 确定性用例；重复执行比对 |
| R9 | 精度满足 COMPLEX64 按 FLOAT32 分量判定：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP；含 Inf/NaN 特殊值规则 | 自测报告；配套用例 |
| R10 | 性能：单 case 平均单次耗时 ≤ GPU 标杆 2.5 倍（≥0.4× GPU）；msprof op 5 次 warmup、10 次有效采样平均 | 自测报告性能截图 |
| R11 | 复用仓内 gemv kernel 直调，实现放在 `blas/gemv_batched/arch35/`，不复制整套第三方源码 | 代码审查；目录结构 |
| R12 | 自验覆盖转置全组合、m/n/batchCount 扫描、标量特殊值、前导维 padding、步长组合（含负步长）、边界负向、Inf/NaN、性能用例 | 测试代码 + 自测报告 |
| R13 | 交付设计文档 PR、自测用例及测试代码、自测报告、待验收代码地址；README 产品支持表标注 Ascend 950PR：支持 | 验收交付件审查 |

## 详细设计（required）

### 算子分析

#### 数学公式

对每个 batch `i ∈ [0, batchCount)`：

```text
y_i = alpha * op(A_i) * x_i + beta * y_i
```

逐元素（以 op=N 为例）：

```text
y_i[r] = alpha * Σ_{c=0}^{n-1} A_i[r][c] * x_i[c] + beta * y_i[r],  r ∈ [0, m)
```

op=T/C 时 A 视为 n×m，`y_i` 长度为 n，`x_i` 长度为 m，按转置/共轭转置索引累加。

#### 接口定义

修正 `include/cann_ops_blas.h:234-237` 为指针数组形态：

```cpp
aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const xarray[], int incx,
    const aclblasComplex* beta,
    aclblasComplex* const yarray[], int incy,
    int batchCount);
```

参数序列与 cuBLAS `cublasCgemvBatched` 一一对应；`Aarray/xarray/yarray` 均为指针数组；`aclblasComplex` 为 complex64（实部/虚部各 float32）。须同步修正 `blas/gemv_batched/arch22/cgemv_batched_host.cpp` 的入参与批量循环，保证 950PR 实现与其它产品线共用同一 API。

#### 数据类型与形状

- 主路径：COMPLEX64（`aclblasComplex`），alpha/beta 为复数标量指针，Host 内存（与 cuBLAS 默认 pointer mode = HOST 一致）。
- `m ≥ 1`、`n ≥ 1`、`batchCount ≥ 1`；零维/零批为非法，返回 `ACLBLAS_STATUS_INVALID_VALUE`（按本仓 arch35 同族口径，比 cuBLAS quick return 更严格）。
- A_i 始终为 m×n 列主序矩阵，`lda ≥ max(1, m)`，与 trans 无关。
- op=N：x 长 n、y 长 m；op=T/C：x 长 m、y 长 n。
- `incx ≠ 0`、`incy ≠ 0`，且均不可为 `INT_MIN`；支持负步长（从向量尾部反向遍历）。
- 不支持超出 lda/incx/incy 语义的非连续访问；不涉及广播；不要求 dynamic shape。

#### 转置语义

- `trans = ACLBLAS_OP_N`：op(A_i) = A_i，y 长 m；
- `trans = ACLBLAS_OP_T`：op(A_i) = A_i^T，y 长 n；
- `trans = ACLBLAS_OP_C`：op(A_i) = A_i^H（复数共轭转置），y 长 n；
- 取值不在 {N, T, C} 返回 `ACLBLAS_STATUS_INVALID_ENUM`。

#### 参数校验顺序

按任务书 §2.5 固定顺序，全部在 Host 侧完成，任一失败均不启动 kernel：

```text
handle == nullptr                         → HANDLE_IS_NULLPTR
m ≤ 0 或 n ≤ 0                            → INVALID_VALUE
lda < max(1, m)                           → INVALID_VALUE
incx == 0 / incy == 0 / INT_MIN
  / batchCount ≤ 0                        → INVALID_VALUE
Aarray / xarray / yarray 为 nullptr       → INVALID_VALUE
（含任一数组元素为 nullptr）              → INVALID_VALUE
alpha / beta 为 nullptr                   → INVALID_VALUE
trans ∉ {N, T, C}                         → INVALID_ENUM
```

指针数组元素为 nullptr 时须在调用前逐元素校验，不得解引用空元素。

### 算子实现

#### Host 侧设计

Host 侧负责接口校验、指针数组处理、tiling 计算和 kernel 下发。

1. **参数校验**：严格按上述顺序实现；`m/n/batchCount` 非正、`lda` 不足、零步长/INT_MIN 步长、数组指针或元素为空、alpha/beta 为空、非法枚举分别返回对应状态码；任一失败直接返回，不下发 kernel、不写 yarray。
2. **指针数组处理**：`Aarray/xarray/yarray` 为 Device 内存中的指针数组，Host 侧逐元素判空但不解引用；Kernel 通过 batchCount 和数组首地址在 Device 侧按 batch 索引读取对应矩阵/向量首地址。
3. **Tiling 计算**：按 m、n、trans、lda、incx、incy、batchCount 生成 tilingkey；按 950PR 的核数、UB 容量、double buffer 需求计算分核与分块参数。batch 维度可映射到 block 维或核内循环：小 batchCount 大向量优先向量内并行，大 batchCount 小向量优先 batch 间并行。负步长通过反向起点 `kx = 1-(n-1)*incx`（或 op=T/C 时的 m 维对应公式）计算，读取范围不变。
4. **Kernel 下发**：通过 handle 绑定 stream，Ascend C kernel 直调方式下发；异步执行，读回 Device 结果前调用方同步 stream。
5. **确定性**：同一输入、同一 stream、同一 tiling 下 kernel 执行顺序固定，不引入原子累加或非确定规约；保证两次执行逐位一致。

#### Kernel 侧设计

Kernel 侧复用仓内 gemv kernel 直调，按 Init + Process 组织，逐 batch 复用同一 gemv 计算模板。

1. **Batch 索引**：从 `Aarray/xarray/yarray` 按 batch 索引读取第 i 个矩阵/向量首地址；batch 循环映射到 block 维或核内循环，由 tiling 决定。
2. **数据搬入**：A_i 按列主序搬入 tile；op=N 直接读，op=T 通过转置索引或 Matmul/向量模板的 transpose 参数实现，op=C 在转置基础上对虚部取负。x_i 按 incx 步长读取，负步长从反向起点开始；COMPLEX64 实部/虚部分别搬入。
3. **矩阵-向量乘**：`alpha * op(A_i) * x_i` 按行（op=N）或按列（op=T/C）归约；复数乘加实部/虚部分离计算；归约顺序固定以保证确定性。
4. **Epilogue**：按 `y_i = alpha * (A_i * x_i) + beta * y_i` 写回；`y_i` 原值按 incy 步长读入参与累加后原地写回；beta 为复数标量；写回范围严格为 y_i 的逻辑长度（m 或 n），不越界写。
5. **边界与尾块**：m/n 非对齐、incx/incy 非 1、负步长、lda > m 的 padding 场景按 tile 切分处理；padding 不参与计算；batchCount 非 2 的幂时按实际数量循环，不越界读取指针数组。
6. **确定性**：归约不使用原子操作或非确定并行规约；同一 tiling 下执行顺序固定。
7. **异步与同步**：kernel 由显式 stream 下发，异步执行；Host 侧不做流同步。

#### 复用同族 gemv kernel

任务书要求对 batch 内各矩阵-向量乘复用 gemv kernel 直调，实现代码放在 `blas/gemv_batched/arch35/`。设计上：

- 优先复用现有 gemv 的 tiling、数据搬入、复数乘加归约和写回框架；
- 新增 `aclblasCgemvBatched` 的 Host 接口封装、指针数组逐元素校验、批量循环和 N/T/C 分支；
- 同步修正 `blas/gemv_batched/arch22/cgemv_batched_host.cpp` 的单指针入参与批量循环，使其与头文件指针数组声明一致，不破坏 arch22 既有功能；
- 不复制整套第三方源码，不新增与同族 gemv 重复的 kernel 主体；
- 若现有模板缺少复数共轭转置或负步长路径，在 arch35 内做最小扩展，保持公共模板默认行为不变并回归同族 gemv 直接调用样例。

#### 性能与内存设计

- 性能目标：单 case 平均单次耗时 ≤ GPU 标杆 2.5 倍（≥0.4× GPU）。标杆为 cuBLAS `cublasCgemvBatched`，NVIDIA 驱动 550.54、CUDA 12.4。采用 msprof op 采集 task duration，自带 5 次 warmup，warmup 后 10 次有效样本取平均；异常样本不计入并说明。
- 优化手段：batch 间并行与向量内并行按规模自适应；满核均分；double buffer；批量搬运；复数实部/虚部分离计算；小向量避免过度分核；大 batchCount 优先 batch 并行；负步长与 padding 场景避免额外拷贝。
- 内存：本任务不设额外内存门禁；不新增与问题规模无关的持久 Device 缓存；指针数组由调用方分配，Kernel 只读。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR | √ |

实现目录为 `blas/gemv_batched/arch35/`；README 产品支持表标注 Ascend 950PR：支持；接口声明放在 `include/cann_ops_blas.h`，供其他产品线共用。自验证与性能测试均在 Ascend 950PR、CANN 9.1.0 环境完成。

### 算子约束限制

- 主路径 COMPLEX64；alpha/beta 为复数标量指针，Host 内存，不可为 nullptr。
- `m ≥ 1`、`n ≥ 1`、`batchCount ≥ 1`；零维/零批为非法，返回 `ACLBLAS_STATUS_INVALID_VALUE`，不启动 kernel、不写 yarray。
- A_i 始终按 m×n 列主序存储，`lda ≥ max(1, m)`，与 trans 无关。
- x/y 通过 incx/incy 取元素，支持正负步长；`incx=0`、`incy=0`、`INT_MIN` 均非法。
- 不支持超出 lda/incx/incy 语义的非连续访问；不涉及广播；不要求 dynamic shape。
- yarray 原地更新：读入原值参与累加后写回同一内存，不返回视图。
- **要求确定性计算**：相同输入多次执行结果逐位一致；归约不使用原子操作或非确定并行规约。
- 接口声明须由单指针修正为指针数组形态，同步 `blas/gemv_batched/arch22/cgemv_batched_host.cpp`；禁止新增 950PR 私有平行接口或保留单指针重载。
- 合入代码遵循 `ops-blas` 仓 `CONTRIBUTING.md`，不得包含冗余文件或无用代码；PR 关联设计文档与任务书，附自验结果。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 cblas（Netlib BLAS 3.10.0 对应复数实现，实部/虚部单标杆比对）生成，输出全向量验证。COMPLEX64 按 FLOAT32 分量判定：rtol=atol=2^-13（1.2e-4）、required_matched_ratio=0.99、max_abs_error_limit=1e-2 或 32 ULP；`|actual-golden| ≤ atol + rtol×|golden|`，用例同时满足 matched_ratio 与 max_abs_error 限值则通过 | 任务书 §3.2；混合容差精度标准 |
| Inf/NaN 规则 | golden 与 actual 同为 NaN、同为同号 Inf，或一方 NaN 另一方 ±Inf，视为等效通过；一方有限另一方 Inf/NaN 不通过；非有限值不计入 matched_ratio 与 max_abs_error，单独记录总数及通过数；有限值满足 matched_ratio≥0.99 且 max_abs_error≤限值，且非有限值全部通过，用例通过 | 任务书 §3.2.4 |
| 性能标准 | Ascend 950PR，COMPLEX64 平均单次耗时 ≤ GPU 标杆 2.5 倍（单 case ≥0.4× GPU）；标杆为 cuBLAS `cublasCgemvBatched`（NVIDIA 驱动 550.54、CUDA 12.4）；msprof op 5 次 warmup、10 次有效样本平均；异常样本不计入并说明 | 任务书 §3.3 |
| 内存标准 | 不涉及额外内存门禁；不新增与问题规模无关的持久 Device 缓存 | 任务书 §3.4 |

自验要求：使用 ops-blas 仓 test 目录测试框架，CSV 描述用例，C++ GTest 加载 CSV 调用 `aclblasCgemvBatched`；golden 由 cblas 复数实现生成。覆盖 trans ∈ {N,T,C} 全覆盖与非法枚举、m/n 尺寸扫描（1、小质数、2 的幂及 ±1、非对齐、大规模）、batchCount 扫描（1、小质数、2 的幂及 ±1、非对齐至百级）、alpha/beta 特殊值（(0,0)、(±1,0)、纯虚数、大值）、lda 覆盖 max(1,m)/m+1/m+8/非对齐（与 trans 正交，trans≠N 时同样以 m 为下界）、incx/incy 覆盖 ±1/±2/±3 及 0/INT_MIN 负向、边界负向（m=0/m<0/n=0/n<0/batchCount=0/batchCount<0 均期望 `ACLBLAS_STATUS_INVALID_VALUE`，**不是 no-op**）、空指针（数组指针本身及任一元素）、Inf/NaN、性能用例；输出自测报告，含用例参数、精度对比结果及截图、性能数据及截图。

### 兼容性分析

- 本任务同时包含**接口修正**与**新增实现**：`include/cann_ops_blas.h:234-237` 的单指针声明修正为指针数组形态，与 cuBLAS `cublasCgemvBatched` 和同族 `aclblasSgemvBatched` 一致；同步修正 `blas/gemv_batched/arch22/cgemv_batched_host.cpp` 入参与批量循环。
- 实现放在 `blas/gemv_batched/arch35/`，复用同族 gemv kernel 模板；不修改公共模板默认行为；若需最小扩展，须回归同族 gemv 直接调用样例。
- 测试代码放在 `test/gemv_batched/cgemv_batched/arch35/`，文件结构参考主仓 blas 算子测试代码结构，包含 CSV。
- 零维/零批按本仓 arch35 同族口径返回 `ACLBLAS_STATUS_INVALID_VALUE`（比 cuBLAS quick return 更严格），该口径在设计与测试中明确，不视为兼容性破坏。
- 不涉及旧版本算子行为兼容；README 产品支持表标注 Ascend 950PR：支持。
- 遵循 `ops-blas` 仓 `CONTRIBUTING.md`，PR 不得包含冗余文件或无用代码；PR 关联设计文档与任务书，附精度、性能自验结果。

## 验收交付件

1. 本设计文档：按模板提交至 `cann-competitions` 仓库 `04_tasks/01_community-task-2026/tasklist` 对应任务目录，以 PR 形式评审合入。
2. 自测用例及测试代码：覆盖任务书配套全部自测用例，清晰列出精度 case 与性能 case；README 说明测试步骤，保证可复现。
3. 自测报告：按模板包含用例参数、精度对比结果及截图、性能数据（有效采样 >10 次平均值）及截图。
4. 待验收代码地址：个人代码仓链接、分支、算子目录；邀请账号 `Ascend-CANN` 作为开发者；README 按 ops-blas 规范提供，产品支持表标注 Ascend 950PR：支持；须包含 `include/cann_ops_blas.h` 中 `aclblasCgemvBatched` 声明由单指针修正为指针数组形态的改动，并同步 `blas/gemv_batched/arch22/cgemv_batched_host.cpp`。
5. PR 合入：测试通过后向 `ops-blas` 提交 PR，算子合入 `blas/gemv_batched/arch35`，测试代码合入 `test/gemv_batched/cgemv_batched/arch35`，遵循代码仓规范与贡献指南。