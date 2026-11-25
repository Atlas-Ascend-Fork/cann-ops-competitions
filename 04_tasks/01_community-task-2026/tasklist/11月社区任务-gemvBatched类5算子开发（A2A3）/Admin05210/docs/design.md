# aclblasSgemvBatched 类 5 算子设计文档

> **算子族**：`aclblasSgemvBatched` / `aclblasHSHgemvBatched` / `aclblasHSSgemvBatched` / `aclblasTSTgemvBatched` / `aclblasTSSgemvBatched`（实数批量矩阵-向量乘，BLAS-2）
> **任务**：11月社区任务-gemvBatched类5算子开发（A2A3）
> **目标架构**：Atlas A2/A3 系列（arch22，Ascend 910B3 等），基于 `cann/ops-blas` 仓 `blas/gemv_batched/arch22/` 开发
> **对标基线**：cuBLAS `cublasSgemvBatched` / `cublasHSHgemvBatched` / `cublasHSSgemvBatched` / `cublasTSTgemvBatched` / `cublasTSSgemvBatched`（数学语义对齐 Netlib BLAS `sgemv.f`）

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 初始方案设计：5 变体统一矩阵与校验顺序、Host 2D 分核与分块策略、Kernel 模板复用架构（OP_N 向量化 AXPY / OP_T(C) 分块点积 / 双缓冲流水 / 退化与 NaN 隔离）、精度与性能验收设计 | 2026-11-25 | Admin05210 |

# 需求背景

## 需求来源

昇腾社区任务（`11月社区任务-gemvBatched类5算子开发（A2A3）`）：在 Atlas A2/A3 系列产品（CANN 9.1.0）上使用 Ascend C 编程语言，基于 `cann/ops-blas` 开源仓句柄式 BLAS 工程模式（通过 handle 绑定 stream 直调 NPU kernel），实现单精度与半精度/BF16 实数批量矩阵-向量乘算子族 `aclblasSgemvBatched` 及其四个混合/半精度变体（HSH / HSS / TST / TSS，共 5 个 API），完成设计、开发与全量自验后合入 `cann/ops-blas` 仓 `blas/gemv_batched/arch22/` 与 `test/gemv_batched/arch22/`。

## 背景介绍

### gemvBatched 算子实现现状分析

当前 `cann/ops-blas` 仓中：
1. **公开接口声明已完备**：`include/cann_ops_blas.h` 已存在 `aclblasSgemvBatched` / `aclblasHSHgemvBatched` / `aclblasHSSgemvBatched` / `aclblasTSTgemvBatched` / `aclblasTSSgemvBatched` 共 5 个公开声明，与其他产品线共用，本任务零改动且不新增私有平行 API。
2. **arch22 实数变体缺失**：`blas/gemv_batched/arch22/` 当前仅有复数实现 `cgemv_batched_*`，缺少上述 5 个实数变体的 Host 入口、Tiling 与 Kernel 实现；`test/gemv_batched/arch22/` 尚无实数 5 变体的测试工程。
3. **跨架构参考关系**：`blas/gemv_batched/arch35/` 与 `test/gemv_batched/arch35/` 已有面向其他架构的 5 变体参考实现，可参考语义与变体分派框架，但需按 Atlas A2/A3（arch22）硬件向量架构、UB 存储层次与任务书状态码/边界规范独立设计实现，不得直接拷贝不兼容代码。

### gemvBatched 算子功能现状分析

对标 cuBLAS 对应 5 个 batched GEMV 接口，数学语义对齐 Netlib `sgemv.f`，批量计算 `y_i = alpha * op(A_i) * x_i + beta * y_i`（`i = 0, 1, ..., batchCount - 1`）。核心接口参数概要如下：

| 参数名 | 方向 | 含义 | 类型与格式 | 约束 | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | ops-blas 上下文句柄（绑定 stream） | `aclblasHandle_t` | 非空有效句柄 | `nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | 属性 | 矩阵运算类型（N / T / C；实数下 `OP_C ≡ OP_T`） | `aclblasOperation_t` | `{OP_N, OP_T, OP_C}` | 非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM` |
| `m`, `n` | 属性 | 物理矩阵 `A_i` 的行数与列数 | `int` | `m >= 0, n >= 0` | `< 0` → `ACLBLAS_STATUS_INVALID_VALUE`；`0` 为 quick return |
| `alpha`, `beta` | 输入 | Host 侧 FLOAT32 标量乘数指针 | `const float*` | 非空 | `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray` | 输入 | Device 指针数组，指向 `batchCount` 个列主序矩阵 `A_i` | 按变体（ND） | 每个 `A_i` 恒为 `lda × n` | 非空计算下 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 属性 | `A_i` 列主序前导维度 | `int` | `lda >= max(1, m)`（与 `trans` 无关） | `< max(1, m)` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `xarray` | 输入 | Device 指针数组，指向输入向量 `x_i` | 按变体（ND） | 长 `n`（OP_N）或 `m`（OP_T/C） | 非空计算下 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx`, `incy` | 属性 | `x_i`、`y_i` 元素步长（支持负步长反向遍历） | `int` | 非零且不为 `INT_MIN` | `0` 或 `INT_MIN` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `yarray` | 输出（原地） | Device 指针数组，指向原地覆写向量 `y_i` | 按变体（ND） | 长 `m`（OP_N）或 `n`（OP_T/C） | 非空计算下 `nullptr` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount` | 属性 | 批次数 | `int` | `batchCount >= 0` | `< 0` → `ACLBLAS_STATUS_INVALID_VALUE`；`0` 为合法 no-op |

# 需求分析

## 需求描述

在 Atlas A2/A3（arch22）上使用 Ascend C kernel 直调方式，在 `blas/gemv_batched/arch22/` 实现一套统一模板架构，完整覆盖 `aclblasSgemvBatched`、`aclblasHSHgemvBatched`、`aclblasHSSgemvBatched`、`aclblasTSTgemvBatched`、`aclblasTSSgemvBatched` 5 个接口，全面对齐 cuBLAS / Netlib `sgemv` 功能语义与任务书状态码约束，满足生态算子精度标准与 `0.8×` GPU kernel-only 性能标杆。

## 需求拆解

1. **多变体统一功能语义**：支持 FP32（S）、FP16（HSH）、FP16→FP32（HSS）、BF16（TST）、BF16→FP32（TSS）5 种精度组合；半精度/BF16 接口以 `uint16_t` 承载位模式，Host 标量 `alpha`/`beta` 恒为 FLOAT32，计算均在 FP32 域完成累加并按输出 dtype 收窄。
2. **布局与转置语义**：`A_i` 按列主序存储，物理尺寸恒为 `lda × n` 且 `lda >= max(1, m)`（与 `trans` 无关）；`trans` 仅决定逻辑算子 `op(A_i)`；实数算子 `OP_C` 与 `OP_T` 完全等价（不做共轭）。
3. **参数校验与状态码**：严格按固定校验序拦截非法入参；`handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法 `trans` 返回 `ACLBLAS_STATUS_INVALID_ENUM`（ops-blas 与 cuBLAS 的有意差异）；其余结构或指针非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. **边界与退化语义**：`batchCount == 0`、`m == 0` 或 `n == 0` 为合法 quick return（不读取 Device 内存、不启动 kernel，此时指针数组允许为 `nullptr`）；`alpha == 0` 时仅计算 `y = beta * y` 且禁止读取 `A`/`x`（隔离 `x` 含 NaN）；`beta == 0` 时禁止读取旧 `y`（隔离 `y` 含 NaN）；支持任意非零正/负步长与非连续 `lda` padding。
5. **精度与性能验收**：精度覆盖 5 变体共 5000 条（每变体 1000 条），满足生态算子开源混合容差标准与测试工程 MERE/MARE 校验；性能覆盖 5 变体共 1000 条（每变体 200 条），以 `msprof op` kernel `Task Duration(us)` 为正式判据，达到 NVIDIA A100 cuBLAS kernel-only 标杆 `0.8×` 及以上。
6. **工程化与可维护性**：采用模板化共享核心实现 5 变体，避免代码冗余；与同目录复数 `cgemv_batched_*` 及 `arch35` 实现零冲突共存；同步补齐 `test/gemv_batched/arch22/` 测试工程与 `blas/gemv_batched/README.md` 产品支持矩阵。

# 详细设计

## 算子分析

### 数学公式

对每个独立批次 `i ∈ {0, 1, ..., batchCount - 1}`：

```text
y_i = alpha * op(A_i) * x_i + beta * y_i
```

其中：
- 物理矩阵 `A_i` 按列主序存储，尺寸恒为 `lda × n`（`lda >= max(1, m)`，列间 padding 元素 `[m, lda)` 不参与计算），第 `(r, c)` 元素位于基址偏移 `c * lda + r`。
- 当 `trans == ACLBLAS_OP_N` 时，`op(A_i) = A_i`（`m × n`），输入 `x_i` 逻辑长度 `n`，输出 `y_i` 逻辑长度 `m`：
  ```text
  y_i[r] = alpha * Σ_{c=0}^{n-1} A_i[r + c * lda] * x_i[c] + beta * y_i[r],   0 <= r < m
  ```
- 当 `trans == ACLBLAS_OP_T` 或 `ACLBLAS_OP_C` 时，`op(A_i) = A_i^T`（`n × m`，实数不做共轭，`OP_C ≡ OP_T`），输入 `x_i` 逻辑长度 `m`，输出 `y_i` 逻辑长度 `n`：
  ```text
  y_i[c] = alpha * Σ_{r=0}^{m-1} A_i[r + c * lda] * x_i[r] + beta * y_i[c],   0 <= c < n
  ```
- 步长索引映射（对齐 Netlib `sgemv`）：对逻辑长度为 `L`、步长为 `inc != 0` 的向量，第 `k ∈ [0, L)` 个逻辑元素对应物理偏移：
  ```text
  phys(k) = k * inc                 (inc > 0)
  phys(k) = (L - 1 - k) * (-inc)    (inc < 0)
  ```
  步长间隙内的物理元素不属于逻辑向量，计算过程中保持原值不改写。

### 支持数据类型

| 变体 | 公开 API 名称 | `A_i` / `x_i` 类型（接口载体） | `y_i` 类型（接口载体） | `alpha` / `beta` | 内部累加域 | 输出收窄 |
| --- | --- | --- | --- | --- | --- | --- |
| **S** | `aclblasSgemvBatched` | FP32（`const float*`） | FP32（`float*`） | FLOAT32（Host） | FP32 | 无（直出 FP32） |
| **HSH** | `aclblasHSHgemvBatched` | FP16（`const uint16_t*`） | FP16（`uint16_t*`） | FLOAT32（Host） | FP32 | FP32 → FP16 |
| **HSS** | `aclblasHSSgemvBatched` | FP16（`const uint16_t*`） | FP32（`float*`） | FLOAT32（Host） | FP32 | 无（直出 FP32） |
| **TST** | `aclblasTSTgemvBatched` | BF16（`const uint16_t*`） | BF16（`uint16_t*`） | FLOAT32（Host） | FP32 | FP32 → BF16 |
| **TSS** | `aclblasTSSgemvBatched` | BF16（`const uint16_t*`） | FP32（`float*`） | FLOAT32（Host） | FP32 | 无（直出 FP32） |

### 支持形状

- `m`、`n`、`lda`、`incx`、`incy`、`batchCount` 均为运行时 `int` 参数，Host 侧按实参动态计算分核与分块（测试覆盖 `m, n ∈ [0, 4096]`、`batchCount ∈ [0, 256]`、紧凑及 `+1/+8/+64` padding 的 `lda`、以及 `±1/±2/±3` 到 `±127` 跨步）。
- `Aarray`、`xarray`、`yarray` 为 Device 侧指针数组，各 batch 数据块之间不要求连续；不支持广播、不涉及 0 维张量。
- 矩阵与跨步向量字节偏移全程使用 `uint64_t` 计算，规避大维度 × 大步长下的 32 位整型溢出。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

Host 入口以统一模板 `GemvBatchedLaunch<InDtype, OutDtype>(...)` 承接 5 个公开 API，完成参数校验、quick return、Tiling 计算与 stream 绑定下发。

**参数校验顺序与状态码表**（严格先验结构参数，结构合法后才允许 quick return）：

| 顺序 | 校验项 | 触发条件 | 返回状态码 |
| --- | --- | --- | --- |
| 1 | `handle` | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `trans` | 不属于 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `m`, `n` | `m < 0 \|\| n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `lda` | `lda < max(1, m)`（与 `trans` 无关） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx`, `incy` | `incx == 0 \|\| incy == 0 \|\| incx == INT_MIN \|\| incy == INT_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `batchCount` | `batchCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `alpha`, `beta` | `alpha == nullptr \|\| beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | 指针数组条件判空 | `batchCount > 0 && m > 0 && n > 0` 时，若 `yarray == nullptr`，或（`*alpha != 0` 时）`Aarray == nullptr \|\| xarray == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 9 | quick return | `batchCount == 0 \|\| m == 0 \|\| n == 0`，或 `*alpha == 0.0f && *beta == 1.0f` | `ACLBLAS_STATUS_SUCCESS`（不启动 kernel） |

> 注：非空计算下若任务书或测试套要求 `alpha == 0` 仍校验三数组非空，亦可在入口策略中按统一条件判空处理并在 kernel 侧保证绝不解引用 `Aarray`/`xarray`，确保既符合参数校验契约又完全隔离 `x` 含 NaN。

##### 1. 分核策略：

- **2D `(batch, output_tile)` 逻辑任务网格**：记输出逻辑长度 `outLen = (trans == OP_N) ? m : n`，归约长度 `kLen = (trans == OP_N) ? n : m`。Host 运行时查询当前设备可用 AIV 核数 `aivNum`（不硬编码），按输出瓦片粒度 `outTile` 将每个 batch 划分为 `tilesPerBatch = ceil(outLen / outTile)` 个互不重叠的输出瓦片，构建总任务数 `totalTasks = batchCount × tilesPerBatch`，以 `(taskId / tilesPerBatch, taskId % tilesPerBatch)` 映射回 `(batchIdx, tileIdx)`，再采用负载均衡的区间切分将 `totalTasks` 均匀分配给 `usedCores = min(totalTasks, aivNum)` 个 AIV 核。
- **场景自适应并行度**：
  - 大 `batchCount` × 小尺寸场景：`tilesPerBatch = 1`，全核沿 batch 维并行，避免过度切分带来的固定启动与循环开销；
  - 小 `batchCount`（如 `batchCount = 1/2/4`）× 大尺寸场景（如 `4096×4096`、`4096×64`、`64×4096`）：沿 `outLen` 维细粒度切分出足够多的 `output_tile`，充分打满多核带宽；
  - 非单位输出步长（`incy != 1`）或输出极短场景：限制单输出向量不跨核写回（`tilesPerBatch = 1`），仅在 batch 维多核并行；单位步长多核切分时确保瓦片边界按 64B cache line 对齐，彻底避免多核写回相邻内存互相覆盖。
- **零跨核依赖**：每个输出元素 `y_i[p]` 仅归属于唯一任务块，全流程无跨核归约、无原子操作、无额外 workspace。

##### 2. 数据分块和内存优化策略：

- **UB 空间规划与二维分块 `(tileM, tileN)`**：根据输入/输出元素位宽（2B/4B）和当前平台 UB 可用容量，在 Host 侧动态确定 `(tileM, tileN)`，为 `A` 输入块、类型转换缓冲（FP16/BF16→FP32）、`x` 向量块、FP32 累加器与旧 `y` 块预留 32B 对齐的静态分区，确保总占用满足 UB 容量上限。
- **Ping-Pong 双缓冲流水**：在归约维循环内对 `A` 分块（及必要时的 `x` 分片）配置双缓冲队列，通过异步搬运与计算解耦，实现 MTE2 搬运与 Vector 向量乘加/归约重叠执行，提升带宽利用率。
- **指针数组批量预取**：当单核处理多个 batch 时，按小批量将 Device 侧 `Aarray`/`xarray`/`yarray` 指针表段连续搬入 UB 缓存并从中读取各 batch 地址，消除逐 batch 访问 GM 指针表的标量延迟。
- **非对齐与尾块处理**：矩阵列段及向量尾块统一使用 `DataCopyPad` 完成 GM↔UB 搬运，原生兼容任意非对齐 `lda`（如 `4095`、+1 padding）与非对齐尺寸，padding 通道显式清零，避免无效数据参与乘加。

##### 3. 路径分派策略（kernel 直调，无 TilingKey）：

本算子为 ops-blas 句柄式 kernel 直调形态，无编译期 TilingKey 注册表；5 种数据类型组合与 `IsTrans`（`OP_N` vs `OP_T/C`）通过 Host 侧编译期模板选择对应核函数实例，其余运行时分支由按值传入 kernel 的轻量 POD `TilingData` 中的 `pathMode` 字段驱动：

| 路径 | 触发条件 | 核心职责 |
| --- | --- | --- |
| `P0` AlphaZero 路径 | `alpha == 0`（`beta == 1` 已在 Host quick return） | 仅执行 `y = beta * y`（`beta == 0` 置零且不读旧 `y`），绝不访问 `Aarray`/`xarray` |
| `P1` OP_N 向量化主路径 | `trans == OP_N` 且单位/规整步长 | 列主序连续读取 `A` 列段，双缓冲 + 向量化 FP32 AXPY 累加 |
| `P2` OP_T/C 向量化主路径 | `trans ∈ {OP_T, OP_C}` 且单位/规整步长 | 列主序连续读取 `A` 列块与 `x` 分片，双缓冲 + 向量化分块点积与归约 |
| `P3` 通用跨步/边界路径 | 非单位步长、负步长、大步长或极小规约/输出形态 | 通用步长映射（含反向索引）+ FP32 累加，保证全域正确性与间隙元素零污染 |

#### 3.2.2 kernel侧设计：

所有 5 个实数变体共享同一 AIV-only 向量核模板 `GemvBatchedKernel<InType, OutType, IsTrans>`，执行 `Init → Process` 两阶段，单次任务循环依次完成 `CopyIn → Cast/Compute → Epilogue/CopyOut`：

1. **列主序 `OP_N` 向量化 AXPY（P1）**：列主序下 `A_i` 同一列内连续存储。对当前 `(batchIdx, rowTile)`，初始化长度 `tileM` 的 FP32 累加器清零；内层沿 `n` 维按 `tileN` 步进，双缓冲连续搬入 `A_i[rowStart : rowStart+tileM, colStart : colStart+tileN]` 与对应 `x_i` 分片；对半精度/BF16 输入先经向量 `Cast` 提升至 FP32，逐列以标量 `x[j]` 对 `A` 列向量执行向量融合乘加（`Axpy`）累入行累加器；全部列遍历完成后执行 `acc = alpha * acc`（若 `beta != 0` 则搬入旧 `y` 转为 FP32 并计算 `acc += beta * y_old`），按输出 dtype 收窄写回 GM。
2. **列主序 `OP_T / OP_C` 向量化分块点积（P2）**：`OP_C` 直接复用 `OP_T` 实例。此时每个输出分量 `y_i[c]` 为物理列 `A_i[:, c]` 与 `x_i` 的内积，列内依然完全连续。对当前 `(batchIdx, colTile)`，外层分配多列局部 FP32 向量累加缓冲，内层沿 `m` 维按 `tileM` 分块将 `x_i` 切片与 `tileN` 列 `A_i` 数据双缓冲搬入 UB，类型提升后执行向量逐元素乘加累入各列部分和向量，减少内层归约频次；沿 `m` 维结束后对每列部分和向量做一次向量归约得到该列内积，最后统一缩放合成 `alpha * dot + beta * y_old` 并按输出 dtype 写回。
3. **FP32 累加域与按输出 dtype 收窄**：HSH / HSS / TST / TSS 均在搬入 UB 后将输入 `A`、`x`（及 `beta != 0` 时的旧 `y`）通过 `Cast` 转换为 FP32 进行全部乘加与标量运算，避免半精度/BF16 累加精度塌缩与中间溢出；写回阶段 S / HSS / TSS 直接输出 FP32，HSH / TST 分别转换为 FP16 / BF16 后写回。
4. **非单位步长、负步长与大步长处理（P3）**：统一通过 `phys(k)` 换算元素物理地址（负步长反向索引、全程 `uint64_t` 防溢出）。对小步长连续窗口可整段预取至 UB 再提取有效元素以降低零散访存次数；对大步长（如 `±64`/`±127`）或负步长按需分片搬运入连续 UB 缓冲参与 FP32 向量计算，写回时仅更新 `y_i` 的有效逻辑位置，绝不触碰步长间隙中的相邻内存。
5. **`alpha = 0` / `beta = 0` 退化与 NaN 隔离**：
   - `alpha == 0` 走 P0 路径，完全不读取 `Aarray`/`xarray`，即使 `A`/`x` 含 `NaN`/`Inf` 也不会影响 `y`；当 `beta == 0` 时直接向 UB 输出缓冲填零写回（不读取旧 `y`），其余 `beta` 读入旧 `y` 经 FP32 缩放后写回。
   - 非 P0 路径下 `beta == 0` 同样完全跳过旧 `y` 读取与 `beta * y_old` 项，彻底阻断旧 `y` 含 `NaN` 时的 `0 × NaN = NaN` 污染；正常输入中的 `Inf`/`NaN` 按 IEEE 754 规则自然传播。
6. **流水与同步规范**：MTE2（GM→UB）、Vector（类型转换/乘加/归约）与 MTE3（UB→GM）间采用显式配对同步原语管理双缓冲槽位复用；不使用 `GlobalTensor::SetValue/GetValue`；尾块读入的无效 lane 在参与归约或点积前显式置零，防止残余脏数据或 `0 × Inf` 引入异常值。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 系列（Ascend 910B3 等，arch22） | √ |
| Atlas A3 系列（arch22） | √ |

## 算子约束限制

1. 仅支持 S（FP32）、HSH（FP16）、HSS（FP16→FP32）、TST（BF16）、TSS（BF16→FP32）5 种实数变体，复数变体不属本任务范围；半精度/BF16 以 `uint16_t` 指针数组承载。
2. 物理矩阵 `A_i` 恒为列主序 `lda × n` 且 `lda >= max(1, m)`（与 `trans` 无关）；不支持超出 `lda`/`incx`/`incy` 语义的非连续布局，不支持广播，不涉及 0 维张量。
3. `y_i` 为原地覆写，不返回视图；调用方需保证各 batch 的 `y_i` 之间互不重叠且与 `A_i`/`x_i` 不重叠。
4. 算子在 `handle` 绑定的 stream 上异步执行，Host 侧读取结果前须完成 stream 同步；浮点并行累加顺序不要求逐位确定性。
5. 负步长支持任意非零 `int`（不含 `INT_MIN`），`batchCount = 0` / `m = 0` / `n = 0` 为合法 quick return，不启动 kernel。

# 可维可测分析

## 精度标准/性能标准

### 1. 精度验收标准

golden 由 CPU 侧 Netlib `cblas_sgemv` 逐 batch 生成（测试工程按 float64 累加后按输出 `y` 的 dtype 收窄，亦可与 float32 `cblas_sgemv` 对照并在自测报告说明差异）。逐元素满足 `|actual - golden| <= atol + rtol * |golden|` 计为 matched，整用例需同时满足 `matched_ratio >= 0.99` 且 `max_abs_error <= max_abs_error_limit`。

**生态算子开源混合容差标准（按输出 dtype）与测试工程 MERE/MARE 口径**：

| 适用变体 | 输出 dtype | 生态标准 `rtol` | 生态标准 `atol` | `required_matched_ratio` | `max_abs_error_limit` | 测试工程 CSV `mere_threshold` / `mare_multiplier` |
| --- | --- | --- | --- | --- | --- | --- |
| **S**（`aclblasSgemvBatched`） | FLOAT32 | `2^-13` (1.2e-4) | `2^-13` (1.2e-4) | 0.99 | `1e-2` 或 `32 * ULP` | `0.00012207031` / `10.0` |
| **HSH**（`aclblasHSHgemvBatched`） | FLOAT16 | `2^-10` (1e-3) | `2^-10` (1e-3) | 0.99 | `1e-1` 或 `32 * ULP` | `0.0009765625` / `32.0` |
| **HSS**（`aclblasHSSgemvBatched`） | FLOAT32 | `2^-13` (1.2e-4) | `2^-13` (1.2e-4) | 0.99 | `1e-2` 或 `32 * ULP` | `0.00012207031` / `10.0` |
| **TST**（`aclblasTSTgemvBatched`） | BFLOAT16 | `2^-7` (7.8e-3) | `2^-7` (7.8e-3) | 0.99 | `1e-1` 或 `32 * ULP` | `0.0078125` / `32.0` |
| **TSS**（`aclblasTSSgemvBatched`） | FLOAT32 | `2^-13` (1.2e-4) | `2^-13` (1.2e-4) | 0.99 | `1e-2` 或 `32 * ULP` | `0.00012207031` / `10.0` |

- **精度覆盖规模**：共 **5000 条**精度用例（5 变体各 1000 条，含 L0 基础、L1 尺寸 0..4096、L2 特殊标量、L3 batch 0..256、L4 正负/大步长、L5 `lda` padding 与极值/Inf/NaN 填充、L6 边界与负向校验、EX 确定性扩展采样），并单独补充 `handle == nullptr` 用例及 `alpha=0, x=NaN` / `beta=0, y=NaN` 隔离验证。

### 2. 性能验收标准与典型标杆 Case

- **正式判据（kernel-to-kernel）**：在 Atlas 800T A2（Ascend 910B3）上使用 `msprof op` 采集单算子 kernel `Task Duration(us)`，warmup 5 次不计入，连续采集 10 次有效样本取算术平均；要求单 case 满足 `NPU kernel avg <= gpu_ms × 1.25`（即达到 NVIDIA A100 cuBLAS kernel-only 基线 `0.8×` 及以上性能），且全量可比 case 平均倍率 `>= 0.8`。
- **辅助交叉验证（非正式判据）**：GTest 预置 Device 缓冲，warmup 5 次后经 `aclrtEvent` 采集连续 10 次下发均值打印 `[PERF] ... avg_us=...`，供 `verify_performance.py` 快速回归交叉验证，不替代 `msprof op` 正式判据。
- **性能覆盖规模**：共 **1000 条**性能用例（5 变体各 200 条 `TC_PF` 用例，覆盖大尺寸 4095/4096、尺寸扫描、多 batch 及跨步/非默认标量场景）。

下表摘录随任务 5 变体 `gpu_baseline.csv` 中的典型 A100 cuBLAS kernel-only 标杆耗时及对应 `0.8×` NPU 验收上限（当前为设计阶段 PR，NPU 实测耗时待实现与真机自验完成后随验收交付件提交）：

| 典型场景 | `trans` / `m × n` / `lda` / `incx,incy` / `batchCount` | S 基线 / 0.8×上限 (us) | HSH 基线 / 0.8×上限 (us) | HSS 基线 / 0.8×上限 (us) | TST 基线 / 0.8×上限 (us) | TSS 基线 / 0.8×上限 (us) |
| --- | --- | --- | --- | --- | --- | --- |
| 瘦高大矩阵 OP_N | `N` / `4096×64` / `4096` / `1,1` / `1` | 15.56 / 19.46 | 9.52 / 11.90 | 9.32 / 11.65 | 11.16 / 13.95 | 9.83 / 12.29 |
| 矮胖大矩阵 OP_T | `T` / `64×4096` / `64` / `1,1` / `1` | 9.63 / 12.03 | 10.96 / 13.70 | 11.26 / 14.08 | 10.65 / 13.31 | 11.06 / 13.82 |
| 大方阵对齐 OP_N | `N` / `4096×4096` / `4096` / `1,1` / `1` | 54.99 / 68.74 | 32.97 / 41.22 | 33.38 / 41.73 | 32.87 / 41.09 | 33.08 / 41.34 |
| 大方阵非对齐 OP_N | `N` / `4095×4095` / `4095` / `1,1` / `1` | 61.75 / 77.18 | 50.48 / 63.10 | 50.69 / 63.36 | 50.38 / 62.98 | 51.51 / 64.38 |
| 中批非对齐 OP_C | `C` / `33×65` / `33` / `1,1` / `64` | 9.52 / 11.90 | 9.93 / 12.42 | 9.83 / 12.29 | 9.63 / 12.03 | 9.83 / 12.29 |
| 大批跨步负载 OP_N | `N` / `4095×65` / `4095` / `2,-3` / `256` | 268.70 / 335.87 | 221.39 / 276.74 | 228.04 / 285.06 | 220.88 / 276.10 | 229.17 / 286.46 |

## 兼容性分析

1. **接口兼容性**：直接复用 `include/cann_ops_blas.h` 已有 5 个公开 API 声明，签名与 cuBLAS 逐参数一致，零头文件改动、零破坏性变更。
2. **目录与多架构共存**：实现代码位于 `blas/gemv_batched/arch22/`，与同目录已有复数 `cgemv_batched_*` 符号隔离共存；与 `blas/gemv_batched/arch35/` 及 `test/gemv_batched/arch35/` 物理隔离，不修改 arch35 代码与既有测试语义。
3. **测试可复现性**：测试工程落位于 `test/gemv_batched/arch22/`，在 `test/gemv_batched/CMakeLists.txt` 独立注册 arch22 目标，支持 CSV 驱动 GTest 精度自验、端到端性能交叉验证与 `msprof op` kernel-only 采集，并随交付包提供可复现说明文档。
