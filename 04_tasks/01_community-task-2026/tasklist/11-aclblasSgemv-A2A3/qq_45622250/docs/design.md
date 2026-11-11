# aclblasSgemv 算子设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 11月社区任务-aclblasSgemv算子开发（A2/A3） |
| 任务 taskId | `a8c0b7d0985349c59503ec0c34ca7344` |
| 任务链接 | [昇腾社区任务中心](https://www.hiascend.com/activities/task-center/details/a8c0b7d0985349c59503ec0c34ca7344) |
| 目标代码仓 | [cann/ops-blas](https://gitcode.com/cann/ops-blas)（master） |
| 实现目录 | `blas/gemv/arch22/` |
| 测试目录 | `test/gemv/sgemv/arch22/` |
| 适配硬件 | Atlas A2 / A3 训练与推理系列产品（arch22；自验覆盖 910B3） |
| CANN 版本 | CANN 9.1.0 |
| 设计文档提交路径 | `04_tasks/01_community-task-2026/tasklist/11-aclblasSgemv-A2A3/qq_45622250/docs/design.md` |
| 文档状态 | 设计评审阶段；本文不宣称代码、精度或性能已完成验收 |

# 需求背景（required）

## 需求来源

本算子来自“11月社区任务-aclblasSgemv算子开发（A2/A3）”（taskId：`a8c0b7d0985349c59503ec0c34ca7344`）。任务要求在 Atlas A2/A3 系列产品上，基于 [ops-blas](https://gitcode.com/cann/ops-blas) 开源仓的 kernel 直调工程模式，使用 Ascend C 实现单精度实数通用矩阵-向量乘法 `aclblasSgemv`，对标 cuBLAS `cublasSgemv`（CUBLAS Library §2.6.2，参数语义、返回码与快速返回口径的唯一依据），验收通过后合入 ops-blas 仓。

## 背景介绍

### 算子定位

`aclblasSgemv` 是 BLAS Level-2 的通用矩阵-向量乘加接口：

```text
y = alpha * op(A) * x + beta * y
```

其中 A 为列主序 `m x n` 矩阵，`op(A)` 由 `trans` 决定：

- `ACLBLAS_OP_N`：`op(A) = A`，`k = n`，`len(x) = n`，`len(y) = m`；
- `ACLBLAS_OP_T`：`op(A) = A^T`，`k = m`，`len(x) = m`，`len(y) = n`；
- `ACLBLAS_OP_C`：`op(A) = A^H`；A 为实数矩阵，`C` 与 `T` 逐位等价，复用同一实现路径。

列主序下逻辑元素 `A[i][j]`（0-based，`i in [0,m)`、`j in [0,n)`）的物理偏移为：

```text
offset(A[i][j]) = i + j * lda
```

要求 `lda >= max(1, m)`；`lda > m` 时相邻列之间存在空洞，空洞不参与计算，且不得读写有效矩阵范围之外的内存。`x`、`y` 为一维逻辑向量，支持任意非零 `int32` 步长（含负步长），负步长按 Netlib 语义从物理 span 尾端起算：

```text
inc > 0: offset(k) = k * inc
inc < 0: offset(k) = (len - 1 - k) * (-inc)
```

y 为原地输入兼输出；`beta == 0` 时 y 的旧值不被读取；`m == 0` 或 `n == 0` 在参数校验通过后为合法快速返回，不修改 y。

### 仓内现状（基于 ops-blas master 核实）

| 项目 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| 公共接口声明 | `include/cann_ops_blas.h:468` 已存在 `aclblasSgemv` 声明，与 950PR 共用 | 直接复用，不新增/不修改公共头，不另立私有平行接口 |
| arch35（950PR） | `blas/gemv/arch35/sgemv_host.cpp`、`sgemv_kernel.cpp`、`sgemv_tiling_data.h` 已有 SIMT 实现 | 语义与校验口径的唯一源码锚点；kernel 实现不可直接用于 arch22 |
| arch22（A2/A3） | `blas/gemv/arch22/` 仅有复数版 `cgemv_host.cpp` 与两个 cgemv kernel，无 sgemv | 本任务开发主体 |
| 产品支持表 | `blas/gemv/README.md` 中 `aclblasSgemv` 标注 Atlas A2/A3 不支持 | 交付时更新为支持 |
| 测试工程 | `test/gemv/sgemv/` 仅有 `arch35/`，共享 `sgemv_param.h`、`sgemv_golden.h` | 随任务新增 `arch22/` 测试工程 |
| 构建机制 | `blas/CMakeLists.txt` 自动递归收集 `*/arch22/*.cpp` | 新增源文件无需修改构建脚本 |
| 测试注册 | 任务书发放基线描述为“同族 CMakeLists 只在 arch35 分支注册 GTest”；当前 master 的 `test/gemv/sgemv/CMakeLists.txt` 为无条件注册 | 交付时按 `test/gemv/cgemv/CMakeLists.txt` 的显式 arch 分支写法整理，确保 A2/A3 走 GTest runner |

> 说明：任务书 §2.3/§7 给出的声明行号为 L461-463 / L457-459；实测当前 master 为 `include/cann_ops_blas.h:468`（上游持续新增接口导致行号漂移）。设计以实际声明为准，函数签名与任务书 §2.3 逐字一致。

### 参考实现与工程骨架

| 参考 | 用途 |
| --- | --- |
| `blas/gemv/arch35/sgemv_host.cpp` 的 `ValidateSgemvParams`（L31）与 `aclblasSgemv` 入口（L92） | 参数校验顺序、返回码、quick return 口径 |
| `blas/gemv/arch22/cgemv_host.cpp`、`cgemv_no_trans_kernel.cpp`、`cgemv_do_trans_kernel.cpp` | 同 arch 的 N/T 两路径拆分、分核方式、UB 手工布局参考 |
| `blas/gbmv/arch22/sgbmv_host.cpp`、`sgbmv_kernel.cpp`、`sgbmv_kernel_l1.cpp` | 当前 arch22 最新工程范式：tiling 按值下发、`GetAivCoreCount()` 分核、`EnsureDefaultWorkspace`/`GetEffectiveWorkspace`、`SyncAll()` 跨核归约 |
| `blas/common/helper/host_utils.h`、`blas/common/helper/aclblas_handle_internal.h`、`blas/common/helper/kernel_constant.h` | 核数查询、handle 默认 32MB workspace、公共常量 |

### 口径裁决

1. **非法 trans 返回 `ACLBLAS_STATUS_INVALID_VALUE`**：任务书 §2.4/§2.5 与 arch35 实数实现的 `ValidateSgemvParams` 均为此口径；不复用复数版 `cgemv` README 中 `INVALID_ENUM` 的写法。
2. **A/x 判空与 alpha 无关**：arch35 Sgemv 在 `m>0 && n>0` 时要求 A 非空、逻辑长度非零时要求 x 非空，不因 `alpha == 0` 放宽；本设计保持一致，不复用 Cgemv 的 alpha 条件判空。
3. **alpha == 0 不做短路**：实现侧始终按 `y = 0 * op(A) * x + beta * y` 计算；golden `cblas_sgemv` 在 alpha=0 时短路为 `y = beta * y`，二者仅在 A/x 含 Inf/NaN 时不同，故该组合不设精度用例（任务书 §2.1/§2.5 已认可）。
4. **采用任务书 `lda` 口径**：`lda >= max(1, m)`，不是 LAPACK 带状存储口径。
5. **异步语义**：`aclblasSgemv` 内不调用 `aclrtSynchronizeStream`，返回成功仅表示已向 handle 绑定的 stream 提交；结果读取由调用方同步。

### 设计阶段范围

本 PR 仅提交设计文档，不包含 ops-blas 算子代码、测试代码与自测报告。代码实现与测试在评审通过后另行提交到 `cann/ops-blas`；本文中的性能与精度数值均为验收目标或公开基线，不是本地/NPU 实测结果。

# 需求分析（required）

## 需求描述

在 `blas/gemv/arch22/` 新增 `aclblasSgemv` 的 host 侧校验、tiling 与 Ascend C kernel，实现：

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

支持 FLOAT32 单精度实数、列主序、`N/T/C` 三种 trans、正负 `incx/incy`、`lda > m` 的列间空洞；y 原地更新；结果通过 handle 绑定 stream 异步下发。

参数序列与 cuBLAS `cublasSgemv` 逐位一致（handle、trans、m、n、alpha、A、lda、x、incx、beta、y、incy 一一对应），无需额外参数映射层。

## 需求拆解

| 编号 | 需求 | 说明 |
| --- | --- | --- |
| FR1 | 数学语义 | `y = alpha * op(A) * x + beta * y`，索引与长度随 trans 变化 |
| FR2 | trans 三态 | `N`、`T`、`C`；实数下 `C ≡ T`，不设共轭分支 |
| FR3 | 数据类型 | A/x/y 为 FLOAT32；alpha/beta 为 Host 内存 FLOAT32 标量指针 |
| FR4 | 参数校验 | 返回码与顺序对齐 arch35 `ValidateSgemvParams`；补齐跨度溢出守卫 |
| FR5 | quick return | 校验通过后 `m == 0 或 n == 0` 返回成功，不访问 A/x/y、不启动 kernel |
| FR6 | 非连续访问 | A 按 `lda` 跨列；x/y 按 `incx/incy` 跨元素；正负步长均生效 |
| FR7 | 原地与 beta 语义 | y 原地覆写；`beta == 0` 不读 y 旧值；`alpha == 0` 不短路 |
| FR8 | 异步执行 | 经 handle stream 下发，不阻塞、不内部同步 |
| FR9 | 精度 | FLOAT32 mixed_tolerance 口径：rtol=atol=2^-13，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32ULP |
| FR10 | 性能 | 910B3 上平均单次耗时不高于 GPU（A100）标杆的 1.25 倍，即 ≥ 0.8× 标杆 |
| FR11 | 交付 | arch22 算子代码、`test/gemv/sgemv/arch22/` 测试代码、CSV 用例、README 支持表、自测报告 |

## 公开接口与范围

接口声明复用 `include/cann_ops_blas.h:468` 既有原型，与 950PR 共用，不新增私有接口。实现范围：

- 新增 `blas/gemv/arch22/sgemv_host.cpp`、`sgemv_kernel.cpp`、`sgemv_kernel.h`、`sgemv_tiling_data.h`；
- 新增 `test/gemv/sgemv/arch22/sgemv_test.cpp`、`sgemv_npu_wrapper.h`、`sgemv_test.csv`；
- 调整 `test/gemv/sgemv/CMakeLists.txt` 的 arch 分支注册；
- 更新 `blas/gemv/README.md` 的产品支持表。

不修改：公共头 `include/cann_ops_blas.h`、arch35 实现、`cgemv` arch22 实现、公共 handle 结构。

## 任务书 §2.5 约束逐项核对

| §2.5 约束项 | 本设计结论 | 落地位置 |
| --- | --- | --- |
| 参数合法性与校验层级 | 完全对齐 arch35 顺序与返回码；新增跨度溢出守卫（arch35 尚未提供）；非法 trans 返回 `INVALID_VALUE` | `sgemv_host.cpp::ValidateSgemvParams` |
| 非连续 Tensor 支持 | 支持：A 按 `lda` 跨列、x/y 按 `incx/incy` 跨元素，正负步长均生效 | host 计算物理基址；kernel 统一 `StridedIndex` |
| broadcast 规则 | 不涉及：BLAS-2 矩阵-向量算子，无操作数间广播 | — |
| dynamic shape 要求 | 不要求：m/n/lda/incx/incy 为运行时入参，host tiling 按实参计算 | `CalcSgemvTiling` |
| 原地与视图语义 | y 原地覆写；`beta != 0` 读旧值，`beta == 0` 不读；A/x 只读 | kernel epilogue |
| 确定性计算要求 | 不要求；实现采用固定分块顺序，单次执行内部确定，但不承诺跨次 bit-exact | kernel 归约顺序 |
| 空 Tensor 处理 | 等价于 `m == 0 或 n == 0`：校验通过后直接返回成功，不访问 A/x/y、不启动 kernel | host quick return |
| 0 维 Tensor 处理 | 不涉及：无 0 维输入，标量由 Host 指针入参承担 | — |
| 异步执行 | 经 handle stream 异步下发；host 内不同步、不 D2H | `LaunchSgemvKernel` |

# 详细设计（required）

## 算子分析

### 数学公式

以 0-based 索引表示：

```text
trans = N:    y[i] = alpha * Σ(j=0..n-1) A[i + j*lda] * x[idx_x(j)] + beta * y[idx_y(i)],  i = 0..m-1
trans = T/C:  y[j] = alpha * Σ(i=0..m-1) A[i + j*lda] * x[idx_x(i)] + beta * y[idx_y(j)],  j = 0..n-1
```

其中：

```text
idx_x(k) = (incx > 0) ? k * incx : (len_x - 1 - k) * (-incx)
idx_y(k) = (incy > 0) ? k * incy : (len_y - 1 - k) * (-incy)
len_x = (trans == N) ? n : m
len_y = (trans == N) ? m : n
```

### 支持数据类型

| 项 | 取值 |
| --- | --- |
| A / x / y | FLOAT32 |
| alpha / beta | FLOAT32 标量，Host 内存指针 |
| 计算类型 | FLOAT32 累加（不做中间降精度） |

### 支持形状与布局

- `m >= 0`、`n >= 0`；`m == 0 || n == 0` 为合法快速返回；
- `lda >= max(1, m)`，支持 `lda > m` 的列间空洞；
- `incx != 0`、`incy != 0`、二者均不为 `INT32_MIN`；正负步长均支持；
- 逻辑向量物理跨度 `1 + (len - 1) * |inc|`，host 用 int64 计算并检查不超过 `INT32_MAX`；
- 不涉及广播、不涉及动态 shape 编译。

### 特殊语义

| 场景 | 行为 |
| --- | --- |
| `m == 0 或 n == 0` | 所有参数校验通过后返回 `ACLBLAS_STATUS_SUCCESS`，不修改 y、不启动 kernel |
| `beta == 0` | 不读取 y 旧值，可直接覆写；y 旧值即使为 NaN/Inf 也不影响结果 |
| `beta != 0` | 读取 y 旧值并按 `beta * y_old` 参与 epilogue |
| `alpha == 0` | 不短路；仍计算 `0 * op(A) * x + beta * y`（A/x 含 Inf/NaN 时按 IEEE 自然传播） |
| A/x/y 含 Inf/NaN | 不做特殊值判断，按 IEEE 规则自然传播；精度判定按任务书 §3.2 的非有限值等价规则 |
| `trans == C` | 实数矩阵共轭不变，逐位等价于 `T`，复用同一 kernel 路径 |

## 算子实现

### 文件与工程改动

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `blas/gemv/arch22/sgemv_host.cpp` | 新增 | 参数校验、quick return、分核/tiling、kernel launch |
| `blas/gemv/arch22/sgemv_kernel.cpp` | 新增 | Ascend C kernel：N 路径、T/C 路径、跨核归约、launcher |
| `blas/gemv/arch22/sgemv_kernel.h` | 新增 | `sgemv_kernel_do` 对外声明，供 host 引用 |
| `blas/gemv/arch22/sgemv_tiling_data.h` | 新增 | Host/Kernel 共享 tiling 结构 |
| `test/gemv/sgemv/arch22/sgemv_test.cpp` | 新增 | CSV 驱动 GTest（A2/A3） |
| `test/gemv/sgemv/arch22/sgemv_npu_wrapper.h` | 新增 | device 缓冲分配、stream 绑定、`INT32_MIN` 短路 |
| `test/gemv/sgemv/arch22/sgemv_test.csv` | 新增 | 随任务提供的 1200 条用例（1000 精度 + 200 性能） |
| `test/gemv/sgemv/CMakeLists.txt` | 修改 | 按 cgemv 写法补 arch22 显式 GTest 注册 |
| `blas/gemv/README.md` | 修改 | `aclblasSgemv` 产品支持表：Atlas A2/A3 改为“支持” |

文件数量与同族 arch22 算子（cgemv 4 个源文件、sgbmv host + kernel + 头文件）保持一致，不引入多余工程文件。

### Host（Tiling）设计

#### 参数校验顺序

校验顺序与 arch35 `ValidateSgemvParams` 一致，并新增任务书要求的跨度溢出守卫：

```text
1. handle == nullptr                                  → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. trans ∉ {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} → ACLBLAS_STATUS_INVALID_VALUE
3. m < 0                                              → INVALID_VALUE
4. n < 0                                              → INVALID_VALUE
5. lda < max(1, m)                                    → INVALID_VALUE
6. incx == 0 || incx == INT32_MIN                     → INVALID_VALUE
7. incy == 0 || incy == INT32_MIN                     → INVALID_VALUE
8. xLen > 0 且 (int64)(xLen - 1) * |incx| > INT32_MAX → INVALID_VALUE   【本任务新增】
9. yLen > 0 且 (int64)(yLen - 1) * |incy| > INT32_MAX → INVALID_VALUE   【本任务新增】
10. alpha == nullptr                                  → INVALID_VALUE
11. beta == nullptr                                   → INVALID_VALUE
12. m > 0 && n > 0 && A == nullptr                    → INVALID_VALUE
13. xLen > 0 && x == nullptr                          → INVALID_VALUE
14. yLen > 0 && y == nullptr                          → INVALID_VALUE
15. 全部通过后：m == 0 || n == 0                      → ACLBLAS_STATUS_SUCCESS（quick return）
```

要点：

1. `|inc|` 必须在排除 `INT32_MIN` 之后计算；`abs(INT32_MIN)` 为未定义行为。
2. 跨度计算一律使用 `int64_t`；`(len - 1)` 在 `len == 0` 时跳过，避免负数误判。
3. A/x/y 判空与 `alpha` 是否为 0 无关，严格对齐 arch35 Sgemv；不采用 Cgemv 的 alpha 条件判空。
4. 所有 `INVALID_VALUE` 分支返回同一错误码，1~14 的内部先后不影响返回码，但保持 arch35 顺序便于逐项对照。
5. `handle` 校验先于一切；`m == 0/n == 0` 的 quick return 位于全部校验之后，因此 `m=0` 时 `lda`、`incx/incy`、`alpha/beta` 仍须合法，A 可为空，逻辑长度为 0 的 x/y 可为空。

#### 分核与 tiling

定义：

```text
outDim = (trans == N) ? m : n      // 输出向量逻辑长度
redDim = (trans == N) ? n : m      // 归约维长度（N 为列方向，T/C 为行方向）
aivCoreNum = GetAivCoreCount()     // 910B3 实测通常为 40，以平台返回值为准；返回 0 → EXECUTION_FAILED
```

**一维分核（常规路径）**：

```text
MIN_OUT_PER_CORE = 32              // 单核至少 32 个输出元素（128B，4 个 32B 块）
outSplit = min(aivCoreNum, max(1, ceil(outDim / MIN_OUT_PER_CORE)))
redSplit = 1
```

每个 block 负责 `outDim` 上一段连续的输出区间，核间无写冲突、无归约、无原子操作。`outPerCore = ceil(outDim / outSplit)` 向上对齐到 8 个 float（32B），最后一个 block 按实际剩余长度截断。

**瘦长输出 + 跨核归约路径（split-K）**：

当 `outDim` 很小而 `redDim` 很大时，仅按输出维分核无法占满 AIV（典型：性能用例 `TC_PF_1025`，m=4、n=1048576，仅 4 个输出，4 核读 16MB 矩阵，带宽严重不足）。此时启用归约维二次切分：

```text
MIN_RED_PER_CORE = 8192            // 单核至少 8192 个归约元素（32KB）
redSplit = min(aivCoreNum / outSplit, max(1, redDim / MIN_RED_PER_CORE))
useCoreNum = outSplit * redSplit
```

- 每个 block 负责一个 `(outGroup, redGroup)` 块，计算该输出区间在归约维子区间上的部分和；
- 部分和写入 handle 默认 workspace，布局为 `partial[redGroup][outDimPadded]`，`outDimPadded` 为 `outDim` 向上对齐到 8；
- 所有 block 执行 `SyncAll()` 后，按 block 轮转分担输出区间的最终归约：对每个输出累加 `redSplit` 个部分和，再应用 `alpha` 与 `beta` 并写回 y；最终归约的输出切片按 8 个 float（32B）对齐切分，避免相邻 block 共享 32B 写扇区产生写竞争；未分到输出切片的 block 跳过写回，但仍须参与 `SyncAll()`；
- 仅当 `redSplit > 1` 才分配 workspace、才执行 `SyncAll()`；常规路径零 workspace、零同步。

workspace 规模上界：`redSplit > 1` 仅发生在 `outSplit < aivCoreNum` 时（本设计参数下 `outDim ≤ 640`）；最坏 `redSplit ≤ aivCoreNum`，所需字节数 ≤ `aivCoreNum * ceil8(outDim) * 4B`（约 ≤ 100KB），远小于 handle 默认 32MB；通过 `EnsureDefaultWorkspace(h, bytes)` 申请，失败（例如用户通过 `aclblasSetWorkspace` 设置了更小的 workspace）时回退 `redSplit = 1`，仅损失极端形状的并行度，不改变正确性。

**分核依据与性能取舍**：

- 本算子为 memory-bound：4096×4096 需读取 A 约 67.1MB，性能判据要求 ≤ 65.8us（GPU 52.63us / 0.8），即 ≥ 1.02TB/s 有效带宽，必须充分占用 AIV 核；
- `MIN_OUT_PER_CORE = 32` 允许 4096 行时使用全部 40 核（每核 103 行、每列片段 412B），同时避免 1×1/2×2 等极小形状启动大量空核；
- 列向 A 块搬运以 `colTile` 列为一组，用二维 DMA 一次搬入多列，摊薄小片段 DMA 开销；
- split-K 仅服务于“输出太小、归约维极长”的形状；常规形状不付 `SyncAll()` 与 workspace 成本（参考 `sgbmv_kernel_l1.cpp` 注释，`SyncAll()` 约 2us 量级，不可无条件加入每次 launch）。

#### tiling 结构与下发

```cpp
struct SgemvTilingData {
    uint32_t m;             // A 行数
    uint32_t n;             // A 列数
    uint32_t lda;           // A 列间距
    uint32_t trans;         // 0 = N，1 = T/C（host 归一化）
    float alpha;            // Host 标量
    float beta;             // Host 标量
    int64_t incx;           // x 步长（含符号）
    int64_t incy;           // y 步长（含符号）
    uint32_t outDim;        // N: m；T/C: n
    uint32_t redDim;        // N: n；T/C: m
    uint32_t useCoreNum;    // 实际 launch 的 block 数
    uint32_t outPerCore;    // 一维分核时单核输出元素数（8 对齐）
    uint32_t tileOut;       // 输出方向向量块长（如 256）
    uint32_t colTile;       // N 路径一次搬入的列数（如 32）
    uint32_t redTile;       // T/C 路径一次搬入的行数（如 1024）
    uint32_t redSplit;      // 归约维切分数，1 表示不切
    uint32_t wsStride;      // workspace 每行跨度（outDimPadded）
};
```

下发方式与 arch22 现行范式一致（参考 `sgbmv_kernel_do`）：host 调用 `sgemv_kernel_do(...)`，由后者在 `sgemv_kernel.cpp` 内完成 launch：

```cpp
// sgemv_kernel.cpp
void sgemv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workspace,
    const SgemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sgemv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workspace, tiling);
}
```

- tiling 按值拷贝进 kernel 参数，不分配 tiling GM、不做 H2C memcpy；
- 仅 split-K 路径从 handle 取 workspace，常规路径传 `nullptr`；
- host 内不做 `aclrtSynchronizeStream`，不做 D2H；
- kernel 入口设置 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，只用 AIV 向量核。

### Kernel 设计

#### 总体结构

单 kernel entry，按 `tiling.trans` 分派到两个设备函数：

```text
sgemv_kernel(a, x, y, workspace, tiling)
 ├─ blkIdx = GetBlockIdx(); if (blkIdx >= useCoreNum) return;
 ├─ trans == 0 → ProcessNoTrans(...)
 └─ trans == 1 → ProcessTrans(...)
```

两条路径共享：

- `StridedIndex(k, len, inc)`：64 位逻辑索引到物理偏移的映射；
- 8 对齐的输出分块与尾块截断；
- `alpha`/`beta` 的 epilogue 公共逻辑；
- `redSplit > 1` 时的 workspace 写入、`SyncAll()` 与最终归约。

#### trans = N 路径（按输出行分核，遍历列累加）

每个 block 负责输出行区间 `[r0, r1)`（以及 split-K 时的列子区间）：

1. 以 `tileOut` 为输出方向块长、`colTile` 为列方向块长，循环处理：
   - 从 GM 搬入 A 块：`tileOut x colTile`，列主序，每列 `tileOut` 个连续 float，列间 `lda`；
   - 搬入 `colTile` 个 x 标量（`incx == 1` 时连续 DMA，否则按 stride 取数）；
   - 对块内每一列执行 `acc += A_col * x[c]` 的向量 FMA（`vmla`/`Axpy`）；
2. 全部列处理完后进入 epilogue：
   - `acc = alpha * acc`；
   - `beta == 0`：不读 y，直接写回；
   - `beta != 0`：读 y 旧值，`y = acc + beta * y_old`，再写回；
3. split-K 时 epilogue 改为：把 `acc` 写入 workspace 的 `[redGroup][r0..r1)` 切片，不写 y。

`lda > m` 时，A 块列间 stride 取 `lda - tileOut`；空洞不读。`lda` 为 8 的倍数且 `r0` 8 对齐时使用多列二维 DMA；否则按列 `DataCopyPad`，保证非对齐 `lda` 也正确。

#### trans = T/C 路径（按输出列分核，遍历行归约）

每个 block 负责输出列区间 `[j0, j1)`（以及 split-K 时的行子区间）：

1. 对每条输出列 j：
   - 以 `redTile` 为行块长，搬入 A 列片段 `A[i0..i1, j]`（列内连续）与对应 x 片段（按 incx）；
   - 向量乘加进入 256B 向量累加器，块间继续累加；
   - 使用 `WholeReduceSum`/`cadd` 树形归约得到标量 `dot`；
2. epilogue 同 N 路径：`y[j] = alpha * dot + beta * y_old[j]`（`beta == 0` 不读旧值）；
3. split-K 时把 `dot` 的部分和写入 workspace，最终统一归约。

列内 A 片段天然连续（长度 m），DMA 效率高；`trans = C` 与 `T` 共用此路径，不设共轭分支。x 在 `m * 4B` 可驻留 UB（如 m ≤ 8192）时整段缓存并跨列复用，否则按 `redTile` 分段搬入；`incx != 1` 时按 stride 取数。

#### 瘦长输出（outDim 很小）的列向量化

当 `outDim < 32`（如 m=4、n=1048576 的 N 路径）时，若仍按“每列搬 4 个 float”处理，单次 DMA 仅 16B，效率极低。该场景启用列向量化：

- 以 `colTile`（如 256）列为一块，将 A 的 `outDim x colTile` 子块一次搬入 UB（`lda == outDim` 时物理连续，可整段 DMA）；
- 对每一行 r，将 A 行（列方向跨步为 `outDim`）与 x 块做向量乘加并按列归约，得到 `outDim` 个部分和；
- 计算量按列方向向量化，DMA 按大块进行，避免小片段搬运；
- 该路径的 `outDim` 个部分和与常规路径共享同一 split-K/workspace 归约框架。

#### 负步长、非对齐与尾块

```cpp
inline int64_t StridedIndex(int64_t k, int64_t len, int64_t inc)
{
    return (inc > 0) ? (k * inc) : ((len - 1 - k) * (-inc));
}
```

- 所有 GM 偏移使用 `int64_t`；即使 host 已将向量跨度限制在 `INT32_MAX` 以内，`j * lda` 等中间量仍可能超出 int32，统一按 64 位计算；
- `outPerCore`、`tileOut` 向上对齐到 8 个 float（32B），最后一块按 `min(tile, outDim - offset)` 截断；
- 列方向尾块（`n % colTile != 0`、`m % redTile != 0`）使用有效长度或 mask，脏数据不参与归约；
- `incy == 1` 时 y 按连续向量读写；`incy != 1` 时按逻辑索引逐元素读写（正确性优先，性能用例均为单位步长）；
- `incx != 1` 时 x 按 stride 取数，`incx == 1` 时连续 DMA。

#### UB 预算与双缓冲

按 910B3 单 AIV 192KB UB 做保守预算（实际可用容量以平台信息为准）：

| 缓冲区 | 典型大小 | 说明 |
| --- | --- | --- |
| A 块（双缓冲） | `tileOut * colTile * 4B * 2`（如 256×32×4×2 = 64KB） | MTE2 与 V 重叠 |
| 输出累加器 | `tileOut * 4B * 2`（如 2KB） | 可拆 2~4 路累加，降低舍入与依赖 |
| x 块 | `colTile * 4B` 或整段 `redTile * 4B` | 单位步长时连续 DMA |
| y 暂存 | `tileOut * 4B` | epilogue 读改写 |
| mask/临时 | ≤ 8KB | 尾块与非对齐路径 |

总占用控制在 128KB 以内，留出余量给 TQue/TBuf 管理与指令临时数据。A 块使用双缓冲（`BUFFER_NUM=2`），保证 MTE2 搬运与向量计算重叠；无双缓冲时带宽通常下降 30% 以上，是本算子达标的前提之一。

#### 同步、确定性与精度

- **同步**：仅 split-K 路径使用 `SyncAll()`；常规路径 kernel 内无跨核同步、无原子操作；
- **确定性**：不承诺跨次 bit-exact；实现采用固定的列/行分块顺序，单次执行内部确定；
- **累加策略**：
  - N 路径每个输出元素累加 `redDim` 项；使用 2~4 路轮转累加器（按列块分组），最后成对合并，缩短舍入链并提升 ILP；
  - T 路径以 256B 向量累加器承载列内部分和，`redTile` 块间累加，最后树形归约，天然接近 pairwise 求和；
  - 全程 FLOAT32，不做 float16/bfloat16 中间降精度；
  - 长归约（如 redDim = 1048576）相对误差估算约 `sqrt(redDim) * 2^-24 ≈ 6e-5`（以 float32 单位舍入 2^-24 计），低于 rtol = 2^-13 ≈ 1.2e-4；若实测个别用例逼近阈值，再评估 Kahan 补偿或增加累加路数（作为优化项，不改变接口语义）。

### 资源与调度

| 项 | 设计 |
| --- | --- |
| 线程/核模型 | AIV-only，`useCoreNum` 个 block；每 block 内单线程向量流水 |
| 双缓冲 | A 块使用 TQue 双缓冲；x 块单缓冲（容量小），y 暂存复用 A 块 buffer |
| workspace | 常规路径无；split-K 路径使用 handle 默认 32MB workspace 的前若干 KB，`EnsureDefaultWorkspace` 申请 |
| 分配与缓存 | 每次调用不 `aclrtMalloc`、不做 tiling H2C；不引入常驻数据缓存 |
| 并发 | 不同 handle 相互独立；同一 handle 遵循仓内既有 workspace 使用约定 |
| 异步 | 仅向 stream 下发 kernel，返回即表示提交成功 |

### 支持硬件

| 芯片版本 | 架构目录 | 状态 |
| --- | --- | --- |
| Atlas A2 训练系列产品（Atlas 800T A2，SOC `ascend910b3`） | arch22 | 首要验证与性能出具设备（910B3） |
| Atlas A2 推理系列产品（Atlas 800I A2，SOC `ascend910b3`） | arch22 | 设计支持，同 arch22 二进制 |
| Atlas A3 训练/推理系列产品（SOC `ascend910_93*`） | arch22 | 设计支持；按 ops-blas 顶层 CMake 的 `ascend910_93` 前缀匹配，随代码 PR 验证 |

任务书 §3.1 的“Atlas 800T A2 / 800I A2”产品别名统一映射到 `ascend910b3` 编译目标，不据此推导不同二进制。

### 算子约束限制

| 约束 | 处理 |
| --- | --- |
| 仅 FLOAT32 | 复数由同仓 `aclblasCgemv` 承担 |
| `lda < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx/incy == 0` 或 `INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 跨度 `(len-1) * abs(inc) > INT32_MAX` | `ACLBLAS_STATUS_INVALID_VALUE`（host int64 校验） |
| 非法 trans | `ACLBLAS_STATUS_INVALID_VALUE`（对齐 arch35 Sgemv） |
| 空 handle | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 正维度下 A/x/y 空指针 | `ACLBLAS_STATUS_INVALID_VALUE`（判空与 alpha 无关） |
| `alpha == 0` 短路 | 不支持/不实现；仍按公式计算 |
| 广播、0 维、动态 shape | 不涉及 |
| 确定性 bit-exact | 不承诺 |
| 非单位步长性能 | 正确性支持；性能用例均为单位步长，非单位步长不作性能承诺 |

# 可维可测分析

## 精度标准与验证

| 项 | 内容 |
| --- | --- |
| Golden | `cblas_sgemv`（Netlib BLAS 实数实现，与仓内 `test/gemv/sgemv/sgemv_golden.h` 一致），逐用例生成 CPU 真值 |
| 判定对象 | y（原地更新后的 FLOAT32 向量），逐元素判定 |
| 阈值 | FLOAT32：`rtol = atol = 2^-13`，`matched_ratio >= 0.99`，`max_abs_error <= 1e-2` 或 `32 * ULP` |
| 判定公式 | `abs(actual - golden) <= atol + rtol * abs(golden)`，同时满足 matched_ratio 与 max_abs_error 上限 |
| Inf/NaN | golden 与 actual 同为 NaN、同为同号 Inf、一方 NaN 另一方 ±Inf 视为等效；一方有限、另一方非有限判不通过；非有限元素不计入 matched_ratio，单独统计并通过规则判定 |
| alpha=0 特例 | 因实现不短路、golden 短路，`alpha=0 && A/x 含 Inf/NaN` 的组合不设精度用例；`alpha != 0` 的 Inf/NaN 用例照常覆盖 |
| 随机算子精度对比 | 不涉及：aclblasSgemv 不含随机数生成，输出由 A/x/alpha/beta 完全确定；任务书 §3.5 的均匀/正态分布仅用于测试输入生成 |
| 覆盖率 | 全量 1000 条精度用例；另按任务书 §3.5 补充正态分布（`GAUSS_NORM_5_2`）用例并纳入自验 |

精度风险与措施：

- 长归约（k 达 1048576）的舍入累积是主要风险；通过 2~4 路累加器 + 树形归约降低误差，全程 FLOAT32 不降精度；
- 若实测个别极端用例逼近阈值，优先增加累加路数、调整分块，再评估 Kahan 补偿；不通过修改判定口径规避。

## 性能标准与验证

| 项 | 内容 |
| --- | --- |
| 设备 | Atlas 800T A2（910B3） |
| 判据 | 平均单次耗时 ≤ GPU（A100）标杆耗时 / 0.8，即 ≥ 0.8× 标杆 |
| 标杆 | `test_cases/gpu_baseline.csv`（NVIDIA A100-SXM4-80GB，CUDA 12.8，cuBLAS 12.8.4，warmup 5 后有效采样均值） |
| 采样（任务书口径） | 按任务书 §3.3：msprof op 采集单算子 Task Duration（us），自带 5 次 warmup，warmup 后有效样本 ≥ 10 次取算术平均；异常样本不计入并在报告说明 |
| 辅助口径 | 测试工程同时用 `aclrtEvent` 打印 `[PERF] avg_us`（warmup 5 后连续下发 20/50 次取平均，含下发开销、不含 H2D/golden）；`verify_performance.py` 以该值与 `gpu_baseline.csv` 比对，用于工程内快速判定与交叉核对。自测报告同时给出两套数据，最终结论以任务书 §3.3 的 msprof 口径为准；若两者结论不一致，逐 case 说明原因 |
| 规模覆盖 | 200 条性能用例：4096×4096 的 N/T/C 锚点、1~4096 规模扫描、非 2 幂（4095/4097）、矩形、极端瘦长（4×1048576、16×65536 等）、trans 三态 |

带宽可行性：

- 4096×4096 锚点：A 约 67.1MB，GPU 52.63us → 达标线 65.8us → 910B3 需 ≥ 1.02TB/s；属纯 bandwidth-bound，需全核并行、A 只读一次、双缓冲重叠；
- 4×1048576：A 约 16.8MB，GPU 29.29us → 达标线 36.6us；若仅按 4 个输出分核（4 核）约需 100us，必然不达标，因此本设计引入 split-K：40 核各读约 0.42MB，理论 10us 量级，加上归约与同步仍有余量；
- 小规模（1×1、2×2 等）受 kernel launch 开销主导：核数按 `outDim` 自适应收缩，不做 split-K、不分配 workspace、不做同步，保持单次 launch 的最小开销。

## 测试工程与覆盖

测试工程位于 `test/gemv/sgemv/arch22/`：

- `sgemv_test.cpp`：复用 `test/gemv/sgemv/sgemv_param.h`、`sgemv_golden.h`，CSV 驱动 GTest；覆盖 `NullHandle` 等无法写入 CSV 的负向场景；
- `sgemv_npu_wrapper.h`：device 缓冲分配、stream 绑定与同步；对 `INT32_MIN` 步长先短路（原样传给接口），避免 wrapper 在 `std::abs(INT32_MIN)` 处未定义行为；
- `sgemv_test.csv`：任务提供的 1200 条用例（1000 精度 + 200 性能），列格式与 arch35 `sgemv_test.csv` 一致；
- `CMakeLists.txt`：按 `test/gemv/cgemv/CMakeLists.txt` 的写法补 arch22 分支，A2/A3 构建走 `ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)`：

```cmake
if("arch22" IN_LIST SOC_ARCH_DIRS OR "arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} sgemv_test)
else()
    ops_blas_add_tests(${OPS_BLAS})
endif()
```

> 任务书 §2.2 指出仓内同族 CMakeLists 只在 arch35 分支注册 GTest、A2/A3 会退回非 GTest runner；当前 master 的 `test/gemv/sgemv/CMakeLists.txt` 为无条件注册。交付时按上述显式分支写法整理（与 cgemv 同构），保证 A2/A3 构建确实走 GTest 并输出 `[PERF] avg_us`，不依赖不同基线版本的无条件注册行为。

覆盖矩阵：

| 维度 | 覆盖 |
| --- | --- |
| trans | N/T/C 三态 + 非法枚举（999） |
| m, n | 0、1、3、64、512、2048、4096、1048576×4、4×1048576、65536×16、16×65536、非方阵、负值 |
| lda | `max(1,m)`、`m+8`、`2m` 空洞、非对齐偏移 |
| incx/incy | 1、2、7、-1、-3、0、INT32_MIN、跨度溢出 |
| alpha/beta | 0.0、1.0、2.5、-1.5、0.1、10.0、Inf、NaN、nullptr |
| 填充 | 均匀分布、正态分布、Inf/NaN、NULLPTR |
| 边界/负向 | quick return（m=0/n=0 × trans 三态 × A/x/y 空指针组合）、负维度、lda 非法、步长非法、标量空指针、A/x/y 空指针 |
| 特例 | beta=0 不读 y、alpha=0 不短路、trans=C 与 T 等价、负步长尾端起算 |

补充要求：

1. **正态分布用例**：`gen_csv.py` 默认仅生成均匀分布；按任务书 §3.5 使用 `python3 gen_csv.py --dist mixed` 补充 `GAUSS_NORM_5_2`（μ∈[-5,5]、σ∈[0.1,2]）用例，或单独补充 CSV；在自测报告中说明补齐方式、条数与覆盖情况。
2. **负向用例缓冲区**：期望返回非 SUCCESS 的用例不得按 `(len-1)*|inc|` 预分配 device 缓冲；以单元素 dummy 缓冲或 `NULLPTR` 占位，由接口在读取数据前返回预期错误码即判定通过。
3. **m=0/n=0 用例**：经 wrapper 直连接口（不分配 device 缓冲），避免误触 kernel；`m=0` 且 `incx=0`、`lda=0`、`beta=nullptr` 等仍为负向用例，`m=0` 且 A=nullptr 等为合法用例。
4. **内存验收**：任务书 §3.4 不涉及内存要求；split-K 的 workspace 使用 handle 默认 32MB 的前若干 KB，自测报告记录实际峰值即可。

### 自验步骤（代码验收阶段执行）

```bash
# 1. 精度（A2 用 ascend910b，A3 用 ascend910_93；脚本会把 CSV 复制到 test/gemv/sgemv/arch22/）
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b --csv ./sgemv_test.csv

# 2. 性能快速判定（读取 GTest 打印的 [PERF] avg_us 并与 gpu_baseline.csv 比对）
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend910b

# 3. msprof op 采集 Task Duration（任务书 §3.3 口径，建议收敛到 TC_PF 子集）
msprof op --application="./build/test/gemv/sgemv/sgemv_test --gtest_filter=*TC_PF*" --output=./prof_sgemv

# 4. 补充正态分布用例（默认仅均匀分布）
python3 gen_csv.py --dist mixed --out ./sgemv_test_mixed.csv
```

自测报告需记录：用例总数与分布（含正态分布补齐条数）、逐用例 PASS/FAIL、精度统计（matched_ratio、max_abs_error、非有限值统计）、性能逐 case 的 msprof Task Duration 与 avg_us、与 `gpu_baseline.csv` 的比值、异常样本说明。

## 兼容性分析

| 项 | 结论 |
| --- | --- |
| 公共接口 | `include/cann_ops_blas.h` 不改动，ABI 与 950PR 一致；不新增私有平行接口 |
| arch35（950PR） | 独立目录编译，`SOC_ARCH_DIRS` 分流；互不引入符号 |
| arch22 Cgemv | 新增 `sgemv_*` 符号，不改动 `cgemv_*`；测试与实现互不影响 |
| README | 仅更新产品支持表，不涉及行为兼容 |
| 多 stream | 依赖 handle 绑定 stream；不同 handle 可并发 |

## 已知限制与规避

| 限制 | 影响 | 规避 |
| --- | --- | --- |
| 长归约误差余量 | 极端 redDim 用例可能逼近 2^-13 阈值 | 多路累加 + 树形归约；全程 FLOAT32；实测超阈值再评估 Kahan |
| 非单位步长走逐元素路径 | 非单位步长性能不保证 | 正确性优先；性能用例全部为单位步长（`incx=incy=1`） |
| `alpha=0` 与 golden 短路口径不一致 | A/x 含 Inf/NaN 时结果不同 | 该组合不设精度用例（任务书已认可）；实现严格按不短路语义 |
| 用户自定义小 workspace | split-K 无法申请部分和缓冲 | 回退 `redSplit=1`，正确性不变；自测报告说明 |
| arch35 缺跨度溢出守卫 | 两代实现行为存在差异 | 本设计在 arch22 补齐；在 PR 说明中标注，供上游决定是否同步 arch35 |
| 极小形状 launch 开销 | 1×1 等用例可能接近 GPU 基线 | 单核、无 workspace、无同步、单次 launch；报告中单列说明 |

## 交付物对照（任务书 §4）

| 交付件 | 本设计对应 |
| --- | --- |
| 算子设计文档 | 本 PR（仅 1 个 `docs/design.md`） |
| 自测用例及测试代码 | 代码 PR：`test/gemv/sgemv/arch22/` + 任务提供的 `test_cases/` 包 |
| 自测报告 | 代码验收阶段：精度/性能数据、采样口径、复现步骤，按模板提交 |
| 待验收代码地址 | 个人代码仓分支 + `blas/gemv/arch22/`、`test/gemv/sgemv/arch22/`；按仓规范提供 README |
| `task_submission` | 验收阶段按模板在私仓根目录提交“自验证步骤说明/精度报告/性能报告/内存报告” |

## AI 辅助披露

本文档撰写过程中使用了 AI 编程助手辅助整理任务书、ops-blas 源码与仓内既有实现，提交者已逐节复核；文中涉及的接口行号、目录结构与校验口径均以 ops-blas 当前 master 源码核对为准。PR 描述中的 AI 参与项按实际情况如实填写。

## 修订记录

| 版本 | 日期 | 说明 |
| --- | --- | --- |
| v1.0 | 2026-11-11 | 初稿：需求背景、需求分析、详细设计、可维可测与 §2.5 约束核对 |
| v1.1 | 2026-11-11 | 修正表格单元格内竖线的 Markdown 解析问题，补充自验步骤与性能采样双口径说明 |

---

*参考：cuBLAS cublasSgemv（CUBLAS Library §2.6.2）、Netlib `sgemv.f`、ops-blas `blas/gemv/arch35/sgemv_host.cpp` 与 `blas/gbmv/arch22/sgbmv_*`、`blas/gemv/arch22/cgemv_*`，以及任务书 `aclblasSgemv_AtlasA2_A3_task_doc.md` 与 `test_cases/README.md`。*
