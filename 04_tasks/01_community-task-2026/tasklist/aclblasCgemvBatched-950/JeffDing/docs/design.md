# 需求背景（required）

## 需求来源

本需求来自 Ascend 950PR 社区任务"aclblasCgemvBatched 算子开发"。目标是在 Ascend 950PR
上使用 Ascend C 补齐单精度复数批量矩阵-向量乘算子，并按 `ops-blas` 的句柄式 BLAS 接口、
工程目录和测试框架完成设计、开发与自验。

实现代码位于 `blas/gemv_batched/arch35/`，测试代码位于
`test/gemv_batched/cgemv_batched/arch35/`，复用公共头文件中已经存在的
`aclblasCgemvBatched` 声明，不增加 950PR 专用平行接口。

## 背景介绍

### CGEMV Batched 算子功能

`aclblasCgemvBatched` 对一组形状相同、地址彼此独立的复数矩阵与向量执行批量 GEMV：

```text
y[b] = alpha * op(A[b]) * x[b] + beta * y[b],   0 <= b < batchCount
```

其中：

- `A[b]` 为 `m x n` 复数矩阵，按 BLAS 约定采用 Column-Major 存储。
- `x[b]`、`y[b]` 为复数向量，支持正负步长。
- `alpha`、`beta` 为 Host 侧复数标量，所有批次共享同一对标量。
- `op(A)` 支持不转置、转置和共轭转置，由 `trans` 参数统一控制所有批次。
- 各批次共用 `m`、`n`、`lda`、`incx`、`incy` 与转置标志，仅设备指针不同，属于
  **uniform batch** 语义（与 `aclblasCgemmBatched` 的 batch 模型一致）。

`op(A)` 的具体语义如下：

| trans | op(A) | x 逻辑长度 | y 逻辑长度 |
| --- | --- | --- | --- |
| `ACLBLAS_OP_N` | `A` | `n` | `m` |
| `ACLBLAS_OP_T` | `A^T` | `m` | `n` |
| `ACLBLAS_OP_C` | `A^H` | `m` | `n` |

复数场景下 `ACLBLAS_OP_C` 必须执行共轭转置，即读取矩阵元素时对虚部取反；不能退化为
普通转置。

### 现状分析

ops-blas 仓已有公共头文件声明 `aclblasCgemvBatched`，但 Ascend 950PR 对应 arch35 实现
和测试用例仍需补齐。单批 `aclblasCgemv` 已在 `blas/gemv/arch35/` 提供实现，本任务在
此基础上增加 batch 维度：

- 输入从单指针变为长度为 `batchCount` 的设备指针数组 `Aarray`、`xarray`、`yarray`；
- 并行空间从一维输出元素扩展为 **二维（批次 × 输出元素）** 任务网格，需要重新设计
  分核与 tiling；
- 需要保证 batch 维度的引入不破坏单批语义，且 `batchCount=0` 时行为与
  `aclblasCgemmBatched` 的 quick return 一致。

# 需求分析（required）

## 需求描述

### 接口原型

```cpp
aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[],
    int lda,
    const aclblasComplex* const xarray[],
    int incx,
    const aclblasComplex* beta,
    aclblasComplex* const yarray[],
    int incy,
    int batchCount);
```

### 参数与异常语义

| 参数 | 位置 | 含义与约束 | 非法行为 |
| --- | --- | --- | --- |
| `handle` | Host | 携带当前执行 stream 的库句柄 | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | Host | 只允许 `ACLBLAS_OP_N/T/C` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m` | Host | A 的行数，须 `>= 0` | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | Host | A 的列数，须 `>= 0` | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | Host | COMPLEX64 标量指针 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray` | Device | 长度为 `batchCount` 的设备指针数组 | `batchCount>0` 且数组为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | Host | Column-Major 前导维，`lda >= max(1, m)` | 不满足返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `xarray` | Device | 长度为 `batchCount` 的设备指针数组 | `batchCount>0` 且数组为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | Host | x 元素步长，`incx != 0` | 零步长返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | Host | COMPLEX64 标量指针 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `yarray` | Device | 长度为 `batchCount` 的设备指针数组，y 原地更新 | `batchCount>0` 且数组为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incy` | Host | y 元素步长，`incy != 0` | 零步长返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount` | Host | 批次数，须 `>= 0` | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |

向量实际存储元素数至少为 `1 + (len - 1) * abs(inc)`。各批次逻辑区域不得相互重叠，
否则行为未定义。

### 快速返回与特殊值

| 条件 | 语义 |
| --- | --- |
| `m==0 || n==0 || batchCount==0` | 合法 no-op，返回成功，不启动计算 Kernel |
| `alpha==(0,0) && beta==(1,0)` | C 不变，直接返回成功 |
| `alpha==(0,0) && beta!=(1,0)` | 不读取 A 和 x，仅执行 `y = beta * y` |
| `beta==(0,0)` | 计算路径不读取旧 y 的输入值 |
| `k==0`（即归约维为空） | 同上，执行 `y = beta * y` |

参数检查先于快速返回；因此除 `batchCount==0` 外，设备指针数组本身仍须满足非空约束。
`alpha`/`beta` 的 Host 值在 quick return 判断前读入，以便识别 `alpha==(0,0)` 分支。

## 需求拆解

1. Host 侧实现参数检查、快速返回、alpha/beta 读入、设备核数查询、二维 tiling 计算与
   Kernel 下发；
2. Kernel 侧实现 `trans=N` 行方向归约（每输出元素为一行点积）；
3. Kernel 侧实现 `trans=T` 列方向归约（每输出元素为一列点积）；
4. Kernel 侧实现 `trans=C` 列方向归约并对 A 元素取共轭；
5. 实现 `beta` 缩放与 `alpha=0` 特例路径，避免无效读 A/x 与非有限值污染；
6. 支持正负 `incx`、`incy`，负步长按 Netlib 语义从反向起点遍历；
7. 支持 uniform batch 语义，各批共用维度、ld、步长与转置标志；
8. 补充 README 参数说明、产品支持表和 CSV 测试用例。

# 详细设计（required）

## 算子分析

### 数学公式

当 `trans == ACLBLAS_OP_N` 时，对批次 `b` 与输出行 `i in [0, m)`：

```text
y[b][iy] = alpha * sum(j=0..n-1, A[b][i + j * lda] * x[b][jx]) + beta * y[b][iy]
```

其中 `jx = xBaseOffset + j * incx`，`iy = yBaseOffset + i * incy`。

当 `trans == ACLBLAS_OP_T` 时，对批次 `b` 与输出列 `j in [0, n)`：

```text
y[b][jy] = alpha * sum(i=0..m-1, A[b][i + j * lda] * x[b][ix]) + beta * y[b][jy]
```

其中 `ix = xBaseOffset + i * incx`，`jy = yBaseOffset + j * incy`。

当 `trans == ACLBLAS_OP_C` 时：

```text
y[b][jy] = alpha * sum(i=0..m-1, conj(A[b][i + j * lda]) * x[b][ix]) + beta * y[b][jy]
```

复数乘法采用：

```text
(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
```

复数加法采用分量相加。共轭操作为：

```text
conj(a + bi) = a - bi
```

### 支持数据类型

| 输入/输出 | 数据类型 |
| --- | --- |
| `Aarray` 元素 | `aclblasComplex` / complex64 |
| `xarray` 元素 | `aclblasComplex` / complex64 |
| `yarray` 元素 | `aclblasComplex` / complex64 |
| `alpha` | `aclblasComplex` / complex64 |
| `beta` | `aclblasComplex` / complex64 |

`aclblasComplex` 的内存布局以 ops-blas 仓 `include/cann_ops_blas_common.h` 定义为准，实部
和虚部均为 float32，物理存储按 `real, imag, real, imag, ...` 交错排列。

### 支持形状

| 场景 | A 有效形状 | A 存储形状 | x 逻辑长度 | y 逻辑长度 |
| --- | --- | --- | --- | --- |
| `trans=N` | `[m, n]` | `[lda, n]` | `n` | `m` |
| `trans=T/C` | `[m, n]` | `[lda, n]` | `m` | `n` |

负步长起点：

```text
xBaseOffset = (incx < 0) ? (lenx - 1) * (-incx) : 0
yBaseOffset = (incy < 0) ? (leny - 1) * (-incy) : 0
```

其中 `lenx`、`leny` 由 `trans` 决定：

```text
if trans == ACLBLAS_OP_N:
    lenx = n
    leny = m
else:
    lenx = m
    leny = n
```

## 算子实现

### 实现方案

#### Host 侧设计

##### 参数检查流程

Host 侧按以下顺序检查，确保错误码稳定：

1. `handle == nullptr`，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `trans` 不属于 `ACLBLAS_OP_N/T/C`，返回 `ACLBLAS_STATUS_INVALID_ENUM`。
3. `m < 0`、`n < 0`、`batchCount < 0`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `alpha == nullptr` 或 `beta == nullptr`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. `lda < max(1, m)`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. `incx == 0` 或 `incy == 0`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. `batchCount > 0` 时，`Aarray`、`xarray`、`yarray` 数组指针本身须非空，否则返回
   `ACLBLAS_STATUS_INVALID_VALUE`。
8. 读入 Host 侧 `alpha`、`beta`，进入 quick return 与退化分支判断。

quick return 和退化分支：

| 条件 | 行为 |
| --- | --- |
| `m == 0 || n == 0 || batchCount == 0` | 返回成功，不 launch kernel |
| `alpha == (0,0) && beta == (1,0)` | 返回成功，不读写 y |
| `alpha == (0,0) && beta != (1,0)` | 仅执行 `y = beta * y`（不读 A、x） |
| `beta == (0,0)` | 计算输出时不读取 y 输入值 |

指针检查策略：

| 分支 | Aarray | xarray | yarray |
| --- | --- | --- | --- |
| quick return | 不检查 | 不检查 | 不检查 |
| `alpha == 0 && beta != 1` | 不检查 | 不检查 | 需非空 |
| 常规 GEMV | 需非空 | 需非空 | 需非空 |

##### 二维 Tiling 参数

与单批 GEMV 一维输出切分不同，Batched GEMV 的并行空间是二维的。Host 侧定义：

```text
leny  = (trans == N) ? m : n            // 单批输出元素数
total = batchCount * leny                // 二维任务网格总任务数
```

Host 侧生成以下 tiling 信息并传入 Kernel：

| 字段 | 说明 |
| --- | --- |
| `m`、`n`、`lda` | 矩阵形状和前导维 |
| `incx`、`incy` | 向量步长 |
| `lenx`、`leny` | 单批 x/y 逻辑长度 |
| `xBaseOffset`、`yBaseOffset` | 负步长时的逻辑起点偏移 |
| `transMode` | `N/T/C` 分支标记 |
| `computeMode` | 常规 GEMV 或仅缩放 y |
| `batchCount` | 批次数 |
| `totalTasks` | `batchCount * leny` |
| `coreNum` | 实际参与计算的 AI Core 数 |
| `tileRed` | 归约维单次搬运元素数（UB 约束导出） |
| `tasksPerCore[]` | 每核任务数（前 `remain` 核多 1 个） |

##### 分核策略

采用 **二维任务网格（批次 × 输出元素）线性化** 的并行策略。每个任务对应一个
`(batch, outIdx)` 对，计算单个输出元素。

总任务数：

```text
totalTasks = batchCount * leny
```

Host 侧根据 `totalTasks` 和平台 AI Core 数确定实际 core 数：

```text
coreNum = min(platformCoreNum, totalTasks)
```

将 `totalTasks` 均分到 `coreNum`：

```text
base = totalTasks / coreNum
remain = totalTasks % coreNum
```

前 `remain` 个 core 每核多处理 1 个任务，保证负载均衡。每核任务区间为
`[startTask, endTask)`。

线性任务 ID 到二维坐标的映射（Host 与 Kernel 一致）：

```text
batch = taskId / leny
outIdx = taskId % leny
```

该映射使同一 core 内连续任务属于同一批次、输出索引递增，从而在 `trans=N` 时能跨多行
复用同一 `x` 分块，在 `trans=T/C` 时能跨多列复用同一 `x` 分块，减少 GM 重复读取。

**边界 case 处理**：

- `totalTasks <= platformCoreNum` 时 `coreNum = totalTasks`，每核恰好 1 个任务，无空闲
  core 浪费。
- `totalTasks` 极大时 `coreNum` 取满平台核数，靠每核多任务吸收负载。

##### 归约维分块（Reduction Tiling）

GEMV 每个输出元素需要遍历整个归约维度（`trans=N` 时为 `n`，`trans=T/C` 时为 `m`）。
受 UB 容量限制，单次只能搬运一部分归约元素到片上。Host 侧根据 UB 大小计算
`tileRed`：

```text
// UB 需容纳：A 分块 + x 分块 + 累加器 + 临时寄存器开销
bufA  = tileRed * sizeof(aclblasComplex)
bufX  = tileRed * sizeof(aclblasComplex)
bufAcc = 若干累加器
tileRed = floor((UB可用字节 - 开销) / (2 * sizeof(aclblasComplex)))
```

`tileRed` 需按硬件要求的边界对齐（16 元素对齐），最终由 `GetCoreMemSize` 查询的 UB
大小与 `BUFFER_NUM`、是否开启 double buffer 综合决定。Kernel 侧按 `tileRed` 循环搬运。

##### TilingKey 规划

| TilingKey | 场景 |
| --- | --- |
| `0` | `trans=N` 常规 GEMV |
| `1` | `trans=T` 常规 GEMV |
| `2` | `trans=C` 常规 GEMV |
| `3` | `alpha=(0,0)` 且 `beta!=(1,0)`，仅缩放 y |

Host 侧根据 `trans` 和 `alpha/beta` 特殊值选择 TilingKey，Kernel 侧使用不同路径减少
分支判断。

#### Kernel 侧设计

Kernel 采用 `Init` + `Process` 结构。`Process` 中遍历本核分配的 `[startTask, endTask)`
任务区间，每个任务按 `CopyIn` → `Compute` → `CopyOut` 执行。

##### 数据访问

`A` 为 Column-Major：

```text
A[b](i, j) = Aarray[b][i + j * lda]
```

向量访问：

```text
x[b](k) = xarray[b][xBaseOffset + k * incx]
y[b](k) = yarray[b][yBaseOffset + k * incy]
```

当 `incx` 或 `incy` 为负数时，base offset 保证逻辑顺序与 Netlib 一致。

##### `trans=N` 计算路径

每个任务负责一个 `(batch, outIdx)`，`outIdx` 即行号 `i`：

```text
sum = 0 + 0i
for r = 0; r < n; r += tileRed:
    rEnd = min(r + tileRed, n)
    // CopyIn: 搬运 A[b][i + r..rEnd-1 * lda] 与 x[b][xBaseOffset + r..rEnd-1 * incx]
    // Compute: 逐元素 sum += A_chunk * x_chunk
out = alpha * sum
if beta != 0:
    out += beta * y[b][yBaseOffset + i * incy]
y[b][yBaseOffset + i * incy] = out
```

**同核多任务复用**：当一个 core 被分配多个任务且这些任务同属一批次（`batch` 相同）
时，`x` 分块在多个输出行之间共享。实现上按 `(batch, outIdx)` 分组，外层遍历 `x` 分块，
内层遍历该批次的所有待处理行，使一次 `x` 搬运服务多行点积，显著降低 `x` 的 GM 读取量。

##### `trans=T` 计算路径

每个任务负责一个 `(batch, outIdx)`，`outIdx` 即列号 `j`：

```text
sum = 0 + 0i
for r = 0; r < m; r += tileRed:
    rEnd = min(r + tileRed, m)
    // CopyIn: 搬运 A[b][r..rEnd-1 + j * lda]（连续）与 x[b][xBaseOffset + r..rEnd-1 * incx]
    // Compute: 逐元素 sum += A_chunk * x_chunk
out = alpha * sum
if beta != 0:
    out += beta * y[b][yBaseOffset + j * incy]
y[b][yBaseOffset + j * incy] = out
```

该路径对 A 的一列连续读取，访存局部性优于 `trans=N`。同批次多列任务可复用 `x` 分块。

##### `trans=C` 计算路径

共轭转置与 `trans=T` 的输出维度一致，仅矩阵元素读取后执行共轭：

```text
sum = 0 + 0i
for r = 0; r < m; r += tileRed:
    a = A[b][r + j * lda]
    a.imag = -a.imag          // 共轭
    sum += a * x[b][xBaseOffset + r * incx]
```

其余 `alpha`、`beta` 缩放与写回逻辑与 `trans=T` 一致。

##### 仅缩放 y 路径

当 `alpha == (0,0)` 且 `beta != (1,0)` 时不读取 A 和 x，遍历本核所有任务对 y 执行：

```text
y[b][offset] = beta * y[b][offset]
```

当 `beta == (0,0)` 时直接写零。特别注意：`batchCount > 0` 且 `n == 0` 时同样走此路径，
此时 `0 * alpha` 不得被计算，避免非有限 `alpha` 把有限结果污染为 NaN。

##### UB 与流水

1. 按 `(batch, outIdx)` 二维网格分配任务到 core，核心间负载均衡。
2. 每个 core 内，按批次分组，外层循环归约维分块（`tileRed`）。
3. 每个分块内 `CopyIn` 将 A 分块与 x 分块搬入 UB，`Compute` 在片上完成复数乘加累积。
4. 一个输出元素归约完成后执行 `alpha` 缩放、`beta` 融合并 `CopyOut` 写回 y。
5. 开启 double buffer 时，前后两个分块的搬运与计算可流水重叠。

对 `trans=T/C`，A 的列方向连续，可采用较大的 `tileRed`；对 `trans=N`，A 按列跨步读取
同一行，首版以正确性与同核复用为主，后续可进一步采用多行并行 tile 提高 A 列块复用。

### 代码组织

| 文件 | 作用 |
| --- | --- |
| `blas/gemv_batched/arch35/gemv_batched_host.cpp` | 参数检查、快速返回、核数查询、tiling 计算、Kernel 下发 |
| `blas/gemv_batched/arch35/gemv_batched_kernel.cpp` | N/T/C 三条计算路径、缩放路径、复数基础运算 |
| `blas/gemv_batched/arch35/gemv_batched_tiling_data.h` | Host/Kernel 共享 tiling 结构体 |
| `test/gemv_batched/cgemv_batched/arch35/cgemv_batched_test.cpp` | CSV 正确性、边界回归和性能契约 |
| `test/gemv_batched/cgemv_batched/arch35/cgemv_batched_test.csv` | 任务输入用例 |
| `test/gemv_batched/cgemv_batched/README.md` | 950PR 构建、运行和结果解释 |

### Host 总体流程

```text
Validate parameters
  -> empty shape or batchCount==0? return SUCCESS
  -> read alpha/beta from Host
  -> alpha==0? beta scaling / return (k==0 同理)
  -> query AI Core count
  -> compute lenx, leny, base offsets, totalTasks
  -> decide coreNum and per-core task range
  -> decide tileRed from UB size
  -> launch kernel on handle stream
```

所有计算 Kernel 都下发到 handle 当前 stream。调用方在 Host 读回或在其他 stream 消费
结果前须同步当前 stream 或建立事件依赖。workspace 由 handle 的现有机制管理；本算子
各路径均不申请额外临时矩阵，因此无额外 workspace 开销。

### Workspace 设计

本算子所有路径均不需要额外 workspace：

| 路径 | workspace 字节数 |
| --- | --- |
| 常规 GEMV | 0 |
| beta 缩放 | 0 |
| quick return | 0 |

Kernel 内部仅使用 UB 临时缓冲区，其大小由 `tileRed` 与核心数决定，不向 Host 暴露。
指针数组 `Aarray`/`xarray`/`yarray` 由调用方提供，本接口不负责其生命周期管理。

### TilingData

`GemvBatchedTilingData` 结构体字段：

| 字段 | 用途 |
| --- | --- |
| `m`、`n`、`lda` | 矩阵形状与前导维 |
| `incx`、`incy` | 向量步长 |
| `lenx`、`leny` | 单批 x/y 逻辑长度 |
| `xBaseOffset`、`yBaseOffset` | 负步长起点偏移 |
| `transMode` | `N/T/C` 编码 |
| `computeMode` | 常规 / 仅缩放 |
| `batchCount` | 批次数 |
| `totalTasks` | 二维任务网格总数 |
| `coreNum` | 实际使用 AI Core 数 |
| `tileRed` | 归约维分块大小 |
| `startTask`、`endTask` | 本核任务区间 |

Host 侧对 `totalTasks` 做 64 位计算并检查是否溢出；所有设备核数均由运行环境查询，
不写死 950PR 核数。

### 错误传播

参数错误使用任务书指定状态码；核数查询、Kernel 调度失败时返回已有库状态或
`ACLBLAS_STATUS_INTERNAL_ERROR`。接口不抛出 C++ 异常跨越 C ABI。Kernel 调度沿用仓内
stream 直调方式。

## 支持硬件

| 硬件 | 支持情况 |
| --- | --- |
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持同一 arch35 实现 |
| Atlas A2/A3 | 本任务实现不支持 |

arch35 的 cgemvBatched 依赖 CANN asc-devkit 9.1 及以上；低版本构建不启用该实现。

## 算子约束限制

| 项目 | 约束 |
| --- | --- |
| 数据类型 | 输入、输出和标量均为 COMPLEX64，计算分量为 FP32 |
| 布局 | Column-Major，复数实虚交错 |
| Batch | uniform batch；各批共用维度、ld、步长和转置标志 |
| 指针 | A/x/y 为设备侧指针数组；alpha/beta 位于 Host |
| 非连续存储 | 只支持 lda/incx/incy 表达的元素间步长与列间 padding，不支持额外 stride |
| 输出别名 | 不允许同一批次内 x 与 y 逻辑区域重叠；不允许不同批次 y 逻辑区域重叠 |
| Broadcast | 不支持，也不属于接口语义 |
| 动态 Shape | m/n/batchCount 为运行时参数，不使用图模式动态 Shape |
| 异步语义 | 计算下发到 handle 当前 stream，结果使用前由调用方同步 |
| 确定性 | 无额外 bit-exact 要求；`alpha=0` 的缩放特例按任务要求验证 EXACT |
| quick return | `m=0`、`n=0`、`batchCount=0` 均为合法 no-op |

# 可维可测分析

## 精度标准/性能标准

| 指标 | 标准 | 标准来源 |
| --- | --- | --- |
| 功能 | 公开接口、N/T/C、标量、batch、padding、边界和状态码符合任务书 | 社区任务书 |
| 精度 | 实部/虚部分别按 FP32：`rtol=2^-10`、`atol=2^-16`、matched ratio >= 99%、最大误差满足 `1e-2` 或逐元素 `32*ULP` | 社区任务书与生态算子开源精度标准 |
| 性能 | warmup 后有效采样大于 50，平均单次耗时不高于任务书标杆 | 任务书附件 |
| 内存 | 无额外 workspace，分析可归因设备内存 | 上述 Workspace 设计 |

逐元素判定公式：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

实部和虚部均满足阈值时，该 complex64 元素判定通过。

性能基准 case（标杆值取自任务书附件，以实际下发为准）：

| case | trans | m | n | batchCount | 标杆耗时 Avg time(us) |
| --- | --- | --- | --- | --- | --- |
| 1 | `ACLBLAS_OP_N` | 512 | 512 | 32 | 待任务书确认 |
| 2 | `ACLBLAS_OP_N` | 2048 | 2048 | 16 | 待任务书确认 |
| 3 | `ACLBLAS_OP_T` | 2048 | 2048 | 16 | 待任务书确认 |

## 自测用例设计

测试用例使用 CSV 驱动，覆盖以下类别：

| 类别 | 覆盖内容 |
| --- | --- |
| 基础功能 | `trans=N/T/C`，小尺寸和常规方阵，单批与多批 |
| batch 扫描 | `batchCount` = 0, 1, 2, 质数、2 的幂、大值 |
| 尺寸扫描 | `m/n` = 1、小质数、2 的幂、2 的幂±1、非对齐值、大尺寸 |
| 非方阵 | `m < n` 宽矩阵、`m > n` 窄矩阵 |
| 标量特殊值 | `alpha/beta` 为 `0`、`1`、`-1`、纯虚数、一般复数 |
| 前导维 | `lda=max(1,m)` 和 `lda>m` padding |
| 步长 | `incx/incy` 为 `±1`、`±2`、`±3` |
| quick return | `m=0`、`n=0`、`batchCount=0`、`alpha=0 && beta=1` |
| 退化计算 | `alpha=0 && beta!=1`、`beta=0`、`n=0` 的非有限 alpha |
| 负向用例 | 空指针、非法枚举、负维度、非法 lda、零步长、负 batchCount |
| 特殊值 | A/x/y 中包含 Inf、NaN 的传播场景 |
| 性能用例 | 任务书指定性能 case 和扩展规模、跨 batchCount 扩展 |

负向用例期望错误码：

| 场景 | 期望 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| `m < 0`、`n < 0` 或 `batchCount < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha == nullptr` 或 `beta == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda < max(1,m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` 或 `incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount > 0` 时数组指针为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 常规计算时 `Aarray/xarray/yarray` 元素为空 | `ACLBLAS_STATUS_INVALID_VALUE` |

测试逐批调用 Netlib `cblas_cgemv` 生成 CPU golden，再执行 NPU 计算。一次 batched API
的所有逻辑输出聚合后统计匹配比例，实部和虚部独立判定；`lda`/`incy` padding 不计入
99% 比例，并由独立 sentinel 语义保护。

## 可观测性分析

1. **维度可观测**：Host 侧 tiling 结构体包含 `totalTasks`、`coreNum`、`tileRed`、每核
   任务区间，可在日志中打印，便于定位负载不均衡与 UB 溢出。
2. **分支可观测**：`transMode` 与 `computeMode` 正交编码，四条 Kernel 路径可通过
   TilingKey 独立计数，单元测试可断言各分支均被执行。
3. **精度可观测**：实部/虚部分别判定，padding 区域由 sentinel 保护，失败用例可按
   GTest 名称与 CSV 行号定位到具体 `(batch, outIdx)`。
4. **回归可观测**：单批 `aclblasCgemv` 的 CSV 用例可复用（`batchCount=1`），保证新增
   batch 维度不破坏既有语义。

## 可维护性分析

1. 复数基础操作（加、乘、共轭、标量缩放）封装为 `AscendC` 内联函数，避免三种转置路
   径重复实现实部/虚部计算。
2. Host 侧将维度、步长、base offset、二维分核结果统一写入 tiling，Kernel 侧只消费
   tiling 数据，不重复计算坐标映射。
3. `trans=N/T/C` 使用独立 TilingKey，便于后续针对不同访存模式做性能优化（如
   `trans=N` 的多行并行 tile）。
4. 新增 `batch` 维度仅扩展 tiling 与任务映射，单批 GEMV 的 Compute 逻辑可复用，降低
   两份实现的维护成本。
5. README、测试 CSV、GTest 入口与实现代码同步提交，保证验收人可复现。

## 兼容性分析

### API 与 ABI

实现复用 `include/cann_ops_blas.h` 已有函数声明和 `aclblasComplex` 类型，不修改公共结构
体布局、枚举值、参数顺序或导出符号，因此不改变其他产品线 API/ABI。

### 源码与构建

变更限定于 arch35 batched GEMV、对应测试、README 和设计文档。根构建配置只在用户未指
定 `CMAKE_BUILD_TYPE` 时默认 Debug；显式传入 Release 时保留 Release，以便执行真实的
性能测试。未改变单批 `aclblasCgemv` 的接口和实现。

### 运行时

实现动态查询 AI Core 核数，使用 handle 当前 stream 和现有 workspace 管理机制，不依赖
950PR 固定核数，也不新增运行时三方依赖。950DT 复用同一 arch35 路径；本任务的功能与
性能验证面向 950PR。

### 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| T/C 混淆 | 共轭组合错误 | Kernel 显式虚部取反，N/T/C 正交 CSV 覆盖 |
| 二维分核负载不均 | 尾核空闲、性能下降 | `totalTasks` 均分 + 前 `remain` 核多 1 任务 |
| UB 溢出 | 大 tileRed 导致编译失败 | 由 `GetCoreMemSize` 动态计算，留足 double buffer 开销 |
| 非有限 alpha 污染 | `n=0` 时产生 NaN | `alpha==0` 分支提前返回，不计算 `0*alpha` |
| 负步长偏移 | 逻辑索引与物理地址不一致 | Host 统一计算 base offset 并传入 Kernel |
| 小尺寸调度抖动 | 独立环境的测量值可能波动 | 固定预热与采样契约，并在设备空闲时测量 |
| 上游主干继续变化 | 合并冲突 | 代码 PR 前同步最新 master，重新执行编译与正确性准入 |

# 参考资料

1. `aclblasCgemvBatched` 社区任务书及随附测试工具；
2. 社区任务设计模板 `04_tasks/01_community-task-2026/resources/design_template.md`；
3. `ops-blas/include/cann_ops_blas.h` 与 `cann_ops_blas_common.h`；
4. 单批参考实现 `blas/gemv/arch35/` 设计文档；
5. `aclblasCgemmBatched` batched 算子设计文档（uniform batch 与 quick return 语义）；
6. Netlib BLAS CGEMV：<https://www.netlib.org/blas/cgemv.f>；
7. cuBLAS batched GEMV：<https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemvbatched>；
8. 生态算子开源精度标准。