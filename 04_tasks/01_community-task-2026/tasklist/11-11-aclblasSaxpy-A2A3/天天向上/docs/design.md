# aclblasSaxpy（Atlas A2/A3）算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasSaxpy` |
| 文档版本 | V1.0 |
| 提交者 | harrynospot |
| 单位 | 个人开发者 |
| 目标硬件 | Ascend Atlas A2/A3（arch22） |
| 软件版本 | CANN 9.1.0 |
| 数据类型 | FLOAT32（单精度实数） |
| 开发方式 | Ascend C Kernel 直调，句柄绑定 Stream |
| 代码规划目录 | `blas/axpy/arch22/` |
| 文档状态 | 设计完成，代码与真机数据待实现和验证 |

# 1 需求背景（required）

## 1.1 需求来源

本任务来源于 CANN 社区任务，要求基于 `cann/ops-blas` 工程，在 Ascend Atlas A2/A3（arch22）上实现单精度实数向量线性组合更新接口 `aclblasSaxpy`。接口语义对齐 cuBLAS `cublasSaxpy`（https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf §2.5.4）和 Netlib BLAS `saxpy.f`（https://netlib.org/blas/saxpy.f）。代码与测试合入 https://gitcode.com/cann/ops-blas。

## 1.2 背景介绍

AXPY 是 BLAS Level 1 中的基础向量算子，用于执行"一个向量乘以标量后累加到另一个向量"。`aclblasSaxpy` 是其单精度实数版本，计算结果原地写回向量 `y`：

```text
y = alpha * x + y
```

当前 `ops-blas` 已有 `arch35`（950PR）的 `saxpy` 实现以及 `arch22` 的其他向量算子（如 `ccopy` 等）。已有 `arch35` 实现可作为工程组织、Stream 绑定、分核和非连续访存的参考，但本任务仍需补齐面向 A2/A3 的完整参数校验、正负步长、INT32_MIN 与跨度溢出检查以及测试能力。

## 1.3 现状与问题

`arch35` 既有 `saxpy` 实现主要面向 950PR，其工程模式（host 入口 + tiling + device kernel）可作为参考，但需适配 A2/A3 的 arch22 工程骨架。设计中需要解决以下问题：

1. `incx`、`incy` 可以为正数或负数，需要正确计算逻辑元素对应的物理地址。
2. 连续数据应使用批量搬运和向量计算，避免逐元素访问造成性能损失。
3. 非单位步长应提供正确的通用路径，且不能越界访问。
4. `n <= 0` 为合法 no-op（返回成功，不访问 `alpha`、`x`、`y`），对齐 Netlib `saxpy.f` 的 `IF (N.LE.0) RETURN` 语义。
5. `alpha = 0.0` 时按 Netlib `saxpy.f` 的 `IF (DA.EQ.ZERO) RETURN` 直接返回，`y` 不变且不读取 `x`。
6. `incx`/`incy` 为 `INT32_MIN` 或 `(n-1)*|inc|` 超出 int32 范围时返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算）。
7. Host 接口应只负责校验、生成 tiling 和异步启动 Kernel，不在正常调用路径中强制同步 Stream。

# 2 需求分析（required）

## 2.1 需求描述

接口执行以下运算：

```text
for i = 0 ... n - 1:
    k = physicalIndex(i, n, incx)
    j = physicalIndex(i, n, incy)
    y[j] = alpha * x[k] + y[j]
```

对于负步长，传入的 Device 缓冲区仍按照物理长度 `1 + (n - 1) * abs(inc)` 分配，逻辑首元素位于相应物理缓冲区的尾部。实现需要把逻辑索引转换为非负的物理索引。

## 2.2 接口定义

接口使用 `ops-blas` 公共头文件中的统一声明（`include/cann_ops_blas.h` L461），不增加 A2/A3 私有平行接口：

```cpp
aclblasStatus_t aclblasSaxpy(
    aclblasHandle_t handle,
    int n,
    const float* alpha,
    float* x,
    int incx,
    float* y,
    int incy);
```

> 注意：仓内 `include/cann_ops_blas.h` 中 `x` 为非 `const` 指针（cuBLAS 为 `const float*`），实现不得写入 `x`。

## 2.3 参数说明

| 参数 | 位置 | 含义 | 约束与异常行为 |
| --- | --- | --- | --- |
| `handle` | Host | ops-blas 句柄，携带执行 Stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | Host | 逻辑元素数量 | `n <= 0` 为合法 no-op（返回成功，不访问 `alpha`、`x`、`y`） |
| `alpha` | Host | FLOAT32 标量指针 | `n > 0` 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；值不校验（含 0.0、Inf、NaN） |
| `x` | Device | 只读 FLOAT32 输入向量 | `n > 0` 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；NaN/Inf 按公式自然传播 |
| `incx` | Host | `x` 的逻辑步长 | `n > 0` 且为 0 或 `INT32_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`(n-1)*|incx|` 超 int32 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | Device | 输入并原地输出的 FLOAT32 向量 | `n > 0` 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；NaN/Inf 按公式自然传播 |
| `incy` | Host | `y` 的逻辑步长 | `n > 0` 且为 0 或 `INT32_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`(n-1)*|incy|` 超 int32 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 2.4 需求拆解

1. 实现 FLOAT32 标量乘加，结果原地写回 `y`。
2. 支持运行时 `n`，覆盖 `n <= 0`（no-op）、小尺寸和大尺寸。
3. 支持任意非零 `incx`、`incy`（非 `INT32_MIN`），包括正负步长组合。
4. 连续访存与非连续访存采用不同执行路径。
5. 使用 handle 中绑定的 Stream 异步启动 Kernel。
6. 精度结果对齐 cblas/Netlib golden。
7. 在 Ascend A2/A3、CANN 9.1.0 环境完成精度和性能验证。

## 2.5 边界语义

参数校验顺序设计如下：

1. 检查 `handle`；为空时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. 检查 `n`；`n <= 0` 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不访问 `alpha`、`x`、`y`，也不启动 Kernel（对齐 Netlib `saxpy.f` 的 `IF (N.LE.0) RETURN`）。
3. 当 `n > 0` 时检查 `alpha`、`x`、`y`，空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 检查 `incx`、`incy`；任一为 0 或 `INT32_MIN` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. 检查跨度溢出：`(n-1) * |incx|` 或 `(n-1) * |incy|` 超出 int32 可表示范围时返回 `ACLBLAS_STATUS_INVALID_VALUE`（host 侧 int64 计算，issue #780/#866 口径）。
6. 读取 `alpha` 值；若 `alpha == 0.0`，直接返回 `ACLBLAS_STATUS_SUCCESS`（对齐 Netlib `saxpy.f` 的 `IF (DA.EQ.ZERO) RETURN`，`y` 不变且不读取 `x`）。

索引与物理跨度计算使用 64 位整数，避免 `n` 与步长乘法产生 32 位溢出；对 `INT_MIN` 步长取绝对值时先提升为 `int64_t`。

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

设 `alpha` 为 FLOAT32 标量，`x`、`y` 为 FLOAT32 向量，则：

```text
y[j] = alpha * x[k] + y[j]
```

其中：

```text
k = 1 + (i - 1) * incx    （i = 1..n，以 1 基索引描述）
j = 1 + (i - 1) * incy
```

以 0 基索引描述（C/C++ 习惯）：

```text
k = i * incx              （i = 0..n-1，incx > 0）
k = (n - 1 - i) * |incx|  （i = 0..n-1，incx < 0）
j 同理
```

每个逻辑元素之间不存在数据依赖，适合按元素分核并行处理。计算仅需 1 次乘法、1 次加法，无复数实虚部交叉运算。

### 3.1.2 数据类型与存储

`alpha` 为 FLOAT32 标量，位于 Host 内存。`x`、`y` 为 FLOAT32 向量，位于 Device 内存，采用连续布局：

```text
x = [x0, x1, x2, ...]
y = [y0, y1, y2, ...]
```

每个元素占 4 Byte。逻辑长度为 `n` 时，向量物理元素数量为：

```text
xSpan = 1 + (n - 1) * abs(incx)
ySpan = 1 + (n - 1) * abs(incy)
```

### 3.1.3 负步长地址映射

设逻辑索引为 `i`，`absInc = abs(inc)`，物理索引计算如下：

```text
inc > 0: physicalIndex(i) = i * absInc
inc < 0: physicalIndex(i) = (n - 1 - i) * absInc
```

分核后，第 `b` 个核处理逻辑区间 `[start, start + count)`。通用路径根据全局逻辑索引计算 `x`、`y` 的物理位置，确保所有地址均落在已分配跨度内。

## 3.2 总体方案

```text
用户调用 aclblasSaxpy
        |
        v
Host 参数校验与 alpha 读取
        |
        +-- n <= 0 -----------------------> 直接返回 SUCCESS（no-op）
        |
        +-- alpha == 0.0 -----------------> 直接返回 SUCCESS（y 不变）
        |
        v
获取 AIV Core 数并生成 TilingData
        |
        +-- incx == 1 && incy == 1 --> 连续 AIV 向量路径
        |
        +-- 其他合法步长 ----------> 通用分步长路径
        |
        v
使用 handle->stream 异步启动 Kernel
        |
        v
Kernel 分核、分 Tile 执行标量乘加并原地写回 y
```

## 3.3 Host 侧设计

### 3.3.1 参数校验

Host 侧实现独立的参数校验函数 `ValidateSaxpyParams`，按照 2.5 节顺序返回明确状态码。`alpha` 位于 Host 内存，在启动 Kernel 前解引用读取 `*alpha`，以值的形式写入 tiling 数据，避免异步执行依赖调用者后续可能失效的 Host 指针。

校验逻辑对齐仓内 950PR 既有实现（`blas/axpy/arch35/saxpy_host.cpp` 的 `ValidateSaxpyParams` 与 `aclblasSaxpy` 入口，L20-160），返回码口径与 cuBLAS 官方文档一致。

### 3.3.2 Tiling 数据

计划使用如下等价信息；最终字段名以代码规范为准：

```cpp
struct SaxpyTilingData {
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t tileSize;
    float alpha;
    int64_t incx;
    int64_t incy;
    uint32_t useCoreNum;
};
```

其中：

- `totalN`：逻辑元素数量。
- `perCoreN`、`remainder`：连续路径的均匀分核参数。
- `tileSize`：单次进入 UB 的 FLOAT32 元素数量。
- `alpha`：Host 标量值（已解引用）。
- `incx`、`incy`：通用路径的地址计算参数。
- `useCoreNum`：实际使用的 AIV Core 数量。

如通用路径采用每核显式区间表，可增加 `startOffset[]` 与 `calCount[]`，但不在 Host 侧申请每次调用都需要手工释放的 Device 辅助缓冲区。

### 3.3.3 分核策略

1. 通过工程已有平台接口获取可用 AIV Core 数量。
2. 实际核数不超过 `n`，避免启动无工作量的 Core。
3. 连续路径以 32 Byte 对齐的元素个数（8 个 FLOAT32）为基本分配单位，最后一个或前若干个 Core 处理余数。
4. 通用路径按逻辑元素数量均分，不按物理跨度均分，保证每个元素只处理一次。
5. `n <= 0` 时不计算 tiling、不启动 Kernel。

### 3.3.4 Stream 与错误处理

Kernel 使用 `handle->stream` 启动。接口正常返回仅表示任务成功下发，不主动调用 `aclrtSynchronizeStream`；调用者在读取 `y` 前负责同步 Stream。Kernel 启动或平台信息获取失败时转换为 `aclblasStatus_t` 中对应的执行错误状态。

## 3.4 Kernel 侧设计

### 3.4.1 连续访存快速路径

适用条件：

```text
incx == 1 && incy == 1
```

处理流程：

1. 每个 Core 根据 `perCoreN` 和 `remainder` 得到自己的逻辑区间。
2. 将 `x`、`y` 的连续 FLOAT32 块从 GM 搬入 UB。
3. 执行 `y = alpha * x + y` 的向量乘加运算（`Axpy` 指令或等价 `Mul` + `Add` 组合）。
4. 将结果搬回原 `y` 地址。
5. 尾块使用带长度信息的搬运接口处理，禁止覆盖有效范围之外的数据。

为减少流水停顿，可为 `x` 和 `y` 使用双缓冲，将 GM 搬入、Vector 计算和 GM 搬出流水化。Tile 大小根据 A2/A3 UB 容量、输入/输出局部张量和中间结果占用动态确定，并向 32 Byte 对齐。

### 3.4.2 通用步长路径

适用条件：除连续路径外的所有合法步长组合。

通用路径优先复用 `arch35` 既有 `saxpy` 的分步长搬运思路：

1. 使用 `abs(incx)`、`abs(incy)` 计算物理跨度。
2. 对正负步长分别计算 Tile 的起始物理位置。
3. 对非单位步长使用 Compact 搬运或等价 Gather 方式，把离散 FLOAT32 元素收集为 UB 中的连续数据。
4. 当输入和输出步长符号不同，保证两者仍按同一逻辑索引配对，必要时对元素进行顺序重排。
5. 在 UB 中执行与连续路径相同的标量乘加。
6. 按 `incy` 将结果写回离散位置。

如果 Compact 搬运在目标 CANN 版本对极端步长存在限制，则使用 SIMT 逐逻辑元素路径作为保正确回退方案；连续大规模用例仍使用 AIV 快速路径保证性能。

### 3.4.3 尾块与对齐

- 完整块使用对齐的 `DataCopy`。
- 不足一个数据块的尾部使用带字节长度的 Pad/扩展搬运接口。
- 对齐偏移测试中只访问请求范围，不对 GM 地址向前取整。
- 所有局部缓冲区按 32 Byte 对齐。

## 3.5 文件结构设计

计划在 `ops-blas` 中新增或修改：

```text
include/cann_ops_blas.h                    # 使用公共 aclblasSaxpy 声明（已有，核对签名）
blas/axpy/arch22/saxpy_host.cpp            # 参数校验、tiling、Kernel 启动
blas/axpy/arch22/saxpy_kernel.cpp          # Ascend C Kernel
blas/axpy/arch22/saxpy_kernel.h            # Kernel 启动声明
blas/axpy/arch22/saxpy_tiling_data.h       # tiling 数据结构
test/saxpy/CMakeLists.txt                  # arch22 分支注册 GTest
test/saxpy/saxpy_param.h                   # CSV 参数定义（复用仓内已有）
test/saxpy/saxpy_golden.h                  # cblas/CPU golden（复用仓内已有）
test/saxpy/arch22/saxpy_npu_wrapper.h      # NPU 调用封装
test/saxpy/arch22/saxpy_test.cpp           # GTest 测试
test/saxpy/arch22/saxpy_test.csv           # 用例集合
```

公共头文件当前已有 `aclblasSaxpy` 声明时只核对签名，不重复定义。

## 3.6 兼容性与资源分析

- 接口签名与其他产品线共用，不引入 A2/A3 私有 API。
- 数据类型固定为 FLOAT32，不涉及隐式类型转换。
- `y` 原地更新，不返回新视图。
- 算子没有跨元素归约，结果具有确定的逐元素计算顺序。
- 正常执行不需要额外 Workspace；tiling 以 Kernel 启动参数或工程统一机制传递。
- `x`、`y` 非完全重叠时按标准输入/输出语义处理；未在任务书定义的复杂别名场景不作为额外保证。

# 4 可维可测分析

## 4.1 精度标准

FLOAT32 输出向量逐元素判定，遵循生态算子开源精度标准（mixed_tolerance 口径）：

| 指标 | 要求 |
| --- | --- |
| `rtol` | `2^-13`，约 `1.22e-4` |
| `atol` | `2^-13`，约 `1.22e-4` |
| 匹配比例 | `matched_ratio >= 0.99` |
| 最大绝对误差 | `<= 1e-2` 或 `<= 32 * ULP` |

单个元素满足：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

golden 使用 cblas/Netlib `saxpy` 以 float32 直接计算（与仓内 `saxpy_golden.h` 一致）。

**Inf/NaN 特殊值判定规则**：当 golden 或 NPU 输出中出现 Inf/NaN 时，不适用上述判定式：golden 与 actual 同为 NaN、同为同号 Inf，或一方为 NaN 而另一方为 +/-Inf，视为等效，判定通过；一方为有限值、另一方为 Inf/NaN，判定不通过；非有限值元素不计入 matched_ratio 与 max_abs_error 统计，单独记录。

## 4.2 功能与异常测试

测试采用 CSV 驱动的 C++ GTest，至少覆盖：

| 类别 | 重点内容 |
| --- | --- |
| 基础用例 | `n=1/8`，步长 `(1,1)/(2,1)/(1,2)/(2,3)` |
| 尺寸扫描 | 0、1、小质数、2 的幂及其相邻值、大尺寸（至 1048576） |
| 步长组合 | `incx`、`incy` 的 `+1/+2/+3/-1/-2/-3` 全组合及混合 |
| alpha 特殊值 | 0、+/-1、0.5、+/-2.5、1e10、1e-30、Inf、NaN |
| 数据填充 | 随机、全零、交替值、极值、Inf、NaN |
| 边界负向 | `n<=0` quick return、空指针、零步长、`INT32_MIN` 步长、跨度溢出 |
| no-op | `n <= 0` 时成功且不修改 `y` |

任务配套 CSV 共 1200 条，其中 1000 条用于精度、功能和异常验证，200 条用于性能测试。固定随机种子保证结果可复现。

随机输入按任务书要求规划为 50% 均匀分布（`RANDOM_NORM_5_5`，[-5,5]）和 50% 正态分布（`GAUSS_NORM_5_2`，mu in [-5,5]、sigma in [0.1,2]）。当前配套生成脚本的默认能力是均匀分布；代码阶段需要扩展测试框架的正态分布填充能力后，再补齐混合分布用例，不能把尚未生成的正态分布用例计为已覆盖。

## 4.3 性能标准

性能测试在 Ascend 800T A2（910B3）、CANN 9.1.0 环境进行。先 warmup 5 次预热，再有效采样 >=10 次取平均，使用 msprof op 采集单算子 task duration（单位 us）。判据：平均单次耗时不高于 GPU 标杆耗时的 1.25 倍（即达到标杆 0.8 倍以上性能）。

| `n` | `incx` | `incy` | GPU A100 标杆 (ms) | NPU 上限 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1048576 | 1 | 1 | 0.01207 | 0.01508 |
| 2097152 | 1 | 1 | 0.01640 | 0.02050 |
| 4194304 | 1 | 1 | 0.03482 | 0.04352 |

性能优化顺序为：先保证结果正确，再优化连续路径的分核、Tile 大小和流水并行；通用步长路径以正确性和无越界为首要目标。

## 4.4 测试执行与证据

测试报告需要记录：

1. 硬件型号、CANN 版本、代码提交号和测试命令。
2. 每类用例的数量、通过数和失败数。
3. 误差统计（max_abs_error、matched_ratio）。
4. 性能用例的预热次数、采样次数和平均耗时。
5. 编译日志、测试日志及真实设备截图。

本文档只描述设计与验证计划，不将尚未执行的测试标记为通过。代码实现完成后再补充真实自测结果。

# 5 风险与应对

| 风险 | 影响 | 应对措施 |
| --- | --- | --- |
| 负步长首地址理解错误 | 结果倒序或越界 | 使用统一物理索引公式，并覆盖四种正负符号组合 |
| 尾块非对齐写回 | 越界或污染邻近数据 | 使用精确字节长度搬运并增加保护区检查 |
| Host 侧强制同步 | 破坏接口异步语义和性能 | 正常路径只下发 Kernel，读取结果前由调用者同步 |
| GTest 耗时代替 Kernel 时间 | 性能结论失真 | 使用 msprof op 采集真实 kernel task duration |
| UB 分配过大 | 编译或运行失败 | 根据平台 UB 容量计算 Tile，并保留安全余量 |
| INT32_MIN 步长 abs 溢出 | 地址计算错误或段错误 | 先提升为 int64_t 再取绝对值 |
| 跨度溢出未检查 | 越界访问 | host 侧 int64 计算 (n-1)*|inc| 并校验不超 INT32_MAX |
| 正态分布用例未补齐 | 精度覆盖不完整 | 扩展 gen_csv.py --dist mixed 生成 GAUSS_NORM_5_2 用例 |

# 6 结论

本设计采用"连续 AIV 向量快速路径 + 通用分步长路径"的双路径方案。Host 侧完成参数校验（含 `n<=0` no-op、`alpha==0` 快速返回、`INT32_MIN` 与跨度溢出检查）、标量读取、分核和 tiling；Kernel 侧按逻辑元素并行执行标量乘加，并通过统一地址映射支持正负步长。该方案与 `ops-blas` 现有 `arch35` 向量算子的工程模式保持一致，可同时满足接口兼容性、功能正确性和连续大规模场景的性能要求。

# 7 参考资料

1. aclblasSaxpy Atlas A2/A3 算子开发任务书。
2. `cann/ops-blas`：<https://gitcode.com/cann/ops-blas>。
3. Netlib BLAS `saxpy`：<https://www.netlib.org/blas/saxpy.f>。
4. NVIDIA cuBLAS AXPY（§2.5.4）：<https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf>。
5. CANN 生态算子开源精度标准（mixed_tolerance）：<https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>。
6. Ascend C 算子开发文档：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html>。
