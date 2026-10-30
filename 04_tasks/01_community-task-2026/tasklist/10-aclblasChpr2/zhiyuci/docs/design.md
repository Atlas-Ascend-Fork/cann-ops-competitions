# 需求背景（required）

## 需求来源

10 月社区任务 aclblasChpr2 算子开发（950），任务讨论：https://gitcode.com/cann/ops-blas/discussions/54 。对应社区任务目录为 `10-aclblasChpr2`，贡献者为 `zhiyuci`。任务要求在 Ascend 950PR 上补齐 complex64 Hermitian packed 秩 2 更新能力，接口与 cuBLAS `cublasChpr2` 对齐。

## 背景介绍

BLAS 的 Hermitian 矩阵只需保存上三角或下三角。packed 格式按列连续保存该三角的 `n(n+1)/2` 个复数，可避免分配完整 `n×n` 矩阵。ops-blas 已有同族实数接口 `aclblasSspr2` 的 arch35 实现，但没有 complex64 的 `aclblasChpr2`。

本任务基于 ops-blas 公共 API 和 handle/stream 框架，新增 `aclblasChpr2`，使用 Ascend C kernel 直调实现原地更新。alpha 位于 Host，x/y/ap 位于 Device；复数实部、虚部均为 float32，类型定义复用 `aclblasComplex`。

## 算子实现现状分析

本算子不属于神经网络图算子，ops-blas 仓内没有对应的 TBE 实现，也不走 ACLNN 或 TBE 算子信息库路径。可复用的现有实现是 `blas/spr2/arch35/` 下的实数 `aclblasSspr2`；它提供了同族接口、参数校验顺序、句柄式启动和测试框架，但仅处理 float32 对称矩阵。

| 项目 | aclblasSspr2 | 本任务 aclblasChpr2 |
|---|---|---|
| 矩阵性质 | 实对称 | 复 Hermitian |
| 元素类型 | float32 | complex64 |
| 更新公式 | `A += alpha*x*y^T + alpha*y*x^T` | `A += alpha*x*y^H + conj(alpha)*y*x^H` |
| 存储 | packed 上/下三角 | packed 上/下三角，无前导维 |
| 对角处理 | 实数，无虚部 | 忽略旧虚部，非 no-op 时写 0 |
| 实现位置 | `blas/spr2/arch35/` | `blas/spr2/arch35/`，复用同族构建与导出 |

# 需求分析（required）

## 需求描述

新增公共接口 `aclblasChpr2`，计算：

```text
A ← A + alpha*x*y^H + conj(alpha)*y*x^H
```

其中 `alpha` 是 Host complex64 指针，`x`、`y` 是 Device complex64 向量，`ap` 是 Device packed 矩阵并原地更新。仅引用和更新 `uplo` 指定的三角，共 `n(n+1)/2` 个复数元素；对角元素旧虚部不参与计算，非 no-op 输出时对角虚部置 0。

## 需求拆解

1. 在 `include/cann_ops_blas.h` 增加与其他产品线共用的 `aclblasChpr2` 声明，不定义 950PR 私有平行接口。
2. 实现 UPPER/LOWER packed 地址映射、复数乘加、对角虚部清零和零列语义，支持任意合法正负 `incx`/`incy`。
3. 严格遵循任务书的参数校验顺序：handle、负 n、n=0、alpha 指针、uplo、步长、x/y/ap 指针、零 alpha。
4. 通过 handle 绑定的 stream 异步启动单个 AIV kernel，不分配额外 workspace。
5. 基于随任务 CSV、Netlib `cblas_chpr2` golden 和 GPU 基线完成精度、异常、性能及内存检查，并提供可复现证据。
6. 按 ops-blas 仓规范提供算子 README、测试 README 和公共头文件声明。

# 详细设计（required）

## 算子分析

### 数学公式

对每个已存储的 `A(i,j)`，令：

```text
t1 = alpha * conj(y[j])
t2 = conj(alpha * x[j])
A'(i,j) = A(i,j) + x[i]*t1 + y[i]*t2
```

非对角结果按 `(old + x[i]*t1) + y[i]*t2` 的 FP32 分组计算；对角只计算旧实部加更新项实部，旧虚部不读取，输出虚部固定为 0。复数乘法拆成四个独立 FP32 乘积后再加减，保留 `0×Inf`、中间溢出和 Netlib 累加顺序的语义。

当 `x[j]` 与 `y[j]` 的实部、虚部同时为零时，按 Netlib 语义跳过该列乘加，只清除对角虚部，避免对无关的 Inf 值执行 `Inf*0` 改变结果。

### 支持数据类型

| 参数 | 类型 | 说明 |
|---|---|---|
| handle | `aclblasHandle_t` | Host 句柄，携带 stream |
| uplo | `aclblasFillMode_t` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | `const int` | 非负矩阵阶数 |
| alpha | `const aclblasComplex*` | Host 单个 complex64 标量 |
| x、y | `const aclblasComplex*` | Device 向量，只读 |
| incx、incy | `const int` | 非零且不为 `INT_MIN`，单位为复数元素 |
| ap | `aclblasComplex*` | Device packed 原地输入输出 |

`aclblasComplex` 实部、虚部均为 FLOAT32；实现以编译期断言检查大小和字段偏移。kernel 使用 `-ffp-contract=off -fno-fast-math`，不复用仅对有限值成立的零乘捷径。

### 支持形状与存储

`x`/`y` 物理长度至少为 `1+(n-1)*abs(inc)`，`ap` 至少包含 `n(n+1)/2` 个复数。下标均为 0-based：

- UPPER：`0≤i≤j<n`，`p = i + j*(j+1)/2`。
- LOWER：`0≤j≤i<n`，`p = i + ((2*n-j-1)*j)/2`，等价于列首 `j*(2*n-j+1)/2` 加 `i-j`。
- 负步长物理起点为 `(n-1)*(-inc)`；逻辑第 k 个元素对应 `start + k*inc`。

所有索引乘积先提升到 64 位，避免 `n*inc` 等中间值溢出。调用方负责保证缓冲区有效；实现不人为设置 `n≤4096` 的 API 上限。

## 算子实现

### 接口

在 `include/cann_ops_blas.h` 增加：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy,
    aclblasComplex* ap);
```

实现位于 `blas/spr2/arch35/`，包含 host、kernel、tiling 数据三个文件；接入同族 CMake 构建、导出和测试入口，接口声明放入公共头文件供其他产品线共用。

### 实现方案

```text
调用 aclblasChpr2
        │
        ▼
Host 按固定顺序校验参数
        │
        ├─ n=0 / alpha=0 / 参数错误 ──► 直接返回，不写 ap
        │
        ▼
计算 64 位起点、packed 索引和分核参数
        │
        ▼
通过 handle stream 启动一个 AIV kernel
        │
        ▼
block 轮转处理列；128 线程处理同列连续 packed 行
        │
        ▼
逐列计算 t1/t2，逐行更新实部/虚部；对角虚部写 0
        │
        ▼
stream 异步完成，调用方同步后读回
```

#### Host 侧设计

Host 校验顺序与任务书一致：

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n<0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `n=0` 返回 `ACLBLAS_STATUS_SUCCESS`，不访问后续参数。
4. alpha 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. uplo 非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. `incx`/`incy` 为 0 或 `INT_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. x、y、ap 任一为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
8. 读取 Host alpha；实部、虚部均为 0 时返回成功，不访问 x/y/ap 内容，也不修改 ap。

通过校验后，Host 计算 `xStart`/`yStart`、64 位索引和分核数。`blockDim` 取 `min(n, AIV 核数)`，tiling 数据只包含 n、uplo、alpha 实虚部、步长和起点，不分配 workspace，不增加 Host 同步。

#### Kernel 侧设计

- 每个 block 处理列 `b, b+gridDim.x, ...`；同一 block 的 128 个 SIMT 线程处理当前列的连续行，使相邻线程访问连续 packed 元素。
- UPPER/LOWER 通过模板参数在编译期特化，避免每个元素重复分支。
- 每个线程计算 `px = x[i]*t1`、`py = y[i]*t2`；非对角先加 px 再加 py，对角只计算 `old_real + (px.real + py.real)`，随后统一写 `ap[2*p+1]`。
- 列首 `x[j]`、`y[j]` 同为零时，仅由 0 号线程将 `ap[2*diag+1]` 置 0，其余线程跳过该列。
- 每个输出元素只由一个线程更新，不需要原子操作或跨核归约。

#### 性能优化方案

1. 列内并行、列间分核：每个 block 负责完整列，线程覆盖连续 packed 行，兼顾负载均衡和访存连续性。
2. 单 kernel、无 workspace、无 Host 同步：减少启动和中间拷贝开销。
3. 预计算列首和 t1/t2：每列只读取一次 `x[j]`、`y[j]`，行循环复用。
4. 编译期特化 UPPER/LOWER，64 位索引避免溢出检查落在内层错误路径。
5. 保持 `-ffp-contract=off` 和独立 FP32 乘积，优先满足 Netlib 特殊值语义；这是有意的精度换性能取舍。
6. 小 n 时自动减少 block 数，避免固定满核启动造成额外调度开销。

### 支持硬件

| 支持的芯片版本 | 支持情况 |
|---|---|
| Ascend 950PR | 支持，arch35，CANN 9.1.0 完成开发和自测 |

### 算子约束限制

- alpha 仅支持 Host 指针，不提供 Device alpha 模式。
- x/y 只读，ap 原地更新；不存在 lda、广播或非连续 Tensor 语义。
- 调用方需保证 x/y/ap 长度和地址有效，并避免 ap 与输入发生有害重叠。
- 读取 Device 结果前需同步 handle 绑定的 stream。
- GPU 基线沿用任务提供的 `gpu_baseline.csv`，本次不重跑 H100。
- 任务书 §3.3 规定 msprof op 采集 10 次有效样本；开发自测另保留 51 次独立采集用于稳定性复核，两类数据不裁剪、不混算，正式报告会明确预热和采样口径。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度 | 实部、虚部分别比较；`rtol=atol=2^-13`，有限值匹配率≥99%，最大绝对误差≤`max(1e-2, 32×ULP)`；NaN/同号 Inf/NaN 与 Inf 按任务规则等效，有限与非有限不等效 | 任务书 §3.2 |
| 性能 | 每个 case 的 kernel task duration 算术平均值≤GPU 基线耗时×2.5 | 任务书 §3.3 |
| 内存容量 | 任务书未设置容量门槛；附加 mssanitizer 冒烟单列披露，不替代容量验收 | 任务书 §3.4 |

测试覆盖冻结原始集 1200 条和独立均匀/正态补充集 1152 条，逐例核对状态、实虚指标、对角虚部、只读输入、输出 canary 及 no-op 不写。性能采集保存逐调用 CSV、GTest XML、样本映射、二进制与工具散列，并由独立核验器复核收据。

## 验证摘要

R6 开发自测结果：NPU 正确性 2352/2352 通过；性能 200 条用例在 10 次、51 次两组独立采集中均为 200/200 通过，最差预算占用分别约 73.57%、73.35%；原装 msprof task-time 交叉采集 200/200 通过；附加 mssanitizer 冒烟覆盖 3 项 GTest、364 次 kernel 调用，未检测到内存错误，控制台另有 1 条目录权限安全 WARNING，已在材料中单独披露。以上为开发自测，不代表平台验收结论。

R6 源码与证据：

- 源码提交：https://gitcode.com/zhiyuci/ops-blas/commit/5deaf3938c9b1b784c6f1cf0a27d22c1f1ed4b04
- 分支：`chpr2-950-r6`
- 基线：`05b96ea22cea926ecfd43fc7ac691a795522bd61`

## 兼容性分析

公共头文件只新增 `aclblasChpr2` 声明，不改变已有接口。arch35 实现和严格 FP 编译选项限定在本算子，不影响其他产品线及 `aclblasSspr2`。profile 工具的兼容性处理保留在测试工具和证据目录，不写入 SDK 安装，不混入算子补丁。
