# aclblasChpmv 算子设计文档（Atlas A2/A3）

## 需求背景

### 需求来源

本设计对应 CANN 社区任务 `aclblasChpmv` Atlas A2/A3 版本。目标是在 `ops-blas` 的 `arch22` 路径补齐单精度复数 Hermitian 压缩矩阵向量乘，接口语义对齐 `cublasChpmv`，复用仓内现有的 `aclblasChpmv` 声明和句柄式 BLAS 调用模式。本文依据本任务的开发任务书编写；具体实现与性能数字须在目标设备上验证。

### 背景介绍

矩阵仅存储上三角或下三角，共 `n(n+1)/2` 个 `aclblasComplex` 元素。相较展开成稠密矩阵，压缩表示减少矩阵存储量，但 Device 端必须正确计算镜像元素的共轭值。目标硬件为 Atlas A2/A3，目标代码目录为 `blas/spmv/arch22/`，测试目录为 `test/spmv/chpmv/arch22/`。

## 需求分析

### 需求描述

实现 `y ← alpha * A * x + beta * y`，其中 `A = Aᴴ`，输入和输出均为 complex64。支持 `UPPER`/`LOWER`、任意非零正负 `incx`/`incy`、Host 与 Device 位置的复数标量指针、原地输出，以及任务书定义的空规模和零标量行为。调用方保证 `y` 的写入区域不与 `AP`、`x` 的读取区域重叠。

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

### 需求拆解

| 项目 | 设计要求 |
| --- | --- |
| 参数与返回码 | 按任务书顺序完成结构参数校验；非法 `uplo` 返回 `INVALID_ENUM`，其余非法值返回对应 `INVALID_VALUE`，空句柄返回 `HANDLE_IS_NULLPTR`。 |
| 计算 | 仅读取已存三角；镜像项取共轭；对角虚部无效；`beta=0` 不读旧 `y`。 |
| 内存访问 | packed 索引和步长偏移使用足够宽的整数中间值；负步长从物理缓冲区末端对应的逻辑首元素开始遍历。 |
| 执行 | 在 handle 绑定的 stream 上异步下发；不在正常路径同步 stream。 |
| 验证 | 采用任务书规定的 CBLAS golden、精度阈值及 NPU/GPU kernel 时间口径。 |

## 详细设计

### 算子分析

#### 数学公式与布局

对每个逻辑行 `i`，计算 `yᵢ = alpha * Σⱼ Aᵢⱼ xⱼ + beta * yᵢ`。`AP` 按列主序存放：

- `UPPER`：当 `i ≤ j` 时，`AP[i + j(j+1)/2] = Aᵢⱼ`。
- `LOWER`：当 `i ≥ j` 时，`AP[i + j(2n-j-1)/2] = Aᵢⱼ`。
- 未存半区按 `Aᵢⱼ = conj(Aⱼᵢ)` 取得。对角项只使用 `real(Aᵢᵢ)`，不让其虚部参与运算。

向量调用方传入缓冲区的物理起点。逻辑元素 `v[k]` 的物理元素偏移为 `base + k*inc`，其中 `base = 0`（正步长）或 `(n-1)*abs(inc)`（负步长）。偏移和 packed 元素数先在宽整数中计算，再转换为设备地址，避免 `int` 乘法溢出。`inc = INT_MIN` 仍按非零步长处理，绝不在有符号 `int` 中直接取绝对值。

#### 特殊语义

1. 校验顺序：`handle` → `n<0` → `uplo` → `incx/incy` → `alpha/beta` 指针 → `n=0` 返回 → `AP/x/y` 指针 → 根据标量值选择计算路径。
2. `n=0` 返回成功，不读取 `AP/x/y`、不下发 kernel，但结构参数仍须有效。
3. `alpha=0, beta=1` 无需访问三个张量；`alpha=0, beta≠1` 仅缩放 `y`，不访问 `AP/x`。
4. `beta=0` 的所有路径均不读取旧 `y`，以支持旧 `y` 为 NaN 的用例。
5. Host 标量可在 Host 侧判断数值并选择路径。Device 标量不得在 Host 侧解引用，须把判定留在 Device 执行路径；不能为识别零值而同步 stream。

### 算子实现

#### Host 侧设计

API 入口复用 ops-blas 的 handle/stream 和 `aclblasComplex` 定义，不新增平行接口。Host 侧按上述顺序检查参数、选择路径和规划 tiling。`AP/x/y` 指针是 Device 侧数据指针；标量位置按仓内既有指针模式规则识别，不假设全部标量位于 Host。

拟采用三种下发路径：

| 路径 | 条件 | 目的 |
| --- | --- | --- |
| 直接行计算 | 小 `n` 或可在单次下发中完成的规模 | 每个输出行只由一个核写入，省去第二次下发与归约。 |
| 列分块与归约 | 大规模、可获得所需 workspace | 第一阶段按列分配 packed 数据并生成各核的部分结果；第二阶段归约并融合 `alpha/beta`。 |
| 仅缩放 | `alpha=0` 且 `beta≠1` | 只处理 `y`，避免矩阵和 `x` 访存。 |

路径切换阈值、核数、列块宽度和 UB tile 长度由目标 SoC 资源及实测确定，不在未测量前固定为某一数值。Host 侧依据 workspace 可用容量限制第一阶段并行核数；空间不足时回退到无需全局部分结果的直接行计算，以保持功能正确。所有设备地址计算使用宽整数，并检查分配大小的乘法溢出。

#### Kernel 侧设计

**直接行计算**：每个核负责互不重叠的输出行。沿逻辑列遍历对应的 packed 索引；未存储侧使用转置位置并取共轭；对角项屏蔽虚部。按 `incx` 收集 `x`，在局部累加复数乘积，最后一次性写回对应 `y`。核间无写冲突，也不需要原子操作。

**列分块与归约**：第一阶段按 packed 列切分工作，每个核读取自身连续列段。对每个非对角存储值 `Aᵢⱼ`，分别累加 `Aᵢⱼ*xⱼ` 到行 `i`、`conj(Aᵢⱼ)*xᵢ` 到行 `j`；对角仅累加 `real(Aⱼⱼ)*xⱼ`。各核写独立的部分结果缓冲，避免并发写 `y`。第二阶段按输出行切分，汇总全部部分结果，再执行 `alpha`、`beta` 融合并按 `incy` 写回。部分结果在写出前必须初始化，不能让未写区域的旧 workspace 数据进入归约。

大 `n` 时按输出行 tile 管理 UB 暂存，不假设整个 `x` 或部分输出都可常驻 UB。列分块优先保持 `AP` 顺序读取；不连续 `x/y` 使用显式步长寻址。针对 `LOWER` 列起点和尾块的非对齐访问，按 Ascend C API 的合法对齐要求设计搬运和尾部屏蔽。双缓冲仅在事件同步、UB 占用及实测收益确认后启用。

两条计算路径都由 Kernel 处理 Device 标量值。Host 标量模式下可使用 Host 已知的 `alpha/beta` 选择更短路径。Kernel 在 `beta=0` 分支不加载 `y`；在 `alpha=0` 分支不加载 `AP/x`。

#### 并行与资源规划

列分块路径的 workspace 规模约为 `核数 × n × sizeof(aclblasComplex)`，另计对齐和必要的归约暂存。Host 根据 handle 提供的工作空间与 SoC 核数限制并行度。直接行路径不使用此部分结果空间。所有路径以运行时 `n` 和 `uplo` 计算 tiling；尾列、尾行与负步长均单独覆盖。

### 支持硬件

| 硬件 | 目标 |
| --- | --- |
| Atlas A2 系列（含 910B3） | `arch22` 实现及自验证目标 |
| Atlas A3 系列 | `arch22` 适配目标 |

### 算子约束限制

- 仅支持 `aclblasComplex`（实部和虚部各为 float32）及列主序 Hermitian packed 布局；无 `lda` 参数。
- `incx`、`incy` 必须非零；向量物理缓冲区须覆盖 `1+(n-1)*abs(inc)` 个元素。
- `y` 的写区域不得与 `AP/x` 的读区域重叠；调用方保证缓冲区长度、可访问性及 AP 内容符合接口约定。
- 不支持广播或任意视图步长；API 返回成功仅表示异步任务已成功下发。

## 可维可测分析

### 精度标准与测试方案

精度 golden 使用 CPU `cblas_chpmv(CblasColMajor, ...)`。实部、虚部分别按 FLOAT32 判定：`rtol=atol=2^-13`，每用例匹配率至少 `0.99`，最大绝对误差满足任务书规定的 `1e-2` 或 `32 ULP` 限制。测试覆盖上下三角、`n=0/1`、奇数与非对齐规模、正负步长、非法参数、对角虚部、`alpha=0`、`beta=0`、NaN/Inf 输入和 Host/Device 标量模式。CSV 提供 1000 条精度用例与 200 条性能用例；需要在 ops-blas 测试工程中实现对应的填充 token、golden 与逐分量比对。

### 性能标准与测量方案

正式性能口径为 Atlas 800T A2（910B3）上 `msprof op` 的 NPU kernel `Task Duration(us)`，预热 5 次后取 10 次有效样本的算术平均。逐条与同参数 GPU kernel-only `gpu_ms` 比较，目标为 `NPU_ms ≤ GPU_ms/0.8`。`aclrtEvent` 结果可用于回归趋势，但不替代正式 kernel-to-kernel 判据。实施阶段分别统计直接行路径、列分块阶段和归约阶段耗时，根据小规模下发开销、大规模带宽与 workspace 流量调整阈值。本文不声称任何尚未取得的精度或性能实测结果。

### 兼容性与风险

复用 `include/cann_ops_blas.h` 已有声明和 handle 的 stream 语义，不改变公共 API。主要风险是 packed 镜像索引、负步长物理起点、Device 标量分支、归约顺序带来的浮点误差，以及大规模两阶段 workspace 和额外下发开销。以上风险以边界用例、CBLAS 对照、内存检查和逐路径 profiling 验证；若实测不达标，再依据瓶颈数据调整 tiling 与内核结构。
