# 需求背景（required）

## 需求来源

- 任务：11月社区任务 `aclblasCrot` 算子开发（Atlas A2/A3）。
- 任务书：`aclblasCrot_AtlasA2_A3_task_doc.md`。
- 目标仓库：`cann/ops-blas`，算子实现目录 `blas/rot/arch22/`，测试目录 `test/rot/crot/arch22/`。
- 参考接口：cuBLAS `cublasCrot`；复数旋转数学语义以 Netlib/LAPACK `crot` 为准。

## 背景介绍

### aclblasCrot 算子实现

`aclblasCrot` 对两个单精度复数向量执行 Givens 平面旋转，向量在设备侧原地更新。任务目标是在 ops-blas 现有 Ascend C BLAS 工程框架中增加句柄式接口，在 Atlas A2/A3（arch22）上通过 handle 绑定的 stream 直调 kernel。

### ops-blas 实现现状分析

按任务书给出的仓库现状：

| 项目 | 现状与设计影响 |
| --- | --- |
| 公共接口 | `include/cann_ops_blas.h` 尚无 `aclblasCrot` 声明；本任务新增公共声明，不定义产品私有平行接口 |
| arch22 算子 | `blas/rot/arch22/` 有 `aclblasCsrot`，可参考其工程集成方式 |
| 实数 rot 参考 | `aclblasSrot` 位于 arch35，可参考其 Host、kernel 和 tiling 组织方式，但参数契约以本任务书的 arch22 要求为准 |
| 测试工程 | 需要新增 `test/rot/crot/arch22/` CSV 驱动 GTest；并在 `test/rot/crot/CMakeLists.txt` 注册 arch22 测试 |
| 标量内存 | `c`、`s` 可来自 Host 或 Device，需沿用仓内运行时指针属性识别与 stream 处理方式 |

### aclblasCrot 算子功能分析

对 `i = 0 ... n-1`，按正负步长映射到 `x`、`y` 的逻辑元素。令 `x0`、`y0` 表示更新前的值，`c` 为实数，`s = sr + j*si` 为复数：

```text
x' = c*x0 + s*y0
y' = c*y0 - conj(s)*x0
```

复数乘法按 `(a+jb)(u+jv) = (au-bv) + j(av+bu)` 计算。`y'` 必须使用更新前的 `x0`。`x`、`y` 均原地更新；算子不返回新 Tensor。

输入 `x`、`y` 为 COMPLEX64 向量；`c` 为 FLOAT32 标量，`s` 为 COMPLEX64 标量。`n`、`incx`、`incy` 为 Host 侧 int 参数。步长支持正、负值；负步长从物理向量尾端起算。动态 shape、广播和额外 leading dimension 不适用。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 Atlas A2/A3（arch22）实现 `aclblasCrot`，在保持 BLAS 句柄和 stream 语义的同时，按 Netlib/LAPACK `crot` 复数公式更新两个 complex64 向量。接口参数顺序与 cuBLAS `cublasCrot` 对齐，并满足任务书指定的错误码、精度、性能和测试要求。

接口原型：

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy,
    const float* c,
    const aclblasComplex* s);
```

## 需求拆解

1. 在 `include/cann_ops_blas.h` 新增公共接口声明，保持 ABI 类型和参数顺序与任务书一致。
2. 在 arch22 Host 入口实现 handle、`n`、步长、跨度和指针校验，并使用 handle 绑定的 stream 发起 kernel。
3. 实现支持正负非零步长的 complex64 原地旋转，确保 `y` 更新使用旧 `x` 且对 `s` 取共轭。
4. 支持 `c`、`s` 分别位于 Host 或 Device 的输入组合。
5. 提供 CSV 驱动的 arch22 GTest，覆盖功能、边界、异常、精度及性能场景。
6. 满足任务书规定的 FLOAT32 精度标准和相对 GPU 基线的性能门槛，并提交可复现的自测材料。

## 需求口径

本设计以本任务书对 A2/A3（arch22）的约束为准：handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n < 0`、步长为 0、步长为 `INT32_MIN` 或跨度溢出返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n = 0` 在 handle 校验后快速返回 SUCCESS。随附测试 README/生成器中将负 `n` 和零步长描述为合法 no-op/路径，与任务书不一致；实现和测试合入前，需按任务书更新 CSV 期望值及测试 runner，或由任务维护者确认正式验收口径。

# 详细设计（required）

## 算子分析

### 数学公式

设复数 `x = xr + j*xi`、`y = yr + j*yi`、`s = sr + j*si`，`c` 为实数。计算为：

```text
xr' = c*xr + sr*yr - si*yi
xi' = c*xi + sr*yi + si*yr
yr' = c*yr - sr*xr - si*xi
yi' = c*yi - sr*xi + si*xr
```

上述 `yr'`、`yi'` 使用旋转前的 `xr`、`xi`。实现可将实部/虚部拆分计算，或用仓内 `aclblasComplex` 对应的复数表示进行等价计算；不得将 `conj(s)` 误用于 `x'`，也不得省略 `y'` 中的共轭。

### 支持数据类型

| 参数 | 类型 | 位置与访问方式 |
| --- | --- | --- |
| `x`、`y` | COMPLEX64（实部、虚部均为 FLOAT32） | Device，原地读写 |
| `c` | FLOAT32 | Host 或 Device，标量只读 |
| `s` | COMPLEX64 | Host 或 Device，标量只读 |
| `n`、`incx`、`incy` | int32 接口参数 | Host 侧校验并传入 kernel 参数 |

### 支持形状与步长

逻辑形状为一维 `[n]`。物理元素跨度分别为 `1 + (n-1)*abs(incx)` 和 `1 + (n-1)*abs(incy)`（`n > 0`）。对第 `i` 个逻辑元素（0-based），定义：

```text
offset_x = (incx < 0 ? (n - 1) * abs(incx) : 0) + i * incx
offset_y = (incy < 0 ? (n - 1) * abs(incy) : 0) + i * incy
```

Host 侧以 int64 计算绝对步长和最大跨度；拒绝 `INT32_MIN`，并拒绝 `(n-1)*abs(inc)` 大于 INT32_MAX 的情况，确保后续地址偏移计算不会发生有符号 int32 溢出。零步长不允许。

### 参数校验与返回值

| 顺序 | 校验/行为 | 返回值 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `n == 0`，不访问 `x`、`y`、`c`、`s` | `ACLBLAS_STATUS_SUCCESS` |
| 4 | `incx` 或 `incy` 为 0、INT32_MIN，或跨度计算溢出 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `n > 0` 时 `x`、`y`、`c` 或 `s` 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | 参数有效，按 handle stream 发起 kernel | `ACLBLAS_STATUS_SUCCESS` 或仓内运行时对应错误码 |

## 算子实现

### 实现方案

#### Host 侧设计

1. 遵守上表的校验先后顺序。尤其 handle 检查必须先于 `n == 0` 快速返回。
2. `n == 0` 不读取任何向量或标量指针，也不发起 kernel。
3. 对 `incx`、`incy` 先拒绝 0 与 INT32_MIN，再用 int64 计算跨度上界；所有检查在访问设备数据之前完成。
4. 用 `aclrtPointerGetAttributes` 判定 `c`、`s` 的内存位置，支持 host/host、host/device、device/host、device/device 组合。Host 标量值作为 launch 参数传递；Device 标量指针传入 kernel 并由 kernel 读取。不得在 Host 上直接解引用 Device 指针。
5. 从 handle 取得绑定 stream，kernel 与标量相关操作均遵守该 stream 顺序，接口保持异步提交语义；Host 输入标量应在调用期间读入 launch 参数，不能依赖调用返回后的 Host 缓冲区生命周期。
6. 本算子不需要临时向量 workspace。若仓内 launch 封装对标量参数或 tiling 有既定结构，复用该结构，避免额外的设备分配和同步。

#### Kernel 侧设计

1. 使用 Ascend C 直调 kernel；一个逻辑元素对应一个处理单元，或者由向量化处理连续的若干逻辑元素。
2. 将 `[0, n)` 的逻辑索引划分到可用 AI Core。每个 core 处理连续的逻辑索引区间；尾块通过有效元素数屏蔽，不越界读写。
3. 依据正负步长计算 `x`、`y` 物理偏移。单位步长是性能主路径，保持相邻逻辑元素的访存连续；非单位步长保持正确性。
4. 每个元素先读取 `x0` 与 `y0` 到寄存器/局部变量，再计算 `x1`、`y1`，最后写回。这样保证 `y1` 使用旧 `x0`，不需要全局临时向量。
5. Host 标量由 kernel 参数直接使用；Device 标量按传入的 device pointer 读取。标量内存位置标志由 Host 端确定，避免对 Host 地址进行设备访存。
6. 对 n 较小的输入仍以单次 kernel 为基线，避免引入额外的调度与同步。具体 block 数、每 block 元素数和向量化宽度通过 A2（910B3）实测调优。

#### Tiling 与分核策略

- tiling 只需传递 `n`、`incx`、`incy`、负步长起点、标量来源标志及设备标量指针/Host 标量值等必要信息。
- 按逻辑元素数分核，而不是按物理跨度分核；避免大步长造成无效任务膨胀。
- 分核区间采用 `base = n / coreNum` 与 `remain = n % coreNum` 均衡分配，前 `remain` 个 core 多处理一个元素，或采用等价的连续分块策略。
- 各 core 写入互不重叠的逻辑元素；尾块以 `min(tileSize, remaining)` 控制有效长度。
- 不使用动态 shape 推导，不需要矩阵类 tiling 或额外 workspace。

#### 并行、精度与性能优化

- 主要性能目标为连续访问（`incx = incy = 1`）的大向量场景。每个逻辑元素读取两个 complex64 并写回两个 complex64，理论数据搬运量约 32 bytes/元素，预期由显存带宽和 kernel launch 开销主导。
- 使用一次 kernel 完成读、计算、写，避免中间张量和多次 kernel launch；尽量提高连续场景的并行度和合并访存效率。
- 计算以 FLOAT32 为准，不做低精度转换。通过旧值暂存及显式复数公式控制运算顺序，避免错误重排导致语义变化。
- 不启用可能改变验收误差特征的激进近似数学选项；是否融合乘加由编译器默认行为决定，并通过规定的逐实部/虚部精度测试验证。

### 测试设计

测试工程放在 `test/rot/crot/arch22/`，按仓内 CSV/GTest 模式接入；arch22 分支在 `test/rot/crot/CMakeLists.txt` 注册 `crot_test`。Golden 按 Netlib/LAPACK `crot` 语义实现，并使用更新前的 `x` 与 `conj(s)` 计算 `y`。

| 测试组 | 覆盖内容 |
| --- | --- |
| 基本功能 | `n=1`、小质数、不同向量值和标量值；核对 x/y 原地结果 |
| shape 扫描 | 2 的幂、2 的幂 ±1、非对齐长度及至少 `2^20` 的大向量 |
| 步长 | `incx`、`incy` 的 `1/2/3/-1/-2/-3` 组合；正负步长独立组合 |
| 标量 | `c` 的 0、1、-1 和常规值；`s` 的 0、实数、纯虚数及常规复数 |
| 标量指针位置 | Host/Host、Host/Device、Device/Host、Device/Device |
| 填充与特殊值 | 零值、交替值、极端值、Inf/NaN；按规范单独统计非有限值 |
| 边界和异常 | `n=0` no-op、负 `n`、空指针、零步长、INT32_MIN、正负步长跨度溢出；错误用例只分配 dummy buffer 或空指针 |
| 分布 | 均匀分布与正态分布；正态输入需在配套生成器或 CSV 中补充，并纳入精度验证 |
| 性能 | `n=1` 至数百万元素；重点检查 `n=2^20/2^21/2^22`，单位步长，采集有效执行时间 |

对有限值，实部和虚部分别按 FLOAT32 精度标准比较，`atol = rtol = 2^-13`，required matched ratio 为 0.99，max absolute error 上限为 `1e-2` 或 32 ULP；Inf/NaN 按任务书的非有限值规则判定。性能按任务书要求在 Atlas 800T A2（910B3）warmup 5 次后至少采样 10 次，以 msprof 单算子 task duration 的算术平均与配套 GPU 基线比较，目标达到至少 0.8 倍 GPU 性能。测试 README 当前的 `aclrtEvent` 判定描述与任务书采样口径不同，最终报告按任务维护者确认后的统一口径生成。

## 支持硬件

| 支持的芯片版本 | 架构目录 | 支持状态 |
| --- | --- | --- |
| Atlas A2（含 Atlas 800T A2 / 910B3） | arch22 | 支持 |
| Atlas A3 | arch22 | 支持 |

自验证至少覆盖一种 A2/A3 款型；性能按任务书要求优先在 Atlas 800T A2（910B3）完成。

## 算子约束限制

- `n` 为 Host 侧 int 参数；负数非法，0 为快速返回。
- `incx`、`incy` 必须为非零 int32 且不等于 INT32_MIN；正负步长均支持。
- `n > 0` 时 `x`、`y`、`c`、`s` 必须非空；向量在 Device，标量允许 Host 或 Device。
- `(n-1)*abs(incx)` 与 `(n-1)*abs(incy)` 必须能由 int32 表示；校验中间值使用 int64。
- x/y 原地更新，不创建输出 Tensor；无广播、batch、额外 leading dimension 或动态 shape 约束。
- kernel 按 stream 异步提交，读回结果前由调用方同步对应 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 复数实部、虚部分别按 FLOAT32 容差检查；matched ratio ≥ 0.99，max absolute error ≤ `1e-2` 或 32 ULP；非有限值按任务书规则单独判定 | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | Atlas 800T A2（910B3）平均 task duration 达到 GPU 基线的至少 0.8 倍；warmup 5 次，至少 10 次有效样本 | 任务书 §3.3、`test_cases/gpu_baseline.csv` |
| 内存标准 | 算子无额外向量 workspace；测试报告记录任务要求的内存数据 | 任务书 §3.4、§4 |

## 兼容性分析

- 新增 `aclblasCrot` 公共声明，不修改现有 BLAS 接口或既有算子行为。
- 参数顺序与 cuBLAS `cublasCrot` 一致；旋转公式遵循 Netlib/LAPACK `crot`，特别是 `y'` 使用 `conj(s)` 以及原始 `x`。
- A2/A3 使用 arch22 实现目录。任务书指定的负 `n`、零步长和溢出处理是该任务的接口约束，不以其他架构或配套 README 中不同的 rot 族描述覆盖。
- 测试工程随实现提交至 ops-blas，测试用例、README 和脚本的期望状态码/计时口径需与最终确认的任务书验收口径一致。
