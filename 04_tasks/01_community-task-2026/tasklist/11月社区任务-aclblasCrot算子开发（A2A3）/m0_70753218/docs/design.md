# aclblasCrot 算子设计文档

## 需求背景

### 需求来源

本文档对应 2026 年 11 月社区任务《11月社区任务-aclblasCrot算子开发（A2A3）》，实现并验证 Atlas A2/A3（arch22）上的单精度复数 Givens 平面旋转算子 `aclblasCrot`。

任务依据：

1. `aclblasCrot_AtlasA2_A3_task_doc.md`。
2. `cann/ops-blas` 仓库当前 `master`，基线提交 `5bd9514`。
3. `cann-ops-competitions` 设计文档模板与 `任务流程细节.md`。
4. `ops-blas` 的 `blas/rot/arch22/csrot_*`、`blas/rot/arch35/crot_*` 及相关测试框架。

### 背景介绍

`aclblasCrot` 对两个 `complex64` 向量执行原地 Givens 旋转。旋转余弦 \(c\) 为 `float32`，旋转正弦 \(s\) 为 `complex64`。与同族 `aclblasCsrot` 的差别在于：`aclblasCrot` 的 \(s\) 有虚部，且 \(y\) 的更新使用 \(\operatorname{conj}(s)\)。

当前仓库已有 `include/cann_ops_blas.h` 中的 `aclblasCrot` 声明及 arch35 实现，缺少 arch22 实现、arch22 GTest/CSV 测试和 README 产品支持信息。本任务保持既有公共声明不变，在 `blas/rot/arch22/` 增加 Atlas A2/A3 实现，并补齐验证与文档。

## 需求分析

### 需求描述

接口签名为：

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

参数约定：

| 参数 | 类型与位置 | 约定 |
|---|---|---|
| `handle` | Host | 必须有效；为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | Host | \(n<0\) 返回 `ACLBLAS_STATUS_INVALID_VALUE`；\(n=0\) 快速返回成功，不读写数据 |
| `x`、`y` | Device `complex64` 向量 | 原地更新；\(n>0\) 时不得为空 |
| `incx`、`incy` | Host `int` | 支持正负非零步长；零步长和 `INT32_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `c` | Host 或 Device `float` 标量 | \(n>0\) 时不得为空 |
| `s` | Host 或 Device `complex64` 标量 | \(n>0\) 时不得为空 |

参数校验顺序固定为：

1. `handle` 是否为空。
2. \(n<0\)。
3. \(n=0\) 快速返回。
4. `incx`、`incy` 的零值、`INT32_MIN` 与跨度溢出。
5. \(n>0\) 时 `x/y/c/s` 的空指针。

其中跨度使用 `int64_t` 计算：

\[
\text{lastX}=(n-1)\times |\text{incx}|,\qquad
\text{lastY}=(n-1)\times |\text{incy}|.
\]

`lastX` 或 `lastY` 超过 `INT32_MAX` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

### 口径核对

随任务提供的 `test_cases/README.md` 将 \(n<0\) 和零步长写成合法输入，与正式任务书 §2.1、§2.4、§2.5 的 arch22 口径冲突。本设计以任务书原件为主判据：

- \(n<0\)：`ACLBLAS_STATUS_INVALID_VALUE`。
- `incx==0`、`incy==0`、`incx==INT32_MIN`、`incy==INT32_MIN`：`ACLBLAS_STATUS_INVALID_VALUE`。
- \(n=0\)：`ACLBLAS_STATUS_SUCCESS`，不修改 `x/y`。

随附 README 的相反描述只作为诊断记录，不进入正式验收门槛。

### 需求拆解

1. 实现 arch22 Host 入口、参数校验、标量位置识别和异步 kernel 下发。
2. 实现单位步长连续双缓冲向量路径，面向 \(n\ge 2^{20}\) 的性能用例。
3. 实现正负非单位步长路径，负步长从物理向量尾端起算。
4. 支持 `c`、`s` 独立位于 Host 或 Device 内存，覆盖 HH、HD、DH、DD 四种组合。
5. 保留任务提供的 1200 条 CSV 用例，补充任务书要求的边界负向和特殊值检查。
6. 对 200 条性能用例输出可被验收脚本解析的 `[PERF] ... avg_us=` 数据。
7. 更新 `blas/rot/README.md` 的 Atlas A2/A3 产品支持与约束说明。

## 详细设计

### 算子分析

#### 数学公式

对逻辑下标 \(i=0,\ldots,n-1\)，复数元素

\[
x_i=xr_i+\mathrm{i}\,xi_i,\qquad
y_i=yr_i+\mathrm{i}\,yi_i,\qquad
s=sr+\mathrm{i}\,si,
\]

执行：

\[
x'_i=c x_i+s y_i,
\]

\[
y'_i=c y_i-\operatorname{conj}(s)x_i.
\]

实部、虚部分开计算：

\[
\begin{aligned}
xr'_i &= c\,xr_i+sr\,yr_i-si\,yi_i,\\
xi'_i &= c\,xi_i+sr\,yi_i+si\,yr_i,\\
yr'_i &= c\,yr_i-sr\,xr_i-si\,xi_i,\\
yi'_i &= c\,yi_i-sr\,xi_i+si\,xr_i.
\end{aligned}
\]

\(y'_i\) 使用旋转前的原始 \(x_i\)。实现先在 UB 中保存四个原始分量并完成两组结果，再写回 `x/y`。

#### 支持数据类型

- `x`、`y`、`s`：`aclblasComplex`，实部与虚部各为 `float32`，内存布局为 `{real, imag}`。
- `c`：`float32`。

#### 支持形状与步长

正步长时，逻辑下标 \(i\) 对应物理复数下标 \(i\cdot \text{inc}\)。

负步长时：

\[
j_i=(n-1-i)\cdot |\text{inc}|.
\]

每个向量的物理元素数为 \(1+(n-1)|\text{inc}|\)。任务按非连续一维向量处理，不涉及 leading dimension、广播或动态 shape。

### 算子实现

#### 文件组织

实现文件：

```text
blas/rot/arch22/
├── crot_host.cpp
├── crot_kernel.cpp
├── crot_kernel.h
└── crot_tiling_data.h
```

测试文件：

```text
test/rot/crot/arch22/
├── crot_npu_wrapper.h
├── crot_test.cpp
└── crot_test.csv
```

#### Host 侧设计

`aclblasCrot` 在 Host 完成参数校验、设备 AIV 核数查询、分核参数计算、`c/s` 指针位置识别和 kernel 下发。

`c` 与 `s` 分别通过 `aclrtPointerGetAttributes` 判断内存位置。Host 指针的值写入 tiling；Device 指针原样转发给 kernel。两种标量位置互不影响。

连续路径采用均分策略：

\[
q=\left\lfloor\frac{n}{B}\right\rfloor,\qquad r=n\bmod B.
\]

前 \(r\) 个核处理 \(q+1\) 个元素，其余核处理 \(q\) 个元素。分核数 \(B=\min(n,C)\)，其中 \(C\) 为当前设备 AIV 核数。

非单位步长路径按 `ceil(n/128)` 估算核数，并受 \(C\) 限制。各核处理互不重叠的逻辑下标区间。

Host 不分配 workspace，不调用 `aclrtSynchronizeStream`。kernel 通过 `handle->stream` 异步下发，读回结果由调用方同步同一 stream。

#### Tiling 设计

`CrotTilingData` 保存：

- 总长度 `totalN`。
- `incx`、`incy`。
- 每核基础元素数 `perCoreN` 与余数 `remainder`。
- Host 侧 `c`、`s` 标量值。
- `c/s` 是否位于 Device。
- 路径选择 `tilingKey`。

| `tilingKey` | 选择条件 | 实现 |
|---:|---|---|
| 0 | 连续路径中不属于 1、2 的输入 | 通用复数双缓冲 Vector 路径 |
| 1 | 任一非单位步长；\(n<64\)；\(64\le n<8192\) 且 Host \(s_i\ne0\) | 标量跨步路径 |
| 2 | `incx==incy==1`、\(n\ge8192\)、Host 标量 \(s_i=0\) | 实数正弦双缓冲 Vector 快路径 |

`tilingKey=2` 是通用公式在 Host 标量 \(s_i=0\) 时的严格特化，仍支持任意 \(c\) 和 \(s_r\)。Device 标量不增加 Host 同步读回：kernel 读取 Device \(s_i\) 后，为零时进入同一实数正弦路径，非零时执行通用路径。

\(64\le n<8192\) 且 \(s_i=0\) 使用 `tilingKey=0`；大 shape 使用 `tilingKey=2`。两条路径都显式保留零乘以 Inf 得到 NaN 的语义。

#### Kernel 侧设计

##### 1. 连续向量路径

连续路径使用输入预取、Vector 计算和输出写回交错执行的双缓冲流水。每个 stage 拥有独立的 `x/y` 输入和 `x` 结果缓冲；前一块计算时预取下一块，`MTE3_MTE2` 事件保证上一块写回完成后才复用同一 stage。

通用复数路径单 tile 处理 3008 个复数元素。AoS 浮点序列按 `{real, imag}` 交错保存，通过 `Gather` 交换每个复数内部的实部和虚部，再对交换后的 lane 施加 `[-s_i,+s_i]` 符号。Host 在 handle 中缓存完整 Gather 字节偏移和 64 个元素的符号种子；Kernel 将符号种子与 \(\pm s_i\) 相乘后，以 `src1RepStride=0` 在每次 Vector repeat 中重复使用 64 lane 系数，避免为每个 tile 保存完整缩放表。

通用复数路径显式 UB 占用为：

\[
(6144+64)\times4+7\times24064+256=193536\ \text{B}.
\]

其中包含约 24 KiB Gather/符号表、7 个 24 KiB tile 缓冲和两个 64 lane 缩放种子。

处理流程：

1. 满 tile 且在 32B 对齐时使用 `DataCopy`；尾块或非 32B 对齐地址使用 `DataCopyPad` 搬运 `x/y`。
2. `Gather` 交换 AoS 复数内部的实部/虚部 lane。
3. 使用独立的 `Muls`、`Axpy`、`Add` 计算四个结果分量，保持已验证的运算顺序。
4. `x` 结果暂存在独立 tile 缓冲，`y` 结果覆盖已消费的输入 tile，保证 \(y'\) 使用原始 \(x\)。
5. 满 tile 使用 `DataCopy`，尾块使用 `DataCopyPad` 写回有效元素。

每个核处理独立且连续的复数区间，核间无重叠；块尾按实际元素数搬运，防止越过逻辑边界。

##### 2. 跨步路径

跨步路径将逻辑下标映射为物理复数下标，读取原始四个分量后计算并写回。

每核每 pass 处理至多 1024 个逻辑元素。结果先写入 32B 对齐的 UB slot，再以每个复数 8B 的 `DataCopyPad` 写回 GM，避免对每个浮点分量单独发起 GM 写操作。

负步长使用：

\[
(n-1-i)\cdot|\text{inc}|.
\]

主机侧已拦截零步长、`INT32_MIN` 和跨度溢出，因此 kernel 中不存在负数取绝对值或 GM 地址窄化溢出。

##### 3. 实数正弦连续快路径

当 \(s_i=0\) 时，公式退化为：

\[
x'=cx+s_r y,\qquad
y'=cy-s_r x.
\]

该路径使用 3072-complex tile 和相同的双缓冲事件序列。主公式执行两次 `Muls` 和两次 `Axpy`；随后用两次 `Gather + Axpy(0)` 显式加入零交叉项，保留 \(0\cdot\infty=\mathrm{NaN}\) 的 IEEE 语义。六个 stage 缓冲、一个 scratch 和 Gather 表的显式 UB 占用为：

\[
8\times3072\times2\times4=196608\ \text{B}.
\]

200 条正式性能用例中 198 条满足 \(s_i=0\)，其余 2 条继续走通用复数路径。

##### 4. 小连续向量路径

连续路径按每核约 1024 个复数限制核数：

\[
B=\min\left(C,\left\lceil\frac{n}{1024}\right\rceil\right).
\]

这样 \(n=64\ldots4096\) 不会启动全部 AIV 核。短向量只加载实际需要的 Gather 偏移和 64 项符号种子，不加载完整 tile 表。\(n<64\) 或 \(64\le n<8192\) 且 \(s_i\ne0\) 的连续小向量归入 `tilingKey=1`，Kernel 直接使用 `GlobalTensor::GetValue/SetValue` 完成标量旋转。

## 支持硬件

| 芯片与软件 | 状态 |
|---|---|
| Atlas 800T A2、910B3、CANN 9.1.0 | 开发与验收目标 |
| Atlas A3 系列、arch22、CANN 9.1.0 | 任务要求支持 |
| Ascend 950PR、arch35、CANN 9.1.0 | 仓库已有独立实现，本设计不改动 |

## 算子约束限制

1. 当 \(n>0\) 时，`x/y/c/s` 必须非空。
2. `incx/incy` 必须为非零 `int32`，且不等于 `INT32_MIN`。
3. 任一侧跨度 \((n-1)|\text{inc}|\) 不得超过 `INT32_MAX`。
4. `x/y` 为原地输入输出；调用方在 stream 完成前保持 Device 内存有效。
5. 本任务支持 `x/y` 各自原地更新。`x/y` 部分重叠的跨步别名场景未列为正式支持范围。

## 可维可测分析

### 精度标准

复数实部、虚部分别按 FLOAT32 混合容差判定：

- `rtol = atol = 2^-13`。
- 匹配比例不小于 0.99。
- 最大绝对误差不超过 \(10^{-2}\) 或 32 ULP。
- Inf/NaN 按任务书 §3.2 的等效规则单独统计。

非有限分量判定为 NaN↔NaN、同号 Inf↔Inf 或 NaN↔±Inf 等效；有限分量单独计算匹配率与最大绝对误差。全 NaN 传播只输出辅助诊断，不作为正式门槛。

测试使用 `crot_golden.h` 中的 CPU 参考实现，公式与 kernel 一致，使用旋转前原始 `x` 计算 `y`。

### 测试用例

正式精度测试保留任务提供的 1200 条 CSV 用例，其中 1000 条精度用例、200 条 `TC_PF` 性能用例。

1000 条精度用例中保留 500 条原均匀分布，另外 500 条的 x、y 改为 `GAUSS_NORM_5_2`。同批 500 条中，250 条的 c、s 分量按 [-5,5] 均匀分布生成，250 条按 μ∈[-5,5]、σ∈[0.1,2] 的正态分布生成。

补充的专项检查：

- `handle=nullptr` 优先级。
- \(n<0\)。
- \(n=0\) 不修改 `x/y`。
- 零步长与 `INT32_MIN`。
- `incx/incy` 跨度溢出。
- Host/Device 混合标量指针。
- x/y 相对 32B 设备基址的 +1/+2/+3 complex 对齐偏移。
- 正负步长、非 1 步长与空洞元素保持不变。
- Inf/NaN 输入。

最终自验证执行 1008/1008 通过。

### 性能标准

任务书 §3.3 与随附 `verify_performance.py` 分别指向 `msprof op` 和 GTest Event。自验证采用仓库脚本及已合并同族任务的口径：200 条 `TC_PF` 的 Event 数据作为脚本主判定，三条典型形状另用 `msprof op` 的 Task Duration 作为正式交叉证据。该解释已在设计 PR 评论中请求维护者确认。

GTest 直接使用驻留 Device 的 `x/y` 和标量：

1. 预热 10 次。
2. 使用 `aclrtEvent` 包围连续 20 次公共 API 下发。
3. 在 stream 同步后计算平均单次耗时。
4. 输出 `[PERF] case=<case_name> ... avg_us=<value>`。

验收判据：

\[
\text{NPU avg\_us}\le \frac{\text{GPU baseline us}}{0.8}.
\]

`verify_performance.py` 读取 200 条 `[PERF]` 记录并与 `gpu_baseline.csv` 逐例比较。`msprof op` 的 Task Duration 用于三档典型形状的交叉验证；两种证据分开保存。

最终 Event 全量判定为 200/200 通过。

Atlas 910B3、CANN 9.1.0 实测三档 `msprof op` task duration 为 22.78 / 39.50 / 72.44 us，上限为 23.7226665 / 56.917334 / 109.333333 us。

### 内存

任务书 §3.4 不设内存阈值。通用复数连续路径单核显式 UB 占用 193536 B，实数正弦快路径占用 196608 B，跨步路径占用 64 KiB；算子不申请 GM workspace。

### 测试执行

精度与性能分开执行、分开判定：

- 精度：排除 `TC_PF`，运行 1000 条 CSV 精度用例及专项 TEST_F。
- 性能：仅运行 200 条 `TC_PF`，解析 GTest 的 `[PERF]` 行。
- Inf/NaN、负数、零步长、跨度溢出使用小缓冲或 dummy 缓冲。

## 兼容性分析

1. `aclblasCrot` 公共声明已在当前仓库存在，签名与任务书一致，本任务不新增私有接口。
2. arch35 实现和测试保持不变。
3. arch22 实现新增于独立目录，由现有源码收集规则编译。
4. `test/rot/crot/CMakeLists.txt` 当前已注册 `crot_test`，新增 arch22 源文件后，`--soc=ascend910b` 会构建 GTest。
5. `blas/rot/README.md` 将 `aclblasCrot` 的 Atlas A2/A3 支持状态改为支持，并明确 arch22 与 arch35 的入参差异。

## 交付件映射

| 任务书要求 | 对应内容 |
|---|---|
| 设计文档 | 本文件，提交至 `cann-ops-competitions` 设计文档 PR |
| 自测用例与测试代码 | `test/rot/crot/arch22/` 及 1200 条 CSV |
| 自测报告 | `task_submission/` 下精度、性能、内存报告与日志 |
| 待验收代码地址 | ops-blas 个人仓分支、完整 commit 链接与算子 README |
| 产品支持 | `blas/rot/README.md` 标注 Atlas A2/A3 支持 |

## 验收判定

正式验收只采用任务书原文和维护者明确答复。自增极端诊断、多轮复测和额外硬件不作为新增门槛。

设计 PR 只包含本设计文档。代码 PR 在验收测试通过后按任务流程提交。
