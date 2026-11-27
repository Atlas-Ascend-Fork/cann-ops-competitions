# aclblasChpr2 算子开发(A2/A3)

## 需求来源

本算子来自 2026 年 11 月 CANN 社区任务“aclblasChpr2 算子开发（A2/A3）”。任务要求在 Atlas A2/A3 系列产品上，基于 ops-blas 句柄式 BLAS 工程，使用 Ascend C 实现单精度复数 Hermitian 压缩存储秩-2 更新算子 `aclblasChpr2`，核心功能与参数语义对标 cuBLAS `cublasChpr2`，数值参考 Netlib CHPR2/CBLAS。

任务依据：

- `aclblasChpr2_AtlasA2_A3_task_doc.md`：功能、接口、精度、性能和交付要求；
- [Netlib CHPR2](https://www.netlib.org/blas/chpr2.f)：计算与压缩存储语义参考；
- ops-blas 已有 `blas/spr2/arch35/` 实现：工程组织与接口行为参考。

## 背景介绍

### aclblasChpr2 算子实现

本任务补齐 arch22（Atlas A2/A3）版本，不新增公共接口声明。代码路径如下：

- 算子实现：`blas/spr2/arch22/`；
- 测试实现：`test/spr2/chpr2/arch22/`；
- 公共接口声明：`include/cann_ops_blas.h`；
- 复数与状态码定义：`include/cann_ops_blas_common.h`。

实现采用 handle 绑定 stream 的 Ascend C kernel 直调模式，不引入私有平行 API。

### 参考实现现状分析

arch35 已有同类实现，可作为接口校验、数据布局和测试工程组织的参考，但 arch22 需要结合 AIV 指令、UB 容量和 DMA 参数限制独立设计分列、分块与跨步搬运方案。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 上下文及 stream | handle | aclblasHandle_t | 非空 | scalar |
| uplo | 指定读写上三角或下三角 | 枚举 | aclblasFillMode_t | UPPER 或 LOWER | 属性 |
| n | Hermitian 矩阵阶数 | 标量 | int32 | n ≥ 0 | 属性 |
| alpha | 复数标量乘数，支持 Host/Device 指针 | 标量指针 | complex64 | 非空 | scalar |
| x、y | 两个输入复数向量 | tensor | complex64 | 任意非零步长，INT_MIN 除外 | 逻辑长度 n |
| incx、incy | x、y 的元素步长 | 属性 | int32 | 非零且不为 INT_MIN | scalar |
| AP | 原地更新的压缩 Hermitian 矩阵 | tensor | complex64 | 仅读写 uplo 指定三角，不得与 x/y 可读写区域重叠 | n(n+1)/2 个复数元素 |

### aclblasChpr2 算子功能分析

算子计算：

```text
AP = alpha × x × y^H + conj(alpha) × y × x^H + AP
```

输入为 `alpha`、`x`、`y` 和旧 `AP`，输出为原地覆写的 `AP`。矩阵按列主序将 uplo 指定三角无间隙压缩存储，无前导维度参数；未存储的另一半由 Hermitian 共轭对称性隐含表示。实际执行更新时，对角元素虚部强制置零。

支持数据类型为 complex64，实部和虚部均为 FLOAT32。不支持广播，x/y 的非连续访问仅由 incx/incy 描述。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 arch22 上实现 `aclblasChpr2`，支持 Atlas 800T A2 和 Atlas 800T A3，适配 CANN 9.1.0。实现需覆盖正负步长、Host/Device alpha、空规模和零标量 quick return、压缩矩阵原地更新、对角虚部处理及规定的参数校验顺序，并满足任务书规定的精度和性能验收标准。

接口定义如下：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```

## 需求拆解

1. **功能正确**：实现 Hermitian packed 秩-2 更新，正确处理 UPPER/LOWER、负步长、对角虚部和特殊值。
2. **接口兼容**：沿用现有公共声明、handle、stream、状态码和 Host/Device 标量语义。
3. **边界完整**：按规定顺序校验参数；n=0 和 alpha=0 时保持 AP 不变，同时不豁免前置校验。
4. **并行与访存合理**：按列划分多核任务，按行块复用 x/y，连续搬运 AP，并处理非对齐和尾块。
5. **结果可验收**：使用 ops-blas ATK/GTest 框架完成 1000 条精度、200 条性能及边界自验；正式性能采用 msprof kernel 耗时。

# 详细设计（required）

## 算子分析

### 数学公式

对所选三角中第 j 列的元素 A(i,j)，令：

```text
t1 = alpha × conj(y[j])
t2 = conj(alpha × x[j])
A(i,j) = A(i,j) + x[i] × t1 + y[i] × t2
```

该形式与 Netlib CHPR2 一致。复数乘法展开为：

```text
(a + i·b)(c + i·d) = (a·c - b·d) + i·(a·d + b·c)
```

不通过代数变形改变运算顺序，以避免引入额外的精度差异或改变 Inf/NaN 传播行为。

### 压缩布局与步长索引

AP 按列主序逐列无间隙打包，仅保存 n(n+1)/2 个复数元素：

| 三角 | 行范围 | AP 中 A(i,j) 的元素下标 |
| --- | --- | --- |
| UPPER | 0 ≤ i ≤ j | `i + j(j+1)/2` |
| LOWER | j ≤ i < n | `i + j(2n-j-1)/2` |

向量逻辑元素 k 的物理下标为：

```text
inc > 0：k × inc
inc < 0：(n - 1 - k) × |inc|
```

列起点、AP 下标和向量物理偏移使用 64 位整数计算，避免大 n 或大步长下中间结果溢出。

### 支持数据类型

- alpha、x、y、AP 均为 complex64；
- 每个复数由两个 FLOAT32 分量按实部、虚部交错存储；
- alpha 支持 Host 和 Device 指针；
- 不涉及其他 dtype 或计算精度转换。

### 支持形状

- n 为非负 int32，AP 的逻辑长度为 n(n+1)/2；
- x、y 的逻辑长度均为 n，物理元素数分别至少为 `1+(n-1)×|incx|` 和 `1+(n-1)×|incy|`；
- 不支持广播，也不要求超出步长语义的任意非连续布局；
- n=0 时逻辑数据为空，完成结构参数校验后直接返回成功。

## 算子实现

### 实现方案

Host 负责参数校验、alpha 位置识别、tiling 和 kernel 发射；Kernel 使用 AIV 多核按列更新 AP，并按最多 1024 个逻辑行分块复用 x/y。Kernel 分为 Init 和 Process 两个阶段，Process 包括 Device alpha 处理、CopyIn、Compute 和 CopyOut。

本实现不使用 tilingkey。uplo、n、步长和 Host alpha 等运行时信息通过 `Chpr2TilingData` 传递，步长分支在 kernel 内选择。

#### 3.2.1 Host 侧设计

##### 1. 参数校验与 quick return

严格按照任务书规定顺序处理：

| 顺序 | 检查或行为 | 返回值/结果 |
| --- | --- | --- |
| 1 | handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | n < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | alpha 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | uplo 非 UPPER/LOWER | `ACLBLAS_STATUS_INVALID_ENUM` |
| 5 | incx/incy 为 0 或 INT_MIN | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | n=0 | 成功返回，不检查 x/y/AP，不启动 kernel |
| 7 | n>0 且 x/y/AP 任一为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | 识别 alpha 的 Host/Device 位置 | 位置检查失败时返回对应错误 |
| 9 | Host alpha=(0,0) | 成功返回，不读取 x/y 数值，AP 完全不变 |
| 10 | 向量物理字节长度无法由 size_t 表示 | `ACLBLAS_STATUS_INVALID_VALUE`，防止地址计算溢出 |

alpha=0 的 quick return 位于 x/y/AP 指针校验之后，因此 alpha=0 叠加空 tensor 指针仍返回 `ACLBLAS_STATUS_INVALID_VALUE`。n=0 或 alpha=0 时不执行对角虚部清零，AP 保持原值。

Device alpha 不在 Host 侧读取或同步。其数值由同一 stream 上的 kernel 读取；若为 (0,0)，kernel 直接结束且 AP 不变，从而保持前序 Device 写 alpha 与本算子的 stream 顺序。

##### 2. 分核与 tiling 策略

Host 从平台信息获取 AIV 核数，并设置：

```text
numBlocks = min(n, AIV 核数)
```

`Chpr2TilingData` 保存 n、uplo、numBlocks、Host alpha 的实部/虚部以及 64 位 incx/incy。Host alpha 为零时已 quick return；Device alpha 场景的标量字段由 kernel 读取后覆盖。

Kernel 通过 handle 绑定的 stream 异步发射。发射失败统一返回 `ACLBLAS_STATUS_EXECUTION_FAILED`；接口成功返回仅表示任务已提交，调用方读取 Device 结果前须同步该 stream。

#### 3.2.2 Kernel 侧设计

##### 1. 初始化与 Device alpha

Init 设置 x、y、AP 的 GM 地址并分配 UB 缓冲区。Process 开始时，如 alpha 位于 Device，则通过 `DataCopyPad` 读取两个 FLOAT32 分量并等待标量可读；alpha=(0,0) 时直接返回，否则写入本核 tiling 副本。

##### 2. 蛇形分列与行分块

每轮每核最多处理一列。第 ord 轮中，核 blockIdx 对应列为：

```text
j = ord × numBlocks + blockIdx                 （ord 为偶数）
j = ord × numBlocks + numBlocks - 1 - blockIdx （ord 为奇数）
```

j ≥ n 时跳过。该蛇形分配使相邻轮的长列和短列尽量组合，缓解 UPPER/LOWER 列长递增或递减造成的负载不均；尾轮按实际剩余列数处理，不保证各核严格等量。每列仅由一个核写入，避免跨核写冲突。

每个核再按 1024 个逻辑行切分 x/y。对当前列只计算其三角行范围与当前行块的交集，不搬运或更新无关 AP 元素。

##### 3. CopyIn：向量与 AP 搬运

- **单位步长**：x/y 使用 `DataCopyPad` 连续搬入当前行块。
- **非单位步长**：优先使用 strided DMA 批量读取复数元素，再通过 Gather 整理为连续布局；负步长在 Gather 索引中完成反序。
- **超大间隔**：当 DMA gap 超出 32 位字段表示范围时，使用 64 位物理下标逐元素精确读取，避免 stride 截断。
- **列系数**：x[j]/y[j] 位于当前行块时直接读取 UB，否则通过标量路径从 GM 读取。
- **AP**：每列每段按有效字节数使用 `DataCopyPad` 连续搬入，支持非对齐起点和尾段。

搬入后将复数的两个交错分量构造为 `(real, imag)` 和 `(-imag, real)` 两种布局，供实数向量指令计算复数乘法。

##### 4. Compute：复数更新与特殊场景

对列系数 t1、t2，非对角段使用 `Muls`、`Axpy` 和 `Add` 完成：

```text
x × t1：x.raw × t1.real + x.sprime × t1.imag
y × t2：y.raw × t2.real + y.sprime × t2.imag
AP：    AP + 第一项 + 第二项
```

其中 `sprime=(-imag, real)`。列系数和对角实部采用标量四项乘加路径；非对角采用上述等价实数展开，不引入近似计算。通过 x/y/AP 分别含 Inf/NaN 的用例验证各路径与 CBLAS golden 的特殊值传播一致。

对角元素不参与虚部结果判定，最终虚部直接写零；对角实部使用旧 AP 实部加两项贡献的实部，忽略输入对角虚部。当 x[j] 和 y[j] 的四个分量均为零时，该列非对角元素不变，仅将实际执行更新范围内的对角虚部清零。

##### 5. CopyOut 与同步

计算后的 AP 段使用 `DataCopyPad` 原地写回。UB 复用前通过 MTE2/MTE3、Vector、Scalar 之间的事件和 `PipeBarrier` 建立依赖，确保：

- GM 数据搬入完成后才进行 Vector/Scalar 读取；
- 上一段计算完成后才覆盖其输入或临时 UB；
- 对角标量修正完成后才写回；
- 下一行块搬入前等待上一阶段不再使用对应缓冲区。

当前采用单 buffer 的保守同步方案，不假设搬运与计算已完全重叠。主要 AP 访问均为列内连续读写，蛇形分列用于改善核间负载；是否达到性能目标以正式 msprof 结果为准。

##### 6. UB 预算

行块大小为 1024 个复数元素，主要缓冲区如下：

| 缓冲区 | 用途 | 单核占用 |
| --- | --- | --- |
| x/y raw 与 sprime | 两个向量的原始及变换布局 | 32 KiB |
| AP、D1、D2、work | AP 段、两项贡献及临时计算 | 32 KiB |
| shiftIdx、packIdx | Gather 与交错布局索引 | 16 KiB |
| splitIdx | 实部/虚部分离索引 | 4 KiB |
| scalar | Device alpha、列系数和标量修正 | 32 Byte |
| strided（可选） | 跨步 DMA 的 32 Byte 块暂存 | 32 KiB |
| **合计** | 单位步长 / 含跨步暂存 | **84 KiB+32 Byte / 116 KiB+32 Byte** |

该预算覆盖索引表和临时空间，不启用 double buffer；A2/A3 使用同一行块上限和缓冲区布局。

## 支持硬件

| 支持的芯片版本 | 架构/SoC | CANN | 涉及勾选 |
| --- | --- | --- | --- |
| Atlas 800T A2 | arch22 / ascend910b3 | 9.1.0 | √ |
| Atlas 800T A3 | arch22 / ascend910_93 | 9.1.0 | √ |

自验证可覆盖其中一种款型；正式性能测试设备为 Atlas 800T A2（910B3）。

## 算子约束限制

| 约束项 | 本算子行为 |
| --- | --- |
| 数据类型 | 仅支持 complex64，实部/虚部均为 FLOAT32 |
| 广播 | 不支持 |
| 非连续访问 | 仅支持 incx/incy 描述的向量步长，不使用 AutoContiguous/ViewCopy |
| 动态形状 | 不涉及框架 dynamic shape；n 和步长均为运行时参数，由 Host 完成 tiling 与 launch |
| 原地与重叠 | AP 原地更新；AP 可写区不得与 x/y 可读区重叠 |
| 空数据 | n=0 完成结构校验后成功返回，不引用 x/y/AP，不启动 kernel |
| 零标量 | alpha=0 时 AP 完全不变；tensor 指针校验先于 alpha=0 quick return |
| 标量位置 | alpha 支持 Host/Device 指针，Device alpha 在绑定 stream 上读取 |
| 地址范围 | incx/incy 不得为 0 或 INT_MIN；向量物理字节长度溢出 size_t 时返回 INVALID_VALUE |
| 异步执行 | 返回成功不代表计算完成，读取结果前须同步 handle 绑定的 stream |
| 确定性 | 不要求跨实现逐位一致，按任务书精度标准验收 |

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | 1000 条 CSV 全量用例；实部、虚部分别与 Netlib CBLAS 3.10.0 的 `cblas_chpr2` 比较。atol=rtol=2^-13，matched_ratio ≥ 0.99；每个有限值元素的绝对误差还须不超过 `max(1e-2, 32×ULP(|golden|))` | 任务书 §3.2、§3.5 |
| 边界标准 | 29 条 CSV 边界用例与 5 条独立接口用例，覆盖校验顺序、quick return、Device alpha、stream 顺序、零列和对角虚部；状态码及 AP 保持/更新行为符合任务书 | 任务书 §2.4、§2.5、§3.5 |
| 性能标准 | 200 条性能用例；Atlas 800T A2 910B3 上使用 msprof op 采集 kernel Task Duration，预热 5 次后取 10 个有效样本的算术平均；单 case NPU 平均耗时 ≤ GPU kernel 基线 × 1.25 | 任务书 §3.3 |
| 内存标准 | 不涉及；精度测试仍通过缓冲区哨兵检查越界读写 | 任务书 §3.4 |

### 测试覆盖与可复现性

1. 使用 ops-blas test 目录的 ATK/GTest 框架加载 CSV，并保留逐用例日志、实部/虚部结果和 XML。
2. 精度覆盖 UPPER/LOWER、n=0 至 4096、奇数与分块尾段、实数/纯虚数/复数 alpha、正负及大步长、均匀与正态输入，以及 Inf/NaN 和非零对角虚部。
3. quick return 用例检查 AP 原值保持不变；Device alpha 用例通过同 stream 的前序拷贝验证执行顺序。
4. 正式性能仅比较 GPU kernel-only 与 NPU msprof kernel-only 耗时；ACL event 计时仅用于快速回归，不与正式判据混用。
5. 测试步骤、环境、用例清单、原始日志和报告归档于 `task_submission/`，保证验收人可复现。

## 兼容性分析

本任务是在已有 ops-blas 公共接口下补齐 arch22 实现，不修改函数签名、参数顺序、复数类型和状态码定义，不新增私有平行 API。arch35 实现保持独立，既有产品线行为不受影响。

AP 布局、负步长、quick return 和 Host/Device alpha 语义与任务书及现有 BLAS 接口约定保持一致。新增测试集中在 `test/spr2/chpr2/arch22/`，通过分组运行确认基础、边界、全量精度和正式性能结果。