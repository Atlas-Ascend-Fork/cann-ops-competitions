# aclblasSgemvBatched 类算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 11 月社区任务 · GEMV Batched 五个实数接口（A2/A3） |
| 工程 | `cann/ops-blas`，实现位于 `blas/gemv_batched/arch22/`，测试位于 `test/gemv_batched/arch22/` |
| 硬件 | Atlas A2/A3（dav-2201），性能验收设备为 Atlas 800T A2（910B3） |
| 软件 | 任务书指定 CANN 9.1.0 |
| 标杆 | cuBLAS 对应接口的 A100 kernel 时间，Netlib CBLAS `sgemv` 逐批精度参考 |

## 一、需求背景

### 1.1 需求来源

在 Atlas A2/A3 上实现句柄式实数批量矩阵向量乘法，覆盖 S、HSH、HSS、TST、TSS 五个变体，接口参数顺序与对应 cuBLAS 接口一致。复用 `include/cann_ops_blas.h` 中的公开声明。工程采用 Ascend C 内核直调，调用者通过 `aclblasSetStream` 绑定 stream。

### 1.2 标杆与参考实现

任务的 BLAS 入口通过 handle 直接下发内核，TBE 实现、TBE 原型和算子信息库不参与该调用链。任务书指定以下参考：

| 用途 | 参考 |
| --- | --- |
| 公开接口与现有行为 | [ops-blas 公共头](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h) |
| 同族 Ascend C 参考 | [arch35 Host](https://gitcode.com/cann/ops-blas/blob/master/blas/gemv_batched/arch35/gemv_batched_host.cpp)、[arch35 Kernel](https://gitcode.com/cann/ops-blas/blob/master/blas/gemv_batched/arch35/gemv_batched_kernel.cpp) |
| 精度标杆与访存语义 | [Netlib sgemv.f](https://www.netlib.org/blas/sgemv.f) 和 [ops-blas CBLAS golden](https://gitcode.com/cann/ops-blas/blob/master/test/gemv_batched/gemv_batched_golden.h) |
| 性能标杆 | [cuBLAS GEMV Batched](https://docs.nvidia.com/cuda/cublas/index.html) 和任务附件各变体的 `gpu_baseline.csv` |

精度参考按 batch 解释指针数组，将窄类型输入解释成 FP32 后调用 `cblas_sgemv`，并按输出类型进行舍入。Netlib 的 N 路径按矩阵列累加，T/C 路径逐列求点积。下图描述公开精度参考的语义与控制流。GPU 时间来自任务包中的 cuBLAS 实测，cuBLAS 内核实现细节未公开。

```mermaid
flowchart TD
    A["组装五个变体的参考输入"] --> B["判断空尺寸、空批次"]
    B -->|"空操作"| Z["返回成功"]
    B -->|"有输出"| C["逐 batch 调用 CBLAS SGEMV"]
    C --> D["按负步长定位 x 和 y 的逻辑起点"]
    D --> E["按 beta 置零或缩放 y"]
    E --> F{"alpha 为零"}
    F -->|"是"| I["按目标 dtype 舍入 y"]
    F -->|"否"| G{"trans"}
    G -->|"N"| H["按物理列累加到 y"]
    G -->|"T 或 C"| J["按物理列计算点积"]
    H --> I
    J --> I
```

## 二、需求分析

### 2.1 数学定义和布局

对 `b = 0, ..., batchCount-1` 执行 `y[b] = alpha · op(A[b]) · x[b] + beta · y[b]`。所有数据缓冲彼此独立寻址。`A[b](r,c)` 的元素偏移是 `c*lda+r`，矩阵物理布局恒为列主序 `lda × n`，`lda >= max(1,m)`。

| trans | op(A) | x 逻辑长度 | y 逻辑长度 |
| --- | --- | ---: | ---: |
| N | A | n | m |
| T、C | A 的转置 | m | n |

长度 `L` 的向量对负步长从 `(1-L)*inc` 开始，其余从 0 开始。位置 `j` 的物理下标为 `start+j*inc`。

### 2.2 接口和精度类型

所有接口共用参数顺序 `handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount`。三个数组都位于 Device 且各含 `batchCount` 个地址，`alpha` 与 `beta` 是 Host FP32 指针。`m`、`n`、`lda`、步长和 `batchCount` 均为 `int`。FP16/BF16 以 `uint16_t` 承载位模式。

| 接口 | A、x | y | 乘加和标量类型 |
| --- | --- | --- | --- |
| `aclblasSgemvBatched` | FP32 | FP32 | FP32 |
| `aclblasHSHgemvBatched` | FP16 | FP16 | FP32 |
| `aclblasHSSgemvBatched` | FP16 | FP32 | FP32 |
| `aclblasTSTgemvBatched` | BF16 | BF16 | FP32 |
| `aclblasTSSgemvBatched` | BF16 | FP32 | FP32 |

### 2.3 参数约束和异常行为

- 按 handle、trans、m/n、lda、incx/incy、batchCount、alpha/beta、Device 指针数组的顺序处理校验。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法 trans 返回 `ACLBLAS_STATUS_INVALID_ENUM`，其他非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- `m`、`n`、`batchCount` 允许零，结构参数校验通过即可快速返回，Device 指针数组可为空且无需 kernel。非空操作要求三个数组参数均有效。步长允许负数，0 和任务包负向用例覆盖的 `INT_MIN` 返回非法参数。
- 沿用仓内可寻址跨度保护。A 的最大访问范围是 `(n-1)*lda+m` 个元素，向量跨度是 `1+(L-1)*abs(inc)`，总输出元素数是 `batchCount*L_y`，各项按 `INT32_MAX` 限制。
- y 原地更新。调用方保证每批 y 与本批的 A/x 以及其他批次的 y 无重叠。Host 只校验指针数组自身。接口支持批次离散存储与矩阵列填充。实数 C 与 T 等价。广播、复数和跨批共享输出区域超出范围。
- 当 `alpha=0` 时只处理 y，当 `beta=0` 时直接以零初始化累加值。`alpha=0,beta=1` 时 Host 可直接返回。调用者在读取 Device 结果前同步 stream。

## 三、详细设计

### 3.1 Host 分派和分核

五个入口转发给同一模板，校验通过后使用 `GetAivCoreCount()` 获取可用 AIV 核数。使用 `trans=0` 表示 N，`trans=1` 表示 T/C。dtype 标识保持仓内编号 `0=HSH,1=S,2=HSS,3=TST,4=TSS`。维度、步长、标量和任务数随 `GemvBatchedTilingData` 按值下发，handle 的 stream 保持调用时异步执行顺序。

令 `Y` 为每个 batch 的逻辑输出长度，`U` 为单次处理的输出上限。N 路径默认 `U=128`。当 batch 数足以占满 AIV、输出不短且 x 维度较小时，Host 将 N 路径的输出块扩大到最多 1024 个元素。对于矩阵列跨度未对齐的单 batch 长输出，Host 按可用 AIV 数计算每核输出行数，再向上对齐到 8 个元素。T/C 路径默认 `U=16`。短归约在单位输出步长下可扩大到 128 个元素，在跨步输出且 `m≤4` 时可扩大到 512 个元素。其余转置路径按输出长度、输出步长和 batch 并行度选择输出块。对于长点积，Host 在 batch 并行度不足时缩小输出块，使更多 AIV 参与计算。

Host 计算 `tiles=ceil(Y/U)`、`tasks=batchCount*tiles` 和 `blocks=min(AIV,tasks)`。N 和短点积路径采用循环分配，核 `b` 依次处理任务号 `b,b+blocks,...`。T/C 的长点积路径将 `batchCount*Y` 个逻辑输出划分成 `blocks` 个近等长连续区间，再按 batch 边界和 `U` 切块。各核拥有互不重叠的输出元素，可独立完成乘加和原地写回。

### 3.2 数据分块与 UB

N 路径的输出块最多包含 1024 个元素，短转置归约最多包含 512 个输出。N 路径每次取至多 2048 个 x 元素，按列主序批量搬入矩阵的连续行片段。单个 A 槽位容纳 8192 个原始元素，每列按 32 字节对齐，再按可容纳的最大二次幂列数分组。T/C 的长点积路径按矩阵列组处理，FP32 输入每个归约段至多 1024 行，FP16/BF16 输入通常至多 2048 行。分组时按 32 字节对齐 FP32 输出起始地址。

单位步长采用连续 `DataCopyPad`。跨步读按至多 512 个逻辑元素分段，选择连续窗口或每元素占一个 32 字节块的多块搬运，随后用 `Gather` 恢复逻辑顺序。跨步写输出由 `Brcb` 扩展到搬运块，再用多块 `DataCopyPad` 写回。负步长先在 UB 中反序，绝对步长为 1 时连续写回。

设 `U` 为输出缓冲的元素容量。跨步暂存包括 `16,384 B` 的输入搬运窗口和 Gather 索引区，索引区正向访问保留 `512×4 B`，反向输出保留 `max(512,U)×4 B`。跨步输出按 Brcb 每次广播 8 个元素分配 `ceil(U/8)×8×32 B`。异步写回使用独立缓冲，同步写回在搬运窗口与广播所需容量中取较大值。

N 路径的 `U=1024` 反向输出异步写回时，矩阵双槽及转换缓冲合计 `65,536 B`，x 原始缓冲及 FP32 缓存合计 `16,384 B`，y 原始缓冲及累加缓冲合计 `8,448 B`。计算、规约、指针、跨步暂存及独立写回缓冲分别占 `32,768 B`、`4,096 B`、`512 B`、`20,480 B` 和 `32,768 B`，合计 `180,992 B`，低于单核 `192 KiB`。

FP16/BF16 转置长点积使用跨步 x 缓存和异步写回时，矩阵双槽及转换缓冲分别为 `65,536 B` 和 `32,768 B`，x 原始缓冲及 FP32 缓存分别为 `8,192 B` 和 `16,384 B`。其余 y 原始缓冲、累加、计算、规约、指针、跨步暂存及写回容量合计 `33,024 B`，合计 `155,904 B`。矩阵双槽交替预取，部分和在 UB 中合并。

### 3.3 Kernel 数据流

N 路径将 x 分段与 A 列组搬入 UB。对齐的二次幂列组用 `Brcb` 广播 x，并执行逐元素乘法和树形加法归约。其他列组逐列执行 `Axpy`。T/C 路径在本核处理同一个 batch 时复用能放入 UB 的 x。短点积批量执行 `WholeReduceSum`，长点积先规约每 64 个 FP32 乘积，再按列和归约段累加。短点积出现非有限中间值时，FP32 和 BF16 路径以 2 的幂缩放后重算，以处理有限输入产生的中间溢出。

大幅值 alpha 走补偿累加，输出前合并点积的乘积残差和加法残差。窄输出用 `Cast(..., CAST_RINT, ...)` 舍入。设备指针由 MTE2 搬入后交给 Scalar 读取，A 预取和输出写回通过核内事件协调。

```mermaid
flowchart TD
    A["校验参数并处理空操作"] --> B["按 dtype、trans 和 batch×输出 tile 下发"]
    B --> C["各 AIV 读取本任务的 y 指针"]
    C --> D["按 beta 初始化 FP32 累加块"]
    D --> E{"alpha 为零"}
    E -->|"是"| H["按 y 类型舍入并按步长写回"]
    E -->|"否"| F["读取本 batch 的 A/x 指针"]
    F --> G{"trans"}
    G -->|"N"| J["批量搬入列片段并累加当前输出块"]
    G -->|"T 或 C"| K["复用 x，按短归约或分段点积计算"]
    J --> H
    K --> H
```

### 3.4 与参考流程的差异

```mermaid
flowchart LR
    A["CBLAS 逐批函数调用"] -->|"一次下发覆盖所有 batch"| B["AIV 并行独占输出 tile"]
    C["标量访存与循环"] -->|"按连续列片段搬运"| D["GM 到 UB 分块及 Vector 计算"]
    E["T/C 逐列串行点积"] -->|"列组并行规约"| F["WholeReduceSum 和段结果累加"]
    G["beta 缩放及 alpha 退化"] -->|"保留未读语义"| H["空操作由 Host 返回，零标量由分派处理"]
```

小形状主要受启动和同步耗时影响。大形状需要关注矩阵读带宽、每组列数和跨步写回开销。N 路径的各输出块独立读取 x，T/C 路径在本核的同一 batch 内复用已缓存的 x。性能比较使用同一任务用例和同一计时口径。

## 四、特性交叉分析

| 交叉维度 | 验证点 |
| --- | --- |
| 五种 dtype × N/T/C | 相同的物理 lda，正确的 x/y 长度和输出类型 |
| 维数 × batch | 0/1/尾块/4095/4096，空、单批和多批，覆盖输出 16、128 与归约 64、2048 的路径边界 |
| lda × 步长 | 紧凑和列填充，正负单位和跨步，非法 lda、零步长及 INT_MIN |
| 标量 × y 更新 | 0、±1、±1e4 和随机值，beta=0 时旧 y 为 NaN，alpha=0 时 x 为 NaN |
| 数值 × 类型 | 均匀和正态数据，FP16/BF16 收窄，Inf/NaN 的传播和未读路径 |
| 架构 × 执行 | A2 与 A3 共用 dav-2201 源码，按 SoC 编译并在 handle stream 同步结果 |

## 五、可维可测分析

### 5.1 精度与性能标准

精度按所有 batch 的逻辑 y 元素比较。满足 `abs(actual-golden) <= atol+rtol*abs(golden)` 的比例至少为 0.99，并检查最大绝对误差上限。特殊值按同类非有限值匹配。任务书允许 CBLAS FP32 参考或 FP64 累加参考，使用的口径随报告注明。

| 输出 dtype | rtol 与 atol | 最大绝对误差规格 |
| --- | --- | --- |
| FP32 | 2^-13 | `max(0.01, 32 ULP)` |
| FP16 | 2^-10 | `max(0.1, 32 ULP)` |
| BF16 | 2^-7 | `max(0.1, 32 ULP)` |

每变体 1000 条精度用例和 200 条性能用例。正式性能在 910B3 使用 `msprof op`，排除 5 次预热后统计连续 10 次成功的 kernel Task Duration 算术平均。单条平均耗时须满足 `NPU_us/1000 <= 1.25*gpu_ms`，匹配键包括 dtype、trans、m、n、lda、incx、incy、batchCount、alpha、beta。GTest `aclrtEvent` 使用同样的 5+10 采样，打印 `[PERF] case=... avg_us=...` 供端到端交叉验证。正式结果以实测报告为准。

### 5.2 构建与兼容性

使用 `bash build.sh --ops=sgemv_batched --soc=ascend910b3` 编译，A3 选择相应 `ascend910_93` SoC。实现复用五个公开签名和共同类型，变体输出格式为 ND，A/x/y 保持各自物理布局。当前验收目标使用 CANN 9.1.0。已在共享 910B3 的 CANN 9.0.0 环境验证算子和 GTest 可构建，9.1.0 的精度与性能数据需按任务标准另行确认。

精度阈值与用例合同见[开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。文档结构遵循[社区任务模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
