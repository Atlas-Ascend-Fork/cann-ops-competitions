# 需求背景（required）

## 需求来源

本算子族来源于 CANN 社区任务 2026（ops-blas 开源仓社区任务：11月社区任务-gemvBatched类5算子开发（A2A3），代码仓：https://gitcode.com/cann/ops-blas ）。要求在昇腾 NPU（Atlas A2/A3 系列产品，含 Atlas 800T A2 / Atlas 800T A3）上，使用 Ascend C 编程语言并基于 ops-blas 开源仓句柄式 BLAS 工程框架，开发单精度/半精度/脑浮点实数批量矩阵-向量乘算子族 `aclblasSgemvBatched`、`aclblasHSHgemvBatched`、`aclblasHSSgemvBatched`、`aclblasTSTgemvBatched`、`aclblasTSSgemvBatched`（共 5 个 API），对标 cuBLAS 同名接口与 Netlib BLAS `sgemv` 计算语义，完成设计、开发与全量自验证后合入昇腾算子开源仓。

## 背景介绍

### aclblasSgemvBatched 类算子实现现状分析

在 ops-blas 开源仓中，5 个实数变体的公开接口声明已存在于 `include/cann_ops_blas.h`（L287–309），`blas/gemv_batched/arch35/` 已具备 Ascend 950PR（`arch35`）参考实现；而面向 Atlas A2/A3（`arch22` / `dav-2201`）的 `blas/gemv_batched/arch22/` 目录此前主要服务于复数 `cgemv_batched_*`，其现有实数原型代码在边界语义与性能上均存在缺口（如未按任务书将 `m=0`/`n=0`/`batchCount=0` 放行为合法 quick return、未支持 `ACLBLAS_OP_C`、未采用批量指针预取与向量化跨步搬运）。本任务需在 `blas/gemv_batched/arch22/` 下补齐并完善满足 A2/A3 任务书契约的 5 个实数变体 Host 与 Kernel 实现，并在 `test/gemv_batched/arch22/` 补齐 5 变体全量测试工程。

当前对标接口（cuBLAS 对应 `Batched` 接口）支持的能力与参数契约如下表所示：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄，携带 stream | aclblasHandle_t | aclblasHandle_t | 非空有效句柄 | 标量 |
| trans | op(A[i]) 的运算类型（N/T/C） | aclblasOperation_t | 枚举 | `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}`；实数下 C≡T | 标量（属性） |
| m | 矩阵 A[i] 的物理行数 | int | int | $m \ge 0$；$m=0$ 为合法 quick return | 标量（属性） |
| n | 矩阵 A[i] 的物理列数 | int | int | $n \ge 0$；$n=0$ 为合法 quick return | 标量（属性） |
| alpha | 标量乘数（Host 内存） | const float* | FLOAT32 | 非空指针；5 个变体均为 FLOAT32 | 标量（Host 内存） |
| Aarray | 输入矩阵 A[i] 的 Device 指针数组 | const TIn* const[] | S: float; HSH/HSS/TST/TSS: uint16_t (FP16/BF16) | 长度 batchCount；逐批 Device 首址 | 列主序，恒为 $\text{lda} \times n$（与 trans 无关） |
| lda | A[i] 的前导维度 | int | int | $\text{lda} \ge \max(1, m)$（与 trans 无关） | 标量（属性） |
| xarray | 输入向量 x[i] 的 Device 指针数组 | const TIn* const[] | 同 Aarray | 长度 batchCount；逐批 Device 首址 | `OP_N`: 长 $n$；`OP_T/C`: 长 $m$ |
| incx | x[i] 的元素步长 | int | int | 任意非零 `int`（不含 `0` 与 `INT32_MIN`）；负值反向遍历 | 标量（属性） |
| beta | 标量乘数（Host 内存） | const float* | FLOAT32 | 非空指针；$\beta=0$ 时不读 y 旧值 | 标量（Host 内存） |
| yarray | 输出向量 y[i] 的 Device 指针数组（原地覆写） | TOut* const[] | S/HSS/TSS: float; HSH/TST: uint16_t (FP16/BF16) | 长度 batchCount；逐批 Device 首址，各 y[i] 互不重叠 | `OP_N`: 长 $m$；`OP_T/C`: 长 $n$ |
| incy | y[i] 的元素步长 | int | int | 任意非零 `int`（不含 `0` 与 `INT32_MIN`）；负值反向遍历 | 标量（属性） |
| batchCount | 批量大小 | int | int | $\text{batchCount} \ge 0$；0 为合法 no-op | 标量（属性） |

计算公式：

$$y_i = \alpha \cdot \operatorname{op}(A_i) \cdot x_i + \beta \cdot y_i, \quad i = 0, 1, \dots, \text{batchCount}-1$$

### aclblasSgemvBatched 类算子功能分析

- **算子功能**：对 `batchCount` 个相互独立的矩阵-向量乘加系统并行执行实数 GEMV 运算，并将结果按 `incy` 步长原地写回各批次的 $y_i$ 向量区域。
- **输入**：`handle`、`trans`、`m`、`n`、`alpha`、`Aarray`、`lda`、`xarray`、`incx`、`beta`、`incy`、`batchCount`。
- **输出**：`yarray`（原地覆写输出，仅修改有效逻辑下标对应的元素，严格保护 `y[i]` 元素间隙及矩阵 padding 不被改写）。
- **支持数据类型**：
  - `aclblasSgemvBatched`（S）：A/x/y 均为 `FP32`；
  - `aclblasHSHgemvBatched`（HSH）：A/x/y 均为 `FP16`（接口以 `uint16_t` 承载）；
  - `aclblasHSSgemvBatched`（HSS）：A/x 为 `FP16`（`uint16_t`），y 为 `FP32`；
  - `aclblasTSTgemvBatched`（TST）：A/x/y 均为 `BF16`（接口以 `uint16_t` 承载）；
  - `aclblasTSSgemvBatched`（TSS）：A/x 为 `BF16`（`uint16_t`），y 为 `FP32`；
  - 5 个变体的 `alpha`、`beta` 均为 `FLOAT32` Host 标量。
- **支持广播**：不支持广播。各批次为独立操作数，shape 之间不进行广播。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3（`arch22` / `dav-2201`）上实现 `aclblasSgemvBatched` 类 5 个实数变体算子：

1. **接口与语义完全对齐**：5 个接口签名与参数顺序直接复用 `include/cann_ops_blas.h`，严格遵循 cuBLAS `cublas<t>gemvBatched`（A 物理维度恒为 $\text{lda} \times n$、$\text{lda} \ge \max(1, m)$，与 `trans` 无关；`OP_C` 与 `OP_T` 等价；`incx/incy < 0` 按 BLAS 反向遍历寻址）。
2. **混合精度高精度累加**：FP16/BF16 输入变体统一在片上无损提升为 FP32 完成点积与标量乘加，仅在最终写回 `HSH`/`TST` 的 $y$ 时按 IEEE 最近偶数舍入（RNE）收窄一次，避免中间截断误差累积。
3. **完备的参数校验与 Quick Return**：严格按契约顺序校验 `handle`、`trans`、`m/n`、`lda`、`incx/incy`（拦截 `0` 与 `INT32_MIN`）、`batchCount`、`alpha/beta` 与 `Aarray/xarray/yarray`；支持 $m=0 \lor n=0 \lor \text{batchCount}=0$ 及 $\alpha=0 \land \beta=1$ 的零开销 quick return；$\alpha=0$ 时不读 `A/x`，$\beta=0$ 时不读旧 `y`。
4. **跨步与非对齐写保护**：全覆盖 `incx/incy` 正负大步长与 `lda > m` padding 场景，确保 `y[i]` 非有效步长间隙（gap）与矩阵外侧字节不被污染。
5. **精度与性能双百达标**：精度全量 5000 条（5 变体 × 1000 条）满足对应输出 dtype 的生态算子开源精度标准；性能全量 1000 条（5 变体 × 200 条）在 Atlas 800T A2（910B3）上通过 `msprof op` 实测达到 GPU cuBLAS 基线的 $0.8\times$ 以上。

## 需求拆解

1. **Host 入口统一校验与分派模板**：抽象模板化校验与参数封装，按规定顺序返回 `HANDLE_IS_NULLPTR` / `INVALID_ENUM` / `INVALID_VALUE` / `SUCCESS`，支持 64 位安全索引跨度校验。
2. **纯 AIV 单次 Launch 多路径架构**：GEMV 算术强度 $< 0.5\text{ FLOP/Byte}$，属于典型延迟受限（Tiny）与 HBM 带宽受限（Large）算子；采用 `KERNEL_TYPE_AIV_ONLY` 架构，按规模与转置模式路由至专用路径，最大化 40 AIV 核 HBM 聚合读带宽并压低小矩阵启动延迟。
3. **批量指针预取与二维持久调度**：单核按连续 Batch/Tile 批量预取 Device 指针数组（每批最多 64 组），结合 `(batchIdx, outTileIdx)` 二维均衡切分，同时兼顾大 batch 小矩阵与小 batch 大矩阵的核间负载均衡与 $x_i$ 片上复用。
4. **单 `TBuf` 统一 UB 池与双缓冲流水**：规避 `dav-2201` 多 `TBuf` 地址重叠问题，仅申请 1 个连续 UB 池按 256B 对齐偏移复用；采用 Ping-Pong 双缓冲实现 MTE2 2D `DataCopyPad` 与 Vector 计算完全掩盖。
5. **免标量停顿跨步向量搬运引擎**：针对 `inc = +1`、`inc = -1`、`2 ≤ |inc| ≤ 16`、`|inc| > 16` 分级采用 `Gather`、连续窗口 DMA 与 32B 间隔槽 2D `DataCopyPad`，消除标量逐元素搬运停顿并精确保护间隙内存。
6. **5 变体构建注册与共享 GTest 框架**：扩展 CMake 构建路由以支持 5 个 `--ops=` 编译测试目标，构建兼容双 CSV 格式、FP64/Netlib 双域 Golden 核对及 `msprof op` 18-Launch 协议的统一测试套件。

# 详细设计（required）

## 算子分析

### 数学公式

对每个批次 $b \in [0, \text{batchCount}-1]$，令 $A_b(r, c) = \text{Aarray}[b][c \cdot \text{lda} + r]$（列主序存储）：

- **非转置模式（`trans == ACLBLAS_OP_N`）**：输出长度 $O = m$，归约长度 $K = n$：
  $$y_b[i] = \alpha \sum_{j=0}^{n-1} A_b(i, j) \cdot x_b[j] + \beta \cdot y_b[i], \quad i \in [0, m-1]$$
- **转置模式（`trans == ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`）**：输出长度 $O = n$，归约长度 $K = m$：
  $$y_b[j] = \alpha \sum_{i=0}^{m-1} A_b(i, j) \cdot x_b[i] + \beta \cdot y_b[j], \quad j \in [0, n-1]$$

**向量步长物理地址映射**：对于逻辑长度为 $L$、非零步长为 $\text{inc}$ 的向量 $v$，第 $k \in [0, L-1]$ 个逻辑元素的物理偏移（以元素计）为：

$$\text{phys}(k) = \begin{cases} k \cdot \text{inc}, & \text{inc} > 0 \\ (L - 1 - k) \cdot |\text{inc}|, & \text{inc} < 0 \end{cases}$$

### 支持数据类型

| 变体接口 | A / x 存储类型 | y 存储类型 | alpha / beta 类型 | 片上累加域 | 输出写回精度策略 |
| --- | --- | --- | --- | --- | --- |
| `aclblasSgemvBatched` | `FLOAT32` (`float`) | `FLOAT32` (`float`) | `FLOAT32` (Host) | `FP32` | 直接写出 `FP32` |
| `aclblasHSHgemvBatched` | `FLOAT16` (`uint16_t`) | `FLOAT16` (`uint16_t`) | `FLOAT32` (Host) | `FP32` | `RoundMode::CAST_RINT` 收窄为 `FP16` |
| `aclblasHSSgemvBatched` | `FLOAT16` (`uint16_t`) | `FLOAT32` (`float`) | `FLOAT32` (Host) | `FP32` | 直接写出 `FP32` |
| `aclblasTSTgemvBatched` | `BFLOAT16` (`uint16_t`) | `BFLOAT16` (`uint16_t`) | `FLOAT32` (Host) | `FP32` | `RoundMode::CAST_RINT` 收窄为 `BF16` |
| `aclblasTSSgemvBatched` | `BFLOAT16` (`uint16_t`) | `FLOAT32` (`float`) | `FLOAT32` (Host) | `FP32` | 直接写出 `FP32` |

### 支持形状

- $m \ge 0, n \ge 0, \text{batchCount} \ge 0$；全面覆盖极小尺寸（`1×1`、`3×5`）、非对齐素数/奇数尺寸（`17×31`、`127×129`、`511×513`、`2047×2049`、`4095×4095`）、细长矩形（`1×2049`、`4096×64`、`2×4095`）及大矩阵（`4096×4096`）；
- $\text{lda} \ge \max(1, m)$（含紧凑 $\text{lda}=m$ 及 $+1/+8/+64$ 等 padding）；
- $\text{incx}, \text{incy} \ne 0, \text{INT32\_MIN}$，涵盖单位步长、负步长（`-1, -2, -3`）及大步长（`7, 8, 16, 31, 64, 127`）；
- $\text{batchCount}$ 涵盖 `0, 1, 2, 3, 8, 16, 64, 256` 等规模。

## 算子实现

### 实现方案

采用 **`KERNEL_TYPE_AIV_ONLY` 多路径自适应路由** 方案，单次非平凡调用优先以单 Kernel Launch、0 额外 GM 申请完成（小 batch 大归约 `N` 场景按需复用 Handle 预分配 Workspace 做 Split-K 归约）：

1. **`ALPHA_ZERO` 路径**（$\alpha = 0$ 且 $\beta \ne 1$）：跳过 `Aarray/xarray` 指针预取与矩阵搬运，仅按输出维度切分执行 $y_i = \beta \cdot y_i$（$\beta = 0$ 时直接写 `0.0f` 不读旧 `y`）。
2. **`TINY_FAST` 路径**（$\alpha \ne 0$，$m, n \le 64$ 且总矩阵量 $\text{batchCount} \cdot m \cdot n \cdot \text{sizeof}(T_{\text{in}}) \le 128\text{ KiB}$）：每 Batch 对应 1 个 Task，将整块 $A_b$ 一次搬入 UB 完成计算与写回，消除分块循环与多级同步开销，压低极小形状延迟。
3. **`SMALL_M_N` 路径**（`OP_N`，非 `TINY_FAST` 且 $m \le 32, n \ge 64$）：每 Batch 1 个 Task，沿归约轴 $n$ 批量连续搬入多列（紧凑时 1D 大块读取），在 UB 内通过 `Gather` 按行抽取后与 $x$ 相乘并归约。
4. **`NORMAL_N` 路径**（其余 `OP_N` 场景）：沿输出轴 $m$ 切分输出行块，沿归约轴 $n$ 分段加载 $x$ 至 UB 复用，对 $A$ 列块采用 Ping-Pong 2D 搬运 + 双累加器（`acc0, acc1`）交替 `Axpy` 隐藏 `PIPE_V` 流水延迟。对于小 batch 大归约尺寸（如 $n \ge 128$ 且有足够可用核心），可借用 Handle 自有 Workspace 做分段 K 归约并通过 `SyncAll` 后按固定顺序累加，消除单核归约长尾。
5. **`SMALL_M_T` 路径**（`OP_T/OP_C`，非 `TINY_FAST` 且 $m \le 32, n \ge 64$）：沿输出轴 $n$ 切分列块，一次加载短向量 $x[0..m-1]$ 常驻 UB，逐行广播并 `Axpy` 累加至对应输出列槽。
6. **`TRANS_T` 路径**（其余 `OP_T/OP_C` 场景）：沿输出轴 $n$ 切分列块，`x` 分段加载后跨多列复用；对 $A$ 的列块采取 Ping-Pong 2D 连续搬运，FP16/BF16 转 FP32 后与 `x32` 逐元素 `Mul`，通过两级 `WholeReduceSum`（配合二分 `Add` 折叠）并行归约多列点积。

#### 3.2.1 host侧设计：

##### tiling策略：
- **输入与输出长度确定**：`outDim = (trans == ACLBLAS_OP_N) ? m : n`，`kDim = (trans == ACLBLAS_OP_N) ? n : m`。
- **路径选择与 Tile 上界**：
  - `TRANS_T` 路径：若点积维 $m \le 4096$，单次缓存全部 $x$ 向量；`rowStride = (mTile <= 512) ? AlignUp(mTile, 64) : AlignUp(mTile, 512)`，`colTile = clamp(16384 / rowStride, 1, 256)`，保证每次 2D DMA 搬入块与 UB 槽容量严格匹配。
  - `NORMAL_N` 路径：`mSubTile = min(outTileLen, 2048)`，`mAligned = AlignUp(mSubTile, 64)`（小输出按 32B 对齐），`kTile = clamp(18432 / mAligned, 1, 256)`。

##### 任务均分：
- 查询物理 AIV 核心数 $C = \text{GetAivCoreCount}()$（Atlas 800T A2 上为 40）。
- `TINY_FAST` 与 `SMALL_M_N` 取每 Batch 1 个 Task（$Q = 1$）；其余路径取输出切分粒度 $h$（N/ALPHA_ZERO 取 $h=64$，T/C 取 $h=16$），令 $G = \lceil O / h \rceil$：
  - 若 $\text{batchCount} \ge C$：优先取 $Q = 1$（仅当存在明显波次尾部闲置且总数据量大时适当拆分 $Q = 2$）；
  - 若 $\text{batchCount} < C$：按目标总任务数铺满 $C$ 个核心（$Q = \min(G, \lceil C / \text{batchCount} \rceil)$ 或凑满 $C$ 的整数倍）。
- 总任务数 $T = \text{batchCount} \times Q$，实际启动核数 $U = \min(C, T)$。
- 每个核按连续区间分配任务（核 $u$ 领 $\lfloor T/U \rfloor + (u < T\%U ? 1 : 0)$ 个连续 Task），保证同核连续处理同一 Batch 时复用已解析指针与 `x` 缓存，同时确保相邻核在同一 Batch 内的写回区间按 32B 块边界对齐，绝不出现跨核踩踏同一个 32B 写入块的情况。

##### 批量搬运：
- **批量指针预取**：在核内按每批最多 64 个 Batch，通过 3 条 `DataCopyPad` 配合 1 次 `MTE2_S` 事件同步将 `Aarray`、`xarray`、`yarray` 的设备指针一次性批量读入 UB 指针槽，消除每 Batch 串行读指针的停顿开销。
- **跨步向量搬运分级**：
  - `inc == 1`：单次 1D `DataCopyPad` 连续读写；
  - `inc == -1`：单次 1D `DataCopyPad` 读入升序区间，UB 内通过 `Gather` 偏移表一次翻转（写回前同样 UB 翻转后 1D 写出）；
  - `2 ≤ |inc| ≤ 16`：读入时按物理窗口跨度 `span` 一次性 1D `DataCopyPad` 读入暂存窗再用 `Gather` 抽取；写回时将结果填入 32B 间隔槽后以 **2D `DataCopyPad`**（`blockLen = sizeof(TOut)`, `srcStride = 0`, `dstStride = (|incy|-1)*sizeof(TOut)`）精确写回 GM，不触碰间隙内存；
  - `|inc| > 16`：读写均采用基于 32B 间隔槽的 2D `DataCopyPad`（超出硬件步长字段上限时退化为 64 位地址分段搬运）。

##### 1. 分核策略：
- **满核优先与按需收缩**：仅当总任务数 $T \ge C$ 时启动全部物理 AIV 核心；当 $T < C$ 时启动 $U = T$ 个核心，避免多余空闲核带来额外的调度与缓存抖动开销。
- **K 分段归约策略（仅限小 batch 大矩阵）**：当 `OP_N` 且 $\text{batchCount} \le 8$、$K$ 较长且 Handle 自有默认 Workspace（非用户小外部缓冲）足够容纳 `batchCount × splitK × AlignUp(m, 16) × 4` 字节时，沿 K 维切分由多核并行累加 FP32 部分和到 Workspace，经 `SyncAll` 后由核按固定树形顺序合并写回；若 Workspace 不足则自动回退到单核或输出轴切分路径。

##### 2. 数据分块和内存优化策略：
1. **单 `TBuf` 统一 UB 内存池（规避 `DAV_2201` 多 `TBuf` 重叠缺陷）**：全 Kernel 仅向 `TPipe` 申请 1 个 `188 KiB`（低于物理 `192 KiB` 上限）的连续 `TBuf<TPosition::VECCALC>` 池，所有逻辑子区（指针批 `2 KB`、Gather 偏移表 `2 KB`、跨步暂存 `4 KB`、`x32Buf` `8~16 KB`、Ping/Pong 双槽 `aBuf0/aBuf1`、`reduceBuf` / `acc0, acc1`、`yTmpBuf`）均采用 256B 对齐的显式字节偏移切分与生命周期复用。
2. **Ping-Pong 双槽异步掩盖**：`aBuf0` 与 `aBuf1` 严格绑定独立的 `MTE2→V` 和 `V→MTE2` 成对事件信号量，实现下一分块搬运与当前分块向量计算完全并行。
3. **零运行期内存申请**：全程不调用同步 `aclrtMalloc` / `aclrtFree`；常规路径 0 Workspace，Split-K 路径直接复用 Handle 内置预分配 Workspace。

##### 3. tilingkey规划策略：
Tiling 结构体 `GemvBatchedTilingData` 保持 8 字节对齐（72 字节以内）经由内核参数寄存器直接值传递，零 GM Tiling 读取开销；通过字段组合在核内静态分发：
- `dtype`（`0=HSH, 1=S, 2=HSS, 3=TST, 4=TSS`）模板实例化不同输入/输出数据类型组合；
- `pathMode`（`ALPHA_ZERO`, `TINY_FAST`, `SMALL_M_N`, `NORMAL_N`, `SMALL_M_T`, `TRANS_T`）分派对应执行引擎；
- `trans`（`0=OP_N`, `1=OP_T/OP_C`）合并 `OP_T` 与 `OP_C`。

##### 数据检测：
Host 侧严格遵循以下确定性校验顺序（不在 Host 扫描 Device 张量内容）：
1. `handle == nullptr` → 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `trans ∉ {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` → 返回 `ACLBLAS_STATUS_INVALID_ENUM`；
3. `m < 0 || n < 0` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `lda < std::max(1, m)` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`（恒与 `trans` 无关）；
5. `incx == 0 || incy == 0 || incx == INT32_MIN || incy == INT32_MIN` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. `batchCount < 0` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
7. `alpha == nullptr || beta == nullptr` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
8. **空维/空批 Quick Return**：`batchCount == 0 || m == 0 || n == 0` → 直接返回 `ACLBLAS_STATUS_SUCCESS`（允许 `Aarray/xarray/yarray` 为 `nullptr`，不启动 Kernel）；
9. `Aarray == nullptr || xarray == nullptr || yarray == nullptr` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
10. **纯恒等 Quick Return**：`*alpha == 0.0f && *beta == 1.0f` → 直接返回 `ACLBLAS_STATUS_SUCCESS`（不启动 Kernel，保持 `y` 原值）；
11. **64 位跨度安全检查**：用 64 位整型校验矩阵与步长向量物理跨度是否超出硬件寻址表达范围，防止整型溢出。

#### 3.2.2 kernel侧设计：

Kernel 执行分为 **Init** 与 **Process** 两个阶段：

1. **Init 阶段**：
   - 解析 `GemvBatchedTilingData`，计算当前 `blockIdx` 领取的连续 Task 区间 `[taskBegin, taskEnd)`；若核空载则直接返回；
   - 将 Device 指针数组基地址 `aArray`, `xArray`, `yArray` 绑定至 `GlobalTensor<uint64_t>`；
   - 初始化唯一连续 `ubPool_`（`188 KiB`），并按路径模式初始化各 256B 对齐的子缓冲视图与 `Gather` 偏移表（倒序索引/行抽取索引）。
2. **Process 阶段**：
   - **批量指针预取与 Batch 切换缓存**：每次批量预取最多 64 个 Task 对应的设备指针；当相邻 Task 属于同一 `batchIdx` 时复用上次地址与 $x$ 向量数据；
   - **CopyIn（搬入）与升精**：
     - 跨步读入 $x_b$ 至 `x32Buf`（FP16/BF16 通过 `Cast(..., RoundMode::CAST_NONE)` 无损转为 FP32；`OP_N` 时预乘 `alpha`）；
     - 矩阵 $A_b$ 通过 Ping-Pong 2D `DataCopyPad` 搬入 `aBuf[slot]`，其中 `DataCopyPad` 尾部补零区间配合掩码或 `Duplicate(0.0f)` 保证严格为 `0.0f`（防止 `0 * NaN` 污染）；
   - **Compute（计算）**：
     - `NORMAL_N`：将块内列向量的 `x` 系数依次通过双累加器 `Axpy(acc0, aCol, xVal)` / `Axpy(acc1, ...)` 交替累加，循环末合并 `acc0 + acc1`；
     - `TRANS_T`：对每列执行向量乘法 `Mul(a32, x32)`，采用两级 `WholeReduceSum` 配合二分 `Add` 折叠归约得到多列点积结果至 `yAccBuf`；
     - `SMALL_M_N` / `SMALL_M_T`：利用 UB `Gather` 实现多列/多行打包紧凑计算，减少循环发射次数；
     - `ALPHA_ZERO`：完全跳过 `A`/`x` 读取与点积计算；
   - **CopyOut（合并与搬出）**：
     - 非 `ALPHA_ZERO` 路径计算 `alpha * acc`；仅当 `beta != 0.0f` 时调用 `LoadStridedVec` 读入旧 `y_b`（`beta == 0.0f` 绝不读取旧 `y_b`，防 `NaN` 传播），升精为 FP32 后计算 `alpha * acc + beta * yOld`；
     - 对于 `HSH`/`TST` 变体，使用 `Cast(..., RoundMode::CAST_RINT)` 按 IEEE RNE 舍入至目标 16 位格式；
     - 调用 `StoreStridedVec` 按逻辑有效长度精确落盘（跨步模式使用 32B 间隔槽 2D `DataCopyPad` 写回，严格不破坏相邻元素间隙）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I A2 推理产品 | √ |
| Atlas 800T A2 训练产品 | √ |
| Atlas 800T A3 训练产品 | √ |

实现严格基于 `arch22`（`dav-2201`）Ascend C AIV 向量与 DMA 指令集，使用 CANN 9.1.0 工具链编译验证。

## 算子约束限制

1. **数据类型**：仅支持 S（FP32）、HSH（FP16）、HSS（FP16→FP32）、TST（BF16）、TSS（BF16→FP32）5 个实数变体；复数批量矩阵向量乘由 `aclblasCgemvBatched` 承担。
2. **数据布局与前导维**：矩阵 $A[i]$ 按列主序存储，物理尺寸恒为 $\text{lda} \times n$，要求 $\text{lda} \ge \max(1, m)$（与 `trans` 无关）。
3. **步长与对齐**：`incx` 与 `incy` 不得为 `0` 或 `INT32_MIN`；负步长严格按照 BLAS 反向遍历规则寻址。
4. **内存重叠**：调用方须保证各批次 $y[i]$ 存储区域互不重叠，且 $y[i]$ 可写区域与 $A[j]/x[j]$ 可读区域不重叠；各批次共享 $m, n, \text{lda}, \text{incx}, \text{incy}, \text{trans}, \alpha, \beta$。
5. **异步执行**：Kernel 在 `handle` 绑定的 stream 上异步下发，Host 读取结果前须先同步 stream。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 1. CPU Golden 由 Netlib BLAS 3.10.0 `cblas_sgemv` 逐 batch 生成（输入按变体 dtype 量化），或按任务书 §3.2 第 4 条采用 float64 累加后按输出 dtype RNE 收窄。<br>2. 按输出 `y` 的 dtype 适用 Mixed Tolerance：<br>   - `S / HSS / TSS`（FLOAT32）：$rtol = atol = 2^{-13} \approx 1.2\times 10^{-4}$，$\text{max\_abs} \le \max(10^{-2}, 32 \times \text{ULP})$；<br>   - `HSH`（FLOAT16）：$rtol = atol = 2^{-10} \approx 9.77\times 10^{-4}$，$\text{max\_abs} \le \max(10^{-1}, 32 \times \text{ULP})$；<br>   - `TST`（BFLOAT16）：$rtol = atol = 2^{-7} \approx 7.8\times 10^{-3}$，$\text{max\_abs} \le \max(10^{-1}, 32 \times \text{ULP})$。<br>3. 所有变体要求 $\text{matched\_ratio} \ge 0.99$；全量 5000 条精度用例（5 变体 × 1000 条）100% PASS，跨步间隙与 padding 哨兵零污染。 | 生态算子开源精度标准（mixed_tolerance_standard）、任务书 §3.2 |
| 性能标准 | 1. 测试设备：Atlas 800T A2 (910B3)。<br>2. 正式判据（kernel-to-kernel）：采用 `msprof op` 真实采集单算子底层 kernel `Task Duration(us)`，warmup 5 次后取连续 10 次有效样本算术平均（兼容统一 18-Launch 协议）。<br>3. 达标门槛：NPU 平均单次耗时 $\le \text{gpu\_ms} / 0.8$（达到 A100 cuBLAS 同参数 `gpu_baseline.csv` 标杆的 $0.8\times$ 以上性能）。<br>4. 辅助判据：GTest 端到端 `[PERF] avg_us`（`aclrtEvent` 计时）仅作交叉验证，不替代正式判据。全量 1000 条性能用例（5 变体 × 200 条）100% 达标。 | 任务书 §3.3、`gpu_baseline.csv` |

## 兼容性分析

1. **接口与头文件兼容**：5 个算子原型完全复用 `include/cann_ops_blas.h`（L287–309）既有声明与 `include/cann_ops_blas_common.h` 状态码定义，无需改动公共头文件，100% 保持 ABI 向后兼容。
2. **已有回归兼容**：在修正 A2/A3 边界语义（合法空操作与 `OP_C`）的同时，完整保留原仓 `gemv_batched_test` 既有回归用例的编译与运行兼容性，且实现与同目录下的复数 `cgemv_batched_*` 及 `arch35` 版本完全解耦共存。
3. **构建系统兼容**：通过在 `blas/CMakeLists.txt`、`cmake/test.cmake`、`test/CMakeLists.txt` 及 `test/gemv_batched/CMakeLists.txt` 注册 5 个变体的 `--ops=` 别名路由，无缝兼容社区标准单算子构建与 ATK 验证脚本调用入口。
