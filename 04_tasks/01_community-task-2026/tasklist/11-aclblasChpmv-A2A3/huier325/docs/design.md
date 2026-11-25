# 需求背景（required）

## 需求来源

本设计文档来源于昇腾 CANN 开源社区任务 **“11月社区任务-aclblasChpmv算子开发（A2/A3）”**（任务 ID：`c596a05d90b9471683e41fcf4d026ae5`），目标仓为 `cann/ops-blas`（`https://gitcode.com/cann/ops-blas`）。

## 背景介绍

### aclblasChpmv算子实现与优化背景

在量子物理模拟、雷达阵列信号协方差处理、无线通信 MIMO 信道估计及复数有限元结构力学等科学计算场景中，大量线性代数核心子程序围绕 **Hermitian 矩阵（共轭对称矩阵，$A = A^H$）** 展开。由于 $n \times n$ Hermitian 矩阵上下三角互为共轭转置且主对角线元素必为实数，采用按列无间隙的压缩存储（Packed Storage，长度仅为 $n(n+1)/2$）可将矩阵内存占用与 HBM 访存带宽需求降低近 50%。

`aclblasChpmv` 是 BLAS Level-2 核心接口之一，对标 NVIDIA cuBLAS `cublasChpmv` 与 Netlib 标准 `cblas_chpmv` 接口，在昇腾 Atlas A2 / A3 系列硬件（对应 `ops-blas` 仓 `arch22` 架构目录）上基于 Ascend C 语言实现单精度复数（`COMPLEX64`）Hermitian 压缩矩阵-向量乘加计算。

### 仓内现状分析

当前 `cann/ops-blas` 仓中相关实现与接口现状如下：

- 头文件声明：`include/cann_ops_blas.h` 与 `include/cann_ops_blas_common.h` 已提供 `aclblasChpmv` 函数原型、`aclblasComplex`（`{float real; float imag;}`）及 `aclblasFillMode_t`、`aclblasStatus_t` 等公共类型定义。
- 参考架构：`blas/spmv/arch35/` 已具备面向 `arch35` 的实现；`blas/spmv/arch22/` 存量初版采用 `TILE_ROWS = 1024` 的两阶段（`accum` + `reduce` 双 kernel）串行搬运方案，存在以下待补齐与优化项：
  1. **参数校验顺序对齐**：`n = 0` 提前返回未先完成 `uplo`、`incx`/`incy`、`alpha`/`beta` 结构参数校验，且误拦截合法负步长极值 `INT_MIN`；
  2. **标量指针位置支持**：未支持 Device 内存位置的 `alpha`、`beta` 标量指针流序读取；
  3. **访存与流水性能瓶颈**：`n > 1024` 时按行分片反复扫描压缩矩阵 `AP`，内层循环存在逐列小粒度同步搬运与频繁 `V_S` / `S_V` 标量-向量流水停顿，且双 kernel 发射在小规模 `n` 下引入额外调度开销；
  4. **测试与自验对齐**：需适配任务书 1,200 条标准 CSV 用例集（含 `RANDOM_GAUSS_0_1`、`RANDOM_DIAG_IMAG` 分布标记及 `[PERF] case=... avg_us=...` 性能输出协议）。

| 参数名 | 输入/输出/属性 | 参数含义 | 数据类型 | dtype | 格式 | 形状 / 长度约束 |
| --- | --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | `ops-blas` 上下文句柄（绑定 `aclrtStream` 与 workspace） | `aclblasHandle_t` | - | - | 非空有效句柄 |
| `uplo` | 属性 | `AP` 存储上三角或下三角模式 | `aclblasFillMode_t` | INT | - | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` |
| `n` | 属性 | Hermitian 方阵 $A$ 的阶数 | `int` | INT | - | $n \ge 0$（$n=0$ 为合法 no-op） |
| `alpha` | 输入 | 复数标量乘子 $\alpha$（支持 Host / Device 指针） | `const aclblasComplex*` | COMPLEX64 | - | 单个复数标量 |
| `AP` | 输入 | 列主序无间隙压缩存储的 Hermitian 矩阵 | `const aclblasComplex*` | COMPLEX64 | ND | $n(n+1)/2$ 个复数元素 |
| `x` | 输入 | 输入复数向量 | `const aclblasComplex*` | COMPLEX64 | ND | 逻辑长度 $n$，物理跨度 $\ge 1+(n-1)|\text{incx}|$ |
| `incx` | 属性 | 向量 $x$ 的访问步长（支持正负非零步长） | `int` | INT | - | $\text{incx} \ne 0$ |
| `beta` | 输入 | 复数标量乘子 $\beta$（支持 Host / Device 指针） | `const aclblasComplex*` | COMPLEX64 | - | 单个复数标量 |
| `y` | 输入/输出 | 原地更新的输出复数向量 | `aclblasComplex*` | COMPLEX64 | ND | 逻辑长度 $n$，物理跨度 $\ge 1+(n-1)|\text{incy}|$ |
| `incy` | 属性 | 向量 $y$ 的访问步长（支持正负非零步长） | `int` | INT | - | $\text{incy} \ne 0$ |

### aclblasChpmv算子功能分析

- 计算表达式：$y \leftarrow \alpha A x + \beta y$，其中 $A \in \mathbb{C}^{n \times n}$ 满足 $A = A^H$。
- 对角线实数约束：Hermitian 矩阵主对角元素虚部理论上为 $0$，算子实现严格**仅读取对角线实部**，不引用对角线虚部内存单元，避免输入对角虚部含 `NaN`/`Inf` 时通过 `0 * NaN` 污染结果。
- 步长空隙保护：当 $|\text{incy}| > 1$ 时仅原地覆写逻辑下标对应的步长采样点，步长间隙内存保持原值逐字节不变。

# 需求分析（required）

## 需求描述

基于 `cann/ops-blas` 开源仓工程框架，在 Atlas A2 / A3 系列产品（`arch22`）上使用 Ascend C 开发句柄式直调算子 `aclblasChpmv`（代码位于 `blas/spmv/arch22/`，测试位于 `test/spmv/chpmv/arch22/`），功能与错误检查语义对齐 cuBLAS `cublasChpmv` 及 Netlib `cblas_chpmv`，并通过 1,000 条精度用例与 200 条性能用例（`msprof op` 单算子耗时 $\le 1.25 \times \text{gpu\_ms}$，即达到 GPU 标杆 $0.8\times$ 以上性能）。

## 需求拆解

1. **接口与错误码对齐**：严格遵循 `handle -> n < 0 -> uplo -> incx/incy != 0 -> alpha/beta 非空 -> n == 0 quick return -> AP/x/y 非空 -> alpha==(0,0)&&beta==(1,0) quick return` 校验顺序。
2. **Host / Device 标量双模式支持**：Host 标量直接随 Tiling 下发；Device 标量在 `handle->stream` 上由核函数按流序读取，避免 Host 侧同步阻塞并保持流内异步语义。
3. **Hermitian 压缩索引与全步长支持**：完整支持 `ACLBLAS_UPPER` / `ACLBLAS_LOWER` 压缩寻址及任意非零正/负 `incx`、`incy`（64 位偏移防溢出，兼容 `INT_MIN`/`INT_MAX`）。
4. **特殊标量短路与防污染语义**：`beta = (0,0)` 时不读取 `y` 旧值（兼容旧 `y` 含 `NaN`/`Inf`）；`alpha = (0,0)` 时不读取 `AP` 与 `x`。
5. **单核/多核自适应单 Kernel 高性能架构**：通过 UB 常驻向量 + 双缓冲流式读取 `AP` + 蛇形均衡分列 + 核内 `SyncAll()` 归约，消除二次 Kernel 启动与冗余 GM 读写。
6. **完备自验与交付**：对齐任务书 1,000 条精度用例（`atol = rtol = 2^{-13}`，匹配率 $\ge 0.99$）与 200 条 `msprof op` 性能用例。

# 详细设计（required）

## 算子分析

### 数学公式

对于阶数 $n$ 的 Hermitian 矩阵 $A \in \mathbb{C}^{n \times n}$（$A_{ij} = \overline{A_{ji}}$，$A_{ii} = \operatorname{Re}(A_{ii}) \in \mathbb{R}$），算子计算：

$$y_{\text{phys}(i)} \leftarrow \alpha \sum_{j=0}^{n-1} A_{ij} x_{\text{phys}(j)} + \beta \, y_{\text{phys}(i)}, \quad i = 0, 1, \dots, n-1$$

其中负步长与正步长的物理元素下标映射统一采用 64 位无溢出公式：

$$\text{phys}(i, n, \text{inc}) = \begin{cases} i \cdot \text{inc}, & \text{inc} > 0 \\ (n - 1 - i) \cdot (-\text{inc}), & \text{inc} < 0 \end{cases}$$

压缩数组 `AP` 按列主序无间隙存放（0 基下标）：

- **`uplo = ACLBLAS_UPPER`**（存上三角 $0 \le i \le j < n$）：

  $$\text{AP}\!\left[\frac{j(j+1)}{2} + i\right] = A_{ij}$$

  按列 $j$ 遍历时：主对角贡献 $z_j \mathrel{+}= \operatorname{Re}(A_{jj}) x_j$；严格上三角段 $i \in [0, j)$ 同时产生纵向向量乘加（AXPY：$z_i \mathrel{+}= A_{ij} x_j$）与横向共轭内积（DOT：$z_j \mathrel{+}= \sum_{i=0}^{j-1} \overline{A_{ij}} x_i$）。

- **`uplo = ACLBLAS_LOWER`**（存下三角 $0 \le j \le i < n$）：

  $$\text{AP}\!\left[\frac{j(2n - j + 1)}{2} + (i - j)\right] = A_{ij}$$

  按列 $j$ 遍历时：主对角位于段首 $\text{colBase}(j)$，贡献 $z_j \mathrel{+}= \operatorname{Re}(A_{jj}) x_j$；严格下三角段 $i \in (j, n)$ 同时产生纵向 AXPY（$z_i \mathrel{+}= A_{ij} x_j$）与横向共轭 DOT（$z_j \mathrel{+}= \sum_{i=j+1}^{n-1} \overline{A_{ij}} x_i$）。

复数乘法的实虚部分量公式（设 $a = a_r + \mathrm{i} a_i,\; b = b_r + \mathrm{i} b_i$）：

- 普通复乘 $a \cdot b = (a_r b_r - a_i b_i) + \mathrm{i}(a_r b_i + a_i b_r)$；
- 共轭复乘 $\overline{a} \cdot b = (a_r b_r + a_i b_i) + \mathrm{i}(a_r b_i - a_i b_r)$。

### 支持数据类型

- `AP`、`x`、`y`、`alpha`、`beta`：`COMPLEX64`（`aclblasComplex`，实部与虚部均为 IEEE-754 `FLOAT32`）。

### 支持形状

- 不支持广播；
- 方阵阶数 $n \ge 0$（$n = 0$ 为合法空操作；性能与常规规模覆盖 $n \in [1, 8192]$，同时通过分块回退支持 $n > 8192$ 任意规模）；
- 步长 $\text{incx} \ne 0,\; \text{incy} \ne 0$（正负单位步长、小步长及大跨步均支持）。

## 算子实现

### 实现方案

整体架构采用 **“单 Kernel 发射 + UB 全常驻向量（$n \le 8192$）+ MTE2 双缓冲流式读取 `AP` + 核内 `SyncAll()` 归约”** 方案，流程如下：

```
aclblasChpmv (Host)
  ├─ 1. 严格序参数校验: handle -> n<0 -> uplo -> incx/incy -> alpha/beta -> (n==0 返回 SUCCESS) -> AP/x/y
  ├─ 2. 指针位置探测 (aclrtPointerGetAttributes):
  │     ├─ Host 标量: 直接读取，命中 alpha=(0,0)&&beta=(1,0) 直接 SUCCESS 返回
  │     └─ Device 标量: 记录标志位交由 Kernel 在绑定 stream 上按流序读取
  ├─ 3. 分核与 Workspace 规划 (EnsureDefaultWorkspace):
  │     ├─ 小规模 (n <= 96) 或 Scale-Only: 单核/多核直出，0 Workspace，无 SyncAll
  │     ├─ 中大规模 (96 < n <= 8192): 满核 AIV，x 与局部累加器 acc 全量常驻 UB (<=128 KiB)
  │     └─ 超大规模 (n > 8192): 行分片 (TILE_ROWS = 4096) 多轮累加回退路径
  └─ 4. 单次发射 chpmv_arch22_kernel<<<usedCores, nullptr, handle->stream>>>
        ├─ Phase A: 流序读取 Device 标量(若启用)；处理 alpha==0 退化分支
        ├─ Phase B: 加载 x 向量至 UB 实虚分离平面 (xRe, xIm)，双缓冲流式遍历本核蛇形分配列:
        │           32B 对齐掩码化处理 LOWER 首部偏移 -> Gather 拆实虚 -> SIMD AXPY + SIMD 共轭 DOT
        ├─ Phase C: 写出本核局部和至 L2 常驻 Workspace，执行硬件 SyncAll() 核间同步
        └─ Phase D: 按输出行分核归约局部和，融合计算 y = alpha * z + beta * y (beta=0 跳过读 y) 并写回
```

#### 3.2.1 host侧设计：

##### 1. 参数校验与标量探测策略：

严格按照任务书 §2.5 与 cuBLAS 一致的顺序执行入口检查：
1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
3. `uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER` 返回 `ACLBLAS_STATUS_INVALID_ENUM`；
4. `incx == 0 || incy == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`（使用 `int64_t` 承接步长计算，合法支持 `INT_MIN`/`INT_MAX`）；
5. `alpha == nullptr || beta == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. 若 `n == 0`，至此全部结构参数合法，直接返回 `ACLBLAS_STATUS_SUCCESS`（豁免 `AP`/`x`/`y` 空指针检查）；
7. `ap == nullptr || x == nullptr || y == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。

随后通过 `CheckPtrLocation`（`aclrtPointerGetAttributes`）判断 `alpha` 与 `beta` 所在内存域：若均在 Host 侧则直接解引用并在 $\alpha=(0,0) \land \beta=(1,0)$ 时零开销快速返回；若任一位于 Device 侧，则将指针直接传入 Kernel 在 `handle->stream` 上读取，无需 `aclrtSynchronizeStream` 阻塞主机线程。

##### 2. 分核策略：

通过 `GetAivCoreCount()` 动态获取当前设备 AIV 核数 $C_{\text{aiv}}$（Atlas 800T A2 为 48 AIV 或 40 AIV）：
- **小规模单核直通路径（$n \le 96$）**：`AP` 大小不超过 $96 \times 97 / 2 \times 8 = 37.2\text{ KiB}$，单核即可将 `x`、`acc` 与 `AP` 全部装入 UB 完成计算，省去多核同步与 Workspace 读写，`usedCores = 1`，单次 Kernel 耗时控制在 $2\sim 3\,\mu\text{s}$。
- **中大规模满核负载均衡路径（$n > 96$）**：启用 `usedCores = min(CeilDiv(n, 4), aivCoreNum)`。由于压缩三角矩阵第 $j$ 列长度为 $j+1$（`UPPER`）或 $n-j$（`LOWER`），直接连续切分列会导致首尾核计算量相差悬殊；本方案采用 **双向蛇形配对分列（Snake Cyclic Mapping）**：

  $$j(\text{core}, k) = k \cdot P + \begin{cases} \text{core}, & k \bmod 2 = 0 \\ P - 1 - \text{core}, & k \bmod 2 = 1 \end{cases}$$

  每相邻两轮迭代中，任意核处理的列长度之和恒等于 $(2k+2)P + 1$（`UPPER`），实现核间任务量 $>99.5\%$ 均衡。

##### 3. 数据分块和内存优化策略：

Atlas A2 / A3 单 AIV 核具备 $192\text{ KiB}$ UB 容量，采用双策略自适应分块兼顾中小规模极低时延与超大规模严格顺序精度：
- **中小规模 UB 全常驻单遍流式策略（$n \le 3072$，`CHPMV_STRATEGY_RESIDENT_SNAKE`）**：取 $n_{\text{pad}} = \operatorname{CeilAlign}(n, 64) \le 3072$。在 UB 中常驻 `xRe[nPad]`、`xIm[nPad]` 及本核局部累加器 `accRe[nPad]`、`accIm[nPad]`，配置 `CHUNK = 1024` 复数元素（每块 $8\text{ KiB}$）的双缓冲槽位 `apSlot[0..1]`。整个矩阵乘法期间 **`x` 仅从 GM 加载 1 次，`AP` 仅单遍流式读取 1 次**；`n <= 96` 时单核零 Workspace、零 `SyncAll()` 直接写回，`96 < n <= 3072` 时经芯片 $192\text{ MiB}$ L2 Cache Workspace（$\le 1.1\text{ MiB}$）完成核间归约。
- **大规模行分块保序流式策略（$n > 3072$，`CHPMV_STRATEGY_ORDERED_TILE`）**：针对 $n=8192$ 等大规模场景下 FP32 顺序累加舍入轨迹敏感性与无上限规模扩展需求，按互斥输出行块 `ORD_TILE = 128` 将行划分至多核（零全局 Workspace、零 `SyncAll()`）。每核在 UB 内维护 $128 \times 128$ 复数矩阵块（$128\text{ KiB}$）与局部行累加器 `out[128]`、`sum[128]`，对直接三角区（列连续）与共轭三角区（行连续读入 + `Gather` UB 原位转置解交织）统一以 8 列批向量乘 + 严格递增列序累加，与 Netlib `cblas_chpmv` 保持一致的顺序舍入语义。

##### 4. tiling规划策略：

直调算子通过值传递结构体 `ChpmvTilingData` 下发运行参数（含 `n`、`nPad`、`uplo`、`usedCores`、`tileRows`、`strategy`、`scalarFlags`、`alphaReal`/`alphaImag`、`betaReal`/`betaImag`、`incx`、`incy`），Kernel 内根据 `scalarFlags`、`strategy`、`usedCores == 1` 与 `uplo` 分派至对应最优执行分支。

#### 3.2.2 kernel侧设计：

Kernel 入口统一为单个核函数 `chpmv_arch22_kernel`（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），分为以下阶段：

1. **标量就绪与快速分支**：
   - 若 `scalarFlags` 指示 `alpha` 或 `beta` 在 Device 侧，各核通过 `DataCopyPad` 在当前 stream 上读取对应 8 字节复数标量；
   - 若 $\alpha = (0,0) \land \beta = (1,0)$ 直接返回；若 $\alpha = (0,0)$ 则直接进入向量缩放写出分支（当 $\beta = (0,0)$ 时直接写零，绝不读取 `y`、`x`、`AP`）。
2. **跨步向量高效加载（`LoadStridedComplexToPlanar`）**：
   - 当 `inc == 1` 时，单条连续 `DataCopyPad` 批量读入并通过 `Gather` 拆分为 `Re` / `Im` 两个连续平面；
   - 当 `inc != 1`（含负步长与大跨步如 `64`、`127`）时，利用 `DataCopyPad` 硬件多突发跨步能力（`blockCount <= 2048, blockLen = 8B, srcStride = (|inc|-1)*8B`）单指令批量抓取跨步复数至 UB 块首，再通过正向（`inc > 0`）或反向（`inc < 0`）偏移表经单条 `Gather` 完成顺序还原与实虚分离，消除逐元素同步 DMA 开销。
3. **32B 对齐掩码化三角列计算（消除非对齐搬移与 `0 * Inf` 风险）**：
   - 对角元素从已预取至 UB 的列块或对齐缓冲中**仅提取实部**参与 `diagRe * x[j]`，不触碰对角虚部；
   - 对于 `LOWER` 模式非对角起点 $i = j+1$ 非 8 元素（32B）对齐的情形，向下对齐至 `baseAligned = (j + 1) & ~7`（偏移 `headSkip = (j + 1) & 7`），利用 `DataCopyPad` 的 `leftPadding = 2 * (headSkip & 3)`（严格 $< 32\text{ B}$）硬件左填充使有效数据精确对齐至 `xRe[baseAligned]` 与 `accRe[baseAligned]`，并用向量掩码将首尾填充位清零后再累加，彻底消除非对齐 `Gather` 与 `V_S`/`S_V` 标量停顿，同时杜绝填充 `0.0f` 与邻近 `Inf`/`NaN` 相乘产生伪 `NaN`。
   - 列内多块共轭内积先在向量寄存器缓冲 `dotVecRe`、`dotVecIm` 中累加，每列仅在末尾执行 1 次 `ReduceSum`，并按批次写回 `acc[j]`，将标量同步次数降低一个数量级以上。
4. **核间 `SyncAll()` 归约与原地写回**：
   - `usedCores == 1` 时直接在 UB 内执行 $y = \alpha \cdot \text{acc} + \beta \cdot y$ 并写回；
   - `usedCores > 1` 时各核将局部 `accRe`、`accIm` 连续写入 L2 中的各自 Workspace 分片，调用 `SyncAll()` 同步后，每核负责互斥行区间 `[rowStart, rowEnd)` 的跨核求和、$\alpha/\beta$ 复标量融合（$\beta=(0,0)$ 时跳过加载 `y`）及步长写回（`incy == 1` 连续 `DataCopyPad` 写出；`incy != 1` 仅写采样物理位置，严守步长间隙不变）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（如 Atlas 800T A2） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（如 Atlas 800T A3） | √ |

## 算子约束限制

- 仅支持 `COMPLEX64`（`aclblasComplex`）数据类型，不支持广播；
- `n >= 0`，`incx != 0`，`incy != 0`，`uplo` 仅限 `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`；
- `y` 为原地读写张量，调用方须保证 `y` 的可写元素与 `AP`、`x` 的可读区域无重叠。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 与 Netlib CBLAS `cblas_chpmv`（`CblasColMajor`）生成的单精度复数 golden 逐分量（实部、虚部分别）比对：`atol = 2^-13`（约 `1.2207e-4`），`rtol = 2^-13`（约 `1.2207e-4`），`matched_ratio >= 0.99` 且 `max_abs_error <= 1e-2`（或 `32 * ULP`）；覆盖 1,000 条全量精度用例（含正负/大步长、对角虚部非零、`beta=0` 旧 `y` 含 `NaN`、结构参数异常及 `Inf`/`NaN` 特殊值）。 | 任务书 §3.2 及 CANN 生态算子混合容差精度标准 |
| 性能标准 | 在 Atlas A2/A3 上采用 `msprof op` 采集单算子 Kernel `Task Duration(us)`（warmup 5 次后连续采样 10 次取算术平均），覆盖 200 条 `TC_PF` 性能用例，单条用例 NPU Kernel 平均耗时 $\le 1.25 \times \text{gpu\_ms}$（即达到 GPU 标杆 `cublasChpmv` Kernel 标杆的 $0.8\times$ 及以上性能）。 | 任务书 §3.3 及 `test_cases/gpu_baseline.csv` |
| 内存标准 | 不涉及（压缩矩阵与向量无 AutoContiguous 视图拷贝，多核归约复用 `handle` 内置 Workspace）。 | 任务书 §3.4 |

## 兼容性分析

- 接口函数原型、枚举值与状态码完全复用 `cann/ops-blas` 既有 `include/cann_ops_blas.h` 声明，不引入任何私有平行 API；
- 实现文件仅作用于 `blas/spmv/arch22/` 与 `test/spmv/chpmv/arch22/`，与 `arch35` 及同目录 `aclblasSspmv` 正交共存、互不影响。
