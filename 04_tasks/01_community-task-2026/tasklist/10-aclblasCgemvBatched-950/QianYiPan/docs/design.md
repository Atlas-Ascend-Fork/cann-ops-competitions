# aclblasCgemvBatched 算子设计文档（Atlas 950PR）

| 版本 | 日期 | 作者 | 说明 |
| :--- | :--- | :--- | :--- |
| v1.0 | 2026-10-30 | QianYiPan | 按照 CANN 社区任务模板规范输出技术设计文档，覆盖算子架构设计、SIMT核函数调度、接口契约与测试方案 |

> **目标算子实现目录**：`blas/gemv_batched/arch35/`  
> **目标测试工程目录**：`test/gemv_batched/cgemv_batched/arch35/`  
> **对标接口**：cuBLAS `cublasCgemvBatched`；CANN 9.1.0；Atlas 950PR（`arch35` / `dav-3510`）

---

# 需求背景（required）

## 需求来源

依据 CANN 社区任务（2026 年 10 月）“aclblasCgemvBatched 算子开发（Ascend 950PR）”，要求在 `cann/ops-blas` 开源仓（https://gitcode.com/cann/ops-blas）工程框架下，使用 Ascend C 编程语言在华为昇腾新一代硬件平台 Atlas 950PR（架构代号 `arch35` / `dav-3510`，软件栈环境 CANN 9.1.0）上实现标准 BLAS Level 2 单精度复数批量矩阵-向量乘算子 `aclblasCgemvBatched`。

本技术设计文档按竞赛规范提交至 `cann/cann-ops-competitions` 的 `04_tasks/01_community-task-2026/tasklist/10-aclblasCgemvBatched-950/QianYiPan/docs/` 目录供社区技术委员会与官方专家进行设计评审。

## 背景介绍

### aclblasCgemvBatched 算子功能定义

`aclblasCgemvBatched` 是通用线性代数子程序库（BLAS）中极其核心的 Level-2 批量矩阵-向量乘运算算子，对标 NVIDIA cuBLAS 库中的 `cublasCgemvBatched` 以及 Netlib BLAS 的复数实现。在深度学习复数注意力网络、雷达信号处理、量子计算模拟及科学计算应用中，批量矩阵向量乘具备极高密度的计算需求。

算子对一组批次数量为 `batchCount` 的独立矩阵与向量执行以下原位更新计算：

$$y_i \leftarrow \alpha \cdot \text{op}(A_i) \cdot x_i + \beta \cdot y_i, \quad \forall i \in [0, \text{batchCount}-1]$$

其中：
- $A_i$ 为第 $i$ 个批次的单精度复数（complex64）矩阵，按 BLAS 国际规范采用**列主序（Column-Major）**连续排布，前导维满足 $lda \ge \max(1, m)$；
- $x_i$ 与 $y_i$ 分别为第 $i$ 个批次的单精度复数向量，支持任意非零步长（包含反向寻址的负步长）；
- $\alpha$ 与 $\beta$ 为 Host 侧单精度复数标量，整个批量计算共享同一对标量乘子；
- $Aarray$、$xarray$、$yarray$ 分别为指向各批次矩阵和向量设备内存块的首地址指针数组（Pointer Arrays）；
- $\text{op}(A_i)$ 由统一枚举参数 `trans` 决定，支持不转置（`ACLBLAS_OP_N`）、转置（`ACLBLAS_OP_T`）以及共轭转置（`ACLBLAS_OP_C`）。

对应数学映射关系如下表：

| `trans` 枚举值 | $\text{op}(A_i)$ 语义 | $x_i$ 逻辑长度 | $y_i$ 逻辑长度 | 典型计算量（FLOPs / batch） |
| :--- | :--- | :--- | :--- | :--- |
| `ACLBLAS_OP_N` (0) | $A_i$（不转置） | $n$ | $m$ | $8mn + 6m$ |
| `ACLBLAS_OP_T` (1) | $A_i^T$（普通转置） | $m$ | $n$ | $8mn + 6n$ |
| `ACLBLAS_OP_C` (2) | $A_i^H$（共轭转置） | $m$ | $n$ | $8mn + 6n$ |

> [!IMPORTANT]
> 在复数线性代数中，`ACLBLAS_OP_C` 必须严格执行共轭转置，即读取矩阵元素时必须对其虚部取反，绝不能退化为普通转置。

### 算子现状分析

1. **接口规范同步缺陷**：
   ops-blas 仓公共头文件 `include/cann_ops_blas.h:234-237` 历史声明为单指针形态（`const aclblasComplex* A` 等），与同族算子 `aclblasSgemvBatched`（指针数组形态）以及行业标准 cuBLAS 均不一致，属于接口历史缺陷。任务书要求必须修正为二级指针数组形态（`const aclblasComplex* const Aarray[]`），并同步修正既有 `arch22` 实现，禁止保留单指针私有重载。
2. **`arch35`（Atlas 950PR）全新算力平台特征**：
   Atlas 950PR（`arch35` / `dav-3510`）搭载 56 个高性能 AI Vector（AIV）Core，具有原生 SIMT 硬件执行模型与高达 256KB 的紧耦合 Unified Buffer（UB）空间。对于 GEMV 这种计算访存比（Arithmetic Intensity）相对偏低的 Memory-Bound 算子，采用 Cube 单元会引入不必要的矩阵分块搬移与流间通信开销，因此采用纯 AIV 的轻量级高并发 SIMT 架构是达到极值吞吐的最优架构方案。

### 算子功能分析

算子需全面覆盖以下功能与参数边界：
1. **多重转置模式**：全量支持 `ACLBLAS_OP_N`（无转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（共轭转置）；
2. **双向步长寻址**：支持正步长（$inc > 0$）与负步长（$inc < 0$，按 BLAS 标准自向量末尾元素向前寻址）；
3. **标量与恒等快速旁路**：支持任意复数标量；当 $\alpha = 0 \land \beta = 1$ 时触发快速返回（No-op）；$\beta = 0$ 时跳过原向量 $y$ 读取，避免潜在的未初始化或 NaN 数据污染；
4. **统一防御性拦截**：严格按照任务书 §2.4 与 §2.5 规定的 8 步次序执行参数校验，针对非法枚举、负维度、空句柄、非法步长及空指针实施拦截；
5. **显存安全性与零 Workspace**：算子内部严禁动态分配额外 GM 工作区显存，实现 0 字节独立 Workspace 依赖。

---

# 需求分析（required）

## 需求描述

在 Atlas 950PR（`arch35`）架构上，使用 Ascend C 编程语言实现符合 BLAS 接口契约的 `aclblasCgemvBatched` 算子。算子通过传入的 `handle` 关联的 `aclrtStream` 异步下发，保证端到端异步非阻塞；计算行为与数值精度以 Netlib BLAS 3.10.0 为 Golden 标杆；性能达到并优于 NVIDIA H100 GPU 标杆要求（耗时不超过 GPU 耗时的 2.5 倍，即加速比 $\ge 0.4$）。

## 需求拆解

| 序号 | 需求维度 | 详细设计对应策略 | 验证与验收出口 |
| :--- | :--- | :--- | :--- |
| **F01** | 公共接口与头文件对齐 | 修正公共头为二级指针数组形态，同步修改 `arch22` 与 `arch35` | 编译头文件检查、动态库符号导出检查 |
| **F02** | 严格顺序防御性参数校验 | Host 侧前置执行固定 8 步参数拦截，支持二级指针非空全量巡检 | L6 边界用例、错误码回归测试 |
| **F03** | 恒等与零标量优化路径 | $\alpha=0 \land \beta=1$ 恒等直接返回；$\beta=0$ 消除 $y$ 读入流水 | 边界用例、Profiling 访存带宽对比 |
| **F04** | 批次维度全局打平调度 | 基于 56 个 AIV Core 构建 `(batch, tile)` 二维持久化任务循环网格 | 多 Batch 规模扫描测试，无多核负载倾斜 |
| **F05** | 快慢路径多模板自适应特化 | 设计 G（通用）、N（RegBase）、TC Short、TC Tree、TC Ordered 五大模板 | 覆盖 1200 条多样化形状与步长用例 |
| **F06** | 共轭转置实虚展开与精度保序 | 高精度展开 $(a-bi)(c+di)$；长归约采用有序保序累加 | 1000 条常规用例 100% 精度通过（rtol/atol 达标） |
| **F07** | 0 Workspace 显存硬契约 | 计算就地在输入与全局寄存器完成，无需分配额外中间工作区 | 显存泄露检测、并发多实例压力测试 |
| **F08** | 高性能极致算力释放 | 宽总线多列配对、双缓冲流水线加载、寄存器打平累加 | 200 条 TC_PF 性能用例 100% 达标 |

## 算子接口原型

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

### 参数说明与异常行为

| 参数名 | 类型 | 内存侧 | 含义与值域约束 | 异常行为及返回值 |
| :--- | :--- | :--- | :--- | :--- |
| `handle` | `aclblasHandle_t` | Host | 算子执行上下文句柄，绑定流 | 为 `nullptr` 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | `aclblasOperation_t` | Host | 操作模式：`ACLBLAS_OP_N/T/C` | 取值非三者枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m` | `int` | Host | 矩阵 $A$ 的行数，约束 $m \ge 1$ | $m \le 0$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | `int` | Host | 矩阵 $A$ 的列数，约束 $n \ge 1$ | $n \le 0$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | `const aclblasComplex*` | Host | 复数标量 $\alpha$，非空指针 | 为 `nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray` | `const aclblasComplex* const[]` | Device | 长度为 `batchCount` 的指针数组 | 指针本身为空或**任一数组元素为空**均返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | `int` | Host | 矩阵 $A$ 的前导维，$lda \ge \max(1, m)$ | $lda < \max(1, m)$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `xarray` | `const aclblasComplex* const[]` | Device | 长度为 `batchCount` 的指针数组 | 指针本身为空或**任一数组元素为空**均返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | `int` | Host | 向量 $x$ 元素存储步长，$incx \ne 0$ | $incx = 0 \lor incx = \text{INT\_MIN}$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | `const aclblasComplex*` | Host | 复数标量 $\beta$，非空指针 | 为 `nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `yarray` | `aclblasComplex* const[]` | Device | 长度为 `batchCount` 的指针数组（原位） | 指针本身为空或**任一数组元素为空**均返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incy` | `int` | Host | 向量 $y$ 元素存储步长，$incy \ne 0$ | $incy = 0 \lor incy = \text{INT\_MIN}$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount`| `int` | Host | 批量大小，约束 $batchCount \ge 1$ | $batchCount \le 0$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 设计范围与约束

1. **统一数据排布**：矩阵 $A_i$ 严格遵循列主序（Column-Major），连续列跨度为 $lda$ 个复数元素；向量 $x_i$ 与 $y_i$ 依据各自步长访问，不支持非常规高维 Tensor 视图或稀疏广播；
2. **异步执行模型**：Host 侧函数负责完成参数校验与核函数分发，随后立即通过 `handle->stream` 异步发射 Kernel，不在 API 内部执行阻塞同步操作；
3. **零显存侵入性**：各批次矩阵、向量内存地址由调用方保证生命周期安全与有效性，算子执行中不踩踏 padding 区域。

---

# 详细设计（required）

## 算子分析

### 数学公式

设复数标量 $\alpha = \alpha_r + i\alpha_i$，$\beta = \beta_r + i\beta_i$；向量元素 $x_k = x_{k,r} + ix_{k,i}$，矩阵元素 $A_{j,k} = A_{j,k,r} + iA_{j,k,i}$。

#### 1. 不转置模式（`ACLBLAS_OP_N`）
对于第 $j$ 个输出元素（$0 \le j < m$）：
$$y_j \leftarrow \alpha \cdot \sum_{k=0}^{n-1} A_{j,k} x_k + \beta \cdot y_j$$
其中内积累加项 $S_j = \sum_{k=0}^{n-1} A_{j,k} x_k$ 展开为：
$$S_{j,r} = \sum_{k=0}^{n-1} (A_{j,k,r} x_{k,r} - A_{j,k,i} x_{k,i})$$
$$S_{j,i} = \sum_{k=0}^{n-1} (A_{j,k,r} x_{k,i} + A_{j,k,i} x_{k,r})$$

#### 2. 转置模式（`ACLBLAS_OP_T`）
对于第 $j$ 个输出元素（$0 \le j < n$）：
$$y_j \leftarrow \alpha \cdot \sum_{k=0}^{m-1} A_{k,j} x_k + \beta \cdot y_j$$
内积累加项 $S_j$ 展开公式同上，矩阵读取索引变更为第 $j$ 列、第 $k$ 行（内存地址物理连续，极利于高速向量化加载！）。

#### 3. 共轭转置模式（`ACLBLAS_OP_C`）
对于第 $j$ 个输出元素（$0 \le j < n$）：
$$y_j \leftarrow \alpha \cdot \sum_{k=0}^{m-1} \overline{A_{k,j}} x_k + \beta \cdot y_j$$
共轭内积累加项 $S_j = \sum_{k=0}^{m-1} (A_{k,j,r} - iA_{k,j,i})(x_{k,r} + ix_{k,i})$：
$$S_{j,r} = \sum_{k=0}^{m-1} (A_{k,j,r} x_{k,r} + A_{k,j,i} x_{k,i})$$
$$S_{j,i} = \sum_{k=0}^{m-1} (A_{k,j,r} x_{k,i} - A_{k,j,i} x_{k,r})$$

> **共轭转置算术特征**：实部为对应分量积之**和**，虚部为交叉分量积之**差**。

#### 4. 负步长物理地址映射
当 $incx < 0$ 时，逻辑索引 $k \in [0, \text{len}-1]$ 映射到物理显存位置为：
$$\text{phys\_idx}(k) = (\text{len} - 1 - k) \cdot |incx|$$

### 支持数据类型

| 运算对象 | 数据类型 | 说明 |
| :--- | :--- | :--- |
| 输入矩阵 $A_i$ | `aclblasComplex` (COMPLEX64) | 连续 `float` 实部与虚部对 |
| 输入向量 $x_i$ / 原位向量 $y_i$ | `aclblasComplex` (COMPLEX64) | 依据步长跨步或连续访问 |
| 标量乘子 $\alpha, \beta$ | `aclblasComplex` (COMPLEX64) | 统一位于 Host 侧 |

### 支持形状与规格

- 维度范围：$m \in [1, 16384]$，$n \in [1, 16384]$；
- 批量范围：$batchCount \in [1, 65536]$；
- 步长范围：$incx, incy \in [-16384, 16384] \setminus \{0, \text{INT\_MIN}\}$；
- 前导维：任意 $lda \ge \max(1, m)$。

---

## 算子实现

### 实现方案

```
                       +-----------------------------------+
                       |    aclblasCgemvBatched (Host)     |
                       +-----------------------------------+
                                         |
                       [8-Step Strict Parameter Validation]
                                         |
                       +-----------------------------------+
                       |  Fast-Path Dispatch & Routing     |
                       +-----------------------------------+
                                         |
     +-------------------+---------------+-------------------+-------------------+
     |                   |                                   |                   |
[alpha=0 & beta=1] [Default General (G)]        [Continuous N-Op (N)]     [Continuous Transpose (T/C)]
     |                   |                                   |                   |
Quick Return         SIMT Generic                   RegBase Kernel       +-------+-------+
                       (inc!=1)                    (64x256 Acc Tile)     |       |       |
                                                                       Short    Tree   Ordered
                                                                      (m<=128) (Tree)  (m>4096)
```

#### 3.2.1 Host 侧设计与调度分发

##### 1. 参数防御校验与合法性拦截流程（对齐任务书 §2.4 / §2.5）
Host 侧以严格的先后顺序执行校验，任何一步不合格即刻拦截返回，不启动 Device Kernel：
1. **Handle 校验**：`handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. **正向维度校验**：$m \le 0 \lor n \le 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
3. **前导维约束**：$lda < \max(1, m)$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. **步长与批次有效性**：$incx = 0 \lor incy = 0 \lor incx = \text{INT\_MIN} \lor incy = \text{INT\_MIN} \lor batchCount \le 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
5. **数组指针本身非空校验**：`Aarray == nullptr || xarray == nullptr || yarray == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. **标量指针非空校验**：`alpha == nullptr || beta == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
7. **操作模式枚举校验**：`trans` 不在 {`OP_N`, `OP_T`, `OP_C`} 枚举集内返回 `ACLBLAS_STATUS_INVALID_ENUM`；
8. **恒等操作快速返回（No-op）**：$\alpha = 0 \land \beta = 1$ 时，无需计算，直接返回 `ACLBLAS_STATUS_SUCCESS`。

##### 2. 快慢路径分流决策（Fast-Path & Split）
根据输入参数特征动态分流到最优 Kernel：
- **快速通路条件（Fast-Path）**：
  若满足 $\alpha = (1, 0) \land \beta = (0, 0) \land incx = 1 \land incy = 1 \land lda = m$：
  - 若 `trans == ACLBLAS_OP_N` 且 $m, n \ge 16$：分流至 **RegBase 矩阵块核函数**；
  - 若 `trans \in {OP_T, OP_C}`：
    - $16 \le m \le 128 \land n \ge 160$：分流至 **TC Short 核函数**；
    - $128 < m \le 4096 \land n \ge 160$：分流至 **TC Tree 核函数**；
    - $m > 4096 \lor n < 160$：分流至 **TC Ordered 核函数**。
- **通用降级通路（General-Path）**：
  凡带有非常规标量、任意步长（$incx \ne 1 \lor incy \ne 1$）或小残余尺寸者，由 **G 通用核函数** 保底执行，确保 100% 场景全覆盖。

##### 3. 多核均匀切分与持久循环网格（Grid Mapping）
Atlas 950PR 拥有 56 个独立 AIV Core。为消除批次间调度开销，Host 侧查询实际可用 AIV 核心数：
- 设总任务块数 $TotalTiles = batchCount \times \lceil OutputLength / TileSize \rceil$；
- 启动 BlockDim 设定为 $\min(56, TotalTiles)$；
- Kernel 内部使用 `GetBlockIdx()` 与 Block-Stride 持久循环：
  ```cpp
  for (int tileId = blockIdx; tileId < TotalTiles; tileId += gridDim) {
      int b = tileId / NumTilesPerBatch;
      int t = tileId % NumTilesPerBatch;
      ProcessOneTile(b, t);
  }
  ```
- 彻底杜绝多 Batch 间的冗余初始化与同步开销。

##### 4. 显存与 Workspace 管控（0 字节独立 Workspace）
算子充分利用寄存器和核内 UB 缓存，不申请任何专有 GM Workspace（独立 Workspace 严格为 0 字节），不引入因显存不足导致的申请失败，契约上限与底层 Handle 默认保持 2 GiB 硬上限统一。

---

#### 3.2.2 Kernel 侧设计与指令流水

##### 1. 核心架构与多模板规划
针对复数乘加的运算特征，Kernel 设计了 5 套专属模版：

| 模板标识 | 适用场景 | 核心架构特色 | 显式 UB 分配 | 线程排布 |
| :--- | :--- | :--- | :--- | :--- |
| **G (General)** | 任意步长、通用标量 | 单 SIMT 线程负责单点归约，灵活安全 | 0（依靠 DCache） | 256 线程 |
| **N (RegBase)** | OP_N，连续单位步长 | 64 行 $\times$ 256 列，利用 RegBase 向量累加 | 约 134 KB | 256 线程 |
| **TC Short** | T/C，$m \le 128$ | 4 线程协同负责 1 个输出，规避线程发散 | 0（依靠 DCache） | 512 线程 |
| **TC Tree** | T/C，$128 < m \le 4096$ | 双缓冲乒乓加载，64-lane 树形固定归约 | 最大 229 KB | 512 线程 |
| **TC Ordered** | T/C，长向量 $m > 4096$ | 预乘积 Scatter 后按行序逐段累加，保序稳定 | 最大 248 KB | 512 线程 |

##### 2. 复数乘加与共轭转置实虚展开
在 SIMT 内核中，利用寄存器级别展开复数积运算，避免调用高开销数学函数：
```cpp
// 以共轭转置为例：(Ar - i*Ai) * (xr + i*xi)
float Ar = A_val.real; float Ai = A_val.imag;
float xr = x_val.real; float xi = x_val.imag;
float acc_r += (Ar * xr + Ai * xi);
float acc_i += (Ar * xi - Ai * xr);
```
单条指令周期内实现双路 FMA，充分打满 AIV 向量浮点吞吐。

##### 3. 各分流模板核心流水设计
- **N 模式 RegBase 流水**：
  利用硬件 RegBase 机制，每个线程维护若干输出行的累加寄存器。沿 $n$ 维度以 256 列为步长推进，外层循环加载 $x$ 向量分段，内层展开与矩阵块相乘并累加到寄存器中，最后一次性将累加结果通过异步向量搬运写回 $y$。
- **TC Tree 树形归约流水**：
  沿 $m$ 轴切分为 64 的倍数，核内 UB 建立双缓冲。AIV 执行向量加载时，当前计算块在寄存器内通过蝶式洗牌指令进行 64-lane 快速收敛，隐藏数据从 GM 到 UB 的搬运延迟。
- **TC Ordered 保序累加流水**：
  在超长行场景下，浮点非结合律易导致精度损失。Ordered 模板按矩阵原有行顺序采用确定性交叉分组累加，确保不同 Batch 之间、同输入不同次运行之间满足逐位确定性（Bitwise Determinism）。

##### 4. 向量化访存优化
- **连续内存加载**：转置模式下矩阵列元素物理地址完全连续，统一采用 128-bit 宽总线对齐加载（一次加载 2 个复数/4 个 float）；
- **步长寻址预计算**：通用路径中对负步长首地址偏移在进入计算循环前完成一次性预结算，循环内部仅保留加法步进，消除循环内乘法寻址指令。

---

## 支持硬件

| 支持硬件平台 | 芯片架构 | CANN 最低版本要求 | 编译器支持 |
| :--- | :--- | :--- | :--- |
| **Ascend 950PR** | `arch35` / `dav-3510` | CANN 9.1.0（asc-devkit 9.1.0） | 毕昇 C/C++ 编译器 |
| **Ascend 910B3 / 910_93** | `arch22` / `dav-2201` | CANN 9.1.0 | 毕昇 C/C++ 编译器 |

> 注：`arch22` 仅保持 API 二级指针签名与参数类型同步调用，核心高性能模板针对 `arch35` 深度定制。

## 算子约束限制

1. **输入合法性**：$m \ge 1$，$n \ge 1$，$batchCount \ge 1$；非正维度一律报错返回；
2. **前导维约束**：矩阵 $A$ 前导维严格遵循列主序规范 $lda \ge \max(1, m)$；
3. **步长约束**：$incx \ne 0 \land incy \ne 0 \land incx \ne \text{INT\_MIN} \land incy \ne \text{INT\_MIN}$；
4. **内存重叠**：同一 Batch 内部以及跨 Batch 之间，输入输出缓冲区不得存在未定义的内存别名踩踏。

---

# 可维可测分析

## 精度标准/性能标准

### 1. 精度验收标准（生态开源精度标准）

输出结果与 Netlib BLAS 3.10.0（`cblas_cgemv`）单标杆比对，复数按单精度浮点分量（实部与虚部独立）比对：

| 校验指标 | 标准要求 |
| :--- | :--- |
| **相对容差（rtol）** | $2^{-13} \approx 1.22 \times 10^{-4}$ |
| **绝对容差（atol）** | $2^{-13} \approx 1.22 \times 10^{-4}$ |
| **有限值匹配率（matched_ratio）** | $\ge 99.0\%$ |
| **最大绝对误差（max_abs_error）** | $\le 1.0 \times 10^{-2}$ 或 $32 \times \text{ULP}$ |

- **判定公式**：逐元素判定满足 $|actual - golden| \le atol + rtol \times |golden|$；
- **全量用例要求**：覆盖 L0 基础、L1 形状、L2 标量、L4 前导维/步长、L5 极值填充、L5b 批次扫描及 L6 异常防御共 1200 条用例，精度达标率必须达到 **100% 通过（0 FAIL / 0 SKIP）**。

### 2. 性能验收标准

- **性能标杆**：NVIDIA H100（CUDA 12.4，`cublasCgemvBatched`）；
- **采样规范**：严格执行 **1 次功能预检 + 5 次预热 + 10 次有效采样**；以 10 次有效采样的算术平均 Task Duration（$\mu s$）作为 NPU 实测耗时；
- **门槛判定公式**：
  $$NPU\_mean\_us \le \frac{GPU\_ms \times 1000}{0.4} = 2.5 \times GPU\_us$$
- **达标要求**：全量 200 条 `TC_PF` 性能用例 100% 达成上述门槛（加速比全部 $\ge 0.40$）。

---

## 兼容性分析

1. **头文件与 ABI 兼容性**：
   将公共头文件 `include/cann_ops_blas.h` 的既有单指针错误声明平滑升级为二级指针数组形态，与同仓同族 `aclblasSgemvBatched` 保持一致，消除接口不对称与潜在链接错绑风险；
2. **多架构源码共存**：
   `blas/gemv_batched/arch22/` 实现同步适配为二级指针批量循环；`arch35` 实现独立存放于 `blas/gemv_batched/arch35/`，各架构按 CMake 条件编译独立装配，互不影响；
3. **框架与运行时兼容性**：
   完全遵从 CANN 9.1.0 标准运行时上下文和 Handle 生命周期模型，不引入任何外部非标三方依赖。
