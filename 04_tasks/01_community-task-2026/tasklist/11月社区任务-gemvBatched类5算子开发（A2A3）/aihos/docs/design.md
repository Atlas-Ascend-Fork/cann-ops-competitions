# 需求背景（required）

## 需求来源

本需求来源于昇腾社区任务[11月社区任务-gemvBatched类5算子开发（A2A3）](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)。目标是在 Atlas A2/A3（`arch22`，`NpuArch::DAV_2201`）系列处理器与 CANN 9.1.0 软件栈环境下，基于 `ops-blas` 开源仓工程（https://gitcode.com/cann/ops-blas ），使用 Ascend C 编程语言开发单精度与半精度实数批量矩阵-向量乘（BLAS-2 batched GEMV）算子族 `aclblasSgemvBatched` 及四个混合/同精度变体（共 5 个公开 API），对标 cuBLAS 对应 5 个变体（`cublasSgemvBatched` / `cublasHSHgemvBatched` / `cublasHSSgemvBatched` / `cublasTSTgemvBatched` / `cublasTSSgemvBatched`）及 Netlib `sgemv.f` 规范，完成技术设计、实现与全量自验，验收通过后正式合入昇腾开源仓。

## 背景介绍

### gemvBatched类算子实现优化

在大语言模型推理、科学计算与现代深度学习推荐系统中，批量矩阵-向量乘（Batched GEMV）是核心的访存密集型（Memory-Bound）基础算子。本任务要求在 Atlas A2/A3 上实现高性能 Ascend C 句柄式接口，通过 handle 绑定 stream 直调 NPU kernel。算子针对 Atlas A2/A3 的 SIMD 矢量计算单元（AIV）架构与片上存储结构进行深度优化：
1. **纯 AIV-Only 高效流式计算**：GEMV 算术强度通常小于 0.5 FLOP/Byte，计算瓶颈在于 HBM 访存带宽与 Kernel 启动延迟。通过 40 个 AIV 核心并发打满硬件 HBM 物理聚合带宽（~1.15–1.3 TB/s），无需申请 Device Workspace（0 Workspace），消除 Cube 单元调度与 Split-K 二次归约核下发延迟；
2. **多变体统一模板与架构复用**：以统一的泛型模板架构同时支持 FP32（S）、FP16（HSH）、FP16 输入/FP32 输出（HSS）、BF16（TST）以及 BF16 输入/FP32 输出（TSS）5 种数据类型，累加域统一维持在 FP32 高精度并在最终输出写回时按最近偶数模式（RNE）收窄，杜绝精度漂移；
3. **消除硬件缺陷与极速访存**：彻底规避 Atlas A2/A3（`DAV_2201`）芯片多 `TBuf` 申请可能导致的片上内存地址重叠硬件缺陷，规划单统一 `TBuf` 连续字节池（实配 188 KB，占 192 KB UB 容量的 95.8%）；集成批量指针预取（Bulk Ptr Prefetch）、二维自适应任务网格调度、Ping-Pong 双缓冲异步掩盖与 4 档非单位步长 DMA 搬运引擎。

### gemvBatched类算子ops-blas现有实现现状分析

当前 `cann/ops-blas` 开源仓中：
1. **公开接口声明完备**：`include/cann_ops_blas.h`（约 L287–308）已存在 `aclblasSgemvBatched`、`aclblasHSHgemvBatched`、`aclblasHSSgemvBatched`、`aclblasTSTgemvBatched`、`aclblasTSSgemvBatched` 共 5 个公开声明，供全系列产品共用，本任务严格保持现有声明不变，不新增私有平行 API；
2. **arch22 实数实现缺失**：`blas/gemv_batched/arch22/` 目录下当前仅有复数算子 `cgemv_batched_*`，缺少 5 个实数变体的 Host 入口、Tiling、Kernel 源码与通用头文件；`test/gemv_batched/arch22/` 亦缺少实数独立测试目标与验证工程；
3. **跨架构参考规范**：仓内 `blas/gemv_batched/arch35/` 与 `test/gemv_batched/arch35/` 已有面向其他芯片架构的实现，可作为参数分派与 GTest 模板参考，但因架构代际、UB 内存体系与底层指令集差异，不得直接复制不兼容代码，必须为 Atlas A2/A3（`arch22`）独立设计。

### gemvBatched类算子功能分析

算子执行批量指针数组形式的矩阵-向量乘加运算：
$$y_i = \alpha \cdot \operatorname{op}(A_i) \cdot x_i + \beta \cdot y_i, \quad i \in [0, \text{batchCount}-1]$$

- **批指针解耦**：输入矩阵与向量基地址由 Device 侧指针数组 `Aarray`、`xarray`、`yarray`（长度均为 `batchCount`）独立给出，各 batch 间数据可在显存中离散存放；
- **存储与转置语义**：矩阵 $A_i$ 恒按**列主序（Column-Major）**存储，物理形状恒为 `lda × n`，且满足 `lda ≥ max(1, m)`（与 `trans` 无关）。转置属性仅作用于数学算子 $\operatorname{op}(A_i)$：
  - `trans == ACLBLAS_OP_N`：$\operatorname{op}(A_i)$ 为 $m \times n$，$x_i$ 逻辑长度为 $n$，$y_i$ 逻辑长度为 $m$；
  - `trans ∈ {ACLBLAS_OP_T, ACLBLAS_OP_C}`：$\operatorname{op}(A_i) = A_i^T$ 为 $n \times m$，$x_i$ 逻辑长度为 $m$，$y_i$ 逻辑长度为 $n$；在实数算子中 `OP_C` 与 `OP_T` 语义完全等价（不做共轭）；
- **步长与非连续寻址**：支持任意非零步长 `incx`、`incy`（支持正负步长，负步长遵循 BLAS 标准反向遍历语义）；支持非紧凑前导维度 `lda`；
- **标量与累加规则**：$\alpha$、$\beta$ 恒为 Host 侧 FLOAT32 标量指针；$\alpha = 0$ 时退化为纯向量缩放 `y = beta * y`，严禁访问 $A_i$ 与 $x_i$；$\beta = 0$ 时严禁读取旧 $y_i$（隔离 NaN/Inf 脏值）。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（`arch22`）处理器上，使用 Ascend C 编程语言开发并交付 5 个高效稳定的批量 GEMV 接口：

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float *const Aarray[], int lda,
    const float *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, uint16_t *const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const uint16_t *const Aarray[], int lda,
    const uint16_t *const xarray[], int incx,
    const float* beta, float *const yarray[], int incy, int batchCount);
```

**输入输出参数规格**：

| 参数名 | 属性/方向 | 描述 | 数据类型与格式 | 约束与取值范围 | 异常行为与状态码 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | 库上下文句柄，携带 stream | `aclblasHandle_t` | 指向有效非空句柄 | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` | 属性 | 矩阵运算模式 | `aclblasOperation_t` | `{OP_N, OP_T, OP_C}` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m`, `n` | 属性 | 矩阵 $A_i$ 的行数与列数 | `int` | $m \ge 0, n \ge 0$ | $<0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$=0$ 触发 quick return |
| `alpha` | 输入 | FLOAT32 标量乘数 | `const float*` (Host) | 有限浮点数 | 为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray` | 输入 | 指向 $A_i$ 的 Device 指针数组 | 按变体 (ND) | 物理尺寸恒为 `lda × n` | 非空计算下为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 属性 | $A_i$ 列主序前导维 | `int` | `lda >= max(1, m)`（与 `trans` 无关） | 低于下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `xarray` | 输入 | 指向 $x_i$ 的 Device 指针数组 | 按变体 (ND) | 逻辑长度 $n$（OP_N）或 $m$（OP_T/C） | 非空计算下为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | 属性 | $x_i$ 元素步长 | `int` | 任意非零整数，且 $\ne \text{INT32\_MIN}$ | 为 0 或 INT32_MIN 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | 输入 | FLOAT32 标量乘数 | `const float*` (Host) | 有限浮点数 | 为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `yarray` | 输出(原地) | 指向 $y_i$ 的 Device 指针数组 | 按变体 (ND) | 逻辑长度 $m$（OP_N）或 $n$（OP_T/C） | 非空计算下为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incy` | 属性 | $y_i$ 元素步长 | `int` | 任意非零整数，且 $\ne \text{INT32\_MIN}$ | 为 0 或 INT32_MIN 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount` | 属性 | 批次数量 | `int` | $\text{batchCount} \ge 0$ | $<0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$=0$ 触发 quick return |

## 需求拆解

1. **5 变体泛型统一架构**：通过 C++ 泛型模板与特化机制，一套核心代码支撑 S、HSH、HSS、TST、TSS 5 个接口，避免代码膨胀；
2. **严密的参数校验与快速返回**：严格执行标准的参数校验流水线。当 `batchCount == 0 || m == 0 || n == 0` 时执行空操作快速返回（返回 SUCCESS，允许指针数组为空，不启动 Kernel）；当 `*alpha == 0.0f && *beta == 1.0f` 时执行纯恒等快速返回；
3. **安全边界与 NaN 隔离**：当 `alpha == 0` 时，仅调度输出向量缩放 `y = beta * y`，严禁加载 `Aarray` 与 `xarray`（隔离 `x` 带 NaN 的场景）；当 `beta == 0` 时，强制跳过旧值 `y` 读取，直接以点积结果覆写（隔离 `y` 带 NaN 的场景）；
4. **二维自适应网格分核（Load Balancing）**：小 batch 场景沿输出维度展开切分，打满 40 个 AIV 物理核心；大 batch 场景保持同 batch 的输出块连续排布在同一核上，最大化片上向量 $x$ 复用率；
5. **单 TBuf 统一片上内存管理（188 KB）**：规避 `DAV_2201` 多 `TBuf` 申请重叠缺陷，通过单连续内存池 `ubPool_` 配合严格 256B 对齐偏移切分，片上内存利用率达 95.8%；
6. **双缓冲流水线与矢量化计算引擎**：
   - `OP_N`：二维 `DataCopyPad` 连续列块搬入，双累加器（`acc0`, `acc1`）交替 `Axpy` 消除 `PIPE_V` 写后读（RAW）数据依赖延迟；细长矩阵切换 `Gather` 抽行向量点积；
   - `OP_T / OP_C`：整列 2D 连续加载，两级批量 `WholeReduceSum` 树形折叠，消除逐列标量归约循环；
7. **免标量停顿的 4 档非单位步长 DMA 引擎**：彻底消灭旧实现中逐元素 `DataCopyPad + MTE2_S` 标量停顿，针对单位步长、倒序步长、小步长窗口 DMA 与大步长 2D 跨步 DMA 实施全向量化与硬件级 DMA 传输；
8. **双百达标验证**：全量 5000 条精度用例 100% 通过（Mixed Tolerance 阈值）；全量 1000 条性能用例 100% 达标（以 `msprof op` kernel Task Duration 为正式判据，单例与均值均达到 GPU kernel-only 标杆 0.8× 以上，并内置 5% 安全余量）。

# 详细设计（required）

## 算子分析

### 数学公式

对每个批次 $i \in [0, \text{batchCount}-1]$，从 Device 侧指针数组获取矩阵与向量基址 $A_i = \text{Aarray}[i]$、$x_i = \text{xarray}[i]$、$y_i = \text{yarray}[i]$：
$$y_i = \alpha \cdot \operatorname{op}(A_i) \cdot x_i + \beta \cdot y_i$$

1. **矩阵元素存储映射**：矩阵 $A_i$ 恒为列主序存储，物理尺寸 $lda \times n$，第 $r$ 行第 $c$ 列元素的物理内存偏移为：
   $$\operatorname{Offset}(A_i, r, c) = c \cdot lda + r, \quad 0 \le r < m, \; 0 \le c < n$$
2. **转置与运算维度映射**：
   - `trans == ACLBLAS_OP_N`：$\operatorname{op}(A_i) = A_i$ 为 $m \times n$。
     $$y_i[r] = \alpha \sum_{c=0}^{n-1} A_i(r, c) \cdot x_i[c] + \beta \cdot y_i[r], \quad 0 \le r < m$$
   - `trans ∈ {ACLBLAS_OP_T, ACLBLAS_OP_C}`：$\operatorname{op}(A_i) = A_i^T$ 为 $n \times m$（实数下 $\operatorname{op}(A_i) = A_i^H \equiv A_i^T$）。
     $$y_i[c] = \alpha \sum_{r=0}^{m-1} A_i(r, c) \cdot x_i[r] + \beta \cdot y_i[c], \quad 0 \le c < n$$
3. **BLAS 标准非零步长反向寻址**：对于逻辑长度为 $L$ 的向量，第 $k \in [0, L-1]$ 个逻辑元素的物理下标偏移为：
   $$\operatorname{phys}(k) = \begin{cases} k \cdot \text{inc}, & \text{inc} > 0 \\ (L - 1 - k) \cdot |\text{inc}| = (k - L + 1) \cdot \text{inc}, & \text{inc} < 0 \end{cases}$$

### 支持数据类型

| 变体名称 | API 名称 | `Aarray` 元素类型 | `xarray` 元素类型 | `yarray` 元素类型 | 标量 $\alpha, \beta$ | 内部累加计算域 | 输出收窄模式 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S** | `aclblasSgemvBatched` | `float` (FP32) | `float` (FP32) | `float` (FP32) | `float` (FP32) | `float` (FP32) | 直写（无需收窄） |
| **HSH** | `aclblasHSHgemvBatched` | `uint16_t` (FP16) | `uint16_t` (FP16) | `uint16_t` (FP16) | `float` (FP32) | `float` (FP32) | RNE (`CAST_ROUND`) |
| **HSS** | `aclblasHSSgemvBatched` | `uint16_t` (FP16) | `uint16_t` (FP16) | `float` (FP32) | `float` (FP32) | `float` (FP32) | 直写（无需收窄） |
| **TST** | `aclblasTSTgemvBatched` | `uint16_t` (BF16) | `uint16_t` (BF16) | `uint16_t` (BF16) | `float` (FP32) | `float` (FP32) | RNE (`CAST_RINT`) |
| **TSS** | `aclblasTSSgemvBatched` | `uint16_t` (BF16) | `uint16_t` (BF16) | `float` (FP32) | `float` (FP32) | `float` (FP32) | 直写（无需收窄） |

### 支持形状

1. **矩阵维度**：$m \ge 0, n \ge 0$。支持常规方形矩阵、高瘦矩阵（$m \gg n$）、细长矩阵（$n \gg m$）以及极小矩阵（$m, n \le 64$）；
2. **前导维度**：$lda \ge \max(1, m)$（恒与 `trans` 无关），支持紧凑排布与带大跨度 Padding 排布；
3. **步长范围**：$\text{incx} \ne 0, \text{incy} \ne 0$，覆盖 $\pm 1, \pm 2, \pm 3, \pm 7, 8, 16, 31, 64, \pm 127$ 等步长；
4. **批次数量**：$\text{batchCount} \ge 0$，覆盖单批（$1$）、小批（$2, 3, 8, 16$）与大批（$64, 256$ 等）；
5. **广播支持**：不支持批间或张量广播，每个 batch 拥有完全独立的 $A_i, x_i, y_i$ 矩阵与向量指针。

## 算子实现

### 实现方案

```
+----------------------------------------------------------------------------------------------------+
|                                    gemvBatched 算子端到端执行流程                                     |
+----------------------------------------------------------------------------------------------------+
  【Host 侧】
    1. 参数合法性严格校验（handle -> trans -> m/n -> lda -> incx/incy -> batchCount -> alpha/beta）
    2. 空维快速返回（batchCount==0 || m==0 || n==0 -> 返回 SUCCESS）
    3. 指针数组非空校验（Aarray, xarray, yarray 顶层非空）与 64 位下标越界防溢出检查
    4. 纯恒等快速返回（*alpha==0.0f && *beta==1.0f -> 返回 SUCCESS）
    5. 动态获取 AIV 核数 C = GetAivCoreCount()，执行 2D 网格分核与 Tiling 路由（6 大路径）
    6. 构造 64B 寄存器直传 TilingData，绑定 stream 异步发射纯 AIV 内核（0 Workspace）
                                     |
                                     v
  【Kernel 侧】
    1. Init 阶段：解析 TilingData，初始化统一 188 KB ubPool_，批量预取 64 组 batch 设备指针
    2. CopyIn 阶段：
       - 加载向量 x 并常驻片上，非 FP32 升精为 FP32
       - 2D DataCopyPad 双缓冲（Ping-Pong）大块连续预取矩阵 A
    3. Compute 阶段：
       - OP_N：双累加器交替 Axpy 掩盖 RAW 冒险；细长矩阵切换 Gather 抽行乘加
       - OP_T/C：整列加载，两级批量 WholeReduceSum 树形折叠，零标量循环
    4. CopyOut 阶段：
       - 融合 alpha 点积缩放 + (beta != 0 ? beta * yOld : 0)
       - RNE 收窄为输出 dtype，按 4 档非单位步长 DMA 引擎精确写回 y，零污染相邻间隙
+----------------------------------------------------------------------------------------------------+
```

#### 3.2.1 host侧设计：

tiling策略：

Host 侧根据算子参数动态决策，不向 Device 申请额外工作区（0 Workspace），将计算所需的切分与调度参数打包为 64 字节、严格 8 字节对齐的 `GemvBatchedTilingData` 结构体，通过 Kernel Launch 参数寄存器直接传递，消除 GM Tiling 结构体的额外读取延迟：

```cpp
struct GemvBatchedTilingData {
    uint64_t totalTasks;       // 总任务数 = batchCount * outTilesPerBatch
    uint32_t m;                // A 矩阵行数
    uint32_t n;                // A 矩阵列数
    uint32_t batchCount;       // 批次数
    uint32_t outTilesPerBatch; // 每个 batch 沿输出维切分的任务段数
    uint32_t usedCores;        // 实际激活的核心数 [1, coreNum]
    uint32_t grain;            // 输出切分粒度 (OP_N 为 64, OP_T 为 16)
    uint32_t route;            // 路由路径枚举 (0~5)
    uint32_t dtype;            // 数据类型变体 (0=HSH, 1=S, 2=HSS, 3=TST, 4=TSS)
    uint32_t transNT;          // 转置模式 (0=OP_N, 1=OP_T/OP_C)
    int32_t lda;               // 列主序前导维 (lda >= max(1, m))
    int32_t incx;              // 输入向量 x 步长
    int32_t incy;              // 输出向量 y 步长
    float alpha;               // Host 标量 alpha
    float beta;                // Host 标量 beta
};
static_assert(sizeof(GemvBatchedTilingData) == 64 && alignof(GemvBatchedTilingData) == 8);
```

任务均分：

动态获取当前平台 AIV 核心数量 $C = \text{GetAivCoreCount()}$（在 Atlas 800T A2 / 910B3 上为 40）。
令输出轴维度为 $\text{outDim} = (\text{trans} == 0) ? m : n$，总任务数 $T = \text{batchCount} \times \text{outTilesPerBatch}$，实际发射核数 $U = \min(C, T)$。
采用标准平摊均分算法将连续任务段分配给各核心，消除负载倾斜：
$$\text{base} = \lfloor T / U \rfloor, \quad \text{rem} = T \bmod U$$
第 $u \in [0, U-1]$ 个 AIV 核心领取的任务区间为 $[\text{taskStart}_u, \text{taskEnd}_u)$：
$$\text{taskStart}_u = u \cdot \text{base} + \min(u, \text{rem}), \quad \text{taskLen}_u = \text{base} + (u < \text{rem} ? 1 : 0)$$

批量搬运：

1. **批量指针预取（Bulk Ptr Prefetch）**：每个核心分配到连续的任务段后，将连续涉及的 batch 指针（至多 64 个）通过 3 条 2D `DataCopyPad` + 1 次 `MTE2_S` 同步一次性读入片上专用缓冲，彻底替代旧实现每 batch 执行 3 次 `MTE2_S / S_MTE2` 串行停顿，指针搬运开销降低最高 64 倍；
2. **矩阵大块连续 DMA 搬运**：利用 2D `DataCopyPad` 的 `srcStride = (lda - rows) * sizeof(T)` 与 `dstStride` 自动补齐能力，每次突发搬运 32~64 KB 大数据块，充分打满硬件 HBM 物理读带宽；
3. **向量复用与局部性保障**：在同一核心连续处理同一 batch 的相邻输出分块时，直接跳过输入向量 $x$ 的加载，实现片上数据 100% 复用。

##### 1. 分核策略：

1. **小批次场景（`batchCount < coreNum`）**：
   - 沿输出维度细粒度展开切分：令粒度 $h = (\text{trans} == 0 ? 64 : 16)$，$G = \lceil \text{outDim} / h \rceil$；
   - 切分段数 $\text{outTilesPerBatch} = \min(G, \lceil C / \text{batchCount} \rceil)$；
   - 总任务数 $T = \text{batchCount} \times \text{outTilesPerBatch}$，激活全部 $U = \min(C, T)$ 个 AIV 核心，打满 40 核并行度，消除小批次核闲置。
2. **大批次场景（`batchCount >= coreNum`）**：
   - 优先每核处理完整 batch（$\text{outTilesPerBatch} = 1$），最大化片上向量 $x$ 复用；
   - 若出现严重尾浪不均（如 $\text{batchCount} = 64$ 在 40 核上产生 2 浪导致 16 核空闲），且总数据量较大时，自动裂变为 $\text{outTilesPerBatch} = 2$，使 128 个任务均匀铺满 40 核（每核 3~4 个任务）。

##### 2. 数据分块和内存优化策略：

Atlas A2/A3（`DAV_2201`）芯片单 AIV 物理可用 UB 容量为 192 KB（196,608 B）。针对已知硬件缺陷（同时申请 $>6$ 个独立 `TBuf` 时可能返回重叠 UB 地址），设计**单统一连续 `TBuf<TPosition::VECCALC> ubPool_` 架构（实配 188 KB = 192,512 B，预留 4 KB 栈安全空间）**，所有子缓冲均以严格互不重叠、256B 对齐的显式字节偏移切分，片上内存利用率高达 95.8%：

| 偏移区间 (KiB) | 缓冲区名称 | 大小 | 数据类型与容量 | 功能与复用说明 |
| :--- | :--- | :---: | :--- | :--- |
| `[0, 2)` | `ptrA / ptrX / ptrY` | 2 KB | `uint64_t[64] × 3` | 批量预取至多 64 组 batch 的 Device 指针数组 |
| `[2, 4)` | `gatherOff` | 2 KB | `uint32_t[512]` | UB `Gather` 专用字节偏移表（倒序/跨步重排/抽行） |
| `[4, 8)` | `stageIO` | 4 KB | `128 × 32B` 槽位 | 跨步向量 2D DMA 暂存区 |
| `[8, 24)` | `x32Buf` | 16 KB | `4096` float | 常驻当前点积轴的 FP32 向量 $x$（支持跨列复用） |
| `[24, 56)` | `aBuf0` (Ping) | 32 KB | `8192` 元素 | 矩阵 $A$ 搬运双缓冲 Ping 槽位 |
| `[56, 88)` | `aBuf1` (Pong) | 32 KB | `8192` 元素 | 矩阵 $A$ 搬运双缓冲 Pong 槽位（加载 $x$ 阶段复用为临时窗口） |
| `[88, 152)` | `a32Work` | 64 KB | `16384` float | 向量计算与 `Cast` 升精工作区（FP32 下复用为 `aBuf` 空间） |
| `[152, 168)` | `reduceBuf` | 16 KB | `4096` float | 两级 `WholeReduceSum` 树形归约中间段和缓冲 |
| `[168, 176)` | `accBuf` | 8 KB | `2048` float | 双累加器输出累加区 |
| `[176, 188)` | `yOutBuf` | 12 KB | `3072` 元素 | 旧 $y$ 读取缓冲及最终输出写回暂存区 |

##### 3. tilingkey规划策略：

根据参数特征，规划 6 条确定性执行路径（`route`）：

| 路径编号 | 路由枚举 | 命中条件 | 核心策略 | 针对目标场景 |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `ALPHA_ZERO` | $\alpha == 0.0\text{f}$ 且未触发恒等快速返回 | 仅调度 $y$ 向量 2D 分核执行 $y = \beta \cdot y$（$\beta = 0$ 写正零），绝对不访问 $A$ 与 $x$ 指针 | $\alpha=0$ 退化场景与 $x$ 带 NaN 场景 |
| **1** | `TINY_FAST` | $\alpha \ne 0$ 且 $\text{batchCount} \times m \times n \times \text{sizeof}(T) \le 128\text{ KB}$ 且 $m, n \le 64$ | 纯 batch 维度分核，整块 $A_i$ 单次搬入 UB 完成计算，零切分循环、极简同步 | 40%~46% 极小规模用例，冲刺 Task Duration $\le 9 \;\mu\text{s}$ |
| **2** | `SMALL_M_N` | `trans == OP_N` 且 $m \le 32, n \ge 64$ | 沿 $n$ 连续搬入多列，UB `Gather` 抽行 + `Mul` + 向量求和 | 细长短行矩阵 |
| **3** | `NORMAL_N` | `trans == OP_N` 其余通用形状 | 2D $(batch, m)$ 分核，双缓冲 2D DMA + 双累加器交替 `Axpy` 消除 RAW 冒险 | 中大矩阵转置前通用场景 |
| **4** | `SMALL_M_T` | `trans != OP_N` 且 $m \le 32, n \ge 64$ | 2D $(batch, n)$ 分核，一次搬入短行多列，常驻 $x$，按行广播 `Axpy` 累加 | 细长矩阵转置场景 |
| **5** | `TRANS_T` | `trans != OP_N` 其余通用形状 | 2D $(batch, n)$ 分核，$x$ 分段常驻，矩阵 2D 连续列块搬运 + 两级 `WholeReduceSum` 树形归约 | 极端带宽受限大矩阵（如 $2047 \times 2048$） |

数据检测：

严格按照以下固定顺序流水线执行 Host 参数检测，非法输入立即返回对应状态码，仅当全部入参合法后才允许执行 quick return：

```
1. handle == nullptr                      --> 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. trans != OP_N && trans != OP_T && OP_C  --> 返回 ACLBLAS_STATUS_INVALID_ENUM (实数 OP_C 合法且等价 OP_T)
3. m < 0 || n < 0                         --> 返回 ACLBLAS_STATUS_INVALID_VALUE
4. lda < max(1, m)                        --> 返回 ACLBLAS_STATUS_INVALID_VALUE (与 trans 无关)
5. incx == 0 || incy == 0 ||
   incx == INT32_MIN || incy == INT32_MIN --> 返回 ACLBLAS_STATUS_INVALID_VALUE (拦截 INT_MIN 防取负溢出)
6. batchCount < 0                         --> 返回 ACLBLAS_STATUS_INVALID_VALUE
7. alpha == nullptr || beta == nullptr    --> 返回 ACLBLAS_STATUS_INVALID_VALUE
8. 【空维快速返回】
   batchCount == 0 || m == 0 || n == 0    --> 返回 ACLBLAS_STATUS_SUCCESS (不启动 Kernel，允许指针为空)
9. Aarray == nullptr || xarray == nullptr
   || yarray == nullptr                   --> 返回 ACLBLAS_STATUS_INVALID_VALUE
10.【恒等快速返回】
   *alpha == 0.0f && *beta == 1.0f        --> 返回 ACLBLAS_STATUS_SUCCESS (不改变 y，原位保持)
11. 64 位下标跨度安全防溢出检查          --> 若跨度超限返回 ACLBLAS_STATUS_INVALID_VALUE
```

#### 3.2.2 kernel侧设计：

Kernel 采用单次 Launch、纯 AIV 向量核（`KERNEL_TYPE_AIV_ONLY`）架构，计算流程分为 `Init`、`CopyIn`、`Compute` 与 `CopyOut` 阶段：

1. **Init 阶段**：
   - 核心根据 `blockIdx` 计算所领取的连续任务区间 $[\text{taskStart}, \text{taskEnd})$；
   - 调用 `pipe_->InitBuffer(ubPool_, 188U * 1024U)` 完成片上连续内存池初始化；
   - 解析首个任务的 `batchIdx`，触发批量指针预取引擎将至多 64 组 batch 的指针一次性搬入 `ptrA/X/Y` 缓冲；
2. **CopyIn 阶段（输入向量与双缓冲矩阵预取）**：
   - **向量 $x$ 加载与复用**：当前 batch 首次处理时，调用 `LoadStridedVec` 搬入 $x_i$ 并在片上将其升精为 FP32 存入 `x32Buf`，同 batch 的后续输出块跳过加载，完全复用；
   - **双缓冲流水线（Ping-Pong）**：对于矩阵 $A_i$，设置槽位 0 与槽位 1。在计算当前槽位矩阵块的同时，异步通过 MTE2 预取下一槽位矩阵块，并通过 `MTE2_V` 与 `V_MTE2` 事件标志位实现 DMA 与 Vector 流水无缝掩盖；
3. **Compute 阶段（全向量化累加与树形归约）**：
   - **`OP_N` 路径**：
     - 若输入为半精度，先执行 `Cast` 升精至 FP32；
     - 采用**双累加器（`acc0`, `acc1`）交替发射 `Axpy` 向量计算**，彻底消除 Vector 流水线 RAW 数据依赖等待；
     - 细长短行矩阵（$m \le 32$）通过预置 `gatherOff` 抽取行向量后与 $x$ 向量点积；
   - **`OP_T / OP_C` 路径**：
     - 连续读取列片段，升精后与 `x32Buf` 对应段做向量逐元素乘法 `Mul`；
     - 采用两级 `WholeReduceSum`（配合 64-elem 段二分 `Add` 折叠），直接归约出当前列的点积标量，**彻底消灭逐列标量操作循环**；
4. **CopyOut 阶段（标量融合与 4 档非单位步长安全写回）**：
   - 执行标量乘加融合：$\text{acc} = \alpha \cdot \text{acc}$；若 $\beta \ne 0$，搬入旧值 $y_i$ 并累加 $\beta \cdot y_{\text{old}}$；
   - 按输出类型收窄：若输出为 FP16/BF16，采用 IEEE 最近偶数舍入模式（`CAST_ROUND` / `CAST_RINT`）精确转换；
   - **4 档非单位步长 DMA 搬运引擎**：
     - `inc == 1`：单次连续 1D `DataCopyPad` 直读直写；
     - `inc == -1`：单次 1D DMA 读入连续区间，UB 内通过预置 `gatherOff` 执行一次 `Gather` 逆序翻转；写回时先在 UB 逆序再 1D DMA 连续落盘；
     - $2 \le |\text{inc}| \le 16$：读取时通过窗口单次 DMA + UB `Gather` 压缩抽取；写回时将数据排布在 `stageIO` 的 32B 间隔槽首，发射一条 2D `DataCopyPad`（`blockLen = sizeof(T), srcStride = 0, dstStride = (|incy|-1)*sizeof(T)`）精确落盘，**绝对不触碰间隙元素**；
     - $|\text{inc}| > 16$：读取时发射单条 2D DMA 读入 32B 间隔槽位后由 `Gather` 一次性压缩；写回时同样经由 2D DMA 跨步精确写回。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | :---: |
| Atlas 800I/T A2 | √ |
| Atlas A3 | √ |

## 算子约束限制

1. **硬件与环境约束**：仅支持 Atlas A2 系列（如 Atlas 800T A2 / `ascend910b3` 等）与 Atlas A3 系列（`ascend910_93` 等），CANN 软件版本要求 CANN 9.1.0；
2. **前导维约束**：矩阵 $A_i$ 恒按列主序存储，前导维度必须满足 `lda >= max(1, m)`（与 `trans` 取值无关）；
3. **步长约束**：向量步长 `incx`、`incy` 必须为非零整数，且不得等于 $\text{INT32\_MIN}$；
4. **内存重叠约束**：各 batch 间的输出向量 $y_i$ 之间不得发生内存重叠；同一 batch 内 $y_i$ 的写目标区域与 $A_i$、$x_i$ 的只读区域不得重叠；
5. **工作空间约束**：算子内部严禁申请 Device 显存，0 Workspace 运行。

# 可维可测分析

## 精度标准/性能标准

### 精度验收标准

1. **比对基线**：以 CPU 侧 Netlib `cblas_sgemv`（3.10.0）逐 batch 计算结果作为主 Golden，输入数据按变体类型量化后在 FP64/FP32 累加域运算并直接收窄为输出数据类型；
2. **精度度量标准**：严格遵守昇腾生态算子开源精度标准（Mixed Tolerance）：
   - **FP32 输出（S / HSS / TSS）**：$rtol = atol = 2^{-13} \approx 1.2 \times 10^{-4}$，最大绝对误差上限 $\le 10^{-2}$ 或 $32 \times \text{ULP}$；
   - **FP16 输出（HSH）**：$rtol = atol = 2^{-10} \approx 9.8 \times 10^{-4}$，最大绝对误差上限 $\le 10^{-1}$ 或 $32 \times \text{ULP}$；
   - **BF16 输出（TST）**：$rtol = atol = 2^{-7} \approx 7.8 \times 10^{-3}$，最大绝对误差上限 $\le 10^{-1}$ 或 $32 \times \text{ULP}$；
   - **判定准则**：逐元素判定 $|actual - golden| \le atol + rtol \cdot |golden|$，每个用例要求匹配率 $\text{matched\_ratio} \ge 0.99$ 且满足最大绝对误差限制；全量 5000 条精度用例必须达到 100% 通过（0 FAIL / 0 SKIP）。

### 性能验收标准

1. **测试环境与基准**：性能基准设备为 Atlas 800T A2（`ascend910b3`），对比基线为 NVIDIA A100 cuBLAS kernel-only 实测耗时（`gpu_ms`）；
2. **正式判据（统一 18-Launch 采集协议）**：
   - 使用 `msprof op` 真实采集底层的单算子 **kernel Task Duration(us)**；
   - 执行统一协议：`1 precheck + 5 warmup + 12 samples = 18 次 Launch`，剔除前 6 次冷态/预热下发，在后 12 次有效样本中剔除 2 个最大瞬态毛刺，对剩余 10 次稳态采样取算术平均作为最终 Task Duration；
   - **达标判据**：NPU kernel 平均耗时 $\le 1.25 \times \text{gpu\_ms}$（即加速比 $\ge 0.8\times$），自测设计预留 **5% 安全余量**（目标 $\ge 0.84\times$），全量 1000 条性能用例 100% 达标；
3. **端到端辅助验证**：测试框架内通过 `aclrtEvent` 对连续 10 次调用计时，输出 `[PERF] case=... avg_us=...`，作为交叉对照验证。

## 兼容性分析

1. **API 与 ABI 兼容性**：完全复用 `include/cann_ops_blas.h` 中已发布的 5 个公开函数签名，参数顺序、类型与状态码与 cuBLAS 严格对齐，不新增私有平行接口；
2. **代码仓既有算子零破坏**：与 `blas/gemv_batched/arch22/` 下既有的复数实现 `cgemv_batched_*` 以及 `arch35/` 实现完全隔离，互不干扰；
3. **构建与测试系统兼容性**：在 `cmake/test.cmake`、`blas/CMakeLists.txt`、`test/CMakeLists.txt` 与 `test/gemv_batched/CMakeLists.txt` 中支持全部 5 个独立 `--ops=` 别名编译构建（`sgemv_batched`、`hsh_gemv_batched`、`hss_gemv_batched`、`tst_gemv_batched`、`tss_gemv_batched`）及聚合测试目标 `gemv_batched_test`，保持原仓 74 条回归用例 100% 正常运行。
