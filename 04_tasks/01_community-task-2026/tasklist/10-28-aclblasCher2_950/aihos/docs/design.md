# aclblasCher2 算子设计文档（Atlas 950PR）

| 版本 | 日期 | 作者 | 说明 |
| :--- | :--- | :--- | :--- |
| v1.0 | 2026-10-28 | aihos | 按照 CANN 社区任务格式要求输出完整技术设计文档，覆盖算子架构设计、AIV向量与SIMT双路径调度、接口契约与测试方案 |

> **目标算子目录**：`blas/syr2/arch35/`  
> **目标测试目录**：`test/syr2/cher2/arch35/`  
> **对标接口**：cuBLAS `cublasCher2`；CANN 9.1.0；Atlas 950PR（`arch35` / `dav-3510`）

---

# 需求背景（required）

## 需求来源

CANN 社区任务（2026 年 10 月）：`aclblasCher2` 算子开发（Ascend 950PR）。任务要求基于 `ops-blas` 开源仓（https://gitcode.com/cann/ops-blas）工程框架，使用 Ascend C 编程语言在 Ascend 950PR（`arch35`，CANN 9.1.0）上实现 BLAS Level 2 单精度复数 Hermitian 秩-2 更新算子 `aclblasCher2`，完成算子设计、开发、测试全流程。验收通过后算子代码合入 `blas/syr2/arch35/`，测试代码合入 `test/syr2/cher2/arch35/`，接口声明新增至公共头文件 `include/cann_ops_blas.h`。

## 背景介绍

### aclblasCher2 算子目标与功能定义

`aclblasCher2` 是 BLAS Level-2 单精度复数（complex64）Hermitian 秩-2 更新算子（Complex Hermitian Rank-2 Update），对标 cuBLAS `cublasCher2` 与 Netlib BLAS `cher2`。算子以 $n$ 维单精度复数向量 $x, y$ 与复数标量 $\alpha$ 对 $n \times n$ 厄米特（Hermitian）矩阵 $A$ 执行**原位（In-place）更新**：

$$A \leftarrow \alpha \cdot x \cdot y^H + \bar{\alpha} \cdot y \cdot x^H + A$$

其中：
- $\alpha$ 为**单精度复数标量**（`const aclblasComplex*`，包含实部 $\alpha_r$ 与虚部 $\alpha_i$），$\bar{\alpha} = \alpha_r - i \alpha_i$ 为其复共轭；
- $x, y$ 为两个 $n$ 维单精度复数向量（每个元素为两个 32-bit 浮点数构成的实部/虚部连续对）；
- $y^H, x^H$ 表示向量 $y, x$ 的共轭转置操作（对列向量取共轭后转置为行向量）；
- $A$ 为 $n \times n$ 厄米特（Hermitian）矩阵，列主序存储（Column-Major），前导维满足 $lda \ge \max(1, n)$；
- 矩阵 $A$ 仅根据 `uplo` 参数指定的上三角（`ACLBLAS_UPPER`）或下三角（`ACLBLAS_LOWER`）区域被引用和覆写，未引用的严格互补三角区域保持原样，不被访问与修改；
- 根据 Hermitian 矩阵代数定义，更新后矩阵 $A$ 的对角线元素必须保持纯实数性质，任务书明确规定：**对角元素的虚部假定为 0，更新后强制赋值为 0.0f**。

### aclblasCher2 算子现状分析

| 项 | 现状 |
| :--- | :--- |
| **公共头文件声明** | `include/cann_ops_blas.h` 已新增标准 10 参数 `aclblasCher2` 签名，供全产品线通用调用。 |
| **`arch35` 架构现状** | 属于 Ascend 950PR 全新开拓的 BLAS Level-2 算子，需基于 `dav-3510` 核心指令与 AIV/SIMT 混合架构从零研发。 |
| **硬件架构特征** | Atlas 950PR（`arch35` / `dav-3510`）搭载 56 个高效 AI Vector（AIV）Core，支持原生轻量级 SIMT 硬件线程束与块调度，配备 192KB+ 高速 Unified Buffer（UB），无 Cube 通信排队与同步开销，特别适合向量级高并发访存。 |

### aclblasCher2 算子功能分析

算子需全面覆盖以下功能与边界维度：
1. **三角引用模式**：`uplo` $\in$ {`ACLBLAS_UPPER` (121), `ACLBLAS_LOWER` (122)}；
2. **双向量步长多样性**：支持任意正步长（$incx > 0, incy > 0$）与负步长（$incx < 0, incy < 0$ 逆序寻址）；
3. **标量自适应与快速旁路**：标量 $\alpha$ 支持非空校验；当 $\alpha = (0.0\text{f}, 0.0\text{f})$ 时触发快速返回（quick return），允许输入矩阵为空且绝不读写修改 $A$；
4. **零维合法操作**：$n = 0$ 时直接返回成功；
5. **严苛防御校验**：针对空句柄、非法维度（$n < 0$）、非法枚举、前导维越界（$lda < \max(1, n)$）、非法步长（$incx/incy = 0 \lor \text{INT\_MIN}$）及空指针进行阶梯式前置拦截；
6. **写保护与对角虚部清零**：非引用三角区域及 padding 字节绝对零修改，对角线虚部严格写回 0.0f。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas 950PR（`arch35`）架构上实现高效句柄式算子 `aclblasCher2`（10 参数原位更新 $A$ 形式），全面支持 COMPLEX64 数据类型。算子通过 `handle` 关联的 `aclrtStream` 异步下发执行，计算行为与数值精度严格对标 cuBLAS `cublasCher2` 与 Netlib BLAS `cher2`。

## 需求拆解

1. **标准接口与全量参数校验**：实现句柄、枚举、维度、双步长、前导维、空指针全量检查；支持 $n=0$ 与 $\alpha=0$ 的快速返回语义；
2. **复数双共轭秩-1 外积展开**：利用单指令多数据展开复数共轭外积，实部两项相加、虚部两项相减；
3. **分级 Kernel 双路径（Dual-Path）架构**：
   - **快路径 AIV 纯向量核（`Cher2AivKernel`）**：针对连续存储且 $n \in [40, 4096]$ 的紧凑矩阵，将 $x, y$ 完整驻留至 Unified Buffer，利用 MicroAPI 64-lane 向量指令集与向量寄存器复用实现超高带宽吞吐；
   - **慢路径 SIMT 通用核（`Cher2Pairs` / `Cher2Tiny`）**：针对非单位步长、大前导维（$lda > n$）及超小方阵（$n \le 32$），采用三角列对折配对（Invariant Sum $n+1$）与小尺寸扁平展开，消除多核负载长尾倾斜；
4. **对角虚部强制置零**：遵循 Hermitian 规范，对角线更新后虚部直接硬赋 `0.0f`；
5. **内存零依赖**：算子纯就地更新，无需申请额外的 Device Workspace 显存，零显存分配失败与 OOM 风险；
6. **双百达标指标规划**：设计完备的自测验证体系，确保全量 1000 项功能/精度用例（满足 0 FAIL / 0 SKIP）与全量 200 个 Profiling 性能用例（目标加速比全部 $\ge 0.4\times$ GPU Baseline）端到端严格达标。

## 算子接口原型

```cpp
aclblasStatus_t aclblasCher2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x,
    int incx,
    const aclblasComplex* y,
    int incy,
    aclblasComplex* A,
    int lda);
```

### 参数说明与异常行为

| 参数 | 参数含义 | 数据类型 | 约束条件 | 异常行为与返回值 |
| :--- | :--- | :--- | :--- | :--- |
| `handle` | 算子执行上下文句柄 | `aclblasHandle_t` | 非空 | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 矩阵 $A$ 三角存储模式 | `aclblasFillMode_t` | `UPPER` (121) 或 `LOWER` (122) | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | 矩阵 $A$ 的阶数与向量 $x, y$ 维数 | `int` | $n \ge 0$ | $n < 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$n = 0$ 合法空操作返回 `ACLBLAS_STATUS_SUCCESS` |
| `alpha` | 复数标量乘系数 | `const aclblasComplex*` | 非空指针 | 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；$\alpha = (0, 0)$ 时快速返回 `SUCCESS` |
| `x` | 复数输入向量 1 | `const aclblasComplex*` | Device 连续或跨步指针 | $\alpha \ne 0 \land n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | 向量 $x$ 元素存储步长 | `int` | $incx \ne 0 \land incx \ne \text{INT\_MIN}$ | 为 0 或 `INT_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | 复数输入向量 2 | `const aclblasComplex*` | Device 连续或跨步指针 | $\alpha \ne 0 \land n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incy` | 向量 $y$ 元素存储步长 | `int` | $incy \ne 0 \land incy \ne \text{INT\_MIN}$ | 为 0 或 `INT_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 复数 Hermitian 矩阵（输入/原位输出） | `aclblasComplex*` | Device 矩阵指针，列主序 | $\alpha \ne 0 \land n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 矩阵 $A$ 的领先维度（Leading Dimension） | `int` | $lda \ge \max(1, n)$ | 不满足下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 设计范围与约束

- **数据类型**：向量 $x, y$、标量 $\alpha$ 与矩阵 $A$ 均为 `aclblasComplex`（COMPLEX64，`{float real; float imag;}`）；
- **存储布局**：矩阵 $A$ 严格要求列主序（Column-Major），支持 $lda > n$ 的 padding 存储；
- **执行语义**：计算在传入句柄的关联流上异步下发，主机端不进行冗余同步；
- **内存安全**：保证原位写回 $A$ 时绝对不踩踏未引用的互补三角与 $lda$ 范围外的 padding 空间；$\alpha=(0,0)$ 时跳过对 $x, y$ 与 $A$ 的读写。

---

# 详细设计（required）

## 算子分析

### 数学公式

对于列主序矩阵的第 $row$ 行、第 $col$ 列元素（物理存储线性偏移为 $col \cdot lda + row$）：
设输入复数元素分别为：
$$x_{row} = x_r + i x_i, \quad y_{row} = y_r + i y_i$$
$$x_{col} = u_r + i u_i, \quad y_{col} = v_r + i v_i$$
$$\alpha = a_r + i a_i, \quad \bar{\alpha} = a_r - i a_i$$

定义列标量预缩放中间变量（由标量与列向量共轭元素相乘）：
$$P = \alpha \cdot \bar{y}_{col} = (a_r + i a_i)(v_r - i v_i) = (a_r v_r + a_i v_i) + i(a_i v_r - a_r v_i) = p_r + i p_i$$
$$Q = \bar{\alpha} \cdot \bar{x}_{col} = (a_r - i a_i)(u_r - i u_i) = (a_r u_r - a_i u_i) - i(a_r u_i + a_i u_r) = q_r + i q_i$$

两个秩-1 共轭外积项对第 $(row, col)$ 位的更新量为：
$$T_1 = x_{row} \cdot P = (x_r + i x_i)(p_r + i p_i) = (x_r p_r - x_i p_i) + i(x_r p_i + x_i p_r)$$
$$T_2 = y_{row} \cdot Q = (y_r + i y_i)(q_r + i q_i) = (y_r q_r - y_i q_i) + i(y_r q_i + y_i q_r)$$

总更新增量 $\Delta A(row, col) = T_1 + T_2$ 的实部与虚部分别为：
$$\Delta A(row, col).real = (x_r p_r - x_i p_i) + (y_r q_r - y_i q_i)$$
$$\Delta A(row, col).imag = (x_r p_i + x_i p_r) + (y_r q_i + y_i q_r)$$

#### 对角线（$row == col$）特殊约束与严格置零
当 $row == col$ 时，两项互为复共轭，因此虚部在数学上恒为 0。根据任务书与 Netlib `cher2` 标准，更新公式为：
$$A(col, col).real \leftarrow A(col, col).real + \Delta A(col, col).real$$
$$A(col, col).imag \leftarrow 0.0f$$

#### 负步长物理索引映射公式（逻辑下标 $k \in [0, n-1]$）：
- 当 $inc \ge 0$ 时：$physIndex = k \cdot inc$；
- 当 $inc < 0$ 时：$physIndex = (n - 1 - k) \cdot (-inc)$。

### 支持数据类型

| 数据类型 | 支持情况 | 说明 |
| :--- | :--- | :--- |
| `aclblasComplex` (COMPLEX64) | 支持 | 标量 $\alpha$、向量 $x, y$ 与矩阵 $A$（连续实部与虚部 float 对） |

### 支持形状

- 允许合法零维（$n = 0$）；
- 标量 $\alpha$ 允许任意复数值（包含 $\alpha = (0, 0)$、$\alpha = (1, 0)$ 及任意正负有限值/Inf/NaN）；
- 矩阵维度支持 $n \in [1, 4096]$，双步长支持 $incx, incy \in [-4096, 4096] \setminus \{0, \text{INT\_MIN}\}$，前导维支持任意 $lda \ge \max(1, n)$。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. tiling策略与参数校验阶梯：
校验顺序严格按规范执行，先完成全量静态校验，再执行合法快速返回：
```
Step 1: handle == nullptr                      --> 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
Step 2: n < 0                                  --> 返回 ACLBLAS_STATUS_INVALID_VALUE (先于 uplo)
Step 3: uplo != UPPER && uplo != LOWER         --> 返回 ACLBLAS_STATUS_INVALID_ENUM
Step 4: alpha == nullptr                       --> 返回 ACLBLAS_STATUS_INVALID_VALUE
Step 5: incx == 0 || incx == INT_MIN           --> 返回 ACLBLAS_STATUS_INVALID_VALUE (防 |INT_MIN| 溢出)
Step 6: incy == 0 || incy == INT_MIN           --> 返回 ACLBLAS_STATUS_INVALID_VALUE (防 |INT_MIN| 溢出)
Step 7: lda < std::max(1, n)                   --> 返回 ACLBLAS_STATUS_INVALID_VALUE
Step 8: Quick Return 合法快速返回:
        if (n == 0 || (alpha->real == 0.0f && alpha->imag == 0.0f))
            --> 直接返回 ACLBLAS_STATUS_SUCCESS (允许 x, y, A 为空指针)
Step 9: 物理缓冲区非空校验:
        if (x == nullptr || y == nullptr || A == nullptr)
            --> 返回 ACLBLAS_STATUS_INVALID_VALUE
Step 10: 显存可寻址空间上限校验 (Cher2Addressable)
```

##### 2. 双路径分流调度逻辑：
Host 侧通过硬件特征动态路由：
```cpp
const bool contiguous = (incx == 1 && incy == 1);
const bool useAiv = contiguous && (lda == n) && (n >= 40 && n <= 4096);
const auto launch = useAiv ? cher2_aiv_kernel_do : cher2_kernel_do;
```

##### 3. 分核策略（Grid & Block Mapping）：
- 快路径 AIV 核：`blocks = cores = 56`，充分铺满 Atlas 950PR 全部物理 AIV Core；
- 慢路径 SIMT 核：`blocks = std::max(1u, std::min(wanted, (pairs + CHER2_GROUPS - 1) / CHER2_GROUPS))`。

##### 4. 零 Workspace 显存管控：
- 算子为纯原地更新，不申请任何外部工作区显存，`WorkspaceSize = 0`。

#### 3.2.2 kernel侧设计：

##### 1. AIV 纯硬件向量核（`cher2_aiv_kernel.cpp`）：
- **Unified Buffer 向量全量驻留**：利用 10 个容量为 `CAPACITY = 4096 + 64` 的浮点缓冲区（总开销约 $166.4\text{ KB} \le 192\text{ KB}$ UB 容量），通过 DMA `DataCopyPad` 一次性将 $x, y$ 载入 UB，并在 UB 内完成共轭标量乘预缩放：
  - $P = \alpha \cdot \bar{y}$ 存入 `pr, pi`；
  - $Q = \bar{\alpha} \cdot \bar{x}$ 存入 `qr, qi`。
- **64-lane MicroAPI 向量指令集优化**：
  - 通过 `MicroAPI::DataCopy<float, DIST_BRC_B32>` 指令，将当前列的标量 `pr[col], pi[col], qr[col], qi[col]` 单拍广播至 64 个向量通道寄存器；
  - 内层行循环中，AoS 复数矩阵切片通过 `MicroAPI::DeInterleave` 拆分为实虚部寄存器，执行向量 FMA 乘加与减法后，通过 `MicroAPI::Interleave` 重新交织写回；
- **LOWER 连续对齐与 Shift 映射**：
  - 针对 LOWER 模式列起点随 $col$ 偏移问题，初始加载引入 `shift = GetBlockIdx() % 8` 对齐 32 字节边界，使得后续所有列子视图起始地址恒为 32B 硬件对齐；
- **对角虚部硬件级置零**：
  - 列计算完成后，直接在 UB 中对角位置写入 `0.0f`（UPPER 位于 `2 * col + 1`，LOWER 位于 `1`），再统一发起 DMA 写回 GM。

##### 2. SIMT 通用核（`cher2_kernel.cpp`）：
- **三角对折配对（`Cher2Pairs`）**：
  - 将上三角或下三角的第 $j$ 列（长度 $j+1$）与第 $n-1-j$ 列（长度 $n-j$）捆绑为一个配对单元，单单元总计算量恒为 $n+1$；
  - 各 Core 均分配对单元，彻底消除上三角递增与下三角递减引起的多核长尾倾斜；
- **超小方阵展平核（`Cher2Tiny`）**：
  - 针对 $n \le 32$ 极小方阵，启动单 Block 展平写入，单拍内直接计算并写回，执行耗时降低至 $3.6\ \mu\text{s}$ 以内。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| :--- | :---: |
| **Ascend 950PR（arch35 / dav-3510）** | **√** |
| Atlas 800I/T A2（arch22） |  |
| Atlas 900 A3 系列（arch22） |  |

## 算子约束限制

1. **数据类型**：仅支持单精度复数（`aclblasComplex`）；
2. **存储布局**：严格要求列主序存储（Column-Major），不支持行主序；
3. **步长约束**：禁止步长为 0 或 `INT_MIN`；
4. **三角隔离**：非引用三角区域及 padding 空间绝对写保护；
5. **对角虚部**：输出矩阵对角线元素虚部强制保持为 0.0f；
6. **异步执行**：结果由 `aclrtStream` 异步保证，主机读回前须调用 `aclrtSynchronizeStream` 同步。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| :--- | :--- | :--- |
| **精度标准** | 采用 BLAS 复数标准混合容差：$\text{rtol} = 2^{-10}$，$\text{atol} = 2^{-16}$；要求单元素匹配率 $\ge 99\%$；最大绝对误差满足 $\le \max(10^{-2}, 64 \times \text{ULP})$；对角虚部误差严格为 0；支持任务书规定的“一方为 NaN 另一方为 $\pm$Inf 等效判定通过”特化逻辑；全量用例 0 FAIL。 | 任务书 §3.2 验收规范与开源算子精度标准 |
| **性能标准** | 覆盖 `gpu_baseline.csv` 全部 200 个基准用例，在 Atlas 950PR 上单算子耗时满足 $\text{NPU 平均单次耗时} \le \frac{\text{Baseline}_{\text{GPU}}}{0.4} = 2.5 \times \text{Baseline}_{\text{GPU}}$（即加速比 $\ge 0.4\times$ GPU H100 标杆）；全量 200 个用例 100% 达标。 | 任务书 §3.3 与 `gpu_baseline.csv` 契约 |

### 精度判定容差阈值表

| 数据类型 | rtol | atol | 要求单元素匹配率 | 最大绝对误差上限 | 对角虚部容差 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `aclblasComplex` (COMPLEX64) | $2^{-10} \approx 9.77 \times 10^{-4}$ | $2^{-16} \approx 1.53 \times 10^{-5}$ | $\ge 99.0\%$ | $\le \max(10^{-2}, 64 \times \text{ULP})$ | 严格为 0.0f |

### 性能验收基线与 4 个典型 Case 考核指标

依据任务书 §3.3 规范，性能测试以 `gpu_baseline.csv` 中 H100 GPU 耗时为基准，要求单算子平均耗时不高于标杆的 2.5 倍（即加速比 $\ge 0.4\times$）。其中 4 个典型 Case 的考核指标如下：

| Case 编号 | 用例名称 | 三角类型 | 矩阵阶数 $n$ | 双向量步长 $incx, incy$ | H100 标杆耗时 (ms) | 标杆微秒 (us) | 验收达标耗时上限 (2.5×, ms) | 达标微秒上限 (us) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 0** | `TC_PF_1001` | UPPER | 512 | 1, 1 | 0.005768 | 5.768 us | $\le 0.014420$ ms | $\le 14.420$ us |
| **Case 1** | `TC_PF_1002` | LOWER | 1024 | 1, 1 | 0.006101 | 6.101 us | $\le 0.015253$ ms | $\le 15.253$ us |
| **Case 2** | `TC_PF_1003` | UPPER | 2048 | 1, 1 | 0.011525 | 11.525 us | $\le 0.028813$ ms | $\le 28.813$ us |
| **Case 3** | `TC_PF_1004` | LOWER | 4096 | 1, 1 | 0.054474 | 54.474 us | $\le 0.136185$ ms | $\le 136.185$ us |

## 测试概览

测试工程落位 `test/syr2/cher2/`（`cher2_golden.h`, `cher2_param.h`, `cher2_verify_ext.h`）与 `test/syr2/cher2/arch35/`（`cher2_test.cpp`, `cher2_npu_wrapper.h`, `cher2_test.csv`）。

随任务提供 1200 条用例（1000 精度 + 200 性能，固定随机种子可严格复现）：

| 类别 | 用例前缀 | 数量 | 覆盖维度与测试目标 |
| :--- | :--- | :--- | :--- |
| **L0 冒烟** | `TC_L0` | 4 | UPPER / LOWER 基础通路与小尺寸正确性 |
| **L1 方阵** | `TC_SQ` | 46 | $n \in [1, 4096]$ 典型方阵，含奇数、2的幂及非对齐尺寸 |
| **L2 广播标量** | `TC_AB` | 40 | $\alpha = (1, 0), (0, 1), (-1, -1), (0, 0)$ 等各类标量组合 |
| **L3 步长组合** | `TC_ST` | 60 | 正步长、非单位步长、负步长（$incx/incy < 0$）跨步组合 |
| **L4 边界负向** | `TC_ED` | 50 | 零维、快速返回、非法步长（0 / `INT_MIN`）、空指针、非法枚举 |
| **L5 填充极值** | `TC_FL` | 200 | 全零、正态分布、交替符号、极端极值、Inf 与 NaN 特化容差场景 |
| **EX 扩展用例** | `TC_EX` | 600 | 尺寸池 × 双步长池 × Padding 池的大规模正交参数化组合 |
| **PF 性能用例** | `TC_PF` | 200 | 4 条典型 Case + $n \in [1, 4096]$ 性能扫描用例 |

## 兼容性分析

- **API 兼容性**：严格按照 `include/cann_ops_blas.h` 公开接口实现，与 cuBLAS `cublasCher2` 及 Netlib BLAS 语义完全对齐；
- **平台兼容性**：实现专属于 Atlas 950PR（`arch35`），通过独立的目录隔离与条件编译解耦，不影响 `arch22` 等其他架构的代码与测试组织。