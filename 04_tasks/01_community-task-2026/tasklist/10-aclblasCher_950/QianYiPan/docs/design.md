# aclblasCher 算子设计文档（Atlas 950PR）

| 版本 | 日期 | 作者 | 说明 |
| :--- | :--- | :--- | :--- |
| v1.0 | 2026-10-28 | QianYiPan | 按照 CANN 社区任务格式要求输出完整技术设计文档，覆盖算子架构设计、SIMT核函数调度、接口契约与测试方案 |

> **目标算子目录**：`blas/syr/arch35/`  
> **目标测试目录**：`test/syr/cher/arch35/`  
> **对标接口**：cuBLAS `cublasCher`；CANN 9.1.0；Atlas 950PR（`arch35` / `dav-3510`）

---

# 需求背景（required）

## 需求来源

CANN 社区任务（2026 年 10 月）：`aclblasCher` 算子开发（Ascend 950PR）。任务要求基于 `ops-blas` 开源仓（https://gitcode.com/cann/ops-blas）工程框架，使用 Ascend C 编程语言在 Ascend 950PR（`arch35`，CANN 9.1.0）上实现 BLAS Level 2 单精度复数 Hermitian 秩-1 更新算子 `aclblasCher`，完成算子设计、开发、测试全流程。验收通过后算子代码合入 `blas/syr/arch35/`，测试代码合入 `test/syr/cher/arch35/`，接口声明新增至公共头文件 `include/cann_ops_blas.h`。

## 背景介绍

### aclblasCher 算子目标与功能定义

`aclblasCher` 是 BLAS Level-2 单精度复数（complex64）Hermitian 秩-1 更新算子（Complex Hermitian Rank-1 Update），对标 cuBLAS `cublasCher` 与 Netlib BLAS `cher`。算子以 $n$ 维单精度复数向量 $x$ 与实数标量 $\alpha$ 对 $n \times n$ 厄米特（Hermitian）矩阵 $A$ 执行**原位（In-place）更新**：

$$A \leftarrow \alpha \cdot x \cdot x^H + A$$

其中：
- $\alpha$ 为**单精度实数**标量（`float`，非复数，符合 Netlib `cher` 接口规范）；
- $x$ 为 $n$ 维单精度复数向量（每个元素为两个 32-bit 浮点数构成的实部/虚部连续对）；
- $x^H$ 表示向量 $x$ 的共轭转置；
- $A$ 为 $n \times n$ 厄米特矩阵，列主序存储（Column-Major），前导维满足 $lda \ge \max(1, n)$；
- 矩阵 $A$ 仅根据 `uplo` 参数指定的上三角（`ACLBLAS_UPPER`）或下三角（`ACLBLAS_LOWER`）区域被引用和覆写，未引用的严格互补三角区域保持原样，不被访问；
- 根据 Hermitian 矩阵代数定义，更新后矩阵 $A$ 的对角线元素必须保持纯实数性质，任务书明确规定：**对角元素的虚部假定为 0，更新后强制赋值为 0.0f**。

### aclblasCher 算子现状分析

| 项 | 现状 |
| :--- | :--- |
| **公共头文件声明** | `include/cann_ops_blas.h` 已新增标准 8 参数 `aclblasCher` 签名，供全产品线通用调用。 |
| **`arch35` 架构现状** | 属于 Ascend 950PR 全新开拓的 BLAS Level-2 算子，需基于 `dav-3510` 核心指令与 SIMT 编程范式从零研发。 |
| **硬件架构特征** | Atlas 950PR（`arch35` / `dav-3510`）搭载 56 个高效 AI Vector（AIV）Core，支持原生轻量级 SIMT 硬件线程束与块调度，无 Cube 通信排队与同步开销，适合向量级规整高并发访存。 |

### aclblasCher 算子功能分析

算子需全面覆盖以下功能与边界维度：
1. **三角引用模式**：`uplo` $\in$ {`ACLBLAS_UPPER` (121), `ACLBLAS_LOWER` (122)}；
2. **向量步长多样性**：支持任意正步长（$incx > 0$）与负步长（$incx < 0$ 逆序寻址）；
3. **标量自适应与快速旁路**：标量 $\alpha$ 支持 Host 与 Device 内存位置自适应判定；$\alpha = 0.0\text{f}$ 时触发快速返回（quick return），允许输入矩阵为空且绝不修改 $A$；
4. **零维合法操作**：$n = 0$ 时直接返回成功；
5. **严苛防御校验**：针对空句柄、非法枚举、非法维度、前导维越界、非法步长（$incx = 0 \lor incx = \text{INT\_MIN}$）及空指针进行阶梯式前置拦截；
6. **写保护与对角虚部清零**：非引用三角区域及 padding 字节绝对零修改，对角线虚部严格为 0.0f。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas 950PR（`arch35`）架构上实现高效句柄式算子 `aclblasCher`（8 参数原位更新 $A$ 形式），全面支持 COMPLEX64 数据类型。算子通过 `handle` 关联的 `aclrtStream` 异步下发执行，计算行为与数值精度严格对标 cuBLAS `cublasCher` 与 Netlib BLAS `cher`。

## 需求拆解

1. **标准接口与全量参数校验**：实现句柄、枚举、维度、步长、前导维、空指针全量检查；支持 $n=0$ 与 $\alpha=0$ 的快速返回语义；
2. **复数共轭乘法与实虚展开**：利用单指令多数据展开复数共轭外积，实部两项相加、虚部两项相减；
3. **分级 Kernel 架构**：
   - 小尺寸平铺核（`CherTinyUnitTriangle`）：针对 $n \le 32$ 矩阵，纯网格扁平化展开，消除循环开销；
   - 典型尺寸专用折叠三角核（`CherUnitAlphaTriangle` / `CherUnitAlphaLowerTriangle`）：针对 $n = 512, 1024, 2048, 4096$ 等性能基准，利用三角对称性将上/下三角映射为 $N \times (N/2)$ 连续矩形，彻底消除线程发散（Divergence）；
   - 列分组与循环列向量化核（`CherUnitAlphaGroupedColumns` / `CherCyclicColumns`）：基于 `float2` 宽总线进行双列配对访存展开，隐藏全局内存访存延迟；
   - 通用 GM 慢路径（`CherGm`）：兼容非单位步长、负步长、padding 及 Device 标量；
4. **对角虚部强制置零**：遵循 Hermitian 规范，对角线更新后虚部直接硬赋 `0.0f`；
5. **内存零依赖**：算子纯就地更新，无需申请额外的 Device Workspace 显存，零显存分配失败与 OOM 风险；
6. **双百达标指标规划**：设计完备的自测验证体系，确保全量 1205 项功能/精度用例（满足 0 FAIL / 0 SKIP）与全量 200 个 Profiling 性能用例（目标加速比全部 $\ge 0.4\times$ GPU Baseline）端到端严格达标。

## 算子接口原型

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float* alpha,
    const aclblasComplex* x,
    int incx,
    aclblasComplex* A,
    int lda);
```

### 参数说明与异常行为

| 参数 | 参数含义 | 数据类型 | 约束条件 | 异常行为与返回值 |
| :--- | :--- | :--- | :--- | :--- |
| `handle` | 算子执行上下文句柄 | `aclblasHandle_t` | 非空 | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 矩阵 $A$ 三角存储模式 | `aclblasFillMode_t` | `UPPER` (121) 或 `LOWER` (122) | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | 矩阵 $A$ 的阶数与向量 $x$ 维数 | `int` | $n \ge 0$ | $n < 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$n = 0$ 合法空操作返回 `ACLBLAS_STATUS_SUCCESS` |
| `alpha` | 实数标量乘系数 | `const float*` | 非空指针（支持 Host/Device 内存） | 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；$\alpha = 0.0\text{f}$ 时快速返回 `SUCCESS` |
| `x` | 复数输入向量 | `const aclblasComplex*` | Device/Host 连续或跨步指针 | $\alpha \ne 0 \land n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | 向量 $x$ 元素存储步长 | `int` | $incx \ne 0 \land incx \ne \text{INT\_MIN}$ | 为 0 或 `INT_MIN` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 复数 Hermitian 矩阵（输入/原位输出） | `aclblasComplex*` | Device 矩阵指针，列主序 | $\alpha \ne 0 \land n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 矩阵 $A$ 的领先维度（Leading Dimension） | `int` | $lda \ge \max(1, n)$ | 不满足下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 设计范围与约束

- **数据类型**：向量 $x$ 与矩阵 $A$ 为 `aclblasComplex`（COMPLEX64，`{float real; float imag;}`）；标量 $\alpha$ 为 `float`（FLOAT32 实数）；
- **存储布局**：矩阵 $A$ 严格要求列主序（Column-Major），支持 $lda > n$ 的 padding 存储；
- **执行语义**：计算在传入句柄的关联流上异步下发，主机端不进行冗余同步；
- **内存安全**：保证原位写回 $A$ 时绝对不踩踏未引用的互补三角与 $lda$ 范围外的 padding 空间；$\alpha=0$ 时跳过对 $x$ 与 $A$ 的读写。

---

# 详细设计（required）

## 算子分析

### 数学公式

对于列主序矩阵的第 $row$ 行、第 $col$ 列元素（内存索引为 $col \cdot lda + row$）：
令暂存标量乘向量 $t_{row} = \alpha \cdot x_{row}$（因 $\alpha \in \mathbb{R}$，故 $t_{row}.real = \alpha \cdot x_{row}.real,\; t_{row}.imag = \alpha \cdot x_{row}.imag$）。

元素级外积增量 $\Delta A(row, col) = t_{row} \cdot \overline{x_{col}}$ 展开为：
$$\Delta A(row, col).real = t_{row}.real \cdot x_{col}.real + t_{row}.imag \cdot x_{col}.imag$$
$$\Delta A(row, col).imag = t_{row}.imag \cdot x_{col}.real - t_{row}.real \cdot x_{col}.imag$$

> **关键共轭特性**：与常规复数乘法相比，共轭乘法实部为两积之**和**，虚部为交叉积之**差**！

对角线元素（$row = col$）特殊更新规则：
$$\Delta A(col, col) = \alpha \cdot x_{col} \cdot \overline{x_{col}} = \alpha (|x_{col}.real|^2 + |x_{col}.imag|^2) \in \mathbb{R}$$
故更新公式为：
$$A(col, col).real \leftarrow A(col, col).real + \alpha \cdot (x_{col}.real^2 + x_{col}.imag^2)$$
$$A(col, col).imag \leftarrow 0.0f$$

负步长（$incx < 0$）物理索引映射公式（逻辑下标 $k \in [0, n-1]$）：
$$physIndex = (n - 1 - k) \cdot |incx|$$

### 支持数据类型

| 数据类型 | 支持情况 | 说明 |
| :--- | :--- | :--- |
| `aclblasComplex` (COMPLEX64) | 支持 | 向量 $x$ 与矩阵 $A$（连续实部与虚部 float 对） |
| `float` (FLOAT32) | 支持 | 实数标量 $\alpha$ |

### 支持形状

- 允许合法零维（$n = 0$）；
- 标量 $\alpha$ 允许任意实数值（包含 $\alpha = 0.0\text{f}$、$\alpha = 1.0\text{f}$ 及任意正负有限值/Inf/NaN）；
- 矩阵维度支持 $n \in [1, 4096]$，步长支持 $incx \in [-4096, 4096] \setminus \{0, \text{INT\_MIN}\}$，前导维支持任意 $lda \ge \max(1, n)$。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. tiling策略与参数校验阶梯：
校验顺序严格按规范执行，先完成全量静态校验，再执行合法快速返回：
```
1. handle == nullptr                      --> 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. n < 0                                  --> 返回 ACLBLAS_STATUS_INVALID_VALUE
3. n == 0                                 --> 直接返回 ACLBLAS_STATUS_SUCCESS（合法 quick return）
4. uplo != UPPER && uplo != LOWER         --> 返回 ACLBLAS_STATUS_INVALID_ENUM
5. lda < std::max(1, n)                   --> 返回 ACLBLAS_STATUS_INVALID_VALUE
6. incx == 0 || incx == INT_MIN           --> 返回 ACLBLAS_STATUS_INVALID_VALUE
7. alpha == nullptr                       --> 返回 ACLBLAS_STATUS_INVALID_VALUE
8. 标量判定与解析 (ResolveCherAlpha)：
   - 通过 CheckPtrLocation 判断 Host / Device 内存位置；
   - 若 alpha == 0.0f：直接返回 ACLBLAS_STATUS_SUCCESS（合法 quick return，允许 x/A 为空）
9. alpha != 0.0f 时：
   - x == nullptr || A == nullptr        --> 返回 ACLBLAS_STATUS_INVALID_VALUE
10. 进入 LaunchCher 发射 SIMT Kernel
```

##### 2. 分核策略（Grid & Block Mapping）：
- 启动网格维度通过 `GetAivCoreCount()` 获取物理 AIV Core 数（Atlas 950PR 为 56 个核心）：
  $$useNumBlocks = \min(static\_cast\langle uint32\_t \rangle(n), aivCoreNum)$$
- 采用循环列跨步调度（Grid-Stride / Cyclic Columns）：第 $k$ 个 Core 处理列 $col = k, k + gridDim.x, k + 2 \cdot gridDim.x, \dots$，确保在上三角或下三角负载动态缩减时，各物理核心的工作量全局均匀配平，消除长尾拖尾。

##### 3. 数据分块与内存优化策略：
- **8 字节对齐向量化访存**：将 Device GM 指针重解释为 `__gm__ float2*`，以 64-bit 总线宽字单位执行读写，访存带宽利用率提升 2 倍；
- **In-place 零显存占用**：直接就地更新，无需申请 Device Workspace，运行期零 `aclrtMalloc` / 零 `aclrtFree`。

##### 4. tilingkey规划策略：
- 采用值传递轻量结构体 `CherTilingData`（`numThreads, n, lda, uplo, alphaReal, alphaIsDevice, incx`）直接下发至 Kernel，无需多模板膨胀。

#### 3.2.2 kernel侧设计：

采用原生 Ascend C SIMT 架构，声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，构建多级流水体系：

| 路径 | 核函数 | 适用条件 | 架构特点 |
| :--- | :--- | :--- | :--- |
| **快路径 1** | `CherTinyUnitTriangle` | $n \le 32 \land incx = 1 \land lda = n \land \alpha = 1.0$ | 1024 线程网格扁平化展开，直接消除循环与边界判断开销 |
| **快路径 2** | `CherUnitAlphaUpperTriangle<SHIFT>` <br> `CherUnitAlphaLowerTriangle<SHIFT>` | $n \in \{512, 1024, 2048, 4096\} \land incx = 1 \land lda = n \land \alpha = 1.0$ | **折叠对称矩形核**：将三角矩阵映射为 $N \times (N/2)$ 连续矩形，线程发散度为 0，吞吐达到硬件峰值 |
| **快路径 3** | `CherUnitAlphaGroupedColumns` / `CherCyclicColumns` | 一般单位步长与对齐矩阵 | 双列配对访存展开（`UpdateCherPair`），寄存器复用行向量 $t_{row} = \alpha x_{row}$ |
| **慢路径** | `CherGm` | 非单位步长、负步长、padding 及 Device 标量 | 全参数通用兼容路径，保证 100% 任意场景功能正确 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| :--- | :---: |
| **Ascend 950PR（arch35 / dav-3510）** | **√** |
| Atlas 800I/T A2（arch22） |  |
| Atlas 900 A3 系列（arch22） |  |

## 算子约束限制

1. **数据类型**：仅支持单精度复数（`aclblasComplex`）与单精度实数标量（`float`）；
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
| **精度标准** | 采用 BLAS 复数标准混合容差：$\text{rtol} = 2^{-10}$，$\text{atol} = 2^{-16}$；要求单元素匹配率 $\ge 99\%$；最大绝对误差满足 $\le \max(10^{-2}, 32 \times \text{ULP})$；对角虚部误差严格为 0；支持任务书规定的“一方为 NaN 另一方为 $\pm$Inf 等效判定通过”特化逻辑；全量用例 0 FAIL。 | 任务书 §3.2 验收规范与开源算子精度标准 |
| **性能标准** | 覆盖 `gpu_baseline.csv` 全部 200 个基准用例，在 Atlas 950PR 上单算子耗时满足 $\text{NPU 平均单次耗时} \le \frac{\text{Baseline}_{\text{GPU}}}{0.4} = 2.5 \times \text{Baseline}_{\text{GPU}}$（即加速比 $\ge 0.4\times$ GPU H100 标杆）；全量 200 个用例 100% 达标。 | 任务书 §3.3 与 `gpu_baseline.csv` 契约 |

### 精度判定容差阈值表

| 数据类型 | rtol | atol | 要求单元素匹配率 | 最大绝对误差上限 | 对角虚部容差 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `aclblasComplex` (COMPLEX64) | $2^{-10} \approx 9.77 \times 10^{-4}$ | $2^{-16} \approx 1.53 \times 10^{-5}$ | $\ge 99.0\%$ | $\le \max(10^{-2}, 32 \times \text{ULP})$ | 严格为 0.0f |

### 性能验收基线与 4 个典型 Case 考核指标

依据任务书 §3.3 规范，性能测试以 `gpu_baseline.csv` 中 H100 GPU 耗时为基准，要求单算子平均耗时不高于标杆的 2.5 倍（即加速比 $\ge 0.4\times$）。其中 4 个典型 Case 的考核指标如下：

| Case 编号 | 用例名称 | 三角类型 | 矩阵阶数 $n$ | 向量步长 $incx$ | H100 标杆耗时 (ms) | 标杆微秒 (us) | 验收达标耗时上限 (2.5×, ms) | 达标微秒上限 (us) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 0** | `TC_PF_1001` | UPPER | 512 | 1 | 0.005574 | 5.574 us | $\le 0.013935$ ms | $\le 13.935$ us |
| **Case 1** | `TC_PF_1002` | LOWER | 1024 | 1 | 0.005937 | 5.937 us | $\le 0.014842$ ms | $\le 14.842$ us |
| **Case 2** | `TC_PF_1003` | UPPER | 2048 | 1 | 0.011113 | 11.113 us | $\le 0.027782$ ms | $\le 27.782$ us |
| **Case 3** | `TC_PF_1004` | LOWER | 4096 | 1 | 0.054050 | 54.050 us | $\le 0.135125$ ms | $\le 135.125$ us |

## 测试概览

测试工程落位 `test/syr/cher/`（`cher_golden.h`, `cher_param.h`, `cher_verify_ext.h`）与 `test/syr/cher/arch35/`（`cher_test.cpp`, `cher_npu_wrapper.h`, `cher_test.csv`）。

随任务提供 1200 条用例（1000 精度 + 200 性能，固定随机种子可严格复现）：

| 类别 | 用例前缀 | 数量 | 覆盖维度与测试目标 |
| :--- | :--- | :--- | :--- |
| **L0 冒烟** | `TC_L0` | 4 | UPPER / LOWER 基础通路与小尺寸正确性 |
| **L1 方阵** | `TC_SQ` | 27 | $n \in [1, 2048]$ 典型方阵，含奇数、2的幂及非对齐尺寸 |
| **L2 边界负向** | `TC_ED` | 150 | 零维、合法/非法快速返回、非法步长（0 / `INT_MIN`）、空指针、非法枚举 |
| **L3 填充模式** | `TC_FL` | 42 | 全零、均匀分布、交替符号、极端极值、Inf 与 NaN 特化容差场景 |
| **EX 确定性扩展** | `TC_EX` | 777 | 尺寸池 × 步长池 × Padding 池的大规模参数化组合 |
| **PF 性能用例** | `TC_PF` | 200 | 4 条典型 Case + $n \in [1, 4096]$ 性能扫描用例 |
| **独立单元测试** | `TEST_F` | 5 | 空句柄、零维快速返回、Host/Device 零标量、对角虚部强制置 0 校验 |

## 兼容性分析

- **API 兼容性**：严格按照 `include/cann_ops_blas.h` 公开接口实现，与 cuBLAS `cublasCher` 及 Netlib BLAS 语义完全对齐；
- **平台兼容性**：实现专属于 Atlas 950PR（`arch35`），通过独立的目录隔离与条件编译解耦，不影响 `arch22` 等其他架构的代码与测试组织。
