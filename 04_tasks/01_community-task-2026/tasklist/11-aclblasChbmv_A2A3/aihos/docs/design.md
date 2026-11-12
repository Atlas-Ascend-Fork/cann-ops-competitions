# aclblasChbmv 算子设计文档（Ascend C / arch22）

> 任务名称：2026 年社区任务 — `aclblasChbmv` 算子开发（A2/A3）  
> 任务书：[aclblasChbmv 官方任务详情](https://www.hiascend.com/activities/task-center/details/ea8efcc9f98c4d54930f8b8e94a59366?menu=guide)  
> 交付模板：[cann-ops-competitions design_template.md](../../../../resources/design_template.md)  
> 目标硬件：Atlas A2 / A3 系列产品（`dav_2201` / `arch22`，性能对标设备 Atlas 800T A2 (910B3)，CANN 版本 9.1.0）  
> 目标仓库：[cann/ops-blas](https://gitcode.com/cann/ops-blas)，落盘路径 `blas/sbmv/arch22/` 与 `test/sbmv/chbmv/arch22/`  
> 提交路径：`04_tasks/01_community-task-2026/tasklist/11-aclblasChbmv_A2A3/aihos/docs/design.md`  

---

# 需求背景（required）

## 需求来源

在昇腾 NPU（Atlas A2/A3 系列产品，含 Atlas 800T A2 / Atlas 800T A3，性能测试对标设备 Atlas 800T A2 (910B3)，CANN 版本 9.1.0）上，基于 Ascend C 编程语言开发单精度复数（COMPLEX64）Hermitian 带状矩阵-向量乘算子 `aclblasChbmv`，执行：
$$y = \alpha \cdot A \cdot x + \beta \cdot y$$
算子完成设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，补齐 BLAS Level-2 算子库能力。

## 背景介绍

### aclblasChbmv 算子目标与功能定义

`aclblasChbmv` 是 BLAS Level-2 单精度复数（complex64）Hermitian 带状矩阵-向量乘算子（Complex Hermitian Banded Matrix-Vector Multiplication），对标 cuBLAS `cublasChbmv` 与 Netlib BLAS `chbmv`。算子以 $n$ 维单精度复数向量 $x, y$ 与单精度复数标量 $\alpha, \beta$ 对 $n \times n$ Hermitian 带状矩阵 $A$ 执行矩阵向量乘累加更新：

$$y \leftarrow \alpha \cdot A \cdot x + \beta \cdot y$$

其中：
- $\alpha, \beta$ 为单精度复数标量（`const aclblasComplex*`，包含实部与虚部）；
- $x, y$ 为 $n$ 维单精度复数向量（每个元素为两个 32-bit 浮点数构成的实部/虚部连续对，步长分别为 `incx` 与 `incy`）；
- $A$ 为 $n \times n$ Hermitian 带状矩阵，具有 $k$ 条超对角线和 $k$ 条次对角线，满足共轭对称性 $A_{i,j} = \overline{A_{j,i}}$（即 $\text{Re}(A_{i,j}) = \text{Re}(A_{j,i}),\; \text{Im}(A_{i,j}) = -\text{Im}(A_{j,i})$）；
- 矩阵 $A$ 按列主序带状格式（Column-Major Banded Format）存储于维度为 $\text{lda} \times n$（$\text{lda} \ge k+1$）的二维连续数组中，仅 `uplo` 指定的带形三角被显式引用，另一侧由 Hermitian 共轭对称性隐含补全；
- **主对角线元素纯实数约束**：根据 Hermitian 矩阵的代数定义，主对角线元素必须恒为纯实数（$A_{j,j} = \overline{A_{j,j}} \implies \text{Im}(A_{j,j}) = 0$）。任务书 §2.1-2 明确规定：**对角元素的虚部假定为 0，实现只取对角元素实部参与计算**（即使输入显存中对角元素虚部为非零随机数、`Inf` 或 `NaN`，也绝不可参与运算）。

### aclblasChbmv 算子现状分析

在 `ops-blas` 开源仓中：
1. `blas/sbmv/` 目录下**当前仅有实对称带状矩阵向量乘算子**（`ssbmv_kernel.cpp` / `ssbmv_host.cpp` / `ssbmv_tiling_data.h`），面向实数单精度 FLOAT32；
2. 仓内公共头文件 `include/cann_ops_blas.h` 中**尚无** `aclblasChbmv` 接口声明，本算子为全新开拓的新增接口；
3. 本任务属于在 Atlas A2/A3（架构目录 `arch22`）上开发首个单精度复数 Hermitian 带状矩阵-向量乘算子，需遵循 Ascend C 向量计算模型（AIV 核心，192KB Unified Buffer），设计高效的复数实虚解交错、带状存储分块投影与无锁行分配并行流水，杜绝跨核原子争用。

### aclblasChbmv 算子输入输出与功能特性分析

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 排布格式 | 维度(shape) | 约束与值域 | 异常行为与返回值 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `handle` | 输入 | ops-blas 句柄，携带 stream | scalar | - | - | 指向有效句柄 | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 输入 | 三角存储模式：`UPPER` 或 `LOWER` | attr | - | - | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | 非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | 输入 | 矩阵 $A$ 阶数及向量 $x, y$ 维数 | scalar | - | - | $n \ge 0$ | $n < 0$ 返回 `INVALID_VALUE`；$n = 0$ 为合法 quick return 返回 `SUCCESS` |
| `k` | 输入 | 半带宽（次/超对角线条数） | scalar | - | - | $k \ge 0$ | $k < 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | 输入 | 复数标量乘数指针 | scalar | - | - | COMPLEX64 | $n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 输入 | Hermitian 带状矩阵，列主序存储 | tensor | ND | $\text{lda} \times n$ | COMPLEX64，仅引用 `uplo` 带形三角 | $n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 输入 | 数组 $A$ 的列主序前导跨度 | scalar | - | - | $\text{lda} \ge k+1$（int64 比较） | $\text{lda} < k+1$ 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | 输入 | 输入复数向量 | tensor | ND | 跨度 $\ge 1+(n-1)\cdot |\text{incx}|$ | COMPLEX64 | $n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | 输入 | 向量 $x$ 存储步长 | scalar | - | - | $\text{incx} \ne 0 \land \text{incx} \ne \text{INT32\_MIN}$ | 为 0 或 `INT32_MIN` 或跨度溢出返回 `INVALID_VALUE` |
| `beta` | 输入 | 复数标量乘数指针 | scalar | - | - | COMPLEX64 | $n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `y` | 输入/原位输出 | 输入输出复数向量，原地覆写 | tensor | ND | 跨度 $\ge 1+(n-1)\cdot |\text{incy}|$ | COMPLEX64 | $n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；$\beta=(0,0)$ 亦不可为空 |
| `incy` | 输入 | 向量 $y$ 存储步长 | scalar | - | - | $\text{incy} \ne 0 \land \text{incy} \ne \text{INT32\_MIN}$ | 为 0 或 `INT32_MIN` 或跨度溢出返回 `INVALID_VALUE` |

- **支持广播**：不涉及（矩阵向量乘运算，输入形状独立固定，无广播）；
- **执行语义**：在 handle 关联的 `aclrtStream` 上以非阻塞方式异步下发执行，主机端零冗余同步。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3（`arch22`）架构上开发单精度复数 Hermitian 带状矩阵-向量乘算子 `aclblasChbmv`，数据类型为 COMPLEX64（实部 FP32 + 虚部 FP32，单元素 8 字节交错存储）。完整支持 `ACLBLAS_UPPER` 与 `ACLBLAS_LOWER` 两种存储模式、任意非负阶数 $n \ge 0$、非负带宽 $k \ge 0$、合法前导维 $\text{lda} \ge k+1$ 以及正负非零步长（$\text{incx} \ne 0, \text{incy} \ne 0$），具备完备的边界防御与零维快速返回能力。

## 需求拆解

1. **数学语义与模式全覆盖**：严格实现 $y = \alpha A x + \beta y$，覆盖 `UPPER` 和 `LOWER` 带状存储，精确展开主对角线纯实数贡献、非对角线正向 AXPY 贡献以及镜像共轭 Dot 贡献；
2. **Hermitian 约束与对角虚部隔离**：主对角线虚部强制假定为 0，核内仅提取对角元素实部参与运算，带外未引用区域与对角虚部的 `Inf/NaN` 绝不污染有效结果；
3. **分级边界与短路径优化**：
   - 零维（$n = 0$）合法快速直返 `ACLBLAS_STATUS_SUCCESS`；
   - 恒等快速直返：$\alpha = (0, 0)$ 且 $\beta = (1, 0)$ 时无损直返；
   - 零覆盖保护：$\beta = (0, 0)$ 时**绝不读取原向量 $y$**（支持原 $y$ 包含未初始化 `NaN`）；
   - 快速缩放：$\alpha = (0, 0), \beta \ne (1, 0)$ 时走轻量 Vector 纯缩放/清零路径；
4. **精度指标达标（100% PASS）**：实部与虚部独立按生态算子开源精度标准（FLOAT32 混合容差）判定：$\text{rtol} = 10^{-3}, \text{atol} = 10^{-6}$，匹配率 $\ge 99.0\%$，逐点最大绝对误差 $\le \max(10^{-2}, 32 \times \text{ULP})$；非有限值遵循任务书 §3.2 等效规则；
5. **性能指标达标（100% 达标）**：全量 200 条性能用例单用例耗时均满足 $\le \text{gpu\_ms} / 0.8$（达到 A100 GPU 标杆 0.8 倍以上性能，内部自测预留 5% 性能安全余量）；
6. **显存零额外占用与硬上限保护**：算子正常计算保持 **新增 Device GM Workspace = 0 字节**；仅在检测到 $x == y$ 原位别名冲突时申请临时 Workspace 备份，并设定 2 GiB 硬上限管控；
7. **接口与工程合规**：新增接口合入 `include/cann_ops_blas.h`，实现代码合入 `blas/sbmv/arch22/`，测试工程合入 `test/sbmv/chbmv/arch22/`。

## 已确认的执行口径

| 项目 | 本方案执行口径 | 依据与说明 |
| :--- | :--- | :--- |
| **校验优先级** | `handle != nullptr` 最优先；紧接着校验 $n \ge 0$，若 $n == 0$ 立即快速返回 `SUCCESS`（不校验后续参数与指针） | 严格遵循社区 issue #765/#861 维护者拍板口径与任务书 §2.5，覆盖 `TC_ED_231`~`238` |
| **非法 uplo 返回值** | 取值不在 `{UPPER, LOWER}` 时统一返回 `ACLBLAS_STATUS_INVALID_VALUE` (3) | 严格遵循任务书 §2.4 与 §7.5 明确约定，覆盖 `TC_ED_251` |
| **指针全量非空校验** | 当 $n > 0$ 时，$\alpha, \beta, A, x, y$ 五个指针均不可为空，$\alpha=(0,0)$ 或 $\beta=(0,0)$ 亦不豁免非空检查 | 严格遵循任务书 §2.4 表格约束，覆盖 `TC_ED_246`~`250` |
| **基线数据绝对保护** | 使用官方随包提供的 CSV，**严禁重新运行 `gen_csv.py`** | 保护 `gpu_baseline.csv` 中已有的 200 条 GPU 实测数据不被空串抹除 |
| **测试集合范围** | 严格执行 `chbmv_test.csv` 全量 1200 条用例（1000 精度 + 200 性能） | 包含正态分布、特殊值、边界与极值，不人为扩缩集合 |
| **性能测试对齐** | 双轨兼容：默认 `chbmv_test.cpp` 预置 Device 缓冲通过 `aclrtEvent` 连续 10 次计时输出 `[PERF]`；同时支持 `CHBMV_PROFILE=1` 环境变量无缝对接 `msprof op` 硬件采集 | 确保无论官方自动化脚本还是底层 Profiling 均 100% 达标 |

---

# 详细设计（required）

## 算子分析

### 数学公式

$$y = \alpha \cdot A \cdot x + \beta \cdot y$$

设复数展开：$A = A_r + i A_i$，$x = x_r + i x_i$，$\alpha = \alpha_r + i \alpha_i$，$\beta = \beta_r + i \beta_i$。
对于一般复数乘法，其单点内积项为：
$$P_r = A_r \cdot x_r - A_i \cdot x_i, \qquad P_i = A_r \cdot x_i + A_i \cdot x_r$$

设中间结果向量 $z = A x \in \mathbb{C}^n$。利用 Hermitian 矩阵共轭对称性 $A_{j,i} = \overline{A_{i,j}} = A_{i,j}^r - i A_{i,j}^i$，对于带状存储数组中存储的第 $j$ 列显式元素：
1. **主对角线贡献（$i = j$）**：
   因 $A_{j,j}$ 为纯实数（虚部强制为 0），设 $d_j = \text{Re}(A_{\text{diag}, j}) \in \mathbb{R}$：
   $$\text{Re}(z_j) \mathrel{+}= d_j \cdot \text{Re}(x_j), \qquad \text{Im}(z_j) \mathrel{+}= d_j \cdot \text{Im}(x_j)$$
2. **非对角线正向贡献（显式存储的 $A_{i,j}$，$i \ne j$）**：
   直接作用于行 $i$（无共轭，正向 AXPY）：
   $$\text{Re}(z_i) \mathrel{+}= \text{Re}(A_{i,j})\text{Re}(x_j) - \text{Im}(A_{i,j})\text{Im}(x_j)$$
   $$\text{Im}(z_i) \mathrel{+}= \text{Re}(A_{i,j})\text{Im}(x_j) + \text{Im}(A_{i,j})\text{Re}(x_j)$$
3. **非对角线镜像共轭贡献（隐式对称位置 $A_{j,i} = \overline{A_{i,j}}$，$i \ne j$）**：
   作用于行 $j$（取复共轭，镜像 Dot 规约）：
   $$\text{Re}(z_j) \mathrel{+}= \text{Re}(A_{i,j})\text{Re}(x_i) + \text{Im}(A_{i,j})\text{Im}(x_i)$$
   $$\text{Im}(z_j) \mathrel{+}= \text{Re}(A_{i,j})\text{Im}(x_i) - \text{Im}(A_{i,j})\text{Re}(x_i)$$
4. **最终 $\alpha, \beta$ 线性组合更新**：
   - 当 $\beta = (0, 0)$ 时，直接覆写：
     $$\text{Re}(y_i) = \alpha_r \text{Re}(z_i) - \alpha_i \text{Im}(z_i), \qquad \text{Im}(y_i) = \alpha_r \text{Im}(z_i) + \alpha_i \text{Re}(z_i)$$
   - 当 $\beta \ne (0, 0)$ 时，读取旧值 $y_i^{\text{old}}$ 累加：
     $$\text{Re}(y_i) = \big(\alpha_r \text{Re}(z_i) - \alpha_i \text{Im}(z_i)\big) + \big(\beta_r \text{Re}(y_i^{\text{old}}) - \beta_i \text{Im}(y_i^{\text{old}})\big)$$
     $$\text{Im}(y_i) = \big(\alpha_r \text{Im}(z_i) + \alpha_i \text{Re}(z_i)\big) + \big(\beta_r \text{Im}(y_i^{\text{old}}) + \beta_i \text{Re}(y_i^{\text{old}})\big)$$

### 支持数据类型

- 标量 $\alpha, \beta$：COMPLEX64（`aclFloatComplex`，包含 `float real` 与 `float imag`）；
- 输入矩阵 $A$、输入向量 $x$、输出向量 $y$：COMPLEX64（单元素 8 字节，实部虚部交错存储）。

### 支持形状

- 矩阵阶数 $n \ge 0$；带宽 $k \ge 0$；
- 矩阵物理存储：二维数组 $\text{lda} \times n$，其中 $\text{lda} \ge k+1$；
- 向量长度：$n$ 维向量，物理跨度满足 $1+(n-1)\cdot |\text{incx}|$ 与 $1+(n-1)\cdot |\text{incy}|$。

### 带状存储语义与地址映射

矩阵 $A$ 以列主序压缩存储于尺寸为 $\text{lda} \times n$ 的二维数组中，第 $j$ 列（$0 \le j < n$）起始物理偏移为 $j \cdot \text{lda}$：

```
+----------------------------------------------------------------------------------------------------+
|                                    列主序带状存储映射模型 (0-based)                                    |
+----------------------------------------------------------------------------------------------------+
  [UPPER: uplo = ACLBLAS_UPPER (0)]
    - 存储上三角带: max(0, j - k) <= i <= j
    - 列内行偏移:   r = k + i - j in [0, k]
    - 线性元素下标: idx = (k + i - j) + j * lda
    - 主对角线位置: A(j, j) 恒位于每列第 r = k 行 (0-based)
    - 未引用填充区: 左上角 j < k 且 r < k - j, 以及 r > k (坚决不予读取)

  [LOWER: uplo = ACLBLAS_LOWER (1)]
    - 存储下三角带: j <= i <= min(n - 1, j + k)
    - 列内行偏移:   r = i - j in [0, k]
    - 线性元素下标: idx = (i - j) + j * lda
    - 主对角线位置: A(j, j) 恒位于每列第 r = 0 行 (0-based)
    - 未引用填充区: 右下角 j > n - 1 - k 且 r > n - 1 - j, 以及 r > k (坚决不予读取)
+----------------------------------------------------------------------------------------------------+
```

### 向量步长与物理定位

向量 $x, y$ 的逻辑长度为 $n$，步长 $\text{inc}$ 允许为负。统一定义 64 位有符号整型安全物理定位公式：
$$\text{start}(n, \text{inc}) = \begin{cases} 0, & \text{inc} > 0 \\ (n - 1) \times (-\text{inc}), & \text{inc} < 0 \end{cases}$$
$$\text{offset}(i, n, \text{inc}) = \text{start}(n, \text{inc}) + i \times \text{inc} \quad (0 \le i < n)$$

API 接收物理缓冲区首地址基指针，负步长逻辑起点由 Kernel 内部基于上述公式统一反向定位，不在 API 入口前移指针，消除双重偏移隐患。

---

## 算子实现

### 实现方案架构

```
+------------------------------------------------------------------------------------+
| Host 侧 (chbmv_host.cpp): 参数阶梯校验 -> 快速路径识别 -> 多核 Tiling -> 异步发射      |
+------------------------------------------------------------------------------------+
                                         | 按值传递 POD ChbmvTilingData
                                         v
+------------------------------------------------------------------------------------+
| Device Kernel 侧 (chbmv_kernel.cpp & chbmv_kernel.h): AIV 向量核流水线加速计算      |
|                                                                                    |
|  [Path 0: Vector Scale/Zero] alpha == (0,0) (不读取 A/x，对 y 执行向量缩放或一键清零) |
|  [Path 1: 行区间归属无锁多核]   通用步长 / 小尺寸 / 列窗复用 (单核负责输出行块，零锁冲突) |
|  [Path 2: 窄带批量列块优化]   k <= 64 且 incx=incy=1 (2D DataCopyPad 一次性搬运多列)  |
|  [Path 3: 宽带分块混合流水]   k > 64 大矩阵 (分块 Gather-Dot + 列窗流水)               |
+------------------------------------------------------------------------------------+
```

#### 3.2.1 host侧设计：

##### 1. 分核策略：
- **优先满核与均衡切分原则**：
  获取当前芯片可用的 AIV 核心数 $N_{\text{max}}$（Atlas 800T A2 通常为 24~40 核）。将输出向量 $y$ 的 $n$ 个元素按可用核数均匀划分为连续行块：
  $$\text{rowsPerCore} = \left\lceil \frac{n}{\text{useCoreNum}} \right\rceil$$
  Core $p$ 独立负责输出行区间 $[p \times \text{rowsPerCore}, \min(n, (p+1) \times \text{rowsPerCore}))$。各核计算输出互斥，**无需任何跨核原子锁与跨核同步**；
- **小尺寸动态限核策略**：
  为避免小规模矩阵调度过多空核带来发射和事件同步延迟，设计动态限核规则：
  $$\text{neededCores} = \left\lceil \frac{n}{32} \right\rceil, \qquad \text{useCoreNum} = \max(1, \min(N_{\text{max}}, \text{neededCores}))$$
  当 $n \le 128$ 时限制使用 1~4 个核心，极大消除调度长尾开销。

##### 2. 数据分块和内存优化策略：
- **核内列窗分块（Column-Window Tile）**：
  对于当前核负责的输出行块 $[i_0, i_1)$（块长 $T_r = i_1 - i_0 \le 256$）：
  能对 $[i_0, i_1)$ 产生贡献的有效列范围为 $j \in [\max(0, i_0 - k), \min(n-1, i_1 - 1 + k)]$。在遍历列 $j$ 时，通过单次 `DataCopyPad` 将第 $j$ 列有效段搬入 UB：
  1. **正向贡献（Vector AXPY）**：该列中属于行 $[i_0, i_1)$ 的显式元素与标量 $x_j$ 进行向量乘累加；
  2. **主对角线贡献**：若 $j \in [i_0, i_1)$，标量读取对角实部 $\text{Re}(A_{\text{diag}, j})$ 累加至第 $j$ 行；
  3. **镜像共轭贡献（Vector Dot）**：若 $j \in [i_0, i_1)$，该列显式存储的非对角段与已缓存在 UB 中的 $x$ 窗口进行共轭点积规约（`ReduceSum`），直接计入第 $j$ 行。
  **一列 $A$ 仅从 Global Memory 读取一次，同时完成正向与镜像双重计算，访存效率达理论最优**；
- **窄带批量列块搬运优化（Multi-Column Batch CopyIn）**：
  当 $k \le 64$ 时，单列有效数据量较小。使用二维 `DataCopyPad`（`blockCount = B_col`，`blockLen = (k+1)*8B`，`srcStride = (lda-(k+1))*8B`），一次 DMA 指令将多列连续搬入 UB，平摊 MTE2 启动延迟；
- **UB 显式空间预算（Atlas A2/A3 单核 UB 容量 192 KB）**：
  设计峰值占用控制在 125 KB，预留 67 KB 安全余量，杜绝栈溢出：

  | 缓冲区名称 | 用途说明 | 元素类型与数量 | 字节大小 | 32B 对齐校验 |
  | :--- | :--- | :--- | :--- | :--- |
  | `inQueueA` (Ping-Pong ×2) | 存放当前搬入的列或多列打包复数带状数据 | `float` × $2 \times 4096 \times 2$ | 32 KB | 严格 32B 对齐 |
  | `xWinRe` / `xWinIm` | 缓存当前行块关联的 $x$ 窗口实虚部 | `float` × $2 \times 4352$ | 34 KB | 严格 32B 对齐 |
  | `accRe` / `accIm` | 当前输出行块累加器实部与虚部 | `float` × $2 \times 512$ | 4 KB | 严格 32B 对齐 |
  | `colRe` / `colIm` | 解交错后的当前列实部与虚部向量 | `float` × $2 \times 2048$ | 16 KB | 严格 32B 对齐 |
  | `tmpBuf0` ~ `tmpBuf3` | 复数乘加与规约临时计算向量 | `float` × $4 \times 2048$ | 32 KB | 严格 32B 对齐 |
  | `yInOutQueue` | 读取旧 $y$ 及组装交错复数写回 GM | `float` × $2 \times 512 \times 2$ | 8 KB | 严格 32B 对齐 |
  | **总计 (Total UB)** | **全流水峰值占用** | — | **~125.0 KB** | **$\le 192\text{ KB}$ (余量 67 KB)** |

##### 3. tilingkey规划策略：
根据矩阵规格与入参属性分发至不同硬件优化分支：
- **`pathType = 0` (ScaleY)**：$\alpha = (0, 0)$，跳过矩阵 $A$ 与向量 $x$ 的读取，直接执行向量化缩放或一键清零；
- **`pathType = 1` (Column-Centric Output-Row)**：通用步长、非连续存储或中等矩阵（$n \le 512$），走标准行分段列窗流水；
- **`pathType = 2` (Narrow-Band Batch)**：窄带连续存储（$k \le 64$ 且 $\text{incx}=\text{incy}=1$ 且 $\text{lda} \times 8\text{B} \equiv 0 \pmod{32}$），启用二维多列打包搬运；
- **`pathType = 3` (Wide-Band Mixed)**：宽带大矩阵（$k > 64, n > 512$），启用大块双缓冲 Gather-Dot 混合流水。

##### 4. 参数校验与防御性策略：
严格按照以下顺序执行阶梯拦截（100% 覆盖 `TC_ED_231`~`263`）：
1. `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. 基础维度校验与快速返回：$n < 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；**$n == 0$ 立即返回 `ACLBLAS_STATUS_SUCCESS`**（不校验后续 uplo/k/lda/指针）；
3. 三角枚举校验：`uplo` 不在 `{UPPER, LOWER}` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. 带宽与前导维校验：$k < 0 \lor \text{lda} < k + 1$（int64 比较防溢出）返回 `ACLBLAS_STATUS_INVALID_VALUE`；
5. 步长校验：$\text{incx} = 0 \lor \text{incy} = 0 \lor \text{incx} = \text{INT32\_MIN} \lor \text{incy} = \text{INT32\_MIN}$ 或跨度超出 `INT32_MAX` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. 指针非空校验：$n > 0$ 时，$\alpha, \beta, A, x, y$ 任一为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；
7. 标量读取与恒等返回：读取 Host/Device 标量，若 $\alpha = (0,0) \land \beta = (1,0)$ 立即返回 `ACLBLAS_STATUS_SUCCESS`；
8. 原位别名保护：检测到 $\alpha \ne (0,0) \land x == y$ 内存重叠时，申请临时 Workspace 进行异步备份，并施加 2 GiB 硬上限管控。

---

#### 3.2.2 kernel侧设计：

算子执行严格分为 `Init`、`Process`（`CopyIn` $\to$ `Compute` $\to$ `CopyOut`）两阶段：

```
+----------------------------------------------------------------------------------------------------+
|                                    Kernel 侧执行流水线分解                                           |
+----------------------------------------------------------------------------------------------------+
  [CopyIn 阶段]
    - 读取输出行块关联的 x 窗口至 UB，解交错为 xWinRe、xWinIm；
    - 若 beta != (0,0)，搬入该核负责的原 y 段并解离；若 beta == (0,0)，严禁读取原 y；
    - 列循环中，双缓冲异步搬入当前列的有效带状数据。
                                     |
                                     v
  [Compute 阶段]
    - 复数实虚分离: 使用 GatherMask 提取偶数下标实部与奇数下标虚部;
    - 正向 AXPY 向量化累加: ProdRe = A_re * x_re - A_im * x_im, ProdIm = A_re * x_im + A_im * x_re;
    - 对角线隔离: 严格标量读取 Re(A_diag)，虚部强制置零，累加至对角行;
    - 镜像共轭 Dot 规约: ConjDotRe = A_re * x_re + A_im * x_im, 通过 ReduceSum 汇聚计入列行;
    - 尾部线性更新: alpha * acc + (beta * y_old)。
                                     |
                                     v
  [CopyOut 阶段]
    - 实虚交错重组: 将更新后的实部与虚部通过双通道交错重组为 [re, im] 物理格式;
    - 连续写回: 32 字节对齐写回 GM; 非连续步长按精确物理索引写回，保持间隙数据不受破坏。
+----------------------------------------------------------------------------------------------------+
```

##### 关键计算与安全设计点：
1. **复数向量化原语（硬件级解交错）**：
   利用 Ascend C `GatherMask`（模式 1 提取实部，模式 2 提取虚部）完成硬件级解交错，利用向量 `Mul`、`Add`、`Sub` 算子打满 AIV 向量计算单元；
2. **主对角线纯实数与带外数据防污染机制**：
   - 针对 `TC_FL_209`（全 `Inf`）与 `TC_FL_212`（全 `NaN`），带状未引用区与主对角线虚部可能存放有 `Inf` 或 `NaN`；
   - 根据 IEEE 754 浮点标准，$0.0\text{f} \times \text{Inf} = \text{NaN}$，因此**严禁采用乘 0.0f 的方式屏蔽对角虚部或带外数据**；
   - Kernel 严格通过精准下标控制循环边界，主对角线仅提取其实部参与实标量乘累加，带外无效数据不参与向量计算，从根本上杜绝 `Inf/NaN` 污染；
3. **$\beta = (0, 0)$ 覆写语义保障**：
   在 `TC_ED_243` 中初始 $y$ 填充为 `NaN`。Kernel 在 $\beta = (0, 0)$ 时直接跳过对原 $y$ 的读取，新计算结果直接覆盖写回，杜绝 $0 \times \text{NaN} = \text{NaN}$。

---

## 支持硬件

| 支持的芯片版本 | 是否涉及 | 验证说明 |
| :--- | :---: | :--- |
| Atlas 800T A2（910B3 训练卡） | **√** | 性能测试对标设备与官方主验证平台 |
| Atlas 800I A2（推理卡） | **√** | 共享 `arch22` 架构指令集 |
| Atlas A3 系列产品（910_93） | **√** | 共享 `arch22` 核心能力，自动适配核心数 |

---

## 算子约束限制

1. **维度与参数范围**：$n \ge 0, k \ge 0, \text{lda} \ge k+1, \text{incx} \ne 0, \text{incy} \ne 0$；
2. **指针重叠约束**：输出向量 $y$ 与输入矩阵 $A$ 不允许内存重叠；若 $y$ 与 $x$ 发生原位重叠（$x == y$），算子将自动申请内部临时显存进行保护（上限 2 GiB）；
3. **非连续访问限制**：不超出 BLAS 标准定义的带状存储与正负步长内存范围。

---

# 可维可测分析（required）

## 精度标准/性能标准

### 1. 精度标准
遵循最新生态算子开源精度标准（COMPLEX64 档，实部与虚部分别按单精度 FLOAT32 独立判定）：
- **逐元素混合容差门限**：
  $$|actual - golden| \le \text{atol} + \text{rtol} \times |golden| \quad (\text{rtol} = 10^{-3}, \text{atol} = 10^{-6})$$
  要求满足条件的元素比例：$\text{matched\_ratio} \ge 0.99$；
- **全张量逐点硬上限（Hard Limit）**：
  每一个有限分量必须同时满足：
  $$|actual - golden| \le \max(10^{-2}, 32 \times \text{ULP}(|golden|))$$
  其中 $\text{ULP} = \text{ldexp}(1.0\text{f}, \text{exp} - 24)$ 按最大误差点 Golden 值的指数动态计算；
- **Inf/NaN 特殊值判定规则**：
  非有限值遵循任务书 §3.2 规则（双方同为 NaN、同号 Inf，或一方为 NaN 另一方为 ±Inf 视为等效 PASS，不计入有限值匹配率分母）；
- **Golden 真值源**：
  调用 Netlib CBLAS `cblas_zhbmv`（输入转 FP64 真值计算，计算后收窄至 FP32 输出），实部与虚部全量 100% 独立验证。

---

### 2. 性能标准
- **性能达标判据**：
  全量 200 条性能用例单用例平均耗时均满足：
  $$t_{\text{NPU, avg}}(\mu s) \le \frac{1000 \times \text{gpu\_ms}}{0.8}$$
  （达到标杆 GPU 的 0.8 倍以上性能，内部自测预留 5% 性能安全余量）；
- **双轨性能采集机制**：
  1. **测试框架直接计时（默认）**：`chbmv_test.cpp` 预置 Device 缓冲，先执行 5 次预热，再通过 `aclrtEvent` 测量连续 10 次下发的总时间取平均，打印 `[PERF]` 标准行，支持 `verify_performance.py` 一键解析；
  2. **硬件 `msprof op` 模式**：测试代码响应环境变量 `CHBMV_PROFILE=1`、`CHBMV_WARMUP`（默认 10）与 `CHBMV_SAMPLES`（默认 60）下发循环，敞开硬件采集通道，由分析脚本剔除前 11 次与 10 个最大毛刺尖刺后取算术平均。

---

### 3. 内存标准与记录口径
1. **任务书依据**：依据任务书 §3.4 规定，本算子**内存要求：不涉及**（官方未设定额外显存阈值）；
2. **算子实现指标**：采用纯 AIV 片上 UB 累加流水，**新增 Device GM Workspace 恒为 0 字节**（无 $x==y$ 冲突时零显存申请）；
3. **自测验收与审计口径**：
   - 记录必要张量物理存储量：
     $$A_{\text{bytes}} = 8 \times \text{lda} \times n$$
     $$x_{\text{bytes}} = 8 \times (1 + (n - 1) \times |\text{incx}|)$$
     $$y_{\text{bytes}} = 8 \times (1 + (n - 1) \times |\text{incy}|)$$
   - 通过 `npu-smi` 实时采样，严格验证初始/峰值/测试后显存，确保显存 0 泄漏。

---

## 固定 CSV 测试工程设计

### 1. 文件冻结与 12 类用例覆盖台账
测试集严格冻结为官方随包提供的 1200 条用例，覆盖分布如下：

| 类别代号 | 类别名称 | 用例条数 | 覆盖特征与说明 |
| :---: | :---: | :---: | :--- |
| **L0** | 基础小用例 | 4 | $n \in \{4, 8\}$，全 uplo 组合，基础带宽 |
| **SQ** | 方阵阶数扫描 | 72 | 阶数从 1 到 4096，含奇偶、质数、2的幂及对齐 |
| **BW** | 带宽极值扫描 | 36 | $k=0$（对角矩阵）、$k=1$、$k=n-1$（满带）等 |
| **AB** | 标量特殊值 | 32 | $\alpha, \beta$ 取 $(0,0)$、$(1,0)$、纯虚数、大值等特殊组合 |
| **LD** | 前导维 padding | 18 | $\text{lda} = k+1$ 紧凑与 $\text{lda} > k+1$ 填充跨度 |
| **INC** | 向量步长组合 | 36 | $\text{incx}, \text{incy} \in \{\pm 1, \pm 2, \pm 3\}$ 全组合（含负步长反向寻址） |
| **FL** | 特殊值填充 | 15 | 极端浮点数、交替值、全 `Inf`、全 `NaN` 鲁棒性验证 |
| **ED** | 边界与负向用例 | 33 | 零维、空句柄、非法 uplo、越界前导维、负维度、零步长、空指针 |
| **EX** | 规模正交扩展 | 754 | 尺寸、带宽、步长组合的大范围随机正交覆盖 |
| **PF** | 性能测试用例 | 200 | 连续访存（$\text{incx}=\text{incy}=1$）下的各种规模与带宽典型性能用例 |
| **合计** | - | **1200** | **1000 条精度/接口测试 + 200 条纯连续性能测试** |

### 2. 负向注入与防 OOM 保护机制
- 当 `expect_result != ACLBLAS_STATUS_SUCCESS` 或 $n \le 0$ 时（如 `TC_ED_260`~`263` 中 $\text{incx} = \pm 2147483647$），测试工程仅分配固定 16 元素 dummy 缓冲，彻底规避主机端内存溢出（OOM）；
- 识别 `"null"` 字符串精准注入 `nullptr`，识别非法枚举 `999` 原样传入接口验证拦截行为。

---

## 兼容性分析

1. **接口规范兼容性**：函数签名与符号命名与 cuBLAS `cublasChbmv` 完全一致，状态码遵循 ops-blas 仓公共定义；
2. **产品线兼容性**：接口声明新增至 `include/cann_ops_blas.h`，供昇腾全产品线共用，杜绝私有接口；
3. **架构与工程兼容性**：代码严格遵循 `ops-blas` CMake `arch22` 自动发现机制，落盘于 `blas/sbmv/arch22/` 与 `test/sbmv/chbmv/arch22/`，公共框架代码零侵入修改，确保社区 PR 能够顺畅合入。
