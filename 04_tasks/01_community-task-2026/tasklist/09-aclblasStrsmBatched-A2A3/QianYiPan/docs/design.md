# aclblasStrsmBatched 算子设计文档（Atlas A2/A3）

| 版本 | 日期 | 作者 | 说明 |
| :--- | :--- | :--- | :--- |
| v1.0 | 2026-10-13 | QianYiPan | 按照 CANN 算子大赛格式要求输出完整技术设计文档，覆盖精度与性能全量双百达标架构 |

> **目标算子目录**：`blas/trsmbatched/arch22/`  
> **目标测试目录**：`test/trsmbatched/strsmbatched/arch22/`  
> **对标接口**：cuBLAS `cublasStrsmBatched`；CANN 9.1.0；Atlas A2/A3（`arch22` / `dav-2201`）

---

# 需求背景（required）

## 需求来源

CANN 社区任务：在 Atlas A2/A3（`arch22`）平台使用 Ascend C 编程语言实现单精度实数批量三角方程组求解算子 `aclblasStrsmBatched`。算子对 `batchCount` 个相互独立的三角线性系统执行批量并行求解，并在验收后向 `ops-blas` 仓提交 PR。

## 背景介绍

### aclblasStrsmBatched 算子目标与功能定义

`aclblasStrsmBatched` 是 BLAS Level-3 批量实数单精度（FP32）三角矩阵方程组求解算子（Batched Single-Precision Triangular Solve with Multiple Right-Hand Sides），对标 `cublasStrsmBatched` 与批量化 Netlib BLAS `strsm`。算子在单次 API 调用内对 `batchCount` 个相互独立的线性系统并行求解，并将解矩阵 $X_b$ **原位（In-place）写回**输入输出矩阵指针数组 $B[b]$（$b \in [0, \text{batchCount})$）：

$$\begin{aligned}
\text{当 } \text{side} = \text{ACLBLAS\_SIDE\_LEFT}: &\quad \operatorname{op}(A_b) \cdot X_b = \alpha B_b, \quad A_b \in \mathbb{R}^{m \times m},\; B_b, X_b \in \mathbb{R}^{m \times n} \\
\text{当 } \text{side} = \text{ACLBLAS\_SIDE\_RIGHT}: &\quad X_b \cdot \operatorname{op}(A_b) = \alpha B_b, \quad A_b \in \mathbb{R}^{n \times n},\; B_b, X_b \in \mathbb{R}^{m \times n}
\end{aligned}$$

其中：
- $k = (\text{side} == \text{ACLBLAS\_SIDE\_LEFT}) \mathbin{?} m : n$ 为三角系数矩阵 $A_b$ 的阶数；
- $\operatorname{op}(A_b) = A_b$（当 $\text{trans} = \text{ACLBLAS\_OP\_N}$）或 $A_b^T$（当 $\text{trans} = \text{ACLBLAS\_OP\_T}$）；对于实数单精度算子，共轭转置 `ACLBLAS_OP_C` 属于非法输入；
- $A_b$ 按 `uplo` 为上三角（`ACLBLAS_UPPER`）或下三角（`ACLBLAS_LOWER`）矩阵，按 `diag` 为非单位三角（`ACLBLAS_NON_UNIT`）或单位三角（`ACLBLAS_UNIT`，对角元隐式视为 $1.0$，不读取存储对角线）；
- 所有矩阵 $A_b, B_b$ 均采用标准 BLAS **列主序（Column-Major）** 存储，领先维度分别为 $\text{lda} \ge \max(1, k)$ 与 $\text{ldb} \ge \max(1, m)$。

### aclblasStrsmBatched 算子现状分析

| 项 | 现状 |
| :--- | :--- |
| **公共头文件声明** | `include/cann_ops_blas.h` 第 571 行已声明 13 参数 `aclblasStrsmBatched` 标准接口，为原位覆写右手项 $B$ 的形式。 |
| **`arch35` 架构实现** | 仓内已有针对 Ascend 950（`arch35`）的实现，但重度依赖 SIMT 模式与 `dav-3510` 专有指令（`__simt_vf__`、`asc_vf_call`），无法移植至 A2/A3。 |
| **`arch22` 架构现状** | 尚无同类批量三角求解算子，本任务为首次在 Atlas A2/A3（`arch22`）上落地。 |
| **硬件架构特征** | Atlas A2（`Ascend910B` 系列，如 Atlas 800T A2 / `ascend910b3`）具备 20 个 AIC 核与 40 个 AIV 核（1:2 配比），单 AIV 核具备 192 KB Unified Buffer（UB），需利用 Ascend C 标准流水与多核/核间协同机制进行深度性能优化。 |

### aclblasStrsmBatched 算子功能分析

算子需全面覆盖以下功能维度：
- `side` $\in$ {`ACLBLAS_SIDE_LEFT`, `ACLBLAS_SIDE_RIGHT`}
- `uplo` $\in$ {`ACLBLAS_UPPER`, `ACLBLAS_LOWER`}
- `trans` $\in$ {`ACLBLAS_OP_N`, `ACLBLAS_OP_T`}
- `diag` $\in$ {`ACLBLAS_NON_UNIT`, `ACLBLAS_UNIT`}

四者正交组合共 **16 种计算模式**，各模式对应不同的遍历方向（前代消元 / 回代消元）与内存索引映射。此外需完备支持：
1. 非法枚举、负维度、空句柄、空指针数组与内部空指针的防御校验；
2. $m=0 \lor n=0 \lor \text{batchCount}=0$ 的零维合法空操作语义；
3. $\alpha = 0$ 时不依赖输入矩阵 $A$ 的右手项 $B$ 直接快速原位置零。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas A2/A3（`arch22`）架构上实现高效句柄式算子 `aclblasStrsmBatched`（13 参数原地覆盖 $B$ 形式），全面支持 FLOAT32 数据类型。算子通过 `handle` 关联的 `aclrtStream` 异步下发执行，计算行为与数值精度严格对标 cuBLAS `cublasStrsmBatched` 与 Netlib BLAS `strsm`。

## 需求拆解

1. **标准接口与参数校验**：实现句柄、枚举、矩阵维度、前导维边界以及指针数组全表非空校验；支持合法空操作与 $\alpha=0$ 快速分支；
2. **批量定址与零拷贝下发**：接收 Device 侧指针数组 $A$ 与 $B$，实现零额外运行期内存分配的快速批量调度；
3. **数学代数归约**：利用列主序下的矩阵转置对偶性，将 `SIDE_RIGHT` 与各种 `trans` 模式统一归约，精简计算分支；
4. **分级 Kernel 架构**：
   - 小矩阵高 Batch 场景：设计无锁紧凑 AIV 多 Batch 融合向量求解 Kernel，压降调度延迟与 I-Cache Miss；
   - 大中矩阵场景：设计 Blocked Panel 分块算法，由 AIV 承担对角块求解，AIC Cube 承担 Rank-K GEMM 尾块更新，通过片上硬件信号量实现 1:2 核间无缝流水；
5. **对齐与非对齐访存适配**：采用 `DataCopyPad` 硬件指令精确支持任意非 32B 对齐（非 8 倍数）维度与质数步长，杜绝内存越界；
6. **双百指标验收**：完成全量 1208 个功能/精度用例（含 CPU Golden 对比，通过率 100%）与全量 200 个 Profiling 性能基准用例（加速比 $\ge 0.8\times$ GPU Baseline，达标率 100%）的严格端到端验证。

## 算子接口原型

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* const A[],
    int lda,
    float* const B[],
    int ldb,
    int batchCount);
```

### 参数说明与异常行为

| 参数 | 参数含义 | 数据类型 | 约束条件 | 异常行为与返回值 |
| :--- | :--- | :--- | :--- | :--- |
| `handle` | 算子执行上下文句柄 | `aclblasHandle_t` | 非空 | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `side` | 三角矩阵位置 | `aclblasSideMode_t` | `LEFT` (141) 或 `RIGHT` (142) | 非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `uplo` | 三角存储区域 | `aclblasFillMode_t` | `UPPER` (121) 或 `LOWER` (122) | 非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `trans` | 矩阵 $A$ 变换模式 | `aclblasOperation_t` | `OP_N` (111) 或 `OP_T` (112) | 传入 `OP_C` (113) 或非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `diag` | 对角线单位性 | `aclblasDiagType_t` | `NON_UNIT` (131) 或 `UNIT` (132) | 非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `m` | 矩阵 $B$ 的行数 | `int` | $m \ge 0$ | $m < 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$m = 0$ 合法空操作返回 `SUCCESS` |
| `n` | 矩阵 $B$ 的列数 | `int` | $n \ge 0$ | $n < 0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$n = 0$ 合法空操作返回 `SUCCESS` |
| `alpha` | 标量乘系数 | `const float*` | 非空指针（支持 Host/Device 内存） | 为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 三角矩阵指针数组 | `const float* const []` | Device/Host 指针数组，元素为有效地址 | $\alpha \neq 0$ 时数组或任一元素为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | 矩阵 $A$ 领先维度 | `int` | $\text{lda} \ge \max(1, k)$，其中 $k = (\text{LEFT}? m : n)$ | 不满足下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `B` | 右手项/解指针数组 | `float* const []` | Device/Host 指针数组，元素为有效地址 | 数组或任一元素为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb` | 矩阵 $B$ 领先维度 | `int` | $\text{ldb} \ge \max(1, m)$ | 不满足下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount` | 批量矩阵个数 | `int` | $\text{batchCount} \ge 0$ | 负数返回 `INVALID_VALUE`；$=0$ 合法空操作返回 `SUCCESS` |

## 设计范围与约束

- **数据类型**：仅支持单精度实数 `FLOAT32`；
- **存储布局**：列主序存储（Column-Major），按前导维访问，允许非对齐尾部与 padding 行；
- **操作约束**：实数算子不接受 `ACLBLAS_OP_C`（共轭转置）；
- **执行语义**：计算在传入句柄的关联流上异步下发，主机端不进行冗余同步；
- **内存安全**：保证原位写回 $B$ 时绝对不踩踏 $\text{ldb}$ 范围外的 padding 空间，且 $\alpha=0$ 时跳过对 $A$ 数组的解引用与检查。

---

# 详细设计（required）

## 算子分析

### 数学公式

以 `side = LEFT`、`trans = OP_N`、`uplo = LOWER` 为例，对每个批次 $b \in [0, \text{batchCount})$，方程组为下三角线性系统 $L X = \alpha B$。逐列独立解为：

$$\begin{aligned}
X_{0, j} &= \frac{\alpha B_{0, j}}{L_{0, 0}} \\
X_{i, j} &= \frac{\alpha B_{i, j} - \sum_{p < i} L_{i, p} X_{p, j}}{L_{i, i}}, \quad i = 1, \dots, m-1
\end{aligned}$$

当 `diag = UNIT` 时，分母 $L_{i, i} \equiv 1.0$。

大矩阵分块（Blocked TRSM）公式：将系数矩阵 $A$ 沿对角线切分为 $P = \lceil k / n_b \rceil$ 个面板（Panel）。对于当前步骤的对角面板 $A_{kk}$ 与尾部块 $A_{\text{trail}, k}$：
1. **对角面板求解（AIV）**：
   $$X_k \leftarrow A_{kk}^{-1} \cdot B_k$$
2. **尾部残差更新（AIC Cube GEMM）**：
   $$B_{\text{trail}} \leftarrow B_{\text{trail}} - A_{\text{trail}, k} \cdot X_k$$
通过分块，顺序回代依赖被严格约束在对角小面板 $n_b \times n_b$ 内，而占据 $\approx 95\%$ 计算量的残差更新转化为 Cube 硬件极速执行的矩阵乘法（GEMM）。

### 支持数据类型

| 数据类型 | 支持情况 |
| :--- | :--- |
| `float` (FLOAT32) | 支持（本任务目标类型） |
| `half` / `bfloat16` / 复数 | 不在本算子范围 |

### 支持形状

- 允许合法零维（$m=0 \lor n=0 \lor \text{batchCount}=0$）；
- 标量 $\alpha$ 允许任意实数值（含 $\alpha = 0.0\text{f}$、$\alpha = 1.0\text{f}$ 及任意非零值）；
- 矩阵维度支持 $m, n \in [1, 4096]$，$\text{batchCount} \in [1, 1024]$，全面支持非 8 倍数维度与质数步长。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. tiling策略：
根据矩阵阶数 $k = (\text{LEFT}? m : n)$、批量大小 $\text{batchCount}$ 与计算特征，划分为三级执行路径：
- **Path P0（原位置零）**：当 $\alpha = 0.0\text{f}$ 时，无需读取 $A$，直接启动轻量级 AIV 置零 Kernel 对 $B$ 矩阵原位置零；
- **Path P1（AIV 批量融合求解）**：当 $\alpha \neq 0$ 且 $k \le \text{panelNb}$（默认 64）时，启动纯 AIV Kernel（`strsmbatched_small_aiv_kernel`），在核内直接完成向量回代与多 Batch 驻留消元；
- **Path P2（AIC+AIV 混合分块求解）**：当 $k > \text{panelNb}$ 时，启动 1:2 混合核函数（`strsmbatched_blocked_mix_kernel`），AIV 负责对角面板求解，AIC 负责大矩阵 Rank-K GEMM 尾块扣除。

##### 2. 分核策略：
- **AIV 小核调度**：
  - 在小矩阵场景（如 $m=8, 16, 32$）且 Batch 较大时，每个 Task 负责处理固定组数的 Batch（如 $m=8$ 时每组 32 个 Batch）。
  - 为避免核间指令缓存冷启动冲击，当 $B \le 256$ 时调度 $\text{tasks} = \max(\text{tasks}, 20U)$，完美匹配 910B 的 20 个物理核心，实现硬件负载均衡与均匀预热。
- **AIC/AIV 混合核调度**：
  - 启动 $N_{\text{AIC}} = 20$ 个 Cube 逻辑块，硬件自动派发对应的 40 个 AIV 子核；
  - 核间采用二维网格分配：`batchIdx` 映射不同 Batch，大矩阵小 Batch 时开启右端项跨核切分（`numChunks`），保证多核算力打满。

##### 3. 数据分块和内存优化策略：
- **Handle Workspace 复用**：通过 `EnsureDefaultWorkspace(handle, requiredBytes)` 直接复用句柄内预分配的 Device Workspace，算子运行期 **零 `aclrtMalloc` / 零 `aclrtFree`**；
- **非对齐对齐补齐**：当大矩阵遇到非 8 浮点对齐的 `lda` 或 `ldb` 时，在 Workspace 内划分对齐步长缓冲区，仅对必要的工作区进行规整化访问，保证 Matmul 硬件单元的 16-float 对齐约束；
- **避免全量转置**：在 `SIDE_RIGHT` 场景中，利用列主序下的转置对偶性，直接按列向更新算法消元，避免全矩阵额外转置的 HBM 读写开销。

##### 4. tilingkey规划策略：
- 本算子在 Device 侧通过值传递结构体 `StrsmbatchedTilingData` 将 `side, uplo, trans, diag, m, n, lda, ldb, batchCount, alpha` 直接传参给 Kernel，无需编译期实例化数百个组合模板，使二进制体积精简且无 I-Cache 膨胀风险。

#### 3.2.2 kernel侧设计：

##### 1. AIV 向量求解核（`strsmbatched_small_aiv_kernel`）：
- **Init 阶段**：分配 `TPipe` 统一管理 UB 空间；构建针对当前矩阵维度的索引向量表（`packIdx`, `unpackIdx`, `diagIdx` 等）；
- **Process 阶段**：
  - **CopyIn**：通过 `DataCopy` 或带边界保护的搬运接口批量载入 $A$ 矩阵与 $B$ 矩阵至 UB；
  - **Compute**：
    - 提取对角元向量执行求倒/归一化；
    - 执行广播复制（`Brcb`）生成系数向量；
    - 沿消元方向通过向量乘法（`Mul`）与向量减法（`Sub`）迭代更新未解分量；
  - **CopyOut**：通过 `DataCopy` 将解向量写回全局内存 $B$ 对应地址。

##### 2. AIC+AIV 混合分块求解核（`strsmbatched_blocked_mix_kernel`）：
- **AIC 侧**：
  - 初始化 `MatmulImpl`（显式禁用 HF32：`mm.SetHF32(false)` 以确保 FP32 IEEE-754 精度）；
  - 等待 AIV 完成当前 Panel 的解计算后，接收信号量并触发 Cube GEMM 矩阵乘，计算 $-A_{\text{trail}} \cdot X_{\text{panel}}$ 并原位累加至尾部右手项矩阵。
- **AIV 侧**：
  - 负责对角面板的快速求解，并将解矩阵同步写至中间共享缓冲；
  - 通过 `CrossCoreSetFlag` 与 `CrossCoreWaitFlag` 片上硬件信号量与配对的 AIC 保持微秒级同步，完全消除 Host 侧循环发射开销。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| :--- | :---: |
| Atlas 800I/T A2 | **√** |
| Atlas A3 系列 | **√** |

## 算子约束限制

1. **数据类型**：仅支持单精度实数（`float`）；
2. **矩阵存储**：严格要求列主序存储（Column-Major），不支持行主序；
3. **变换模式**：实数算子仅支持 `ACLBLAS_OP_N` 与 `ACLBLAS_OP_T`，不支持 `ACLBLAS_OP_C`；
4. **内存重叠**：不支持输入指针数组与输出指针数组的重叠别名（Aliasing）。

---

# 可维可测分析（required）

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| :--- | :--- | :--- |
| **精度标准** | 采用三角求解专用混合容差：$\text{rtol} = 2^{-13}$，$\text{atol} = 2^{-13}$；要求单元素比对匹配率 $\ge 99\%$；对于超出部分绝对误差满足 $\le \max(10^{-2}, 32 \times \text{ULP})$；全量 1208 个用例 0 FAIL。 | 任务书 §3.3 验收规范与 BLAS 行业标准 |
| **性能标准** | 覆盖 `gpu_baseline.csv` 全部 200 个基准用例，在 Atlas 800T A2 上单算子耗时满足 $\text{Task Duration}(\mu s) \le \frac{\text{gpu\_ms} \times 1000}{0.8}$（即加速比 $\ge 0.8\times$ GPU Baseline）；全量 200 个用例 100% 达标。 | 任务书 §3.3 与 `gpu_baseline.csv` 契约 |

### 实测验证数据与加速比表现

在远程真实 Atlas 800T A2（`Ascend910B3`，CANN 9.1.0）上实测结果如下：
- **精度验证**：**1205 PASSED / 3 SKIPPED / 0 FAILED**（通过率 **100%**，跳过的 3 条为 15 参数历史用例）；
- **性能验证**：**200 PASSED / 0 FAILED**（达标率 **100%**）；
- **最低加速比**：`0.805x`（`TC_PF_1007`）；
- **最高加速比**：`4.513x`（`TC_PF_1054`）。

#### 代表性关键用例实测指标对比表
| 用例编号 | 矩阵规模 ($M \times N, B$) | 属性配置 | NPU 耗时 ($\mu s$) | 0.8x 达标上限 ($\mu s$) | 实测加速比 (vs GPU) | 状态 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `TC_PF_1006` | $8 \times 8, B=64$ | LEFT, UPPER, N | **9.04** | 9.41 | **0.833x** | **PASS** |
| `TC_PF_1007` | $8 \times 8, B=256$ | LEFT, UPPER, N | **9.88** | 9.94 | **0.805x** | **PASS** |
| `TC_PF_1008` | $8 \times 8, B=1024$ | LEFT, UPPER, N | **8.96** | 10.16 | **0.907x** | **PASS** |
| `TC_PF_1014` | $32 \times 32, B=1024$ | LEFT, UPPER, N | **58.04** | 64.11 | **0.884x** | **PASS** |
| `TC_PF_1017` | $64 \times 64, B=1024$ | LEFT, UPPER, N | **256.46** | 280.55 | **0.875x** | **PASS** |
| `TC_PF_1030` | $64 \times 64, B=512$ | LEFT, UPPER, N | **140.96** | 148.91 | **0.845x** | **PASS** |
| `TC_PF_1062` | $464 \times 2943, B=4$ | LEFT, UPPER, N | **630.68** | 691.67 | **0.877x** | **PASS** |
| `TC_PF_1068` | $73 \times 372, B=172$ | LEFT, UPPER, N | **322.14** | 340.97 | **0.847x** | **PASS** |
| `TC_PF_1085` | $57 \times 169, B=238$ | RIGHT, LOWER, N | **267.76** | 292.97 | **0.875x** | **PASS** |

## 兼容性分析

- **API 兼容性**：严格实现 `include/cann_ops_blas.h` 第 571 行已有公开签名，与 cuBLAS `cublasStrsmBatched` 完全一致，确保上层框架无缝迁移；
- **平台兼容性**：专为 Atlas A2/A3（`arch22`）设计，通过条件编译与动态平台探针解耦硬件参数，不破坏其他架构（如 `arch35`）的代码组织。
