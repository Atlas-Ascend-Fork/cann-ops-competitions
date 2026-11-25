# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务 2026 算子开发任务（任务编号：11-aclblasSCHgemm_A2A3），官方任务书为《aclblasSgemm / aclblasCgemm / aclblasHgemm 类算子开发任务书（Atlas A2/A3）》。

任务要求在昇腾 NPU（Atlas A2/A3 系列产品，含 Atlas 800T A2 / Atlas 800T A3）上，使用 Ascend C 编程语言并基于 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）工程框架，开发通用矩阵乘算子族：
- `aclblasSgemm`（FLOAT32 实数矩阵乘）
- `aclblasCgemm`（COMPLEX64 复数矩阵乘）
- `aclblasHgemm`（FLOAT16 半精度矩阵乘，`aclblasHalf`）

分别对标 NVIDIA cuBLAS 的 `cublasSgemm` / `cublasCgemm` / `cublasHgemm` 接口。采用 ops-blas 仓统一的句柄式 BLAS 接口工程模式，通过 `handle` 绑定 `stream` 直调 NPU kernel，完成设计、开发、测试全流程工作，验收通过后合入 ops-blas 开源仓。

## 背景介绍

### aclblasSCHgemm 算子实现优化

通用矩阵乘（GEMM: General Matrix Multiply）是高性能计算、数值代数与深度学习的基础核心计算单元（BLAS-3 核心）。基于 Ascend C 编程语言在 Atlas A2/A3 架构（arch22）上开发句柄式 BLAS 接口，能够充分发挥昇腾 AI Core 中 Cube 矩阵计算单元与 Vector 向量计算单元的硬件算力。

在 ops-blas 开源仓中，算子实现与测试代码组织如下：
- 实现代码路径：`blas/gemm/arch22/`
- 测试代码路径：`test/gemm/arch22/`（复用仓内通用 `gemm_test.cpp` 与测试框架）
- 公开接口头文件：`include/cann_ops_blas.h`
- 架构参考实现：`blas/gemm/arch35/cgemm_host.cpp`（Cgemm 参考）、`blas/gemm/arch35/gemm_host.cpp`（Sgemm 参考）、`blas/gemm/arch35/hgemmStridedBatched_host.cpp` / `hgemmStridedBatched_kernel.cpp`（FP16 计算路径参考）

### aclblasSCHgemm 算子实现现状分析

通过对 ops-blas 仓内 GEMM 算子族的功能分析与实现调研，当前支持的能力与接口现状如下：
1. `blas/gemm/arch22/gemm_host.cpp` 已导出 `aclblasSgemm`，但对于大矩阵全网格（WholeGrid）、非对齐列主序 leading dimension 以及 K 方向切分归约（Split-K）仍需进一步调优；
2. `blas/gemm/arch35/cgemm_host.cpp` 具备 Ascend 950PR 架构下的复数 GEMM 参考，但 arch22 尚未正式导出并完善适配 Atlas A2/A3 的 `aclblasCgemm`；
3. `blas/gemm/arch35/hgemmStridedBatched_host.cpp` 包含 FP16 计算路径参考，但在 `include/cann_ops_blas.h` 中非 batch 的 `aclblasHgemm` 接口声明尚不存在，且 arch22 缺乏针对 FP16 的直调高性能 Cube 矩阵乘实现；
4. 三个算子均采用统一的 `aclblasHandle_t` 句柄模式，接口参数序列与 cuBLAS 对应接口保持一一映射。

通过对 `aclblasSgemm`、`aclblasCgemm`、`aclblasHgemm` 算子功能分析，支持能力如下表所示：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | ops-blas 库上下文句柄，携带 stream | aclblasHandle_t | aclblasHandle_t | 非空有效句柄 | scalar |
| transa | op(A) 运算类型 | aclblasOperation_t | enum | {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} | scalar |
| transb | op(B) 运算类型 | aclblasOperation_t | enum | {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} | scalar |
| m | op(A) 与 C 的行数 | int | int32 | m ≥ 0 | scalar |
| n | op(B) 与 C 的列数 | int | int32 | n ≥ 0 | scalar |
| k | op(A) 的列数、op(B) 的行数 | int | int32 | k ≥ 0 | scalar |
| alpha | 标量乘数，Host 内存 | const T* | S: float; C: aclblasComplex; H: aclblasHalf | 非空指针 | scalar |
| A | 输入矩阵 A，列主序，Device 内存 | const T* | S: float; C: aclblasComplex; H: aclblasHalf | m>0 且 n>0 且 k>0 且 alpha≠0 时不可空 | op(A): m×k，物理: lda×k(N) 或 lda×m(T/C) |
| lda | A 的前导维度 | int | int32 | transa=N 时 lda ≥ max(1, m)；transa=T/C 时 lda ≥ max(1, k) | scalar |
| B | 输入矩阵 B，列主序，Device 内存 | const T* | S: float; C: aclblasComplex; H: aclblasHalf | m>0 且 n>0 且 k>0 且 alpha≠0 时不可空 | op(B): k×n，物理: ldb×n(N) 或 ldb×k(T/C) |
| ldb | B 的前导维度 | int | int32 | transb=N 时 ldb ≥ max(1, k)；transb=T/C 时 ldb ≥ max(1, n) | scalar |
| beta | 标量乘数，Host 内存；beta=0 时不读 C 旧值 | const T* | S: float; C: aclblasComplex; H: aclblasHalf | 非空指针 | scalar |
| C | 输出矩阵 C，列主序，Device 内存，原地覆写 | T* | S: float; C: aclblasComplex; H: aclblasHalf | m>0 且 n>0 时不可空，与 A/B 存储不重叠 | m×n，物理: ldc×n |
| ldc | C 的前导维度 | int | int32 | ldc ≥ max(1, m) | scalar |

计算公式：
$$C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C$$

### aclblasSCHgemm 算子功能分析

算子功能：计算矩阵乘加运算 $C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C$。

输入：`handle`、`transa`、`transb`、`m`、`n`、`k`、`alpha`、`A`、`lda`、`B`、`ldb`、`beta`、`C`（原地累加时作为旧值读入）、`ldc`。

输出：`C`（计算结果原地覆写写回 Device 内存）。

支持数据类型：
- `aclblasSgemm`: FLOAT32（`float`）
- `aclblasCgemm`: COMPLEX64（`aclblasComplex`，实部 real 与虚部 imag 各占一个 float32）
- `aclblasHgemm`: FLOAT16（`aclblasHalf`，IEEE 754 16位半精度浮点）

支持广播：不支持（标准 BLAS-3 语义，各操作数形状独立确定，不支持维度广播）。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，在昇腾 Atlas A2/A3 系列芯片（CANN 9.1.0，arch22）上开发句柄式接口 `aclblasSgemm`、`aclblasCgemm`、`aclblasHgemm`：
1. 接口与参数语义与 cuBLAS `cublasSgemm` / `cublasCgemm` / `cublasHgemm` 完全对齐，数学基准参考 Netlib BLAS `sgemm` / `cgemm` 与 IEEE 754 标准规范；
2. 矩阵均按列主序（Column-Major）存储，支持带 padding 的前导维度 `lda, ldb, ldc` 访问；
3. 支持 `transa/transb ∈ {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}`。其中 `aclblasCgemm` 中 `OP_C` 为共轭转置（虚部取反）、`OP_T` 为普通转置；`aclblasSgemm` 与 `aclblasHgemm` 为实数算子，`OP_C` 与 `OP_T` 语义等价；
4. 支持快速返回（quick return）边界条件：
   - $m = 0$ 或 $n = 0$ 为合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不引用 A/B/C 内存，不下发 NPU kernel；
   - $k = 0$ 或 $\alpha = 0$ 时不进行矩阵乘累加，退化为标量缩放 $C = \beta \cdot C$，此时允许 A/B 为 `nullptr`；
   - $\beta = 0$ 时不读取矩阵 C 的旧值（允许 C 初始包含非有限值 NaN/Inf）；
5. 精度 100% 满足开源精度标准（FLOAT32/COMPLEX64 满足 rtol=1.2e-4, atol=1.2e-4; FLOAT16 满足 rtol=1e-3, atol=1e-3; 匹配率 ≥ 99.0%），全量 1000 条用例 100% PASS；
6. 性能 100% 达标：在 Atlas 800T A2 (910B3) 上使用 `msprof op` 采集真实硬件底层 Task Duration，全量 200 条性能用例单次 kernel 平均耗时均满足 $\le 1.25 \times \text{gpu\_ms}$（即加速比 $\ge 0.8\times$），自测预留 5% 性能安全余量。

## 需求拆解

1. **接口声明与统一定义**：在 `include/cann_ops_blas.h` 中新增非 batch 版本的 `aclblasHgemm` 声明，保持已有 `aclblasSgemm`（L575）和 `aclblasCgemm`（L579）接口签名与符号不变，返回值统一为 `aclblasStatus_t`。
2. **Host 端严格防御性校验流水线**：遵循先结构性参数（handle -> transa/transb -> m/n/k -> lda/ldb/ldc -> alpha/beta 指针 -> A/B/C 指针）后 quick return 的原则，准确返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、`ACLBLAS_STATUS_INVALID_ENUM`、`ACLBLAS_STATUS_INVALID_VALUE` 等状态码。
3. **数据类型分发与多流水架构设计**：
   - **Sgemm（FLOAT32）**：深度利用 arch22 平台 Cube 单元的 Matmul 高性能计算，针对大尺寸全网格（WholeGridSquare）采用规则大瓦片切分，针对长 K 场景支持 Split-K 与双缓冲（Double Buffer），隐藏内存搬运延迟。
   - **Cgemm（COMPLEX64）**：复数按实部和虚部交错存储（Interleaved Complex: `re, im`）。采用实数解耦与复数四实乘分解流水（非 3M 分解）：Phase 0（AIV 向量解交错拆分）+ Phase 1（利用高度调优的实数 Sgemm Cube 路径并行执行 4 次实数乘）+ Phase 2（AIV 向量合并 Epilogue，精确处理 $\alpha, \beta$ 缩放与共轭符号因子），实现高吞吐。
   - **Hgemm（FLOAT16）**：针对 FP16 数据类型，深度发挥 Atlas A2 Cube 单元原生的 16-bit 矩阵乘极致算力（16×16×16 分形乘加），支持 FP32 高精度累加输出与 FP16 结果写回，大幅提升运算吞吐。
4. **内存与 Workspace 边界管控**：显式预估并申请 Scratch 空间，严格管控全局显存上限，杜绝越界写与内存泄漏；对 `ldc` 的 padding 区域实施严格的哨兵保护。
5. **可维可测与验收闭环**：基于 ATK 测试框架，覆盖精度 1000 条用例与性能 200 条用例，提供标准化复现步骤。

# 详细设计（required）

## 算子分析

### 数学公式

通用矩阵乘加数学公式为：
$$C = \alpha \cdot \text{op}(A) \cdot \text{op}(B) + \beta \cdot C$$

其中 $\text{op}(A)$ 的逻辑维度为 $m \times k$，$\text{op}(B)$ 的逻辑维度为 $k \times n$，$C$ 的逻辑维度为 $m \times n$。

#### 1. 实数路径（Sgemm / Hgemm）
对于实数元素 $i \in [0, m-1], j \in [0, n-1]$：
$$C_{i,j} = \alpha \sum_{p=0}^{k-1} \text{op}(A)_{i,p} \cdot \text{op}(B)_{p,j} + \beta \cdot C_{i,j}$$
其中：
- $\text{transa} = \text{ACLBLAS\_OP\_N}$ 时，$\text{op}(A)_{i,p} = A_{i,p}$；$\text{transa} \in \{\text{ACLBLAS\_OP\_T}, \text{ACLBLAS\_OP\_C}\}$ 时，$\text{op}(A)_{i,p} = A_{p,i}$；
- $\text{transb} = \text{ACLBLAS\_OP\_N}$ 时，$\text{op}(B)_{p,j} = B_{p,j}$；$\text{transb} \in \{\text{ACLBLAS\_OP\_T}, \text{ACLBLAS\_OP\_C}\}$ 时，$\text{op}(B)_{p,j} = B_{j,p}$。

#### 2. 复数路径（Cgemm）
复数矩阵按实部与虚部交错格式存储，设矩阵元素为 $A = A_r + i A_i$，$B = B_r + i B_i$。
对于转置操作，定义符号因子 $s_a, s_b \in \{+1, -1\}$：
- 当 $\text{trans} = \text{ACLBLAS\_OP\_C}$（共轭转置）时，$s = -1$，$\text{op}(X) = X_r^T - i X_i^T$；
- 当 $\text{trans} = \text{ACLBLAS\_OP\_T}$（普通转置）时，$s = +1$，$\text{op}(X) = X_r^T + i X_i^T$；
- 当 $\text{trans} = \text{ACLBLAS\_OP\_N}$（不转置）时，$s = +1$，$\text{op}(X) = X_r + i X_i$。

复数矩阵乘积 $P = \text{op}(A) \cdot \text{op}(B)$ 的实部与虚部分量展开为四次实数矩阵乘法：
$$\text{Re}(P) = \text{op}(A_r) \cdot \text{op}(B_r) - (s_a \cdot s_b) \cdot \text{op}(A_i) \cdot \text{op}(B_i)$$
$$\text{Im}(P) = s_b \cdot \text{op}(A_r) \cdot \text{op}(B_i) + s_a \cdot \text{op}(A_i) \cdot \text{op}(B_r)$$

最终结合复数标量 $\alpha = \alpha_r + i \alpha_i$ 与 $\beta = \beta_r + i \beta_i$ 原地更新矩阵 $C$：
$$C_{\text{new}} = (\alpha_r + i \alpha_i)(P_r + i P_i) + (\beta_r + i \beta_i) C_{\text{old}}$$
当 $\beta = 0$ 时，旧矩阵 $C_{\text{old}}$ 的值不被读取，直接计算 $C_{\text{new}} = \alpha \cdot P$。

### 支持数据类型

- `aclblasSgemm`: $\alpha, \beta, A, B, C$ 均为 FLOAT32（`float`，单精度浮点数）；
- `aclblasCgemm`: $\alpha, \beta, A, B, C$ 均为 COMPLEX64（`aclblasComplex`，实部/虚部各为 float32，按交错连续存放）；
- `aclblasHgemm`: $\alpha, \beta, A, B, C$ 均为 FLOAT16（`aclblasHalf`，IEEE 754 16位半精度浮点数，计算过程采用 FP32 累加保证精度）。

### 支持形状

- 矩阵尺寸 $m, n, k$ 为运行时入参，支持非负整数范围：$m \ge 0, n \ge 0, k \ge 0$；
- 矩阵存储格式为列主序（Column-Major），带前导维度 $lda, ldb, ldc$：
  - $\text{transa} = \text{ACLBLAS\_OP\_N}$ 时，物理行容量 $lda \ge \max(1, m)$；$\text{transa} \in \{\text{ACLBLAS\_OP\_T}, \text{ACLBLAS\_OP\_C}\}$ 时，$lda \ge \max(1, k)$；
  - $\text{transb} = \text{ACLBLAS\_OP\_N}$ 时，物理行容量 $ldb \ge \max(1, k)$；$\text{transb} \in \{\text{ACLBLAS\_OP\_T}, \text{ACLBLAS\_OP\_C}\}$ 时，$ldb \ge \max(1, n)$；
  - 输出矩阵 $C$ 的物理行容量 $ldc \ge \max(1, m)$；
- 覆盖典型边界形状（0、1、2、质数、2的幂次、2的幂加减1，以及 4096 规模大矩阵）；
- 不支持维度广播，不支持非连续任意 stride，输出矩阵 $C$ 与输入矩阵 $A, B$ 物理存储不得重叠。

## 算子实现

### 实现方案

总体架构采用“Host 侧参数校验与动态 Tiling 调度 + Cube 矩阵乘硬件流水 + AIV 向量拆解/合并”的多层协同架构：
1. **Host 侧**：负责入参防御性校验、quick return 快速路径识别、平台资源查询（AICore/AIV 核心数与片上内存）、动态 Tiling 生成及任务流下发；
2. **Kernel 侧**：
   - 对于实数 Sgemm / Hgemm，直接驱动 AIC Cube 单元执行高效分块矩阵乘法与 Double Buffer 流水线；
   - 对于复数 Cgemm，构建 Phase 0（AIV 解交错拆分）-> Phase 1（AIC 4 次实数 GEMM）-> Phase 2（AIV 向量合并与共轭缩放）三阶段流水线，复用高度优化的 Sgemm Cube 核。

#### 3.2.1 host侧设计：

tiling策略：

1. **防御性校验顺序**：
   - 检查 `handle` 指针：为 `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - 检查 `transa, transb` 枚举值：不在 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` 时返回 `ACLBLAS_STATUS_INVALID_ENUM`；
   - 检查维度尺寸：$m < 0$ 或 $n < 0$ 或 $k < 0$ 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - 检查前导维度：$lda, ldb, ldc$ 低于合法下界时返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - 检查标量指针：`alpha` 或 `beta` 为 `nullptr` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；
   - 检查数据指针：当 $m > 0$ 且 $n > 0$ 时，$C$ 不可为空；当 $m > 0$ 且 $n > 0$ 且 $k > 0$ 且 $\alpha \ne 0$ 时，$A, B$ 不可为空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. **快速返回与路径分流**：
   - $m = 0$ 或 $n = 0$：合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不下发任何 Kernel；
   - $k = 0$ 或 $\alpha = 0$：跳过矩阵乘阶段，直接调用向量标量缩放 Kernel（`aclblasSscal` / `aclblasCscal` / `aclblasHscal`）执行 $C = \beta \cdot C$（若 $\beta = 1$ 则直接 no-op）；
   - $\beta = 0$：通知计算 Kernel 关闭 $C$ 旧值加载，直接写回新值，防止 $C$ 初始 NaN/Inf 污染结果。
3. **列主序到硬件友好布局转换**：
   根据列主序矩阵乘性质，计算 $C = \text{op}(A) \cdot \text{op}(B)$ 等价于其行主序视角的 $C^T = \text{op}(B)^T \cdot \text{op}(A)^T$。Host 侧根据转置类型与尺寸，调整操作数映射与 stride 参数，使 Cube 单元以最高吞吐的连续行块进行计算。

任务均分：
coreNum 根据输入规模与片上切分粒度动态调整。获取当前芯片物理可用 AI Core 数量（Atlas 800T A2 为 24 或 30 核），将 $M \times N$ 分解为二维网格宏块。总任务块数 $T = \lceil M / M_{\text{tile}} \rceil \times \lceil N / N_{\text{tile}} \rceil$。分核数取 $\text{blockDim} = \min(T, \text{coreNum})$。各核通过循环调度均匀处理各自的任务块，余数块轮转分摊，确保满核运行与核间负载严格均衡。

批量搬运：
利用 Ascend C 的 `DataCopy` / `DataCopyPad` 接口实现 32 字节（FLOAT32 为 8 个元素、COMPLEX64 为 4 个复数、FLOAT16 为 16 个元素）连续对齐传输。针对列主序 $lda, ldb, ldc$ 带来的非连续步长，在 UB 内开辟连续缓冲，按块批量搬运有效数据，尾块及非整块通过硬件 Mask 保护，杜绝越界内存踩踏。

##### 1. 分核策略：
遵循优先占满核原则。
- **2D 网格切分**：将输出矩阵 $M \times N$ 划分为 $M_{\text{tile}} \times N_{\text{tile}}$ 宏块：
  - 若核间能均分，各核处理的任务块数一致；
  - 若核间不能均分，通过扁平化一维任务编号 `taskId` 进行模调度，多出的任务块平摊到前几个核心；
- **长 K 维度 Split-K 分核**：当 $M, N$ 极小而 $K$ 极大导致输出网格无法铺满所有核心时，启用 Split-K 策略：将 $K$ 维切分为 $K_{\text{splits}}$ 段，分配至不同核心分别计算局部部分和，最终通过专用归约 Kernel 或原子累加合并至矩阵 $C$。

##### 2. 数据分块和内存优化策略：
遵循充分使用片上缓存（UB、L1、L0A、L0B、L0C）原则。
- **分块尺寸规划**：
  - Sgemm（FLOAT32）：设置基础分块 $M_{\text{tile}} = 128, N_{\text{tile}} = 128, K_{\text{tile}} = 64$（或针对特大方阵启用 WholeGridSquare $256 \times 256$ 切分）；
  - Hgemm（FLOAT16）：设置分块 $M_{\text{tile}} = 128, N_{\text{tile}} = 256, K_{\text{tile}} = 64$，充分释放 Cube 单元 16×16×16 分形矩阵乘加极致吞吐；
  - Cgemm（COMPLEX64）：通过 Workspace 分配按 64 字节对齐的 $A_r, A_i, B_r, B_i$ 连续实数平面，使得 Phase 1 的实数 GEMM 能复用 Sgemm 经过高度调优的 Cube 分块流水。
- **Double Buffer（双缓冲）机制**：
  在片上缓冲为输入数据和累加输出开辟 Ping-Pong Buffer，使 MTE2 搬运、Cube 矩阵计算和 MTE3 写回形成三级流水，实现计算时延与搬运时延完全掩盖。
- **显式 Workspace 显存上限管控**：
  通过 `EnsureDefaultWorkspace` 集中申请与管理临时工作区，设定全局安全硬上限（如 2 GiB），超出时主动拦截并返回 `ACLBLAS_STATUS_ALLOC_FAILED`，防止静默内存溢出。

##### 3. tilingkey规划策略：
根据 Host 侧获取的算子类型、数据类型、矩阵尺寸及转置模式分发不同的核函数分支：
- `tilingKey = 0`：退化/微形状快速路径（$k=0$ 或 $\alpha=0$ 的向量标量缩放路径，或微小尺寸纯向量计算）；
- `tilingKey = 1`：标准通用分块 Cube 矩阵乘路径（双缓冲通用矩阵乘流水）；
- `tilingKey = 2`：大方形全网格专用路径（WholeGridSquare，极大矩阵的高效对齐流水）；
- `tilingKey = 3`：Split-K 路径（长 K、短 M/N 场景，核间分段并行累加）；
- `tilingKey = 4`：复数专用 AIV 解交错与合并路径（Cgemm split & combine）。

#### 3.2.2 kernel侧设计：

算子 Kernel 采用标准 `Init` 与 `Process` 两阶段设计，其中 `Process` 阶段包含数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三级流水：

1. **Sgemm Kernel 设计**：
   - **CopyIn**：通过 MTE2 将全局内存中 $A$ 和 $B$ 的分块数据经由 L1 搬入 L0A 与 L0B 缓冲，支持转置分形重排；
   - **Compute**：驱动 Cube 单元执行矩阵乘累加 $C_{\text{tile}} = A_{\text{tile}} \times B_{\text{tile}}$；
   - **CopyOut**：通过 Fixpipe 将 L0C 中累加结果搬运至 UB 或 GM，结合 Vector 单元完成 $\alpha$ 乘法与 $\beta \cdot C$ 原地合并，精准写回有效区域，保护 $ldc$ padding 列不被污染。
2. **Cgemm Kernel 设计**：
   - **Phase 0（AIV 解交错拆解）**：启动 AIV 向量核，批量读取交错存储的复数矩阵 $A$ 与 $B$，通过向量解交错指令将实部与虚部分离，写入 Workspace 中 64 字节行对齐的实数平面 $A_r, A_i, B_r, B_i$；
   - **Phase 1（AIC Cube 实数 GEMM）**：在绑定的执行流中下发 4 次实数 Sgemm 计算：$A_r B_r$、$A_i B_i$、$A_r B_i$、$A_i B_r$；
   - **Phase 2（AIV 向量合并 Epilogue）**：AIV 向量核读取 4 个中间结果平面，根据共轭符号因子 $s_a, s_b$ 进行交叉相加减，完成复数标量 $\alpha, \beta$ 缩放，并原地写回目标复数矩阵 $C$。
3. **Hgemm Kernel 设计**：
   - 充分发挥 Atlas A2 Cube 单元原生 FP16 矩阵计算能力，输入为 FP16，在 L0C 内部以 FP32 高精度累加，最后在 Fixpipe/UB 阶段量化为 FP16 写回，兼顾极致性能与数值精度稳定性。
4. **核内同步与内存屏障**：
   通过 `SetFlag` / `WaitFlag` 和 `PipeBarrier` 严格维护各级存储间事件同步，杜绝 RAW/WAR 竞争冒险。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 800I/T A3 | √ |

## 算子约束限制

1. 矩阵数据必须按列主序（Column-Major）存储，矩阵行间仅支持通过前导维度 $lda, ldb, ldc$ 描述的规则间隙，不支持任意非连续 stride 或多维张量视图；
2. 输出矩阵 $C$ 的物理存储区域不得与输入矩阵 $A$ 或 $B$ 重叠（严格执行原地更新 $C$ 语义）；
3. 不支持维度广播运算；
4. 异步执行约束：算子在 `handle` 绑定的 stream 上异步执行，接口返回表示成功下发，Host 侧读取计算结果前必须显式同步 stream；
5. 输入矩阵各元素须为可表示的有限数值（特殊用例中对 NaN/Inf 按照 BLAS 规范进行浮点传递）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 1. `aclblasSgemm`：以 CPU Netlib `cblas_sgemm` 为参考，按 FLOAT32 阈值（rtol=1.2e-4, atol=1.2e-4, 匹配率 ≥ 99.0%，最大绝对误差限 ≤ 1e-2 或 32 ULP）比对；<br>2. `aclblasCgemm`：以 CPU Netlib `cblas_cgemm` 为参考，实部与虚部分量分别按 FLOAT32 阈值比对，匹配率 ≥ 99.0%；<br>3. `aclblasHgemm`：以高精度 Float64/FP32 累加收窄至 FP16 的 CPU 参考（对齐 cuBLAS `cublasHgemm`），按 FLOAT16 阈值（rtol=1e-3, atol=1e-3, 匹配率 ≥ 99.0%，最大绝对误差限 ≤ 1e-1 或 32 ULP）比对。<br>全量 1000 条用例 100% PASS。 | 生态算子开源精度标准（Mixed Tolerance 混合容差标准）与官方任务书 §3.2 |
| 性能标准 | 在 Atlas 800T A2 (910B3) 上，性能测试在预置 Device 缓冲上执行，通过 `msprof op` 真实采集硬件底层单算子 kernel Task Duration（微秒），采用统一标准采样协议（冷态单拉，5 次 warmup + 10 次以上有效样本取算术平均作为 Avg time）。<br>验收判据：全量 200 条性能用例单次 kernel 平均耗时均 $\le 1.25 \times \text{gpu\_ms}$（即加速比 $\ge 0.8\times$），自测严格预留 5% 性能安全余量（加速比 $\ge 0.84\times$）。GTest 端到端耗时 `[PERF] avg_us` 作为辅助交叉验证。 | 官方任务书 §3.3 与 `test_cases/gpu_baseline.csv` |

## 兼容性分析

1. **接口与 ABI 兼容性**：算子接口严格遵循 ops-blas 仓现有的 C ABI 规范与 `include/cann_ops_blas.h` 声明，`aclblasSgemm` 与 `aclblasCgemm` 保持原有函数签名不变，`aclblasHgemm` 按照同族规范新增声明，与既有系统二进制完全兼容；
2. **源码隔离与模块解耦**：所有实现代码严格放置在 `blas/gemm/arch22/` 目录下，与 Ascend 950（arch35）实现物理隔离，不引入 arch35 私有指令与头文件，确保 Atlas A2/A3 平台上独立、纯净、平稳地编译构建与运行；
3. **测试兼容性**：复用 ops-blas 仓内通用的 `gemm_test.cpp` 与 ATK 测试基础设施，测试用例驱动与验证脚本与仓内其他算子完全一致。
