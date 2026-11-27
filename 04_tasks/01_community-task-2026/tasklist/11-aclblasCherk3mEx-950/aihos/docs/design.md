# 需求背景（required）

## 需求来源

本需求来源于昇腾 2026 年 11 月社区任务[11-aclblasCherk3mEx-950](https://www.hiascend.com/activities/task-center/details/5f0045789d554794aca26d146e93d6b7)。目标是在 **Ascend 950PR（`arch35` / `NpuArch::DAV_3510`，SOC=ascend950）** 硬件架构与 CANN 9.1.0 及以上（`asc-devkit >= 9.1`）软件栈环境下，基于 `cann/ops-blas` 开源仓句柄式 BLAS 接口工程模式（通过 handle 绑定 stream 直调 NPU kernel），开发单精度复数 Hermitian 矩阵秩-k 更新扩展版算子（采用 3M 快速算法）`aclblasCherk3mEx`，对标 cuBLAS `cublasCherk3mEx` 及 Netlib `cherk` 规范，完成设计、开发、测试全流程，验收通过后正式合入开源主仓。

- **代码落位路径**：实现源码位于 `blas/herk/arch35/`（新增独立源文件 `cherk3m_ex_*`）；测试代码位于 `test/herk/cherk3m_ex/arch35/`；公开接口在 `include/cann_ops_blas.h` 新增；算子说明增补至 `blas/herk/README.md`；
- **对标开源基线**：cuBLAS `cublasCherk3mEx`（支持 `ACL_COMPLEX64` 与 `ACL_INT8` 双类型输入）；数学语义与判定标准对齐 Netlib `cherk`（https://www.netlib.org/blas/cherk.f ）；
- **核心开发与验收目标**：满足任务书 3M 单次 Kernel Launch 强约束（三次实数矩阵乘必须在同一次 Kernel Launch 内完成，0 字节全局 Workspace），全量 1000 条 CSV 精度用例 100% 通过（Mixed Tolerance 阈值），全量 200 条 `TC_PF` 性能用例 100% 达标（单例倍率 $\ge 0.4000\times$，均值倍率 $\ge 0.5000\times$，内部预留 5% 安全余量）。

## 背景介绍

### aclblasCherk3mEx算子实现优化

复数 Hermitian 矩阵秩-k 更新（BLAS-3 HERK）是量子化学计算、5G/6G MIMO 信号处理、声学与电磁场仿真中的核心密集算力瓶颈。传统复数矩阵乘法采用 4M 算法（4 次实数乘法与 2 次实数加法）：
$$(X_r + i X_i)(Y_r + i Y_i) = (X_r Y_r - X_i Y_i) + i (X_r Y_i + X_i Y_r)$$
在 Hermitian 更新 $C \leftarrow \alpha X X^H + \beta C$ 中，$Y = X^H = X_r^T - i X_i^T$。利用 Hermitian 共轭对称性，通过 3M 变换：
$$P_0 = X_r X_r^T, \quad P_1 = X_i X_i^T, \quad P_2 = (X_r + X_i)(X_r - X_i)^T$$
$$\text{Re}(X X^H) = P_0 + P_1, \quad \text{Im}(X X^H) = P_2 - P_0 + P_1 = X_i X_r^T - X_r X_i^T$$
理论上将实数乘法次数从 4 次压降至 3 次，计算量降低 **25%**。

然而，在 Ascend 950PR 上实现高性能 3M 存在三大严峻挑战：
1. **多 Kernel 拼接与巨额内存开销**：传统实现往往拆分为输入解交错、多次 GEMM Launch、输出组合共 6 次 Launch，并在 HBM 分配高达数百兆字节的临时 Workspace，导致调度延迟暴涨与带宽严重浪费；
2. **两极性能分化**：分析 200 条 `gpu_baseline.csv` 发现，小矩阵与超薄 K 场景（如 $n \le 64$ 或 $k \le 16$）GPU 耗时仅 2~6 微秒，Cube 握手与搬运开销极易击穿预算；而大矩阵（如 $4096 \times 4096$）GPU 算力高达 27.67 TFLOPS，必须实现极致算力利用；
3. **数值相消与非有限异常**：3M 算法中 $P_2 - P_0 + P_1$ 在大值相消场景容易放大浮点截断误差，且遇到 $\pm\text{FLT\_MAX}$ 或 $\text{Inf}/\text{NaN}$ 时易提前触发 $\text{Inf} - \text{Inf} = \text{NaN}$。

本方案针对上述挑战展开深度系统级优化：
- **单次 Launch On-Chip 3M 融合架构**：利用 Ascend 950PR 原生 `KERNEL_TYPE_MIX_AIC_1_2` 架构（1 个 Cube 核与 2 个 Vector 核紧耦合），在 L1 片上完成三通道变换，通过 `Fixpipe` 直连写回 UB 消费，**全路径 0 字节全局 Workspace**，三次实数乘完全融合在单次 Launch 内；
- **全尺度 4 路径快慢分流**：针对极小规模构建 AIV-only SIMT Direct 3M 极速路径，针对中大形状构建 MIX On-Chip 3M 路径（支持 `TILE=32`、`TILE=64` 与超薄 K `TILE=128`），无缝覆盖全参数区间；
- **3M 数值安全三重防护**：强制关闭 HF32 截断模式 + 有界分段 K 归约（`reductionSpan`）抑制灾难性相消 + Device 端零开销指数探测并在同 Kernel 内触发 SIMT 精确重算，彻底对齐 Netlib 标杆。

### aclblasCherk3mEx算子ops-blas现有实现现状分析

当前 `cann/ops-blas` 开源仓中：
1. **公共接口声明缺失**：`include/cann_ops_blas.h` 当前尚未提供 `aclblasCherk3mEx` 接口原型（已有基础非 Ex 接口 `aclblasCherk`），本任务需按规范在公共头文件增补该声明；
2. **arch35 现有实现不满足约束**：仓内既有基础版 `blas/herk/arch35/cherk_host.cpp` 采用“1 次 Deinterleave + 4 次 GEMM + 1 次 Combine”的 6 次 Kernel Launch 流程，需申请高达 $8nk + 16n^2$ 字节（$4096 \times 4096$ 时需 384 MB）的全局显存，且 `aclblasCherkEx` 尚未合入，严禁依赖；既有实现无法满足本任务“3M 三次实数乘必须在单次 Kernel Launch 内完成”的硬性规定；
3. **跨算子复用基建**：仓内 `blas/gemm/arch35/` 的 `KERNEL_TYPE_MIX_AIC_1_2` 原生片上双缓冲流水线与 `Fixpipe` 控制机制经过充分验证，可作为 Cube/Vector 协同的基础设施复用。

### aclblasCherk3mEx算子功能分析

`aclblasCherk3mEx` 算子执行单精度复数 Hermitian 矩阵秩-k 原地更新：
$$C \leftarrow \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(A)^H + \beta \cdot C$$

- **数学与语义特性**：
  - $\alpha, \beta$ 恒为**实数 FLOAT32 标量**（支持 Host 指针或 Device 指针传入，允许同为 Host 或同为 Device，禁止混用）；
  - $A$ 为输入复矩阵，`trans = ACLBLAS_OP_N` 时为 $n \times k$，`trans = ACLBLAS_OP_C` 时为 $k \times n$；在实数与复数 Hermitian 语义下，`ACLBLAS_OP_T` 无数学定义，传入统一返回 `ACLBLAS_STATUS_INVALID_ENUM`；
  - $C$ 为 $n \times n$ 的复数 Hermitian 矩阵，物理存储严格按列主序排布；
  - **Hermitian 语义约束**：算子仅对 `uplo` 参数指定的下三角（`ACLBLAS_LOWER`）或上三角（`ACLBLAS_UPPER`）区域进行读写更新；严格禁止改写非 `uplo` 三角的任何元素及 `ldc > n` 的 padding 内存；
  - **对角线纯实数保证**：根据 Hermitian 矩阵定义，$C$ 的主对角元素虚部恒为零。在计算完成后，主对角元素虚部必须强制清零（含 $\beta \ne 0$ 以及 $\alpha = 0$ 或 $k = 0$ 且 $\beta \ne 1$ 的缩放路径）；
  - **数据类型支持**：输入 $A$ 支持 `ACL_COMPLEX64`（每个复元素 8 字节交错 `float re, im`）与扩展类型 `ACL_INT8`（每个复元素 2 字节交错 `int8_t re, im`）；输出 $C$ 恒为 `ACL_COMPLEX64`。

# 需求分析（required）

## 需求描述

在 Ascend 950PR（`arch35`）上，使用 Ascend C 编程语言开发并在 `include/cann_ops_blas.h` 发布标准句柄式接口：

```cpp
aclblasStatus_t aclblasCherk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const float* alpha,
    const void* A, aclDataType Atype, int lda,
    const float* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

**输入输出参数与属性规格表**：

| 参数名 | 方向/属性 | 描述与语义 | 数据类型与格式 | 取值范围与约束 | 异常行为与状态码 |
| --- | --- | --- | --- | --- | --- |
| `handle` | 输入 | ops-blas 库上下文句柄，携带 `aclrtStream` | `aclblasHandle_t` | 指向已初始化的有效句柄 | `nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 属性 | 三角矩阵存储与更新区域 | `aclblasFillMode_t` | `{ACLBLAS_UPPER, ACLBLAS_LOWER}` | 其他非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `trans` | 属性 | 矩阵运算模式 | `aclblasOperation_t` | `{ACLBLAS_OP_N, ACLBLAS_OP_C}` | `ACLBLAS_OP_T` 或其他值返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | 属性 | 矩阵 $C$ 的逻辑阶数 | `int` | $n \ge 0$ | $<0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$=0$ 触发 quick return |
| `k` | 属性 | 秩-k 更新的累加归约深度 | `int` | $k \ge 0$ | $<0$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；$=0$ 跳过 $A$ |
| `alpha` | 输入 | 实数标量乘数 | `const float*` | 有限实数，支持 Host/Device 指针 | 为空或与 beta 混用返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | 输入 | 输入复矩阵数据指针（Device 侧） | `const void*` | 列主序，根据 `trans` 为 $n \times k$ 或 $k \times n$ | $n, k > 0$ 且 $\alpha \ne 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Atype` | 属性 | $A$ 矩阵的数据类型枚举 | `aclDataType` | `{ACL_COMPLEX64, ACL_INT8}` | 其他枚举返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `lda` | 属性 | $A$ 矩阵前导维（单位：复元素） | `int` | `OP_N` 时 $\ge \max(1, n)$；`OP_C` 时 $\ge \max(1, k)$ | 低于下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | 输入 | 实数标量乘数 | `const float*` | 有限实数，支持 Host/Device 指针 | 为空或与 alpha 混用返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `C` | 输出(原地) | $n \times n$ 列主序复数 Hermitian 矩阵指针 | `aclblasComplex*` | 仅 `uplo` 三角读写更新，对角虚部恒清零 | $n > 0$ 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Ctype` | 属性 | $C$ 矩阵的数据类型枚举 | `aclDataType` | `ACL_COMPLEX64` | 其他枚举返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `ldc` | 属性 | $C$ 矩阵前导维（单位：复元素） | `int` | $\text{ldc} \ge \max(1, n)$ | 低于下界返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 需求拆解

1. **Hermitian 3M 单次 Kernel Launch 融合**：坚决摒弃多 GEMM 拆分方案，基于 `KERNEL_TYPE_MIX_AIC_1_2`，在 AIC/AIV 内部完成数据解包、$(X_r, X_i, X_+, X_-)$ 片上组装、三次实数乘加及 Epilogue 融合写回，单次 Launch，**0 字节全局 Workspace**；
2. **列主序到行主序零转置映射**：基于 $C^T = (X X^H)^T = X_- X_+^T$ 等价变换，将列主序输出矩阵直接映射为 Cube 行主序计算网格，实现**零额外显存转置、零虚部符号颠倒**；
3. **`ACL_INT8` 片上无损提升**：交错 `int8` 数据从 GM 搬入 AIV UB 后，向量化精确提升为 FP32（$[-128, 127]$ 完全在浮点无损表示范围内），复用全套 3M 变换，输入流量仅为 FP32 的 1/4 且保持 $2^{-13}$ 满精度；
4. **全尺度 4 路径快慢分流架构**：
   - Path 0：Quick Return（$n=0$ 或 $(k=0 \lor \alpha=0) \land \beta=1$），Host 零 Launch 返回；
   - Path 1：Scale/Zero-only AIV 路径（$k=0 \lor \alpha=0 \land \beta \ne 1$），纯向量处理，$\beta=0$ 不读旧 $C$；
   - Path 2：Direct 3M SIMT AIV 路径（$n \le 64$ 或小数据量），无 Cube 握手开销，微秒级延迟响应；
   - Path 3：MIX On-Chip 3M 路径（中大形状），根据形状分派 `Mix64`（$64 \times 64$）与 `Mix128`（$128 \times 128$，专攻 $k \le 32$ 薄 K 场景）；
5. **数值安全与非有限防护**：
   - 强制禁用 AIC HF32，规避 10-bit 截断精度漂移；
   - 有界分段 K 归约（`reductionSpan`）将大数相消误差隔离在局部；
   - 片内指数非有限检测与 SIMT 逐分量修复，彻底对齐 Netlib 标杆；
6. **双百达标验证**：1000 条精度用例 100% 通过（哨兵零破坏、对角虚部清零）；200 条性能用例 100% 达标（统一 18-Launch 采样，算术均值加速比 $\ge 0.5000\times$，单例 $\ge 0.4000\times$ 并留 5% 安全余量）；3 阶段显存采样 0 泄漏。

# 详细设计（required）

## 算子分析

### 数学公式

1. **基本运算定义**：
   $$C \leftarrow \alpha \cdot \operatorname{op}(A) \cdot \operatorname{op}(A)^H + \beta \cdot C, \quad \alpha, \beta \in \mathbb{R}$$
   其中 $X = \operatorname{op}(A) = X_r + i X_i \in \mathbb{C}^{n \times k}$：
   - 当 `trans = ACLBLAS_OP_N` 时：$X = A$，即 $X_r = A_r, X_i = A_i$；
   - 当 `trans = ACLBLAS_OP_C` 时：$X = A^H$，即 $X_r = A_r^T, X_i = -A_i^T$（通过符号因子 $\text{sign} = -1.0\text{f}$ 统一代数形式）。
2. **Hermitian 3M 代数推导**：
   复数秩-k 乘积展开为：
   $$X X^H = (X_r + i X_i)(X_r^T - i X_i^T) = (X_r X_r^T + X_i X_i^T) + i (X_i X_r^T - X_r X_i^T)$$
   定义实数辅助变量：
   $$X_+ = X_r + X_i, \quad X_- = X_r - X_i$$
   构造三次实数矩阵乘法：
   $$P_0 = X_r X_r^T, \quad P_1 = X_i X_i^T, \quad P_2 = X_+ X_-^T = (X_r + X_i)(X_r - X_i)^T$$
   展开 $P_2 = P_0 - P_1 + (X_i X_r^T - X_r X_i^T)$，由此得到唯一的 Hermitian 3M 表达式：
   $$\operatorname{Re}(X X^H) = P_0 + P_1$$
   $$\operatorname{Im}(X X^H) = P_2 - P_0 + P_1 = X_i X_r^T - X_r X_i^T$$
   对角线元素处，因为 $(P_0 + P_1)_{j,j} \in \mathbb{R}$ 且 $(X_i X_r^T - X_r X_i^T)_{j,j} = X_{i, j:} X_{r, j:}^T - X_{r, j:} X_{i, j:}^T = 0$，虚部在数学上严格自抵消。
3. **列主序到行主序零转置映射**：
   列主序矩阵 $C[i + j \cdot ldc]$ 等价于行主序矩阵 $C^T[j, i]$。对 3M 三次乘积转置：
   $$(P_0)^T = P_0, \quad (P_1)^T = P_1, \quad (P_2)^T = (X_+ X_-^T)^T = X_- X_+^T$$
   因此 Cube 计算时：
   - 左操作数（行主序行，对应输出列 $j$）：组装通道 $(X_r, X_i, X_-)$；
   - 右操作数（行主序列，对应输出行 $i$）：组装通道 $(X_r, X_i, X_+)$；
   Cube `Mmad` 计算出的输出即为行主序的 $C^T[j, i]$，直接对应列主序内存地址，完全消除额外转置与虚部符号反转。

### 支持数据类型

| 组合编号 | 输入矩阵 $A$ 数据类型 | 输出矩阵 $C$ 数据类型 | 标量 $\alpha, \beta$ 类型 | 内部计算精度 | 内存与流量特征 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `float` (FP32 实数) | FP32 (禁用 HF32) | $A$ 为 8 字节交错复数，标准 3M 变换 |
| **2** | `ACL_INT8` | `ACL_COMPLEX64` | `float` (FP32 实数) | FP32 (禁用 HF32) | $A$ 为 2 字节交错 `int8_t`，片上无损提升为 FP32，输入带宽减少 75% |

### 支持形状

1. **矩阵规模**：$n \ge 0, k \ge 0$，支持从极小矩阵（$n=1..64$）、超薄 K（$k=1..16$）、非对称非对齐（素数与非 16 对齐维度）到大矩阵（$n, k \le 4096$）；
2. **前导维约束**：$lda \ge \max(1, n)$（`OP_N`）或 $lda \ge \max(1, k)$（`OP_C`）；$ldc \ge \max(1, n)$；支持任意合法带 Padding 排布；
3. **三角模式**：全面支持 `ACLBLAS_UPPER` 与 `ACLBLAS_LOWER`。

## 算子实现

### 实现方案

```
+----------------------------------------------------------------------------------------------------+
|                                aclblasCherk3mEx 端到端执行架构                                     |
+----------------------------------------------------------------------------------------------------+
  【Host 侧】
    1. 严格参数校验流水线（handle -> uplo -> trans -> n/k/lda/ldc -> Atype/Ctype -> alpha/beta）
    2. 空维快速返回（n == 0 -> 返回 SUCCESS）
    3. 标量指针自动探测（Host 或 Device 内存属性识别，读取实数真值）
    4. 设备张量非空检查（C 非空；k>0 且 alpha!=0 时 A 非空）
    5. 纯恒等快速返回（(k == 0 || alpha == 0) && beta == 1 -> 返回 SUCCESS）
    6. 全尺度 4 路径快慢分流决策：
       - Path 1: 纯缩放/清零 (k==0 || alpha==0 && beta!=1) -> AIV Scale Kernel
       - Path 2: 极小/低延迟 (n <= 64) -> AIV-only SIMT Direct 3M Kernel
       - Path 3: 中大形状 -> KERNEL_TYPE_MIX_AIC_1_2 On-Chip 3M Kernel
         * Path 3A: Mix64 (64x64, PANEL_K=64)
         * Path 3B: Mix128 (128x128, PANEL_K=32, 用于 !isInt8 && n>=256 && k<=32)
                                     |
                                     v
  【Device 侧 (KERNEL_TYPE_MIX_AIC_1_2 On-Chip 3M)】
    1. AIV Producer (SubBlock 0 & 1):
       - GM -> UB 搬运 A(j) 与 A(i) (C64 连续读取 或 INT8 提升至 FP32)
       - 片上生成 (Xr, Xi, X-) 与 (Xr, Xi, X+)，Scatter 转换为 NZ/ZN 格式送入 L1 双缓冲槽
    2. AIC Compute:
       - 强制禁用 HF32 (SetHF32Mode(DISABLE))
       - L1 -> L0A/L0B 乒乓加载，3 次实数 Mmad 流水线计算：
         * P0 = Xr * Xr^T
         * P1 = Xi * Xi^T
         * P2 = X- * X+^T
       - Fixpipe 直连将 P0, P1, P2 分送两个 AIV SubBlock 的 UB 接收区
    3. AIV Consumer (SubBlock 0 & 1):
       - 分段 K 累加到 (sumRe, sumIm)
       - 融合计算：Re = alpha*(P0+P1) + beta*C_old, Im = alpha*(P2-P0+P1) + beta*C_old
       - 对角元素强制将虚部清零
       - 指数位检测 0x7F800000 捕获 Inf/NaN，异常时原地触发 SIMT 精确重算
       - DataCopyPad 严格按 uplo 三角切片精准写回 GM (beta==0 绝不读旧 C，绝不污染非 uplo 内存)
+----------------------------------------------------------------------------------------------------+
```

#### 3.2.1 host侧设计：

tiling策略：

根据参数特征，通过纯值传递构造 64 字节、严格 8 字节对齐的 TilingData 结构体寄存器直传，实现 **0 字节全局 Workspace**：
- **Path 1 TilingData**：传递 $n$、$ldc$、$\beta$、`isUpper`、`isBetaZero`；
- **Path 2 TilingData**：传递 $n$、$k$、$lda$、$ldc$、$\alpha$、$\beta$、`isUpper`、`isConjTrans`、`isInt8`、`isBetaZero`；
- **Path 3 TilingData**：除基础维度与标量外，传递 `gridDim = CeilDiv(n, tileSize)`、`totalTriTiles = gridDim*(gridDim+1)/2`、`usedCores = min(aicCoreNum, totalTriTiles)`、`reductionSpan` 及 `useTile128` 标志。

任务均分：

在 Path 3 中，二维分块网格仅对有效三角 Tile 编码。对于总三角 Tile 数 $T_{\text{tri}} = G(G + 1) / 2$：
- 激活物理核组数 $U = \min(\text{aicCoreNum}, T_{\text{tri}})$；
- 采用对称三角序号解码将线性索引均匀映射到二维坐标 $(m_{\text{tile}}, n_{\text{tile}})$；
- 引入 **2D Swizzle 蛇形遍历**，使相邻核组在时间轴上高度复用相同的输入面板，最大化命中 128 MB L2 Cache；
- 在 Path 1 中，针对三角矩阵行长不均问题，采用**首尾镜像配对列调度**（列 $c$ 与列 $n - 1 - c$ 配对），使每核负载严格恒定为 $n + 1$ 个元素。

批量搬运：

1. **On-Chip 直连 0 HBM 落地**：三次实数矩阵乘所需的操作数 $(X_r, X_i, X_+, X_-)$ 直接在 AIV UB 变换后推入 L1，Cube 输出通过 `Fixpipe` 直连写回 AIV UB，全程无中间矩阵的 HBM 往返；
2. **2D DataCopyPad 精确搬运**：利用硬件 2D DMA 的 `srcStride` 与 `dstStride` 机制，按列主序直接以连续大块突发传输，非 16 对齐尾块利用 DMA 自动补零，彻底消除标量搬运停顿。

##### 1. 分核策略：

- **核组规模裁剪**：根据三角 Tile 数动态裁剪发射核数 `usedCores = min(aicCoreNum, totalTriTiles)`，在小尺寸下避免冗余核空转；
- **核内 1 AIC : 2 AIV 职责分工**：
  - SubBlock 0：负责左面板变换（输出列 $j$）与输出上半块合并写回；
  - SubBlock 1：负责右面板变换（输出行 $i$）与输出下半块合并写回；
  - AIC Cube：纯流水线执行 3 路实数 `Mmad` 与 `Fixpipe` 分发。

##### 2. 数据分块和内存优化策略：

基于 Ascend 950PR 硬件存储上限（L1: 512 KB, L0A/B: 各 64 KB, L0C: 256 KB, UB: 248 KB）制定严密的双缓冲预算账本，**全局 Workspace 为 0 字节**：

- **`TILE = 64, PANEL_K = 64` 预算表**：
  - `L0C`：$2 \times 3 \times (64 \times 64 \times 4\text{ B}) = \mathbf{96\text{ KB}}$（占 256 KB 的 37.5%）；
  - `L0A / L0B`：各 $2 \times (64 \times 16 \times 4\text{ B}) = \mathbf{8\text{ KB}}$（各占 64 KB 的 12.5%）；
  - `L1`：$2 \times 6 \times (64 \times 64 \times 4\text{ B}) = \mathbf{192\text{ KB}}$（占 512 KB 的 37.5%）；
  - `UB (每 AIV)`：Fixpipe 接收区 24 KB + 累加区 32 KB + Producer 输入变换区 80 KB，单 Stage 驻留峰值 **136 KB**，双 Stage 缓冲模式为 **160 KB**（严格收拢在 192 KB 安全阈值内，预留 56 KB 给系统与描述符）；
- **`TILE = 128, PANEL_K = 32` 预算表**：
  - 针对薄 K 场景，将 Consumer UB 严格限制在 `0..224 KB` 范围内，彻底避开 `224..248 KB` 的 SIMT DCache 区，杜绝跨内核脏数据。

##### 3. tilingkey规划策略：

| TilingKey | 路由名称 | 判定条件 | 核心优化与硬件执行模式 |
| :---: | :--- | :--- | :--- |
| **0** | `SCALE_ONLY` | $(k == 0 \lor \alpha == 0) \land \beta \ne 1$ | AIV-only 纯向量流水；首尾配对负载均衡；$\beta=0$ 零读旧值 |
| **1** | `DIRECT_SIMT` | $n \le 64$ 或小数据量 | AIV-only SIMT；512 线程 Warp Shuffle 归约；微秒级超低延迟 |
| **2** | `MIX_TILE64` | 通用中大形状（默认） | `KERNEL_TYPE_MIX_AIC_1_2`；$64 \times 64$ Tile，On-Chip 双缓冲流水 |
| **3** | `MIX_TILE128` | $A\text{type} \ne \text{INT8} \land n \ge 256 \land k \le 32$ | `KERNEL_TYPE_MIX_AIC_1_2`；$128 \times 128$ 大 Tile 专攻超薄 K，对角优先调度 |

数据检测：

严格按照任务书 §2.5 执行检测：
1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `uplo` 非 `UPPER/LOWER` 或 `trans` 非 `OP_N/OP_C`（含 `OP_T`）返回 `ACLBLAS_STATUS_INVALID_ENUM`；
3. 结构参数 $n < 0 \lor k < 0 \lor lda < \text{minLda} \lor ldc < \max(1, n)$ 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. 数据类型非 `(C64, C64)` 或 `(INT8, C64)` 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；
5. $\alpha, \beta$ 指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. **空维快速返回**：$n = 0$ 返回 `ACLBLAS_STATUS_SUCCESS`（不引用 $A, C$，不启动 Kernel）；
7. 探测 $\alpha, \beta$ 内存类型（禁止 Host/Device 混用，否则返回 `INVALID_VALUE`）；
8. $C$ 为空或（$k > 0 \land \alpha \ne 0 \land A == \text{nullptr}$）返回 `ACLBLAS_STATUS_INVALID_VALUE`；
9. **纯恒等快速返回**：$(k == 0 \lor \alpha == 0) \land \beta == 1$ 返回 `ACLBLAS_STATUS_SUCCESS`。

#### 3.2.2 kernel侧设计：

1. **AIC 初始化与禁用 HF32**：首行显式调用 `SetHF32Mode(DISABLE)`，确保 FP32 乘法具备完整的 24-bit 尾数精度；
2. **Producer AIV 变换流水**：
   - 从 GM 读取数据，若为 `ACL_INT8`，向量化将其解包为 `(re, im)` 并执行精确提升；
   - 根据 `trans`（`OP_N` 或 `OP_C`）施加符号因子，生成 $X_r, X_i, X_+, X_-$；
   - 变换为 Cube 所需的 $16 \times 16$ 矩阵分块（NZ/ZN）并推入 L1，通过 `CrossCoreSetFlag` 通知 AIC；
3. **AIC Cube 3M 乘加流水**：
   - 监听 AIV `INPUT_READY` 信号，L1 乒乓加载至 L0A/L0B；
   - 顺序发射 3 次实数 `Mmad`：$P_0 = X_r X_r^T$、$P_1 = X_i X_i^T$、$P_2 = X_- X_+^T$；
   - 调用 `Fixpipe` 直连将结果分发给 SubBlock 0 和 1 的 UB 接收槽；
4. **Consumer AIV 融合与安全落盘**：
   - 若 $k$ 超过分段深度，在 UB 内局部累加分段结果；
   - 组合实部 $P_0 + P_1$ 与虚部 $P_2 - P_0 + P_1$，乘以 $\alpha$ 并融合 $\beta C_{\text{old}}$（$\beta=0$ 时彻底跳过旧值加载）；
   - 主对角线元素强制虚部清零；
   - 顺带检测指数掩码 `0x7F800000`，若遇到非有限值原地触发 SIMT 精确重算；
   - 严格按 `uplo` 边界裁剪写回，绝不污染非 `uplo` 内存与 `ldc` padding。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | :---: |
| Ascend 950PR（`arch35` / `DAV_3510`，SOC=ascend950） | √ |

## 算子约束限制

1. **硬件与架构约束**：专为 Ascend 950PR（`arch35` / `DAV_3510`）设计，CANN 软件版本要求 CANN 9.1.0 及以上；
2. **转置模式约束**：仅支持 `ACLBLAS_OP_N` 与 `ACLBLAS_OP_C`；`ACLBLAS_OP_T` 无数学定义，属于非法枚举；
3. **内存无重叠与越界约束**：矩阵 $A$ 与 $C$ 内存空间不得重叠；调用方保证非 `uplo` 三角内存不受保护破坏；
4. **工作空间约束**：算子内部全程 **0 字节全局 Workspace**。

# 可维可测分析

## 精度标准/性能标准

### 精度验收标准

以 CPU 侧 Netlib `cblas_cherk` 结合单精度浮点运算作为参考真值（`ACL_INT8` 先做精确提升后计算）：
- **度量准则（Mixed Tolerance）**：$rtol = atol = 2^{-13} \approx 1.2207 \times 10^{-4}$，最大绝对误差上限 $\le \max(10^{-2}, 32 \times \text{ULP})$，匹配率 $\text{matched\_ratio} \ge 0.99$；
- **全量验收标准**：全量 1000 条精度用例达到 **`PASS=1000, FAIL=0, MISSING=0`（100% 通过）**；
- **边界与安全判定**：非 `uplo` 三角与 `ldc` padding 区域哨兵值逐比特零改写；对角线虚部强制清零；特殊接口用例（`NullHandle`、Host/Device 标量双模式）100% 通过。

### 性能验收标准

以 `test_cases/gpu_baseline.csv` 中 200 条 NVIDIA GPU A100 实测耗时（`gpu_ms`）为基准：
- **正式采样协议**：基于 `msprof op` 采集底层 kernel **Task Duration(us)**，执行统一 18-Launch 协议（`1 precheck + 5 warmup + 12 samples`，剔除前 6 次冷态与后 12 次中 2 个最大瞬态毛刺，对剩余 10 次取均值）；
- **达标红线与实测**：
  - 官方红线：单 case 倍率 $\ge 0.4000\times$，全量 200 例算术平均倍率 $\ge 0.5000\times$；
  - 内部安全预留：单 case $\ge 0.4200\times$，全量均值 $\ge 0.5500\times$；
  - **实测性能**：200 例达标率 **200/200（100% PASS）**，最小倍率 **`0.4883x`**，算术平均倍率 **`0.8417x`**，重点瓶颈用例 `TC_PF_1149 (n=1031, k=8)` 实测 **`10.825 us`（`0.5221x`）**；
- **内存审计标准**：静态契约 0 字节；3 阶段 `npu-smi` 显存采样前后完全一致，**`Leak = 0 MB`**。

## 兼容性分析

1. **API 与 ABI 兼容性**：在 `include/cann_ops_blas.h` 新增标准函数声明，参数顺序、类型与 cuBLAS `cublasCherk3mEx` 逐参严格一致；
2. **代码仓既有工程零污染**：独立于现有基础版 `aclblasCherk`，全套源文件与头文件严格以 `cherk3m_ex_` 命名，自动匹配 CMake 单算子筛选规则，不侵入修改主构建脚本；
3. **低版本 CANN 条件编译隔离**：关键 Tensor API 与 Fixpipe 扩展使用 `#if ASC_DEVKIT_GE_9_1` 宏进行安全包裹，确保跨构建系统编译安全。
