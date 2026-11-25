# aclblasCgemm3m 算子设计文档（Atlas A2/A3，arch22）

| 项 | 内容 |
| --- | --- |
| 算子名 | `aclblasCgemm3m` |
| 任务 | 社区任务 — aclblasCgemm3m 算子开发（A2/A3） |
| 目标仓 | [ops-blas](https://gitcode.com/cann/ops-blas)，实现 `blas/gemm/arch22/`，测试 `test/gemm/cgemm3m/arch22/` |
| 对标基线 | cuBLAS `cublasCgemm3m`（[cuBLAS 官方文档](https://docs.nvidia.com/cuda/cublas/index.html)），数学语义参考 Netlib `cgemm`（<https://www.netlib.org/blas/cgemm.f>） |
| 数据类型 | 输入/输出矩阵 `COMPLEX64`（`aclblasComplex`，实部/虚部各 FP32，列主序）；标量 `alpha` / `beta` 为 Host 侧 `COMPLEX64` |
| 目标硬件 | Atlas A2 训练/推理系列、Atlas A3 训练/推理系列（arch22） |
| CANN 版本 | 9.1.0 |
| 自验证范围 | 仅需覆盖一种款型，精度与性能自验证设备均为 Atlas 800T A2（910B3）（任务书 §3.1） |
| **本文档状态** | **设计评审稿（不含 NPU 实测数据）。** 接口原型、校验口径、arch35 既有实现差异、任务包规格核查与 A100 性能预算均已依据 `ops-blas` 仓源码、任务书与 `test_cases/gpu_baseline.csv` 逐项核定；最终 NPU 精度、性能与 profiling 实测结果将在自验证阶段独立出具于自测报告与原始日志中。 |

---

# 1 需求背景（required）

## 1.1 需求来源

社区任务「aclblasCgemm3m 算子开发（A2/A3）」。当前 `ops-blas` 仓仅存在面向 Ascend 950PR（arch35）的 `blas/gemm/arch35/cgemm3m_host.cpp` 实现与 `test/gemm/cgemm3m/arch35/` 测试工程，Atlas A2/A3（arch22）在复数三乘法通用矩阵乘（3M GEMM）上为空缺。本任务补齐 `aclblasCgemm3m` 的 arch22 实现与测试工程，并将文档与验证套件完整对齐仓库规范。

同时存在两处工程约束须在立项阶段明确：

1. **测试构建缺口**：`test/gemm/cgemm3m/CMakeLists.txt` 当前仅在 `"arch35" IN_LIST SOC_ARCH_DIRS` 时注册 `cgemm3m_test` 目标，arch22 构建会跳过全部 `cgemm3m` 测试。须在不改动 arch35 行为的前提下，将条件扩展为 `if("arch35" IN_LIST SOC_ARCH_DIRS OR "arch22" IN_LIST SOC_ARCH_DIRS)`。
2. **目录放置边界（任务书 §2.2）**：实现代码必须放在 `blas/gemm/arch22/`，测试代码放在 `test/gemm/cgemm3m/arch22/`。**严禁**放入 `blas/gemm3m/`——该目录是实数 `aclblasSgemm3m`（沿 K 维拼接三个实数矩阵的 3M 接口）的实现目录，与复数 `aclblasCgemm3m` 属于不同接口族。

## 1.2 背景介绍

复数通用矩阵乘（CGEMM）的直接 4M 算法需要 4 次实数矩阵乘法：

$$\operatorname{Re}(A B) = A_r B_r - A_i B_i,\qquad \operatorname{Im}(A B) = A_r B_i + A_i B_r$$

Gauss 3M 算法利用代数恒等式将 4 次实数矩阵乘降为 3 次实数矩阵乘，从而将理论 Cube 矩阵乘计算量降低 **25%**（$6MNK$ vs $8MNK$ 实数 FLOPs）：

$$P_0 = \operatorname{op}_R(A_r)\cdot\operatorname{op}_R(B_r),\quad P_1 = \operatorname{op}_R(A_i')\cdot\operatorname{op}_R(B_i'),\quad P_2 = \operatorname{op}_R(A_r + A_i')\cdot\operatorname{op}_R(B_r + B_i')$$
$$\operatorname{Re}(\operatorname{op}(A)\cdot\operatorname{op}(B)) = P_0 - P_1,\qquad \operatorname{Im}(\operatorname{op}(A)\cdot\operatorname{op}(B)) = P_2 - (P_0 + P_1)$$

其中 $A_i' = s_A A_i$、$B_i' = s_B B_i$，$s_A = -1$ 当且仅当 `transa == ACLBLAS_OP_C`（否则 $+1$），$s_B = -1$ 当且仅当 `transb == ACLBLAS_OP_C`（否则 $+1$），$\operatorname{op}_R$ 表示纯实数转置（`OP_N` 不转置，`OP_T`/`OP_C` 转置）。

在 Atlas A2/A3（arch22）上落地高性能且全量过精度门禁的 `aclblasCgemm3m`，存在三个核心工程难点：

1. **3M 数值抵消与长 K 漂移**：与逐 $k$ 步执行复数乘加的 `cblas_cgemm`（4M）不同，3M 在中间和矩阵 $A_r + A_i'$、$B_r + B_i'$ 上引入额外一次 FP32 舍入，且 $P_2$ 在非零均值分布（如任务包 `GAUSS_NORM_0_1`）下的量级可达 $P_0, P_1$ 的 4 倍。长 $K$ 归约末尾再做 $P_2 - P_0 - P_1$，或被大标量 $|\alpha| = 10^4$ 放大，或在交替符号输入下发生有效位抵消时，极易突破 `max(1e-2, 32·ULP)` 误差上限或 `matched_ratio ≥ 0.99` 门禁。
2. **AIV 预处理/后处理与 AIC Cube 的协同吞吐**：中大矩阵（$M,N,K \ge 512$）由 AIC Cube 算力主导，而瘦长/短 K 形状（如 $64\times 4096\times 64$、$4096\times 4096\times 64$、$31\times 79\times 1368$）及极小矩阵（$1^3 \sim 64^3$）受 AIV 内存访存与核启动延迟主导，要求统一 Cube 发射、批量 2D DMA/1D 合并双缓冲向量搬运以及小形状零 workspace 单核/单发射快路径。
3. **arch22 硬件能力与 arch35 的架构差异**：arch22 不具备 arch35 的 SIMT/MicroAPI 与 `tensor_api` 能力，且 FP32 Cube 在部分 CANN/SoC 组合下默认开启 HF32 截断，必须显式关闭 HF32 并基于 Ascend C `MatmulImpl` 与向量原语构建自研流水。

### 1.2.1 arch35 与仓内既有 arch22 实现现状分析

已通读 `blas/gemm/arch35/cgemm3m_host.cpp`、`blas/gemm/arch35/cgemm_host.cpp`（L1-783）、`blas/gemm/arch35/cgemm_epilogue_kernel.cpp`、`blas/gemm/arch35/cgemm3m_rounding_kernel.cpp` 及同目录 `blas/gemm/arch22/cgemm_host.cpp`、`blas/gemm/arch22/gemm_kernel.cpp`，关键事实汇总如下：

| 既有仓内事实 | 源码位置 | 本 arch22 设计的处理 |
| --- | --- | --- |
| arch35 `aclblasCgemm3m` 经 `LaunchComplexGemm(..., forceThreeProducts=true)` 分发，支持非融合 3M（$P_0-P_1, P_2-P_0-P_1$）与融合 3M 两条路径 | `blas/gemm/arch35/cgemm3m_host.cpp` L17-23；`cgemm_host.cpp` L701-730 | arch22 独立实现 `blas/gemm/arch22/cgemm3m_*`，采用统一对称 Gauss 3M 管线 + 小形状融合 AIV 3M 路径，不触碰 arch35 文件 |
| arch35 在 `LaunchComplexGemm` L713 将 `forceThreeProducts` 下的 `ACLBLAS_STATUS_INVALID_ENUM` 改写为 `ACLBLAS_STATUS_INVALID_VALUE` | `blas/gemm/arch35/cgemm_host.cpp` L713 | **刻意不同（按任务书与 A2/A3 任务包核定）**：任务书 §2.3/§2.4 及任务包 `TC_ED_396`/`TC_ED_397` 明确要求非法 `transa`/`transb` 返回 `ACLBLAS_STATUS_INVALID_ENUM`，arch22 严格返回 `INVALID_ENUM`（详见 §3.3.2、§6.2） |
| arch35 标量 `alpha`/`beta` 直接在 Host 侧解引用；arch22 `cgemm_host.cpp` 却通过 `CBlas3ReadScalarsFromDevice` 同步拷贝设备标量 | `arch35/cgemm_host.cpp` L719-720；`arch22/cgemm_host.cpp` L464 | 按任务书 §2.4「alpha/beta 位于 Host 内存」，在 Host 侧直接解引用，避免多余的 H2D/D2H 与流同步开销 |
| arch35 在长 K / 抵消场景引入 K 分区（`kPartitions`）以及基于无参考能量/抵消比的稀疏顺序重算守卫（`CorrectableEpilogue` / `cgemm3m_rounding_kernel`） | `arch35/cgemm_host.cpp` L251-275；`cgemm_epilogue_kernel.cpp` L130-215 | 借鉴同一数学原理，但针对 arch22 向量架构做大幅简化：采用定长 K 分块每块先抵消再累加 + 向量快筛稀疏抵消纠正，不依赖 SIMT 指令 |
| arch22 `aclblasCgemm` 通过子句柄 4 次调用 `aclblasSgemmEx` 且存在 `K > 1024` 分块规避（ops-blas#1077），且 `aclblasSgemm` 对非对齐尺寸退化明显 | `blas/gemm/arch22/cgemm_host.cpp` L62-69、L353 | **不委托 `aclblasSgemm`/`aclblasSgemmEx`、零 CATLASS 依赖**：自研纯 Ascend C `MatmulImpl` 单次发射 3 乘积统一 Cube Kernel，并在预处理阶段完成 16/64 对齐零填充，彻底消除非对齐退化与 3 次独立启动开销 |
| arch22 `gemm_kernel.cpp` 显式调用 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)` 防止 Cube FP32 退化为 HF32 | `blas/gemm/arch22/gemm_kernel.cpp` L111-113、L295 | 同样在所有 Cube Kernel 入口第一时间显式关闭 HF32 |

### 1.2.2 标杆接口（cuBLAS `cublasCgemm3m`）与预算特征

```cpp
cublasStatus_t cublasCgemm3m(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha,
    const cuComplex *A, int lda, const cuComplex *B, int ldb,
    const cuComplex *beta, cuComplex *C, int ldc);
```

依据任务包 `test_cases/gpu_baseline.csv` 提供的 200 条 A100 kernel-only 基线数据，不同规模档对应的 1.25× 耗时预算（即 ≥ 0.8× A100 标杆）呈现三段截然不同的瓶颈特征（详见 §3.5.2）：

- **极小规模段（$1^3 \sim 64^3$）**：延迟主导，最紧预算为 $1\times 1\times 1$ 的 **9.86 us** 与 $64\times 64\times 64$ 的 **18.69 us**，无法承受三阶段多 kernel 启动与 workspace 往返。
- **瘦长/短 K/中等规模段（如 $64\times 4096\times 64$、$63\times 4095\times 63$、$4096\times 4096\times 64$、$31\times 79\times 1368$、$256^3 \sim 426^3$）**：AIV 预处理/后处理访存带宽与小列流水停顿、以及 Cube 核占用率共同主导，最紧预算 25.47 ~ 47.74 us。
- **大计算量段（$512^3 \sim 4096^3$，含 $4095^3$）**：Cube 算力主导，峰值等效复数算力需求约 **20.11 TFLOPS**（对应 3M 实数 Cube 约 **15.08 TFLOPS**，预算 27.34 ~ 27.55 ms），需要高 L0C 复用大 tile 与对齐访存。

## 1.3 算子原型

接口声明已存在于 `include/cann_ops_blas.h`（L584-587），与 arch35 共用，本任务**不修改**公开头文件（任务书 §2.3 / §7.3）。

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

| 参数 | 方向 | 位置 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | Host | `aclblasHandle_t` | ops-blas 上下文句柄，绑定 stream 与内部 workspace |
| `transa` | 输入 | Host | `aclblasOperation_t` | $\operatorname{op}(A)$ 类型：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` |
| `transb` | 输入 | Host | `aclblasOperation_t` | $\operatorname{op}(B)$ 类型：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C` |
| `m` | 输入 | Host | `int` | $\operatorname{op}(A)$ 与 $C$ 的行数，$m \ge 0$ |
| `n` | 输入 | Host | `int` | $\operatorname{op}(B)$ 与 $C$ 的列数，$n \ge 0$ |
| `k` | 输入 | Host | `int` | $\operatorname{op}(A)$ 的列数与 $\operatorname{op}(B)$ 的行数，$k \ge 0$ |
| `alpha` | 输入 | Host | `const aclblasComplex*` | 复数标量乘数指针，不可为 `nullptr` |
| `A` | 输入 | Device | `const aclblasComplex*` | 列主序复数矩阵，物理尺寸 `lda × k`（N）或 `lda × m`（T/C），只读 |
| `lda` | 输入 | Host | `int` | $A$ 的前导维，`transa==N` 时 $\ge \max(1, m)$，否则 $\ge \max(1, k)$ |
| `B` | 输入 | Device | `const aclblasComplex*` | 列主序复数矩阵，物理尺寸 `ldb × n`（N）或 `ldb × k`（T/C），只读 |
| `ldb` | 输入 | Host | `int` | $B$ 的前导维，`transb==N` 时 $\ge \max(1, k)$，否则 $\ge \max(1, n)$ |
| `beta` | 输入 | Host | `const aclblasComplex*` | 复数标量乘数指针，不可为 `nullptr`；`beta == 0` 时不读取 $C$ 旧值 |
| `C` | 输入/输出 | Device | `aclblasComplex*` | 列主序原地输出矩阵，物理尺寸 `ldc × n`，仅覆写逻辑 $m \times n$ 区域 |
| `ldc` | 输入 | Host | `int` | $C$ 的前导维，$\ge \max(1, m)$ |

---

# 2 需求分析（required）

## 2.1 需求描述

在 Atlas A2/A3（arch22）NPU 上实现列主序单精度复数三乘法通用矩阵乘 `C = alpha * op(A) * op(B) + beta * C`，支持 `N`、`T`、`C` 全部 9 种转置/共轭组合以及带 padding 前导维布局，通过 `aclblasHandle_t` 绑定的 stream 异步直调 Ascend C kernel。

## 2.2 需求拆解

| # | 要求 | 任务书出处 |
| --- | --- | --- |
| 1 | 核心计算：`C = alpha * op(A) * op(B) + beta * C`，COMPLEX64 列主序，采用 Gauss 3M 三次实数矩阵乘分解 | §2.1 |
| 2 | 9 种转置/共轭组合：`OP_C` 为共轭转置、`OP_T` 为非共轭转置，二者数值不同 | §2.1 |
| 3 | 参数校验顺序：handle → `transa`/`transb` 枚举（`INVALID_ENUM`）→ `m`/`n`/`k < 0` → `lda`/`ldb`/`ldc` 下界 → `alpha`/`beta` 判空 → quick return（`m==0 \|\| n==0`）→ `C` 判空 → 有乘积时 `A`/`B` 判空 | §2.3 / §2.4 / §2.5 |
| 4 | 边界与退化语义：`m==0 \|\| n==0` 合法 no-op（不访问 A/B/C、不启动 kernel）；`k==0 \|\| alpha==0` 仅做 `C = beta * C`（允许 A/B 为 `nullptr` 或含 NaN，且 `beta==1` 时 no-op）；`beta==0` 时不读取 C 旧值（支持 C 含 NaN） | §2.1 / §2.4 / §2.5 |
| 5 | 写保护语义：仅覆写 $C$ 的逻辑 $m \times n$ 区域，`ldc` 超出 $m$ 的 padding 行保持原值不被改写；$C$ 不与 $A/B$ 存储重叠 | §2.1 / §2.5 |
| 6 | 精度验收：以 CPU `cblas_cgemm` 为 golden，实部与虚部分别按 FLOAT32 混合容差判定（`atol = rtol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ max(1e-2, 32·ULP)`），Inf/NaN 按非有限值规则独立判定 | §3.2 |
| 7 | 性能验收：Atlas 800T A2（910B3）上正式判据为 `msprof` kernel Task Duration 与 `gpu_baseline.csv` 的 `gpu_ms` 做 kernel-to-kernel 对比（warmup 5 次 + 有效采样 10 次取均值，单 case 及平均倍率均 ≥ 0.8）；GTest `[PERF] avg_us` 作为辅助交叉验证 | §3.3 |
| 8 | 内存验收：不涉及，但交付包须按模板出具 `4.1/4.2` 内存自验证文件并说明依据 | §3.4 / §4 |
| 9 | 工程落位：实现代码位于 `blas/gemm/arch22/`（禁止放入 `blas/gemm3m/`），测试代码位于 `test/gemm/cgemm3m/arch22/` 并注册至 `test/gemm/cgemm3m/CMakeLists.txt` | §2.2 |
| 10 | 接口兼容：沿用 `include/cann_ops_blas.h` L584 既有声明，禁止新增私有平行 API；更新 `blas/gemm/README.md` 补充 A2/A3 支持 | §2.2 / §2.3 / §7.3 |
| 11 | 异步执行：所有 kernel 下发至 `handle->stream`，Host 侧不做隐式流同步 | §2.5 |

## 2.3 与 cuBLAS `cublasCgemm3m` 的差异

| 项 | cuBLAS `cublasCgemm3m` | 本算子 `aclblasCgemm3m`（arch22） | 原因 |
| --- | --- | --- | --- |
| 非法 `transa`/`transb` | 返回 `CUBLAS_STATUS_INVALID_VALUE` | 返回 `ACLBLAS_STATUS_INVALID_ENUM` | 任务书 §2.3/§2.4 明文规定为 ops-blas 约定，属有意差异 |
| `alpha`/`beta` 指针模式 | 随 `cublasPointerMode` 支持 Host 或 Device 指针 | 固定为 Host 内存指针 | 任务书 §2.4 与 ops-blas 接口约定（与 cuBLAS 默认 Host mode 一致） |
| 返回码类型 | `cublasStatus_t` | `aclblasStatus_t`（`SUCCESS` / `HANDLE_IS_NULLPTR` / `INVALID_ENUM` / `INVALID_VALUE` / `ALLOC_FAILED` / `EXECUTION_FAILED` / `INTERNAL_ERROR`） | ops-blas 统一错误码体系 |
| 3M 数值舍入 | GPU Tensor Core / FP32 3M 累加序 | NPU AIC Cube（禁用 HF32）+ 分块 3M 抵消与稀疏纠正 | 任务书 §2.1 / §3.2 明确不要求与 cuBLAS/cgemm 逐位一致，按混合容差阈值判定 |

---

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式与列主序映射

**1. Gauss 3M 复数乘法分解与共轭符号吸收**

令 $A = A_r + i A_i$、$B = B_r + i B_i$，定义共轭符号 $s_A = (\text{transa} == \text{OP\_C}) ? -1 : +1$、$s_B = (\text{transb} == \text{OP\_C}) ? -1 : +1$，在预处理阶段直接构造三张实数平面：

$$A^{(0)} = A_r,\qquad A^{(1)} = s_A A_i,\qquad A^{(2)} = A_r + s_A A_i$$
$$B^{(0)} = B_r,\qquad B^{(1)} = s_B B_i,\qquad B^{(2)} = B_r + s_B B_i$$

由此将 `OP_C` 的共轭语义完全吸收进预处理阶段，后续实数矩阵乘仅区分非转置（`N`）与转置（`T`/`C`）两类物理布局。三次实数矩阵乘定义为：

$$P_0 = \operatorname{op}_R(A^{(0)})\cdot \operatorname{op}_R(B^{(0)}),\quad P_1 = \operatorname{op}_R(A^{(1)})\cdot \operatorname{op}_R(B^{(1)}),\quad P_2 = \operatorname{op}_R(A^{(2)})\cdot \operatorname{op}_R(B^{(2)})$$

复数乘积组装为：

$$R = \operatorname{Re}(\operatorname{op}(A)\operatorname{op}(B)) = P_0 - P_1,\qquad I = \operatorname{Im}(\operatorname{op}(A)\operatorname{op}(B)) = P_2 - (P_0 + P_1)$$

> **减法顺序要点**：虚部刻意写为 `P2 - (P0 + P1)` 而非 `(P2 - P0) - P1`。当 $P_0 \approx -P_1$ 且 $|P_2| \ll |P_0|$（如 `RANDOM_ALTER` 交替输入中 $B_r + B_i = -1$ 而 $|B_r|, |B_i| \sim 2000$）时，先计算 $P_0 + P_1$ 可使大数先抵消，避免 $(P_2 - P_0)$ 先吞没 $P_2$ 的有效尾数位。

最终复数仿射缩放与原地累加：

$$C_r^{\text{new}} = (\alpha_r R - \alpha_i I) + (\beta_r C_r - \beta_i C_i),\qquad C_i^{\text{new}} = (\alpha_r I + \alpha_i R) + (\beta_r C_i + \beta_i C_r)$$

**2. 列主序到行主序 Cube `MatmulImpl` 的零转置映射**

Ascend C `MatmulImpl`（`CubeFormat::ND`）按行主序解释矩阵。利用列主序矩阵 $X \in \mathbb{R}^{r \times c}$ 在内存中的字节序列等价于行主序矩阵 $X^T \in \mathbb{R}^{c \times r}$ 的恒等式，列主序实数乘积 $P = \operatorname{op}_R(A) \cdot \operatorname{op}_R(B)$ 等价于行主序矩阵乘：

$$P^T_{[N_{\text{pad}} \times M_{\text{pad}}]} = \big(\operatorname{op}_R(B)\big)^T_{[N_{\text{pad}} \times K_{\text{pad}}]} \cdot \big(\operatorname{op}_R(A)\big)^T_{[K_{\text{pad}} \times M_{\text{pad}}]}$$

由此确定 `MatmulImpl` 内部左操作数（来自 $B$）、右操作数（来自 $A$）与输出（$P$）的转置标志与维度映射：

| 公开 `transB` / `transA` | 物理列主序形状 | 内存等价行主序形状 | 内部 `LeftTrans`（$B$） / `RightTrans`（$A$） |
| --- | --- | --- | --- |
| `transB == N` | $K_{\text{pad}} \times N_{\text{pad}}$ | $N_{\text{pad}} \times K_{\text{pad}}$ | `LeftTrans = false` |
| `transB == T / C` | $N_{\text{pad}} \times K_{\text{pad}}$ | $K_{\text{pad}} \times N_{\text{pad}}$ | `LeftTrans = true` |
| `transA == N` | $M_{\text{pad}} \times K_{\text{pad}}$ | $K_{\text{pad}} \times M_{\text{pad}}$ | `RightTrans = false` |
| `transA == T / C` | $K_{\text{pad}} \times M_{\text{pad}}$ | $M_{\text{pad}} \times K_{\text{pad}}$ | `RightTrans = true` |

内部输出的物理行主序 $[N_{\text{pad}} \times M_{\text{pad}}]$ 正是列主序 $P_{[M_{\text{pad}} \times N_{\text{pad}}]}$（前导维 $ld_P = M_{\text{pad}}$），全程无需任何物理显式转置。

**3. 3M 数值误差来源分析**

设沿 $K$ 维分为 $S$ 段、每段长度为 $K_{\text{span}}$，则每段实数乘积 $P_0^{(s)}, P_1^{(s)}, P_2^{(s)}$ 的累加深度仅为 $K_{\text{span}}$。将复数组装前移至每个 K 分块内部：

$$R = \sum_{s=0}^{S-1} \big(P_0^{(s)} - P_1^{(s)}\big),\qquad I = \sum_{s=0}^{S-1} \big(P_2^{(s)} - (P_0^{(s)} + P_1^{(s)})\big)$$

- **消除长 K 均值膨胀漂移**：在非零均值输入（均值 $\mu \neq 0$）下，整段 $K$ 累加会使 $P_2 \sim 4 \mu^2 K$ 随 $K$ 线性增长，当 $K=4096$ 时单个 ULP 即可超过抵消后 $\operatorname{Im}(AB)$ 的 $32\cdot\text{ULP}$ 容差。通过限制单块归约深度 $K_{\text{span}} \le 256$ 并逐块做 $P_2^{(s)} - (P_0^{(s)} + P_1^{(s)})$ 抵消，将单次抵消前的中间量级压低 $K / 256$ 倍（$K=4096$ 时压低 16 倍），使长 K 方阵与矩形用例直接收敛于精度阈值内。
- **消除极端标量与强抵消点误差**：当 $|\alpha|$ 极大（如 $|\alpha| = 10^4$）或单个输出元素满足 $|R| \ll |P_0| + |P_1|$、$|I| \ll |P_2| + |P_0| + |P_1|$、$|\operatorname{Re}(\alpha(R+iI))| \ll |\alpha_r R| + |\alpha_i I|$ 时，3M 舍入误差可能超过该元素的 $32\cdot\text{ULP}$ 或 `1/8192` 相对容差。通过向量快筛估计单元素舍入噪声上界，仅对极少数高风险抵消点从已拆分的实数平面 $(A_r, A_i', B_r, B_i')$ 执行有序复数乘加纠正（详见 §3.2.2）。

### 3.1.2 支持数据类型

| 项 | 类型 | 说明 |
| --- | --- | --- |
| 矩阵 `A`、`B`、`C` | `aclblasComplex`（COMPLEX64） | 交错存储 `{float real, float imag}`，每元素 8 字节；支持全有限值域及 Inf/NaN 传播 |
| 标量 `alpha`、`beta` | `aclblasComplex`（Host） | 实部、虚部独立为 FP32 |
| 内部实数平面与累加器 | `float`（FP32） | 禁用 HF32，保持完整 IEEE-754 binary32 尾数精度 |

### 3.1.3 支持形状

| 维度 / 属性 | 支持范围 | 说明 |
| --- | --- | --- |
| `m`, `n`, `k` | 任意非负 `int32` | `m==0 \|\| n==0` quick return；`k==0` 仅缩放 $C$；负值返回 `INVALID_VALUE` |
| `lda`, `ldb`, `ldc` | 满足 BLAS 前导维下界 | 支持紧凑及任意大于下界的带 padding 布局；低于下界返回 `INVALID_VALUE` |
| `transa`, `transb` | `{N, T, C} × {N, T, C}` 全部 9 种 | 非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 任务包覆盖规模 | 精度 $0 \sim 4096$（含 $4095^3/4096^3$）；性能 $1 \sim 4096$（含 $4095^3/4096^3$） | `test_cases/README.md` |

### 3.1.4 目标硬件约束与运行时平台查询

本设计面向 Atlas A2 / Atlas A3（arch22，分离式 AIC Cube + AIV Vector 架构），严格遵守「核数运行时查询、不写死芯片常量」原则：

- **核数查询**：Host 侧通过 `GetAicCoreCount()` 与 `GetAivCoreCount()`（底层调用 `platform_ascendc::PlatformAscendCManager`）动态获取当前设备的 AIC 与 AIV 核数；返回 0 时返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。
- **禁用 HF32 约束**：arch22 的 FP32 `MatmulImpl` 在默认配置下可能启用 HF32（10-bit 尾数），会导致长 K 乘加严重偏离 IEEE FP32 golden。所有 Cube Kernel 入口第一条语句必须执行 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)`。
- **对齐与零填充约束**：`MatmulImpl` 在 M/N 对齐 16、K 对齐 32/64 时可满载分形与突发访存，而任意非对齐尺寸会导致边界搬运与算力大幅退化。由于 3M 本身必经 AIV 预处理拆分 $(A_r, A_i', A_r+A_i')$ 与 $(B_r, B_i', B_r+B_i')$，在预处理写出阶段顺带对齐到 $M_{\text{pad}} = \lceil m/16 \rceil \cdot 16$、$N_{\text{pad}} = \lceil n/16 \rceil \cdot 16$、$K_{\text{pad}} = \lceil k/K_{\text{align}} \rceil \cdot K_{\text{align}}$（`isTransA || k >= 64` 时 $K_{\text{align}}=64$，否则 $32$）并补零，即可让非对齐形状（如 $4095^3$、$63\times 4095\times 63$）无损享受满对齐 Cube 吞吐。
- **arch22 `DataCopyPad` 与流水同步约束**：在 arch22 上，`DataCopyPad` 的 `paddingValue` 对非 32 B 对齐尾部不可依赖，因此所有尾部补零均在 UB 内显式清零后再经 MTE3 写出；同时标量读写（`SetValue`/`GetValue`）、向量指令与 MTE2/MTE3 之间严格配对 `SetFlag`/`WaitFlag` 与 `PipeBarrier<PIPE_V>`。

## 3.2 核心设计决策

### 3.2.1 独立纯 Ascend C 3M 管线（零 CATLASS / 零 `aclblasSgemm` 委托）

相比仓内 `blas/gemm/arch22/cgemm_host.cpp` 委托 `aclblasSgemmEx` 的方案，本设计采用完全自研的纯 Ascend C 3M 管线：

1. **融合预处理（1 次 AIV 启动）**：单次 `cgemm3m_preproc_kernel` 同步完成 $A$ 与 $B$ 的实虚解交错、共轭符号吸收、3M 和矩阵构造与对齐零填充，避免两次独立拆分 kernel 启动。
2. **统一 3-Product Cube Kernel（1 次 AIC 启动）**：单次 `cgemm3m_cube_*_kernel` 将 $P_0, P_1, P_2$ 与 Split-K 分区的所有输出 tile 扁平化为统一任务队列 `totalTasks = 3 * kParts * nTiles * mTiles`，按核轮询调度。相比分别发射 3 次实数 GEMM，不仅将 Cube 发射开销降为 1/3，而且在中小输出形状（如 $256^3$、$64\times 4096\times 64$）下将并行任务数扩大 3 倍，显著提高 AIC 核利用率。
3. **融合后处理（1 次 AIV 启动）**：单次 `cgemm3m_combine_kernel` 完成 K 分区先抵消再归约、复数 `alpha`/`beta` 缩放、稀疏抵消纠正与交错写回 $C$。

### 3.2.2 面向精度收敛的 K 分块抵消与向量快筛稀疏纠正

针对 §3.1.1 识别的两类 3M 数值风险，设计两层完全数据无关、通用化的精度保障机制：

1. **K 分块先抵消再累加（控制单块归约深度 $K_{\text{span}} \le 256$）**：
   - 当 $K_{\text{pad}} > 256$ 时，将 K 维切分为若干不超过 256 的片段。
   - 若单次 Split-K 所需产品缓冲 $3 \cdot k_{\text{parts}} \cdot M_{\text{pad}} N_{\text{pad}} \cdot 4\text{ B} \le 128\text{ MiB}$ 且 $k_{\text{parts}} \le 16$，则由**单次** Cube Kernel 并发输出所有 `kParts` 片段，再由单次 `combine` kernel 逐 `kp` 累加 $P_0^{(kp)} - P_1^{(kp)}$ 与 $P_2^{(kp)} - (P_0^{(kp)} + P_1^{(kp)})$（兼顾精度与 Split-K 核占用率）。
   - 若输出较大（如 $2048^3$、$4096^3$），则由 Host 按 K 块循环发射 Cube + 轻量级中间累加 kernel，将每块的 $(P_0 - P_1)$ 与 $(P_2 - (P_0 + P_1))$ 累加进常驻实部/虚部缓冲 `accRe` / `accIm`，同时保持 workspace 峰值有界（$4096^3$ 约 704 MiB，远低于 2 GiB 上限）。
2. **向量快筛 + 稀疏有序重算守卫（`cgemm3m_combine_kernel`）**：
   - 在后处理每个 UB 分块内，顺带统计每个输出元素的 3M 中间量级 $S_P = \max_{kp}(|P_0^{(kp)}| + |P_1^{(kp)}| + |P_2^{(kp)}|)$（纯 UB 向量指令，几乎零额外访存），并按无参考误差模型估算 3M 舍入噪声上界：
     $$\hat{\varepsilon} = c_0 \cdot \sqrt{K_{\text{chunks}}} \cdot (|\alpha_r| + |\alpha_i|) \cdot S_P$$
   - 将 $\hat{\varepsilon}$ 与逐元素容差保守门限（同时覆盖 `max(1e-2, 32·ULP)` 门限及中小输出的 `1/8192` 匹配门限）做向量比较并通过 `ReduceMax` 得到整块标量标记。
   - **快路径零损耗**：常规输入与正常标量下几乎全部 UB 块标记为 0，直接跳过纠正分支。
   - **稀疏纠正**：仅对标记为 1 的极少数高风险抵消元素，直接从已拆分的实数平面 $(A_r, A_i', B_r, B_i')$ 执行严格有序（`#pragma clang fp contract(off)`）复数点积重算，并在应用 $\alpha, \beta$ 后写回对应位置，消除 3M 抵消超差。

### 3.2.3 面向延迟与带宽的多级执行路径

依据输入形状与标量特征，Host 侧确定性分派到三条执行路径：

| 路径 | 触发条件 | 启动 Kernel 数 | Workspace 占用 | 设计目标 |
| --- | --- | --- | --- | --- |
| **路径 1：退化缩放快路径** | `k == 0 \|\| alpha == (0,0)` | 0（`beta==(1,0)`）或 1（`combine`） | 0 B | 不读 A/B，`beta==1` 零启动，`beta==0` 不读旧 C |
| **路径 2：小形状融合 AIV 3M 路径** | `m <= 64 && n <= 64 && k <= 64` | 1（`cgemm3m_small_kernel`） | 0 B | 覆盖 $1^3 \sim 64^3$ 延迟敏感区，零 workspace、单次发射、逐 $k$ 步 3M 抵消 |
| **路径 3：通用 Cube 3M 流水路径** | 其余所有有效形状 | 3（单 K 轮）或 $1 + 2 \cdot K_{\text{loops}}$（大输出多 K 轮） | 按需复用 handle workspace | 兼顾瘦长/短 K 高效向量搬运与大方阵高 Cube 吞吐 |

在路径 3 的向量阶段，进一步区分三种访存布局优化：

- **1D 连续合并双缓冲快路径**：当物理前导维紧凑且无行填充（预处理 `ld == rows && rowsPad == rows`；后处理 `ldc == m && ldP == m`）时，将二维矩阵展平为一维连续向量，按 2048 元素分块并采用 ping-pong 双缓冲重叠 MTE2 / Vector / MTE3，彻底消除小列循环停顿。
- **2D 多列批量搬运路径**：当存在行对齐填充或非紧凑前导维且单列较短（`rowsPad <= 1024` / `ldP <= 1024`）时，每轮按 `colsPerBatch = min(cols, 1024 / stride)` 批量打包多列，一次 2D `DataCopyPad` + 一次批量向量计算 + 一次写出，将流水停顿次数降低 4× ~ 16×。
- **单位标量直出分支**：后处理针对高频的 `alpha == (1,0) && beta == (0,0)` 场景直接将 `P0 - P1` 与 `P2 - (P0 + P1)` 写入 `crci` 缓冲，省去多余的复数标量乘法与 UB 拷贝。

### 3.2.4 多档 Cube Tile 与 Split-K 并行度策略

在 `cgemm3m_kernel.cpp` 中静态实例化 3 档输出 tile × 4 种转置组合共 12 个 `MatmulImpl` 专用 Cube Kernel：

| Tile 档位 | `tileM × tileN`（内部 $N_{\text{pad}} \times M_{\text{pad}}$） | `baseM × baseN × baseK` | 适用条件 | 设计考量 |
| --- | --- | --- | --- | --- |
| **LARGE** | $256 \times 256$ | $64 \times 256 \times 32$（TN 为 $64$） | `mPad >= 256 && nPad >= 256 && 3*nTiles*mTiles >= 2*aicCores` | L0C 双缓冲 ping-pong，最大化 $A/B$ L1 复用与大矩阵算力 |
| **MEDIUM** | $128 \times 128$ | $64 \times 128 \times 32$（TN 为 $64$） | `mPad >= 128 && nPad >= 128 && 3*nTiles*mTiles >= aicCores` | 中等方阵与矩形的核占用率与复用平衡 |
| **SMALL** | $64 \times 128$ | $64 \times 128 \times 32$（TN 为 $64$） | 瘦长矩阵与小矩阵（`min(mPad, nPad) < 128` 或任务数不足） | 避免在 $m=64$ 或 $n=64$ 等瘦长形状上浪费 75% tile 宽度，最大化并行任务数 |

当输出 tile 乘 3 仍不足以填满全部 AIC 核（`3 * nTiles * mTiles < aicCores`）且 $K_{\text{pad}} \ge 256$ 时，自动扩大 `kParts` 提升 AIC 核占用率，同时受每段最小跨度（64 对齐）与 workspace 预算约束。

## 3.3 Host 侧设计

### 3.3.1 目录与文件划分

在 `blas/gemm/arch22/` 下新增 4 个文件，职责清晰、无冗余代码，严格遵循仓库函数长度与头文件规范（`#pragma once`）：

| 文件 | 职责 |
| --- | --- |
| `cgemm3m_tiling_data.h` | 定义 4 个 Host/Device 共享 Tiling 结构体（`Cgemm3mPreprocTiling`、`Cgemm3mCubeTiling`、`Cgemm3mCombineTiling`、`Cgemm3mSmallTiling`）及公共常量 |
| `cgemm3m_kernel.h` | 声明 4 个 kernel 启动封装函数（`cgemm3m_preproc_kernel_do`、`cgemm3m_cube_kernel_do`、`cgemm3m_combine_kernel_do`、`cgemm3m_small_kernel_do`） |
| `cgemm3m_host.cpp` | 对外 `aclblasCgemm3m` C 入口：参数校验、快速返回、核数查询、Tile/Split-K/K-Chunk 规划、workspace 溢出校验与分配、三级路径发射 |
| `cgemm3m_kernel.cpp` | 实现 12 个特化 AIC Cube Kernel 及 3 个 AIV Vector Kernel（`preproc`、`combine`、`small`）与 launcher 分发 |

### 3.3.2 参数校验顺序与错误码

严格按任务书 §2.3 / §2.4 / §2.5 规定的顺序在 Host 侧完成全部参数校验，不扫描 Device 内存：

1. `handle == nullptr` ⇒ `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `transa ∉ {N, T, C}` 或 `transb ∉ {N, T, C}` ⇒ **`ACLBLAS_STATUS_INVALID_ENUM`**
3. `m < 0 || n < 0 || k < 0` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`
4. 前导维下界校验：
   - `lda < max(1, transa == N ? m : k)` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`
   - `ldb < max(1, transb == N ? k : n)` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`
   - `ldc < max(1, m)` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`
5. `alpha == nullptr || beta == nullptr` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`
6. **Quick Return**：`m == 0 || n == 0` ⇒ 直接返回 `ACLBLAS_STATUS_SUCCESS`（不解引用 A/B/C，不启动任何 kernel）
7. `C == nullptr` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`（此时必有 `m > 0 && n > 0`）
8. 在 Host 侧读取 `alphaVal = *alpha`、`betaVal = *beta`，计算 `hasProduct = (k > 0) && !(alphaVal.real == 0 && alphaVal.imag == 0)`；当 `hasProduct && (A == nullptr || B == nullptr)` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`

| 典型边界/负向组合 | 期望返回码 | 决定步骤 |
| --- | --- | --- |
| `handle == nullptr`（任意维度，含 `m=n=k=0`） | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 步骤 1 |
| `transa=999` 或 `transb=999`（即使 `m=0`） | `ACLBLAS_STATUS_INVALID_ENUM` | 步骤 2 |
| `m=0, n=8, A=B=C=nullptr, lda=ldb=ldc=1` | `ACLBLAS_STATUS_SUCCESS` | 步骤 6 先于步骤 7/8 |
| `m=8, n=8, k=0, A=B=nullptr, C!=nullptr` | `ACLBLAS_STATUS_SUCCESS`（仅做 `beta*C`） | 步骤 8 跳过 A/B 判空 |
| `m=8, n=8, k=8, alpha=(0,0), A=B=nullptr, C!=nullptr` | `ACLBLAS_STATUS_SUCCESS`（仅做 `beta*C`） | 步骤 8 跳过 A/B 判空 |
| `m=8, n=8, k=0, C=nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 步骤 7 |

### 3.3.3 分核与 K 分块规划策略

- **AIV 退化缩放 / 后处理**：按 1D 连续任务数或 2D 批量任务数 `totalTasks`，设 `blocks = min(aivCores, max(1, totalTasks))`。
- **AIV 小形状路径**：设 `blocks = min(aivCores, max(1, n))`（每核负责若干完整输出列；`m=n=k=1` 时自动收缩为 1 核）。
- **AIV 预处理路径**：合并 $A$ 任务与 $B$ 任务至同一任务空间 `totalTasks = aTasks + bTasks`，设 `blocks = min(aivCores, max(1, totalTasks))`。
- **AIC Cube + K 分块规划**：
  1. 对齐维度：`mPad = AlignUp(m, 16)`，`nPad = AlignUp(n, 16)`，`kAlign = (isTransA || k >= 64) ? 64 : 32`，`kPad = AlignUp(k, kAlign)`。
  2. 按 §3.2.4 规则选定 `tier`（`LARGE` / `MEDIUM` / `SMALL`）与 `tileM / tileN`。
  3. 计算单轮所需 K 切分：以精度安全上限 `K_ACCURACY_SPAN = 256`（且兼顾占用率）确定总 K 段数 `totalKParts = max(CeilDiv(kPad, 256), occupancyParts)`。
  4. 判断采用单轮 Split-K 还是多轮 Host K-Chunk：
     - 若单轮产品区 $3 \cdot \text{totalKParts} \cdot M_{\text{pad}} N_{\text{pad}} \cdot 4\text{ B} \le 128\text{ MiB}$ 且 $\text{totalKParts} \le 16$，则 `kLoops = 1`、`kParts = totalKParts`，一次 Cube + 一次 Combine 完成。
     - 否则进入多轮 Host K-Chunk 模式（每轮 `kPartsPerLoop` 满足 $3 \cdot k_{\text{partsPerLoop}} \cdot M_{\text{pad}} N_{\text{pad}} \cdot 4\text{ B} \le 192\text{ MiB}$，并分配 2 张常驻累加平面 `accRe`、`accIm` 及 1 张峰值量级平面 `accScale`），预处理仍只执行 1 次。

### 3.3.4 Workspace 布局与溢出守卫

Workspace 复用 handle 托管内存（`EnsureDefaultWorkspace` / `GetEffectiveWorkspace`），不在接口内额外 `aclrtMalloc`/`aclrtFree`，避免热路径分配开销。全部字节计算使用 `CBlas3TryMulU64` 与 `CBlas3TryAddU64` 溢出检查，溢出或超出 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）时返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

| 子区 | 大小（float 数量） | 用途 |
| --- | --- | --- |
| `aPlanes` | $3 \times M_{\text{pad}} \times K_{\text{pad}}$ | 连续存放 $A_r$、$s_A A_i$、$A_r + s_A A_i$ 三张实数平面 |
| `bPlanes` | $3 \times K_{\text{pad}} \times N_{\text{pad}}$ | 连续存放 $B_r$、$s_B B_i$、$B_r + s_B B_i$ 三张实数平面 |
| `pPlanes` | $3 \times k_{\text{partsPerLoop}} \times M_{\text{pad}} \times N_{\text{pad}}$ | 存放当前轮次各 K 分区的 $P_0, P_1, P_2$ 实数乘积 |
| `accPlanes`（仅 `kLoops > 1`） | $3 \times M_{\text{pad}} \times N_{\text{pad}}$ | 多 K 轮常驻累加区：`accRe`、`accIm`、`accScale`（最大块中间量级，供末轮稀疏抵消快筛） |

**最大测试规模核算（$4096 \times 4096 \times 4096$）**：
- `aPlanes` = $3 \times 64\text{ MiB} = 192\text{ MiB}$
- `bPlanes` = $3 \times 64\text{ MiB} = 192\text{ MiB}$
- `pPlanes`（`kPartsPerLoop = 1`，`kSpan = 256`，`kLoops = 16`）= $3 \times 64\text{ MiB} = 192\text{ MiB}$
- `accPlanes` = $3 \times 64\text{ MiB} = 192\text{ MiB}$
- **合计 768 MiB**，远低于 handle workspace 上限 2 GiB 及 Atlas 800T A2 的 64 GiB HBM 容量。

### 3.3.5 Tiling 结构体定义

```cpp
struct Cgemm3mPreprocTiling {
    uint32_t aRows, aCols, lda, aRowsPad, aColsPad, conjA;
    uint32_t bRows, bCols, ldb, bRowsPad, bColsPad, conjB;
    uint64_t planeAFloats, planeBFloats;
};

struct Cgemm3mCubeTiling {
    uint32_t mPad, nPad, kPad, transA, transB;
    uint32_t tileM, tileN, nTiles, mTiles;
    uint32_t kOffset, kParts, kPartSpan, tier;
    uint64_t planeAFloats, planeBFloats, planePFloats;
};

struct Cgemm3mCombineTiling {
    uint32_t m, n, k, ldc, ldP;
    uint32_t kParts, totalKChunks;
    uint32_t hasProduct, mode; // 0: direct final, 1: accum first, 2: accum mid, 3: accum final
    uint32_t transA, transB, kPad;
    uint64_t planeAFloats, planeBFloats, planePFloats;
    float alphaRe, alphaIm, betaRe, betaIm;
};

struct Cgemm3mSmallTiling {
    uint32_t m, n, k, lda, ldb, ldc, transA, transB;
    float alphaRe, alphaIm, betaRe, betaIm;
};
```

## 3.4 Kernel 侧设计

### 3.4.1 `cgemm3m_preproc_kernel`（AIV_ONLY）

- **职责**：将交错存储的复数矩阵 $A$ 和 $B$ 同步拆分至 `aPlanes` 与 `bPlanes`，吸收共轭负号，生成 $A_r + s_A A_i$ 与 $B_r + s_B B_i$，并对超出逻辑行列的填充区写 0。
- **执行分支**：
  1. **1D 连续双缓冲快路径**（`ld == rows && rowsPad == rows && colsPad == cols`）：将整张矩阵按 `PREPROC_1D_CHUNK = 2048` 复数元素切分，利用双 UB 缓冲（Slot 0/1，总 UB 约 100 KB）流水化执行 `DataCopy(GM→UB)` → `GatherMask(re/im)` + 可选 `Muls(-1)` + `Add(sum)` → `DataCopy(UB→GM)`，掩盖访存延迟。
  2. **2D 多列批量路径**（`rowsPad <= 1024` 且存在 padding 或非紧凑 `ld`）：每任务处理 `colsPerBatch = min(colsPad, 1024 / rowsPad)` 列。先将 UB 缓冲置零，再用单次 2D `DataCopyPad`（`blockCount = validCols`）搬入有效列，整批执行 `GatherMask`、`Muls(-1)`、`Add`，显式清零行尾 `[rows, rowsPad)`，最后 3 次连续 `DataCopy` 写出整批至 `dst0/dst1/dst2`。
  3. **通用长列分块路径**（`rowsPad > 1024`）：按 `PREPROC_CHUNK = 1024` 逐行段搬运与补零，保证任意大维度 UB 占用恒定。

### 3.4.2 `cgemm3m_cube_*_kernel`（AIC_ONLY）

- **职责**：单次下发完成当前 K 轮次内全部 $3 \times k_{\text{parts}} \times n_{\text{tiles}} \times m_{\text{tiles}}$ 个输出 tile 的 FP32 实数矩阵乘。
- **核心流程**：
  1. 首行执行 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)`。
  2. 初始化静态 `MatmulImpl`（`mm.Init`, `mm.DisableBias`, `mm.SetOrgShape(nPad, mPad, kPad, kPad, mPad)` 置于循环外避免重复配置）。
  3. 每 AIC 核以步长 `GetBlockNum()` 遍历任务索引 `task`，解码 `(prodIdx, kPart, nTile, mTile)`，其中 `kBase = t.kOffset + kPart * t.kPartSpan`。
  4. 按 `LeftTrans`/`RightTrans` 编译期分支计算 `left`（来自 $B^{(prodIdx)}$）、`right`（来自 $A^{(prodIdx)}$）与 `output`（写入 $P^{(prodIdx, kPart)}$）的精确偏移，调用 `mm.SetSingleShape` → `mm.SetTensorA` → `mm.SetTensorB` → `mm.IterateAll(output, 0)`。

### 3.4.3 `cgemm3m_combine_kernel`（AIV_ONLY）

- **职责**：执行 K 分区先抵消再归约、中间多 K 轮累加、复数 $\alpha/\beta$ 缩放、稀疏抵消纠正以及交错写回 $C$（严格保护 `ldc` padding 行）。
- **核心流程**：
  1. **中间累加模式（`mode == 1 / 2`）**：按 1D 连续 2048 元素双缓冲读入当前 K 轮的 $P_0, P_1, P_2$，计算 `dRe = P0 - P1`、`dIm = P2 - (P0 + P1)` 与 `dScale = |P0| + |P1| + |P2|`，首次覆盖（`mode == 1`）或累加（`mode == 2`）至 `accRe / accIm / accScale`，不触碰 $C$。
  2. **末轮 / 单轮输出模式（`mode == 0 / 3`）**：
     - 对所有 `kp = 0..kParts-1` 逐段计算 `re += P0[kp] - P1[kp]`、`im += P2[kp] - (P0[kp] + P1[kp])` 并维护块内最大中间量级 `maxAbsP`。
     - 若为 1D 连续且单位标量（`ldc == m && ldP == m && alpha==(1,0) && beta==(0,0)`），直接将 `re, im` 写入 `crci` 缓冲并双缓冲 `Gather` 交错写回 $C$；否则执行一般复数 `ApplyComplexAlpha` 与 `ApplyComplexBeta`（`beta == (0,0)` 时不读取 $C$ 旧值）。
     - **稀疏抵消守卫**：基于 `maxAbsP`、`totalKChunks` 与 `|alpha|` 计算噪声估计向量 `noise`，经向量比较 + `ReduceMax` 判定当前 UB 块是否存在潜在超差元素。若无（绝大多数分块），直接进入 `StoreInterleavedC`；若有，则仅对触发的单个 `(row, col)` 从 `aPlanes`/`bPlanes` 读取 $(A_r, A_i', B_r, B_i')$ 执行有序复数点积并覆写 `cr[idx]` / `ci[idx]`，再随整块交错写出至 $C$。
  3. **多列批量路径**：对 `m < 1024` 且存在 padding 的形状，批量读取多列并利用预构建交错索引打包后一次性写回，大幅降低小列停顿。

### 3.4.4 `cgemm3m_small_kernel`（AIV_ONLY）

- **职责**：零 workspace 单 kernel 处理 $m, n, k \le 64$ 的小形状矩阵乘。
- **核心优化**：
  - 当 `transA == N && lda == m` 时，单次 `DataCopyPad` 载入全部 $m \times k$ 个复数元素（最大 $64 \times 64 \times 8\text{ B} = 32\text{ KB}$），一次 `GatherMask` 解交错，将 $A$ 载入 DMA 次数从 $k$ 次降至 1 次。
  - 沿 $k$ 维逐层执行 3M 外积：`p0 = ar_l * b0`、`p1 = ai_l * b1`、`p2 = (ar_l + ai_l) * (b0 + b1)`，以 `stepRe = p0 - p1`、`stepIm = p2 - (p0 + p1)` 立即累加，从源头避免跨 $k$ 步抵消。
  - 当单步 $B$ 分量发生强抵消（$|b_0 + b_1| < 0.25(|b_0| + |b_1|)$，如 `RANDOM_ALTER`）时，自动切换为共轭差形式或直接实虚外积以守住 `matched_ratio ≥ 0.99`。

## 3.5 性能设计

### 3.5.1 判定口径与双套采集方案

依据任务书 §3.3 与 `test_cases/README.md`，性能验收包含两套互补口径：

| 口径 | 采集方式 | 采样约定 | 角色 |
| --- | --- | --- | --- |
| **Kernel 耗时（正式判据）** | `msprof` 采集设备侧 `Task Duration(us)`，对每次 `aclblasCgemm3m` API 调用内发射的全部 kernel 耗时求和（`CGEMM3M_PROFILE=1` 模式下禁用事件包裹、精确同步发射） | warmup 5 次不计入；后续连续 10 次有效样本取算术平均，转为 ms 与 `gpu_ms` 对比 | **正式验收判据**（单 case 倍率 ≥ 0.8 且全部平均倍率 ≥ 0.8） |
| **端到端事件计时（交叉验证）** | GTest 在预置 Device 缓冲上使用 `aclrtEvent` 夹住连续多次 API 调用，打印 `[PERF] case=<name> ... avg_us=<数值>`，由 `verify_performance.py` 解析判定 | warmup ≥ 5 次，每批 ≥ 10 次调用（小形状采用多批取中位数抗 Host 调度抖动） | **辅助交叉验证与快速回归**（不替代正式判据） |

> 注：若目标环境 `msprof op` 模式对 GTest 复合二进制仅产出 `NA`（CANN 常见现象），则采用 `msprof --application="..." --ai-core=on --task-time=on` 采集完整 `op_summary_*.csv` 的 `Task Duration(us)`，按 API 调用分组求和并严格执行 warmup 5 + 10 样本算术平均，并在自测报告中如实说明。

### 3.5.2 性能预算分析（由 `test_cases/gpu_baseline.csv` 推导）

下表全部数值由任务包冻结的 200 条 A100 基线 `test_cases/gpu_baseline.csv` 按 1.25× 耗时预算（≥ 0.8× A100）反算，用于指导前述架构与优化目标（不含 NPU 实测值）：

| 规模档 | 代表用例（`gpu_baseline.csv`） | A100 `gpu_ms` | 1.25× 预算 | 需求特征 | 对应设计落点 |
| --- | --- | --- | --- | --- | --- |
| **极小延迟档** | $1^3$（最紧延迟） | 7.88 us | **9.86 us** | 纯 kernel 启动延迟 | §3.2.3 路径 2：零 workspace 单发射 `cgemm3m_small_kernel`（1 核） |
| **小规模 AIV 档** | $64^3$ | 14.95 us | **18.69 us** | 单次载入 + 低停顿向量外积 | §3.4.4 `transA=N` 单次 DMA 载入 A + 精简流水屏障 |
| **瘦长/短 K 访存档** | $64\times 4096\times 64$ / $63\times 4095\times 63$ | 20.48 / 21.81 us | **25.60 / 27.26 us** | AIV 拆拼带宽与列循环开销 | §3.4.1/3.4.3 1D 连续双缓冲 + 2D 多列批量搬运 + `SMALL` Tile |
| **大输出短 K 档** | $4096\times 4096\times 64$ | 384.10 us | **480.13 us** | 写回 C 带宽（约 320 MB HBM） | §3.4.3 1D 双缓冲 + 单位标量直写快路径 |
| **瘦长长 K 档** | $64\times 4096\times 4096$ / $31\times 79\times 1368$ | 452.51 / 21.20 us | **565.63 / 26.50 us** | Preproc 带宽 + AIC Split-K 并行度 | §3.2.4 `SMALL` Tile + Split-K + 1D/2D 批量预处理 |
| **中等方阵档** | $256^3$ / $271^3$ / $512^3$ | 26.01 / 25.60 / 66.66 us | **32.51 / 32.00 / 83.33 us** | 三阶段均衡，启动+拆拼+Cube | 统一 3-Product Cube + 批量预处理/后处理 |
| **大算力峰值档** | $1024^3$ / $2048^3$ / $4096^3$ / $4095^3$ | 0.459 / 3.389 / 21.88 / 21.93 ms | **0.573 / 4.236 / 27.34 / 27.42 ms** | 最高需 20.11 T 复数算力（15.08 T 3M 实数 Cube） | §3.2.4 `LARGE` $256\times 256$ L0C 双缓冲 + 16/64 对齐零填充 + K 分块归约 |

## 3.6 支持硬件

| 硬件 | 支持 | 说明 |
| --- | --- | --- |
| Atlas A2 训练/推理系列 | 是 | arch22；自验证与性能验收设备为 Atlas 800T A2（910B3） |
| Atlas A3 训练/推理系列 | 是 | arch22，与 A2 共用本实现；核数运行时查询自动适配 |
| Ascend 950PR / 950DT | 已有实现 | 由既有 `blas/gemm/arch35/` 承担，本任务不改动 |

`blas/gemm/README.md` 的 `aclblasCgemm3m` 章节须随本任务更新，补充 Atlas A2/A3（arch22）支持情况与实现约束说明。

## 3.7 算子约束限制（任务书 §2.5 逐项核对）

| 约束项 | 任务书内容 | 本设计结论 | 落点 |
| --- | --- | --- | --- |
| 参数合法性与校验层级 | handle、transa/transb、m/n/k、lda/ldb/ldc、alpha/beta/A/B/C 指针均在 Host 侧校验并返回确定错误码；不扫描 Device 元素 | **满足**，8 步固定校验顺序全在 Host 侧执行 | §3.3.2 |
| 非连续 Tensor 支持 | 按 lda/ldb/ldc 列主序带填充布局访问，不使用 AutoContiguous/ViewCopy | **满足**，全路径原生支持任意合法 `lda`/`ldb`/`ldc` 并保护 padding 行 | §3.4.1 / §3.4.3 |
| broadcast 规则 | 不支持 | **不涉及**，三操作数矩阵不做广播 | — |
| dynamic shape 要求 | 不要求，运行时 Host 参数驱动 tiling | **满足**，`m/n/k/lda/ldb/ldc` 与 `transa/transb` 均为运行时入参 | §3.3.3 |
| 原地与视图语义 | C 原地覆写，不返回视图；调用方保证 C 与 A/B 存储不重叠 | **满足**，A/B 全程只读，C 仅覆写逻辑 $m \times n$ 区域 | §3.4.3 |
| 确定性计算要求 | 不要求；按形状与片上容量选择路径，不按测试编号或随机种子分派 | **满足**，路径选择仅依赖 `(m, n, k, trans, alpha, beta)` 与硬件核数，且同一输入重复执行确定 | §3.2.3 / §3.3.3 |
| 空 Tensor 处理 | `m==0 \|\| n==0` 合法 no-op，允许 A/B/C 为空；`k==0 \|\| alpha==0` 只做 `beta*C`，允许 A/B 为空但 C 非空 | **满足**，在 §3.3.2 步骤 6~8 精确区分 | §3.3.2 |
| 0 维 Tensor 处理 | 不涉及 | **不涉及**，矩阵以指针与维度参数描述 | — |
| 异步执行 | 绑定 `handle->stream` 异步执行，Host 读回前须同步 stream | **满足**，所有 kernel 异步下发至 `handle->stream`，Host 侧无隐式同步 | §3.3.1 |

---

# 4 特性交叉分析

| 特性 | 与本算子的交叉与处理 |
| --- | --- |
| 9 种转置/共轭组合 | `OP_C` 共轭负号在 `preproc`/`small` 吸收进虚部平面；4 种实数布局组合对应 4 组静态 `MatmulImpl`（`NN/NT/TN/TT`），确保 `OP_C` 与 `OP_T` 行为正确分离 |
| 非对齐形状（如 $4095^3$、质数维） | 预处理阶段零填充对齐到 $M_{\text{pad}}/N_{\text{pad}}$（16 对齐）与 $K_{\text{pad}}$（32/64 对齐），Cube 满效率计算，后处理严格按逻辑 $m \times n$ 写回 |
| `lda`/`ldb`/`ldc` Padding | 预处理与后处理按 `lda`/`ldb`/`ldc` 寻址，仅覆写 $C$ 的 `[0, m) × [0, n)`，保护 `[m, ldc)` 哨兵区域不被改写 |
| `m == 0 \|\| n == 0` | 结构与标量指针合法后立即返回 `SUCCESS`，不访问 A/B/C、不启动 kernel |
| `k == 0 \|\| alpha == (0,0)` | 跳过预处理与 Cube；`beta == (1,0)` 直接 no-op；否则仅启动 1 次 `combine` 执行 `C = beta * C`，允许 `A/B` 为 `nullptr` 或含 NaN |
| `beta == (0,0)` × 旧 C 含 NaN | `ApplyComplexBeta` 与单位标量快路径在 `beta == (0,0)` 时完全不读取 $C$ 旧值，避免 `0 * NaN = NaN` 污染 |
| 特殊标量（纯虚、负值、$\pm 10^4$、随机复数） | `ApplyComplexAlpha`/`ApplyComplexBeta` 完整支持一般复数乘法；大 $|\alpha|$ 放大 3M 抵消时由稀疏纠正守卫保障精度 |
| 长 $K$ × 非零均值输入（`GAUSS_NORM_0_1`） | K 分块上限 256 + 逐块先抵消再累加，防止 $P_2$ 幅值随 $K$ 膨胀突破 `32·ULP` |
| 交替符号输入（`RANDOM_ALTER`） | `P2 - (P0 + P1)` 顺序 + 强抵消步保护，防止 $P_0 \approx -P_1$ 吞没 $P_2$ 低位 |
| Inf / NaN 传播 | 有效计算路径不拦截 Inf/NaN，自然参与 3M 乘加与传播；非有限值按任务书等价规则判定 |
| 小形状（$1^3 \sim 64^3$） | 走零 workspace 单发射 `cgemm3m_small_kernel`，满足 9.86 ~ 18.69 us 延迟预算 |
| 大形状（$4096^3$） × Workspace 上限 | 多 K 轮累加将峰值 workspace 控制在 768 MiB，低于 2 GiB handle 限制，且通过溢出检查防越界 |
| 异步执行 × Stream 复用 | 预处理 → Cube → 后处理依次下发至同一 `handle->stream`，硬件流内保序，Host 零隐式同步 |

---

# 5 可维可测分析

## 5.1 精度标准 / 性能标准 / 内存标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| **精度标准** | 以 CPU `cblas_cgemm`（Netlib BLAS 3.10.0 参考语义）为 golden，实部、虚部分别按 FLOAT32 混合容差判定：`atol = rtol = 2^-13`（约 `1.2207e-4`），`matched_ratio ≥ 0.99`，且 `max_abs_error ≤ max(1e-2, 32·ULP(golden))`；Inf/NaN 按非有限值规则独立判定；`ldc` padding 哨兵逐字节不变 | 任务书 §3.2；[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| **性能标准** | 正式判据：Atlas 800T A2（910B3）上 `msprof` 采集单次 API 全部 kernel 的 `Task Duration(us)` 之和（warmup 5 次 + 10 次有效样本算术平均），须满足单 case 及全部 200 条平均倍率均 ≥ 0.8（即 NPU ≤ 1.25 × A100 `gpu_ms`）；辅助交叉验证：`verify_performance.py` 解析 GTest `[PERF] avg_us` 同口径比对 | 任务书 §3.3；`test_cases/gpu_baseline.csv` |
| **内存标准** | **不涉及**（任务书 §3.4 明文）。交付件仍按模板提供 `4.1 内存自验证报告.xlsx` 与 `4.2 内存自验证日志.log` 并注明依据 | 任务书 §3.4 / §4 |

> 本节为设计阶段验收契约；实际 NPU 精度通过率、性能倍率与 profiling 数据将在完成实现并实机运行后，完整记录于 `task_submission/` 自测报告与原始日志中。

## 5.2 兼容性分析

- 公开接口声明沿用 `include/cann_ops_blas.h` L584，**零改动**，与 arch35 共用。
- 新增实现局限于 `blas/gemm/arch22/cgemm3m_*`，新增测试局限于 `test/gemm/cgemm3m/arch22/*`，不修改任何 arch35 文件、公共测试头（`cgemm3m_param.h` / `cgemm3m_golden.h` / `cgemm3m_verify.h`）及 `blas/gemm3m/`。
- `test/gemm/cgemm3m/CMakeLists.txt` 仅扩展 `arch22` 构建分支，arch35 保持完全兼容。
- 更新 `blas/gemm/README.md` 的 `aclblasCgemm3m` 章节，补充 Atlas A2/A3 支持说明。

## 5.3 测试设计

### 5.3.1 测试工程设计

测试工程位于 `test/gemm/cgemm3m/arch22/`，直接复用公共头 `cgemm3m_param.h`、`cgemm3m_golden.h`、`cgemm3m_verify.h` 与 `test/frame/fill.h`：

| 测试模块 | 覆盖内容 | 说明 |
| --- | --- | --- |
| `Cgemm3mArch22Test.*`（8 条接口与语义单测） | 1. `NullHandleReturnsHandleIsNullptr`<br>2. `InvalidEnumReturnsInvalidEnum`<br>3. `EmptyOutputPreservesSentinel`<br>4. `ZeroProductAllowsNullInputsAndScalesC`（含 `beta=0` 配 NaN）<br>5. `ConjugateVsTransposeDistinct`<br>6. `PaddingSentinelPreservedAcrossSmallAndCubePaths`<br>7. `SplitKPathMatchesGolden`<br>8. `CsvCaseCountAndUniqueness` | 补齐 CSV 无法表达的 `handle == nullptr`、跨路径 padding 保护与 Split-K 回归 |
| `Cgemm3m/Cgemm3mTest.CsvDriven/*`（1000 条精度用例） | 覆盖任务包 `cgemm3m_test.csv` 全部 1000 条非 `TC_PF_` 用例，校验返回码、实部/虚部混合容差与 `ldc` padding 写保护 | 由 `verify_accuracy.py` 驱动执行 |
| `TC_PF_*` 双模式性能运行（200 条性能用例） | 默认事件计时模式（warmup 5 + 连续调用取均值并打印 `[PERF] case=... avg_us=...`，供 `verify_performance.py` 交叉验证）；设置 `CGEMM3M_PROFILE=1` 时切换为纯净同步发射模式（warmup 5 + samples 10，无事件包裹，供 `msprof` 正式采集 kernel Task Duration） | 不修改任务包任何脚本或 CSV |

### 5.3.2 任务包静态核查与规格偏差记录

在对任务包 `test_cases/` 进行只读核查时，发现以下三项规格与脚本偏差（遵循「严禁改动任务包原始文件」红线，在设计文档与自测报告中如实记录并在测试封装层妥善处理）：

1. **`gen_csv.py` 短转置字符导致 `TC_PF_*` 前导维不匹配**：`test_cases/gen_csv.py` L97-102 的 `rows_a`/`rows_b` 判断 `ta == "ACLBLAS_OP_N"`，但 `PERF_CASES`（L117-316）使用单字符 `'N'`/`'T'`/`'C'`，导致 `rows_a('N', m, k)` 误返回 `k`、`rows_b('N', n, k)` 误返回 `n`。200 条 `TC_PF_*` 中有 110 条的 `lda`/`ldb` 与紧凑列主序不符，其中 **60 条严格违反 BLAS 下界**（例如 `TC_PF_0005`：`N,N,4096,64,64,lda=64 < 4096`）。而 `gpu_baseline.csv` 已按紧凑 GEMM 采集全部 200 条 `gpu_ms`，且 `verify_performance.py` 按 CSV 原始列做精确键匹配。**处理方案**：保持 `cgemm3m_test.csv` 与 `gpu_baseline.csv` 逐字节不变，仅在 `test/gemm/cgemm3m/arch22/cgemm3m_test.cpp` 的 `RunPerfCase()` 中通过 `NormalizePerfParam()` 将 `TC_PF_*` 的 `lda`/`ldb`/`ldc` 归一化为紧凑下界（与 `test_cases/README.md`「性能用例均为连续访存」一致）；全部 1000 条精度与负向用例完全使用 CSV 原始 `lda`/`ldb`/`ldc`，不受任何影响。
2. **`test_cases/README.md` 分类条数与实际 CSV 略有出入**：实际 1000 条精度用例分布为 `TC_L0=18`、`TC_SQ=110`、`TC_MN=91`、`TC_AB=88`、`TC_LD=16`、`TC_FL=21`、`TC_CV=32`、`TC_ED=38`、`TC_EX=586`（README 旧表写为 `AB=32, EX=642`），总条数均为 1000 精度 + 200 性能 = 1200 条。
3. **`test_cases/README.md` 残留 `gpu_ms` 占位说明**：README L161 仍称 `gpu_baseline.csv` 的 `gpu_ms` 为空，实际核查该文件 200 条 `gpu_ms` 均已完整回填（含 25 条非默认 `alpha`/`beta` 用例），可直接严格验收（不加 `--allow-noref`）。

### 5.3.3 用例分层对应

| 分层 | 覆盖内容 | 对应条目 |
| --- | --- | --- |
| 基础与全转置组合 | 9 种 `N/T/C` 组合 × 小尺寸 | `TC_L0`（18） |
| 方阵与矩形尺寸扫描 | $1 \sim 4096$（含质数、$2^p \pm 1$、$1000\times 1001\times 999$、$4095^3$、$4096^3$） | `TC_SQ`（110）+ `TC_MN`（91）+ `TC_CV`（32） |
| 标量与填充覆盖 | 特殊标量（0/1/-1/纯虚/$\pm 10^4$/随机复数）、7 种矩阵填充（均匀/高斯/全零/交替/极值/Inf/NaN） | `TC_AB`（88）+ `TC_FL`（21）+ `TC_EX`（586） |
| 前导维 Padding | `lda`/`ldb`/`ldc` 非紧凑偏移与哨兵写保护 | `TC_LD`（16）+ 单元回归 |
| 边界与负向 | `m/n/k=0`、`alpha=0`、`beta=0` 配 NaN、空指针组合、非法枚举 `999`、负维度、非法前导维、`handle=nullptr` | `TC_ED`（38）+ `Cgemm3mArch22Test.*`（8） |
| 性能全集 | 极小延迟档 + 瘦长/短 K 档 + 方阵对数扫描 + 矩形 + $4095^3/4096^3$ + 12.5% 非默认标量 | `TC_PF`（200） |

## 5.4 可测性与复现步骤

- **构建命令**：`bash build.sh --soc=ascend910b3 --ops=cgemm3m`（需 `libblas-dev` / `libopenblas-dev` 提供 `cblas_cgemm` 参考库）。
- **精度与单测复现**：
  1. 单元测试：`./build/test/gemm/cgemm3m/cgemm3m_test --gtest_filter="Cgemm3mArch22Test.*"`
  2. 全量精度：`python3 test_cases/verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 0 --timeout 7200`
- **性能复现**：
  1. 正式 kernel-to-kernel 采集：`CGEMM3M_PROFILE=1 CGEMM3M_WARMUP=5 CGEMM3M_SAMPLES=10 msprof --application="./build/test/gemm/cgemm3m/cgemm3m_test --gtest_filter=*TC_PF*" --output=./prof_cgemm3m --ai-core=on --task-time=on`
  2. 端到端交叉验证：`taskset -c 0-7 python3 test_cases/verify_performance.py --repo /path/to/ops-blas --soc ascend910b3 --skip-build --device 0 --timeout 3600`

---

# 6 交付与合入

## 6.1 交付物清单

| 交付物 | 位置 |
| --- | --- |
| 公开接口声明 | `include/cann_ops_blas.h` L584（已存在，本次**无改动**） |
| 算子实现 | `blas/gemm/arch22/` 共 4 个文件：`cgemm3m_tiling_data.h`、`cgemm3m_kernel.h`、`cgemm3m_host.cpp`、`cgemm3m_kernel.cpp` |
| 算子文档 | `blas/gemm/README.md`（更新 `aclblasCgemm3m` 章节补充 Atlas A2/A3 支持） |
| 测试工程 | `test/gemm/cgemm3m/arch22/cgemm3m_test.cpp`、`test/gemm/cgemm3m/arch22/cgemm3m_test.csv` 及 `test/gemm/cgemm3m/CMakeLists.txt`（新增 `arch22` 分支） |
| 设计文档 | 本文档（提交至竞赛/任务设计文档仓库） |
| 自验证交付包 | 个人代码仓根目录 `task_submission/`（含任务书 §4 规定的全部 7 个文件） |

`task_submission/` 目录结构：

```text
task_submission
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx      # 按「不涉及」口径出具（任务书 §3.4）
└── 4.2 内存自验证日志.log        # 同上
```

## 6.2 规格歧义的核定结果

| # | 事项 | 核定结论 | 依据 |
| --- | --- | --- | --- |
| a | 非法 `transa`/`transb` 返回码 | **`ACLBLAS_STATUS_INVALID_ENUM`**。arch35 `cgemm_host.cpp` L713 将 `cgemm3m` 的 `INVALID_ENUM` 改写为 `INVALID_VALUE`，但 A2/A3 任务书 §2.3/§2.4 及任务包 `TC_ED_396`/`TC_ED_397` 均要求返回 `ACLBLAS_STATUS_INVALID_ENUM`，故 arch22 严格按任务书与用例契约实现 | 任务书 §2.3/§2.4、`cgemm3m_test.csv` `TC_ED_396/397`、`arch35/cgemm_host.cpp` L713 |
| b | `alpha`/`beta` 内存域 | **Host 内存指针**。arch22 `cgemm_host.cpp` L464 从 Device 读取标量，但任务书 §2.4 与 `arch35/cgemm_host.cpp` L719-720 明确 `alpha`/`beta` 为 Host 指针；arch22 `aclblasCgemm3m` 直接在 Host 侧解引用 | 任务书 §2.4、`arch35/cgemm_host.cpp` L719-720 |
| c | `TC_PF_*` CSV 前导维缺陷 | 保持原始 `cgemm3m_test.csv` 与 `gpu_baseline.csv` 逐字节不变，仅在 `RunPerfCase()` 中将 `TC_PF_*` 归一化为紧凑 `lda/ldb/ldc`，精度用例维持原始值 | `test_cases/gen_csv.py` L97-102、L117-316、`test_cases/README.md` L44 |
| d | 禁止放置于 `blas/gemm3m/` | `blas/gemm3m/` 为实数 `aclblasSgemm3m` 专属目录；复数 `aclblasCgemm3m` 实现置于 `blas/gemm/arch22/` | 任务书 §2.2 |
| e | 性能正式判据口径 | 正式判据为 `msprof` kernel-to-kernel Task Duration（单次调用内全部 kernel 求和）对比 `gpu_ms`；`verify_performance.py` 的 `[PERF] avg_us` 仅作交叉验证 | 任务书 §3.3、`test_cases/README.md` L10、L124 |

## 6.3 已知限制与后续优化方向

| 项 | 说明 | 后续优化方向 |
| --- | --- | --- |
| Workspace 占用 | 通用 Cube 3M 路径需在 handle workspace 中开辟三组实数平面与中间产品区（$4096^3$ 峰值约 768 MiB），受 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）约束；超出时返回 `ACLBLAS_STATUS_ALLOC_FAILED` | 若需支持更大方阵（如 $8192^3$），可在 Host 侧增加 M/N 外层分块（Panel Blocking），以常数级 workspace 覆盖任意维度 |
| 多阶段跨核融合 | 当前大/中矩阵采用 `preproc`（AIV）→ `cube`（AIC）→ `combine`（AIV）三阶段分离发射，通过 HBM 交换实数平面 | 后续可探索 Mix-Core（CV 融合）在 L1/UB 间直接流水衔接，进一步削减中等规模的 HBM 读写流量 |
| 极端抵消点纠正开销 | 稀疏有序重算仅针对向量快筛命中的高风险抵消元素，在常规与性能分布下几乎零触发；若用户构造全矩阵极端抵消配超大 $|\alpha|$，AIV 重算耗时将随命中比例上升 | 可对高命中率块切换为向量化双精度/补偿（Dekker）点积以提升最坏情况吞吐 |

---

# 7 参考资料

1. cuBLAS 官方文档（`cublasCgemm3m` 对标接口）：<https://docs.nvidia.com/cuda/cublas/index.html>
2. Netlib BLAS 参考实现（`cgemm.f`）：<https://www.netlib.org/blas/cgemm.f>
3. ops-blas 开源仓（含 `blas/gemm/arch35/cgemm3m_host.cpp` 与 `blas/gemm/arch22/`）：<https://gitcode.com/cann/ops-blas>
4. ops-blas 贡献指南：<https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md>
5. 生态算子开源精度标准（`mixed_tolerance_standard.md`）：<https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>
6. Ascend C 算子开发文档与 API 参考：<https://www.hiascend.com/document>
7. 本任务需求与测试规范：`aclblasCgemm3m_AtlasA2_A3_task_doc.md`、`test_cases/README.md`
