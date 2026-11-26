# aclblasCgemm3m 算子设计文档（Atlas A2/A3）

## 1. 需求来源与背景

### 1.1 任务来源与标杆
- **任务名称**：`11月社区任务-aclblasCgemm3m算子开发（A2A3）`
- **目标硬件**：Atlas A2 / Atlas A3（`arch22`，SOC 架构以 `Ascend910B3` 为典型测试平台，支持 CANN 9.1+）。
- **功能参考与精度标杆**：Netlib Reference-LAPACK v3.10.0 `cblas_cgemm3m`（复数单精度矩阵乘 3M 算法等价语义）。
- **性能标杆**：任务书附带 NVIDIA A100 GPU 基准耗时（`gpu_baseline.csv`，共 200 个 `TC_PF_*` 性能用例），要求 Atlas A2/A3 NPU 实现达到 **单用例性能比（`gpu_ms / npu_ms`）≥ 0.8 且全部 200 个性能用例平均性能比 ≥ 0.8**（5 次预热 + 10 次正式执行算术平均，`msprof` / `aclrtEvent` 交叉验证）。

### 1.2 现有能力与本任务目标
- 仓库 `cann/ops-blas` 现有 `cgemm3m` 实现与测试用例主要面向 `arch35`，在 `arch22` 目录下尚无正式交付版本，且原有原型存在：
  1. `transa`/`transb` 非法枚举返回码误用 `ACLBLAS_STATUS_INVALID_VALUE`（规格要求 `ACLBLAS_STATUS_INVALID_ENUM`）；
  2. 大规模奇偶非对齐维度与非默认 `alpha`/`beta` 组合场景缺乏高吞吐路径和全面自测覆盖；
  3. 需要严格遵循 `repo-coding-rules`（函数复杂度 NBNC ≤ 50、嵌套深度 ≤ 5、`#pragma once` 等）。
- 本任务目标：在 `blas/gemm/arch22/` 下使用 Ascend C + CANN Catlass 原生接口实现高性能、高精度、低复杂度且完全满足 `repo-coding-rules` 与安全红线的 `aclblasCgemm3m` 算子，并在 `test/gemm/cgemm3m/arch22/` 建立完整 1200 条 CSV + 单元测试自测体系。

---

## 2. 接口与功能约束

### 2.1 函数原型（`include/cann_ops_blas.h`，只读不改）
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

### 2.2 参数与数据类型约束
- **数据类型与布局**：复数单精度 `aclblasComplex`（交织存储 `[real, imag]`，等价 `float2`），矩阵 `A`/`B`/`C` 均为 **Column-Major（列主序）**。
- **操作枚举**：`transa`/`transb` 支持 `ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）、`ACLBLAS_OP_C`（共轭转置）。
- **标量指针**：`alpha`、`beta` 支持 Host 或 Device 侧指针，通过内部异步流读回 helper 统一解析。
- **矩阵尺寸与 Leading Dimension**：
  - `m >= 0, n >= 0, k >= 0`；
  - `lda >= max(1, (transa == ACLBLAS_OP_N) ? m : k)`；
  - `ldb >= max(1, (transb == ACLBLAS_OP_N) ? k : n)`；
  - `ldc >= max(1, m)`。

### 2.3 异常与边界语义
严格按参数校验优先级返回状态码：
1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `transa` 或 `transb` 不在 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` → **`ACLBLAS_STATUS_INVALID_ENUM`**；
3. `m < 0 || n < 0 || k < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；
4. `lda`/`ldb`/`ldc` 小于最小允许步长 → `ACLBLAS_STATUS_INVALID_VALUE`；
5. `alpha == nullptr || beta == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
6. 若 `m == 0 || n == 0`：合法空输出，不访问 `A`/`B`/`C` 直接返回 `ACLBLAS_STATUS_SUCCESS`（`ZeroProductAllowsNullInputs` 与 `EmptyOutputPreservesSentinel` 契约）；
7. `C == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
8. 当 `k == 0` 或 `alpha == (0, 0)` 时：不访问 `A`/`B`（允许 `A`/`B` 为空），直接执行 `C = beta * C`（若 `beta == (1, 0)` 则 no-op），返回 `ACLBLAS_STATUS_SUCCESS`；
9. 其余情况检查 `A == nullptr || B == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
10. Workspace 超出 `ACLBLAS_MAX_WORKSPACE_SIZE` → `ACLBLAS_STATUS_ALLOC_FAILED`；硬件核数获取失败 → `ACLBLAS_STATUS_INTERNAL_ERROR`。

---

## 3. 算法与参考

### 3.1 3M 复数矩阵乘数学公式
设 $\text{op}(A) = A_r + i A_i \in \mathbb{C}^{m \times k}$，$\text{op}(B) = B_r + i B_i \in \mathbb{C}^{k \times n}$（若操作为共轭转置 `ACLBLAS_OP_C`，则预处理拆分阶段对虚部取反）。

构造中间实矩阵和：
$$A_s = A_r + A_i \in \mathbb{R}^{m \times k}, \qquad B_s = B_r + B_i \in \mathbb{R}^{k \times n}$$

通过 **3 次 FP32 实数 GEMM**（禁用 HF32，使用严格 IEEE FP32 MAC 精度）计算三个实数中间结果：
$$P_0 = A_r B_r, \qquad P_1 = A_i B_i, \qquad P_2 = A_s B_s$$

最终复数乘积为：
$$\text{Re}(\text{op}(A)\text{op}(B)) = P_0 - P_1, \qquad \text{Im}(\text{op}(A)\text{op}(B)) = P_2 - P_0 - P_1$$

并结合复数标量 $\alpha = \alpha_r + i\alpha_i$、$\beta = \beta_r + i\beta_i$ 回写：
$$C \leftarrow \alpha \cdot \big((P_0 - P_1) + i(P_2 - P_0 - P_1)\big) + \beta \cdot C$$

### 3.2 大幅抵消（Catastrophic Cancellation）精度保护策略
标准 3M 算法利用 $(A_r + A_i)(B_r + B_i)$ 计算虚部时，当行/列交叉项 $A_r B_i + A_i B_r$ 相比单独实部/虚部能量 $|A_r||B_r| + |A_i||B_i|$ 发生**段级或元素级大幅相消**（例如 `SegmentCancellation` 与病态长 $k$ 矩阵）时，3M 中间和 $P_2 - (P_0 + P_1)$ 的舍入误差可能超过标杆容限。为此设计两级轻量级 AIV 保护：
1. **能量边界预估（`cgemm3m_energy_ab_kernel`）**：在 $m,n,k \ge 1536$ 时启用，统计 $\text{op}(A)$ 每行的复数 $L_1$/$\max$ 范数与 $\text{op}(B)$ 每列的 $L_1$/$\max$ 范数；
2. **向量筛选与 Double-Double 精确重算（`cgemm3m_screen_cancel_kernel`）**：用向量比较 `min(rowL1*colMax, rowMax*colL1) > 262144 * max(1, |P0-P1| + |P2-P0-P1|)` 快速定位存在高风险相消的少量输出位置，仅对命中位置以 Double-Double（`TwoSum` / `TwoProd` + `#pragma clang fp contract(off)`）重算精确值，其余绝大多数位置零额外开销。

### 3.3 选择理由与和传统 4M 的差异
- 相比传统 4M 算法（4 次实数 GEMM），3M 将 Cube 计算量降低 **25%**（仅 3 次 Batched/Grouped 实数 GEMM），在大尺寸矩阵上可显著逼近并对齐 GPU cuBLAS 3M 基准（`ratio >= 0.8`）；
- 通过关闭 Cube `HF32Mode::DISABLE` 并叠加 `1536+` 选择性 DD 抵消保护，兼顾了 3M 的性能收益与 4M/参考 Netlib 同等的高数值精度。

---

## 4. Host 与 Kernel 设计

### 4.1 总体执行流与调用路径
Host 入口 `aclblasCgemm3m`（[cgemm3m_host.cpp](file://blas/gemm/arch22/cgemm3m_host.cpp)）按职责拆分为参数校验 `ValidateCgemm3mArgs`、尺寸与对齐规划 `BuildCgemm3mDims` / `BuildWorkspacePlan`、工作区切分 `SliceCgemm3mWorkspace` 与 Kernel 调度 `ExecuteCgemm3mKernels`：
1. **`k == 0 || alpha == 0` 快速路径**：跳过 Workspace 分配与 Split/Cube，直接启动 `cgemm3m_scale_c_do` 执行 $C = \beta C$（若 $\beta = (1,0)$ 直接返回）；
2. **Step 1 — 复数拆分与对齐填充（`cgemm3m_split_ab_kernel`，AIV Only）**：
   - 将交织存储的 `A`/`B` 按列批量搬入 UB，使用 `GatherMask` 分离实部、虚部（`OP_C` 虚部乘 `-1`）并向量 `Add` 得到 $A_s, B_s$；
   - 按 **64 元素对齐**写入 Workspace 中连续 3 个平面 $[A_r, A_i, A_s]$ 与 $[B_r, B_i, B_s]$，消除非对齐 leading dimension 与尾块给 Cube 带来的 Bank 冲突或越界访问。
3. **Step 2 & 3 — 两条计算/回写调度路径**：
   - **融合路径（`cgemm3m_mix_cube3_combine_do`，`KERNEL_TYPE_MIX_AIC_1_2`，当 `!enableGuard` 且 Cube Tile 总数 $\ge$ `aicCores`）**：
     - 1 个 AIC 核依次完成 3 个 Batch（$P_0, P_1, P_2$）对应输出 Swizzle Tile 的矩阵乘，并通过 `CrossCoreSetFlag<2, PIPE_FIX>` 通知配对的 2 个 AIV 子核；
     - 2 个 AIV 子核通过 `CrossCoreWaitFlag<2, PIPE_MTE2>` 按列对半切分直接从 $P_0, P_1, P_2$ 合并计算 $\alpha \cdot \text{prod} + \beta \cdot C$ 并使用 `Gather` 交织回写 `C`，减少一次全局读后写落盘延迟。
   - **分离路径（`cgemm3m_cube3_do` + 可选保护 + `cgemm3m_combine_do`）**：
     - 当启用大矩阵相消保护或小矩阵 Tile 数不足以喂满 `MIX_AIC_1_2` 时，先以 `KERNEL_TYPE_AIC_ONLY` 调度 3 Batch Swizzle Batched Cube（`Catlass::Gemm::MmadAtlasA2Pingpong<true>`，列主序转置等价映射 Row-Major $C^T = B^T A^T$），必要时插入 `cgemm3m_energy_ab_kernel` + `cgemm3m_screen_cancel_kernel`，最后由 `cgemm3m_combine_do` 合并输出。

### 4.2 分块（Tiling）与动态核数策略
- **Shape 自适应 Cube 分块（`DispatchShape`）**：依据 `(m, n, k)` 尺度静态实例化并动态路由 `<TM, TN, TK>` 模板：
  - 瘦长边界矩阵（`n <= 32` 或 `m <= 32`）：优先切分 `TK=256`（`k>=256`）或 `TK=64` 的细粒度 `<32,32,256>` / `<32,64,256>` / `<64,32,256>`，提高核间并行度；
  - 中小方阵（`m,n <= 256`）：`<64, 128, 64>`；
  - 浅 `k` 大输出矩阵（`k <= 96, m*n >= 262144`）：`<128, 64, 64>` 或 `<128, 128, 64>`；
  - 常规中大矩阵：`<128, 128, 64>`（`m,n <= 512`）或最大吞吐 `<256, 128, 64>`。
- **Swizzle Tile 调度（`kSwizzle = 4`）**：`DecodeSwizzleTile` 按 4 行 Swizzle 组织 L2 Cache 复用，提升 $A/B$ 跨 Tile 重用率。
- **Vector 动态分核与 UB 管理**：
  - 使用 `GetAivCoreCount()` / `GetAicCoreCount()` 与 `GetBlockNum()` / `GetBlockIdx()` 动态分配核数，禁止硬编码核数；
  - AIV Kernel 使用局部变量 `TPipe` + `TQue`/`TBuf`（`ChooseDynamicTile32` 动态在 UB 容量内限幅 `kMaxSplitTile = 4096`、`kMaxVecTile = 2048`、`kGuardTile = 2048`），对于支持列步长连续搬运且 `Align8(rows) <= tile/2` 的情况启用 2D 批量 `DataCopyPad`（配合 `ComplexUbColGap32B` 自动对齐 32B 偏移）降低 MTE 发射开销。

---

## 5. 硬件支持与兼容性

- **目标硬件与架构目录**：Atlas A2 / A3（`arch22`，覆盖 Ascend 910B 系列）；代码置于 `blas/gemm/arch22/`，通过 `blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 自动纳入且不干扰现有 `arch35`（Ascend 950）逻辑，不新建 `blas/gemm3m/` 目录、不修改 `include/cann_ops_blas.h`。
- **关键依赖**：
  - CANN Toolkit >= 9.1.0（Ascend C SDK）；
  - Catlass header-only 算子模板库（通过 `cmake/third_party/gemmex-catlass.cmake` 的固定 commit 接入）。
- **Workspace 约束**：总中间缓冲区需小于 `ACLBLAS_MAX_WORKSPACE_SIZE`，且所有字节对齐到 `512 B`。

---

## 6. 自测与性能方案

### 6.1 测试数据集与覆盖率
- 基于任务书 `test_cases/test_case.csv` 构建 `test/gemm/cgemm3m/arch22/cgemm3m_test.csv`，共 **1200 条 CSV 用例**（1000 条精度用例 + 200 条 `TC_PF_1001`~`TC_PF_1200` 性能用例），并保留 8 条 C++ 补充契约/边界/抵消/空指针单测，总计 **1208 条 GTest 用例**：
  - 覆盖 9 种 `transa x transb`（`N/T/C`）组合、6 种 `m/n/k` 对齐尾块及素数维度、`lda/ldb/ldc` 非紧致跨距；
  - 覆盖 4 类典型分布（正态 `RANDOM_NORM_5_5`、零均值正态 `GAUSS_NORM_0_1` 50%/50%、非有限数 `Inf/NaN` 构造等）；
  - 覆盖 `TC_PF_*` 中 25 条非默认 `alpha`/`beta` 性能用例与边界异常返回码 `ACLBLAS_STATUS_INVALID_ENUM`（`TC_ED_396`/`TC_ED_397`）。

### 6.2 Golden 与精度通过准则
- **Golden 参考**：独立编译链接 **Netlib Reference-LAPACK v3.10.0**（`libcblas.so.3.10.0` + `libblas.so.3.10.0`，通过 `LD_PRELOAD` 与编译期 `REFBLAS_*` 变量锁定，严禁使用 OpenBLAS）。
- **数值判定口径**：按 `aclblasCgemm3m` A2/A3 任务书逐元素校验，要求 **所有 1000 条高精度 CSV 用例及补充单测 100% 通过**。

### 6.3 性能采集与达标口径
- **执行参数**：每个 `TC_PF_*` 用例执行 **5 次预热（Warmup）+ 10 次正式执行（Sample）**，取 10 次算术均值；
- **双路度量**：通过 `CGEMM3M_PROFILE=1` 打印 `[PERF] case=<name> avg_us=<val>` 供 `aclrtEvent` 事件核验，并结合 `msprof op` / `msprof` 采得 `Task Duration` 作为硬件核函数实际执行时间，与任务书 `gpu_baseline.csv`（A100 `gpu_ms`）对比计算 `ratio = gpu_ms / npu_ms`；
- **达标目标**：全部 200 条 `TC_PF_*` 用例 **单用例比值 ≥ 0.8 且平均比值 ≥ 0.8**。
- **调优策略**：先以覆盖主要 shape/trans 维度的 1/10 典型子集（20 条）定位瓶颈并完成轮次优化，验证达标后再展开 200 条全量性能基准与全量 1208 条精度回归。
- **内存检查**：Atlas A2/A3 本次任务书明确标明内存安全检查**不涉及**。

---

## 7. 评审问题与变更

- **待确认/评审跟踪**：
  - [x] `transa`/`transb` 非法值返回码已由 `ACLBLAS_STATUS_INVALID_VALUE` 修正为 `ACLBLAS_STATUS_INVALID_ENUM`（与任务书 §2.3/§2.4 及 `TC_ED_396`/`TC_ED_397` 对齐）；
  - [x] `cgemm3m_host.cpp` / `cgemm3m_kernel.cpp` / `cgemm3m_cube3_kernel.cpp` 所有长函数已按 `repo-coding-rules` 拆分至 NBNC ≤ 50、嵌套深度 ≤ 5；
  - [ ] 待收集设计评审意见及后续 A2/A3 1/10 与全量性能数据同步回填。
