# aclblasSgemm / aclblasCgemm / aclblasHgemm 算子设计文档（Atlas A2/A3）

团队及开发者：katiyaaa。设计日期：2026-11-25。

# 需求背景（required）

## 需求来源

依据 11 月社区任务随包任务书 `aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md` 以及 `test_cases/{sgemm,cgemm,hgemm}/` 下的测试指导、CSV 用例集、GPU 基线与验收脚本，设计文档提交至 `cann/cann-ops-competitions` 竞赛仓评审，算子源码、测试与文档在完成开发及自验后贡献至 [ops-blas](https://gitcode.com/cann/ops-blas)。本任务已进入算子开发阶段，本文为拟实施与对齐基线后的设计方案，尚未包含最终全量真机验收结论。

## 背景介绍

### 算子功能与价值

通用矩阵乘（GEMM）是 BLAS Level-3 的核心基础算子，计算列主序矩阵乘加 `C = alpha * op(A) * op(B) + beta * C`，广泛用于科学计算、信号处理及大模型/深度学习推理与训练底层加速。本任务在昇腾 Atlas A2/A3 系列（`arch22`，性能验证设备为 Atlas 800T A2 / 910B3）上基于 CANN 9.1.0 与 Ascend C 句柄式直调工程，完成以下 3 个单矩阵 GEMM 算子的功能补齐、语义对齐与性能优化：

- `aclblasSgemm`（FLOAT32，对标 cuBLAS `cublasSgemm`、Netlib `sgemm`）
- `aclblasCgemm`（COMPLEX64 / `aclblasComplex`，对标 cuBLAS `cublasCgemm`、Netlib `cgemm`，直接复数乘加，非 3M 分解）
- `aclblasHgemm`（FLOAT16 / `aclblasHalf`，对标 cuBLAS `cublasHgemm`）

### ops-blas 工程现状与基线审计

核对 `ops-blas` 上游基线提交 `55ef791b77f3677b45da3468b0a8b17941424f84`（`master`），本任务并非 TBE 迁移，而是基于仓内既有 BLAS 句柄直调架构开展增量补齐与加固。基线现状与任务书要求的核对如下：

| 检查项 | 仓内基线现状（`55ef791b`） | 本任务拟处理方案 |
| --- | --- | --- |
| 公开接口声明 `include/cann_ops_blas.h` | `aclblasSgemm`（L575）与 `aclblasCgemm`（L579）已声明；无非 batch `aclblasHgemm` 声明 | 复用 S/C 既有签名；按同族风格在 `include/cann_ops_blas.h` 新增 `aclblasHgemm` 声明，复用 `cann_ops_blas_common.h` 的 `aclblasHalf` 与 `aclblasComplex` |
| `blas/gemm/arch22/` 实现现状 | 任务书撰写时仅含 `aclblasSgemm`；最新基线已存在 `gemm_host.cpp` / `gemm_kernel.cpp`（Sgemm）与 `cgemm_host.cpp` / `cgemm_kernel.cpp`（Cgemm），仍缺失 `aclblasHgemm` | 在 `blas/gemm/arch22/` 补齐 `aclblasHgemm`，并对既有 `aclblasSgemm`、`aclblasCgemm` 按任务书逐项修复语义偏差与性能瓶颈 |
| 现有 `arch22` Sgemm 已知缺口 | `ValidateSgemm` 将 `m==0` 或 `n==0` 置于 `lda/ldb/ldc` 校验之前；`k==0` 或 `alpha==0` 且 `beta!=0` 时按列循环调用 `n` 次 `aclblasSscal` 且未短路 `beta==1`；`cgemm_host.cpp` 注释记录 `ops-blas#1077`（`K>1024` 非 whole-grid 及部分转置切片路径精度隐患） | 前移 `lda/ldb/ldc` 校验；补齐单次 AIV 缩放与 `beta==1` no-op；排查并修复长 K / 转置 / `beta!=0` 路径缺陷并验证全量精度与性能 |
| 现有 `arch22` Cgemm 已知缺口 | `cgemm_host.cpp` 使用 `CBlas3ReadScalarsFromDevice` 以 D2H + `aclrtSynchronizeStream` 读取 `alpha`/`beta`（违背 Host 指针与异步执行语义）；`ValidateCgemmPointers` 未豁免 `alpha==(0,0)` 时的空 `A`/`B`；按 `CGEMM_K_CHUNK=1024` 多次调用 `aclblasSgemmEx` 与多次 AIV 累加带来显著启动开销 | 改为直接读取 Host 标量、去除额外全流同步；补齐 `alpha==(0,0)` 时允许 `A`/`B` 为空的校验语义；消除冗余 K 切片与回退路径，优化 4M 管线性能 |
| 测试工程 `test/gemm/arch22/` | 仅有面向 `sgemm` 的 `gemm_test.cpp` 与 `gemm_test.csv`，计时输出为 `PERFORMANCE_SUMMARY_JSON` 而非 `[PERF] case=... avg_us=...` | 按任务书要求在 `test/gemm/arch22/` 原地扩展（不新建 `test/gemm/sgemm/arch22/`），支持 S/C/H 三类 CSV Schema、精度比对、padding 哨兵及 `[PERF] avg_us` 输出 |
| 算子文档 `blas/gemm/README.md` | Sgemm 仍标示 A2/A3 不支持；Cgemm 仍标示 A3 不支持且精度表沿用旧 `experimental_standard.md`；无 Hgemm 章节 | 同步更新 S/C 的 Atlas A2/A3 支持说明与精度口径，补充 `aclblasHgemm` 完整接口章节 |

# 需求分析（required）

## 需求描述

在 Atlas A2/A3 系列硬件与 CANN 9.1.0 环境下，采用 Ascend C kernel 直调模式实现并加固 `aclblasSgemm`、`aclblasCgemm`、`aclblasHgemm`，通过 `aclblasHandle_t` 绑定 `aclrtStream` 异步发射，完全对齐任务书功能、参数校验、边界语义、精度标准（`mixed_tolerance_standard.md`）和性能要求（Atlas 800T A2 / 910B3 上 `msprof op` kernel 耗时达到 A100 cuBLAS 标杆的 0.8 倍以上）。接口原型如下：

```cpp
aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc);

aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);

aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda,
    const aclblasHalf* B, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc);
```

## 材料差异核对与口径澄清

在进入编码与联调前，核对任务书与 `test_cases/{sgemm,cgemm,hgemm}/` 随包文件，确认以下口径差异及处理原则（随包原始 CSV 与基线保持原件不变）：

1. **性能基线回填状态**：三个子目录 README 中部分段落仍提示 `gpu_ms` 待回填，但实际核查 `gpu_baseline.csv` 全部 **200/200 条**（Sgemm 67 / Cgemm 67 / Hgemm 66，各含 8 条约 12% 非默认 `alpha`/`beta`）均已回填有效 `gpu_ms` 数值。验收直接使用随包已回填基线，不做删改。
2. **用例总数与子目录统计表微差**：实际随包精度 CSV 共 **1000 条**（Sgemm 334 / Cgemm 333 / Hgemm 333，各含 22 条负向用例）、性能 CSV 共 **200 条**，合计 **1200 条**，与任务书 §3.5 口径一致；子目录 README 中个别前缀细分计数与 CSV 原件略有出入，执行与统计均以随包 CSV 原件为准。
3. **性能双口径区分**：任务书 §3.3 与 §7 明确正式判据为 **`msprof op` 采集的 kernel Task Duration（us）与 `gpu_ms`（kernel-only）做 kernel-to-kernel 对比**；`verify_performance.py` 解析的 GTest `aclrtEvent` `avg_us` 含下发开销，仅作交叉验证与回归辅助，不替代正式判据。两套结果在自测报告中分别留存、独立展示。
4. **精度标准口径统一**：正式精度验收严格采用任务书 §3.2 指定的 `mixed_tolerance_standard.md` 阈值，不以 CSV 中仅供测试框架参考的 `mere_threshold`/`mare_multiplier` 或旧 README 中的 `experimental_standard.md` 覆盖。
5. **内存交付项口径**：任务书 §3.4 标明内存要求“不涉及”，但 §4 `task_submission` 模板包含 `4.1 内存自验证报告.xlsx` 与 `4.2 内存自验证日志.log`；交付时保留该两项文件并如实注明“不涉及”，补充 workspace 预算与越界校验说明，不伪造内存阈值。

## 需求拆解

| 维度 | 规格与约束 |
| --- | --- |
| 数据类型 | S：`FLOAT32`；C：`COMPLEX64`（`aclblasComplex`，实虚各 FP32）；H：`FLOAT16`（`aclblasHalf`，IEEE FP16 位模式）；各算子 `alpha`/`beta` 与 `A`/`B`/`C` 同类型 |
| 内存域与布局 | `alpha`/`beta` 位于 Host 内存；`A`/`B`/`C` 位于 Device 内存，按列主序带 `lda`/`ldb`/`ldc` 填充存储；不支持任意非连续视图与广播；`C` 原地覆写且不得与 `A`/`B` 重叠 |
| 运算属性 | `transa`/`transb ∈ {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}`；S/H 中 `OP_C` 等价 `OP_T`；C 中 `OP_C` 为共轭转置、`OP_T` 为非共轭转置，结果不同 |
| 维度与前导维 | `m, n, k ≥ 0`；`lda ≥ max(1, transa==N ? m : k)`，`ldb ≥ max(1, transb==N ? k : n)`，`ldc ≥ max(1, m)`；运行时动态传入，不按用例编号分派 |
| 退化与空输入 | `m==0` 或 `n==0` 合法 quick return（结构与标量校验通过后不访存、不启核，允许 `A`/`B`/`C` 为空）；`k==0` 或 `alpha==0` 仅执行 `C = beta * C`（不读 `A`/`B`，允许 `A`/`B` 为空，但 `m>0 && n>0` 时 `C` 必非空）；`beta==0` 禁止读取 `C` 旧值 |
| 错误状态码 | `handle==nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM`；负维度、前导维不足、非法空指针或地址溢出 → `ACLBLAS_STATUS_INVALID_VALUE` |
| 异步执行 | 经 `aclblasSetStream` 绑定 stream 异步下发，不引入额外全流同步，由调用方在读回前同步 stream |
| 禁止事项 | 不定义私有平行 API；不把 `aclblasCgemm` 混入 `cgemm3m` 3M 路径；不迁移仅限 arch35/950 的不兼容接口；不为单算子擅改全仓公共编译策略 |

# 详细设计（required）

## 算子分析

### 数学公式

对逻辑索引 `0 <= i < m`、`0 <= j < n`：

```text
C[i, j] = alpha * sum_{p=0}^{k-1} op(A)[i, p] * op(B)[p, j] + beta * C_old[i, j]
```

- **列主序物理寻址（0-based）**：
  - `op(A)[i, p]`：`transa == N` 时取 `A[i + p * lda]`；`transa == T` 时取 `A[p + i * lda]`；`transa == C` 时在 Cgemm 中取 `conj(A[p + i * lda])`，在 S/H 中同 `T`。
  - `op(B)[p, j]`：`transb == N` 时取 `B[p + j * ldb]`；`transb == T` 时取 `B[j + p * ldb]`；`transb == C` 时在 Cgemm 中取 `conj(B[j + p * ldb])`，在 S/H 中同 `T`。
  - `C[i, j]`：读写 `C[i + j * ldc]`；每列 `m <= i < ldc` 的 padding 区域严禁修改。
- **复数直接乘加（4M，非 3M）**：令 `A = Ar + i*Ai`、`B = Br + i*Bi`，定义共轭符号 `s_a = (transa == OP_C) ? -1 : +1`、`s_b = (transb == OP_C) ? -1 : +1`，则：
  - `Re(op(A) * op(B)) = op_r(Ar) * op_r(Br) - (s_a * s_b) * op_r(Ai) * op_r(Bi)`
  - `Im(op(A) * op(B)) = s_b * op_r(Ar) * op_r(Bi) + s_a * op_r(Ai) * op_r(Br)`
  - 其中 `op_r` 仅执行 `N` 或 `T`，共轭语义由符号 `(s_a, s_b)` 完整承载；随后按复数乘法 `(alpha_r + i*alpha_i)(R + i*I) + (beta_r + i*beta_i)(C_r + i*C_i)` 更新 `C`。
- **列主序到 Cube 行主序的等价变换**：列主序 `C(m x n)` 内存布局等价于行主序 `C^T(n x m)`，因此底层 Cube 计算 `C^T = op(B)^T * op(A)^T`，内部形状为 `M_int = n`、`N_int = m`、`K_int = k`。

### 支持数据类型

- **Sgemm（FLOAT32）**：输入输出与标量均为 FP32，全链路保持 FP32 精度；在 `arch22` Cube 路径中显式执行 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)`，避免 HF32 尾数截断在大 `K`（最高 4096）下相对 FP32 golden 漂移。
- **Cgemm（COMPLEX64）**：`aclblasComplex`（实部与虚部分量均为 FP32，交错存储，8B/元素）；四个实数乘积同样关闭 HF32，避免大 `K` 或相消场景精度超差。
- **Hgemm（FLOAT16）**：`aclblasHalf`（按 IEEE-754 half 位模式解释，2B/元素，范围 `[-65504, 65504]`，最小次正规数 `2^-24`）。Cube 内部采用原生 `half x half -> float`（L0C FP32 累加），epilogue 在 FP32 域完成 `alpha * prod + beta * C_old` 后按 IEEE round-to-nearest 收窄为 FP16，既保留高吞吐又避免 `K` 归约过程过早溢出，同时保持超 `65504` 溢为 `±Inf`、次正规数及 `Inf/NaN` 的 IEEE 传播语义。

### 支持形状

单矩阵 2D 列主序输入输出，`op(A)` 为 `m x k`、`op(B)` 为 `k x n`、`C` 为 `m x n`，物理维度由 `lda`/`ldb`/`ldc` 描述，不支持广播及任意非连续视图。运行时支持任意合法非负 `(m, n, k)`；验收测试池覆盖小质数与非对齐维度（`1, 2, 3, 5, 7, 31, 33, 63, 65, 127, 129, 255, 511, 1023, 2047, 4095`）、2 的幂（最高 `4096^3`）、极端瘦长矩阵以及 `lda/ldb/ldc` 紧凑与 `+1/+8/+16/+32/+64` padding 组合。

## 算子实现

### 实现方案

#### 1. Host 侧设计

##### 参数校验顺序与状态码

三个算子统一遵循任务书 §2.5 的严格校验顺序（不扫描 Device 张量内容）：

1. `handle == nullptr` → 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `transa` 或 `transb` 不在 `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` → 返回 `ACLBLAS_STATUS_INVALID_ENUM`；
3. `m < 0 || n < 0 || k < 0` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. 前导维下界校验：`lda < max(1, transa == N ? m : k)`、`ldb < max(1, transb == N ? k : n)`、`ldc < max(1, m)` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`（**早于 `m==0 || n==0` 短路**）；
5. 标量指针校验：`alpha == nullptr || beta == nullptr` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`（**早于 `m==0 || n==0` 短路**，满足 `null_alpha_m0` / `null_beta_n0`）；
6. **空矩阵 Quick Return**：`m == 0 || n == 0` → 直接返回 `ACLBLAS_STATUS_SUCCESS`，不校验且不访问 `A`/`B`/`C`，不发射 kernel；
7. 直接读取 Host 侧 `*alpha`、`*beta`（Cgemm 同步移除旧版 `CBlas3ReadScalarsFromDevice` 的 D2H 拷贝与流同步；Hgemm 在 Host 将 `aclblasHalf` 按 IEEE-754 位模式解包为 `float` 供零值判断与 kernel 传参）；
8. Device 指针校验：`C == nullptr` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；仅当 `k > 0 && !IsZero(alpha)` 时校验 `A == nullptr || B == nullptr` → 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
9. **64 位地址与字节溢出检查**：所有物理跨度 `(cols - 1) * ld + rows`、对齐后维度与 workspace 需求均先提升至 `uint64_t` 执行溢出检查，超界提前返回 `ACLBLAS_STATUS_INVALID_VALUE` 或 `ACLBLAS_STATUS_ALLOC_FAILED`。

##### 退化与零缩放快路径

当 `k == 0 || IsZero(alpha)` 时，严格不读取 `A`/`B`（隔离 `A`/`B` 为空或含 NaN）：

- `IsOne(beta)`（S/H 为 `1.0f`，C 为 `(1.0f, 0.0f)`）：直接返回 `ACLBLAS_STATUS_SUCCESS`，零 kernel 启动；
- `IsZero(beta)`（S/H 为 `0.0f`，C 为 `(0.0f, 0.0f)`）：发射单次 AIV 清零 kernel，仅向 `C` 的 `m×n` 逻辑区写零，不读 `C` 旧值（隔离 `C` 含 NaN），不改 `ldc` padding；
- 其他 `beta`：发射单次 AIV 二维列主序缩放 kernel（替换旧 Sgemm 循环调用 `n` 次 `aclblasSscal` 的做法），按列并行计算 `C = beta * C` 并严格保护 `ldc` padding。

##### 分核、Tiling 规划与 Workspace 管理

- **动态硬件查询**：通过仓内 `GetAicCoreCount()`、`GetAivCoreCount()` 及平台内存接口获取运行时 AIC/AIV 核数与片上 UB 容量，保证同一份 `arch22` 二进制同时适配 Atlas A2（如 910B3）与 Atlas A3，不硬编码核数。
- **分核策略**：优先满核并行，避免小问题过分核导致启动与搬运开销反超：
  - Cube 计算阶段：按内部输出矩阵 `(M_int=n, N_int=m)` 的 2D tile 网格或行波次分配到 AIC 核，当某轴过窄（如瘦长矩阵 `n ≤ 128` 且 `m, k` 较大）时切换到 2D 网格切分以提升 AIC 占用率；
  - AIV 阶段（Pack / Split / Accum / Epilogue / Zero / Scale）：按列块或逻辑元素块均分至不超过 `GetAivCoreCount()` 个核，尾块按实际有效长度收敛。
- **TilingKey / 路径分派**：仅依赖运行时几何形状 `(m, n, k)`、前导维紧凑性、`transa/transb` 与标量类别（`alpha=1, beta=0` 零拷贝直写 vs 通用缩放），严禁依赖用例名称或数据内容分派。
- **Workspace 规划**：统一通过 handle 的 `EnsureDefaultWorkspace` / `GetEffectiveWorkspace` 管理，满足 32B/64B 对齐，不在异步 stream 完成前释放；Cgemm 复用底层实数 GEMM 时通过 user-owned 子句柄严格隔离 scratch 切片，避免覆盖仍在使用的实虚平面。

#### 2. Kernel 侧设计

##### `aclblasSgemm`（FP32，加固与性能优化）

复用 `blas/gemm/arch22/` 既有 `MatmulImpl`（`GM ND float` → `L0C float`）架构，按问题规模与对齐性分层分派：

- **直通 Cube 快路径**：当 `m, n, k` 满足 Cube 对齐要求且 `lda`/`ldb` 紧凑时，利用 4 种转置特化（`NN/NT/TN/TT`）直接读取 `A`/`B`；若同时满足 `alpha == 1.0f && beta == 0.0f && ldc == m`，Cube 直接写回 `C`，实现单 kernel 闭环；
- **大矩阵 Whole-Grid 路径**：覆盖 `2048～4096` 等大尺寸（含 `4096³`），按 `256×256` 网格并行发射全部 AIC 核；对非对齐维度、带 `ld` padding 或 `trans` 场景通过 AIV Pack 将操作数规整到对齐连续布局（新增填充坐标严格补零），Cube 完成后由 AIV Epilogue 执行 `alpha * prod + beta * C_old`（`beta==0` 时跳过读 `C`）；
- **通用分块路径与缺陷修复**：针对非对齐（如 `4095³`、质数维）、`K > 1024` 场景，重点排查并修复现有基线中的长 K 归约（`ops-blas#1077`）及 `transB=T` 行切片基址对齐问题，统一使用 64 位偏移计算；K 方向分块累加时保持 FP32 累加缓冲，禁止引入低精度中间缓存。

##### `aclblasCgemm`（COMPLEX64，4M 管线重构与提速）

保持 `C = alpha * op(A) * op(B) + beta * C` 直接复数乘加语义，不采用 3M 分解：

- **Phase 0（AIV 去交错拆分）**：按列并行将交错 `aclblasComplex` 输入拆分为 64B 对齐的连续实/虚平面 `(Ar, Ai)` 与 `(Br, Bi)`；仅搬运物理有效行数，不读取 `lda`/`ldb` 越界区；UB 缓冲控制在安全容量内并采用精确流水同步（`MTE2 → V → MTE3`）；
- **Phase 1（AIC 四次实数乘积）**：基于共轭符号 `(s_a, s_b)` 计算 `Re = Ar*Br - (s_a*s_b)*Ai*Bi` 与 `Im = s_b*Ar*Bi + s_a*Ai*Br`：
  - 修复 `arch22` 实数 GEMM 长 K 与转置缺陷后，解除现有 `CGEMM_K_CHUNK=1024` 强制切片限制，单次完成全 K 归约（仅当总 workspace 超出设备/句柄安全预算时按 K 分块），避免 4096³ 下乘以 4 倍的重复发射与累加开销；
  - 避免经由 `aclblasSgemmEx` 对 `|alpha| > 1` 或中小规模误回退至慢速 AIV 路径；对 `Re`/`Im` 的第二项乘加优先评估直接复用 Cube 累加或单次融合 AIV 累加，减少中间缓冲往返；
  - 对极小规模矩阵评估单核/少量 AIV 直接复数乘加路径，降低多阶段 kernel 启动总延迟以匹配 A100 低至 `8.5 us` 的小规模基线；
- **Phase 2（AIV 复数 Epilogue）**：按列并行加载 `(Re, Im)`，仅当 `beta != (0,0)` 时读取 `C_old`；在 UB 内完成复数 `alpha` 与 `beta` 乘加并交错写回 `C` 的逻辑 `m×n` 区域；尾块严格以有效字节长度写回 GM，保护 `ldc` padding。

##### `aclblasHgemm`（FLOAT16，原生 FP16 Cube 加速）

`aclblasHgemm` 为本任务在 `arch22` 的全新实现，核心目标是发挥 Atlas A2/A3 的 FP16 Cube 峰值算力并严格满足 IEEE FP16 语义：

- **原生 FP16 Cube 计算核**：采用 `half × half → float`（L0C FP32 累加）的 Ascend C `MatmulImpl`（或经仓内基线验证的 `arch22` CATLASS/Ascend C Cube 模板），覆盖 `NN/NT/TN/TT` 四种物理转置组合（`OP_C` 在 Host 映射为 `OP_T`）；
- **单核/多核分块规划**：
  - 大中型矩阵（如 `512³～4096³`）：采用 2D 输出网格切分（如 `128×256` / `128×128` / `256×128` 块，依据 `M_int, N_int, K_int` 与 AIC 核数均衡调度，避免尾部空核），K 轴采用 `baseK` 双缓冲掩盖 MTE2 搬运延迟；
  - 紧凑对齐且 `alpha == 1.0f && beta == 0.0f && ldc == m` 时，Fixpipe 直接完成 L0C FP32 → GM FP16（IEEE round-to-nearest）输出，零额外 AIV kernel 启动与 workspace 搬运；
  - 含 `lda/ldb/ldc` padding、非对齐维度（如 `4095³`、质数维）或非默认 `alpha`/`beta` 时：Cube 输出 FP32 中间乘积至 workspace（或经轻量 Pack 对齐输入，填充位严格写零），随后由单次 AIV Epilogue 在 FP32 域完成 `alpha_f32 * prod_f32 + beta_f32 * cast_f32(C_old)`（`beta == 0` 不读 `C_old`）并经 `Cast<half, float, RoundMode::CAST_RINT / CAST_NONE>`（按 CANN 9.1.0 实际 IEEE 最近偶数舍入重载核对）写回 `C` 逻辑区；
  - 极小规模（如 `1×1×1`、`4³`、`8³`、`TC_ED_042～049` 特殊值）：采用轻量 AIV FP32 累加路径，避免 Cube 对齐打包开销，同时天然保证 `±65504`、次正规数 `5e-8`、`Inf`、`NaN` 严格按 IEEE-754 规则收窄传播。
- **同步与尾块 DMA 安全**：所有 AIV/AIC 流水严格按真实数据生产-消费及缓冲区复用插入细粒度流水事件同步，不使用 `PIPE_ALL`；尾块搬运逐方向核对 GM 字节步长与 UB 32B 块步长，写 GM 不扩大有效写入范围。

#### 3. 工程组织与构建测试集成

严格遵循任务书 §2.2 与 `ops-blas` 仓目录规范，最小必要文件集合如下：

```text
ops-blas/
├── include/
│   └── cann_ops_blas.h              # 复用 aclblasSgemm/Cgemm；新增 aclblasHgemm 声明
├── blas/
│   ├── CMakeLists.txt               # 核对 --ops=sgemm,cgemm,hgemm 单算子筛选依赖（不改全仓编译策略）
│   └── gemm/
│       ├── README.md                # 更新 S/C 的 A2/A3 支持与精度表，补充 aclblasHgemm 章节
│       └── arch22/
│           ├── gemm_host.cpp        # 加固 aclblasSgemm 校验/退化/分派路径
│           ├── gemm_kernel.cpp/.h   # 加固长 K/转置/缩放路径
│           ├── cgemm_host.cpp       # 修复 Host 标量与 alpha=0 校验，优化 4M 管线
│           ├── cgemm_kernel.cpp/.h  # 复数拆分/累加/组合 kernel
│           ├── cgemm_tiling_data.h
│           ├── hgemm_host.cpp       # 新增 aclblasHgemm Host 入口与 tiling
│           ├── hgemm_kernel.cpp/.h  # 新增 aclblasHgemm Cube + AIV kernel
│           └── hgemm_tiling_data.h  # 按需复用或独立定义，保持声明头不含全局 using namespace
└── test/
    └── gemm/
        └── arch22/
            ├── gemm_test.cpp        # 扩展支持 S/C/H 三算子、三类 CSV、padding 哨兵、显式选卡与 [PERF] 输出
            └── gemm_test.csv        # 仓内默认回归用例（开发与自验阶段兼容 --csv 替换执行全量 1200 用例）
```

- 编译与测试命令：`bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`（A3 使用对应 `--soc` 取值）；同时核查默认全仓构建不引入编译告警或符号冲突。
- 测试工程遵循显式设备选择（响应 `ASCEND_DEVICE_ID` / `ASCEND_RT_VISIBLE_DEVICES`），在分配 Device 内存前完成设备设置；一次性自测报告、日志与本地辅助脚本仅放于交付目录 `task_submission/`，不混入 `ops-blas` 源码 PR。

## 支持硬件

| 产品系列 | 支持设计 | 说明 |
| --- | --- | --- |
| Atlas A2 系列（含 Atlas 800T A2 / 910B3） | √ | `arch22` 实现；计划作为精度与性能主自验平台 |
| Atlas A3 系列（含 Atlas 800T A3） | √ | 共用 `arch22` 兼容实现，动态查询核数与片上容量 |
| Ascend 950PR / 950DT | 维持既有 | 不改动 `arch35` 既有实现，保持公共头文件兼容 |
| 其他产品系列 | × | 不在本任务适配范围 |

## 算子约束限制

- 仅支持列主序连续带 `lda`/`ldb`/`ldc` 填充布局，不支持任意非连续视图与 shape 广播；
- `alpha`/`beta` 为 Host 指针，`A`/`B`/`C` 为 Device 指针；`C` 原地更新且不得与 `A`/`B` 内存重叠；
- 不保证多核/长 K 归约的跨次执行逐位确定性，精度验收以 `mixed_tolerance_standard.md` 阈值为准；
- 不采用 3M 分解实现 `aclblasCgemm`；不依赖 `arch35` 专属 SIMT/HF32 特性；
- 当极端大矩阵所需 workspace 超出句柄/设备可用上限时，按仓内约定返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 算子 | 验收标准 | 标准来源 |
| --- | --- | --- | --- |
| 精度（Sgemm） | `aclblasSgemm` | `rtol = atol = 2^-13 (1.22e-4)`，`matched_ratio ≥ 0.99`，且 `max_abs_error ≤ max(1e-2, 32·ULP)`；golden 为 `cblas_sgemm`（float64 累加后收窄 FP32） | 任务书 §3.2 / `mixed_tolerance_standard.md` |
| 精度（Cgemm） | `aclblasCgemm` | 实部、虚部分别按 FLOAT32 档判定：`rtol = atol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ max(1e-2, 32·ULP)`；golden 为 `cblas_cgemm`（float64 累加后收窄 complex64） | 任务书 §3.2 / `mixed_tolerance_standard.md` |
| 精度（Hgemm） | `aclblasHgemm` | `rtol = atol = 2^-10 (9.77e-4)`，`matched_ratio ≥ 0.99`，且 `max_abs_error ≤ max(1e-1, 32·ULP)`；golden 按 FP32（float64 累加）后 IEEE round-to-nearest 收窄 FP16 | 任务书 §3.2 / `mixed_tolerance_standard.md` |
| 性能（正式判据） | S / C / H | Atlas 800T A2（910B3）上 `msprof op` 采集 kernel Task Duration（us），warmup 5 次后取连续 10 次有效样本算术平均；逐条满足 `NPU_avg_us ≤ gpu_ms * 1000 / 0.8`（即 ≥ 0.8× A100 cuBLAS 标杆），且全部可比用例平均倍率 ≥ 0.8 | 任务书 §3.3 / `gpu_baseline.csv` |
| 性能（辅助交叉验证） | S / C / H | GTest 预置 Device 缓冲、warmup 5 次后 `aclrtEvent` 夹住连续 10 次输出 `[PERF] case=... avg_us=...`，由 `verify_performance.py` 比对，不替代正式判据 | 任务书 §3.3 / 随包脚本 |
| 内存 | S / C / H | 不涉及（无专项内存门槛；单用例测试集 A/B/C 控制在 ≤ 512MB） | 任务书 §3.4 |

## 测试覆盖与自验计划

完整保留并执行随包 1200 条官方用例（不篡改 CSV 或基线）：

- **精度全量 1000 条**（Sgemm 334 / Cgemm 333 / Hgemm 333）：覆盖 `TC_L0`（9 种转置组合）、`TC_MN`（1～4096 方阵含 `4095³`/`4096³` 与极端矩形）、`TC_AB`（特殊与随机标量）、`TC_LD`（单维与全维 padding）、`TC_FL`（50% `RANDOM_NORM_5_5` + 50% `GAUSS_NORM_0_1` 及全零/交替/极端/Inf/NaN）、`TC_CV`、`TC_ED`（`m/n/k=0`、`alpha=0` 叠加 `A/B=NaN/nullptr`、`beta=0` 叠加 `C=NaN`、`handle=nullptr`、非法枚举、负维度、前导维不足、Cgemm `OP_C` 非对称共轭验证、Hgemm `±65504`/次正规数 `5e-8`/Inf/NaN）与 `TC_EX`。所有正向用例均验证 `ldc` padding 哨兵不被改写；负向用例仅校验返回码，不因非法维度申请巨量缓冲。
- **补充回归测试**：针对 `m=0/n=0` 叠加非法 `lda/ldb/ldc`、`K > 1024` 非 whole-grid（`512×512×1536` 等）及 64 位跨度溢出补充 GTest 单元测试，验证修复完全闭合。
- **性能全量 200 条**（Sgemm 67 / Cgemm 67 / Hgemm 66，按 small ≤64KiB / medium ≤2MiB / large >2MiB 分层且各含 8 条非默认 `alpha/beta`）：
  1. 执行 `verify_performance.py` 采集端到端 `aclrtEvent` `avg_us` 做快速回归与交叉验证；
  2. 使用 `msprof op` 逐条采集 `OpBasicInfo.csv` 中的 `Task Duration(us)`，严格剔除前 5 次 warmup、取后 10 次有效样本均值；对多 kernel 路径（如 pack + cube + epilogue 或 Cgemm 多阶段管线）逐次调用累加全部计算 kernel 耗时，严禁仅摘取单个最快 kernel 上报；若存在异常样本单独记录并说明。

## 兼容性分析

- **接口兼容性**：保持 `aclblasSgemm`（`include/cann_ops_blas.h` L575）与 `aclblasCgemm`（L579）既有函数签名不变；新增 `aclblasHgemm` 复用公共 `aclblasHalf` 与错误码体系，与其他产品线共享声明，不新增私有平行接口。
- **架构与既有实现兼容**：不改动 `blas/gemm/arch35/` 与 `test/gemm/arch35/` 既有 Ascend 950 实现；`blas/gemm/arch22/` 动态查询运行时 AIC/AIV 核数与片上容量以兼顾 Atlas A2 与 Atlas A3。
- **行为变更点说明**：现有 `arch22` `aclblasCgemm` 将 `alpha`/`beta` 当作 Device 指针并触发同步的偏差在本任务中修正为 Host 指针异步语义（与任务书 §2.4 及仓内 README 参数说明一致）；`aclblasSgemm` 补齐 `m==0` 或 `n==0` 时的 `lda`/`ldb`/`ldc` 前置校验。

## 风险控制与交付清单

- **主要风险与对策**：
  1. *Sgemm 长 K / 转置缺陷*：优先通过 `512×512×1536` 等回归定位 `ops-blas#1077` 根因并修复，再放开 Cgemm 复用；
  2. *Cgemm 4M 多阶段开销*：通过取消不必要 K 切片、减少中间 workspace 写回、为极小规模提供低启动路径控制总耗时；
  3. *Hgemm 4096³ 高标杆（A100 1.68 ms）与小尺寸低延迟（约 10 us）*：大模型依赖原生 FP16 Cube 2D 网格与直接 Fixpipe 写回，小模型通过轻量路径与核数裁剪控制启动开销；
  4. *FP16 极值与大倍率误差*：全链路 FP32 累加 + IEEE 舍入并针对 `±65504`、`5e-8`、`Inf/NaN` 专项验证；所有性能数字以真机实测结果为准，不在设计阶段预先承诺。
- **验收交付清单**：验收阶段在私仓根目录创建 `task_submission/`（含 `1 自验证步骤说明.md`、`2.1/2.2 精度自验证报告与日志`、`3.1/3.2 性能自验证报告与日志`、`4.1/4.2 内存自验证报告与日志`），记录源码 commit、算子库 SHA256、设备/CANN 版本与完整命令，同时按流程邀请 `Ascend-CANN` 为开发者。

## 参考资料

1. 任务书：`aclblasSgemm_Cgemm_Hgemm_AtlasA2_A3_task_doc.md`
2. ops-blas 开源仓：https://gitcode.com/cann/ops-blas
3. 生态算子开源精度标准（`mixed_tolerance_standard.md`）：https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md
4. cuBLAS 参考文档：https://docs.nvidia.com/cuda/cublas/index.html
5. Netlib BLAS 参考实现（`sgemm.f` / `cgemm.f`）：https://www.netlib.org/blas/
