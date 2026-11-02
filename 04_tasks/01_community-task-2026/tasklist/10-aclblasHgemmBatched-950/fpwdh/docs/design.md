# 需求背景

## 需求来源

CANN 社区任务 2026 年 10 月任务：在 Ascend 950PR 上使用 Ascend C 开发半精度批量矩阵乘算子
`aclblasHgemmBatched`，补齐 `cann/ops-blas` 仓 Level-3 GEMM 族的 FLOAT16 批量能力，验收通过后合入主仓。
对标接口为 cuBLAS `cublasHgemmBatched`，参数序列一一对应。

## 背景介绍

### 算子功能

对每个 batch i，原地计算：

```
C_i = alpha * op(A_i) * op(B_i) + beta * C_i ，i ∈ [0, batchCount)
```

- `op(X)`：`transa/transb = ACLBLAS_OP_N` 时不转置，`ACLBLAS_OP_T` 转置，`ACLBLAS_OP_C` 共轭转置；
  FLOAT16 为实数，T 与 C 等价。
- A_i 为 m×k（N）或 k×m（T/C），B_i 为 k×n（N）或 n×k（T/C），C_i 恒为 m×n，全部按**列主序**存储。
- `Aarray`/`Barray`/`Carray` 是设备侧指针数组，第 i 个元素指向第 i 个 batch 矩阵首地址，
  前导维分别为 `lda`/`ldb`/`ldc`。
- `alpha`、`beta` 为 FLOAT16 Host 标量指针，整批共用。

### ops-blas 仓现状分析

主仓（master `05b96ea`，2026-10-28）已有同族批量 GEMM 实现，且**内核已按 dtype 模板化**：

| 组件 | 位置 | 现状 |
| --- | --- | --- |
| FP32 批量接口 | `aclblasSgemmBatched`（`blas/gemm_batched/arch35/gemm_batched_host.cpp`） | 已合入 |
| 复数批量接口 | `aclblasCgemmBatched`（同文件） | 已合入，4M 分解复用实数 GEMM |
| cube 批量内核 | `blas/gemm_batched/arch35/gemm_batched_cube_impl.h` | `GbDtypeTraits<0/1/2>` = FP32/FP16/BF16，`C0`/`L0 base K` 按 dtype 静态选择 |
| kernel 入口表 | `blas/gemm_batched/arch35/gemm_batched_kernel.cpp` | 已实例化 `gemm_batched_gemm_kernel_fp16_{nn,nt,tn,tt}`、`gemm_batched_alpha_beta_kernel_fp16`、`gemm_batched_early_exit_kernel_fp16` |
| tiling 结构 | `blas/gemm_batched/arch35/gemm_batched_tiling_data.h` | `GemmBatchedGemmTilingData::dtypeCase` 已预留 `GEMM_BATCHED_DTYPE_FP16 = 1` |
| FP16 精度后处理 | `gemm_batched_alpha_beta_kernel_fp16` | 以 FLOAT32 临时结果计算 `alpha*T + beta*C`，写回时一次性 `Cast` 到 FLOAT16 |

即：三段式执行链（cube GEMM → alpha/beta 后处理 → early-exit）的 FLOAT16 实例已经存在并可编译，
本算子的工作集中在 host 侧与接口层。

### 缺口分析

1. `include/cann_ops_blas.h` 无 `aclblasHgemmBatched` 声明；
2. `include/cann_ops_blas_common.h` 无 FLOAT16 标量类型 `aclblasHalf`（仅有 `aclblasComplex`/`aclblasDoubleComplex`）；
3. host 侧 `CalcGemmTiling` 把 `dtypeCase` 固定为 FP32，缺少 FLOAT16 的 tiling/workspace/kernel 下发入口；
4. `blas/gemm_batched/arch35/gemm_batched_kernel.cpp` 在单算子构建（`--ops`）时会被源文件过滤器排除，
   新增依赖方需登记到 `blas/CMakeLists.txt` 的依赖表；
5. 缺少 FLOAT16 的测试工程、CSV 用例与 README。

# 需求分析

## 需求描述

新增句柄式 BLAS 接口：

```cpp
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const* Aarray, int lda,
    const aclblasHalf* const* Barray, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const* Carray, int ldc,
    int batchCount);
```

通过 handle 绑定的 stream 直调 NPU kernel；声明放入公共头 `include/cann_ops_blas.h`，与其他产品线共用，
不定义 950PR 私有平行接口。

## 需求拆解

| 编号 | 需求 | 设计落点 |
| --- | --- | --- |
| R01 | 新增 `aclblasHalf` 类型与 `aclblasHgemmBatched` 公共声明 | `include/cann_ops_blas_common.h`、`include/cann_ops_blas.h` |
| R02 | 参数校验顺序与错误码严格按任务书 §2.5 | `ValidateHgemmBatchedParams` |
| R03 | `m=0`/`n=0`/`batchCount=0` 合法 no-op，不访问 A/B/C | 入口短路（校验之后） |
| R04 | `k=0` 逐 batch `C_i = beta*C_i`（beta=0 置零），不引用 A/B | `gemm_batched_early_exit_kernel_fp16` |
| R05 | `alpha=0` 退化为 `C_i = beta*C_i`，与 Netlib cblas 的 alpha==0 短路语义一致 | 同上 |
| R06 | transa/transb 全 9 组合、任意 m/n/k、lda/ldb/ldc padding | 复用 `NDExtLayoutPtn`/`DNExtLayoutPtn` 四套 FP16 cube kernel |
| R07 | 精度达 FLOAT16 档：rtol=atol=2⁻¹⁰、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-1 或 32·ULP | FLOAT32 累加 + 写回一次舍入 |
| R08 | 性能：单 case ≥ 0.4×、全部性能 case 平均 ≥ 0.5× H100 | 同族 tiling + L1 双 bank + batch×M/N 任务切分 |
| R09 | 1000 条精度 + 200 条性能 CSV 驱动 GTest，可复现 | `test/gemm_batched/hgemm_batched/` |
| R10 | README 产品支持表标注 Ascend 950PR：支持 | `blas/gemm_batched/README.md` |

# 详细设计

## 算子分析

### 数学公式与索引

对 i ∈ [0, batchCount)、r ∈ [0, m)、c ∈ [0, n)：

```
C_i[r][c] = alpha × Σ_{p=0}^{k-1} op(A_i)[r][p] × op(B_i)[p][c] + beta × C_i[r][c]
```

列主序存储下的元素地址：

- `transa = N`：`A_i` 为 m×k，`A_i[r][p]` 位于 `r + p*lda`
- `transa ∈ {T,C}`：`A_i` 为 k×m，`op(A_i)[r][p] = A_i[p][r]` 位于 `p + r*lda`
- B 同理；`C_i[r][c]` 位于 `r + c*ldc`

### 支持数据类型

| 角色 | 类型 | 说明 |
| --- | --- | --- |
| A_i、B_i、C_i | FLOAT16 | IEEE 754 binary16，2 字节 |
| alpha、beta | FLOAT16（Host 指针） | 整批共用 |
| 内部累加 | FLOAT32 | Cube L0C 累加；与 golden 口径一致 |

本算子不设 dtype 轴，全部固定 FLOAT16。`aclblasHalf` 采用 16 位整型存储（`typedef uint16_t`），
位模式与 `aclFloat16`、Ascend C `half` 完全一致，公共头不依赖编译器 `_Float16` 扩展。

### 参数约束与异常行为

| 参数 | 约束 | 违反时返回 |
| --- | --- | --- |
| handle | 非空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| alpha / beta | 非空（no-op 场景同样校验） | `ACLBLAS_STATUS_INVALID_VALUE` |
| m / n / k | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| batchCount | ≥ 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| transa / transb | ∈ {N, T, C} | `ACLBLAS_STATUS_INVALID_ENUM` |
| lda | ≥ max(1, A_i 物理行数) | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldb | ≥ max(1, B_i 物理行数) | `ACLBLAS_STATUS_INVALID_VALUE` |
| ldc | ≥ max(1, m) | `ACLBLAS_STATUS_INVALID_VALUE` |
| Aarray / Barray / Carray | `batchCount > 0` 时非空（k=0、m=0、n=0 也不例外） | `ACLBLAS_STATUS_INVALID_VALUE` |

A_i 物理行数：N → m，T/C → k；B_i 物理行数：N → k，T/C → n。

### 参数校验顺序（冻结）

| 顺序 | 检查 | 返回码 |
| --- | --- | --- |
| 1 | `handle != nullptr` | `HANDLE_IS_NULLPTR` |
| 2 | `alpha != nullptr`、`beta != nullptr` | `INVALID_VALUE` |
| 3 | `m >= 0`、`n >= 0`、`k >= 0`、`batchCount >= 0` | `INVALID_VALUE` |
| 4 | `transa/transb ∈ {N,T,C}` | `INVALID_ENUM` |
| 5 | `lda/ldb/ldc` 前导维 | `INVALID_VALUE` |
| 6 | `batchCount > 0` 时数组指针非空 | `INVALID_VALUE` |
| 7 | no-op 短路：`m==0`、`n==0` 或 `batchCount==0` | `SUCCESS` |

**多种非法参数同时存在时，以上表顺序为准**（返回表中第一个命中的错误码）。

### 转置组合表（9 种，全部覆盖）

| transa | transb | op(A_i) | op(B_i) | cube kernel 布局（A,B） | 覆盖用例 |
| --- | --- | --- | --- | --- | --- |
| N | N | m×k | k×n | `NDExt`, `NDExt` → `_nn` | TC_L0 × 2 尺寸 |
| N | T | m×k | n×k | `NDExt`, `DNExt` → `_nt` | TC_L0 × 2 尺寸 |
| N | C | m×k | n×k（实数等价 T） | `_nt` | TC_L0 × 2 尺寸 |
| T | N | k×m | k×n | `DNExt`, `NDExt` → `_tn` | TC_L0 × 2 尺寸 |
| T | T | k×m | n×k | `DNExt`, `DNExt` → `_tt` | TC_L0 × 2 尺寸 |
| T | C | k×m | n×k | `_tt` | TC_L0 × 2 尺寸 |
| C | N | k×m | k×n | `_tn` | TC_L0 × 2 尺寸 |
| C | T | k×m | n×k | `_tt` | TC_L0 × 2 尺寸 |
| C | C | k×m | n×k | `_tt` | TC_L0 × 2 尺寸 |

### 路径状态表

| 路径 | 状态 | 判定条件 |
| --- | --- | --- |
| no-op 早退 | **已有代码**（本次新增 host 分支） | `m==0 || n==0 || batchCount==0` |
| early-exit（beta 缩放/置零） | **已有代码**（FP16 实例已编译，本次新增调用） | `k==0 || alpha==0`；`beta==1` 时直接返回 |
| cube GEMM 主路径 | **已有代码**（FP16 实例已编译，本次新增 tiling/下发） | `k>0 && alpha!=0` |
| alpha/beta 后处理（FP32 临时 + 一次舍入） | **已有代码**（FP16 实例已编译，本次新增调用） | 同主路径（FLOAT16 输出必须） |
| 直接 L0C→FLOAT16 GM 输出（省一次 kernel） | **拟实施，待 NPU 验证** | 非默认路径；验证通过前不进入承诺范围 |

> 说明：本设计**不新增矩阵乘算法**，只新增 FLOAT16 的 host 入口；上表「已有代码」指内核实现已在
> master 的 `blas/gemm_batched/arch35/` 中存在，本次改动为调用与参数化。「拟实施」路径不代表当前代码已有该分支。

## 算子实现

### 落位说明（口径）

任务书 §2.2/§5 写「实现代码放在 `blas/gemm/arch35/`」。仓内同族实现（`aclblasSgemmBatched`、
`aclblasCgemmBatched`）与官方测试指导（`test_cases/README.md`）均指向 `blas/gemm_batched/arch35/`、
`test/gemm_batched/`。本设计取**任务书目录**作为 host 入口落位（`blas/gemm/arch35/hgemm_batched_host.cpp`），
通过相对路径复用 `gemm_batched` 的 tiling 结构与内核 launcher；测试工程按官方测试指导落位
`test/gemm_batched/hgemm_batched/`（否则 `build.sh --ops=hgemm_batched` 与随任务提供的
`verify_accuracy.py`/`verify_performance.py` 的路径解析都不成立）。两处差异记录于《口径声明》。

### 文件清单

| 文件 | 职责 | 类型 |
| --- | --- | --- |
| `include/cann_ops_blas_common.h` | 新增 `aclblasHalf` | 修改（新增 5 行） |
| `include/cann_ops_blas.h` | 新增 `aclblasHgemmBatched` 声明 | 修改（新增声明） |
| `blas/gemm/arch35/hgemm_batched_host.cpp` | FLOAT16 host：校验、tiling、workspace、三套 kernel 下发 | 新增 |
| `blas/gemm_batched/README.md` | 接口说明、产品支持表、精度口径、调用示例 | 修改 |
| `blas/CMakeLists.txt` | 单算子构建时保留被依赖的 `gemm_batched_kernel.cpp` | 修改（新增 10 行） |
| `test/gemm_batched/hgemm_batched/*` | CSV 驱动 GTest + golden + README | 新增 |

### host 侧设计

#### 1. 总体流程

```
aclblasHgemmBatched
  ├─ ValidateHgemmBatchedParams        // §2.5 顺序
  ├─ no-op 短路（m/n/batchCount == 0） → SUCCESS，不访问 A/B/C
  ├─ alpha/beta：FLOAT16 → FLOAT32（host 标量换算，供 kernel tiling 使用）
  ├─ k==0 || alpha==0 → LaunchEarlyExit（C = beta*C；beta==1 直接返回）
  └─ ExecuteHgemmBatched
       ├─ CalcGemmTiling(dtypeCase = FP16)
       ├─ PrepareTempWorkspace（指针数组 64B 对齐 + 每 batch FLOAT32 临时矩阵）
       ├─ ApplyColumnMajorSwap（C^T = op(B)^T·op(A)^T，A/B 指针互换）
       ├─ gemm_batched_gemm_kernel_do(dtypeCase=FP16 → cube FP16 kernel)
       └─ gemm_batched_alpha_beta_kernel_do(dtypeCase=FP16 → FLOAT32 累加 + 一次 Cast 到 FLOAT16)
```

#### 2. tiling

复用 `GemmBatchedGemmTilingData`（FLOAT16 取值）：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `dtypeCase` | `GEMM_BATCHED_DTYPE_FP16 = 1` | 选择 FP16 内核实例 |
| `tileM` / `tileN` | 128 / 128 | 与 FP32 共用默认块 |
| `tileKChunk` | 128 | K 向 L1 分块 |
| L0 base K | `GEMM_BATCHED_FP16_L0_BASE_K = 64` | 由 `GbDtypeTraits<1>` 在 kernel 内静态选择 |
| C0 | `GEMM_BATCHED_FP16_C0 = 16` | 分形对齐单位 |
| `mBlocks`/`nBlocks` | `CalcMnBlockSplit` | 取最小的 M/N 切分使 `batchCount × spatialTasks` 为 Cube 核数整数倍，消除尾波 |
| `singleCoreM`/`singleCoreN` | 按 16 对齐后截断到 m/n | 分形对齐 |
| `totalTasks` | `batchCount × mBlocks × nBlocks` | > UINT32_MAX 返回 INVALID_VALUE |
| `usedAicCoreNum` | `min(cubeCoreNum, totalTasks)` | 不空转 |

#### 3. workspace

主路径需要两块设备临时空间，复用 handle 的库工作区（`EnsureDefaultWorkspace`，不足时按倍增策略扩容）：

1. **临时 C 指针数组**：`batchCount × sizeof(void*)`，向上对齐到 64B；
2. **FLOAT32 临时结果**：`batchCount × n × tempRowStride × sizeof(float)`，其中
   `tempRowStride = CeilAlign(m, 16)`，保证每个 batch 的临时矩阵起点 64B 对齐、fixpipe 不写非对齐地址。

临时区固定 FLOAT32，不随输入 dtype 变化：alpha/beta 后处理内核以 `__gm__ float*` 解释该区域，
在 FLOAT32 下完成 `alpha*T + beta*C` 后一次性舍入到 FLOAT16 写回。因此整条链路只有「写回 C」一次舍入，
与 golden「FLOAT32 计算 + 写回舍入」的口径一致。

#### 4. 分核

- 调度单位 = 一个 batch 内的一块 `singleCoreM × singleCoreN` 输出；
- `totalTasks = batchCount × mBlocks × nBlocks`，各核按 `blockIdx` 步进领取，天然负载均衡；
- 核数取 `min(AIC 核数, totalTasks)`；小 batch×大矩阵靠 M/N 切分填核，大 batch×小矩阵靠 batch 维填核；
- 后处理 AIV 网格取 `min(AIV 核数, ceil(batch×m×n / 16384))`，小用例不空转。

### kernel 侧设计

复用 `blas/gemm_batched/arch35/` 三段式内核，FP16 同一套模板实例化（逐字调用，不改内核）：

| kernel | 实例 | 作用 | 线程/流水 |
| --- | --- | --- | --- |
| `gemm_batched_gemm_kernel_fp16_{nn,nt,tn,tt}` | `GbDtypeTraits<1>` | 按 `totalTasks` 解出 `(batch, mBlock, nBlock)`，指针数组取 A/B 首地址；16×16 分形搬入 L1/L0，K 向累加；结果写 FLOAT32 临时区 | AIC-only，L1 双 bank ping/pong，L0A/L0B 双缓冲 |
| `gemm_batched_alpha_beta_kernel_fp16` | `GemmBatchedAlphaBetaOp<1>` | 读 FLOAT32 临时结果与旧 C（`beta!=0` 时），FLOAT32 计算 `alpha*T+beta*C`，`Cast` 到 FLOAT16 写回；按列条带多核并行 | AIV，`DataCopyPad` 对齐读写 |
| `gemm_batched_early_exit_kernel_fp16` | `GemmBatchedEarlyExitOp<1>` | `k=0`/`alpha=0`：`C = beta*C`，`beta=0` 置零，不访问 A/B | AIV |

FP16 关键常量：`C0 = 16`、`L0 base K = 64`（FP32 为 8 与 16），单次 K 向装载覆盖更长的累加链。

### 精度设计

- Cube 内 FP16 输入提升到 FLOAT32 累加（L0C 仅支持 FP32 累加），`SetHF32Mode(DISABLE)` 不启用 HF32 快径；
- 临时结果与 `beta*C` 的合成在 FLOAT32 下完成，写回 C 时按 IEEE 754 就近舍入（CAST_ROUND）一次；
- golden：逐 batch FLOAT32 `cblas_sgemm`（输入为量化到 FLOAT16 的精确值），结果同样一次舍入到 FLOAT16；
- 误差来源仅为累加顺序差异，量级约 `O(√k)·ε_fp32`，远小于 2⁻¹⁰ 门限；
- 若 `k` 很大导致 FLOAT16 累加溢出（任务书 §3.2.4 场景），在自测报告中给出最大误差随 k 的增长曲线与
  累加档位说明，不放宽阈值。

### 性能设计与理论上限

按随任务提供的 H100 基线（`gpu_baseline.csv`）反推各典型 case 对 NPU 的最低吞吐要求
（`FLOPs = 2·m·n·k·batch`，门限耗时 = `gpu_ms / 0.4`）：

| case | FLOPs | H100 耗时 | 门限耗时（0.4×） | 门限所需吞吐 |
| --- | --- | --- | --- | --- |
| 256³×16 (NN) | 1.074 GFLOP | 7.249 µs | ≤ 18.1 µs | ≥ 59 TFLOPS |
| 512³×8 (NN) | 2.147 GFLOP | 8.647 µs | ≤ 21.6 µs | ≥ 99 TFLOPS |
| 1024³×4 (TN) | 8.590 GFLOP | 26.125 µs | ≤ 65.3 µs | ≥ 131 TFLOPS |
| 2048³×2 (NT) | 34.36 GFLOP | 55.413 µs | ≤ 138.5 µs | ≥ 248 TFLOPS |

结论与打法：

1. 大 shape（≥1024³）是**吞吐受限**，靠 Cube 满核 + L1 双 bank + 大 tile 逼近峰值；
2. 小 shape（≤512³、esp. 8³×1024）是**启动开销受限**，门限只有十几 µs，必须控制 kernel 数量与下发开销。
   当前实现为 2 个 kernel（GEMM + 后处理），属于保守上界；「直接 L0C→FLOAT16 输出」的单 kernel 路径
   列为拟实施项，需真机验证 fixpipe 量化后用 A/B 对比决定是否启用；
3. 只保「最差 case 达标」，不追平均余量：均值门限 0.5× 由大 shape 拉动，小 shape 保 0.4× 即可。

# 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / Ascend 950DT（arch35 / dav-3510） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 本期不涉及 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 本期不涉及 |

前置条件：CANN 9.1.0，asc-devkit ≥ 9.1（与 `aclblasSgemmBatched` 相同）。

# 算子约束限制

1. 不支持 batch 间 stride 非连续访问；各 batch 矩阵按列主序连续存储，仅支持 lda/ldb/ldc 前导维语义；
2. 不支持广播：各 batch 的 A_i/B_i/C_i 为独立矩阵；
3. m/n/k/batchCount 为运行时入参，无编译期 shape 特化要求；
4. 确定性计算不作要求；
5. `beta = 0` 时 C_i 不必是有效输入；
6. `batchCount > 0` 时三个指针数组必须非空（含 `k=0`/`m=0`/`n=0`）；
7. 内存：单次调用峰值设备侧临时区 `batchCount × m × n × 4B`（FLOAT32 临时 C），受 handle workspace 上限约束。

# 可维可测分析

## 测试设计

### 用例构成（随任务提供，1000 精度 + 200 性能）

| 类别 | 前缀 | 条数 | 覆盖点 |
| --- | --- | --- | --- |
| L0 基础 | TC_L0 | 18 | transa×transb 全 9 组合 × {4³, 8³} × batch=2 |
| 尺寸扫描 | TC_MN | 47 | 1→2048 方阵 + 12 组非方阵矩形 |
| 标量 | TC_AB | 10 | FLOAT16 可精确表示的 alpha/beta 组合 |
| 前导维 | TC_LD | 15 | lda/ldb/ldc 同时 padding 与单维 padding |
| 填充值域 | TC_FL | 18 | A/B/C 分别取 6 种填充 token（含 INF/NAN/EXTREME） |
| batch 覆盖 | TC_CV | 21 | batchCount ∈ {1,2,3,4,5,8,16,32,64,128,256,512,1024} |
| 边界负向 | TC_ED | 37 | no-op、k=0 缩放、空指针、非法枚举/前导维/负维度、空 handle、65504 边界 |
| 扩展精度 | TC_EX | 834 | 尺寸池 × 转置 × 标量 × batchCount 确定性采样 |
| 性能 | TC_PF | 200 | 4 条任务书典型 case + 小尺寸大 batch + 转置全组合 + 8192² 单 batch |

### golden 与判定

- golden：逐 batch FLOAT32 `cblas_sgemm`（输入为先量化到 FLOAT16 的精确值）→ 一次 RNE 舍入到 FLOAT16；
- 逐元素：`|actual - golden| ≤ atol + rtol·|golden|`，`atol = rtol = 2⁻¹⁰`；
- 用例级：有限值 `matched_ratio ≥ 0.99` 且全部有限值元素 `≤ max(1e-1, 32·ULP(golden))`；
- NaN/Inf：同为 NaN、同为同号 Inf、或一方 NaN 一方 ±Inf 视为等效；一方有限判失败；
  非有限值不计入 matched_ratio 与 max_abs_error，单独统计总数与通过数；
- 性能用例（`TC_PF_`）只执行、不参与精度分母。

### 契约审计（真机前可执行）

`tools/audit_csv_contract.py` 按 §2.5 校验顺序独立推算 1200 条用例的期望返回码，与 CSV 的
`expect_result` 比对。当前结果：**1199/1200 一致**，唯一差异为 `TC_ED_138`
（`k=0, beta=0, Aarray/Barray=NULLPTR, batchCount=2`，CSV 期望 SUCCESS，任务书 §2.5 要求 INVALID_VALUE），
处置见《口径声明》。

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT16 档：rtol = atol = 2⁻¹⁰，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-1 或 32·ULP | 生态算子开源精度标准 `mixed_tolerance_standard.md` |
| 性能标准 | 单 case `GPU/NPU ≥ 0.4`；全部性能 case 倍率算术平均 ≥ 0.5 | 任务书 §3.3 |
| 性能采集 | msprof op task duration，自带 5 次 warmup，warmup 后连续 10 次有效样本取均值 | 任务书 §3.3.1 |

## 兼容性分析

1. 新增接口，不修改任何既有函数签名与语义；`aclblasSgemmBatched`/`aclblasCgemmBatched` 行为不变；
2. `aclblasHalf` 为新增公共类型，与 `aclblasComplex` 并列，不影响既有头文件使用者；
3. 复用 `gemm_batched` 内核时 FP32/BF16 实例零改动，仅新增 host 入口与 tiling 参数；
4. `blas/CMakeLists.txt` 只在单算子构建（`--ops`）过滤表新增一项依赖，全量构建路径不变。

# 口径声明

| 争点 | 各档口径 | 采用 | 依据 |
| --- | --- | --- | --- |
| host 实现目录 | 任务书 §2.2：`blas/gemm/arch35/`；仓内同族：`blas/gemm_batched/arch35/` | 采用任务书目录，相对包含复用同族 tiling 与 launcher | 交付验收以任务书为主（主提示词 §0）；同族代码可直接复用，无需复制内核 |
| 测试工程目录 | 任务书 §5：`test/gemm/hgemmBatched/arch35`；官方测试指导与 `build.sh`/`verify_*.py` 路径解析：`test/gemm_batched/hgemm_batched/` | 采用 `test/gemm_batched/hgemm_batched/` | 任务书 §3.5 要求「根据随任务提供的自测用例与测试指导完成自测」；另一路径无法被 `--ops=hgemm_batched` 解析 |
| `k=0` 且 A/B 数组为空 | CSV `TC_ED_138` 期望 SUCCESS；任务书 §2.5/§3.5 要求 INVALID_VALUE | 采用任务书：返回 `INVALID_VALUE` | 任务书为主依据；与 `aclblasSgemmBatched` 已合入行为一致；测试对该行显式 SKIP 并留痕 |
| alpha=0 是否读取 A/B | 任务书未禁止短路；Netlib `cblas_sgemm` 在 alpha==0 时短路 | 采用短路（C = beta*C），不引用 A/B | 与 golden 语义一致；`SgemmBatched` 同款实现 |
| 性能采样口径 | 任务书：msprof op，5 warmup + 10 有效样本；随附脚本：GTest 墙钟（含 host+golden） | 以任务书 msprof 为判定口径；GTest 墙钟仅作保守上界参考 | 任务书 §3.3.1 |
| FLOAT16 标量类型 | `_Float16`（原生）vs `uint16_t`（位模式） | 采用 `uint16_t` | 公共 C 头不依赖编译器扩展；与 CANN `aclFloat16` 一致 |
