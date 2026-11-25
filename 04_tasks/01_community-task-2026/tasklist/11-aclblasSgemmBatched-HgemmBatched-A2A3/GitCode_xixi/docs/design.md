# 需求背景（required）

## 需求来源

CANN 社区任务 2026 — `aclblasSgemmBatched` 与 `aclblasHgemmBatched` 算子开发（Atlas A2 / Atlas A3，`arch22`）。

## 背景介绍

### 算子功能概述

`aclblasSgemmBatched` 与 `aclblasHgemmBatched` 是 `ops-blas` 仓 BLAS Level-3 核心批量矩阵乘接口，分别对应单精度浮点（FP32）和半精度浮点（FP16 / `aclblasHalf`）列主序一般矩阵的批量乘法与线性叠加操作：

$$C[i] = \alpha \cdot \mathrm{op}(A[i]) \cdot \mathrm{op}(B[i]) + \beta \cdot C[i], \quad i \in \{0, 1, \dots, \text{batchCount} - 1\}$$

其中 `Aarray`、`Barray`、`Carray` 均为位于 Device 侧内存的指针数组，数组中每个元素分别指向对应 batch 的列主序矩阵 $A[i]$、$B[i]$、$C[i]$。该设计在指针数组形式上兼容非等步长或分散内存排布的多矩阵并发计算场景，广泛应用于科学计算、推荐排序与深度学习小/中/大批量 GEMM 推理融合。

### 公共头文件与接口定义

公共接口已在 `include/cann_ops_blas.h` 中声明（无需修改头文件）：

```cpp
// FP32 批量 GEMM
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount);

// FP16 批量 GEMM
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const Aarray[], int lda,
    const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const Carray[], int ldc,
    int batchCount);
```

| 参数 | 参数含义 | 数据类型 | 指针所在内存 | 约束 |
| --- | --- | --- | --- | --- |
| `handle` | BLAS 上下文句柄 | `aclblasHandle_t` | Host 对象指针 | 非空，含 `aclrtStream` 与 workspace |
| `transa` / `transb` | $\mathrm{op}(A[i])$ / $\mathrm{op}(B[i])$ 转置选项 | `aclblasOperation_t` | 值传递 | `{ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` |
| `m`, `n`, `k` | 矩阵逻辑维度 | `int` | 值传递 | $\ge 0$ |
| `alpha`, `beta` | 标量缩放系数 | `const float*` / `const aclblasHalf*` | Host 或 Device 指针 | 非空（包括退化返回路径前亦须校验） |
| `Aarray`, `Barray` | 输入矩阵指针数组 | `const T* const[]` | Device 侧数组（元素为 Device 地址） | 当 `batchCount>0 && m>0 && n>0 && k>0 && alpha!=0` 时非空 |
| `Carray` | 输出矩阵指针数组 | `T* const[]` | Device 侧数组（元素为 Device 地址） | 当 `batchCount>0 && m>0 && n>0` 时非空 |
| `lda`, `ldb`, `ldc` | 列主序 Leading Dimension | `int` | 值传递 | $\ge \max(1, \text{physRows})$ |
| `batchCount` | 批次数 | `int` | 值传递 | $\ge 0$ |

### 仓内现状与实现范围

`ops-blas` 仓已有面向 Atlas A2/A3（`arch22`）的 `blas/gemm_batched/arch22/`（复数 `cgemm_batched`）及面向 Atlas 950（`arch35`）的 `sgemm_batched` / `hgemmBatched` 参考目录，但 `arch22` 平台尚未补充 `aclblasSgemmBatched` 与 `aclblasHgemmBatched` 的实现与独立测试。本次开发在 `blas/gemm_batched/arch22/` 与 `test/gemm_batched/` 中补齐 `arch22` 版本的实数批量 GEMM 完整实现与全量自验体系。

---

# 需求分析（required）

## 需求描述

在 `ops-blas` 仓中面向 Atlas A2 / Atlas A3（`arch22`）架构基于 Ascend C 实现 `aclblasSgemmBatched`（FP32）与 `aclblasHgemmBatched`（FP16），满足：
1. 与 cuBLAS `cublasSgemmBatched` / `cublasHgemmBatched` 及 `ops-blas` `arch35` 接口语义一致，完整支持所有 `transa × transb` 组合、任意合法 `m/n/k/batchCount/lda/ldb/ldc`、Host/Device 双侧标量指针以及边界退化语义。
2. 精度通过全部 1000 条（SGEMM 500 + HGEMM 500）功能、边界与异常用例，与 FP32/FP64 累加的 Host CPU `cblas_sgemm` 参考结果误差在门限内。
3. 在 `ascend910b3` 上性能（各 100 条 `TC_PF_*` 性能用例）单条及平均倍率均达到对标 NVIDIA A100（`gpu_ms`）的 $0.8\times$ 以上。

## 需求拆解

1. **Host 参数校验与退化语义对齐**：严格按序校验 `handle` -> `trans` -> 维度与 `ld` 下界 -> `alpha/beta` 非空 -> No-op（`batchCount==0 || m==0 || n==0`） -> `Carray` 非空 -> 标量读取 -> `Aarray/Barray` 非空（仅 `k>0 && alpha!=0`）。
2. **退化与快速向量路径**：`k==0` 或 `alpha==0` 时不触碰 `Aarray/Barray`，直接由 AIV Kernel 对 `C[i]` 原地完成 `beta == 0` 置零或 `beta * C[i]` 缩放（`beta == 1` 时零拷贝跳过）。
3. **高性能双缓冲 Pingpong Cube 核函数**：针对 Atlas A2/A3 的 Cube 单元（AIC）与 Vector 单元（AIV）分离硬件架构，采用手写纯 Ascend C `Mmad` 流水线（`MTE2 -> MTE1 -> M -> FIXPIPE`）与 L0A/L0B/L0C 双缓冲设计，支持全 `NN/NT/TN/TT`（`OP_C` 复用 `OP_T`）、非对齐尾部（Zero-pad / MTE2 Nd2Nz）、L2 缓存友好块调度与小输出块 L0C 写后重用同步。
4. **通用标量与 Epilogue 融合**：当 `alpha == 1 && beta == 0`（且 `k` 满足直出范围）时，`Fixpipe` 直接从 `L0C` 写出至 `C[i]`，省去 workspace 与二次读写；当 `alpha != 1 || beta != 0` 时，Cube 将未缩放高精度 `FP32` 累加结果写出至 chunked workspace，再由 AIV 向量 epilogue 融合完成 `C[i] = alpha * Temp[i] + beta * C[i]` 并支持输出舍入回 FP16 / FP32。
5. **大 Batch 内存管控**：按单次最大 workspace 上限（如 512 MB）或用户托管 workspace 容量自动切片 `firstBatch + curBatches` 分轮发射，确保大 `batchCount` 场景不触发 HBM OOM。

---

# 详细设计（required）

## 算子分析

### 数学公式与列主序映射

由于 Ascend C `Mmad` 与列主序的等价变换关系：列主序下的 $C = \mathrm{op}(A) \cdot \mathrm{op}(B)$ 等价于行主序视角下的：

$$C^T_{N \times M} = \mathrm{op}(B)^T_{N \times K} \cdot \mathrm{op}(A)^T_{K \times M}$$

据此直接映射至 `Mmad(dstL0C, srcB_L0A, srcA_L0B)`（硬件视角左矩阵为 `op(B)^T`、右矩阵为 `op(A)^T`，结果在 L0C 经 `Fixpipe` 直接按 `ND/Col-Major` 步长 `outputLd` 写回 GM，无需显式整体转置开销）：

| 接口 `transa, transb` | 实数等价 | GM 存储排布 | `Nd2NzParams`（B -> L1A, A -> L1B） | `LoadData2D` transpose |
| --- | --- | --- | --- | --- |
| `(N, N)` | `(N, N)` | `A: m×k`, `B: k×n` | `B: n×k` 直读, `A: k×m` 直读 | 视 L0A/L0B 朝向按 Fractal 排布加载 |
| `(N, T/C)` | `(N, T)` | `A: m×k`, `B: n×k` | `B` K-major 加载, `A` normal | 对应通道启用 Fractal 转置或 L0B/L0A 重排 |
| `(T/C, N)` | `(T, N)` | `A: k×m`, `B: k×n` | `B` normal, `A` M-major 加载 | 同上 |
| `(T/C, T/C)` | `(T, T)` | `A: k×m`, `B: n×k` | 两者均按非主维对齐加载 | 均按硬件转置模式加载入 L0A/L0B |

### 支持数据类型与精度保护

- **输入输出类型**：
  - `aclblasSgemmBatched`：输入 $A, B, \alpha, \beta$ 与输出 $C$ 均为 `FP32`（`float`）。在 Atlas A2/A3 Cube 上利用 FP32 `Mmad` 指令累加至 FP32 `L0C`。
  - `aclblasHgemmBatched`：输入 $A, B, \alpha, \beta$ 与输出 $C$ 均为 `FP16`（`aclblasHalf` / `half`）。Host 将 `alpha`/`beta` 转 FP32 传递；Cube 以 FP16 输入、**FP32 `L0C` 高精度累加**；`directOut == 1` 时由 `Fixpipe`（`QuantMode_t::F322F16`）硬件舍入直出至 FP16，非默认标量 `directOut == 0` 时经 `QuantMode_t::NoQuant` 保留 **FP32** 中间结果至 workspace，由 AIV 在 FP32 下完成 $\alpha \cdot \text{Temp} + \beta \cdot \text{FP32}(C)$ 后再 `Cast` 回 FP16，避免二次半精度截断误差。
- **特殊数值保护**：`alpha == 0` 时严禁乘读 $A/B$（防止 $0 \times \text{NaN} / \text{Inf} = \text{NaN}$）；`beta == 0` 时直接在 workspace 写入或 AIV `Duplicate(0)` 覆盖 $C$（防止 $0 \times C_{\text{NaN/Inf}} = \text{NaN}$）。

## 算子实现

### 总体目录与代码组织

实数批量 GEMM 复用同源架构设计，公共 Host 逻辑与双精度/半精度分离核函数集中放置在 `blas/gemm_batched/arch22/`：

- `gemm_batched_real_tiling.h`：Host 与 Kernel 共享的 `GemmBatchedReal22Tiling` POD 结构体与数据类型常量。
- `gemm_batched_real_host_common.h`：统一参数校验 `ValidateRealBatchedParams`、标量预读（Host/Device 兼容）、退化分发 `LaunchDegenerateBeta`、Chunked Workspace 管理 `PrepareChunkWorkspace` 及主执行入口 `ExecuteRealBatched`。
- `sgemm_batched_host.cpp` / `hgemmBatched_host.cpp`：C ABI 入口封装，标量类型解码与异常安全边界。
- `gemm_batched_real_kernel.h` / `gemm_batched_real_kernel.cpp`：核函数发射桥接与后处理 epilogue / Cube wrapper / MatmulImpl 通用兜底。
- `gemm_batched_real_pingpong_fp32.h` / `gemm_batched_real_pingpong_fp16.h`：面向 A2/A3 的纯 Ascend C 高性能双缓冲 Pingpong Cube 模板核函数及向量 Epilogue。

### 实现方案

#### 1. Host 侧设计（`gemm_batched_real_host_common.h`）

1. **校验流水线（`ValidateRealBatchedParams`）**：
   - 步骤 1：`handle == nullptr` -> `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
   - 步骤 2：`transa` / `transb` 合法性 -> `ACLBLAS_STATUS_INVALID_ENUM`。
   - 步骤 3：`m < 0 || n < 0 || k < 0 || batchCount < 0` -> `ACLBLAS_STATUS_INVALID_VALUE`。
   - 步骤 4：`lda >= max(1, (transa==N)?m:k)`、`ldb >= max(1, (transb==N)?k:n)`、`ldc >= max(1, m)` -> `ACLBLAS_STATUS_INVALID_VALUE`。
   - 步骤 5：`alpha == nullptr || beta == nullptr` -> `ACLBLAS_STATUS_INVALID_VALUE`（确保 `batchCount==0` 或 `m==0` 时空标量依然报错）。
   - 步骤 6：`batchCount == 0 || m == 0 || n == 0` -> 返回 `ACLBLAS_STATUS_SUCCESS`（允许 `Aarray/Barray/Carray` 为空）。
   - 步骤 7：`Carray == nullptr` -> `ACLBLAS_STATUS_INVALID_VALUE`。
   - 步骤 8：入口函数通过 `ReadScalarFromHostOrDevice` 读取 `alpha`、`beta` 值；仅当 `k > 0 && alphaVal != 0.0F` 时检查 `Aarray != nullptr && Barray != nullptr`。
2. **执行路径路由（`ExecuteRealBatched`）**：
   - **退化分支**（`alphaVal == 0.0F || k == 0`）：直接调用 `LaunchDegenerateBeta` 启动 AIV 核函数，按列/分组映射对每个 $C[i]$ 执行 `beta == 0` 置零或 `beta != 1` 原地缩放（`beta == 1` 无算子发射，直接返回成功）。
   - **Direct-Out 分支**（`alphaVal == 1.0F && betaVal == 0.0F`）：`directOut = 1`、`outputLd = ldc`，零额外 workspace 占用，单次 `GemmBatchedReal22CubeDo` 直接完成所有 batch 计算。
   - **非默认标量 / Workspace Epilogue 分支**（`alphaVal != 1.0F || betaVal != 0.0F`）：设 `directOut = 0`、`outputLd = CeilAlign(m, 16)`、`batchTempStride = CeilAlign(n * outputLd, 128)`，根据 512 MB 上限或用户提供的 workspace 容量计算单轮可容纳批次数 `chunkCap`，循环分片调用 `GemmBatchedReal22CubeDo`（写 FP32 累加到临时 workspace）与 `GemmBatchedReal22CombineDo`（AIV 完成 $\alpha \cdot \text{Temp} + \beta \cdot C$ 并覆盖写出至 `C[i]`）。
3. **Tiling 结构定义（`GemmBatchedReal22Tiling`）**：

```cpp
struct GemmBatchedReal22Tiling {
    uint32_t m, n, k;
    uint32_t lda, ldb, ldc, outputLd;
    uint32_t firstBatch, batchCount;
    uint32_t transA, transB;
    uint32_t dtype;      // 0: FP32, 1: FP16
    uint32_t directOut;  // 1: Cube 直写 C[i]; 0: Cube 写 FP32 Temp -> AIV Combine
    float alpha, beta;
    uint64_t batchTempStride;
};
```

#### 2. Kernel 侧设计（Cube Pingpong + AIV Epilogue）

```
  [Device Ptr Arrays: Aarray, Barray, Carray]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  (alpha==1 && beta==0)      (alpha!=1 || beta!=0)
  directOut = 1 (零 WS)       directOut = 0 (Chunked FP32 WS)
        │                           │
        ▼                           ▼
  Cube Pingpong Kernel       Cube Pingpong Kernel ──► Temp (FP32, L0C->GM NoQuant)
  (L0C -> Fixpipe C[i])             │
                                    ▼
                             AIV Combine Kernel (UB 4K 分块)
                             C[i] = alpha*Temp + beta*C[i]
```

1. **指针数组解引用**：Kernel 通过 `GlobalTensor<uint64_t>` 读取各 AIC/AIV 核当前负责的 `batch = firstBatch + localBatch` 的 `Aarray[batch]`、`Barray[batch]`、`Carray[batch]` 指针，动态 `SetGlobalBuffer` 绑定各自地址。
2. **Tile 家族与核间负载均衡**：
   - 根据问题形状自动选择适配的 Cube Base Block 与 Tile 组合（如 FP16 支持多档高性能大块 `128×256`、`176×176`、`128×128` 及 `TN` / 小矩阵专属配置；FP32 支持 `128×128`、`64×128`、`128×64` 等）。
   - 全局任务空间 `totalJobs = curBatches × ceil(m / tileM) × ceil(n / tileN)` 在全部 AIC 核心（20 核 on 910B3）间均匀 stride 分发。
   - 针对大矩阵（如 $4096^3$），在 $(M, N)$ Tile 网格上采用 **M-band Swizzle（例如 `M_BAND = 4`）** 调整发射遍历序，提升 192 MB L2 缓存对 A/B 块的重用率，降低跨片 HBM 访存带宽压力。
3. **K-Loop 四级双缓冲流水线与同步**：
   - 使用两套显式 L1 Buffer（L1A_0/L1A_1 与 L1B_0/L1B_1）和 L0A/L0B Ping-Pong 槽位配合 `TQue` / `SetFlag-WaitFlag` 同步：
     - `MTE2` 通过 `DataCopy(Nd2NzParams)` 直接将列主序非对齐或带 padding 的分块异步搬入 L1，并在 K 未对齐 16 时按需零填充尾块避免脏数据混入累加。
     - `MTE1` 通过 `LoadData`（或转置加载路径）将 Fractal 数据块搬入 `L0A` / `L0B`。
     - `M` 引擎执行 `Mmad` 连续 K 步累加至 FP32 `L0C`；利用双 L0C 槽位或 `WaitFlag<HardEvent::FIX_M>` 进行硬件级流水交叠。
     - **L0C 小矩阵复用 Hazard 防护**：当单块元素数 $\le 4096$（如 $m \times n \le 64 \times 64$，常见于多 batch 中小矩阵连续复用同一 L0C 槽位）时，`Mmad` 必须确保前一次 `Fixpipe` 写出已完成或执行显式核内同步，避免单 L0C 写后覆写竞态导致局部精度错误或 AICore 异常。
   - 对非双缓冲覆盖不到的极窄/特殊边缘形状（如任意超小不规则 `ld` 步长超 `uint16_t` 限制），保留经实测验证的 `MatmulImpl` + `Combine` 兜底路径确保功能 100% 完备。
4. **AIV Epilogue（`Combine` 与 `BetaScale`）**：
   - 采用固定 4096 元素/chunk 向量流水（`CopyIn -> Vec Mul/Axpy/Muls -> Cast -> CopyOut`），按 `m <= 512` 对齐分列合并发送或逐列独立映射到 `aivCores`（40 核），并严格在 `alpha == 0` 或 `beta == 0` 分支隔离 `0 * NaN / Inf` 传播。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 说明 |
| --- | --- | --- |
| Atlas A2 / Atlas 800T A2 / 910B3 (`arch22`) | √ | 目标开发与性能验收平台（20 AIC @ 1800 MHz，40 AIV，192 MB L2，64 GB HBM） |
| Atlas A3 (`arch22`) | √ | 同属 `arch22` 指令集与存储层级架构 |

## 硬件资源评估（以 Atlas 910B3 为基准）

- **片上缓存单核容量**：L1 = 512 KB，L0A = 64 KB，L0B = 64 KB，L0C = 128 KB，UB = 192 KB，全局共享 L2 = 192 MB。
- **FP16 Pingpong 资源预算示例（`tileM=128, tileN=256, kL1=64` 双缓冲）**：
  - L1 双缓冲：$2 \times (128 \times 64 \times 2\text{B} + 256 \times 64 \times 2\text{B}) = 96\text{ KB} \ll 512\text{ KB}$（可进一步放大 `kL1` 至 128/256 占约 192–384 KB 以降低 GM 访存次数）。
  - L0A/L0B 双缓冲：`kL0=64` 时单槽 A/B 各约 $16\text{ KB}$ / $32\text{ KB}$，双槽合计 $\le 64\text{ KB}$；单 Tile 输出占 L0C $= 128 \times 256 \times 4\text{B} = 128\text{ KB}$（`128×128` 块占 64 KB，支持完整 `dbL0C = 2` 双 L0C 流水）。
- **FP32 Pingpong 资源预算示例（`tileM=128, tileN=128, kL1=64` 双缓冲）**：
  - L1 双缓冲：$2 \times (128 \times 64 + 128 \times 64) \times 4\text{B} = 128\text{ KB} \ll 512\text{ KB}$。
  - L0C 占用：$128 \times 128 \times 4\text{B} = 64\text{ KB}$，完整适配 128 KB L0C 双缓冲；若使用 `64×128` / `128×64` 则单槽仅占 32 KB。
- **AIV Epilogue UB 预算**：`AIV_CHUNK_ELEMS = 4096`（16 KB per buffer），双缓冲输入 Temp + C + Cast 中间向量共约 96 KB，远低于 192 KB UB 上限。

## 算子约束限制

1. `Aarray`、`Barray`、`Carray` 指针数组及其指向的矩阵数据必须位于 Device 全局内存（HBM）；`alpha`、`beta` 支持 Host 或 Device 指针。
2. 当 `alpha != 1 || beta != 0` 且使用临时 workspace 时，Host 自动申请或复用内部/用户 workspace，单轮切片上限按 512 MB 控制；极端情况下若用户传入极小维度外的高维单矩阵无法满足单 batch 临时空间需求，将返回 `ACLBLAS_STATUS_ALLOC_FAILED`。
3. 工程编译依赖 CANN 9.1.0 工具链与 `bisheng` 编译器（`arch22`）。

---

# 可维可测分析

## 编译与测试验证方案

- **编译命令**：
  ```bash
  bash build.sh --soc=ascend910b3 --ops=sgemm_batched,hgemmBatched
  ```
- **测试用例集（S/H 各自 600 条，共 1200 条）**：
  - 功能与精度用例（各 500 条：`TC_L0`、`TC_MN`、`TC_AB`、`TC_LD`、`TC_FL`、`TC_CV`、`TC_ED`）：覆盖四类转置组合、$1\sim 4096$ 奇偶质数/方形/极瘦长形状、非零 `alpha/beta`（含 0/负数/极值）、`lda/ldb/ldc` 非紧致步长、随机/零值/NaN/Inf 填充及全部非法/退化边界状态码校验。
  - 性能对标用例（各 100 条 `TC_PF_*`）：覆盖小中大矩阵、不同 `batchCount`、非默认 `alpha/beta`（如 `alpha=2,beta=0`、`alpha=1.5,beta=-0.5`）及 $4095^3 / 4096^3$ 大矩阵；采用 5 次 warmup + 10 次 sample 均值统计 `msprof` `Task Duration(us)` 及 GTest `[PERF] avg_us`。

## 精度标准/性能标准

| 验收项 | 目标标准 | 标准来源 |
| --- | --- | --- |
| SGEMM 精度 | 对标 CPU FP32（FP64 累加）`cblas_sgemm`；MERE $\le 2^{-13}$，MARE 倍率 $\le 10.0$ | `aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md` |
| HGEMM 精度 | 对标 CPU FP32 计算后转回 FP16；MERE $\le 2^{-10}$，MARE 倍率 $\le 32.0$ | `aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md` |
| 性能表现 | Ascend 910B3 对标 NVIDIA A100（`gpu_ms`）：单条 `TC_PF_*` 用例倍率 $\ge 0.8\times$ 且 100 条平均倍率 $\ge 0.8\times$（S/H 均达标） | `aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md` |

## 兼容性分析

- 本次在 `arch22` 新增实现与测试，公共头文件 `include/cann_ops_blas.h` 签名零改动；`arch22` 原有 `cgemm_batched` 与 `arch35` 既有实现及构建脚本保持完全向后兼容。

## 交付文件清单

```
ops-blas/
├── blas/gemm_batched/
│   ├── README.md
│   └── arch22/
│       ├── gemm_batched_real_tiling.h
│       ├── gemm_batched_real_host_common.h
│       ├── sgemm_batched_host.cpp
│       ├── hgemmBatched_host.cpp
│       ├── gemm_batched_real_kernel.h
│       ├── gemm_batched_real_kernel.cpp
│       ├── gemm_batched_real_pingpong_fp32.h
│       └── gemm_batched_real_pingpong_fp16.h
└── test/gemm_batched/
    ├── sgemm_batched/...
    └── hgemm_batched/...
```
