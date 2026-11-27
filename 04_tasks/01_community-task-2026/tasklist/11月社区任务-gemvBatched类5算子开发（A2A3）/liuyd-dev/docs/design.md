# aclblasSgemvBatched 类 5 个算子（Atlas A2/A3 / arch22）设计文档

> **文档状态**：设计方案初稿（纯设计阶段，不含未实测的 NPU 性能/内存数值；实测结果待开发与验收阶段回填）
> **目标仓库**：`cann/ops-blas`（代码路径 `blas/gemv_batched/arch22/`、测试路径 `test/gemv_batched/arch22/`）
> **目标硬件**：Atlas A2 系列（Ascend 910B3，dav-2201，40 AIV / 20 AIC，单核 UB 192 KiB）及 Atlas A3 系列
> **软件栈**：CANN 9.1.0，Ascend C Kernel 直调模式

---

## 1. 需求背景与任务分析

### 1.1 需求来源

本设计面向 2026 年 11 月 CANN 社区任务《11月社区任务-gemvBatched类5个算子开发（A2/A3）》（任务 ID：`d82eff6522c34a1a89710f93bb8926b7`），在 Atlas A2/A3（`arch22`）平台上对标 NVIDIA cuBLAS 实数批量矩阵-向量乘族接口，实现并优化以下 5 个 API：

| 算子 API | 变体代号 | `A[i]` / `x[i]` dtype | `y[i]` dtype | `alpha` / `beta` | 计算累加精度 |
| --- | --- | --- | --- | --- | --- |
| `aclblasSgemvBatched` | `S` | `FP32` | `FP32` | `FLOAT32` | `FP32` |
| `aclblasHSHgemvBatched` | `HSH` | `FP16` (`uint16_t`) | `FP16` (`uint16_t`) | `FLOAT32` | `FP32` 累加后收窄至 `FP16` |
| `aclblasHSSgemvBatched` | `HSS` | `FP16` (`uint16_t`) | `FP32` | `FLOAT32` | `FP32` 累加直接输出 `FP32` |
| `aclblasTSTgemvBatched` | `TST` | `BF16` (`uint16_t`) | `BF16` (`uint16_t`) | `FLOAT32` | `FP32` 累加后收窄至 `BF16` |
| `aclblasTSSgemvBatched` | `TSS` | `BF16` (`uint16_t`) | `FP32` | `FLOAT32` | `FP32` 累加直接输出 `FP32` |

### 1.2 数学公式与语义约定

对每个 `i = 0, 1, ..., batchCount - 1` 独立计算：

```text
y_i = alpha * op(A_i) * x_i + beta * y_i
```

关键语义约束（与 cuBLAS `cublas<t>gemvBatched` 及任务书 2026-11-24 修订口径严格一致）：

1. **矩阵物理存储（与 `trans` 无关）**：每个 `A_i` 按列主序存储，物理维度恒为 `lda × n`，`lda >= max(1, m)`。`trans` 仅作用于逻辑运算 `op(A_i)`，不改变 `A_i` 的物理布局。
2. **向量长度**：
   - `trans == ACLBLAS_OP_N`：`op(A_i) = A_i`（`m × n`），`x_i` 逻辑长度 `n`，`y_i` 逻辑长度 `m`；
   - `trans == ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`：`op(A_i) = A_i^T`（`n × m`），`x_i` 逻辑长度 `m`，`y_i` 逻辑长度 `n`。实数算子中 `ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 完全等价（不做共轭）。
3. **非单位与负步长**：`incx` / `incy` 为任意非零 `int`（不含 `0` 与 `INT_MIN`）。负步长遵循标准 BLAS 反向索引语义：第 `k` 个逻辑元素（`0 <= k < len`）位于物理偏移 `(len - 1 - k) * |inc|`。
4. **退化与快速返回语义**：
   - `batchCount == 0`、`m == 0` 或 `n == 0`：合法 quick return / no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不解引用 `Aarray`/`xarray`/`yarray`（三者允许为 `nullptr`），不启动 kernel；
   - `alpha == 0.0f`：仅执行 `y_i = beta * y_i`，禁止读取 `A_i` / `x_i`（保障 `alpha=0` 且 `x` 含 `NaN` 不污染 `y`）；
   - `beta == 0.0f`：禁止读取 `y_i` 旧值（保障 `beta=0` 且 `y` 含 `NaN` 不污染输出）；
   - `alpha == 0.0f && beta == 1.0f`：原地无变化，可直接返回 `ACLBLAS_STATUS_SUCCESS`（或空操作）。

---

## 2. 仓内源码现状核查（`ops-blas` `upstream/master`）

开发前已对 `cann/ops-blas` 当前 `upstream/master` 逐文件核查源码行号与行为差异：

### 2.1 公开接口声明与公共辅助

- `include/cann_ops_blas.h`（L287–L308）：5 个实数变体声明（`aclblasSgemvBatched` L287、`aclblasHSHgemvBatched` L291、`aclblasHSSgemvBatched` L296、`aclblasTSTgemvBatched` L301、`aclblasTSSgemvBatched` L306）均已存在，本任务无需修改头文件。
- `blas/common/helper/gemv_batched_span_check.h`（L1–L39）：提供 `GemvBatchedSpanExceedsInt32(isOpN, elemFactor, m, n, lda, incx, incy, batchCount)` 公共 64 位跨度校验函数。注意在 `m=0`/`n=0`/`batchCount=0` 快速返回分支之后调用，且需提前拒绝 `incx == INT_MIN || incy == INT_MIN` 防止取负溢出。

### 2.2 `arch35` 参考实现事实（`blas/gemv_batched/arch35/`）

- `blas/gemv_batched/arch35/gemv_batched_host.cpp`（共 329 行）：
  - `ValidateGemvBatchedParams`（L89–L117）：将 `m <= 0 || n <= 0` 与 `batchCount <= 0` 判为 `INVALID_VALUE`，且仅允许 `OP_N / OP_T`（拒绝 `OP_C`），与 A2/A3 任务书 §2.4/§2.5 存在差异；
  - `CalTilingTranspose`（L120 起）：围绕 950 架构 UB 与 RegBase 路径切分 `dotTile`/`outTile`。
- `blas/gemv_batched/arch35/gemv_batched_kernel.cpp`（共 630 行）：使用了 950 专属特性与 SIMT/RegBase 路径，不可直接照搬至 dav-2201。

### 2.3 当前 `arch22` 基线现状与待补齐缺口（`commit feb4f9c4`）

仓内 `blas/gemv_batched/arch22/gemv_batched_host.cpp`（149 行）与 `gemv_batched_kernel.cpp`（415 行）已有早期基础版本，但相对本任务书验收标准存在以下关键缺口，均需在本方案中重构补齐：

| 维度 | 当前 `arch22` 现状（`feb4f9c4`） | 任务书 §2/§3 要求 | 本设计改进方案 |
| --- | --- | --- | --- |
| 参数校验顺序 | `handle` → `m/n/lda/incx/incy/batchCount` → 指针 → `trans` | `handle` → `trans` → `m/n` → `lda` → `incx/incy` → `batchCount` → `alpha/beta` → `Aarray/xarray/yarray` → quick return | 严格按任务书 §2.5 校验顺序重构 `ValidateGemvBatchedParams` |
| `ACLBLAS_OP_C` | 误返回 `ACLBLAS_STATUS_INVALID_ENUM` | 实数算子 `OP_C ≡ OP_T` | 放行 `OP_C` 并统一映射为转置路径 |
| Quick Return | `m==0`/`n==0`/`batchCount==0` 误返 `INVALID_VALUE` | 合法 no-op，返回 `SUCCESS`，允许三指针数组为空 | 结构参数合法性校验通过后优先执行 quick return |
| `INT_MIN` 步长 | 未显式拦截 `incx/incy == INT_MIN` | 负步长取绝对值溢出，须返回 `INVALID_VALUE` | 显式拦截 `incx == 0 || incx == INT32_MIN || incy == 0 || incy == INT32_MIN` |
| 退化标量语义 | `alpha==0` 仍读 `A`/`x`；`beta==0` 路径未彻底隔离旧 `y` | `alpha=0` 不读 `A/x`（防 NaN 传播）；`beta=0` 不读 `y` 旧值 | 独立 `scale_only` 路径处理 `alpha==0`；主核显式隔离 `beta==0` 读路径 |
| 多核并行维度 | **仅按 `batchCount` 分核**（`batchCount=1` 时只用 1/40 核） | 全尺寸（含 `batchCount=1`、`4096×4096`）均需 ≥ 0.8× A100 kernel 性能 | 改为 **2D 网格分核 `(batch, outTile)` + 窄输出 Split-K**，吃满全部 40 个 AIV 核 |
| 热路径寄存器与流水 | 单一大核混含多 dtype 与标量/向量分支，易产生寄存器溢出与同步停顿 | 带宽受限场景接近 910B3 HBM 有效带宽上限 | 按 `trans=N`、`trans=T`、`scale_only`、`strided` 拆分零溢出专用 `__global__` 核函数，连续路径启用双缓冲对齐 `DataCopy` |

---

## 3. 详细架构设计

### 3.1 总体执行与分派流程

```text
aclblas{S,HSH,HSS,TST,TSS}gemvBatched(...)
  │
  ├─► [1] Host 侧严格顺序参数校验（ValidateGemvBatchedParams）
  │     handle → trans(N/T/C) → m,n(>=0) → lda(>=max(1,m)) → incx,incy(!=0,!=INT_MIN)
  │     → batchCount(>=0) → alpha,beta(!=null)
  │     → quick return check (batchCount==0 || m==0 || n==0 => SUCCESS)
  │     → Aarray,xarray,yarray(!=null)
  │     → 64-bit 跨度校验（A/x/y 寻址统一在 kernel 使用 uint64_t / int64_t 防 32 位溢出）
  │
  ├─► [2] 退化分支直派
  │     ├─ alpha == 0.0f && beta == 1.0f  => 直接返回 SUCCESS（不启动 kernel）
  │     └─ alpha == 0.0f                  => 启动 gemv_batched_scale_kernel（只写 y = beta*y，不读 A/x；beta==0 时直接写 0）
  │
  └─► [3] 计算主核分派（KERNEL_TYPE_AIV_ONLY，按 dtype 模板实例化）
        ├─ 连续单位步长热路径（incx == 1 && incy == 1）：
        │    ├─ trans == N  => gemv_batched_n_contig_kernel（按 batch × mTile 二维分核）
        │    └─ trans != N  => gemv_batched_t_contig_kernel（按 batch × nTile 二维分核；极小 n 大 m 走 Split-K 或多段流水）
        └─ 通用非单位/负步长路径（incx != 1 || incy != 1）：
             └─ gemv_batched_strided_kernel（支持任意非零 ±incx/±incy，保持列主序读 A 连续）
```

### 3.2 Host 侧参数校验与边界处理

校验顺序严格对齐任务书 §2.5，避免任何非法结构参数被 quick return 吞掉：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C` → `ACLBLAS_STATUS_INVALID_ENUM`
3. `m < 0 || n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
4. `lda < std::max(1, m)` → `ACLBLAS_STATUS_INVALID_VALUE`
5. `incx == 0 || incy == 0 || incx == INT32_MIN || incy == INT32_MIN` → `ACLBLAS_STATUS_INVALID_VALUE`
6. `batchCount < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
7. `alpha == nullptr || beta == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`
8. **Quick return**：`if (batchCount == 0 || m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;`
9. `Aarray == nullptr || xarray == nullptr || yarray == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`（注：`alpha == 0` 场景下，为兼容任务书 §2.4「`batchCount>0 且 m,n>0` 时 `Aarray/xarray/yarray` 为空返回 `INVALID_VALUE`」，入口仍校验数组指针非空，但 kernel 内部绝不解引用 `Aarray[i]` / `xarray[i]`）。

### 3.3 多核 2D Tiling 与访存流水设计（910B3 40 AIV 满载）

910B3 单芯片配备 **40 个 AIV 核**、单核 UB **192 KiB**。GEMV 属典型 **Memory-Bandwidth-Bound** 算子（算术强度 < 1 FLOP/Byte），性能上限由 HBM 有效读带宽与 40 核负载均衡共同决定。

#### 3.3.1 `trans = N` 连续路径（`y_i[m] = alpha * A_i[m×n] * x_i[n] + beta * y_i[m]`）

- **天然列主序优势**：`A_i` 列主序存储，第 `j` 列的连续一段 `A_i[m0 : m0 + tileM, j]` 在 HBM 中物理连续！
- **2D 分核策略**：
  - 将 `m` 按 `tileM` 切分为 `mTiles = ceil(m / tileM)` 块，总逻辑任务块 `totalTiles = batchCount * mTiles`；
  - `usedCoreNum = min(40, totalTiles)`，每个 AIV 核通过 `for (uint64_t tid = blockIdx; tid < totalTiles; tid += usedCoreNum)` 领取 `(b = tid / mTiles, mTileIdx = tid % mTiles)`；
  - **动态 `tileM` 选择**：优先使 `totalTiles >= 40` 以喂满全部 40 AIV 核，同时对齐 32B/256B（FP32 取 64 元素倍数如 `tileM ∈ {64, 128, 256, 512, 1024}`，FP16/BF16 取 128 元素倍数）；当 `batchCount=1, m=4096` 时，`tileM = 128` 产生 `32` 块（或 `tileM = 128`/`64` 视 `n` 调整为 `40` 核均衡分摊），无跨核通信、无原子操作！
- **内核计算流水（单核内）**：
  1. 一次批量读取当前 `x_i[k0 : k0 + kBlk]`（`kBlk = 256`）至 UB 并转 `FP32`（`H/T` 变体用 `Cast<float, TIn>`）；
  2. 初始化 `acc[tileM]` 为 `0.0f`（`Duplicate`）；
  3. 对 `kBlk` 内各列 `j`：双缓冲 `DataCopy` 预取 `A_i[m0 : m0 + tileM, j]`（列内连续！），`H/T` 变体 `Cast` 到 `FP32`，执行向量 `Axpy(acc, aCol32, xVal, tileM)`；
  4. 所有 `n` 列累加完成后：`Muls(acc, acc, alpha)`；若 `beta != 0.0f`，读取 `y_i[m0 : m0 + tileM]` 转 `FP32` 做 `Axpy(acc, yOld32, beta, tileM)`（`beta == 0.0f` 完全跳过 `y` 读取）；
  5. 若输出为 `FP16/BF16`（`HSH`/`TST`），执行 `Cast<TOut, float>(outBuf, acc, RoundMode::CAST_RINT / CAST_NONE)` 后写回 `y_i`（32B 对齐走 `DataCopy`，尾块用 `DataCopyPad`）。

#### 3.3.2 `trans = T / C` 连续路径（`y_i[n] = alpha * A_i^T[n×m] * x_i[m] + beta * y_i[n]`）

- **天然列主序优势**：`op(A_i)` 的第 `j` 行正是物理矩阵 `A_i` 的第 `j` 列 `A_i[0 : m, j]`，沿归约维 `m` **100% 物理连续**！
- **2D 分核策略**：
  - 总输出元素数为 `batchCount * n`，按 `tileN` 切分为 `nTiles = ceil(n / tileN)` 块，`totalTiles = batchCount * nTiles`；
  - `usedCoreNum = min(40, totalTiles)`，每个 AIV 核处理若干输出列 `j`；
  - 内层循环沿 `m` 按 `tileK`（如 FP32 `tileK = 2048` = 8 KiB，FP16/BF16 `tileK = 4096` = 8 KiB）分块：
    - 一次加载 `x_i[k0 : k0 + kLen]`，复用给同一核负责的 `colsPerPass`（如 4~8 列）个 `A_i[k0 : k0 + kLen, j]` 列，显著降低 `x_i` 重复读带宽；
    - 逐列双缓冲 `DataCopy` 读 `A_i[:, j]`，与 `x32` 做 `Mul` 后累加到向量部分和 `partSum[tileK]`（避免每块都调用高延迟 `ReduceSum`！），整条 `m` 扫完后仅做 **1 次 `WholeReduceSum` / `ReduceSum`** 得到精确标量点积；
  - **极端瘦输出特例**（`batchCount * n < 40` 且 `m >= 2048`，如 `batchCount=1, n=1..16, m=4096`）：若单列按 `n` 分核核数不足，可进一步引入按 `m` 切段的 `Split-K` 规约（利用 `handle` 既有 workspace 或原子/二阶段汇总）或提高每核双缓冲深度，确保小 `n` 大 `m` 仍满足 0.8× A100 门槛。

#### 3.3.3 通用步长路径（`incx != 1 || incy != 1`）

- 保持 `A_i` 列主序连续读取不变，仅对 `x_i`（读）与 `y_i`（读写）做带步长（含负步长反向偏移 `base + (len - 1 - idx) * |inc|`）的窗口收集/写回；
- 当 `|incx| <= 16` 且窗口连续段适合 UB 时，批量读入窗口再提取；大跨步时安全标量加载/存储，严禁越界触碰步长空洞之外的内存。

### 3.4 片上 UB 静态内存预算（单核上限 192 KiB）

所有主核均严格控制单核 UB 峰值 ≤ 160 KiB（留出 ≥ 32 KiB 余量给编译器栈与临时缓冲）：

| Kernel 路径 | 主要 UB 缓冲分配（最重场景：FP16/BF16 需额外 FP32 Cast 缓冲） | 峰值 UB 估算 | 是否 ≤ 192 KiB |
| --- | --- | --- | --- |
| `trans=N` 连续双缓冲 (`tileM=1024`, `kBlk=256`) | `aInQ` (2×4 KiB) + `a32` (4 KiB) + `xIn` (1 KiB) + `x32` (1 KiB) + `acc` (4 KiB) + `yIn/yOut` (8 KiB) + `ptrBuf` (2 KiB) | ~28 KiB（可按需放大 `tileM=2048/4096` 至 ~96 KiB） | 是（留余量 > 90 KiB） |
| `trans=T` 连续双缓冲 (`tileK=2048`, 4 列复用 `x`) | `xIn/x32` (16 KiB) + `aInQ` (2×8 KiB) + `a32` (8 KiB) + `partSum[4]` (32 KiB) + `work` (8 KiB) + `yBuf` (4 KiB) | ~84 KiB | 是（留余量 > 100 KiB） |
| `scale_only` (`tile=2048`, 双缓冲) | `yInQ` (2×8 KiB) + `yOutQ` (2×8 KiB) + `y32` (8 KiB) | ~40 KiB | 是 |
| `strided` 通用路径 (`tile=1024`) | `aIn/a32` (8 KiB) + `xWin/x32` (16 KiB) + `acc/y` (16 KiB) | ~48 KiB | 是 |

### 3.5 代码文件规划（`ops-blas` 仓）

- `blas/gemv_batched/arch22/gemv_batched_tiling_data.h`：Host/Device 共享 `GemvBatchedTilingData` 结构体（使用 `#pragma once`，包含 `dtype`、`trans`、`m`、`n`、`lda`、`incx`、`incy`、`batchCount`、`alpha`、`beta`、`tileM/tileK`、`usedCoreNum`、`mode` 等，64 位安全索引域）。
- `blas/gemv_batched/arch22/gemv_batched_kernel.h`：声明拆分后的专用核函数 launch 封装接口（`#pragma once`）。
- `blas/gemv_batched/arch22/gemv_batched_host.cpp`：5 个公开 API 入口、严格顺序参数校验、quick return、退化分支分派、2D Tiling 计算与 kernel 发射错误透传。
- `blas/gemv_batched/arch22/gemv_batched_kernel.cpp`：纯 AIV（`KERNEL_TYPE_AIV_ONLY`）专用热路径核函数 + 通用步长核函数 + `scale_only` 核函数，全路径 64 位偏移计算。
- `blas/gemv_batched/README.md`：核对并补齐 Atlas A2/A3 5 个变体的产品支持与调用说明。
- `test/gemv_batched/arch22/`：支持 5 个变体 CSV 驱动精度与 `[PERF]` 性能采集的 GTest 工程（及 `handle=nullptr`、`y` 哨兵保护等接口单测）。

---

## 4. 精度、性能与测试验证规划

### 4.1 精度验证计划

- **用例规模**：5 个变体（`sgemv_batched`、`hsh_gemv_batched`、`hss_gemv_batched`、`tst_gemv_batched`、`tss_gemv_batched`）各 1000 条 CSV 驱动精度用例（共 5000 条）+ 接口/语义单元测试（覆盖 `handle=nullptr`、`OP_C==OP_T`、`batchCount/m/n=0`、`INT_MIN` 步长、`alpha=0 & x=NaN`、`beta=0 & y=NaN`、非连续 `lda` 哨兵保护）。
- **Golden 与阈值**（任务书 §3.2）：CPU Netlib `cblas_sgemv` 逐 batch（FP64 累加后按输出 dtype 收窄）；
  - `S / HSS / TSS`（FP32 输出）：`atol = rtol = 2^-13`，`matched_ratio >= 0.99`，`max_abs_error <= max(1e-2, 32*ULP)`；
  - `HSH`（FP16 输出）：`atol = rtol = 2^-10`，`matched_ratio >= 0.99`，`max_abs_error <= max(1e-1, 32*ULP)`；
  - `TST`（BF16 输出）：`atol = rtol = 2^-7`，`matched_ratio >= 0.99`，`max_abs_error <= max(1e-1, 32*ULP)`。

### 4.2 性能验证计划

- **用例规模**：5 个变体各 200 条 `TC_PF_*` 性能用例（共 1000 条），全部对标任务包 `gpu_baseline.csv`（A100 `gpu_ms` 已 200/200 回填）。
- **正式判据**（任务书 §3.3）：使用 `msprof` 采集 NPU kernel `Task Duration(us)`（warmup 后 10 次有效样本算术平均），逐条满足 `NPU <= 1.25 * GPU`（等价 `GPU/NPU >= 0.80`）且整体平均倍率 `>= 0.80`。
- **辅助交叉验证**：运行任务包原版 `verify_performance.py`（解析 `[PERF] avg_us`，不使用 `taskset` 限核），与正式判据交叉核对。
- **内存验收**：任务书 §3.4 明确「不涉及」，提交件按「不涉及」口径出具，不填报未实测 GB 数值。

### 4.3 后续演进与实测回填计划

1. 本初稿聚焦架构正确性、语义合规与 40 AIV 满载设计，用于指导 `arch22` 实现与社区评审；
2. 代码在 Atlas 800T A2（910B3）完成冻结构建、5000 条精度与 1000 条性能实测后，如实补充性能分析、瓶颈剖析及交付件清单；
3. 若极端小 `n` 大 `m` 转置用例或大 `batchCount` 小矩阵用例在实机上出现核数饥饿或指针数组读延迟瓶颈，将在代码迭代中针对性引入 `Split-K` 归约或批量指针预取优化。
