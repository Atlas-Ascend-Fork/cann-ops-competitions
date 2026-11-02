# aclblasHgemmBatched 算子设计文档

---

# 1 需求背景（required）

## 1.1 需求来源

| 项目 | 内容 |
|------|------|
| 任务来源 | 社区任务「10月社区任务-aclblasHgemmBatched算子开发（950）」 |
| 任务书 | `aclblasHgemmBatched_Atlas950PR_task_doc.md` |
| 主仓 | https://gitcode.com/cann/ops-blas |
| 评测集模式 | 模式 A（有评测集，任务方随任务下发） |
| 评测集材料 | `test_cases/hgemm_batched_test.csv`（1200 条功能用例）、`test_cases/gpu_baseline.csv`（200 条性能基线）、`verify_accuracy.py`、`verify_performance.py`、`gen_csv.py` |

## 1.2 背景介绍

本任务为补齐昇腾算子开源仓（ops-blas）Blas 仓能力，在 Atlas 950PR（Ascend 950PR）上基于 Ascend C 编程语言实现半精度实数（FLOAT16）批量通用矩阵乘算子 `aclblasHgemmBatched`。

- **算子定位**：Contraction（GEMM 族，Level-3 BLAS，半精度批量通用矩阵乘）。
- **对标基线**：cuBLAS `cublasHgemmBatched`，参数序列一一对应。
- **目标芯片/架构**：Ascend 950PR / arch35（NpuArch `DAV_3510`）。
- **CANN 版本**：9.1.0（`/usr/local/Ascend/cann-9.1.0`）。
- **工程模式**：Ascend C kernel 直调（基于 ops-blas 同族 GEMM batch kernel 直调，复用 Gemm 模板），实现代码放 `blas/gemm/arch35/`。
- **代码架构**：SIMD（实现载体：Cube）；Cube 实现路径：Blaze / tensor_api 路线。
- **golden 说明**：golden 非评测集预冻结值，由测试工程按任务书 §3.2 规定方法现场生成——Netlib `cblas_sgemm`（BLAS 3.10.0）逐 batch FLOAT32 计算后按 IEEE 754 就近舍入（round-to-nearest-even）转换为 FLOAT16。测试用例、性能基线与验收脚本为任务方下发、提交方不得改写。

# 2 需求分析（required）

## 2.1 需求描述

实现半精度实数批量通用矩阵乘算子 `aclblasHgemmBatched`：对每个 batch i（i∈[0,batchCount)），

```
C_i = alpha × op(A_i) × op(B_i) + beta × C_i
```

其中 A_i、B_i、C_i 均为 FLOAT16（半精度）矩阵，alpha、beta 为 FLOAT16 标量（以 `aclblasHalf` 指针传入），Aarray/Barray/Carray 为指向 batchCount 个矩阵的指针数组。

### 2.1.1 接口签名（schema）

严格对齐 cuBLAS `cublasHgemmBatched` 参数序列，alpha/beta 用 `const aclblasHalf*` 指针，Aarray/Barray/Carray 用指针数组，维数与 batchCount 用 `int`：

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

声明落点：`include/cann_ops_blas.h`，与既有 `aclblasSgemmBatched`（L480）/`aclblasCgemmBatched`（L487）同头、风格一致，与其他产品线共用同一 API，**禁止定义 950PR 私有平行接口**。

### 2.1.2 参数规格

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| handle | Host | 输入 | `aclblasHandle_t` | 库上下文句柄，携带 stream；nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| transa | Host | 输入 | `aclblasOperation_t` | op(A_i) 转置模式，枚举 {N,T,C}；非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM` |
| transb | Host | 输入 | `aclblasOperation_t` | op(B_i) 转置模式，枚举 {N,T,C}；非法枚举 → `ACLBLAS_STATUS_INVALID_ENUM` |
| m | Host | 输入 | `int` | C_i 行数 / op(A_i) 行数，≥0；m<0 → INVALID_VALUE；m=0 合法 no-op |
| n | Host | 输入 | `int` | C_i 列数 / op(B_i) 列数，≥0；n<0 → INVALID_VALUE；n=0 合法 no-op |
| k | Host | 输入 | `int` | 内维，≥0；k<0 → INVALID_VALUE；k=0 逐 batch 仅 beta 缩放 |
| alpha | Host | 输入 | `const aclblasHalf*` | FLOAT16 标量指针，不可 nullptr；nullptr → INVALID_VALUE |
| Aarray | Device | 输入（只读） | `const aclblasHalf* const*` | batchCount 个矩阵指针数组；batchCount>0 时不可 nullptr |
| lda | Host | 输入 | `int` | A_i 前导维（列主序），lda ≥ max(1, 行数)；否则 INVALID_VALUE |
| Barray | Device | 输入（只读） | `const aclblasHalf* const*` | batchCount 个矩阵指针数组；batchCount>0 时不可 nullptr |
| ldb | Host | 输入 | `int` | B_i 前导维（列主序），ldb ≥ max(1, 行数)；否则 INVALID_VALUE |
| beta | Host | 输入 | `const aclblasHalf*` | FLOAT16 标量指针，不可 nullptr；beta=0 时 C_i 不必有效输入 |
| Carray | Device | 输入/输出（原地） | `aclblasHalf* const*` | batchCount 个矩阵指针数组，原地覆写；batchCount>0 时不可 nullptr |
| ldc | Host | 输入 | `int` | C_i 前导维（列主序），ldc ≥ max(1, m)；否则 INVALID_VALUE |
| batchCount | Host | 输入 | `int` | 参与计算的矩阵个数，≥0；<0 → INVALID_VALUE；=0 合法 no-op |

返回值：`aclblasStatus_t`，语义与 `include/cann_ops_blas_common.h` 一致（`ACLBLAS_STATUS_SUCCESS` / `INVALID_VALUE` / `INVALID_ENUM` / `HANDLE_IS_NULLPTR`）。

### 2.1.3 边界语义

- `m=0` / `n=0` / `batchCount=0`：合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不访问 Aarray/Barray/Carray。
- `k=0`：逐 batch 仅执行 `C_i = beta × C_i`（`beta=0` 时置零），不引用 A/B，返回 SUCCESS。
- `alpha=0`（k>0）：退化为 `C_i = beta × C_i`，不引用 A/B。

### 2.1.4 精度与性能目标

- 精度（FLOAT16 档）：`rtol = atol = 2^-10`；`required_matched_ratio = 0.99`；`max_abs_error_limit = 1e-1 or 32*ULP`；判定式 `|actual - golden| ≤ atol + rtol × |golden|`。
- 性能（Ascend 950PR）：单 case 倍率 ≥ 0.4× GPU 且全部 case 倍率算术平均 ≥ 0.5× GPU，两项须同时满足。

## 2.2 需求拆解

1. 算子名 `aclblasHgemmBatched`，实现半精度批量 GEMM：`C_i = alpha*op(A_i)*op(B_i) + beta*C_i`。
2. `op()` 为转置 N/T/C（FLOAT16 实数下 T 与 C 等价）；A_i 为 m×k（N）或 k×m（T/C），B_i 为 k×n（N）或 n×k（T/C），C_i 为 m×n；列主序存储。
3. Aarray/Barray/Carray 为指向 batchCount 个矩阵的指针数组；前导维 lda/ldb/ldc。
4. 边界：m=0/n=0/batchCount=0 合法 no-op 返回 SUCCESS；k=0 逐 batch 仅 `C_i=beta×C_i`。
5. alpha/beta 不可 nullptr；batchCount>0 时 Aarray/Barray/Carray 不可 nullptr。
6. 对标基线 cuBLAS `cublasHgemmBatched`，参数序列一一对应。
7. 接口声明入 `include/cann_ops_blas.h` 共用同一 API，禁止 950PR 私有平行接口。
8. 工程模式：kernel 直调（复用 Gemm 模板），实现代码放 `blas/gemm/arch35/`。
9. 参数校验与异常码：handle 空→HANDLE_IS_NULLPTR；非法枚举→INVALID_ENUM；非法值→INVALID_VALUE。
10. 目标硬件 Ascend 950PR，CANN 9.1.0；golden 由 Netlib cblas_sgemm 逐 batch FLOAT32 计算后 IEEE 754 就近舍入到 FP16。
11. 精度（FLOAT16 档）：rtol=atol=2^-10；matched_ratio≥0.99；max_abs_error≤1e-1 或 32*ULP；判定式 `|actual-golden|≤atol+rtol×|golden|`。
12. Inf/NaN 特殊值规则：golden 与 actual 同为 NaN/同号 Inf/一方 NaN 另一方 ±Inf 视为等效；非有限值不计入 matched_ratio/max_abs_error，单独统计。
13. k 过大导致 FLOAT16 溢出超差属非实现问题，须举证（CPU golden 对比、误差随 k 曲线、累加档位），不得放宽阈值。
14. 性能：msprof op 模式，warmup 5 次不计 + 有效采样 10 次取平均；单 case ≥0.4× GPU 且全部 case 平均 ≥0.5× GPU。
15. 性能基线：`gpu_baseline.csv`（200 条已回填 gpu_ms）。
16. 内存要求：不涉及。
17. 自验：ops-blas test 框架 CSV+GTest，覆盖 transa/transb 全组合、尺寸扫描、batchCount 边界、标量特殊值、padding、边界负向、Inf/NaN、性能用例。
18. 落点：声明入 `include/cann_ops_blas.h`；实现 `blas/gemm/arch35/`；测试 `test/gemm_batched/hgemm_batched/arch35/`（含 csv，落位说明见 §6.1.3）。
19. 交付件：算子设计文档（PR 入 cann-competitions）、自测用例及测试代码、自测报告、待验收代码地址。
20. 代码合入遵循仓规范，不含冗余，kernel 设计合理清晰。

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

```
// 输入输出
输入: A_i (op=N: m×k；op∈{T,C}: k×m，FLOAT16，列主序)  逐 batch i∈[0,batchCount)
      B_i (op=N: k×n；op∈{T,C}: n×k，FLOAT16，列主序)
      C_i (m×n，FLOAT16，列主序，输入旧值 + 原地覆写输出)
      alpha、beta（FLOAT16 标量，以 aclblasHalf* 指针传入）
输出: C_i (m×n，FLOAT16，列主序，原地覆写)

// 数学公式（对每个 batch i）
op(A_i)[p][q] = A_i[p][q]（op=N）或 A_i[q][p]（op∈{T,C}；实数下 T 与 C 等价）
C_i[i][j] = alpha × Σ_{p=0}^{k-1} op(A_i)[i][p] × op(B_i)[p][j] + beta × C_i[i][j]
            i∈[0,m)，j∈[0,n)，逐 batch i∈[0,batchCount)
```

### 3.1.2 支持数据类型

| 数据类型 | 说明 |
|----------|------|
| FLOAT16（`aclblasHalf`） | 输入 A/B/C 与标量 alpha/beta 均为 FLOAT16；累加在 Cube 侧为 FP32，输出回写 FLOAT16 |

- 单一 FLOAT16 档（A/B/C 与 alpha/beta 全为 `aclblasHalf`），无 dtype 分派；compute 累加固定 FP32。
- 类型 `aclblasHalf` 仓内当前不存在，需在 `include/cann_ops_blas_common.h` 新增 `typedef uint16_t aclblasHalf;`（IEEE 754 binary16，2 字节，与 device 侧 `half` 逐位兼容）。host 侧 FP16↔FP32 转换复用 CANN `aclFloat16ToFloat` / `aclFloatToFloat16`。

### 3.1.3 支持形状

- m/n/k/batchCount 为运行时入参，编译期 shape 未知（不要求 dynamic shape）。
- m/n/k ≥ 0，batchCount ≥ 0；支持 m=0/n=0/k=0/batchCount=0 边界语义（见 §2.1.3）。
- 支持转置 N/T/C 全组合（transa×transb = 3×3 = 9 组）。
- 支持列主序前导维 padding（lda/ldb/ldc ≥ max(1, 对应行数)）。
- 不支持 broadcast（各 batch 的 A_i/B_i/C_i 为独立矩阵，无维度广播）。

## 3.2 算子实现

### 3.2.1 实现方案

本算子采用与同族 `aclblasCgemmBatched` 一致的「tensor_api 直接调用」形态（Blaze 三层抽象栈中的 Tensor API 层）：

```
┌─ 主计算（AIC，Cube）：tensor_api（AscendC::Te）─────────────────────┐
│  GM(A/B FP16) --CopyGM2L1--> L1(NZ/ZN) --CopyL12L0A/B--> L0A/L0B   │
│  --MmadAtom/Mmad(FP32 累加)--> L0C --CopyL0C2GM(fixpipe)--> GM      │
└─────────────────────────────────────────────────────────────────────┘
┌─ 后处理（AIV，Vector）：经典 AscendC LocalTensor 接口───────────────┐
│  temp(FP32 D) / C_orig(FP16) --DataCopyPad--> UB --Cast/Muls/Add--> │
│  FP32 epilogue --Cast(FP32→FP16)--> UB --DataCopyPad--> GM(C FP16)   │
└─────────────────────────────────────────────────────────────────────┘
```

两个 kernel（均 device 侧，非 host 计算）：
- **Kernel 1（GEMM，AIC-only，tensor_api）**：`op(A_i) × op(B_i) → D_i`（FP32 累加），写 temp FP32 或直接写 C（fast path）。
- **Kernel 2（alpha/beta epilogue，AIV-only，经典 AscendC）**：`C_i = alpha·D_i + beta·C_i`（FP32 计算 + 单次 FP32→FP16 舍入）。

#### 3.2.1.1 host 侧设计

##### （1）参数校验（含校验顺序）

校验顺序（任务书 §2.5）：**alpha/beta 判空与 Aarray/Barray/Carray 判空先于 m/n/batchCount 短路判定**；其中 Aarray/Barray/Carray 仅在 batchCount>0 时校验，故 m=0 或 n=0 且 batchCount>0 时数组指针仍不可为 nullptr，batchCount=0 时不校验数组指针。

| 校验项 | 异常码 |
|--------|--------|
| handle == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| transa / transb 非法枚举 | `ACLBLAS_STATUS_INVALID_ENUM` |
| alpha / beta == nullptr（判空先于短路） | `ACLBLAS_STATUS_INVALID_VALUE` |
| m / n / k < 0；batchCount < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda < max(1, 行数)、ldb < max(1, 行数)、ldc < max(1, m) | `ACLBLAS_STATUS_INVALID_VALUE` |
| batchCount > 0 且 Aarray/Barray/Carray == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |

##### （2）分核策略

- **总任务划分**：`totalTasks = batchCount × mBlocks × nBlocks`；taskId → `(batch, mBlock, nBlock)` 由 `GbResolveTaskCoordinates` 解析（batch 为最外维，空间分块为内维；奇偶 mBlock 蛇形反转 nBlock 均衡负载）。
- **AIC 核分配**：`usedAicCoreNum = min(cubeCoreNum, totalTasks)`；kernel 内 `for (taskId = GetBlockIdx(); taskId < totalTasks; taskId += GetBlockNum())` 轮转，每核处理多个 (batch, mBlock, nBlock) 任务。
- **AIV 核分配**（epilogue/early-exit）：按列均分，`totalCols = batchCount*n`，`startCol = blockIdx * ceil(totalCols/blockNum)`；`usedAivCoreNum = min(aivCoreNum, max(1, ceil(totalElements/16384)))`（每核 ≥16384 元素摊薄 launch 开销）。
- **负载均衡**：batch 为一级并行 + 空间分块填满剩余核；`CalcMnBlockSplit` 保证 `batchCount × spatialTasks` 尽量为 cubeCoreNum 的整倍数，消除尾部半波。

##### （3）数据分块与内存优化（Tiling 策略）

- **切分维度**：三维——batchCount（batch 间独立）+ M×N 空间分块（`mBlocks × nBlocks`）+ K 方向 tiling（`tileKChunk`）。
- **tile 大小**：`tileM=128`、`tileN=128`、`tileKChunk=128`。
- **硬件基形**：`baseM=16`、`baseN=16`（FRACTAL）；FP16 L0 `baseK=64`、`C0=16`、L0C `C0=16`（L0C cube 边长恒 16）。
- **K 方向 L1 迭代**：`kL1Iter = ceil(k / tileKChunk)`；L1 内再 `kSteps = ceil(curK / baseK)` 迭代进 L0A/L0B。
- **尾块处理**：M/N 尾块 `Min(tileM, mEnd-mOff)`；K 尾块 `Min(baseK, curK - kStep*baseK)`；`Slice` 自动夹断；`singleCoreM/N` 按 `CeilAlign(ceil(m/mBlocks), baseM)` 上取整后 clamp 到 m/n。
- **空间分块选择**：`CalcMnBlockSplit`——以 batch 为一级并行，`spatialCoreTarget = cubeCoreNum / gcd(cubeCoreNum, batchCount)`，枚举 `mBlocks×nBlocks ≤ spatialCoreTarget`，优先最大化 `util=mBlocks*nBlocks`，避免 batch 大时过度碎片化。
- **列主序适配**：`ApplyColumnMajorSwap`（m↔n、lda↔ldb、isTransA↔isTransB、singleCoreM↔N、mBlocks↔nBlocks）+ 交换 A/B 指针（详见 §3.2.1.2）。

**Buffer 规划**：

| Buffer | 位置 | 用途 | 大小（FP16） |
|--------|------|------|--------------|
| L1 A ping/pong | L1（512KB，2 bank） | GM→L1 的 A tile（NZ） | `RoundUp(curTileM,16) * RoundUp(curK,C0) * 2` 字节/半区 |
| L1 B ping/pong | L1 | GM→L1 的 B tile（ZN） | `RoundUp(curTileN,16) * RoundUp(curK,C0) * 2` 字节/半区 |
| L0A ping/pong | L0A（64KB） | L1→L0A 的 A 块 | `curTileM * curBaseK * 2`（≤ 半区 32KB） |
| L0B ping/pong | L0B（64KB） | L1→L0B 的 B 块 | `curBaseK * curTileN * 2`（≤ 半区 32KB） |
| L0C ping/pong | L0C（256KB） | MMAD 累加器（FP32） | `RoundUp(curTileM,16) * RoundUp(curTileN,16) * 4`（≤ 半区 128KB） |
| temp D（workspace） | GM | post-process 路径的 FP32 中间结果 | `batchCount * n * tempRowStride * 4` 字节（`tempRowStride=CeilAlign(m,16)`），仅 `needPostProcess` 时分配 |
| UB（AIV kernel） | UB | epilogue 各 `TQue`（各 2 深 ping-pong，tile 256 元素） | `GB_AB_TILE(=256)*sizeof*2 + 32` 对齐，各 Queue 独立 |

**Buffer 复用/扩满**：Cube 侧 L1/L0A/L0B/L0C 全部 ping-pong 双缓冲（`GEMM_BATCHED_L1_BUF_NUM=2`），K 方向多 tile 时 MTE1/MTE2/Cube 三级流水重叠；AIV 侧各 `TQue` 2 深，`DataCopyPad` 与 `Cast/Muls/Add` 流水；L1 内 A/B 同 bank 连续、A 后紧贴 B，避免 MTE1 bank 冲突。

**workspace 规划**：
- **fast path（alpha==1 且 beta==0）**：无 workspace。
- **post-process path**：`PrepareTempWorkspace` → `totalWorkspaceBytes = align64(batchCount*sizeof(void*)) + batchCount*n*tempRowStride*4`；`EnsureDefaultWorkspace`（库工作区 32MiB 起，倍增扩容，上限 2GiB）；`workspace[0..align64) = batchCount 个 temp 指针数组（H2D memcpy）`，`workspace[align64..) = FP32 temp D`；`tilingData.ldc = tempRowStride` 供 GEMM kernel 写 temp。
- **early-exit / k==0 / alpha==0**：无 temp，`LaunchEarlyExit` 直接 `C = beta*C`。

##### （4）tilingkey 规划

**本算子不使用 tilingkey**。direct launch 结构中 Tiling 计算在 host 实现文件内（`hgemm_batched_host.cpp` 的 `CalcGemmTiling` 等），无独立 tilingkey 引擎。transa/transb 的 4 种组合（NN/NT/TN/TT）在 host 侧由 `LaunchGemmFp16` 依 `isTransA/isTransB` 分派到 4 个独立 `__global__` 入口（`gemm_batched_gemm_kernel_fp16_{nn,nt,tn,tt}`）；其余参数（m/n/k/batchCount/ld*/tile 尺寸）全部随 `GemmBatchedGemmTilingData` 结构体传入，无需 key 查表。

host 侧 Tiling 计算逻辑：

```
CalcGemmTiling(m, n, k, batchCount, lda, ldb, ldc, isTransA, isTransB, cubeCoreNum, tiling):
    tiling.{m,n,k,lda,ldb,ldc,isTransA,isTransB,batchCount} = 入参
    tiling.tileM = 128; tiling.tileN = 128; tiling.tileKChunk = 128
    CalcMnBlockSplit(m, n, batchCount, cubeCoreNum, mBlocks, nBlocks)   # 空间分块
    tiling.singleCoreM = CeilAlign(ceil(m/mBlocks), 16) 后 clamp 到 m
    tiling.singleCoreN = CeilAlign(ceil(n/nBlocks), 16) 后 clamp 到 n
    tiling.totalTasks = batchCount * mBlocks * nBlocks   # overflow 检查 → INVALID_VALUE
    tiling.usedAicCoreNum = min(cubeCoreNum, totalTasks)
```

#### 3.2.1.2 kernel 侧设计

##### （1）Cube 实现路径选型

- **代码架构**：SIMD（实现载体：Cube）。主计算形态为 Matmul 类（GEMM），规则访存（列主序矩阵乘），目标芯片 ascend950（DAV_3510），主计算走 Cube 为最优匹配；不触发 Attention 类例外，不涉及通算融合（MC2）。
- **Cube 实现路径**：**Blaze / tensor_api 路线**（直接调用 `AscendC::Te` tensor_api）。适用性依据（来自 `ascendc-blaze-best-practice` 查询）：① Blaze skill 适用范围「Ascend 950 / DAV_3510 平台上基于 Blaze/tensor_api 开发 Batch 等 MatMul 类算子」；② FP16 dtype 支持（C0=16）；③ 转置 N/T/C 的 LayoutPtn 映射；④ Batch 维度 kernel 入口；⑤ 标准 alpha/beta epilogue 即基础 GEMM 语义，无 Vector 后处理，属 blaze_native 候选；⑥ 同族 `aclblasCgemmBatched`（`blas/gemm_batched/arch35/`）已采用 tensor_api 路线且 ops-tensor 依赖已接入构建。

**关键 API 族（Cube 侧，全部经 `ascendc-blaze-best-practice` 确认）**：

| API 族 | 用途 |
|--------|------|
| `MakeMemPtr<Location, T>(byteOffset)` | 构造硬件感知指针（GM/L1/L0A/L0B/L0C），offset 为字节 |
| `MakeFrameLayout<Pattern, Std::Int<C0>>(shape)` | 构造强类型 Layout（Pattern 决定编译期派发） |
| `MakeTensor(ptr, layout)` / `Slice(coord, shape)` | Tensor 构造与切块（自动夹断尾块） |
| `MakeCopy(CopyGM2L1{})` / `Copy(atom, dst, src)` | GM→L1（ND→NZ / DN→ZN 自动转换） |
| `MakeCopy(CopyL12L0A{})` / `CopyL12L0B{}` | L1→L0A/L0B（routing 按 L1 pattern 自动派发） |
| `MmadAtom<MmadTraits<MmadOperation>>{}.with(MmadParams)` / `Mmad(...)` | 矩阵乘累加（FP32 L0C），末拍 `unitFlag=FINAL_ACCUMULATION` |
| `MakeCopy(CopyL0C2GM{})` + `FixpipeParams{FINAL_ACCUMULATION}` | L0C→GM（fixpipe 一条龙，FP32→FP16 cast） |
| LayoutPtn：`NDExtLayoutPtn` / `DNExtLayoutPtn` / `NZLayoutPtn` / `ZNLayoutPtn` | GM 入口（列主序 trans 映射）与 L1/L0 分形 |
| `SetFlag` / `WaitFlag<HardEvent>` | 事件同步（tensor_api 不接管，调用方负责配对） |

**列主序 → LayoutPtn 映射**（`MakeGmLayout` + `ApplyColumnMajorSwap` 策略）：

| 操作数 | 存储（列主序） | trans | GM LayoutPtn | 说明 |
|--------|---------------|-------|--------------|------|
| A（m×k / k×m，lda） | 列主序 | false | `DNExtLayoutPtn(lda, k)` | stride(1, lda)：元素 A[i][p] 在 p·lda+i |
| A | 列主序 | true | `NDExtLayoutPtn(m, lda)` | stride(lda, 1)：op(A)[i][p]=A[p][i] 在 i·lda+p |
| B（k×n / n×k，ldb） | 列主序 | false | `DNExtLayoutPtn(ldb, n)` | 同 A 的非转置 |
| B | 列主序 | true | `NDExtLayoutPtn(k, ldb)` | 同 A 的转置 |
| C（m×n，ldc，输出） | 列主序 | — | `NDExtLayoutPtn(m, ldc)` | 经 `ApplyColumnMajorSwap` + 交换 A/B 后按行主写，与列主序 C 逐字节一致 |

**batch（指针数组）语义**：host 侧将 `Aarray/Barray/Carray`（Device 指针数组）原样传入 kernel（`__gm__ uint64_t* aPtrArr`），kernel 内 `GbResolveTaskCoordinates` 按 taskId 解析 `(batch, mBlock, nBlock)`，再 `aPtrArr[batch]` 取该 batch 矩阵首地址。无需在 host 侧物化中间矩阵。

##### （2）数据分块

- L1 双缓冲（ping-pong）半区偏移 = `bufId * (TOTAL_L1_SIZE >> 1)`，A/B 在同一 bank 内连续排布（`[Ping A][Ping B]`）；L0A/L0B/L0C 各自 ping-pong 半区 = `half * (TOTAL_L0x_SIZE >> 1)`（半区偏移是字节偏移）。
- FP16 分形常量：`C0=16`、`FRACTAL=16`、L0C cube 边长恒 16。
- 尾块：`curTileM/N = Min(tileM/N, mEnd/nEnd - mOff/nOff)`，`Slice` 自动夹断；K 方向 `baseK=64`，`curBaseK = Min(baseK, curK - kStep*baseK)`。

kernel 侧 tiling 参数消费：

```
kernel(gemm)：baseK = 64（FP16）；kL1Iter = ceil(k / tileKChunk)
  for taskId in [GetBlockIdx(), totalTasks, step GetBlockNum()]:
      (batch, mBlock, nBlock) = ResolveTaskCoordinates(taskId)
      (mStart,mEnd,nStart,nEnd) = ResolveTaskBounds(...)
      aGm = aPtrArr[batch]; bGm = bPtrArr[batch]; cGm = cPtrArr[batch]
      ProcessMNTiles(...): GbLoadL1Chunk（K 方向 tileKChunk 切）
        → GbProcessKL1Loop（L1 双缓冲）→ GbProcessL0Loop（baseK 切 L0）
        → Mmad（cmatrixInitVal/unitFlag 判定）→ CopyL0C2GM 写回（ping-pong L0C）
```

##### （3）FP16 epilogue

数据流（搬入 → 计算 → 搬出）：

```
【主计算 path（fast：alpha==1 且 beta==0）】
GM A/B (FP16) ─CopyGM2L1(ND→NZ / DN→ZN)─> L1 ─CopyL12L0A/B─> L0A/L0B
  ─Mmad(FP32 累加，K 迭代 unitFlag 累加/末拍 FINAL)─> L0C
  ─CopyL0C2GM(fixpipe，FP32→FP16 cast)─> GM C (FP16，原地覆写)

【post-process path（alpha!=1 或 beta!=0）】
Kernel1：GM A/B → … → L0C ─CopyL0C2GM─> GM temp D (FP32，workspace)
Kernel2：GM temp D (FP32) ─DataCopyPad─> UB，C_orig (FP16) ─DataCopyPad─> UB
  ─Cast(FP16→FP32,CAST_NONE)─> UB，Muls(alpha)、Muls(beta)、Add（全 FP32）─> UB
  ─Cast(FP32→FP16,RNE)─> UB ─DataCopyPad─> GM C (FP16)

【early-exit path（k==0 或 alpha==0）】
Kernel(early-exit AIV)：C_orig (FP16) ─DataCopyPad─> UB ─Cast(FP32)─> Muls(beta)
  ─Cast(FP32→FP16,RNE)─> UB ─DataCopyPad─> GM C
  （beta==1 时 host 直接返回 SUCCESS，不 launch）
```

**FP32→FP16 舍入语义（与 golden 对齐）**：golden = `cblas_sgemm` FP32 计算后 IEEE 754 就近舍入（RNE）到 FP16，device 须在 FP32 全链累加 + 单次 RNE 降精度。A/B FP16→FP32 在 MMAD 内精确（FP16⊂FP32），FP32 累加与 golden 同精度。**同族参考实现的 FP16 输出 cast 用 `CAST_ROUND`（away-zero），与 golden 的 RNE 语义不符**，本方案改 `CAST_RINT`（RNE）或 `CAST_EVEN` 作为 FP32→FP16 输出舍入；fast path 的 `CopyL0C2GM` 默认 `RoundMode::DEFAULT` 须穿刺确认为 RNE，两 cast 路径（fixpipe DEFAULT 与 AIV `CAST_RINT`）须 RNE 一致。

**性能优化项**（均面向计算/标量/搬入主 bound）：O1 三级 ping-pong 双缓冲流水重叠；O2 fast path 直接 fixpipe cast 写 C 跳过 temp 与第二 kernel；O3 列主序适配免显式转置；O4 AIV 核数摊薄 launch 开销；O5 空间分块去碎片化；O6 k==0/alpha==0 早期退化 early-exit kernel；O7 beta==0 时 epilogue 不读 C_orig（保证 beta=0 且 Carray=NaN 不传播）。

# 4 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（Ascend 950PR，arch35 / NpuArch `DAV_3510`） | √ |

CANN 版本：9.1.0（`/usr/local/Ascend/cann-9.1.0`）。

# 5 算子约束限制

| 约束项 | 内容 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 参数合法性 | m/n/k ≥ 0；batchCount ≥ 0；lda ≥ max(1, 行数)、ldb ≥ max(1, 行数)、ldc ≥ max(1, m)；alpha、beta 不可为 nullptr；batchCount > 0 时 Aarray/Barray/Carray 不可为 nullptr；非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，handle 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。校验顺序：alpha/beta 判空与 Aarray/Barray/Carray 判空先于 m/n/batchCount 短路判定；Aarray/Barray/Carray 仅在 batchCount > 0 时校验 |
| 非连续 Tensor 支持 | 不支持，各 batch 矩阵按列主序连续存储，仅支持 lda/ldb/ldc 前导维语义 |
| broadcast 规则 | 不支持，各 batch 的 A_i/B_i/C_i 为独立矩阵，无维度广播 |
| dynamic shape 要求 | 不要求，m/n/k/batchCount 为运行时入参，编译期 shape 未知 |
| 原地与视图语义 | Carray 各 batch 原地覆写，不返回视图 |
| 确定性计算要求 | 不要求 |
| 空 Tensor 与 0 维处理 | `m=0`、`n=0` 或 `batchCount=0` 为合法 no-op，返回成功且不执行计算；`k=0` 时逐 batch 仅做 beta 缩放 `C_i = beta × C_i`（`beta=0` 时置零），不引用 A、B，返回成功 |
| 内存要求 | 不涉及（任务书 §3.4「不涉及」；workspace 为 post-process 路径的内部实现机制，fast path 无 workspace） |

# 6 可维可测分析

## 6.1 精度标准/性能标准

### 6.1.1 精度标准

golden 由 FLOAT32 参考实现（Netlib `cblas_sgemm`，逐 batch 计算）得到后按 IEEE 754 就近舍入转换为 FLOAT16，再逐元素比对，输出各 batch 矩阵 C_i（m×n）分别全矩阵验证（仅逻辑区域，ldc padding 区域不参与比对）。输入输出为 FLOAT16，按半精度档判定：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
| -------- | ---- | ---- | ---------------------- | ------------------- |
| FLOAT16 | **2^-10** | **2^-10** | 0.99 | 1e-1 or 32 * ULP |

- 判定公式：逐元素通过条件 `|actual - golden| ≤ atol + rtol × |golden|`；当用例同时满足 `matched_ratio ≥ 0.99` 且 `max_abs_error ≤ max_abs_error_limit` 时，判定该用例精度通过。
- Inf/NaN 特殊值规则（不适用上述判定式）：golden 与 actual 同为 NaN、同为同号 Inf，或一方为 NaN 而另一方为 ±Inf，视为等效通过；一方为有限值、另一方为 Inf/NaN，判定不通过；非有限值元素不计入 matched_ratio 与 max_abs_error 统计，单独记录总数与通过数。用例通过条件 = 有限值满足 0.99 + max_abs_error ≤ 限，且非有限值全部按规则通过。
- `k` 过大导致 FLOAT16 累加溢出超差属非实现问题，须在自测报告举证（CPU golden 单标杆对比、最大误差随 k 增长曲线、累加精度档位），不得放宽阈值代替举证。
- 精度阈值以 **2^-10** 为准（约 9.765625e-4），随任务用例 CSV 全表 `mere_threshold=0.0009765625`、`mare_multiplier=32.0`。

### 6.1.2 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 性能标准 | FLOAT16 场景平均单次耗时（Avg time，us），单 case 倍率 ≥ 0.4× GPU 且全部 case 倍率算术平均 ≥ 0.5× GPU（两项须同时满足） | 任务书 §3.3；`gpu_baseline.csv`（200 条，`gpu_ms` 0.004438~1.49708 ms） |

- 测试设备：Ascend 950PR。性能数据为 FLOAT16 输入场景下的平均单次耗时。
- 采样方式：msprof op 模式采集单算子 task duration（us）作为单次耗时；msprof op 自带 **5 次 warmup（不计入样本）**，warmup 后连续执行 **10 次作为 10 个有效样本，取算术平均**作为 Avg time；算子返回 SUCCESS 方计入有效样本，异常样本不计入并须在自测报告中说明。
- 判据：① 单 case 倍率（`GPU 标杆耗时 / NPU 实测耗时`）≥ 0.4× GPU（等价 Avg time ≤ 2.5× GPU 标杆）；② 全部性能 case 倍率算术平均 ≥ 0.5× GPU。两项须同时满足。

### 6.1.3 测试方案（分级用例）

> 评测来源模式 A：分级用例 = 评测集 cases 核对 + 补充。1200 条功能用例（`hgemm_batched_test.csv`，含 1176 SUCCESS + 24 反向）由任务方下发，映射到 L0/L1/L2 并核对覆盖；200 条 TC_PF 性能用例单列性能口径。

| 分级 | 用例数 | 说明 |
|------|--------|------|
| L0 门槛 | 18 | TC_L0（transa×transb 全组合 3×3=9 × 小尺寸 4³/8³，batchCount=2） |
| L1 功能 | 945 | TC_MN(47，尺寸扫描 1→2048 + 矩形) + TC_AB(10，标量组合) + TC_LD(15，前导维 padding) + TC_FL(18，填充值域) + TC_CV(21，batchCount 扫描) + TC_EX(834，确定性随机采样) |
| L2 异常 | 37 | TC_ED（13 正向边界 + 24 反向异常） |
| 性能（单列） | 200 | TC_PF（不参与精度比对） |
| **功能合计** | **1000** | = 18 + 945 + 37（其中 SUCCESS 976 + 反向 24） |
| **总计** | **1200** | 功能 1000 + 性能 200 |

**24 条反向用例判定口径（期望错误码比对）**：

| 期望错误码 | 条数 | 用例 |
|-----------|------|------|
| `ACLBLAS_STATUS_INVALID_VALUE` | 21 | null_alpha、null_beta、null_alpha_batch0、null_beta_m0、null_Aarray、null_Barray、null_Carray_beta1、m0_null_arrays、n0_null_arrays、k0_beta0_Carray_null、k0_beta1_Carray_null、alpha0_null_Aarray、invalid_lda_N、invalid_lda_T、invalid_ldb_N、invalid_ldb_T、invalid_ldc、neg_m、neg_n、neg_k、neg_batchCount |
| `ACLBLAS_STATUS_INVALID_ENUM` | 2 | invalid_transa（999）、invalid_transb（999） |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 1 | handle_null |

反向用例对 `expect_result != SUCCESS` 仅调用一次 `aclblasHgemmBatched` 并 `EXPECT_EQ(返回值, expectResult)`，不进入 golden 生成与精度比对。

**测试场景覆盖**（§3.5 自验要求）：transa/transb 全组合、尺寸扫描（m/n/k）、不同 batchCount（含 1 与多批）、标量特殊值（alpha/beta 取 0、±1、大值）、前导维 padding、边界与负向用例（m=0/n=0/batchCount=0 no-op 期望 SUCCESS；k=0 期望 SUCCESS 且 `C_i=beta×C_i`；alpha/beta 空指针期望 INVALID_VALUE；batchCount>0 时数组空指针期望 INVALID_VALUE；batchCount=0 不校验数组指针；非法枚举/非法前导维/负维度/batchCount<0 对应错误码）、Inf/NaN 特殊值用例，以及性能用例。

**测试工程落点（落位说明）**：测试代码放 `test/gemm_batched/hgemm_batched/arch35/`（含 csv），与同族 `test/gemm_batched/cgemm_batched/` 完全同构。

该落点与任务书 §5 字面所列 `test/gemm/hgemmBatched/arch35/` 存在差异，原因如下，均为可复核的技术约束而非风格取舍：

1. 随任务下发的 `verify_accuracy.py` 将算子名固定为 `OP = "hgemm_batched"`，并以 `bash build.sh --soc=<soc> --ops=hgemm_batched` 触发编译（`verify_accuracy.py` L22、L41-42）。
2. 编译系统对 `--ops` 值采用**目录名精确匹配**：先判 `EXISTS test/<name>`，再遍历家族子目录判 `EXISTS test/<family>/<name>`（`test/CMakeLists.txt` L20-31）。Linux 文件系统大小写敏感，`hgemmBatched` 与 `hgemm_batched` 不匹配。
3. 匹配失败时直接 `FATAL_ERROR` 中止配置（`test/CMakeLists.txt` L40-44），即在编译阶段即失败，而非仅在验收阶段找不到产物。
4. 该脚本 `install_csv()` 还会将用例 CSV 写入 `test/gemm_batched/hgemm_batched/arch35/hgemm_batched_test.csv`（L31-38），`find_binary()` 的 4 个候选路径亦全部位于 `build/test/gemm_batched/hgemm_batched/`（L24-29）。

因此任务书 §5 的驼峰路径在本仓编译系统下不可实现；为不改动官方 `test/CMakeLists.txt`（侵入编译系统，超出算子任务范围）、亦不改动官方验收脚本（任务明确禁止），测试工程按上述脚本路径落地，可零适配命中编译目标与全部产物路径。实现代码仍严格按任务书 §5 落 `blas/gemm/arch35/`，文件名以 `hgemm_batched_` 为前缀以满足 `cmake/test.cmake` L42 的源码探测规则。

本项差异已显式披露，供评审确认；如需调整，仅需移动测试目录，实现代码与设计结论不受影响。

## 6.2 兼容性分析

- 新算子，接口声明放入 `include/cann_ops_blas.h`，与既有 `aclblasSgemmBatched`/`aclblasCgemmBatched` 同头共用、风格一致，与其他产品线共用同一 `aclblasHgemmBatched` API，**禁止定义 950PR 私有平行接口**。
- 参数序列与对标接口 cuBLAS `cublasHgemmBatched` 一致（handle 及参数顺序一一对应），无需额外映射说明。
- README 产品支持表标注 Ascend 950PR：支持。
- 无既有历史版本（非 TBE 迁移），不涉及历史算子版本兼容性分析。
