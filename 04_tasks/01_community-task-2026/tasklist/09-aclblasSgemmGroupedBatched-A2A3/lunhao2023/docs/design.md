# 需求背景（required）

## 需求来源

9 月社区任务《aclblasSgemmGroupedBatched 算子开发(A2A3)》：在昇腾 NPU（Atlas 800I A2 / Atlas 800I A3，arch22）上基于 Ascend C 开发单精度实数（FLOAT32）分组批量矩阵乘算子 `aclblasSgemmGroupedBatched`，完成算子设计、开发、测试全流程。

- 任务书：`aclblasSgemmGroupedBatched_A2A3_task_doc.md`（本文唯一语义真值源；配套 `test_cases/README.md` 及 CSV 若与任务书冲突，以任务书为准）
- 设计文档提交仓：`cann-ops-competitions`（任务书中的 `cann-competitions` 为其别名）
- 提交路径：`04_tasks/01_community-task-2026/tasklist/09-aclblasSgemmGroupedBatched-A2A3/lunhao2023/docs/design.md`
- 代码合入仓：`https://gitcode.com/cann/ops-blas`（基线 master `96302d5d`）
  - 实现：`blas/gemm_grouped_batched/arch22/`
  - 测试：`test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`
- 对标接口：cuBLAS `cublasSgemmGroupedBatched`（CUDA ≥ 12）；精度 golden 为 Netlib `cblas_sgemm` 逐 batch
- 工程模式：kernel 直调（句柄式 BLAS），不走 ACLNN / tilingKey 注册

## 背景介绍

### aclblasSgemmGroupedBatched 算子实现

在 ops-blas 开源仓框架下，用 Ascend C 在 Atlas A2/A3（架构目录 `arch22`，`ArchType::ASCEND_V220`）实现 FLOAT32 分组批量矩阵乘：多组形状、转置、scale、批量大小完全独立的矩阵乘打包成一次组计算。对第 g 组（`g = 0..groupCount-1`）内第 j 个 batch（扁平下标 `idx = Σ_{t<g} groupSize[t] + j`）：

```
C[idx] = alpha_g * op(A[idx]) * op(B[idx]) + beta_g * C[idx]
op(X)  : ACLBLAS_OP_N → X ，ACLBLAS_OP_T → Xᵀ
```

所有矩阵列主序；`Aarray`/`Barray`/`Carray` 为 Host 侧扁平指针数组（长度 `Σ groupSize[g]`，元素指向 Device 矩阵）；cuBLAS `cublasSgemmGroupedBatched` 无 stride 语义。

### 同类算子现状分析

ops-blas master `96302d5d` 相关基线：

| 算子 / 目录 | 数据类型 | 仓内现状 | A2/A3 |
| --- | --- | --- | --- |
| `aclblasSgemmGroupedBatched`（`blas/gemm_grouped_batched/arch35/`） | FLOAT32 | host(712 行) + kernel(567 行) + tiling(80 行)；纯 AIV 向量 GEMM，"一个 batch 一个核" | 不支持 |
| `blas/gemm_grouped_batched/` | — | 仅 `arch35/`，无 `arch22/` | — |
| `test/gemm_grouped_batched/` | — | 共享 `gemm_grouped_batched_param.h` / `_golden.h` / `arch35/`（GTest + CSV） | — |
| `blas/syrk/arch22/`、`blas/symm/arch22/` | FLOAT32 | arch22 Cube 先例：静态 tiling Matmul API + AIV 前后处理 + workspace chunk | 支持 |
| `blas/trsmbatched/arch22/` | COMPLEX64 | arch22 batch 指针数组先例：host 扁平化指针数组、kernel 按 batchIdx 解引用 | 支持 |

结论：

1. 接口声明 `include/cann_ops_blas.h:462` 已存在（arch35 引入），本任务**无需改动公共头**；README 产品支持表需将 Atlas A2/A3 更新为支持。
2. arch35 是语义蓝本而非实现蓝本：其 host 校验顺序、组语义、no-op 分支、指针数组 flatten/H2D、`GroupParam` 用 GM 地址承载的 tiling 设计可复用；但 kernel 是纯向量实现且按 batch 整体分核，arch22 性能标杆（§3.3 case 5 等效吞吐约 29 TFLOPS）超出 AIV FP32 能力，计算主体必须落到 AIC Cube。
3. arch22 FP32 三级算子已验证「静态 tiling Matmul + workspace chunk + AIV epilogue」路线（`complex_blas3_arch22.h` / `csyrk` / `ssymm`），本任务沿用该仓内正统路线。

### aclblasSgemmGroupedBatched 算子功能分析

- **功能**：对多组相互独立、组内批量大小一致的 FLOAT32 矩阵乘做分组批量计算，结果写入各组 `C`。
- **输入**：handle、groupCount、逐组 transa/transb/m/n/k/alpha/lda/ldb/beta/ldc/groupSize 数组（均 Host 内存）、扁平指针数组 Aarray/Barray/Carray。
- **输出**：各组 `C[idx] = alpha_g·op(A[idx])·op(B[idx]) + beta_g·C[idx]`；`C` 与 `A`/`B` 不允许内存重叠，不返回视图。
- **支持数据类型**：仅 FLOAT32。
- **支持广播**：不涉及。无 dynamic shape 要求（各维为运行时 Host 标量，由 host tiling 支持）。
- **边界语义**（任务书 §2.1/§2.5）：`m_g=0`、`n_g=0` 或 `groupSize_g=0` 为该组合法 no-op；`k_g=0` 等效 `C=beta_g·C`；`alpha_g=0` 跳过矩阵乘仅执行 `C=beta_g·C`。
- **异步**：依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步。
- **确定性**：不要求（浮点累加顺序不保证逐位一致）。

# 需求分析（required）

## 需求描述

使用 Ascend C 在 ops-blas 的 `arch22` 路径实现 `aclblasSgemmGroupedBatched`，接口签名与仓内既有声明完全一致，逐组数组语义与 cuBLAS `cublasSgemmGroupedBatched` 对齐（仅 groupCount 参数位置、命名风格 2 处非语义差异），覆盖 NN/NT/TN/TT 四种转置组合、任意分组形状，精度达到生态 FLOAT32 混合容差标准，性能不超过任务书 §3.3 达标耗时。

## 需求拆解

1. 实现放 `blas/gemm_grouped_batched/arch22/`，文件名前缀 `gemm_grouped_batched_`，由顶层 CMake 按 SOC_ARCH_DIRS=arch22 编入；README 产品支持表更新 Atlas A2/A3 为支持。
2. Host 参数校验与错误码严格按任务书 §2.4/§2.5：非法枚举（实数无共轭语义，`ACLBLAS_OP_C` 判非法）返回 `ACLBLAS_STATUS_INVALID_VALUE`（任务书口径；arch35 现返回 `INVALID_ENUM`，以任务书为准，配套 CSV 同步修正）。
3. Kernel 直调：handle 取 stream 与 workspace；Aarray/Barray/Carray 为 Host 侧指针数组，Host 侧完成逐元素非空校验后扁平化为 Device 数组供 kernel 解引用。
4. 计算主体走 AIC Cube：静态 tiling Matmul API，AIV 做 epilogue；覆盖小 shape 的 AIV 直算路径。
5. 边界全覆盖：no-op 组、`k=0`、`alpha=0`、lda/ldb/ldc padding、负维/空指针/非法枚举负向用例。
6. 精度：cblas `sgemm` 逐 batch 作 golden，FLOAT32 混合容差。
7. 性能：Atlas 800T A2 (910B3) 上 5 条标杆 Avg time 不超过任务书 §3.3 表。
8. 测试：`test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`，CSV 驱动 GTest；以任务配套 `sgemm_grouped_batched_test.csv` 为底，冲突项按任务书修正。

# 详细设计（required）

## 算子分析

### 数学公式

记扁平下标 `idx = Σ_{t<g} groupSize[t] + j`，组内形状（列主序）：

| 操作数 | transa/transb=N | transa/transb=T |
| --- | --- | --- |
| `A[idx]` 存储形状 | `m_g × k_g`（lda ≥ max(1,m)） | `k_g × m_g`（lda ≥ max(1,k)） |
| `B[idx]` 存储形状 | `k_g × n_g`（ldb ≥ max(1,k)） | `n_g × k_g`（ldb ≥ max(1,n)） |
| `C[idx]` 存储形状 | `m_g × n_g`（ldc ≥ max(1,m)） | 同左，与 trans 无关 |

```
C[idx] = alpha_g * op(A[idx]) * op(B[idx]) + beta_g * C[idx]
```

`op(A)` 为转置时，按「转置视图」读取：物理上仍按列主序从 `A[idx]` 取数，逻辑行列互换，不需要显式转置拷贝。

边界：

- `m_g=0 / n_g=0 / groupSize_g=0`：该组 no-op，不产生任何访存与工作项。
- `k_g=0`：等效 `C = beta_g·C`。
- `alpha_g=0`：跳过矩阵乘（不读 A/B），等效 `C = beta_g·C`。
- `Inf/NaN`：不做清洗，按浮点运算自然传播。

### 支持数据类型

仅 FLOAT32。按生态精度标准 FLOAT32 档混合容差判定。

### 支持形状

- `groupCount ≥ 0`；各组 `m/n/k ≥ 0`、`groupSize ≥ 0`
- 前导维约束：`lda ≥ max(1, transa=N ? m : k)`；`ldb ≥ max(1, transb=N ? k : n)`；`ldc ≥ max(1, m)`
- 不支持超出 lda/ldb/ldc 语义的非连续访问，不做 broadcast
- 组间形状、scale、batch 任意独立；`Σ groupSize[g]` 受指针数组长度与 Device 内存约束

## 算子实现

### 实现方案

工程文件（任务书 §5 + ops-blas 目录规范）：

```
blas/gemm_grouped_batched/arch22/
├── gemm_grouped_batched_host.cpp          # 校验、组语义展开、tiling、workspace、launch
├── gemm_grouped_batched_kernel.cpp        # AIC Cube GEMM + AIV epilogue / 小 shape 直算
├── gemm_grouped_batched_kernel.h          # kernel_do launcher 声明
└── gemm_grouped_batched_tiling_data.h     # Host/Device 共享 POD（GroupParam + TilingData）

test/gemm_grouped_batched/sgemm_grouped_batched/arch22/
├── gemm_grouped_batched_npu_wrapper.h
├── gemm_grouped_batched_test.cpp
└── gemm_grouped_batched_test.csv
```

共享的 `gemm_grouped_batched_param.h` / `gemm_grouped_batched_golden.h` 复用 `test/gemm_grouped_batched/` 既有文件，不重复造轮子。

#### 3.2.1 host 侧设计

入口 `aclblasSgemmGroupedBatched` 流程：

```
1. handle == nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2. 校验 groupCount / 各组枚举与维度 / alpha·beta·指针数组非空
3. groupCount == 0 或 Σ groupSize == 0 → SUCCESS（不发射 kernel）
4. 跳过 no-op 组；逐组在 Host 完成：
   a. 路径选择（Cube / AIV 直算 / 仅 beta 缩放）
   b. Cube tile 选择与 work item 展开
5. 扁平化 A/B/C 指针数组并逐元素判空 → H2D 拷入 workspace
6. 组装 GroupParam / TilingData → H2D
7. 按 launch 轮次在 handle->stream 上发射 AIC / AIV kernel，异步返回
```

**参数校验（任务书 §2.4，全部在计算前完成）：**

| 条件 | 返回码 |
| --- | --- |
| handle 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| groupCount < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| transaArray[g]/transbArray[g] ∉ {OP_N, OP_T}（含 OP_C） | `ACLBLAS_STATUS_INVALID_VALUE` |
| mArray[g]/nArray[g]/kArray[g]/groupSizeArray[g] < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| alphaArray/betaArray/m/n/k/lda/ldb/ldc/groupSizeArray/Aarray/Barray/Carray 任一为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| Aarray/Barray/Carray 中任一元素为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda/ldb/ldc 不满足前导维约束 | `ACLBLAS_STATUS_INVALID_VALUE` |

说明：Aarray/Barray/Carray 是 Host 侧数组，逐元素判空在 Host 内存上完成，无 D2H 开销（较 arch35 增强，对齐任务书「其中某地址为 nullptr 返回 INVALID_VALUE」）。「不可传入 Device 侧指针数组」为文档化契约，不做运行时探测（同 arch35 注释口径）。

**路径选择（逐组独立）：**

| 路径 | 条件 | 行为 |
| --- | --- | --- |
| no-op | m=0 或 n=0 或 groupSize=0 | 不产生工作项 |
| 仅 beta | k=0 或 alpha=0 | AIV 缩放 kernel：`C ← beta·C` |
| AIV 直算 | `m×n×k < AIV_SMALL_THRESHOLD`（初值 32³）或 m、n 均小于最小 Cube tile（32） | 单 batch 整体在 AIV 上按 arch35 的列行点积方式直算，省 AIC 发射开销 |
| Cube | 其余（覆盖 §3.3 全部性能 case） | AIC Matmul + AIV epilogue |

**Cube 分块与 work item：**

- tile 档位沿用仓内 `complex_blas3_arch22.h` 单一来源常量：`TILE_M × TILE_N = 128×128`，`BASE_K = 64`（L0 双缓冲上限）。128×128 FP32 累加器占 L0C 64KB ≤ 128KB，整个 k 维可在 Matmul 内部循环完成累加，无需 K 向 workspace 切块。
- 逐组选档：host 按 m/n 从 {128×128, 64×64, 32×32} 就小选择（tile 不超过该组 m/n 的 2 倍），小形状避免整块 128×128 浪费算力；tileK 恒为 64。
- work item = `(group, batchInGroup, tileRow, tileCol)`，总数 `Σ_g ceil(m_g/tileM)·ceil(n_g/tileN)·groupSize_g`。Host 将 item 依组内 tile 数降序排列（大组在前）后 round-robin 均分到 AIC，余数给前若干核；同组形状一致天然负载均衡，异组场景靠降序排列抑制长尾。
- `blockIdx → (item)` 映射写在 tiling：`usedBlocks`、`itemsPerBlock`、`itemTail`，kernel 内 for 循环顺序消费，超额 round-robin。

**指针数组与 workspace 布局：**

```
workspace（handle 托管，不足则提示 aclblasSetWorkspace）:
┌────────────────────────────┬───────────┐
│ TilingData                 │ 256B 对齐 │
│ GroupParam[groupCount]     │           │
│ aPtrArray[totalBatch]      │ uint64    │
│ bPtrArray[totalBatch]      │ uint64    │
│ cPtrArray[totalBatch]      │ uint64    │
│ gemmTemp（环形 tile 缓冲）   │           │
└────────────────────────────┴───────────┘
```

- `gemmTemp` 大小 = `itemsPerRound × tileM·tileN·4B`；每轮 `itemsPerRound = min(totalItems, usedBlocks × ITEMS_PER_BLOCK)`（初值 ITEMS_PER_BLOCK=4，保证 AIC 单次 launch 摊薄发射开销，同时 temp ≤ usedBlocks×4×64KB ≈ 5MB 量级）。AIC 与 AIV 在同一 stream 上串行，轮间 temp 安全复用。
- Host launch 前完成容量检查，不足返回 `ACLBLAS_STATUS_EXECUTION_FAILED` 并提示所需字节数（同 arch35 口径）。

**tiling 结构（无 tilingKey，pathId + 标志位）：**

```cpp
struct GroupParam {                       // 逐组，GM 承载（同 arch35 设计）
    uint32_t transa, transb;              // 0=N, 1=T
    uint32_t m, n, k;
    int32_t  lda, ldb, ldc;
    uint32_t groupSize, groupOffset;      // 组内批量数 / 扁平 batch 起始下标
    float    alpha, beta;
    uint32_t tileM, tileN, tileK;         // 本组 Cube 档位
    uint8_t  pathId;                      // 0=beta-only, 1=aiv-small, 2=cube
};
struct GemmGroupedBatchedTilingData {
    uint64_t groupParamsGmAddr, aPtrArrayGmAddr, bPtrArrayGmAddr, cPtrArrayGmAddr, gemmTempGmAddr;
    uint32_t groupCount, totalBatchCount;
    uint32_t usedBlocks, itemsPerBlock, itemTail;   // work item 分核
    uint32_t itemsPerRound, totalItems;             // launch 轮次
    uint32_t aivCoreNum;                            // AIV epilogue / 小 shape 分核
};
```

字段 Host/Device 字节布局一致（`#pragma pack(4)`，同 arch35）。

**核数获取**：`GetAicCoreCount()` / AIV 核数经平台信息获取，为 0 时报 `ACLBLAS_STATUS_INTERNAL_ERROR`（同 `csyrk_host.cpp:152` 口径）。

#### 3.2.2 kernel 侧设计

**1. AIC Cube GEMM kernel（pathId=2，每轮一次 launch）：**

- 入参：tilingGm、groupParamsGm、三个指针数组 Gm、gemmTempGm。
- 每个 AIC block 依 tiling 领取自己的 item 区间，逐 item：
  1. 由 item 反解 `(g, batchInGroup, tileRow, tileCol)`，读 `GroupParam[g]` 与 `aPtrArray/bPtrArray[groupOffset+batchInGroup]`；
  2. `MatmulImpl` 静态 tiling（NN/NT/TN/TT 四个模板特化，`GetMatmulApiTiling` 编译期生成，同 `complex_blas3_arch22.h` 三种 MmTrans/MmPlain 组合的用法），A/B 直接从 GM 读取（FP32 无需预打包，无 Phase0），Matmul 内部完成 GM→L1→L0A/L0B 与 k 维累加；
  3. 结果 tile 由 L0C fixpipe 写 `gemmTemp[itemSlot]`（`itemSlot = item - roundBase`）。
- 无核间数据依赖：不同 item 读写不同 temp 槽位，AIC 核间无同步需求。

**2. AIV epilogue kernel（每轮一次 launch，AIC 之后）：**

逐 item 将 `C_tile = alpha·gemmTemp[slot] + beta·C_old_tile` 写回 `cPtrArray[idx]` 对应 tile：

- `beta=0`（`isBetaZero` 标志）：跳过 C_old 读，仅写 `alpha·temp`；
- `alpha=0` 不会进入该路径（host 已分流至仅 beta 路径）；
- `DataCopyPad` 处理 ldc padding，只写逻辑 m 行/n 列；读 C_old 与写 C 按列分块流水，tile 内列间无依赖可并行；
- alpha/beta 缩放用 `Muls` 完成（同 `csyrk` combine kernel 的标量折叠写法），避免构造乘法向量。

**3. AIV 仅 beta kernel（pathId=0）：**

工作项 `(batch, 列块)` round-robin 到 AIV 核，读 `C[idx]` 列、`Muls(beta)`、原位写回。覆盖 `k=0` 与 `alpha=0` 两种组语义，不读 A/B。

**4. AIV 小 shape 直算 kernel（pathId=1）：**

单 batch 一个工作项，按 arch35 kernel 的列行点积方式在 UB 内完成 `C = alpha·op(A)·op(B) + beta·C`：整矩阵或行列分块搬入 UB，`Muls` 先行折叠 alpha 到 A 或 B 面板（同 `ssymm` pack 折叠写法），beta 直接作用于 C 初值，向量指令完成累加。仅用于 `AIV_SMALL_THRESHOLD` 以下形状，规避 AIC 发射开销。

**5. 同步与发射序列：**

同一 stream 顺序发射，轮次循环：

```
for round in 0..ceil(totalItems/itemsPerRound)-1:
    gemm_kernel_do(..., roundBase, roundItems, blocks=usedBlocks, stream)   # AIC
    combine_kernel_do(..., roundBase, roundItems, blocks=aivCoreNum, stream) # AIV
```

AIC 与 AIV 天然由 stream 串行化，无需核间 flag；pathId=0/1 的组在第一轮前单独发射对应 kernel。混合分组场景按「仅 beta 组 → 小 shape 组 → Cube 轮次」顺序发射，组间无数据依赖，顺序仅影响时间不影响正确性。

**6. Inf/NaN：** 不做额外清洗；负向与特殊值用例按任务书 §3.5 允许场景比对。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2（910B / arch22） | √ |
| Atlas A3 训练/推理系列（arch22） | √ |
| Ascend 950PR/DT（arch35） | 已有实现，本任务不改动 |

自验证覆盖一种款型即可；精度建议 910B3 或 A3，性能建议 910B3。

## 算子约束限制

- 仅 FLOAT32，列主序，扁平指针数组逐 batch 独立寻址，无 stride/broadcast 语义
- `C` 与 `A`/`B` 不允许内存重叠；不返回视图；Aarray/Barray/Carray 为 Host 侧指针数组，不可传入 Device 侧指针数组（文档化契约，不做运行时探测）
- 非法枚举（含 `ACLBLAS_OP_C`）按任务书返回 `ACLBLAS_STATUS_INVALID_VALUE`；配套 CSV 若期望 `INVALID_ENUM` 以任务书为准修正
- `m/n/groupSize = 0` 为合法 no-op；`k=0`、`alpha=0` 等效 `C = beta·C`
- 确定性计算不要求
- CANN 9.1.0

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden = 逐 batch `cblas_sgemm`；比对各组 C 逻辑 m×n 全矩阵。FLOAT32：rtol=atol=`2^-13`≈1.22e-4，`matched_ratio≥0.99`，`max_abs_error ≤ 1e-2 或 32*ULP`。逐元素 `\|actual-golden\| ≤ atol + rtol×\|golden\|`；含大数规约时 max_abs 可按任务书说明放宽至 2ULP | 任务书 §3.2；[混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 设备 Atlas 800T A2 (910B3)。FLOAT32，warmup 后有效采样 >10 次取 Avg time (us)，不超过下表 | 任务书 §3.3 |
| 内存标准 | 不涉及（workspace 用量随调用输出，无泄漏路径） | 任务书 §3.4 |

性能标杆（与配套 `gpu_baseline.csv` 五条 case 一致）：

| case | groupCount | groupSizeArray | m/n/k | transa/transb | 达标耗时 (us) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | [64,64] | 256×256×256 | N/N | 351.3 |
| 2 | 2 | [128,64] | 512×512×512 | N/N | 3714 |
| 3 | 3 | [64,128,64] | 1024×1024×1024 | N/T | 37652 |
| 4 | 2 | [128,128] | 2048×2048×2048 | T/N | 298380 |
| 5 | 2 | [64,64] | 4096×4096×4096 | N/N | 1188440 |

五条 case 全部走 Cube 路径（m/n ≥ 128 档位或 64 档位），无 AIV 直算介入。

测试工程：CSV + C++ GTest（`test/gemm_grouped_batched/sgemm_grouped_batched/arch22/`），复用仓内 csv_loader/fill/verify 测试框架，golden 在 Host 调 cblas。用例覆盖（任务书 §3.5）：

- 小 shape 基础（配套 `TC_L0_*`，NN/NT/TN/TT 转置正交）
- shape 扫描：小质数、2 的幂及 ±1、非对齐、直至大规模；lda/ldb/ldc 多档 padding
- 边界与负向：groupCount/m/n/k/groupSize 负值、各数组 nullptr、指针数组元素 nullptr、非法枚举（OP_C）、零维 no-op、`k=0`、`alpha=0`
- 特殊值：Inf/NaN 传播用例；alpha/beta 取 0/1/-1
- 性能/内存用例：上述 5 条标杆 + `gpu_baseline.csv` 参考 case，msprof 采样 Avg time

负向用例只查返回码，不比数值。

## 兼容性分析

新增 `arch22/` 实现与既有 `arch35/` 共存：同一接口声明按 SOC 编译期选择架构目录，不改实数/950 行为；公共头 `include/cann_ops_blas.h` 无改动；`blas/gemm_grouped_batched/README.md` 产品支持表追加 Atlas A2/A3「支持」。测试新增 `sgemm_grouped_batched/arch22/` 子目录，共享 param/golden 文件只复用不修改。
