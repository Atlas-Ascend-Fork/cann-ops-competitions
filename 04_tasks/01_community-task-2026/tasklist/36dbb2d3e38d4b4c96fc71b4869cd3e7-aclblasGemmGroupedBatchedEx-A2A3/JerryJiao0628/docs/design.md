# aclblasGemmGroupedBatchedEx 算子设计文档

> 昇腾社区 2026 年 10 月社区任务 · `aclblasGemmGroupedBatchedEx`（Atlas A2/A3 补齐）
> 任务 ID：`36dbb2d3e38d4b4c96fc71b4869cd3e7`
> 目标仓：`https://gitcode.com/cann/ops-blas`（分支 `feature/aclblasGemmGroupedBatchedEx-arch22`）
> 文档模板：`cann-competitions/04_tasks/01_community-task-2026/resources/design_template.md`

---

# 需求背景（required）

## 需求来源

昇腾社区「10 月社区任务」发布的 `aclblasGemmGroupedBatchedEx` 算子开发任务书（Atlas A2/A3 系列）。任务要求在
Ascend C 编程语言下，于 ops-blas 开源仓中补齐 `aclblasGemmGroupedBatchedEx` 这一 cuBLAS 风格
**分组批量矩阵乘**接口的 **Atlas A2 / A3** 实现，并完成设计、开发、测试全流程，最终以 PR 形式合入
`cann/ops-blas`。

与同批的 `aclblasLtMatmul` 相比，本接口的语义面更宽：**一次调用内既有分组（group）维度、又有组内批数
（batch）维度，且每个分组可以有不同的转置组合、不同的 alpha/beta、不同的形状与行前导维**。这使得
tiling、分核、kernel 特化都要按「组」而不是按「调用」来组织。

## 背景介绍

### cuBLAS `cublasGemmGroupedBatchedEx` 语义对齐

`cublasGemmGroupedBatchedEx` 计算

```
C[i] = alpha[i] * op(A[i]) * op(B[i]) + beta[i] * C[i]
```

其中 `i` 取遍**所有**矩阵（problem）。矩阵被组织成 `groupCount` 个组，每组有 `groupSize[i]` 个矩阵。
所有 shape 参数（`mArray/nArray/kArray/ldaArray/ldbArray/ldcArray/alphaArray/betaArray/
transaArray/transbArray`）都是**逐组**给出的，即**同组内所有矩阵共享同一组参数**；不同组之间参数可以
不同（例如第 0 组 `transa=N`、第 1 组 `transa=T`）。

本任务要求逐项对齐该语义：接口签名与 `include/cann_ops_blas.h` 中既有声明一致，**与 950PR 共用同一声明，
不新增产品私有平行 API**。

### ops-blas 仓现状

| 路径 | 现状 |
| --- | --- |
| `include/cann_ops_blas.h` | `aclblasGemmGroupedBatchedEx` 已有声明，签名与任务书一致，**与 950PR 共用，不得重复定义** |
| `blas/gemm_grouped_batched_ex/arch35/` | 950（Ascend 350 / A5）的实现**已存在**：手写 Mmad 内核 + epilogue，支持 FP16/BF16/FP8，**不支持 FP32** |
| `blas/gemm_grouped_batched_ex/arch22/` | **缺失** —— 本任务的主要缺口 |
| `test/gemm_grouped_batched_ex/arch35/` | 950 的测试工程与 CSV 已存在（列名为蛇形） |
| `test/gemm_grouped_batched_ex/arch22/` | **缺失** —— 本任务的主要缺口 |

结论：**接口层不需要重写，需要补齐的是 A2/A3 侧的设备端实现 + 一套能直接吃随包官方 CSV 的测试工程**。

### 为什么 arch22 不能照抄 arch35

1. **950 的内核是手写 Mmad**（`AscendC::Mmad` + ND2NZ + `LoadData2DParamsV2`），这些是 350 专有 API，
   在 A2/910B 上不可用。
2. **950 的 epilogue 是 SIMT 实现**（`__simt_vf__` / `asc_vf_call`），A2 上不存在。
3. arch35 路径的 dtype 集合里**没有 FP32**，而本任务的主路径就是 FP32（任务书 §2.4「主路径为单精度实数」）。
4. A2 上已有**仓内验证过的高层 Cube Matmul API 用法**（`blasLt/matmul/fp32/arch22/matmul_fp32_kernel.cpp`、
   `blas/gemm_strided_batched/arch22`、`blas/trmm/arch22` 等），且 `CubeFormat::ND` 原生支持
   「行主序 + 任意行前导维」。

因此 **arch22 采用高层 Cube Matmul API 路线**（`lib/matmul_intf.h` + `MatmulImpl` +
`GetMatmulApiTiling`），并**自研 column-major → row-major 的映射**与**独立的向量 epilogue**。

---

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言，在 ops-blas 仓中补齐 `aclblasGemmGroupedBatchedEx` 的 Atlas A2/A3（arch22）实现：

- 复用 `include/cann_ops_blas.h` 既有声明，**不新增产品私有平行接口**；
- 主路径数据类型：FLOAT32（`Atype/Btype/Ctype = ACL_FLOAT`，`computeType = ACLBLAS_COMPUTE_32F`）；
- 布局：**列主序**（column-major），`lda/ldb/ldc` 为列主序行前导维；
- 支持 `ACLBLAS_OP_N / ACLBLAS_OP_T`；`ACLBLAS_OP_C`（共轭转置）对实数矩阵等价于 `OP_T`，按 `OP_T` 处理；
- 分组维度：`groupCount` 个组，每组 `groupSize[i]` 个矩阵，**组内矩阵共享该组的全部参数**；
- 对标基线：cuBLAS `cublasGemmGroupedBatchedEx`，精度 golden 用 Netlib / OpenBLAS 实数实现（`cblas_sgemm`）逐组生成。

## 需求拆解

1. **接口与校验层对齐**
   - 复用既有声明，校验顺序与任务书 §2.5 一致：
     `handle → dtype → groupCount → host 数组非空 → 逐组值 → 指针数组`；
   - `groupCount == 0` 直接返回 `SUCCESS`（短路，不再校验其余参数）；
   - `groupCount < 0`、逐组 `m/n/k/groupSize < 0`、非法行前导维 → `ACLBLAS_STATUS_INVALID_VALUE`；
   - 非法 `transa/transb` 枚举 → `ACLBLAS_STATUS_INVALID_ENUM`；
   - 不支持的 dtype / computeType → `ACLBLAS_STATUS_NOT_SUPPORTED`。

2. **column-major → row-major 映射**
   - 列主序 `C = op(A)·op(B)`（`A` 为 m×k、`B` 为 k×n、`C` 为 m×n）等价于行主序
     `Cᵀ = op(B)ᵀ·op(A)ᵀ`；
   - 因此在 host 侧把 `m↔n`、`lda↔ldb` 交换，并把转置标志互换后交给 Cube；**左操作数是原始 B**。

3. **按组的 kernel 特化**
   - 转置组合（2×2）与尺寸档（2）共 8 种特化。转置组合与尺寸档都是**组级属性**，
     所以 host 侧一次下发全部「有任务」的特化，各特化在 kernel 内部按组挑任务；
   - host 侧用 `specMask` 位图只下发**真正有任务**的特化：8 个特化全量下发时，7 个空跑特化各要 ~2 µs，
     在小 shape 上会占掉整个算子耗时的一半以上。

4. **向量化 epilogue**
   - Cube 把行主序乘积写进 workspace，epilogue 再按列主序把它合并回 C：
     `C = alpha·product + beta·C`；
   - `beta == 0` 时不得读 C（任务书允许此时 C 指针为空）。

5. **测试工程**
   - 直接支持随包官方 CSV（驼峰列名 + `NULLPTR` 负向编码），验收脚本 `verify_accuracy.py` 可原样装载；
   - golden 由 `cblas_sgemm` 逐组生成；精度判据按任务书 §3.2 口径执行。

---

# 详细设计（required）

## 算子分析

### 数学公式

对第 `g` 个组（组内第 `b` 个矩阵，`i = b` 为组内下标）：

```
C[g][i] = alpha[g] · op(A[g][i]) · op(B[g][i]) + beta[g] · C[g][i]
```

其中 `op(X)` 为 `X`（`OP_N`）或 `Xᵀ`（`OP_T`/`OP_C`），`A[g][i]` 为 m[g]×k[g]（列主序，行前导维 lda[g]），
`B[g][i]` 为 k[g]×n[g]（列主序，行前导维 ldb[g]），`C[g][i]` 为 m[g]×n[g]（列主序，行前导维 ldc[g]）。

### 支持数据类型

| 组合 | Atype | Btype | Ctype | computeType | 说明 |
| --- | --- | --- | --- | --- | --- |
| 主路径 | ACL_FLOAT | ACL_FLOAT | ACL_FLOAT | ACLBLAS_COMPUTE_32F | **支持** |
| 其它 | 其它 | 其它 | 其它 | 其它 | 返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |

arch35 路径支持 FP16/BF16/FP8，两条路径的 dtype 集合是**并集**，互不影响。

### 支持形状

- `m/n/k`：任意正整数（0 维在 `m=0` 或 `n=0` 时只做 `beta·C` 缩放，`k=0` 时乘积项为 0）；
- `lda ≥ max(m, k)`（`transa=N`）/ `lda ≥ k`（`transa=T`）；`ldb ≥ max(k, n)`（`transb=N`）/ `ldb ≥ n`（`transb=T`）；
  `ldc ≥ m`；
- `groupCount ≥ 0`；`groupSize[i] ≥ 0`；
- 理论上限受 workspace 与 `uint32` 任务计数限制（host 侧显式检查溢出）。

## 算子实现

### 实现方案

整体是 **host 侧 tiling + 一次 cube kernel + 一次向量化 epilogue kernel**：

```
aclblasGemmGroupedBatchedEx
        │
        ├─ host：参数校验（任务书 §2.5 顺序）
        ├─ host：逐组初始化 group 记录（形状/转置/alpha/beta/批数）
        ├─ host：切分 tile、划分 cube 任务与 epilogue 任务、计算 workspace 布局
        ├─ host：生成 specMask（哪些特化有任务）
        ├─ device：tiling 结构（header + group 数组）H2D
        ├─ launch：cube kernel（只下发 specMask 命中的特化）
        └─ launch：epilogue kernel（向量核，按列合并回 C）
```

#### 3.2.1 host 侧设计

**tiling 结构**（`gemm_grouped_batched_ex_tiling_data.h`）：

```cpp
struct GroupedGemmTilingHeader {
    uint32_t groupCount;          // 组数
    uint32_t problemCount;        // 总矩阵数 = Σ groupSize
    uint32_t totalCubeTasks;      // cube 任务总数
    uint32_t totalEpilogueTasks;  // epilogue 任务总数
    uint32_t epilogueTile;        // epilogue 行块高度
    uint32_t specMask;            // 有任务的 cube 特化位图
};
// 之后紧跟 groupCount 个 GroupedGemmGroupData
```

每个 `GroupedGemmGroupData` 记录一组的：行主序映射后的 `m/n/k/lda/ldb/ldc`、转置标志 `isTransA/isTransB`、
tile 划分 `mTiles/nTiles/tileM/tileN`、批数区间 `batchStart/batchCount`、cube/epilogue 任务区间、
原始列主序形状 `originalM/originalN/originalLdc`、workspace 偏移、`alpha/beta`。

**1. 分核策略：**

- cube：任务总数 `Σ batchCount·mTiles·nTiles`；下发的 block 数 `min(cubeCoreCount, totalCubeTasks)`，
  kernel 内用 **grid-stride** 循环领任务（`taskId = blockIdx; taskId < total; taskId += blockNum`），
  任务在核间均衡。
- epilogue：任务总数 `Σ batchCount·ceil(originalN/8)·ceil(originalM/512)`
  （一个任务负责 C 的 8 列与一段行）；
  下发的 block 数 `min(aivCoreCount, totalEpilogueTasks)`；**每个 block 领一段连续任务**
  （`[blockIdx·chunk, (blockIdx+1)·chunk)`），使组记录可以按组缓存。

**2. 数据分块和内存优化策略：**

- **tile 几何**（沿用仓内已验证的 arch22 FP32 路径 `blasLt/matmul/fp32/arch22/matmul_fp32_kernel.cpp`）：
  - `BASE_M = 256`、`BASE_N = 256`：Cube 侧一次编译出 256 列宽（L1 tile）与 128 列宽两套配置；
  - `tileM = min(m, 256)`，再向下取到 64 的整数倍（MMA 行粒度）；`tileM` 为 0 时退化为 64；
  - `tileN`：`n > 128` 用 256 宽（large 特化），`n ≤ 128` 用 128 宽（small 特化）；再向下取到 64 的整数倍；
  - `mTiles = ceil(m/tileM)`、`nTiles = ceil(n/tileN)`，每组 `batchCount·mTiles·nTiles` 个 cube 任务。
- **workspace**：每个矩阵的乘积（m×n FP32）单独放在 workspace 的 256 B 对齐位置；
  `alpha == 0` 或 `k == 0` 的组不分配乘积缓冲（`hasGemm = 0`），epilogue 直接用 0 参与合并。
- **UB**：epilogue 的三个队列（product / C / out）各 2 槽，每槽 `8 列 × (512 行 + 8)` 个 float
  （8 是向量指令 64 bit 对齐的余量），约 100 KB，在 A2 的 192 KB UB 容量之内。

**3. 异步语义与 workspace 归属：**

- 接口是**异步**的：算子只做「校验 → 建 tiling → 取 workspace → 下发 kernel」，
  **不调用 `aclrtSynchronizeStream`**，调用方在读回 Device 结果前自行同步 stream
  （任务书 §2.5）。
- 因此 tiling 结构**不能**用「每次调用 `aclrtMalloc`、返回时释放」的临时缓冲：
  函数返回时 kernel 还没跑完，缓冲已被释放。tiling 放在 **handle 的 workspace** 里
  （乘积缓冲之后、16 B 对齐处），workspace 归 handle 所有、生命周期长于调用。
- 代价：调用方在两次调用之间需要自行同步，避免后一次调用覆盖前一次仍在使用的数据；
  这与其它 aclBLAS 算子使用 handle workspace 的约定一致。

**4. Device 侧缩放系数：**

- 任务书 §2.4 允许 `alphaArray` / `betaArray` 位于 Host **或** Device 内存。
- Host 侧用 `aclrtPointerGetAttributes` 判定系数数组的位置：
  - Host：host 直接读出系数，写进 group 记录；
  - Device：group 记录里带上系数数组的设备地址（`alphaGm` / `betaGm`），
    **epilogue 按 group index 直接从 Device 读该组的系数**——不回拷、不等 event、
    **不排空 stream**，整条路径仍然是异步的。
- Host 看不到 Device 系数时按「该组有乘积」规划（`hasGemm = 1`），并关闭
  `directWrite` 捷径（该捷径要求 `alpha == 1 && beta == 0`，Host 看不到就不能用）；
  `alpha == 0` 时 kernel 读到 0，`0 · product` 即为 0，合并结果仍然正确。

**5. tilingkey 规划策略：**

本算子**不使用 tilingkey**，改用「特化入口 + 位图」：

- 8 个特化 = `transA(2) × transB(2) × 尺寸档(2)`，各自是独立的 kernel 入口；
- host 侧根据每组的 `isTransA/isTransB` 与 `tileN` 算出该组落在哪个特化，置位到 `specMask`；
- 下发时只 launch `specMask` 命中的特化，`specMask` 本身也写进 tiling header 便于排查。

> 为什么不用 tilingkey：tilingkey 只能让**同一个 kernel** 走不同分支，而这里不同分支的
> `SetOrgShape` / `SetTensorA/B` 参数类型与编译期常量不同，需要**不同的 kernel 实例**才能把
> Cube 的形状配置固化到编译期。

#### 3.2.2 kernel 侧设计

**cube kernel**（`gemm_grouped_batched_ex_kernel.cpp`）：

1. `RunGroupedCubeTasks<TRANS_A, TRANS_B, SMALL>` 是模板化的任务循环：
   - 读 GM 上的 tiling header 与 group 数组；
   - grid-stride 遍历 `totalCubeTasks`，用**二分查找**定位任务所属的组；
   - **只执行与本特化匹配的任务**（`isTransA/isTransB` 匹配、`tileN ≤ 128` 匹配尺寸档），其余跳过；
   - 匹配上之后调用对应模板参数的 `RunOneGroupedTask`。
   - **组记录本地缓存**：二分查找命中后，组的描述（`GroupedGemmGroupData`）被拷到设备侧
     的本地存储并被后续同组的任务复用，避免每个任务都从 GM 重新读一遍组记录；
     特化过滤（`isTransA/isTransB`/`tileN`）在**拷贝组记录之前**完成，
     因此不属于本特化的组不付这次拷贝的代价；
   - 查找复杂度为 O(log groupCount)，与组数无关。
2. **Cube 实例按 block 复用**：`TPipe` 与 `MatmulT` 实例在**每个 block 内只建一次**，
   tile 只改变 launch 的形状，不重建 Cube 的资源。实测重建的开销在小 shape 上与 tile
   本身相当，这是把 7 条不达标的性能用例拉回标杆以内的关键改动之一。
3. **乘积直写 C（`directWrite`）**：`alpha == 1 && beta == 0 && ldc == m` 时，
   Cube 产出的行主序乘积块与列主序 C 的内存布局完全一致，乘积**直接写进 C**，
   省掉整趟转置 epilogue。该捷径需要 Host 看到系数，因此只在 Host 系数时启用。
4. `RunOneGroupedTask`：
   - 用 `mm.Init(GetMatmulApiTiling(...), &pipe)` 初始化 Cube，`SetHF32(false)` 关掉 HF32
     （HF32 会截断尾数，在长 K 归约上会明显漂移）；
   - `SetOrgShape` 设置行主序形状与行前导维（A/B 已按 host 侧的映射交换）；
   - 按 `kLaunchChunkK = 4096` 分块推进 K：首块覆盖写，后续块累加（`IterateAll(dst, accumulate)`）；
   - 乘积写入 workspace（行主序，行前导维 = `originalM`）。

**epilogue kernel**（`gemm_grouped_batched_ex_epilogue_kernel.cpp`）：

- 沿用仓内已验证的 arch22 向量 epilogue 写法（`blasLt/matmul/epilogue/arch22/epilogue_alpha_beta_kernel.cpp`）：
  - GM→UB 用 **`DataCopyPad` 且 `isPad = true`**（列主序 C 的一列地址/长度天然不是 32 B 对齐，
    不带 pad 的 `DataCopy` 会静默不搬运）；
  - MTE2→VEC 的同步走 `TQue` 队列的 `EnQue/DeQue`；
  - 向量指令的元素数一律取 8 的整数倍（`alignedCount`），不足处用 `rightPadding` 补齐；
  - **float→float 的拷贝用 `Adds(dst, src, 0.0f)` 而不是 `Cast`**：`Cast` 不支持同类型转换，
    实测输出为未定义值。
- 一个任务负责 C 的**一组列**（8 列）与它们的**一段行**（≤512 行）：
  - `mid = alpha · product`（`Muls`）；`beta != 0` 时再 `mid += beta · C`（`Muls` + `Add`）；
  - `beta == 0` 且该问题的 C 指针为空时，**直接返回**，不读不写 C（任务书语义）；
  - 结果写回 C（UB→GM 用不带 pad 参数的 `DataCopyPad` 重载）。
- **连续块单次 DMA**：乘积缓冲是行主序 `[originalN][originalM]`，当任务覆盖整行
  （`count == originalM`）且行距已 32 B 对齐时，一组列在源上就是**一段连续区间**，
  用一次 `DataCopy` 搬完，而不是一列一次；C 侧在 `ldc == m` 时同样成立。
  两条路径统一收敛到 `CopyColumnsFromGm` / `CopyColumnsToGm`。
- **epilogue 是标量瓶颈，因此按标量开销来设计**（这一点由 msprof 的单元分解实测确认：
  优化前 epilogue 的 `aiv_scalar_time` 几乎等于它的总耗时，vector 与 DMA 单元大部分时间在空转）：
  1. **列块化**：C 是列主序，**列**才是连续方向。一个任务处理 `8 列 × 512 行`
     （约 4096 个元素，与仓内已验证的 arch22 LtMatmul epilogue 的 1024 元素块同一量级）；
     若一列一任务，窄矩阵（m = 16）每 64 字节就要付一次完整的队列往返，
     任务数会随 n 线性膨胀。
  2. **每个 block 领一段连续任务**（而不是 grid-stride），于是任务所属的组在一段范围内不变，
     **组记录只在换组时从 GM 取一次**（原来每个任务都要 13 次标量 GM 加载复制整条组记录，
     是 epilogue 的头号开销）；
  3. **结果直接从 VECOUT 队列的 tensor 写回**：中间的累加缓冲由 VECOUT 队列分配，
     省掉一次 `Adds` 拷贝和一对队列同步；
  4. 任务解码的除数（`tasksPerProblem` / `rowTiles` / `colBlocks`）随组记录一起缓存到本地；
  5. **tiling header 的大小保持 8 字节对齐**：header 后面紧跟 group 数组，
     kernel 按 64 位块读取 group 记录，header 若不是 8 的整数倍，group 数组就会落在
     错位的地址上，导致 64 位 GM 访问触发 AI Core 异常（实测踩过：加了一个 `uint32_t`
     字段后 header 变成 28 字节，整算子直接返回 `ACLBLAS_STATUS_EXECUTION_FAILED`）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas 300I/800T A3 | √ |

## 算子约束限制

- 不支持 FP16/BF16/FP8/FP64（返回 `ACLBLAS_STATUS_NOT_SUPPORTED`）；这些 dtype 由 arch35 路径覆盖。
- 不支持 `ACLBLAS_OP_C` 与 `OP_T` 的区分（实数下两者等价，按 `OP_T` 处理）。
- 乘积缓冲占用 workspace，workspace 不足时返回 `ACLBLAS_STATUS_ALLOC_FAILED`（**不降级、不截断**）。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | `atol = rtol = 2⁻¹³`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ max(1e-2, 32×ULP(max|golden|))`；`alpha = 0` 的用例要求 `D = beta·C` **位精确** | 任务书 §3.2（生态算子开源精度标准 FLOAT32 档位） |
| 性能标准 | 平均单次耗时 ≤ GPU 标杆 × 1.25（即 ≥ 0.8× GPU） | 任务书 §3.3 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

**golden 生成**：按任务书 §3.1，逐组用 `cblas_sgemm`（OpenBLAS 0.3.25 实数实现）生成参考值；
`alpha == 0` 或 `k == 0` 时跳过乘积项，`beta == 0` 时不读 C。

**非有限值的处理**：`RANDOM_EXTREME` 填充会让 FP32 中间结果溢出成 ±Inf，FP32 累加顺序与 golden 可能
给出**同为 Inf 但符号相反**的形态。任务书 §3.2 明确「最终形态取决于累加顺序与实现路径」，
因此这类分歧用 **double 精度真值仲裁**：用 double 重算一遍，判断究竟是谁伪溢出。
有限值的判据**不放宽**。

## 实测结果（Atlas A2 · Ascend 910B1 · CANN 8.5.0）

> 被测提交 `dfbe769`（代码 MR !752 head，消除 CI CodeCheck 的 2 条「重复代码[C++]」告警：分组二分查找提取为共享头模板 `FindGroupIndexByTaskStart<StartPolicy>`、测试侧提取 `FillFloatDefaults`；**算子数值逻辑与用例判定逻辑均未改变**），下表数据在该提交对应的构建（`build_gg11`）上实跑采集；
> 其中 2 条用例因首次采集时 msprof 采样文件异常（`device_5` 缺失）已剔除后补采，
> 最终参与判定的为随包 CSV 的全部 200 条。

测试代码随本算子一同提交（`test/gemm_grouped_batched_ex/arch22/`），
随包 200 条用例（1200 条 CSV：1000 条精度 + 200 条性能）逐条执行。

| 验收项 | 判据 | 实测 |
| --- | --- | --- |
| 精度（§3.2） | `matched_ratio ≥ 0.99`、`max_abs_error ≤ max(1e-2, 32×ULP)`、`alpha=0` 位精确 | **1200 / 1200 全部通过**；`matched_ratio` 最小 **0.9926**，`max_abs_error` 最大 **2.69e-03** |
| 性能（§3.3） | 平均单次耗时 ≤ GPU 标杆 × 1.25 | **200 / 200 全部达标**；`msprof op` 单算子 Task Duration 与 GPU 标杆之比中位 **0.266**、最大 **0.623**（最小 0.036） |
| 内存（§3.4） | 不涉及 | workspace/输入输出数据量最大 **0.0508**（≤ 1.2） |

性能采样口径为 §3.3.1 规定的 `msprof op` 单算子 Task Duration，每例 warmup 5 次后取 50 个有效样本的均值；
GPU 标杆取自随包 `gpu_baseline.csv`。判据为**逐条**判定（`NPU 耗时 ≤ 1.25 × GPU 标杆`），200 条全部通过。

**测试侧计时口径的两点说明**（算子实现未因此改动）：

1. **计时区间只含 1 次算子调用**：Device 缓冲在 suite 级一次性预留并复用，算子的 Device 侧准备、
   矩阵上传与回拷全部排除在计时区间之外；冷启动由 suite 级预热吸收，逐条采集时关闭 suite 预热，
   改用 in-case warmup 5 次 + 50 个有效样本，取算术平均。优化前的历史版本数字（比值中位 0.373、
   最大 1.140）差异**全部来自测试侧**，算子实现一行未改，两种口径下均为 200/200 达标。
2. **随包官方 `verify_performance.py` 的物理上限**：该脚本取 GTest 输出的**整例墙钟**且只报整毫秒，
   200 条中 180 条的 GPU 标杆 < 0.8 ms（预算 < 1 ms），报 0 ms 判 `NO_REF`、报 1 ms 必判 `FAIL`，
   因此**在整数毫秒口径下最多只有约 4~5 条能判 PASS**，与算子实现无关。经计时口径修正后，
   机器空闲时在最终构建 `build_gg11` 上连跑 3 次，官方脚本实测结果均为
   **PASS=3 / FAIL=0 / NO_REF=197、rc=0**（逐条一致）；任务书 §3.3.1
   指定的 `msprof op` 口径为 **200/200 PASS**。另有一轮与其它回归任务并发的实跑读作
   **PASS=3 / FAIL=1 / NO_REF=196、rc=1**，唯一一条 FAIL（`TC_PF_1065`）是 GTest 整毫秒墙钟
   读数在 0 ms / 2 ms / 3 ms 之间跳变造成的（判定预算 2.96 ms），该用例的算子真实耗时为
   864.9 µs、`msprof op` 口径比值 0.365，达标。逐条对照见交付包
   `tables/口径对照_官方脚本_vs_msprof.csv`。

关键设计点在实测中的收益：

- **乘积直写 C**：`alpha == 1 && beta == 0 && ldc == M` 时跳过转置 epilogue，
  实测 200 条性能用例的 `msprof` 分解中 **epilogue kernel 全部消失**，
  算子耗时相对上一版中位下降 13.6%、最大下降 85.7%。
- **Cube 实例按 block 复用**：每个 block 只建一次 `TPipe`/`MatmulT`，所有形状普遍受益。
- **接口异步 + Device 侧缩放系数**：不排空 stream，Device 系数由 kernel 直接读取，
  全程无同步点；对应测试用例 `DeviceScales` 对同一问题分别用 Host 系数与 Device 系数各跑一遍，
  两者都与 `cblas_sgemm` golden 一致。

**补充验证**（对应 ops-blas !752 评审意见）：

- **K 维分块累加路径**：内核按 `kLaunchChunkK = 4096` 推进 K（首块覆盖写、后续块累加）。
  为覆盖 `k > 4096` 的多段累加分支，补充 3 条正向用例——`LargeKChunkedNN`（256×256×5000）、
  `LargeKChunkedTransA`（256×128×5000）、`LargeKChunkedTransB`（128×256×8292），
  实测 `matched_ratio` 0.9981 / 0.9981 / 0.9975，`max_abs_error` 3.30e-03 / 3.17e-03 / 3.78e-03，
  均满足 §3.2 判据。
- **测试工程门禁**：新增 `test_gate.py` / `run_self_test.py` / `run_memcheck.py` 三件套，
  对齐上游 master `test/gemm_grouped_batched/sgemm_grouped_batched/` 的范式：
  `run_self_test.py` 断言 CSV 用例数并全量执行（发现 1039 / 执行 1039 / 失败 0 / 跳过 0），
  `run_memcheck.py` 通过 `mssanitizer --tool=memcheck` 插桩（60 用例全执行、878 次 kernel 插桩、
  11636 条内存记录、0 error / 0 warning），两者均输出 `*_summary.json` 作为可核验证据。

## 兼容性分析

- 新增目录 `blas/gemm_grouped_batched_ex/arch22/` 与 `test/gemm_grouped_batched_ex/arch22/`，
  不修改 arch35 的任何文件；
- `include/cann_ops_blas.h` 的既有声明**不变**（与 950PR 共用）；
- 共享的 `test/gemm_grouped_batched_ex/gemm_grouped_batched_ex_param.h` 与 `test/frame/verify.h`
  **不变**（arch35 路径仍在用），arch22 自带参数头与精度判据头；
- CMake **无需改动**：`arch22` 已在 `ARCH_SPECIFIC_DIRS`，`get_soc_arch_dirs` 对 `ascend910b*` 映射到 arch22；
  精度测试所需的 cblas 经 `${REFBLAS_LIB}` 自动链接。
