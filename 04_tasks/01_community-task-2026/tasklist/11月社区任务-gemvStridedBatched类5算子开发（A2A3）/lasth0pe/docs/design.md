# gemvStridedBatched 类 5 个算子设计文档

## 一、需求背景

### 1.1 需求来源

本需求来源于社区任务[11月社区任务-gemvStridedBatched类5个算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)。目标是在 Atlas A2/A3（arch22）、CANN 9.1.0 环境下，基于 `ops-blas` 工程实现 5 个跨步批量矩阵-向量乘算子的 Atlas A2/A3 高性能版本与完整测试：`aclblasSgemvStridedBatched`、`aclblasHSHgemvStridedBatched`、`aclblasHSSgemvStridedBatched`、`aclblasTSTgemvStridedBatched`、`aclblasTSSgemvStridedBatched`。

### 1.2 背景介绍

该组算子对连续或等距跨步存放的 `batchCount` 个矩阵-向量对执行 `y_b = alpha * op(A_b) * x_b + beta * y_b`，通过显式 `strideA`、`stridex`、`stridey` 在基址上定位每个 batch，覆盖 FP32、FP16（同精度与混合精度输出）及 BF16（同精度与混合精度输出）五种组合。

仓库已有 `blas/gemv_strided_batched/arch35/` 与 `blas/gemv/arch22/` 实现，可参考其参数校验、负步长寻址及列主序高效分块计算模式。Atlas A2/A3（arch22）采用 Ascend C AIV 向量核与片上 UB 双缓冲分块模型，复用 `include/cann_ops_blas.h` 与仓库 GTest/CSV 框架。

## 二、需求分析

### 2.1 接口定义

公共声明位于 `include/cann_ops_blas.h`，5 个接口参数顺序与既有架构声明一致：

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* A, int lda, int64_t strideA, const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount);

aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount);
```

| 参数 | 位置与含义 | 约束 |
| --- | --- | --- |
| handle | Host，库上下文句柄 | 非空有效句柄 |
| trans | Host，矩阵转置模式 | `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`（实数算子等价 `ACLBLAS_OP_T`） |
| m、n | Host，矩阵行数与列数 | `m >= 0`，`n >= 0`（任一为 0 时快速返回 `ACLBLAS_STATUS_SUCCESS`） |
| lda | Host，列主序前导维度 | `lda >= max(1, m)` |
| strideA、stridex、stridey | Host，批间元素步长 | `> 0`；且当 `batchCount > 1` 且 `m > 0, n > 0` 时满足不重叠约束 `strideA >= lda * n`、`stridex >= (dimX - 1) * abs(incx) + 1`、`stridey >= (dimY - 1) * abs(incy) + 1` |
| alpha、beta | Host，FP32 标量指针 | 非空且位于 Host 内存 |
| A、x、y | Device，批数据首地址 | `m > 0 && n > 0 && batchCount > 0` 时非空 |
| incx、incy | Host，向量内元素步长 | 非零且非 `INT_MIN`，支持正负值 |
| batchCount | Host，批数量 | `batchCount >= 0`（为 0 时快速返回 `ACLBLAS_STATUS_SUCCESS`） |

### 2.2 功能要求

1. 支持 `m == 0 || n == 0 || batchCount == 0` 合法快速返回（在结构与标量校验完成后、指针判空前触发）。
2. 支持五种精度组合（S/HSH/HSS/TST/TSS），统一在 FP32 域完成完整点积累加后再乘 `alpha` 并按目标输出类型舍入（FP16 使用 `CAST_ROUND`，BF16 使用 `CAST_RINT`）。
3. 支持 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 与 `ACLBLAS_OP_C`（等价 `ACLBLAS_OP_T`），支持正负元素步长、`beta = 0` 不读取旧 y 且 `abs(incy) > 1` 时保留间隔元素。
4. 支持 `alpha == 0 && beta == 1` Host 侧快速返回，`alpha == 0 && beta != 1` Kernel 侧跳过矩阵与 x 向量读取，仅执行 `beta * y` 缩放。

### 2.3 支持范围

目标架构为 Atlas A2/A3（`arch22`，`SOC_VERSION=ascend910b*`/`ascend910_93*`），默认验证平台为 `Atlas 800T A2`（Ascend 910B2/B3）与 Atlas A3，软件基线为 CANN 9.1.0。

## 三、详细设计

### 3.1 算子分析

第 b 个 batch（`0 <= b < batchCount`）基址为 `A + b * strideA`、`x + b * stridex`、`y + b * stridey`（64 位地址增量）。无论 `trans` 取值如何，物理矩阵 `A` 按列主序存储为 `lda * n`（`lda >= max(1, m)`）。负步长向量逻辑起点为 `-(int64_t(dim) - 1) * int64_t(inc)`。

### 3.2 Host 侧设计

#### 3.2.1 参数校验

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `trans` 非 `ACLBLAS_OP_N`/`ACLBLAS_OP_T`/`ACLBLAS_OP_C` | 返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `m < 0` 或 `n < 0` 或 `lda < max(1, m)` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `incx`/`incy` 为 0 或 `INT_MIN`，或 `strideA/stridex/stridey <= 0`，或 `batchCount < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `batchCount > 1` 且 `m > 0 && n > 0` 时 `strideA/stridex/stridey` 不满足不重叠下界 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `alpha`/`beta` 为空或标量位于 Device 内存 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `m == 0` 或 `n == 0` 或 `batchCount == 0` 或 `*alpha == 0.0f && *beta == 1.0f` | 快速返回 `ACLBLAS_STATUS_SUCCESS` |
| 8 | `A`/`x`/`y` 为空 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

#### 3.2.2 二维自适应分核与模板选择

通过 `GetAivCoreCount()` 动态获取核数 `coreNum`，联合 `batchCount`、转置标志 `isTrans` 与输出维度 `outDim` 构建二维任务网格 `(batchIdx, outSplitIdx)`：
- 当 `batchCount < coreNum` 时按转置方向差异化切分：`ACLBLAS_OP_N` 取每块最小 64 元素、16 元素对齐，保障连续行突发长度与 64B/32B 对齐访问；`ACLBLAS_OP_T` 取最小 16 元素、1 元素步长切分，打满全部 AIV 核且写回区间互不重叠。
- 当 `coreNum <= batchCount < 2 * coreNum` 且不能整除时，将 `outDim` 二分消除核间尾块负载不均；其余大批次按 batch 切分。
- 总任务 `totalTasks = batchCount * outSplits` 均匀分发到 `usedCoreNum = min(coreNum, totalTasks)` 个核；同一 batch 内的相邻输出分块在同一核内连续执行，复用已加载到 UB 的 `x` 向量。

### 3.3 Kernel 侧设计

Kernel 为 `KERNEL_TYPE_AIV_ONLY`，按 `trans` 区分独立 UB 内存布局（片上总占用不超过 128 KB）：
1. **`ACLBLAS_OP_N` 路径（二维 DMA 双缓冲 + `Brcb` 广播树形归约）**：
   - 输出行按 `MAX_TILE_M = 128` 分块（FP32 按 8 元素、FP16/BF16 按 16 元素对齐为 `rowsAligned`），点积列块动态取 `tileK = min(512, AlignDown(8192 / rowsAligned, 8))`。
   - `n <= 4096` 时按 batch 整段缓存 `x`；`n > 4096` 时按 4096 分段加载。
   - 使用 `TQue<QuePosition::VECIN, 2>` 双缓冲重叠二维 `DataCopyPad` 预取与计算（FP32 尾行块启用 DMA `rightPadding` 补零）；非 FP32 输入整块或按有效 `rows` 分段 `Cast` 为 FP32，8 的倍数列采用 `Brcb + Mul + 广义二叉树 Add` 矢量归约，余数 `< 8` 列按精确 `rows` 长度执行 `Axpy`。
2. **`ACLBLAS_OP_T` 路径（二维 DMA 双缓冲 + `WholeReduceSum` 多列并行归约）**：
   - 点积轴按 64 对齐分段（单段上限 `kStrideMax = min(AlignUp(m, 64), 512)`），输出列批大小动态取 `colTile = min(128, 8192 / kStrideMax)`。
   - `TQue<QuePosition::VECIN, 2>` 双缓冲加载二维 `(kCnt x cols)` 块，非 FP32 满片整块 `Cast`、尾片按列仅转换有效 `curCnt` 个元素后清零 `mul` 尾区并按精确 `curCnt` 执行 `Mul` 与 `Add`，避免未初始化 UB 读取。
   - 依 `segments = kAligned64 / 64` 分别使用单次 `WholeReduceSum`（1 段）、两级 `WholeReduceSum`（8 段）或分段 `WholeReduceSum + Add`（2–7 段）完成并行归约。
3. **写回与非单位步长处理**：
   - `incy == 1` 使用单次连续 `DataCopyPad`；`incy != 1` 使用 32B 对齐的 UB staging 区配合二维带步长 `DataCopyPad` 或逐元素 staging DMA，全程由精确 `HardEvent`（`MTE2_V`/`MTE2_S`/`V_S`/`S_V`/`S_MTE3`/`V_MTE3`/`MTE3_V`/`MTE3_S`/`MTE3_MTE2`）同步，消除 `PIPE_ALL` 全局停顿。

### 3.4 性能设计

1. **小批大矩阵多核扩展**：二维 `(batch, outSplit)` 分割使单批及少批场景充分利用全部 AIV 核。
2. **双缓冲访存计算掩盖**：`TQue<QuePosition::VECIN, 2>` 重叠二维 DMA 与矢量乘加，配合 `Brcb` 树归约及 `WholeReduceSum` 消除标量归纳串行瓶颈。
3. **跨分块向量缓存与零工作区**：单核顺序遍历同批输出切块复用 UB 中的 `x` 向量，不依赖额外临时 GM workspace。
4. 依据官方 `test_cases/*_gemv_strided_batched/gpu_baseline.csv`（每变体 200 条 `TC_PF_*`，共 1000 条），在 Atlas 800T A2（CANN 9.1.0）上通过 `msprof op` 采集核函数 `Task Duration(us)`，结合实测完成分核与分块调优。

### 3.5 工程集成

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 共用既有 5 个 `aclblas*gemvStridedBatched` 公共声明 |
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_host.cpp` | 5 接口入口、参数校验、二维任务划分与启动 |
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_kernel.cpp` | 双缓冲二维分块 `OP_N`/`OP_T` AIV 矢量内核 |
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_kernel.h` | `gemv_strided_batched_arch22_kernel_do` 声明 |
| `blas/gemv_strided_batched/arch22/gemv_strided_batched_tiling_data.h` | Tiling 结构与分块常量 |
| `test/gemv_strided_batched/arch22/` | `arch22` NPU 封装、GTest（含二维切分与多块覆盖用例）与 CSV |
| `blas/gemv_strided_batched/README.md` | 补充 5 个接口的 Atlas A2/A3 支持说明 |

## 四、可维可测分析

### 4.1 精度验证

基于任务附件 `test_cases/` 中 `sgemv_strided_batched`、`hsh_gemv_strided_batched`、`hss_gemv_strided_batched`、`tst_gemv_strided_batched`、`tss_gemv_strided_batched` 共 5 套官方测试集（每套 1000 条精度用例 + 200 条 `TC_PF_*` 性能用例，总计 6000 条）执行全量验证；参考标杆采用 Netlib 双精度累加/`cblas_sgemv`，精度阈值满足：FP16 输出 `rtol = atol = 2^-10`、BF16 输出 `rtol = atol = 2^-7`、FP32 输出（S/HSS/TSS）`rtol = atol = 2^-13`，并通过 `verify_accuracy.py` 全量校验。同时补充多批次重叠边界、负步长 `incy`、`OP_C` 等价性及 `alpha=0` 缩放单测。

### 4.2 性能验证

依据任务附件各变体目录的 `gpu_baseline.csv`（共 1000 条性能用例），在 Atlas 800T A2（CANN 9.1.0）环境下使用 `msprof op`（5 次预热 + 10 次采样算术平均）采集 kernel `Task Duration(us)`，并结合 `aclrtEvent` 通过 `verify_performance.py` 验证逐例满足 `NPU / GPU_baseline <= 1.25`。

### 4.3 兼容性与维护

新增代码均位于 `arch22` 子目录内，不影响 `arch35` 既有实现，支持 `--ops=sgemv_strided_batched`、`--ops=gemv_strided_batched` 等独立编译与 `blas_all_test` 聚合运行，并通过 `mssanitizer`（`memcheck`、`racecheck`、`initcheck`）0 错误验证。

### 4.4 交付内容

交付 5 个接口的 `arch22` 源码、测试与 CSV、README、设计文档、私有代码仓及验收报告包。

## 五、参考资料

1. 官方任务详情：[11月社区任务-gemvStridedBatched类5个算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)
2. 讨论区来源：`https://gitcode.com/cann/ops-blas/discussions/91`
3. 同算子另一架构参考：`blas/gemv_strided_batched/arch35/gemv_strided_batched_host.cpp`、`blas/gemv_strided_batched/arch35/gemv_strided_batched_kernel.cpp`
4. 同架构高性能参考：`blas/gemv/arch22/sgemv_host.cpp`、`blas/gemv/arch22/sgemv_kernel.cpp`
