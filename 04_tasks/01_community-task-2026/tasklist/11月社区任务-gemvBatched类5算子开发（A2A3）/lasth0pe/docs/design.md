# gemvBatched 类 5 个算子设计文档

## 一、需求背景

### 1.1 需求来源

本需求来源于社区任务[11月社区任务-gemvBatched类5个算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)。目标是在 Atlas A2/A3（arch22）、CANN 9.1.0 环境下，基于 `ops-blas` 工程实现 5 个批量矩阵-向量乘算子接口的 Atlas A2/A3 高性能版本与完整测试：`aclblasSgemvBatched`、`aclblasHSHgemvBatched`、`aclblasHSSgemvBatched`、`aclblasTSTgemvBatched`、`aclblasTSSgemvBatched`。

### 1.2 背景介绍

该组算子执行批量指针数组形式的矩阵-向量乘加运算：`y[i] = alpha * op(A[i]) * x[i] + beta * y[i]`（`0 <= i < batchCount`），输入矩阵与向量地址以 Device 侧指针数组 `Aarray`、`xarray`、`yarray` 传入，覆盖 FP32、FP16（同精度与混合精度输出）及 BF16（同精度与混合精度输出）五种组合。

仓库已有 `blas/gemv_batched/arch35/gemv_batched_host.cpp`、`blas/gemv_batched/arch35/gemv_batched_kernel.cpp` 与 `blas/gemv_batched/arch22/cgemv_batched_host.cpp`，可参考其参数校验、32 位跨度上界检查及 Device 指针数组读取逻辑。Atlas A2/A3（arch22）采用 Ascend C AIV 向量核与片上 UB 双缓冲分块模型，复用 `include/cann_ops_blas.h` 与仓库 GTest/CSV 框架。

## 二、需求分析

### 2.1 接口定义

公共声明位于 `include/cann_ops_blas.h`，5 个接口参数顺序与既有架构声明一致：

```cpp
aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta,
    float* const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* const Aarray[], int lda, const uint16_t* const xarray[], int incx, const float* beta,
    uint16_t* const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* const Aarray[], int lda, const uint16_t* const xarray[], int incx, const float* beta,
    float* const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* const Aarray[], int lda, const uint16_t* const xarray[], int incx, const float* beta,
    uint16_t* const yarray[], int incy, int batchCount);

aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* const Aarray[], int lda, const uint16_t* const xarray[], int incx, const float* beta,
    float* const yarray[], int incy, int batchCount);
```

| 参数 | 位置与含义 | 约束 |
| --- | --- | --- |
| handle | Host，库上下文句柄 | 非空有效句柄 |
| trans | Host，矩阵转置模式 | `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`（实数算子等价 `ACLBLAS_OP_T`） |
| m、n | Host，矩阵行数与列数 | `m >= 0`，`n >= 0`（任一为 0 时快速返回 `ACLBLAS_STATUS_SUCCESS`） |
| lda | Host，列主序前导维度 | `lda >= max(1, m)` |
| alpha、beta | Host，FP32 标量指针 | 非空且位于 Host 内存 |
| Aarray、xarray、yarray | Device，批指针数组 | 非空，`batchCount` 个有效指针 |
| incx、incy | Host，向量步长 | 非零且非 `INT_MIN`；按 64 位绝对值计算跨距，并检查 INT32_MAX 上限 |
| batchCount | Host，批数量 | `batchCount >= 0`（为 0 时快速返回 `ACLBLAS_STATUS_SUCCESS`） |

### 2.2 功能要求

1. 支持五种精度组合（S/HSH/HSS/TST/TSS），统一在 FP32 累加域完成完整点积后再乘 `alpha` 并按目标输出类型舍入（FP16 使用 `CAST_ROUND`，BF16 使用 `CAST_RINT`）。
2. 支持 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 与 `ACLBLAS_OP_C`（等价于 `ACLBLAS_OP_T`），支持正负步长向量寻址，保证 `beta = 0` 不读取 y 旧值且 `abs(incy) > 1` 时不污染间隔元素。
3. 支持 `m == 0 || n == 0 || batchCount == 0` 合法快速返回与 `*alpha == 0.0f && *beta == 1.0f` Host 侧快速返回；`alpha == 0 && beta != 1` 时 Kernel 跳过矩阵与 x 向量访存仅执行 `beta * y` 缩放。
4. 严格按规范顺序校验 `handle`、`trans`、`m`/`n`、`lda`、`incx`/`incy`、`batchCount`、`alpha`/`beta`（含 Host 内存检查）、快速返回、`Aarray`/`xarray`/`yarray` 及 32 位跨度上界，非法入参返回对应状态码。

### 2.3 支持范围

目标架构为 Atlas A2/A3（`arch22`，`SOC_VERSION=ascend910b*`/`ascend910_93*`），默认验证平台为 `Atlas 800T A2`（Ascend 910B2/B3）与 Atlas A3，软件基线为 CANN 9.1.0。

## 三、详细设计

### 3.1 算子分析

对第 b 个 batch（`0 <= b < batchCount`），从 Device 指针数组槽位读取 `A_b = Aarray[b]`、`x_b = xarray[b]`、`y_b = yarray[b]`：
- 无论 `trans` 取值如何，物理矩阵 `A_b` 均按列主序存储为 `lda * n`（列主序元素 `A_b(i, j)` 偏移 `j * lda + i`），`lda >= max(1, m)`。
- `ACLBLAS_OP_N` 时输入向量长度 `n`、输出向量长度 `m`；`ACLBLAS_OP_T` / `ACLBLAS_OP_C` 时输入向量长度 `m`、输出向量长度 `n`。
- 负步长逻辑起点为 `-(int64_t(dim) - 1) * int64_t(inc)`。

### 3.2 Host 侧设计

#### 3.2.1 参数校验

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `trans` 非 `ACLBLAS_OP_N`/`ACLBLAS_OP_T`/`ACLBLAS_OP_C` | 返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 3 | `m < 0` 或 `n < 0` 或 `lda < max(1, m)` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `incx`/`incy` 为 0 或 `INT_MIN`，或 `batchCount < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `alpha == nullptr` 或 `beta == nullptr` 或标量位于 Device 内存 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `m == 0` 或 `n == 0` 或 `batchCount == 0` 或 `*alpha == 0.0f && *beta == 1.0f` | 快速返回 `ACLBLAS_STATUS_SUCCESS` |
| 7 | `Aarray`/`xarray`/`yarray` 为空或 `GemvBatchedSpanExceedsInt32` 超限 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

#### 3.2.2 二维自适应分核与模板选择

通过 `GetAivCoreCount()` 动态获取可用核数 `coreNum`，联合 `batchCount`、转置标志 `isTrans` 与输出维度 `outDim` 构建二维任务网格 `(batchIdx, outSplitIdx)`：
- 当 `batchCount < coreNum` 时，按转置方向差异化切分 `targetSplits = min(CeilDiv(outDim, minSplit), CeilDiv(coreNum, batchCount))`：
  - `ACLBLAS_OP_N` 沿列主序连续轴切分行输出，取 `minSplit = 64`、`splitAlign = 16`，保障单次 MTE2 突发长度与 64B/32B 对齐访问效率，避免小突发争用；
  - `ACLBLAS_OP_T` 点积沿连续列展开，输出 `outStart` 无 GM 对齐约束，取 `minSplit = 16`、`splitAlign = 1`，在小/单批场景下充分打满全部 AIV 核并消除尾部核闲置。
- 当 `coreNum <= batchCount < 2 * coreNum` 且不能整除时，将 `outDim` 二分以消除核间尾块不均；其余大批次按 batch 切分。
- 总任务 `totalTasks = batchCount * outSplits` 均匀分配到 `usedCoreNum = min(coreNum, totalTasks)` 个核；单核处理连续 `[taskBegin, taskEnd)` 任务段，同一 batch 的相邻输出分块共享已缓存在 UB 的输入向量 `x`。

### 3.3 Kernel 侧设计

Kernel 为 `KERNEL_TYPE_AIV_ONLY`，`OP_N` 与 `OP_T` 分设独立 UB 布局，片上占用不超过 128 KB（在 192 KB UB 容量内留足余量）：
1. **`ACLBLAS_OP_N` 路径（二维 DMA 双缓冲 + `Brcb` 广播树形归约）**：
   - 输出行按 `MAX_TILE_M = 128` 分块（FP32 按 8 元素、FP16/BF16 按 16 元素对齐为 `rowsAligned`），点积列块动态取 `tileK = min(512, AlignDown(8192 / rowsAligned, 8))`。
   - 当 `n <= 4096` 时整段缓存 `x` 于 UB 并复用；`n > 4096` 时按 4096 分段缓存 `x`。
   - 使用 `TQue<QuePosition::VECIN, 2>` 双缓冲并发发射二维 `DataCopyPad`（`blockCount = cols`、`srcStride = (lda - rows) * sizeof(T_IN)`、`dstStride` 对应 32B 块对齐，FP32 尾块行不足 8 的倍数时启用 DMA `rightPadding` 补零），MTE2 预取下一列块与 PIPE_V 计算当前列块流水重叠。
   - 非 FP32 输入整块或按有效行分段 `Cast` 为 FP32（尾行不足对齐且触发 `Brcb` 时预先对计算缓冲清零），对 8 的倍数列 `colsBase` 利用 `Brcb` 广播 `x`、整块 `Mul` 与广义二叉树 `Add` 消除 `Axpy` 写后读流水停顿，余数 `< 8` 列按精确 `rows` 长度执行 `Axpy` 累加。
2. **`ACLBLAS_OP_T` 路径（二维 DMA 双缓冲 + `WholeReduceSum` 多列并行归约）**：
   - 点积轴按 64 对齐分片（单片上限 `kStrideMax = min(AlignUp(m, 64), 512)`），输出列批大小动态取 `colTile = min(128, 8192 / kStrideMax)`，单次最多并行处理 128 列。
   - `TQue<QuePosition::VECIN, 2>` 双缓冲预取 `(kCnt x cols)` 二维列块，非 FP32 输入在满片时整块 `Cast`、不足 64 尾片时按列仅转换有效 `curCnt` 个元素，随后清零 `mul` 尾区并按精确 `curCnt` 执行 `Mul` 与 `Add` 累加，消除任何 UB 未初始化读。
   - 依据 `segments = kAligned64 / 64` 分路径执行归约：`segments == 1` 单次 `WholeReduceSum` 一次归约全部 `cols` 列；`segments == 8` 两级 `WholeReduceSum` 归约；`2 <= segments <= 7` 按段发射间隔步长 `WholeReduceSum` 后一次性 `Add` 合并。
3. **写回与非单位步长处理**：
   - `incy == 1` 使用单次 `DataCopyPad` 读取/写回；`incy != 1` 时借助按 32B 步距排布的 UB staging 区执行二维带步长 `DataCopyPad`（`incy > 0` 且跨度在 32 位域内）或单元素 staging DMA（负步长），全程通过精确 `HardEvent`（`MTE2_V`/`MTE2_S`/`V_S`/`S_V`/`S_MTE3`/`V_MTE3`/`MTE3_V`/`MTE3_S`/`MTE3_MTE2`）同步，不使用 `PIPE_ALL`。

### 3.4 性能设计

1. **小批大矩阵打满多核**：二维 `(batch, outSplit)` 分割彻底解决 `batchCount < 40` 时 AIV 核闲置问题，且 16 元素输出对齐消除写穿缓存行竞争。
2. **大批小矩阵零冗余开销**：单次启动覆盖全部 batch，无额外工作空间分配；输入向量 `x` 与设备指针数组读取按 batch 复用。
3. **访存计算重叠与矢量满带宽**：二维 `DataCopyPad` + 双缓冲隐藏 GM 访存延迟；`OP_N` 通过 `Brcb + Mul + 树形 Add`、`OP_T` 通过 `WholeReduceSum` 取代逐列标量归约。
4. 依据官方 `test_cases/*_gemv_batched/gpu_baseline.csv`（每变体 200 条 `TC_PF_*`，共 1000 条），在 Atlas 800T A2（CANN 9.1.0）上通过 `msprof op` 采集核函数 `Task Duration(us)`，对所有维度与转置组合完成基于测量证据的分核与 Tile 优化。

### 3.5 工程集成

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 共用既有 5 个 `aclblas*gemvBatched` 公共声明 |
| `blas/gemv_batched/arch22/gemv_batched_host.cpp` | 5 接口入口、参数校验、二维任务划分与启动 |
| `blas/gemv_batched/arch22/gemv_batched_kernel.cpp` | 双缓冲二维分块 `OP_N`/`OP_T` AIV 矢量内核 |
| `blas/gemv_batched/arch22/gemv_batched_kernel.h` | `gemv_batched_arch22_kernel_do` 启动声明 |
| `blas/gemv_batched/arch22/gemv_batched_tiling_data.h` | Tiling 结构与分块常量 |
| `test/gemv_batched/arch22/` | `arch22` NPU 封装、GTest（含二维切分与多块覆盖用例）与 CSV |
| `blas/gemv_batched/README.md` | 补充 5 个接口的 Atlas A2/A3 支持说明 |

## 四、可维可测分析

### 4.1 精度验证

基于任务附件 `test_cases/` 中 `sgemv_batched`、`hsh_gemv_batched`、`hss_gemv_batched`、`tst_gemv_batched`、`tss_gemv_batched` 共 5 套官方测试集（每套 1000 条精度用例 + 200 条 `TC_PF_*` 性能用例，总计 6000 条）执行全量验证；参考标杆采用 Netlib 双精度累加/`cblas_sgemv`，精度阈值满足：FP16 输出 `rtol = atol = 2^-10`、BF16 输出 `rtol = atol = 2^-7`、FP32 输出（S/HSS/TSS）`rtol = atol = 2^-13`，并通过 `verify_accuracy.py` 全量校验。同时补充参数边界、负步长间隔保护、`OP_C` 等价性及 `alpha=0` 缩放等单测。

### 4.2 性能验证

依据任务附件各变体目录的 `gpu_baseline.csv`（共 1000 条性能用例），在 Atlas 800T A2（CANN 9.1.0）环境下使用 `msprof op`（5 次预热 + 10 次采样算术平均）采集 kernel `Task Duration(us)`，并结合 `aclrtEvent` 通过 `verify_performance.py` 验证逐例满足 `NPU / GPU_baseline <= 1.25`。

### 4.3 兼容性与维护

保持 `blas/gemv_batched/arch22/cgemv_batched_*` 与 `arch35/` 既有文件完全不变，支持 `--ops=sgemv_batched`、`--ops=gemv_batched` 等独立编译与 `blas_all_test` 聚合运行，并通过 `mssanitizer`（`memcheck`、`racecheck`、`initcheck`）0 错误验证。

### 4.4 交付内容

交付 5 个接口的 `arch22` 源码、测试与 CSV、README、设计文档、私有代码仓及验收报告包。

## 五、参考资料

1. 官方任务详情：[11月社区任务-gemvBatched类5个算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)
2. 讨论区来源：`https://gitcode.com/cann/ops-blas/discussions/90`
3. 同算子另一架构参考：`blas/gemv_batched/arch35/gemv_batched_host.cpp`、`blas/gemv_batched/arch35/gemv_batched_kernel.cpp`
4. 同架构高性能参考：`blas/gemv/arch22/sgemv_host.cpp`、`blas/gemv/arch22/sgemv_kernel.cpp`、`blas/gemv_batched/arch22/cgemv_batched_host.cpp`
