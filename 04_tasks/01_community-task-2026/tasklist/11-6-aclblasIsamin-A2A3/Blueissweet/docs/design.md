# aclblasIsamin 算子设计文档（Atlas A2/A3）

## 1. 需求背景

本任务对应 2026 年 11 月 CANN 社区任务第 6 项。`ops-blas` 已提供 `aclblasIsamin` 的公开声明和 950PR（arch35）实现。本设计补齐 Atlas A2/A3（arch22）上的实现和测试，复用公开接口，以 cuBLAS `cublasIsamin` 和仓内 arch35 实现为语义参考。

## 2. 需求分析

在单精度实数向量中查找绝对值最小元素的第一个逻辑位置。对 `n > 0` 且 `incx > 0`，以 0 为起始下标访问 `x[k * incx]`，其中 `0 <= k < n`。在符合任务书 NaN 规则的候选中最小化 `(abs(x[k * incx]), k)`，写出 `k + 1`。正常计算的有效索引范围为 `[1, n]`。

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

| 参数 | 约束 |
| --- | --- |
| `handle` | Host 上的库句柄，提供 stream 和工作区。 |
| `n` | INT32 逻辑元素数。 |
| `x` | Device 上的 FP32 一维 ND 向量，计算路径的有效跨度为 `(n - 1) * incx + 1` 个元素。 |
| `incx` | INT32 元素步长，正值用于计算。 |
| `result` | Device 上独立的 INT32 输出，形状为 `[1]`。 |

需求拆分为 Host 校验与快速返回、按 AIV 分核、连续及非连续向量的核内归约、跨核稳定合并，以及与 CPU golden 和 A100 基线的对比验证。

## 3. 详细设计

### 3.1 Host 侧

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。`n < 0` 或 `result == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. `n == 0` 或 `incx < 1` 时，在句柄绑定的 stream 上调用 `aclrtMemsetAsync` 写出 0。`0` 是快速返回的哨兵值，表示本次调用无元素位置可用，位于正常计算的有效索引范围之外。此路径允许空 `x`，不启动计算 kernel。写入提交失败时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
3. 计算路径用 INT64 计算 `(n - 1) * incx`。`x == nullptr` 或末元素偏移超过 `INT32_MAX` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `n <= 2048` 使用单个 AIV。更大规模使用 `min(可用 AIV 数, ceil(n / 2048))` 个 AIV。按每组 64 个逻辑元素分配完整组，前若干核各多领一组，末核处理不足一组的尾部。多核路径要求句柄有效工作区至少为 `参与核数 × 32` 字节。

以 `IsaminTilingData` 按值传递 `n`、`incx`、每核基础组数和额外组数，核数由启动参数指定。使用同一个 kernel 入口，在核内按步长和区间长度选择执行路径。

### 3.2 Kernel 侧

| 场景 | 核内路径 |
| --- | --- |
| 区间长度不超过 16，或 `incx > 16` | 按逻辑索引标量扫描。 |
| `incx == 1` 的向量路径 | 每 tile 最多 16,320 个元素，通过 `DataCopyPad` 搬运有效长度后做向量归约。 |
| `2 <= incx <= 16` 的向量路径 | 以 64 个逻辑元素为单位缩小 tile，搬入有效物理跨度，在 UB 中 `Gather` 出参与比较的元素。 |

向量路径清除 FP32 符号位，用幅值的整数编码保留顺序。首轮将 NaN 的幅值编码映射为 Inf 的编码，将低端编码提升至下界，再减去一个指数步长，使 `ReduceMin` 的比较键落在正的正规 FP32 范围内。首轮命中下界时，若候选幅值为零可直接使用该索引，否则重新编码低端区间并归约，以保留次正规数的顺序。

每轮使用带索引的 `ReduceMin(dst, src, tmp, count, true)`，最后一个参数为 `calIndex`。输出的第一个元素是最小键，第二个元素保存 tile 内的 0-based 索引位模式。`ReduceWithBias` 用 `ReinterpretCast<int32_t>()` 读取第二个元素，并加上 tile 起始逻辑位置。多个元素的最终比较键相等时，该 API 返回其中最小的索引。比较键只编码幅值，索引由 API 单独输出。tile 和核间按逻辑顺序合并，仅当幅值严格减小时更新候选。[Ascend C ReduceMin API](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMin.html#约束说明)

全 NaN 输入返回 1 的依据是本任务书 §2.1 的 NaN 规则。各区间的初始候选幅值为 Inf，索引为区间起点。全 NaN 时，向量路径归约得到的候选幅值为 Inf，标量路径的 NaN 幅值位模式大于 Inf。两条路径的候选均保持初始值，按区间顺序合并后保留全局逻辑位置 0，最终写出 1。

多核以 batch 调度。`ISAMIN_WORKSPACE_SLOT_BYTES` 取一个 Ascend C 搬运 DataBlock 的大小（32 字节），作为各核在 GM 中的槽间距，并与汇总时 UB 中的块布局一致。每个核通过 `DataCopyPad` 写入槽内前 8 字节（32 位幅值和 32 位逻辑索引），然后执行 `SyncAll`。第 0 核按 32 字节间距从 GM 逐块读回这 8 字节，读入 UB 时在右侧补齐 24 字节。第 0 核按区间顺序合并候选并写出 4 字节结果。单核直接写回结果。[Ascend C 搬运基本概念](https://asc.gitcode.com/api/SIMD-API/basic_api/data_move_guide/overview/data_move_concept.html)

### 3.3 硬件与约束核对

设计目标为 Atlas A2/A3（arch22）。公开接口与 arch35 实现共用声明。自验证使用 `ascend910b3` 编译目标，采集记录标识为 WebIDE 容器设备 0，材料中暂无独立的设备型号截图。

| 任务书 §2.5 约束 | 处理方式 |
| --- | --- |
| 参数合法性和返回码 | 校验顺序、空指针规则及跨度上限见 §3.1。输入元素值无需校验。 |
| 非连续 | 支持所有合法正步长。大步长标量路径以正确性为主。 |
| 广播 | 一维单输入归约，无广播。 |
| 动态 shape | 每次调用使用运行时 `n` 和 `incx` 计算 tiling。图模式动态 shape 不在任务范围内。 |
| 原地与视图 | `x` 只读，结果写到独立缓冲区。 |
| 确定性与特殊值 | 相等绝对值取最先出现的位置，`+0` 和 `-0` 的幅值相等。按任务书 §2.1，首元素为 NaN 时以 `FLT_MAX` 为严格更新阈值，后续 NaN 不更新候选。全 NaN 时的索引推导见 §3.2。 |
| 空 Tensor 与 0 维 | `n == 0` 走快速返回。输入只支持一维向量。 |
| 异步执行 | Host 返回成功表示提交成功。调用方同步对应 stream 后再读 `result`。 |

调用方通过 `aclblasSetWorkspace` 替换工作区时，需预留每个参与核 32 字节。性能基线使用 `incx = 1`，非连续访问由精度用例覆盖。

## 4. 可维可测分析

实现放在 `ops-blas` 的 `blas/iamin/arch22/`，测试放在 `test/isamin/arch22/`，通过 `test/isamin/CMakeLists.txt` 注册 arch22 GTest 目标。

- **精度**。成功路径将 INT32 索引与 CPU golden 逐项精确比较，快速返回检查输出 0，负向测试检查状态码。验证覆盖连续及非连续输入、并列最小值、NaN、Inf、次正规数和边界参数。1,000 条 CSV 精度用例中，403 条使用均匀分布，403 条使用正态分布，其余覆盖固定填充、快速返回和负向场景，另有 5 项 GTest 边界测试。
- **性能**。对 200 条 `TC_PF` 用例逐条采集 `msprof op` 的 kernel `Task Duration(us)`，每条预热 5 次并取 10 个有效计时的平均值。逐条将 NPU 均值与匹配的 A100 基线比较，要求 NPU/A100 耗时比不超过 1.25。
- **内存**。按任务书 §3.4，无内存自验证用例。多核工作区大小按参与核数计算并在 Host 校验。
- **兼容性**。复用 `include/cann_ops_blas.h` 的原有声明、handle 的 stream 及工作区机制。arch22 和 arch35 分别构建。测试通过独立精度和性能入口复现。

2026-11-12 在 CANN 9.1.0 的 WebIDE 环境以 `ascend910b3` 构建，精度 1,005/1,005 项通过，性能 200/200 条通过，共取得 2,000 个有效计时。最慢的相对比值为 1.156430（`TC_PF_1147`）。复现步骤、日志及逐项数据随验收提交。
