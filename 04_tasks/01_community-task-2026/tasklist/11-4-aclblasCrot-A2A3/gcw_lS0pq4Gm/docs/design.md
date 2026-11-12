# aclblasCrot（Atlas A2/A3）算子设计

## 需求背景

本设计对应昇腾任务广场「11月社区任务-aclblasCrot算子开发（A2/A3）」。在 `cann/ops-blas` 的 `arch22` 路径增加 Ascend C 直调实现，支持 Atlas A2（`ascend910b`）和 Atlas A3（`ascend910_93`）。接口语义参考 cuBLAS `cublasCrot` 和 Netlib/LAPACK `CROT`，验收依据为本期任务书及随附数据。

- [昇腾任务广场任务](https://www.hiascend.com/activities/task-center/details/7a42143a91d54df58451707188133781?menu=tasks)
- [任务书和测试附件](https://www.hiascend.com/p/resource/202611/4251fdbf6acd4a7b83cbfd245ae80016.zip)
- [任务讨论](https://gitcode.com/cann/ops-blas/discussions/75)

## 需求分析

在 `include/cann_ops_blas.h` 声明公共接口，内核放在 `blas/rot/arch22/`，CSV 驱动的 GTest 放在 `test/rot/crot/arch22/`。测试构建在 `test/rot/crot/CMakeLists.txt` 的 arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)`。

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n,
    aclblasComplex* x, int incx,
    aclblasComplex* y, int incy,
    const float* c, const aclblasComplex* s);
```

`x` 和 `y` 是 Device 上的 complex64 向量，实部和虚部各用一个 FP32 分量。步长以复数元素为单位，物理存储长度分别为 `1 + (n − 1) × |incx|` 和 `1 + (n − 1) × |incy|`（`n > 0`）。`c` 为 FP32，`s` 为 complex64，两者的指针位置独立识别，支持 Host 和 Device 混用。输出原地写回，跨步留下的存储位置保持原值。

## 详细设计

### 数学与索引

以零起点逻辑下标 `k ∈ [0, n)` 表示配对。步长为 `inc` 的物理下标在正步长时为 `k × inc`，在负步长时为 `(n − 1 − k) × |inc|`。读出该对输入的原值 `X = xr + i·xi` 和 `Y = yr + i·yi`，令 `s = sr + i·si`，其中 `i` 为虚数单位。

| 输出分量 | 计算式 |
| --- | --- |
| x 实部 | `c·xr + (sr·yr − si·yi)` |
| x 虚部 | `c·xi + (sr·yi + si·yr)` |
| y 实部 | `c·yr − (sr·xr + si·xi)` |
| y 虚部 | `c·yi − (sr·xi − si·xr)` |

`y` 的两项使用更新前的 `X`，因此两路输入在写回前均已读出。参考公式为 `X′ = cX + sY`、`Y′ = cY − conj(s)X`。 [Netlib/LAPACK CROT](https://www.netlib.org/lapack/explore-html/da/df6/group__complex__blas__level1_ga4b55e64b6b4f48e1c6e25f3dc53b5b06.html)

### Host 侧

先检查 `handle`，空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。有效句柄下，`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`，`n = 0` 返回 `ACLBLAS_STATUS_SUCCESS`，跳过数据和标量访问。其余调用中，空指针、步长为 `0` 或 `INT32_MIN`、任一路 `(n − 1) × |inc|` 大于 `INT32_MAX` 均返回 `ACLBLAS_STATUS_INVALID_VALUE`。跨度在 Host 端用 int64 计算，内核将逻辑位置转换成 64 位地址偏移。

`aclrtPointerGetAttributes` 分别识别 `c` 和 `s`。Host 标量随 tiling 参数传值，Device 标量在 handle 对应的 stream 内由内核加载。Host 查询可用 AIV 核数后发起一次 kernel 调用，使用 handle 所绑定的 stream。调用者在该 stream 同步后读取结果，Device 标量存储须有效至流上的本次执行完成。内核使用每核 UB 缓冲，算子无需申请 GM 工作区。

### Kernel 侧

连续路径处理 `(incx, incy) = (1, 1)` 或 `(−1, −1)`。两路同时反向时，每个物理位置上的输入配对相同，因此同一套连续地址读写即可。分核以 32 个复数为单元，每个完整 tile 含每路 2,048 个复数。输入和输出队列各有两个 slot，同一 slot 存放 `x`、`y` 两路数据。复用相邻实虚分量的 gather 索引，按块执行乘加。尾块初始化参与向量指令的补齐分量，搬出只覆盖有效字节。双缓冲、交换缓冲、索引和对齐空间共预留 181,248 字节，每核 192 KiB UB 可容纳。

当 `s.imag == 0` 时，连续路径逐块检查两路输入的 FP32 指数位，有限块按实标量更新。若遇到 Inf 或 NaN，从该块起使用完整复数运算，使乘零仍按浮点语义传播非有限值。通用跨步路径按逻辑区间分核，依照 64 位地址逐对精确读写，每个复数搬运 8 字节。每对输入先读完，再写回两个结果。

### 支持硬件

| 硬件 | 构建目标 |
| --- | --- |
| Atlas A2 系列，性能验收设备为 Atlas 800T A2（910B3） | `ascend910b` |
| Atlas A3 系列 | `ascend910_93` |

## 可维可测分析

测试工程通过公共接口运行附件 CSV，精度标杆由 Netlib/LAPACK CROT 语义生成。对 `x`、`y` 的实部和虚部分别按任务书 FLOAT32 混合容差判定，`atol = rtol = 2⁻¹³`，有限元素匹配率至少 0.99，最大绝对误差按任务书的 `1e−2 或 32 ULP` 上限判定。Inf、NaN 独立计数并逐元素判定。同为 NaN、同号 Inf、NaN 与 Inf 配对允许通过，有限值与非有限值配对判为失败。

用例覆盖正负步长、错位、尾块、大尺寸、两路 Host/Device 标量位置组合、非有限值、quick return 和非法入参。负向用例用单元素缓冲或空指针，按预期状态码判定。精度用例检查跨步空洞以及 quick return 下的输出存储。附件原件保留，执行表中的 `TC_ED_123` 至 `TC_ED_126` 按任务书 §2.4 和 §2.5 纠正为 `ACLBLAS_STATUS_INVALID_VALUE`，其中 `TC_ED_123` 的 `n < 0`，其余三条的 `n > 0` 且至少一路步长为零。另补充 982 个 `GAUSS_NORM_5_2` 向量用例，其中标量的均匀和正态采样各 491 组。自测报告按 case ID 对应执行日志。

性能按任务书 §3.3 在 Atlas 800T A2（910B3）采集 `msprof op` 单算子 task duration。每例去掉 5 次 warmup，取至少 10 次成功调用的算术平均值，异常样本记录在自测报告中。按 case ID 对齐附件 `gpu_baseline.csv` 中的 A100 基线，统一为微秒后逐例判定 `NPU_avg_us ≤ GPU_avg_us / 0.8`。附件 Python 脚本读取 GTest 的 `[PERF] avg_us`（`aclrtEvent` 测得，含下发开销），该结果单独保留，便于核对两种计时口径。

arch22 通过仓内公共头文件和测试注册接入，现有其他架构的实现各自沿用对应目录的语义。设计 PR 只提交本设计文档。实现、自测用例与运行说明按本期流程交付。
