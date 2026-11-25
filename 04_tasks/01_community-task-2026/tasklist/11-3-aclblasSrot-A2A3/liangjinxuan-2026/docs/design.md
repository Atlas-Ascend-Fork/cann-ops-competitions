# aclblasSrot（Atlas A2/A3）算子设计

## 1. 需求背景

在 `ops-blas` 的 Atlas A2/A3（arch22）上补齐 FP32 实数向量原地旋转。沿用 `include/cann_ops_blas.h` 中的公开入口，按 [Srot 任务讨论 #74](https://gitcode.com/cann/ops-blas/discussions/74) 对应的 11 月任务书实现。950PR/950DT 的 arch35 Srot 提供接口和标量下发参考，arch22 Csrot 提供搬运参考。

## 2. 需求分析

```cpp
aclblasStatus_t aclblasSrot(aclblasHandle_t handle, int n, float* x, int incx,
                          float* y, int incy, const float* c, const float* s);
```

对每个逻辑下标 `i ∈ [0, n)`，先读取原值 `x0 = x[ix]`、`y0 = y[iy]`，再写入 `x[ix] = c*x0 + s*y0`、`y[iy] = c*y0 - s*x0`。向量是一维 FP32 Device 缓冲区，每侧物理长度至少为 `1 + (n-1)*abs(inc)`。每侧独立计算物理下标：正步长取 `i*inc`，负步长取 `(n-1-i)*abs(inc)`。跳过的元素不写回。以 Netlib SROT 的合法输入计算作为参照。

| 条件（按校验顺序） | 返回值或行为 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n == 0` | `ACLBLAS_STATUS_SUCCESS`，不读取 x/y/c/s、不校验步长、不启动 kernel |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n > 0` 且 x/y/c/s 任一为空，或任一步长为 0 / INT_MIN | `ACLBLAS_STATUS_INVALID_VALUE` |
| 任一 `(n-1)*abs(inc)` 大于 INT32_MAX | `ACLBLAS_STATUS_INVALID_VALUE` |
| 其余合法输入 | 在句柄绑定的 stream 下发计算；读取输出前由调用方同步 |

对每侧先排除 INT_MIN，再提升到 int64 后计算绝对步长和末端偏移。最大合法物理下标仍以 64 位进行元素索引及字节偏移运算，避免将 INT32_MAX 个元素误当成 INT32_MAX 字节。c/s 数值不做校验，不以 c=1、s=0 提前返回，保留非有限值参与运算的行为。

## 3. 详细设计

### 3.1 Host 参数与 tiling

- Host 完成校验后分别用 `aclrtPointerGetAttributes` 识别 c、s。Host 指针在入口读取一次并将值写入按值传递的 tiling；Device 指针随 kernel 参数原样下发，在同一 stream 的 kernel 内读取。两者可以独立来自 Host 或 Device。属性查询失败采用仓内同族的运行时错误处理，不猜测指针位置。
- tiling 包含 n、两侧有符号步长、每核分段信息、UB tile 长度、Host c/s 值及两个 Device 标志。只在 `incx == 1 && incy == 1` 时走连续路径。下发沿用按值 tiling 的直调模式，不为每次调用分配 Device tiling，也不做 Device-to-Host 拷贝或 stream 同步。
- 连续路径按 32 B（8 个 FP32）划分工作单元。核数上限取平台可用 AIV 数与有效单元数中的较小值，前面的核各多领一个余下单元，最后一个单元按实际 n 截断。小向量按有效工作量少启核。非连续路径将逻辑区间按整除和余数分给各核，按上述映射独立计算 x/y 物理位置。

### 3.2 Kernel 计算与搬运

- **连续路径。** 两条输入流水将 x/y tile 搬入 UB，保留原值直至两个输出都算完。按 `Muls(x,c) + Muls(y,s)`、`Muls(y,c) - Muls(x,s)` 计算，复用一块乘积临时区；用事件保护搬入、计算与写回。以五块 FP32 缓冲（两输入、两输出、一临时）估算单缓冲 tile，若双缓冲则同时计入相应队列深度，以实际 UB 容量和对齐向下取整。完整块连续搬运，尾部用 `DataCopyPad` 按实际元素数搬出，避免写越界。用 910B3 数据决定分核、tile 和是否启用双缓冲。
- **非连续路径。** 采用 arch22 AIV。每个逻辑点先读两侧旧值并计算，再从各自 32 B 对齐的 UB 槽按 4 B 有效载荷写回。用分批写回和标量至搬出管线的同步保护槽位复用；不对 stride 间隙做整块覆盖。此路径先保证正、负和混合步长正确，再用真机数据判断是否值得批量搬运。任务包的 200 条性能用例均走双单位步长。
- 按任务书支持 Atlas A2/A3、FP32 ND、运行时 n 和步长、原地输出、同一输入重复调用的确定性计算。无需广播、图模式动态 shape、空 Device 缓冲访问或零维 Tensor。普通合法输入下，各核只更新自己负责的元素；方案以两个独立向量缓冲区为前提，不为重叠缓冲区定义跨元素写入顺序。

### 3.3 工程集成

实现仅加入 `blas/rot/arch22/`，沿用已有公开符号，更新该目录上层 README 的 A2/A3 支持说明。测试加入 `test/rot/srot/arch22/`。当前 `test/rot/srot/CMakeLists.txt` 已调用 `ops_blas_add_gtest_tests(${OPS_BLAS} srot_test)`，复用现有注册并验证 arch22 选中正确源文件。不修改 arch35 实现或其测试契约。

## 4. 可维可测分析

| 项目 | 计划及判定 |
| --- | --- |
| 精度 | 910B3、CANN 9.1.0，运行 1000 条精度用例。对合法输入用 Netlib BLAS 3.10.0 的 FP32 cblas_srot 作 golden。x、y 分别校验：atol/rtol = 2⁻¹³，matched ratio ≥ 0.99，最大绝对误差采用任务书的 1e-2 或 32 ULP 口径。按任务书独立统计 NaN/Inf，不计入有限值匹配率。检查跳步元素和边界哨兵。 |
| 边界 | 覆盖 n=0、负 n、空 handle/x/y/c/s、步长 0/INT_MIN、偏移溢出、正负混合、非对齐尾块，以及 c/s 四种 Host/Device 组合。非法输入以单元素占位，直接校验返回码，不按非法步长分配。n=0 校验输出未改且未启动计算。 |
| 性能 | 在 Atlas 800T A2 910B3 对 200 条 TC_PF 分别执行 `msprof op`，取 5 次预热后不少于 10 个成功样本的 `Task Duration(us)` 均值。按 (n, incx, incy, c, s) 对照 A100 实测基线，将 `gpu_ms` 转为 μs，逐条检查 NPU 均值 ≤ 1.25× 基线。不把 Event 时间作为最终判据。 |
| 内存 | 任务书无独立内存验收阈值。保留 UB 预算与越界检查；按验收模板说明不适用项。 |

随包旧 README/CSV 仍沿用 arch35 的负 n 快返与零步长合法口径。保留原始包供对照，arch22 的期望值按主任务书 §2.4/§2.5 修正；对合法负步长仍按真实物理 span 分配。用确定性生成补齐任务书要求的均匀/正态分布，保留原有 200 条性能参数及基线。提交自验报告时附用例差异、逐条精度和性能原始数据，以及可复现的构建与采集命令。

## 5. 参考

- [NVIDIA cuBLAS 手册](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rot)
- [Netlib SROT](https://netlib.org/blas/srot.f)
- [ops-blas rot 源码与跨架构行为说明](https://gitcode.com/cann/ops-blas/tree/master/blas/rot)
- [CANN 混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
