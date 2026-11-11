# aclblasIcamin 算子设计文档（Atlas A2/A3）

# 需求背景（required）

## 需求来源

CANN 社区任务：使用 Ascend C 在 [ops-blas](https://gitcode.com/cann/ops-blas) 中实现 Atlas A2/A3 的 `aclblasIcamin`。参与账号为 `T350380`。

## 背景介绍

算子查找单精度复数向量中 `abs(real) + abs(imag)` 最小的元素，返回 1-based 逻辑索引。相同模值返回最小索引。例如输入 `[3+4i, 1-2i, -1+2i]`、`incx=1`，输出为 `2`。

复用仓内句柄、stream、构建和 GTest 框架。公共接口已有声明，本任务补齐 `arch22` 实现，不新增平行接口。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle, int n,
    const aclblasComplex* x, int incx, int* result);
```

| 参数 | 位置与类型 | 语义 |
| --- | --- | --- |
| `handle` | Host，`aclblasHandle_t` | 提供执行 stream；空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n` | Host，`int` | 逻辑复数元素数；负数返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | Device，`COMPLEX64` | 实虚部交错存储，各为 FP32；正常计算时须非空 |
| `incx` | Host，`int` | 以复数元素为单位的物理步长 |
| `result` | Device，`int*` | 一个 INT32 输出；任意路径空指针均返回 `ACLBLAS_STATUS_INVALID_VALUE` |

## 需求拆解

1. `n=0` 或 `incx<1` 时，在绑定 stream 上异步置 `result=0`，返回成功，不启动计算 kernel，也不读取 `x`。负 `n` 仍返回非法值。
2. 正常路径检查 `x` 非空，并用 int64 计算 `(n-1)*incx`；超过 `INT32_MAX` 返回非法值。GM 字节地址计算保持 64 位。
3. 支持运行时 `n` 和正步长，输出与 CPU golden 的整数索引逐位一致。调用方同步 stream 后读回结果。
4. 采用向量计算、批量 DMA 和分层归约，避免逐元素标量搬运与归约。

# 详细设计（required）

## 算子分析

### 数学公式与数据类型

对零起始逻辑位置 `j=0..n-1`：

```text
m[j] = abs(x[j*incx].real) + abs(x[j*incx].imag)  // FP32
result = 1 + 最早达到最小模值的 j
```

输入为 `COMPLEX64`，输出为 INT32 标量，不涉及广播、原地更新或矩阵布局。

CPU golden 以首元素初始化候选，仅在后续模值严格更小时更新。首元素模值为 NaN 时结果保持 `1`；其他位置 NaN 忽略。Inf 正常参与比较，全 Inf、全零及相同模值均返回最早索引。kernel 各级归约统一保留该语义，不用 `FLT_MAX` 替代有效的 Inf 候选。

## 算子实现

### Host 侧设计

参数检查依次覆盖句柄、输出指针、负 `n`、quick return、正常输入指针和跨度。quick return 使用 `aclrtMemsetAsync`。正常路径从 handle 取得 stream，计算 tiling、workspace 并提交 kernel，不增加 Host 同步。

Tiling 根据 `n`、`incx`、平台 AIV 核数和 UB 容量选择有效核数及 tile 大小。小向量采用单核，大向量按逻辑范围均分。核数限制在硬同步支持的同驻留范围，避免等待未调度核。每核 workspace 保存局部模值和逻辑索引，另预留同步区域；不生成全量模值中间张量。

### Kernel 侧设计

| 条件 | 数据搬运与计算 |
| --- | --- |
| `n=1` | 直接写回索引 `1` |
| `incx=1` | 批量连续 DMA；`Abs` 后用相邻实虚分量求和得到模值；大 tile 双缓冲重叠搬运与计算 |
| 小正步长 | 搬入连续物理段，用 `Gather` 压紧有效复数，再执行向量模计算 |
| 大正步长 | `DataCopyPad` 按复数块批量搬入 UB，用 `Gather` 压紧行内有效数据，避免逐元素标量加载 |

所有路径按有效长度处理尾块。搬运范围不得越过最后一个有效复数，UB 对齐填充位置不参与归约。Gather 偏移、DMA 字段范围和 UB 占用共同约束 tile 大小，超出单批范围时分批搬运。

每个 tile 先向量求最小模值，再通过相等掩码选择最早的逻辑索引。后续 NaN 以有效掩码排除，Inf 候选保持有效；首元素 NaN 单独形成输出选择掩码。索引始终按 INT32 位表示保存与选择，不转成 FP32，以覆盖大于 `2^24` 的逻辑位置。

多个 tile 的 `(value,index)` 候选先批量写入 UB，再向量合并。各核将最终候选写入独立 workspace 槽位。完成 GM 写入与可见性同步后，所有参与核执行硬 `SyncAll`，由 0 号核批量读取候选并执行同规则的向量归约，最终写回一个 1-based 索引。单核路径直接完成输出，不产生跨核等待。同步状态和寄存器按 Ascend C ABI 要求恢复。

### 工程组织

| 路径 | 改动职责 |
| --- | --- |
| `blas/iamin/arch22/` | Host、tiling 和 Ascend C kernel |
| `test/iamin/icamin/arch22/` | CSV 驱动 GTest、CPU golden 和边界用例 |
| `test/iamin/icamin/CMakeLists.txt` | 在 arch22 分支注册 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)` |
| `blas/iamin/README.md` | 补充 A2/A3 支持范围、接口约束与运行说明 |

沿用同族算子的构建组织。已有公共声明及其他平台实现保持共用接口。

## 支持硬件

| 产品 | 目标架构 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800T A2 / 910B3） | arch22 |
| Atlas A3 系列（含 Atlas 800T A3） | arch22 |

## 算子约束限制

正常路径要求 `n>0`、`incx>=1`、`(n-1)*incx<=INT32_MAX`，并提供覆盖物理跨度的 Device 输入缓冲区。负步长仅执行 quick return，不反向遍历。输入输出使用独立缓冲区。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验证方案与标准 |
| --- | --- |
| 环境 | 按任务书使用 CANN 9.1.0；性能设备为 Atlas 800T A2（910B3） |
| 精度 | 测试工程提供 cblas 风格 CPU 循环，FP32 计算 `abs(real)+abs(imag)`；使用 `EXPECT_EQ(actual,golden)` 比较整数索引 |
| 用例 | 保留原始 1200 条用例，区分精度与 200 条性能用例；补充正态分布、相同模值跨 tile/跨核、首 NaN/后续 NaN、全 Inf、尾块、偏移、跨度溢出及空指针组合 |
| 负向分配 | 非法参数及 quick return 不按大跨度分配输入，使用 dummy 或空指针验证状态码与输出 |
| 性能 | `msprof op` 预热 5 次，有效采样至少 10 次取算术平均；逐例要求 `NPU_us <= 1.25 * GPU_us`，比较原始 GPU 基线并核对单位 |
| 内存与同步 | 独立插桩构建运行 mssanitizer，检查尾块、步长搬运、workspace 与同步状态；核对 error/warning |

精度与性能分开执行。性能采样仅纳入接口成功且 CSV 完整的记录，统计完整 API 的全部 kernel。报告保存源码版本、构建配置、CANN/设备信息、实际样本数和原始结果。本文件描述设计与验收方案，尚无本次真机精度或性能结论。

## 兼容性分析

沿用 `aclblasIcamin` 公共 ABI、句柄和异步 stream 语义，仅新增 A2/A3 路径。连续和步长路径使用相同的特殊值及最早索引规则。路径阈值依据 UB、访存跨度和工作量确定，具体参数经目标设备验证后收敛。
