# 需求背景（required）

## 需求来源

昇腾 CANN 社区任务 08-46 要求在 ops-blas 中实现 Atlas A2/A3 的 `aclblasIcamax`，按任务书提供精度、性能和自验证交付件。[社区任务](https://www.hiascend.com/activities/task-center/details/3187f64ea116468190cabc7779a7cb03)

## 背景介绍

`aclblasIcamax` 是 BLAS Level 1 复数向量归约接口。ops-blas 已有 Ascend 950PR 实现和公共声明，Atlas A2/A3 复用公共接口及 handle、stream、工作区机制，在 `arch22` 增加 Ascend C kernel 直调实现。功能对齐 cuBLAS `cublasIcamax` 和 Netlib `ICAMAX`。[Netlib ICAMAX](https://www.netlib.org/blas/icamax.f)

# 需求分析（required）

## 需求描述

输入为 Device 侧 COMPLEX64 向量、Host 侧逻辑长度 `n` 与步长 `incx`。沿 handle 绑定的 stream 计算复数 1-范数最大的逻辑元素，向 Device 侧 `int32` 标量写入最小的 1-based 索引。调用者同步 stream 后读回结果。

## 需求拆解

1. 复用 `include/cann_ops_blas.h` 中的声明，处理空指针、负长度、零长度、非正步长及地址跨度溢出。
2. 提供连续与正步长两条向量路径，覆盖尾块、未对齐输入、并列最大值、NaN 和 Inf。
3. 在 `test/iamax/icamax/arch22/` 注册 GTest，用官方及补充 CSV 验证精确索引，用 910B3 逐项对照官方 GPU 耗时。

# 详细设计（required）

## 算子分析

### 数学公式

正常路径对每个逻辑索引 `i = 1…n`，在物理位置 `k = 1 + (i - 1) * incx` 读取复数并计算 `m(i) = abs(Re(x[k])) + abs(Im(x[k]))`。返回使 `m(i)` 最大的最小 `i`。第一项模值为 NaN 时返回 1，否则跳过后续 NaN。多个 Inf 或有限最大值并列时取首项。

### 支持数据类型

| 参数 | 数据类型 | 地址空间 |
| --- | --- | --- |
| `handle` | `aclblasHandle_t` | Host |
| `n`、`incx` | `int` | Host |
| `x` | `const aclblasComplex*`，实部和虚部以 FP32 交错存储 | Device |
| `result` | `int*`，单个 INT32 索引 | Device |

### 支持形状

`x` 为逻辑形状 `[n]` 的一维 ND 向量。计算路径需要 `1 + (n - 1) * incx` 个物理复数元素。输出为标量，正常范围为 `[1, n]`，快速返回写 0。

## 算子实现

### 实现方案

`blas/iamax/arch22/icamax_host.cpp` 处理入口、分核参数与 stream 下发。`icamax_kernel.cpp` 使用 AIV 完成分块向量归约，在一次 kernel 下发中完成核间合并。

#### 3.2.1 host侧设计

| 校验顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` 或 `result == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `n == 0` 或 `incx <= 0` | 在绑定 stream 异步写 0。写入失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 4 | `x == nullptr` 或 `(n - 1) * incx > INT32_MAX` | 返回 `ACLBLAS_STATUS_INVALID_VALUE`，乘积以 `int64_t` 计算 |

##### 1. 分核策略

`n <= 4096` 使用单核。更长的输入按 `min(AIV 核数, ceil(n / 4096))` 选核，AIV 数量查询为 0 时返回内部错误。每核对应连续逻辑区间，连续输入的中间分界按 64 个复数即 512 字节向下对齐。相邻核使用相同分界，最后一核处理余数。

##### 2. 数据分块和内存优化策略

多核归约使用 handle 的有效工作区，每核 32 字节，保存模值和全局索引。工作区不足或未分配时用单核完成全向量归约。连续路径每 tile 8,128 个复数，步长路径每 tile 512 个复数，均使用两个输入缓冲区。连续输入达到 32 MiB 时禁用该输入的 L2 缓存分配。尾块只搬运真实输入，UB 按向量宽度补齐。

##### 3. tilingkey规划策略

直调参数携带 `n`、`incx`、核数、工作区和 stream。设备端依长度与步长选择单元素、连续、可编码步长和超大步长路径。步长字节间隔超过 MTE2 的 `uint32_t` 表示范围时，合法输入跨度限制了元素数，使用逐项读取。

#### 3.2.2 kernel侧设计

1. 搬入阶段以双缓冲交叠 DMA 与向量计算。连续输入按实际有效字节执行 `DataCopyPad`。正步长输入按复数间隔搬运，UB 中每个复数占一个 32 字节块。
2. 连续路径用 `Abs` 和 `PairReduceSum` 求模值。步长路径用 `Gather` 提取实部、虚部，分别取绝对值后相加。NaN 掩码保留首项特例，其余 NaN 的比较值置为 -1。
3. `ReduceMax` 在有效元素范围内产生块最大值和局部索引。每批最多 16 个 tile 共用一次向量到标量同步，按逻辑顺序以严格大于更新获胜索引。
4. 多核把局部候选写到各自工作区，`SyncAll` 后由核 0 按核序合并，严格大于才更新。首元素 NaN 时直接保留索引 1。结果只写回 4 字节。

## 支持硬件

| 支持的芯片版本 | 验证范围 |
| --- | --- |
| Atlas A2，含 Atlas 800T A2 910B3 | CANN 9.1.0 真机精度与性能验证 |
| Atlas A3，含 Atlas 800T A3 | `ascend910_93` 目标编译与链接验证 |

## 算子约束限制

仅支持 COMPLEX64 一维输入和独立 INT32 输出，不涉及广播或原地更新。非正步长走快速返回。计算路径的逻辑跨度须能以 INT32 表示，设备侧字节地址使用 64 位计算。跨核归约结果遵守首个最大值语义。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 与 Netlib BLAS 3.10.0 `cblas_icamax` 的 0-based 结果转换后的 1-based 索引逐项精确相等。快速返回和异常输入检查输出及状态码。 | 任务书 §3.2、§3.5 |
| 性能 | 910B3 用 `msprof op` 为 200 条官方用例逐项采集 `Task Duration(us)`，预热 5 次后取 16 个有效样本求平均。按 `(n, incx)` 在官方标杆中的出现顺序配对，GPU 毫秒转微秒，逐条报告 NPU 均值、GPU 标杆、GPU/NPU 与原始样本。通过线为 0.8。 | 任务书 §3.3、`test_cases/gpu_baseline.csv` |
| 内存 | 交付模板的内存报告与日志注明不涉及。 | 任务书 §3.4 |

精度覆盖 1,000 条官方用例、434 条补充正态分布用例和 4 项工程检查，包含未对齐输入、跨核并列、非法参数、NaN/Inf 及非规格化浮点。正态输入实部与虚部独立生成，使用固定种子。性能用例也逐条核对输出索引，保留完整运行日志。

## 兼容性分析

公共 ABI、状态码和 stream 语义沿用 ops-blas。A2/A3 按 `arch22` 注册独立实现与测试，Ascend 950PR 继续使用 `arch35` 路径。两个平台分支使用同一公共声明。
