# 需求背景（required）

## 需求来源

11 月社区任务要求在 Atlas A2/A3 上提供 `aclblasIcamax`。固定基线为 `cann/ops-blas@5bd95145a9c499e43ebad44a313a1029c63d54bc`。[社区任务](https://www.hiascend.com/activities/task-center/details/3187f64ea116468190cabc7779a7cb03)

## 背景分析

公开声明见 `include/cann_ops_blas.h:242`。基线已有 `blas/iamax/arch35/icamax_host.cpp` 及 950PR kernel，`blas/iamax/arch22/` 有 Isamax 的 AIV 数据搬运、每核结果合并和 4 字节结果写回示例。许可证为 CANN Open Software License Agreement Version 2.0。

# 需求分析（required）

## 功能规格

- 输入为交错 FP32 实虚分量。逻辑第 `i` 个元素的物理偏移为 `i*incx`（`i` 从 0 开始）。正常输出 `1+argmax_i(abs(real)+abs(imag))`，相同值选择较小逻辑索引。
- `handle=nullptr` 返回 `HANDLE_IS_NULLPTR`。负 `n`、空 `result`、正常路径空 `x` 或最后一个复数偏移大于 `INT32_MAX` 返回 `INVALID_VALUE`。`n=0` 或 `incx<=0` 在绑定流写 0。首个元素含 NaN 时返回 1，其余 NaN 忽略，多个 `+Inf` 取最早位置。参考 Netlib `v3.10.0@aa631b4b4bd13f6ae2dbab9ae9da209e1e05b0fc:BLAS/SRC/icamax.f:92–120`。

## 性能要求

- 200 个正式性能 case 都是 `incx=1`，GPU 参考 `8.14080003–28.4501333 μs`。每例按实际任务耗时均值对各自 GPU 参考判定，性能上限 `10.1760000375–35.562666625 μs`。

# 详细设计（required）

## 算子分析

公开参数及结果布局与基线 arch35 相同。候选比较可用两个整数字段表达非负浮点模值的位模式和逻辑索引。索引 0 表示无候选，首项 NaN 以优先于正无穷的独立位模式标记。正常模值计算使用 FP32 加法。按逻辑顺序扫描块，跨块严格比较模值可保留第一个并列者。

## 方案实现

### Host 侧设计

`blas/iamax/arch22/icamax_host.cpp` 使用已有 `handle->stream` 和有效 workspace。quick return 用 `aclrtMemsetAsync`。跨度用 `int64_t` 计算。正常 launch 用 `n`、`incx`、核数标量参数。`n<=1024` 单核，较大输入按 `min(ceil(n/1024), AIV核数, workspaceBytes/32, 64)` 分核。workspace 不足两个记录时单核执行。

### Kernel 侧设计

`blas/iamax/arch22/icamax_kernel.cpp` 以 32 字节输入组分核。连续路径默认 4096 个复数一块，`n>=3000000` 用 8192。8192 满块使用按元素数的 `DataCopy` 搬运 65,536 B，尾块以 `DataCopyPad` 搬运真实字节数。`Abs` 和 `PairReduceSum` 计算复数模值，8192 满块的 `PairReduceSum` 拆为 255+1 个 repeat。把非首 NaN 排除后，用 counter 模式 `WholeReduceMax` 和 `GetReduceMaxMinCount` 获取 tile 最大值及最早 lane，复用同仓 arch22 Isamax 的读取方式。每个连续分片首次找到 `+Inf` 后结束扫描，并列时保留更早的逻辑索引。正步长使用从 `uint64_t(index)` 开始计算的 GM 偏移。首项 NaN 的核仍参加多核同步。

单核直接写 4 字节索引。多核分别写 32 字节对齐的 workspace 槽，等待 MTE3 结束后全核 `SyncAll()`，第 0 核顺序合并。连续路径使用单输入缓冲。B4 的 64 KiB 双缓冲和四槽预取在 4M 随机输入分别为 51.886 和 52.058 µs，单缓冲 SPR 对照为 51.218 µs。

## 支持硬件

Atlas A2 `ascend910b*` 与 Atlas A3 `ascend910_93*` 使用 arch22，现有 950PR 实现继续由 arch35 构建。CANN 9.1.0。本次自验证在 Ascend 910B4 执行，使用 5 次 profiler 预热及任务书 §3.3 的冻结逐点门槛。

## 使用约束说明

正常输入要求 `n>0`、`incx>0`，并满足 `(n-1)*incx <= INT32_MAX`。`x` 的设备缓冲应包含最后一个被访问的 COMPLEX64 元素，`result` 应指向可写的 4 字节设备缓冲。输出在 handle 绑定流上异步写入。

# 可维可测分析

## 精度标准与性能标准

- 原始 1000 精度与 200 性能用例保留。精度补充 434 条显式 Gaussian 用例及覆盖首/后 NaN、跨核 tie、未对齐 stride 与非法参数的定点输入。正常输出与 Netlib CBLAS 3.10.0 返回值 `+1` 后精确比对。quick return 和负向检查使用原合同的常量或状态码。
- 固定 GPU 输入 SHA-256 `c15d6c794559f77f1fb6ab2aa1078a37e3983f6f7e9c75aa3cb8538cdbc6001e`。正式采样以 `msprof op` 的 `Task Duration(us)` 为准，5 次预热后每例至少 10 次有效样本，算术平均 `<= 1.25×` 对应 GPU。重复 shape 按稳定 case ID 保留各自门槛。
- 保留采样的重放参数。910B4 上 `TC_PF_1003` 同二进制在 kernel 重放为 `51.160 µs`，application 重放为 `31.670 µs`。官方实现文档注明两种模式的 L2 状态不同。B4 application 采集器 `639c2e1` 用 GTest PID、profile 原始路径、逐序号分析日志与 CSV 核对来源，保留原始 `Pid=NA`。
- `44f253f` 同构建在 Ascend 910B4 / CANN 9.1.0 通过 200/200 性能点、1,451/1,451 精度项和 5/5 长分块探针，取得 3 组内存实测。最大 NPU/GPU 比值 `1.16179033835`。结果和复现步骤在送审代码仓的 `task_submission/`。

## 可维护性分析

Arch22 的实现与测试均独立于现有 arch35 路径。`test/iamax/icamax/arch22/` 保留原始 CSV 和补充 CSV，分开记录冻结用例与扩展用例。冻结性能用例通过 case ID 关联 GPU 基线，重复 shape 仍保留各自的判据。
