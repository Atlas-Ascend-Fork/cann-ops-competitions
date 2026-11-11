# aclblasSaxpy（Atlas A2/A3）算子设计

| 项目 | 内容 |
| --- | --- |
| 社区任务 | [11月社区任务-aclblasSaxpy算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/6ffdb3f781a044588935d6c7d33c3b96) |
| 需求讨论 | [ops-blas discussion 73](https://gitcode.com/cann/ops-blas/discussions/73) |
| 贡献者 | 2501_91839129 |
| 目标硬件 / 架构 | Atlas A2 / A3，arch22 |
| 开发版本 | CANN 9.1.0，Ascend C，kernel 直调 |
| 数据类型 | FLOAT32 |
| 实现 / 测试目录 | `blas/axpy/arch22/` / `test/saxpy/arch22/` |
| 当前阶段 | ops-blas原生工程A3全量验证完成 |

# 需求背景（required）

## 需求来源

本任务补齐 ops-blas 在 Atlas A2/A3 上的单精度实数 AXPY 能力，复用公开的 `aclblasSaxpy` 接口与 handle/stream 管理方式，为上层提供符合 BLAS 语义的一维向量原地更新操作。

## 功能与设计目标

Saxpy 执行 `y ← alpha × x + y`。每个元素读取一个 x、一个旧 y，再写回一个新 y，逻辑数据流量为 `12n` 字节；运算强度约 `1/6 FLOP/Byte`，因此设计重点是连续访存、合理分核、减少调用资源开销，以及非连续场景的正确地址映射。

本方案采用两条明确路径：

- **连续路径：**DMA 搬运与 SIMD 计算，按规模选核，支持任意长度及精确尾写。
- **通用步长路径：**按 y 的物理顺序分片，使用有界 span 搬运与 UB 内 gather/compute/scatter，支持正负混合步长并保持间隙数据不变。

两条路径均为单 kernel、无 workspace、无每次调用的 Device 内存分配；标量及切分参数按值下发，计算使用 handle 绑定的 stream 异步执行。

## 参考实现与本次增量

在上游 `5bd95145a9c499e43ebad44a313a1029c63d54bc` 基线上，公开接口和 arch35 Saxpy 测试框架可以复用；A2 Caxpy 提供了本架构步长访问与工程组织参考。本任务新增 arch22 实数实现、参数约束、自测接入及文档，不依赖 arch35 的 SIMT 设备代码。

参考前期工作包括 [Saxpy !171](https://gitcode.com/cann/ops-blas/pull/171)、[负步长地址映射 !627](https://gitcode.com/cann/ops-blas/pull/627)、[A2 Caxpy 步长实现 !334](https://gitcode.com/cann/ops-blas/pull/334)。本方案独立组织实数连续/步长内核，不引入复数运算、atomic 写回及额外缓存资源。

# 需求分析（required）

## 接口与数学定义

公共 ABI 沿用 `include/cann_ops_blas.h` 的既有声明：

```cpp
aclblasStatus_t aclblasSaxpy(
    aclblasHandle_t handle,
    int n,
    const float* alpha,
    float* x,
    int incx,
    float* y,
    int incy);
```

虽然公共声明中的 x 为 `float*`，实现不通过 x 路径写入，不改变公开 ABI；x/y 完全相同视图时，共享存储随 y 更新。

采用 0-based 逻辑索引 `i ∈ [0,n)`。以调用者提供的物理数组首地址为基址：

```text
ax = abs(int64(incx))
ay = abs(int64(incy))
px(i) = (incx > 0 ? i : n-1-i) * ax
py(i) = (incy > 0 ? i : n-1-i) * ay
y[py(i)] = alpha * x[px(i)] + old_y[py(i)]
```

负步长从对应物理 span 的尾端起算，x、y 分别映射。分核后仍使用全向量 n，不能把核内长度代入负步长基址公式。

| 参数 | 位置 | 语义与范围 |
| --- | --- | --- |
| handle | Host | 有效库句柄，携带 stream |
| n | Host | int；n≤0 为成功空操作 |
| alpha | Host | FLOAT32 标量指针；n>0 时非空，值不作有限性限制 |
| x | Device，只读 | 物理长度至少 `1+(n-1)*ax` 个 float |
| incx | Host | n>0 时非零且不为 INT32_MIN，正负均支持 |
| y | Device，输入/输出 | 原地更新，物理长度至少 `1+(n-1)*ay` 个 float |
| incy | Host | n>0 时非零且不为 INT32_MIN，正负均支持 |

物理数组长度及字节数采用 `uint64_t` / `size_t`；地址乘法先扩展到 64 位。

## 参数校验与快速返回

顺序固定为：handle → n → 非空指针 → 步长及跨度 → alpha 值 → launch。

```cpp
if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
if (n <= 0) return ACLBLAS_STATUS_SUCCESS;
if (!alpha || !x || !y) return ACLBLAS_STATUS_INVALID_VALUE;
if (incx == 0 || incy == 0 || incx == INT32_MIN || incy == INT32_MIN)
    return ACLBLAS_STATUS_INVALID_VALUE;
int64_t ax = incx < 0 ? -int64_t(incx) : int64_t(incx);
int64_t ay = incy < 0 ? -int64_t(incy) : int64_t(incy);
if ((int64_t(n)-1)*ax > INT32_MAX || (int64_t(n)-1)*ay > INT32_MAX)
    return ACLBLAS_STATUS_INVALID_VALUE;
float a = *alpha;
if (a == 0.0f) return ACLBLAS_STATUS_SUCCESS;
// 按值下发 n/a/stride/分核参数，在 handle->stream 上执行。
```

n≤0 不读取 alpha/x/y、不检查步长、不启动 kernel。n>0 的非法指针或步长不会因 alpha=0 而被放行。合法 alpha=±0 时 y 逐 bit 保持不变，也不读取 x/y；NaN alpha 不满足零值判断，按正常路径计算。

## 约束核对

| 约束项 | 设计行为 |
| --- | --- |
| 非连续访问 | 通过 incx/incy 表达，支持非零正负步长 |
| 空输入 | n≤0 成功返回，y 不变 |
| 原地更新 | y 复用原缓冲；每个元素使用更新前的 x/y |
| x/y 完全相同视图 | 同地址、同步长时先读后写，按同一乘加顺序处理 |
| 其他交叠视图 | 不扩大 BLAS 接口对任意部分重叠的保证；不新增私有错误码 |
| 广播 | 不涉及，一维向量由 n 和步长确定 |
| 动态 shape | n 为运行时参数，一次实现适配合法规模，无图模式依赖 |
| 特殊值 | 不拒绝 Inf/NaN；正常运算遵循 FLOAT32 特殊值传播与验收判定 |
| 确定性 | 无跨核浮点归约，同一元素由唯一执行区间更新 |
| 异步 | 计算提交到句柄 stream；输出读取和缓冲释放由调用方同步 |
| 内存资源 | 无全局 workspace、无 GM tiling 上传、无逐调用 malloc/free |

# 详细设计（required）

## 方案选择

| 路线 | 特点 | 选择 |
| --- | --- | --- |
| 新建 A2 SIMD 连续路径＋有界 span 步长路径 | 直接使用 A2 原语；控制流、UB 和写回所有权清楚 | 采用 |
| 基于完整复数框架裁剪 | 可参考步长公式和同步，但实数操作不需要复数拆交错及复杂派发 | 提取通用经验 |
| 将 arch35 SIMT 作为主内核 | 设备实现依赖另一架构能力 | 不作为 arch22 交付路线 |

连续与步长路径共用数值区间选择：alpha=1使用向量Add，alpha=-1使用向量Sub；一般数值区间使用Muls后接Add，并建立V流水依赖。对潜在乘积溢出、大数抵消或非有限值的tile，以整数保存精确24×24位乘积、符号、保护位和sticky位，执行binary32 FMA及最近偶数舍入。完全同视图仍按原始乘加语义处理。

数值区间选择在UB内用Abs/ReduceMax获取幅度上界，不改变输入、不增加全局workspace。alpha±1直接进入向量快路径；200条原始性能case的alpha均为1，因此该常见参数路径无需归约筛选。精确计算仅用于被筛选出的tile。

独立主机检查以std::fma为参考，覆盖7392738组随机位模式、边界及构造抵消输入，FMA不一致和筛选后任务容差违反数均为0；相同检查通过UndefinedBehaviorSanitizer。设备侧混合极值回归覆盖连续/负stride、多核、跨tile、尾部、NaN/Inf以及x/间隙保持；配合属性补充与重复执行检查，纳入原生1977项精度集合。

## Host 与 tiling

最终工程定义小型 POD tiling，按值传给 kernel；不将 Host 栈地址当作 GM tiling 指针。

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| n | uint32_t | 全向量逻辑长度 |
| incx / incy | int32_t | 带符号步长 |
| alpha | float | Host 一次读取的标量副本 |

当前工程将n/alpha/stride作为标量kernel参数按值传递，无GM tiling表。实际核数通过launch维度提供，路径由dense/stride两个入口区分，tile长度由路径和步长推导。

读取实际可用 AIV 数 C 后，初始分核策略为：

```text
candidate = min(C, max(1, ceil(n/4096)))
blocks = min(candidate, max(1, floor(n/8)))
```

4096 是当前每核工作量阈值，已用于本轮A3全量验证。小规模使用单核，较大规模逐步增加并行度。y 基址非 32B 对齐时采用单核路径，支持合法子视图并简化边界所有权。

以 `B=floor(n/8)` 个完整 32B 块分配工作：

```text
q = B / blocks; r = B % blocks
beginBlock(k) = k*q + min(k,r)
countBlock(k) = q + (k<r)
start(k) = 8*beginBlock(k)
count(k) = 8*countBlock(k) + (k==blocks-1 ? n%8 : 0)
```

所有核的逻辑区间不相交，只有末核接收不足 8 个 float 的尾部。未对齐 y 采用单核时，不需要跨核边界证明。

## 连续 SIMD 路径

派发条件为 `incx==1 && incy==1`。

### UB 布局

采用两个输入队列与一个输出队列，每个队列深度为 2；tile=4096 float：

| 缓冲 | 数量 | 单块大小 | 合计 |
| --- | ---: | ---: | ---: |
| x 输入 | 2 | 16 KiB | 32 KiB |
| y 输入 | 2 | 16 KiB | 32 KiB |
| y 输出 | 2 | 16 KiB | 32 KiB |
| 队列 payload | 6 | — | 96 KiB |
| 数值区间归约scratch | 1 | 16 KiB + 32 B | 16 KiB + 32 B |
| 合计 payload | — | — | **112 KiB + 32 B** |

该预算对应当前原生工程实现；早期原型的三队列预算为96KiB。输出独立分配，减少输入生命周期与输出搬运的耦合，不依赖未经验证的队列别名复用。工具链及队列管理开销不计入上表 payload，正式 tiling 需为其保留余量。

### 内核阶段

```text
CopyIn:  x/y 精确字节 DMA → qx/qy EnQue
Compute: qx/qy DeQue → alpha±1向量Add/Sub，或幅度筛选 → 常规Muls/Add或精确FMA → outLocal
CopyOut: qo EnQue/DeQue → 精确字节 DMA 写 y → FreeTensor
```

队列建立 MTE2→V、V→MTE3 及缓冲复用依赖。双缓冲使输入和输出可在独立流水上调度；实际重叠程度以 profiler 分析为准，不把队列深度本身等同于已经实现满流水。

尾部使用已在 A3/CANN 9.1.0 编译运行的 `DataCopyPad` / `DataCopyExtParams` 重载，按 `validCount*sizeof(float)` 指定字节数，不向 GM 尾部写入补齐值。向量 count 只包含合法元素。

## 通用步长路径

为同时满足负步长、间隙保持和明确的核间写所有权，按 **y 的物理升序坐标 j** 分片：

```text
i(j) = incy>0 ? j : n-1-j
py(j) = j*ay
px(j) = (incx>0 ? i(j) : n-1-i(j))*ax
```

每核的 j 范围沿用前述 8 元素分块。对一个长度为 t 的 tile：

```text
xSpan = (t-1)*ax + 1
ySpan = (t-1)*ay + 1
xBase = min(px(j), px(j+t-1))
yBase = j*ay
reverseX = (incx<0) XOR (incy<0)
```

从 xBase/yBase 加载两个有界 span，在 UB 中按 `reverseX` gather x、按 ay gather y；SIMD 计算紧凑结果后，仅替换 ySpan 中对应的逻辑位置，其余间隙沿用原始值，最后精确字节回写 ySpan。

### span 上限和 UB 预算

两个 span 各不超过 4096 float；三个紧凑缓冲（x/y/out）各不超过 512 float：

```text
maxInc = max(ax,ay)
t = min(512, 1 + floor(4095/maxInc))
if (t>=8) t = floor(t/8)*8
```

span 的最大长度为4096，因此基础 payload 为 `2*4096*4 + 3*512*4 = 38 KiB`，另加`(512+8)*4`字节数值区间归约scratch，总计 **40 KiB + 32 B**。较大步长自动收缩 tile，极大步长退化为单元素访问，仍不分配与完整 span 成比例的 UB。

### 写回所有权证明

1. 多核时 y 基址实际 32B 对齐，各核 j 起点为 8 的倍数，所以 `yBaseAddress + 4*j*ay` 也是 32B 对齐。
2. 某核末逻辑坐标为 e，最大写入位置为 `e*ay`；下一核从 `(e+1)*ay` 开始。两核 ySpan 不交叠，且下一核起点处于独立 32B 边界。
3. 核内 tile 可能因收缩而共享边界 DMA 块，因此通用路径在上一 tile 的 MTE3 完成后，才读取下一 tile 的 ySpan；不直接沿用连续路径的自由双缓冲次序。
4. UB gather、SIMD、scatter、MTE3 之间建立明确依赖。实现采用阶段barrier保证次序，kernel内同步覆盖相邻tile的写回依赖。
5. 非32B对齐 y 单核处理，间隙和保护区通过 bitwise 检查验证。

此策略保留连续 DMA 的工程可行性，避免大规模非连续输入依赖逐元素 GM 往返。完全相同视图先加载当前 tile 的 x/y，再写结果；其他部分重叠不作额外快照承诺。

## 异步与资源生命周期

alpha 在 API 内读取一次；之后只传其数值，因此 Host alpha 生命周期无需延长到 kernel 完成。n、stride 和 tiling 同样由 launch 参数复制，不持有 Host 栈指针。

x/y 由调用方持有到所属 stream 完成。算子不在内部调用 `aclrtSynchronizeStream`，不在返回前释放尚被使用的 Device 资源。校验完成且非零 alpha 的正常路径只提交一个 kernel；Host参数校验通过并执行launch后返回SUCCESS；异步设备错误由调用方同步时处理。

# 可维可测分析

## 精度与边界验证计划

正式验证覆盖随包1000条精度用例，并补充正态分布和以下接口级检查：

| 类别 | 重点检查 |
| --- | --- |
| 尺寸 | 0、负值、1、小质数、2的幂±1、32B/tile/核边界、最大验收规模 |
| 步长 | ±1/±2/±3/7、正负混合、INT32_MIN、0、跨度等于/超过INT32_MAX |
| 指针 | handle/alpha/x/y空指针；n≤0快速返回组合 |
| alpha | ±0、±1、±2.5、小值、大值、Inf、NaN |
| 输入分布 | 均匀与正态覆盖；零、交替、极值、Inf/NaN |
| 内存保持 | x不变；y间隙及前后保护区逐bit保持；合法偏移视图 |
| 流与参数 | 同流不同参数连续提交、Host alpha返回后修改、双流独立调用 |

负向用例先判断预期返回，再分配 dummy 缓冲，禁止在测试层对 INT32_MIN 调用 `abs(int)` 或按溢出跨度预分配。正常负步长计算须分配完整物理 span。

FLOAT32逐元素条件为 `abs(actual-golden) ≤ atol+rtol*abs(golden)`，`atol=rtol=2^-13`；同时满足任务书要求的 `matched_ratio≥0.99` 和 `max_abs_error≤1e-2 或 32×ULP`。非有限值按任务书等效规则单列，不并入有限误差统计。

原生工程同时保留 `cblas_saxpy` FLOAT32参考与FP64计算后收窄的参考，两路分别检查任务规定的匹配率、误差上限和非有限值规则，并记录x、y间隙及保护区结果。原始1000条精度CSV、8条任务指定属性补充、964条成对GAUSS补充及5项原生接口/数值fixture，共1977项已在A3通过。

## 性能验证计划

任务书指定性能自验设备为 **910B3**；本文实测记录对应 **Ascend910_9382（A3）**。每 case 预热5次，至少10次有效采样取算术平均，逐case满足：

```text
NPU_avg_us ≤ gpu_ms * 1000 / 0.8
```

所有200条性能输入为连续步长；同时另外验证一般alpha值的正确性。造数、H2D/D2H、golden和event初始化置于计时窗口外。重复计算采用同一stream和预分配缓冲；每轮正式采样前在窗口外恢复y；每case在计时前执行一次完整精度核对，计时调用逐次检查提交返回码并在末尾同步。

分别保留两种计时结果：任务书指定的 msprof task duration，以及配套验证工具解析的 aclrtEvent avg_us；报告中明确字段和单位，不相互替代。对于单kernel实现，设备侧无需跨kernel累加。

基线关联固定为 `case_name → baseline_id`，同时核对 n/incx/incy/alpha。4M有三条同参数但不同填充的独立case，分别保留对应基线，不按shape覆盖：

| case | baseline_id | NPU上限（µs） |
| --- | --- | ---: |
| TC_PF_1003 | saxpy-base-002 | 43.520001 |
| TC_PF_1028 | saxpy-perf-027 | 43.328001 |
| TC_PF_1029 | saxpy-perf-028 | 43.242667 |

验收报告检查1000精度和200性能的完整case清单，逐项记录结果；缺失结果、未关联基线不能视作通过。

## 原生工程A3验证

已在Ascend910_9382、CANN9.1.0、48 AIV环境完成ops-blas原生CMake编译，参考依赖为Netlib BLAS3.10.0。精度1977项通过，包含任务指定alpha=-1.5/0.1/10、步长7、混合非有限值/溢出抵消、多核及跨tile和重复执行bit保持回归。A2编译目标ascend910b/dav-2201也已构建通过。

1950项有效正长度case分别保存FP64计算后收窄、CBLAS FLOAT32及设备输出，共5850文件、976480464字节；参考落盘回读后参与原生判定，独立归档检查两路均1950/1950通过。两次独立运行全部5850文件的SHA256完全一致。完整二进制保存在代码仓外，逐文件清单及复核结果随标准交付材料归档。

性能使用原始200条case，msprof op BasicInfo、kernel replay按50case一批完成四批采集，共3200条原始Task。每case排除一次精度调用与5次GTest预热，保留10条有效设备Task取算术平均；工具每次另预热5次。200/200满足参考上限和额外5%内部目标，相对参考上限的最小余量16.944%。

容器逻辑设备0对应物理设备2，采集显式使用工具侧设备查询适配，仅在msopprof调用栈中返回采集接口所需的逻辑编号，不改变kernel、launch、频率或原始耗时。适配源码、每批精确命令、完整日志、原始counter及CSV均归档；检查器核对Task ID完整性、case顺序、核数、有效样本和原始基线。采集时的event包含工具开销，不用于性能判定。

独立辅助记录包括四轮无采集event（每case1000次，逐case最差轮均值最小余量36.674%）以及常规task-time（201200原始Task，200000有效样本，最小余量58.704%）；两者均200/200达到5%目标，并按各自采集口径分列。

本轮实测源码对应`0c54a02`，实际源文件SHA256逐项核对一致；源码/二进制摘要、原始任务记录和逐case结果随标准交付包归档。个人fork：[2501_91839129/ops-blas](https://gitcode.com/2501_91839129/ops-blas)，实现分支`feat/saxpy-arch22-20261111`；标准报告放在`delivery/saxpy-arch22-20261111`的`task_submission/`。Ascend-CANN已具有Developer权限。

## 工程组织与交付

```text
blas/axpy/arch22/
  saxpy_host.cpp
  saxpy_kernel.cpp
  saxpy_kernel.h
  saxpy_tiling_data.h
  saxpy_float_math.h

test/saxpy/arch22/
  saxpy_test.cpp
  saxpy_test.csv
  saxpy_required_cases.csv
  README.md
```

复用既有公共声明和CSV参数/参考框架，按最新CMake结构接入arch22测试；README产品表标记A2/A3支持。交付设计说明、自测代码、完整case清单、精度/性能报告、复现步骤及对应日志。代码、测试及详细验证材料由个人fork保存；本设计PR仅保留本篇设计文档。

## 兼容性

不更改其他产品线接口、结构体或arch35设备实现，不引入第三方kernel依赖。新增代码遵循ops-blas贡献规范，单核/多核使用同一数学语义和尾部规则。

## 参考资料

1. [cuBLAS Saxpy](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-saxpy)
2. [Netlib saxpy.f](https://www.netlib.org/blas/saxpy.f)
3. [ops-blas](https://gitcode.com/cann/ops-blas)
4. [社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
5. [生态算子精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
