# 需求背景（required）

## 需求来源

11月社区任务「aclblasIsamin 算子开发（Atlas A2/A3）」任务书。
开发账号：`chilpirate`。实现目标仓库：`cann/ops-blas`，开发基线为 master。

## 背景介绍

仓库已有 `blas/iamin/arch35/isamin_host.cpp` 和 `test/isamin/arch35/`，
本任务补齐 Atlas A2/A3 的 Ascend C kernel 直调实现及 GTest，复用公开的
`include/cann_ops_blas.h` 接口，不改变 arch35 实现。

Isamin 对 FLOAT32 实数向量求绝对值最小元素的首个逻辑位置，返回 INT32、1-based 索引。
任务中的原始用例包包含 1000 条精度用例、200 条性能用例和逐行对应的 GPU 基线。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIsamin(
    aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

| 参数 | 位置与类型 | 语义 |
| --- | --- | --- |
| handle | Host，aclblasHandle_t | 提供 stream、设备上下文及 workspace |
| n | Host，int32 | 逻辑元素数，负数非法 |
| x | Device，FLOAT32 ND 向量 | 正步长下读取 `x[i * incx]`，只读 |
| incx | Host，int32 | 以 float 元素为单位的步长；小于 1 快速返回 |
| result | Device，INT32 [1] | 独立输出；正常为 [1,n]，快速返回为 0 |

正常路径计算 `argmin_{0<=i<n} abs(x[i*incx]) + 1`，并列取最小逻辑下标。
输入 FLOAT32 数值不做容差比较，输出 INT32 必须与 CPU golden 完全相等。

### 状态码与校验顺序

1. handle 为空：`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0 或 result 为空：`ACLBLAS_STATUS_INVALID_VALUE`，包括快速返回时 result 为空。
3. n=0 或 incx<1：在绑定 stream 上 `aclrtMemsetAsync` 写 result=0，成功返回；
   不读 x，不启动计算 kernel，此时 x 可以为空。
4. 正常路径 x 为空，或以 int64 计算的 `(n-1)*incx > INT32_MAX`：`INVALID_VALUE`。
5. 设备核心数获取失败：`INTERNAL_ERROR`；多核 workspace 不足或异步清零下发失败：
   `EXECUTION_FAILED`；正常下发成功：`SUCCESS`。

Host 不解引用 result，不在接口内分配内存或同步 stream。调用方读取结果前同步该 stream。

### NaN、Inf 与并列行为

按任务书 §2.1 和仓内 `test/isamin/isamin_golden.h` 的顺序比较实现：
首元素绝对值为 NaN 时，将初始基准设为 FLT_MAX、下标设为 0；后续 NaN 跳过，
仅当绝对值严格更小时更新。因此全 NaN 返回 1；首 NaN 后只有 Inf、FLT_MAX、NaN 时也返回 1；
首 NaN 后有小于 FLT_MAX 的有限数时，返回其中绝对值最小者的首个位置。
全 Inf、全 FLT_MAX、重复最小值及 ±0 均返回首个位置。

## 需求拆解

- 添加 `blas/iamin/arch22/isamin_{host.cpp,kernel.cpp,tiling_data.h}`。
- 添加 `test/isamin/arch22/` 的 CSV、GTest，并补齐 `test/isamin/CMakeLists.txt` 的 arch22 注册。
- 保留原始 1200 行 CSV 和 GPU 基线，独立钉死数量、哈希与运行集合。
- 补齐正态输入、步长、尾块、NaN、空指针、跨度溢出和 stream 顺序测试。
- 以 msprof op 的 Kernel Task Duration 核验全部 200 条性能，逐例 GPU/NPU≥0.8。
- 按任务书模板提供步骤说明、精度/性能报告和日志，以及标明不涉及的内存报告与日志。

# 详细设计（required）

## 算子分析

### 数据类型与形状

只实现任务要求的 FLOAT32 输入和 INT32 输出。n、incx 为运行时 int32 标量，
正常路径物理长度为 `1+(n-1)*incx`。支持任意正步长，不对输入施加超出 float 自然对齐的要求。
GM 索引使用 uint64，避免合法元素跨度乘字节数后发生 int32 溢出。

### 并行归约的等价性

每核处理连续的逻辑区间，局部维护 `(value,index)`，按 value 更小、或相同 value 下 index 更小选择。
普通核初始值为 +Inf、下标为本核起点；第 0 核若首元素为 NaN，初始值改为 FLT_MAX、下标为 0。
向量计算中 NaN 替换为 +Inf，参与最小值比较的候选不携带 NaN。
本核起点按序递增，跨核相等时选择首核，与全局最早下标一致。

第 0 核的 FLT_MAX 种子保证首 NaN 后仅有 FLT_MAX/Inf 的情形仍保留下标 0；
其他核若存在更小有限值，第二阶段仍可选中，不会把首 NaN 误判为无条件返回 1。

## 算子实现

### Host 与 tiling

从平台获取 AIV 核数 A。设置逻辑分块上限 T=8192、向量对齐 L=64：

```text
wanted   = ceil(n / 8192)
cores0   = min(wanted, A)
perCore  = align_up(ceil(n / cores0), 64)
cores    = ceil(n / perCore)
start(k) = k * perCore
count(k) = min(perCore, n - start(k))
```

使用实际非空核数启动。n≤8192 使用一核直接输出；多核写每核 32 字节的 workspace 槽，
再由同一 stream 上的第二个 kernel 归约。workspace 需求为 `cores*32` 字节，使用 handle 的既有缓冲区。
两阶段之间依赖 stream 顺序，无跨核自旋或 Host 回读。

| incx | 逻辑 tile 大小 | 读入方式 |
| --- | --- | --- |
| 1 | 8192 | DataCopyPad 连续搬运，Abs 后 ReduceMin |
| 2..128 | `floor(8192/incx/64)*64` | 搬运有限物理窗口，Gather 取逻辑元素后 Abs |
| >128 | 64 | uint64 GM 地址逐元素读入紧凑 tile，随后向量 Abs/ReduceMin |

步长路径不会依据大步长分配大 UB。Gather 偏移仅作用于最多 8192 个 float 的局部窗口。
输入双缓冲交错搬运和计算；最多 32 个 tile 的归约结果成批读取到标量，降低流水线切换次数。

### Kernel 与 UB

扫描核使用深度 2 的输入队列、绝对值缓冲、Gather 偏移、比较掩码、归约 scratch、
32 个中间槽和一个输出槽。最坏 UB 分配为：

| 缓冲 | 最大字节数 |
| --- | --- |
| 双缓冲输入 | 2×8192×4=65536 |
| values | 32768 |
| Gather offsets | 32768 |
| NaN mask | 8192 |
| ReduceMin scratch | 32768 |
| 32 个 tile 槽 | 1024 |
| 输出槽 | 32 |
| 合计 | 173088，低于 192 KiB |

搬入长度为 `(count-1)*incx+1` 个 float，不越过最后一个有效元素。
未满 tile 的逻辑位置填 +Inf；大步长分支不读 padding 对应的 GM。
ReduceMin 同时返回最小值及首个局部索引，映射回逻辑下标后比较。
写回结果使用 DataCopyPad 精确写 4 字节，避免覆盖输出相邻内存。

多核最终归约把每核槽中的 value Gather 到连续 UB，ReduceMin 选择首个最小值核，
再读取该核保存的逻辑下标，加 1 后写回 result。

## 支持硬件

| 硬件 | 适配范围 | 验证口径 |
| --- | --- | --- |
| Atlas A2，910B3 | arch22 | 按 §3.1/§3.3/§7 出具正式精度及性能数据 |
| Atlas A3 | arch22 | 支持开发调试，记录运行时实际 SoC，不代替 910B3 数据 |

编译、运行使用 CANN 9.1.0。数据采集前记录 `npu-smi`、`aclrtGetSocName` 和 SDK 版本，
避免把共同的 arch22 编译架构当成相同设备型号。

## §2.5 约束逐项核对

| 约束 | 实现与核对结论 |
| --- | --- |
| 合法性与校验层级 | 按上述先 handle、n/result、快速返回、x/跨度的顺序处理 |
| 非连续 Tensor | 正步长逻辑索引；间隔元素不参与归约；incx<1 写 0 |
| broadcast | BLAS-1 单向量，不涉及 |
| dynamic shape | n/incx 为运行时参数，Host 每次计算 tiling，无图模式要求 |
| 原地与视图 | x 只读，result 独立输出；测试核验输入及间隔字节保持不变 |
| 确定性 | 同值取最早逻辑索引，各归约阶段保持相同规则 |
| 空 Tensor | n=0 不读 x、不发计算核，异步写 0 |
| 0 维 | 向量接口，不涉及 |
| 异步执行 | kernel/清零使用 handle stream；调用内部无同步 |

## 已知限制与规避

不支持 `(n-1)*incx>INT32_MAX` 的跨度，Host 返回 INVALID_VALUE；不把负步长解释为反向遍历。
输入及独立输出应为当前设备的有效地址，并在异步执行完成前保持有效。
调用方自行保证 result 与 x 不重叠；输入值全域由比较规则处理，不进行额外数值合法性检查。
大步长使用标量 GM 读入作为通用路径，性能验收仍逐例采用随包固定基线。

# 可维可测分析

## 精度标准与覆盖

CPU golden 使用仓内 `isamin_golden`，按顺序逐元素计算 abs 和严格小于比较。
Netlib 3.10.0 没有 isamin.f，不以其他 BLAS 接口冒充对应 golden；测试工程仍沿用仓内参考库依赖。

原始 CSV 逐字节保留，SHA256 为
`95c1b1c324e8ad8c6c940a8a67489737ec5c60b15921994dd2d855635fa51373`。
C++ 和 Python 独立检查 1200 条=1000 精度+200 性能、case_name 唯一性；
严格 CSV 解析遇到截断、字段缺失或重复表头立即失败。

补充 884 条：820 条均匀精度用例的正态对应项、48 条契约与边界、16 条成对错位地址用例。
随机精度用例中均匀和正态各 828 条；有效读取输入的随机用例也按两类等量配对。
μ 在 [-5,5]、σ 在 [0.1,2] 采样，记录种子及实际分布参数。
另外提供空 handle 和同 stream 连续提交/快速返回覆盖 2 项，总精度执行数为 1886。

| 覆盖域 | 检查内容 |
| --- | --- |
| 普通及尾块 | n=0/1/2/3、64 对齐边界、8192 分块边界及大尺寸 |
| 跨步 | incx=1/2/7/16/127/128/129/65536，间隔最小值不应被选中 |
| 特殊值 | 首/中/末 NaN、全 NaN/Inf/FLT_MAX、首 NaN 后 Inf/FLT_MAX、±0、subnormal |
| 并列 | 同 tile、跨 tile、跨核最小值选择首个逻辑索引 |
| 负向 | n<0、result=null（含快速返回）、x=null、两类指定跨度溢出 |
| 地址与完整性 | 合法 4 字节对齐的偏移、输入/输出前后 guard、输入及间隔逐字节不变 |

负向和快速返回仅分配单元素 dummy，不按非法 n/incx 分配巨量内存。
Python 逐项校验 XML 的完整执行集合、运行状态、actual/golden 日志及分布证据，
任何丢行、跳过、重复、错误返回或输出不相等均判失败。

## 性能标准与采样

固定 200 条 TC_PF_ 与 200 行 GPU 基线按行一一对应，校验原始基线哈希
`3d3c973201676715fc22e8dbda31115c522a79aa0220d5427c85f763eeb68d62`。
设备为任务书指定 910B3，主指标为 `msprof op` 的 `Task Duration(us)`。
每例先做 5 次 API 预热，再在 MSTX range 内提交 10 次有效 API；工具另设 warm-up=5，均不纳入有效样本。
只计返回 SUCCESS 的调用，多核的 scan+reduce 两阶段逐次求和，再取 10 次的算术平均。
逐例 `GPU_us / NPU_mean_us >= 0.8` 才通过，不以整体均值或 ACL Event 耗时替代。
保留命令、调用日志、XML、每个原始 OpBasicInfo CSV 与样本到原始 CSV 的映射。

## 内存、交付与兼容性

任务书 §3.4 内存验收不涉及，仍按七文件模板保留内存 XLSX 和日志并明确标为 N/A。
输入/输出 guard 和只读检查属于精度自验，不将其包装为额外内存验收报告。

实现只新增 arch22 Isamin 文件并注册现有 GTest target，保留 arch35 入口、测试和共享 ABI。
算子 README 提供构建、精度、性能和复现命令。正式交付前确认设计 PR 已实际合入，
私仓根目录同步七个 task_submission 文件并邀请 Ascend-CANN 为开发者。
代码 PR 按任务平台的官方验收安排提交，不能把开发者自测等同于官方验收通过。
