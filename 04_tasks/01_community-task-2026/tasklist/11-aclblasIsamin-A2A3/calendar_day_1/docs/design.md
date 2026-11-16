# 需求背景（required）

## 需求来源

[11 月社区任务：aclblasIsamin 算子开发（Atlas A2/A3）](https://www.hiascend.com/activities/task-center/details/b8d4f23bc41c4c58b64414e16dcf9e34)，来源仓库为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。本方案基于该仓 commit `9565ef46a98f82e8884ee084aea96ec62e3e993e` 的公共接口、handle 和测试框架独立设计。

本文按 `04_tasks/01_community-task-2026/resources/design_template.md` 的章节组织，提交账号为 `calendar_day_1`。本次仅提交设计说明；实现代码、自测报告及原始数据按照任务流程另行交付，设计文档的评审或合入不表示任务验收完成。

## 背景介绍

### Isamin 算子能力补齐

`aclblasIsamin` 从 FP32 向量中返回绝对值最小元素最早出现的 1-based 索引。固定基线已存在公共接口及 arch35 实现，本任务新增 Atlas A2/A3 对应的 arch22 实现，不新增平行公共 API，不改变 arch35 或 Icamin 的行为。

### 现状分析

| 项目 | 固定基线能力与新增范围 |
| --- | --- |
| 公共接口 | 复用 `include/cann_ops_blas.h` 中的 `aclblasIsamin` 声明 |
| 工程模式 | Ascend C kernel 直调，Host 校验及调度，Device 计算 |
| 新增实现 | `blas/iamin/arch22/` 的 Host、kernel 和内部启动声明 |
| 新增测试 | `test/isamin/arch22/` 的 CSV 与 GTest，补齐 arch22 注册 |
| 现有参考 | 保留 `test/isamin/isamin_golden.h` 的纯值/索引参考，不使用其校验顺序不同的旧 API wrapper |

### 功能分析

输入为 Device FP32 向量和 Host 标量 `n`、`incx`；输出为独立 Device INT32 标量。正步长按 `x[i * incx]` 访问；相等绝对值取最早位置。NaN、Inf、正负零和次正规数的处理遵循任务书及固定参考实现。

# 需求分析（required）

## 需求描述

实现公共接口 `aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result)`。计算路径以 Ascend C 在 NPU 上完成，支持连续与非连续输入；Host 不拷回输入、不归约数据、不分配每次调用的 workspace、不在接口内部同步 stream。

输出索引必须精确，不允许使用浮点索引导致大于 `2^24` 的下标失真。对连续大向量采用多 AIV 分块和单 AIV 合并；通过真实 `msprof op` task duration 按任务的全部 200 条性能配置逐项验收。

## 需求拆解

1. 按任务书规定顺序处理空 handle、负 `n`、空 `result`、quick return、空 `x` 及跨度溢出。
2. 以整数位序计算 FP32 绝对值的最小 key，保留次正规数、NaN 规则和最早平局。
3. 连续输入采用 16384 元素 tile、整数折半 `Min` 和两级 EQ bitmap；非连续输入采用 NPU 整数标量路径。
4. 大向量复用 handle workspace 存放每核结果，并在同一 stream 上合并。输出仅写 4 字节，保证同地址重复调用的结果可见性。
5. 保留官方精度/性能配置并补齐正态分布、参数及边界测试；精度整数完全相等，性能严格按指定工具和标杆判据判定。

# 详细设计（required）

## 算子分析

### 数学公式

对 `n > 0`、`incx >= 1`，结果为使 `|x[(i - 1) * incx]|` 最小的最早 `i`，其中 `1 <= i <= n`。`n == 0` 或 `incx < 1` 时输出 0；`n < 0` 是非法参数。

首元素为 NaN 时，参考算法将比较种子设为 `FLT_MAX`，初始索引保留为 1；后续 NaN 不更新结果，后续值仅在绝对值严格更小时更新。例如 `[NaN, +Inf]` 和 `[NaN, FLT_MAX]` 均返回 1，全 NaN 也返回 1。此语义不能简化为先删除全部 NaN 再做普通 argmin。

### 支持数据类型

| 参数 | 内存位置 | 类型 | 作用与约束 |
| --- | --- | --- | --- |
| handle | Host | `aclblasHandle_t` | 非空上下文，提供 stream 及 workspace |
| n | Host | `int` | 元素个数；负数非法 |
| x | Device | FP32 | 一维输入，只读；需要计算时不得为空 |
| incx | Host | `int` | 元素步长；小于 1 时 quick return |
| result | Device | INT32，shape `[1]` | 独立输出；任意路径均不得为空 |

本任务只有 FP32 输入，不由测试框架可识别的其他 dtype 推导算子支持能力。

### 支持形状

逻辑输入为长度 `n` 的一维向量，正步长计算需要物理存储至少 `1 + (n - 1) * incx` 个元素。Host 以 64 位整数计算末元素偏移，并要求其不大于 `INT32_MAX`；Device 地址和步长偏移使用足够宽的整数，不把该元素偏移约束误当成 32 位字节地址。

无广播、无图模式动态 shape、无 0 维 Tensor 输入。`n`、`incx` 是运行时参数，合法规模由同一接口处理。`result` 不与输入重叠，不返回输入视图。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

参数检查依次为：

1. `handle == nullptr`：返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n < 0`：返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `result == nullptr`：返回 `ACLBLAS_STATUS_INVALID_VALUE`，包括 quick return 情况。
4. `n == 0 || incx < 1`：在 handle stream 上通过 `aclrtMemsetAsync` 将 Device `result` 的 4 字节写为 0；不检查或读取 `x`，不启动计算 kernel。ACL 提交失败映射为执行失败。
5. 其余路径要求 `x != nullptr`；用 `int64_t(n - 1) * int64_t(incx)` 检查末元素偏移，超出 `INT32_MAX` 时拒绝。

检查成功后异步下发 kernel。成功仅表示 Host 已提交；读取结果前由调用者同步 stream。输入、输出及 workspace 在异步执行完成前必须保持有效。

##### 1. 分核策略

`incx != 1` 使用一个 AIV 的 NPU 标量路径。连续输入 `n <= 16384` 采用一个 AIV 并直接写输出。其余情况以 `ceil(n / 16384)` 为期望 block 数，取其与平台实际 AIV 数、64 的最小值；实际 AIV 数查询为 0 时返回内部错误。

令总 tile 数为 `ceil(n / 16384)`，第 `b` 个 block 处理从 `floor(totalTiles * b / blocks)` 到 `floor(totalTiles * (b + 1) / blocks)` 的连续 tile 区间。多核各自形成局部最优记录，再启动一个合并 kernel；同一 stream 保证这两个 kernel 的顺序。

##### 2. 数据分块和内存优化策略

tile 大小固定为 16384 个 FP32 元素。各 tile 的有效归约宽度取不小于有效元素数、且至少为 64 的 2 次幂。输入 UB 先填不可能胜出的 key 哨兵，`DataCopyPad` 只复制有效输入字节，尾部填充不读取 GM 越界地址。

归约 kernel 的局部缓冲区如下：

| 缓冲区 | 字节 |
| --- | ---: |
| 输入位模式，16384 个 uint32 | 65536 |
| 折半工作区 A，8192 个 int32 | 32768 |
| 折半工作区 B，8192 个 int32 | 32768 |
| 元素 bitmap | 2048 |
| bitmap 二级摘要 | 64 |
| 写回记录 | 32 |
| 合计 | 133216 |

以上合计小于 A2 目标的 184 KiB 用户 UB。此方案为单输入缓冲，使用显式事件屏障保证 MTE2、向量、标量与 MTE3 的依赖；不声明 DMA 双缓冲重叠。

多 block workspace 从 handle 获取，每个 block 使用 32 字节，最多 2048 字节。例如 40 个 block 需要 1280 字节。空 workspace 或容量不足时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。算子不改变 workspace 所有权，不进行每次调用的申请释放。调用者应保证相同 workspace 的使用满足 stream 顺序；不承诺跨 stream 并发复用同一 workspace 的安全性。

##### 3. tilingkey规划策略

本任务采用 kernel 直调，不新增图模式 TilingKey、TilingContext 或算子原型。Host 传入 `n`、`incx`、`blocks` 和 workspace 指针，Device 根据 `incx` 选择连续归约或步长标量路径；`blocks` 决定是否需要独立合并 kernel。只使用既有公共 C ABI，内部 kernel 启动声明不作为额外公共接口。

#### 3.2.2 kernel侧设计

**连续输入的精确 key。** FP32 原始位模式通过整数移位清除符号位，得到 `0..0x7fffffff` 的非负 key。对有限数、正负零和无穷大，该 key 与绝对值顺序一致。`key > 0x7f800000` 对应 NaN。全过程不对原值做浮点算术或近似转换，因此保留次正规数顺序；所有有效 key 均可由非负 int32 精确表示。

**整数最小值归约。** 两个工作区交替执行 `Min<int32_t>` 折半，剩余 8 个元素后用整数标量求最小值。tile 最小值为 NaN key 时不更新。包含首元素的 block 以首元素 key 初始化；若其为 NaN，则替换为 `FLT_MAX` key 并保留索引 0，精确遵循参考算法的特殊种子。

**两级 bitmap 找最早索引。** 只有 tile 中出现严格更小的有效 key 时，才对输入 key 使用 `CompareScalar<int32_t>(EQ)` 产生元素 bitmap。将 bitmap 的 int32 word 再做 `EQ 0` 生成摘要；摘要 word 按位取反后，第一个置位指出第一个非零 bitmap word，再取该 word 的最低置位。最大扫描 16 个摘要 word 和 1 个元素 bitmap word，不使用 int32 NE/LT 模式或浮点索引。

短 tile 的 bitmap 对齐范围先清零，再执行全重复宽度的摘要比较，避免读取未初始化填充。有效最小 key 不可能等于输入 padding 哨兵，并显式检查局部索引小于有效元素数。tile 按升序遍历且仅严格改善时更新，保留核内最早平局。

**核间合并与输出。** 每核在独立 32 字节槽位写入最佳 key 与 0-based 全局索引。合并 kernel 在 NPU 上按 `(key, index)` 的字典序选优，再写回 1-based 结果。单 block 直接写输出。最终输出通过 MTE3 只拷贝 4 字节，并等待相应写回依赖，避免缓存标量 GM 写导致同地址重复调用读取旧结果，也不改写输出前后保护字节。

**非连续路径。** `incx != 1` 时用 NPU 整数标量按 64 位元素偏移读取原始位模式，应用相同 NaN、严格比较和最早索引规则。步长间隔中的值完全不参与归约。该路径不使用 CPU、TBE、其他架构实现或厂商 BLAS fallback。

## 支持硬件

| 硬件 | 适配目标 | 自验证范围 |
| --- | --- | --- |
| Atlas 800I/T A2 | arch22 Ascend C 实现 | 按任务书在 910B3、CANN 9.1.0 自验 |
| Atlas A3 系列 | arch22 Ascend C 实现 | 不用 B3 结果冒充 A3 独立实测；任务书仅要求自验覆盖 910B3 |
| Ascend 950PR / 950DT | 沿用现有 arch35 | 本设计不修改、不扩展声明 |

## 算子约束限制

任务书 §2.5 的逐项核对如下：

| 约束项 | 本方案对应行为与核对结论 |
| --- | --- |
| 参数合法性与校验层级 | 按上述 5 步 Host 顺序检查；result 在 quick return 之前判空，跨度在 64 位整数域计算；元素值不在 Host 校验 |
| 非连续 Tensor 支持 | 支持正 `incx` 的 NPU 步长访问；`incx < 1` 为写 0 的合法 quick return |
| broadcast 规则 | 不涉及，单输入 BLAS-1 向量无广播 |
| dynamic shape 要求 | 不要求图模式动态 shape；运行时 `n`、`incx` 决定 Host 调度，不引入无关 `incy` |
| 原地与视图语义 | `x` 只读，result 独立输出，不复用输入、不返回视图 |
| 确定性计算要求 | 任务不额外要求确定性模式；严格改善与 `(key,index)` 合并保证相同输入的最早平局索引 |
| 空 Tensor 处理 | `n == 0` 不读取 x、不启动计算 kernel，异步写 Device result 为 0；仍要求有效 result |
| 0 维 Tensor 处理 | 不涉及；输入为逻辑向量，Host 标量参数不是 0 维 Tensor |
| 异步执行 | 所有计算及 Device 输出写入在 handle stream 下发；接口内部不做 stream 同步，调用者读取前同步 |

已知限制与规避：

- `incx != 1` 的标量 NPU 路径以语义完整为目标，不能由连续向量性能推导其性能；任务 200 条性能配置均为 `incx == 1`。
- FP32 是本接口唯一输入 dtype。框架的其他 token 不表示本算子支持其他数据类型。
- 对不能提供所需 workspace 的 handle 明确返回错误，不退回 Host 或其他算法。保证 workspace 生命周期并按 stream 顺序使用。
- A3 独立设备实测及最终官方验收结果另行记录，不由架构共用推定已完成。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计与判定口径 | 标准来源 |
| --- | --- | --- |
| 功能及精度 | INT32 输出与固定 CPU 纯值参考逐项完全相等；负向用例精确核对状态码；不使用浮点容差放宽错位索引 | 任务书 §2.1、§2.4、§3.2 |
| 性能 | 全部 200 条原配置逐项满足 `AvgNPU_us <= GPU_baseline_ms * 1000 * 1.25`；使用指定设备及真实 task duration | 任务书 §3.3 与 `gpu_baseline.csv` |
| 样本口径 | 排除 5 次 warmup，至少 10 次有效调用取算术平均；只纳入成功调用，异常样本保留并说明 | 任务书 §3.3 |
| 多 kernel 口径 | 一次 API 的归约与合并 kernel 均计入；核对全 kernel 采集、启动数与调用数，防止只测第一阶段 | kernel 直调设计与任务书单次 API 耗时口径 |
| 内存专项 | 任务书不涉及内存专项；仍测试输出保护字节、输入只读、重复地址和 workspace 合法性，不虚构内存报告测量 | 任务书 §3.4 |

### 测试用例与复现规范

1. 保留官方原始 1200 条 CSV 参数：1000 条精度/参数配置、200 条性能配置。负向及 quick return 测试不按超大跨度分配输入，使用最小 dummy 或空指针；由 API 先校验再访问。
2. 对有效计算域中的 806 条 `RANDOM_NORM_5_5` 精度配置逐一追加对应的 806 条 `GAUSS_NORM_5_2` 配置，保留同一组 shape、stride 与种子覆盖。两个随机分布各占此配对有效随机域的 50%；原有特殊值、负向和 quick return 配置独立保留。仓库 GAUSS 生成路径逐用例选择均值及标准差，并将样本裁剪到 `[-5,5]`，不将其描述为无裁剪正态分布。
3. 2006 条 CSV 加 5 个直接 GTest 共注册 2011 项。精度入口覆盖 1806 条 CSV 非性能配置与 5 个直接测试；200 条 PF 在精度 GTest 中明确跳过，跳过不等于性能通过，必须单独逐项实际采集。
4. 5 项直接 GTest 与独立追加边界检查共同覆盖完整参数检查顺序、64 位跨度、首 NaN/全 NaN/Inf/次正规数/正负零、最早平局、跨 bitmap word/tile/核边界、4 字节偏移视图、输入只读、同地址交替结果、输出 guard、步长间隔值及整数索引 `16777217`。追加边界检查独立记录，不计入上述原有 2011 项注册数。
5. 性能自验使用任务要求的 `msprof op`，保存原始采集文件、每例参数、全部 kernel 的 task duration、预热和有效样本数量、工具与运行时版本、实际库映射，按原始 case ID 逐项核对官方判据。配套脚本的格式及重复 shape 归并局限如实记录，不以其输出代替 200 个原始 CSV case ID 的独立核验。工具兼容修复或组合须明确披露，是否可用于正式验收以任务方意见为准；不以 Host wall time、Event、空 CSV、工具 exit0 或少量通过行替代全量结果。
6. 正式工程补齐 `test/isamin/CMakeLists.txt` 的 arch22 GTest 注册，按原仓标准目标编译及显式分配设备运行，分别保存编译、精度和性能日志。不把独立 kernel/测试入口验证冒充原仓完整标准 target 已通过。

完整精度及 200 项性能的最终结论以原始回执和正式自测材料为准。本设计不预填测试通过数量或最终性能达标结论，不作为验收申请报告。

## 兼容性分析

- 公共函数签名、返回码类型、handle 及 stream 机制保持既有 ABI。仅新增 arch22 实现与测试，不改动 arch35 或 Icamin 源码。
- CANN 9.1.0 为任务目标环境。共享 CSV 测试框架中的可选 ACL 枚举通过当前选定头文件的编译探测控制，不使用版本字符串猜测存在性；不修改 FP32 token、参考计算或未知 token 的异常行为，也不藉此声明新的产品 dtype 能力。
- 不依赖整数到浮点的近似转换、浮点索引或原始输入浮点算术的 FTZ 行为。整数 `Min` 与 int32 EQ 路径使用目标 CANN/arch22 公开接口。
- 实现代码、设计文档合入、标准测试、自验报告及正式验收遵循任务要求的先后流程；报名、评审或提交本身不代表官方验收、奖励确认或到账。
