# 需求背景（required）

## 需求来源

本设计对应《11月社区任务-aclblasIcamin算子开发（A2A3）》任务书，面向 Atlas A2/A3 系列产品，在 ops-blas 工程中使用 Ascend C 实现 `aclblasIcamin`。采用句柄绑定 stream 的 kernel 直调方式，复用公共 BLAS 接口和 workspace 管理机制。

文档遵循社区任务仓 `resources/design_template.md`，覆盖接口语义、Host/Tiling、Kernel、硬件限制以及可维可测设计。实现归属 `blas/iamin/arch22/`，测试工程归属 `test/iamin/icamin/arch22/`。

## 背景介绍

### aclblasIcamin 算子实现目标

复数向量极值索引是 BLAS 一级运算，可用于复数信号处理、数值算法中的元素选择。算子返回复数分量绝对值之和最小的元素位置，不返回最小模值，也不计算平方和开方。

公开入口复用 `include/cann_ops_blas.h` 的已有声明。`aclblasComplex` 使用 `include/cann_ops_blas_common.h` 的定义，实部和虚部均为 FLOAT32，按交错格式存储。A2/A3 实现通过 arch22 构建路径接入，与 arch35 实现共享接口而不重复定义私有 API。

### 同族实现与功能分析

`blas/iamin/arch35/` 的同族归约可供接口组织参考；A2/A3 的向量指令、搬运和归约接口按 arch22 能力设计，不能直接假设其他架构的寄存器指令可用。

| 对象 | 类型及位置 | 功能与约束 |
| --- | --- | --- |
| handle | Host 句柄 | 提供设备上下文、stream 和有效 workspace |
| n、incx | Host INT32 标量 | 指定逻辑长度和复数元素步长 |
| x | Device COMPLEX64 向量 | 只读，实虚交错，逻辑形状 `[n]` |
| result | Device INT32 标量 | 输出 1-based 逻辑索引；快速返回时为 0 |

普通输入的最小模、并列取最小索引和 1-based 语义对齐 [cuBLAS cublasI&lt;t&gt;amin](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin)。任务书进一步规定 `n < 0` 返回参数错误、输出仅使用 Device 指针，以及首元素 NaN 的 CPU golden 语义；这些任务约束在下文单独定义，不将其扩大解释为 cuBLAS 文档对全部异常行为的保证。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

参数顺序与对标接口一致，不引入额外属性、输出 Tensor 或平行接口。

| 参数 | 规格 | 异常及边界行为 |
| --- | --- | --- |
| handle | 有效 ops-blas 句柄 | 空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 逻辑复数元素数，正常范围 `[1, INT32_MAX]` | 负数返回 `ACLBLAS_STATUS_INVALID_VALUE`；0 进入快速返回 |
| x | ND 一维复数向量；正步长时物理长度至少 `1 + (n - 1) * incx` 个复数 | 正常计算时空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`；快速返回不读取 x |
| incx | 按复数元素计数的步长 | 小于 1 时快速返回；不执行反向遍历 |
| result | 独立的 Device INT32 标量，可写 4 字节 | 任意路径的空指针均返回 `ACLBLAS_STATUS_INVALID_VALUE` |

Host 使用 int64 计算 `(n - 1) * incx`，仅在 `n > 0 && incx > 0` 时计算。超过 INT32_MAX 返回参数错误。后续换算 float 下标及字节地址使用 64 位整数，不因复数展开再发生 32 位溢出。

## 需求拆解

1. 功能：实现 COMPLEX64 到 INT32 的最小索引归约；相同模值选最小逻辑索引，结果与 CPU golden 精确相等。
2. 特殊值：明确首元素 NaN、后续 NaN、Inf、正负零、上溢及非正规数行为，避免把硬件默认浮点行为当作接口契约。
3. 边界：覆盖空向量、负长度、非正步长、空指针、最大地址跨度和 workspace 容量检查。
4. 并行：小向量减少启动开销，大向量按可用 AIV 核数分片；核内分块，核间归约不使用竞争写 result。
5. 性能目标：按照任务书逐用例比较，NPU 平均耗时不超过 GPU 标杆的 1.25 倍。
6. 工程：复用公共声明、handle、stream、workspace 和 CSV/GTest 框架，保留 arch35 构建分支。

# 详细设计（required）

## 算子分析

### 数学公式

采用 C/C++ 的 0-based 逻辑位置 `j` 表示输入访问：

```text
p(j) = j * incx                         0 <= j < n
m(j) = RN_float32(abs(Re(x[p(j)])) + abs(Im(x[p(j)])))
j*   = 最小的、使 m(j) 最小的有效逻辑位置
result = j* + 1
```

`RN_float32` 表示 binary32 最近偶数舍入。结果是逻辑序号，不是物理位置 `p(j) + 1`。不使用欧几里得模、双精度累加或容差比较替代上述计算。

特殊值规则以任务书 CPU 顺序参考为契约：从第一个元素初始化，只在 `m(j) < best` 时更新。

| 场景 | 设计行为 |
| --- | --- |
| 首个逻辑元素任一分量为 NaN | 固定返回 1；该全局条件优先于分块归约 |
| 首元素非 NaN，后续出现 NaN | 后续 NaN 不作为有效候选；不能用各块首元素重新初始化 NaN 种子 |
| 相等的有限模值 | 返回最小逻辑索引，包括跨 tile、跨核的并列情况 |
| 全部有效模值为正无穷 | 保留第一个有效元素；全局首元素非 NaN 时结果为 1 |
| 有限值相加上溢 | 模值为正无穷，参与正常并列规则，不饱和为 FLT_MAX |
| 正零、负零 | 绝对值之后均按零比较，取最小索引 |
| 非正规数及极小正规数 | 按 binary32 语义保留对最小值和并列关系的影响 |

全局首元素规则处理之后，候选统一表达为 `(magnitudeBits, logicalIndex)`。非负且非 NaN 的 FLOAT32 位模式可按无符号整数排序；同值比较索引。无效候选采用 `(+Inf, UINT32_MAX)`，真实 Inf 必须携带有效索引。所有全局索引维持整数表示，避免大于 `2^24` 时经 FLOAT32 转换丢失精度。

### 支持数据类型

输入仅支持 COMPLEX64，即两个 FLOAT32 分量；输出仅支持 INT32。模值在 FLOAT32 域计算，特殊精度路径以 UINT32 位模式实现等价运算。无 FLOAT16、BFLOAT16、COMPLEX128 的类型分支。

### 支持形状

输入为运行时长度 n 的一维向量，输出为单标量。`incx = 1` 表示连续访问，正的非单位步长表示规则抽样。物理地址范围为 `[0, (n - 1) * incx]` 个复数元素，输入间隙不参与归约。无广播、多维 stride、原地更新或额外 leading dimension 规则。`n = 0` 不启动计算 kernel。

计算量为 O(n)，输入有效数据量为 `8n` 字节，输出 4 字节；连续分段搬运的跨步路径还会读取跨度中的间隙。主要开销来自 GM 搬运、向量归约和 kernel 启动，而非 Cube 计算。

## 算子实现

### 实现方案

```text
aclblasIcamin
  -> 参数校验
  -> quick return：同 stream 异步清零 result
  -> Host 分核、容量检查、TilingData
  -> scan kernel：各核 CopyIn -> 模值计算 -> tile/核内归约
       单核：直接输出索引
       多核：每核写独占 workspace 槽
  -> merge kernel：同 stream 读取所有槽 -> 全局最小索引 -> result
```

文件职责如下，公共 API 只保留一个入口。

| 文件 | 职责 |
| --- | --- |
| `blas/iamin/arch22/icamin_host.cpp` | 入参校验、分核、workspace 校验和启动封装调用 |
| `blas/iamin/arch22/icamin_tiling_data.h` | tile 常量及无数组的标量 TilingData 定义 |
| `blas/iamin/arch22/icamin_kernel.h` | Host 可见的直调封装声明 |
| `blas/iamin/arch22/icamin_kernel.cpp` | scan、merge、向量流程及特殊值精确处理 |
| `test/iamin/icamin/arch22/` | CSV 驱动测试、CPU golden 和验证脚本 |

#### 3.2.1 host侧设计：

##### 数据检测与返回顺序

1. 检查 handle；为空立即返回句柄错误。
2. 检查 `n < 0` 和 `result == nullptr`；任一满足即返回参数错误，包括 `n < 0 && incx <= 0` 的组合。
3. 当 `n == 0 || incx < 1`，使用 `aclrtMemsetAsync` 在 handle 的 stream 上清零 result 的 4 字节；不读取 x、不申请本算子临时空间、不启动 scan/merge。提交清零失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
4. 正常路径检查 x 非空，再以 int64 检查跨度上限。负步长已在上一步返回，不对 INT32_MIN 做 abs。
5. 获取平台 AIV 数量与 UB 容量，构造分片；平台不可用或资源参数不满足设计下限时返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。
6. 多核时检查有效 workspace 指针、容量和对齐约束；要求起始地址至少 32 字节对齐，以满足 merge 的整块搬运。指针为空、容量不足或未满足该对齐时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，不提交部分计算。
7. 向同一 stream 顺序提交 scan 和必要的 merge。可同步获知的启动错误映射为执行错误；异步执行错误由调用方同步 stream 时处理。成功提交不表示结果已可在 Host 直接读取。

##### 1. 分核策略：

以最大 tile 容量 `B = 4096` 个逻辑复数作为分核粒度，平台可用 AIV 数记为 A：

```text
C0 = (n <= B) ? 1 : min(A, ceil(n / B))
perCore = align_up(ceil(n / C0), 4)
cores = ceil(n / perCore)
start(c) = c * perCore
count(c) = min(perCore, n - start(c))
```

核心数量通过仓内 `GetAivCoreCount()` / 平台信息获取，不写死具体卡型核数。只启动 `cores` 个非空分片。每核起点按 4 个复数对齐，单位步长下对应 32 字节；输入基地址仍可能有合法偏移，搬运接口必须处理其实际对齐情况。末核按有效数量截断，不访问 n 之外的逻辑元素。

单核路径只启动 scan；多核采用两个独立 kernel，省去 kernel 内全局自旋屏障及最后完成核竞争。

##### 2. 数据分块和内存优化策略：

连续或小步长 `1 <= incx <= 8` 时，核内 tile 的有效复数容量为：

```text
tileComplex = floor(floor(B / incx) / 64) * 64
spanBytes(v) = 8 * ((v - 1) * incx + 1)
```

v 为该 tile 的实际有效数量。该公式限制跨步输入跨度在单份输入缓冲容量以内；64 对齐用于 FLOAT32 向量归约。末块先在 UB 填充无效通道，只搬运 `spanBytes(v)` 个真实输入字节。GM 不为对齐而扩读，UB 补齐通道不作为候选。

输入队列深度为 2，在处理当前块时预取下一块。每批最多收集 32 个 tile 的候选后再进行标量级候选合并，减少向量流水与标量流水的切换。大步长 `incx > 8` 使用稀疏精确路径，只读取每个逻辑元素的两个分量，避免读取跨度内大量无关数据；该路径的耗时需要独立纳入性能覆盖，不能按连续路径带宽估算。

scan 的 UB 预算如下，均为设计分配值：

| 缓冲 | 数量及容量 | 字节数 |
| --- | --- | --- |
| 双缓冲交错输入 | `2 * B * 2 * sizeof(float)` | 65536 |
| 实部/模值 | `B * sizeof(float)` | 16384 |
| 虚部/选择临时值 | `B * sizeof(float)` | 16384 |
| Gather 偏移或索引 | `B * sizeof(uint32_t)` | 16384 |
| 比较掩码 | `B / 8` | 512 |
| ReduceMin 临时空间 | `B * sizeof(float)` | 16384 |
| tile 候选槽 | `32 * 32` | 1024 |
| 输出临时块 | 32 字节 | 32 |
| 合计 | 不含编译器/框架保留空间 | 132640 |

平台信息通过 `GetCoreMemSize(CoreMemType::UB, ...)` 获取，扣除框架保留量后校验上述分配及归约 API 临时空间要求。固定 B 只用于满足该预算的目标配置，不静默使用超出 UB 的 tile。merge 的 UB 需要 `32 * cores + 32` 字节，也在启动前检查。

workspace 按每核一个 32 字节槽布局：第一个 UINT32 存模值位模式，第二个 UINT32 存全局 0-based 索引，其余作为对齐保留区并初始化。需求量为 `32 * cores` 字节，单核路径为 0。每核只写自己的槽；merge 只读取本次 cores 个已完成槽，不依赖前次内容。workspace 复用 handle 管理的 Device 空间，生命周期覆盖两个 kernel；调用期间不得跨 stream 并发复用同一段空间，也不得与 x/result 重叠。

##### 3. tilingkey规划策略：

本算子采用 kernel 直调，不使用图模式 TilingKey 注册；以 TilingData 标量和入口数量分流。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| n | uint32_t | 校验后的逻辑长度 |
| incx | uint32_t | 校验后的正步长 |
| perCore | uint32_t | 对齐后的单核分片容量 |
| cores | uint32_t | 实际启动的非空分片数 |

`cores == 1` 直接输出，`cores > 1` 启动 merge；`incx == 1` 用连续解交错路径，`2 <= incx <= 8` 用 Gather 路径，较大步长用稀疏精确路径。核内少量元素（不超过 8 个）使用精确处理避免向量初始化开销。TilingData 不包含数组、设备指针或动态容器。

#### 3.2.2 kernel侧设计：

##### Init 与 CopyIn

kernel 为 AIV-only，TPipe 在 kernel 入口栈上创建并贯穿 Init/Process 生命周期，通过引用传入计算对象；队列和缓冲随本次调用结束释放。Init 绑定 x/result/workspace，计算核内区间，初始化无效候选和 Gather 偏移。

CopyIn 使用支持精确字节数及非对齐输入的 `DataCopyPad`，源地址按 `uint64(logicalIndex) * incx * 8` 计算。`incx == 1` 以 GatherMask 分离实部和虚部；小正步长以字节偏移 `j * incx * 8` 和该偏移加 4 的 Gather 读取两分量。偏移在 UB 内有界，不能把全局地址截断为 Gather 的 32 位局部偏移。

##### Compute 与局部归约

1. 先检查全局第一个逻辑元素的两分量位模式。存在 NaN 时，各核发布全局索引 0 的一致候选，最终写 1，不将此行为用于其他块的首元素。
2. 对实际有效输入执行 Abs、Abs、Add，生成 FLOAT32 模值，保证上溢为 Inf。
3. 通过 `m == m` 生成非 NaN 掩码，与有效长度掩码共同约束候选。无效位置以正无穷参与值归约，但不能凭无效索引胜出。
4. 使用 `ReduceMin` 获取最小模值，再对等于该值且有效的通道生成局部位置，取最小位置，显式保证并列时选择最早元素。该二次归约可复用虚部缓冲，用 FLOAT32 精确表示 `[0, B)` 的局部位置，以正无穷屏蔽其他通道，再调用 FLOAT32 ReduceMin；恢复为整数之后才叠加全局起点，不将全局索引转换为浮点数。当最小模值为 Inf 时，通过原始分量的 NaN 有效性进行精确筛选，避免 NaN 替换值和真实 Inf 混淆；全部无效则发布无效候选。
5. tile 内局部索引以整数恢复为 `start + tileOffset + localIndex`；tile 间、核间均按模值位模式优先、索引次优先合并。

尾块填充值只存在于 UB。即使全部真实模值为 Inf，也必须让真实元素胜过填充元素；若一个非首块全部是 NaN，该块输出无效候选。禁止将 FLT_MAX 作为无穷哨兵，否则有限上溢和真实 Inf 的次序会被改变。

##### 特殊精度路径

设备浮点单元可能将非正规数刷新为零，进而改变极小模值的排序。精确路径从原始分量位模式清除符号位，处理 NaN/Inf，按指数对齐尾数，保留 guard/round/sticky 位，完成加法、规格化和最近偶数舍入，返回非负 binary32 模值位模式。

小规模和大步长路径使用上述精确运算。向量路径在 tile 的候选模值小于 `2^-102` 且非零时重新精确扫描该 tile；候选为零时先精确检查对应元素，若原始模值非零也重新扫描。真实零可直接作为 tile 最小值。该阈值是保守的精度保护边界，保证可能受输入非正规数影响的最小候选进入补偿路径；不以浮点容差掩盖索引差异。

精确处理只保留有限的标量循环边界：小片段、稀疏输入、特殊精度 tile，以及每批/每核候选合并。连续普通输入使用向量搬运和向量归约，不逐元素从 GM 读取。

##### CopyOut、同步与核间合并

队列 EnQue/DeQue 管理输入搬运可见性；向量算子之间保留必要依赖屏障。UB 尾部初始化后再启动覆盖该区间的 DMA，使用 V→MTE2 事件；读向量归约输出前使用 V→S 事件；标量写候选后使用 S→MTE3 事件。缓冲复用必须在前一消费者完成后进行，不能只依赖 C++ 语句顺序。

单核 scan 使用精确 4 字节写回输出。多核 scan 写各自的独占槽，随后同一 stream 上的单核 merge 执行完整槽读取，MTE2→S 同步后按统一候选规则合并，只由 merge 写 result。多核结果不依赖原子最小值、不通过覆盖写输出裁决并列、不使用核间忙等。

kernel 完成前输入、输出及 workspace 均保持有效；调用方在读取结果或跨 stream 消费前完成 stream 同步或建立事件依赖。

## 支持硬件

| 支持的芯片/产品系列 | 设计支持 | 使用单元 |
| --- | --- | --- |
| Atlas A2 训练系列，含 Atlas 800T A2（910B3） | 是 | AIV、UB、GM、MTE |
| Atlas A3 系列，含 Atlas 800T A3 | 是 | AIV、UB、GM、MTE |

软件基线为 CANN 9.1.0，性能目标设备为 Atlas 800T A2（910B3）。不使用 Cube、矩阵布局转换或仅 arch35 提供的指令。按平台信息检查 AIV 数量和 UB 容量，归约与搬运 API 的具体签名及支持范围以 CANN 9.1.0 为准。

## 算子约束限制

- 仅支持单向量 COMPLEX64、INT32 索引；不支持批量、多维广播、混合精度或 64 位公开索引接口。
- 正常路径要求有效的 x/result Device 指针、正步长和受限物理跨度。实际缓冲区大小由调用方保证，Host 不遍历 Device 内存检查容量。
- result 与 x、workspace 不重叠；输入只读。跨步间隙即使包含更小值或 NaN，也不得影响输出。
- `n < 0` 为参数错误，`n == 0` 或 `incx < 1` 为快速返回；先验证 result，快速返回允许不提供可读的 x。
- 同一 handle 的 stream/workspace 在本次异步执行期间保持一致；并发调用需遵循 handle 的线程与资源隔离约束。
- 索引必须精确，不能使用近似模值或人为容差改变并列关系。首元素 NaN 规则独立处理之后才能使用可并行合并的候选关系。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计判据与方法 | 标准来源 |
| --- | --- | --- |
| 精度 | 每例 result 与 CPU golden 的 INT32 值逐位相等；状态码单独比较，不采用浮点容差放宽索引结果 | 任务书 §3.2 |
| 性能 | 每例 `Avg_NPU <= 1.25 * Avg_GPU`；GPU 基准使用配套 `test_cases/gpu_baseline.csv`，不混用 shape 或步长 | 任务书 §3.3 |
| 采样 | msprof op 模式，5 次 warmup 后采集至少 10 次有效执行，单位 us，取算术平均；异常执行不计入有效样本 | 任务书 §3.3、§7 |
| 内存正确性 | 输入保持只读；输出仅写 4 字节；核内缓冲、尾块和 workspace 无越界；任务书无独立内存性能门槛 | 任务书 §3.4 及接口范围 |

性能归因覆盖完整一次接口计算：单核统计 scan，多核将同次调用的 scan 与 merge task duration 相加后再求平均，不能只取扫描 kernel；Host 初始化、CPU golden 和数据准备不计入设备计算耗时。快速返回单独验证接口行为，不与正常归约混合计算性能比值。

### 测试工程与参考计算

在 `test/iamin/icamin/CMakeLists.txt` 的 arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`，复用 CSV 参数加载方式，保留 arch35 注册。参考实现自行计算复数分量绝对值之和并以严格小于更新最早索引；Netlib BLAS 3.10.0 不提供标准 icamin 例程，不能调用不存在的 `cblas_icamin` 或把其他范数接口作为 golden。

CPU golden 使用 FLOAT32 运算与最近偶数舍入，不启用改变 NaN、Inf、非正规数规则的 fast-math。正常路径初始化首元素后顺序扫描；对负 n、非正步长及空指针组合按 Host 校验优先级产生预期状态与结果。

### 覆盖矩阵

| 维度 | 计划覆盖 | 关注的不变量 |
| --- | --- | --- |
| 长度 | 0、1、2/3/5/7、2 的幂及其前后邻值、4096 邻域、至少 `2^20` 的大向量 | 小向量分流、tile/核边界、尾块和大索引 |
| 步长 | 1、2、3、8、9 及更大正值；0、-1/-2/-3、INT32_MIN | 连续/Gather/稀疏分支、快速返回、无负步长反向访问 |
| 输入分布 | 均匀分布 `[-5,5]` 与正态分布各占随机分布用例的一半；后者 `μ∈[-5,5]、σ∈[0.1,2]`，两分量独立采样 | 普通数据覆盖、随机种子可复现 |
| 填充模式 | 全零、全相等、正负交替、首尾最小、跨 tile/跨核相等最小、跨步间隙放置干扰值 | 并列选最小逻辑索引且忽略间隙 |
| 特殊值 | 首/中/尾 NaN、非首块全 NaN、NaN 与 Inf 混合、全 Inf、有限上溢、±0、非正规数、极小值并列 | 不依赖分块顺序；特殊精度与 CPU golden 一致 |
| 对齐 | 复数对齐但非 32 字节对齐的输入偏移、向量宽度前后长度、最小合法输出缓冲 | DMA 不扩读，补齐通道不获胜，输出不覆盖相邻字节 |
| 负向 | 空 handle/x/result、负 n、快速返回与空 result 组合、跨度等于/超过 INT32_MAX、workspace 不足 | 校验优先级、无非法读取或无界分配 |
| 异步与复用 | 非默认 stream、连续调用使用不同输入、独立 handle/独立 workspace 并发 | 结果依赖 stream 顺序且无历史候选污染 |

补充正态分布使用 `GAUSS_NORM_5_2` 填充规则并纳入精度用例。任务配套全部用例与上述边界补充共同构成覆盖集合；覆盖范围按语义分类，不将设计阈值当成测试结果。

负向及快速返回用例只分配单元素 dummy 缓冲或按条件传空指针，不按非法 n/步长预分配巨大数组；跨度合法但体量很大的边界可在 Host 参数层验证算术，Device 访问测试则必须提供真实合法缓冲，不能用 dummy 触发读取。

### 可维护性

Host 校验、分核、模值计算、候选比较和输出写回分别封装；tile 容量、稀疏路径阈值和槽大小集中定义。候选比较规则由 scan 和 merge 共用，避免两阶段对并列和无效项的解释不同。诊断信息仅描述参数范围、资源容量及运行时错误码，不打印输入向量内容。性能优化调整缓冲或路径阈值时，保持接口语义和覆盖矩阵稳定。

## 兼容性分析

公共签名、复数布局和状态码沿用 ops-blas 定义。arch22 文件按目标架构编译，arch35 继续使用原有实现，防止重复导出同名符号。输入 ND 向量不增加框架图模式注册要求，句柄和 stream 的生命周期沿用仓库机制。

与 cuBLAS 一致的部分为参数顺序、复数模定义、正步长寻址、最小并列索引及 1-based 返回。任务要求的差异为：负 n 返回 INVALID_VALUE，结果限定 Device 指针，NaN 行为依据任务书 CPU golden 明确化。测试框架需要区分这些差异，不能直接将 GPU 的全部异常返回码替换为本接口预期。

## 设计依据

- 《11月社区任务-aclblasIcamin算子开发（A2A3）》任务书：接口、特殊值、精度及性能标准。
- [社区设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)：章节结构。
- [ops-blas](https://gitcode.com/cann/ops-blas)：公共类型、句柄、架构目录与测试接入机制。
- [cuBLAS cublasI&lt;t&gt;amin](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-amin)：普通输入的数学语义与索引约定。
- [Ascend C ReduceMin](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0077.html)：向量归约接口及参数约束；正式实现以 CANN 9.1.0 配套接口为准。
- [GetCoreMemSize](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_1034.html)：平台内存容量查询机制。
