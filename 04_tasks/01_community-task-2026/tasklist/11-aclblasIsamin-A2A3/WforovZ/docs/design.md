# aclblasIsamin 算子设计文档（Atlas A2/A3）

> 任务：11月社区任务-aclblasIsamin算子开发（A2/A3）
> 目标源码仓：`cann/ops-blas`
> 实现目录：`blas/iamin/arch22/`
> 测试目录：`test/isamin/arch22/`
> 提交账号：`WforovZ`

## 需求背景（required）

### 需求来源

本设计依据 Ascend 社区任务
[「11月社区任务-aclblasIsamin算子开发（A2/A3）」](https://www.hiascend.com/activities/task-center/details/b8d4f23bc41c4c58b64414e16dcf9e34?menu=guide)
及对应任务书 `aclblasIsamin_AtlasA2_A3_task_doc.md`。目标是在 `cann/ops-blas`
中为 Atlas A2/A3 增加 `arch22` 的 Ascend C kernel 直调实现，保留既有 Atlas 950
`arch35` 实现和公共接口。

### 背景介绍

仓内已有 `blas/iamin/arch35/` 与 `test/isamin/arch35/`，但 A2/A3 的 `arch22`
缺少对应实现和 GTest 注册。本任务补齐 host 入口、运行时 tiling、AIV kernel、
CSV 驱动测试和性能验证，使 `aclblasIsamin` 在 A2/A3 上具备与 cuBLAS
`cublasIsamin` 一致的功能和错误处理语义。

算子在 FLOAT32 向量中查找绝对值最小元素的首个逻辑位置，输出 INT32、
1-based 索引。任务随包提供 1000 条精度用例和 200 条性能用例，共计 1200 条；
内存验证不涉及。

## 需求分析（required）

### 需求描述

接口沿用 `include/cann_ops_blas.h` 中的既有声明，不新增私有平行接口：

```cpp
aclblasStatus_t aclblasIsamin(
    aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

设逻辑下标 `i` 的有效范围为 `[0, n)`，实际元素地址为 `x[i * incx]`：

```text
result = 1 + argmin(i=0..n-1) |x[i * incx]|
```

正常路径下 `result` 属于 `[1, n]`；绝对值相等时选择最小逻辑下标。NaN
不参与最小值竞争，全 NaN 输入返回 1。`n=0` 或 `incx<1` 时通过 handle
绑定的 stream 将 `result` 异步写为 0，并返回成功。

### 参数与契约

| 参数 | 类型 | 位置 | 语义与约束 |
| --- | --- | --- | --- |
| `handle` | `aclblasHandle_t` | Host | 提供 stream 和 workspace；不得为空 |
| `n` | `int32_t` | Host | 逻辑元素数；负值非法 |
| `x` | `const float*` | Device | FLOAT32 只读向量；正常路径不得为空 |
| `incx` | `int32_t` | Host | 按 float 元素计的步长；`incx<1` 快速返回 |
| `result` | `int32_t*` | Device | 独立单元素输出；任何路径均不得为空 |

Host 校验顺序固定如下：

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `result == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. `n == 0 || incx < 1` 时异步写 `result=0`，返回
   `ACLBLAS_STATUS_SUCCESS`；不访问 `x`，不启动计算 kernel。
5. 正常路径 `x == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. 使用 int64 计算 `(n-1) * incx`；超过 `INT32_MAX` 返回
   `ACLBLAS_STATUS_INVALID_VALUE`。

上述顺序保证 quick return 用例的 `result` 空指针仍会报错，同时不会按非法
`n` 或巨大步长分配输入缓冲区。Host 不读取 Device 结果，调用返回成功只表示
任务已提交到 handle stream。

### 需求拆解

- 新增 `blas/iamin/arch22/isamin_host.cpp`，实现参数校验、tiling 和 kernel 下发。
- 新增 `isamin_kernel.cpp/.h`，实现单核 direct kernel 与多核 scan kernel。
- 新增 `isamin_tiling_data.h`，定义运行时 tiling 参数和分块常量。
- 新增 `test/isamin/arch22/` 的 GTest、1200 条 CSV 和 200 行 GPU 基线。
- 在 `test/isamin/CMakeLists.txt` 的 `arch22` 分支注册
  `ops_blas_add_gtest_tests(${OPS_BLAS} isamin_test)`。
- 更新 iamin 与 arch22 README，补充接口、tiling、约束和复现方式。
- 按官方模板在私仓根目录维护 `task_submission/` 七文件交付件。

## 详细设计（required）

### 算子分析

#### 数据规模与寻址

输入仅包含 FLOAT32，输出仅包含 INT32。`n` 和 `incx` 均为运行时 int32
标量，正常路径的物理跨度为 `storageN = (n-1) * incx + 1` 个 float。
逻辑下标、核边界和 workspace 偏移使用 uint32；GM 地址由 uint64 计算，
避免合法元素的字节地址乘法溢出。负步长不解释为反向遍历，而是按任务书
要求走 quick return。

#### NaN、Inf 与并列

标量路径的更新条件为：

```text
value 不为 NaN
且（尚未获得候选值
    或 value < bestValue
    或 value == bestValue 且 index < bestIndex）
```

因此 NaN 被跳过，并列值保留更早的逻辑下标，±0 的绝对值相等时也保留首个
位置。Inf 是合法参与比较的值。若某一核没有任何有限或 Inf 候选，其
workspace 值写为 NaN 作为“空候选”标记，最终归约跳过该槽。

### 算子实现

#### Host 校验与 tiling

Host 实现位于 `blas/iamin/arch22/isamin_host.cpp`。通过
`GetAivCoreCount()` 获取 AIV 核数后，按以下规则计算 tiling：

```text
coreLimit     = min(n, aivCoreNum)
perCoreN      = align_up(ceil(n / coreLimit), 64)
useCoreNum    = min(coreLimit, ceil(n / perCoreN))
lastCoreN     = n - (useCoreNum - 1) * perCoreN
tileSize      = 16320
```

每核处理连续逻辑区间，`perCoreN` 按 64 个 FP32 元素对齐，以便热点连续路径
使用 vector repeat。尾核只访问剩余逻辑元素。正常多核路径的 workspace 为
每个核一个 float 值和一个 int32 索引，总大小按 64 个 float 对齐，并在下发
kernel 前检查 handle workspace 的大小和地址。

Host 不进行设备内存分配、不执行同步，也不启动 CPU、TBE、vendor 或 arch35
fallback。设备核心数获取失败返回 `INTERNAL_ERROR`；workspace 或快速返回
异步清零失败返回 `EXECUTION_FAILED`；任务成功提交后返回 `SUCCESS`。

#### Direct kernel

仅当 `useCoreNum == 1` 且 `n <= tileSize` 时启动
`isamin_direct_kernel`。单核在本地维护最佳值与最佳逻辑下标，计算完成后
直接向 `result[0]` 写 1-based 索引，不访问 workspace。

#### Scan kernel

其他正常路径启动 `isamin_scan_kernel`，启动核数为 `useCoreNum`：

1. 核 `k` 的起点为 `k * perCoreN`，尾核使用 `lastCoreN`。
2. 每核完成本区间的连续或步长扫描，保存 `(bestValue, bestIndex)`。
3. 各核经 `DataCopyPad` 将候选写入 handle workspace。
4. 同一 stream 内的 kernel 执行 `SyncAll`。
5. 0 号核按核编号顺序归约；值更小则更新，值相等时选择更小逻辑下标。
6. 0 号核向 `result[0]` 写最终值加 1。

候选中没有合法值时写入 NaN，归约阶段显式跳过。若所有输入都是 NaN，
归约初值保持为 `(FLT_MAX, 0)`，最终写出 1，满足任务书全 NaN 语义。

#### 连续向量路径

当 `incx == 1` 且本核逻辑元素数至少为 64 时，使用向量化热点路径：

1. 每轮取不超过 16320 个 FP32 元素，并按 64 元素整块执行 `DataCopy`。
2. 计算绝对值后，通过 `CompareScalar(..., GE, 0)` 生成有限值掩码，
   将 NaN 替换为 `FLT_MAX` 哨兵。
3. 对值取负，使用 `WholeReduceMax` 同时取得绝对值的最大值及其块内索引。
4. 将块内索引映射回全局逻辑下标，并沿用“更小值、并列更早下标”规则更新。
5. 不足 64 元素的尾段或哨兵命中场景回退到标量扫描，保证特殊值正确。

该路径是 200 条性能用例的主要热点。输入 double buffer 为
`2 * 16320 * 4 = 130560` 字节，另有 NaN 掩码和 32 字节归约输出缓冲。

#### 正步长路径

当 `incx > 1` 或区间不足 64 元素时，按逻辑下标逐元素读取：

```text
xGM[(startOffset + i) * incx]
```

标量路径只读取有效元素，不依赖间隔 padding 的内容。该路径优先保证通用
正步长正确性；官方 200 条性能用例均为连续访问。

### 异步与确定性

kernel 和 quick-return 清零均提交到 `handle->stream`，接口内部不调用
stream 同步。调用方读取 `result` 前应执行 `aclrtSynchronizeStream` 或等价
同步。每个核内部按逻辑下标递增扫描，阶段归约按核编号和更早索引更新，
因此单次执行结果确定，并列时始终选择首个位置。

## 支持硬件

| 产品/架构 | 支持路径 | 验证范围 |
| --- | --- | --- |
| Atlas 800I/T A2，910B3 / `arch22` | 本设计新增实现 | CANN 9.1.0 真机全量自验 |
| Atlas A3 / `arch22` | 与本实现共用 arch22 源码 | 支持 A3 构建；正式数据按任务要求覆盖 910B3 |
| Atlas 950 / `arch35` | 既有实现 | 保留，不由本任务修改或替代 |

## 约束核对

| 任务书约束 | 实现与验证结论 |
| --- | --- |
| 参数合法性与校验层级 | handle、n、result、quick return、x、跨度依次校验，返回码与任务书一致 |
| 非连续 Tensor | 支持任意 `incx >= 1` 的正步长访问；`incx < 1` 写 0 快速返回 |
| broadcast | BLAS-1 单向量归约，不涉及多操作数广播 |
| dynamic shape | n、incx 为运行时标量，Host 每次重新计算 tiling |
| 原地与视图 | x 只读；result 独立输出，不复用输入内存 |
| 确定性 | 严格小于比较，并列选择最小 1-based 下标 |
| 空 Tensor | n=0 不访问 x、不启动计算 kernel，异步写 result=0 |
| 0 维 Tensor | 向量算子不涉及；空向量由 n=0 契约承担 |
| 元素值校验 | 不额外校验元素；Inf 参与比较，NaN 不参与最小值竞争 |
| 异步执行 | 使用 handle stream，不在接口内阻塞或回读结果 |

## 已知限制与规避

- `(n-1) * incx` 必须可表示为 INT32；超限输入由 Host 返回
  `ACLBLAS_STATUS_INVALID_VALUE`。
- 性能热点针对连续向量；正步长路径以通用正确性为主，不承诺所有大跨度
  组合达到连续路径的吞吐。
- 输入、输出必须为当前设备有效地址，并在异步任务完成前保持有效。
- 调用方应保证 `result` 与输入不重叠。
- 不使用未入仓测试、golden、GPU 基线或任务文件参与运行时计算。

## 可维可测分析

### 精度验证

测试位于 `test/isamin/arch22/isamin_test.cpp`，复用仓内 CSV loader、CPU
golden 和 NPU wrapper。自验覆盖：

- 官方 1000 条精度 CSV 和 200 条性能 CSV。
- n=0、n=1、正步长 1/2/7/16、负 n、空 handle、空 x、空 result。
- quick return、INT32 跨度溢出、Inf、NaN、全 NaN、±0、重复最小值。
- 均匀分布与正态分布按 50%/50% 补齐；正态参数满足
  μ∈[-5,5]、σ∈[0.1,2]。
- result 为 INT32，逐条与 golden 精确相等，不应用浮点容差。

GTest 共 1006 项，其中 1000 项由 CSV 驱动，另含 6 项显式 API 契约与边界
检查。报告只写实际执行得到的 `matched_ratio` 和 `max_abs_error`，
不以单元测试替代完整 1000 条 CSV 验证。

### 性能验证

性能设备为 Atlas 800I A2 / Ascend 910B3，环境为 CANN 9.1.0。200 条
`TC_PF_` 用例与 `gpu_baseline.csv` 按 `(n, incx)` 顺序逐条对应。
按任务书采用 `msprof op` 采集 `Task Duration(us)`，预热 5 次，
每条至少采集 10 个有效样本并取算术平均。

判据为单条用例 NPU/GPU 平均耗时比不高于 1.25。目前全量结果为
200/200 PASS、2000 个有效样本，最大比值为 `1.101241`。报告保留
完整 200 行结果、终端截图和原始日志。

### 内存验证

任务书 §3.4 明确内存测试不涉及。`4.1 内存自验证报告.xlsx` 与
`4.2 内存自验证日志.log` 按模板保留并标注 N/A，不将输入/输出 guard
检查包装成额外内存验收。

## 交付件与提交

代码与测试随 `cann/ops-blas` PR 提交：

- `blas/iamin/arch22/isamin_host.cpp`
- `blas/iamin/arch22/isamin_kernel.cpp`
- `blas/iamin/arch22/isamin_kernel.h`
- `blas/iamin/arch22/isamin_tiling_data.h`
- `blas/iamin/arch22/README.md`
- `test/isamin/arch22/` 下的 GTest、CSV 和基线
- `test/isamin/CMakeLists.txt` 的 arch22 GTest 注册
- iamin 公共 README、arch35 公共 golden/wrapper 契约修正

个人仓根目录提交以下官方七文件：

```text
task_submission/
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx
└── 4.2 内存自验证日志.log
```

验收前在任务 IT 系统提交七文件、个人仓分支、算子目录、代码 commit
链接和 PR 链接，并邀请 `Ascend-CANN` 为个人仓开发者。自验结果不等同
官方验收结论，最终以任务平台测试和官方代码评审结果为准。
