# aclblasIsamin 算子设计文档

## 文档信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | aclblasIsamin |
| 任务类型 | CANN 社区任务 2026 年 11 月 BLAS 算子开发 |
| 目标仓库 | `cann/ops-blas` |
| 目标硬件 | Atlas A2/A3 系列，自验证硬件为 Atlas 800I A2 910B3 |
| 目标 CANN 版本 | CANN 9.1.0 |
| 实现语言 | Ascend C |
| 交付目录 | `blas/iamin/arch22/`、`test/isamin/arch22/` |

## 1. 需求背景

`aclblasIsamin` 对标 cuBLAS `cublasIsamin`，用于在单精度实数向量中查找绝对值最小元素的位置。该接口是 BLAS-1 的基础索引归约能力，返回 1-based 索引，与现有 `aclblasIsamax` 构成对称能力。

`ops-blas` 仓已有面向 Ascend 950PR 的 `blas/iamin/arch35/` 实现，但 Atlas A2/A3 使用的 `arch22` 目录尚不存在。本任务为 Atlas A2/A3 新增 kernel 直调实现、host 参数校验、tiling、GTest 用例和算子文档，使同一公开接口 `aclblasIsamin` 在两代硬件上可共同使用。

## 2. 需求分析

### 2.1 数学语义

设输入向量为 \(x\)，元素个数为 \(n\)，步长为 \(incx\)。结果定义为：

$$
result = \operatorname*{arg\,min}_{i=1,\ldots,n}\left|x[1+(i-1)\times incx]\right|
$$

结果使用 1-based 索引。比较采用严格小于语义，多个元素绝对值相等时返回最小索引。NaN 元素不参与最小值更新；首个元素为 NaN 时以 \(FLT\_MAX\) 作为初始最小值；如果全部元素均为 NaN，返回 1。

当 \(n=0\) 或 \(incx<1\) 时执行快速返回，将 `result` 写为 0 并返回成功。

### 2.2 接口定义

```cpp
aclblasStatus_t aclblasIsamin(
    aclblasHandle_t handle,
    int n,
    const float* x,
    int incx,
    int* result);
```

接口声明沿用 `include/cann_ops_blas.h`，不新增私有平行接口。

### 2.3 参数与支持范围

| 参数 | 方向 | 说明 | 类型/位置 | 支持范围与约束 |
| --- | --- | --- | --- | --- |
| handle | 输入 | 库句柄，携带异步 stream | `aclblasHandle_t`，Host | 必须非空；为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 输入 | 向量元素个数 | `int`，Host | 支持非负整数；\(n<0\) 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| x | 输入 | 单精度实数向量首地址 | `const float*`，Device | 正常计算要求非空；支持单位与非单位正步长 |
| incx | 输入 | 相邻逻辑元素间隔 | `int`，Host | \(incx<1\) 走快速返回；正常计算要求 \((n-1)incx\) 不超过 `INT32_MAX` |
| result | 输出 | 最小绝对值元素的 1-based 索引 | `int*`，Device | 必须非空；正常计算取值 \([1,n]\)，快速返回写 0 |

数据类型固定为 FLOAT32 输入、INT32 输出，不涉及广播、dynamic shape、原地计算或视图语义。

### 2.4 验收口径

精度正式集为 1000 条 CSV 用例，输出为整数索引，要求与实际 CPU golden 完全一致。`RANDOM_NORM_5_5` 和 `GAUSS_NORM_5_2` 各 500 条；零、Inf、NaN、并列最小值、quick return 和参数负向路径通过分层特殊注入覆盖。

性能正式集为 200 条连续访存用例。每条用例独立执行 `msprof op BasicInfo`，在 5 次 warmup 后采集 15 个有效 kernel `Task Duration(us)` 样本，全量 3000 个样本。平均 task duration 不得超过对应 GPU A100 基准的 1.25 倍，即达到 A100 性能的 0.8 倍以上。

内存专项不涉及。正式自验证硬件为 910B3。

## 3. 总体设计

算子采用 kernel 直调模式，由 host 入口完成参数校验和 tiling 计算，再由单次 kernel 下发完成多核归约。host 返回成功仅表示任务已提交到 handle 绑定的 stream，不在库内同步。

```text
Host: 参数校验 -> tiling -> workspace 检查 -> kernel launch
                                      |
Device: 多 AIV 核分片 -> 各核 indexed ReduceMin -> GM 写回 pair
                                      |
                               SyncAll
                                      |
                        core 0 汇总 -> result
```

该方案一次 launch 完成全部计算，避免“多核部分归约 + 单独 reduce kernel”带来的第二次下发开销。各核的内部临时结果写入 handle 已管理的 workspace，不新增每次调用级别的设备内存申请和释放。

## 4. Host 侧设计

### 4.1 参数校验

校验顺序固定为：

1. `handle == nullptr`，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. \(n<0\)，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. `result == nullptr`，返回 `ACLBLAS_STATUS_INVALID_VALUE`；该检查覆盖快速返回路径。
4. \(n=0\) 或 \(incx<1\)，进入快速返回。
5. 正常计算时 `x == nullptr`，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. 使用 int64 计算 \((n-1)incx\)，超过 `INT32_MAX` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

快速返回通过 `aclrtMemsetAsync(result, 0, sizeof(int), handle->stream)` 写回设备结果，随后返回成功。

### 4.2 分核与 tiling

host 查询 AIV 核数，并按计算量选择实际使用核数：

$$
useCoreNum=\min\left(AIVCoreNum,\left\lceil\frac{n}{4096}\right\rceil\right)
$$

同时保证 \(useCoreNum\geq1\)。每核处理元素数为：

$$
perCoreN=\left\lceil\frac{n}{useCoreNum}\right\rceil
$$

最后一个核按实际剩余元素数截断。该策略使小尺寸用例只使用必要核数，大尺寸用例使用完整 AIV 并行度。

`IsaminTilingData` 只包含总元素数、步长、实际核数、每核元素数和单核 tile 大小。tiling 通过值参数传给 kernel，避免每次调用额外复制 tiling 到设备。

### 4.3 Workspace

每个 AIV 核向 workspace 写入一对 `{bestValue, indexBits}`。workspace 需求为：

$$
workspaceBytes=useCoreNum\times2\times sizeof(float)
$$

host 使用 handle 当前有效 workspace，并在下发前检查容量。910B3 的 handle 默认 workspace 足以覆盖最多核数场景。

## 5. Kernel 侧设计

### 5.1 连续访存路径

当 \(incx=1\) 时，输入按连续区间分段。单核以 16320 个 FP32 元素为一个 tile，处理流程为：

1. `DataCopy` 搬入完整 tile，尾 tile 使用 `DataCopyPad`。
2. `Abs` 计算绝对值。
3. 调用带索引输出的 `ReduceMin`，得到 tile 最小值和 tile 内首个最小索引。
4. 将 tile 结果转换为全局索引，并按值、索引的二元组更新核内最优结果。

尾 tile 使用 `+Inf` 填充，并将索引不小于有效元素数的结果丢弃。小尾块可以直接在 UB 上标量扫描，保证 `n=1` 以及极短尾块的结果稳定。

更新规则使用严格小于和索引优先规则：

```text
if candidateValue is NaN: ignore
if no candidate exists and candidateValue >= FLT_MAX: ignore
if candidateValue < bestValue: update
if candidateValue == bestValue and candidateIndex < bestIndex: update
```

### 5.2 非连续路径

当 \(incx>1\) 时，按逻辑元素序号 \((start+i)\times incx\) 从 GM 读取元素，先 gather 到 UB，再复用同一套绝对值和 indexed ReduceMin 逻辑。host 已保证最大逻辑偏移不超过 `INT32_MAX`。

性能正式集全部为连续访存；非连续路径属于功能支持范围，不使用性能门槛衡量其吞吐。

### 5.3 核间汇总

每个核处理后向 workspace 写入 `{bestValue, indexBits}`。索引以 float 位模式保存，保证 pair 可以按 float 搬运；汇总时再恢复为 uint32 索引。

所有核写完后执行 `SyncAll()`。core 0 搬入全部核结果并执行第二次标量二元组归约，最终写入：

```text
result = hasFiniteCandidate ? bestIndex + 1 : 1
```

因此全部 NaN 时返回 1，并列最小值时返回最小全局索引。

### 5.4 异步执行

kernel 通过 handle 绑定的 stream 下发。host 侧不执行 `aclrtSynchronizeStream`，不读取设备结果，也不依赖调用后的 host 状态。调用方读取 `result` 前需自行同步 stream。

## 6. 测试设计

### 6.1 测试工程

在 `test/isamin/arch22/` 新增 CSV 驱动的 GTest：

- `isamin_test.cpp`：加载 CSV、分配设备内存、执行 NPU/CPU golden 对比和性能采集。
- `isamin_test.csv`：1000 条精度用例和 200 条性能用例。
- `gpu_baseline.csv`：性能用例对应的 A100 基线。
- `verify_performance_msprof.py`：逐例采集 200 条正式性能并归档原始 `OpBasicInfo`。

`test/isamin/CMakeLists.txt` 同时注册 arch22 和 arch35 的 `isamin_test`，A2/A3 构建不再退回非 GTest runner。

### 6.2 精度用例

正式精度集覆盖以下边界：

- \(n=0,1,2,3\) 及 2 的幂、2 的幂加减一、非对齐尺寸和大尺寸。
- \(incx\in\{1,2,7,16\}\) 以及 quick return 的 0、负数和 `INT32_MIN`。
- 全零、正负交替、重复最小值、首元素 NaN、全 NaN、全 Inf、`[NaN, +Inf, -Inf]`、`[NaN, FLT_MAX]` 和混合特殊值。
- `handle` 为空、`n<0`、x 为空、result 为空和跨步溢出。

`RANDOM_NORM_5_5` 和 `GAUSS_NORM_5_2` 各 500 条。正态分布覆盖 \(\mu\in[-5,5]\)、\(\sigma\in[0.1,2]\)。特殊值通过 CSV `description` 中的 `special=<tag>;` 注入，并计入对应 500 条分层。输出按整数索引精确比较，不应用浮点 tolerance。

### 6.3 性能用例

200 条性能用例全部使用连续访存。`aclrtEvent` 路径用于 200 条快速筛查，不作为正式判定。正式性能逐例启动独立 `msprof op`：

1. `--aic-metrics=BasicInfo`。
2. `--warm-up=5`。
3. `--launch-count=15`。
4. `--replay-mode=kernel`。
5. `--kernel-name=isamin_kernel`。

GTest profile 模式同步执行 5 次预热和 15 次有效调用。每例必须解析出 15 个正有限 `Task Duration(us)` 样本并取算术平均；缺少样本、非有限值、GTest 失败或 msprof 错误均判采集失败。

正式判定使用 `gpu_baseline.csv` 中按 `(n, incx)` 匹配的 A100 耗时：

$$
msprof\_mean\_us \leq \frac{GPU\_baseline}{0.8}
$$

逐例保存原始 `OpBasicInfo_*.csv`、`command.json`、stdout/stderr、逐样本 CSV 和 200 例汇总。GTest `aclrtEvent` 结果单独保留为快速筛查记录。

### 6.4 内存验证

任务书 §3.4 明确内存专项不涉及，因此不构造内存验收门槛；仅通过 workspace 容量检查和正常设备运行验证资源使用。

## 7. §2.5 约束逐项核对

| 约束项 | 本设计结论 |
| --- | --- |
| 参数合法性与校验层级 | 按 §4.1 顺序校验；快速返回写设备 0；溢出使用 int64 判断 |
| 非连续 Tensor | 支持正步长访问；\(incx<1\) 直接快速返回 |
| broadcast | 不涉及 |
| dynamic shape | 不涉及图模式；运行时 n 和 incx 由 host 生成 tiling |
| 原地与视图语义 | x 只读，result 为独立 Device 输出 |
| 确定性 | 单次执行使用固定分核和严格二元组比较，结果确定 |
| 空 Tensor | 等价 \(n=0\)，不启动计算 kernel，快速返回成功 |
| 0 维 Tensor | 不涉及 |
| 异步执行 | 仅向 handle stream 下发，不在库内同步 |

## 8. 已知限制与规避

1. 非连续路径的 gather 以功能正确性为主，不进入性能正式集。
2. 最大逻辑偏移超过 `INT32_MAX` 的参数组合在 host 侧提前拒绝，避免 device 侧整数回绕。
3. 尾块填充值不能成为有效结果；kernel 同时检查最小值和索引有效范围。
4. 全部元素为 NaN 时没有有效候选，core 0 汇总路径显式输出 1。

## 9. 交付件

- `blas/iamin/arch22/`：host 入口、tiling 定义、kernel 和 README 更新。
- `test/isamin/arch22/`：GTest、正式 CSV、性能基线和逐例 msprof 采集脚本。
- 设计文档、自验证步骤、精度/性能日志和报告。
- 个人代码仓分支和待验收 commit。
