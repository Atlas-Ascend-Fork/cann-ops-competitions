# aclblasIcamax 算子设计文档

> 任务：`11月社区任务-aclblasIcamax算子开发（A2/A3）`
> 目标硬件：Atlas A2 / Atlas A3（arch22），CANN 9.1.0
> 代码落点：`ops-blas/blas/iamax/arch22/`，测试落点：`ops-blas/test/iamax/icamax/arch22/`
> 模板：cann-competitions `04_tasks/01_community-task-2026/resources/design_template.md`

# 需求背景（required）

## 需求来源

社区任务《11月社区任务-aclblasIcamax算子开发（A2/A3）》要求在 Atlas A2/A3 上用 Ascend C
实现 `aclblasIcamax`，对标 cuBLAS `cublasIcamax`，验收通过后合入
[ops-blas](https://gitcode.com/cann/ops-blas) 的 `blas/iamax/arch22/`。

任务书随附：自测用例（1200 条 CSV）、精度/性能验收脚本、GPU（A100）性能基线。

## 背景介绍

### 对标接口与参考实现

| 参考 | 语义 | 与本任务的关系 |
|---|---|---|
| cuBLAS `cublasIcamax` | 返回 `|Re|+|Im|` 最大元素的 **1-based** 索引，并列取最小索引 | 功能与接口语义对标对象 |
| Netlib `icamax.f` | `IX=1`；`I=2..N` 用 `ABS(REAL)+ABS(AIMAG)` 与 `X(IX)` 比较，**严格大于**才更新 | 比较语义与特殊值的最终依据 |
| cblas `icamax`（BLAS 3.10.0） | 与 Netlib 同源，返回 **0-based** | 精度 golden 来源（+1 转 1-based） |

关键结论：复数“模”是 **1-范数** `|Re| + |Im|`（Netlib `SCABS1`），不是欧氏模；比较必须
“严格大于才更新”，这同时决定了并列取最小索引、首个 `Inf` 胜出、`NaN` 不参与更新三条特殊语义。

### 仓内现状分析

| 项 | 现状 |
|---|---|
| 接口声明 | `include/cann_ops_blas.h:242` 已有 `aclblasIcamax`，与 `aclblasIsamax` 参数类型一致；**不新增、不定义私有接口** |
| arch22 实现 | **缺失**。`blas/iamax/arch22/` 只有 `isamax_host.cpp` / `isamax_kernel.cpp` / `isamax_kernel_impl.h` |
| arch35 实现 | 已有 `icamax_*`（Ascend 950PR），基于 SIMT / RegBase（`RegTensor`、`DeInterleave`、`asc_vf_call`），**arch22 无 SIMT，不能复用其 kernel** |
| arch22 测试 | **缺失**。`test/iamax/icamax/CMakeLists.txt` 只在 `arch35` 分支注册 GTest，A2/A3 构建没有本算子测试目标 |
| 产品支持表 | `blas/iamax/README.md` 把 `aclblasIcamax` 标为 950PR 支持、**A2/A3 不支持**，本任务后需修正 |
| 同族可复用资产 | `blas/iamax/arch22/isamax_*` 的直调框架、分核 tiling、value+index 成对归约、`SyncAll` + 核 0 跨核归约模式 |

`isamax` arch22 实现中有两处**不能照搬**：

1. `isamax_kernel_impl.h:181` 解析了 `incx` 但全文未使用，即 arch22 isamax 实际只支持连续输入；
2. `isamax_host.cpp:221-227` 在 quick return 时对 `result == nullptr` 仍返回成功，且用 host 语句
   `*result = 0` 写 Device 指针，与本任务书（`result` 为空一律 `INVALID_VALUE`）和 arch35 实现都不一致。

`isamax_kernel_impl.h:351-371` 已含复数分支（`Abs` 全量 + `PairReduceSum` 得 `|Re|+|Im|`），
但 host 恒置 `dtypeFlag = 0`，该分支在本仓从未被实例化。本任务复用的是**这段思路**，不是它的
调用约定。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIcamax(
    aclblasHandle_t handle,
    int n,
    const aclblasComplex* x,
    int incx,
    int* result);
```

- 输入 `x`：Device 内存，COMPLEX64（实部/虚部各 FP32，交错 8B），逻辑长度 `n`，物理跨度
  `1 + (n-1)*incx` 个复数（`incx >= 1`）。
- 输出 `result`：Device 内存，单个 INT32，**1-based** 索引。
- 计算 `result = 1 + argmin_i { i | s(i) = max_j s(j) }`，`s(i) = |Re(x[i*incx])| + |Im(x[i*incx])|`，
  `i = 0..n-1`。
- 异步执行：设备工作加入 `handle` 绑定的 stream，读回 `result` 前由调用方同步 stream。

## 需求拆解

1. **功能语义**：1-范数比较、1-based 输出、并列取最小索引、首个 `Inf` 胜出、`NaN` 规则、quick return。
2. **参数与错误码**：`handle` 空、`n < 0`、`result` 空、正常路径 `x` 空、`(n-1)*incx` 溢出。
3. **精度**：与 cblas golden 精确相等（bit-exact），无浮点容差。
4. **性能**：A2/A3 单 case 平均耗时 ≤ 1.25 × GPU 基线（即 ≥ 0.8×），性能优先、不设优化轮次上限。
5. **交付件**：设计文档、arch22 实现、CSV 驱动 GTest（含性能采集）、自测报告、README 产品表更新。

## 语义定义（实现与验证的共同口径）

### 普通有限值输入

```text
s(i) = |Re(x[i*incx])| + |Im(x[i*incx])|,  i = 0..n-1
M    = max s(i)
r    = min{ i | s(i) = M } + 1
```

不使用 `sqrt`、不计算 `Re²+Im²`；比较对象是 `(s, i)` 二元组，先比 `s` 大、再比 `i` 小。

### 特殊值

| 输入 | 结果 | 依据 |
|---|---|---|
| 含多个 `+Inf` 比较值 | 第一个 `Inf` 所在索引 | 严格大于：`Inf > Inf` 为假 |
| 首元素比较值为 `NaN` | `1` | Netlib 以 `IX=1` 为种子，`s(i) > NaN` 恒假 |
| 后续元素比较值为 `NaN` | 该元素不参与，其余元素按普通规则 | 同上 |
| 比较值为 `NaN` 且首元素不为 `NaN` | 等价于“忽略所有 `NaN` 后取最大值” | 由上一行推出 |

由此得到一条**可直接实现**的等价规则，本设计据此实现：

```text
result = 1                        若 s(0) 为 NaN
result = 1 + 最小索引(argmax over { s(i) | s(i) 非 NaN })   否则
```

### 边界与错误码

| 条件 | 行为 |
|---|---|
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `result == nullptr`（任意 `n`，含 quick return） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `n == 0` 或 `incx <= 0` | quick return：不读 `x`，stream 上异步写 `result = 0`，返回 `SUCCESS` |
| `n > 0 && incx >= 1 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `(n-1) * incx` 超出 int32 元素索引范围 | `ACLBLAS_STATUS_INVALID_VALUE`（issue #745） |

校验顺序固定为 `handle → n<0 → result 空 → quick return → x 空 → 跨度溢出`，与 arch35
`icamax_host.cpp:28-63` 一致：`result` 空必须在 quick return 之前判定，否则 `n=0` 的负向用例会被
quick return 掩盖。`incx <= 0` 是**合法 quick return**，不是参数错误；本算子不做负步长反向遍历。

# 详细设计（required）

## 算子分析

### 数学公式

见“语义定义”。补充实现层面的两个等价变形：

- `|Re| + |Im|` 对交错布局天然可由“整段取绝对值 + 相邻成对求和”得到，不需要拆成实部/虚部两份数组；
- “严格大于”的归约等价于“求最大值 + 在等于最大值的元素中取最小索引”，可用向量 `Max`/`ReduceMax`
  加一次等值定位实现，不需要逐元素标量比较。

### 支持数据类型

| 参数 | dtype | 说明 |
|---|---|---|
| `x` | COMPLEX64 | 实部/虚部各 FP32，交错存储，8B/元素 |
| `result` | INT32 | 单个标量 |
| `n` / `incx` | int（Host） | `n >= 0`；`incx <= 0` 走 quick return |

### 支持形状 / 布局

逻辑一维 `[n]`。物理跨度 `1 + (n-1)*incx`。`incx = 1` 走连续路径（性能主路径），
`incx >= 2` 走步长路径。不做广播、不做负数步长、不支持 `n` 超过 int32 元素索引范围。

### 访存与复杂度

- 必读数据量：`8n` 字节（每个复数只读一次）；写出 4 字节结果 + 至多 `8 × 核数` 字节的核间摘要。
- 计算量：每元素 2 次绝对值 + 1 次加法 + 归约比较，算术强度约 0.25 FLOP/B，**纯访存 bound**。
- 因此性能上限由 HBM 带宽决定，设计目标是让 MTE2 成为唯一瓶颈（计算与同步不成为串行段）。

## 算子实现

### 总体结构（单 kernel）

```text
host：校验 → quick return（stream memset）或 tiling → 一次 kernel launch（tiling 按值传参）
                    │
kernel（每个 AIV 核）：
  核心 0：读 x[0] 判定首元素 NaN（仅 2 个标量读）
  ├─ incx == 1：循环 tile：MTE2 搬入 → Abs(2m) → PairReduceSum(m) → ReduceMax(值)
  │              → 标量比较更新 (bestValue, bestIdx)；值更新时在本 tile 内定位最小索引
  └─ incx > 1 ：循环 tile：逐元素 gather 到 UB → 同上（同一套向量计算与归约）
  各核把 (value, index) 写入 workspace → SyncAll → 核 0 按核序归约 → 写 result = index + 1
```

只启动一次 kernel：核内归约与跨核归约在同一次 launch 内完成，避免第二次 launch 的固定开销
（`blas/iamax/arch22/isamax_*` 已验证同一模式）。

### host 侧设计

#### 参数校验与 quick return

按上表顺序校验。quick return 不启动 kernel、不读 `x`，用
`aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream)` 在 stream 上写 0，
与 arch35 `WriteIcamaxQuickReturn` 一致；这样 `result` 也不需要 host 侧可写。

#### tiling 策略

设 `G = 32`（32 个复数 = 256B，满足 32B/512B 对齐与 `PairReduceSum` 的整 repeat 要求）：

```text
groups          = n / G
tail            = n % G
maxUsefulCores  = groups + (tail > 0 ? 1 : 0)
usedCoreNum     = min(aivCoreNum, maxUsefulCores)        // A3/A2 上 aivCoreNum 由平台查询得到
perCoreGroups   = groups / usedCoreNum
remainderGroups = groups % usedCoreNum
```

核 `c` 的逻辑区间：

```text
prefix = min(c, remainderGroups)
start  = (c * perCoreGroups + prefix) * G
count  = perCoreGroups * G + (c < remainderGroups ? G : 0) + (c == usedCoreNum-1 ? tail : 0)
```

即“整组均分 + 余组轮转给前 `remainderGroups` 个核 + 不足一组的尾巴并入末核”。这样：

- 每个核的起点都是 256B 对齐，连续路径可直接 `DataCopy`；
- 尾块只出现在末核，且长度 < 32 复数，由一次 `DataCopyPad` 补齐处理；
- 不会出现“最后一个核几乎空转”或“尾块单独占一个核”的失衡。

**小 n 退化**：`usedCoreNum` 自然被 `maxUsefulCores` 限制；当 `n < 2G` 时只有 1 个核，
省掉跨核 `SyncAll` 与 workspace 往返。

**步长路径**：同样的分核公式，但每核内部按逻辑索引取数（物理位置 `start+i` 乘 `incx`），
不要求 GM 侧连续。

#### tilingkey / 分支

| 分支 | 条件 | 说明 |
|---|---|---|
| `incx == 1` 连续路径 | `incx == 1` | 主性能路径，`DataCopy` + 向量归约 |
| `incx > 1` 步长路径 | `incx >= 2` | 逐元素 gather 到 UB，同一套向量归约 |
| 首元素 NaN | 核 0 读到 `x[0]` 任一分量为 NaN | 跳过扫描，最终写 1（仍参与 `SyncAll`） |
| 无候选（全 NaN） | 所有核摘要均为哨兵 | 写 1（与首元素 NaN 同解，属兜底） |

分支由 host 侧 `incx` 与 kernel 内标量判断决定，不需要 tilingkey 枚举；kernel 内分支都是
“每核一次”或“每 tile 一次”，不进入逐元素循环。

#### workspace 与启动

- workspace：`usedCoreNum × 2 × 4B`（`value` + `index`），从 handle 级 workspace 取
  （`GetEffectiveWorkspace` / `GetEffectiveWorkspaceSize`，默认 32MB），**每次调用不 malloc**；
  启动前校验容量，不足返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
- tiling：结构体按值传给 kernel（arch22 已支持，见 `blas/tpsv/arch22/stpsv_kernel.cpp:560`），
  **不做 per-call `aclrtMalloc` + H2D**。
- 不在 host 侧 `aclrtSynchronizeStream`：保持异步语义，同步责任交给调用方。

### kernel 侧设计

#### 常量与 UB 预算（单个 AIV）

| 项 | 值 | 依据 |
|---|---|---|
| `TILE_COMPLEX` | 4096 复数（32KB/缓冲） | 双缓冲 64KB；`2×4096/64 = 128` 个 repeat，低于 `PairReduceSum` 单次 255 repeat 上限 |
| 输入缓冲 | 2 × 32KB | 双缓冲，MTE2 与 V 重叠 |
| work 缓冲 | `(4096/64 + 8) × 4B = 288B` | `ReduceMax` 级联工作区 |
| 掩码缓冲 | `4096/8 = 512B` | `Compare/CompareScalar` 的按位掩码（1 bit/元素） |
| 核间摘要缓冲 | `40 × 2 × 4B = 320B` | 核 0 读回各核 (value,index) |
| 输出缓冲 | 2 × 32B | 摘要写回与结果写回各一份 |

总占用约 65.5KB，低于 arch22 单 AIV 的 UB 容量，tile 还有加大余量（候选 4）。

#### CopyIn

- 连续路径：`DataCopy(inLocal, xGm[offset*2], 2*count)`；`count` 已按 32 复数对齐，天然 256B 对齐。
- 尾块（末核 `tail < 32`）：`DataCopyPad` 一次搬入 `2*tail` 个 float，`paddingValue = -FLT_MAX`。
  补位值不可能成为最大值（真实比较值 `>= 0`），因此不需要额外掩码参与归约。
- 步长路径：`for i in 0..count-1: inLocal.SetValue(2i, xGm.GetValue(2*(start+i)*incx)); ...`。
  这是仓内 arch22 处理 `incx > 1` 的既有做法（`blas/asum/arch22/sasum_kernel.cpp:163-171`），
  优先保证正确性与代码可读性；向量化 gather 作为性能候选单独 A/B（见第 5 节）。

#### Compute

每个 tile：

```text
1) Abs(inLocal, inLocal, 2*count)                    // |Re|, |Im| 就地取绝对值
2) PairReduceSum(score, inLocal, repeat, 64, 1, 1, 8) // 相邻成对求和 => |Re|+|Im|，结果在 inLocal 前半
3) 不足 64 的尾部补 -FLT_MAX（见下）
4) Compare(mask, score, score, NE, aligned)          // NaN 位置（位掩码，1 bit/元素）
   Select(score, mask, score, -FLT_MAX, VSEL_TENSOR_SCALAR_MODE, aligned)
5) ReduceMax(tileMax, score, work, aligned, false)   // 只取值
6) 标量：if (tileMax > bestValue) { bestValue = tileMax; 在本 tile 内定位最小索引 }
```

第 4) 步是**无条件执行**的：把 NaN 替换为 `-FLT_MAX` 后，`ReduceMax` 得到的就是“非 NaN 元素的最大值”，
与 Netlib“NaN 永不满足严格大于”完全等价，不依赖硬件 `vmax` 的 NaN 语义。代价是每 tile 多两次
向量 pass（Compare + Select），而这两次只消耗向量流水、不消耗 HBM 带宽，在 MTE2 为瓶颈时可以被
覆盖。若实测该开销不可忽略，可改为“仅当 `tileMax` 为 NaN 时才对当前 tile 补掩码重算”的条件分支
（列为本设计的性能候选，见第 5 节候选 2）。

填充槽位统一为 `-FLT_MAX`：真实比较值恒 `>= 0`，因此补位既不会在 `ReduceMax` 中胜出，也不会在
下面的等值定位中被误判。`count` 不是 64 的整数倍时，先把 `[count, aligned)` 补成 `-FLT_MAX` 再参与
归约与比较，保证所有向量 API 的元素数都是 64 的整数倍。

**tile 内最小索引定位**（只在 `bestValue` 被严格更新时执行，随机输入下每核约 3~5 次）：

```text
CompareScalar(mask, score, bestValue, CMPMODE::EQ, aligned)   // 位掩码，1 bit/元素
按 32 位字扫描掩码，取第一个非零字，字内最低置位即局部索引 → 全局索引 = start + tileOffset + idx
```

这样**不依赖 `ReduceMax`/`WholeReduceMax` 在并列时返回首元素还是末元素**，并列语义由软件固定，
与硬件实现无关。`count` 为 0 的空 tile 直接跳过。

#### CopyOut / 跨核归约

- 每核把 `(bestValue, bestIndex)` 写成两个 32 位字到 workspace 的第 `c` 槽（`DataCopyPad`，8B）。
- 全核写完后 `SyncAll()`，核 0 读取 `usedCoreNum` 个槽：
  按核序（等价于索引升序）用**严格大于**比较，取最大 `value` 及其 `index`；全部为哨兵时输出 1。
- 结果 `DataCopyPad` 写 4B 到 `result`（不整块写 32B，避免越界写）。

#### 同步与流水

| 交接 | 保护手段 | 说明 |
|---|---|---|
| MTE2 → V | `TQue` 的 `EnQue/DeQue`（双缓冲） | 搬入与上一 tile 计算重叠 |
| V → S | `SetFlag/WaitFlag<V_S>`，每 tile 一次 | 只为读取 `ReduceMax` 的标量结果 |
| S → V | `SetFlag/WaitFlag<S_V>`，仅索引定位后 | 标量写回的掩码/比较在被向量使用前 |
| 核间 | `SyncAll()` | workspace 摘要的跨核可见性 |
| S → MTE3 | `SetFlag/WaitFlag<S_MTE3>` | 结果标量写入 UB 后搬出 |

每个 tile 只有 1 次 V→S 同步（读最大值），没有逐元素同步；`PipeBarrier` 只放在
`Abs/PairReduceSum/ReduceMax` 之间确有依赖处。

### 支持硬件

| 芯片 | 支持 | 说明 |
|---|---|---|
| Atlas 800T A2（910B3） | √ | 任务书推荐性能设备 |
| Atlas A3 训练/推理系列（910_93） | √ | 本任务自验设备（hidevlab 单卡） |
| Ascend 950PR | 已有独立 arch35 实现 | 本设计不改动 |

arch22 实现只使用 v220 向量指令与 MTE2/MTE3，不依赖 SIMT/RegBase API。

### 算子约束限制

- 不支持负步长反向遍历（`incx <= 0` 为合法 quick return，结果 0）。
- 不保证核间/片间的归约顺序固定，但**结果**必须满足“最小索引”语义，与 golden 精确一致。
- `n` 与 `incx` 的组合必须满足 `(n-1)*incx <= INT32_MAX`（元素索引），否则返回 `INVALID_VALUE`。
- 输入、输出与 handle workspace 在调用期间必须有效；调用方不得并发使用同一 workspace。

# 可维可测分析

## 精度标准

| 项 | 标准 |
|---|---|
| golden | cblas（Netlib BLAS 3.10.0）`icamax`，**0-based 结果 +1** 后比对 |
| 判定 | NPU 输出与 golden **精确相等**（整数索引，无容差列，等价 matched_ratio = 1） |
| dtype 档 | 输入数据按 FLOAT32 档（rtol/atol = 2^-13）生成；该档只约束造数，不用于输出比对 |
| 负向用例 | 按 `expect_result` 比对状态码；quick return 用例直接期望 `result = 0` |
| 补充说明 | 误差只可能来自比较语义（欧氏模、并列取错索引、0/1-based 混淆），正常实现应全部精确通过 |

## 性能标准

| 项 | 标准 |
|---|---|
| 设备 | Atlas 800T A2（910B3）（任务书推荐）；当前以 A3 单卡采集并在报告注明设备 |
| 采集 | `msprof op` 单算子 task duration（us），5 次 warmup 不计入，≥10 次有效样本取算术平均 |
| 基线 | `test_cases/gpu_baseline.csv`（A100 / CUDA 12.8 / cuBLAS 12.8.4）：n=1M→17.46us、2M→19.64us、4M→27.63us |
| 判据 | 单 case `avg_us <= gpu_ms * 1000 / 0.8`，即 ≥ 0.8× GPU |

## 测试方案

### 用例组织

- 目录：`test/iamax/icamax/arch22/`，CSV 与 arch35 同格式（`case_name,description,n,incx,x,result,random_seed,expect_result`）。
- 随任务 `icamax_test.csv`（1000 精度 + 200 性能）作为基础集；`TC_PF` 前缀为性能/内存用例，均为 `incx=1`。
- 覆盖：`n = 0/1/2/3/7/2^k±1/≥2^20`、`incx = 1/2/3`、`incx = 0/-1/-2/-3`（quick return）、
  负 `n`、`x` 空、`result` 空、跨度溢出、全零/常量（并列最小索引）、正负交替、极端值、`Inf`、`NaN`。

### 必须补充的用例（随任务 CSV 未覆盖）

1. **`GAUSS_NORM_5_2` 正态分布**：随任务生成器只产出 `RANDOM_NORM_5_5`（均匀），任务书 §3.5 要求
   正态（μ∈[-5,5]、σ∈[0.1,2]）。补约 60 条，覆盖尺寸扫描 × 步长。
2. **混合 `NaN` 语义**：随任务 CSV 的 130 条 NaN 用例全是 `VALUE_NORM_NAN`（常量 NaN，结果恒为 1），
   覆盖不到“首元素 NaN → 1”和“后续 NaN 被忽略”。补：首元素 NaN、尾元素 NaN、中间 NaN、
   NaN 与 `Inf` 混合、首个 `Inf` 与多个 `Inf`，约 12~16 条，golden 由 cblas 生成。
3. **性能用例的 `[PERF]` 输出**：`verify_performance.py` 只解析 GTest 打印的
   `[PERF] case=<case> ... avg_us=<数值>`；arch35 的测试没有该输出，arch22 测试必须自带
   （warmup 5 次，`aclrtEvent` 夹住连续 ≥10 次下发取均值，H2D/D2H 与 golden 不计入计时窗口）。

### 负向用例的缓冲区规则

期望返回值非 `SUCCESS` 的用例不得按 `(n-1)*|incx|` 预分配：测试封装在 `n <= 0`、`incx <= 0`
或 `x == nullptr` 时**不分配**输入缓冲，以 `nullptr` 或单元素 dummy 传入，由接口先行返回错误码。

### 复现命令

```bash
bash build.sh --soc=ascend910b --ops=icamax          # A2；A3 用 --soc=ascend910_93
python verify_accuracy.py --repo <repo> --soc ascend910b --csv <csv> --skip-build
python verify_performance.py --repo <repo> --soc ascend910b --skip-build
```

## 兼容性分析

- 仅新增 `blas/iamax/arch22/icamax_*.{h,cpp}`、`test/iamax/icamax/arch22/` 与 CSV，并给
  `test/iamax/icamax/CMakeLists.txt` 增加 arch22 分支；不改动 arch35 的 icamax 实现与测试。
- 接口沿用 `include/cann_ops_blas.h:242` 已有声明，不新增私有 API、不改参数类型。
- `blas/iamax/README.md` 的产品支持表把 `aclblasIcamax` 的 A2/A3 改为“支持”，并补充约束说明。
- `SyncAll`/workspace 只使用本算子自己的槽位，不与同 handle 上其他算子共享状态。

# 性能优化路线

## 判定方法

1. **先定天花板**：用同尺寸纯 `DataCopy` 读 kernel 测 8n 字节的搬运下界（copy-bound baseline），
   确认 4M case 在目标设备上的物理下限，再判断 0.8× 是否可达；不可达先与需求方对齐口径。
2. 每个候选：固定设备/CANN/编译选项/输入，5 次 warmup + ≥10 次采样，取均值并记录中位数/最小值；
   每轮只改一个维度，保留 A/B 数据与失败原因。
3. 每轮优化后跑全量精度/负向用例，精度不过不进入下一轮。

## 候选清单

| # | 类别 | 候选 | 预期收益 / 风险 |
|---|---|---|---|
| 1 | 算法 | `Max + 最小索引` 取代逐元素严格比较；`|Re|+|Im|` 免 `sqrt`、免平方 | 去掉标量分支与乘法，是能上向量归约的前提 |
| 2 | 算法 | NaN 掩码由“每 tile 无条件”改为“仅当 tile 最大值为 NaN 时对该 tile 重算” | 省每 tile 两次向量 pass；前提是实测确认硬件 `vmax` 对 NaN 的行为（当前实现无条件掩码，正确性优先） |
| 3 | 切分 | 256B 整组均分 + 余组轮转 + 尾块并入末核；小 n 单核 | 消除末核长尾与空核；小 n 省 `SyncAll` |
| 4 | 搬运 | 32KB tile 双缓冲，MTE2 与 V 重叠；尾块一次 `DataCopyPad` | 让 MTE2 成为唯一瓶颈；tile 加大可再降循环开销（UB 有余量） |
| 5 | 计算 | `Abs(2n)` + `PairReduceSum(n)` 一趟出比较值；`ReduceMax` 只取值 + 更新时定位索引 | 中间结果不落 GM、不建索引向量；索引定位每核仅数次 |
| 6 | 同步 | 单 kernel + `SyncAll` + 核 0 小规模归约（≤2×核数字） | 省一次 launch；核间只搬几十字节 |
| 7 | host | handle workspace + tiling 按值传参，零 per-call 分配；quick return 只 memset | 去掉 16MB `aclrtMalloc`/H2D/`aclrtSynchronizeStream` |
| 8 | 搬运/布局 | `incx > 1` 用向量化 stride gather（`GatherMask` + 掩码缓存，参照 `blas/axpy/arch22/caxpy_host.cpp`）替代逐元素 gather | 只影响步长用例（非 TC_PF 判定），需与标量版 A/B，赢才留 |
| 9 | 备选 | 两 kernel 分阶段（每核摘要 + 单核 reduce） | 仅当单 kernel 的 `SyncAll` 成为瓶颈时采用，需计入 launch 开销 |
| 10 | 备选 | 原子操作（`SetAtomicMax`） | 无法成对携带 index，并列语义无法保证，预计淘汰，保留结论 |

# 风险与备选方案

## 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| HBM 带宽不足导致 0.8× 不可达 | 4M case 需要 ≥1.28× GPU 带宽才达标 | 先做 copy-bound 测量；数据写入报告并如实说明，不通过改口径“达标” |
| `NaN` 的 `vmax` 语义与预期不同 | 特殊值结果错误 | 实现不依赖硬件 NaN 语义：每个 tile 无条件把 NaN 替换为 `-FLT_MAX`；用混合 NaN 用例覆盖“首元素 NaN / 后续 NaN / 全 NaN” |
| 并列最大值依赖硬件归约索引 | 取错索引 | 不依赖 `WholeReduceMax` 的索引，软件用等值掩码 + 首个置位定位 |
| `incx > 1` 走标量 gather 较慢 | 步长用例耗时长（非性能判据） | 作为候选 8 做 A/B；必要时按 `caxpy` 的掩码 gather 方案实现 |
| CMake arch22 分支漏注册 | A2/A3 静默退回非 GTest runner，精度用例不生效 | 显式 `ops_blas_add_gtest_tests(${OPS_BLAS} icamax_test)`，并在 A3 上确认测试目标存在 |
| A3 与 910B3 差异 | 性能数据设备口径不同 | 报告注明设备；如验收要求 910B3 数据则补测并标注 |
| `n` 接近 int32 上限 | 地址越界 | host 侧 int64 计算 `(n-1)*incx` 并拒绝越界（issue #745） |

## 备选实现

1. **单 kernel（本设计）**：核内归约 + `SyncAll` + 核 0 跨核归约。优点：一次 launch，摘要仅几十字节。
2. **两 kernel**：每核写摘要后由独立 reduce kernel 合并。优点：避免 `SyncAll`；缺点：多一次 launch，
   小 n 时固定开销占比高。仅在 1 的 `SyncAll` 被测量证明是瓶颈时启用。
3. **步长路径两选**：逐元素 gather（本设计，简单、与仓内先例一致） vs 掩码向量 gather（快、复杂）。
   先实现前者保证正确性，再按 A/B 数据决定是否替换。

# 附：与参考实现的借鉴与差异

| 来源 | 借鉴点 | 差异 |
|---|---|---|
| `blas/iamax/arch22/isamax_*` | 直调框架、256B 整组分核、value+index 成对归约、`SyncAll`+核 0 归约、双缓冲 tile | 补齐 `incx` 支持；不复用其 quick return 的 `result` 空指针行为；不复用其每调用 malloc/sync；`isamax` 的复数分支在其 host 中未启用，本算子自行实现 |
| `blas/iamax/arch35/icamax_*` | 校验顺序与错误码、quick return 用 stream memset、`(n-1)*incx` 溢出守卫、“首元素 NaN → 1”的判定 | arch22 无 SIMT/RegBase，`DeInterleave`/`asc_vf_call` 不可用；改用 `Abs`+`PairReduceSum`；NaN 不做整段重扫 |
| Netlib `icamax.f` | 严格大于更新、`IX=1` 种子、1-范数比较 | 1-based 输出、整数索引 bit-exact 比对 |
| cuBLAS `cublasIcamax` | 接口参数顺序、quick return 语义、最小索引承诺 | 不承诺负步长反向遍历 |
