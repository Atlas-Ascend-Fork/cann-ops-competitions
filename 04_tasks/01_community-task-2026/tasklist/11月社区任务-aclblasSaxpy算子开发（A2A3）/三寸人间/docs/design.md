# aclblasSaxpy 算子设计文档

文档头信息表（任务、团队、仓库、目录与状态）：

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 11月社区任务-aclblasSaxpy 算子开发（A2/A3） |
| 提交团队 | 三寸人间 |
| 任务讨论 | [ops-blas discussions #73](https://gitcode.com/cann/ops-blas/discussions/73) |
| 设计文档提交目录 | `04_tasks/01_community-task-2026/tasklist/11月社区任务-aclblasSaxpy算子开发（A2A3）/三寸人间/docs/design.md` |
| 适配硬件 | Atlas A2/A3，`arch22`；首要验证平台 `ascend910b3`，A3 使用 `ascend910_93*` |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas`（开发分支 `master`） |
| 代码目标目录 | `blas/axpy/arch22/`、`test/saxpy/arch22/` |
| 文档状态 | 设计评审阶段；本文只描述设计，代码与自验证证据随后续 `ops-blas` 代码 PR 提交 |

# 需求背景（required）

## 需求来源

本算子来自 2026 年 11 月 CANN 社区任务「11月社区任务-aclblasSaxpy算子开发（A2A3）」，目标是在 `cann/ops-blas` 中补齐 Atlas A2/A3（`arch22`）上的单精度实数 AXPY 接口（BLAS-1），完成设计评审、算子开发、自验证与社区 PR 合入。

语义基准有三处，冲突时按下列顺序裁决：

1. cuBLAS `cublasSaxpy`（CUBLAS_Library §2.5.4）：参数顺序、返回码与快速返回口径；
2. Netlib `saxpy.f`：`IF (N.LE.0) RETURN`、`IF (DA.EQ.ZERO) RETURN` 的 quick-return 语义；
3. 仓内 950PR 既有实现 `blas/axpy/arch35/saxpy_host.cpp`（`ValidateSaxpyParams` 与 `aclblasSaxpy` 入口，L20-160）：校验顺序与返回码落地口径。

## 背景介绍

### aclblasSaxpy 算子实现

算子完成 `y ← alpha·x + y` 的向量原地更新：`x`、`y` 为一维 FLOAT32，`alpha` 为 Host 内存标量，`y` 原地覆写。接口沿用仓内公共声明 `include/cann_ops_blas.h`：

~~~cpp
aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha,
                             float* x, int incx, float* y, int incy);
~~~

头文件中 `x` 为非 `const` 指针（cuBLAS 为 `const float*`），属既有 ABI，本任务不改动，实现内部禁止写入 `x`。

### 现状分析

- `ops-blas` 已有公共声明 `include/cann_ops_blas.h` 与 `blas/axpy/README.md`；`blas/axpy/arch35/` 为 950PR 实现，`blas/axpy/arch22/` 目前只有复数版本 `caxpy`，缺少单精度实数实现。
- 测试侧 `test/saxpy/` 原有用例在 `arch35`，`arch22` 用例随本任务补齐；`test/saxpy/CMakeLists.txt` 需在 `ascend950` 或 `SOC_ARCH_DIRS` 含 `arch22` 时注册 GTest（本任务随代码 PR 补齐该注册），否则 A2/A3 构建退回非 GTest runner。
- `arch35` 的非连续路径依赖 SIMT 线程级逐元素寻址，`arch22` 无 SIMT，不能复用；其 Host 校验与快速返回口径可直接对齐。

### 算子功能分析

| 参数 | 含义 | 类型/位置 | 约束与异常行为 |
| --- | --- | --- | --- |
| handle | BLAS 句柄，携带 stream | Host | 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | 向量元素个数 | Host `int` | `n <= 0` 为合法 no-op，不访问 alpha/x/y，返回 SUCCESS |
| alpha | 标量乘数 | Host `const float*` | `n > 0` 时不可为空；值不校验（含 0、Inf、NaN） |
| x | 输入向量 | Device `float*` | `n > 0` 时不可为空；物理跨度 `1+(n-1)×abs(incx)` |
| incx | x 的步长（元素） | Host `int` | 非 0、非 `INT32_MIN`，正负均可 |
| y | 输入兼输出向量 | Device `float*` | `n > 0` 时不可为空；原地覆写；跨度 `1+(n-1)×abs(incy)` |
| incy | y 的步长（元素） | Host `int` | 同 incx |

## 口径裁决

1. `n <= 0` 一律返回 SUCCESS 且不访问 `alpha/x/y`，即使同时传入空指针或零步长；该裁决优先于所有参数校验（Netlib、cuBLAS、仓内 arch35 一致）。
2. `n > 0` 时 `alpha`、`x`、`y` 任一为空，或 `incx == 0`、`incy == 0`、`incx == INT32_MIN`、`incy == INT32_MIN`，均返回 `ACLBLAS_STATUS_INVALID_VALUE`，不进入 kernel。
3. `(n-1)×abs(inc)` 超出 int32 可表示范围（`> INT32_MAX`）返回 `ACLBLAS_STATUS_INVALID_VALUE`；Host 侧统一用 int64 计算后再比较，避免 int 溢出（issue #780/#866 口径）。
4. `alpha == 0.0F`（含 `-0.0F`）按 Netlib `IF (DA.EQ.ZERO) RETURN` 直接返回 SUCCESS：`y` 不变且不读取 `x`；空指针仍按第 2 条先校验，故该快速返回不会静默接受空指针。
5. 负步长语义：物理 span 固定为 `[0, (n-1)×abs(inc)]`，逻辑第 `i`（0 基）个元素位于 `(n-1-i)×abs(inc)`（自 span 尾端起算）；`x`、`y` 各自独立映射。
6. `alpha` 由 Host 同步解引用后按值下发；NaN/Inf 按 IEEE 规则自然传播，不做特殊处理。
7. 返回码优先级：`HANDLE_IS_NULLPTR` > `INVALID_VALUE`（参数） > `NOT_INITIALIZED`（stream 未初始化） > `INTERNAL_ERROR`（核数获取失败）。

## 设计阶段范围

本 PR 只提交设计文档，不提交 `ops-blas` 算子代码、测试工程、CSV 用例或自验证日志。算子代码、测试工程、CSV 与自验证证据在 `ops-blas` 的独立代码 PR 中提交并引用本文；本文不写验收截图，也不宣称 PASS 数量、实测耗时或性能提升比例，只给出可判定的验收标准与判据。

# 需求分析（required）

## 需求描述

使用 Ascend C kernel 直调方式，在 `blas/axpy/arch22/` 实现 `aclblasSaxpy`，由句柄绑定的 stream 异步启动 NPU kernel，仅支持 FLOAT32；不新增私有平行接口、不修改公共 ABI 与参数顺序、不使用额外 workspace。

## 需求拆解

1. 实现 `y = alpha·x + y` 的原地向量计算，语义与 cuBLAS/Netlib/仓内 arch35 一致。
2. 完成 Host 参数校验、`n <= 0` quick return、`alpha == 0` 快速返回与标准状态码。
3. 支持正负步长的非连续访存，`(n-1)×abs(inc)` 跨度在 Host 侧按 int64 校验。
4. 连续路径做 SIMD 双缓冲流水；非连续路径在 arch22 无 SIMT 的前提下用「物理 span 连续搬运 + UB 索引压缩」实现。
5. 以 `cblas_saxpy` 为 golden，满足 FLOAT32 mixed_tolerance 精度标准与 Inf/NaN 等价判定。
6. 在 Atlas A2（910B3）上按任务书采样口径完成性能自测，平均单次耗时达到 A100 标杆 0.8 倍以上。
7. 交付 `blas/axpy/arch22/` 算子代码、`test/saxpy/arch22/` 测试工程、CSV 用例、算子 README、自测报告与个人仓分支。

# 详细设计（required）

## 算子分析

### 数学公式

~~~text
for i = 1..n:
    k = 1 + (i - 1) * incx      # x 的元素位置
    j = 1 + (i - 1) * incy      # y 的元素位置
    y[j] = alpha * x[k] + y[j]
~~~

`n <= 0` 时不修改 `y`；`alpha == 0` 时不修改 `y` 且不读取 `x`。kernel 内改用 0 基下标：`y[j0] = alpha * x[i0] + y[j0]`，`i0`、`j0` 由下节映射得到。每对元素只做一次乘加，`y` 单遍原地更新，无临时累加量。

### 向量步长

逻辑下标（0 基）到物理下标的映射：

~~~text
inc > 0: offset(i) = i * abs(inc)
inc < 0: offset(i) = ((n - 1) - i) * abs(inc)
~~~

由此得到两条不变式：

- 物理 span 恒为 `[0, (n-1)×abs(inc)]`，长度 `1+(n-1)×abs(inc)`，与步长符号无关；
- 负步长只是逻辑遍历方向相反，不需要对数据整体反序。

kernel 内按 tile 处理时（`lo` 为 tile 起始逻辑下标，`cnt` 为 tile 元素数，`N = n`）：

~~~text
inc > 0: spanStart = lo * abs(inc);                spanLen = (cnt - 1) * abs(inc) + 1
inc < 0: spanStart = (N - lo - cnt) * abs(inc);    spanLen = (cnt - 1) * abs(inc) + 1
tile 内第 i 个元素的 span 下标：inc > 0 ? i * abs(inc) : (cnt - 1 - i) * abs(inc)
~~~

### 支持数据类型和形状

- `alpha`、`x`、`y` 均为 FLOAT32；`n`、`incx`、`incy` 为运行时参数，不要求 dynamic-shape 编译支持。
- 逻辑形状一维 `[n]`，物理长度 `1+(n-1)×abs(inc)`；任意 `n > 0`、任意非零且非 `INT32_MIN` 的正负步长均可。
- 不涉及 broadcast、多维张量与视图语义；`y` 原地输出，不返回新视图；不使用 workspace。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计

Host 侧完成校验、快速返回、tiling 与异步派发，入口 `aclblasSaxpy` 的顺序固定为：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `n <= 0` → `ACLBLAS_STATUS_SUCCESS`，不访问 `alpha/x/y`；
3. 空指针与步长校验（`ValidateSaxpyParams`）：`alpha`、`x`、`y` 依次判空；`incx`、`incy` 判 `0` 与 `INT32_MIN`；`SaxpySpanOverflow` 判 `(n-1)×abs(inc) > INT32_MAX`（int64 计算）；任一不满足返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `handle->stream == nullptr` → `ACLBLAS_STATUS_NOT_INITIALIZED`；
5. `alphaHost = *alpha`；`alphaHost == 0.0F` → `ACLBLAS_STATUS_SUCCESS`（不读 `x`）；
6. `coreNum = GetAivCoreCount()`；为 0 → `ACLBLAS_STATUS_INTERNAL_ERROR`；
7. `CalSaxpyTilingData` 计算分核参数，`saxpy_kernel_do` 以 `activeBlocks` 个 AIV 核在 `h->stream` 上异步下发。

该顺序保证错误路径不解引用空指针，且 `n <= 0` 叠加空指针或零步长时仍返回 SUCCESS；`alpha == 0` 判定放在校验之后，避免空指针被静默接受。`alpha` 为 Host 内存，第 5 步才第一次解引用。

Host 与 Kernel 共享的 tiling 结构按值传入 kernel（避免每次下发额外的 GM tiling 拷贝），字段与 arch35 公共字段保持同名：

~~~cpp
struct SaxpyTilingData {
    uint32_t totalN;        // 逻辑元素个数 n（> 0）
    uint32_t activeBlocks;  // 本次下发的 AIV 核数
    uint32_t perBlockN;     // 每核基础元素数
    uint32_t remainder;     // 余数
    uint32_t tileSize;      // 连续路径每 UB 分块元素数（12288）
    float    alpha;         // 已解引用的标量
    int64_t  incx;          // x 步长（带符号）
    int64_t  incy;          // y 步长（带符号）
};
~~~

分核常量：`SAXPY_UB_BYTES = 192 KiB`、`SAXPY_QUEUE_NUM = 2`、`SAXPY_BUFFER_NUM = 2`，`SAXPY_TILE_ELEMS = 192 KiB / (2×2×sizeof(float)) = 12288`（单缓冲 48 KiB）；`SAXPY_BLOCK_ALIGN = 32 B / 4 B = 8` 元素。

两套分核策略由 `dense = (incx == 1 && incy == 1)` 选择，Host 与 Kernel 必须使用同一判定：

| 策略 | 条件 | 核数 blocks | perBlockN | remainder 归属 |
| --- | --- | --- | --- | --- |
| 连续均分 | `incx == incy == 1` | `min(coreNum, max(1, n/8))` | `floor(n/blocks/8)×8` | 全部归最后一个核，保证每核 GM 起点 32B 对齐 |
| 非连续均分 | 其余 | `min(coreNum, max(1, n))` | `n/blocks` | 分散到前 remainder 个核，负载均衡 |

对应 kernel 取块口径：

~~~text
dense  : start = perBlockN * blockIdx;  count = perBlockN + (blockIdx == activeBlocks-1 ? remainder : 0)
strided: start = perBlockN * blockIdx + min(blockIdx, remainder);
         count = perBlockN + (blockIdx < remainder ? 1 : 0)
~~~

小 shape 时 `blocks` 自动收缩（例如 `n < 8` 时连续路径只有 1 个核），避免启动开销超过计算量；`tileSize` 由 Host 写入、kernel 不再重复推导，保证两侧常量一致。

#### 3.2.2 kernel 侧设计

kernel 入口 `saxpy_kernel` 声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，`GetBlockIdx() >= activeBlocks` 直接返回；按 `incx == 1 && incy == 1` 分两条路径。

**连续路径（SIMD 双缓冲流水）**：`SaxpyAIV` 用 `TPipe` 建两条 VECIN 队列（x、y），各 2 个缓冲 × 48 KiB，即 4 × 48 KiB 共占 192 KiB UB 预算。单个 tile：

1. `todo = min(tileElems, count - done)`，`aligned = todo - todo % 8`，`tail = todo - aligned`；
2. 对齐段 `DataCopy(xLocal, xGM[start+done], aligned)`、`DataCopy(yLocal, ...)`；
3. 尾块 `DataCopyPad`：`DataCopyExtParams{1, tail×4, 0, 0, 0}` 配 `DataCopyPadExtParams<float>{true, 0, 8-tail, 0.0F}`（右侧补 0，只计算有效长度）；
4. `EnQue`/`DeQue` 建立 MTE2→V 同步，`Axpy(yLocal, xLocal, alpha, todo)` 原地完成 `y += alpha·x`；
5. `SetFlag<HardEvent::V_MTE3>` / `WaitFlag` 后 `DataCopy`/`DataCopyPad` 回写 `y`；
6. `FreeTensor` 后 `SetFlag<HardEvent::MTE3_MTE2>` / `WaitFlag`，缓冲可被下一 tile 复用；事件 ID 由 `GetTPipePtr()->FetchEventID(...)` 获取。

该路径承担全部性能用例（性能子集恒为 `incx = incy = 1`）。

**非连续路径（物理 span 连续搬运 + UB 索引压缩）**：`SaxpyStridedAIV` 不使用 `TPipe`，直接在 UB 上划分两块 span 缓冲：x 位于偏移 0（`8192×sizeof(float) = 32 KiB`），y 位于偏移 32 KiB，合计 64 KiB，其余 UB 留给编译器。每个 tile：

1. `tileCount = (8192 - 1) / max(abs(incx), abs(incy)) + 1`，保证 `(cnt-1)×abs(inc) + 1 ≤ 8192`；
2. `asc_copy_gm2ub_align` 把 x、y 的物理 span 连续搬到 UB，GM 只经 DMA 访问；
3. `set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)` / `wait_flag` 建立搬运→标量同步；
4. 在 UB 内按 `idx = inc > 0 ? i×abs(inc) : (cnt-1-i)×abs(inc)` 压缩索引并原地计算 `ubY[idxY] = alpha×ubX[idxX] + ubY[idxY]`；
5. `set_flag(PIPE_S, PIPE_MTE3)` / `wait_flag` 后 `asc_copy_ub2gm_align` 把 y span 连续写回；
6. `set_flag(PIPE_MTE3, PIPE_MTE2)` / `wait_flag` 复用 UB。

x 与 y 在同一 tile 内都按逻辑升序处理，因此正负步长共用同一套索引公式，不需要对 UB 数据整体反序。

### 设计取舍

1. **单 kernel 双路径，而非一套通用慢路径**：连续路径用 `DataCopy`/`Axpy` 满带宽流水；非连续路径的标量索引压缩逐元素成本 O(cnt)，若合并成一条路径会把对齐、尾块和标量循环的开销带进性能热点。
2. **非连续路径不使用标量直接写 GM**：arch22 上标量写 GM 依赖 data cache，即使追加 `DataCacheCleanAndInvalid` + `DataSyncBarrier`，多核并发下仍会出现部分元素未落盘（同用例重复运行结果不稳定）。因此 GM 只经 DMA 读写，计算全部落在 UB，结果可复现。
3. **连续路径显式事件同步，而非依赖 auto-sync**：TQue 只保证 MTE2→V 的入队/出队同步；V→MTE3 与 MTE3→MTE2 用显式 `SetFlag`/`WaitFlag` 更稳妥，避免回写读到未完成的向量结果或缓冲提前复用。
4. **tiling 按值下发**：结构仅约 40 字节，按值传参省掉一次 tiling GM 分配与拷贝，也避免 AIV-only kernel 的 tiling 解析开销。
5. **不设 workspace、不做分段累加**：逐元素单遍计算，无中间累加量，FLOAT32 单指令完成，与 cuBLAS 的精度档位对齐。
6. **非连续路径不列入性能验收**：任务书性能用例恒为单位步长，非连续路径只需功能正确与结果可复现；其 span 搬运已按 32 KiB 缓冲对齐，不再为它引入更复杂的 UB 分块。

## 支持硬件

| 支持芯片 | 架构标识 | 状态 |
| --- | --- | --- |
| Atlas A2（任务书性能设备 Atlas 800T A2 / 910B3） | `ascend910b3` / `arch22` | 首要验证 |
| Atlas A3 系列 | `ascend910_93*` / `arch22` | 设计支持，代码 PR 验证 |

`ascend910_93*` 表示按 `ops-blas` 顶层 CMake 的 `ascend910_93` 前缀匹配；提交代码 PR 时以实际 CANN 工具链接受的完整 SOC 名为准。任务书对 A2 存在 800T/800I 两种产品别名，本文不据此推导不同编译目标，统一使用 `ascend910b3`。

## 算子约束限制

| 约束 | 处理 |
| --- | --- |
| 仅 FLOAT32 | 编译期固定接口类型，不做 dtype 分派 |
| `n <= 0` | 合法 no-op，返回 SUCCESS，不访问 alpha/x/y |
| `alpha/x/y` 为空（`n > 0`） | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == 0` 或 `incy == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx == INT32_MIN` 或 `incy == INT32_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| `(n-1)×abs(inc) > INT32_MAX` | `ACLBLAS_STATUS_INVALID_VALUE`（Host int64 判定） |
| `alpha == 0` | 返回 SUCCESS，`y` 不变且不读取 `x` |
| 广播、多维张量、视图语义 | 不支持 |
| 额外 workspace | 不使用 |
| 非连续路径 tile 上限 | `(cnt-1)×max(abs(incx),abs(incy))+1 ≤ 8192` 元素（32 KiB span 缓冲） |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 |
| --- | --- |
| Golden | `cblas_saxpy`（Netlib BLAS 3.10.0 参考实现，FLOAT32），比对原地更新后的 `y`，按 `abs(incy)` 跨步逐元素比较 |
| FLOAT32 精度 | mixed_tolerance：`rtol = atol = 2^-13`（约 1.2e-4），`required_matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 `≤ 32×ULP` |
| 判定公式 | `abs(actual - golden) ≤ atol + rtol×abs(golden)`，同时满足匹配率与最大绝对误差上限即通过 |
| Inf/NaN | 同为 NaN、同为同号 Inf、一方 NaN 一方 ±Inf 视为等效；一方有限一方非有限判不通过；非有限值不计入匹配率与最大误差，单独统计 |
| 精度用例 | 1000 条（任务书口径），覆盖尺寸扫描、正负步长、alpha 取值、填充模式与负向边界 |
| 性能用例 | 200 条连续访存用例（`TC_PF`，`n ∈ [1, 4194304]`，`incx = incy = 1`） |
| 性能标杆 | `gpu_baseline.csv` 的 A100 实测值（A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4） |
| 性能判据 | 平均单次耗时 ≤ 标杆耗时 / 0.8，即 `ratio = 标杆/NPU ≥ 0.8`（达到 A100 0.8 倍以上） |
| 采样口径 | warmup 5 次（不计入样本）后有效采样 ≥ 10 次取算术平均；实现按连续 20 次下发用 `aclrtEvent` 取设备侧均值，可用 `msprof` 的 Task Duration 交叉验证 |
| 内存 | 不涉及：BLAS-1 向量算子无 workspace 判据，不单设内存上限 |

以上为判据；实测数据、逐条 PASS/FAIL 与日志在 `ops-blas` 代码 PR 的自测报告中给出，本文不预设结论。

## 兼容性分析

- 公共接口只使用 `include/cann_ops_blas.h` 已声明的 `aclblasSaxpy`（L461），不新增私有平行 API、不修改参数顺序与返回码类型，与 950PR 共享同一声明。
- 实现放在 `blas/axpy/arch22/`，测试放在 `test/saxpy/arch22/`，沿用仓内 CMake 自动收集规则；`test/saxpy/CMakeLists.txt` 在 `ascend950` 或 `SOC_ARCH_DIRS` 含 `arch22` 时注册 GTest，与 arch35 共享 `saxpy_test` 目标。
- `test/saxpy/saxpy_param.h`、`saxpy_golden.h` 为 arch22/arch35 共用，CSV 列名与 `ReadMap` 键保持一致，新增用例不改列格式。
- 正负步长、原地覆写、quick return 与返回码口径与 cuBLAS/Netlib 一致，上层调用无需适配。

## 测试设计与自测流程

测试资产由任务包 `test_cases/` 提供，仓库侧测试工程位于 `test/saxpy/arch22/`：

- `saxpy_test.csv`：1200 条 CSV 驱动用例（1000 精度 + 200 性能），列格式 `case_name,description,n,incx,incy,alpha,x,y,random_seed,mere_threshold,mare_multiplier,expect_result`；`gen_csv.py` 固定种子 `20260823`，同种子两次生成逐字节一致。
- `gpu_baseline.csv`：200 条 A100 性能基线（任务包提供，`gpu_ms` 为 A100 实测值），与 `TC_PF` 按 `(n, incx, incy, alpha)` 匹配。
- `verify_accuracy.py`：安装 CSV、编译并运行 GTest，逐条解析 PASS/FAIL。
- `verify_performance.py`：运行 `TC_PF` 用例、解析 `[PERF] ... avg_us`，换算为 ms 后按 `ratio = 标杆耗时/实测耗时 ≥ 0.8` 判定，未采到样本按 FAIL 处理。
- `saxpy_test.cpp`：CSV 驱动的 `TEST_P`；精度用例走 `AclblasSaxpyOnDevice` + `applyMixedTolerance(ACL_FLOAT)` + `Verifier::verifyVector`，性能用例走 `RunPerfCase`（预置 device 缓冲、warmup 5、20 次连续下发取均值），另有 `NullHandle/NullAlpha/NullX/NullY` 四个接口级用例。

负向与 quick-return 用例的缓冲区规则：期望返回非 SUCCESS 或 `n <= 0` 的用例不按 `(n-1)×abs(inc)` 预分配（`INT32_MIN` 与跨度溢出的乘积在 int 域即溢出），统一走 `AclblasSaxpyValidation`，以 64 元素 dummy 缓冲或 `NULLPTR` 占位，由接口在读取数据前返回预期错误码。

正态分布覆盖：`gen_csv.py --dist mixed` 追加 `GAUSS_NORM_5_2` 填充（μ∈[-5,5]、σ∈[0.1,2]）；仓内 CSV 中带正态填充的用例共 184 条，其余为 `RANDOM_NORM_5_5` 均匀填充（[-5,5]）。逐用例的 x/y 填充按确定性规则二选一，代码 PR 的自测报告需按实际填充统计说明覆盖比例。

代码 PR 的目标验证流程（设计阶段只描述流程，执行证据随代码 PR 提交）：

~~~bash
cd ${REPO_ROOT} && source /opt/home/developer/Ascend/cann-9.1.0/set_env.sh
bash build.sh --ops=saxpy --soc=ascend910b
python3 ${TASK_ROOT}/test_cases/verify_accuracy.py --repo ${REPO_ROOT} --soc ascend910b \
  --skip-build --csv ${TASK_ROOT}/test_cases/saxpy_test.csv --device 0 --timeout 3600
python3 ${TASK_ROOT}/test_cases/verify_performance.py --repo ${REPO_ROOT} --soc ascend910b \
  --skip-build --device 0 --timeout 1800
~~~

A3 验证把 `--soc` 换成 `ascend910_93`，执行前需确认 `build.sh` 接受该 SOC 名称且验证脚本的架构映射落到 `arch22`。

### 测试覆盖矩阵

| 类别 | 用例前缀 | 覆盖内容 | 目标数量 |
| --- | --- | --- | ---: |
| 基础通路 | `TC_L0` | n=1/8，步长 (1,1)/(2,1)/(1,2)/(2,3) | 8 |
| 尺寸扫描 | `TC_SQ` | 1 到 1048576，含质数、2 的幂 ±1、非对齐值（38 条为 `GAUSS_NORM_5_2` 正态填充） | 38 |
| 步长 | `TC_INC` | 正步长 3×3 组合、负步长、正负混合 | 17 |
| alpha 取值 | `TC_AB` | 0、±1、±2.5、0.5、1e10、1e-30、Inf、NaN | 12 |
| 填充模式 | `TC_FL` | 0、交替值、极值、Inf、NaN，n=64/1024 | 24 |
| 边界与负向 | `TC_ED` | n≤0、零步长、空指针、`INT32_MIN`、跨度溢出 | 22 |
| 扩展采样 | `TC_EX` | 尺寸×步长×alpha×填充的确定性采样（含 146 条正态填充） | 879 |
| 性能 | `TC_PF` | 连续访存规模扫描、特殊填充、n∈[1,4194304] | 200 |
| **合计** | — | 精度 1000 + 性能 200 | **1200** |

## 设计评审与代码交付关系

1. 本 PR 只提交设计文档，目标文件为 `04_tasks/01_community-task-2026/tasklist/11月社区任务-aclblasSaxpy算子开发（A2A3）/三寸人间/docs/design.md`。
2. 设计评审通过后，算子代码提交到 `cann/ops-blas` 的 `blas/axpy/arch22/`，测试代码提交到 `test/saxpy/arch22/`，并在 PR 描述中引用本设计文档。
3. 代码 PR 附编译结果、精度/性能自测结果（内存不涉及）、采样次数与 `msprof` 交叉验证数据；设计阶段不填写未经实测的 PASS 数量、平均耗时或性能提升比例。
4. 交付件按任务书 §4 提交：设计文档、测试用例包、自测报告、个人仓地址/分支/算子目录，并邀请 `Ascend-CANN` 为开发者。

## 参考资料

1. [cuBLAS `cublasSaxpy`（§2.5.4）](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)
2. [Netlib `saxpy.f`](https://netlib.org/blas/saxpy.f)
3. [ops-blas 仓 axpy 目录](https://gitcode.com/cann/ops-blas/tree/master/blas/axpy)
4. [ops-blas 编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)
5. [生态算子开源精度标准 mixed_tolerance](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
6. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
