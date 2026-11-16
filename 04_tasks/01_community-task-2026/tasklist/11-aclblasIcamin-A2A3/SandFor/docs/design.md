**aclblasIcamin for Atlas A2/A3**

# 需求背景（required）

## 需求来源

November community task `6ff54aa152cf4edbbaf516fa36024893` requires
`aclblasIcamin` in `cann/ops-blas`. The existing public declaration and complex
layout are shared with arch35. This design adds arch22 implementation and tests;
it does not introduce a second API.

## 背景介绍

任务要求在 A2/A3 上提供与现有复数 BLAS 接口一致的最小模索引功能。
现有 arch35 SIMT 路径不能直接作为 arch22 的交付实现；本次设计在
`blas/iamin/arch22/` 补充 Ascend C 直调路径，并在
`test/iamin/icamin/arch22/` 注册精度测试。输入保持交错 complex64，
输出保持设备侧 int32 索引，调用方无需改变 handle、stream 或内存布局。

The arch35 implementation uses SIMT and is not an A2/A3 implementation.
The existing complex test registration selects GTest only for arch35. Add the
arch22 registration and retain the existing arch35 sources unchanged.

### 现状、动机与范围

Icamin supplies the index used by BLAS callers to select the smallest complex
magnitude without copying the input vector to the CPU. A2/A3 support closes the
architecture gap between the shared public API and the existing arch35 backend.
The result is a selection index, not a Euclidean norm: using sqrt(re*re+im*im)
would change the ordering and violate the API even if a reduction were faster.

The change is limited to arch22 host dispatch, kernels, test registration and
product support documentation. Existing arch35 behavior and the shared handle,
stream, complex storage and error enumeration remain the compatibility boundary.
No matrix shapes, batching, negative-stride traversal or new public parameters
are introduced. The design can be reviewed independently of code acceptance;
all device results remain outstanding and are not implied by this document.

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n,
                            const aclblasComplex* x, int incx, int* result);
```

The API takes a valid handle, `int n`, device `aclblasComplex* x`, `int incx`, and
a device `int* result`. Each complex element consists of two interleaved FP32
components. Return the smallest 1-based logical index minimizing
`abs(real) + abs(imag)`. Equal magnitudes retain the earliest index.

Validate handle first, then reject negative n or null result. For n=0 or incx<1,
write zero asynchronously on the handle stream without launching an operator
kernel. For normal calls reject null x and `(n-1)*incx > INT32_MAX`, calculating
the product in int64. The handle error is `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`;
invalid arguments return `ACLBLAS_STATUS_INVALID_VALUE`.

Seed semantics follow the task's CPU loop: if the first logical element contains
NaN, return 1; later NaNs do not become winners. Infinity, zero, overflow to
infinity in the magnitude sum, and ties must preserve the smallest index.

## 需求拆解

| 参数 | 含义与类型 | 形状/布局 | 约束及错误处理 |
| --- | --- | --- | --- |
| handle | 既有 aclblasHandle_t | 含调用流及有效 workspace | 空 handle 优先返回 HANDLE_IS_NULLPTR |
| n | int，逻辑复数元素数 | 一维向量长度 | n<0 返回 INVALID_VALUE；n=0 走 quick return |
| x | 设备侧 const aclblasComplex* | 每个元素两个 FP32，物理下标 i*incx | 正常计算不可为空；quick return 不读取 x |
| incx | int，复数元素步长 | 正常输入为正跨度 | incx<1 写0；以 int64 校验 (n-1)*incx 不超过 INT32_MAX |
| result | 设备侧 int* | 单个 int32 | 包括 quick return 在内均不可为空；有效结果为1-based下标 |

需求分为接口校验、带步长读取、最早索引归约和设备验收四部分。
精度的验收对象是整数索引，不能使用 FP32 误差容限；性能按每个 GPU
基线用例分别判断，不能以平均加速比掩盖慢用例。内存无数值门槛，
仍需提交实际内存报告和日志。设计评审、报名获批、设备验收各自独立。

| Requirement | Design decision | Required verification |
| --- | --- | --- |
| FP32 complex vector, positive stride | Read logical element i from physical complex offset i*incx | CPU oracle versus device integer output |
| Earliest 1-based index | Keep an integer index and compare magnitude/index pairs | Ties within a tile, across tiles and across cores |
| Asynchronous execution | Use the handle stream for both phases; no host result readback | Repeated calls on the same stream and synchronization only in the harness |
| Invalid arguments and quick return | Validate before address arithmetic; enqueue exactly four output bytes for valid quick returns | Null handle/result/x, negative n, nonpositive incx and overflow boundary |
| Exact correctness | Never convert global indices to FP32 | Include lengths around 2^24 where float indices lose integer precision when feasible |
| Per-case performance limit | Account for every operator phase | Every supplied GPU baseline row passes the mean-duration threshold |
| No numerical memory threshold | Bound UB and workspace and report observations | Allocation accounting, tail guards and raw memory logs |

Input and output addresses refer to device memory. The normal input extent is
`((n-1)*incx+1)*sizeof(aclblasComplex)`; the output extent is `sizeof(int)`.
The host rejects the task's int32 stride-product overflow before launching.
Quick returns do not require a valid input pointer because they do not read x.
A null result remains an error even for quick returns. Exact failure status and
validation precedence are covered separately from numerical tests.

# 详细设计（required）

## 算子分析

### 数学公式

正常调用的逻辑输入为 `z_i=x[i*incx]`，`0<=i<n`。
计算 `m_i=abs(real(z_i))+abs(imag(z_i))`，返回使 m 最小的最早逻辑下标加1。
例如 `(3,0),(2,2)` 的 BLAS 模分别为3和4，答案为1；若误用欧氏模，
答案会变为2。并列例 `(1,2),(-3,0)` 返回1。

数据类型为 complex64 输入、FP32 模值、int32 输出；不支持广播、反向遍历、
批处理或额外布局。有效长度至 int32 接口范围，实际地址使用64位计算。
每核处理连续逻辑区间，比较 `(模值,全局整数下标)` 时以模值优先、下标次优；
因此按任意分核边界合并均保留最早索引。首元素 NaN 是 CPU 顺序循环的特殊
种子语义，必须单独传递覆盖结果，不能把它当普通的正无穷候选。
后续 NaN 被忽略；全 Inf、正负零或模值加法溢出为 Inf 时仍选择最早元素。

For logical zero-based i, let `m(i)=abs(x[i*incx].real)+abs(x[i*incx].imag)`.
For a finite first magnitude, the oracle starts with `(m(0),0)` and replaces the
pair only when a later magnitude is strictly smaller. The final answer is i+1.
A NaN first magnitude makes every strict comparison false, hence answer 1;
a later NaN is ignored. This seed rule must be preserved separately from vector
NaN masking. Replacing every NaN with infinity alone is insufficient when the
first element is NaN and a later element is finite.

Within each slice, merge candidates using strict magnitude comparison in logical
order. Across slices, select by magnitude first and then by the smaller integer
index. Infinity initialization requires an explicit first valid candidate so an
all-infinity slice does not publish an invalid sentinel index. The global seed
NaN override is applied only for logical element zero, not the first element of
each slice. Signed zeros compare equal and retain the first logical index.

Arithmetic work is O(n), with two absolute values and one addition per element;
there is no multiplication or square root. Contiguous input has approximately
8*n input bytes and four output bytes. For stride s<=2048, copying the physical
span increases transferred bytes toward s times the useful input bytes; Gather
reduces computation on unused components but cannot remove those transfers.
For s>2048, scalar reads avoid a mostly unused 2048-element DMA window but may be
latency-bound. These are performance risks to measure, not accepted exceptions.

A one-stage single-core path avoids the second launch for small inputs. The
multi-core path trades an extra launch and O(active_cores) workspace traffic for
parallel scans. The initial 2048 cutoff is a tunable implementation starting
point; retain it only if the required per-case measurements support it. Any
adjustment must be retested on the final revision without changing case coverage
or the 1.25-times-GPU limit. Workspace grows with active cores, not vector length.

## 算子实现

### 实现方案

#### host 侧设计与 tiling 策略

Use the existing handle stream and effective workspace. Partition n into
contiguous logical slices. The initial core count is the smaller of the AIV core
count and `ceil(n/2048)`; ceil-divide slices and recompute the active core count.
A single-core input writes the final result directly. Larger inputs launch one
scan kernel followed by one reduction kernel on the same stream. The latter
provides ordering between phases without a cross-core spin barrier.

The workspace has one 32-byte slot per active core, storing an FP32 minimum and
an integer index in separate 4-byte fields. Integer indices are not converted to
FP32 numerical values. The existing effective workspace supplies storage; no
per-call allocation or host synchronization is introduced.

#### kernel 侧设计

Each core processes tiles with at most 2048 physical complex elements. For
stride s<=2048, a tile has `floor(2048/s)` logical elements. Copy only the
physical span through the last required complex element using DataCopyPad;
Gather byte offsets select real and imaginary components into separate local
buffers. Abs and Add produce the BLAS magnitude. Compare/Select replace NaNs
with positive infinity. Reduction pads the magnitude tile to 64 lanes with infinity. ReduceMin returns
a tile value and earliest local index;
short tiles use scalar comparisons of the computed local magnitudes. Strict
less-than comparisons merge tiles and cores in increasing logical-index order.
For s>2048 the initial implementation uses scalar strided reads with uint64
address arithmetic. This path remains subject to the same acceptance criteria.

The first-element NaN rule is checked before publishing the final index. The
all-infinity case retains index 1. Each workspace write is isolated by 32 bytes.
The final output copy writes exactly four bytes, including for unaligned logical
input pointers; padding must not cause input overreads or output overwrites.

## 支持硬件

| 硬件 | 设计用途 | 验证状态 |
| --- | --- | --- |
| Atlas A2/A3（arch22） | 精度支持至少一种 | 待真机验证 |
| Atlas 800T A2（910B3） | 任务指定性能验收 | 待真机验证 |

## 算子约束限制

Target Atlas A2/A3, with one model sufficient for accuracy self-validation.
Required performance hardware is Atlas 800T A2 (910B3); CANN version is 9.1.0.
No broadcast, reverse traversal, in-place update, or extra tensor layout is
required. Memory has no numerical acceptance threshold. Actual environment and
memory observations will be recorded rather than inferred from a selectable
image. Fixed UB allocation in the draft is 51,232 bytes per core, excluding
implementation-private API scratch/TPipe bookkeeping; verify it during build.

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 标准来源与证据要求 |
| --- | --- | --- |
| 精度 | 所有有效用例索引与 CPU golden 逐位相同 | 本题任务书；记录输入、期望/实际整数及状态 |
| 非法输入 | 状态码、校验优先级、quick return 输出符合接口约定 | 本题任务书；独立记录负向用例，不能只比较结果值 |
| 性能 | 每行 mean_us <= GPU_ms*1000*1.25 | 本题 GPU 基线；5次预热后至少10个SUCCESS有效样本，多阶段求和 |
| 内存 | 任务本体无数值阈值，仍检查越界与 workspace/UB 用量 | 本题交付模板；实际日志及内存报告，不填造测量值 |
| 环境 | CANN 9.1.0；A2/A3至少一种精度验证，910B3性能验证 | 本题任务书；保存设备型号、版本、命令和源码SHA |

All 1200 supplied CSV cases remain unchanged. Add 1141 cases, including Gaussian
counterparts, mixed NaNs and complex-aligned but DMA-unaligned inputs. There are
1105 uniform and 1105 Gaussian cases. Negative and quick-return tests use a dummy
input allocation instead of sizing buffers from invalid dimensions or strides.
GTest compares results with the shared CPU reference using exact integer equality.

Before accepting a revision, build on the required toolchain, execute every
accuracy case and supplement any missing boundary coverage. On 910B3, use msprof
op task-duration samples in microseconds after five warmups, retaining at least
ten successful samples. For two-kernel calls include both scan and reduction
durations. Each GPU baseline row must satisfy `mean_us <= gpu_ms*1000/0.8`.
Keep raw profiler output, per-case sample counts and failures. GTest wall time
is not a performance acceptance measurement.

## 精度用例与复现策略

The future implementation keeps the supplied CSV unchanged and generates
supplemental cases with recorded seeds and distribution parameters. The proposed
2341-case inventory includes 1105 uniform and 1105 Gaussian cases; edge/error
cases are recorded separately from that random-distribution balance. Historical
draft counts are planning data, not proof that an executable suite is delivered.

The harness allocates valid dummy storage for invalid dimensions/strides and
checks both status and output behavior. It computes the reference with the
specified CPU loop and compares device output as an integer after stream
synchronization. Use fixed patterns to force the winner at index 1, the last
index, a tile boundary and a core boundary. Include repeated minima, all-zero,
all-infinity, leading/later NaN, magnitude overflow, positive/negative alternation,
lengths 0/1/small primes/powers of two and adjacent lengths through at least 2^20,
strides 1/2/3 and invalid 0/-1/-2/-3. Check overflow cases without allocating their
nominal enormous input. Complex-aligned input offsets exercise DMA misalignment.

Guard the input tail and four-byte result with sentinel storage; verify no
unexpected writes and exercise tails shorter than the copy/reduction alignment.
For asynchronous ordering, queue input initialization, the operator and result
copy on the same stream, then synchronize once. Run repeated calls with different
inputs to detect stale workspace state. No acceptance case is dropped because a
fast path or the scalar-stride path is slower.

## 构建、性能采样与故障定位

Record the exact source SHA, build command and exit code, CANN/compiler version,
OS, chip model, physical-device identity and runtime configuration alongside each
run. Use the repository's arch22 build and the registered Icamin GTest target;
the concrete target/filter and msprof command must be taken from the implemented
build and installed profiler help rather than invented here. Preserve those
literal commands in the validation report so a second developer can rerun them.

For each performance row, retain five warmups followed by at least ten successful
samples, units, phase names, correlation identifiers and raw profiler files.
For multi-kernel invocations, pair scan/reduce phases from the same invocation
and sum their task durations before computing the mean. Compare against the
matching GPU row after converting milliseconds to microseconds. Missing samples,
non-success task status or an unmatched phase fail the measurement rather than
reducing its denominator. GTest process time and launch-only host time are not
substitutes. Keep slower cases in the result table with their ratios and causes.

A failure record contains input seed/shape/stride/offset, expected/actual status
and index, source SHA, environment, raw log and a minimal rerun command. Diagnose
numerical failures by separating host validation, slice bounds, tail handling,
local tie selection, workspace publication and final reduction. Investigate
performance failures by size/stride/active-core grouping and phase durations.
After a fix, rerun affected cases and then the full mandatory final-revision
suite; evidence from a previous SHA cannot certify the changed revision.

## 可维护性与交付边界

Keep arch22 implementation and tests in the stated existing operator hierarchy;
reuse the handle/workspace contract and product registration instead of adding a
parallel API or framework. Keep tiling parameters and workspace fields explicit
so logs can identify the selected path. Documentation changes use the single
`docs/design.md` required by the design PR; code, profiler output and acceptance
packages do not enter this PR. CLA/CI status, design review and device acceptance
are independent gates. Platform enrollment is currently rejected; this revision
only addresses feedback on the already submitted design contribution and does
not resume implementation, device allocation or platform delivery.

## 兼容性分析与交付

The public signature and arch35 implementation remain shared/unchanged. Add code
in `blas/iamin/arch22/` and GTest/CSV in `test/iamin/icamin/arch22/`; update the
existing CMake registration and complex-interface product support table.

Submit this design through the required design PR and obtain review/merge.
After final-revision validation and enrollment approval, deliver the private
code commit, required developer invitation and template-named `task_submission`
materials through the task system and read back the receipt. Submit the upstream
code PR after platform testing succeeds. Do not merge on the user's behalf.

## 当前验证状态

This is a design draft. The historical local draft was inspected for CSV
preservation and balanced distributions, but its unsubmitted implementation and
test artifacts were removed following enrollment rejection. No CANN compilation,
NPU accuracy, performance or memory measurement has passed yet. Remaining work
includes verifying Ascend C tail
handling, toolchain API signatures, profiler invocation/output mapping, and
resource shutdown protection before starting a metered test window.
