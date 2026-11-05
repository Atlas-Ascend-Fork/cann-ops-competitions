# 需求背景（required）

## 需求来源

华为社区任务《torch_npu 支持 kaiser_window 接口（A2A3A5）》。需求依据为
[官方任务页面](https://www.hiascend.com/activities/task-center/details/3e07081ee52b4043a030ec48d3f0ec34?menu=guide)提供的 `KaiserWindow_task_doc(A2A3A5).md`。
章节结构采用 cann-ops-competitions（原 cann-competitions）官方
`04_tasks/01_community-task-2026/resources/design_template.md`，模板参考提交
`01f0c677c8a2a3f33d228a6696541f5f1d16d4bc`。

## 背景介绍

在同一 SiP 仓新增 Ascend C KaiserWindow 算子和 `torch_sip.asd_kaiser_window`
工厂接口，为 A2/A3 的 arch22 与 A5 的 arch35 提供独立设备实现。
输出为当前或显式指定 NPU 上新建的一维连续张量。

# 需求分析（required）

## 需求描述

完整支持 window_length、periodic、beta、dtype、layout、device 和 requires_grad。
dtype 仅支持 float16、bfloat16、float32，缺省 float32；layout 仅支持 strided。
零长度返回空张量，一长度返回 1。负 beta 按绝对值计算，长度至少为 2 时
NaN 或正负无穷 beta 返回全 NaN。非法长度、dtype、device 不静默回退。

## 需求拆解

1. SiP C 接口提供输出描述符、stream、workspace，并将 workspace 大小恒定为零。
2. PTA 用 NPUGuard、at::empty 和 EXEC_FUNC 对接；注册 PrivateUse1 及工厂调用路径。
3. A2/A3 和 A5 分别满足原始任务精度与性能判据，不更改用例或标杆。
4. 运行期仅分配输出 GM，临时计算驻留 UB 或寄存器；多线程、多流输出独立。

# 详细设计（required）

## 算子分析

### 数学公式

对 L≥2，N 在 periodic=True 时为 L，否则为 L−1。
`out[i] = I0(abs(beta) * sqrt(abs(1 - (2*i/N - 1)^2))) / I0(abs(beta))`。
只写入 i∈[0,L)；L=0/1 使用独立短路规则。

### 支持数据类型与形状

输出为 `[window_length]`，支持 FLOAT16、BFLOAT16、FLOAT32。计算域为 FP32，
输出按目标类型舍入；任务长度覆盖 0、短窗口至 4194304，host 长度上限为 2^31−1。
没有张量输入或广播需求。

## 算子实现

The operator is a factory: it has no tensor inputs and writes a single new,
contiguous NPU tensor. `asdKaiserWindowGetWorkspaceSize` always returns zero.
The length-zero path returns before constructing an operation or launching a
kernel. Length one and beta zero use a constant-one kernel mode; nonfinite beta
with length at least two uses a constant-NaN mode.

### host 与 PTA 集成

`core/base/kaiser_window.cpp` validates the output descriptor, length, supported
dtype, one-dimensional shape and stride. `KaiserWindowOperation` selects the
arch22 kernel on ASCEND_910B (A2/A3), or the independent arch35 kernel on
ASCEND_950 (A5). The same tiling data format is used on both architectures.

`sip_pta/csrc/base/kaiser_window.cpp` caches the current stream and device,
guards only an explicitly requested different device, allocates the output with
`at::empty`, queries workspace and submits on the target device's current stream.
The optional guard restores the caller's device after submission. The default
false gradient flag is already supplied by `at::empty`; only a true flag calls
`set_requires_grad`, avoiding redundant AutogradMeta creation.
It registers both PrivateUse1 and CompositeExplicitAutograd; the latter is
required because the dispatcher cannot infer a backend from tensor inputs.
`EXEC_FUNC` converts and frees an aclTensor descriptor without retaining output
storage in the helper's thread-local tensor holder. KaiserWindow opts into
`SIP_PTA_USE_OPAPI_SUBMISSION` before including the shared helper: the same
`EXEC_FUNC` conversion, captured arguments, descriptor release and error check
are retained, while submission uses torch_npu's `OpCommand::RunOpApiV2`.
Other translation units keep the existing `OpCommand` instance path. This is
an internal change to the shared wrapper, not an unchanged upstream wrapper;
the opt-in and both branches are included in the source patch for review.
The Python wrapper exposes
the task signature and explicitly sets the requested leaf gradient flag.
It omits identical default arguments on the current-NPU path to reduce Python
dispatcher conversion overhead; explicit other devices and layouts still use
the registered operator's normal validation.
For the pinned PyTorch 2.7.1, the wrapper binds the registered overload's
underlying callable once, with a public-overload fallback. The SiP V2 path
retains one initialized KaiserWindow kernel per submitting thread, keyed by
ACL context, stream and all operator parameters. Every call still infers the
output descriptor and supplies fresh launch/run objects. The cache contains
only host execution metadata and never retains output storage or addresses.

### kernel 数值方案

Let N = L for a periodic window and N = L-1 for a symmetric window. Each tile's
first numerator is evaluated as `int32_t(2LL * offset - N)`, then an integer
progression with step two generates exact `2*i-N`. Only afterwards is it
converted to float32 and multiplied by the float32 reciprocal of N.

The kernel computes `t = sqrt(abs(1-r*r))` and the scaled expression
`i0e(abs(beta)*t) * exp(abs(beta)*(t-1)) / i0e(abs(beta))`. This avoids the large
unscaled I0 intermediate near beta=88. The reciprocal denominator is a single
host scalar in the tiling data; no denominator tensor is allocated.

The I0e approximation uses PyTorch/Cephes piecewise Chebyshev coefficients:
`y=x/2-2` on [0,8] and `y=32/x-2` above 8, the latter divided by sqrt(x).
The retained polynomials use 12 small-argument coefficients and five
large-argument coefficients. The sums of omitted coefficient magnitudes bound
their respective absolute I0e errors by 2.3e-5 and 2.7e-8 on the approximation
intervals. Output error also depends on denominator normalization and is checked
against the task's mixed tolerances. Mixed tiles clamp both branches to their
own interval, so an unused branch cannot overflow at x=0 or x=88.
On arch22, abs(beta) <= 5 instead uses a degree-nine Horner series in x*x/4
for I0(x), multiplied by exp(-x) to retain the same scaled-I0e formulation.
The truncation tail is below 2.7e-7 relative on [0,5]; FP32 rounding and the
shared denominator approximation are additionally checked against task tolerances.
This shortens the small-beta vector dependency chain without adding GM storage.
`check_fp32_model.py` reads the actual C++ coefficients
and models separate FP32 operations against float64 PyTorch golden, including
output narrowing (`--arch 22` or `--arch 35`). This check is supplemental and
is never labeled NPU testing.

### 分核、分块与内存方案

A2/A3 uses Ascend C vector instructions and explicit vector dependency
barriers. The arch22 CompareScalar count is rounded up to a full 64-float
repeat within allocated UB; selection and output DMA use the real tail length.
Chebyshev state is rotated by tensor handles, avoiding state-copy vector
instructions. Host-computed guarded boundaries allow wholly small-argument or
large-argument tiles to evaluate one polynomial only. For large windows with
beta > 8, cyclic tile assignment spreads expensive endpoint work across cores.
A5 uses CANN 9.2's `AscendC::Reg` MicroAPI inside a SIMD VF. The A5
Chebyshev recurrence is expanded through template recursion, keeping all
coefficients as compile-time scalar constants and all intermediate vectors in
registers. A5 conversion uses round-to-nearest-even and packed 16-bit stores.

Tiles hold at most 4096 output elements on A2/A3 and 1024 on A5. The small-branch
result shares the final result buffer; the float32 path uses 180 KiB of UB.
Blocks cover
disjoint ranges, with 256-element aligned contiguous partitions or disjoint
cyclic tiles on A2/A3. The final partial tile uses byte-counted
`DataCopyPad`, preventing writes past the supplied tensor. VECOUT queues order
computation, DMA submission and output-buffer reuse. A2/A3 also explicitly
completes vector conversion and output DMA before subsequent buffer reuse;
this is required for correctness when successive tiles take different
polynomial branches. Temporary storage is UB
only on arch22 and register/UB only on arch35; neither path creates intermediate
GM tensors. The caller owns the output and stream lifetime.

## 支持硬件

| 芯片与任务档位 | 设备实现 |
| --- | --- |
| Ascend910B，A2/A3 | arch22 Ascend C AIV |
| Ascend950，A5；实测设备 Ascend950PR | arch35 RegBase/MicroAPI |

## 算子约束限制

不支持 FLOAT64、整型、BOOL、复数及非 NPU 输出。仅支持一维连续输出。
任务精度域为 abs(beta)≤88，非有限值按专门语义检查；不据此承诺超出任务范围
的有限大 beta 精度。调用方负责输出与 stream 生命周期。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 原始1000条用例，FP64 CPU真值收窄到输出dtype后执行混合容差检查 | 任务书§3.2及verify_accuracy.py |
| 性能 | 原始200条，预热10次、采集60次取均值；A2/A3倍率≥0.8，A5≥0.4 | 任务书§3.3及verify_performance.py |
| 内存 | workspace=0、无中间GM、输出独立，分配器对齐用同尺寸torch.empty对照 | 任务书§3.4及补充contract检查 |

### 验证边界与复现条件

Task fixtures under `test_cases/` are preserved unchanged. NPU precision uses
the task's float64 CPU golden narrowed to the actual output dtype; performance
uses its original events, warmups, repetitions and supplied GPU baselines.
Each physical hardware tier requires its own test results. Cross-compiling
arch35 on A3 establishes compiler compatibility only, not A5 accuracy or speed.
Recorded performance runs use `PYTHONMALLOC=malloc`, `TASK_QUEUE_ENABLE=2`,
`OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`; these settings must be reproduced.
On the available Ascend950PR device, original torch_npu 2.7.1 fails during
native `torch.empty` initialization with an unsupported-SoC error. A5 validation therefore uses official torch_npu 2.7.1.post11. This is an
explicit task-version deviation requiring reviewer confirmation; these results
do not establish compatibility of original 2.7.1 with this A5 device.

`check_npu_contract.py` additionally verifies the zero workspace C API, raw
factory dispatch, leaf gradient accumulation, default dtype, two devices when
available, nondefault streams, explicit allocation on another device with caller
device/stream restoration, output independence, determinism and allocated output
memory after cache warmup.
`check_plan_reuse.py` additionally checks four submitting threads with two
streams each, changes parameters and dtype between repeated calls, keeps 192
independent outputs live, and compares each output to both CPU golden and a
sequential NPU invocation. This is supplemental to the unchanged task fixtures.

## 兼容性分析

新增工厂算子保留现有其他算子的 EXEC_FUNC 提交默认分支。仅当前翻译单元
启用提交优化；不修改其他算子的转换、错误处理或队列行为。A2/A3 的软件
版本按任务书固定；A5 post11 差异、原版失败日志和实际加载库版本随报告交付。


## 已完成的开发自验与复现条件

本节记录正式构建安装后的开发自验，尚未经任务方验收。
三平台使用 CANN 9.2.0、torch 2.7.1+cpu，经 SiP 与 sip_pta 同源构建，
执行 sip_pta/build.sh --install 后从 site-packages 加载扩展。

| 平台 | 实测设备 | torch_npu | 精度 | 性能三轮 | 各轮最低倍率 |
| --- | --- | --- | --- | --- | --- |
| A2 | Ascend910B3 | 2.7.1 | 1000/1000 | 200/200、200/200、200/200 | 0.8224、0.8301、0.8208 |
| A3 | Ascend910_9382 | 2.7.1 | 1000/1000 | 200/200、200/200、200/200 | 0.9802、0.9777、0.9727 |
| A5 | Ascend950PR_9579 | 2.7.1.post11 | 1000/1000 | 200/200、200/200、200/200 | 0.4954、0.4951、0.4958 |

三平台 API 各23项通过，零workspace、输出分配/释放后无输出残留、可用设备上的流/设备契约检查与192个独立输出并发检查均通过。
性能使用任务原始200项、预热10次、采集60次取均值；没有修改脚本、基线或阈值。

上述结果均使用 PYTHONMALLOC=malloc、TASK_QUEUE_ENABLE=2、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2。
A2 进程固定 CPU 192–199，A3 固定 CPU 0–7，A5 未固定 CPU。
A2/A3 在未固定CPU配置下曾出现个别性能项失败，原始轮次保留；上表结论仅适用于明确记录的配置。
不以固定核后的通过结果证明此前尖峰来自网络或证明主机调度是唯一原因。

供评审确认的实现选择包括：仅 KaiserWindow 翻译单元启用的 EXEC_FUNC 内部 RunOpApiV2 提交分支；
由 EXEC_FUNC 的 ConvertType 创建并释放描述符，未直接调用任务书提及的 CreateAclTensorFromAtTensor，
以避免其线程局部holder保留输出存储。这些选择和 A5 软件版本差异均随文档公开，不声称任务方已批准。
