# aclblasIcamin 算子设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | aclblasIcamin 算子开发（Atlas A2/A3） |
| 任务编号 | 任务书未给出 IT 编号。已有 icamin 目录均为 950 任务，本设计落入新建的 A2/A3 目录 `11-aclblasIcamin-A2A3` |
| TeamName | ATEAM |
| 设计文档路径 | `04_tasks/01_community-task-2026/tasklist/11-aclblasIcamin-A2A3/ATEAM/docs/design.md` |
| 目标代码仓 | `https://gitcode.com/cann/ops-blas` |
| 实现目录 | 拟新建 `blas/iamin/arch22/`；测试拟新建 `test/iamin/icamin/arch22/` |
| 目标硬件 / arch | Atlas A2 / Atlas A3 系列，`arch22`，`dav-2201`（`ascend910b*`、`ascend910_93*`）。性能验收覆盖任务书规定的 A2 款型 |
| 对标接口 | cuBLAS `cublasIcamin`。Netlib 无 icamin，golden 用测试工程内 cblas 风格 CPU 循环 |
| 基线 | ops-blas `master` 上已合入的 arch35 `aclblasIcamin`。下文接口、Host 校验和 arch35 行为以该实现为准 |
| CANN | 任务书规定 9.1.0 |

# 需求背景（required）

## 需求来源

- 任务书：`aclblasIcamin_AtlasA2_A3_task_doc.md`。
- 目标仓 ops-blas，目标硬件 Atlas A2/A3 系列（arch22）。性能验收覆盖任务书规定的 A2 款型。自验证覆盖一种款型。
- 对标接口 cuBLAS `cublasIcamin`。
- 依据优先级：任务书 > `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h` > 仓内 arch35 实现与 `blas/iamin/README.md` > cuBLAS 公开文档。
- 符号：任务书写声明在 `include/cann_ops_blas.h` 第 241 行。基线里该声明在第 244 行，签名已是下面这一条，与任务书逐参数一致，并与 950 共用。本任务沿用该 ABI，不新增符号。

```cpp
aclblasStatus_t aclblasIcamin(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result);
```

任务书要求对齐 aclblasIsamin 的 README 约束与 arch22 实现。当前 `blas/iamin/` 只有 `arch35/`。Host 语义对齐 `blas/iamin/README.md` 以及 arch35 `ValidateIcaminParams`（`blas/iamin/arch35/icamin_host.cpp`）。

## 背景介绍

`aclblasIcamin` 查找单精度复数向量中 1-范数模最小元素的 1-based 索引：

```text
result = argmin_i (|Re(x[k])| + |Im(x[k])|)
i = 1..n，k = 1+(i-1)*incx
```

模值相同取最小索引。`aclblasComplex` 在 `include/cann_ops_blas_common.h` 中是两个 `float`（`real`、`imag`）交错存放，n 个复数对应 2n 个 float。

基线已在 `blas/iamin/arch35/` 实现该接口，供 Ascend 950PR / 950DT。`blas/iamin/README.md` 将 Atlas A2、Atlas A3 标为不支持。根 `CMakeLists.txt` 把 `ascend910b*` 与 `ascend910_93*` 编进 `arch22`（`NPU_ARCH=dav-2201`），A2/A3 构建因此没有 `aclblasIcamin` 的定义。

arch35 kernel 的步长路径和 NaN 回退使用 `__simt_vf__`、`asc_vf_call`，tile 按 dav-3510 的 248KB UB 取值（`blas/common/arch/hardware.h` 中 `HardwareInfo<ASCEND_V350>::ubSize`）。同一头文件的缺省 `ubSize` 是 `192 * 1024`，arch22 算子按 192KB 预算。本任务在 arch22 上另写 kernel。Host 的错误码、quick return、NaN 与平局语义与 arch35 和任务书一致。

本任务不涉及 TBE、OPP、Python 或 ATen。

| 项 | 当前基线 | 本任务 |
| --- | --- | --- |
| 接口 | `cann_ops_blas.h` 已声明，定义只在 arch35 | 沿用声明，为 arch22 提供定义 |
| 硬件 | README：950 支持，A2/A3 不支持 | A2/A3 改为支持；950 行保持支持 |
| 功能 | arch35 已实现任务书语义 | arch22 拟实现同等语义，源码待新建 |
| 测试 | `test/iamin/icamin/CMakeLists.txt` 只在 arch35 注册 GTest | arch22 改为 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`，并新增 `test/iamin/icamin/arch22/` |

# 需求分析（required）

## 需求描述

在 Atlas A2/A3（arch22，dav-2201）上用 Ascend C 实现已声明的 `aclblasIcamin`：handle 绑定 stream，直调 NPU kernel。输入是 Device 上的 COMPLEX64 向量，逻辑长度 n、步长 incx，ND 交错存储；输出是 Device 上的一个 INT32 索引。支持 n ≥ 0 且 incx ≥ 1 的计算，以及 n = 0 或 incx < 1 的 quick return。本提交范围不含 arch35 kernel 改动、其他 dtype、负步长反向遍历、广播、动态 shape 推导、原地更新和 Python 入口。

## 需求拆解

1. 接口和 ABI：链接已有 `aclblasIcamin`。返回 `aclblasStatus_t`，码值取自 `cann_ops_blas_common.h`。
2. 功能、dtype、shape、format：COMPLEX64 入、INT32 标量出；逻辑 shape `[n]`，物理元素数 `1+(n-1)*incx`（n > 0 且 incx ≥ 1）；ND 交错 complex。
3. Host 校验和错误行为：顺序见「Host」。quick return 返回 `ACLBLAS_STATUS_SUCCESS`。
4. Kernel、Tiling、分核：连续路径按 AIV 核切分，每核一个 `(模值, 索引)` 槽，再归并。不使用 TilingKey，分支在 launch 函数内按 `incx == 1` 与 `useCoreNum` 选择。
5. 边界与 quick return：n = 0 或 incx < 1（含 0、负值、`INT32_MIN`）不启动 kernel，Device 上的 `result` 写 0。`(n-1)*incx` 用 int64 计算，超出 int32 可表示范围返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. 精度和性能验证：索引与 CPU golden bit-exact。性能用 msprof op 的 task duration，门限为 GPU 标杆的 1.25 倍。NPU 耗时待测。
7. 兼容与回归：arch35 源码保持不动。`SOC_ARCH_DIRS` 使 arch22 与 arch35 不会进入同一次链接。
8. 上层入口：验收入口就是 `aclblasIcamin`。不涉及 aclnn、TBE 或 PyTorch。

# 详细设计（required）

## 算子分析

### 语义

n > 0 且 incx ≥ 1 时：

```text
abs1(z) = |Re(z)| + |Im(z)|
best = 0
for i = 1 .. n-1:          # 0-based
    if abs1(x[i * incx]) < abs1(x[best * incx]):
        best = i
result = best + 1          # 1-based
```

比较是严格小于，平局保持先出现的索引。分块顺序不改变最终索引，验收用整数 `==`。

NaN 与任何值的 `<` 都为假，所以 NaN 模不会替换已有最小值。首元素实部或虚部为 NaN 时结果为 1，全 NaN 也是 1。这与 `test/iamin/icamin/icamin_golden.h` 的 `aclblasIcamin_cpu` 一致：先取 `abs1(x[0])`，只在 `absVal < bestVal` 时更新。

`|Inf|` 与有限数相加仍是 Inf，不会小于有限模。两个 `+Inf` 不满足严格小于，保持较小索引。

### 存储、workspace、异步

| 参数 | 内存 | dtype | 布局 | 说明 |
| --- | --- | --- | --- | --- |
| handle | Host | `aclblasHandle_t` | 标量 | 携带 stream。空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n | Host | int | 标量 | 复数个数 |
| x | Device，只读 | COMPLEX64 | ND，`{real, imag}` 交错 | n > 0 且 incx ≥ 1 时不可为空 |
| incx | Host | int | 标量 | 以复数为单位的步长 |
| result | Device | INT32 | 单值 | 1-based 索引，或 quick return 的 0 |

多核路径的 workspace 存放每核一对 `(float 模值, uint32 索引)`。`aclblasCreate` 分配库属 workspace，大小 `ACLBLAS_DEFAULT_WORKSPACE_SIZE`（32 MiB，`blas/common/helper/aclblas_handle_internal.h`）。拟从 `GetEffectiveWorkspace` 取用，长度不足返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。算子内不 `aclrtMalloc`。`useCoreNum == 1` 时直接写 `result`，不用这块槽。调用方若用 `aclblasSetWorkspace` 换成更小缓冲，长度须覆盖本次槽位。

kernel 与 quick return 的 `aclrtMemsetAsync` 都挂在 `handle->stream` 上。接口返回前不同步。调用方读回 `result` 前自行同步。

## 算子实现

`blas/iamin/arch22/` 尚不存在。Host 契约按已合入的 arch35 `aclblasIcamin` 写。Kernel 是拟实现，UB 上的 tile 元素个数留到实现时按 192KB 预算计算，不把 arch35 的 `ICAMIN_TILE_COMPLEX = 4096` 写成 arch22 参数。

### 总体架构

```text
aclblasIcamin
  └─ Host：校验 → quick return，或计算 tiling 并提交 kernel
       ├─ incx == 1 且 useCoreNum == 1：per-core kernel 直接写 result
       ├─ incx == 1 且 useCoreNum > 1：per-core kernel 写 workspace，再单核归并
       └─ incx > 1：按步长扫描，再按同样规则归并
```

不使用 TilingKey。launch 函数里按 `incx == 1` 和 `useCoreNum` 分支，层次与 arch35 `icamin_kernel_do` 相同，入口换成 arch22 kernel。

### Host

拟新建 `blas/iamin/arch22/icamin_host.cpp`。校验顺序与 arch35 `ValidateIcaminParams` 相同：

| 顺序 | 条件 | 行为 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `result == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE`（含随后的 quick return 条件） |
| 4 | `n == 0` 或 `incx < 1` | 不启动 kernel。`aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream)`。失败返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，成功返回 `ACLBLAS_STATUS_SUCCESS` |
| 5 | `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `last = int64(n - 1) * int64(incx)` 小于 0，或大于 `INT32_MAX`，或 `last > INT64_MAX / 2` | `ACLBLAS_STATUS_INVALID_VALUE` |

第 6 步与 arch35 相同：后面的复数寻址还要乘 2，先用 int64 挡住 int32 回绕。通过后：

- `aivCoreNum = GetAivCoreCount()`，为 0 时返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。
- `numBlocks = min(n, aivCoreNum)`。
- incx == 1 时，每核切片按 4 个复数（32 字节）向上对齐，使整段 GM 地址和长度满足 DataCopy。对齐只会减少实际核数。
- incx > 1 时按逻辑元素均分，余数在最后一核。
- tiling 按值传入 kernel，字段为 `totalN`、`perCoreN`、`lastCoreN`、`useCoreNum`、`tileSize`、`incx`。不含 arch35 的 `nthreads`。
- 多核路径的 workspace 需求是每核 2 个 float，整体向上对齐到 32 字节。不足返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
- 提交 kernel 后返回 `ACLBLAS_STATUS_SUCCESS`，不调用 `aclrtSynchronizeStream`。

同仓 arch22 的 `aclblasIsamax`（`blas/iamax/arch22/isamax_host.cpp`）在接口内同步 stream，并当场 `aclrtMalloc` workspace，空 handle 返回 `ACLBLAS_STATUS_NOT_INITIALIZED`，quick return 写 Host 指针。本算子的 Host 按上一张表实现。

### Kernel

拟新建 `blas/iamin/arch22/icamin_kernel.cpp`，类分为 Init 与 Process。SIMT 接口不进入该文件。

连续路径（incx == 1）对每个 UB tile 做 CopyIn、Compute、发布，多核再加一次归并：

1. CopyIn。`DataCopy` 搬入交错的 float。末块不是 32 字节整数倍时，`DataCopyPad` 补齐最后一块，更长的对齐尾巴用 `Duplicate` 写成 `+Inf` 的复数对（实部、虚部都是 `+Inf`），避免脏数据赢得最小值。填充按复数对进行，保证后面的相邻求和仍然一项对应一个复数。
2. Compute。`Abs` 作用在交错 float 上，再 `PairReduceSum` 把相邻的 `|Re|`、`|Im|` 收成 `abs1`。一项 `abs1` 对应一个复数，其后归约得到的索引就是复数下标，加上本核 `startOffset` 后是全局 0-based 索引。这两步是 arch22 `blas/iamax/arch22/isamax_kernel_impl.h` 里复数分支已经使用的 API；本算子用它们求最小值，不采用该文件的 repeat 切分常量。
3. NaN。`Abs` 不改变 NaN。`Compare(v, v, CMPMODE::EQ)` 只在 NaN 上为假，arch22 `blas/gbmv/arch22/cgbmv_kernel.cpp` 已用这个比较识别 NaN。拟把 NaN 模改成 `+Inf` 再归约，避免 `ReduceMin` 对 NaN 的结果差异。负责全局元素 0 的核在发布前再读 `x[0]`：实部或虚部为 NaN 时，该核发布 `(NaN, 索引 0)`，归并后 1-based 结果为 1。
4. 索引。拟调用与 arch22 `WholeReduceMax<T, false>` 对称的 `WholeReduceMin`，再用 `GetReduceMaxMinCount` 读出值和索引（isamax 在 `WholeReduceMax` 之后这样取值）。仓内已有的 `WholeReduceMin(..., ORDER_ONLY_VALUE)` 和 `ReduceMin(..., false)` 只产值，不作为本算子的取值方式。平局必须是最小索引。实现时用全相等向量核对 `GetReduceMaxMinCount` 的索引；若它不是最小下标，在该 UB tile 内从起点找到第一个等于该最小值的位置。
5. CopyOut。`useCoreNum > 1` 时把本核 `(模值, 索引位型)` 写入 workspace。归并 kernel 把各槽读入 UB，按严格小于和最小索引得到 0-based 结果，加 1 写入 `result`。`useCoreNum == 1` 时跳过 workspace，直接写 `result`。`Duplicate` 在 V 管，写出在 MTE3，发布前做管道屏障。

队列深度取 1。192KB UB 要同时放下输入槽、`abs1`、NaN mask 和归约临时区。代码未开双缓冲，本文不写搬运与计算重叠。`tileSize` 取实现时能放进该预算、且为 4 个复数整数倍的值。

步长路径（incx > 1）不做三段式。逻辑元素间隔 `incx` 个复数，连续 `DataCopy` 不能表示这条向量。拟按逻辑下标标量读 GM，核内维护 `(bestVal, bestIdx)`，再写入本核槽位。负 incx 不会进入 kernel。任务书的性能用例全部是 incx = 1，步长路径只承担功能正确性。

kernel 内不重复 Host 已做的空指针、负 n、步长和溢出检查。

### 目录与构建

```text
ops-blas/
├── include/cann_ops_blas.h                 # 已有声明，不改签名
├── blas/iamin/README.md                    # A2/A3 产品支持改为支持
├── blas/iamin/arch22/                      # 待新建
│   ├── icamin_host.cpp
│   ├── icamin_kernel.cpp
│   ├── icamin_kernel.h
│   └── icamin_tiling_data.h
├── blas/iamin/arch35/                      # 保持不动
└── test/iamin/icamin/
    ├── CMakeLists.txt                      # arch22 注册 GTest
    ├── icamin_golden.h                     # 已有，arch22 复用
    ├── icamin_param.h                      # 已有
    ├── icamin_npu_wrapper.h                # 已有，调用 aclblasIcamin
    ├── arch35/                             # 保持不动
    └── arch22/                             # 待新建
        ├── icamin_test.cpp
        └── icamin_test.csv
```

`blas/CMakeLists.txt` 按 `SOC_ARCH_DIRS` 收集 `*/arch22/*.cpp`。`ascend910b*`、`ascend910_93*` 只有 arch22，`ascend950*` 只有 arch35。iamin 根目录没有同名 `.cpp`，不存在根目录文件与 arch 文件争符号的问题。

`test/iamin/icamin/CMakeLists.txt` 现在只在 arch35 调用 `ops_blas_add_gtest_tests(${OPS_BLAS} icamin_test)`，其他 SOC 走 `ops_blas_add_tests`。拟让 arch22 同样注册该 GTest 目标，源文件放在 `test/iamin/icamin/arch22/`。

## 支持硬件

| 芯片 | 本任务 |
| --- | --- |
| Atlas A2 训练 / 推理系列 | 拟支持。性能验收覆盖任务书规定的 A2 款型 |
| Atlas A3 训练 / 推理系列 | 拟支持，与 A2 共用 arch22 源。自验证覆盖一种款型，另一种待测 |
| Ascend 950PR / 950DT | 已有 arch35。本提交不改该实现，也不计入 A2/A3 验收 |

## 算子约束限制

- 只支持 COMPLEX64 输入、INT32 索引输出。
- n < 0、`result` 为空、计算路径上 `x` 为空、`(n-1)*incx` 超出 int32：返回对应错误码。
- n = 0 或 incx < 1：成功，`result = 0`，不读 x。负步长不反向遍历。
- 不支持广播、动态 shape 推导、leading-dimension padding、原地更新。
- 调用方保证 handle 已创建；计算路径上 x、result 是当前 device 的有效指针；读回前同步已绑定的 stream。
- 多核路径使用 handle workspace。默认 32 MiB 足够。调用方自备 workspace 时须覆盖每核一对 float 再按 32 字节对齐的长度。
- 模值相同返回最小 1-based 索引。
- 本提交范围不含 arch35 kernel 修改、SIMT，以及其他复数精度。性能数字待测。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出是一个 INT32，通过条件为 `actual == golden`。任务书给出的 FLOAT32 档 `rtol = atol = 2^-13`、`required_matched_ratio = 0.99` 描述输入分量档位；索引比对是逐值相等，一条用例通过则 ratio = 1。quick return 期望 `result == 0`。错误路径期望对应状态码 | 任务书 §3.2；任务书所给 opbase mixed tolerance 文档。golden 为 `icamin_golden.h` 的严格小于循环 |
| 性能标准 | 设备为任务书规定的 Atlas A2 性能款型，COMPLEX64，incx = 1。msprof op 的 task duration（us）：工具自带 5 次 warmup 不计入；其后 ≥10 次返回 SUCCESS 的样本取算术平均。平均耗时 ≤ GPU 标杆的 1.25 倍。标杆是任务随附 `gpu_baseline.csv` 的 `gpu_ms`，共 200 行。前三条：n = 1048576，`0.0181589339`；n = 2097152，`0.0216234665`；n = 4194304，`0.0366080012`。其余行用同一公式。NPU 耗时待测 | 任务书 §3.3；任务随附 `gpu_baseline.csv` |

内存：任务书 §3.4 为不涉及。验收模板仍有内存报告，设计阶段不填占用数字。

任务书以 msprof task duration 为性能判定。随任务的 `test_cases/README.md` 以 GTest `avg_us` 为判定、以 msprof 为交叉验证。验收按任务书。自测报告可并列两列。arch35 测试注释中的 `gpu_ms * 1000 / 0.4` 属于 950，本任务系数为 1.25。

## 测试设计

用例文件 `test_cases/icamin_test.csv` 共 1200 行：精度 1000，`TC_PF` 200。列与 `test/iamin/icamin/icamin_param.h` 一致。当前文件的前缀计数：`TC_L0` 6、`TC_SQ` 38、`TC_INC` 17、`TC_FL` 14、`TC_ED` 17、`TC_EX` 908、`TC_PF` 200。`test_cases/README.md` 里 FL/ED/EX 的条数与该 CSV 不同，以 CSV 为准。

| 维度 | 当前 CSV | arch22 拟执行 |
| --- | --- | --- |
| 正向 | `TC_L0`、`TC_SQ`、正步长的 `TC_INC`、incx ≥ 1 的 `TC_EX`。填充含均匀、全零、交替、极端值、Inf、NaN | GTest 调用 `aclblasIcamin`，与 `aclblasIcamin_cpu` 做 `EXPECT_EQ` |
| 边界 | n = 0 有 7 行。incx < 1 时期望 SUCCESS 且索引 0，含 `INT32_MIN` 步长；该类 x 用 `NULLPTR` 占位 | 比对状态码和结果 0 |
| 负向 | `ACLBLAS_STATUS_INVALID_VALUE` 共 9 行，其中负 n 3 行、x 为 `NULLPTR` 6 行、result 为 `NULLPTR` 3 行，属性有重叠，所以三者之和大于 9 | 只比对状态码。不按 `(n-1)*\|incx\|` 分配大缓冲 |
| 一致性 | golden 与 NPU 都是严格小于 | bit-exact |
| 特殊值 | `VALUE_NORM_NAN` 24 行，`VALUE_NORM_INF` 24 行，description 含 nan 的 5 行。arch35 测试按 `mix_nan_first_` / `mix_nan_mid_` 改写元素 | arch22 保留该改写 |
| 回归 | arch35 源码及其 CSV 不改 | A2/A3 构建不链入 arch35 的 icamin；950 构建不链入 arch22 |
| 性能 | `TC_PF` 200 行全部 incx = 1，前三条 n 为 1048576、2097152、4194304 | msprof op，待测 |
| handle 空指针 | CSV 不表达。arch35 有独立 `NullHandle` | arch22 同样单列，期望 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |

正态：任务书要求一部分用例使用 `GAUSS_NORM_5_2`（μ ∈ [-5, 5]，σ ∈ [0.1, 2]）。`test/frame/fill.h` 已识别 `GAUSS`。当前 CSV 中 `GAUSS` 行数为 0；`gen_csv.py --dist mixed` 仍只生成 `RANDOM_NORM_5_5`。正态行待补充，补充后纳入精度比对，并在自测报告写明条数。

精度和性能都待在 A2 或 A3 上实测。

## 兼容性分析

- 新增：arch22 host/kernel 与 arch22 GTest。头文件符号已存在，不新增 ABI。
- 增量架构：增加 `arch22` 目录，并修改测试 CMake 的 arch22 分支。`SOC_ARCH_DIRS` 与 arch35 互斥。
- dtype / shape：仍是 COMPLEX64 到 INT32，n 为运行时标量。本提交不扩展 dtype，也不做广播。
- 并行贡献：与 950 共用 `include/cann_ops_blas.h` 的同一声明。本任务不修改 `blas/iamin/arch35/`。README 把 A2/A3 两行改为支持，950 行保持支持。
- 不涉及 TBE 插件、Python 包和旧私有 API。

## 风险与对策

| 风险 | 对策 |
| --- | --- |
| arch35 的 SIMT 路径不能在 dav-2201 上作为本算子实现 | arch22 使用 `Abs`、`PairReduceSum`、`WholeReduceMin`、`GetReduceMaxMinCount`。步长路径用标量扫描 |
| `GetReduceMaxMinCount` 在最小值归约后给出的索引不是最小下标 | 全相等用例核对。不符合时，在该 UB tile 内从起点找第一个等于最小值的位置 |
| NaN 改成 `+Inf` 后，首元素 NaN 被后面的有限值替换 | 拥有全局下标 0 的核在发布前检查 `x[0]`，NaN 时强制索引 0。用 `mix_nan_first_` 验收 |
| 写出与 `Duplicate` 之间没有屏障，`result` 读到旧 UB | 发布前做 V 到 MTE3 的屏障 |
| arch22 继续走非 GTest runner | `CMakeLists.txt` 的 arch22 分支改为 `ops_blas_add_gtest_tests` |
| 正态用例还没有生成 | 补充 `GAUSS_NORM_5_2`。补充前在自测报告标明缺口 |
| 性能门限尚未测量 | 先完成 bit-exact，再按 msprof 口径测量。超门限时再改 tile。本文不把双缓冲写成已有优化 |
| 调用方 workspace 小于每核槽位 | 返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。默认 32 MiB 足够 |
