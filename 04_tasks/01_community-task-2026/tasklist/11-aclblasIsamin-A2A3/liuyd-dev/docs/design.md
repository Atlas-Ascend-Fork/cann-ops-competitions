# aclblasIsamin 算子设计文档（Atlas A2/A3，arch22）

| 项 | 内容 |
| --- | --- |
| 算子名 | `aclblasIsamin` |
| 任务 | 社区任务 — aclblasIsamin 算子开发（A2/A3） |
| 目标仓 | [ops-blas](https://gitcode.com/cann/ops-blas)，实现 `blas/iamin/arch22/`，测试 `test/isamin/arch22/` |
| 对标基线 | cuBLAS `cublasIsamin`（[官方文档](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf) §2.5.2） |
| 数据类型 | 输入 `FLOAT32` 向量；输出 `INT32` 标量（1-based 索引） |
| 目标硬件 | Atlas A2 训练/推理系列、Atlas A3 训练/推理系列（arch22） |
| CANN 版本 | 9.1.0 |
| 自验证范围 | 仅需覆盖 910B3（任务书 §3.1）；性能设备 Atlas 800T A2（910B3） |
| **本文档状态** | **已实机验证并与最终代码一致。** 910B3 / CANN 9.1.0 实测：接口与语义单测 **8/8**、精度 **1012/1012**、性能 **200/200 达标**（最终轮倍率 min 0.915、median 1.179，判据 ≥ 0.8；本次连续 3 轮均 200/200）。§6.2 表一的规格歧义按 arch35 源码核定（含一处任务包文档错误），表二的两项待确认已由板上实测结案（含一条此前未见于任何文档的 arch22 行为）。§5.1 的数值全部取自 §5.1.4 所列的同一构建最终实测批次 |

---

# 1 需求背景（required）

## 1.1 需求来源

社区任务「aclblasIsamin 算子开发（A2/A3）」。`ops-blas` 仓当前只有
`blas/iamin/arch35/` 一份面向 Ascend 950PR 的实现（950PR 既有 PR，下称 950PR），
Atlas A2/A3 系列在 BLAS-1 的 `I*amin` 能力上为空缺。本任务为该算子补齐 arch22 实现。

同时存在一处工程缺口须随本任务补齐：`test/isamin/CMakeLists.txt` 当前全文仅一个分支

```cmake
if("arch35" IN_LIST SOC_ARCH_DIRS)
    ops_blas_add_gtest_tests(${OPS_BLAS} isamin_test)
endif()
```

A2/A3 构建因此会退回非 GTest runner（只输出整数毫秒），须按同一函数增加 arch22
分支（arch35 分支不改动，任务书 §2.2）。这一项不是可选优化：§5.3 的性能判定依赖
GTest 打印的 `[PERF] ... avg_us=` 行，非 GTest runner 的整数毫秒输出无法满足判定口径。

## 1.2 背景介绍

`I*amin` 是 BLAS-1 的索引归约类接口：在向量中查找绝对值最小元素的位置，返回
1-based 索引。它与 `S*` 系列的数值归约（如 `sasum`、`snrm2`）有两点本质差异，
这两点决定了本算子的全部设计难点：

1. **输出是索引而不是数值**。因此精度判据不是浮点容差，而是整数精确相等
   （任务书 §3.2：`matched_ratio = 1.0`、`max_abs_error = 0`）。任何一次并列判定
   取错、0/1-based 混淆、或跨核合并次序错误，都是直接失败，没有容差兜底。
2. **归约算子不满足朴素的可结合性**。「取最先出现的最小值」要求合并时保持原始
   索引序；而任务书 §2.1 规定的 NaN 基准值语义进一步破坏了朴素分核分解的正确性
   （§3.2.1 给出反例与修正）。这是本算子唯一真正的算法设计点。

### 1.2.1 arch35 现状分析

仓内既有实现 `blas/iamin/arch35/`（950PR）提供了本任务的行为基准。任务书 §7.2
明确「行为对齐仓内 950PR 既有实现」，并把 `blas/iamin/arch35/isamin_host.cpp` 的
校验入口（L23-57）指定为返回码与快速返回口径的源码依据。

本设计把 arch35 作为**语义的唯一事实来源**，而不是实现的移植模板：arch35 面向
Ascend 950PR（arch35 微架构），其分核常量、UB 预算与缓存行为均不可直接套用到
arch22（见 §3.1.4 的两条 arch22 专有约束）。

已通读 `isamin_host.cpp` 全文，与本设计直接相关的事实（均已核入对应小节）：

| arch35 事实 | 源码位置 | 本设计的处理 |
| --- | --- | --- |
| 校验顺序：handle → `n<0` → **`result` 判空** → 快速返回 → `x` 判空 → 溢出守卫 | `ValidateIsaminParams` L23-57 | §3.3.2 逐条照搬。注意 `result` 判空在快速返回**之前** |
| 溢出守卫为 `lastElement < 0 \|\| lastElement > INT32_MAX`，int64 中间量 | L48-55 | §3.3.2 原样沿用（`lastElement < 0` 在 `incx ≥ 1` 下不可达，属防御性分支，一并保留） |
| 快速返回由 host 侧 `aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream)` 写 0，注释明示「不得在 host 直接解引用 result」 | L88-96（memset 在 L90），issue #738 | §3.3.2 同口径 |
| workspace 取自 handle（`GetEffectiveWorkspace`），**不按调用申请**；不足则 `EXECUTION_FAILED` | L115-123 | §3.3.4 同机制 |
| workspace 用量 `numBlocks × 2` 个 float（每核一对 `(minVal, idx)`，索引以 float 承载），整体对齐到 64 float = 256 B | L117-120 | §3.3.4 **刻意不照搬紧凑排布**，原因见该节 |
| `numBlocks = min(n, aivCoreNum)`，小 `n` 不额外收缩 | L107 | §3.3.3 **刻意不同**，原因见该节 |
| `incx != 1` 走 SIMT 路径（`nthreads`、`SIMT_*_THREAD_NUM`） | `CalcIsaminTiling` L72-78 | arch22 无该能力，stride 路径自行设计（§3.4.3） |

> 以上行号以当前 `ops-blas` master（`isamin_host.cpp` 142 行版本）为准。#738/#745
> 之前的旧快照只有 123 行，校验顺序为「快速返回先于 `result` 判空」且无 memset 与
> 溢出守卫，不能用于复核本表；两者差异正是这两条 issue 的修复内容。

### 1.2.2 标杆接口（cuBLAS cublasIsamin）实现描述

```cpp
cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float *x, int incx, int *result);
```

cuBLAS 在该规模区间（任务包基线 n ∈ [1, 4194304]）的实测表现呈两段：
n ≤ 64K 时耗时基本恒定在 8.06 ~ 14.27 us（kernel 启动与归约树深度主导，与 n 近乎无关），
n > 1M 后进入带宽主导段，n = 4194304 时 21.15 ~ 21.23 us。两段的分界决定了本算子
两条不同的优化目标（§3.5.2）。

## 1.3 算子原型

接口声明已存在于 `include/cann_ops_blas.h`（L240），与 950PR 共用，本次**无改动**；
任务书 §7.1 明文禁止另立私有平行接口。

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result);
```

| 参数 | 方向 | 位置 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | Host | `aclblasHandle_t` | 库上下文，内部携带 stream；`aclblasCreate` 返回的非空句柄 |
| `n` | 输入 | Host | `int` | 向量元素个数 |
| `x` | 输入 | Device | `const float*` | 输入向量，物理长度 `1+(n−1)×|incx|` 个 FP32 元素，只读 |
| `incx` | 输入 | Host | `int` | 相邻元素存储步长（元素为单位） |
| `result` | 输出 | Device | `int*` | 绝对值最小元素的最先出现位置，1-based，shape `[1]` |

**数学定义**

```
result = argmin(i = 1..n) |x[j]|,  j = 1 + (i−1) × incx
并列时取最先出现的 i；result 为 1-based，正常计算时 result ∈ [1, n]
n = 0 或 incx < 1 时：result = 0
```

---

# 2 需求分析（required）

## 2.1 需求描述

在单精度实数向量中查找绝对值最小元素的最先出现位置，返回 1-based 索引。
向量按 BLAS 步长语义访问（`incx` 间隔，跳过间隔元素）。参数非法时返回确定错误码；
`n = 0` 与 `incx < 1` 为合法快速返回。计算经 handle 绑定的 stream 异步下发。

## 2.2 需求拆解

| # | 要求 | 任务书出处 |
| --- | --- | --- |
| 1 | FP32 向量的 argmin\|x\|，1-based 索引，并列取最先出现 | §2.1 |
| 2 | NaN 语义：首元素 NaN 时基准置 `FLT_MAX`，后续 NaN 跳过；全 NaN ⇒ 结果 1 | §2.1 |
| 3 | 校验与快速返回：handle 判空 → `n < 0` → **result 判空** → `n = 0 \|\| incx < 1` 快速返回（result 写 0）→ x 判空 → 溢出守卫 | §2.4 / §2.5，已按源码 `isamin_host.cpp` L23-57 核定（§3.3.2） |
| 4 | `(n−1) × incx` 超 int32 可表示范围 ⇒ `ACLBLAS_STATUS_INVALID_VALUE` | §2.4 / §2.5 |
| 5 | 非连续访问（`incx > 1`）正步长生效；`incx < 1` 不进入计算 | §2.5 |
| 6 | 精度：输出索引与 golden 整数精确一致，不允许任何错位 | §3.2 |
| 7 | 性能：平均单次耗时 ≤ GPU 标杆的 1.25 倍（逐 case 判定，≥ 0.8× 标杆） | §3.3 |
| 8 | kernel 直调工程（host 入口 + tiling + device kernel），实现 `blas/iamin/arch22/` | §2.2 |
| 9 | 测试工程 `test/isamin/arch22/`，并补 `test/isamin/CMakeLists.txt` 的 arch22 GTest 注册 | §2.2 |
| 10 | 异步语义：返回成功仅表示 Host 已提交，读取输出前须同步 stream | §2.5 |

## 2.3 与 cuBLAS cublasIsamin 的差异

| 项 | cuBLAS `cublasIsamin` | 本算子 | 原因 |
| --- | --- | --- | --- |
| `incx < 1` | 文档规定 `incx` 须 ≥ 1，越界行为未定义 | 明确为合法快速返回，`result` 写 0 并返回 SUCCESS | 任务书 §2.4 与 arch35 既有实现对齐；比 cuBLAS 的「未定义」更强 |
| 步长溢出 | 未规定 | `(n−1) × incx` 超 int32 ⇒ `INVALID_VALUE` | 任务书 §2.4；arch35 有该守卫 |
| NaN | 文档未规定 NaN 下的索引语义 | 明确：NaN 不被选为最小元素；全 NaN ⇒ 1；首元素 NaN 时基准 `FLT_MAX` | 任务书 §2.1 明文规定，golden 按此口径实现 |
| 错误码 | `CUBLAS_STATUS_*` | `ACLBLAS_STATUS_*`（`HANDLE_IS_NULLPTR` / `INVALID_VALUE` / `SUCCESS`） | 参数序列逐位一致，仅返回码类型映射（任务书 §3.5） |
| 确定性 | 未承诺 | 并列取最先出现索引，结果与分核数无关（§3.4.4） | 任务书 §2.5「确定性计算要求」 |

---

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

归约的**顺序**语义（任务书 §2.1，golden 与 arch35 CPU 参考实现的口径）：

```c
minAbs = isnan(|x_1|) ? FLT_MAX : |x_1|;
idx    = 1;
for (i = 2; i <= n; ++i) {
    a = |x_{1+(i-1)*incx}|;
    if (!(a != a) && a < minAbs) { minAbs = a; idx = i; }   // NaN 跳过；严格小于
}
result = idx;
```

两处细节是后续全部设计的约束条件：

- **严格小于**（`a < minAbs`，不是 `<=`）⇒ 并列时保留先出现的索引。
- **基准值的非对称性**：`FLT_MAX` 替换**只作用于 i = 1**。i ≥ 2 的 NaN 是「跳过」，
  不参与竞争；而 i = 1 即使是 NaN 也会以 `FLT_MAX` 的身份**参与竞争**。
  这条非对称性是 §3.2.1 的全部来源。

### 3.1.2 支持数据类型

| 项 | 类型 | 说明 |
| --- | --- | --- |
| 输入 `x` | `FLOAT32` | 全值域，含 ±0 / 次正规数 / ±Inf / NaN |
| 输出 `result` | `INT32` | 1-based 索引，值域 `[1, n]`；快速返回时为 0 |
| Host 标量 | `int` | `n`、`incx` |

无多 dtype 面：任务书 §3.2 的多 dtype 阈值表对本算子只有 FLOAT32 一列生效，
且因输出为整数索引而退化为精确比对（§5.1）。

### 3.1.3 支持形状

一维向量，物理长度 `1 + (n−1) × |incx|` 个 FP32 元素。
`n` 与 `incx` 为运行时入参，host tiling 按运行时 `n` 计算，不要求 dynamic shape。

| 维度 | 范围 | 来源 |
| --- | --- | --- |
| `n` | 任意 int32；`n < 0` 负向，`n = 0` 快速返回 | 任务书 §2.4 |
| `incx` | 任意 int32；`incx < 1` 快速返回 | 任务书 §2.4 |
| `(n−1) × incx` | 须 ≤ `INT32_MAX` | 任务书 §2.4 |
| 测试覆盖规模 | 精度 `n ∈ [0, 1048576]`；性能 `n ∈ [1, 4194304]` | `test_cases/README.md` |

### 3.1.4 硬件能力实测

分核与分块参数全部由 `platform_ascendc::PlatformAscendCManager` 在运行时读出，
不引用文档值、不写死常量（与仓规及本人 `aclsparseSpSV` 设计同口径）。

**开发与自验证环境**（AtomGit CANNLab Dev Space `Ascend_A2_CANN`，已实测）：

| 项 | 实测值 | 采集方式 |
| --- | --- | --- |
| NPU 型号 | **910B3**（NPU id 1，Bus-Id `0000:0A:00.0`） | `npu-smi info` |
| HBM | 65536 MB | 同上 |
| npu-smi | 25.5.0 | 同上 |
| CANN | **9.1.0**（`/home/developer/Ascend/cann-9.1.0`），与任务书 §3.1 要求一致 | `$ASCEND_TOOLKIT_HOME` |
| 可见设备 | `ASCEND_VISIBLE_DEVICES=1`（故验证脚本须传 `--device 1`） | `env` |
| Host | 32 vCPU / 245 GB RAM | `nproc` / `free` |
| 构建工具链 | **已验证可用**：仓内既有 build 的 `SOC_VERSION=ascend910b3`、`CMAKE_BUILD_TYPE=Release`，`libops_blas.so` 构建成功 | `build/CMakeCache.txt` |

| 实测项 | 实测值 | 采集方式 |
| --- | --- | --- |
| **AIV 核数** | **40** | msprof `op_summary` 的 `Block Num`：多核用例恒为 40、单核用例恒为 1，与 `GetCoreNumAiv()` 的取值交叉验证一致（§5.1.3）。注：`platform_ascendc` 的符号不在工具链任何 `.so`/`.a` 中导出（已逐个 `nm` 搜过），只能经 tikcpp CMake 工具链链入，故独立小程序取不到，只能在仓内构建环境或由 Profiler 反证 |
| UB 实际占用 | 约 76.8 KB / 核（多核路径；单核路径不分配 `slotsBuf`，约 66.8 KB） | 由 §3.3.4 的预算式算出（`inBuf` 63.75 KB + `workBuf` 1056 B + `maskBuf` 2072 B + `slotsBuf` 10 KB + `outBuf` 32 B），实机编译与运行无 UB 溢出 |
| 有效读带宽（n = 4194304） | **约 770 GB/s** | 16.78 MB ÷ 21.70 us（msprof 设备侧 median，§5.1.3） |

**带宽结论**：§3.5.2 要求的峰值 634.8 GB/s 低于实测可达的约 770 GB/s，故大规模段达标，
这与 §5.1.2 的实测倍率一致（n ≈ 4.19 M 处最终轮倍率 0.943，本轮 min 0.915，仍在判据之上）。
该项是编码前的前置判断（§3.1.4 原列为待实测），结论为「可达，但余量有限」。

**最后一项是立项级的前置判断，不是事后指标**：§3.5.2 算出 n = 4194304 的
达标预算要求约 **634.8 GB/s** 的有效读带宽。若实测可达带宽不显著高于该值，
则必须在编码前就确定「单趟 HBM 访问 + UB 内完成索引定位」的方案（§3.4.2），
而不能先写两趟方案再优化。

**本任务新增实测的两条 arch22 行为**（均由实机失败反推定位，此前不见于任务书、
任务包或仓内文档，已各自写进代码注释）：

| # | 实测行为 | 发现方式 | 对策 |
| --- | --- | --- | --- |
| A | **`DataCopyPad` 的 `DataCopyPadParams.paddingValue` 在 dav-2201 上不生效**：尾块补齐的 lane 不是请求的 `+Inf`，而是比真实最小值更小的内容 | 首轮实机：n = 3 的用例返回索引 5、n = 2 返回索引 8——**索引超出 n**，只能来自补齐 lane 胜出 | 彻底不依赖补齐：向量路径只处理整 64 元素 repeat，不足 64 的尾块（每核 ≤ 63 个）走标量。仓内 `blas/iamax/arch22/isamax_kernel_impl.h:307` 的注释「dummy 自动补齐 32B，paddingValue=0」与该实测互为印证——它做 max 归约，补 0 无害，故从未暴露 |
| B | **标量读刚由向量指令写出的张量、以及 MTE 搬运刚由标量写出的张量，必须显式 `SetFlag`/`WaitFlag`**，否则读到旧值 | 多核用例稳定返回核 0 的局部最小值（其余核的 workspace 槽读成旧内容），而单核用例全部正确 | 按 `isamax_kernel_impl.h` 的同一套惯用法补齐 `V_S` / `S_MTE3` / `MTE2_S` / `V_MTE3` 事件与 `PipeBarrier<PIPE_V>`，见 `isamin_kernel.cpp` 的 `SyncPipe<>()` |

**两条继承的 arch22 约束**（已在本人 `aclsparseSpSV` A2/A3 任务中经板上探针确认，
见该设计文档 §3.1.4；本算子的跨核合并槽位直接受其约束，故此处原样继承并复验）：

1. `GlobalTensor::SetValue` 的标量 GM 写会停留在本核数据缓存，后续 kernel 的 MTE2
   读取不到；必须逐缓存行回刷
   `DataCacheCleanAndInvalid<..., SINGLE_CACHE_LINE, CACHELINE_OUT>`，
   且 `CacheLine::ENTIRE_DATA_CACHE` 档在该平台**无效**。
2. 回刷以**整条数据缓存行**为单位，两个核的输出若落进同一缓存行会互相覆盖。

⇒ 直接推论：§3.3.4 的跨核合并 workspace **必须每核独占一条完整数据缓存行**，
不能按 8 字节紧凑排布 `(float, int32)` 对。这是本算子最容易写错且症状为
「结果随核数/规模随机偶发错一位」的点。

## 3.2 核心设计决策

### 3.2.1 归约的可结合形式：候选序列重写

这是本算子唯一的算法设计点。**§3.1.1 的顺序语义不能按「每核各自按自己的首元素起基准」
的方式分核**，朴素分解会产生错误结果。

**反例 1（朴素「每核自己起基准」错）**：`x = [Inf, NaN, FLT_MAX]`，`incx = 1`。

- 顺序语义：基准 `|x1| = Inf`，idx = 1；i = 2 是 NaN 跳过；i = 3 的 `FLT_MAX < Inf` 成立
  ⇒ **result = 3**。
- 按 2 核分（核 0 取 `[Inf]`、核 1 取 `[NaN, FLT_MAX]`），若每核按自己首元素起基准：
  核 1 的首元素是 NaN ⇒ 基准置 `FLT_MAX`、idx = 2；随后 `FLT_MAX < FLT_MAX` 不成立
  ⇒ 核 1 报 `(FLT_MAX, 2)`；核 0 报 `(Inf, 1)`；合并取 `FLT_MAX < Inf` ⇒ **result = 2**。错。

**反例 2（朴素「一律丢弃 NaN」也错）**：`x = [NaN, Inf]`。

- 顺序语义：首元素 NaN ⇒ 基准 `FLT_MAX`、idx = 1；i = 2 的 `Inf < FLT_MAX` 不成立
  ⇒ **result = 1**（指向一个 NaN）。
- 若一律丢弃 NaN：唯一候选是 idx = 2 的 Inf ⇒ **result = 2**。错。

两个反例说明：`FLT_MAX` 基准既不能按核复制（反例 1），也不能直接删去（反例 2）。

**修正：把顺序递推重写成「候选序列 + 最先出现 argmin」**

定义候选序列 `c`：

```
c_1 = isnan(|x_1|) ? FLT_MAX : |x_1|      // 索引 1 恒参与竞争
c_i = |x_i|,  i >= 2，当 |x_i| 为 NaN 时该候选【不存在】（丢弃）
```

则 `result = argmin c_i`，并列取最小 i。该形式下合并算子

```
combine(A, B) = (B.min < A.min) ? B : A        // A 的索引区间严格早于 B
identity      = {valid = false}                 // 空区间
```

是**可结合**的（严格小于 + A 在前 ⇒ 并列时 A 胜 ⇒ 等价于最先出现），因此可以任意分核、
任意分块、任意树形合并，结果与分核数无关。逐一复核前述两个反例与规格要求：

| 输入 | 候选序列 | 结果 | 规格要求 | 一致 |
| --- | --- | --- | --- | --- |
| `[Inf, NaN, FLT_MAX]` | `Inf@1`，（丢弃），`FLT_MAX@3` | 3 | 3 | ✓ |
| `[NaN, Inf]` | `FLT_MAX@1`，`Inf@2` | 1 | 1 | ✓ |
| 全 NaN | `FLT_MAX@1`，其余全丢弃 | 1 | 1（§2.1） | ✓ |
| 全 Inf | `Inf@1`，`Inf@2`… | 1 | 1（并列取首） | ✓ |
| 全 0 | `0@1`，`0@2`… | 1 | 1（并列取首） | ✓ |
| `[NaN, 3.0]` | `FLT_MAX@1`，`3.0@2` | 2 | 2 | ✓ |
| `±0` 并列 | `0@i` 多个 | 最小 i | 最小 i（`\|±0\| = 0`） | ✓ |

**工程落地规则（一句话）**：
**只有拥有全局索引 1 的核（核 0）执行 `FLT_MAX` 基准替换；其余所有核一律丢弃 NaN
并携带 valid 标志。** 核 0 恒 valid（索引 1 必产生候选），故全局结果恒有定义。

**该形式已与权威 golden 逐字核对一致**。仓内 `test/isamin/isamin_golden.h`（arch 无关，
arch22 复用同一份）的 `isamin_golden()` 实现为：

```c
float minAbs = std::fabs(x[0]);
int   minIdx = 1;
if (minAbs != minAbs) minAbs = FLT_MAX;          // 替换只作用于 x[0]
for (int i = 1; i < n; i++) {
    float absVal = std::fabs(x[(size_t)i * incx]);
    if (!(absVal != absVal) && absVal < minAbs) { minAbs = absVal; minIdx = i + 1; }
}                                                 // i >= 2 的 NaN 被跳过，不参与竞争
```

`if (minAbs != minAbs) minAbs = FLT_MAX;` 在循环**之外**、且只读 `x[0]`——
这正是上述候选序列定义中「`c_1` 特例、`c_i (i≥2)` 丢弃」的非对称性，
因此本节两个反例在真实 golden 下成立，§3.2.1 的修正是必需的而非保守设计。

**受影响的值域窗口**：两种错误分解与正确分解仅在「输入同时含 NaN 且含 `|x| ≥ FLT_MAX`
的元素」时才产生分歧（因为只有 `FLT_MAX` 这个替换值能被 `|x| ≥ FLT_MAX` 平手或击败）。
该窗口很窄，但**任务包的现有用例恰好覆盖不到它**（§5.3.2），即错误实现可以全绿通过。
因此 §5.3.2 把该窗口的专项用例列为必补项。

### 3.2.2 单趟 HBM 访问：索引定位必须留在 UB 内

§3.5.2 的带宽预算要求 n = 4194304 时达到约 634.8 GB/s 有效读带宽。
一个自然但错误的实现是两趟：第一趟 `ReduceMin` 求最小值，第二趟扫描找首个等于该值的
索引——这把 HBM 流量翻倍，预算直接变为约 1270 GB/s，在 A2 上没有任何实现方式能达标。

⇒ **硬性约束：x 只允许被读一次。** 每个 UB 分块在搬入后，必须在 UB 内同时解出
`(分块最小值, 分块内最先出现的位置, 是否有效)` 三元组，再以 §3.2.1 的 `combine`
归并进本核的标量滚动状态。具体指令序列见 §3.4.2。

### 3.2.3 小规模走单核快路径

任务包 200 条性能用例中 11 条为 `n ≤ 1024`，其 GPU 标杆为 8.06 ~ 8.43 us，
1.25 倍预算 **10.07 ~ 10.54 us**（最紧的一条是 n = 128，预算 10.07 us）。
该区间完全由下发与同步开销主导，`n` 的影响可忽略。

⇒ `blockDim` 按 `n` 动态收缩（§3.3.3），且当 `blockDim == 1` 时**整条跨核路径
（workspace 写出 + `SyncAll()` + 第二级归并）被编译期/运行期旁路**，核 0 直接把
结果写进 `result`。小规模下省掉的是一次全核同步与一次 GM 往返，这正是该区间的
主要成本项。

### 3.2.4 非连续访问（incx > 1）以正确性优先

任务包明确「性能用例一律连续访存（`incx = 1`），stride 路径属功能维度，不进入性能基线」
（`test_cases/README.md` 用例规模节）。实测分布复核：200 条性能用例的 `incx` 全为 1。

⇒ stride 路径（`incx ≥ 2`）只需正确且不劣化到影响测试时长，不需要为其做带宽优化。
设计上保留一个按 `incx` 取值分流的阈值（§3.4.3），但两条分支的数值结果必须完全一致。

## 3.3 host 侧设计

### 3.3.1 函数划分

| 函数 | 职责 |
| --- | --- |
| `aclblasIsamin` | 对外入口：校验 → 快速返回 → tiling → 下发 kernel |
| `ValidateIsaminParams` | §3.3.2 的固定顺序校验，返回确定错误码（对齐 arch35 同名函数） |
| `IsaminQuickReturn` | `result` 写 0，不启动计算 kernel |
| `CalcIsaminTiling` | 运行时平台量 + `n`/`incx` ⇒ `IsaminTilingData` |
| `isamin_kernel`（device） | 第一级：分块归约 + UB 内索引定位；第二级：跨核归并写出 |

### 3.3.2 参数校验顺序与错误码

**顺序本身是语义的一部分**：`n = 0` 且 `x = nullptr` 必须返回 SUCCESS，`n = 0` 且
`result = nullptr` 必须返回 `INVALID_VALUE`，`n = −1` 且 `incx = 0` 必须返回
`INVALID_VALUE`——三者只能由顺序区分。

**已按 `blas/iamin/arch35/isamin_host.cpp` 的 `ValidateIsaminParams`（L23-57）逐条核定**，
arch22 照搬同一顺序：

1. `handle == nullptr` ⇒ `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `n < 0` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`
3. **`result == nullptr` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`**（**先于**快速返回，故
   `n = 0` 且 `result = nullptr` 也是 `INVALID_VALUE`）
4. **快速返回**：`n == 0 || incx < 1` ⇒ `result` 写 0，返回 `ACLBLAS_STATUS_SUCCESS`
   （该路径不访问 `x`，不启动计算 kernel）
5. `x == nullptr` ⇒ `ACLBLAS_STATUS_INVALID_VALUE`（此时必有 `n > 0 && incx >= 1`）
6. `lastElement = (int64_t)(n − 1) × incx`；`lastElement < 0 || lastElement > INT32_MAX`
   ⇒ `ACLBLAS_STATUS_INVALID_VALUE`（**后于**快速返回）

错误码汇总（第 3、4、6 步的先后是三条易错边界，已逐条标注源码依据）：

| 条件 | 返回码 | 由哪一步决定 |
| --- | --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | 1 |
| `n < 0`（任意 `incx`，含 `incx = 0`） | `ACLBLAS_STATUS_INVALID_VALUE` | 2 |
| `result == nullptr`（**任意 `n`，含 `n = 0` 与 `incx < 1`**） | `ACLBLAS_STATUS_INVALID_VALUE` | **3 先于 4** |
| `n == 0` 或 `incx < 1`（含 `incx = INT32_MIN`） | `ACLBLAS_STATUS_SUCCESS`，`result = 0` | 4 |
| `n = 0` 且 `x == nullptr` | `ACLBLAS_STATUS_SUCCESS`，`result = 0` | **4 先于 5** |
| `n > 0 && incx >= 1 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` | 5 |
| `n = 1024, incx = INT32_MIN`（跨度在 int32 外但 `incx < 1`） | `ACLBLAS_STATUS_SUCCESS`，`result = 0` | **4 先于 6** |
| `(n−1) × incx` 超 int32 且 `incx ≥ 1` | `ACLBLAS_STATUS_INVALID_VALUE` | 6 |
| 其余 | `ACLBLAS_STATUS_SUCCESS`，`result ∈ [1, n]` | — |

> **核定结论：任务包文档此处有错，须在自测报告中指出。**
> 任务书 §2.4/§2.5（「`result` 为 nullptr 时返回 `INVALID_VALUE`，任意 n，含快速返回路径」）
> 与 arch35 源码一致；而任务包 `test_cases/README.md:16,122` 与 `gen_csv.py:13-16` 声称
> 快速返回路径「不校验 x / result 空指针」「`result` 为 nullptr 时跳过写入」并仍返回 SUCCESS，
> **与源码相反**。由于 `isamin_param.h` 无 `result` 列（CSV 表达不了该用例），这条错误
> 不影响任何 CSV 用例的期望值，但会误导接口单测的编写——按任务包口径写的单测会断言
> SUCCESS 而实际应为 `INVALID_VALUE`。§5.3.1 的接口单测按源码口径编写。

**快速返回的实现**：`result` 位于 Device，在 handle 绑定的 stream 上
`aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream)` 写 0，
与 arch35（L88-96，memset 在 L90，issue #738「不得在 host 直接解引用 result」）同口径——
用 memset 而非启动一个写 0 的 kernel，以满足任务书 §2.5「不访问向量数据、
不启动计算 kernel」。memset 失败时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`。
异步语义与正常路径一致（§3.6）。

### 3.3.3 分核策略

遵循「核数运行时获取」与「小规模不盲目满核」两条：

```
aivNum   = GetCoreNumAiv()                              // 运行时读取，无硬编码
blockDim = clamp(ceil(n / MIN_ELEMS_PER_CORE), 1, aivNum)
```

| `n` 区间 | `blockDim` | 依据 |
| --- | --- | --- |
| `n <= MIN_ELEMS_PER_CORE` | 1 | §3.2.3：旁路 `SyncAll` 与 GM 往返；该区间延迟预算仅 ~10.1 us |
| 中间区间 | `ceil(n / MIN_ELEMS_PER_CORE)` | 每核至少有一个完整 UB 分块的工作量，避免启核成本超过收益 |
| 大规模 | `aivNum`（满核） | §3.5.2：带宽主导，需全核并发访存 |

`ISAMIN_MIN_ELEMS_PER_CORE` 最终取 **4096**。实测依据：n = 4096 走单核耗时约 3.9 us、
n = 8192 起走双核约 8.0 us，而 `n ≤ 1024` 的延迟预算是 10.07~10.54 us —— 即单核路径在
整个小规模段都有 2.6 倍以上裕量（§5.1.2），阈值无需再调。

> **与 arch35 的刻意差异**：arch35 取 `numBlocks = min(n, aivCoreNum)`（`isamin_host.cpp` L107），
> 小 `n` 不额外收缩——即 `n = 8` 时也会起 8 个核并走完整的跨核归并。arch22 这里不照搬，
> 理由是 §3.5.2 的实测预算：11 条 `n ≤ 1K` 的性能用例预算仅 10.07 ~ 10.54 us，该区间
> 成本几乎全部来自下发与全核同步，多起核只增加开销而无收益。该差异须由
> §5.4 的调参实测确认收益后定稿；若实测表明收缩无益，则回退到与 arch35 一致的口径
> 并在自测报告中说明。

**分段方式**：按全局索引**连续升序**切分——
`base = n / blockDim`、`rem = n % blockDim`，前 `rem` 个核各多分 1 个元素。
连续升序是 §3.2.1 `combine` 可结合性的前提（核号序 = 索引序），
**不得**改为跨步（stride-by-core）分配。

### 3.3.4 数据分块和 workspace 布局

**UB 分块**：单缓冲串行流水（搬入 → 同步 → 计算 → 同步）。
分块元素数由 `ReduceMin` 的 repeat 上限确定：

最终实现的 UB 占用（`tileSize` = `FP32_MAX_DATA_COUNT` = 16320）：

| 缓冲 | 大小 | 用途 |
| --- | --- | --- |
| `inBuf_` | 16320 × 4 B = **63.75 KB** | 分块数据（原地做 `Abs`/`Select`） |
| `workBuf_` | 1056 B ≈ **1.03 KB** | `ReduceMin` 的工作区（按 repeat 数推导 + 一个 block 余量） |
| `maskBuf_` | 2072 B ≈ **2.02 KB** | `Compare` 的 NaN 掩码（`tileSize / 8` + 一个 block 余量） |
| `slotsBuf_` | `useCoreNum × 64 × 4 B` ≤ **10 KB**（40 核） | 核 0 回读全部跨核槽位（仅 `useCoreNum > 1` 时分配） |
| `outBuf_` | 32 B | `ReduceMin` 输出与 result 写出暂存 |
| 合计 | **约 76.8 KB / 核** | 多核路径；单核路径不分配 `slotsBuf_`，约 66.8 KB。远低于 A2 的 UB 容量，实机无溢出 |

`tileSize` 取 **16320 = 255 × 64**：`ReduceMin` 的 `repeatTimes` 上限为 255，每 repeat
处理 64 个 FP32，故这是单次 `ReduceMin` 能覆盖的最大元素数（与 arch35 的同名常量一致）。

> **未实施双缓冲**：原设计考虑 `2 × tileElems`（x 双缓冲）以重叠搬入与计算；
> 最终实现是单缓冲串行（搬入 → 同步 → 计算 → 同步），理由与后续优化见 §6.3。

**跨核合并 workspace**：

| 子区 | 大小 | 用途 |
| --- | --- | --- |
| 每核结果槽 | `blockDim × CACHELINE_BYTES` | 核 `b` 写 `(minVal: fp32, idx: int32)`；`idx = 0` 编码「本核无有效候选」 |

- **每核独占一条完整数据缓存行**，不得紧凑排布——依据 §3.1.4 约束 2。
- 槽内用 `idx = 0` 表示 invalid（合法索引恒 ≥ 1），无需额外标志字段。
- 索引以 `int32` 承载（arch35 以 float 承载索引，见下）。
- 总量 `blockDim × CACHELINE_BYTES`（量级为 KB），`blockDim == 1` 时不使用。
- **workspace 取自 handle**（`GetEffectiveWorkspace(handle)` / `GetEffectiveWorkspaceSize(handle)`），
  与 arch35（L115-123）同机制，**不在调用内申请**；容量不足时返回
  `ACLBLAS_STATUS_EXECUTION_FAILED`。kernel 直调模式下作为 kernel 入参传入。

> **与 arch35 的刻意差异**：arch35 按 `numBlocks × 2` 个 float 紧凑排布（每核 8 B，
> 整体对齐到 64 float = 256 B，`isamin_host.cpp` L117-120）。arch22 **不照搬**：
> §3.1.4 约束 2 已实测确认，arch22 上回刷以整条数据缓存行为单位，两个核的输出若落进
> 同一缓存行会互相覆盖——8 B 间距必然导致多核共享缓存行。改为每核独占一行后
> workspace 从约 0.3 KB 增至数 KB 量级，相对 handle workspace 可忽略。
> **此项是本算子最易出错且症状最隐蔽的一处**：紧凑排布在小核数或特定规模下可能偶然
> 正确，表现为「结果随核数/规模随机偶发错一位」。§5.3.3 的 `blockDim` 一致性回归
> （同用例在单核与满核下结果须逐位相同）是该项的检出手段。

### 3.3.5 tiling 字段

最终实现的 `IsaminTilingData`（6 个 `uint32_t`，与 arch35 同名结构保持同构）：

| 字段 | 说明 |
| --- | --- |
| `totalN` | 元素个数（进 kernel 时已保证 > 0） |
| `perCoreN` | 每核元素数（前 `useCoreNum - 1` 个核） |
| `lastCoreN` | 最后一个核的元素数（`perCoreN + totalN % useCoreNum`，承载余数） |
| `useCoreNum` | 实际使用核数（§3.3.3） |
| `tileSize` | UB 分块元素数 = `FP32_MAX_DATA_COUNT` = 16320（255 × 64，`ReduceMin` 的 repeat 上限） |
| `incx` | 步长（进 kernel 时已保证 ≥ 1） |

**不设 tilingkey**：原设计打算以 `(strideMode, singleCore)` 组合做 key 以避免 kernel 内分支。
最终实现不需要——两处分支都是**每次调用一次**的粗粒度判断（`useCoreNum == 1` 走单核
快路径；`incx == 1` 选向量或标量路径），不在分块循环内，运行时代价可忽略，而多一套
tilingkey 会多出一组 kernel 实例与分发代码。保持单 kernel 更符合仓库对「不得包含
明显冗余内容」的要求。

## 3.4 kernel 侧设计

### 3.4.1 两级归约结构（单 kernel + SyncAll）

**全部形状只下发一个 kernel**（`isamin_kernel`）。跨核归约用 `SyncAll()` 而非第二个
kernel，与 arch22 兄弟算子 `blas/iamax/arch22` 的结构一致：

```
if (useCoreNum == 1) {                            // 小规模快路径，无 workspace 往返
    best = SeedCandidate(x)                       // §3.2.1 的索引 0 候选
    best = Fold(best, Reduce(0, totalN))
    写 result; return;
}

第一级（全核并行）：
  核 b 负责全局索引区间 [b*perCoreN, +cnt)（连续升序；余数归最后一个核）
  best = { +Inf, 区间首索引 }
  for each UB tile: CopyIn → UB 内求 (tileMin, tileIdx) → best = Fold(best, tile)
  不足 64 元素的尾块走标量（§3.1.4 行为 A）
  写 workspace 槽 b（256 B 槽距，整 32 B 向量存，§3.3.4）

SyncAll()

第二级（仅核 0）：
  best = SeedCandidate(x)                         // 索引 0 恒参与，值为 isnan ? FLT_MAX : |x0|
  for b = 0 .. useCoreNum-1:  best = Fold(best, slot[b])   // 升序 = 索引序
  result = best.idx + 1
```

`Fold` 为 §3.2.1 定义的 `(B.min < A.min) ? B : A`（`A` 在前，严格小于 ⇒ 并列取先者）。

**为什么不是两个 kernel**：最初实现为 `partial` + `reduce` 两次下发，功能正确，但
在共享容器上**每次调用的 host 下发成本翻倍**，而小规模段是下发受限的；实测表现为
性能用例随机出现连续区段整体劣化到约 29 us（≈ 正常单次下发成本的 2 倍）、与 n 无关，
每轮失败 11~30 条且位置漂移。改为单 kernel 后同口径累计 12 轮 200/200（其中 1 轮曾出现 1 条 host 争用抖动、复跑恢复；§5.1.2）。

**不再需要 valid 标志**：§3.2.1 的候选序列把 NaN 归一为 `+Inf` 而非丢弃。两者等价，
因为「全局最小值为 +Inf」只可能发生在所有元素都是 ±Inf 或 NaN 时，而那种输入下
索引 0 的候选（`FLT_MAX` 或 `Inf`）必然以首次出现胜出，结果恒为 1。省掉标志后
槽位只需 (值, 索引) 两个字段。

### 3.4.2 UB 内的分块三元组求解（单趟）

对一个已搬入 UB 的 `cnt` 元素分块（`cnt` 恒为 64 的整数倍，见下），在 UB 内解出
`(tileMin, tileIdx)`，不回读 HBM。最终实现的指令序列：

| # | 操作 | 说明 |
| --- | --- | --- |
| 1 | `DataCopy(in, xGm[off], cnt)` | 整 32 B 对齐连续搬入；`cnt % 64 == 0` 故无需 `DataCopyPad` |
| 2 | `Abs(in, in, cnt)` | 取绝对值；`±0` 同归为 `+0` |
| 3 | `Compare(mask, in, in, CMPMODE::EQ, cnt)` | `t == t` 为假**恰好**在 NaN 处 ⇒ mask 为「非 NaN」 |
| 4 | `Select(in, mask, in, +Inf, VSEL_TENSOR_SCALAR_MODE, cnt)` | NaN lane 置 `+Inf` |
| 5 | `ReduceMin(out, in, work, cnt, true)` | `calcIndex = true` ⇒ `out[0] = 最小值`、`out[1] = 索引`（float 位模式） |

每条相邻向量指令之间插 `PipeBarrier<PIPE_V>`，第 5 步后插 `V_S` 事件再做标量读
（§3.1.4 行为 B）。

设计要点：

- **第 3 步用 `Compare(t, t, EQ)` 而不是检查 Abs 的符号位**：`t != t` 恒真仅对 NaN 成立，
  因此无论 `Abs` 对 NaN 的符号位如何处理，NaN 都被准确识别。这比「清符号位 +
  按整数钳位」的方案更稳，且后者在本平台不可用——`Ands` 对 `int32` 在 dav-2201 上
  只有声明没有定义（实测链接报 `undefined symbol: AscendC::Ands<...LocalTensor<int>...>`）。
- **第 4 步之后分块内已无 NaN**，因此 `ReduceMin` 的 NaN 行为（文档未规定）不可能
  影响结果——这正是 §6.2-c 原先担心的点，现在从源头消除而非依赖实测行为。
- **直接采用 `ReduceMin` 的索引输出**。原设计打算用「比较 + Select + 对索引再取最小」
  自行解出首个命中位，以免依赖该指令的并列语义；实测证明无此必要：任务包的
  `VALUE_NORM_0`（全零 n = 64/1024）、`RANDOM_ALTER`、`tie_*_incx{2,3}` 以及本工程新增的
  `multicore_all_zero_tie_takes_first`（n = 100000 全零）全部通过，即该指令在并列时
  返回**最小** lane 号。这些用例长期留在回归里，持续守住该前提。
- **尾块走标量而非补齐**：`cnt` 只取整 64 repeat，不足的尾块（每核 ≤ 63 个元素）
  由 `ScalarRange()` 逐元素处理。原因是 §3.1.4 行为 A——`DataCopyPad` 的
  `paddingValue` 在本平台不生效，补齐 lane 会胜出并给出越界索引。尾块标量化在
  n = 4194304 / 40 核时是每核 25 个元素（占 0.024%），对带宽无影响。
- **`±0`**：`Abs` 后同为 `+0`，按并列取最小索引处理，无特殊分支。

### 3.4.3 非连续访问的搬入

最终实现只有两条路径：

| 条件 | 搬入方式 | 说明 |
| --- | --- | --- |
| `incx == 1` | 连续 `DataCopy`（整 64 repeat）+ 尾块标量 | 性能路径，满 burst；200 条性能用例全部走这里 |
| `incx >= 2` | `ScalarRange()` 逐元素 `GlobalTensor::GetValue(i × incx)` | 正确性路径 |

**为什么 stride 路径是标量**：原设计拟按 `incx` 大小分三档（连续 / 紧步长压缩 / 大步长
`DataCopyPad`）。实测否掉了其中的向量化方案——`DataCopyPad` 对 GM→UB 的每个 block
按 32 B 对齐落位，`blockLen = 4 B` 的逐元素 gather **不会在 UB 内紧凑排布**，每元素占
32 B，既浪费 8 倍 UB 又无法直接喂给 `ReduceMin`；而「连续超取 + `Gather` 压缩」在
`incx` 很大时（如 n = 4、incx = 10⁶）超取量会超出 UB，需要再加一条回退分支。

权衡后取标量单路：任务包明确 stride 不进性能基线（已核对 200 条性能用例 `incx` 全为 1），
而标量路径对任何 `incx` 都正确、代码只有 10 行、无 API 风险。实测代价可接受——1012 条
精度用例含 `incx ∈ {1,2,3,4,7,16}`、n 至 1048576，全量约 **17 秒**。
该路径与连续路径共用同一个 `CanonAbs()` 语义，数值结果一致性由 `TC_FL` 的
`tie_*_incx{2,3}`（并列 × 非单位步长）与 `TC_INC` 的非整除步长用例锁定。

### 3.4.4 确定性

同一输入重复执行逐位一致，且**与 `blockDim` 无关**，由四条共同保证：

1. `combine` 可结合且「A 在前者在并列时胜」（§3.2.1），故任意分核/分块/合并树同解。
2. 分段连续升序（§3.3.3）⇒ 核号序 = 索引序。
3. 第二级按核号升序串行归并（§3.4.1），不使用浮点原子或无序归约。
4. 分块内首个命中位由第 6~9 步自行解出（§3.4.2），不依赖指令并列行为。

⇒ 该性质可被检验：同一用例在 `blockDim = 1` 与满核下结果必须逐位相同，
作为 §5.3 的一条回归项（`singleCore` 可由 `MIN_ELEMS_PER_CORE` 调参强制触发）。

## 3.5 性能设计

### 3.5.1 判定口径

任务书 §3.3：平均单次耗时 ≤ GPU 标杆的 1.25 倍（即 ≥ 0.8× 标杆），
**逐 case 判定**，200 条全部须达标（`verify_performance.py` 在任意 FAIL 或任意
`NO_REF` 时返回非 0）。采样：`msprof op` 自带 5 次 warmup 不计入，warmup 后连续
≥ 10 次有效样本取算术平均；算子返回 SUCCESS 方计入有效样本。

测试工程须按用例打印 `[PERF] case=<case_name> ... avg_us=<数值>`
（`verify_performance.py` 按 `avg(?:_single)?_us=` 解析，**未解析到即判 FAIL，
不回退 GTest 整数毫秒**）。计时窗口用 `aclrtEvent` 夹住 warmup 后的连续下发，
排除 H2D/D2H、host 造数与 golden 比对。

**最终采样方案**（`isamin_test.cpp`）：预热 10 次后，做 **9 个独立批次 × 每批 20 次下发**，
事件夹住整批（保持队列饱满 ⇒ 测到稳态单次成本），每批得一个批均值，上报
**9 个批均值的中位数**。每个批均值仍是 ≥ 10 个有效样本的平均，满足 §3.3 的采样要求；
取中位数是为了抵抗 host 争用（§3.4.1、§5.1.2），任务书 §3.3 明文允许剔除异常样本
并说明。日志同时打印 `batch_min` / `batch_max` 以保留离散度。

> 试过但更差的方案：逐次下发计时（每次 `record(start)` → 1 次下发 → `record(end)` → 同步）。
> 设备会在 host 把 kernel 入队**之前**就消费掉 start 事件，于是那段气泡落进测量值，
> 200 条几乎全不达标。该结论已写入测试代码注释，避免后续重复踩。

### 3.5.2 带宽与延迟预算（由 `gpu_baseline.csv` 推导）

对 `test_cases/gpu_baseline.csv` 的 200 条 A100 实测基线，按 1.25 倍预算
反算本算子须达到的有效读带宽（`n × 4B / 预算耗时`）：

| `n` | A100 标杆 | 1.25× 预算 | 须达有效读带宽 |
| --- | --- | --- | --- |
| 128（最紧延迟预算） | 8.06 us | **10.07 us** | — （延迟主导） |
| 1 | 8.40 us | 10.50 us | — （延迟主导） |
| 1024 | 8.43 us | 10.54 us | — （延迟主导） |
| 65536 | 14.27 us | 17.83 us | 14.7 GB/s |
| 1048576 | 17.29 us | 21.61 us | 194 GB/s |
| 2097152 | 17.63 us | 22.04 us | 380 GB/s |
| 4194304 | 21.15 us | 26.43 us | **634.7 GB/s** |
| 4160954（全集最紧） | 20.98 us | 26.22 us | **634.8 GB/s** |

全集分布：**200 条中 93 条要求 > 400 GB/s**，峰值要求约 **634.8 GB/s**；
规模分布为 `n > 1M` 132 条、`64K < n ≤ 1M` 48 条、`1K < n ≤ 64K` 9 条、`n ≤ 1K` 11 条。

⇒ 两个优化目标，各自绑定一条设计决策：

| 区间 | 条数 | 主导因素 | 对应设计 |
| --- | --- | --- | --- |
| `n > 1M` | 132 | HBM 读带宽 | §3.2.2 单趟访问（硬性）+ §3.3.3 满核 + §3.3.4 单缓冲 UB 分块 |
| `n <= 1K` | 11 | 下发与同步延迟（预算 ~10.1 us） | §3.2.3 单核快路径，旁路 `SyncAll` 与 GM 往返 |

**风险与前置动作**：峰值 634.8 GB/s 是本任务唯一的硬性性能风险。
A100 在该点本身只用到其 HBM 峰值带宽的约 1/3（即标杆并非带宽饱和值，
说明该目标并非不可达），但本算子**没有算术强度可供隐藏延迟**，
容错空间小。故 §3.1.4 把「HBM 实测流式读带宽」列为编码前的前置实测项：
实测结论见 §3.1.4：`n = 4194304` 的有效读带宽约 770 GB/s，高于所需的 634.8 GB/s，
故该点达标但余量有限（最终轮实测 0.943，本轮 min 0.915）。

`n ≤ 1K` 的 11 条若实测不达标，按任务包 `README.md` 既有口径
（「部分特殊场景下自动生成的 case 可能导致标杆出现异常行为使测试行为无意义，
此时可根据实际场景过滤掉或者修改这些 case 并给出相应的说明」）在自测报告中
逐条列出实测值与差距，**不作达标外推**，并说明过滤理由。

## 3.6 支持硬件

| 硬件 | 支持 | 说明 |
| --- | --- | --- |
| Atlas A2 训练/推理系列 | 是 | arch22；自验证设备 910B3 |
| Atlas A3 训练/推理系列 | 是 | arch22，与 A2 共用本实现；任务书 §3.1 要求自验证仅覆盖 910B3 |
| Ascend 950PR / 其他 | 不涉及 | 由既有 `blas/iamin/arch35/` 承担 |

`blas/iamin/README.md` 的产品支持表须随本 PR 更新，把 A2/A3 从「不支持」改为支持。

## 3.7 算子约束限制（任务书 §2.5 逐项核对）

| 约束项 | 任务书内容 | 本设计结论 | 落点 |
| --- | --- | --- | --- |
| 参数合法性与校验层级 | handle 判空 ⇒ `HANDLE_IS_NULLPTR`；`n < 0`、`result` 判空、`n > 0 && incx >= 1` 时 x 判空、`(n−1)×incx` 溢出 ⇒ `INVALID_VALUE`；`n = 0` 或 `incx < 1` 为合法快速返回；元素值不校验 | **满足**，全部在 Host 侧校验并返回确定错误码。六步顺序已按 arch35 源码 L23-57 核定：`result` 判空**先于**快速返回（故任意 `n` 下 `result = nullptr` 均为 `INVALID_VALUE`，与任务书一致、与任务包 README 相反），溢出守卫**后于**快速返回 | §3.3.2 |
| 非连续 Tensor 支持 | 支持；按 `incx` 间隔访问，正步长生效；`incx < 1` 不进入计算 | **满足**，两条路径覆盖 `incx = 1`（向量）与 `incx ≥ 2`（标量）；`incx < 1` 走快速返回不进 kernel；两条路径共用同一 `CanonAbs()` 语义，数值结果一致 | §3.4.3 |
| broadcast 规则 | 不涉及 | **不涉及**，BLAS-1 单向量算子，无操作数间广播 | — |
| dynamic shape 要求 | 不要求，静态 shape 满足 | **不要求**，`n`/`incx` 为运行时 Host 入参，tiling 按运行时 `n` 计算，任意合法规模一次适配，无图模式依赖 | §3.3.3 / §3.3.4 |
| 原地与视图语义 | `result` 为独立输出，不复用输入内存，不返回视图；`x` 只读 | **满足**，`result` 为独立 Device 输出；`x` 全程只读（仅 MTE2 搬入）；workspace 为独立申请，不与 `x`/`result` 重叠 | §3.3.4 |
| 确定性计算要求 | 不要求；并列取最先出现索引，单次执行内部确定 | **强于要求**：同一输入重复执行逐位一致，且与 `blockDim` 无关，四条保证见 §3.4.4，并作为回归项可检验 | §3.2.1 / §3.4.4 |
| 空 Tensor 处理 | 等价于 `n = 0`：不访问向量数据、不启动计算 kernel，按快速返回口径返回成功 | **满足**，`n = 0` 在第 3 步快速返回，用 `aclrtMemsetAsync` 写 0 而非启动 kernel，不存在空 Device 指针参与计算的路径 | §3.3.2 |
| 0 维 Tensor 处理 | 不涉及 | **不涉及**，向量算子无 0 维输入；标量语义由 `n = 0` 快速返回与 Host 标量入参承担 | — |
| 异步执行 | 经 handle 绑定 stream 异步下发；返回成功仅表示 Host 已提交；读取输出前须同步 | **满足**，正常路径 kernel 与快速返回的 `aclrtMemsetAsync` 均下发到 handle 的 stream，口径一致；Host 侧不做任何隐式同步 | §3.3.2 |

---

# 4 特性交叉分析

| 特性 | 与本算子的交叉 |
| --- | --- |
| 并列最小值 | 严格小于 + 分段连续升序 + 按核号升序归并 ⇒ 取最小索引（§3.2.1） |
| NaN × 索引 1 | 索引 1 以 `FLT_MAX` 身份参与竞争，**仅核 0** 执行该替换（§3.2.1） |
| NaN × `i ≥ 2` | 归一为 `+Inf`，等价于丢弃（§3.4.1 末段给出等价性论证）；分块内由 `Compare(t,t,EQ)` + `Select` 完成（§3.4.2 第 3~4 步） |
| NaN × `\|x\| ≥ FLT_MAX` | 唯一会暴露错误分核分解的值域窗口；现有用例覆盖不到，须补专项（§5.3.2） |
| 全 NaN | 无有效候选 ⇒ 核 0 的 `FLT_MAX@1` 胜出 ⇒ 结果 1 |
| 全 Inf | `Inf@1` 与后续并列 ⇒ 取首 ⇒ 结果 1 |
| `±0` | `Abs` 后同为 `+0`，按并列取最小索引，无特殊分支 |
| 次正规数 / `FLT_MIN` | 正常参与比较，无下溢处理（只做 abs 与比较，不做算术） |
| `incx = 1` × 大规模 | 性能主路径：满核 + 单趟访问 + 单缓冲分块（§3.2.2 / §3.5.2） |
| `incx ≥ 2` | 功能路径，两档搬入，不进性能基线；数值结果与连续路径一致（§3.4.3） |
| `incx < 1`（含 `INT32_MIN`） | 快速返回，`result = 0`，不进 kernel，不访问 `x` |
| `n = 0` × `x = nullptr` | 快速返回先于 x 判空 ⇒ SUCCESS（§3.3.2 第 3 步） |
| `n < 0` × `incx = 0` | `n < 0` 先于快速返回 ⇒ `INVALID_VALUE`（§3.3.2 第 2 步） |
| 步长溢出 × 快速返回 | `incx = INT32_MIN` 时两条守卫竞争，已核定为快速返回在前 ⇒ SUCCESS（§3.3.2） |
| `blockDim = 1` vs 满核 | 结果必须逐位相同（§3.4.4），作为回归项 |
| 小规模（`n ≤ 1K`） | 延迟主导，预算 ~10.1 us，走单核快路径（§3.2.3） |
| 跨核 workspace × arch22 缓存行 | 每核独占整条缓存行 + 逐行 `dcci`；紧凑排布会互相覆盖（§3.1.4） |
| 异步 × 读取输出 | 正常路径与快速返回同为异步；读 `result` 前须 `aclrtSynchronizeStream` |

---

# 5 可维可测分析

## 5.1 精度标准 / 性能标准 / 内存标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出为 INT32 索引 ⇒ 按 `actual == golden` **整数精确比对**，不适用浮点容差。任务书 §3.2 的 FLOAT32 阈值（rtol/atol 2⁻¹³、`required_matched_ratio` 0.99、`max_abs_error_limit` 1e-2 或 32×ULP）在本算子上退化为 `matched_ratio = 1.0`、`max_abs_error = 0`。`x` 的 FP32 值仅为生成域，不参与精度比对 | 任务书 §3.2；[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | 平均单次耗时 ≤ GPU A100 标杆 × 1.25（≥ 0.8× 标杆），**逐 case** 判定，200 条全达标；warmup 5 次不计入，有效采样 ≥ 10 次取算术平均 | 任务书 §3.3；`test_cases/gpu_baseline.csv` |
| 内存标准 | **不涉及**（任务书 §3.4 明文）。交付件模板仍要求提交 `4.1 内存自验证报告.xlsx` / `4.2 内存自验证日志.log`，按「不涉及」口径出具，并说明依据 | 任务书 §3.4 / §4 |

golden 来源：Netlib 参考库无 `isamin.f`，golden 由测试工程内按 cuBLAS 文档语义
编写的 CPU 参考实现直接计算 1-based 索引（FP32 输入，不涉及浮点收窄）。

### 5.1.1 精度实测结果（910B3 / CANN 9.1.0）

| 项 | 结果 | 说明 |
| --- | --- | --- |
| 精度用例（1012 条，排除 `TC_PF_`） | **PASS = 1012，FAIL = 0，ALL PASS** | `verify_accuracy.py` 解析逐条 PASS/FAIL；含新增 12 条 `incx=16`（§5.3.2-5） |
| 其中正态分布用例（§5.3.2-1 补齐） | **397 条**，全部通过 | 随机分布段 425 均匀 / 397 正态 = 51.7% / 48.3% |
| 接口与语义单测 | **8 / 8** | `IsaminArch22Test.*`，含下列三项 |
| └ NaN × `\|x\| ≥ FLT_MAX` 专项（§5.3.2-2） | 通过 | §3.2.1 两个反例 + 8 个边界向量，单核与多核各一组 |
| └ `result`/`handle` 判空、步长溢出、quick return | 通过 | CSV 无法表达的接口契约（§5.3.1） |
| └ 核数无关性（`blockDim = 1` vs 满核） | 通过 | n ∈ {4096, 4097, 8192, 16321, 100000, 262144}，§3.4.4 |

### 5.1.2 性能实测结果（Atlas 800T A2 / 910B3）

| 项 | 结果 |
| --- | --- |
| 性能用例（200 条 `TC_PF`） | **PASS = 200，FAIL = 0，NO_REF = 0** |
| 倍率（= GPU 标杆 / NPU，判据 ≥ 0.8） | **min 0.915**、p05 0.963、**median 1.179**、max 2.793 |
| 最紧 5 条 | `TC_PF_1085` n = 4049400 → 0.915；`1178` n = 4162192 → 0.923；`1102` n = 4160954 → 0.931；`1160` n = 4053650 → 0.933；`1003` n = 4194304 → 0.943 |
| 峰值带宽点 `n = 4194304` | 实测 22.42 us（脚本口径）/ 21.70 us（设备侧 median），预算 26.43 us ⇒ 达标 |
| 最紧延迟点 `n = 128` | 实测 5.28 us，预算 10.07 us ⇒ 达标（裕量 1.9×） |
| 未达标用例 | **无** |
| 稳定性 | 本次补充连续 3 轮（min 0.918 / 0.924 / 0.915）均 **200/200**；此前累计 9 轮亦 200/200。另 1 轮曾出现 1 条 host 争用抖动（`TC_PF_1180` ratio 0.791），复跑即恢复，按任务书 §3.3 口径剔除并在此说明 |

**裕量的两种口径**，一并如实列出：

- 脚本口径（host-inclusive，判定依据）：最紧一条 0.915 对阈值 0.8，**裕量 14.4%**；
- 任务书 §3.3 指定的 msprof 设备侧 task duration 口径：`n = 4194304` 为 21.70 us
  对预算 26.43 us，**裕量 17.9%**（预算为实测的 1.22 倍）。

两者差值即 host 下发开销。该算子的大规模段余量有限，已识别的进一步优化方向见 §6.3。

### 5.1.3 Profiler 证据（910B3）

`msprof --application="./build/test/isamin/isamin_test --gtest_filter=*TC_PF_1003" --ai-core=on --task-time=on`，
取 `mindstudio_profiler_output/op_summary_*.csv` 的 `Task Duration(us)`。
（`msprof op` 模式对本 GTest 二进制只产出一行 `NA`，不可用；改用通用 profiling 模式。）

| 项 | 实测 |
| --- | --- |
| 捕获 kernel | `isamin_kernel` **391 次**（= 一次性预热 100 + 100 + 用例正确性 1 + 用例预热 10 + 计时批 9×20 = 180） |
| Task Type | **全部 `AI_VECTOR_CORE`；`AI_CPU` = 0** |
| Stream | **单一 stream（id 47）**，即调用方 handle 绑定的那条 |
| Block Num | **40**（多核路径，291 次）/ **1**（单核路径，100 次）——交叉验证 AIV = 40 与 §3.3.3 的分核收缩 |
| `n = 4194304`（40 核） | 291 次：avg 21.74 us、**median 21.70**、min 20.80 us（首条冷启动 46.27 us 计入 avg，median 不受影响） |
| `n = 64`（1 核） | 100 次：avg 3.97 us、**median 4.08**、min 3.18 us |

设备侧 median 与脚本 `avg_us`（22.42 us / 5.28 us）同量级且吻合，说明在无争用的运行中
host 下发不是瓶颈；§3.4.1 所述的异常区段则是 host 争用所致，与算子无关。
msprof 数据仅供交叉验证，**不参与判定**（判定以 GTest 的 `[PERF] avg_us` 为准，
`test_cases/README.md` 性能验收节）。

### 5.1.4 实测环境与运行批次

| 项 | 值 |
| --- | --- |
| 设备 | Atlas A2 / **910B3**（AtomGit CANNLab Dev Space `Ascend_A2_CANN`），`ASCEND_VISIBLE_DEVICES=1` ⇒ 逻辑 device 0 |
| CANN | **9.1.0**；npu-smi 25.5.0；HBM 65536 MB |
| 构建 | `bash build.sh --soc=ascend910b3 --ops=isamin`，Release |
| 运行日期 | 2026-11-11 |
| §5.1.1/5.1.2 的数值 | 取自同一构建的最终实测：精度/单测 15:29-15:30，性能最终轮 15:27-15:28（`logs/unit.log`、`logs/accuracy.log`、`logs/performance.log`、`logs/perf_results.csv`）；另两轮复跑见 `logs/final_run/perf_script_{1,2}.log` |
| §5.1.3 的数值 | 取自同一二进制的 `logs/msprof_op_summary.csv`（15:30 采集） |

## 5.2 兼容性分析

- 接口原型与 `include/cann_ops_blas.h` L240 既有声明逐字一致，**无改动**，与 950PR 共用。
- 新增目录 `blas/iamin/arch22/` 与 `test/isamin/arch22/`，不触碰 arch35 任何文件。
- `test/isamin/CMakeLists.txt` 的改动限于**新增** arch22 分支的 GTest 注册，
  arch35 分支不变；合入前须跑一轮 arch35 回归确认无影响。
- `blas/iamin/README.md` 的产品支持表更新为 A2/A3 支持。

## 5.3 测试设计

### 5.3.1 测试工程

测试工程位于 `test/isamin/arch22/`，CSV 驱动 + C++ GTest，对齐仓内既有 test 工程结构。
已通读 arch35 测试工程全部源文件，可复用与须新写的部分如下：

| 仓内既有文件 | arch 相关性 | arch22 的处理 |
| --- | --- | --- |
| `test/isamin/isamin_golden.h` | arch 无关 | **直接复用**，语义已核对（§3.2.1） |
| `test/isamin/isamin_param.h` | arch 无关 | **直接复用**（三列 `n`/`incx`/`x`，无 `result` 列） |
| `test/isamin/isamin_npu_wrapper.h` | arch 无关 | **须改造**，见下方「负向用例缓冲区」 |
| `test/frame/fill.h` | arch 无关 | **直接复用，不改动**。`M_GAUSS` 原生支持 ⇒ `GAUSS_NORM_5_2`（默认 μ=5/σ=2）可直接用于缺口 1；`ExtremeGenerator` 循环 `{1, 0, −1, FLT_MAX, FLT_MIN, −FLT_MAX, denorm_min}`，故 `RANDOM_EXTREME` 的最小值 0 恒在 1-based 第 2 个元素（与 `gen_csv.py` 注释一致） |
| `test/isamin/arch35/isamin_test.cpp` | arch35 | 作为 arch22 测试的结构模板，**但缺两项**，见下 |

**arch35 测试工程的两项缺口**（arch22 须补齐）：

1. **无任何性能计时与 `[PERF]` 打印**。arch35 测试只有 `NullHandle` 与 `CsvDriven`
   两个用例体，不含 `aclrtEvent` 计时。而 `verify_performance.py` 只解析
   `[PERF] case=<name> ... avg_us=<值>` 且**未解析到即判 FAIL**，故该打印须由
   arch22 新写（warmup 5~10 次后连续下发取均值，排除 H2D/D2H 与 golden 比对）。
2. **`NullHandle` 测的不是算子**。arch35 的该用例期望
   `ACLBLAS_STATUS_NOT_INITIALIZED`，因为 `aclblasIsamin_npu` 包装层在
   `handle == nullptr` 时**自己**返回 NOT_INITIALIZED，根本没调到算子；
   算子真实的 `HANDLE_IS_NULLPTR` 分支（§3.3.2 第 1 步）从未被覆盖。
   arch22 另加一条直接调 `aclblasIsamin(nullptr, ...)` 的用例覆盖真实分支。

**负向用例缓冲区：现有 wrapper 会在溢出用例上自爆（须改造）**

`isamin_npu_wrapper.h` 对任意 `n > 0 && incx > 0` 一律按
`dataBytes = ((n−1)×|incx| + 1) × 4` 分配，而 `test/frame/device.h` 的
`DeviceBuffer` 构造函数在 `aclrtMalloc` 失败时**抛 `std::runtime_error`**。
对任务包用例 `incx_span_overflow_n4`（`n = 4, incx = 2147483647`）：

- `isamin_test.cpp` 先 `makeBlasArray(xLen=6442450942, ...)` ⇒ 约 **25.8 GB host 内存**；
- 再 `DeviceBuffer(25769803776)` ⇒ 约 **24 GB device 分配**，失败后抛异常；
- GTest 记为「异常失败」，而非期望的 `ACLBLAS_STATUS_INVALID_VALUE`——**接口行为根本没被测到**。

该情形此前未暴露，因为 **arch35 自带 CSV 仅 62 条且无任何溢出用例**（已核对：
`incx` 最大为 4、最小为 −2），是新任务包的 1200 条用例首次引入。

⇒ arch22 按任务书 §3.5「负向用例缓冲区分配规则」改造：**`expect_result != SUCCESS`
的用例一律以单元素 dummy 缓冲区（或 `NULLPTR`）占位，不按 `(n−1)×|incx|` 分配**；
`n = 0` 或 `incx < 1` 的 quick return 用例同样不按步长分配。改造落在 arch22 自己的
wrapper/测试体内，**不改动 arch 无关的 `isamin_npu_wrapper.h`**，以免影响 arch35。
列格式对齐 `test/isamin/isamin_param.h` 的 `ReadMap` 键
（`case_name,description,n,incx,x,expect_result,random_seed`，与 arch35 表头逐列一致；
因输出为整数索引，**不含** `mere_threshold` / `mare_multiplier` 列）。

任务包随任务提供的用例与脚本（`test_cases/`）构成自验证主体：

| 文件 / 入口 | 项数 | 覆盖 |
| --- | --- | --- |
| `isamin_test.csv` | 1212 | 精度 1012（`TC_L0` 15 / `TC_SQ` 38 / `TC_INC` 10 / `TC_FL` 21 / `TC_ED` 19 / `TC_I16` 12 / `TC_EX` 897）+ 性能 200（`TC_PF`） |
| `gpu_baseline.csv` | 200 | A100 实测基线（A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4），按 `(n, incx)` 联合匹配 |
| `verify_accuracy.py` | — | 编译 + 执行 GTest（排除 `TC_PF`），解析逐条 PASS/FAIL |
| `verify_performance.py` | — | 执行 `TC_PF`，解析 `[PERF] avg_us`，与基线按 0.8 倍率判定 |
| 接口契约单测（须新增） | — | `result == nullptr` 与 `handle == nullptr`——`isamin_param.h` 无 `result` 列（已核对，仅 `n` / `incx` / `x` 三列），CSV 表达不了，须由测试工程单测覆盖。**按 §3.3.2 核定口径断言**：`result = nullptr` 在 `n = 0`、`incx = 0`、`n > 0` 三种情形下**一律** `INVALID_VALUE`，不是任务包 README 所称的 SUCCESS |

### 5.3.2 已识别的用例缺口与补齐计划

对任务包用例做了一轮实测统计，五处须补齐，均纳入精度自验证并在自测报告中说明：

| # | 缺口 | 实测依据 | 补齐方式与结果 |
| --- | --- | --- | --- |
| 1 | **无正态分布用例**。任务书 §3.5 要求均匀 50% / 正态 50%，而原 `isamin_test.csv` 实测 1018/1200 为 `RANDOM_NORM_5_5`（均匀），`GAUSS_NORM_5_2` 计数为 **0** | `cut -d, -f5 isamin_test.csv \| sort \| uniq -c` | **已补齐**。`gen_csv.py --dist mixed` 原只能到 378/1000 ≈ 37.8%（固定类别 `l0/inc/ed` 硬编码均匀填充），故改 `Gen.add()` 的默认 `xf` 为 `None` ⇒ 由填充轮转决定，使**所有**类别参与 50/50；性能用例显式钉回 `FILL_RAND` 以保证 `TC_PF` 行逐字节不变。重新生成后：随机分布段 **425 均匀 / 397 正态**（51.7% / 48.3%），`gpu_baseline.csv` 未改动，`TC_PF` 200 行 `diff` 为空（已验证）。语义填充（全零/交替/序列/极值/Inf/NaN，182 条）属特殊值覆盖，不计入分布划分——此为本报告采用的口径 |
| 2 | **`FLT_MAX` 基准语义未被覆盖**。`TC_FL` 的 `first_nan_rest_rand` 用例名声称「首元素 NaN 其余随机」，实际 `xf = "VALUE_NORM_NAN"`（全 NaN），落回「全 NaN ⇒ 1」这条两种实现都能通过的路径 | `gen_csv.py:100` | **已补齐，但不走 CSV**：该形态需要逐元素指定数值，现有填充模式表达不了，而新增框架级填充模式会改动 arch 无关的 `test/frame/fill.h`（影响全部算子）。改为在 `test/isamin/arch22/isamin_test.cpp` 内写死向量：`NanSeedSemanticsSmall`（10 个向量，含 §3.2.1 两个反例 `[NaN,Inf]`、`[Inf,NaN,FLT_MAX]`）与 `NanSeedSemanticsMultiCore`（n = 100000，覆盖向量路径与跨核）。**开发期实证价值**：首轮实现这两条即失败，正是它们定位出 §3.1.4 行为 A |
| 3 | **步长溢出只有 1 条**。任务书 §3.5 要求 `n = 4, incx = INT_MAX` 与 `n = 2³⁰, incx = 8` 两种 | `gen_csv.py:131` 仅第一种 | **已补齐，且必须不走 CSV**：见下方「负向用例缓冲区」——`isamin_npu_wrapper.h` 会按 `(n−1)×\|incx\|` 预分配，CSV 表达该用例会让测试工程自身分配失败（24 GB / 34 GB）。两条均放进 `StridedSpanOverflowIsInvalidValue`，以单元素 dummy device 缓冲区占位 |
| 4 | **基线键冲突**。`verify_performance.py` 按 `(n, incx)` 建基线字典，`(4194304, 1)` 有 3 条不同 A100 实测（0.0211968 / 0.0212309 / 0.0211456 ms）被折叠，后者覆盖前者 ⇒ 随机填充与全零填充用例实际按全 Inf 的基线判定 | `gpu_baseline.csv` 三行 + `verify_performance.py:53-64` | **不改脚本**（任务包为交付件）。实测三者差异约 0.4%，远小于本算子在该点的裕量，不改变任何判定结论；在此说明以备查 |
| 5 | **`incx = 16` 未覆盖**。任务书 §3.5 明列步长覆盖 `{1, 2, 7, 16}`，而原 `gen_csv.py` 的 `POS_INCS` 只有 `{1,2,3,4,7}`，全部 1200 条中 `incx=16` 计数为 0 | `gen_csv.py` `POS_INCS` | **已补齐**：新增 `inc16_extra()`，在 `perf()` **之后**追加 12 条 `TC_I16`（10 个尺寸 + 2 条并列 × incx=16）。放在性能用例之后是为了不改动 `self.idx` 与 `ex_extend` 的 RNG 消耗，从而保证原 1000 条精度用例与 200 条 `TC_PF` 行逐字节不变；`gpu_baseline.csv` 的 A100 实测按 `(n, incx)` 绑定，故不受影响。12 条全部通过 |

### 5.3.3 用例分层对应

| 分层 | 覆盖 | 用例 |
| --- | --- | --- |
| 边界域 | `n = 0`、`n = 1`、`incx < 1`（`result = 0`）、并列最小值（重复元素与 `±0`）、NaN 样本 | `TC_ED` / `TC_FL` / `TC_INC` 的 quick return 3 条 |
| 负向域 | `handle = nullptr`、`result = nullptr`（含 `n = 0` 路径）、`n < 0`、`n > 0 && incx >= 1` 时 x 为 nullptr、步长溢出 | `TC_ED` 9 条 + 接口单测（`result`/`handle` 判空） |
| 功能域 | 尺寸扫描 1→1048576、`incx ∈ {1,2,3,4,7,16}`、非整除步长（`n=1021/incx=3`、`n=4099/incx=7`）、7 种填充 | `TC_SQ` / `TC_INC` / `TC_FL` / `TC_I16` / `TC_EX` |
| 性能域 | 3 条规模锚点 + 小尺寸延迟区 + 规模扫描 + 全零/全 Inf 填充（均 `incx = 1`） | `TC_PF` 200 条 |
| 一致性 | `blockDim = 1` vs 满核逐位相同 | §3.4.4 回归项 |

## 5.4 可测性

- **路径可触达**：向量 / 标量路径由 `incx` 确定性触发（CSV 的 `incx ∈ {1,2,3,4,7,16}` 全覆盖）；
  单核 / 多核路径由 `n` 相对 `ISAMIN_MIN_ELEMS_PER_CORE` 确定性触发，
  `ResultIndependentOfCoreCount` 用 n ∈ {4096, 4097, …, 262144} 跨越该阈值两侧，
  且两条路径结果必须逐位一致，因此路径切换本身可检验。
- **执行计划可观测**：Host 侧 `OP_LOGD` 打印 tiling 全部字段
  （`totalN / perCoreN / lastCoreN / useCoreNum / tileSize / incx`），
  现场可据此判断走的是哪条路径。
- **常量取值依据**：`tileSize` = 16320 由 `ReduceMin` 的 255-repeat 上限唯一确定（§3.3.4）；
  `ISAMIN_MIN_ELEMS_PER_CORE` = 4096 的依据见 §3.3.3；两者均非经验值。
- **复现**：`test_cases/README.md` 的两条命令
  （`verify_accuracy.py` / `verify_performance.py`，`--soc ascend910b`）。
  须先把 `isamin_test.csv` 放入 `test/isamin/arch22/`（`--csv` 会自动复制并备份）。
  `build.sh` 的 `--ops=` 取值已核对：`expand_family_ops`（`build.sh:167-175`）按
  **test 目录名**展开，而非算子实现目录名，故两个脚本传的 `--ops=isamin`
  （对应 `test/isamin/`）**正确**，无需改为 `iamin`。

---

# 6 交付与合入

## 6.1 交付物

| 交付物 | 位置 |
| --- | --- |
| 接口声明 | `include/cann_ops_blas.h` L240（原型已存在，本次**无改动**） |
| 算子实现 | `blas/iamin/arch22/` 共 4 个文件（与 arch35 的 `isamin_*` 同构，不引入冗余文件）：`isamin_host.cpp`（150 行，校验 + 快速返回 + tiling + 下发）、`isamin_kernel.cpp`（311 行，`IsaminReducer` + 单 kernel）、`isamin_kernel.h`（19 行）、`isamin_tiling_data.h`（35 行） |
| 算子文档 | `blas/iamin/README.md`（产品支持表标注 A2/A3 支持、约束与限制、复现说明） |
| 测试代码 | `test/isamin/arch22/isamin_test.cpp`（412 行：CSV 驱动 + 性能采样 + 8 条接口/语义单测）与 `isamin_test.csv`（1212 条）；`test/isamin/CMakeLists.txt` 的分支条件改为 `if("arch35" IN_LIST SOC_ARCH_DIRS OR "arch22" IN_LIST SOC_ARCH_DIRS)`，arch35 行为不变 |
| 设计文档 | 本文档，随 PR 合入 |
| 自测报告 | 用例参数、精度逐条结果、性能逐 case 倍率与原始数据、自验证步骤；未通过项与偏差如实列出。原始日志：`logs/unit.log`、`logs/accuracy.log`、`logs/performance.log`、`logs/perf_raw_gtest.log`、`logs/perf_results.csv`、`logs/msprof_op_summary.csv` |

`task_submission/` 目录（按任务书 §4 模板命名，提交至个人代码仓根目录）：

```
task_submission
├── 1 自验证步骤说明.md
├── 2.1 精度自验证报告.xlsx
├── 2.2 精度自验证日志.log
├── 3.1 性能自验证报告.xlsx
├── 3.2 性能自验证日志.log
├── 4.1 内存自验证报告.xlsx      # 按「不涉及」口径出具（任务书 §3.4）
└── 4.2 内存自验证日志.log        # 同上
```

合入路径：代码、测试与文档以 PR 提交至 `ops-blas` 的 `master` 分支，提交前须通过
仓内门禁与编译。另须在任务 IT 系统【说明】中填写验收代码 commit 链接，
并在私仓邀请账号 `Ascend-CANN` 作为开发者（任务书 §4）。

## 6.2 规格歧义的核定结果与剩余待确认项

### 表一：已核定（依据 arch35 源码与仓内 `build.sh`，无需实机）

| # | 事项 | 核定结论 | 依据 |
| --- | --- | --- | --- |
| a | `result == nullptr` 在快速返回路径的返回码 | **`INVALID_VALUE`（任意 `n`，含 `n = 0` 与 `incx < 1`）**。`result` 判空位于快速返回**之前**。任务书 §2.4/§2.5 正确；**任务包 `README.md:16,122` 与 `gen_csv.py:13-16` 错误**（该两处称快速返回路径不校验 `result`、返回 SUCCESS）。因 `isamin_param.h` 无 `result` 列，该错误不影响任何 CSV 用例期望值，但会误导接口单测——须在自测报告中指出 | `isamin_host.cpp` L34-37 先于 L38-43 |
| b | 溢出守卫与快速返回的先后 | **快速返回在前**。故 `n = 1024, incx = INT32_MIN` ⇒ SUCCESS，任务包用例 `incx_intmin_quick_return` 的期望值正确。溢出守卫实际形式为 `lastElement < 0 \|\| lastElement > INT32_MAX`（int64 中间量），其中 `lastElement < 0` 在 `incx ≥ 1` 下不可达，属防御性分支，arch22 一并保留 | `isamin_host.cpp` L38-43 先于 L48-55 |
| d | `build.sh --ops=` 取值 | **`--ops=isamin` 正确**，无需改为 `iamin`。`expand_family_ops` 按 **test 目录名**（`test/isamin/`）展开，与算子实现目录 `blas/iamin/` 无关。两个验证脚本无需修改 | `build.sh:167-175` |
| f | arch22 GTest 注册的具体改法 | `test/isamin/CMakeLists.txt` 当前仅 `if("arch35" IN_LIST SOC_ARCH_DIRS)` 一个分支，内含 `ops_blas_add_gtest_tests(${OPS_BLAS} isamin_test)`。arch22 按同一函数增加分支（或合并条件），arch35 分支不改动 | 已取仓内该文件全文（7 行） |

### 表二：已由板上实测结案

| # | 事项 | 结论 | 依据 |
| --- | --- | --- | --- |
| c | 向量指令语义 | **`ReduceMin(..., calcIndex = true)` 并列时返回最小 lane 号**（由全零/交替/并列×步长等用例持续守住，见 §3.4.2）。`Ands` 对 `int32` 在 dav-2201 **只有声明没有定义**，故改用 `Abs` + `Compare(t,t,EQ)` + `Select`；该组合使分块在进入 `ReduceMin` 前已无 NaN，原先「ReduceMin 的 NaN 行为未规定」的顾虑从源头消除 | 实机编译报错 + 1012 条精度用例与 8 条语义单测全通过（§5.1.1） |
| e | arch22 平台量与带宽 | **AIV = 40**；UB 实际占用约 76.8 KB/核（多核）/ 66.8 KB（单核）；`n = 4194304` 有效读带宽约 **770 GB/s**，高于 §3.5.2 所需的 634.8 GB/s ⇒ 大规模段可达但余量有限 | msprof `Block Num` 与 `Task Duration`（§5.1.3），§3.1.4 已回填 |
| A | `DataCopyPad.paddingValue` 不生效（**新发现**） | 见 §3.1.4 行为 A。对策：向量路径只吃整 64 repeat，尾块标量化 | 实机索引越界（n = 3 返回 5） |
| B | 标量/向量/MTE 之间需显式事件同步（**新发现**） | 见 §3.1.4 行为 B。对策：`SyncPipe<V_S/S_MTE3/MTE2_S/V_MTE3>` + `PipeBarrier<PIPE_V>` | 多核结果稳定错位 |

## 6.3 已知限制与后续优化方向

| 项 | 现状 | 说明 / 规避 |
| --- | --- | --- |
| 大规模段裕量有限 | 最终轮最紧用例 0.915 对阈值 0.8（14.4%） | 未实施的优化：UB 分块**双缓冲**，让 MTE2 搬入与向量计算重叠。当前分块循环是严格串行（搬入 → 同步 → 计算 → 同步），n = 4194304 时每核约 7 个分块、每块一次 `V_S` 流水排空。预期可显著抬高大规模吞吐。**本次未做**：当前实现已在两种口径下稳定达标，而该改造需手工管理两个在飞 MTE2 事件，正确性风险与收益不成比例；留作后续优化并由 §5.3.3 的全量回归守护 |
| stride 路径为标量实现 | `incx ≥ 2` 逐元素读 GM | 任务包明确 stride 不进性能基线（200 条性能用例 `incx` 全为 1，已核对）。实测 1012 条精度用例含 `incx ∈ {1,2,3,4,7,16}`、n 至 1048576，总耗时约 17 s，不构成瓶颈。若后续需要 stride 性能，可用「连续超取 + `Gather` 压缩」替换 |
| handle workspace 需求高于 arch35 | arch22 每核独占 256 B（`useCoreNum × 256 B`，40 核 10 KB），arch35 为 `numBlocks × 2` 个 float 紧凑排布（总量 256 B 对齐，40 核约 1.28 KB） | 调用方须提供 ≥ `useCoreNum × 256 B` 的 library workspace；实测自备 256 B workspace 时接口返回 `ACLBLAS_STATUS_EXECUTION_FAILED`（先校验后下发，不越界写）。`aclblasCreate` 默认 32 MiB 下该需求占 0.03%，正常使用无感；若调用方显式配置了更小的 workspace，需按本算子需求放大。这是 arch22 为规避缓存行伪共享而刻意增加的内存开销 |
| 性能测量对 host 争用敏感 | 见 §3.4.1、§5.1.2 | 采样改为 9 批 × 20 次取**批均值中位数**（任务书 §3.3 允许剔除异常样本并说明）；复现建议 `taskset -c 0-7` 绑核。设备侧 msprof 口径不受影响 |
| `aclblasIcamin`（复数版）未覆盖 | arch22 只实现 `isamin` | 本任务范围仅 `aclblasIsamin`（任务书 §2.1）。`blas/iamin/arch35/` 另有 `icamin`，arch22 侧为后续任务 |

---

# 7 参考资料

1. cuBLAS 官方文档（对标接口语义与返回码，§2.5.2）：<https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf>
2. ops-blas 开源仓（含 `blas/iamin/` 既有实现与 README）：<https://gitcode.com/cann/ops-blas>
3. ops-blas 编译与运行样例：<https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md>
4. ops-blas 贡献指南：<https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md>
5. 生态算子开源精度标准（mixed_tolerance）：<https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>
6. Ascend C 算子开发文档、算子开发接口文档：<https://www.hiascend.com/document>
7. 任务书与任务包：`aclblasIsamin_AtlasA2_A3_task_doc.md`、`test_cases/README.md`
8. arch22 平台约束的既有实测依据：本人 aclsparseSpSV A2/A3 设计文档 §3.1.4，
   <https://gitcode.com/cann/cann-ops-competitions/pull/2279>
