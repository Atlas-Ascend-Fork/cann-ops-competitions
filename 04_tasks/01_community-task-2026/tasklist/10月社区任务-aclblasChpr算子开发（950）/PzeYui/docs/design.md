# aclblasChpr 算子设计文档（Ascend 950PR）

作者：PzeYui；日期：2026-10-28。

本文为拟实施设计，依据随包 `aclblasChpr_Atlas950PR_task_doc.md`、1200 条 CSV 用例及 200 条 H100 基线编写。本文中的性能数值为验收目标或由基线换算的上限；本设计 PR 不包含算子实现或真机测试结果。

# 需求背景（required）

## 需求来源

10 月社区任务“aclblasChpr 算子开发（950）”要求在 Ascend 950PR 上，使用 CANN 9.1.0 和 Ascend C kernel 直调模式，为 [cann/ops-blas](https://gitcode.com/cann/ops-blas) 补齐 `aclblasChpr` 的 arch35 实现。代码在 [PzeYui/ops-blas](https://gitcode.com/PzeYui/ops-blas) 中开发，最终实现目录为 `blas/hpr/arch35/`，测试目录为 `test/hpr/chpr/arch35/`。

设计文档按 [2026 社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md) 编写。提交目录使用任务书完整名称和个人账号 PzeYui；不推测任务编号。

## 背景介绍

### 算子功能

CHPR 是 BLAS Level 2 的单精度复数 Hermitian 压缩存储秩 1 更新：

```text
A ← A + alpha · x · xᴴ
```

`alpha` 为 FLOAT32 实数，`x` 和 `A` 为 COMPLEX64。只保存、引用和更新 `uplo` 指定的三角，输出覆盖原 packed 数组 `ap`；另一半通过 Hermitian 共轭对称性隐含。

### 上游实现现状

源码调研固定在 ops-blas 提交 `05b96ea22cea926ecfd43fc7ac691a795522bd61`。以下为该基线事实与本任务拟新增部分。

| 基线文件、符号 | 现状 | 本任务设计 |
| --- | --- | --- |
| [`include/cann_ops_blas.h`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/include/cann_ops_blas.h)：`aclblasChpr` | 已有公共接口声明 | 直接复用，不增加公开 API 或重复声明 |
| [`include/cann_ops_blas_common.h`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/include/cann_ops_blas_common.h)：`aclblasComplex` | 实部、虚部各一个 float，已有公共状态码 | 保持类型、布局和 ABI |
| [`blas/hpr/arch22/chpr_host.cpp`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/hpr/arch22/chpr_host.cpp)：`aclblasChpr` | A2/A3 Host 校验与 stream 直调；handle 先于 n 校验 | 新增 arch35 Host，按本次任务书的 n/quick return 优先级校验 |
| [`blas/hpr/arch22/chpr_kernel.cpp`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/hpr/arch22/chpr_kernel.cpp)：`ChprKernel` | packed 列处理、分块及正负步长寻址 | 参考索引语义；重新规划 arch35 分核和资源，不复制架构常量 |
| [`blas/spr/arch35/sspr_kernel.cpp`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/spr/arch35/sspr_kernel.cpp)：`SsprGmImpl`、`SsprUb` | arch35 SIMT 调用和实数 packed 更新参考 | 参考 `asc_vf_call` 工程用法；Hpr 保留复数共轭和对角语义 |
| [`blas/CMakeLists.txt`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/CMakeLists.txt)、`build.sh` | 按 SOC 收集 arch35 文件，支持 `--ops=chpr` | 通过规范命名接入，检查编译源和导出符号仅出现一次 |
| [`test/hpr/chpr/chpr_param.h`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/test/hpr/chpr/chpr_param.h) | 读取 `x`、`ap`，alpha 空指针只识别 `NULLPTR` | arch35 适配随包 `x_fill`、`ap_fill` 和 `alpha=null`，保留 arch22 行为 |
| [`test/hpr/chpr/arch22/chpr_test.cpp`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/test/hpr/chpr/arch22/chpr_test.cpp) | 有 CSV/GTest 和空 handle 单测；实虚合并检查，长度计算使用 `std::abs(int)` | arch35 单独实现安全长度规划、实虚分开统计、INT_MIN 负向测试 |
| [`test/hpr/chpr/chpr_golden.h`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/test/hpr/chpr/chpr_golden.h)：`aclblasChpr_cpu` | 通过 `cblas_chpr` 生成 golden | 复用 CBLAS 计算，任务特定参数判定在测试装配层处理 |
| [`blas/hpr/README.md`](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/hpr/README.md) | 950PR 标记为不支持 | 实现和验证完成后更新 950PR 产品支持与调用说明 |

# 需求分析（required）

## 需求描述

复用公共签名：

```cpp
aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* ap);
```

| 参数 | 类型、位置 | 约束与作用 |
| --- | --- | --- |
| handle | Host，`aclblasHandle_t` | 上下文与绑定的 stream；有效计算时不能为空 |
| uplo | Host，`aclblasFillMode_t` | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | Host，int | 矩阵阶数；n<0 非法，n=0 quick return |
| alpha | Host，`const float*` | 实数标量；n>0 必须非空；通过校验后 alpha=0 原样返回 |
| x | Device，`const aclblasComplex*` | 逻辑长度 n；物理长度至少 `1+(n-1)*abs(incx)` |
| incx | Host，int | 任意非零步长，支持负步长；`INT_MIN` 非法 |
| ap | Device，`aclblasComplex*` | 输入/输出，长度至少 `n*(n+1)/2`；packed、列主序、原地更新 |

## 需求拆解

| 编号 | 要求及任务书依据 | 设计落点 | 验证方法 |
| --- | --- | --- | --- |
| R1 | 共用 API、复数类型和 Hpr 目录；§2.1–2.3 | arch35 Host/kernel，复用公共声明 | 编译、符号和路径核对 |
| R2 | 固定校验顺序、合法空操作、返回码；§2.4–2.5 | Host 分支，不触碰 Device 数据 | 25 条边界用例及补充复合异常单测 |
| R3 | 上/下三角 packed 地址、原地更新；§2.1 | 64 位列基址、互不重叠的输出区间 | 小矩阵人工 golden、双 uplo 全量比对、边界哨兵 |
| R4 | 实数 alpha、复数共轭、对角虚部；§2.1 | FP32 实虚公式和对角独立覆盖 | 实虚分别统计、对角输入虚部污染测试 |
| R5 | 任意正负非零 incx、拒绝 INT_MIN；§2.1、§2.4 | 64 位逻辑到物理索引，通用 SIMT 路径 | ±1/±2/±3，补充大步长和 n=1 |
| R6 | stream 绑定、异步执行；§2.5 | 单次 kernel 下发，无内部 stream 同步 | 非默认 stream、完成后读回 |
| R7 | 950PR、CANN 9.1.0、Netlib 3.10.0；§3.1 | arch35 构建和真实环境记录 | 设备、驱动、toolkit、CBLAS 版本证据 |
| R8 | 全量 packed、FLOAT32 精度与特殊值；§3.2 | 任务精度比较器、非有限值独立统计 | 1000 条功能/精度用例全部通过 |
| R9 | 单 case 性能≥0.4×H100；§3.3 | 分核、单 kernel、连续块读写 | 200 条逐项 msprof 数据及基线关联 |
| R10 | 无独立内存门槛；§3.4 | 无 Device workspace；有界 UB tile | 无越界、无泄漏，记录内部缓冲规模 |
| R11 | 输入分布、负向及 Inf/NaN 覆盖；§3.5 | 保留随包 CSV，另补正态分布与 API 单测 | 原始 1200 条与补充用例分别计数 |
| R12 | 设计、代码、测试及可复现材料；§4–5 | 算子 README、测试步骤、固定版本日志 | 对应同一源码与构建产物的结果索引 |

# 详细设计（required）

## 算子分析

### 数学公式

令逻辑向量 `x[i]=r_i+i*s_i`，每列先计算 `t=alpha*conj(x[j])`：

```text
t_r = alpha * r_j
t_i = -alpha * s_j
ap.real += r_i * t_r - s_i * t_i
ap.imag += r_i * t_i + s_i * t_r
```

这个计算顺序对应 Netlib CHPR 的 `TEMP=ALPHA*CONJG(X(J))`，减少与单精度 golden 的中间舍入差异。对角 `i=j` 只读取旧实部，以同一实部公式更新，并直接写 `imag=+0.0f`，忽略旧对角虚部。

`x[j]` 的实部和虚部均为零时，遵循 Netlib 的零列分支：不计算 `alpha*conj(x[j])`，不修改该列非对角元素，只把该列对角虚部清零。该分支对 alpha 或其他输入含 Inf/NaN 时尤其重要；不扩展为“所有零行跳过”。`alpha=0` 则在 Host 返回，整个 ap 保持原样，对角也不改写。

### 支持数据类型

仅实现 alpha 为 FLOAT32、x/ap 为 COMPLEX64 的已有接口。内部实虚分量保持 FP32，不转换为 FP16/BF16，不使用 Cube 低精度乘法。常规更新无归约，每个输出元素有唯一写者。

### 支持形状和布局

令 `P=n*(n+1)/2`。对 0-based 列号 j，定义第 j 列起点：

```text
B_upper(j) = j*(j+1)/2
B_lower(j) = j*(2*n-j+1)/2
```

UPPER 的 `i∈[0,j]`，`q=B_upper(j)+i`；LOWER 的 `i∈[j,n-1]`，`q=B_lower(j)+(i-j)`。两种模式均满足 `B(0)=0`、`B(n)=P`，列内连续。

负步长不移动调用者传入的物理首地址。统一寻址为：

```text
XPos(k) = k*incx                              , incx>0
XPos(k) = (n-1-k)*(-int64_t(incx))            , incx<0
```

所有乘法先提升到 64 位；物理长度、packed 元素数和字节数采用检查溢出的计算。测试包的 n 上限是覆盖规模，不作为公开接口的硬编码限制。调用者负责提供足够的 Device 内存。

## 算子实现

### 实现方案

整体执行链为：Host 校验与退化返回 → 选择分核和计算路径 → 在 handle 的 stream 上下发一个 AIV kernel → 调用者同步后读回。计算复杂度为 `O(n²)`，不构造完整 `n*n` 矩阵，不展开未存储三角，不使用跨核归约或原子更新。

#### 3.2.1 Host 侧设计

**参数校验。** 按任务书 §2.5 的顺序处理，避免读取空 alpha 或计算非法步长的绝对值：

1. `n<0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. `n==0` 返回 `ACLBLAS_STATUS_SUCCESS`，不访问 handle/alpha/x/ap，不启动 kernel。
3. 有效计算需要非空 handle，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
4. 校验 uplo 为 UPPER/LOWER，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。
5. 校验 incx 不是 0 或 INT_MIN，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。
6. 校验 alpha/x/ap 非空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`。
7. 读取 Host 标量 alpha；等于 `+0.0f` 或 `-0.0f` 时直接成功，不修改 ap。
8. 对非退化输入生成 tiling，查询实际 AIV 资源并下发；资源或下发失败按仓库约定返回相应失败状态。

任务书对 n 的优先级高于其余参数校验，因此本设计将 n<0、n=0 也置于 handle 检查前；将 `n=0/null handle`、`n<0/null handle` 单独列入复合异常测试。arch22 现有校验顺序保持原样。

**Tiling 信息。** 使用仅 arch35 内部可见的 `ChprTilingData`，保存 n、uplo、alpha、64 位 incx、64 位 packed 元素数、核数、计算路径和 tile 容量。字段同时被 Host 与 kernel 包含，避免依赖不一致的结构体布局。无需新增图模式 tiling 注册或用户可见 tiling key。

**分派候选。** 小问题启动少量 AIV；连续大问题按输出量扩展到实际可用核数。SIMT 线程数候选为 128/256，向量 tile 候选为 512/1024 个复数，切换阈值通过 950PR 实测决定。初始小尺寸候选为 `n≤64`，所有阈值仅作为调优参数，不影响输入合法性。

| 路径 | 适用场景 | 核内处理 |
| --- | --- | --- |
| SIMT 通用路径 | 小尺寸、incx≠1，或向量资源条件未满足 | 线程遍历 packed 元素，64 位寻址后直接复数更新 |
| 向量连续路径 | incx=1 且输出规模足够 | packed 连续区间分核，按列边界与固定 tile 切分后流水搬运和计算 |

**均衡分核。** 按 packed 输出元素数分配，不按相同列数分配。以 4 个 complex64、即 32 字节为分组单位，令 `G=ceil(P/4)`。B 个核按商和余数各持有连续的 group 区间，最后一组只处理有效元素。每个核的起点除总尾部外均对齐 32 字节，可减少相邻核写同一搬运块的风险。B 同时受可用 AIV 数和每核最小工作量限制。

#### 3.2.2 Kernel 侧设计

**SIMT 通用路径。** 使用 arch35 的 `asc_vf_call` 机制。线程以 grid-stride 遍历自身核的 packed 元素 q；在单调列起点函数 B(j) 上进行整数二分，求唯一的 `B(j)≤q<B(j+1)`，再得到行 i。避免用单精度平方根反解三角数导致列边界误判。

线程按 `XPos(i)`、`XPos(j)` 读取 x；执行零列分支或复数更新，独占写回一个 complex64。对角线程只读旧实部并写零虚部。该路径以正确支持任意步长为基础，不为性能用例 ID 或有限步长值设置特判。

**向量连续路径。** 每核拥有 packed 区间 `[start,end)`，以整数二分定位首列 j，之后顺序推进列号。当前处理段长度为：

```text
count = min(end-q, B(j+1)-q, tileCapacity)
```

段内列号不变，因此 `t_r/t_i` 为标量；行号连续，x 和 ap 都可连续搬运。完成一列后递增 j，无需逐元素反解列号。同一列允许被两个核分段处理，但每个 packed 元素只有一个写者。

1. CopyIn：通过 `DataCopyPad` 按实际字节数读取 ap 和连续 x 行片段，尾段不向上扩大 GM 访问范围。
2. Compute：将 AoS 的实虚交错布局拆成 FP32 分量，分别计算两次标量乘法、实虚组合与旧 ap 相加；MTE2 到 Vector 的依赖由队列事件管理。
3. 对角修正：若当前段包含对角，在 Vector 结果对 Scalar 可见后，把对应虚部分量设为 +0；旧对角虚部不参与结果。
4. CopyOut：重新交错输出，通过 `DataCopyPad` 写有效元素。等待写回完成后复用缓冲；不同段不依赖输出 padding。

连续 x 的列值在每列或每段读取一次并复用。可在实测确认收益后把较小 x 缓存在 UB；初版不要求全量 x 驻留，避免把 n 上限绑定为 UB 容量。

**UB 和同步。** 采用 `TPipe/TQue/TBuf` 分配缓冲，事件由资源管理器分配，不与硬编码 event ID 混用。候选 tile 为 1024 个复数：x 输入双缓冲 16 KiB、ap 输入双缓冲 16 KiB、输出双缓冲 16 KiB、计算临时分量 16 KiB，数据缓冲预算约 64 KiB；额外算子临时空间、对齐、事件资源由实际 API 和编译资源报告复核。资源不足时减小 tile 或回退通用路径。

UB 是每核私有临时存储；无 GM workspace、无全局 scratch、无跨核屏障。实现中不调用 `aclrtMalloc`、`aclrtFree`、D2H 拷贝或 stream 同步。调用返回后，调用者按原有 BLAS 异步约定等待 stream 完成。

**精度及特殊值。** 优先用独立 FP32 乘法和加减保持 Netlib 的运算结构；如 SIMT 编译器收缩 FMA 导致越界误差，在本算子编译范围处理，不修改全仓浮点选项。保留 NaN/Inf，不做任意截断或替换；零列跳过、对角写零和 alpha=0 分别按各自条件执行。

#### 3.2.3 工程接入与维护

```text
blas/hpr/
├── README.md
├── arch22/                         # 保留已有实现
└── arch35/
    ├── chpr_host.cpp
    ├── chpr_kernel.cpp
    └── chpr_tiling_data.h

test/hpr/chpr/
├── CMakeLists.txt                   # 复用 ops_blas_add_gtest_tests
├── chpr_param.h                    # 已有公共部分按需兼容
├── chpr_golden.h                   # 已有 CBLAS golden
└── arch35/
    ├── chpr_test.cpp
    ├── chpr_npu_wrapper.h
    └── chpr_test.csv
```

仅在实际需要时增加本算子的精度辅助或 profiling 汇总文件。测试参数适配可放在 arch35 内，若改动公共参数头，兼容旧 x/ap 列名和 arch22 用例。按 arch35 源码收集方式构建，不增加重复的算子 CMake 工程，不修改公共 API。

## 支持硬件

| 支持的芯片版本 | 本任务范围 | 软件环境 |
| --- | --- | --- |
| Ascend 950PR / arch35 | 是 | CANN 9.1.0，配套驱动、固件与编译器 |
| A2/A3 / arch22 | 保留已有实现 | 本次不修改、不扩展其验证结论 |
| 其他产品 | 不扩展支持表 | 按各产品原有实现 |

## 算子约束限制

- Host alpha；Device x/ap；alpha 为实数，ap 为列主序 packed，接口没有 lda、beta 或 batch 参数。
- 不涉及广播、额外张量视图或超出 incx 的非连续布局；n 为运行时参数。
- ap 原地更新，x 只读；本次不扩展 x 与 ap 相互重叠的别名能力，测试使用独立操作数。
- 仅更新指定三角；输入对角虚部忽略，普通计算后置零，alpha=0 时完整保留输入。
- 不要求跨实现 bit-exact；每个输出仍采用唯一写者和固定的本地运算结构。
- 测试前记录真实设备型号、CANN 版本和选用设备；性能运行期间避免其他作业占用同一设备。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 环境 | Ascend 950PR，CANN 9.1.0；golden 使用 Netlib BLAS 3.10.0 的 `cblas_chpr` | 任务书 §3.1 |
| 常规精度 | packed 输出全量检查；实部/虚部分别按 FLOAT32：`atol=rtol=2^-13`，`abs(actual-golden)≤atol+rtol*abs(golden)`；元素匹配率≥0.99，并满足最大误差上限 `max(1e-2,32×ULP(g_low))` | 任务书 §3.2、指定 opbase 精度标准 §2.1.2 |
| 非有限值 | NaN/NaN、同号 Inf/Inf、NaN 与任一符号 Inf 按任务书等效规则通过；有限/非有限不匹配及异号 Inf 判失败。非有限值独立统计且全部符合规则 | 任务书 §3.2 |
| 工程精度配置 | 保留 CSV 的 `mere_threshold≈2^-13`、`mare_multiplier=10.0`，分别报告工程 MERE/MARE 与任务书 mixed tolerance | 随包测试 README |
| 性能 | 200 条逐 case：`ratio=T_gpu_us/T_npu_us≥0.4`，即 `T_npu_us≤2.5*T_gpu_us`；NPU 使用 msprof op 的 kernel task duration | 任务书 §3.3 |
| 内存 | 无独立内存验收指标；随包 Host 单用例≤512 MB 是用例生成预算。检查实际分配、安全访问和资源释放 | 任务书 §3.4、随包测试 README |

有限值与非有限值集合分别检查。有限值为空时不做除零或空集合最大值归约，只检查非有限值规则；总输出为空时按 quick return 验证。对角虚部、no-op 输入保持与返回码属于独立的严格功能检查，不使用 99% 元素匹配率放宽。

任务书引用的固定版本精度标准明确：表格中的 `or` 表示取较大者；`g_low` 是该实部或虚部分量集合中最大绝对误差点的 golden，按实际输出 FLOAT32 表示。统计时保存最大误差点的坐标、golden、ULP 和计算后的上限，使 32×ULP 判据可以复核。Inf/NaN 使用本任务书列出的专门规则。

### 性能基线与采样

随包 `gpu_baseline.csv` 共 200 行，`gpu_ms` 已填值。单位先从 ms 转为 us，再比较；基线来源按随包说明为 H100、CUDA 12.4，不在本设计阶段重新测量 GPU。

| 规模与模式，incx=1 | H100 基线 / us | NPU 平均耗时上限 / us |
| --- | --- | --- |
| n=512，UPPER | 11.909 | 29.7725 |
| n=1024，LOWER | 30.999 | 77.4975 |
| n=2048，UPPER | 96.491 | 241.2275 |
| n=4096，LOWER | 570.764 | 1426.9100 |

小尺寸的调优重点是单次下发开销，连续大尺寸的重点是 packed 读改写带宽和核间工作量。上述四点之外仍逐条检查全部 200 条性能用例，任何单条失败均单独报告。

采样材料有三处口径：任务书 §3.3 指定 5 次 warmup 后 10 次有效调用，§7 写有效调用超过 10 次，随包 README 写超过 50 次。计划保存 5 次 warmup 后至少 51 次有效调用的逐次原始记录，同时输出前 10 次与全量均值；最终采用哪组作为正式验收结果，提交验收前按评审口径确认。warmup 不计入样本；异常调用在报告中保留并记录原因，不以减少样本数掩盖失败。

测试必须区分 kernel、API 与端到端耗时。性能测试对每个 case 只在采样前设置输入、采样后校验输出；连续原地调用可能累积更新，应按固定重复次数检查最终输出，并在每组测量开始前恢复相同 ap。H2D 初始化、CBLAS golden 和结果比对不计入 kernel task duration。

随包脚本当前把 GTest 整项整数毫秒数作为 NPU 耗时，正式性能统计另行接入 `OpBasicInfo.csv` 的 `Task Duration(us)`。单用例采集并保存 kernel 名、case、样本序号及耗时的映射，原始脚本输出一并保留。

## 测试用例与验证计划

### 随包 1200 条用例

| 类别 | 前缀 | 数量 | 主要覆盖 |
| --- | --- | --- | --- |
| 基础 | TC_L0 | 4 | 两种 uplo，小尺寸 |
| 尺寸 | TC_SQ | 46 | n=1 到 2048，质数、非对齐、幂及附近尺寸 |
| 标量 | TC_AB | 16 | alpha 为 0、±1、小数、大值 |
| 步长 | TC_INC | 12 | incx 为 ±1/±2/±3 × 两种 uplo |
| 填充 | TC_FL | 12 | 随机、零、交替、极值、Inf、NaN |
| 覆盖 | TC_CV | 16 | 中等尺寸与 uplo 组合 |
| 边界 | TC_ED | 25 | n=0、alpha=0、空指针、非法枚举、n<0、incx=0/INT_MIN |
| 扩展精度 | TC_EX | 869 | 尺寸、标量、步长的固定种子组合 |
| 性能 | TC_PF | 200 | n=1 到 4096，两种 uplo，均为 incx=1 |
| 合计 | — | 1200 | 精度/功能 1000 条，性能 200 条 |

CSV 中共 11 条预期 `ACLBLAS_STATUS_INVALID_VALUE`，其余 1189 条预期成功。负向用例以返回正确状态码为通过；在计算长度、申请内存前先识别非法维度和步长，尤其不能对 INT_MIN 先做 `std::abs(int)`。

适配 `x_fill/ap_fill`、`alpha=null`、tensor `NULLPTR`；保持固定 case 名与随机种子。输入 ap 生成后，按实际 uplo 把对角虚部设为零。补充测试再注入非零或非有限对角虚部，检查实现确实忽略该输入。

### 补充测试

1. `handle=nullptr` 及其与 n=0、n<0 的组合，alpha=0 与各非法参数的组合；no-op 用字节级对照检查 ap 未改写。
2. n=1、两种 uplo、incx 大正值/大负值、x 非逻辑位置哨兵，核分段和 tile 边界前后一元素，ap 两端保护区。
3. 根据任务书补齐均匀/正态各 50% 的输入生成，实部虚部独立采样，保存分布参数与种子。现有 `--dist mixed` 仅预留开关，保留原 1200 条用例并另记补充覆盖。
4. x 零列与 alpha=Inf/NaN、其他 x 元素 Inf/NaN 的组合；对角虚部 NaN/Inf；确保零列跳过、对角忽略、alpha=0 三种语义分别覆盖。
5. 非默认 stream、重复调用、不同独立输入的并行 stream，结果只在同步后读回；核验输入 x 未修改。
6. 精度统计实部和虚部分开，记录匹配率、最大绝对误差、ULP 和非有限值通过数；同时保留 GTest 的工程精度结论。

补充测试独立命名、独立计数，不替换或删除随包用例。完整性检查要求实际执行数与选择的 CSV 清单一致，跳过和缺失单独列出；核对 GTest 进程退出码及逐条结果。随包精度脚本目前未强制数量完整性、性能脚本可把无参考记录纳入成功退出，需由测试汇总层补齐这些检查，保留原始输出。

### 构建与复现入口

计划使用仓库已有工程：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --soc=ascend950 --ops=chpr
./build/test/hpr/chpr/chpr_test --gtest_filter=-*TC_PF*
./build/test/hpr/chpr/chpr_test --gtest_filter=*TC_PF*
```

`set_env.sh` 使用真实 CANN 9.1.0 安装提供的入口；测试二进制位置以该构建实际产物为准。msprof 逐条收敛采集；输出中保留真实命令、源码提交、设备、软件版本、CSV 清单、逐样本耗时、逐用例判定和 GTest 原始日志。

实现阶段先验证基本 packed 数学与边界语义，再完整跑 1000 条功能/精度；性能调优后用同一最终源码和构建产物复跑 1200 条及补充测试。实现代码 README 与自测步骤随代码一并维护。

## 兼容性分析

本任务只新增 arch35 实现，复用已有 `aclblasChpr` ABI、handle 和 stream 接口。按 SOC 编译 arch22 或 arch35，避免两个实现同时定义同一符号；公共参数解析若需扩展，保持旧 CSV 和旧架构的默认行为。产品支持表仅在目标产品验证后更新。没有新增 workspace 参数、私有 950PR 接口或图模式算子接口。

## 待评审确认项

1. 性能有效采样次数的正式口径：保留 10 次与至少 51 次两套统计，按最终口径出具报告。
2. n 与空 handle 同时出现时的优先级：本稿按任务书 §2.5 的 n 优先顺序设计；通过复合异常单测固定确认后的行为。

## 参考资料

- 随任务包：`aclblasChpr_Atlas950PR_task_doc.md`、`test_cases/README.md`、`chpr_test.csv`、`gpu_baseline.csv`、两个验证脚本。
- [cuBLAS hpr 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpr)。
- [Netlib CHPR 参考实现](https://www.netlib.org/blas/chpr.f)。
- [任务书指定的 opbase mixed tolerance 标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
- [2026 社区任务目录与提交说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
- [社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)，2026-10-28 通过官方 API 读取正文。
