# 需求背景（required）

## 需求来源

本方案面向 **aclblasChpr2 算子开发（Atlas 950PR）**，依据任务书、配套测试和 ops-blas 工程编写。

| 来源 | 版本与用途 |
| --- | --- |
| 任务附件 | `6933b64564804b64902e0b26ec9fc2a9.zip`，SHA-256 `0df98fe0d51b28a68f2a744eb612409eeee1bd8cab5419d4644f3d1cbabd2d5c`；任务书与 `test_cases/` 为功能、测试和交付依据 |
| [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/72ca53083296fb2cc81c80e9cf4cf14bddbdaa20/04_tasks/01_community-task-2026/resources/design_template.md) | 模板最后修改 commit `72ca53083296fb2cc81c80e9cf4cf14bddbdaa20` |
| [ops-blas 源码](https://gitcode.com/cann/ops-blas/tree/05b96ea22cea926ecfd43fc7ac691a795522bd61) | 参考基点 `05b96ea22cea926ecfd43fc7ac691a795522bd61` |
| [固定版本精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) | 任务书 §3.2 指定版本，FLOAT32 混合容差和 ULP 上限定义 |
| [Netlib CHPR2](https://www.netlib.org/blas/chpr2.f) | packed 下标、负步长、零列分支和复数运算参考；正式 golden 按任务书绑定 Netlib BLAS 3.10.0 |

任务讨论：[aclblasChpr2 算子开发（950）](https://gitcode.com/cann/ops-blas/discussions/54)。

## 背景介绍

Chpr2 将两个复数向量的 Hermitian 秩 2 更新直接作用于 packed 三角存储，补齐 ops-blas 的 BLAS Level-2 能力。调用方不需要展开成稠密矩阵，也不需要额外提供 lda。

工程接入复用 `blas/spr2/arch35/sspr2_*` 的句柄、stream 和直接启动模式；复数 SIMD 读写参考 `blas/gerc/arch35/cgerc_kernel.cpp`，packed 索引参考 `hpr/arch22/chpr_*`。本次新增公共 `aclblasChpr2` 声明、arch35 实现及配套测试。

# 需求分析（required）

## 需求描述

在 Ascend 950PR、CANN 9.1.0 上，以 Ascend C kernel 直调方式实现单精度复数 Hermitian packed 秩 2 更新：

\[
A \leftarrow A + \alpha x y^H + \overline{\alpha} y x^H.
\]

接口新增到 `include/cann_ops_blas.h`，所有产品线共用该声明：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy,
    aclblasComplex* ap);
```

`aclblasComplex` 复用 `include/cann_ops_blas_common.h`，实部、虚部均为 float32。实现放在任务指定的 `blas/spr2/arch35/`，测试放在 `test/spr2/chpr2/arch35/`，采用统一的句柄式 BLAS API。

## 需求拆解

| 编号 | 要求 | 设计落点 | 原始来源 |
| --- | --- | --- | --- |
| F01 | complex64，复数 alpha，Host 标量指针 | Host 快照 alpha 两个 float，Device 以 FP32 复数展开计算 | 任务书 §2.1、§2.3–2.5 |
| F02 | UPPER/LOWER 按列 packed，ap 原地更新，无 lda | 连续 packed 分片；精确列区间定位 | §2.1.2、§2.4 |
| F03 | x/y 任意非零合法步长，支持负值，排除 INT_MIN | 64 位逻辑到物理下标；有界搬运及重排 | §2.4、§2.5 |
| F04 | 固定错误检查顺序与状态码；合法 no-op 不读写 Device 数据 | API 早返回 + Validate + Launch 分层 | §2.5 |
| F05 | 正常更新忽略对角输入虚部，输出虚部为 0 | 对角单独计算及精确零写入；no-op 优先 | §2.1.3、§2.1.5 |
| F06 | handle stream 异步启动 | 同一 stream 单 kernel，无每调用同步或 Device 标量回读 | §2.2、§2.5 |
| V01 | Netlib 3.10.0 cblas golden，packed 全量，实/虚分别判定 | 自建 Chpr2 GTest、固定参考库、混合容差比较器 | §3.1、§3.2、§3.5 |
| V02 | 覆盖全部附带用例及原文边界/分布 | 1000 条精度、200 条性能；补齐分布和接口单测 | §3.5、§4；测试 README |
| P01 | 每条平均 kernel 耗时 ≤ 对应 GPU×2.5 | msprof op task duration，逐 case 汇总 | §3.3 |
| D01 | README、测试步骤、自测报告、同版代码入口 | README、测试、报告和源码版本关联 | §4、§5 |

## 输入输出与边界

令 `P=n(n+1)/2`，`Lx=1+(n-1)|incx|`，`Ly=1+(n-1)|incy|`（后两式用于 n>0）。

| 参数 | 存储与用途 | 约束 |
| --- | --- | --- |
| handle | Host 上下文，含 stream | nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | Host 枚举 | UPPER/LOWER；非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| n | Host int | n<0 非法；n=0 合法早返回 |
| alpha | Host complex64 标量 | n>0 时指针非空；零值在其他参数检查后早返回 |
| x / y | Device，只读，分别至少 Lx/Ly 个复数 | 任意非零正/负步长，排除 INT_MIN |
| ap | Device，P 个复数，原地读写 | 保证实际分配足够；只存储并更新指定三角 |

n 和 inc 为运行时参数，当前测试覆盖至 n=4096。接口范围为 Host alpha、packed ap 和由 inc 描述的向量布局。任务书未定义 x/y 与 ap 重叠的语义；x 与 y 可共用同一只读向量。

# 详细设计（required）

## 算子分析

### packed 和步长映射

0-based 列起点与列长为：

\[
C_U(j)=j(j+1)/2,\quad \ell_U(j)=j+1;
\qquad C_L(j)=j(2n-j+1)/2,\quad \ell_L(j)=n-j.
\]

UPPER 的 `k=C_U(j)+i`；LOWER 的 `k=C_L(j)+(i-j)`。后者等价于任务书的 `i+(2n-j-1)j/2`。两种情况下对 `[0,P)` 均为无间隙、一一对应的分区。

逻辑第 i 项的物理索引：

\[
s(i,inc)=\begin{cases}i\,inc,&inc>0\\(n-1-i)(-inc),&inc<0.\end{cases}
\]

调用指针指向物理分配起点，负步长从逻辑首元素对应的物理偏移开始。排除 INT_MIN 后，将 n、inc 提升至 int64，再计算 n+1、2n 和乘积；元素数、packed 下标使用 uint64。字节跨度转换为 size_t 前检查可表示性，DMA 长度按本地 tile 分段。

### 复数计算与特殊值

按 Netlib 参考路径，列 j 的系数：

```text
t1 = alpha * conj(y[j])
t2 = conj(alpha * x[j])
p  = x[i] * t1
q  = y[i] * t2
```

若 alpha=(ar,ai)，列值为 xj=(xr,xi)、yj=(yr,yi)：

```text
t1.real = ar*yr + ai*yi       t1.imag = ai*yr - ar*yi
t2.real = ar*xr - ai*xi       t2.imag = -(ar*xi + ai*xr)
```

每个复数乘法采用 `real=u.r*v.r-u.i*v.i`、`imag=u.r*v.i+u.i*v.r`。非对角输出按 `(ap+p)+q` 形成；对角按 `ap.real + (p.real+q.real)` 形成，并令 imag=+0。对角计算仅读取输入实部，输入虚部中的 NaN/Inf 同样被忽略。

保留 Netlib 的零列分支：当 x[j]、y[j] 两个复数都精确为零时，不计算 t1/t2、不改变该列非对角元素，只清对角虚部。否则 `0×Inf` 可能把参考实现保持的有限值变成 NaN。alpha=0 的整算子 no-op 则完全不更新 ap，包括对角虚部。

初版采用显式 FP32 Mul/Add/Sub，保留上述乘法、加法顺序和舍入边界。编译配置与生成指令共同控制浮点融合；后续融合或系数复用优化按混合容差和特殊值用例复核。

## 算子实现

### Host 设计

API 仅处理调用级顺序，校验与启动分别收敛到 `ValidateChpr2Params`、`LaunchChpr2Kernel`：

```text
if handle == nullptr: HANDLE_IS_NULLPTR
if n < 0: INVALID_VALUE
if n == 0: SUCCESS
ValidateChpr2Params:
    alpha 指针 → uplo → incx/incy 非零且非 INT_MIN → x/y/ap 非空
alphaValue = *alpha
if alphaValue.real == 0 and alphaValue.imag == 0: SUCCESS
LaunchChpr2Kernel(handle.stream, alphaValue, ...)
```

n=0 直接返回；alpha=0 在参数检查完成后返回。Host 将 alpha 值复制进 tiling 参数，调用返回后 alpha 本地变量即可结束生命周期；x/y/ap 须保持有效直到 stream 上本调用完成。

launch 使用句柄 stream，通过仓内 `GetAivCoreCount()` 等平台入口查询可用核数和资源；查询失败沿用仓内失败返回约定。基础路线为单 kernel 异步启动，额外 GM workspace 为零。调用方在 stream 同步时检查设备执行状态并读回结果。

tiling 最小字段：n、uplo、alpha 实/虚、int64 incx/incy、uint64 P、uint64 tileCount、tileElems、blockCount、实际启用的向量缓存模式。tile 循环变量 q 及区间乘积均用 uint64，防止大 n 下 tileCount 截断；tileElems、blockCount 则按实际资源范围使用定宽字段。上下三角在 kernel 分派后使用独立模板/分支，减少逐元素条件判断。

### 分核与分块

主线为 **SIMD regbase + packed 连续 tile**。按 packed 元素数均衡分核，在 tile 起点定位列，再按列片段计算，以控制三角列长差异和索引开销。

1. tile 长 T 是实际搬运/向量要求的公倍数，以复数元素计。初始调优集合为 256/512/1024。
2. `Q=ceil(P/T)`，`B=min(实际可用 AIV 数,Q)`。小规模采用 B=1，阈值由本算子 profiling 确定。
3. 第 b 核处理 `q=b,b+B,...<Q`；拥有 `[q*T,min((q+1)*T,P))`。不同核写区间不重叠，负载相差至多一个 tile，无 atomic、无核间归约。
4. 对 tile 起点 k，用整数二分查找最大满足 `C(j)≤k` 的 j；列起点和哨兵 C(n)=P 均用 64 位计算，确保索引计算精确。
5. 令本段 `m=min(tileEnd-k,C(j+1)-k)`，由 k 得到 i 的起点，在 UB 中更新该列的 m 个连续元素；走到列尾后 j++。一个 tile 可以跨多列。

该映射把每个 packed 元素分配给唯一的核、唯一的 tile 和唯一的列片段，对角元素也由所属片段处理。UPPER 对角 k=C_U(j)+j，LOWER 对角 k=C_L(j)。整 tile 搬出，避免每个短列片段分别做重叠的对齐写回。

```mermaid
flowchart LR
    H[Host 顺序校验与 alpha 快照] --> T[计算 packed tile 与分核]
    T --> K[在 handle stream 启动单 kernel]
    K --> A[搬入所属 ap tile]
    A --> C[定位列片段并读取 x/y]
    C --> V[FP32 SIMD 复数更新]
    V --> D[处理零列与对角]
    D --> O[精确写回所属 ap 区间]
```

### Device 搬运与 SIMD 计算

GM 保留 `aclblasComplex` 的 AoS `[real,imag,...]` 表示，UB 暂存 AoS。SIMD 从 UB 载入时解交错为实、虚寄存器，计算后交错写回；原型参考同架构 `cgerc_kernel.cpp` 的 `MicroAPI::DataCopy` 与 `LoadDist::DIST_DINTLV_B32` / `StoreDist::DIST_INTLV_B32`。

- **ap**：每 tile 读写精确 `8*validElems` 字节，使用支持非对齐长度的 DataCopyPad 类入口。UB 按完整向量读所需范围分配并初始化补齐位置；尾向量用 mask 约束写回，GM 搬运仅覆盖有效区间。
- **x/y 单位正步长**：列片段行区间连续搬入；列 j 的 x/y 值用于 t1/t2，列内广播。若该核缓存了完整逻辑向量，则从缓存取值。
- **一般步长**：根据物理索引构造有界连续包络，搬入后在 UB 中抽取每 abs(inc) 项；负步长同时逆转逻辑顺序。包络大小为 `8*(1+(m-1)*abs(inc))`，受 scratch 预算约束，超过预算时减少 m。任意合法大步长可退到每段 m=1 的精确元素搬运。
- **重排 API**：UB 内 Gather/置换的目标版本签名、索引宽度和对齐需最小探针验证；超出硬件 gather 索引范围时分更小段。先保证有界逐元素搬入/装配可以表达任意步长，再为小 abs(inc) 启用批量路径。该步骤属于搬运装配，主算术仍用 SIMD。
- **列边界**：tile 内片段起点可能不是寄存器对齐边界。给 SIMD 的 x/y/ap 片段使用对齐 UB scratch，并在 tile 缓冲与片段 scratch 之间按有效长度拷贝。
- **对角**：同一核用独立算式覆盖 real，并写 imag=0；对角计算在所属片段内完成。

流水初版为 CopyIn → 完成等待 → Compute → 完成等待 → CopyOut → 复用缓冲前等待。使用 MTE2→Vector、Vector→MTE3 和标量访问 UB 所需的事件，保证搬运、标量与向量访问之间的可见性。双缓冲仅在正确性确认后加入，等待每个 buffer 的上次写出完成才覆盖。跨核只读共享 x/y，写 ap 区间互斥，无全局 barrier。

### 内存与资源模型

输入/输出 GM 占用为 `8*(P+Lx+Ly)` 字节，额外 GM workspace 为零。Host 使用 O(1) 的 tiling 与系数快照。

设 A(z) 为按 UB 对齐取整的字节数，T 为 tile 元素数，S≤T 为片段元素数，V 为一次解交错向量读取所覆盖的复数元素数，`Sv=ceil(S/V)*V`。E 为每个 stride 包络元素预算，I 为重排索引字节数，R 为平台/API 所需预留空间。保守单缓冲预算可写为：

\[
U=A(8T)+3A(8S_v)+2A(8E)+A(I)+U_{xy}+R.
\]

各项依次为 ap tile、x/y/ap 片段 scratch、x/y 原始包络、重排索引、可选完整逻辑向量缓存和预留。三份 scratch 均包含完整向量读取所需的 padding。`U_xy=0` 或 `2*A(8n)`，缓存仅存逻辑 n 项，计算经已补齐的片段 scratch 读取。R 列明 API 额外临时 UB，寄存器驻留数据单独计入寄存器资源；双缓冲按实际重复项重新计算预算。

Host 根据实际 UB 可用量选择 T/S/E，并在预算内启用 x/y 缓存。n=4096 时两条逻辑向量共 65,536 字节；超过缓存预算时使用片段模式。

### 性能分析与优化

ap 必须读一次写一次，最低矩阵流量为 16P 字节。未缓存行向量的基础路径约再读 16P 字节 x/y，加上列系数读取、短片段和尾部开销。

性能包均为 incx=incy=1、alpha=(1,0)。通用路径的优化顺序为：小规模启动核数 → packed tile 大小 → 预算内只读 x/y 复用 → 搬运与计算重叠。每轮控制一个主要变量，记录对应版本的完整结果。

单 kernel 路线将本次更新所需计算包含在一个 profiling task 中。

### 工程接入

| 路径 | 计划内容 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增统一声明；不改既有类型、枚举、结构体布局 |
| `blas/spr2/arch35/chpr2_host.cpp` | Validate/Launch、Host alpha、tiling、stream 启动与 OP_LOG 日志 |
| `blas/spr2/arch35/chpr2_kernel.h` | Host/Device 共用 kernel_do 声明 |
| `blas/spr2/arch35/chpr2_tiling_data.h` | 固定宽度 tiling 字段和资源参数 |
| `blas/spr2/arch35/chpr2_kernel.cpp` | SIMD packed 分块、复数计算、尾部和一般步长 |
| `blas/spr2/README.md` | 增补 Chpr2 参数、调用示例、产品支持情况；保留已有 Sspr2 内容 |
| `test/spr2/chpr2/CMakeLists.txt` | 注册 chpr2_test，沿用现有测试发现机制 |
| `test/spr2/chpr2/chpr2_param.h` / `chpr2_golden.h` | CSV 装配和指定 cblas golden |
| `test/spr2/chpr2/arch35/` | wrapper、GTest、原始 CSV/基线、接口边界与性能采集入口 |

沿用仓内 `chpr2_` 文件前缀收集机制和 `ops_blas_add_gtest_tests` 测试注册入口。测试 README 提供构建后实际二进制路径与复现命令。

## 支持硬件

| 项目 | 目标配置 |
| --- | --- |
| 硬件 | Ascend 950PR / arch35 |
| SDK | CANN 9.1.0 |
| 精度参考库 | Netlib BLAS 3.10.0 |

编译使用目标环境对应的 `ascend950...` SoC 参数；核数和 UB 容量由平台接口查询。

## 算子约束限制

本任务支持 complex64、UPPER/LOWER packed、Host alpha 和原地 ap；inc 以复数元素为单位。仅引用和更新已存储三角，额外 workspace 为零。调用者负责分配容量、Device 指针有效性、stream 生命周期和读回同步。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 本任务口径 | 标准来源 |
| --- | --- | --- |
| 精度 golden | Netlib BLAS 3.10.0 的 `cblas_chpr2`，packed 全量、实/虚分别比较 | 任务书 §3.1、§3.2 |
| 有限值 | atol=rtol=2^-13；matched_ratio≥0.99，且最大绝对误差≤max(1e-2,32×ULP(g_low)) | 任务书 §3.2；固定 opbase 标准 §2.1.2、§2.2 |
| 非有限值 | 同 NaN、同号 Inf、NaN 与任意符号 Inf 等效；有限/非有限及异号 Inf 不通过；不计入有限值统计，全部非有限点单独检查 | 任务书 §3.2 特殊规则 |
| 对角与 no-op | 普通更新输出对角虚部为零；合法 no-op 完全保持 ap | 任务书 §2.1、§2.5 |
| 性能 | 每 case `mean_us ≤ 1000*gpu_ms/0.4`，msprof op task duration | 任务书 §3.3；原始 gpu_baseline.csv |
| 内存 | 无独立内存验收指标；验证访问不越界、资源生命周期正确 | 任务书 §3.4 |

`g_low` 为最大误差点 golden 的 FLOAT32 值，ULP 按固定标准定义。比较器记录最大误差位置、ULP、动态上限和非有限点计数；没有有限点时有限部分记 N/A，由非有限规则决定。CSV 的 MERE/MARE 配置独立保留为框架诊断信息。

### 正确性与自验计划

1. 按 CSV 完整执行 1000 条精度用例，另行采集 200 条性能用例；case 名、源行、参数与结果一一对应。负向用例按预期错误码判定。
2. GTest 对非法维度、非法步长和空指针直接验证 API 返回，合法输入按物理布局分配。覆盖空 handle、校验优先级及 no-op；普通结果同步 stream 后读回。
3. golden 和 NPU 使用相同的初始 x/y/ap/alpha。正负 inc 均传入物理分配首地址；通过 `REFBLAS_LIB` / `REFBLAS_INCLUDE_DIR` 显式绑定 Netlib 3.10.0，并记录实际链接库。
4. 在原始用例之外补齐均匀/正态各 50% 和随机复数 alpha，实部、虚部独立采样，以固定 seed 复现。保留已有 GPU 基线。
5. 补充针对性用例：空 handle 与 n=0，alpha=0 与 null y，负步长，packed 三角边界，非零/非有限对角输入虚部，零列混合其他位置 Inf/NaN，输入只读和尾部哨兵。同一 stream 连续调用及两个独立 stream 的独立输出验证资源隔离。
6. 运行器检查子进程退出码、超时、预期 case 集合和实际终态集合。原始 Python 包装器中的结果解析与超时处理由测试工程完善。

### 性能采集

逐 case profiling，绑定 `case_id → API 调用 → kernel task`，以 `Task Duration(us)` 作为有效样本。记录工具自动 warmup 行为及有效样本边界，计算算术平均；Host 分配、数据传输、输入恢复、golden 和同步墙钟单独记录。

重复 in-place 调用前，将相同初始 ap 恢复至工作 buffer；恢复与计算在同一 stream 有序执行，profiling 提取对应计算 task。各候选使用相同输入管理方式，报告记录其缓存影响。失败调用保留错误记录，并按任务书从有效样本中排除。

全部 200 个性能 case 分别关联原始 GPU 基线行，包含参数键重复的用例。`gpu_ms` 统一乘 1000 转为 μs，再按 2.5 倍上限逐条判断。

**采样次数待确认：** 任务书 §3.3 为 warmup 5 次后有效采样 10 次，§7.4 为 >10 次，测试 README 为 >50 次。采集器将样本数参数化，最终采用任务方确认的口径。

### 验证安排与当前进展

| 验证对象 | 方法 |
| --- | --- |
| AoS SIMD 与 FP32 计算 | CANN 9.1.0 编译；n=1/3/33 用例比较实部、虚部和对角结果 |
| 非对齐与尾部 | 精确长度 GM 搬运、UB 对齐及 padding 检查，分配边界哨兵验证 |
| 负步长与大 stride | ±1/±2/±3 和较大步长，核对 Gather/置换签名、索引及 stride 字段范围 |
| 资源与同步 | 查询实际 UB/核数，检查缓冲预算、标量/向量可见性与 stream 隔离 |
| 特殊值 | 零列、Inf/NaN、非零对角虚部以及 alpha=0 路径 |
| 性能 | 950PR 单负载环境下执行全部 200 条性能用例 |

设计阶段的 CPU 模型已通过 1680 组布局、90 个大索引点、504 组有限值差分和 4 项语义检查。CANN 编译、NPU 精度与性能验证将在实现阶段执行。

## 兼容性分析

公共头增加统一函数声明，保持现有 ABI、枚举和 `aclblasComplex` 布局。实现文件以 chpr2 前缀隔离，与同目录 Sspr2 共存；README 增补 950PR 产品支持项。

每次调用使用独立 tiling 和核内缓冲，不同 handle/stream 的独立输出可以并发。接口保持任务书规定的参数顺序和错误返回优先级。原始测试数据与运行输出分开保存；共用测试模块的修改按影响范围执行回归。
