# aclblasChpr 算子设计文档（Atlas 950PR）

# 需求背景（required）

## 需求来源

- [任务广场](https://www.hiascend.com/activities/task-center/details/f783b3800ad744ef8af7f50ba237de04?menu=tasks)及其附件《aclblasChpr 算子开发任务书》、`test_cases/README.md`；[任务讨论帖](https://gitcode.com/cann/ops-blas/discussions/53)。
- 工程参考：[ops-blas 固定版本](https://gitcode.com/cann/ops-blas/tree/05b96ea22cea926ecfd43fc7ac691a795522bd61)。
- 数学参考：[Netlib BLAS 3.10.0 CHPR](https://github.com/Reference-LAPACK/lapack/blob/aa631b4b4bd13f6ae2dbab9ae9da209e1e05b0fc/BLAS/SRC/chpr.f)。
- 精度标准：[任务指定的混合容差规范](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
- 文档格式：[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/e1ab79ca0d495afce93dfb1e85f95f546b74d2cf/04_tasks/01_community-task-2026/resources/design_template.md)。

## 背景介绍

`aclblasChpr` 对单精度复数向量执行 Hermitian 秩 1 更新，并原地累加到三角 packed 矩阵。参考版本已有公共 API 和 `blas/hpr/arch22/` 实现；本任务补齐 Atlas 950PR 的 `blas/hpr/arch35/` 实现与 `test/hpr/chpr/arch35/` 测试，沿用公共复数类型、handle、stream 和构建框架。

packed 存储按列压缩三角矩阵，列长随位置变化。本方案按连续输出区间分块，使多核写入互不重叠，再在块内按列组织计算，兼顾负载均衡与向量复用。

# 需求分析（required）

## 需求描述

面向 Ascend 950PR、CANN 9.1.0，以 Ascend C kernel 直调方式实现现有接口：

```cpp
aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* ap);
```

| 参数 | 类型与位置 | 语义 |
|---|---|---|
| handle | Host 句柄 | 使用已绑定的 stream |
| uplo | Host 枚举 | UPPER 或 LOWER，指定 packed 三角 |
| n | Host int | 矩阵阶数，n≥0 |
| alpha | Host float32 指针 | 实数标量，支持 FLOAT32 全集 |
| x | Device complex64，只读 | n>0 时逻辑长度 n，物理长度至少 `1+(n-1)×abs(incx)` |
| incx | Host int | 非零正/负步长，INT_MIN 非法 |
| ap | Device complex64，原地读写 | packed 长度 `M=n(n+1)/2` |

## 需求拆解

| 开发内容 | 设计目标 | 依据 |
|---|---|---|
| 工程接入 | 复用公共 API，在 hpr/arch35 实现 | 任务书 §2.2–2.3、§5 |
| 接口与地址 | 参数校验、no-op、正负步长、宽整数地址 | 任务书 §2.4–2.5 |
| 数值计算 | Hermitian packed 更新、共轭与对角处理 | 任务书 §2.1；Netlib CHPR |
| 精度验证 | Netlib 3.10.0 golden，实部/虚部分别全量检查 | 任务书 §3.1–3.2 |
| 性能验证 | 每 case 平均耗时≤GPU 基线的 2.5 倍 | 任务书 §3.3 |
| 测试与文档 | 全部附带用例、输入分布、复现步骤和自测报告 | 任务书 §3.5、§4 |

# 详细设计（required）

## 算子分析

### 数学公式

对所存三角执行：`A(i,j) ← A(i,j) + alpha×x_i×conj(x_j)`。

采用 Netlib 的计算次序，先得到 `t=alpha×conj(x_j)`，再乘 `x_i`。设 `x_i=(a,b)`、`x_j=(c,d)`：

```text
tr = alpha * c
ti = alpha * (-d)
delta_real = a*tr - b*ti
delta_imag = a*ti + b*tr
out_real = old_real + delta_real
out_imag = old_imag + delta_imag     # 非对角
out_imag = 0                       # 对角
```

当 `x_j.real==0 && x_j.imag==0` 时，跳过该列非对角更新，对角保留旧实部并清零虚部；有符号零按零处理，NaN 不属于零。合法 `alpha=0` 在 Host 返回，整个 ap 保持原值。

运算次序影响极端值：float32 的 `alpha=2^-100、x_j=2^100+0i` 先乘 alpha 得到有限结果，先平方会溢出；零列中的 `0×Inf` 也可能改变应保留的 ap。初版显式分开乘法和加减，FMA 或表达式重排优化需核对这些数值行为。

### 支持数据类型

alpha 为 float32 实数；x/ap 采用仓内 `aclblasComplex { float real; float imag; }`，每元素 8 字节。GM 使用实虚交织的 AoS 布局，AIV 路径在 UB 中拆成实虚数组计算，再合并输出。实现以静态断言核对复数布局。

### 支持形状

令 `C(j)` 为第 j 列基址，`C(n)=M`。对 `0≤k<M` 找到唯一 `C(j)≤k<C(j+1)`：

| 布局 | C(j) | 行号 i | 对角位置 |
|---|---|---|---|
| UPPER | `j(j+1)/2` | `k-C(j)` | `C(j)+j` |
| LOWER | `j(2n-j+1)/2` | `j+k-C(j)` | `C(j)` |

初版以整数二分定位列，tile 内后续列递增推进。n、M、列乘积、tile 编号和元素偏移使用 64 位整数；字节长度采用受检转换。附件最大 n=4096 是测试范围，接口按合法 n 与实际缓冲区容量处理。

x 的逻辑元素 i 对应物理偏移：incx>0 时为 `i×incx`，incx<0 时为 `(n-1-i)×abs(incx)`；incx 先扩展到 int64 再取绝对值。

## 算子实现

### 实现方案

采用**单次 kernel 启动、连续 packed tile 唯一输出归属、零额外 GM workspace**。同一入口根据尺寸、步长和能力条件选择一条路径：通用 SIMT 覆盖所有合法步长与小规模输入；incx=1 的大规模场景候选使用 AIV 向量路径。T、核数和切换阈值在设备实测后确定。

#### 3.2.1 host侧设计：

有效 handle 下按任务书 §2.5 执行以下顺序：

| 顺序 | 条件 | 行为 |
|---|---|---|
| 1 | n<0 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 2 | n=0 | 返回 `ACLBLAS_STATUS_SUCCESS`，ap 不变 |
| 3 | n>0 | 依次校验 uplo、incx、alpha/x/ap；非法返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | 参数合法 | 读取 Host alpha；alpha=±0 时返回 SUCCESS，ap 不变 |
| 5 | 有效更新 | 计算 tiling，在 handle 绑定的 stream 启动 kernel |

空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；其与其他条件组合时的优先级见文末评阅事项。指针校验先于解引用，步长校验先于地址计算，no-op 先于设备资源查询。

复用 `GetAivCoreCount()`，通过 `GetCoreMemSize` 查询 UB。根据预算选定 tile 长度 T，计算 `Q=ceil(M/T)`、`B=min(可用AIV核数,Q)`；小问题可仅启动一个 block。资源查询和启动异常按仓内错误约定处理。

n、M、uplo、int64 incx、alpha、T、模式等组成小型 tiling struct，沿用现有 wrapper 按值传递，在指定 stream 启动一次 kernel。正常调用保持异步；API 返回状态与后续 stream 同步状态由测试分别检查。

#### 3.2.2 kernel侧设计：

**1. 输出分核。** block b 处理 `q=b,b+B,b+2B,…<Q` 的 tile，输出区间为 `[qT,min((q+1)T,M))`。T 为实际向量和搬运粒度的公倍数，各 tile 完整覆盖 `[0,M)` 且互不重叠，每个输出元素只有一个写入者。

**2. 通用 SIMT。** 采用仓内 `asc_vf_call` 路线，线程分担 tile 内不同 k，经整数索引恢复 i/j，读取正负步长对应的 x，再执行 Netlib 数值分支。每个线程独立更新自己的复数元素，无需原子加或跨核归约。此路径同时承接向量路径未覆盖的地址组合。

**3. AIV 向量路径。** 仅用于 incx=1，按以下步骤处理一个 tile：

1. 设有效长度为 valid，向量粒度为 G，UB 长度 `P=ceil(valid/G)×G≤T`。精确搬入 valid 个 ap 元素，并初始化 UB 填充区；以对齐长度 P 将 AoS 拆为 apReal/apImag。
2. 定位起始列 j，把 tile 按列切段。每段复用一个 x_j 及 `t=alpha×conj(x_j)`；x_j 为零时只处理该段可能包含的对角。
3. 对非零列，设段在 apReal 中的偏移为 r、长度为 L，划分 `head=min(L,(-r) mod G)`、`body=floor((L-head)/G)×G`、`tail=L-head-body`。ap body 从 `r+head` 开始，满足对齐。
4. 将 body 对应的逻辑 x 区间单独搬到 xAoS scratch 的 **0 偏移**，再拆到 xReal/xImag 的 **0 偏移**，保证向量运算两侧均对齐。head/tail 各自读取所需 x_i，在 ap 的 UB 缓冲内有界标量计算。候选 G=8，对应 float 数组的 32 字节对齐；最终取值同时满足所用 API 的要求。
5. 对角实部沿用同一 t 运算次序，虚部写零。向量与标量访问同一 UB 缓冲时显式同步。
6. 全 tile 完成后按 P 合并输出；拆合 API 所需的第二个 UB 输出视图从 **P 个 float** 偏移处开始。GM 只写回 **8×valid 字节**。内部列段统一在 UB 完成，整 tile 写回，避免段间扩大写入相邻输出。

精确搬运参考 [ccopy](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/copy/arch35/ccopy_kernel.cpp) 的 `DataCopyPad`，AoS 拆合和单入口派发参考 [caxpy](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/axpy/arch35/caxpy_kernel.cpp)。任意 GM 起点、尾部传输及 UB 视图组合需在 CANN 9.1.0 上确认支持。MTE2→计算→MTE3→缓冲复用之间建立明确同步。

**4. UB 预算。** 初版采用单缓冲：

| 缓冲 | 字节数 |
|---|---:|
| ap AoS，搬入后复用为输出 | 8T |
| apReal、apImag | 8T |
| x AoS scratch | 8T |
| xReal、xImag | 8T |
| 两个 float32 乘法暂存数组 | 8T |
| API 工作区、对齐及编译资源余量 | S |

峰值预算为 `40T+S`。候选 T=1024/2048/4096 的显式部分分别为 40/80/160 KiB；按平台 UB 与编译资源选择，后续增加双缓冲时重新计算峰值。GM 无额外工作区，有效 ap 读写流量约 `16M` 字节；n=4096 时约 128 MiB。

**5. 调优顺序。** 先落实通用路径与向量路径的数值、地址和同步，再比较 T、B 和模式阈值。大规模关注连续访存和负载均衡，小规模关注启动及列映射开销。依据实测考虑 x 局部复用、减少二分和双缓冲，最终配置以全部性能 case 逐项验证。

#### 工程组织

实现放在 `blas/hpr/arch35/`，包含 Host、kernel、wrapper 声明和 tiling 数据；测试在 `test/hpr/chpr/arch35/` 接入现有 GTest 注册。架构由仓内 CMake 选择，源码按现有规则收集。

参数读取兼容附件的 `x_fill/ap_fill` 与旧 `x/ap` 字段，以及 alpha 的 `null/NULLPTR` 表达；冲突字段和非法数值返回装配错误。负向测试在容量计算之前分流，避免 INT_MIN 或负 n 引发测试框架溢出。golden 复用数学调用，状态期望按任务条款判定。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Atlas 950PR（arch35） | √ |

目标软件环境为 CANN 9.1.0。

## 算子约束限制

- alpha 为 Host float32；x/ap 为 Device complex64。x 只读，ap 原地更新，调用方提供足够长度并保持异步执行期间缓冲区有效。
- n≥0，incx≠0 且 incx≠INT_MIN，uplo 为 UPPER/LOWER；n=0 和校验通过的 alpha=0 按 no-op 处理。
- 仅处理指定三角 packed 存储，无 lda、广播或 incx 之外的通用非连续布局。x/ap 按独立缓冲区设计。
- 非零 alpha 的有效更新将对角虚部置零。同一 ap 的跨 stream 访问由调用方建立依赖。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_chpr`，实部/虚部分别全量比对 | 任务书 §3.1–3.2 |
| 精度 | atol=rtol=2^-13；每分量有限值匹配率≥0.99；最大绝对误差≤`max(1e-2,32×ULP(g_low))` | 任务书 §3.2；固定精度规范 §2.1.2、§2.2 |
| 性能 | 每 case 的 NPU 平均 task duration≤GPU 基线耗时的 2.5 倍 | 任务书 §3.3 |
| 内存 | 无单独容量验收项；检查有效地址范围与 UB 预算 | 任务书 §3.4 |

`g_low` 是该分量最大绝对误差点的 FLOAT32 golden。比较器以 double 计算原 float32 值之间的误差，逐点条件为 `abs(actual-golden)≤2^-13+2^-13×abs(golden)`。ULP 按 binary32 指数计算，零/次正规为 2^-149；并列最大误差暂取 packed 顺序首点并记录锚点。

非有限值单独统计：NaN/NaN、同号 Inf/Inf、NaN/任意符号 Inf 按任务规则等效；有限/非有限、反号 Inf/Inf 不等效。非有限位置须全部通过，且不计入有限值分母。无有限位置时有限统计记 N/A。MERE/MARE 作为测试工程统计分栏记录。

### 测试方案

保留原始 **1000 条精度/边界用例、200 条性能用例**及其参数、期望状态和基线。测试显式绑定 Netlib 3.10.0；从同一 ap0 生成 golden 与 NPU 输入，负步长按原基址和 incx 传给 Netlib。

重点覆盖：

- 上下三角、尺寸边界、正负步长、非法参数及 no-op；n=0/alpha=0 检查 ap 字节不变，正常计算检查对角虚部为零、x 及步长间隙不变。
- 零列叠加 Inf/NaN、非有限 alpha、先乘 alpha 与先平方的溢出差异；两条 kernel 路径分别对比同一 golden。
- 跨列 tile、短段、两侧 UB 对齐、最后尾块、多核唯一归属。保护区检查越界写，索引和搬运范围检查结合可用设备诊断核对越界读。
- 普通随机集合按相同参数骨架成对生成均匀/正态输入。alpha 及 x/ap 的实虚分量均匀范围为 [-5,5]，正态参数为 μ∈[-5,5]、σ∈[0.1,2]；各输入及实虚分量独立采样，最后清零 ap 对角虚部。记录分布参数、种子和实际输入；正态默认不截断，特殊输入单列。

测试结果包含实际执行清单、进程退出码、real/imag 指标、非有限统计和失败定位。缺失/重复 case、零执行、超时、解析失败及缺基线均保留独立状态；运行成功要求选定用例完整完成且相应判断通过。

### 性能采集方案

采用 msprof op 的 task duration，warmup 5 次，有效样本数按评阅结论配置。每次调用前在同一 stream 恢复 ap0，记录调用序号并关联计算 task；恢复、拷贝、golden 和 warmup 在计算样本之外。采集适配需确认 profiler 自动重放边界及逐次恢复能力，warmup 由一个明确控制者执行。

有效样本要求 API 返回 SUCCESS 且同步无错误；异常保留原因并按有界策略补样。原始 duration 全部留存，以算术平均作比较：`npu_avg_us≤gpu_ms×1000/0.4`。保留 200 行基线 identity，包括参数相同的重复行，使用未舍入值逐行判定。

报告绑定同一源码和测试版本、环境、编译参数、输入/基线身份、命令及原始结果，给出实虚精度结果和逐 case 性能表。

### 评阅事项

| 事项 | 依据与待明确内容 | 当前方案及影响 |
|---|---|---|
| 有效采样次数 | 任务书 §3.3 为 10 次，§7 第 4 条为 >10 次，测试说明“性能验收”为 >50 次；请确认统一次数 | 采样器参数化，影响正式采样配置与报告 |
| 空 handle 的组合优先级 | 任务书 §2.1、§2.4–2.5 分别规定 n=0 quick return 与空 handle 错误；请明确空 handle 叠加 n=0、n<0 或其他非法参数的返回值 | 暂拟沿用 arch22 的 handle-first，影响 Host 校验与对应状态单测 |
| 均匀/正态 50/50 的统计分母 | 任务书 §3.5 未明确按 case、元素还是批次计数，以及原始 fixture、特殊输入如何计入 | 暂拟普通随机 case 成对生成两种分布，特殊输入单列；影响覆盖统计 |

### 当前验证状态

方案设计已完成。CANN 9.1.0 编译、实际算子与 Netlib 对比、NPU 正确性和性能验证将在实现阶段执行。

## 兼容性分析

沿用公共 API、状态枚举、complex ABI 和 handle。arch35 通过现有架构选择独立编译，保持 arch22 实现；集成时检查导出符号唯一和架构目录选择。

共享测试参数若有扩展，保留旧字段表达并回归 arch22 CSV；本任务的精度及分布规则在算子测试内适配。产品支持文档分别列示 950PR 与 950DT，按各自验证结果维护。
