# aclblasChpmv 算子设计文档（Ascend 950PR）

版本：v1.1（设计评审稿），2026-10-28  
代码仓：cann/ops-blas，新增 `blas/spmv/arch35/` 复数实现  
目标环境：CANN 9.1.0；Ascend 950PR；arch35 / dav-3510

## 1. 需求背景（required）

### 需求来源

- 任务：`aclblasChpmv`，Ascend 950PR，
- 当前状态：设计初稿；未完成设备编译、NPU 精度、性能或外部评审。本文的路径选择和收益均为待验证方案。

### 背景介绍

在 `blas/spmv/arch35/` 增加单精度复数 Hermitian packed 矩阵与向量乘法，补充现有实数 `aclblasSspmv` 家族。公开接口声明放在 `include/cann_ops_blas.h`，不新增产品私有 API。

cuBLAS `cublasChpmv` 用于核对接口和 packed/Hermitian 语义；正式精度结果只能与任务指定的 Netlib BLAS 3.10.0 `cblas_chpmv(CblasColMajor, ...)` 比较。NumPy、高精度计算和其他 BLAS 只能用于诊断。

## 2. 需求分析（required）

### 需求描述

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* AP,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

`aclblasComplex` 使用主仓定义的 `{float real; float imag;}`，每个元素 8 字节。alpha、beta 位于 Host；AP、x、y 位于 Device。读取 Host 标量后按值传入 kernel，接口异步使用 handle 绑定的 stream。

| 参数 | 存储位置与含义 | 约束 |
| --- | --- | --- |
| handle | Host 句柄，携带 stream | 非空；n<0 的错误优先返回 |
| uplo | Host 枚举 | ACLBLAS_UPPER / ACLBLAS_LOWER |
| n | Host int | n>=0；n=0 时不访问其他计算参数 |
| alpha / beta | Host complex64 指针 | n>0 时非空；读取值后按值传递 |
| AP | Device 只读 packed 矩阵 | n>0 时非空，至少 n(n+1)/2 个复数，无 lda |
| x / incx | Device 只读向量及 Host 步长 | incx 非零，可正可负；分配长度见下文 |
| y / incy | Device 原地输出及 Host 步长 | incy 非零，可正可负；beta=0 不读旧 y |

错误返回分别采用 `ACLBLAS_STATUS_INVALID_VALUE`、`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；合法执行及空操作返回 `ACLBLAS_STATUS_SUCCESS`。n=0 必须先满足 handle 非空。

### 需求拆解

1. 支持 UPPER/LOWER packed 布局、复数标量、正负非零步长和原地更新 y。
2. 对角虚部不参与运算；`beta=0` 不读取 y 旧值；合法空任务不访问设备缓冲区。
3. 严格采用参数检查顺序：`n<0` → handle → `n==0` 返回成功 → uplo、incx、incy、alpha、beta、AP、x、y。
4. 实部、虚部分别完整比较，覆盖原始 1200 条用例以及有效补充用例。200 条性能形状也先完整检查精度。
5. 正式性能使用 950PR 的 `msprof op` task duration；以全部 200 条 GPU 对照数据逐项判断，不以 GTest 墙钟时间代替。
6. 代码、设计、日志、截图和报告对应同一候选源码及构建。源码、推送及外部 PR 由用户手动提交。

#### 已发现的附件问题及处理

| 位置 | 事实与影响 | 处理 |
| --- | --- | --- |
| `verify_accuracy.py` | 忽略测试进程退出码，未核对完整名单；超时返回空结果后可能退出 0 | 新增严格运行入口，保留退出码、原始输出、XML，并核对逐项名称与状态 |
| `verify_accuracy.py` | 默认排除 TC_PF，不能证明大性能形状数值正确 | 正式精度包括原始全部 1200 条，再叠加固定测试与有效反例 |
| `verify_performance.py` | 以 GTest 整项毫秒值作为 NPU 耗时；无参考或缺项也可能成功 | 使用 profiler 的微秒数据，缺项、失败或样本不足一律不能通过 |
| `gpu_baseline.csv` | 200 行对照数据含重复形状，不能用形状字典覆盖先前行 | 按 PF 行序与对照 id 序号关联，同时核对 uplo/n/incx/incy，逐行保留原值 |
| 任务 §3.3、§7.4、附件 README | 分别要求 10 次、超过 10 次、超过 50 次有效样本 | 计划保存 51 次有效样本，分别报告前 10 次及全部 51 次的平均值；不挑选快样本。具体 profiler 预热控制由实际版本帮助信息确认 |
| 数据生成器 | 当前随机填充只有均匀分布，`--dist mixed` 未实现正态分布 | 原始 CSV 保留不变；每个合法随机用例追加同参数正态配对，另补充随机复数标量，记录 mu、sigma、seed 与实际比例 |
| 附件精度说明 | 同时描述 MERE/MARE 与生态混合容差 | 采用任务 §3.2 的混合容差作为正式判据；MERE/MARE 若保留只列辅助统计，不代替任务判据 |
| 通用 opbase 与任务特殊值条款 | 任务明确允许 NaN 与任意符号 Inf 等效 | 本任务按任务 §3.2 单独实现；不直接套用更严格的通用分类比较器 |

## 3. 详细设计（required）

### 算子分析

#### 数学公式

`y_i = alpha * sum_j(A_ij * x_j) + beta * y_i`。

packed 索引使用 0 基，所有乘法先提升到 64 位：

- UPPER：`s(j)=j*(j+1)/2`；`i<=j` 时读取 `AP[s(j)+i]`，否则读取 `conj(AP[s(i)+j])`。
- LOWER：`s(j)=j*(2*n-j+1)/2`；`i>=j` 时读取 `AP[s(j)+i-j]`，否则读取 `conj(AP[s(i)+j-i])`。
- 对角：仅读取 `AP[s(i)+diag_offset].real`，虚部不参与任何乘法，包括不使用 `0*imag`。
- 向量物理位置：`offset(i,inc) = i*inc`（inc>0），否则 `(n-1-i)*(-int64_t(inc))`。参数指针指向物理分配起点，不预先重定向到逻辑首元素。

可手算示例：

```text
A = [[2, 1+2i], [1-2i, 3]], x=[1+i, 2-i], alpha=1, beta=0
UPPER AP=[2, 1+2i, 3]
LOWER AP=[2, 1-2i, 3]
两种布局均得到 y=[6+5i, 9-4i]。
```

#### 支持数据类型与形状

- complex64；AP 长度至少 `n*(n+1)/2`，无 lda。
- x/y 分配长度至少 `1+(n-1)*abs(inc)`，无广播、无额外 Tensor stride。
- n 为非负 int，incx/incy 为非零 int。`INT_MIN` 不能仅因取绝对值溢出而被自行判非法，先提升为 int64_t；至少验证 n=1 与极端步长的合法情况。
- AP/x/y 按任务约定为独立张量，不扩展为输入输出相互别名的支持声明。

### 算子实现

#### 实现方案

##### Host 侧设计

1. 按规定顺序验证参数。负 n 优先于空 handle；合法 handle 且 n=0 时即使其他参数非法也直接成功。
2. 验证通过后复制 alpha/beta 到局部值，获取 stream 和实际 AIV 核数。
3. alpha=0、beta=1 可直接返回；alpha=0 的其他情况进入同一 kernel 内的缩放分支，alpha=0 且 beta=0 时直接写复零，不读取 AP/x/y。AP/x/y 指针仍先按 n>0 规则检查。
4. 普通路径计算 `blocks=min(ceil(n/128), AIV核心数)`，线程数向 128 对齐并限制在主仓支持范围。该参数为首个候选，后续按实际 profile 调整。
5. Tiling 按值携带 n、uplo、incx/incy、alpha/beta、线程数，避免异步 Host 参数地址生命周期问题。
6. 首个候选不申请 workspace，不进行设备同步；同步由调用者或测试端完成。

数据流为 `Host 参数检查 → 复制标量及分块参数 → stream 上下发单个 AIV kernel → SIMT 线程独占输出行 → 调用者同步后读回`。首个候选的 Tiling 参数为 48 字节；启动参数按值传递，不在 handle 工作区保存临时 Host 指针。

##### Kernel 侧设计：逐行有序通用路径

首个候选采用当前主仓 arch35 已使用的 Ascend C SIMT 接口 `asc_vf_call`。一个线程独占一个逻辑输出行，按 grid-stride 处理后续行。不同线程不读写彼此的 y，不使用原子累加或跨核共享临时结果。

为降低与 Netlib 的舍入差异，按原算法分解每个独立输出，而非直接对整个点积先求和后乘 alpha：

```text
先：out = beta==0 ? 0 : beta*y_i；beta==1 可保留旧值。
UPPER：
  t = 顺序累加 j<i 的 conj(AP[s(i)+j])*x_j
  out = (out + (alpha*x_i)*diag_real) + alpha*t
  顺序处理 j>i：out += (alpha*x_j)*AP[s(j)+i]
LOWER：
  顺序处理 j<i：out += (alpha*x_j)*AP[s(j)+i-j]
  out += (alpha*x_i)*diag_real
  t = 顺序累加 j>i 的 conj(AP[s(i)+j-i])*x_j
  out += alpha*t
写回 y 的逻辑位置。
```

复数乘法显式分为实部/虚部的 FP32 运算。对角实数乘法单独实现，避免将对角虚部或虚构的零虚部带入非有限值运算。保留 beta=0 与 alpha=0 分支，不用乘零代替不读取。

上述顺序用于构造首个正确性候选，不声称已证明与参考编译结果逐位一致。实际 Netlib 构建中的乘加收缩、复数乘法与次正规数行为必须自检；不修改参考库迎合设备结果。

##### 多核分工、存储与同步

- 每个逻辑 y 元素由唯一 SIMT 线程写入；使用实际元素地址访问，不对用户 AP/x/y 向上扩展搬运范围。
- y 步长空隙、前后 guard 以及 AP/x 的只读内容单独比较，不计入精度分母。
- 首个候选不使用 TQue/TBuf/共享 UB，不预设课程示例的 UB 容量；每线程仅保留复数累加器与索引。寄存器压力、编译器溢出到局部存储及真实占用待设备编译后确认。
- 同一缓存块内不同线程的 SIMT 元素写入需以当前 SDK 语义和真机 guard/重复调用测试确认；若优化改用 DMA 输出，重新设计物理写入归属，不将逻辑不重叠直接视为物理安全。
- 无跨核同步，无临时设备分配，无输入预处理缓存，连续调用会重新读取当前输入。

##### 优化候选与采用条件

| 候选 | 预计减少的成本 | 正确性与成本检查 |
| --- | --- | --- |
| 调整每核线程数及行数 | 小尺寸调度开销、空闲线程或不均匀负载 | 验证分派前后、n=0/1、奇数、负步长；不按 case id 或 seed 分派 |
| 行块与列块协作搬运 | packed 长列重复读取、x 的重复访存 | 严格限制用户搬运范围；计算 UB/事件预算；确认共轭侧地址与尾块 |
| UPPER/LOWER、连续步长模板分派 | 内循环分支与地址运算 | 保留任意合法非零步长的通用路径；新增路径全部参与完整精度 |
| SIMD 有序累加 | 每行循环的 SIMT 指令成本 | 保持每个输出的运算顺序；验证寄存器/UB 复用与生产消费事件 |
| 分段规约或 packed 展开 | 长规约、非连续共轭访存 | 会改变舍入或增加 kernel/工作区；只有全量精度通过且包括辅助 kernel 的总耗时改善才采用 |

首个路径复杂度 O(n²)，AP 每个非对角元素对应两个逻辑输出，访问量估计接近 `8*n²` 字节；x 重复读取另计。n=8192 时 AP 原始存储为 268,468,224 字节。是否受带宽、指令发射或地址计算限制以 profile 为准，不预填性能成绩。

连续两轮无可靠收益、同类精度问题重复或 profile 与假设不符时，记录当前证据并回查 learning-hub 对应章节，再开展新的可证伪实验。

## 4. 支持硬件与算子约束（required）

| 产品 | 本任务目标 | 当前验证状态 |
| --- | --- | --- |
| Ascend 950PR / arch35 / dav-3510 | 支持 | 已识别设备及 CANN；算子尚未设备验证 |

已连接环境为 Ubuntu 22.04.5 LTS / x86_64，设备由 npu-smi 识别为 Ascend950PR，CANN 路径为 `/usr/local/Ascend/cann-9.1.0`。已确认 Ascend C 编译器、msprof op、GCC/GFortran 11.4.0、CMake 3.22.1 与 GTest 可用。精确 SDK 组件、核心数及算子构建结果在正式验证前继续记录；Netlib 3.10.0 尚未完成构建及行为核对。主仓会拉取固定版本 ops-tensor 依赖，离线环境须提前准备。

### 算子约束限制

无 lda、broadcast 和额外 Tensor stride；仅支持 complex64。调用者负责合法 Device 分配及异步生命周期。不会将 CPU 数学模型、模拟执行或其他型号设备的结果写为 950PR 通过。

## 5. 可维可测分析（required）

### 精度标准/性能标准

#### 参考库身份

从官方 [Reference-LAPACK v3.10.0 发布版本](https://github.com/Reference-LAPACK/lapack/releases/tag/v3.10.0) 中的 Reference BLAS 与 CBLAS 源码构建任务指定库，固定版本、构建选项及真实产物路径。测试启动记录文件 SHA256，使用 `dladdr` 及必要的动态绑定信息核对 `cblas_chpmv` 和其内部 Fortran `chpmv_` 的实际提供者。若来源、ABI 或行为不符合要求则停止正式验证。小输入自检包含上/下 packed、负步长、对角虚部毒值、beta=0，以及用于识别乘加收缩差异的构造输入。

主仓通用 CMake 默认会查找 OpenBLAS/BLAS；本算子测试必须显式绑定任务指定库，不能因为文件名为 libblas.so 或导出 cblas_chpmv 就认定来源正确。只在本算子范围配置，不改其他算子参考库。

#### 全输出比较

1. 实部/虚部分别计算有限值统计，atol=rtol=2^-13，有限值 matched_ratio 至少 0.99。
2. 最大有限值绝对误差单独判断；上限为 `max(0.01, 32*ULP(golden_at_max_error))`，ULP 按 FLOAT32 最大误差位置的 golden 确定。若最大误差并列，记录各位置，采用这些位置所得上限中的最小值。零及次正规数的 ULP 为 2^-149；其他有限值为 `2^(ilogb(abs(golden))-23)`，不以另一个大数位置放宽上限。
3. 特殊值按任务条款：同 NaN、同号 Inf、NaN 与任意 Inf 等效；有限与非有限不等效，正负 Inf 互相不等效。非有限值单列统计并要求全部等效。
4. 合法空输出只判语义；全为非有限值时有限值匹配率记为空，不伪造 100%，仍要求所有特殊值通过。
5. 比较器独立故障注入覆盖有限误差、比例不足、硬上限、特殊值、guard 改写、全非有限与空输出；这些自检不计入算子精度通过数。

#### 完整精度清单

- 原始 1200 条 CSV 逐项执行：包括 1000 条常规/负向与 200 条性能形状；原始输入和配置不得改写。
- 固定 API 用例：规定校验顺序、空 handle、n=0 叠加非法值、标量/张量空指针、正负步长、n=1 配合 INT_MIN。
- 数值与存储补充：手算非对称输入、对角虚部 NaN/Inf/非零毒值、beta=0 配合 NaN 旧 y、极端有限值、次正规数、正负 Inf、相消、分派阈值前后、非对齐合法起点、guard/只读区与 y 步长空隙。
- 每个含随机张量的合法原始用例追加同参数正态配对用例，只替换该配对中标为随机的填充，保留固定值、极端值及其他参数。以每个随机张量的调用数和有效元素数分别统计均匀/正态 50%/50%；另以等量配对补充随机 alpha/beta。实际 mu/sigma/seed 记录在结果中，复数实虚独立采样，矩阵对角虚部按任务置零。固定毒值及故障注入自检不混入随机比例。
- 连续调用、修改同地址输入后再次调用、绑定不同 stream、同步前 Host 标量失效的生命周期测试。
- 每个冻结候选从头执行全部清单，核对 CSV、实际注册名单、XML、退出码、参数与日志。失败、遗漏、跳过或超时均不得进入正式性能。

#### 正式性能

使用同一通过完整精度的计算源码和构建产物。每条 PF 用例按原序号映射 GPU CSV，并将 `gpu_ms` 乘 1000 转为微秒。判据为 `NPU_avg_us <= GPU_ms*1000/0.4`。

先检查 950PR 身份与设备占用，再按已安装 `msprof op` 帮助信息配置 5 次预热与有效样本。计划保存 51 次有效样本，分别报告首 10 次与全部 51 次均值及各自判定。附加统计不能替换指定的首 10 次口径。所有异常样本保留原因；不能静默删慢样本。

当前 PF 全部 beta=0，反复调用不依赖旧 y；如补充原地累积型性能场景，每次恢复同一初始 y。恢复及数据准备不计入 kernel 时间。若使用多个计算 kernel，汇总单次 API 内全部计算任务，保留 task 对应关系，不只统计主 kernel。

前 5 条典型性能目标由原始对照 CSV 换算如下，其余 195 条逐行记录在随附 `performance_requirements.csv`。以下均为目标值，尚无本算子实测性能：

| n | uplo | GPU 平均耗时（us） | NPU 平均耗时上限（us） |
| --- | --- | --- | --- |
| 512 | UPPER | 31.209 | 78.0225 |
| 1024 | LOWER | 59.440 | 148.6000 |
| 2048 | UPPER | 121.020 | 302.5500 |
| 4096 | LOWER | 503.773 | 1259.4325 |
| 8192 | UPPER | 1161.690 | 2904.2250 |

#### 工程与交付

计划新增：`blas/spmv/arch35/chpmv_{host,kernel}.cpp`、`chpmv_tiling_data.h`、`test/spmv/chpmv/arch35/` 下的测试及原始 CSV，更新公共头文件和家族 README。实际构建按主仓当前自动发现规则接入，不凭课程目录猜测。

README 写清支持范围、环境、构建、指定 Netlib 库路径、完整精度入口、性能入口、结果位置和已执行范围。设计 PR、代码 PR、个人仓推送、系统验收和最终合入分别跟踪。用户手动推送；当前所有外部提交状态均为未执行。

## 6. 兼容性分析

新增同族公开 API，不修改 aclblasComplex、已有实数算子或公共 handle 行为。新增实现只位于 arch35。其他架构不在本任务实现范围，不能生成同签名私有替代接口或冒充已支持。

### 当前待完成项

- 已连接 950PR / CANN 9.1.0 环境；继续记录核心数、SDK 组件及正式构建输入。
- 完成并核对 Netlib 3.10.0 构建、符号来源、浮点行为与本算子测试的专用链接。
- 设计文档由用户按任务流程手动提交外部评审，目前未评审。
- 设备编译、完整精度、正式性能、材料和外部验收均未完成。
