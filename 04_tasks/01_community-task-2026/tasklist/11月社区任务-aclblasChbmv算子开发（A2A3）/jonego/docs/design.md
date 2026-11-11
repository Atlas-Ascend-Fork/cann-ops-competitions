# aclblasChbmv 算子设计文档（Atlas A2/A3）


## 1. 需求背景

根据《aclblasChbmv_AtlasA2_A3_task_doc.md》，在 `cann/ops-blas` 补齐单精度复数 Hermitian 带状矩阵向量乘能力。使用 Ascend C kernel 直调模式，通过公共 BLAS handle 绑定的 stream 执行，支持 Atlas A2/A3。



| 依据 | 版本或原文位置 | 对本设计的作用 |
| --- | --- | --- |
| 本次任务书 | 2026-11-11 收到的文件；§2、§3、§4、§5、§7 | 接口、存储、错误返回、环境、精度、性能和交付要求 |
| 附件 `test_cases/README.md` | 第 107 行及精度、性能指导章节 | 明确 FP64 golden 由 `cblas_zhbmv` 计算；与任务书冲突处按任务书执行 |
| 附件两个 CSV | `chbmv_test.csv`、`gpu_baseline.csv` | 原始 1,200 个任务用例与 200 个 GPU 耗时 |
| 目标仓 | `cann/ops-blas`，`master`，`5bd95145a9c499e43ebad44a313a1029c63d54bc` | 公共类型、handle、构建、测试及同族 `Ssbmv` 组织方式 |
| 精度标准 | [opbase 固定版本](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) | 混合容差和最大误差点 ULP；特殊值按本任务明确规则执行 |

## 2. 需求分析

### 2.1 计算与接口

计算 `y_out = alpha × A × x + beta × y_in`。A 为 n 阶 Hermitian 带状矩阵，`A(j,i)=conj(A(i,j))`，主对角只使用实部。输入输出为 COMPLEX64，每个复数由两个 FLOAT32 分量组成，采用仓库 `aclblasComplex`，不另定义公共复数类型。

在 `include/cann_ops_blas.h` 新增公共声明：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

| 参数 | 所在内存与布局 | 约束 |
| --- | --- | --- |
| handle | Host；携带 stream | 已创建的有效句柄 |
| uplo、n、k、lda、incx、incy | Host，`int` | UPPER/LOWER；n、k 非负；lda ≥ k+1；步长可正可负但不能为 0 或 INT32_MIN |
| alpha、beta | Host 指针，复数标量 | n>0 时非空；启动前读取值并放入 Tiling 数据 |
| A | Device，只读，列主序带状数组，lda×n | 只引用指定带形三角；对角虚部和无效区域不参与计算 |
| x | Device，只读 | 容量至少 `1+(n-1)*abs(incx)` 个复数 |
| y | Device，原地输出 | 容量至少 `1+(n-1)*abs(incy)` 个复数；仅逻辑元素允许改变 |

接口参考是 [cuBLAS `cublasChbmv`](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hbmv) 及 Netlib CHBMV 的布局语义。**数值参考单独固定为 Netlib BLAS 3.10.0 的 FP64 复数计算**，详见 §3.7；不以 cuBLAS 或其他 BLAS 替换 golden。

### 2.2 边界与返回值

按任务书 §2.5 的顺序检查：

1. handle 为空，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. n=0，直接返回 `ACLBLAS_STATUS_SUCCESS`，不访问其他指针、不启动 kernel。
4. n>0，再检查 uplo、k、lda、incx、incy、alpha、beta、A、x、y。任何规定的非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。

`k+1`、绝对步长和 `(n-1)*abs(inc)` 在 int64 域计算。步长跨度超过 INT32_MAX 时拒绝；INT32_MIN 在求绝对值前拒绝。非法 uplo 按任务书 §7.5 返回 INVALID_VALUE，不照搬其他算子的 INVALID_ENUM。

允许 `k>n-1`：计算有效带宽 `h=min(k,n-1)`，但 **UPPER 的对角位置仍为原始 k，lda 仍须满足原始 k+1**。alpha=0、beta=0 等快路径位于合法性检查之后，不能掩盖非法参数或空指针。

beta=0 时不读取 y 旧值，也不执行 `0*y_in`；alpha=0 时执行 `beta*y_in`，允许实现直接跳过 A/x 计算。对角项采用实数乘复数，不能将带 NaN/Inf 的对角虚部送入一般复数乘法再乘零。

任务不涉及广播、批量、额外 Tensor 视图、超出 lda/inc 的布局；A/x/y 按任务书的独立对象使用，不增加 x/y 重叠支持。输出 y 原地更新。调用异步执行，调用者读回结果前同步对应 stream；不以 Host 端同步隐藏资源生命周期问题。

## 3. 详细设计

### 3.1 索引与公式

以下使用从 0 开始的下标。令 `B[p]` 为带状存储数组，`0≤i,j<n`，仅遍历 `|i-j|≤h`。

| 存储方式 | 条件 | 逻辑矩阵元素 |
| --- | --- | --- |
| UPPER | i≤j | `B[j*lda + k+i-j]` |
| UPPER | i>j | `conj(B[i*lda + k+j-i])` |
| LOWER | i≥j | `B[j*lda + i-j]` |
| LOWER | i<j | `conj(B[i*lda + j-i])` |

当 i=j，使用上述位置的实部并直接按实数处理。向量物理位置为：

```text
pos(t, inc) = t*inc                  （inc>0）
pos(t, inc) = (n-1-t)*(-inc)          （inc<0）
j0 = max(0, i-h)
j1 = min(n-1, i+h)
s_i = sum(A(i,j) * x[pos(j,incx)], j=j0..j1)
y[pos(i,incy)] = alpha*s_i + beta*y_in[pos(i,incy)]
```

负步长以分配数组的低地址为传入基址，不能先移动指针后又重复应用负步长起点。设备与 golden 使用完全一致的原始基址语义、参数及输入。

用如下非对称复数例子检查共轭和布局：

```text
n=3, k=1, lda=2, alpha=1+0i, beta=0+0i
A = [ 2       1+2i      0   ]
    [ 1-2i    3        -2+i ]
    [ 0      -2-i      4   ]
x = [1+i, 2-i, -1+2i]
y = [6+5i, 9-9i, -9+8i]

UPPER 各列：[无效,2]、[1+2i,3]、[-2+i,4]
LOWER 各列：[2,1-2i]、[3,-2-i]、[4,无效]
```

将无效区、lda padding 和对角虚部填入不同的干扰值，结果应保持不变。针对同一逻辑矩阵再改变正负步长、增加 lda，以及令 k>n-1，分别检查地址公式。

### 3.2 工程与职责

| 位置 | 计划内容 |
| --- | --- |
| `include/cann_ops_blas.h` | 共用 API 声明及接口注释 |
| `blas/sbmv/arch22/chbmv_host.cpp` | 校验、读取标量、选择路径、计算 Tiling、在 handle stream 启动 kernel |
| `blas/sbmv/arch22/chbmv_tiling_data.h` | n/k/h/lda、步长与起点、复数标量、核数、行块/列块、路径标识 |
| `blas/sbmv/arch22/chbmv_kernel.h`、`chbmv_kernel.cpp` | 启动声明与 Vector kernel 实现；避免不必要的文件拆分 |
| `test/sbmv/chbmv/` | 公共参数和 FP64 golden 适配、GTest 构建入口 |
| `test/sbmv/chbmv/arch22/` | 公共 API 封装、GTest、原始和补充 CSV、精度/性能运行说明 |
| 算子 README、`docs/zh/api_list.md` | 参数、存储示例、产品支持表、测试复现入口 |

沿用目标仓现有源文件收集方式和 `ops_blas_add_gtest_tests`，不另建平行工程。实现前复核当时的仓库变化和同名接口，避免重复定义。现阶段不修改已有 `Ssbmv` 计算逻辑和公共比较器行为。

Host 负责跨类型长度计算和资源预算，Kernel 只消费已确认合法的 Tiling。GM 元素偏移采用可容纳实际数组的宽类型；UB 局部索引在已证明范围内使用 32 位。`lda*n`、乘 8 字节及地址加法分别检查溢出，不先在 int 域计算再转换。公共 API 不携带容量参数，实际分配容量由调用者保证，测试程序显式核对。

### 3.3 通用路径

初始实现采用 **按输出行归属的单 kernel 方案**，在 AIV 上进行 FLOAT32 复数运算，不把 COMPLEX64 转为低精度 Cube 输入。

每个输出行分成主对角、直接存储部分、共轭映射部分。对有效带内列分块，把 A 和 x 的有效数据搬入 UB，拆分实虚部后计算：

```text
product_real = ar*xr - ai*xi
product_imag = ar*xi + ai*xr
```

行内分段归约后，合成 alpha、beta 项并写回。只有实际参与计算的下标进入搬运和归约。尾部掩码之外的临时位置在使用前初始化；不能用 `0*Inf` 代替排除无效元素。

通用路径支持所有合法 lda 和步长。优先按真实连续区间搬运；跨列访问拆成合法片段并在 UB 整理。GM 不扩读到数组外，UB 对齐补齐不改变 GM 有效长度。对于输出非单位步长，初稿采用单核并对每个逻辑复数执行精确长度写回，保持间隙不变，也避免多个核通过标量缓存更新同一物理缓存行。

### 3.4 优化路径与多核分块

正式性能附件的 200 个用例均为 `incx=incy=1`、`lda=k+1`、alpha=1、beta=0，可使用由这些公开参数决定的通用优化路径；不得根据用例编号、随机种子或预期答案选择实现。

| 路径 | 条件与方案 | 需要验证的风险 |
| --- | --- | --- |
| 零维/标量退化 | n=0 不启动；alpha=0 只处理 beta*y；k=0 仅取实对角 | 参数校验顺序；beta=0 不读旧值；特殊值处理 |
| 窄带多行 | 一个核负责 R 行，缓存相交列块及 x 窗口，按带内偏移组织多行 Vector 运算 | 打包和 Gather 成本；共轭侧边界；带宽增大后 UB 容量 |
| 中宽带分块 | 每核拥有输出行块，按列块复用 x；A 分列搬运到 UB 后用于本核行归约 | A 重复读取、尾块负载和归约误差 |
| 大带宽候选 | 只有 profiler 证明必要时，研究更大的二维分块或分段归约 | 新 workspace、额外 kernel、同步和计时成本；不能仅测主 kernel |

初稿不做跨核原子累加。每个 y 逻辑元素由一个核完成全部归约，避免复杂数值顺序及输出竞争。多核数取实际 AIV 可用核数与工作量限制的较小值，不硬编码课程中的核数。

对 `incy=1` 且输出基址满足搬运对齐要求的路径，以 32 字节、即 4 个复数为分配粒度切分行块；内部分界不共享搬运块，最后一个核独占尾部。小 n 或不满足多核对齐条件时使用单核精确长度写回。这里的 32 字节只针对 DMA 路径，不作为标量 GM 缓存行大小的假设。

分块负载先按每行有效长度 `L_i=1+min(h,i)+min(h,n-1-i)` 估计，再在允许的输出边界上划分。小输入避免核数过多；窄带避免每个复数发起一次搬运。行块 R、列块 C、核数以及单/双缓冲由 UB 预算和实测决定；不继承 `Ssbmv` 的经验耗时常数。

### 3.5 UB、workspace 与生命周期

通用路径首版不申请 GM workspace。A/x 双缓冲、实虚部分离和局部累加在核内 UB 完成。设每次最多 C 个复数参与行内分块，R 个输出行驻留，`align32(b)` 表示字节数向上对齐到 32：

| UB 用途 | 保守预算 |
| --- | --- |
| A/x 输入双缓冲 | `2*(align32(8*C)+align32(8*C))` |
| A/x 四个 FLOAT32 分量 | `4*align32(4*C)` |
| 两个乘积分量与一个计算临时区 | `3*align32(4*C)` |
| 两组局部偏移 | `2*align32(4*C)` |
| 行累加、旧 y、输出暂存 | `3*align32(8*R)` |
| 归约及运行时需要的临时空间 | 按 CANN 9.1.0 所选接口和实际分配逐项加入 |

不计对齐及接口额外空间时为 `68*C+24*R` 字节。只有在生存期不重叠且同步关系成立时才复用缓冲；若引入补偿累加，要另加补偿向量预算。多行打包优化按实际驻留 A 元素数重新计账，不能继续套用单行 C 的容量公式。

通过平台接口获取可用 UB，检查全部 `InitBuffer` 之和后决定 R/C。减少块大小可退回通用路径，不能靠向 GM 多分配或越界访问解决尾部问题。

输入标量按值传递，启动后不再访问临时 Host 指针。设备 A/x/y 的有效期由调用者覆盖 stream 执行期。首版无临时 GM 分配；若后续引入 workspace，则使用仓库 handle 管理的资源，明确最后使用的 stream 和释放时点，不在异步任务完成前释放。

### 3.6 读写归属与同步

| 数据流 | 必须成立的先后关系 |
| --- | --- |
| GM → UB → Vector | MTE2 完成后 Vector 才读取；使用配对队列或 MTE2_V 事件 |
| Vector → UB → GM | Vector 计算完成后 MTE3 才读取输出；使用 V_MTE3 或输出队列 |
| Scalar 填索引/读取局部结果 | Scalar 写后 Vector 读、Vector 写后 Scalar 读分别同步；仅在实际存在依赖处使用 S_V / V_S |
| 双缓冲循环复用 | 消费者结束后才能覆盖同一槽位；输入槽等待 Vector，输出槽等待 MTE3 |
| 同一 stream 多次调用 | 保持 stream 顺序，不使用跨调用共享的未保护临时数据 |

队列分配与释放配对；TBuf 按 TPipe 生命周期管理，不对 TBuf 使用队列出入队或 FreeTensor。首版采用清晰的流水顺序，在数据依赖得到验证后开启双缓冲，不为消除提示随意增删同步。

A、x 全程只读；y 的间隙、前后保护区不允许改写。非单位步长首版不采用整段 y 读改写，避免 beta=0 时读取不需要的旧逻辑输出。跨核无共享可写 UB/GM 临时区，无需跨核同步。

### 3.7 数值方案与指定参考库

**golden：Netlib BLAS 3.10.0，`cblas_zhbmv(CblasColMajor, ...)`，输入各 FLOAT32 分量精确提升为 double，计算完成后逐分量收窄为 FLOAT32，再与 NPU 输出比较。**

任务正文用 `cblas_chbmv` 描述算子语义，同时要求 FP64 计算。附件 README 第 107 行明确由双精度复数接口 `cblas_zhbmv` 承担该计算，因此不能调用 FLOAT32 `cblas_chbmv` 冒充高精度参考。

已核对官方 v3.10.0 源码：[CBLAS 包装](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/CBLAS/src/cblas_zhbmv.c)调用 Fortran ZHBMV；[ZHBMV](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/BLAS/SRC/zhbmv.f)使用 COMPLEX*16、共轭映射、对角实部、负步长起点和 beta=0 分支。这仅确认源码行为，远端库尚未构建或验证实际加载。

参考库由固定 v3.10.0 源码独立构建，启用 CBLAS、使用自带 Netlib BLAS，禁止 `USE_OPTIMIZED_BLAS` 或外部 BLAS 覆盖。记录构建参数、编译器及依赖；运行时同时确认 CBLAS 包装和底层 ZHBMV 的实际来源。通过受控动态链接与符号检查防止 OpenBLAS/MKL 等同名符号混入；仅设置一个库路径不足以证明正确。必要时将 golden 放在隔离的参考进程中，仍由 GTest 控制同一输入和比较。

参考计算不改变公共 API 参数，不将负向用例交给 Netlib 的错误处理代替 NPU API 返回值检查。收窄使用正常的最近偶数舍入，并记录非有限值，避免启用破坏特殊值语义的编译选项。

NPU 首版使用 FLOAT32 乘法及分段、成对归约，减少长串顺序累加误差。特别观察相消、宽带、极端有限值与缩放造成的误差；如不能满足精度，再试补偿累加或二的幂缩放，均作为需验证的实验。补偿只能改善舍入，不能自动解决中间乘积溢出。不能假设数学等价就能通过 FP64 golden，也不能用降低输入幅度或替换参考库解决失败。

## 4. 支持硬件与约束

目标支持 Atlas A2、Atlas A3，共用 `arch22` 路径；不复制此前 Atlas 950 的架构或编译选项。

**本任务自验证准备一套 Atlas 800T A2（910B3）即可，不要求同时准备 A3。** 依据为任务书 §3.1 第 80 行“自验证只需覆盖一种款型”，以及 §3.3 第 108 行“测试设备：Atlas 800T A2 (910B3)”。A2 单机验证不等于 A3 已实测；产品支持范围与实际测试记录分别说明。

CANN 固定为 **9.1.0**。上机时核实芯片型号、驱动、固件、Linux、编译器、SDK 路径、Python、GTest 和 profiler 的实际版本；驱动和固件遵守 CANN 9.1.0 的兼容要求，不从旧任务填入版本。仓库现有 `ascend910b` / `ascend910_93` 构建映射用于 A2/A3，实际 `--soc` 以实机及 SDK 支持值核对后使用。


## 5. 可维护性与可测试性分析

### 5.1 先确认测试可信

正式精度前完成以下检查并保存证据：

1. 记录源码版本、未提交差异、构建命令、二进制和完整用例清单；检查运行进程实际加载的是本次 `libops_blas.so`，而不是 SDK 安装目录中的旧库。
2. 核对 Netlib CBLAS 和底层 BLAS 的来源、版本及运行时符号，运行可手算布局例子；检查 FP64 提升和 FLOAT32 收窄确实发生。
3. 包装层逐项透传 uplo、n/k/lda、步长、标量和空指针，不修正非法参数、不替代 API 返回成功。设备和参考使用同一份初始输入副本。
4. 比较器注入错误自检：只改实部、只改虚部、修改首尾元素、违反匹配比例、违反最大误差限制、有限值改非有限值、相反符号 Inf，均必须被发现；符合任务规则的 NaN/Inf 等效组合应被接受。
5. 将预期清单与 GTest 注册、执行、比较和结果逐项关联，检查退出码、超时、遗漏和跳过。比较器自检与预期负向错误单独统计。

当前附件和目标仓存在下列需处理的差异，不能原样运行脚本后直接认定通过：

| 已发现差异 | 本任务的处理方式 |
| --- | --- |
| 附件精度脚本排除 PF 用例，且主要汇总解析到的输出 | 全部 200 个 PF 用例也先做精度；将完整预期清单与实际结果逐项匹配 |
| 附件性能指导使用 aclrtEvent 平均值 | 正式结果采用任务书规定的 `msprof op` task duration；event 仅作调试辅助 |
| 附件性能匹配未包含 lda，未知芯片有默认架构分支 | 匹配 uplo/n/k/lda/incx/incy 和固定标量；未知芯片停止，不默认为其他架构 |
| 当前公共比较器使用逐元素 ULP 限制及较严格的非有限值规则 | 在 Chbmv 测试内实现任务专用策略，按最大误差点和任务特殊值规则判定，不改变其他算子 |
| 附件脚本设置 ASCEND_DEVICE_ID，目标测试框架读取 BLAS_DEVICE_ID | 采用当前框架实际读取的设备号并在日志记录，不能只设置无效变量 |
| 原始生成器主要使用均匀分布 | 保留原始 CSV，补充正态分布及标量分布覆盖，纳入完整精度清单 |

### 5.2 完整精度范围

原始附件已静态核对：共 **1,200** 个不同名称的用例，其中非 PF 1,000 个、PF 200 个；预期状态为 1,184 个 SUCCESS 和 16 个 INVALID_VALUE。SUCCESS 中包含合法空输入，不能将此数直接写为非空输出精度比较数。

保留上述全部用例，再补充以下固定测试；补充后的准确数量由最终登记清单给出，不在设计阶段虚报：

- UPPER/LOWER 的可手算例子；n=0/1、质数、2 的幂及前后值、非对齐规模；k=0/1/n-1/>n-1；lda 最小值和 padding。
- incx/incy 的 ±1/±2/±3，步长间隙和尾部；INT32_MIN、跨度超限、k+1 溢出、空 handle/其他指针、n=0 与非法其他参数的优先级。
- 对角虚部、未引用带角、padding 的干扰值；A/x 只读验证；y 间隙及前后保护区；重复调用与指定 stream。
- alpha/beta 为 0、1、纯虚数、大值；beta=0 时旧 y 为非有限值；不组合 alpha=0 与含 Inf/NaN 的 x。
- 有限极值、相消、次正规数、正负零及任务允许的非有限值。
- 对随机类输入建立均匀/正态成对覆盖，A/x/y 的实虚部独立采样；alpha/beta 另补两类分布，正态 μ∈[-5,5]、σ∈[0.1,2]。固定种子并记录每类数量，特殊值和固定标量用例单列，不能用同一随机数同时填充实虚部。
- 每条实现路径及切换点前后、单核/多核和 UB 尾块；经确认有效的失败反例永久纳入清单。

每个候选冻结源码、构建、依赖与输入后，从头跑完整清单；全部精度符合要求才进入正式性能。实现、测试或依赖变化后重新完成全套，不能拼接不同版本的通过项。

### 5.3 精度判定

对每个非空用例的 n 个复数输出逐项检查，实部、虚部分开统计并分别通过。有限分量用 double 计算误差统计，避免统计过程自身溢出：

```text
e_i = abs(actual_i - golden_float32_i)
逐项容差 = 2^-13 + 2^-13 * abs(golden_float32_i)
matched_ratio = 有限分量中满足逐项容差的数量 / 有限分量数量
e_max = max(e_i)
L = max(0.01, 32 * ULP(golden_float32_at_max_error))
通过条件：matched_ratio >= 0.99 且 e_max <= L
```

ULP 按 FLOAT32 表示计算；0 和次正规数使用 `2^-149`。最大误差并列时记录所有并列点并检查各点限值，避免由遍历顺序选择更宽松值。CSV 的 MERE/MARE 字段不替代上述规则。

非有限值单独逐项判定：NaN/NaN、同号 Inf/Inf、NaN 与任一符号 Inf 等效；相反符号 Inf 不等效；有限与非有限不等效。所有非有限分量必须通过，并从有限统计中排除。全部为非有限值时记录有限统计“不适用”和实际特殊值数量，仍须逐项验证；合法 n=0 单独记录 quick return。缺输出、漏比较或空统计不能伪装成正常精度通过。

### 5.4 正式性能与优化实验

200 个 PF 参数与 GPU CSV 的 200 个不同参数组合已逐项对应。正式性能设备为 910B3，采集使用 `msprof op` 的 task duration，预热 5 次不计入；之后连续取得至少 10 个有效样本，首轮计划取 10 个，报告算术平均值与每次原始值。工具自带预热和应用循环的关系须通过实际 CANN 9.1.0 的帮助与输出核实，避免重复预热或把预热算入样本。

实机 `msprof op --help` 已确认：`--warm-up` 默认 5，`--launch-count` 默认只有 1，`--replay-mode` 默认 kernel。正式采集必须显式配置有效采样范围，并用实际记录确认至少 10 次；不能只用默认命令取一个 kernel 记录就宣布满足采样要求。帮助检查尚不等于已验证实际采样程序。

逐项判据：`NPU_avg_us <= GPU_ms*1000/0.8`，即 NPU 平均耗时不超过 GPU 的 1.25 倍。不能用全部用例平均值掩盖个别不达标项。前四条目标示例如下，均为根据附件计算的目标，**不是 NPU 实测**：

| 用例 | uplo / n / k | GPU 耗时（us） | NPU 上限（us） |
| --- | --- | ---: | ---: |
| TC_PF_1001 | U / 512 / 8 | 9.881600 | 12.352000 |
| TC_PF_1002 | U / 1024 / 16 | 15.872000 | 19.840000 |
| TC_PF_1003 | L / 2048 / 32 | 28.006400 | 35.008000 |
| TC_PF_1004 | L / 4096 / 64 | 53.026133 | 66.282666 |

每次样本须核对 API SUCCESS、设备执行结束及 profiler 记录完整。异常样本另存并说明原因，不能删除后挑选最快结果。若一个 API 包含多个辅助 kernel，计入该次算子的完整 kernel 耗时并保留关联，不能只报主 kernel。当前 PF 的 beta=0，每轮不依赖旧 y；其他含 beta 的性能场景在规定计时范围之外恢复相同初值。

初始瓶颈假设为：窄带的小片段搬运和标量地址计算，中宽带的 A/x 重复读取、归约与核间负载不均。以 profiler 的 MTE2、Vector、Scalar、等待时间及实际访存支持或否定假设；不预填带宽和加速比。

| 参考资料及已读位置 | 可借鉴机制 | 差异与下一项实验 |
| --- | --- | --- |
| 目标仓 `ssbmv_host.cpp`、`ssbmv_kernel.cpp` 的分块、FinalizeChunk、StoreStridedChunk | 行块归属、按 UB 预算切块、明确搬运同步 | 实数变复数且 beta=0 不能读取旧逻辑 y；分别比较行块/列块和核数，不复制经验常数 |
| 田纳西大学 ICL [MAGMA zhemv](https://github.com/icl-utk-edu/magma/blob/master/magmablas/zhemv.cu)，非对角块与两阶段归约 | 缓存 x、复用一个块及其共轭、局部归约 | 原实现为 CUDA 稠密双精度；本任务先限制在带内及核内，只有收益证据充分时才试带 workspace 的两阶段方案 |
| 得州大学奥斯汀分校 [BLIS gemv_unb_var1](https://github.com/flame/blis/blob/master/frame/2/gemv/bli_gemv_unb_var1.c) | 由行点积组织输出、显式传递步长与共轭 | CPU 微内核不能搬到 Ascend；用来比较逐行归约与多行复用的取舍 |
| 学习库 Ascend C 03.04、03.05 与 08.02、08.04 的相关正文和示例 | Tiling、TBuf/队列生存期、profiling、双缓冲、搬运粒度 | 课程环境和 Add 示例不是本任务规格；在完整精度通过的候选上逐项改变缓冲或块大小 |

后续继续按实际瓶颈检索高校项目、论文和竞赛实现，记录来源、适用条件、与本任务的差异及实验结果。连续两轮没有可靠收益、重复同类失败或 profiler 与假设冲突时，先回到相关资料并形成下一项可检验假设。

### 5.5 交付与当前状态

设计按任务书 §4 在社区任务仓提交 PR 并评审；代码按 §5 进入 ops-blas。本次平台展示完整任务名，未提供独立数字编号，因此设计提交目录采用 `11月社区任务-aclblasChbmv算子开发（A2A3）/jonego/docs/design.md`，位于社区仓 `04_tasks/01_community-task-2026/tasklist/` 下；A2/A3 在目录名中写作 A2A3，避免斜杠成为路径分隔符。若维护者指定统一目录，再按评审意见调整。提交、推送、PR、评论、邀请开发者和官网提交均由任务参与者手动完成。

代码仓交付实现、测试、CSV 和复现文档；官网材料另行整理。任务书同时要求在个人仓根目录 `task_submission/` 放置自验证步骤、精度 xlsx/log、性能 xlsx/log、内存 xlsx/log 共 7 项。内存专项标注“任务不涉及”，不编造工具通过记录。精度报告含实部和虚部结果及真实截图，性能报告含原始数据及真实截图；不复用示例包中的其他算子图片。

## 6. 兼容性分析

新增 API 复用公共 handle、状态码和 `aclblasComplex`，不修改已有接口签名。Host/Kernel 放置于 arch22，与同族 Ssbmv 并存；功能代码不依赖本机路径、下载资料或个人账号。

A2/A3 使用各自平台提供的核数、UB 和 SDK 能力；A2 的调优参数只在适用条件内启用，其他情况回退到通用路径。未在 A3 上运行的内容如实标注，不把 A2 的测试记录复制为 A3 成绩。

测试中新增的 FP64 参考适配和比较策略局限于 Chbmv；与仓库其他算子共享的解析、构建或数据填充若必须调整，独立说明影响并运行相关回归。不会因本任务特殊值规则而放宽其他算子的比较。
