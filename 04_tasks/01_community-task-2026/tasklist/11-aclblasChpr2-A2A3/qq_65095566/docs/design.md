# aclblasChpr2 算子开发设计文档（A2/A3）

| 项目 | 内容 |
| --- | --- |
| 交付阶段 | 设计评审稿；已完成 A2 实现与自验证，社区评审、验收尚未完成 |
| 修订日期 / 提交者 | 2026-11-26 / qq_65095566（Flipped） |
| 目标环境 | CANN 9.1.0，Atlas A2 / A3 系列，ops-blas `arch22` |
| 公开接口 | `aclblasChpr2`，句柄绑定 stream，原地更新单精度复数 Hermitian 压缩矩阵 |
| 提交位置 | `04_tasks/01_community-task-2026/tasklist/11-aclblasChpr2-A2A3/qq_65095566/docs/design.md` |

## 1 需求背景

### 1.1 需求来源

依据随赛题提供的《aclblasChpr2 算子开发任务书》§2–§3，为 ops-blas 补齐 A2/A3 的 BLAS Level 2 运算。数学语义对标 NVIDIA cuBLAS 的 `cublasChpr2`，CPU 精度标杆取 Netlib BLAS 3.10.0 的 `cblas_chpr2`。公开原型、错误码和快速返回的校验顺序以本题任务书和 ops-blas 公共头文件为准。[cuBLAS HPR2](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hpr2) · [Netlib CHPR2](https://www.netlib.org/blas/chpr2.f)

### 1.2 背景介绍与实现现状

Hermitian 矩阵满足 `A[j,i] = conj(A[i,j])`，对角为实数。调用方只存上三角或下三角，已包含重建整个矩阵所需的信息。以 `n=4096` 为例，压缩矩阵有 8,390,656 个复数，占 67,125,248 字节。若先展开成方阵再计算，仅一个稠密复数矩阵就需要 134,217,728 字节，还需重新打包。两个秩一贡献按列可复用系数，因此本设计直接遍历压缩数组，每个被更新的位置只需读取一次旧值、写入一次结果。

任务书指定公共接口已在 `include/cann_ops_blas.h` 声明，复数结构与状态码定义位于 `include/cann_ops_blas_common.h`。仓内 `blas/spr2/arch35/` 可供比对接口接入和算法语义。A2/A3 的核数、UB（核内统一缓冲区）、搬运和指令按 `arch22` 设计。实现归入 `blas/spr2/arch22/`，测试归入 `test/spr2/chpr2/arch22/`。附件 `dir_structure.md` 用于理解通用职责分工，具体落位遵循 BLAS 任务书。[ops-blas](https://gitcode.com/cann/ops-blas)

## 2 需求分析

### 2.1 需求描述与接口

在 handle 绑定的 stream 上，使用 Ascend C 直接启动 NPU Kernel，将两个互为共轭转置的秩一更新累加到 AP。沿用现有公开声明。调用者提供整段物理存储的起始地址，步长以复数元素为单位。

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```

| 参数 | 方向与位置 | 数据类型、形状与约束 |
| --- | --- | --- |
| handle | 输入，Host | 已创建的 `aclblasHandle_t`，使用其绑定的 stream |
| uplo | 属性 | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，指定 AP 中实际存放的三角 |
| n | 属性 | `int`，矩阵阶数，`n ≥ 0` |
| alpha | 输入，Host 或 Device | 指向一个 `aclblasComplex` 的有效指针，即使 `n=0` 也须非空 |
| x、y | 只读，Device | 各有 `n` 个逻辑复数。对 `n>0`，物理存储至少分别有 `1+(n−1)·abs(incx)`、`1+(n−1)·abs(incy)` 个元素 |
| incx、incy | 属性 | 各自独立的 `int`。支持正、负步长，排除 0 与 `INT_MIN` |
| AP | 输入/输出，Device | 一维连续压缩数组，`P=n(n+1)/2` 个 `aclblasComplex`。旧值参与累加，结果写回原地址 |

### 2.2 需求拆解

| 要求 | 设计落点 | 验证依据 |
| --- | --- | --- |
| 上下三角与复数共轭 | 用两套列首公式，每列分别复用两个复系数 | 逐列 packed golden、二阶手算例、纯虚系数 |
| 负步长及非单位步长 | 对 x、y 分别求首偏移和有符号索引 | 独立正负步长组合、大步长及孔洞填充值 |
| 错误优先级和快速返回 | Host 按任务书 §2.5 排序；Device 标量在同一 stream 上取值 | 组合非法参数、零长度、零系数与空指针 |
| 原地、异步与写入边界 | 每个 packed 位置有唯一写核；尾段精确搬运 | 前后哨兵、重复调用、同流改写 Device 标量 |
| 精度和性能 | 全程 FP32 分量计算；分核和向量缓存依据 UB 取值 | 1,000 条精度及边界用例、200 条逐例 kernel 耗时对标 |

采用 Ascend C Kernel 直调方式接入已有 BLAS 公共接口。多核切分、核内分块、缓冲预算和运行分支见 §3.2.1；数据搬运与流水依赖见 §3.2.2；精度、用例和性能采集见 §4。测试阈值采用本赛题规定值。

## 3 详细设计

### 3.1 算子分析

#### 数学公式

设上标 H 为共轭转置，横线为复共轭。完整矩阵的数学更新为

$$
A' = A + \alpha xy^{H} + \overline{\alpha}\,yx^{H}.
$$

固定列 `j`，定义 `c1 = alpha · conj(y[j])`、`c2 = conj(alpha · x[j])`。两个系数可供该列所有行复用。对非对角存储元素，计算 `A'[i,j] = (A[i,j] + x[i]·c1) + y[i]·c2`。对角实部计算 `Re(A[j,j]) + Re(x[j]·c1 + y[j]·c2)`，虚部直接写 `+0.0f`。旧对角虚部不参与实部运算。两项更新互为共轭转置，只更新已存三角就完整地定义了结果。[Netlib CHPR2 源码](https://www.netlib.org/blas/chpr2.f)

#### 支持数据类型与形状

alpha、x、y、AP 的元素均为 COMPLEX64，每个元素由实、虚两个相邻的 FLOAT32 分量组成，合计 8 字节。实现采用仓内 `aclblasComplex`，编译时验证大小及实、虚偏移与打包路径一致。x、y 为一维逻辑向量，AP 为一维列主序 packed 数组；接口没有 `lda`、批维或广播参数。

以下索引从 0 开始。`S(j)` 为第 `j` 列首元素位置，令 `S(n)=P`。每列占据半开区间 `[S(j), S(j+1))`。

| uplo | 合法行 i | 列长 | 列首 S(j) | 元素位置 k | 对角位置 |
| --- | --- | ---: | --- | --- | --- |
| UPPER | 0 ≤ i ≤ j | j+1 | j(j+1)/2 | S(j)+i | S(j+1)−1 |
| LOWER | j ≤ i < n | n−j | j(2n−j+1)/2 | S(j)+(i−j) | S(j) |

例如 `n=3` 时，上三角顺序为 `[A00, A01, A11, A02, A12, A22]`，下三角为 `[A00, A10, A20, A11, A21, A22]`。对于步长 `s` 和逻辑下标 `t`，令 `base = (s < 0) ? (1−n)·s : 0`，则物理位置为 `base+t·s`。例如 `n=3, s=−2`，依次访问物理位置 `4, 2, 0`。传入指针仍指向物理缓冲起点。

**二阶验算。** 取 `alpha=1+i`、`x=(1,i)`、`y=(1+i,2)`，并有意在对角输入中放入非零虚部。两种存储应产生共轭对应的非对角结果。

| 模式 | 输入 AP | 更新后的 AP |
| --- | --- | --- |
| UPPER | `[3+8i, 2−i, 5−7i]` | `[7, 4−i, 1]` |
| LOWER | `[3+8i, 2+i, 5−7i]` | `[7, 4+i, 1]` |

### 3.2 算子实现

**实现方案。** 使用 `arch22` AIV 直调 Kernel，Host 计算核数、块长与缓存布局。Kernel 按规模选择单元素、小矩阵、跨列 packed 向量块或按列分段路径；各核原地写入互不重叠的 AP 区间。外部 GM workspace 为 0。

#### 3.2.1 Host 侧设计与 Tiling

**入口校验。** 按下表自上而下执行。状态简称统一加前缀 `ACLBLAS_STATUS_`。

| 顺序 | 条件 | 行为 / 状态 |
| ---: | --- | --- |
| 1 | handle 为空 | `HANDLE_IS_NULLPTR` |
| 2 | n < 0 | `INVALID_VALUE` |
| 3 | alpha 指针为空 | `INVALID_VALUE` |
| 4 | uplo 非法 | `INVALID_ENUM` |
| 5 | incx 或 incy 为 0 或 `INT_MIN` | `INVALID_VALUE` |
| 6 | n = 0 | `SUCCESS`；不读取 alpha 值或 x/y/AP，不启动 Kernel |
| 7 | x、y、AP 任一指针为空 | `INVALID_VALUE`，包括 alpha 实际为零的情形 |
| 8 | 向量跨度、packed 字节长度或地址加法不能表示 | `INVALID_VALUE` |
| 9 | 查询标量位置失败 | 沿用仓内 `CheckPtrLocation` 的返回状态 |
| 10 | alpha 两个分量均为 0 | `SUCCESS`；保持 AP 全部字节，包括旧对角虚部 |

**标量位置与异步。** Host alpha 直接复制到启动参数；Device alpha 仅传地址，由同一 stream 的 Kernel 先搬入 8 字节并等待完成，再读取核内快照、判断零值。后者不做 Host 同步或 D2H 拷贝。即使 alpha 为零，n>0 的空 x/y/AP 仍先报错。启动使用 `handle->stream`；入口检查当前 launch 错误，执行期错误由调用方同步该流时检查。输入、输出和 Device alpha 在任务完成前保持有效。

**64 位寻址。** packed 长度为 `uint64_t(n)*(n+1)/2`，步长先扩展至 `int64_t` 再取绝对值；负步长物理下标为 `(n-1-row)*abs(inc)`。GM 偏移与字节范围检查使用 64 位。DMA 的窄字段不能表示步长时，按 64 位地址逐项搬运。4096 是原题测试规模边界，不是接口上限；n=4097 已覆盖流式路径。

**核数选择。** 从平台查询可用 AIV 核数 `C`、UB 容量 `U`，查询结果不满足执行要求时返回 `EXECUTION_FAILED`。令 `P=n(n+1)/2`、`H=ceil(P/8)`。

| 条件 | 实际 blockDim B |
| --- | --- |
| 1≤n≤4 | 1 |
| 5≤n≤16 | `min(H,C)` |
| n>16 | `min(H,C,n)` |
| AP 首址不是 64 字节对齐，或 Device alpha 首址位于 AP 可写范围内 | 覆盖上述选择，使用 1 核 |

每组 8 个复数对应 64 字节。AP 非对齐时使用单核精确搬运；Device alpha 指向 AP 内元素时，单核先保存系数快照，避免各核读到不同版本。调用方保证 AP 与 x/y 不重叠，x=y 可以作为只读别名。

**分段策略。** 小矩阵 packed 路径和跨列向量路径按组均分。核 b 的起点为
`8*(floor(H/B)*b+min(b,H%B))`，终点用 b+1 代入并截断至 P。
一般按列路径使用以下边界函数，首核从 0 开始，末尾固定为 P：

1. 当 `n≤L` 且 `n≤768` 时，先均分列号，再将列首 packed 下标向下对齐到 8 个复数。
2. 其余情况按 `W=P+L*n` 建模，将每列初始化成本折算为 L 个元素。将 W 均分给 B 核，以整数二分反解 `S(j)+L*j≤target`；段内偏移为 `max(target-S(j)-L*j-L,0)`，最后按 8 个复数向下对齐。
3. 每核二分定位首列，随后沿列推进，实际段长取本核剩余量、本列剩余量和 L 的最小值。边界可以落在列中，区间首尾相接，每个 packed 元素有唯一写核。

n>768 时始终采用工作量模型，包括 n≤L 的情况。这样可避免 n=1024 LOWER 简单均分列数造成首核计算量偏大。

**UB 布局。** 当前实现使用 `TPipe` 管理的 `TBuf` 单缓冲，配合显式屏障/事件。令 L 为 `tileRows`，V 为 `cacheStride`，均以元素数计。V 初选为 n 向上对齐至 8（n≤4096），否则为 0。

| 缓冲 | 分配字节数 | 用途 |
| --- | ---: | --- |
| scalar | 64 | 两个互异的 32 字节 DMA 标量槽、归约结果 |
| stage | `S*L` | 向量搬入、跨步整理及可复用中间量 |
| raw AP / output | `8L + 8L` | packed 输入、输出；跨列计算中分时复用 |
| planes | `32L` | 8 条 FP32 工作平面 |
| index | `16L` | 分离、重排、工作索引 |
| cache | `K*V` | x/y 四分量；跨列路径另缓存四条系数和零列标记 |

| 规模 | S | K | 总用户缓冲预算 |
| --- | ---: | ---: | --- |
| 64≤n≤768 | 64 | 36 | `128L+36V+64` |
| 其他 n≤1024 | 96 | 16 | `160L+16V+64` |
| n>1024 | 32 | 16 | `96L+16V+64` |

Host 要求预算不超过 `U-8192`。本次 910B3 的 U=196608，用户缓冲上限为 188416 字节。8 KiB 余量来自当前工具链 n=767/768 回归的实测规避，不将其声明为 A2/A3 的统一官方保留区规格。

L 初始根据 n 及本核 packed 片段估计，上限 1024，再选择 64、256 或 1024 档；不足时按 8 元素对齐逐次减半。若缓存连同最小块也放不下，先关闭缓存，再回算 L，仍不足则报错。Kernel 为 L=64/256/512/1024 提供编译期特化，其他块长使用通用分支。分区大小和起点均满足 32 字节对齐。

**启动参数与分支。** `Chpr2TilingData` 携带 `packed, incx, incy, n, lower, numBlocks, tileRows, cacheStride, deviceAlpha, alphaReal, alphaImag`。Device alpha 地址作为单独 Kernel 入参。直调实现由这些字段选择运行分支，无需图注册 TilingKey。

#### 3.2.2 Kernel 侧设计

| 路径 | 条件与处理 |
| --- | --- |
| 单元素 / tiny | n=1 专用；2≤n≤4 单核处理，省去向量索引初始化和缓存归约 |
| small packed | 5≤n≤16，按 64 字节 packed 组分核，标量计算本核片段；本机对齐场景每核最多 8 个复数，别名/非对齐时仍可单核处理全部片段 |
| flat packed SIMD | 64≤n≤768、缓存启用、L 是 64 的倍数，且 alpha/x/y 通过普通幅度路径判断；一个向量块可跨多列 |
| 按列缓存 / 流式 | 其余情况按列分段；缓存可用时复用 x/y，否则按块搬入行向量和当前列标量 |
| 特殊值恢复 | 对需要处理的大幅度/非有限分量，在 UB 内使用标量复数恢复与软件 FP32 融合乘加，保留普通输入的 SIMD 路径 |

**按列计算。** 固定 j 计算 `c1=alpha*conj(y[j])`、`c2=conj(alpha*x[j])`，段内依次累加两项复数贡献。四个独立乘法同阶段下发，再形成实部差、虚部和，最后累加到 AP；三个依赖阶段分别保留 `PIPE_V`。不改变各分量的运算次序。对角独立使用 `oldReal+(firstReal+secondReal)`，虚部写正零。

若 `x[j]`、`y[j]` 的四个分量均为零，保留非对角 AP，仅清零对角虚部。该分支优先于行向量计算，避免对含 Inf/NaN 的行向量执行额外的零乘法。整次调用的 alpha=0 分支又优先于零列处理。

**跨列 packed SIMD。** 在受限规模内，packed 索引可被 FP32 精确表示。以平方根估计列号，随后用三角数关系校正，得到每个 lane 的行列；LOWER 通过镜像 packed 序号复用该定位过程，镜像与字节偏移合并为整数计算。缓存 x/y 的四个分量、每列的四个系数分量及零列标记，用 Gather 组成本块的数据。分别选择普通元素、对角和零列的结果，再交错回写。

向量比较按 64 个 float 补齐，补齐索引截断至合法元素；AP 的 GM 读写长度仍严格为 `8*validCount` 字节。只构造本核片段与输入缓存实际需要的索引，省去随后会被覆盖的通用行索引初始化。复乘和 AP 累加的实、虚部独立运算同阶段下发；已结束使用的系数区承接虚部中间量和对角零值，避免覆盖尚未完成读取的缓冲。

**步长搬入。** x、y 各自选择读法。inc=+1 连续搬入；绝对步长≤32 时按 stage 容量读取连续物理窗口，再按逻辑索引 Gather（负步长倒序复数对）；其他可表示步长使用每块 8 字节的多块 `DataCopyPad`，按 UB 中 32 字节槽距压紧。超出 DMA 步长字段范围时按 64 位物理地址逐项精确搬入。孔洞不参与运算，最末窗口不超出逻辑向量所需的物理跨度。

**流式标量同步。** x[j]、y[j] 分别占据 scalar 的两个独立 32 字节槽。复用前用 S→MTE2 事件等待标量读完成，搬入后用 MTE2→S 事件等待 DMA 完成。禁止把两个列标量连续搬入同一槽后假定第二次标量读取必然刷新；n=4097 的固定输入回归已验证该风险和修复。

**精度恢复。** 普通分量采用 FP32 向量乘加；大幅度判断以分量绝对值达到 2^30 为保守阈值，并包含 Inf/NaN。为匹配本次 AArch64 Netlib 参考实现中有限数溢出附近的融合舍入，特殊路径用整数有效位运算实现单次舍入的 FP32 FMA，并在必要时执行复数 NaN/Inf 恢复。该软件 FMA 与主机 `std::fma` 对比 2,013,824 组，忽略 NaN 负载差异后无差异；最终仍以真实 NPU 的逐分量容差和特殊值分类测试为准。

**数据所有权与流水。** 搬入、计算、搬出及共享临时区复用之间保留实际依赖屏障。AP 对角、尾段均按有效字节通过 DMA 写回，不对相邻核的地址执行整块读改写。每个输出元素单核负责，无原子加、核间归约或卡间通信。

**优化依据。** 小规模优先减少索引、归约和列初始化成本；中等规模通过跨列块复用系数；较大规模同时平衡 packed 元素与列开销。本次 BasicInfo 提供整核耗时，不将其解释为管线利用率或指令时间轴。逐例收益由 §4 的完整性能采样判定。


### 3.3 支持硬件

| 目标 | 适配与自验安排 |
| --- | --- |
| Atlas A2 系列，含 Atlas 800T A2 | 目标为 `arch22`；正式性能指定 Atlas 800T A2（910B3），可在该设备同时完成精度自验 |
| Atlas A3 系列，含 Atlas 800T A3 | 目标为 `arch22`；按 A3 的平台核数、UB、编译目标生成启动参数。建议用短列、尾块、Device 标量和跨步用例做移植回归 |

工作任务书允许自验证覆盖一种款型。交付报告填实际机型、SoC、驱动、CANN 9.1.0、ops-blas commit、编译选项和测试用例版本。A2 与 A3 共用接口和算法；硬件相关参数由各自平台查询。

### 3.4 算子约束限制

| 主题 | 契约 |
| --- | --- |
| 数据布局 | AP 按列连续压缩；x/y 仅由 n 和各自元素步长描述。接口无任意张量视图、广播、AutoContiguous 或 ViewCopy |
| 动态形状 | n 和步长在调用时给出，Host 根据实参 Tiling；工程无需图模式动态 shape 推导 |
| 原地与别名 | 调用方保证 AP 可写范围与 x、y 的可读范围不重叠；x 和 y 均只读，可以指向同一向量。输出就是入参 AP |
| 输入值与对角 | 普通分量采用 SIMD，大幅度/非有限值按特殊路径处理。Hermitian 输入由调用方构造，Host 不扫描 Device 数据；正常更新忽略旧对角虚部 |
| 执行与确定性 | 算子无随机源，不要求浮点结果跨执行逐位一致；数值按 §4.1 判定。调用方在回读、释放或跨流复用缓冲前建立相应同步关系 |
| 内存 | 任务书 §3.4 无独立内存达标数值。本设计仅使用核内缓冲和启动参数。测试说明的单用例 Host 512 MB 是造数预算 |

## 4 可维可测分析

### 4.1 精度标准与性能标准

| 验收项 | 判定方法 | 标准来源 |
| --- | --- | --- |
| 常规结果精度 | 使用 Netlib BLAS 3.10.0 的 `cblas_chpr2(CblasColMajor, …)` 从同一份初始 AP 生成单精度复数 golden。对整个 packed 输出的实、虚部分别计算匹配率及最大绝对误差，两部分均须通过 | 任务书 §3.2；生态混合容差标准 |
| 精确语义 | 正常更新后每个对角虚部均为 0；零 alpha 时 AP 全字节一致；带步长只使用逻辑元素；错误码与优先级逐案相等 | 任务书 §2.4–§2.5、§3.5 |
| 正式性能 | 910B3 上逐条比较。5 次预热后取连续 10 个成功执行的有效 Kernel 样本；`mean(Task Duration[µs])/1000 ≤ 1.25 × gpu_ms` | 任务书 §3.3；附件 `test_cases/gpu_baseline.csv` |
| 内存安全 | 检查 AP 前后哨兵、x/y 原值、最短合法物理存储、核边界和尾块，记录异常地址与用例参数 | 任务书 §2.4–§2.5；本设计的回归检查 |

**精度判定。** 对每个有限 FP32 分量，记输出为 `a`、golden 为 `g`，满足 `abs(a−g) ≤ 2⁻¹³ + 2⁻¹³·abs(g)` 即为该点匹配。每个分量的整体匹配率须 ≥0.99，且有限点的最大绝对误差须 ≤`max(1e−2, 32·ULP(gₘ))`；`gₘ` 取该分量最大绝差点的 FP32 golden。按两侧分类逐点核对 NaN 位置和 Inf 符号，特殊值不参与有限最大绝差；全无有限点时单列记录。空输入、空指针和零 alpha 有各自的返回值或内容断言。随包 CSV 的 `mere_threshold`、`mare_multiplier` 留作测试工程的补充统计。正式报告保留实、虚部各自的匹配率、最大误差、索引及 golden 值。[任务书指定的混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) · [ULP 硬上限计算说明](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)

**逐例性能上限。** 随包 `chpr2_test.csv` 含 1,000 条精度及边界用例和 200 条 `TC_PF` 用例。实查 200 条性能用例的 `(uplo,n,incx,incy)` 均能唯一匹配到一条非空 `gpu_ms`；其中 188 条的两个步长均为 +1。下表末列为从附件计算的验收上限。

| 用例 | uplo | n | incx / incy | 附件 GPU Kernel（ms） | NPU 平均耗时上限（µs） |
| --- | --- | ---: | --- | ---: | ---: |
| TC_PF_1001 | UPPER | 512 | 1 / 1 | 0.0206592 | 25.824000 |
| TC_PF_1002 | LOWER | 1024 | 1 / 1 | 0.0491967 | 61.495875 |
| TC_PF_1003 | UPPER | 2048 | 1 / 1 | 0.1640791 | 205.098875 |
| TC_PF_1004 | LOWER | 4096 | 1 / 1 | 0.8605967 | 1075.745875 |
| TC_PF_1007 | UPPER | 4096 | 127 / 1 | 0.9266986 | 1158.373250 |

精度执行采用 ops-blas ATK/GTest 加载原题 CSV，保留用例、随机种子与阈值。正式性能从 `msprof op` 的 `OpBasicInfo*.csv` 读取 `Task Duration(us)`，不使用包含下发开销的 ACL event 值替代。命令在完成 CANN 9.1.0、Netlib BLAS 3.10.0、CMake 与 GTest 依赖准备后的 ops-blas 源码根目录运行：

```bash
bash test/spr2/chpr2/arch22/validation/build_chpr2.sh
source test/spr2/chpr2/arch22/validation/env.sh
mkdir -p ../logs
CHPR2_FREEZE_DIR=../selftest-package \
  build/test/spr2/chpr2/chpr2_test --gtest_output=xml:../logs/full.xml \
  > ../logs/full.log 2>&1
python3 test/spr2/chpr2/arch22/validation/run_profile_batches.py \
  --fix-visible-device-id --output ../profile-op-review
python3 test/spr2/chpr2/arch22/validation/package_cases.py \
  --root ../selftest-package --xml ../logs/full.xml --profile-dir ../profile-op-review
```

上述脚本与算子代码随验收交付，本次设计 PR 仅包含本文档。采集使用新的空输出目录；每个 case 预热 5 次、随后连续运行 10 次，按数字 launch ID 与真实 GTest 顺序关联原始记录。完整集必须具备 3,000 条原始数据，其中 2,000 条有效样本计入各自用例的均值。缺失记录、样本归属不明、非正耗时、执行错误或版本不一致时拒绝给出通过结论。均值不超过 `gpu_ms*1250` 微秒才判定达标。

**本次工具兼容处理。** 验证环境为 CANN 9.1.0、msopprof 26.1.0。容器内可见设备为 0，物理编号为 6；工具的编号转换错误调用了物理编号接口。进程内兼容层仅修正 `DeviceContext::GetRunningDeviceId()` 使用可见编号转换，先校验安装库版本哈希，不改计时器、采样值或驱动错误码。其他版本不能直接套用该开关。[对应上游源码](https://github.com/Ascend/msopcom/blob/b4ffdb3d6ba7acdb7bbfcf359f6e4dd5b587a4f4/csrc/runtime/inject_helpers/DeviceContext.cpp)

同版本导出解析器只匹配 1–3 位 launch ID，因此完整集串行分为 4 批，每批 50 个 case、750 次采样，规避四位编号被遗漏的问题。每个 case 的 5+10 规则、参数和性能阈值不变。正式数据保留原始 CSV、逐样本用例映射、SHA-256、工具及算子二进制哈希；报告生成前重新核算全部均值与判定。常规 msprof 的独立采集和 GTest ACL event 值分别记录为交叉验证与趋势参考。

### 4.2 用例覆盖与可定位性

| 检查组 | 当前覆盖与检查点 |
| --- | --- |
| 原题全量 | 1,000 条精度/边界用例、200 条性能用例；全部实际执行并核对 XML 条数，性能用例也完成 golden 正确性比对 |
| 数学语义 | UPPER/LOWER、一般复数系数、负步长、零列、非零输入对角虚部、alpha=0 的 AP 字节保持 |
| 错误优先级 | n<0、非法 uplo、inc=0/INT_MIN、空指针、n=0 叠加非法结构参数；补充空 handle 及组合错误测试 |
| 路径边界 | 两种三角覆盖 n=3/4/5/7/8/9/15/16/17/63/64/65/767/768/769，含零列和 AP 的 Inf/NaN；另测 n=4097 流式路径 |
| 标量与别名 | Host、ACL Host、Device alpha；同流先生成 Device alpha 再调用；负步长、8 字节偏移地址、x=y、Device alpha 指向 AP 内元素；该补充组含 17 个场景 |
| 大幅度数值 | n=1/2/4/16/65 的大 alpha 与 Inf/NaN，检查融合舍入和复数恢复；实、虚部分别记录匹配率、最大绝差及特殊值一致性 |
| 内存保护 | 设备缓冲前后 canary、x/y 只读、非法参数与 no-op 的字节保持。没有运行 mssanitizer，不将 canary 等同于所有越界读取均不存在的证明 |
| 可复现性 | 原始 CSV 不变，输入/golden 冻结为小端 interleaved COMPLEX64；记录 case、seed、dtype、物理跨度、文件哈希、环境和源码版本 |

原题 CSV 的 SHA-256 为 `83fc4fe9038ee0ea6ac1bb2e0f263bd572379d4792b36519a563a10c74c656b9`。原始附件的快速验证脚本存在零用例也可返回成功、性能入口优先读取 arch35 CSV 等问题，本次以完整 GTest XML 与 arch22 专用采集脚本为准，不以进程退出 0 代替覆盖数量检查。

任务指定的 `repo-task-case-gen` 未在附件和已安装工具中找到。目前附带 `package_cases.py` 对实际冻结输入/golden 生成 `cases.json`、`perf/cases.json` 与 `facts.json`，明确标记生成工具差异，不宣称已由指定工具生成。完整输入/golden 共 6,000 个二进制文件，目录、字节长度与哈希已保存；该差异在后续验收材料中继续披露。

### 4.3 设计级验证记录

在 CPU 上用固定随机种子 `20261125` 和 NumPy complex128 建立两个独立模型。一侧按压缩索引、列系数和物理步长更新，另一侧直接求稠密外积再按列枚举取得三角。阶数取 `1,2,3,5,7,16,31,32,33,63,64,65`，分别覆盖两种 uplo 及 `{-3,-2,-1,1,2,3}` 的 36 种步长配对。864 组共比较 577,080 个 packed 元素，最大绝对复误差约 `5.44×10⁻¹⁵`；按 `10⁻¹²` 计超限 0 个，输出对角虚部非零 0 个。§3.1 的二阶样例也得到表中结果。此项核对验证代数与索引；随后完成的 FP32 设备精度、边界及耗时自验证记录如下。

### 4.4 A2 实现自验证记录

验证日期为 2026-11-25，使用 Ascend910B3 / CANN 9.1.0 / Netlib BLAS 3.10.0，编译目标 `ascend910b3`。A3 使用相同 arch22 路线，本次没有 A3 真机运行记录；任务书 §3.1 允许自验证只覆盖一种款型。以下结果为开发者自验证，不代表社区评审或验收已通过。

| 项目 | 实测结果 |
| --- | --- |
| 完整 GTest | 1204/1204 通过，失败 0，耗时 246.830 秒 |
| 原题用例 / 补充组 | 1,200 条 / 4 组；包含前述 1,000 条精度和 200 条性能用例 |
| 实部最低匹配率 / 最大绝差 | 100.000000% / 6.103515625e-05 |
| 虚部最低匹配率 / 最大绝差 | 100.000000% / 9.155273438e-05 |
| 特殊值分类不一致 | 实部 0、虚部 0 |
| msprof op 性能 | 200/200 达标，0 条超限；3,000 条原始记录，其中 2,000 条计入均值 |
| 常规 msprof 独立交叉验证 | 200/200 达标，不能替代上一行的正式口径 |
| 内存性能 | 任务书 §3.4 不涉及；额外 canary、只读与 no-op 检查随完整测试通过 |

共 1,078 条原题用例产生数值比对记录，剩余原题用例采用状态码或 no-op 字节保持断言，不将其记成数值误差为零。

以下列出最终同一版本完整采集中的部分用例，单位均为微秒：

| case | n / uplo / incx / incy | kernel 平均耗时 | 题目上限 | 结论 |
| --- | --- | ---: | ---: | --- |
| TC_PF_1001 | 512 / UPPER / 1 / 1 | 20.692414 | 25.824000 | PASS |
| TC_PF_1002 | 1024 / LOWER / 1 / 1 | 45.670914 | 61.495875 | PASS |
| TC_PF_1004 | 4096 / LOWER / 1 / 1 | 432.714655 | 1075.745875 | PASS |
| TC_PF_1011 | 8 / UPPER / 1 / 1 | 5.368107 | 7.276000 | PASS |
| TC_PF_1012 | 16 / UPPER / 1 / 1 | 6.470129 | 8.392000 | PASS |
| TC_PF_1090 | 248 / LOWER / 1 / 1 | 11.414228 | 12.392250 | PASS |

优化集中于小矩阵固定成本、n>768 的工作量分核，以及跨列索引和独立向量运算调度。最终全部功能与性能数据重新采集，未混用旧二进制结果。同步候选分析与数据流分析没有高级候选；静态分析仅辅助检视，不代替 NPU 回归。

实现基于 ops-blas 提交 `6df9aa04a1d37288966e1b97a20ae8033ca2ce3c`。最终 kernel、host、GTest 源码 SHA-256 分别为：

```text
c969bb14cd9523d4f8a748a3ddbafd580cd5a33e4ff83d6294db99e04cc4a906  chpr2_kernel.cpp
1fcdcdaf69ba7c0621f728f650be6e81e31e5b57bcb0ec089f5733a97a00dfde  chpr2_host.cpp
31cf65993800be93c6155a30cff15dae4c0a38c4370d410d58b9ea4312a5249c  chpr2_test.cpp
```

完整精度/性能/内存 XLSX 报告、原始日志、样本清单及复现步骤已形成 `task_submission`，随后续验收交付。

### 4.5 兼容性分析

- **公共接口。** 复用 ops-blas 已有符号、参数次序、复数布局和状态码，arch22 与 arch35 使用同一公开头文件。非法 uplo 按 ops-blas 返回 `ACLBLAS_STATUS_INVALID_ENUM`；cuBLAS 文档采用 `CUBLAS_STATUS_INVALID_VALUE`。参考实现用于对齐数学结果，状态码按目标库验证。
- **硬件和工具链。** 核数、UB 容量、DMA 字段和向量指令上限按 CANN 9.1.0 的目标 SoC 编译和检查。packed 地址计算、步长寻址和 UB 预算使用明确的整数范围；平台差异集中在平台查询、launch 参数与目标编译选项处。
- **用例和后续交付。** 设计评审提交至团队 `docs/design.md`。当前 arch22 实现、ATK/GTest、复现步骤与报告已完成开发者自验，后续按任务流程提交私仓验收材料。设计评审、开发者自验与社区验收分别留档。
