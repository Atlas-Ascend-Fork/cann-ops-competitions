# aclblasChbmv 算子设计文档（Atlas A2/A3）

# 需求背景（required）

## 需求来源

任务为“11月社区任务-aclblasChbmv算子开发（A2A3）”，贡献者为 Ivyalways。本设计对应随任务发布的 `aclblasChbmv_AtlasA2_A3_task_doc.md` 和 `test_cases/`，目标是在 [ops-blas](https://gitcode.com/cann/ops-blas) 现有 `aclblasChbmv` 公共接口下补齐 Atlas A2/A3 的 arch22 实现，保留已合入上游的 Ascend 950PR（arch35）支持。本文描述已推送代码提交 `bab0239b3d617623770b9c6966e32e875052ee02` 的 v16 架构，设计评审与合入仍待完成。合并上游前的 v16 开发轮 clean build、2262 项完整精度（包含 95 个内部动态子场景）均 PASS，24 条性能诊断子集为 24 PASS。合并提交的新 checkout、空构建目录复现及全部 200 条正式性能门禁仍为 PENDING；开发子集不代表正式验收通过。此前 v12 正式完整性能结果为 184 PASS、16 FAIL，总体 FAIL，不能与本轮子集拼接为通过集合。设计贡献者与署名保持为 Ivyalways。

依据及版本：

| 材料 | 固定版本或 SHA-256 |
| --- | --- |
| 任务书 | `e03a2b9997011132a33916b74138e5b3bc6615cfbf47fa280b99b7a1b75d6a3b` |
| 原始 `chbmv_test.csv` | `190f8cd07fa4691051280ae421ecb8c721e63dd4abecc519e0beaf35613b658f` |
| 原始 `gpu_baseline.csv` | `dd2cb87a50589bde139bd42014e911eb4c8c05f31aac0fbdfb0633359bad4369` |
| ops-blas 调查基线 | `master`，`5bd95145a9c499e43ebad44a313a1029c63d54bc` |
| 当前已推送代码提交 | [`bab0239b3d617623770b9c6966e32e875052ee02`](https://gitcode.com/Ivyalways/ops-blas/commit/bab0239b3d617623770b9c6966e32e875052ee02)，`feat/chbmv-arch22`；包含 v16 实现及上游普通合并 |
| 已合并上游基线 | [`9c83813ef011ddd7b124557d228f8c74b55fea3d`](https://gitcode.com/cann/ops-blas/commit/9c83813ef011ddd7b124557d228f8c74b55fea3d)，已提供 Ascend 950PR（arch35）Chbmv；为当前代码提交的第二父提交 |
| 社区设计仓调查基线 | `master`，`f90187f2646933961fa7e3c15fca2036f6c6fdc6` |

设计路径遵循[2026 社区任务 README](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)规定，由贡献者创建任务目录，采用任务完整标题和账号目录；章节沿用[当前设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。设计需经评审并合入后才满足最终验收交付状态。

需求跟踪为 [ops-blas #885](https://gitcode.com/cann/ops-blas/issues/885)，设计 PR 为 [cann-ops-competitions !2784](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2784)。本次在原设计 PR 更新 v16 架构与验证边界，不新建平行设计；PR 尚未合入。最终交付前需重新核验代码提交、完整自验和评审状态。

## 背景介绍

### 算子功能与当前仓库现状

计算 `y = alpha * A * x + beta * y`。A 为 n 阶 Hermitian 带状矩阵，仅存储上或下三角的 k 条带和主对角；A、x、y、alpha、beta 均为 complex64。Hermitian 意味着 `A(i,j) = conj(A(j,i))`，主对角只取实部。

当前上游 `9c83813ef011ddd7b124557d228f8c74b55fea3d` 已声明并登记 `aclblasChbmv` 公共接口，并在 `blas/sbmv/arch35/` 提供 Ascend 950PR 实现。本贡献复用这一签名、复数布局和导出符号，补齐 `blas/sbmv/arch22/`，不重复声明接口或替换已有 arch35 实现。arch22 同族实数 Ssbmv 的句柄、stream、AIV 核数查询及构建约定可作为集成参考；其对称实数计算不能直接替代复数 Hermitian 运算。

除明确描述共享数学语义和现有产品支持的段落外，下文的算法、184 KiB 资源上限、Host 附加范围校验、数值守卫、FP64 golden、95 个内部动态子场景及性能门槛均限定于本任务的 arch22 实现。已有 arch35 实现和测试遵循其各自文档，本设计不将 arch22 的实现或实测结论外推到 Ascend 950PR。

功能基准为 [cuBLAS cublasChbmv](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hbmv)，存储和共轭语义参考 [Netlib CHBMV](https://www.netlib.org/blas/chbmv.f)。本次 A2/A3 任务书明确的校验次序和错误码约束 arch22 实现；合并时保留已有 arch35 实现的行为与独立测试。

# 需求分析（required）

## 需求描述

在 Atlas A2/A3、arch22 上以 Ascend C kernel 直调实现公共 `aclblasChbmv` API。工具链为正式 CANN 9.1.0，当前自验证 SoC 为 Ascend910B3。任务允许只覆盖一种款型实机；Atlas A3 仍为编译适配目标，不声称已有 A3 实测。机箱商品型号没有单独核验时只记录已确认的 SoC。实际 SoC、CANN、driver、compiler 和 msprof 版本随最终报告提供，其他版本的探索结果不替代正式验收证据。

## 需求拆解及追踪

| 任务要求 | 设计落点 | 实现位置 | 验证 |
| --- | --- | --- | --- |
| 共用复数 BLAS API | 复用上游统一接口合同 | `include/cann_ops_blas.h` | 签名/导出符号一致、安装后链接、既有架构集成回归 |
| 上下三角、共轭、对角实数 | 数学公式、核内数据通路 | `blas/sbmv/arch22/chbmv_kernel.cpp` | 全量输出双分量比对、对角虚部污染检查 |
| 步长、padding、超宽带 | 寻址公式、分段搬运 | host、kernel | 正负步长正交组合、k > n-1、非对齐尾块 |
| 参数与 quick return | 校验顺序 | `chbmv_host.cpp` | CSV 负向例及直接 API 单测 |
| 精度门槛 | 精度标准 | `test/sbmv/chbmv/` | 原始 1000 精度例、正态补充、特殊值 |
| 单 case 性能门槛 | 性能测量 | arch22 性能测试与采集脚本 | 全部 200 例 msprof op 平均 task duration |
| 可复现交付 | 交付计划 | README、`task_submission/` | 精确提交、输入哈希、原始结果、报告一致性 |

# 详细设计（required）

## 算子分析

### 数学公式与带存储

以下下标均从 0 开始。记 `w = min(k, n-1)`，对每个输出 i：

`y_new[i] = alpha * sum(A_full(i,j) * x[j], j=max(0,i-w)..min(n-1,i+w)) + beta * y_old[i]`。

令 `B(q,j) = A[j*lda+q]`，原始 k 用于上三角存储定位，不可用 w 替换：

| 模式 | 有效元素 | `A_full(i,j)` |
| --- | --- | --- |
| UPPER | `i < j`、`j-i <= w` | `B(k+i-j,j)` |
| UPPER | `j < i`、`i-j <= w` | `conj(B(k+j-i,i))` |
| LOWER | `j < i`、`i-j <= w` | `B(i-j,j)` |
| LOWER | `i < j`、`j-i <= w` | `conj(B(j-i,i))` |
| 对角 | `i == j` | UPPER 为 `(real(B(k,i)),0)`；LOWER 为 `(real(B(0,i)),0)` |

复数乘法使用 FP32 向量运算：`(a+ib)(c+id) = (ac-bd) + i(ad+bc)`；归约和跨分段累加保留 FP32，不降到 FP16。对角项单独按实数乘复数处理，避免让被忽略的虚部或 `0*Inf` 污染结果。未存储三角通过共轭表达，不显式展开 n×n 稠密矩阵。

### 支持数据类型、形状和参数

复数布局复用 `cann_ops_blas_common.h` 的 `aclblasComplex { float real; float imag; }`，GM 中为交错实虚部。以下函数签名与上游 arch35 共用；参数支持表和后续校验顺序描述本任务 arch22 合同，尤其 `INT32_MIN` 与逻辑跨度上限不表述为所有架构的统一新增限制。

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

| 参数 | 存储与所有权 | 支持范围 |
| --- | --- | --- |
| handle | Host，调用方创建并管理 | 有效 ops-blas 句柄，携带 stream |
| uplo | Host 枚举 | `ACLBLAS_UPPER`、`ACLBLAS_LOWER` |
| n、k、lda | Host int | `n>=0`、`k>=0`、`int64(lda)>=int64(k)+1` |
| alpha、beta | Host 指向 complex64 | n>0 时非空，调用期间读取并复制到 launch 参数 |
| A | Device，只读，调用方持有 | 列主序 `lda*n` 个 complex64，padding 不参与计算 |
| x | Device，只读，调用方持有 | n 个逻辑元素，存储至少 `1+(n-1)*abs(incx)` |
| y | Device，原地读写，调用方持有 | n 个逻辑元素，存储至少 `1+(n-1)*abs(incy)` |
| incx、incy | Host int | 非 0、非 INT32_MIN；`(n-1)*abs(inc)<=INT32_MAX` |

负步长起点为 `start = (inc<0) ? (n-1)*(-int64(inc)) : 0`，逻辑元素 j 位于 `start+j*int64(inc)`。步长乘加、GM 元素地址和字节地址在 64 位域计算；UB 内偏移先由已验证 tile 容量限定，再缩窄到硬件接口支持的整数范围。

n/k 为运行时值。无广播、batch、transpose 参数或独立动态 shape 要求。调用方须保持 Device 存储有效直到 stream 完成。y 原地更新是既定合同，A/x 与 y 的重叠不作为本任务新增支持；实现不增加未经任务定义的别名行为。

## 算子实现

### 3.2.1 Host 侧设计

本任务 arch22 的校验顺序固定为：

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（9），包括 n=0 情况。
2. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`（3）。
3. `n == 0` 返回 `ACLBLAS_STATUS_SUCCESS`（0），不读其他参数、不启动 kernel。
4. 校验 uplo、k、lda、incx、incy、alpha/beta/A/x/y；任一非法统一返回 INVALID_VALUE。尤其非法 uplo 不沿用相邻接口的 INVALID_ENUM。
5. lda 与 k+1 用 int64 比较；步长先排除 INT32_MIN，再在 int64 计算跨度，超 INT32_MAX 返回 INVALID_VALUE。
6. 读取 Host alpha/beta。通过上述全部校验后，`alpha=(0,0), beta=(1,0)` 可直接成功返回并保持 y 原样；其他情况查询 AIV 核数、构造 tiling，使用 handle 绑定的 stream 直调 kernel。核数查询失败返回执行失败，不将未执行的设备工作记为成功。

beta=0 时不读取 y 旧值，但 n>0 时 y 仍须非空。Device 端零值判断检查 FP32 去符号位后的位模式，正负零等价，非规格化非零值不会被误判为零。`beta=(1,0)` 直接加旧 y，避免零虚部与旧 y 的 Inf 相乘，污染另一有限分量。alpha=0 使用纯缩放路径；它仍遵守 n>0 的全部参数校验。按任务书要求，测试不将 alpha=0 与任何含 Inf/NaN 的 x 存储组合，也不从该优化推导更宽的接口保证。

#### 分核和分块

按输出的物理存储顺序划分连续行块，每个逻辑 y 元素只由一个 AIV 核写回。有效核数来自平台查询。packed 的最少行工作量为 `max(4,min(16,w))`，其他路径为 4；先据此和 AIV 核数确定分块，再令 `rowsPerCore=roundUp(ceil(n/blocks),4)`，实际 launch 核数按该行块覆盖范围确定。核起点对齐四个 complex64，避免核间共享同一个 32B 输出块。负 incy 将物理行块映射到逆序逻辑行，计算和写回仍保持逻辑对应。

按以下优先级选择路径，条件只取决于参数语义、访问模式和 UB 容量：

| 路径 | 通用条件 | 每核组织 |
| --- | --- | --- |
| 完整列 packed | `n>=4`，完整 A 窗口与 x 记录均满足容量条件 | 4096 lane，上限 64 行；满足对齐条件时复用 A 环形窗口 |
| compact packed | 完整列模式不满足；span 为 256/512/1024/2048，完整组内列、halo 条带及 x 联合容量满足 | 每 tile 16/8/4/2 行，每 tile 重载；大 x 窗口复用 A 尾部 |
| x 缓存行点积 | 未选 packed、`n<=4096`、`abs(incx)==1` | 每核缓存完整 x；每个 A 点积段最多 768 个复数 |
| 通用行点积 | 其余合法参数 | A/x 分段，单段最多 1024 个复数 |

packed 的共享 Host 计算函数 `ChbmvPackedSpan(w)` 先取 `combinedSpan=max(8,nextPow2(2w+1))`。仅当删除对角后确实增加 `min(64,4096/span)` 的输出行数时，才把对角移到独立向量阶段；等价判据为 `combinedSpan>4096/64 && 2w==combinedSpan/2`，满足时将 span 减半。跨度虽能缩小、但行数仍受 64 行上限限制时，保留对角同算，避免增加独立对角阶段。w=0 直接走实对角向量计算。

最终 `tileRows=min(64,4096/span)`。完整列模式先用 uint64 校验必要条件 `2*lda*(w+1)<=15488`；合法 lda>=w+1 因而限制 w<=87，避免在大带宽上运行未经约束的 uint32 倍增。再验证 `2*lda*min(n,tileRows+w)<=15488` 及 `xRecordFloats*min(n,tileRows+2w)<=1280`。单位正负 incx 的 xRecordFloats 为 2，其他为 8；自动 32B 尾补齐不会超过这些已对齐容量。完整列支持 SPAN 8/16/32/64/128/256，256 使用 Combined。

完整列不满足时，compact 前置检查 `2ULL*w<=2048`，再计算共享 span，允许 256/512/1024/2048。令 `R=4096/span`、`round8(v)=ceil(v/8)*8`，容量以 float 数计：

```text
fullFloats = round8(2*R*lda)
stripFloats = max(8,2*R)
aFloats = fullFloats + w*stripFloats
xFloats = round8(min(n,R+2*w) * (abs(incx)==1 ? 2 : 8))
```

若 xFloats<=1280，使用原独立 x 区且要求 aFloats<=15488；否则 x 放在 A 的 aFloats 偏移处，要求 aFloats+xFloats<=15488。所有 Host 乘加使用 uint64；未满足条件的合法输入回退行点积。R=2 仍将每个 halo 条带补成 8 float，不能按四个 float 低估容量。以上界限来自固定 UB 资源，与输入内容或性能 case 无关。

compact 资源预算通过后，再比较可复用行数：若 `4096/span<4` 且 `2*w<span`（仅两行且存在补齐 lane），同时具备完整 x 缓存和四行 A 条带缓存能力，则优先选择缓存行点积。能力条件为 `n<=4096`、`abs(incx)==1`、`w>=4`、`w+4<=1248`、`lda>4` 且 `(lda-5)*8<=UINT32_MAX`。该选择比较 4096-lane 的补齐开销与一个32B记录容纳四个complex64的复用粒度，不依赖某个case。Separate2048的非对角lane恰好填满跨度，保留compact；完整x缓存不可用时，也保留原compact/通用回退选择。

入口使用 `ChbmvPackedKernel<SPAN, MODE, COMPACT>`；MODE 为 Combined、Separate 或 Only。compact 四种跨度均有 Combined/Separate 模板，原纯对角保留在完整列路径。模式、布局与跨度编译期固定，核内仅保留相应处理。没有 case ID、精确样本表、随机 seed、输入分布或填充值查表分支。

四行 A 共享缓存是 x 缓存行点积内部的搬运优化，按 UB 容量和 DMA 能力选择，不增加公共 API 或独立 Host 分核方式。

#### Tiling 数据、资源和调度

`ChbmvTilingData` 保存 n、原始 k、lda、uplo、rowsPerCore、packedMode、int64 incx/incy，以及 alpha/beta 的实虚部；有效带宽、负步长起点和模板跨度由这些参数推导。packedMode 取 0/1/2 分别表示行点积、完整列 packed、compact packed；不扩展 tiling ABI。结构按值传入 launch，不为每次调用分配 Device tiling。

按正式 SDK 的 arch22 资源约束核算：UB 总量 192 KiB，其中用户可用区 184 KiB、框架保留区 8 KiB。每个 kernel 使用一个 TPipe，分配与 Host 容量守卫一致。

| 路径 | 资源组成 | 用户 UB 峰值 |
| --- | --- | ---: |
| packed | A 分配 19584 float，其中活动 A 或 A+x 联合区最多 15488 float、其余 4096 float 作最大幅值平面；独立 x 为 1280 float；6×4096 float 共享向量平面；832 float 小缓冲；4×512B 掩码 | 187136 B，182.75 KiB |
| x 缓存行点积 | 主计算 `32×768×4` B（其中39 KiB复用为四行A缓存）；x cache `4096×8` B；恢复 `56×256×4` B | 188416 B，184 KiB |
| 通用行点积 | 主计算 `32×1024×4` B；恢复 `56×256×4` B | 188416 B，184 KiB |

packed 的坐标、四个输入分量和乘积按生命期共享六个向量平面。小缓冲以 float 为单位依次安排实和 `[0,64)`、虚和 `[64,128)`、归约暂存 `[128,256)`、行守卫 `[256,320)` 和旧 y/输出共享区 `[320,832)`。归约结束后，部分暂存复用为 tile 风险标志；旧 y 的向量及标量读取结束后，才允许输出覆盖同一存储。

恢复辅助函数借用 A 分配的前 14336 float，即 56 KiB；最大幅值平面从 float 15488 开始，两者不重叠。结果、行守卫和旧 y 在借用区之外。完整列模式的 x 也在借用区外；compact 的大 x 可能位于借用区，但恢复在本 tile 全部输入 Gather、独立对角和缩放结束后发生，此时该缓存已不再读取。恢复使用原 GM 地址。调用恢复即使未最终替换结果，也会使完整列 A 环形缓存失效；compact 本就每 tile 重载 A/x。

x 缓存行点积的四行 A 缓存占主计算区 `[3*TILE,16*TILE)`，大小为 `13*768*4=39936 B`，即39 KiB。有效缓存组内，连续镜像侧 A 最多使用 `[0,2*TILE)`，范围最大值工作区使用 `[2*TILE,3*TILE)`。每个缓存record为32B、保存四个complex64，因此缓存容量为1248列。它包含在原 96 KiB 主计算区中，与 32 KiB 完整 x 缓存、56 KiB 独立恢复区合计仍为 184 KiB。

以 TILE 个 float 为一个平面、从 0 编号，整行实、虚累加使用主计算区平面 20、21，范围风险累积使用平面 31；四行 A 缓存占平面3至15，范围最大值工作区占平面2。它们与批量 y 缓冲、恢复辅助区互不重叠。行点积恢复使用独立 UB，不覆盖四行 A 缓存。

所有路径均为带宽分段或固定 UB tile，不构造 n×n 矩阵，不分配调用级 GM workspace，不需要全局原子或跨核归约。函数异步返回，读取 y 前由调用方同步绑定 stream；库内不逐次同步 stream。

### 3.2.2 Kernel 侧设计

#### packed 的单侧 A 窗口与双侧 x 窗口

设本 tile 的逻辑输出行区间为 `[r,r+R)`，`w=min(k,n-1)`：

| 数据 | 搬运列区间 |
| --- | --- |
| x | `[max(0,r-w), min(n,r+R+w))` |
| UPPER 的 A | `[r, min(n,r+R+w))` |
| LOWER 的 A | `[max(0,r-w), r+R)` |

完整列模式中，A 只需输出行及存储侧的一个 halo，因为 UPPER 的实际存储列为 `max(i,j)`，LOWER 为 `min(i,j)`。A 按完整 lda 列连续搬运；DMA 可复制已分配数组内的 padding/无效角原始字节，但 Gather 和算术只使用定义的矩阵元素，不要求无效区域提供有限数值。

用 `aBegin` 表示 A 窗口起点、`xBegin` 表示 x 窗口起点，核内坐标相对 aBegin：`rowBase=r-aBegin`、`xBias=aBegin-xBegin`。x 的合法相对列为 `[-xBias,xColumns-xBias)`；其记录索引为 `col+xBias`，负 incx 时再变为 `xColumns-1-col-xBias`。这些相对坐标受 UB 容量约束，为 FP32 可精确表示的小整数；GM 总体地址仍在 64 位整数域计算。

UPPER 的 A 相对位置为 `max(row,col)*lda+k-abs(row-col)`，LOWER 为 `min(row,col)*lda+abs(row-col)`。原始 k 始终保留。未满足有效掩码的 lane 改为指向自身实对角，避免越界或引用无效角；最终乘积再 Select 为零，防止 `0*Inf/NaN` 进入归约。对角虚部的 Gather 指向实部槽，绝不把物理对角虚部作为数值输入。

A 窗口容量为 `capA=min(n,tileRows+w)`。仅当 incy>0、w 和 capA 都为四列倍数、且每核需多个 tile 时启用环形复用；否则每 tile 一次完整连续重载。复用时随窗口起点推进 ringOffset，只搬入新进入的列，回绕最多分为两段 DMA。四列 complex64 边界对任意 lda 都是 32B 对齐；尾部自动 padding 留在当前窗口的空闲环形区内。负 incy 不启用该缓存。

x 每 tile 重载。单位正负 incx 一次连续 DMA，UB 记录间隔 8B；其他步长按物理升序搬入 32B record，逻辑负序通过 Gather 索引恢复。两种布局共用记录间隔参数，x 不进行会改变 subnormal 位模式的预变换。

#### compact 的完整组内列与 halo 条带

设本 tile 实际逻辑行区间为 `[g,g+r)`，1<=r<=R。始终以 g 作为 A 坐标原点，rowBase=0、xBias=g-xBegin；x 仍覆盖双侧 halo。先连续搬运真实存在的 r 个完整 A 列 `[g,g+r)`，但空间按固定 R 个列预留并 round8。UPPER 外条带列为 `[g+r,min(n,g+r+w))`，LOWER 为 `[max(0,g-w),g)`。对外条带列 c，只需存储对应输出组内的 r 个复数，固定 record 为 stripFloats；不再把 halo 的完整 lda 列全部复制。

UPPER 的有效组内行为 `j0=max(0,c-g-w), j1=r`；LOWER 为 `j0=0, j1=min(r,c+w+1-g)`。其 GM 首复数索引分别为 `c*lda+k+g+j0-c` 和 `c*lda+g+j0-c`。完整条带可用一条 2D DataCopyPad：blockLen=r*8 字节，srcStride=(lda-1-r)*8 字节，dstStride=(stripFloats-round8(2*r))/8 个32B块。每端不足 r 行的三角边界按实际有效长度复制；UB 起点先按 8 float 对齐，leftPadding 只保留 `2*j0 mod 8` 个 float，不把 padding 单位误写为字节。尾补齐均留在当前 record 内。

Gather 先按 UPPER 的 max(row,col) 或 LOWER 的 min(row,col) 选实际存储列。落入 `[g,g+r)` 时使用完整列索引；否则使用 `fullFloats+(c-haloBegin)*stripFloats+2*(row-g)` 的 float 偏移。最终 Select 只保留合法非负索引，再转换为字节偏移，原始 k 仅在完整列/GM 带坐标中使用，不被裁剪带宽替换。无效 lane 仍指向自己的实对角并在乘积后清零。

compact 不使用环形复用。每 tile 在 V_MTE2 后重载 A/x，输入 Gather 前 MTE2_V 收束。span=2048 时 R=2，单核可能分两次写同一个32B物理输出块；两次 Store 已按 MTE3 依赖串行，同一块不会跨核共享，因为核行块起点仍按四行对齐。此处是资源和所有权设计，实际 DMA 与数值行为仍需设备验证。

#### packed 乘加、归约与独立对角

同算路径以 `d=lane-w` 展开每行，保留带内和矩阵边界掩码；独立对角路径以 `d=lane-w+(lane>=w)` 覆盖负、正带区各一次，不含零位移。独立对角只在上述资源收益条件成立时启用，不针对特定输入配置。

四个输入分量 Gather 完成后，坐标平面死亡。先生成 `ar*xr` 与 `ar*xi`，再复用平面形成 `ai*xr`、`ai*xi`，按共轭符号合成实虚乘积。同算路径的对角交叉项在乘法后清零，保留实对角贡献；独立对角路径不需要这两次 Select。无效 lane 均在乘积阶段清零。

span 不超过64时按行 WholeReduceSum；128先生成两个64-lane部分和再PairReduceSum；256生成四个部分和，以两级PairReduceSum合并。compact span512/1024/2048先生成每行8/16/32个部分和，再以第二级WholeReduceSum按行归约，源repeat stride为1/2/4个32B块；行范围最小值使用同样两级WholeReduceMin并显式ORDER_ONLY_VALUE。所有跨度首级最多64个部分和/repeat，向量比较按 64 float 对齐，比较的额外源 lane 先初始化，归约和写回按真实有效行数执行。Gather 的偏移单位为字节；PairReduceSum 的 repeat/mask 参数顺序与 WholeReduceSum 不同，按正式 SDK 签名调用。

独立对角在非对角归约结束后执行，此时四个输入平面可复用。只 Gather `A(i,i).real`、x 实部及虚部，合并同样的幅值/下溢守卫，再将 `ar*xr`、`ar*xi` 加到该行实虚和。w=0 直接写出这两个乘积和行守卫。完成 alpha/beta 缩放与必要恢复后，按 incy 重排并写回实际逻辑元素。

#### 行点积、x cache 与批量 y

通用行点积将带区分为两侧，固定存储列的一侧连续搬运，跨列一侧使用按 lda 推导的等间距 DMA。各段完成解交织、共轭和复数乘法，乘积在 UB 中跨段累加，整行完成后统一归约。单位物理步长采用连续 DataCopyPad，其他步长使用 8B 有效值加 32B UB record；DMA stride 字段不能容纳合法地址间隔时降为逐 record 搬运，保持 64 位地址，不截断或拒绝合法参数。

缓存路径每核一次复制完整 x 的原始物理存储，保持位模式，负单位步长通过索引反转。x 在缓存后先扫描范围风险，只有首轮检出风险才进行忽略精确零的第二次确认；其风险标志保守影响该核的所有行，即使某个异常 x 不属于该行带区也可能触发恢复。后续各段只重复检查 A 的幅值范围，避免为每个输出行重新扫描相同 x。通用路径则在每段检查实际使用的 A/x。

每行先将实、虚 lane 累加平面清零，将范围累积平面置为中性值 1。带区两侧按 TILE 分段，各段独立执行 Gather、对角处理、共轭、范围守卫和复数乘法；乘积按段内 lane 累加到持久实、虚平面。段尾不读取标量归约结果，待整行全部段完成后才归约实和、虚和及范围风险，并通过一次 `V_S` 同步读取结果。短段只更新真实有效 lane，其他 lane 保留先前段贡献；所有实际带内元素均恰好参加一次。

整行 lane 累加改变了相对于逐段标量求和的 FP32 加法顺序，精度验证覆盖该归约顺序。最小值守卫在整行范围内累计；缓存 x 的首轮扫描每核执行一次，只有风险时再确认，后续段只检查 A。alpha/beta 位模式判断、对角实部语义、beta=0 不读旧 y、beta=(1,0) 直接加旧 y，以及触发风险后的恢复流程均适用于该路径。

单次归约采用正式 arch22 SDK 的 `WholeReduceSum` 和 `WholeReduceMin(..., ORDER_ONLY_VALUE)`。有效元素不超过 64 时直接归约；较长向量先按 64 lane 生成最多 16 个部分结果，尾段写入单独的 32 B 对齐槽 16，再归约前 17 个槽。未使用槽预填求和中性值 0 或求最小值中性值 FLT_MAX，避免尾部及部分结果之间的空槽污染。`ORDER_ONLY_VALUE` 下 `dstRepStride=1` 表示相邻一个 float 的输出间隔，源 repeat 仍按 SDK 的 32 B 单位设置。部分结果的生产、消费及 scratch 复用之间保留 `PIPE_V` 屏障。

两类行点积都按物理输出顺序批量管理 y：缓存路径每 chunk 最多 96 行，通用路径最多 128 行，分别复用 `TILE` 个 float 的 out 缓冲。beta 非零时先批量搬入旧 y；单位 incy 使用 8B 紧凑记录，非单位使用 32B 记录。每行从自己的 UB 槽读旧值并写结果，后续未处理槽保持原样；一个 chunk 只做一次批量写回与对应同步。负 incy 只改变逻辑行映射。beta=0 不读旧 y；此处是内部输出批处理，不新增公共 batch API。

#### 四行 A 共享缓存

跨矩阵列的一侧按单行搬运时为每个输出行搬入一个 complex64，并在 UB 内补齐为一个 32 B record。四行共享缓存将相邻四个逻辑输出行在同一矩阵列中的有效元素合并搬运，使一个完整 32 B record 同时服务四行；固定存储列的一侧继续使用连续搬运。数学计算、守卫与恢复仍复用行点积实现。

启用条件为：已经选择完整 x 缓存路径、`w>=4`、`w+4<=1248`、`lda>4`，且 `(lda-5)*8` 能由 DMA 的 uint32 stride 字段表示。每组必须正好有四行；不足四行的末组、容量不足或 DMA 字段不可表示时均回退单行搬运路径。行数 4 来自 `32 B / sizeof(complex64)`，条件由资源及 DMA 能力推导，不限定某个问题尺寸或性能用例。

设本组逻辑行 `[g,g+4)`，列编号为 c，组内行编号为 j。缓存仅覆盖行点积的跨列存储侧；原始 k 继续用于 UPPER 地址，不能用裁剪后的 w 替换：

| 模式 | 缓存列区间 | 列 c 中有效组内行区间 `[j0,j1)` | 首个有效复数在 A 中的元素偏移 |
| --- | --- | --- | --- |
| UPPER | `[g,min(n,g+4+w))` | `[max(0,c-g-w), min(4,c-g+1))` | `c*lda+k-(c-(g+j0))` |
| LOWER | `[max(0,g-w),g+3)` | `[max(0,c-g+1), min(4,c-g+w+1))` | `c*lda+(g+j0-c)` |

UPPER 缓存包含对角所在 record，LOWER 的跨列侧不含对角。物理对角虚部可能随完整复数搬入 UB，但行点积的对角处理保证它不作为数值输入，不改变被忽略虚部为 NaN/Inf 时的语义。矩阵边界和带角处只搬入真实有效元素，padding 槽不会被有效 Gather 引用。

完整四行列区间内，每列只需 `DataCopyPad` 的 32 B block；相邻列首地址相差 `(lda-1)*8` 字节，因此 block 间 GM gap 为 `(lda-5)*8` 字节。区间两端各最多三列不完整，分别搬入 `8*(j1-j0)` 字节，并以 float 元素数设置左 padding `2*j0`、右 padding `2*(4-j1)`。缓存记录恒为 32 B，后续行访问使用 `32*(c-cacheBegin)+8*(row-g)` 的字节基址，再分别 Gather 实部及基址加 4 B 的虚部。

每个物理输出 chunk 按四行分组。正 incy 的逻辑组起点为当前物理组起点；负 incy 取该物理组对应的最小逻辑行作为 g，标量行循环仍可按逻辑逆序访问缓存。g 不要求是四的倍数。alpha 两个分量均为零时不调用缓存 Prepare；判断采用去符号零位模式，subnormal 非零 alpha 不会被误判。测试继续遵守任务书关于 alpha=0 不与非有限 x 组合的限制。

有效缓存组的跨列存储侧必须始终读取 A 缓存；另一侧连续 GM 搬运每段最多 `2*TILE` 个float，因而不覆盖平面2及缓存。缓存禁用或不足四行时，Prepare先将valid置false，原始带stride的A搬运可以使用前8个平面并覆盖旧缓存和幅值区；两个A分量Gather完成后，才允许幅值工作区覆盖这些已消费的原始字节。下一有效组重新完整加载。正常Segment和仅范围确认Segment都在幅值Compare结束后保留V_MTE2，再进入下一段DMA。

缓存重载前以 `V_MTE2` 等待旧组 Gather 完成，重载后以 `MTE2_V` 保证新组可见。恢复辅助区独立，不需要像 packed 借用 A slab 时那样使该缓存失效。共享缓存减少的是跨列小 record 的重复搬运；持续存在的连续侧搬运、四行各自的复数算术和范围守卫不因此消失，实际性能收益须由 msprof 验证。

#### 数值范围守卫与指数桶恢复

仅检查最终 Inf/NaN 不能发现“中间溢出而缩放后有限”“极小乘积被大 alpha 放大”或正常有限乘积相消丢失残差。快路径因此同时检测有效输入的最小和最大绝对值、标量位模式及输出指数。

一般最小幅值阈值为 `2^-126`；当 alpha 任一分量绝对值大于 `2^32` 时采用更保守的 `2^-63`。完整列 packed 保留实际零触发保守恢复；compact 和行点积增加位模式零识别，避免把精确零误当下溢输入。最大幅值门槛 `OperandLimit` 由 alpha/beta 的指数推导：矩阵乘积考虑两个 tensor 因子，beta*y 考虑一个因子；它约束局部复数乘积的舍入影响。packed 使用独立 min/max 累积平面，每个输入的 Abs 后并行更新两个累积量，最后一次 Compare/Select 将超限标记为零哨兵，再排除无效 lane。

compact 在每个原始输入被浮点 Abs 处理前，以 `(bits<<1)==0` 的整数比较识别正负零。仅将该操作数下界中的真实零替换为中性1，再与其他操作数取Min；原始绝对值仍进入Max。最大幅值/非有限的零哨兵在全部零归一后设置，无效lane最后置中性值，防止将风险哨兵误抹成1。被忽略的ai对角已指向ar实槽，其Max只重复同一个合法实值；其Min另设中性1。不会用可能flush subnormal的浮点EQ0替代位判断。

行点积正常轮次保持原有FP32乘加顺序。若行范围最小值触发下界风险，先保存已求得的实虚和，再重跑仅检查范围的模板：逐原始操作数忽略精确零，不重新累加乘积。确认仍有风险才进行指数桶恢复；缓存x使用同样首轮/确认两阶段，正常输入不无条件执行位分类。该流程的索引scratch在Gather后已死，真实最大幅值mask独立保留，不覆盖持久实虚和、范围累积、A缓存或批量y。

alpha/beta 的 subnormal 或大幅值、相关旧 y 的风险，以及输出非有限指数同样触发检查。beta=(1,0) 不再执行 beta*y 复乘。packed 先判断整个 tile 的风险，只在需要时逐行进入恢复；不以浮点 Max 是否保留 NaN 作为正确性依据，最终非有限检测直接检查指数位。

`ChbmvCompensated` 使用独立 256 元素段、56 KiB UB，把有限 binary32 解码为符号、整数有效数及二进制指数，采用字节分解乘积、指数桶归约与有符号整数合成，保留跨指数范围的残差；随后合成 alpha*Ax 与 beta*y 并构造最终 binary32。tensor 搬运和部分乘积仍在 AIV 上向量化，标量操作用于桶、归约结果及行控制，不是 Host 计算回退。实际引用的输入包含 NaN/Inf 时，恢复可以返回“不替换”，保留快路径特殊值结果；被忽略的对角虚部不参加检查。

**这些局部幅值守卫不是任意长度点积误差界，也不是任意输入均正确舍入的证明。** 指数桶恢复用于已识别风险，不能据它的存在推断所有安全分支自动满足误差门槛。最终仍须用独立 FP64 golden、长点积/相消/极端值回归及全量验收验证；发现新的有限误差反例应修改通用算法或守卫，不能放宽容差或按 case 特判。

#### 流水、API 与架构依据

packed 中旧 y 最小幅值、输出最大指数和行风险最小值等短归约也使用低级 WholeReduce API。无读写依赖且目标分离的归约可以顺序下发，随后统一等待结果；不删除存在 RAW 依赖的屏障。事件只在对应生产者、消费者之间成对设置和等待；下一次 DMA 覆盖 raw 区之前仍保留 `V_MTE2`，以保护上一段 Gather 和范围检查。

参考 CANNBot 的 `ascendc-performance-best-practices` 中 arch22 代际边界、通用 Scalar 分析，以及 `ops-profiling` 的 `msprof-op-guide`；其建议不覆盖任务书明确的采样协议。实现能力以正式 CANN 9.1.0 实际 SDK 头文件与构建结果为准，使用 `DataCopyPad`、`Gather`、`CreateVecIndex`、`Cast`、FP32 向量算术、`Compares/Select`、`WholeReduceSum/Min/Max`、`PairReduceSum`、`ReduceSum/Min/Max`、`TPipe/TBuf` 和类型化硬事件。

当前 kernel 为 AIV_ONLY，仅使用 TBuf，不使用 TQue 或 AllocEventID；显式事件固定使用 ID0，每次 SetFlag/WaitFlag 就地成对消费。该选择依赖当前事件生命周期，未来引入队列、延后等待或改变事件分配方式时须重新审查。packed 入口统一执行一次 `ICachePreLoad(2)`，按正式 arch22 SDK 的单位预取 4 KiB 指令；参数不随 case 变化，接口语义及可用性以正式 SDK 为准。

MTE2→V、V→MTE3、MTE3→V，以及标量读写与 DMA 的依赖使用成对 SetFlag/WaitFlag；向量 RAW/WAR 依赖和平面复用用 PIPE_V 收束。行点积 batch y 的 MTE2→S、S→MTE3、MTE3→S 同样明确。UB 比较尾、归约有效 lane、负向索引和 32B DMA 边界均纳入验证。

本实现不使用 GlobalTensor 单元素 GetValue/SetValue 算术循环，不走仅适用于 arch35 的 Reg/MicroAPI/VF 或 DSL 路线。若今后更换 SDK、SoC 或 API，需要重新核验可用性、对齐和资源预算。

### 代码与构建落位

```text
include/cann_ops_blas.h                     复用统一公共 API 声明，保持 ABI
blas/sbmv/README.md                         扩展 A2/A3 支持，保留 Ascend 950PR
blas/sbmv/arch22/chbmv_host.cpp             参数校验、tiling、stream 下发
blas/sbmv/arch22/chbmv_kernel.cpp           行点积/cache/batch y 与模板 launch
blas/sbmv/arch22/chbmv_packed_kernel.h      完整列/compact模板、AIV计算与资源布局
blas/sbmv/arch22/chbmv_band_cache.h        四行共享 A 缓存、能力条件与 DMA/Gather 地址
blas/sbmv/arch22/chbmv_compensated.h        范围风险与指数桶恢复
blas/sbmv/arch22/chbmv_tiling_data.h        Host/Device 共享参数
test/sbmv/chbmv/CMakeLists.txt             按 arch22/arch35 注册各自测试
test/sbmv/chbmv/arch22/chbmv_param.h       arch22 CSV 参数解析
test/sbmv/chbmv/arch22/chbmv_golden.h      arch22 Netlib 双精度参考
test/sbmv/chbmv/arch22/                    CSV、GTest、NPU wrapper、README
test/sbmv/chbmv/verify_accuracy.py         精度证据与完整性校验
test/sbmv/chbmv/profile_performance.py     msprof op 采集与汇总
```

共享 `test/sbmv/chbmv/chbmv_param.h` 与 `chbmv_golden.h` 保留上游版本供 arch35 使用；本任务的解析和 FP64 参考放在 `arch22/` 同目录，避免一套 helper 悄然改变另一架构的测试语义。测试根 README 提供架构导航，保留原 Ascend 950PR 说明，并链接 arch22 专用步骤。

实现文件随仓库 `blas/CMakeLists.txt` 自动发现；测试按架构选择相应目录，以 `ops_blas_add_gtest_tests()` 注册。遵循公共 helper、日志与资源管理约定，使用仓库构建入口选择 `chbmv` 与 `ascend910b`，A3 使用映射到 arch22 的 `ascend910_93` 配置。精确构建和测试命令在测试 README 固定，并核验实际共享库符号、测试程序动态链接库及安装后调用。

## 支持硬件

| 产品 | 实现目标 | 自验证计划 |
| --- | --- | --- |
| Atlas A2 系列，含 Atlas 800T A2 | 支持，arch22 | Ascend910B3 v16 开发轮 2262 项精度 PASS，24 条性能子集 PASS；合并提交的完整 200 条正式性能与 fresh 复现仍 PENDING |
| Atlas A3 系列，含 Atlas 800T A3 | 支持，arch22 | 编译适配；任务允许仅一种款型实机自验证，不伪造 A3 实测 |
| Ascend 950PR | 保留上游已有 arch35 实现 | 不属于本次新增实现或实机验收范围；不将 A2/A3 结果视为其验证证据 |

## 算子约束限制

本 arch22 实现仅支持任务定义的 complex64 列主序带存储和 lda/incx/incy 步长；不声明额外 Tensor 视图、广播或混合精度。k>n-1 合法，但仍以原始 k 校验 lda 和定位 UPPER 主对角。错误码、n=0 优先级及 beta=0 语义不因优化路径改变。设备内存容量仍由调用方负责；非法跨度用例必须在数据读写前拒绝。

# 可维可测分析

## 精度标准与测试方法

本节仅规定 Atlas A2/A3（arch22）的任务验收。已有 Ascend 950PR（arch35）测试继续使用其原有参考实现及测试协议，不共用本节的场景计数或精度证据。

使用 ops-blas 的 C++ GTest CSV 框架执行原始 1000 条精度 case 和 1256 条补充精度 case，另保留全部 200 条性能 case。当前 API/比较合同集合为 6 个 GTest，精度入口应枚举到 2262 个测试；`FiniteDynamicRange` 内含 95 个动态/资源边界子场景，不能把它们再计为95个额外GTest。集合从当前 `verify_accuracy.py::CONTRACTS` 读取，门禁逐项核对名称、状态、退出码和输出，不只比较总数。负向 case 仅配置最小 dummy 缓冲，避免测试框架在非法大步长上提前申请巨量内存；handle 空指针、判空优先级和 n=0 组合由 API 专项单测覆盖。

golden 使用 Netlib BLAS 3.10.0：输入 complex64 精确提升为 complex128，经 `cblas_zhbmv(CblasColMajor,...)` 执行与 CHBMV 相同的 Hermitian 带状运算，最后实部、虚部分别收窄到 float32。`cblas_chbmv` 本身不能执行 FP64 运算，采用其双精度对应接口正是随包 README 对任务“fp64 真值”的解释。不得用 NPU 输出或自写同算法参考替代该独立 golden。

均匀输入和正态输入按任务要求补齐：保持原始 CSV 及 200 条基线配对不变，对原始随机精度配置补充等量正态变体，实虚部独立、mu∈[-5,5]、sigma∈[0.1,2]，使用固定 seed 和 `GAUSS_NORM_5_2` 标识。均匀/正态随机生成输入按 50%/50% 组织；特殊值、确定性填充、非法输入另计。报告写明补充数量、两种分布覆盖以及 seed，不将分布选择与生产 kernel 分支绑定。

全量 n 个逻辑输出按实部/虚部分别统计，使用任务指定的[混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)：

- 有限值逐元素 `abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)`；每分量 matched_ratio >= 0.99。
- 每分量最大绝对误差同时满足 `max_abs_error <= max(1e-2,32*ULP(golden_at_max_error))`，ULP 在该分量最大误差点的 golden 上动态计算。
- 非有限值单独计数：同为 NaN、同号 Inf，或 NaN 与任意符号 Inf 组合等效；有限与非有限不等效，异号 Inf 不等效。非有限点不进入有限值统计，全部非有限点须通过；无有限点时只判非有限规则。
- CSV 的 MERE/MARE 保留字段不替代或放宽上述门槛。

补充检查包括对角虚部放置非零/NaN、带外和 lda padding 污染、beta=0 旧 y 非有限值、正负步长间隙哨兵、任意非样本 shape、合法超宽带、重复调用及非默认 stream。测试输入恢复和 oracle 对齐以每次调用的初始 y 为准。

最初19个动态数值子场景覆盖：中间溢出后缩放、alpha*Ax 与 beta*y 大项相消、乘积下溢后放大、subnormal 输入、正常有限乘积的残差丢失、subnormal alpha/beta、beta=(1,0) 对旧 y 非有限分量的处理，以及有限 beta*y 相消；相关场景还覆盖正负非单位步长与间隙哨兵。alpha=0 场景的整个 x 存储（含间隙）保持有限。另保留12个带精确零、非零subnormal或超限复数相消的rowdot子场景，以足够大lda保证资源回退；相同构造增加12个compact子场景。52个随机资源边界子场景覆盖完整列/compact/rowdot切换、八种compact模式、A尾x、联合预算拒绝、正负与非单位步长、单行尾、k>n和lda padding。新增边界分别检查完整x缓存的n=4096/4097，以及四行A缓存最后可用带宽和超预算回退。Combined2048由n超过完整x缓存容量的输入继续覆盖；不声称模板模式与全部步长组合穷举。合计19+12+12+52=95；原CSV、seed和阈值不变，非零subnormal及相消残差还做严格FLOAT_EQ检查，避免通用绝对容差掩盖flush到零。

正式精度入口为 `python3 test/sbmv/chbmv/verify_accuracy.py --device <逻辑设备> --build-log <本次clean-build日志> --output-dir <新证据目录>`。结果同时检查进程返回码、GTest XML、日志、每分量结构化统计和 actual/golden 原始向量，运行前后绑定源码、二进制、实际加载库及输入哈希。CPU 地址模型和精确整数构造用于检查索引/别名/分派，不能替代这条实机门禁。

v16 开发轮 clean build 与全部 2262 项精度已 PASS，其中包含上述 95 个内部子场景；日志/XML 复核和运行身份检查通过。该运行来自合并上游前的开发源码快照。当前发布提交已整合 arch35 支持并调整测试 helper 所在目录，不能将开发构建改标为合并后的 fresh 结果。正式复现须从 `bab0239b3d617623770b9c6966e32e875052ee02` 的新 checkout 和空构建目录开始，重新绑定全部源码、构建日志、二进制、实际加载库及输入；该完整复现仍为 PENDING。

## 性能标准、采集与优化

本节门槛与采样流程仅适用于本次 Atlas A2/A3（arch22）任务；不改写既有 Ascend 950PR 任务的独立性能门槛。以本任务随包 A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4 的全部 200 条基线为比较对象。原始 CSV 与基线保持不变；按基线包含的完整参数键 `(uplo,n,k,lda,incx,incy)` 唯一关联。alpha/beta、填充规则与其他输入字段随原始 CSV 哈希和调用日志绑定，并验证 beta=0 的采样前提；统一将 GPU 毫秒转换为微秒。

每例都须满足 `NPU_avg_us <= 1.25*GPU_us`，即 `GPU_us/NPU_avg_us>=0.8`。平均加速比、代表形状或多个版本各自通过的子集不能替代单一最终候选的 200 例逐项门禁。

正式测量使用 `msprof op` 的 kernel task duration。每例应用先执行 5 次 warmup，再连续执行 10 次采样；API 返回成功、同步成功且 task 记录可唯一关联时才是有效样本，最后取算术平均。工具显式记录 `--warm-up=5`、`--launch-skip-before-match=5`、`--launch-count=10` 与实际 replay 模式；额外 profiler 预热或重放不扩充有效样本。日志必须保留每次调用的 phase/index/status，capture 0..9 对应应用采样 launch 5..14。

全量默认使用 BasicInfo，逐 case 独立输出；Default 等深度指标若分多轮 application replay，每轮都核验 5+10 成功调用，只按相同 launch 身份取得 10 条时长，不能将多轮重复记录累加为更多样本。异常退出、缺少 task、重复捕获、混入其他 kernel 或状态不完整均不得判 PASS，不择优删除慢样本。H2D/D2H、造数与 golden 不计入 kernel duration。随包 event 计时与任务书冲突时，以任务书的 msprof op 为准。

原始 200 条性能输入的 beta 为零，采集工具先验证此前提，因此连续调用不依赖或递推旧 y。扩展到 beta 非零的测量时必须恢复同一初始 y，另行验证测量协议。当前公共调用只启动一个目标 kernel；若未来拆成多个，必须计入完整 API 的设备任务，不仅采最快 kernel。

正式入口为 `python3 test/sbmv/chbmv/profile_performance.py --device <逻辑设备> --output <新证据目录> --metrics BasicInfo`；不指定 case 子集时采集全部 200 例。续跑仅允许输入、源码、二进制、实际加载库、SDK、设备与测量配置身份一致，重新从原始日志和 CSV 核验已完成结果，拒绝陈旧数据。分析保留原始 OpBasicInfo、工具版本、完整调用映射、参数键、样本数、退出码和结构化结论。

优化以逐核 PipeUtilization、MTE、Vector、Scalar、Memory、L2Cache 和指令缓存等可用证据区分瓶颈；流水 busy 比例不能简单相加换算耗时。当前架构分别减少重复 A/x 搬运、展开补齐 lane、按行重复的控制以及 y 小块写回，资源收益由通用分派判据表达。任何变更先回归相关语义，再以同一构建身份重新执行全量门禁。

v16 开发轮的 24 条性能诊断子集全部 PASS，采样共 240 条原始 task duration；仍未完成本候选全部 200 条正式门禁。v12 旧正式完整 200 例为 184 PASS、16 FAIL，总体 FAIL，不与新子集拼接。当前合并提交 `bab0239b3d617623770b9c6966e32e875052ee02` 的 fresh 完整精度与 200 条正式性能为 PENDING。正式结果须绑定同一发布源码、二进制、实际加载库、输入及测量身份；CPU 模型、历史构建和性能子集均不替代最终验收。

## 资源、兼容性与风险

| 风险 | 检测与措施 |
| --- | --- |
| UPPER k>w 时对角定位错误 | 原始 k 与有效 w 分字段；k>n-1 专项回归 |
| 整数跨度与 GM 索引宽度 | Host int64 校验，GM uint64/int64；INT32_MIN/INT32_MAX 边界用例 |
| 尾部、对角虚部或旧 y 污染 | 有效区裁剪，对角单独处理，beta=0 跳过读取；污染/哨兵测试 |
| 负步长错误或跨核覆盖 y 间隙 | 固定逻辑起点，独占输出行，只写逻辑元素；全缓冲哨兵比对 |
| UB 分配或 DMA 字段越界 | 15488-float 活动A或A+x联合区、对齐strip与独立max平面、四行缓存1248列及 DMA stride 守卫、184 KiB 上限与 64 位容量校验；对齐尾块和近容量 lda 回归 |
| 数据竞争或缺少同步 | 事件明确生产消费关系；重复调用与不同 stream 测试 |
| FP32 累加、相消或非有限传播差异 | 局部守卫不充当完整误差证明；独立 FP64 golden、95个内部动态/资源场景与长点积回归，失败不得放宽门槛 |
| 小规模固定开销、宽带不连续搬运 | msprof op 度量后优化通用核数与块布局，不只优化典型四例 |
| 环境与验收版本不一致 | 报告固定实际 CANN/driver/SoC，区分探索结果和正式验收 |
| 共享接口、构建或测试 helper 影响已有架构 | 统一公共声明与导出符号，保持复数结构/handle ABI；按架构选择实现与测试，保留上游 arch35 源码及其 helper；执行相关集成回归 |

任务书 §3.4 内存专项为“不涉及”，交付模板中的内存报告和日志仍保留并写 N/A 及依据，不制造内存门禁 PASS。实现自身的越界防护、UB 容量和资源释放仍必须验证。

## 交付与状态管理

最终提交包括 arch22 Host/Kernel/共享 tiling、统一公共声明的兼容性整合、算子 README（增加 A2/A3 并保留 Ascend 950PR 支持）、arch22 GTest/CSV/参考实现、可复现步骤、结构化精度与性能结果以及任务要求截图。个人代码仓 `task_submission/` 使用任务模板名称：

```text
1 自验证步骤说明.md
2.1 精度自验证报告.xlsx
2.2 精度自验证日志.log
3.1 性能自验证报告.xlsx
3.2 性能自验证日志.log
4.1 内存自验证报告.xlsx
4.2 内存自验证日志.log
```

报告绑定代码提交 SHA、输入 manifest 哈希、公开设备型号、实际工具链、命令、退出状态及原始证据。PASS/FAIL/SKIP/TIMEOUT/PENDING/N/A 分开记录。最终交付只引用同一发布源码及构建产物的完整门禁；本设计稿不发布尚未完成的验收结论。

设计 PR 合入、全部自验门槛、代码 PR/仓库 CI、任务平台验收材料和要求的仓库开发者邀请分别跟踪。平台提交、账号权限变更以及维护者评审/合入须遵循对应授权和实际状态，不能以文件已上传代替。生产代码 PR 在测试达到任务门槛后提交，按仓库贡献指南触发和核验 CI；不自动批准或合并 PR。
