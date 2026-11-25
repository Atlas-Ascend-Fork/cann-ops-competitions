# aclblasChpmv（Atlas A2/A3）设计文档

# 需求背景（required）

## 需求来源

11 月社区任务《aclblasChpmv_AtlasA2_A3_task_doc.md》，团队 `gcw_EzaQN38n`。实现与测试分别位于 `cann/ops-blas` 的 `blas/spmv/arch22/` 和 `test/spmv/chpmv/arch22/`，目标分支 `master`。

## 背景介绍

接口完成单精度复数 Hermitian packed 矩阵向量乘 `y = alpha*A*x + beta*y`，与 cuBLAS `cublasChpmv` 的布局和数学语义对应。复用 ops-blas 已声明的句柄式接口、stream、workspace 和构建测试框架。

任务书描述 arch22 尚待补齐；开发时读取的上游基线 `55ef791b77f3677b45da3468b0a8b17941424f84` 已包含 arch22 的两阶段 SIMD 实现。本任务在该实现上补齐任务契约并验证，不重复引入平行接口。初步核查发现：`n=0` 过早返回而遗漏结构检查；`INT_MIN` 步长被拒绝；Host 直接解引用 alpha/beta；现有 CSV 和输入分布没有完整覆盖本任务的 1,000 条精度及 200 条性能用例。

# 需求分析（required）

## 需求描述

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

`aclblasComplex` 沿用公共头文件，实部和虚部均为 float32。AP 包含 `n*(n+1)/2` 个复数；向量物理长度分别至少为 `1+(n-1)*abs(incx)`、`1+(n-1)*abs(incy)`。不增加声明，不接受广播或任意视图；y 原地更新且不得与 AP/x 重叠。alpha/beta 支持 Host 与 Device 地址。

## 需求拆解

1. 按任务顺序检查：handle → n<0 → uplo → incx/incy 非零 → alpha/beta 非空 → n=0 返回 → AP/x/y 非空 → 数值快速返回。非法枚举用 INVALID_ENUM；空句柄用 HANDLE_IS_NULLPTR；其余结构错误用 INVALID_VALUE。
2. 支持 UPPER/LOWER 列主序 packed、Hermitian 共轭镜像、忽略对角虚部、正负步长和完整非零 int 步长域。偏移计算先提升为 64 位，避免 `-INT_MIN` 的 int 溢出。
3. `alpha=0,beta=1` 保持 y；`alpha=0` 不引用 AP/x 内容；`beta=0` 不读 y 旧值，包括 NaN；`n=0` 不访问张量或发射 kernel。
4. CANN 9.1.0，A2/A3（DAV_2201、arch22、设备编译宏 2201）SIMD 路线；自验在 Ascend910B3 完成，A3 兼容性由同代代码与构建目标保证，未上板不能声明 A3 已实测。

# 详细设计（required）

## 算子分析

### 数学公式与布局

UPPER：`AP[i+j*(j+1)/2]=A(i,j), i<=j`。LOWER：`AP[i+j*(2*n-j-1)/2]=A(i,j), i>=j`。另一半使用共轭；对角只用实部。负步长逻辑元素 i 对应 `(n-1-i)*(-int64_t(inc))`，正步长为 `i*int64_t(inc)`。

只支持 complex64。实虚分离后使用 float32 SIMD 乘加/归约，不使用 FP16 矩阵乘替代精度域。packed 三角的列长不等，按交错列分配平衡各核工作量。

## 算子实现

### Host 侧设计

保留现有 API、日志、GetAivCoreCount 和有效 workspace 管理。修正参数校验顺序，`n=0` 在检查结构参数后直接返回。Host 标量读取入 tiling；Device 标量仅探测位置并把地址和位置标记传入 kernel，在绑定 stream 上读取，避免为了读标量而同步 Host。混合 Host/Device 地址同样支持。位置探测失败及 kernel launch 错误按仓库错误码传播。

Host 标量已知时进行 no-op/scale-only 分流；Device 标量在 kernel 内判断相同数学分支。Device 标量的读取与先前同 stream 写操作有序，结果仍异步输出。快速返回不跳过任务规定的 AP/x/y 指针检查。

### 按规模分派、Tiling 与 workspace

当前实现按规模选取三条数值路径，全部保持公共API、标量位置、packed布局和步长语义。

| 范围/条件 | 当前实现 | kernel与分核 |
| --- | --- | --- |
| n=0；或Host alpha=0、beta=1 | 结构校验后的no-op | 无kernel |
| 1≤n≤128 | `ChpmvSmallAIV`；packed AP与x装入UB，输出行由核独占 | `chpmv_small_arch22_kernel`；`min(n,AIV核数)` |
| 129≤n≤4095，一般alpha | 列分核累加与行分核归并 | `chpmv_accum_arch22_kernel` + `chpmv_reduce_arch22_kernel` |
| 129≤n≤4095，Host alpha=0 | 只缩放y | `chpmv_scale_arch22_kernel` |
| n≥4096 | `ChpmvOrderedAIV`；行块独占，顺序两链SIMD | `chpmv_ordered_arch22_kernel` |

小规模常规路径Gather完整行后执行复数乘积与归约；当 `max(abs(x))>1e30` 或该最大值为NaN时，采用UB内有界顺序求值分支，保留参考运算顺序处理极端值。判断依据为实际输入数值，不依赖用例名或随机种子。

中等规模仍以交错列平衡三角工作量，每核写私有实虚部分和，归并核合并并应用alpha/beta。两个kernel位于同一stream，提供全局依赖，无跨核原子写y。`nPad=align_up(n,8)`，累加核数为 `min(n,AIV核数,workspaceBytes/(nPad*8))`，使用handle的 `accumBlocks*nPad*8` 字节workspace；容量不足返回错误，UB tile为1024个复数。

大规模在 `n>128*AIV核数` 时选 `ROWS=208, KS=80`，否则为 `ROWS=128, KS=128`；核数为 `min(ceil(n/ROWS),AIV核数)`，每核循环处理剩余行块。ROWS是并行输出行数，KS为列搬运块长；无跨核workspace。

### 大规模顺序两链与四列批处理

每条SIMD lane对应一个输出行，分别维护输出/AXPY链和共轭dot链。UPPER先求较小列dot，再按参考顺序加入对角及alpha*dot，最后加入较大列AXPY；LOWER先处理较小列AXPY，再加入对角，最后累计较大列dot并应用alpha。输出初始化遵守beta=0不读旧y。

四列优化只批量执行独立Gather、乘法与FMA，然后仍对四列逐项执行Add和Vector barrier；不先求四列部分和，不改成树形归约。三角边界或不足四列使用逐列掩码路径。复数乘法采用独立乘积与 `MulAddDst`，保留固定GNU Fortran Netlib构建的舍入次序；不得通过降低精度门槛抵消重排误差。

矩阵Gather源包括填充行均初始化。208行模板的SIMD宽度补齐到256，额外lane地址钳制在已分配行内；对角路径只消费实部，三角掩码阻止无效lane污染累加。x/y按实际步长搬运，输出只写逻辑元素位置。

### UB预算与有界跨步访问

`CHPMV_ARCH22_WIN_MAX_CPLX=2048`，即16 KiB。跨步包围窗口在上限内整体DMA搬入，超过时按实际元素位置搬运；不得扩大窗口越过分配空间或把步长孔洞作为逻辑输入。

| 路径 | 基础UB字节 | 含16 KiB可选窗口 | 用途 |
| --- | ---: | ---: | --- |
| 小规模n=128上界 | 75,008 | 91,392 | packed矩阵、x/行向量、索引、归约及标量 |
| 两阶段累加 | 72,192 | 88,576 | tile=1024；实虚平面、乘积、搬运/索引、归约 |
| 两阶段归并 | 53,504 | 69,888 | 部分和、输出平面、交织搬运及标量 |
| 有序128×128 | 162,816 | 179,200 | 矩阵tile、20向量平面、系数、索引和IO |
| 有序208×80 | 177,664 | 194,048 | SIMD宽度256；距192 KiB上限仅2,560字节 |

表中是当前显式分配预算；构建时仍核对目标芯片容量与分配对齐。208×80余量较小，新增buffer或修改分块必须重算，不能沿用旧预算。

### 显式同步与编译范围

CMake只对 `blas/spmv/arch22/chpmv_kernel.cpp` 设置源文件级 `--cce-auto-sync=off`，不扩大为全仓开关。小规模、有序、两阶段及scale-only都在该翻译单元编译，所有路径均须显式保证流水依赖，不能只检查有序路径。

`ChpmvWaitEvent` 成对使用SetFlag/WaitFlag建立MTE2→Vector/Scalar、Vector→MTE2/MTE3、Scalar→Vector/MTE3及回写完成依赖，`PipeBarrier<PIPE_V>`保护向量临时区、系数与累加链。buffer复用前等待上次消费者结束，尾块DataCopyPad只使用有效字节。关闭自动同步后的完整精度、保护区及数值顺序复验是必需项。

### 风险与最终验证计划

| 风险 | 处理与待验证证据 |
| --- | --- |
| float32重排、强消去和极端值 | 顺序两链、小规模数值驱动分支及固定Netlib参考；全部原始实虚分量与特殊值判据复验 |
| UB余量、填充lane、尾块与大步长 | 重新核对208×80容量、Gather边界与2048窗口；以保护区/输入不变性/步长孔洞测试及源码审计验证 |
| 关闭auto-sync后的复用依赖 | 审查小、中、大、scale-only全部路径的生产者/消费者事件，不依赖编译器自动插入barrier |
| Device标量同stream更新 | Host/Device混合、调用前异步更新、no-op及beta=0专项，不同步拷回或缓存标量 |
| 性能样本与调用错配 | 按真实kernel routing保留15次调用，明确5次warmup；两kernel同次耗时求和，200×10样本逐项回查 |
| 分片精度耗时冒充正式性能 | 4进程只用于精度/接口；正式性能串行运行原生binary与msprof，二者不同时执行 |

当前方案已落地；最终同轮完整精度、正式性能及交付验证仍需真实执行完成，不在此提前填入PASS。

## 支持硬件与约束

支持 Atlas A2/A3，当前实测设备为 Ascend910B3；CANN 9.1.0。n>=0、步长非零，调用方保证实际分配范围和不重叠约束。保持公共 ABI 与 arch35 分派独立，仅修改本算子必需文件及测试接入。

# 可维可测分析

## 用例与 golden

原始任务 CSV 共 1,200 条：精度 1,000，性能 200；逐项保留身份、参数和种子，不从已通过结果生成预期列表。实际边界类为 28 条，README 的 25 条为陈旧说明。200 行 GPU 基线已填入有限正值，与性能 CSV 的 uplo/n/incx/incy 逐条吻合。

测试沿用仓内GTest/CSV框架，原始身份、参数与seed不变。普通CSV的AP对角虚部置0，`RANDOM_DIAG_IMAG`写有限非零的 `1+i%7`；NaN/Inf对角虚部由额外API用例覆盖，不把普通输入全部改成特殊值。当前另有4项API测试：NullHandle、ZeroDimensionStructuralValidation、FullIntegerStrideDomainAtN1、DeviceAndMixedScalarsFollowStreamUpdates，总计1204个独立注册身份；覆盖Host/Device混合与同stream更新、INT_MIN/INT_MAX的n=1、n=0结构错误及优先级，并检查AP/x不变性、步长孔洞与输出保护区。

golden为固定Netlib BLAS/CBLAS 3.10.0的 `cblas_chpmv`；参考GNU Fortran保持 `-frecursive -O2` 与共享库PIC，不引入fast-math或随意改变浮点选项。Oracle通过 `dlmopen(LM_ID_NEWLM)` 隔离加载并用dladdr核对CBLAS/Fortran符号真实来源，再做UPPER和LOWER/负步长/beta=0已知结果检查；不回退到不明系统BLAS。

| 验收标准 | 判据 | 来源 |
| --- | --- | --- |
| 精度 | 实虚分别atol=rtol=2^-13；有限值匹配率≥0.99；每元素误差≤max(0.01,32×该golden元素ULP)；NaN类别/Inf符号独立核对 | 任务书§3.2 |
| 性能 | 每例 warmup 5 次，再有效采样 10 次，取算术平均；NPU≤1.25×GPU | 任务书§3.3、原始 gpu_baseline.csv |
| 计时 | msprof op 的 kernel task duration；一调用若有多个 kernel，累加对应 kernel 耗时后比较 | 任务书§3.3、测试README |
| 内存专项 | 当前不涉及；仍验证接口保护区和步长孔洞 | 任务书§3.4 |

完整精度先用原生 `--gtest_list_tests` 独立列出1204个身份并核对CSV。`check_validation.py gtest` 调用 `tools/gtests_sharded.py` 调度4个原生GTest进程，各CSV身份只执行一次，写入新建CHPMV_CAPTURE_DIR的不同case子目录。保留4个真实退出码、原始XML/日志/status、二进制hash与调度manifest，原样合并suite后再核对完整expected。合并日志保留四份实际输出及调度JSON，不制造单条原生“全量通过”记录。

正式性能使用 `collect_performance.py` 串行运行原生测试binary与msprof op，不用分片包装器，不与精度并行。routing manifest逐例指定准确GTest身份和真实kernel名称；一调用含accum/reduce时累加二者duration。保留全部15次原始调用、标记前5次warmup，后10次归一化为us并用完整性检查器核对200个身份。事件计时只供开发交叉检查。

`check_validation.py` 来自cann-community-task随包检查器，保留来源与适用许可；调度、采集和冻结脚本为本任务适配器。它们的完整性通过不能替代golden、参考库或profiler真实性审阅。实现、测试、判据或运行配置改变后使用新轮次完整复验，不跨轮拼接。

## 兼容性与交付

不新增公共接口；错误优先级与本任务书一致，修复上游过早 no-op 的差异。公共测试辅助代码变更需保证 arch35 调用仍可编译，arch35 运行状态另行记录。

设计 PR 仅含本 design.md，提交后评论 `compile` 并通知 `@condfuse_3 @fullt @Ascend-CANN`；开发不等待评审。验收源码、测试与根目录 `task_submission/` 提交到个人私仓验收分支，私有状态回读后才推送；仅邀请 `Ascend-CANN` 为开发者。按任务七项模板生成报告/日志，不涉及内存专项写明依据。ZIP 从仓内同一目录生成并逐字节核对，输出精确 commit 链接及平台说明。

## 冻结用例包

本轮每例捕获真实AP/x/y_input；普通数值成功路径另保存NPU actual与Netlib golden。合法n=0保存0字节actual/golden；quick return读回实际y并保存原始y作为不变性预期。负向用例保存输入与预期API状态，缺少数值golden时明确不适用。

`freeze_case_package.py` 在完整PASS后核对身份、字节长度、捕获时间与Netlib来源，用同卷硬链接构建1000精度cases.json、200性能perf/cases.json、inputs/golden/results及facts/readme，逐文件记录SHA256。旧no-op捕获如从y_input导出预期必须标DERIVED_NOOP，不冒充Netlib计算。任务材料未提供其宣称的repo-task-case-gen输出，facts诚实说明由CSV与本轮原生GTest冻结生成，未调用该工具。

原始二进制约24 GB只在本地交付，不加入Git；私仓保留重建脚本、CSV、facts及轻量索引，不声称raw bin已上传。硬链接共享inode，不能覆盖原capture目录或就地修改链接。七项task_submission报告与本地raw包分别管理，最终实际结果由同轮验收证据补充。

## AI 辅助与测试状态

本文及后续实现使用 AI 辅助设计、编码与验证，贡献者负责最终审核。变更类型：已有 arch22 实现的契约修复、验证完善及必要性能优化。当前实现方案已落地，最终同一轮完整精度、串行正式性能与交付核对仍未完成；本文更新不提前声明自验或官方验收通过，最终结论仅以同轮真实证据为准。
