# 需求背景（required）

## 需求来源

10月社区任务：Csyr2 / Cher2k / Cherk 算子精度修复（A2/A3）。

任务页面：https://www.hiascend.com/activities/task-center/details/fc6145cabc15459ab08b3996cad3b8f9?menu=guide

实现仓：https://gitcode.com/cann/ops-blas

个人实现分支：https://gitcode.com/billion4562/ops-blas/tree/codex/triangular-bitwise-a2a3

## 背景介绍

三个算子均已有 arch22 实现。本任务修复操作区的 complex64 数值结果，以及为满足 UB 对齐而写入相反三角或 leading-dimension padding 的问题。修改保留公开函数签名、枚举、状态码、标量位置和 stream 异步语义。

旧 LOWER 路径将写回起点向前取整，再重写旧 C 前缀，或将前缀置零后使用 atomic-add。这些做法会触及非操作区；浮点加零还可能改变负零、NaN payload 或非规范数。完整对角 tile 与某些小 K 路径也存在整块写回超过 uplo 边界的问题。

# 需求分析（required）

## 需求描述

操作区满足官方 FLOAT32 混合容差；相反三角和 lda/ldc padding 与调用前逐 bit 相同，且写回范围不覆盖这些地址。两个硬件均提交完整功能、精度和 bit 检查，A2 910B3 作为性能主门禁，A3 记录性能。

## 需求拆解

| 算子 | 更新关系 | 修复范围 |
| --- | --- | --- |
| aclblasCsyr2 | A ← A + αxyᵀ + αyxᵀ | LOWER 前缀、分块/直接写回、原负步长和非有限语义 |
| aclblasCher2k | C ← αop(A)op(B)ᴴ + conj(α)op(B)op(A)ᴴ + βC | SG3/direct/cube epilogue 裁剪、小 n 长 K 和极值精度 |
| aclblasCherk | C ← αop(A)op(A)ᴴ + βC | direct/cube 对角写回、尾块归约、小 n 极值精度 |

正常 Hermitian 更新将对角虚部设为零。合法 quick return 对整个矩阵缓冲保持原始位模式。测试保留原 CSV、随机种子、非法参数和性能用例，追加任务书要求的 TC_BIT 哨兵。

# 详细设计（required）

## 算子分析

### 数学公式

Csyr2 是复对称更新，使用转置，不作共轭。对列 j，UPPER 仅更新 0≤i≤j，LOWER 仅更新 j≤i<n。负步长仍通过原逻辑起点与步长读取 x/y。

Cher2k 与 Cherk 沿用各自支持的 op(A) 语义和原参数检查。Hermitian 更新只计算指定三角；β 为实数，对角虚部按 BLAS 约定处理。

### 支持数据类型

矩阵和复数向量为 complex64（两个 float32）。Csyr2 和 Cher2k 的 α 为 complex64；Cher2k 的 β、Cherk 的 α/β 为 float32。保持已有 Host/Device 标量归属约定。

### 支持形状

保留上游接口的 n/k/leading dimension/步长约束，不增加测试用例之外的接口限制。非整 tile 尾部、非 32 字节对齐的 GM 对角起点，以及 lda/ldc>n 的 padding 均按真实逻辑范围处理。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

##### 1. 分核策略

Csyr2 保留上游 unit-stride 的分核框架和设备可用 AIV 上限，按已有核数表选择已测候选，不增加公开参数。直接路径按列组分核，LOWER 的相邻波次反向分配以平衡三角列长。direct-eight 处理全部 floor(n/8) 个完整组，只留下 n%8 个单列尾；反向部分末波后，空闲核足够时将尾列放在紧邻部分波边界的空闲核上，这些核先前未配对前向波的工作较短，避免把尾列叠到最长工作量的低编号核。

Cher2k/Cherk 复用现有 tiny、direct、SG3、cube 路径。小 n 路径使用对应固定几何；Cher2k 的既有 packed OP_C 路径处理非 128 整除尾部，保留原 k/标量限制。

##### 2. 数据分块和内存优化策略

Csyr2 的 tiling 缓存只保存数值，并检查逻辑设备号，不保存设备内存或 stream。tiny n≤8、单位步长、α=(1,0) 在全部原参数检查后分派到编译期 Order 核，单个 AIV 顺序处理各选中列并直接读写 GM float32，不申请 UB staging 或额外 Device GM。各元素保留原乘加顺序及零乘非有限交叉分量的传播规则，Host 不增加同步。

LOWER direct 按完整 UB 占用选择 8/4/3/2 列组。令 P=512×ceil(n/64) 字节，B 为列组宽度，短头缓冲 H 在 B=8 时为 512 字节，B≤4 时为 128 字节；所需 UB 为 (5+2B)P+3392+2H，超过 192 KiB 时回退到原分块路径。

n≥640 的 direct-eight 实现仅保留该路径使用的 TBuf 与状态。X/Y 和旋转后的复数分量常驻 UB；两个输出槽用对应 V→MTE3、MTE3→V 事件保护复用。复数系数保留原始 uint64_t 对，按当前计算层展开，避免同时保留四组浮点系数数组。大 n 容量无法容纳既有 spacing 时，旋转输入使用原缓冲余量偏移 256B，调整旋转输入与输出的相对布局；X/Y 保持既有布局。

TPipe 仍负责 UB 分配与生命周期；X/Y、双输出槽、旋转输入及原 padding/spacing 合并在一个 TBuf arena 中，索引使用另一个 TBuf，所有视图保留原字节偏移。输出槽由递增提交计数的奇偶选择，第三次提交起等待两次提交前同槽的 MTE3→V 事件；退出时等待所有已使用槽。没有绕过 TPipe 直接构造裸 UB 地址。

Host 对 ceil(n/64) 为13、16、17的物理 padding 桶使用编译期 PaddedFloats；每桶保留真实 n 和 lda，而非只支持某一个测试阶数。仅 lda*8 可表示为 uint32 时采用该特化；巨大合法 leading dimension 仍走原64位路径。n%8≥3时最后完整八列组至少11行，因此可省掉不可达的逐列短带分支，其余余数仍保留回退。

##### 3. tilingkey规划策略

沿用现有 unit/strided/windowed 分流及公开 launch 语义。新写回和 direct-eight 只挂接 arch22 的对应内部路径，不新增公开枚举或调用方开关。

#### 3.2.2 kernel侧设计：

**精确短头写回。** arch22_complex_store.h 的 StoreComplexRange 接收 32 字节对齐的 UB 源及实际选中范围。当 LOWER 起点不按 4 个 complex64 对齐时，最多将 3 个选中元素的 uint64_t 位模式搬到已丢弃的 UB 前缀，再以短 DataCopyPad 写到准确 GM 地址。剩余范围的 UB 起点已对齐，单独提交。V→S、S→MTE3 事件保证标量搬运与 DMA 次序；此过程不读取 GM 旧前缀，不提交非操作地址。

**输入固定重排。** n≥640 的 direct-eight 共享一份字节索引 4×(i xor 1)，由 CreateVecIndex、奇 lane 加 -2、左移两位构造，再以两次 Gather 交换 X/Y 中相邻 float32 的实虚部位置，保留每个 float 内部位模式；随后沿用原偶 lane 乘 -1。索引准备不依赖输入 DMA，在 Gather 前使用 MTE2→V 等待输入就绪；Gather 后及符号变换后均保留 PIPE_V，之后才复用 indices 缓冲。原逻辑尾部只做逐元素置换，不作归约，padding 不进入有效复数。

每核c的首个列组从8c开始，后续组号严格递增，尾列在全部完整组之后。输入搬入和旋转因此从firstFloat=(16c)&~63开始，裁掉永远不会读取的前缀；所有数据仍放在原UB绝对偏移。该起点按64个float对齐，保留复数实虚部偶数相位，旋转长度为原padded长度减firstFloat。原TPipe容量、计算范围、系数规则、输出地址与同步不变。该路径的Host核数和n≥640保证每核首组合法。

**多列写回。** Csyr2 4/8 列完整组分为对角带和所有列共有的矩形后缀。对角带中的补零及与后缀重叠的地址均在操作区内部；相反三角和 padding 没有 DMA 覆盖。八列短头用预备的列号/偏移表和一次 Gather 构造；索引按 (4×pitch−56)×q+4×t 计算，与原表达式 4×pitch×q+4×(t−14×q) 等价。剩余行数≥15时一次提交8×64B带；11–14行时拆成前四列4×64B和后四列4×32B，两次DMA最晚结束于组内行10。两种带均按原掩码清零操作区内的重叠补位。剩余8–10行或DMA stride超限时精确逐列提交，每列仅8-k个complex64且不需要补零。公共后缀为零行时不提交零长DMA。真正的余数尾列按原精确单列短头处理。

**其余写回路径。** Cher2k 的 SG3/direct/cube epilogue、Cherk 的 direct/cube 对角 tile 均按 uplo 拆成选中列范围；完整操作 tile 保留二维 DMA。低秩 UPPER 路径只提交选中列前缀。不会用整块“读旧值再写回”来模拟非操作区保持。

**数值处理。** Csyr2 保留乘法/累加顺序和 α=(1,0) 遇到 NaN/Inf 的分量传播。Cher2k 小 n 长 K 使用融合列计算与补偿归约，有限极大输入按 2^-80 缩放后平衡累加并恢复尺度。Cherk 的尾部 ReduceSum 只处理真实元素，排除 DMA padding；长 K 对角范数使用分段归约和补偿累加；有限极大输入缩放两个因子并分步恢复。真正非有限输入走对应分量处理路径。

实现不增加与问题规模无关的持久 Device GM 缓存，不增加 Host 同步或 CPU 计算回退。

## 支持硬件

| 平台 | 编译 SOC | 软件 | 验收角色 |
| --- | --- | --- | --- |
| 单个 A2 910B3 逻辑设备 | ascend910b3 | CANN 9.1.0，Netlib BLAS 3.10.0 | 精度、bit 与性能主门禁 |
| 单个 A3 逻辑设备 | ascend910_93 | CANN 9.1.0，Netlib BLAS 3.10.0 | 精度、bit 门禁及性能记录 |

## 算子约束限制

公开参数、非法枚举/步长/leading dimension 的检查顺序和返回码沿用上游。仅修改 arch22 生产实现，以及三个算子的 arch22 测试、专用检查工具；arch35 实现和公开接口头不变。具体文件范围以代码 PR diff 为准。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 标准 |
| --- | --- |
| 操作区实部、虚部 | 分别以 atol=rtol=2^-13 判定，匹配比例≥0.99 |
| 最大有限误差点 | 误差≤max(0.01,32 ULP(g_low)) |
| 非有限值 | 按官方标准处理分类及 double 收窄到 float 的情况 |
| 非操作三角/ld padding | 快照原始字节逐 bit 一致，要求 100% |
| 性能 | 原 200 个 TC_PF；Csyr2 固定100次预热、其余固定10次预热，均51次有效调用；A2每项满足原标杆，A3记录 |

精度标准：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md

Cher2k/Cherk 使用同版本 Netlib BLAS 3.10.0 double 主参考：将原 float 输入与标量精确提升，调用 zher2k/zherk/zgemm；保留原 float 参考的交叉对照及此前 FAIL。Cherk 参考按 256 边长三角分块，最多 8 个 Host 线程。该方案涉及原 float golden 的参考链修订，门限、输入与用例不变，请在设计评审中重点确认这一口径。

同一向量流水的计算层按数据依赖插入 PIPE_V。UPPER 列对的两条同层指令写入不同输出段，每层开始统一同步，输出就绪由已有队列的 V→MTE3 事件保护；LOWER 初次覆盖输出前由对应 MTE3→V 事件保护槽复用，三个累加层保留真实依赖屏障。tiny 由单个 AIV 按原算术顺序计算并以 float32 访问 GM，入口 InitSocState、出口 PIPE_ALL。需要原子累加的路径保持写回完成后再恢复非原子模式。

GTest 对相反三角和 padding 使用调用前快照；TC_BIT 包含±0、多个 NaN payload、±Inf、非规范数和有限值。quick return 检查整个缓冲区。按原 CSV 测试，Csyr2 为 1222 条（含22 BIT）、Cher2k 为 1238 条（含24 BIT及上游额外14条）、Cherk 为 1224 条（含24 BIT）。另保留上游启用的 4/9/5 条原生回归与 Csyr2 原有两条 disabled 的单独记录。

测试工具从 XML 核对全部 CSV ID、缺失/跳过/重复与失败。GTest 分片只调度同一用例，功能并发运行的墙钟时间不作为性能成绩。性能单独串行；Csyr2 按官方任务包允许的 msprof 精确设备计时，取完整 Task Duration(us) 的51次算术平均，保留原 avg_us 的 Host API 对照，不采用更短的 AIV 活跃时间。运行前核对200个PF ID，运行后核对XML和每例203条任务，只使用固定100次预热后的51次batched调用。原10次预热的实测窗口内曾出现系统性耗时变化，100次固定预热及完整Task主计时均已获任务开发者明确批准。原初始输入、seed和门限不变；C仍按原方式连续累加，精度核对另用原始快照，增加预热会相应增加计时前的累加次数。Cher2k/Cherk继续采用原10/51的 average_us / CHERK_PERF avg_us，逐算子注明计时来源。全部与原GPU基线ms/0.8及任务书前三锚点对照；旧协议API/Task FAIL全部保留，请社区评审确认此官方允许的预热和测量口径。

以下结果对应代码提交fba52cc0a964c21f3f2b3b8b9c5d037dbf5f9f70。Csyr2使用本次完整矩阵；未改动的Cher2k/Cherk生产实现沿用各自已完成的整组功能与性能记录。最后在两卡分别核对源码及冻结动态库后，对三算子同一动态库执行70条BIT，均70/70通过。

| 设备 | 算子 | 完整CSV精度 | 启用原生回归 | 原200项性能 |
| --- | --- | --- | --- | --- |
| A2 910B3 | Csyr2 | 1222/1222 | 4/4 | 200/200 |
| A2 910B3 | Cher2k | 1238/1238 | 9/9 | 200/200 |
| A2 910B3 | Cherk | 1224/1224 | 5/5 | 200/200 |
| A3 | Csyr2 | 1222/1222 | 4/4 | 199/200（记录） |
| A3 | Cher2k | 1238/1238 | 9/9 | 200/200（记录） |
| A3 | Cherk | 1224/1224 | 5/5 | 197/200（记录） |

Csyr2原有两条disabled回归按上游状态单独记录，不计入通过数量。A2 Csyr2 TC_PF_1167（LOWER n1011）完整Task均值9.306666667us，原门限9.595us；A3同项9.637823529us，保留FAIL。A3其余未达标项在逐例报告中完整列出，不改作通过。原协议API/Task数据与旧FAIL均保留，不跨实现拼接单项成绩。参考链及测量口径仍请社区评审确认；当前尚未提交平台验收。

## 兼容性分析

本任务修复已有算子，无接口签名变更。arch22 共用短头写回工具只由本次修复路径调用。原非法参数、正负步长、Host/Device 标量与 stream 异步行为由原回归覆盖；未更改 arch35 的参考链或测试框架。任务包 Csyr2 TC_ED_151 的返回码文字与任务书存在差异，沿用现有参数校验与原 CSV ExpectedStatus，不修改公开语义以调整结果。

复现步骤随代码中的 test/README_TRIANGULAR_ARCH22.md 及各算子 arch22/README.md 交付；任务包原 test_cases 的22个脚本/CSV/基线/参考检查文件原样保留并随验收材料提供。GTest的逐bit实现位于 test/utils/untouched_bitwise.h；原任务参考 check_untouched_bitwise.h 另随任务包保留。
