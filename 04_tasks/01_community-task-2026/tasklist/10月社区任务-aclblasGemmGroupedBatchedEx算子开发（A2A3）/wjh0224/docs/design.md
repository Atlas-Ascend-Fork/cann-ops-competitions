# 需求背景（required）

> 当前修订（2026-10-29）：用户已向CANN小助手确认可在HiDevLab 910B4上验证。有限值误差上限锚定最大误差点golden，取max(0.01,32×ULP)；非有限值按任务书§3.2判定。已完成的设备测试范围及剩余验收项在本文末尾注明。

## 需求来源

任务为“10月社区任务-aclblasGemmGroupedBatchedEx算子开发（A2A3）”，依据用户提供的压缩包及其中的 `aclblasGemmGroupedBatchedEx_A2_task_doc.md`、测试 CSV、GPU 参考 CSV 和测试指导编写。设计日期为2026年10月28日，团队/提交目录统一为 wjh0224；历史本地Git作者 wblxr408 不作为提交目录账号。

本文采用官方设计文档模板的章节顺序和表头。任务原件由用户提供，原始1200条CSV保存在代码仓库测试目录。开发环境为HiDevLab 910B4、Ubuntu 22.04.5、CANN 9.1.0，golden固定为Netlib BLAS 3.10.0 CBLAS。

技术设计与实施计划已完成，现已按设计实现并在910B4上自测。下文“支持”为目标能力；实际通过范围见文末测试复现与报告入口，不能据设计文字推定全部验收通过。

## 背景介绍

### aclblasGemmGroupedBatchedEx算子实现背景

分组批量 GEMM 在一次接口调用中完成多个尺寸和转置方式不同的矩阵乘。同组共享 m/n/k、前导维、alpha/beta 和转置属性，各矩阵使用独立地址，适合异构小批量线性代数工作负载。

已读取 ops-blas 本地基线 `62fc37e3fe1350f46ac10249ee730a96ff2b8553`：公共头文件已有本接口；`blas/gemm_grouped_batched_ex/arch35/` 提供 FP16/BF16/FP8 路径，README 标注 A2/A3 不支持；本任务补充 `arch22` 的 FP32 路径。现有 `blas/gemm/arch22/` 可参考 FP32 Matmul、转置映射及 HF32 控制，不能将其普通 GEMM 或现有 arch35 混合精度实现直接作为本任务完成结果。

### aclblasGemmGroupedBatchedEx标杆现状分析

功能参考 [cuBLAS cublasGemmGroupedBatchedEx](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmGroupedBatchedEx)，精度 golden 使用 Netlib BLAS 3.10.0 的 `cblas_sgemm(CblasColMajor, ...)` 逐组逐矩阵生成。数值分支以[固定3.10.0版SGEMM源码](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/BLAS/SRC/sgemm.f)及实际链接结果为依据，避免引用其他版本的分支行为。

附件实际为1000条精度/边界用例与200条性能用例，共1200条，并非任务书数处所写的200条总用例。GPU CSV 有200条已填写耗时，单位为 ms；测试指导分别声称 A100 和 P102-100，CSV 本身未携带设备、CUDA/cuBLAS版本或采样来源。因此保留为附件参考值，正式性能验收前需确认基线来源；不根据耗时数值推断设备。

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| handle | 携带stream及workspace的Host句柄 | aclblasHandle_t | 不涉及数值类型 | 正groupCount时有效 | 标量句柄 |
| transaArray | 每组A的操作 | const aclblasOperation_t[] / Host | N、T、C | 枚举有效，实数C等价T | groupCount |
| transbArray | 每组B的操作 | const aclblasOperation_t[] / Host | N、T、C | 同上 | groupCount |
| mArray | 每组C行数 | const int[] / Host | int32 | 每项≥0 | groupCount |
| nArray | 每组C列数 | const int[] / Host | int32 | 每项≥0 | groupCount |
| kArray | 每组归约长度 | const int[] / Host | int32 | 每项≥0 | groupCount |
| alphaArray | 每组乘法系数，连续标量数组 | const void* / Host或Device | float32 | 正groupCount非空，按位置读取 | groupCount |
| Aarray | Device矩阵指针数组 | const void* const[] / Device | 矩阵float32 | problemCount>0时外层非空 | problemCount个指针 |
| Atype | A类型 | aclDataType | ACL_FLOAT | 其他类型返回NOT_SUPPORTED | 标量 |
| ldaArray | A列主序前导维 | const int[] / Host | int32 | N时≥max(1,m)，T/C时≥max(1,k) | groupCount |
| Barray | Device矩阵指针数组 | const void* const[] / Device | 矩阵float32 | problemCount>0时外层非空 | problemCount个指针 |
| Btype | B类型 | aclDataType | ACL_FLOAT | 其他类型返回NOT_SUPPORTED | 标量 |
| ldbArray | B列主序前导维 | const int[] / Host | int32 | N时≥max(1,k)，T/C时≥max(1,n) | groupCount |
| betaArray | 每组加法系数，连续标量数组 | const void* / Host或Device | float32 | 正groupCount非空，beta=0不读旧C | groupCount |
| Carray | Device原地输出矩阵指针数组 | void* const[] / Device | 矩阵float32 | problemCount>0时外层非空，输出矩阵不重叠 | problemCount个指针 |
| Ctype | C类型 | aclDataType | ACL_FLOAT | 其他类型返回NOT_SUPPORTED | 标量 |
| ldcArray | C列主序前导维 | const int[] / Host | int32 | ≥max(1,m) | groupCount |
| groupCount | 分组数量 | int / Host值 | int32 | ≥0；dtype合法后0直接成功 | 标量 |
| groupSize | 每组矩阵数 | const int[] / Host | int32 | 每项≥0；使用64位求和 | groupCount |
| computeType | 计算类型 | aclblasComputeType_t | ACLBLAS_COMPUTE_32F | 本轮不声明FAST或混合类型 | 标量 |

计算公式：`C[g,i] := alpha[g] * op(A[g,i]) * op(B[g,i]) + beta[g] * C[g,i]`。

### aclblasGemmGroupedBatchedEx功能分析

接口沿用 `include/cann_ops_blas.h` 的共享声明，参数共20个，包含handle；除handle外为19个参数。alphaArray/betaArray均解释为连续FP32标量数组，不解释为标量指针数组。cuBLAS该接口的标量数组位置为Host；任务书另允许Device，本设计额外覆盖Device标量，明确区别于对标接口。

输入为句柄、逐组属性、A/B指针数组、系数及C旧值，输出为原地C。A/B/C的外层指针数组与内层矩阵均在Device；Host不能通过`Aarray[p]`直接读取矩阵地址。dtype是整次调用的公共属性，不能在组间混用。

本轮主路径为 `ACL_FLOAT/ACL_FLOAT/ACL_FLOAT + ACLBLAS_COMPUTE_32F`，覆盖N/T/C全部9种组合、padding和异构组数/批数。不新增FP16/BF16/FP8/INT8的arch22支持声明；附件的这些类型负向用例预期NOT_SUPPORTED。无广播、图注册、ACLNN两段式调用或TBE实现。

共享函数原型：

```cpp
aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray, void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[], aclblasComputeType_t computeType);
```

# 需求分析（required）

## 需求描述

| 需求 | 实现约束与输出 |
| --- | --- |
| 接口与产品 | 公共ABI复用，新增A2/A3 arch22；950实现按架构继续选择 |
| 数学与排布 | 列主序、逐组形状/系数、Device指针数组、C原地更新 |
| FP32 | 乘法及累加保持FP32语义；不默认降为FP16/BF16/HF32 |
| 边界与错误 | dtype先验、零组快速返回、分组全量校验后提交；错误码按任务书 |
| 零值分支 | k=0/alpha=0不引用A/B数据，beta=0不读旧C，beta=1的缩放空操作不写C |
| 执行与资源 | handle stream顺序提交，复用已有workspace；设备标量在kernel读取 |
| 精度与性能 | 全部1200条附件保留；精度全矩阵比较；性能使用msprof的真实kernel记录 |
| 交付 | 官方模板、仓库原生目录和README；真实结果形成后再制作验收报告 |

## 需求拆解

1. 在`blas/gemm_grouped_batched_ex/arch22/`新增Host、kernel、tiling头；需要Cube与Vector分开编译时增加一个epilogue源码，按仓库CMake源文件规则收集。
2. 实现完整参数校验与独立FP32缩放路径；Host和Device标量分别处理，避免对Device地址解引用或为取零系数而同步读回。
3. 先完成FP32通用计算行为，再以Cube分块和小矩阵Vector路径降低批量调度成本；按用户最新要求在HiDevLab实际910B4上实测，记录与任务书910B3的设备差异。
4. 在`test/gemm_grouped_batched_ex/arch22/`隔离任务CSV、参数读取、CBLAS golden和NPU wrapper，复用公共测试框架；共享解析器目前默认FP16、snake_case及长度广播，不能直接加载本包。
5. 使用独立arch22测试覆盖本功能的行为、设备精度和性能，补足Device系数、padding保护和多次调用；附件原脚本的参数与计时差异由本项目适配，不直接使用GTest墙钟作性能证据。

# 详细设计（required）

## 算子分析

### 数学公式

```text
P = sum(groupSize[g])
S[g] = sum(groupSize[h], h < g)
p = S[g] + i, 0 <= i < groupSize[g]
Carray[p][r + c*ldc[g]] = alpha[g] * sum(Aop(r,t)*Bop(t,c), t=0..k[g]-1)
                              + beta[g]*Cold(r,c)
Aop(r,t) = Aarray[p][r + t*lda[g]]       (N)
         = Aarray[p][t + r*lda[g]]       (T/C)
Bop(t,c) = Barray[p][t + c*ldb[g]]       (N)
         = Barray[p][c + t*ldb[g]]       (T/C)
```

逻辑输出范围为`0≤r<m, 0≤c<n`，leading dimension间隙保持原值。k=0或alpha=0进入缩放分支：beta=1不读写C；beta=0直接写正零；其他beta执行一次FP32乘法。零乘法分支不先计算`alpha*0`，因而不会将Inf alpha与k=0组合误变为NaN。普通乘法路径的beta=0也不读C旧值。

### 支持数据类型

| 对象 | 本轮arch22支持范围 |
| --- | --- |
| A、B、C | float32 / ACL_FLOAT |
| alpha、beta | float32连续数组，可在Host或Device，允许两数组位置不同 |
| computeType | ACLBLAS_COMPUTE_32F |
| 尺寸、前导维、组数和批数 | API int32，内部元素/字节/前缀索引使用经溢出检查的64位 |

既有arch35支持表按产品分别描述，不将其FP8等能力推广到A2/A3。不支持的dtype组合与computeType返回NOT_SUPPORTED。

### 支持形状

每组逻辑A为m×k、B为k×n、C为m×n。物理列主序存储：A在N时为lda×k，否则lda×m；B在N时为ldb×n，否则ldb×k；C为ldc×n。支持满足前导维的padding，不额外支持任意strided tensor。

groupCount和groupSize可为0；附件正向groupCount最大8，性能典型方阵到4096³。这些是附件覆盖范围，不成为API硬编码上限。测试程序单用例Host预算512MiB，应按实际同时驻留的A/B/C/golden/还原副本统计；不足时逐矩阵生成与释放golden，不能只计算A+B+C的192MiB就认定4096用例满足预算。

## 算子实现

### 实现方案

采用Ascend C kernel直调。Host按同一转置模式把最多4组描述符按值传递，kernel通过Device指针数组寻址。一般矩阵执行输入分类、Cube FP32 Matmul与Vector epilogue；HF32关闭。若包内有K≥512，使用按K递增的128×64输出块、K32有序Vector路径；K>8192等不满足快速路径条件时回退通用Scalar。分类出的非有限值或幅值溢出风险使用参考分支回退。Tiny路径覆盖TA=N、m/n≤16、K≤32且ldc满足对齐的8×8小块，并有数值风险回退。

有序分支用于控制相对固定Netlib的累加误差。大幅值交替输入反例已通过幅值上界分派修复，并在910B4上定向验证；这不代表完整精度集已通过。当前性能基线整体失败，Tiny仅完成两项性能改善。失败的Direct、K1、K4/K2候选未进入生产。

列主序映射沿用仓库已有的`Cᵀ = op(B)ᵀ * op(A)ᵀ`方案：内部M=n、N=m，左右操作数为B、A，对应转置属性交换，原lda/ldb按交换后的操作数保留。当前Cube使用有界乘积窗口并由epilogue处理系数、padding及尾块；不接入实验中的全量输入打包布局。

不把逐矩阵调用aclblasSgemm作为最终批量方案：其重复下发、寻址和workspace准备难以满足小矩阵性能目标。复用其数学映射与已存在的通用工具，避免复制整套普通GEMM源码。

#### 3.2.1 host侧设计：

tiling策略：全量校验后建立组前缀与每组二维输出tile任务区间。按shape、转置、ld和缩放方式选路径，将多组描述符装入一个有界参数包；容量以CANN 9.1.0的kernel实参限制及编译结果确定，初始候选4组。超过容量则拆包，不能拒绝大groupCount。系数为Host时按值快照，Device时传地址和组偏移，在设备上消费。

##### 1. 分核策略：

每个普通GEMM任务对应`(组g, 组内矩阵i, tileM, tileN)`。按所有组tile数量前缀定位任务，核数取`min(任务数, 实际可用AIC核数)`，grid-stride覆盖剩余任务，避免每个小矩阵单独launch。Vector任务按输出连续行片段和矩阵编号分配，使用实际AIV核数。组数较少时线性定位、较多时在参数包内查前缀，均不按最大shape给所有矩阵补零。

当前公开C输出块为128×64；Cube内部列主序映射交换M/N，有序Vector使用K32递推。块参数以已编译源码为准。细长矩阵按较长输出维切分。当前不采用Split-K，单输出tile由单任务负责，避免原子累加、跨核归约和额外精度差异。Tiny阈值已按上述限制集成，更多形状的扩展需要独立精度与性能证据。

##### 2. 数据分块和内存优化策略：

FP32 Cube显式关闭降低输入精度的HF32模式，参考仓内arch22 GEMM已有的`SetHF32Mode(DISABLE)`用法；精确可用入口以9.1.0安装头文件及编译为准，不以旧版API文档推断通过。A/B在片内按K块复用，边界按有效m/n/k搬运；不得越界读取padding外的元素来凑整块。

一般alpha/beta路径先生成FP32乘积窗口，再由Vector执行`alpha*P + beta*C`。workspace来自既有`GetEffectiveWorkspace/EnsureDefaultWorkspace`约定，内容包含可选打包块与乘积窗口；不分配全批次的`Σ batch*m*n`乘积。工作窗口预算从句柄实际容量扣除辅助区后确定，按矩阵/tile顺序重复使用，工作核与epilogue均放在同一stream。必要的系统Matmul workspace计入总字节数。

Tiny直接写C，不分配完整矩阵乘积临时空间；一般路径单窗口乘积为`4*windowM*windowN*windowBatch`字节，加上对齐打包、系统workspace。所有乘法、加法和对齐先检查溢出。API不新加workspace参数；用户提供workspace太小时遵循既有返回约定，不能越界使用。

句柄既有workspace扩容会同步stream后重分配；不能把首次扩容描述为完全无同步。预先准备容量后的正常提交不添加结果同步。同句柄同stream的后续任务按序复用，同句柄并发Host调用或在未完成时切换stream/workspace由调用者按既有句柄规则串行化。临时缓冲随句柄/用户workspace存活，禁止局部RAII在kernel未完成时释放。

Host描述符按值传给launch，避免异步H2D复制尚未完成就销毁Host vector，也避免同一设备描述符地址在前次任务执行前被覆盖。若实际参数包超过SDK限制，应减小包容量，保持该生命周期方案。

##### 3. tilingkey规划策略：

本算子为直调接口，不注册ACLNN/TBE TilingKey。当前分派包含通用Vector（含SCALE和Scalar回退）、Tiny8×8、输入分类、长K有序Vector、Cube加epilogue。转置归一化为NN/NT/TN/TT四类，同包最多4组且转置模式一致。Direct写回仅为未接纳实验。路径标识属于内部实现，不进入公共ABI。

数据检测：

1. 先校验整次dtype/computeType，非法返回NOT_SUPPORTED；继而groupCount<0返回INVALID_VALUE，groupCount=0直接SUCCESS，后者不读取handle或任何数组。dtype不支持且groupCount=0仍返回NOT_SUPPORTED，依据任务书§2.5的显式顺序。
2. 正groupCount先检查handle，再检查全部Host属性数组及alpha/beta外层非空；逐组检查转置枚举、非负m/n/k/groupSize及最小lda/ldb/ldc，包括零维/零批组。非法转置返回INVALID_ENUM，其他非法值返回INVALID_VALUE。正groupCount空handle叠加其他错误时优先HANDLE_IS_NULLPTR为本设计约定。
3. 所有组都合法后，以64位检查problemCount、物理跨度和任务数，不在发现后组错误前提交前组计算。problemCount>0时Aarray/Barray/Carray外层全部非空，即使全部组均为零维或alpha=0；problemCount=0允许这三个外层数组为空。外层非空检查不等于读取内层A/B数据。
4. 按仓内`CheckPtrLocation`识别alpha/beta位置；Device系数在同stream kernel中读，禁止为快速分支强制D2H同步。k=0无须读取alpha数值；alpha=0/k=0不读取A/B矩阵及对应无须使用的内层指针。
5. C的有效输出存储由调用者提供。内层Device指针的非法地址无法靠Host普通解引用同步校验，错误由设备执行及调用者同步获得；不承诺把所有悬空指针转成同步INVALID_VALUE。Host接口也不能发现普通C数组的实际分配长度；CSV长度不一致只能在测试解析层拒绝，不能对短数组越界后假称API可检测。

#### 3.2.2 kernel侧设计：

Init阶段读取按值tiling、Device矩阵指针数组和必要的Device系数，绑定每个任务的GM窗口。Process阶段按以下路径处理：

1. `SCALE`：m/n/batch为空直接退出；beta=1无写入；beta=0不读旧C直接写0；其他beta用FP32逐元素乘法，保持ldc间隙。k=0或alpha=0时不发射有效A/B读取。
2. `VECTOR_SMALL`：连续输出行块在UB内累加FP32结果，按K递推加载A行片段与B系数，模板处理NN/NT/TN/TT。保留k=0/alpha=0和beta=0/1的独立分支，尾行用有效长度屏蔽；不依赖950特有SIMT能力。
3. 长K有序Vector与Tiny8×8：前者按128×64输出块及K32递推控制顺序；后者在小矩阵条件满足时直接更新逻辑C。数值风险均保留参考回退。`CUBE_DIRECT`不在当前实现内。
4. `CUBE_EPILOGUE`：Cube只写当前乘积窗口，随后Vector从窗口读取FP32乘积，应用组系数并写回C。一个输出元素仅一个写入者，同stream kernel边界提供先后依赖，不使用跨block自旋屏障。

普通路径不通过FP16中转换取吞吐。默认HF32禁用，是否增加精度受控的性能路径需要独立类型/算法契约，本轮不纳入。

Inf/NaN按任务书的分类规则比较。已核实固定Netlib3.10.0的N/T主路径均计算零B对应的乘积；alpha非零且k>0时，A=Inf、B=0会形成非有限结果，不能额外跳过零B将其改成有限C。实施时用一次输入分类kernel识别含非有限值或按幅值上界可能累加溢出的矩阵，分类前先遵守alpha=0/k=0/beta=0的不读取规则。每个当前窗口实例的分类标志存放于句柄workspace辅助区，同stream后续Cube及Vector读取：普通实例走Cube和epilogue，回退实例跳过Cube并由遵循固定参考分支及乘加次序的Vector路径写C，避免重复写入。分类成本须纳入整次调用性能；不得由CSV标签控制生产kernel算法。若FP32 Cube在特殊输入上已有等价行为，可在对应行为证据充分后缩减该回退。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2系列产品（含Atlas 800T A2，910B3） | √ |
| Atlas A3系列产品（含Atlas 800T A3） | √ |

目标实现目录为arch22。用户已向CANN小助手确认可在HiDevLab 910B4上验证；任务书允许自验证覆盖一种款型。A3支持范围仍以对应工具链构建与实际后续验证记录为准，不以A2结果冒充A3实测。950使用既有arch35路径。

## 算子约束限制

全部矩阵为列主序FP32；无广播、复数或其他dtype正向承诺。C矩阵之间不得重叠，C不能覆盖执行中仍要读取的A/B存储；调用者负责实际分配大小、Device可访问性与生命周期。A/B为只读，同组shape一致且外层指针按组连续排列。

Host属性与Host系数在调用期间可读，Device系数/指针数组/矩阵持续有效至stream完成。同stream前序kernel更新的Device系数应在本次执行中可见。dtype和枚举合法性按前述顺序处理，groupCount=0的quick return不替代dtype检查。

接口返回SUCCESS表示Host校验及任务提交成功，异步执行错误仍由调用者同步获取。workspace首次创建/扩容沿用句柄既有同步规则。512MiB仅为测试Host预算，不限制API的数学尺寸。大幅值反例已定向通过；当前完整精度与性能验收尚未完成。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度golden | Netlib BLAS 3.10.0 CBLAS sgemm，ColMajor，逐组逐矩阵全逻辑C比较；padding另核对 | 任务书§3.1、§3.2 |
| FP32有限值 | atol=rtol=2^-13；matched_ratio≥0.99；max_abs_error≤max(0.01,32×ULP(g_low))，g_low为最大误差点收窄后FP32 golden；逐元素上限仅作诊断 | 任务书§3.2及用户提供opbase标准§2.1.2/§2.2 |
| 非有限值 | NaN对NaN、同号Inf对Inf及NaN对任一符号Inf等效；有限与非有限不等效；单独统计，有限集合为空时不除0 | 任务书§3.2 |
| 零值路径 | alpha=0时有限值缩放按FP32 EXACT；非有限值按分类规则；beta=1不写C且保留原位模式，beta=0不读旧C；k=0不读A/B；padding保持原值 | 任务书§2.1、§3.2 |
| 原始覆盖 | 保留1000精度/边界+200性能；N/T/C、组数/批数、padding、负参数和非有限值 | 附件CSV实际内容 |
| 补充行为 | Device系数含同stream前序更新、长groupCount分包、零alpha下A/B不可访问、beta=0下C旧值NaN、各ld间隙、后组错误时不改前组C | 任务书语义与本设计 |
| 输入生成 | 固定种子；均匀与正态各50%，按任务书生成alpha/beta和矩阵，分布及参数写入日志 | 任务书§3.5 |
| 性能设备 | HiDevLab实际910B4；用户已向CANN小助手确认可在B4验证；任务书原定910B3差异明确记录；单次API全部kernel计入 | 用户确认及任务书§3.3 |
| 性能判据 | 用户确认按§3.3的T_NPU≤1.25*T_GPU（ratio≥0.8）验收；报告同时列出≥1的参考数据 | 用户确认及任务书§3.3 |
| 采样 | 预热5次；建议保留51个有效调用样本，同时导出前10均值与51均值，满足10、>10、>50三种材料记载的取数需求 | 任务书§3.3、§4、§7与附件README |
| 采样完整性 | msprof真实task duration按一次API内全部分类/打包/Cube/epilogue/缩放kernel求和；不读取GTest整用例毫秒数作为算子耗时 | 任务书msprof要求、附件脚本审阅 |
| 状态与资源 | API SUCCESS并最终同步成功、结果正确才算有效样本；记录workspace需求与实际容量；GPU workspace未给出时比值填待提供，不能填0或通过 | 任务书§4与本设计 |

对每次性能调用，先在计时范围外将C恢复为相同C0，并保证恢复先于本次kernel；Host系数/Device系数模式及warmup由同一层控制，避免工具与程序重复预热。若一次API发射多个kernel，第j次样本为`T_j=Σ task_duration`，使用关联ID/调用边界确定归属；不能只取最快Cube记录，也不能把来自不同调用的记录混作一个样本。采集模式是否重放应用、导出文件名和warmup责任以目标环境msprof版本确认为准。

附件verify_performance.py目前解析GTest“[ OK ] (... ms)”作为NPU耗时，不满足上述标准；正式实现要替换取数和完整性核对，保留原件不运行。基线ms转us为`gpu_ms*1000`。除groupCount/trans/m/n/k/groupSize外，还须在报告记录alpha/beta、ld、computeType和缓存/重置方式；原GPU CSV缺少这些元数据，不能仅凭不完整键宣称条件一致。

## 兼容性分析

公共头文件已有20参数声明，复用原ABI；生产实现按SOC架构扫描选择arch22或arch35。README将支持矩阵拆为产品×dtype，新增A2/A3 FP32，不覆盖950原类型表。测试增加独立arch22数据和解析适配，不把FP32解析补丁直接改成所有平台的新默认。

采用C++/Ascend C及仓库CMake/GTest，保留已有handle/workspace机制；Python只承担CSV与报告编排。该选择满足功能、复用、部署兼容、维护与退出约束，无新增服务、数据库或运行时。性能、资源效率及特值分类开销须以设备数据评估，不能以框架名称作保证。实际构建使用Ubuntu 22.04.5、CANN 9.1.0和配套910B算子包。

新增源码、头文件和CMake使用仓库完整CANN Open Software License Agreement Version 2.0文件声明；README按官方接口说明和完整调用示例编写。设计沿用受邀项目既有wjh0224任务目录，代码位于ops-blas原生arch22目录。设计与代码已上传受邀仓库master，此状态不等同于上游PR或正式验收申请。

本次更新汇总已实现路径、当前精度标准与实际证据边界；2383项历史精度、37项目标验证及200项性能基线不混算为同一版本通过。当前源码已在910B4重建，29项定向回归通过，含大幅值交替输入与高斯093/095；完整精度与200项性能验收仍在进行，当前结论为未通过任务总验收。


测试复现与报告入口：[测试README](https://atomgit.com/wjh0224/ops-blas/blob/master/test/gemm_grouped_batched_ex/README.md)、[自测报告](https://atomgit.com/wjh0224/ops-blas/blob/master/test/gemm_grouped_batched_ex/SELF_TEST_REPORT.md)。报告按官方模板字段整理，保留当前精度反例、性能失败及workspace数据缺失。
