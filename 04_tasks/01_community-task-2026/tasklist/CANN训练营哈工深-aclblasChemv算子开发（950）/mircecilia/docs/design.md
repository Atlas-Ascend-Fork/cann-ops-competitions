# 需求背景（required）

## 需求来源

[CANN训练营哈尔滨工业大学（深圳）第03节任务引导](https://gitcode.com/org/cann/discussions/335#tid-4fb9043b18e940628d1b353eac906b90)，
[aclblasChemv（950）社区任务](https://www.hiascend.com/activities/task-center/details/00932c7652e946c7a7184aa98339a35a)。
本文按官方`design_template.md`的章节组织，尚未提交设计PR或获得评审通过。

## 背景介绍

ops-blas已提供实数Ssymv和复数对称Csymv。本任务补齐complex64 Hermitian矩阵向量乘，
适配Ascend 950PR和CANN 9.1.0，公共API与其他产品线共用。

### 现状分析

基线仓库：`https://gitcode.com/cann/ops-blas.git`，commit
`708887f379bb89d35381fd80dde0c1f8d792986b`。本次获取时官方仅有master分支，无版本标签；
已经在CANN 9.1.0实机编译确认兼容性。

| 对象 | 数据类型 | 内存/布局 | 约束 |
|---|---|---|---|
| alpha | complex64 | Host标量 | 不可为空 |
| beta | complex64 | Host/Device标量 | 不可为空 |
| A | complex64 | Device，列主序lda×n | 仅引用选定三角，忽略对角虚部 |
| x/y | complex64 | Device，带正/负步长 | 存储长度≥1+(n−1)×abs(inc) |
| n/lda/inc/uplo | int/枚举 | Host属性 | 按任务书校验 |

# 需求分析（required）

## 需求描述

实现`y=alpha*A*x+beta*y`，A满足`A=A^H`；实现位于`blas/symv/arch35/`，
API声明位于`include/cann_ops_blas.h`，测试位于`test/symv/chemv/arch35/`。

## 需求拆解

1. 实现上下三角、共轭、列主序padding、正负步长和y原地更新。
2. 严格实现参数校验顺序、n0和alpha0/beta1 quick return、alpha0不读A/x、beta0不读旧y。
3. 使用handle绑定stream直调Ascend C kernel，Host常规路径异步执行。
4. 覆盖原始1000条精度和200条性能用例，提供设计、复现步骤、日志和报告。
5. 每条性能用例达到H100基线0.4倍以上，不使用平均倍率代替逐条判断。

# 详细设计（required）

## 算子分析

### 数学公式

`y_i = alpha * sum_j(H_ij * x_j) + beta * y_i`

UPPER时：i≤j读`A[i+j*lda]`，i>j读`conj(A[j+i*lda])`；LOWER反之。
对角项仅使用`real(A_ii)`。负步长逻辑i对应物理偏移`(n−1−i)*(-int64_t(inc))`。

### 支持数据类型

complex64，实部/虚部分别为float32；`aclblasComplex`使用仓库公共定义。

### 支持形状

n×n方阵，lda≥max(1,n)；x/y逻辑长度n。n为运行时参数；不涉及广播和额外视图语义。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

校验顺序固定为：handle判空→n<0→n0成功→uplo/lda/步长/标量指针→读标量→
alpha0/beta1成功→必需数据指针。INT_MIN按照随附负向CSV拒绝，偏移和长度计算使用64位。

alpha按任务书Host标量处理；beta查询指针位置。Device beta依照handle stream读回后再判定
quick return及数据指针条件，该额外兼容路径有stream同步；Host路径无此同步。

分派基于实际参数和可用workspace，不依赖测试名称或数据内容：

| 路径 | 选择条件 | 分核/工作空间 |
|---|---|---|
| 一般复数标量 | 非alpha1/beta0 | 每线程一行，128线程/block，不使用workspace |
| 小规模/带步长 | alpha1/beta0，分块条件不满足 | warp协作一行，128线程/block，不使用workspace |
| SIMD顺序累加 | alpha1/beta0，incx=incy=1，n≥128 | 每组64个输出行，按列依次累加；不使用GM workspace |

T=ceil(n/64)，使用不超过AIV核心数的blocks。超过一轮但不超过两轮时，
将直接三角占比较高的两个输出tile分配给同一个核心，其余较贵的共轭tile各占一个核心，
减少尾部少核阶段；上下三角采用相反的行tile映射。

A矩阵按列分块搬入UB，使用double buffer。矩阵UB列pitch为136个float，
避免幂次stride的访问冲突。镜像部分在UB中转为实部/虚部分离、pitch65的布局，
减少跨列gather冲突。总UB约105KiB，不申请GM workspace。

#### 3.2.2 kernel侧设计

**一般复数标量路径**保留Netlib的运算组织：直接三角项先计算alpha*x，再与A相乘累加；
镜像三角项先对conj(A)*x求和，再乘alpha。关闭FP contraction，减少复杂标量在实虚部相消时
因重排产生的绝对误差。该路径从三角合法位置获取数据，对角虚部不读取。

**协作路径**采用32lane处理一行不同列，warp内归约；用于alpha1/beta0的小规模及带步长输入。

**SIMD顺序累加路径**使用64条寄存器lane对应64个输出行，沿逻辑列顺序累加，
在每个64列块之间保留实虚部累加器。直接三角按列DMA到UB；镜像三角仅读取合法三角块，
在UB中转为带skew的平面布局，再顺序加载并取共轭参与计算。
列循环按4次展开以减少控制开销，保留每个累加器的运算先后，完整精度及op性能在冻结版本上复测。

**对角块**先清零UB，再按列只搬运所选三角区间，禁止读取另半三角后再掩码。
对角虚部不参与乘法；对角内用lane mask区分直接项、对角项和镜像项，
维持Netlib的累加组织。所有mask计数使用独立变量，避免UpdateMask修改循环边界。

**流水线与输出**A和x输入队列各使用两个buffer，使下一块DMA与当前VF计算重叠。
上下三角的块顺序和尾块长度按运行时n/lda确定。计算结束后只写回有效y元素，
整个API仅启动一个kernel；无原子累加和跨核归约workspace。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR | √ |
| Ascend 950DT | 未验证 |
| A2/A3 | 未实现 |

## 算子约束限制

- incx/incy≠0且非INT_MIN；后者是随附用例的增强约束。
- 不支持广播；A/x为只读，y原地覆写；读回前同步handle stream。
- 调用者应遵循公共handle和stream的并发使用约定。
- 不以确定性或bit-exact为目标；浮点精度依照任务书规则验证。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| golden | Netlib BLAS 3.10.0的cblas_chemv，CblasColMajor | 任务书§3.1/3.2 |
| 精度 | 实虚部分别atol=rtol=2^-13，有限值matched_ratio≥0.99，最大误差≤max(1e-2,32×ULP)，ULP锚点为最大误差点的golden | 任务书§3.2 |
| 非有限值 | 与有限值统计分开；NaN/Inf等效规则按任务书；所有特殊位置必须通过 | 任务书§3.2 |
| 性能 | 每case平均task duration≤H100耗时×2.5 | 任务书§3.3 |

测试使用原始CSV和固定seed，并补充32组均匀/正态各半的复标量、Device beta及空handle测试。
将未引用三角、padding和对角虚部填NaN；检查A/x未修改、y步长间隙未覆盖。
GTest XML核对1200个CSV case身份（1000条原始精度用例及200条性能输入的附加精度检查），
另核对3项补充测试身份；零执行、漏测、跳过、异常退出均失败。

性能保留60个有效样本和前10样本均值；样本次数在任务书/README中表述不一致，因此完整保留数据。
环境预装msprof op组件将容器逻辑设备0转换为物理设备1，再将1传给要求逻辑编号的接口，
导致驱动channel查询失败（h2d_type，dev_id=1，ret=8）。已在独立目录回移官方msopcom的
逻辑编号转换修复，恢复op模式OpBasicInfo采集；系统驱动及算子动态库未修改。
采集保留kernel replay、warmup与逐次CSV，完整结果见自测报告；正式提交时披露工具修复与来源，
其接受性以任务方评审为准。此前应用级task duration保留为独立诊断数据，不与op模式混算。

## 兼容性分析

新增公共C API，既有接口签名和行为不变。实现仅在arch35下收集编译，
不满足SIMD分派条件的输入使用通用路径。设计评审、代码上库和正式验收尚未完成。
