# 单精度复数LU分解与批量求逆（950PR）算子设计文档

作者：TreamTik

# 需求背景（required）

## 需求来源

本设计对应[10月社区任务：单精度复数LU分解、求解和批量求逆（950）](https://www.hiascend.com/activities/task-center/details/237f144aab534448b58a123650b2fb35)。目标代码仓为[cann/ops-solver](https://gitcode.com/cann/ops-solver)，适配Ascend 950PR。

## 背景介绍

LU分解将矩阵转换为三角因子，用于线性代数中的求逆等运算。单矩阵求逆和基于LU因子的批量求逆复用分解结果，小矩阵直接批量求逆则合并分解与求逆过程。

接口采用列主序COMPLEX64矩阵，支持Device指针、可变leading dimension、主元与info。Host负责参数检查和任务调度，核心计算由Ascend C Kernel执行。

# 需求分析（required）

## 需求描述

| 接口 | 功能 | 对标接口 |
|---|---|---|
| aclsolverCgetrf_bufferSize | 查询LU分解工作空间 | cusolverDnCgetrf_bufferSize |
| aclsolverCgetrf | 原地LU分解，可选部分主元 | cusolverDnCgetrf |
| aclsolverCgetri_bufferSize | 查询单矩阵求逆工作空间 | LAPACK cgetri工作空间查询语义 |
| aclsolverCgetri | 根据LU因子和主元原地求逆 | LAPACK cgetri |
| aclsolverCgetriBatched | 根据LU因子和主元批量求逆 | cublasCgetriBatched |
| aclsolverCmatinvBatched | 小矩阵直接批量求逆 | cublasCmatinvBatched |

公开签名按任务书2.3定义，统一返回`aclsolverStatus_t`。四个计算接口与两个查询接口作为整体交付，复用`aclsolverCreate`、`aclsolverDestroy`、`aclsolverSetStream`和`aclsolverGetStream`。

## 需求拆解

1. 公共头文件定义`aclFloatComplex`，以`extern "C"`保护C ABI；维数使用32位`int`。
2. 单矩阵接口支持合法padding；批量接口接收Device指针数组，允许各矩阵位于不同地址。
3. Host校验标量参数并计算workspace；Kernel读取真实Device数据，写回主元和info。
4. 根据形状、容量、对齐与并行任务数选择路径，兼顾小矩阵启动成本和大矩阵计算效率。
5. 按任务附件规划全量精度、性能与边界用例，验证接口语义和确定性。

# 详细设计（required）

## 算子分析

### 数学公式

LU分解采用逐列部分主元。记正向行交换的乘积为Q，则`Q*A=L*U`；等价写为`A=P*L*U`，其中`P=Q^-1`。`devIpiv[j]`记录第j步交换的1-based行号。空主元指针表示不选主元，此时`A=L*U`。

分解结果原地写入A：对角及上方存U，严格下方存L，L的单位对角不显式存储。主元采用确定的比较规则，相同输入和stream串行重复执行时保持结果一致。

单矩阵求逆和基于LU的批量求逆通过三角求解获得`C=A^-1`，再按主元顺序执行置换。直接批量求逆对原始矩阵完成分解与三角求解，输出独立逆矩阵。奇异矩阵写入对应的正info。

复数乘法采用`(a+bi)(c+di)=(ac-bd)+(ad+bc)i`。复数除法根据分母实虚部大小选择比例计算，减少直接平方求和产生的溢出风险。

### 支持数据类型与布局

| 数据 | 类型与位置 | 约束 |
|---|---|---|
| 矩阵与workspace | COMPLEX64，Device | 列主序，实虚各float32 |
| 维数与leading dimension | int32，Host | 非负维数；leading dimension合法 |
| 主元与info | int32，Device | 主元为1-based；info真实写回 |
| 批量矩阵地址表 | Device指针数组 | 表项指向Device矩阵；不要求连续batch布局 |
| handle与stream | Host句柄 | 计算使用调用方stream |

### 支持形状

Cgetrf支持m×n矩形及方阵，`lda>=max(1,m)`。Cgetri支持n×n方阵，`lda>=max(1,n)`。CgetriBatched分别使用`lda>=max(1,n)`和`ldc>=max(1,n)`。CmatinvBatched要求`0<=n<=32`，输入和输出leading dimension分别检查。

维数和batch上限以任务书及附件用例共同确定，测试规划覆盖附件中的宽矩阵、大batch与padding，不根据用例编号选择实现。零维和零batch按空问题处理。计算地址偏移和workspace大小时使用宽整数，并检查乘加与接口返回范围。

## 算子实现

### 实现方案

#### host侧设计

**参数与错误处理。** 检查handle、维数、leading dimension、必需指针和workspace容量。非法参数返回对应状态，在info可写时写入负参数序号。记录带接口名和关键参数的错误信息。分配、拷贝或平台查询失败时向上返回状态。

**工作空间。** bufferSize查询按同一分块规划计算空间，单位为`aclFloatComplex`元素个数。工作空间包括临时矩阵、分块中间结果、主元归约及阶段同步所需存储。查询不读取Device矩阵内容。调用方按查询值分配，计算接口复用该空间。

**调度与分核。** 根据m、n、batch、对齐、UB容量与可用核心数生成任务。小batch优先控制启动开销；大batch将矩阵或矩阵行片分配到各核。分块更新依据tile几何选择向量或Cube计算。任务映射在各阶段复用，尾块使用相同的有效尺寸约束。

| 运算场景 | 设计路径 | 主要目的 |
|---|---|---|
| 小阶LU | 寄存器/UB驻留，核内完成pivot与消元 | 减少GM往返和多次启动 |
| 宽矩阵LU | 小面板分解，协作完成右侧三角求解 | 保持连续访问和并行度 |
| 中大矩阵LU | 面板分解、确定性pivot、TRSM与分块更新 | 用Cube承担较大更新块 |
| 小阶LU求逆 | 寄存器或分组UB驻留 | 批量复用数据并减少标量访问 |
| 中阶LU求逆 | LU驻留，输出按行分片；容量紧张时压缩三角存储 | 平衡并行度与UB占用 |
| 大阶LU求逆 | 对角块求逆、分块三角求解与Cube更新 | 提高计算密度 |
| 小阶直接批量求逆 | 核内融合分解与求逆；合适几何下跨batch SIMD | 减少中间结果与kernel次数 |

**容量与对齐。** UB预算同时计入实虚分量、驻留因子、输出行片、临时向量和同步开销。先判断稠密驻留是否满足预算，再判断压缩三角布局，最后选择通用分块路径。指针对齐由当前调用实际地址确定；不满足宽读写条件时采用标量宽度读写模板，保持相同计算语义。

#### kernel侧设计

**搬入。** 从Device地址表读取矩阵地址，按列主序和leading dimension搬运有效区。使用连续或分块搬运，保留padding。批量相邻矩阵在核内成组处理，摊薄地址解析与启动成本。

**LU计算。** 在面板内搜索主元，确定性归约选择主元行，完成换行与消元。面板完成后执行右侧三角求解，再更新尾部矩阵。大更新块将复数运算拆为实数矩阵乘法，在保持精度要求的模式下使用Cube。

**求逆计算。** 根据U、L与主元完成三角求解。小阶路径尽量保留因子和中间结果；中阶复用同一LU处理多个输出行片；大阶按对角块和非对角块推进。直接批量求逆将小矩阵LU与求逆融合，独立写回每个矩阵的info。

**同步与搬出。** CopyIn、计算与CopyOut之间使用明确的数据依赖。跨核阶段仅在生产者数据就绪后启动消费者。提前返回路径与正常路径恢复所使用的寄存器及同步状态。输出只写有效矩阵区，输入保持和padding由测试检查。

## 支持硬件

Ascend 950PR（Atlas 950），CANN 9.0.0及以上配套版本。具体分核、UB和Cube参数从目标平台能力及工具链约束推导。

## 算子约束限制

- 输入为有限COMPLEX64数据，不支持broadcast。
- Cgetri和CgetriBatched输入为LU因子及对应主元；CmatinvBatched输入为原始矩阵。
- Cgetri原地输出；批量求逆输出与输入不重叠，批内输出之间也不重叠。
- CmatinvBatched在n>32时返回非法参数，不转调其他接口。
- info=0表示成功，负值表示参数非法，正值表示奇异位置。
- no-pivot遵循对标语义，调用方负责其数值适用性。

# 可维可测分析

## 精度标准/性能标准

令`eps=2^-24`，矩阵1范数按任务书定义。CPU采用同精度COMPLEX64参考链路。

| 接口 | 归一化残差 | 通过标准 |
|---|---|---|
| Cgetrf | 按主元还原后，`||L*U-A||1/(n*||A||1*eps)` | `ratio<=max(5*ratio_cpu,0.5)` |
| Cgetri / CgetriBatched | `||I-C*A||1/(n*||A||1*||C||1*eps)` | `ratio<=max(5*ratio_cpu,0.05)` |
| CmatinvBatched | 同逆矩阵残差，batch内逐矩阵判断 | `ratio<=max(5*ratio_cpu,0.5)` |

性能逐case与附件`bench_result.json`的平均耗时关联，要求`T_NPU<=T_GPU/0.35`。用msprof采集单次API的全部计算kernel耗时并求和，排除预热、Host调度与数据传输。测试报告记录预热、计时次数、软硬件版本与GPU基准来源。任务书3.4未设置内存容量门槛。

## 测试规划

全量附件包含652条：Cgetrf 148条、Cgetri 200条、CgetriBatched 89条、CmatinvBatched 215条。测试计划逐条列明形状、主元选项、leading dimension和性能金标，不以汇总平均代替逐条门槛。

专项覆盖矩形、padding、离散Device地址表、对齐边界、分块边界、零维、零batch、非法参数、奇异和近奇异矩阵、no-pivot、非默认stream及重复执行确定性。数值测试覆盖任务书要求的输入分布与量级边界；内存插桩检查越界、同步和退出状态。测试结果单独记录，不写入设计文档。

## 兼容性分析

保留ops-solver的handle、stream和构建体系。公开接口使用C ABI，参数顺序与Device/Host归属以任务书2.3为准。调用方先查询并分配workspace，再在同一stream中提交计算与后续操作。原有简化原型的调用者按公开签名同步迁移，避免混用Host指针和连续batch旧约定。
