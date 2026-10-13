# 需求背景（required）

## 需求来源

CANN社区2026年9月任务81：aclblasSgemmGroupedBatched算子开发(A2/A3)。
基于ops-blas提交617bb065849dc60fbc543c364dcf36f6e7660905，新增arch22实现；上游已有arch35接口。
参与者：shine4real（个人参与，提交目录名为GitCode用户名）。
设计文档个人仓：https://gitcode.com/shine4real/cann-ops-competitions 。
代码个人仓：https://gitcode.com/shine4real/ops-blas 。
本文用于申请设计评审，尚未获得审核通过结论；实现和实测范围见下文。
当前开发实现尚未推送到上述个人代码仓，已测源码按固定上游基线和SHA256识别。

## 背景介绍

Grouped GEMM在一次BLAS调用中承载形状、转置、缩放及批量各自独立的矩阵组，适合异构小批量矩阵计算。
A/B/C输入是Host扁平指针数组，元素是Device地址，组内不要求地址相邻；前导维支持列主序padding。
arch35实现使用MicroAPI，arch22采用CANN9.1.0 FP32 Matmul与AIV向量接口。

# 需求分析（required）

## 需求描述

每组执行C[idx]=alpha[g]*op(A[idx])*op(B[idx])+beta[g]*C[idx]，idx为前组groupSize之和加组内下标。
FP32，列主序，N/T两种转置；OP_C与其他枚举按任务书返回INVALID_VALUE。
保持已有公共API签名、handle/stream和workspace机制。

## 需求拆解

1. 下发前检查全部组、数组、前导维、矩阵地址与总batch溢出；无效输入不部分执行。
2. groupCount=0且handle有效时直接成功；m/n/groupSize=0合法no-op，空组仍保留扁平槽位。
3. alpha=0或k=0只计算beta*C；beta=0不读取初始C。NaN/Inf及极值语义与Netlib参考对齐。
4. 以Cube输出分块并行跨batch调度；以AIV执行alpha/beta合并和特殊数值路径。
5. 保存精度、msprof设备时间、输入/源码校验值与复现信息，按任务交付目录准备材料。

# 详细设计（required）

## 算子分析

### 数学公式

C_gj=alpha_g*op(A_gj)*op(B_gj)+beta_g*C_gj。
列主序转换为Cube的行主序乘法：C^T=op(B)^T*op(A)^T，调换A/B输入和相应转置配置。

### 支持数据类型

输入输出FP32，alpha/beta为Host FP32值数组；形状和批量为Host int数组。

### 公共接口及参数合同

沿用上游include/cann_ops_blas.h中的声明，不增加产品专属API：

```cpp
aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray,
    const float* const* Barray, const int* ldbArray,
    const float* betaArray, float* const* Carray, const int* ldcArray,
    const int* groupSizeArray);
```

| 参数类别 | 合同 |
|---|---|
| handle | nullptr返回ACLBLAS_STATUS_HANDLE_IS_NULLPTR；有效handle绑定调用stream |
| groupCount | 非负；为0时不解引用其他数组 |
| 逐组数组 | groupCount>0时不能为空；每项形状、前导维、系数、转置及batch各自独立 |
| 形状与batch | m/n/k/groupSize非负；总batch不超过INT_MAX |
| 转置 | 仅N/T；实数OP_C及未知枚举返回INVALID_VALUE |
| 前导维 | lda>=max(1,transa=N?m:k)，ldb>=max(1,transb=N?k:n)，ldc>=max(1,m) |
| A/B/C指针数组 | 位于Host，元素为Device矩阵地址；长度为sum(groupSize)，元素均非空 |
| 空操作 | m/n/groupSize为0不写C；空形状组仍按groupSize保留扁平地址槽位 |
| 系数 | alpha/k为0时只计算beta*C；beta为0不读取初始C；特殊值按已验证参考语义处理 |
| 输出与同步 | C不得与A/B重叠；调用异步，读取C前同步绑定stream |

负维度、非法前导维、空数组/矩阵地址及无效枚举返回ACLBLAS_STATUS_INVALID_VALUE。
参数校验完成后才下发计算；正常完成下发或合法no-op返回ACLBLAS_STATUS_SUCCESS。

### 支持形状

不同组独立m/n/k、转置、lda/ldb/ldc与groupSize。紧凑布局及leading-dimension padding，无stride接口。
C不能与A/B重叠，不提供超出前导维语义的视图/广播。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

先对所有组和地址作完整合法性检查，之后再下发；扁平索引按groupSize累加。
优先选择顺序AIV路径：k>=512，且需要alpha/beta合并或m/n小于16，系数均在(-65536,65536)、
alpha非零、workspace至少64字节。其余满足m/n至少16和资源容量的普通系数走Cube，剩余情况功能回退。
每次按值传入最多64套矩阵地址和该组参数，结构小于3072字节。无需Host临时元数据持有到异步执行结束。

alpha=1,beta=0时Cube直接输出到C；其余普通系数在handle workspace写入临时AB结果。
临时前导维对齐到8个FP32元素。batch容量取min(64,workspace字节数/(alignedM*n*4))，不足则功能回退。
后续组/调用复用workspace时由绑定stream保持Cube与合并kernel顺序；无额外Host同步。

##### 分核策略

128×128输出tile按矩阵索引及行/列tile编码成线性任务，轮询分配到动态查询的AIC。
数值检测/合并kernel将不同矩阵分配到AIV，每个矩阵单核拥有，避免标量store跨核共用缓存行。
顺序AIV路径按每列64行输出块跨矩阵分配任务，沿K按参考顺序累加。
其他小形状和不满足资源门槛的输入走单矩阵单核功能路径。

##### 数据分块和内存策略

当前两台平台API实测24AIC、48AIV、每核UB196352字节。
检测kernel使用5×4096×4=81920字节UB；紧凑矩阵按线性4096元素DMA扫描，padding矩阵按有效列扫描。
将FP32绝对值位模式转换为非负int32再cast到float规约，覆盖全部NaN/Inf及极大有限值。
不扫描padding作为有效数值。检测阈值约6e29，检测到后才执行Netlib顺序的标量重算。
合并阶段重用检测UB片区，256元素分块计算alpha*temp+beta*C，beta=0不读C，写回有效行。

#### 3.2.2 kernel侧设计

Cube采用128×128×64基本分块，归约分段上限4096，关闭HF32，提供NN/NT/TN/TT模板分支。
直接输出路径之后检测输入；遇到极值重算C。带beta路径保留初始C，重算时从初始C构造结果。
标量路径对非转置A先缩放B再累加，对转置A先规约再缩放；末尾显式写回GM缓存。
大规约与一般缩放系数直接在顺序AIV路径计算，避免Cube规约重排及先算AB后乘alpha导致的误差。
K按256元素搬运；A使用64×256 FP32，B使用256×8 FP32，辅助区5×64 FP32，总UB75008字节。
转置A以Gather抽取行；转置B以带padding的二维DMA收集K片段。
先检测极值，每矩阵使用独立64字节workspace标志；异常输入单核重算，普通输出任务跳过已重算矩阵。
transa=N时先构造beta*C，再按K缩放B并累加；transa=T时先累加，再做alpha/beta合并。
所有kernel使用调用时捕获的参数和地址，不从可能已失效的Host数组读数据。

## 支持硬件

| 产品 | 当前验证 |
|---|---|
| Atlas A2 | Ascend910B2，CANN9.1.0，最新版400条回归及1000条原任务、3项专项测试通过 |
| Atlas A3 | Ascend910_9382，最新版400条回归和完整965条正态扩展测试通过 |

## 算子约束限制

Host数组指针及元素必须符合API合同；不探测每个地址是否属于Host分配。
C与A/B不得重叠；同handle跨stream并发共享workspace不在当前已验证范围。
workspace不足会走功能回退，该路径可能较慢，不以小样例通过证明全规格性能。

# 可维可测分析

## 精度标准/性能标准

精度：Netlib BLAS3.10.0 cblas_sgemm，全有效输出mixed tolerance：rtol=atol=2^-13、匹配率>=0.99，最大绝对误差<=max(1e-2,32*ULP(golden))，golden取最大绝对误差处。
NaN/Inf需一致，padding与no-op存储逐字节保护。ULP规则采用任务指定的
[官方混合容差标准](https://gitcode.com/cann/opbase/blob/a33c711799bfec3fbb043bc3bc6bd9ab02d5649a/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
原始任务精度1000条、性能200条保持原字节；新增965条正态配对用例，使用未截断normal_distribution。
正态输入mu在[-5,5]、sigma在[0.1,2]取样；正常输入配对采用相同形状和不同固定种子。

性能：msprof设备Task Duration合并每次API的所有kernel，不包含Host/传输；预热10、采样60、去最大10取50均值。
同时保留60次均值和全部原始采样；正式任务采样口径及910B3设备结果尚未完成。
最新版A3五个典型case自测：163.17560/1457.90906/10672.81972/72182.93258/237488.82272us。
上述结果为Ascend910_9382周期输入下的设备kernel耗时合计，不包括Host、传输或kernel间隔，
不用于910B3正式验收结论。原始日志、全部70次调用采样及分析脚本已本地归档，
后续随个人代码仓task_submission交付；当前尚无这些证据的公开链接。
其余195条性能用例待测，不把不同硬件或口径的数据作为正式验收通过结论。

## 兼容性分析

公共接口及arch35实现保持原样。arch22测试使用公共REFBLAS_LIB，取消对本架构额外强制OpenBLAS链接。
代码新增在既有上游自动收集路径blas/gemm_grouped_batched/arch22与test/gemm_grouped_batched/arch22。
任务书多一层sgemm_grouped_batched的测试目录要求与当前上游框架不同，交付前需对齐。
没有Git提交、推送、PR或设计评审结论；源码版本以基线完整提交号及SHA256清单标识。

## 已测实现标识

本次开发尚无最终提交号；下面是已在A2/A3核对一致的算子源码SHA256，不以基线号替代开发提交号。

| 文件（blas/gemm_grouped_batched/arch22/） | SHA256 |
|---|---|
| gemm_grouped_batched_kernel.cpp | `062023bd4e965e5b0312a8895835f03798205e55bb62eea5d2147b1d31457eee` |
| gemm_grouped_batched_host.cpp | `606ba676ef45577a478cc426d6247c329be71fd6e2fbfe04c91793fbd0827a26` |
| gemm_grouped_batched_tiling.h | `efeac923e5f890b8e0835c9d0992be1ba64c45ff902b805066bd64b47e02afd9` |
