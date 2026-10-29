# 需求背景（required）

## 需求来源

2026-10-28 社区任务：[aclblasGemmBatchedEx算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/d945b8eabdc44fa98939d23f5cef0517)。任务书原始ZIP SHA256为a46f326502f9b8a8abc39865e20331117b8a82893ffbb1b219df4b6e63f4c38b。

## 背景介绍

ops-blas已有aclblasGemmBatchedEx共用接口和arch35实现，本设计新增arch22 FLOAT32实现，不修改公开API签名，不影响arch35声明的混合类型能力。批次由Device指针数组描述，各矩阵可以分散分配，不能当作固定stride的一块连续矩阵处理。

# 需求分析（required）

## 需求描述

对所有批次i，计算列主序C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]。A/B/C均FLOAT32，computeType=ACLBLAS_COMPUTE_32F，algo=ACLBLAS_GEMM_DEFAULT。实数路径的OP_C与OP_T等价，共覆盖9组公开转置组合。

## 需求拆解

1. 维度、前导维、类型、枚举、标量及数组空指针校验在quick return之前完成，返回值与任务书§2.4一致。
2. m/n/batchCount为0时合法no-op；k=0或alpha=0仅处理beta缩放，不引用A/B矩阵内容；beta=0不读取旧C；缩放路径beta=1不改写C原始位。
3. Host标量按调用时值传入；Device标量由设备内核按当前stream顺序读取，不为读取标量做Host同步，不缓存数值。
4. 支持非紧凑leading dimension，保护C每列padding。kernel不得读Host数组中的矩阵地址，Device数组不拷回CPU逐batch下发。
5. CANNLab 910B3 A2作为主要实测设备，CANN9.1.0，CPU golden为Netlib BLAS3.10.0 cblas_sgemm。

# 详细设计（required）

## 算子分析

### 数学公式与布局

利用C^T = op(B)^T * op(A)^T，将列主序BLAS映射到Matmul的ND行主序。Matmul左操作数来自B，右操作数来自A；四种实数内核特化覆盖9种公开枚举组合。物理lda/ldb/ldc分别传入SetOrgShape，不使用逻辑m/n/k替代padding stride。

### 支持形状与类型

非负m/n/k/batchCount，满足任务书前导维约束；主路径FLOAT32。不宣称arch22支持可选FP16/BF16/FP8/INT8路径。返回NOT_SUPPORTED时不进行设备计算。

## 算子实现

### Host侧设计

从handle读取绑定stream，完成参数校验，识别标量位置，将Host值或Device地址放入小型POD tiling。运行时查询实际Cube/Vector核心数，不固化48核或9382环境。

矩阵乘任务为(batch, reductionPart, nTile, mTile)，按block-stride均分给实际Cube核心。每核从Device指针数组读取所属批次真实地址。所有整数地址偏移和任务数量使用64位；工作区大小计算检查溢出。分块依赖形状、转置与实际核心数，不依赖case编号或固定随机种子。

Host beta=0且只有一个K分段，或Host alpha=1、beta=0的长K顺序分段路径，Matmul产品直接写入C，无产品GM workspace；不需要输入整理的普通Host alpha=1、beta=0调用只下发一次Cube kernel。m/n/k均不超过32且Host alpha绝对值不超过1时，采用单次Vector小矩阵内核。其余非单位alpha采用同stream epilogue；beta非零或未知Device beta时临时保存产品，再与原C组合。

为控制长K归约误差，K>2048时按不超过1024的K段计算。Host alpha=1、beta=0时，每个输出tile由同一Cube核按K段顺序处理：首段覆盖写C，后续段通过AtomicAdd累加到同一C tile；不存在跨核抢写同一输出，且beta=0不读旧C。该路径不再分配各段完整产品数组。其余需要alpha/beta组合的长K调用保留独立产品区、TwoSum补偿合并和缩放。新版本完整1000+200精度通过，200性能过程handle保留容量最大为33554432 bytes（32 MiB，库默认容量）；此前4096³、batch4会保留1GiB。容量口径不等同于当前算法最小请求或总HBM。

部分较长K的转置输入在同stream上先整理物理步长并补齐K。A/B分别从设备指针数组获取地址，独立计算其原始与整理后leading dimension，不改变输入内容。未转置路径与不获益的短K形状保留直接读取。申请工作区时计入所有产品段和输入整理区，检查整数溢出，复用handle工作区并遵守用户设置的容量。

较大缩放（Host alpha绝对值>1）、Device alpha或K超过静态Matmul的8192范围时采用顺序K的Vector路径，以独立输出行并行。真机逐元素诊断显示A2的Axpy/MulAddDst先舍入乘积，Netlib使用FMA；因此该路径通过按尾数位拆分的TwoProduct及TwoSum残差补偿减少二次舍入，保持参考K顺序，无CPU计算兜底。选择条件仅基于参数和数值属性，不依赖测试编号或矩阵内容。

### Kernel侧设计

1. 已选定的原生FP32实现使用Ascend C Matmul API、FP32输入及累加，明确禁用HF32。多分量近似计算仍属独立实验，未纳入该冻结版本。
2. 使用常规、狭长、短K和窄输出形状分块。当前已验证的短K方案单核输出128×256、base tile为128×128×64；常规路径根据转置采用不同的FP32分块。尾块用SetSingleShape传实际范围，SetOrgShape保留真实物理步长。后续静态K分段优化须在独立版本实测后才能纳入最终实现。
3. 直接写C的每个输出tile只有一个Cube所有者，长K各段由其按固定顺序累加。其它产品区按part/batch/tile独立分配。Device指针数组及A/B矩阵只读，输出无跨核重叠。
4. Vector epilogue每次处理单列至多1024个FLOAT32元素，DataCopyPad按实际字节数搬运，七个4KiB UB缓冲，并用MTE2/V/MTE3事件保护读、算、写和缓冲复用。
5. alpha=0/k=0走缩放，beta=1直接返回，beta=0写零且不读旧C；保留quick return的NaN payload、负零及padding位。
6. 输入整理、Cube计算和epilogue均在handle stream上，利用stream顺序建立依赖。长K单位alpha/零beta走单核输出tile的顺序分段，其他长K各段写入独立产品区再统一归约；均无跨核输出重叠、无全设备同步、无CPU计算兜底。同一Matmul对象在全部非分K输出块完成后统一End，分K时保留逐段End。
7. 小矩阵路径把8列×32行映射到256个Vector lane，A/B以物理stride搬入UB、Gather定位元素，K按参考顺序累加。按任务书§3.2的非有限值等价规则，将溢出结果规范化为NaN，避免依赖并行归约的溢出符号；有限值仍执行完整误差检查。
8. epilogue及顺序K路径同时保护MTE3→V和MTE3→MTE2，防止下个任务提前覆盖仍被计算使用的UB输入。

### 调优计划

以完整200条固定性能case为准，比较tile尺寸、batch/矩形任务排布及小矩阵单核开销，保留通用的shape分段依据，不根据case编号或测试数据选择实现。先验证正确性，再按msprof实际瓶颈调整。

## 支持硬件

Atlas A2/A3系列产品（arch22）。任务书§3.1允许自验证一种款型；选择910B3 A2，CANN9.1.0。无需A5、9382指定子型号、独立GPU或特定OS镜像；标杆采用官方提供的GPU CSV，不将该数据冒充本项目现场实测GPU结果。

## 算子约束限制

不改公开接口；Aarray/Barray/Carray必须是Device数组，数组元素指向Device矩阵。无输出重复地址及输入输出重叠承诺。非零beta的产品workspace按实际容量分配，空间不足返回库对应错误，不静默回退或减少验证矩阵范围。

# 可维可测分析

## 精度标准/性能标准

- 完整保留官方CSV的1000条精度/边界/负向用例，200条性能用例也验证结果。Netlib3.10.0 cblas_sgemm逐batch生成golden，比较所有有效元素，padding单独逐位检查。
- FLOAT32 atol=rtol=2^-13，matched_ratio>=0.99，另外逐元素要求abs(error)<=max(0.01,32*ULP(expected))；Inf/NaN单独按§3.2等价规则计数，不能跳过有限/非有限不匹配。alpha=0,beta=1另做原始位检查。
- 补测Device标量原地变化、Host标量调用后修改、Host/Device混合标量、同stream异步标量拷贝顺序、独立批次分配、offset、输入只读、两个stream、用户workspace不足及多次执行、beta=0忽略旧NaN输出。
- 保存完整精度日志和完整性能日志、每case参数、全部有效样本及msprof原始文件。200条表逐行匹配GPU CSV，列gpu_us、npu_us、ratio=GPU/NPU、样本数和结论，缺行/重复匹配/缺原始数据均不能判通过。
- 任务书§3.3为ratio>=0.8，但§7.4写>=GPU；开发目标按更严格ratio>=1.0，同时报告两种阈值。先warmup5次，再51次有效调用采样，不删除慢样本。
- GPU CSV来源说明存在A100/P102-100表述差异，保留官方文件和哈希，在自测报告如实列明，不编造GPU型号或额外测量。

## 兼容性分析

仅新增arch22实现与该架构测试，复用现有公开头文件、handle和stream管理。arch35原有实现与测试保持兼容。设计PR与代码PR分开；测试完成后记录最终代码HEAD/tree，设计合入及代码变更时检查是否需要复测。

当前状态（2026-10-29）：冻结的native_serial_parts原生FP32版本在910B3/CANN9.1.0上完成全部1000条精度/边界/负向及200条性能用例的全矩阵精度验证，共1200/1200通过；42条基础/工程及Netlib并行参考逐位一致性检查通过。同一冻结版本完整200性能按5次warmup、51个有效样本和完整设备跨度统计，198/200达到GPU/NPU>=0.8，195/200达到>=1.0，最低0.6843953940。1111、1115仍低于0.8；严格1.0另有1087、1129、1199未达标，不以四舍五入计通过。200条测得最大handle保留工作区为32 MiB。完整精度、200行性能比值、全部51采样、原始task_time和冻结源码证据SHA256：6ebdc6b3a64bce6f13bfb8ab63bdf5fd1b0905f054fb193e532682228d02ed8e。三份阶段XLSX已按此版本重新生成，仍明确标记性能缺口和GPU工作区基线缺失，不作为全部达标的最终材料。此前reuse_norm256和mid_base_fallback证据作为历史阶段独立保留，不混用最优数据。

任务书要求workspace相对GPU不超过1.2倍，但官方GPU CSV仅含耗时、没有GPU工作区基线。报告如实记录NPU分配与容量，明确GPU数据缺失，不能据此声称该比值已通过。
