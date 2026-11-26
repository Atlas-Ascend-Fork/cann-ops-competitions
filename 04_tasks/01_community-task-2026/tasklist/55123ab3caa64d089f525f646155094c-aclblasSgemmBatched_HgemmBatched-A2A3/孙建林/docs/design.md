# aclblasSgemmBatched / aclblasHgemmBatched（A2/A3）设计方案

日期：2026-11-25；提交者：孙建林。状态：用户已确认方案，提交社区设计评审；尚未审核或合入。

任务页面：https://www.hiascend.com/activities/task-center/details/55123ab3caa64d089f525f646155094c?menu=rankings
目录前缀取自该页面的任务ID，不宣称另有数字认领编号。

# 需求背景（required）

## 需求来源

唯一验收依据：随任务交付的 `aclblasSgemmBatched_HgemmBatched_AtlasA2_A3_task_doc.md`。通用规范与附件冲突时以此任务书为准，差异记录在过程文档，不改写原任务书。

官方代码基线：`55ef791b77f3677b45da3468b0a8b17941424f84`。现有 arch22 仅见 CgemmBatched；S/H 的公共声明已存在，保持 ABI 不变。arch35 可参考 Host 组织，但其中 tensor_api/simt_api 等不能直接假定在 CANN 9.1.0 的 A2/A3 可用。

## 背景介绍

补齐列主序 Device 指针数组批量 GEMM：`C[i]=alpha*op(A[i])*op(B[i])+beta*C[i]`。S 为 FP32；H 的矩阵和标量为 IEEE FP16。实数 OP_C 等价 OP_T。各矩阵可独立分配、带 leading-dimension padding，不能假设跨 batch 连续。

# 需求分析（required）

| 任务要求 | 实现与验证落点 |
| --- | --- |
| §2.3 公共 API | 直接实现既有两个导出符号，不新增私有平行 API |
| §2.4/2.5 校验 | Host 校验顺序及 no-op、空指针组合回归 |
| Device 指针数组 | Kernel 在 Device 取每个 batch 的真实地址，Host 不读回/扫描数组元素 |
| 正常转置与 padding | N/T/C 九组合，逻辑输出与 padding 哨兵分别验证 |
| beta=0 不读取 C | 走不加载旧 C 的分支，以 NaN C 验证 |
| alpha=0/k=0 | 不引用 A/B；仅清零或缩放 C，不能计算 0*NaN |
| §3.2 精度 | 两种 dtype 各自混合容差，非有限值单独分类 |
| §3.3 性能 | 910B3；原版 msprof task duration 正式指标、事件 avg_us 辅助指标 |
| §3.5 数量 | 1000 精度（S/H 各500），200 性能（各100），覆盖不能以固定回归替代 |
| §4/5 流程 | 独立设计 PR、个人仓自验证材料、验收后代码 PR与CI |

## 方案比较与选择

1. 推荐：共享 Host 契约 + Vector 小矩阵/退化路径 + Cube 中大矩阵路径。减少小矩阵固定开销，同时适应 4096³；需要建立清楚的路径边界。
2. 全量 Vector：容易验证 FP32 算术和尾部，但大矩阵吞吐风险高，不作为正式主路径。
3. 直接迁移 arch35：可复用调度思想，但 SDK、Tensor API 和硬件指令存在差异，禁止原样照搬。

推荐方案只按 dtype、形状、batch、转置、leading dimensions 和硬件容量选择路径；不按用例编号、seed、输入数值特征或 GPU 基线特判。

# 详细设计（required）

## Host/API

校验顺序严格为 handle → transa/transb → m/n/k → lda/ldb/ldc → batchCount → 标量与数组本体。非法枚举返回 INVALID_ENUM，其余非法结构返回 INVALID_VALUE；空 handle 返回 HANDLE_IS_NULLPTR。

- alpha/beta 指针按任务书无条件非空校验；m/n/batch 为零时无需解引用其值，不启动 Kernel，三个数组可空。
- 正常非空输出首先要求 Carray 非空。k=0 不引用 A/B；k>0 再依据 Host alpha 判断是否需要 Aarray/Barray。
- alpha=0 或 k=0：beta=1 不启动 Kernel（仍完成结构检查），beta=0 只写零，否则缩放 C。Carray 有效性的要求不因 beta=1 被绕过。
- H 标量按 half 位模式解码，不能把 aclblasHalf 当整数值转成 float。
- 元素地址偏移、矩阵跨度、tile 总数和 workspace 字节数用受检查的 64 位运算；不扫描 Device 数组元素、不新增 Host 同步。发射错误按仓内约定返回，不能假 SUCCESS。

## Kernel 与 tiling

### 共享数据流

结构参数与 Host 标量通过 tiling 按值传入，Device 保留三个指针数组。将 `(batch,tile_m,tile_n)` 展成任务号，多核做区间/循环分派；batch 小而矩阵大时增加空间 tile，batch 大而矩阵小时每核处理多个 batch。C 的逻辑区域由唯一 tile 拥有，不使用原子累加或跨核 split-K，避免额外归约与非确定性干扰。

### Vector 路径

处理 1×1×1 等小规模、多 batch 场景及清零/缩放。按列主序布局直接访问真实指针、按逻辑范围输出，使用有效长度与掩码处理尾部。小 GEMM 的切换阈值只在实机测量后确定，不能预先宣称性能达标。

### Cube 路径

优先复用仓内 arch22 已验证的 Matmul 模式，独立核实其 CANN 9.1.0 可编译性。列主序可按 `C^T=op(B)^T*op(A)^T` 交换操作数与 M/N，逐一推导 N/T/C 的物理形状与 stride，不将 padding 当有效元素。

初始评估 64×64、128×128 的 M/N tile 和 64/128 K tile，按 dtype、对齐和片上容量约束选择；这些是验证候选，不是已测参数。K 维在核内分块累加，H 使用 FP32 累加后按规定收窄。尽量融合 alpha/beta epilogue；如 CANN 9.1.0 API 无法正确支持部分尾块或缩放，使用受控 workspace 与独立后处理，并把所有 Kernel 计时纳入正式调用成本。

FP32 首先验证 float Matmul 并显式禁用 HF32，不能为性能默认转 FP16/HF32。若精度或吞吐不满足，再提交精度补偿分解的方案变更及误差测试，不悄悄放宽阈值。

### 内存与同步

L1/L0/UB 根据当前 SoC 查询并预算双缓冲、Matmul 管理开销；不复制 arch35 容量常量。尾部搬运与写回不能覆盖相邻列和 padding。workspace 只使用既有 handle 管理机制，记录实际字节数与生命周期，不能在异步任务完成前复用/释放；无 workspace 的路径不分配临时空间。

## 工程边界

候选文件为 `blas/gemm_batched/arch22/gemm_batched_host.cpp`、`gemm_batched_kernel.cpp` 及必要的共享 tiling/helper 头文件；S/H 共用模板和契约，避免两份同构实现。保持现有 cgemm_batched_* 不变，注册逻辑以最新仓库自动发现规则为准。

测试统一放在 `test/gemm_batched/sgemm_batched/arch22/` 并由该目录上级 CMake 注册。附件两个 CSV 不带 op 列，接入时保留原始附件，生成带 op=S/H 的统一用例清单，以 `(op,case_name)` 唯一标识；不能让两个算子的同名 TC_PF 互相覆盖。允许两个源 CSV作为统一 loader 输入，正式交付清单必须能逐项对应。

## 支持硬件与约束

Atlas A2/A3，CANN 9.1.0。本次按§3.1在 910B3 完成单款型自验证即可，精度与正式性能均使用现有 A2；A3 为适配目标，补充实测有资源时再开展。alpha/beta 为 Host 标量，三个数组本体及矩阵为 Device 数据。C 不重叠、数组元素有效由调用方保证；不进行 Host 元素级扫描。不承诺跨 batch 广播或任意视图布局。

# 可维可测分析

## 精度与 golden

S 主参考采用 Netlib BLAS 3.10.0 cblas_sgemm；明确它是 FP32 BLAS 实现，不能把 cblas_sgemm 描述为 float64 累加。另以显式 float64 累加后收窄为任务书允许的交叉参考，并分别记录 golden来源、版本和舍入。H 使用 float64 累加 → FP32 → FP16 round-to-nearest-even。对特殊值、上限、次正规及零分支单独固定输入，避免不同 golden 分支对 0*NaN 产生误判。

S 的 rtol=atol=2^-13，H 为2^-10；matched_ratio≥0.99。采用任务书指定的[opbase固定版本28a1ce52标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)§2.1.2：在actual和golden均为有限值的点中取最大绝对误差，以该点golden按实际输出dtype RNE收窄后的值g_low为锚点，case cap=max(S 1e-2/H 1e-1,32*ULP(g_low))；不是每个点各自cap全部通过。ULP(x)=2^(max(floor(log2(abs(x))),emin)-fractionBits)，零/次正规统一取2^(emin-fractionBits)，S fractionBits=23/emin=-126，H为10/-14。多个点并列最大误差时统一取其中最小cap，作为标准未指定tie情形的保守、与遍历顺序无关规则。

有限值误差及matched仍对原golden计算，仅ULP锚点收窄。按标准§2.3，actual为Inf而golden有限时将该golden直接RNE收窄，同号Inf通过；其他非有限值按NaN对NaN、同号Inf匹配，分类错配独立硬失败，不能被99%比例或有限max=0掩盖。无有限对时max=0、cap取固定floor，不对Inf/NaN求ULP；有限max点golden收窄溢为Inf而actual仍有限的未定义锚点拒绝，不采用无限上界。C padding逐bit判定，不能稀释matched分母。保留初版strict fixed cap作为诊断视图，正式结果明确记录mixed argmax策略、max finite abs error/cap/anchor、matched比例、逐点cap诊断与非有限错配数。CSV的MERE/MARE不替代标准。

## 性能

在同一预置 Device 缓冲上，成功 warmup5次后采样10次；正式数据来自 msprof op 的 Task Duration(us)。若一次 API 调用产生多个 Kernel，按调用序列把全部任务求和后构成该次样本，不能只取主 GEMM 或选最快任务。保留原始任务表、调用与用例映射，任何缺失、失败或非有限耗时都使该次验收失败并留痕。

同时用 aclrtEvent记录连续10次API下发均值，并输出任务要求的 `[PERF] case=... avg_us=...`。该指标含下发开销，不作为正式判据。每例正式均值≤gpu_ms*1000/0.8，逐例检查200条，不用平均倍率掩盖单例失败。

性能对比使用任务书提供的 gpu_baseline.csv，按 op/trans/m/n/k/batch/alpha/beta/LD 完整键匹配 gpu_ms，并固定文件hash。附件“非默认标量待复测”备注保留在基线来源说明中，结论限定为对所给版本基线的比较，不宣称本人采集了A100数据，不额外要求GPU硬件或增加验收门槛。社区若发布修订基线，再固定新版本统一重做对比。

## 覆盖与复现

包括九种转置、4095/4096立方和尾块、65536个小batch、padding保护、无效参数优先级、m/n/batch=0、k/alpha=0、beta=0且C为NaN、A/B为NaN的不引用分支、H极值与次正规、非默认标量。增加 batch 指针数组乱序、不同间距、独立分配与同批重复输入地址的合法场景，确认实现依据实际指针而非假设固定batch stride。固定seed、记录输入及golden哈希。缺少的 cases.json/inputs/golden/facts.json 必须补齐并通过独立复核，不能以“CSV存在”替代冻结交付包。

## 流程与阶段出口

先按社区模板形成设计并提交 cann/cann-competitions 的设计 PR；落实需求 Issue/SIG 讨论，记录评审结论。实现按测试先行推进，先Host契约，再退化/小矩阵，再Cube与性能；收到社区设计意见先更新方案。验收前必须满足设计评审及交付包标准。代码PR按任务书§5在测试通过及社区验收通知要求满足后提交至 cann/ops-blas，不能将未验证代码PR冒充交付。

个人 fork 的 task_submission 提供七项标准文件；内存按§3.4写不涉及，不能伪造采样值。设计 PR、代码 PR、验收单、最终 commit 和CI运行链接分别留档，CLA真实邮箱与贡献身份一致。源码审查只放必要文件，大日志和原始profiler档案放个人验收材料，保留固定commit/hash映射。

## 2026-11-25 数值兼容方案修订

固定37项Cube回归通过后，原S500严格fixed cap测试有2个极值失败，诊断存在FP32归约/overflow顺序差异，不能用ULP处理Inf反号。计划保持Host几何调度与pack/cube/epilogue三Kernel：S的pack访问原逻辑A/B时另生成每物理列maxAbs统计，beta非零时检查Cold逻辑区，padding统计为0、非有限或统计和溢出以Inf sentinel强制兼容；epilogue每worker每batch按≤1024个float分块读取并缓存统计，风险输出从packed原输入按锁定Netlib序列重算，普通路径保留Cube。Vector fallback也覆盖同一风险语义，beta零不读Cold、退化不读A/B、padding不变。统计空间纳入workspace与chunk预算，所有Kernel成本纳入正式task duration；不按测试seed/case/16立方几何回避测试。

Host只依据K及Host scalars计算保守范围：input limit=sqrt(FLT_MAX/(8*K*max(1,abs(alpha))))，Cold limit=FLT_MAX/(8*max(1,abs(beta)))；非有限scalars强制兼容。此界只约束overflow风险，不能当作任意强消去输入满足ULP的数学证明。Device检测器、UB/事件及padding已在有界实机回归验证；普通输入性能仍需独立采样。

已完成的CPU证据：独立Netlib3.10构建中190,512次调用/1,381,212输出，覆盖九转置、4种形状及LD padding、三batch、14×14标量与9类数据，AlwaysFma+BetaFma候选全0 mismatch（NaN按类、其余bits）。N-A先beta*C，再每K FMA(A,FP32_RNE(alpha*B),acc)，不跳过B零；T/C-A先逐K FMA dot，beta非零再FMA(beta,Cold,FP32_RNE(alpha*dot))。该结论限定锁定库构建与诊断网格，不泛称所有发行Netlib逐位相同。整数FMA/Multiply各百万随机位及边界对照独立std::fma/volatile乘法通过。数值兼容Device已有有界实测，后续执行优化和完整精度/性能验证仍在进行。


### 输出相关风险评分原型与验证边界

仅overflow界不能捕获有限消去；旧Cube对2^50,1,-2^50的18次九转置/标量组合实际误差超过固定cap。进一步使用预先固定c=1的经验评分 E=2^-24*(K*max(1,abs(alpha))*maxAbsA*maxAbsB + max(1,abs(beta))*maxAbsCold)，beta=0忽略Cold。对alpha/beta组合后的输出y，以 E>atol+rtol*max(0,abs(y)-E) 筛选兼容复算；非有限与原overflow条件仍强制。Device按显式FP32 RNE计算评分，CPU double候选仅作诊断。该评分不是严格误差上界，不承诺任意FP32输入的全域证明，也不按失败case调整系数。

实际CANN9.1/910B3原型已通过46项固定/新增回归，无跳过，包含长K平台变种、有限/非有限消去、真实NaN首中末统计、多worker、chunk复用与Vector后备；原两条S极值CSV及两条H代表CSV全部通过。16条受控PF完成精度与5warmup/10sample事件辅助计时，但未完成200条msprof正式验收。global batch max有误触发和软件复算成本，当前只属正确性原型，先测统计与小规模耗时，再优化并运行完整1000精度和200正式性能；未将本里程碑写成合入条件已满足。

### 兼容执行路径的有界优化检查点

仅改变输入搬运和执行成本，保持上述评分c=1、ReferenceElement逐K数学顺序、原golden与判据。CANN9.1/910B3的局部contract(on)标量表达式在独立1,153,088个边界/随机输入上与std::fma 0差异，Multiply符号零上下文与独立volatile乘法0差异；这不是全输入域证明。未采用链接失败的builtin或存在差异的Vector FMA候选。Host继续整数参考。

S Cube兼容getter增加固定256逻辑元素的DMA缓存，A/B各8KiB UB，每输出重新构造身份；连续单块与跨步4byte+padding7布局由独立90组合/20,961原始字及输出尾哨兵逐bit验证。超uint32字节stride保留GM fallback，trueK尾部不越界，Cold beta0不读，Vector/H路径不改。

实际SDK49项固定/数值测试与4条代表CSV全部RUN/COMPLETED、0fail/0skip，独立规格/质量复核均通过。部分低score特殊值测试会跳过repair，getter原始位保持引用独立DMA探针，不能夸大覆盖。S_PF007辅助event由13,319.2207us经原生FMA降至8,315.8457us，再由DMA降至3,172.76001us；仍未达标。三kernel诊断各含16任务（含精度/预热），只定位瓶颈，不作200项正式结论；完整1000精度、冻结包运行桥接及200项正式采样均未完成。
