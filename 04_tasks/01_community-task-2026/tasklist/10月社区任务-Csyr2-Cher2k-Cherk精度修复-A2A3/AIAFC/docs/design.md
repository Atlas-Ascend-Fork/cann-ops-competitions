# Csyr2 / Cher2k / Cherk 精度修复（A2/A3）设计文档

任务：fc6145cabc15459ab08b3996cad3b8f9，贡献者AIAFC。本文按CANN社区任务设计模板说明修复方案。上游基线为ops-blas `05b96ea22cea926ecfd43fc7ac691a795522bd61`。任务书ZIP SHA256：`caa0ee0cb7c81239e4b1c9c5875d08c2ad7b3c78e21e546d8242c2d7625cc57d`。

## 需求背景（required）

### 需求来源

- [本题任务页面](https://www.hiascend.com/activities/task-center/details/fc6145cabc15459ab08b3996cad3b8f9)及其2026年10月任务包。
- [官方讨论](https://gitcode.com/cann/ops-blas/discussions/66)。
- [设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)和[设计提交说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。

### 背景介绍

本题在已合入的三个BLAS接口上修复精度与写回语义，保持调用方式和arch22支持范围。

### 已有实现的问题

原实现部分路径向未选中三角写入零增量，或合并旧矩阵后整块写回。原子加零会改变负零、非规范数或NaN payload；即使合并后值相同，整块写回也不符合任务要求的未操作区零写。另有小矩阵长归约的消减误差，以及极端有限输入在中间乘法中溢出的问题。

修复仅作用于既有arch22实现，保留公开API、列主序、负步长、stream及标量指针模式。Csyr2在`blas/syr2/arch22`，Cher2k在当前上游的`blas/her2k/arch22`，Cherk在`blas/herk/arch22`。不更改arch35实现，不增加平行API。

## 需求分析（required）

1. 保持原有算子公式、公开签名、Host/Device参数归属、列主序、quick return及异步stream语义。
2. 未操作三角与leading dimension padding零写且逐bit不变，覆盖负零、NaN payload、Inf和非规范数。
3. 操作区实部/虚部满足任务书FLOAT32混合容差，Hermitian对角虚部在非no-op计算后为零。
4. A2/A3均完成全部功能、精度、bit验证；按原性能阈值逐项报告，A3运行并记录。任务书指定910B3，本次按用户决定使用910B2推进并披露型号差异；三个算子作为同一任务交付。

## 详细设计（required）

### 算子分析

采用`aclblasComplex`（实部/虚部各FLOAT32），不扩展dtype或广播能力。

| 算子 | 数学公式 | 标量归属与形状 |
| --- | --- | --- |
| Csyr2 | `A := alpha*x*y^T + alpha*y*x^T + A`，普通转置，不共轭 | 复数alpha在Host；x/y在Device，逻辑长度n，incx/incy非零且支持负值；A为列主序lda×n |
| Cher2k N | `C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C` | 复数alpha/实数beta在Device；A/B为n×k，C物理缓冲ldc×n |
| Cher2k C | `C := alpha*A^H*B + conj(alpha)*B^H*A + beta*C` | A/B为k×n，其他同上 |
| Cherk N/C | `C := alpha*A*A^H + beta*C`或`alpha*A^H*A + beta*C` | 实数alpha/beta在Device；A分别为n×k或k×n，C物理缓冲ldc×n |

n/k均为非负int；Csyr2的lda及Hermitian输出ldc≥max(1,n)，Hermitian输入的lda/ldb在N时≥max(1,n)，C时≥max(1,k)。uplo仅UPPER/LOWER；Hermitian trans仅N/C，T返回INVALID_VALUE，非法枚举返回INVALID_ENUM。具体参数顺序复用[公开头文件](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h)，不修改签名或新增API。

Csyr2在n=0或alpha=(0,0)时不改矩阵；其对角虚部按普通对称公式计算，不清零。Cher2k/Cherk在n=0或(alpha=0或k=0)且beta=1时不执行计算、不修改原缓冲；其他合法计算的对角虚部为0，beta=0时不依赖旧C数值。调用方在读回前同步stream，生产Host入口不新增同步。

### Host、分核与tiling设计

继续使用现有Ascend C kernel直调和handle绑定stream。Host只缓存SoC名称与完整参数对应的tiling数值，命中时避免重复查询同一SoC的平台属性；不保留Device分配、context或stream资源。沿用既有核间分工与workspace生命周期，不新增外部tiling key。Csyr2对满足单位正步长、alpha=(1,0)、完整窗口和容量约束的LOWER输入，在n≥64时复用已有Direct路径；上界和其余约束保持原值，避免中小矩阵额外分块暂存。Cher2k小n且长k路径按归约长度启用补偿累加，并允许合法LOWER及非紧凑ld进入对齐列输出路径。片段边界由实际uplo、列号、行起点和尾长计算，不依赖用例ID或固定验收尺寸。

保留现有UB和核间所有权；新增临时量只在本次kernel作用域。Csyr2复用已有tiling UB，Hermitian写回复用当前输出UB；扩展范围重算使用局部标量，不缓存输入结果。

### Kernel：选中三角写回

Csyr2按每列合法三角范围裁剪DMA/原子写入。LOWER首部在UB非对齐时，使用4个独立32字节头部暂存槽，另32字节存Gather偏移；每头以V→MTE3保证Gather完成，仅在槽回卷复用前等待MTE3→V。LOWER暂存合计160字节，比原布局增加96字节，已计入Host/Device的generic、window及packed容量计算；使用头槽的kernel入口结束前排空DMA。连续矩形完全属于操作区时保留批量DMA。原始x/y已完整保存在交织UB的路径复用该缓存判断零列，使用位模式区分±0与非规范数；其他路径保持原始GM读取。某列x和y同时为复数零时遵循Netlib ssyr2分支，不计算或写回该列，避免零乘非有限输入造成错误。

Cher2k/Cherk的部分输出块按列提交选中片段。非对齐UB首部按位搬到对齐位置，再提交剩余连续片段；不把未操作区旧值合并回GM。输出区每个有效元素仍由原有分工拥有，padding不参与写回。

### Kernel：数值路径与性能优化

Cher2k小矩阵在长k归约时使用补偿累加；该路径支持LOWER和合法非紧凑leading dimension。有限极端输入造成非有限输出时，从已驻留的planar输入按原公式重新计算。

Cherk generic Cube路径对输出块做非有限探测，仅对选中三角中的非有限结果进行重算。有限输入重算使用两项float有效数加显式二进制指数，整数分解保留两个24位float乘积的有效位，避免中间范围溢出。alpha、beta及旧C在该扩展范围内合并后收窄。原输入含Inf/NaN时保留Netlib复数运算顺序；Hermitian对角虚部按规则置零。重算全部在Device执行，不使用CPU替代DUT，不增加持久Device缓存。

扩展范围算术共用`blas/common/helper/complex_extended_sum.h`。真实AI Core不支持本题所需的原生double算术；编译探针失败日志已保留。主机测试以独立长双精度公式校验累加、溢出和消减，以`scalbn`校验按位缩放；这不替代NPU结果。

## 可维可测分析

### 精度标准与验证链

Cher2k/Cherk参考固定Netlib3.10.0的zher2k/zherk，输入由FP32精确扩展，参考输出保留FP64以符合所引用标准的高精度单标杆。Csyr2采用独立索引和Netlib ssyr2分支语义的复数参考。独立长双精度公式核对参考链，不调用DUT作为golden。

每个SUCCESS用例在调用前从Device复制完整ld*n缓冲，在调用及同步后再次复制。未操作区和padding逐bit比较，quick return比较整个缓冲；哨兵覆盖随机有限值、正负零、不同NaN payload、Inf和非规范数。操作区实部/虚部分别使用FLOAT32混合容差、99%比例及最大误差硬上限，对角虚部单独检查。

原CSV全部保留，每平台3000个原功能/精度用例、70个新增bit用例及600个性能用例。正文9项性能门槛按原表，扩展项保留官方CSV基线；warmup后有效采样至少51次，记录调用计时范围。任务书指定的A2性能设备为910B3，本次实测为910B2；按用户决定继续推进并在报告注明差异。A3独立构建并覆盖要求。最终候选覆盖不能跨版本拼接。

### 性能标准

有效采样至少51次。完整调用批次的wall-clock均值包括API下发和最终stream完成，不包括分配、H2D/D2H、golden与校验；另记录逐次同步和Device event区间，不能用GTest整例毫秒数代替算子耗时。正式结果绑定实测设备、CANN、驱动、Netlib、源码与产物。

| 算子 | 正文3项（按原表顺序，μs） | 指定性能设备 |
| --- | --- | --- |
| Csyr2 | 7.54 / 9.72 / 19.05 | Atlas 800I A2（910B3） |
| Cher2k | 1458.91 / 5240.68 / 1081.90 | 同上 |
| Cherk | 738.09 / 2616.28 / 485.94 | 同上 |

其余性能项采用原包gpu_baseline.csv及0.8倍率规则，正文典型值不再除倍率。目标是零写回修复后仍满足原表，先优化非对齐头部和已完整选中的批量DMA；数值重算仅针对异常结果，不改变普通有限输出的主体计算路径。

### 支持硬件、兼容性与可维护性

支持Atlas A2/A3，固定CANN9.1.0与Netlib3.10.0；本次A2为910B2，任务书指定的910B3性能尚未实测；按用户决定不因此停止开发与验收准备。A3按实机SoC独立构建arch22。只改现有arch22调用路径；公共扩展算术辅助头仅被本题arch22包含，arch35无调用变化。保留源码版权头、原错误码及资源释放路径。任务书§3.4不设额外内存测试门槛，但输出边界、同步及无持久缓存需随实现核验。

### 当前开发验证与待评审事项

开发期完整非性能验证已覆盖910B2和Ascend910_9382：Cher2k/Cherk各1024项通过，Csyr2唯一失败为下述返回码冲突。R12在A3及B2分别完成Csyr2全部1224项，均1223通过、唯一TC_ED_151；200性能项精度/bit全部通过。A3的R12 LOWER n1024完整调用从R11的11.4784降至8.6051微秒。R13将既有Direct LOWER扩展至中小规模，已完成A3构建、83项smoke（含63/64边界及84次混合零列/步长/alpha重复调用）和19项代表用例精度/bit；受影响中小LOWER代表项完整调用进一步改善约49%–57%。R13双平台全量仍待补齐。自测报告保留原始失败和最终候选结果，不跨候选拼接PASS；设计文档不作为已完成验收的声明。

Csyr2 TC_ED_151的CSV要求INVALID_VALUE，而正文要求INVALID_ENUM；上游实际返回后者。保留失败并等待该题官方裁定，不改CSV制造通过。请评审裁剪写回、数值重算及此错误码差异。后续完成同一最终候选的双平台覆盖、性能与设计评审合入，并按活动要求准备验收材料；B2实测型号与任务书B3差异随报告说明。
