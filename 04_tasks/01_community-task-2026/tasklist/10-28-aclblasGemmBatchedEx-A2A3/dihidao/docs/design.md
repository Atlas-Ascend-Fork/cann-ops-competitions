# aclblasGemmBatchedEx（arch22）R50设计文档

修订：2026-10-30，R50实测版本。依据原任务书与官方设计模板编制。只描述本轮冻结代码及证据，不改写旧版本失败结论。

# 需求背景（required）

## 需求来源

原附件 `ce754b8845084306b84305ac68a84a5b (1).zip`，SHA256 `a46f326502f9b8a8abc39865e20331117b8a82893ffbb1b219df4b6e63f4c38b`。其不带“(1)”副本字节完全一致。任务书正文SHA256为 `ca854ab126f9e3524c521ebf393db4d7dbed63114da0d982cf7cbc6410449266`。
官方设计模板：<https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>。

## 背景介绍

在ops-blas的Atlas A2/A3 arch22目录补齐float32批量矩阵乘扩展接口。复用公共声明、handle及绑定stream，保持950PR等产品线共用接口。实现采用Ascend C kernel直调，未改全局OPP/SDK或原CMake。

# 需求分析（required）

## 需求描述

对每个batch计算 `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`。矩阵列主序；每batch独立Device指针数组，batch间共享m/n/k、标量和leading dimension。N为不转置，T为转置，实数C与T等价。

## 需求拆解

| 需求 | 实现范围 |
|---|---|
| 数据类型 | A/B/C为ACL_FLOAT，computeType=ACLBLAS_COMPUTE_32F，algo=ACLBLAS_GEMM_DEFAULT；未宣称FP16/BF16/FP8/INT8正向支持 |
| 尺寸及存储 | m/n/k/batchCount非负；满足lda/ldb/ldc约束；支持leading dimension padding，不支持超出该语义的任意stride/broadcast |
| 边界 | 参数先校验；m/n/batch为0合法no-op；k或alpha为0仅缩放C；alpha/beta空及正batch空数组返回错误 |
| 标量/stream | Host或Device标量；kernel使用handle绑定stream；Device标量读取可能同步，结果读回前需同步stream |
| 精度 | 独立Netlib BLAS3.10逐batch全矩阵；遵守任务书FLOAT32混合容差、非有限值及EXACT规则 |
| 性能 | 当前按用户确认及正文§3.3的0.8×，预热5次后连续10次算术均值；另披露60样本结果与原文冲突 |

# 详细设计（required）

## 算子分析

`op(A)`形状m×k、`op(B)`形状k×n、C形状m×n。非转置A要求lda≥max(1,m)，转置A要求lda≥max(1,k)；非转置B要求ldb≥max(1,k)，转置B要求ldb≥max(1,n)；ldc≥max(1,m)。支持C原地覆写；beta=0时不依赖C旧值。

## 算子实现

### Host侧设计

1. 按handle、转置枚举、维度、leading dimension、空指针、dtype/compute/algo的顺序校验，再处理no-op。
2. 读取alpha/beta。`(k=0或alpha=0)且beta=1`直接返回；其它缩放由Vector完成，不引用A/B数据。
3. 按公开shape/标量参数选择tiny、Vector、Cube、split-K及混合policy路径。最多20 AIC、40 AIV；实际核数取平台接口值并限幅。
4. 对需要的policy、split临时区申请或使用handle workspace，校验整数溢出、容量与对齐。未修改调用方矩阵布局。

### TilingData

结构`GemmBatchedEx22Tiling`为96字节。m/n/k、lda/ldb/ldc、转置、outputLd、batch范围、alpha/beta描述计算；splitK/parts/firstTask/taskCount描述分K任务；policy/policyCores描述统计区；taskRotation复用末尾原padding。源码对size及72/80/88/92字节关键偏移有static_assert。

### Kernel侧设计与主要路径

| 路径 | 触发/实现 |
|---|---|
| tiny8/tiny16 | alpha=1、beta=0、NN及8³/16³公开shape条件，使用独立静态入口；保留原统计、安全谓词、树归约及不安全输入的NPU参考顺序回退 |
| Vector | 缩放、非单位alpha/非零beta、小工作量；按列主序leading dimension访存，处理转置、padding与尾部 |
| 通用Cube | 静态Matmul配置按shape选择宏块；FP32输入输出、HF32关闭；以完整输出tile分配给AIC |
| 长K | K≥2048使用splitK=512、parts=ceil(K/512)；临时区受workspace容量约束，Vector完成合并/必要处理 |
| policy与Cube重叠 | 收集输入sum/abs-sum与非有限状态，同时执行矩阵路径；参考覆盖必须等待Cube写出完成 |
| 固定NT Norm | Host `UseFixedNT5`选择的窄N、中等K、batch1场景；Norm宏块128×64、basicK64；PF1091为此路径 |
| 短K尾部调度 | 根据公开shape和任务负载选择保留的宏块及必要旋转，保持真实leading dimensions和非对齐尾块范围 |

内部Cube输入视图通过B/A交换及转置模板适配外部列主序；对外仍为原C矩阵。路径选择不能被理解为改变算子数学定义。

### 数值策略和同步

policy计算两输入的`abs(sum)/sum(abs)`，出现非有限统计量或比值乘积大于2^-12时选择NPU上的参考顺序路径。该判据是实现中的路径选择策略，精度结论来自独立Golden的实际覆盖，不能仅由判据推导任意输入的误差界。NPU回退不是CPU Golden，也不以旧输出作为参考。

R50相对R49仅把Norm的`enUnitFlag`恢复true，保持basicK64、宏块和任务分配。默认同步`IterateAll`、`mm.End`及原flags8/9/10完成握手保留；所有相关Cube写出完成后才能执行参考覆盖。真实SDK UnitFlag CO1路径仍有最终FIX_M等待，不把API/队列数量当作净省时。未证明最后一项改动的独立配对贡献。

### 缓冲区与workspace

NT配置双L0A为65536字节、双L0B为32768字节；L1/L0C和实际baseK、双缓冲边界由当前SDK编译断言检查。tiny缩小静态UB配置，其余分支保留各自配置。GM policy预算为`batchCount*(64+policyCores*32)`字节，额外最多63字节对齐；split单任务临时区为`parts*128*128*sizeof(float)`，按实际容量分批提交。测试提供的32MiB workspace是测试夹具容量，不能冒充算子峰值占用。

## 支持硬件与约束

目标为Atlas A2/A3系列、arch22。本轮实机为Ascend910B3/CANN9.1.0，20 AIC/40 AIV；A3未另做实机验证。任务书§3.1允许自验证覆盖一种款型。不宣称本轮验证了950PR或未支持的混合精度类型。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 本轮证据与结果 |
|---|---|
| 原附件用例 | 原1200条的ID、shape、转置、leading dims、dtype、状态、标量和seed映射核对通过，包含200性能用例 |
| 精度 | 2393原身份主CSV+144固定标量+72新随机标量=2609条全部通过；旧删除的72条精确seed身份未冒充复原 |
| 输入分布 | Gaussian扩展保留；新增72标量按用例计均匀36、正态36，含9转置组合及Host/Device标量，参数与seed冻结 |
| 计时输出 | 200×85=17000引用全部通过独立Golden；同输入85次输出200/200逐位一致 |
| 稳定性/双流 | PF1008/PF1009/PF1091及各自Gaussian的246次调用通过；不宣称已证明物理时间重叠 |
| 0.8×性能 | 10样本主窗口200/200；同批原始计时中26–85次的60样本补充窗口也200/200，无新增计时 |
| 1.0×原文 | 主窗口189/200，不宣称达到全部用例1.0×；未过清单见测试报告 |
| 内存 | §3.4不涉及与§4 workspace≤GPU1.2×冲突；缺GPU数据，按用户要求暂不处理并保留未验收 |

## 测试范围与证据

完整逐例参数、精度匹配率/max abs和原始日志见官方XLSX与对应日志；负向/no-op仅检查预期状态和guards。FLOAT32使用atol=rtol=2^-13，有限值matched_ratio≥0.99，硬界max(0.01,32ULP)。同NaN、同号Inf、NaN与Inf按任务书等效；有限与非有限不等效。alpha=0缩放遵守EXACT。所有200性能结束并过门后才执行数值普查，运行时/guards故障即时停止，数值首败保留。

# 兼容性分析

公共接口声明与原CMake保持；仅增加/修改本任务arch22文件及对应测试。继续使用handle/stream/workspace契约。未改全局SDK/OPP；未以CPU计算替代NPU算子。设计与实现分别提交社区任务仓和ops-blas进行评审；本地自验证通过不代表官方验收、CI通过或合入。开发者邀请及正式验收单独处理。

## 版本绑定

- 源码manifest：`4319aac746179fee1b6af06f59a40055f15bd185a1f61509974e46866999807b`。
- 测试binary：`fb1c095f732d4f97ee41811aad013f7c30f426ee233cfcd4a9ca199556d2f583`。
- operator library：`a17b0425daaf7373883de8af351927081497badb7f43f3901ee2201d10e297a0`。
- Netlib：`97ca2ad4238ef999db726302d19be95ab95cfffcb0b3a4f0d04c47d25f518d19`。

对应实现位于ops-blas的`blas/gemm_batched_ex/arch22/`，测试位于`test/gemm_batched_ex/arch22/`。源码manifest含5项实测文件，发布仅补文档和独立CSV，没有改变这些文件身份。
