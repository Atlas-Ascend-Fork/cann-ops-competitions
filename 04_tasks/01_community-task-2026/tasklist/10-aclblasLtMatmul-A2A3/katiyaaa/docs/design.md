# 需求背景（required）

## 需求来源

本设计对应 10 月社区任务“aclblasLtMatmul 算子开发（A2A3）”，提交团队为 katiyaaa。依据任务附件 `aclblasLtMatmul_A2_task_doc.md` 及 `test_cases/` 中的 README、生成器、精度/性能驱动和两份 CSV，补齐 Atlas A2/A3 上 FLOAT32 的 BLASLt 矩阵乘能力。

本文是待评审的实现与验证方案，不代表已完成开发、真机测试或性能达标。设计采用官方模板的需求背景、需求分析、详细设计、可维可测分析四部分；不复用其他算子的接口或验收结论。

源码调研基线为 ops-blas `05b96ea22cea926ecfd43fc7ac691a795522bd61`；竞赛仓模板基线为 `e1ab79ca0d495afce93dfb1e85f95f546b74d2cf`。未发现本任务已分配的 tasklist 数字序号，沿用已有 `10-算子名-A2A3/团队/docs/design.md` 形式，不自造任务序号。

## 背景介绍

### 功能与价值

实现 `D = alpha * op(A) * op(B) + beta * C`。沿用已有 16 参数 `aclblasLtMatmul` 公共接口、句柄、描述符、算法对象、workspace 和显式 stream，支持 N/T/C 转置组合、行主序 padding、C/D 原地与异地、零维和标量退化路径。实数 C 转置等价 T。该算子不产生随机数；随机分布仅属于测试数据生成方式。

### 现有实现分析

最新源码的公共声明位于 `include/cann_ops_blasLt.h`，公共入口为 `blasLt/api/aclblaslt_matmul.cpp`，经 `BuildMatmulProblem` 调用 `blasLt/matmul/engine/matmul_engine.cpp`。当前 FP32 实现位于 `blasLt/matmul/fp32/arch35/`，测试位于 `test/blasLtMatmul/arch35/`，A2/A3 对应 arch22，不能直接编译 arch35 内核作为本任务实现。

任务书提及的 `blas/lt_matmul/arch22/` 在该基线不存在。当前仓库通过 `blasLt/CMakeLists.txt` 自动收集源文件、按架构筛选，并由根构建文件区分 Host CXX 与 Device ASC。测试目标为 `blasLtMatmul_test`，链接已有 Lt 库。设计优先延续当前调用链和目录，目录差异提请维护者评审，不在旧路径重复定义公共符号。

现状还包括：公共校验拒绝 OP_C；无条件要求 A/B 非空；部分校验顺序、scaleType 错误码与本任务不一致；现有测试 golden 使用 double 累加循环，不能代替任务要求的 Netlib 3.10.0 `cblas_sgemm`。这些均作为 arch22 接入和回归的检查项，不能只补 Kernel 而绕开公共入口。

# 需求分析（required）

## 需求描述

| 项目 | 本任务目标 |
| --- | --- |
| 硬件/软件 | Atlas A2/A3，CANN 9.1.0；正式性能在 910B3 实测 |
| 类型 | A/B/C/D 与 alpha/beta 均为 FLOAT32；computeType 为 ACLBLAS_COMPUTE_32F，scaleType 为 ACL_FLOAT |
| 布局 | 四个 layout 为 ORDER_ROW，batchCount=1，stridedBatchOffset=0；允许独立 ld 与 padding |
| 转置 | transA/transB 各为 N/T/C，实数 C 与 T 等价 |
| 输出 | 独立 D；C==D 时布局一致，C!=D 时 shape/dtype/order 一致，允许 ldc 与 ldd 不同 |
| 范围 | m/n/k 非负、合法描述符可表达的尺寸；进行容量和索引溢出检查，不以用例最大 4096 作为功能上限 |
| 主路径 | EPILOGUE_DEFAULT；不扩展 Bias/ReLU/GELU、混合精度、多 batch、广播或任意 stride |
| 异步 | 所有设备任务提交到显式 stream，支持默认 stream；接口不为计时而强制同步 |

## 材料差异及处理原则

以下问题在设计评审时显式确认。原始任务材料、CSV、golden 要求和阈值不修改；未确认项不能在验收报告中写成已满足。

| 差异 | 证据与拟采用口径 |
| --- | --- |
| 用例数量 | 任务书部分段落写 200 条；实读 CSV 为 1200 条：1000 精度、200 性能。全部执行，不缩减精度覆盖；性能 case 也校验结果 |
| 性能倍率 | §3.3 和脚本为 GPU/NPU≥0.8；§7 为≥1.0。并列报告两档，开发目标取较严的逐例≥1.0，正式门槛请任务方确认 |
| 采样次数 | §3.3 为 10 次，§4/§7 为>10，测试 README 为>50。计划 warmup 5 次、有效采样 60 次，并保存前 10 次与完整 60 次均值；不挑选较快样本 |
| GPU 身份 | README 前部称 A100，性能章节称 P102-100，CSV 无型号列。保留 200 条原始数值，注明基线硬件待确认，不将其自行标为 A100实测 |
| 内存预算 | §3.4 为输入输出总量的1.2倍，§4 又要求相对GPU workspace≤1.2倍；CSV无GPU workspace。分别记录，后者需任务方补充，不能用输入大小替代GPU workspace |
| Golden | 概述允许 Netlib/OpenBLAS，验收章节明确 Netlib BLAS 3.10.0。正式验收锁定 Netlib，OpenBLAS仅可诊断；仓内double循环也不能作为正式替代 |
| C为空 | 任务书一处“不作判定”，附件 `null_C_beta_nonzero` 要求 INVALID_VALUE。方案在读取C前拒绝 beta!=0 的空C，beta=0允许空C，按附件覆盖并请确认 |
| 指针模式 | 参数表涉及Host/Device，自验覆盖HOST。本次arch22先支持HOST；DEVICE不在已承诺范围，明确返回INVALID_VALUE，提请评审确认是否要求扩展；不影响arch35已有DEVICE支持 |
| 测试/实现目录 | 任务书旧路径与当前仓结构不同。拟采用当前 `blasLt/matmul/fp32/arch22/`、`test/blasLtMatmul/arch22/`，评审确认后实施 |
| 自验脚本 | 性能脚本解析GTest整例毫秒而非kernel时长；精度脚本未严格校验子进程退出码/完整数量。保留原脚本，同时增加独立审计和msprof采集，禁止把脚本退出0当作完整通过 |
| 基线重复key | 200条性能与200条基线均仅198个唯一五元key。按文件顺序、case编号、参数及重复出现次数一一匹配，不用字典覆盖重复项 |

任务 README 引用的 `FORMAT_BASELINE.md` 未随本目录提供；本文按实际收到材料制定方案，不推断该文件的额外规则。生成器会重写 GPU 基线并清空测量列，因此只在临时目录核对生成结果，禁止覆盖已有测量基线。

## 需求拆解

1. 梳理公共 API、描述符、heuristic、engine、arch22 launcher、内核和测试的完整调用链。
2. 实现正确性优先的 FLOAT32 主路径及 zero/copy/scale 分支，补充错误码与边界测试。
3. 结合 Cube/AIV、分块和数据搬运优化，逐条对照全量 Netlib golden 与性能基线。
4. 保证 heuristic 报告容量、默认算法、显式算法和实际分配一致，不在内部偷偷申请未计入的 GM workspace。
5. 完成 A2/A3 编译、910B3 真机自验、arch35 兼容性回归及可复现交付；没有 A3 设备时明确只证明编译兼容。

# 详细设计（required）

## 算子分析

### 数学公式与地址映射

对 `0<=i<m, 0<=j<n`：

```text
P[i,j] = sum(p=0..k-1, op(A)[i,p] * op(B)[p,j])
D[i,j] = alpha * P[i,j] + beta * C[i,j]
A_N(i,p) = A[i*lda+p]; A_T/C(i,p) = A[p*lda+i]
B_N(p,j) = B[p*ldb+j]; B_T/C(p,j) = B[j*ldb+p]
C(i,j) = C[i*ldc+j]; D(i,j) = D[i*ldd+j]
```

ld=0 归一为对应物理矩阵的 `max(1,cols)`，不产生广播。A物理shape为(m,k)或(k,m)，B为(k,n)或(n,k)，C/D为(m,n)。所有地址乘加、对齐和字节数先在检查过的64位范围计算，再按设备接口限制安全窄化。只写逻辑输出元素，保持padding与哨兵区域不变。

### 接口

直接包含已有头文件，不新增同名声明或产品专属平行API：

```cpp
aclblasStatus_t aclblasLtMatmul(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc,
    const void* alpha, const void* A, aclblasLtMatrixLayout_t Adesc,
    const void* B, aclblasLtMatrixLayout_t Bdesc, const void* beta,
    const void* C, aclblasLtMatrixLayout_t Cdesc, void* D,
    aclblasLtMatrixLayout_t Ddesc, const aclblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, aclrtStream stream);
```

句柄与描述符由现有创建/设置/销毁接口管理。alpha/beta在HOST模式按float读取；禁止把DEVICE指针当Host地址解引用。Cdesc即使C无需读取也不可省略；C==D传入相同布局的描述符，不要求两个句柄对象地址必须相等。输出与A/B重叠及C/D部分重叠不属于任务支持范围。

## 算子实现

### Host侧设计

#### 校验顺序与状态码

arch22分支保留单一公共入口；平台分派在解引用标量和执行当前arch35专用校验之前完成，避免本任务OP_C被共用旧逻辑提前拒绝。尽量复用无架构差异的解析工具；有语义差异的校验限定在本任务分支，回归arch35，不全局放开复数共轭能力。

按任务§2.5依次处理：

1. handle为空返回NOT_INITIALIZED。
2. computeDesc/Adesc/Bdesc/Cdesc/Ddesc任一为空返回INVALID_VALUE。
3. alpha或beta为空返回INVALID_VALUE；不受支持的pointerMode返回INVALID_VALUE。
4. 从D布局获得m/n；m=0或n=0直接SUCCESS，不访问矩阵、workspace或启动内核，后续布局错误不覆盖该no-op优先级。
5. 非空路径检查trans枚举、compute/scale/dtype、shape、order、batch、ld和溢出。非法trans返回INVALID_ENUM；非ROW及不支持的数据类型/计算类型返回NOT_SUPPORTED；非法shape/ld/batch/stride返回INVALID_VALUE。负维度在描述符构造或解码层拒绝，测试需区分设置失败与算子返回，不能跳过该case。
6. 检查D；只有k>0且alpha!=0才检查/读取A、B；只有beta!=0才检查/读取C。所需指针为空返回INVALID_VALUE。
7. 检查算法支持性、workspace非空时16B对齐、实际需求与传入容量；所有检查在首次下发前完成。显式algo容量不足返回INVALID_VALUE；未知/不支持配置拒绝，不悄悄忽略。
8. 调用arch22 launcher；运行时下发错误映射EXECUTION_FAILED。异步执行错误由测试在同步指定stream时另行检查，不能把“下发成功”当作执行完成。

#### 退化路径

| 条件 | 行为与禁止访问 |
| --- | --- |
| m=0或n=0 | no-op，矩阵指针可空；必要描述符和标量指针仍先校验 |
| k=0或alpha=0，beta=0 | 输出逻辑区域写正零，不读取A/B/C；不能计算0*NaN |
| k=0或alpha=0，beta=1，C==D | 不改动数据，不读取A/B，不下发无用计算 |
| k=0或alpha=0，beta=1，C!=D | 按ld复制C到D；不能因“C不变”错误地跳过D写入 |
| k=0或alpha=0，其余beta | AIV逐元素一次FP32乘法写D；不读取A/B |
| 正常乘积，beta=0 | 不读取C，包括C=nullptr或C含NaN |

alpha=0缩放采用EXACT口径，复制保持位模式；测试包含正负零、Inf/NaN、subnormal与padding。k=0不执行alpha乘零，避免alpha=Inf引入非预期NaN。

#### 算法选择与workspace

算法策略按m/n/k、转置、ld、数据类型、可用workspace和平台容量分派，不读取case名称或硬编码官方用例编号。

- 无workspace通用基线：AIV按输出tile遍历K，FP32累加并执行epilogue，可处理默认algo=nullptr、padding、转置、原地及小矩阵；本路径首先保证完整语义，不预设其性能已达标。
- Cube主路径：复用仓内arch22 GEMM的Ascend C Matmul配置经验，建立Lt的行主序接入；不能直接套用传统SGEMM列主序接口或其内部申请空间的流程。按A/B物理stride取块；必要时打包到调用方workspace，输出product后执行epilogue。
- heuristic只返回arch22真实实现的策略，编码tile、精度策略、workspace需求及版本；限制候选数量和内部资源。algo=nullptr在传入预算内选择可执行算法，无可用工作空间时使用无workspace路径。
- 首版不采用全量split-K临时数组；输出tile由唯一核负责。后续若引入split-K，必须显式报告部分和空间及归约kernel时长，重新跑完整精度/性能。

设按16B对齐后的临时需求为 `W=align(WA)+align(WB)+align(WP)+align(Wscratch)`。A/B只在需打包时分配，WP只在需要临时product时分配；原地C在epilogue读入前不能被product覆盖。heuristic返回的容量与launcher实际需求使用同一计算函数，并校验与`algo->max_workspace_bytes`一致。不因默认算法或原地模式暗中分配GM。

当打包空间超过预算时缩减tile或使用直接读取路径；C==D且beta!=0时保证先保存当前tile的C再写D，或使用WP隔离。workspace不足不得在已经写坏D之后报错。报告同时记录调用方分配容量、算法实际峰值和设备系统临时空间；`workspaceSizeInBytes=1`官方负向须结合真实heuristic需求验证，不能为迎合用例虚报内存。

#### 分核与分块

输出分为 `ceil(m/BM)*ceil(n/BN)` 个tile，按平台Cube/AIV数限制block数，采用轮转分配覆盖任意尾tile。每核顺序遍历K块，K方向累加完成后只执行一次alpha/beta epilogue。候选BM/BN取32、64、128，BK按CANN9.1.0目标架构支持的布局及L0/L1容量确定；这些是待测候选而非固定性能结论。

Host读取实际UB/L1/L0容量，预算包含输入缓冲、FP32累加、转置/打包临时区、输出、事件和双缓冲。尾块先置零再搬入有效范围，DMA按合法长度搬运，不能把逻辑padding当作有效乘积输入。大尺寸与乘加索引使用64位，硬件窄字段溢出时分段处理或明确返回INVALID_VALUE。

### Kernel侧设计

采用Ascend C直调，由Host传递只含已验证标量元数据的tiling结构：m/n/k、lda/b/c/d、转置位、alpha/beta、tile尺寸、任务数、路径标识和workspace偏移。

1. CopyIn按行主序和转置地址映射搬运有效A/B块，必要时转换到Cube支持的布局。多buffer采用显式队列或事件维护MTE2、计算、MTE3生命周期，不遗漏最后一块同步。
2. Compute将乘积在FP32中累加。先保留严格FP32向量基线；Cube路径参考当前arch22 GEMM的 `AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE)` 明确关闭HF32，并在CANN9.1.0实际目标编译和实测确认，不能默认HF32/TF32或FP16截断能通过Netlib门槛。若需高低分量补偿，先独立验证抵消、极端值、subnormal及非有限值，再做全量回归，禁止仅用大随机矩阵平均误差放行。
3. Epilogue先保存所需C，再执行alpha乘积与beta缩放；beta=0不搬C；退化路径独立zero/copy/scale。不得用一般乘加代替alpha=0的EXACT语义。
4. CopyOut按ldd写逻辑结果，保留尾部、行padding和周边哨兵；不同核不得写同一输出元素。

对于极端范围，允许输出按任务规则溢出为Inf/NaN，但不得把有限golden变成非有限。最终数值判定以固定Netlib 3.10.0主标杆为准，不能以“数学上更精确”的double累加结果替代验收结果。

### 工程组织与调用链

拟采用当前主仓结构，等待设计评审确认任务书旧路径差异：

```text
include/cann_ops_blasLt.h                     # 保持原有声明
blasLt/api/aclblaslt_matmul.cpp               # 公共入口，arch22条件分派
blasLt/api/aclblaslt_matmul_algo.cpp          # 架构候选和真实workspace预算
blasLt/matmul/engine/                        # 保留公共问题描述与分派
blasLt/matmul/fp32/arch22/
  matmul_fp32_host.cpp                       # Host/tiling/launcher
  matmul_fp32_host.h
  matmul_fp32_kernel.cpp                     # GEMM及退化分支
  matmul_fp32_kernel.h                       # 必要时使用
  matmul_fp32_tiling_data.h
test/blasLtMatmul/arch22/
  blasLtMatmul_test.cpp
  blasLtMatmul_npu_wrapper.h
  blasLtMatmul_test.csv
  blasLtMatmul_golden.h                      # arch22 Netlib适配，不替换arch35 golden
test/blasLtMatmul/CMakeLists.txt              # 复用注册，必要时条件链接Netlib
```

README沿用实际Lt目录的说明入口，增加A2/A3支持表、限制、workspace、构建及例程。新增文件保持最小范围，不复制整套API/描述符/框架；不增加无用途的CMake、源码副本或根目录交付文件。共享engine当前包含arch35头文件，需要架构条件隔离、明确唯一launcher定义，并验证未选架构不参与编译/链接；不能只创建arch22文件而遗漏构建安装选择。

## 支持硬件

| 产品 | 设计支持 | 验证计划 |
| --- | --- | --- |
| Atlas A2（含800T A2，910B3） | 是，arch22 | CANN9.1.0编译、完整精度/性能/内存真机测试 |
| Atlas A3（含800T A3） | 是，arch22 | 独立目标编译；有设备后补实机，不以A2测试冒充A3 |
| Ascend950 | 本次不新增 | 保护arch35既有路径、公共API及测试的兼容性 |

## 算子约束限制

主路径仅FP32、HOST标量、ROW、单batch和DEFAULT epilogue。支持ld表达的行间padding，不支持额外stride、广播和输出与A/B别名。并发调用共享只读描述符时不得修改其内容，算法缓存若引入需包含设备和所有影响tiling的属性并保证线程安全；首版不依赖全局可变缓存。调用方负责维持输入、输出、workspace至stream完成的生命周期。

# 可维可测分析

## 精度标准/性能标准

| 验证项 | 判据及记录 |
| --- | --- |
| 正式golden | Netlib BLAS 3.10.0的cblas_sgemm，固定库路径、版本、构建选项与SHA256；检查动态加载，禁止意外解析到OpenBLAS |
| 有限FP32 | atol=rtol=2^-13，matched_ratio≥0.99；按opbase标准，硬上限取max(1e-2,32*ULP(g_low))，g_low为最大绝对误差点的golden收窄为FP32后的值；保存该点、ULP和最终界，不能取整个矩阵最大值的ULP放宽限制 |
| 非有限值 | 同NaN、同号Inf、NaN与±Inf按任务规则等效；相反符号Inf失败；有限与非有限失败。记录总数及匹配数，有限集合为空时不做除零统计 |
| EXACT | alpha=0的缩放/复制逐位核对，beta=0不读C；必要特殊值单列位模式与分类结果 |
| 性能 | 全部200条每例GPU_us/NPU_us；报告≥0.8及≥1.0，不能用均值或几何均值替代逐例门槛 |
| workspace | 每性能case报告分配/实际峰值、输入输出字节和比值；与GPU workspace之比待基线补充，不写成已通过 |

正式golden按行主序cblas调用，C!=D先在独立CPU输出缓冲复制C有效区，再执行SGEMM；beta=0不读取C旧值。C==D使用输入快照生成golden，不能用被NPU覆写后的C生成。输入与标量使用CSV规定的填充、种子和FP32舍入；Netlib例程本身采用列主序，但通过CBLAS_ROW_MAJOR正确映射，不能直接使用任务书行宽作为Fortran列主序lda。

## 测试覆盖与执行审计

实读附件分类：L0 13、MN 116、AB 24、LD 14、FL 24、CV 16、ED 49、EX 744、PF 200。前八类合计1000精度，其中33条期望非SUCCESS；全部1200条的期望状态分别为SUCCESS 1167、INVALID_VALUE 25、NOT_SUPPORTED 5、INVALID_ENUM 2、NOT_INITIALIZED 1。

CSV缺失的行为以独立补充case测试，不重写官方case：9种转置组合的非方阵；k=0/alpha=0配A/B空指针和NaN毒化；beta=0配C空/NaN；异地beta=1复制；正负零；独立ldc/ldd与padding哨兵；m/n=0同时含非法布局的优先级；workspace错位/不足/边界；算法损坏/不支持配置；非默认stream、连续重复调用和并发不同stream。非法stream在隔离测试进程验证运行时返回，不能用野指针导致测试进程崩溃后算作通过。

负维度、非法枚举或类型可能在描述符setter阶段就失败，测试报告分开记录构造状态和算子调用状态；对需覆盖公共API的路径用测试专用受控元数据夹具，不在产品接口放宽参数校验。每case记录是否真正调用目标符号、同步结果、返回码、数值与padding检查。

保留原始脚本执行输出，另以独立审计验证退出码、预期case ID集合、每ID恰好一次和总条数。超时、进程崩溃、零case、过滤器误匹配、NO_REF均不视为通过。新增驱动只补充采集与审计，不改变原始CSV、期望值、golden或容差。

## 性能与内存采集

910B3固定设备、CANN版本、源码commit及库hash，先做正确性验证，再在单独性能阶段预分配数据/描述符/workspace；不把CPU golden、H2D和GTest整例耗时当kernel时长。每case warmup5次后采60次，原始msprof文件保留kernel名、调用序号和Task Duration(us)。工具自动warmup与应用warmup需通过trace核对，避免重复计入有效样本。

同一次API若含pack、GEMM、epilogue或归约等多个kernel，按调用归组累加全部设备任务，不能只挑最快或主kernel。以60次总和/60为正式测量，并列前10次均值用于§3.3口径；ms换算为us乘1000。性能CSV的200行通过case顺序及五元参数一致性验证逐行关联，保留重复参数的两份独立测量，不压缩成198条。

内存总量先按A/B物理行数×ld和C/D行数×ld计算，再区分逻辑大小与实际分配。C==D重复存储只计一次；报告定义清楚。若任务方未提供GPU workspace，不用0或推算值填充该列。

使用msprof分析搬运、Cube利用率、AIV epilogue、调度和同步瓶颈：小矩阵减少启动和转置开销；中大矩阵优化tile与双缓冲；skinny形状重新分配输出tile。每次改变精度模式、分块归约或算法分派后先全量正确性，再跑完整200性能与内存；任何一条失败继续定位，不筛case、不换golden、不只提交最好一次结果。

## 兼容性与风险

- 公共API符号和结构体ABI保持不变；arch22专用配置不覆盖arch35算法对象解释。显式检查同名符号、头文件搜索路径及最终共享库加载路径。
- CANN9.1.0与最新主仓可能存在工具链适配差异。独立构建A2/A3目标以及共享API所影响的arch35目标；编译失败按工具链/框架/算子分别定位，不能跳过目标后宣称支持。
- FP32累加与Netlib顺序存在差异，Cube精度与速度需实测取舍；设计不保证候选策略天然达标，未全量通过前不交付验收结论。
- 任务性能、基线硬件、GPU workspace及目录差异须在评审中确认。当前设计按较严性能目标推进，但正式验收口径变更必须有任务方依据。
- 设计文档只提交竞赛仓；未来代码PR按维护者要求附设计链接，不将设计长文、截图、XLSX、运行日志或验收压缩包放入源码树。

## 验证步骤及交付清单

1. 冻结任务书、6个测试材料文件的SHA256、两仓commit和环境信息。原始材料只读保留，不在运行时覆盖生成的GPU基线。
2. 按当前构建流程验证 `bash build.sh --soc=ascend910b3 --ops=blasLtMatmul`，A3采用对应实际SoC；确认arch22目标不是被跳过、公共符号来自本次安装库。
3. 运行原精度驱动全部1000条和独立状态/数量审计，再对200性能用例检查正确性；补充case单独统计。链接固定Netlib3.10.0并留存`ldd`及符号解析证据。
4. 使用独立采集驱动和msprof获得全部200条、每条60有效样本的设备耗时与workspace，输出参数、GPU/NPU耗时、两档倍率判据、内存及证据路径。
5. 交付官方模板XLSX、Markdown步骤说明、源码commit链接、完整用例/测试代码、原始精度与性能日志、真实终端截图、文件校验和与验收说明。截图须来自对应实测，不用重排或手工伪造输出替代证据。
6. 验收上传包检查ZIP格式、大小和平台文件数限制；保留完整归档并准备直接可读的精简上传包。私仓邀请Ascend-CANN在提交验收前完成；设计评审、平台验收、上游CI和代码合入分别跟踪。

## 参考资料

- [社区任务目录与提交规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
- [ops-blas贡献指南（调研版本）](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/CONTRIBUTING.md)
- [现有Lt公共接口](https://gitcode.com/cann/ops-blas/blob/05b96ea22cea926ecfd43fc7ac691a795522bd61/blasLt/api/aclblaslt_matmul.cpp)
- [cuBLASLt官方接口文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-LtMatmul)
- [Netlib SGEMM语义参考](https://www.netlib.org/blas/sgemm.f)；正式golden锁定任务指定3.10.0，不以该在线最新版替代版本锁定。
- [生态算子混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
