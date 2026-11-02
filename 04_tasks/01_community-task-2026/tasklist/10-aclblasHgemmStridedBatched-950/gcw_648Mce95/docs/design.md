# aclblasHgemmStridedBatched 算子设计

## 需求背景

### 需求来源

本设计对应[10月社区任务-aclblasHgemmStridedBatched算子开发（950）](https://www.hiascend.com/activities/task-center/details/39ca031614a04829b74c47a68265db99)，依据[官方任务书及测试附件](https://www.hiascend.com/p/resource/202610/1f9095be1df840d0a86484c2c49bbc3c.zip)。目标平台为 Ascend 950PR，软件版本为 CANN 9.1.0，使用 Ascend C kernel 直调方式补齐 ops-blas 公共接口。

### 背景介绍

固定步长批量 GEMM 对多组矩阵执行相同的矩阵乘加。各批次共享矩阵维度、转置方式及标量，通过 stride 定位所在缓冲区，不需要设备指针数组。A、B、C 及 Host 侧 alpha、beta 均为 FLOAT16。

在 [ops-blas@e56c4cd6](https://gitcode.com/cann/ops-blas/tree/e56c4cd6f41ea63712f1c5c123a596004ce2e574) 中，同族 `gemm_strided_batched_ex/arch35` 已有 half 输入、float 累加的 Cube 设备路径，包含跨 batch 的任务分配和 leading dimension 寻址。本接口复用该设备计算能力，单独实现任务书规定的参数检查和 FP16 输出语义。

## 需求分析

### 需求描述

对第 b 个批次执行：

```text
A_b = A + b * strideA
B_b = B + b * strideB
C_b = C + b * strideC
C_b = alpha * op(A_b) * op(B_b) + beta * C_b
```

矩阵按列主序存储。`op` 支持 N、T、C；对于实数输入，C（共轭转置）与 T（转置）的数值结果相同。stride 的单位为元素个数，类型为 `long long`。

公开接口按任务书 §2.3 新增至 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasHgemmStridedBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, long long strideA,
    const aclblasHalf* B, int ldb, long long strideB,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc, long long strideC,
    int batchCount);
```

| 参数 | 语义与检查 |
| --- | --- |
| handle | 使用绑定 stream 的有效句柄；空句柄返回 HANDLE_IS_NULLPTR |
| transa、transb | N/T/C；非法值返回 INVALID_ENUM |
| m、n、k、batchCount | 非负；m、n 或 batchCount 为 0 时按规定顺序检查后返回成功 |
| lda | N 时至少 max(1,m)，T/C 时至少 max(1,k) |
| ldb | N 时至少 max(1,k)，T/C 时至少 max(1,n) |
| ldc | 至少 max(1,m) |
| strideA、strideB、strideC | 非负的 64 位元素偏移；负值在零批次场景仍返回 INVALID_VALUE |
| alpha、beta | Host FLOAT16 标量指针；即使是 no-op 场景也必须非空 |
| A、B | 非 no-op 且 k>0 时必须非空，包括 alpha=0 的情况 |
| C | 非 no-op 时，若 k>0 或 beta≠0 则必须非空；k=0、beta=0 时按任务书允许为空 |

### 需求拆解

1. 新增公共接口和统一的 `aclblasHalf` 类型支持，满足 C/C++ 调用及 Host/Device 二进制表示要求。
2. 实现参数检查、零维返回、k=0 与 alpha=0 的退化计算。
3. 复用 FP16 Cube 计算路径，实现批次和矩阵分块调度、转置及非紧凑布局。
4. 实现 FP32 标量收尾和一次 FP16 舍入，保护 padding 和批次间隙。
5. 覆盖任务附带全部用例，按设备耗时测量性能，提交可复现的自测材料。

## 详细设计

### 算子分析

每个输出元素包含长度为 k 的点积。输入和标量按 FP16 表示，乘积累加和 alpha/beta 收尾在 FP32 域进行，最终输出按 round-to-nearest-even 转为 FP16。中间乘积不能先转成 FP16 再做 alpha/beta 收尾，以免增加一次舍入。

逻辑形状为 `op(A): m×k`、`op(B): k×n`、`C: m×n`。物理 A 在 N 模式为 m×k，T/C 模式为 k×m；物理 B 在 N 模式为 k×n，T/C 模式为 n×k。批次首地址、leading dimension 与转置共同决定有效元素位置。

任务书允许运行时输入 m/n/k/batchCount；实现不能只适配测试表中的形状。偏移、任务总数、工作区字节数使用带溢出检查的 64 位运算，非法维度先返回，不能先分配缓冲或计算无符号长度。

### Host 侧设计

#### 校验顺序与退化分支

检查顺序为 handle、转置枚举、m/n/k/batchCount、lda/ldb/ldc、stride、alpha/beta 指针。完成这些检查后，m=0、n=0 或 batchCount=0 直接返回成功，不访问 A/B/C。非 no-op 再执行设备指针检查：k>0 时 A/B 必须非空；k>0 或 beta≠0 时 C 必须非空。

| 条件 | 行为 |
| --- | --- |
| k=0、C为空、beta=0 | 按任务书 §2.4 和原用例 TC_ED_145 返回成功，不启动 kernel |
| k=0、C非空、beta=0 | 仅将每批逻辑 C 元素置零，不读旧 C，不访问 A/B |
| k=0、beta=1 | 保持 C 不变；仍按规定检查指针 |
| k=0、其他 beta | FP32 计算 beta×C，再舍入为 FP16；不读取 A/B 或 alpha 数值 |
| k>0、alpha=0 | 完成 A/B/C 指针检查后仅处理 beta×C，不读取 A/B 数据 |
| 普通 GEMM、beta=0 | 不读取旧 C，避免旧 C 的 NaN 进入结果 |

`aclblasHalf` 在公共类型头中统一定义，首版提案为 `typedef _Float16 aclblasHalf;`，与同批 Hgemm/HgemmBatched 接口共用。新接口仅在编译器提供 IEEE binary16 数值类型时启用，检查有效尾数11位、最大指数16、sizeof为2；C和C++调用方均需验证alignment及与Device half一致的位表示。缺少该类型的编译器应对本接口给出不支持诊断，既有接口头文件仍应可用；不以整数typedef替代浮点数值语义。公共定义形式由评审确认，远端编译验证是采用该提案的条件。

alpha、beta按Host标量处理。通过必要检查后，将所需标量值复制到本次调用的tiling参数，再异步下发；kernel不保存或解引用调用者的Host指针。k=0分支只读取beta数值。

#### 代码复用与文件位置

实现按任务书 §2.2 放在 `blas/gemm/arch35/`，测试按 §5 放在 `test/gemm/hgemmStridedBatched/arch35/`，公共接口放在 `include/cann_ops_blas.h`。对应测试目录提供必要的 CMake 注册与独立运行入口。附件脚本使用另一套目录查找规则，需要独立适配 runner；官方 CSV 和基线文件保持原样。

拟复用 [gemm_strided_batched_ex 的 FP16 设备实例](https://gitcode.com/cann/ops-blas/blob/e56c4cd6f41ea63712f1c5c123a596004ce2e574/blas/gemm_strided_batched_ex/arch35/gemm_strided_batched_ex_kernel.cpp)。该文件已有 half×half、float 累加、float 或 half 输出以及设备内批次循环。只复用所需内部计算入口和 tiling 字段，不调用 Ex 公共接口的参数检查：Ex Host 对多批次零 stride 的限制与本任务附件不同。实现前验证内部入口的链接关系；需要提取共享模板时保留原 Ex 接口行为并回归受影响路径。

#### 批次调度与分块

初版将 `batchCount × ceil(m/BM) × ceil(n/BN)` 组成任务域，设备核心以 grid-stride 方式遍历。核数取平台可用 AIC 数与任务数的较小值，避免 Host 为每个小矩阵单独启动 kernel。首版采用 splitK=1，每个逻辑输出 tile 由一个任务完成，减少额外归约与工作区。

列主序映射沿用已核查的同族方案：交换 A/B、m/n、对应 ld 与 stride 后调用内部 Cube 入口；N/T/C 归并为四种实际转置组合。初始候选 tile 使用现有 FP16 实例的 64×64 或 128×128 空间分块，BK 使用现有 16/32 分块。最终选择根据片上容量、总任务数和 950PR 测量确定，不按 case 名称或输入数据特判。

小矩阵大批次优先利用 batch 并行度，大矩阵使用空间分块。只有完整测量表明现有路径不能满足小矩阵性能时，才增加单独的小矩阵路径；新增路径与通用路径共享接口语义和精度检查。

#### 分组寻址

设当前批次组起点为b0，组内批次为g；输出块从逻辑行r0、列j0开始，大小为Mr×Ns。令b=b0+g、i=r0+iLocal、j=j0+jLocal，t为K维下标，则元素地址为：

```text
A，N：   b*strideA + i + t*lda
A，T/C： b*strideA + t + i*lda
B，N：   b*strideB + t + j*ldb
B，T/C： b*strideB + j + t*ldb
C：      b*strideC + i + j*ldc
```

传给本组计算的A起点增加`b0*strideA + (transa=N ? r0 : r0*lda)`，B起点增加`b0*strideB + (transb=N ? j0*ldb : j0)`。外部逻辑维度为Mr、Ns、k，lda/ldb和批次stride仍取调用者传入值；内部入口交换A/B后，维度变为Ns、Mr、k，对应ld与stride一并交换。

临时乘积P采用列主序，`ldP=align_up(Mr,16)`，`colsP=align_up(Ns,16)`，`strideP=ldP*colsP`。逻辑元素地址为`g*strideP + jLocal*ldP + iLocal`。内部输出ld使用ldP，批次偏移使用strideP，不能沿用C的ldc和strideC。收尾kernel只读取Mr×Ns有效元素，按上述C地址写回。

### Kernel 侧设计

#### 搬入与累加

A/B 由 GM 按真实 ld、stride 和转置方式搬入 L1，再进入 L0A/L0B。复用已有双缓冲和事件依赖，保持搬运完成、计算完成与缓冲重用的顺序。M/N/K 尾块只读取有效 GM 元素，片内不足位置补零；不能为了凑齐 tile 跨批次或越界读取。

Cube 使用 FP16 输入和 FP32 Mmad 累加。常规路径先将乘积输出至 FP32 工作区，专用 AIV 收尾计算 `alpha×P+beta×C`。beta=0 时跳过旧 C 的加载。最终转换采用 IEEE round-to-nearest-even，保留任务要求的 Inf/NaN 行为；现有 Ex 收尾中的 `__float2half_rn_sat` 不直接作为已满足该语义的依据。

alpha=1、beta=0 时可以评估 FP16 Fixpipe 直接写回，以减少收尾 kernel 和工作区。该路径必须先验证尾块、带 padding 的 ld、溢出、非有限值与舍入结果；通过后才能用于性能候选。

#### 工作区与写回

工作区管理向本次调用提供基址、实际可用字节数W和所有权记录。扣除对齐及其他临时区后得到Wusable，乘积缓冲须满足`groupBatch*ldP*colsP*4 <= Wusable`，所有乘法先检查溢出。给定输出块尺寸后，取`groupBatch=min(剩余批数, floor(Wusable/(ldP*colsP*4)))`。

若一整个批次放不下，先缩小列块Ns；仍放不下时缩小行块Mr，按分组寻址公式推进r0、j0。每次至少处理一个合法tile。若连最小tile所需临时空间都无法取得，返回`ACLBLAS_STATUS_EXECUTION_FAILED`，不能启动部分计算后报成功。地址或字节数计算溢出返回`ACLBLAS_STATUS_INVALID_VALUE`。alpha=1、beta=0的直接写回路径无需乘积缓冲。

每次调用持有独占的工作区记录，绑定下发stream和完成事件。同一调用在同一stream依次执行“Cube→收尾→下一组”，可复用同一地址。最后一个收尾任务完成后才释放该记录；Host返回不代表设备已完成。另一stream只能使用独立工作区，或在设备侧等待前一完成事件后接管；同一handle切换stream也遵循这条规则。优先复用仓内已有事件与临时资源设施；若缺少上述所有权保证，需补齐本接口的管理适配，并验证切换stream及连续调用的生命周期，不增加全局同步。

写回只覆盖各批次 C 的 m×n 逻辑元素，ldc padding 和批次间隙保持不变。批次地址与 tile 地址均使用检查过的 64 位偏移。

### 支持硬件与约束

支持 Ascend 950PR，软件版本为 CANN 9.1.0。

| 项目 | 处理方式 |
| --- | --- |
| 输入布局 | 列主序，通过 ld 和固定 stride 描述；无其他非连续视图 |
| A/B stride=0 | 按附件正向用例复用同一只读输入地址；小于矩阵跨度的只读重叠也按真实地址读取 |
| C 重叠输出 | 原用例未构造；任务书的 stride≥0 表述需与无广播约束进一步确认。不能自行改 stride、默默拒绝或允许并行写竞争 |
| 确定性 | 任务书不额外要求；首版splitK=1，不使用跨任务浮点原子累加 |
| CPU 路径 | CPU只用于测试golden，不作为被测算子的计算回退 |

## 可维可测分析

### 精度标准与用例

保留原始 1000 条精度用例和 200 条性能用例，并对性能用例验证计算结果。输入矩阵和标量先量化为 FP16，再提升到 FP32，逐 batch 调用 Netlib BLAS 3.10.0 `cblas_sgemm`，结果按 round-to-nearest-even 转回 FP16，逐 batch 对逻辑 C 矩阵验证。

任务书 §3.2 的有限值判据为：

```text
abs(actual-golden) <= atol + rtol*abs(golden)
rtol = atol = 2^-10
matched_ratio >= 0.99
max_abs_error_limit = max(0.1, 32 * ULP(g_low))
max_abs_error <= max_abs_error_limit
```

依据任务书引用的[生态精度标准§2.1.2、§2.2](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)，`or`取两项较大者；g_low是最大绝对误差点的golden经FP16舍入后的值。每个batch独立计算有限值的通过比例和最大绝对误差，差值与阈值运算至少使用FP32。FP16正规数的ULP为`2^(floor(log2(abs(g_low)))-10)`；零及次正规数的ULP为`2^-24`。

若一个batch没有有限值比对点，只检查其所有非有限值是否符合下述规则，不计算0/0的比例；空输出按接口返回和不写内存验证。MERE/MARE可以作为附加诊断，不能替代上述生态判据。任务专用检查显式使用2^-10。

golden 与 actual 同为 NaN、同为同号 Inf，或一方为 NaN 而另一方为 Inf，按任务书视为等效；异号Inf、一方有限而另一方非有限均判失败。非有限元素单独统计并全部满足规则，不能从结果中删除。大 k 场景若出现任务书 §3.2.4 所述问题，提交逐元素误差、累加方式及 CPU 对比证据，不自行放宽阈值。

补充测试覆盖参数检查顺序、m/n/k尾块、非方阵、9种转置组合、ld/padding、A/B共享地址、beta=0且旧C为NaN、k=0且alpha为NaN及FP16极值。强制工作区小于整批乘积，覆盖至少两个批次组、两个列块、行分块以及转置B；核对第二组和第二条带的地址及padding。通过连续调用和切换stream检查工作区回收；资源不足和64位运算溢出另设错误返回用例。

原始1000条精度和200条性能用例保持不变。为采用均匀随机填充的原矩阵场景配套增加同形状、同布局的正态随机场景，A、B、C分别记录分布计数。另设64条随机标量用例，alpha、beta各包含32个均匀样本和32个正态样本，采样后量化为FP16；种子固定，逐条记录。各参数的随机场景中均匀与正态各占50%；原有固定标量及特殊值单列，不计入比例。均匀值域为[-5,5]；正态参数取μ∈[-5,5]、σ∈[0.1,2]。

补充规模边界分别覆盖m=16384、n=16384、k=16384和batchCount=65536，使用其他维度较小的非方阵或小矩阵控制内存量。补充集只用于功能和精度覆盖，不改变200条官方性能用例的统计分母。

### 性能标准与采集

正式测量使用 msprof op 设备 task duration，不采用 GTest 的整数毫秒总耗时。一次接口调用包含多个 kernel 时，保留该调用全部任务的映射与耗时分解，并报告累计设备耗时；不会只选矩阵乘 kernel、遗漏收尾或批次分组。

任务书 §3.3 要求每条倍率 `GPU_us/NPU_us≥0.4`，全部性能用例倍率的算术平均≥0.5；§7又写“达到≥GPU数据”。[任务讨论#61](https://gitcode.com/cann/ops-blas/discussions/61)中，condfuse_3 对上述差异回复“都要达标”，并明确“msprof op 自带5次warmup后，连续采集10个有效样本取平均”。

据此，正式统计采用5次预热及10个有效样本，保留原始数据。开发目标暂按每条倍率≥1.0准备，同时完整报告§3.3的逐条0.4和平均0.5指标；每条≥1.0是避免遗漏较严条款的设计取值，请评审确认最终验收是否要求逐条达到GPU，及多kernel累计方式。额外重复测量用于稳定性检查，单独记录，不混入这10个样本。

200条原性能用例全部保留。相同参数的两条用例仍按各自case ID统计，平均值分母为200；单位统一为us，计算使用未舍入值。缺case、NO_REF、无效样本、测试进程失败或实际调用未执行都不能判通过。任何性能结论必须同时给出源码、编译配置、二进制版本、设备信息及原始profile。

### 兼容性与评审确认

新增接口与half类型在公共头中统一维护。原有GEMM及Ex接口行为保持兼容，测试只调整本任务必要的注册与适配，不能为单个候选更改共享编译优化级别。共享设备模板发生修改时，回归对应的现有调用路径。

本次设计评审需确认：公共 `aclblasHalf` 定义形式；C重叠输出的合法性及处理语义；性能是否逐条达到GPU以及多kernel统计方式；附件脚本与任务书指定测试目录的适配方式。其余接口、参数检查与测试覆盖按任务书执行。
