# 单精度实数 LU 分解、求解和批量求逆（950）设计文档

## 需求背景（required）

### 需求来源

本设计对应 [10 月社区任务：单精度实数 LU 分解、求解和批量求逆（950）](https://www.hiascend.com/activities/task-center/details/cd66edbecb8840b3bbcd39035146a970?menu=tasks)，任务编号为 `10-1`，提交人为 `cuizihua`，GitCode 账号为 `cuiwen123`。要求以广场提供的[任务书及测试附件](https://www.hiascend.com/p/resource/202610/9f2b9148f987448eaea2dff10e16b60d.zip)为准，实现交付仓为 [cann/ops-solver](https://gitcode.com/cann/ops-solver)。

### 背景介绍

LU 分解通过部分主元选取将一般实矩阵分解为置换、单位下三角与上三角因子，可用于构造逆矩阵。对于许多独立的小矩阵，批量接口可减少逐矩阵调用和调度开销；对已分解的 LU 因子求逆还可避免重复分解。

ops-solver 已有实数 `Sgetrf`、`Sgetri` 原型。以提交 `0bf7eebb6a400ca33c135867eae5ce080ff1b13d` 为参照，旧原型使用 Host 矩阵指针、内部内存分配和同步，限制 `lda == n`，缺少公开 workspace 查询，且单矩阵求逆接口不接受主元。本任务需升级公开接口与数据语义，并补齐实数批量求逆接口。

## 需求分析（required）

### 功能与交付范围

按任务书第 2.3 节，设计覆盖以下四个计算接口和两个 workspace 查询接口，作为同一任务交付：

| 接口 | 功能与对标 |
| --- | --- |
| `aclsolverSgetrf`、`aclsolverSgetrf_bufferSize` | 对一般 `m × n` 实矩阵进行带部分主元或无主元 LU 分解，对标 cuSolver DN `Sgetrf`。 |
| `aclsolverSgetri`、`aclsolverSgetri_bufferSize` | 根据 LU 因子及对应主元原地求逆，语义对标 LAPACK `SGETRI`。 |
| `aclsolverSgetriBatched` | 根据各矩阵的 LU 因子和对应主元，生成独立的批量逆矩阵，对标 cuBLAS `SgetriBatched`。 |
| `aclsolverSmatinvBatched` | 对 `n <= 32` 的原始小矩阵直接批量求逆，对标 cuBLAS `SmatinvBatched`。 |

矩阵类型为 FLOAT32，输入有效元素为有限浮点数，允许构造奇异矩阵以验证 info；主元与信息输出为 INT32。矩阵采用列主序，元素 `(i,j)` 位于 `A[i + j * lda]`，其中 `i,j` 为从零开始的下标。所有有效矩阵元素、主元、info 和计算 workspace 均位于 Device。维数、handle、枚举以及查询接口输出 `Lwork` 位于 Host；`Lwork` 表示 float 元素个数，所需字节数为 `Lwork * sizeof(float)`。

批量接口接收 Device 上的指针数组；数组中的每个元素指向一个 Device 矩阵，不能将其解释成连续的 `[batch,n,n]` 张量。批量求逆为独立输出，输入与对应输出的存储区域不得重叠。

### 支持规模与边界

| 接口 | 任务书规模下限与布局要求 |
| --- | --- |
| `Sgetrf` | `m,n ∈ [1,4096]`，包括方阵与矩形；`lda >= max(1,m)`，允许 padding。 |
| `Sgetri` | `n ∈ [1,4096]`；`lda >= max(1,n)`，允许 padding。 |
| `SgetriBatched` | `n ∈ [1,4096]`，`batchSize ∈ [1,1000000]`；`lda,ldc >= max(1,n)`。 |
| `SmatinvBatched` | `n ∈ [1,32]`，`batchSize ∈ [1,1000000]`；`lda,lda_inv >= max(1,n)`；`n=33` 必须报错。 |

在其他参数满足合法性要求时，`m=0`、`n=0` 或 `batchSize=0` 按对应空问题成功返回，不访问空矩阵或空主元数组。单矩阵接口仍按任务书检查 `devInfo`，并在可写时将其置零；批量 `batchSize=0` 时没有需要写入的状态元素，`n=0,batchSize>0` 时则将各矩阵的 info 置零。非空计算必须支持非默认 stream，不要求广播或图融合。

官方附件另有 getrf 的 `n > 4096` 用例和 matinv 的 `batchSize > 1000000` 用例。这些用例纳入附件验证，不按上表静默过滤；具体差异在文末列出供评审确认。维数、跨度和 workspace 的乘加均使用经过溢出检查的宽整数计算，避免 32-bit 公共维数导致内部地址回绕。

### 接口范围与基线的待确认事项

广场简介出现 `Sgetrs`，任务书多处写“五个接口”，但其第 2.3 节只定义上述四个计算接口，且没有给出 `Sgetrs` 的公开原型、精度判据或测试附件。本设计以任务书已明确定义的接口为对象，请评审确认是否还需交付 `Sgetrs`；若需要，应以补充的正式定义对齐，不能从其他接口自行推定。

## 详细设计（required）

### 算子分析

#### 数学定义

LU 分解满足 `P A0 = L U`，其中 `A0` 为原始 `m × n` 矩阵，`k=min(m,n)`，`L` 为 `m × k` 的单位下三角或梯形矩阵，`U` 为 `k × n` 的上三角或梯形矩阵。A 原地保存打包的 L/U：严格下三角保存 L 的非对角元素，对角线及其上方保存 U。

`devIpiv[j]` 保存第 `j+1` 步与当前行交换的 1-based 行号。部分主元从当前列尚未消元的行中选取绝对值最大者；并列时选最小行号。`devIpiv == nullptr` 时不选主元，数学关系为 `A0 = L U`。

单矩阵求逆与基于 LU 的批量求逆均计算 `C = A0^-1`。由 `P A0 = L U` 可得 `L U C = P`，因此将单位阵按主元正序进行行交换得到 P，再依次求解 `L Y = P` 和 `U C = Y`。无主元时令 `P=I`。`Sgetri` 原地覆盖 LU 因子；`SgetriBatched` 保留输入 LU，并写入独立的 C。

`SmatinvBatched` 的输入是原始矩阵，在每个 batch 内完成部分主元 LU 与求逆。它与 `SgetriBatched` 的输入契约不同，且 `n > 32` 不得通过转调其他接口绕开限制。

#### 公开接口

复用 `aclsolverHandle_t`、`aclsolverStatus_t` 与 `aclsolverCreate/Destroy/SetStream/GetStream`。在公共类型头文件中补充任务书要求的枚举：

```cpp
typedef enum {
    ACLSOLVER_OP_N = 0,
    ACLSOLVER_OP_T = 1,
    ACLSOLVER_OP_C = 2
} aclsolverOperation_t;
```

上述枚举的 N/T/C 对应不转置、转置与共轭转置，实数情况下 C 与 T 语义相同。本任务已列出的计算原型不增加 `trans` 参数。

```cpp
aclsolverStatus_t aclsolverSgetrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSgetrf(
    aclsolverHandle_t handle, int m, int n, float *A, int lda,
    float *Workspace, int *devIpiv, int *devInfo);

aclsolverStatus_t aclsolverSgetri_bufferSize(
    aclsolverHandle_t handle, int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSgetri(
    aclsolverHandle_t handle, int n, float *A, int lda, int *devIpiv,
    float *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverSgetriBatched(
    aclsolverHandle_t handle, int n, const float *const Aarray[], int lda,
    const int *PivotArray, float *const Carray[], int ldc,
    int *infoArray, int batchSize);

aclsolverStatus_t aclsolverSmatinvBatched(
    aclsolverHandle_t handle, int n, const float *const A[], int lda,
    float *const Ainv[], int lda_inv, int *info, int batchSize);
```

参数顺序、类型和含义保持任务书定义，不将 workspace 或主元从公开接口中隐藏。

| 参数 | 内存归属、方向与约束 |
| --- | --- |
| `handle` | Host 句柄，必须有效；计算使用其当前 stream。 |
| `m,n,lda,ldc,lda_inv,batchSize` | Host `int` 标量，尺寸非负，leading dimension 满足对应行数约束。 |
| 单矩阵 `A` | Device FLOAT32，getrf 输入原矩阵并输出 LU，getri 输入 LU 并输出逆矩阵；物理布局为 `lda × n`。 |
| `Workspace` | 调用方分配的 Device workspace，实际存储至少覆盖查询结果；不得与输入输出重叠。 |
| 查询的 `Lwork` | Host `int*` 输出，不能为空；查询只计算布局，不读取矩阵内容。 |
| getri 的 `Lwork` | Host `int` 输入，必须不小于该规模的查询结果。getrf 计算接口没有此参数，分配长度由调用方依据查询保证。 |
| `devIpiv` | Device INT32，getrf 输出长度为 `min(m,n)`，getri 输入长度为 n；空指针表示对应的无主元路径。 |
| `devInfo` | Device INT32 标量输出，非空；不能在 Host 直接解引用。 |
| `Aarray,Carray,A,Ainv`（批量） | Device 指针数组，每个元素为 Device 矩阵地址；输入为 const，输出独立存储，按各自 leading dimension 访问。 |
| `PivotArray` | Device INT32 输入，按 `PivotArray[b*n+j]` 保存第 b 个矩阵主元；空指针表示全部矩阵采用无主元因子。 |
| `infoArray,info`（批量） | Device INT32 输出，长度为 batchSize，各矩阵独立产生状态。 |

#### 状态与异常行为

公开返回值使用 `aclsolverStatus_t`。空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`；上下文未初始化、分配失败、不支持的设备和执行下发失败分别映射为已有的对应状态码。可同步判定的非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`，不启动数值计算；有可写的 Device info 时，在调用方 stream 上写入首个非法参数的 `-i`，参数编号不计 handle。

例如，getrf 的非法 `m,n,A,lda,Workspace,devInfo` 分别对应 `-1,-2,-3,-4,-5,-7`；getri 的非法 `n,A,lda,devIpiv,Workspace,Lwork,devInfo` 对应 `-1` 至 `-7`。合法的空主元指针不构成错误；查询接口没有 info 参数，错误通过公开返回值报告。

成功时 info 为零。若 U 的第 i 个对角元素为零，则 info 为最早的正整数 i；这是数值奇异信息，正常下发的接口可返回 SUCCESS，由调用方在 stream 完成后读取 Device info。getrf 保留可形成的因子，遇到零主元时不做除零；求逆对奇异矩阵停止该矩阵的求解，不将其输出解释为有效逆矩阵，也不影响其他 batch 的计算。

对 Device 主元中的越界索引、Device 指针数组中的空元素及对应输入输出重叠，先由校验 kernel 写入各矩阵的 info 和一个汇总状态。为兑现任务书要求的非法参数返回值，将该状态字回读 Host，在调用方 stream 完成参数校验后返回 INVALID_VALUE 或继续下发计算。该同步仅用于 Device 内容相关的参数检查，不在 LU 迭代之间同步，不把矩阵复制回 Host。此校验 kernel 计入对应调用的 NPU 耗时，Host 校验时延另外记录。调用方仍须提供真实有效的 Device 分配；地址范围和分配大小按 ACL 可用的指针属性检查能力验证，不能通过试读未知地址校验内存有效性。

### Host 侧设计

Host 负责参数校验、设备与 stream 获取、workspace 布局、分块参数选择和 kernel 下发，不承担 LU、主元搜索或三角求解计算。采用 ops-solver 的 Host C API 和 AscendC/CATLASS kernel 直调模式，保持源码、测试与构建接入该仓工程。

`_bufferSize` 与对应计算接口共用一个尺寸计算函数。工作区按对齐后的矩阵临时区、面板与更新临时区、置换及状态临时区分段，包含段间对齐开销，最终向上折算为 float 元素个数。所有尺寸在转换为公开 `int Lwork` 前检查溢出。空问题查询返回零；非空查询不依赖 A 的数值、主元数据或前次调用结果。getrf 的同一规模查询覆盖有主元与无主元路径，因查询接口没有主元参数。

getrf 的工作区用于面板、需要的矩阵打包以及核间交换数据；getri 的工作区保存仍要读取的 LU 因子、分块右端项和中间结果，防止原地写逆矩阵破坏尚未使用的因子。调用方在 stream 完成前保持 A、主元和 workspace 存活。不同并发调用各自使用独立的可写工作区，不能把上一次调用的状态作为本次结果的前提。

批量接口没有公开 workspace 参数。小矩阵以核内存储为主；超过核内容量的批量 LU 因子求逆，按一批可驻留的矩阵分配内部 Device 临时区并循环处理 batch 区间，避免分配 `batchSize*n*n` 的额外完整副本。内部临时区必须覆盖其最后一次异步使用：优先采用配套 ACL 支持的 stream 顺序分配和释放；若配套版本不支持，则在该 stream 上等内部任务完成后释放，并把这一资源生命周期同步作为接口开销记录，不提前释放，也不调用全设备同步。调用方输入输出不经 Host 中转。

分块尺寸和核数根据当前 m/n、batchSize、UB 容量与目标设备可用核心数选择。小矩阵优先减少 kernel 启动和核间通信；大矩阵将尾随子矩阵更新分配到多个核。所有临时区初始化与计算下发使用同一调用方 stream，顺序依赖由 stream 或受支持的核间同步机制保证。

### Kernel 侧设计

#### LU 分解

采用带部分主元的分块 LU。每个面板按以下顺序执行：对未消元列进行主元搜索，按确定规则归约；交换当前矩阵相应的整行，包括已生成的 L 因子部分；保存 1-based 主元；计算该列的 L 元素；完成面板内更新，再计算块行的三角求解和尾随矩阵更新。

尾随更新为 `A22 := A22 - L21*U12`。面板的主元与数据依赖按顺序完成，独立的尾随 tile 并行更新；前一阶段完成后才消费新面板。对矩形矩阵只推进 `min(m,n)` 个消元步骤，最后不足整块的行列采用实际有效范围，不能越过 leading dimension 或修改 padding。

主元归约采用固定的分组和合并顺序，不用非确定原子竞争决定主元。同一输入、同一 stream 串行重复调用时，L/U、主元和 info 均需 bit-wise 一致。小主元用 FLOAT32 除法形成倍数，不能未经验证改为先求倒数再相乘而导致可表示结果溢出。零主元与无主元分支均独立处理。

#### 单矩阵与基于 LU 的批量求逆

先检查 U 的对角线并报告最早的零对角位置。对非奇异矩阵，按列块构造单位阵右端项，正序应用输入主元，依次进行单位下三角求解和上三角求解，将解按列写为逆矩阵。主元只用于置换右端项，不能把 LU 因子当原始矩阵再次分解。

单矩阵原地求逆先保存所需 LU 数据到调用方 workspace，分块写回 A；批量求逆从每个 `Aarray[b]` 读取 LU，并按 `Carray[b]` 和 ldc 写出结果，不修改输入。小 n 时以一个核或一个固定协作组处理一组矩阵，大 n 时按三角块依赖进行多核求解。所有主元读取使用本矩阵的 `b*n` 偏移，每个 batch 有独立的 info。

#### 小矩阵直接批量求逆

`SmatinvBatched` 对每个原始矩阵在核内构造 LU 及置换，再通过三角求解得到逆矩阵。多个独立矩阵按 batch 区间分配到不同核；对每个核的尾部矩阵按真实数量处理。核内临时区按 n 和实际 UB 容量确定一次处理的矩阵数，不假定整个 batch 可放入 UB。

该路径始终使用部分主元，单个奇异矩阵仅写自己的正 info，并停止对该矩阵进行会除零的求逆步骤。输入和输出按不同的 leading dimension 访问，输入数据及 padding 均不被输出覆盖。

### 性能与数值精度设计

大矩阵主要通过分块三角求解与矩阵乘更新提高数据复用，结合双缓冲搬运降低 GM 访问等待；面板宽度受串行主元成本、UB 容量和更新粒度共同约束。矩阵乘使用目标 CANN/CATLASS 支持的路径，不将较低乘法精度模式默认当作 FLOAT32 等价实现；每种候选均以任务书残差验证，不能只凭 FLOAT32 累加判断精度满足要求。未满足残差判据时采用完整 FLOAT32 更新路径。

小矩阵和大 batch 优先融合主元、分解、求解等核内阶段，减少启动次数与中间矩阵搬运；按 n、leading dimension 和 batchSize 选择每核矩阵数，避免只优化单个典型规格。确定性要求同时约束归约、分块和结果写入，不因性能优化改变公开语义。

性能目标为逐用例达到任务书 GPU 基线的至少 0.35 倍，是否达标以 950PR 上的官方 profiling 结果判定，不以理论计算量替代实测。

### 支持硬件与兼容性

目标为 Ascend 950PR（Atlas 950），CANN 9.0.0 及以上，实际软件组合与 ops-solver README 已验证版本配套。核心计算在 NPU AI Core 完成，不使用 CPU fallback，也不依赖 CUDA 作为 NPU 实现。NumPy/SciPy/LAPACK 与 CUDA 库仅用于测试参考和性能对标。

公开声明位于 `include/cann_ops_solver.h`，类型位于 `include/cann_ops_solver_common.h`；实现和测试分别进入 `src/sgetrf`、`src/sgetri`、`src/sgetri_batched`、`src/smatinv_batched` 及对应 `test` 目录，并同步 API 列表与算子文档。

旧 getrf/getri 原型与新接口的参数个数、类型及语义不同，不能在同一 C 符号下同时保留两套 ABI。按任务书升级声明、定义及仓内相关调用样例和测试，要求旧调用方重新编译并改为 Device 指针和显式 workspace 流程；不将旧 Host 指针入口继续作为新接口使用。不改动无关复数接口的公开契约。

## 可维可测分析

### 测试工程与覆盖

使用 ops-solver 仓现有 CMake、测试二进制和各算子测试目录，接入官方附件用例；不以其他工程的运行成功替代仓内测试。每个用例分别记录接口返回值、Device info、主元、有效输出、精度残差与性能结果。

| 测试类别 | 验证内容 |
| --- | --- |
| 基础功能 | 验证方阵 LU、矩形 LU、getrf 后 getri、已有 LU 后 getriBatched、原矩阵 matinvBatched 的完整语义。 |
| 布局 | 覆盖最小 leading dimension、加 8 和加 32 padding；检查输入输出有效区，并以哨兵检查 padding 不被改写。 |
| Device 指针数组 | 在 Device 构造非连续矩阵地址数组，验证逐矩阵地址解析、主元偏移与独立输出，检查非法的对应输入输出重叠。 |
| 主元 | 覆盖有主元和空主元、首行非最大主元以及绝对值并列时取最小下标，验证行置换和逆矩阵置换方向。 |
| 信息与错误 | 覆盖成功零 info、精确奇异矩阵的正 info，以及非法尺寸、leading dimension、workspace、主元和可写 info 的负参数序号。 |
| 边界 | 覆盖 `n=1`、matinv 的 `n=32`、非法 `n=33`、`batchSize=1` 与空问题；按任务书覆盖 2 的幂及其减一的尺寸。 |
| 确定性与 stream | 相同输入在同一 stream 串行重复运行，逐字节比较数值输出、主元和 info；非默认 stream 通过事件和调用方同步确认依赖正确。 |
| 组合路径 | getrf 的 Device 因子及主元直接交给 getri/getriBatched，覆盖 padding、有主元与无主元，不用 Host 重构因子替代组合调用。 |

自构造数据遵循任务书的均匀/正态分布与对角占优规则，并另覆盖奇异、近奇异及 info 行为。batchSize 覆盖 `1,8,128,1024,3000` 和零；自构造数据遵守任务书的内存限制。原文出现的 `nrhs/trans/B` 场景随 `Sgetrs` 的接口范围一并确认，不向现有四个计算接口添加不存在的参数。

### 精度标准

矩阵 1-范数取各列绝对值之和的最大值，FLOAT32 单位舍入取 `ε=2^-24`。CPU 比较链路使用 FLOAT32 LAPACK 分解或求逆，分别计算同用例的 `ratio_cpu`，不能用 float64 分解误差直接代替。残差计算对 CPU 与 NPU 输出使用相同的高精度评估方法，减少检验计算自身的舍入影响。

| 接口 | 残差与判据 |
| --- | --- |
| `Sgetrf` | 从打包因子构造 L/U，按主元逆序回放行交换恢复为原矩阵行序，得到 R；`ratio=norm1(R-A0)/(n*norm1(A0)*ε)`，其中矩形时 n 为列数；要求 `ratio <= max(5*ratio_cpu,0.3)`。 |
| `Sgetri` | A0 按任务书由输入 LU 与主元还原，C 为输出逆矩阵；`ratio=norm1(I-C*A0)/(n*norm1(A0)*norm1(C)*ε)`，要求 `ratio <= max(5*ratio_cpu,0.05)`。 |
| `SgetriBatched` | 对每个 batch 使用上述求逆残差及自身的 CPU 参考，要求 `ratio_i <= max(5*ratio_cpu_i,0.05)`，任一矩阵失败则该用例失败。 |
| `SmatinvBatched` | A0 为该 batch 原始输入，采用同样的求逆残差，要求 `ratio_i <= max(5*ratio_cpu_i,0.5)`。 |

对于奇异矩阵，检查正 info 及不影响其他 batch 的行为，不把不存在的逆矩阵代入精度判据。零矩阵或其他导致归一化分母为零的情况单独验证精确重构与 info，避免 NaN 被比较逻辑误判通过。

### 性能标准与测试方法

性能基线取各接口附件 `bench_result.json` 中与用例完整参数对应的 `perf.avg_ms`。在相同 shape、FLOAT32、列主序及可比预热/计时条件下，逐用例要求 `T_NPU <= T_GPU/0.35`，不能用平均加速比掩盖单个失败规格。getrf 的查询不计入 kernel 耗时；getri 仅计从已分解 LU 求逆的阶段，不把 getrf 混入。批量 getri 的预先分解同样属于准备阶段，matinv 则须计入其自身的分解与求逆全过程。

使用官方 `msopprof`（目标配套版本的 `msprof op` 入口）对测试二进制采集，单次采集收敛到一个用例，根据 `OpBasicInfo.csv` 的 `Task Duration(us)` 关联实际调用。若一次接口调用包含参数校验、初始化、布局变换、面板、更新和求解等多个 kernel，按一次调用汇总其必要 kernel，再对完整迭代求平均；不只选择最快 kernel 或某一个阶段。GPU 的毫秒值换算为微秒后再比对。Host 调度、内部分配与必要同步另行报告，防止将端到端时间与 kernel 时间混为一谈。

覆盖附件 getrf 148 条、getri 200 条、getriBatched 149 条、matinvBatched 215 条用例，并补齐任务书 P-01 至 P-09 的典型规格、合法 padding 与接口边界。报告记录 CANN、950PR 型号、驱动以及 CUDA/cuSolver/cuBLAS 版本和计时口径；没有有效基线的用例保持缺失状态，不能填零、编造数据或计为性能通过。

### 附件差异及评审确认项

以下差异依据 2026-10-15 核对的原始附件，不修改任务书或删减官方用例。

1. 接口范围需确认：任务书第 2.3 节定义四个计算接口和两个查询接口，广场简介与“五个接口”文案中的 `Sgetrs` 缺少正式定义。
2. getrf 有 49 条附件用例的 n 超过 4096，最大为 8168；matinv 有 44 条用例的 batchSize 超过 1000000，最大为 10000000。请确认附件全部规格是否作为本次验收范围；本设计保留对这些用例的验证要求，不以规模表为由过滤。
3. matinv 的 `n=2,batch=10000000` 与 `n=4,batch=10000000` 两条 GPU 记录为 `perf:null,error:2`。任务书 P-03、P-06、P-07、P-08、P-09 的完整规格也没有对应的附件 GPU 记录，需要补充有效基线或明确验收方式。
4. getri 的 GPU 结果标签为 `cusolverDnSgetri`，任务书则以分解后求解单位阵作为对标。请确认已有基线确实排除了 getrf；设计与自验始终保持“已有 LU 后求逆”的接口计时范围。

### 参考资料

1. [任务书及测试附件](https://www.hiascend.com/p/resource/202610/9f2b9148f987448eaea2dff10e16b60d.zip)与[任务讨论帖](https://gitcode.com/cann/ops-solver/discussions/2)定义本任务要求与补充说明。
2. [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)与[社区任务流程](https://gitcode.com/org/cann/discussions/39)定义设计提交和评审流程。
3. [cuSolver DN 文档](https://docs.nvidia.com/cuda/cusolver/index.html)提供 getrf 的接口与语义对标。
4. [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getribatched)提供 getriBatched 与 matinvBatched 的接口及约束对标。
5. [LAPACK SGETRI](https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html)、任务书引用的 SGET01/SGET03 以及 [ops-solver](https://gitcode.com/cann/ops-solver)提供求逆、残差和工程接入依据。
