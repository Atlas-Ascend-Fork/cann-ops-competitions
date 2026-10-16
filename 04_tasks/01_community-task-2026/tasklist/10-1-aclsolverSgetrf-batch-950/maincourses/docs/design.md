# Atlas 950 单精度实数 LU 分解及求逆接口设计

作者：maincourses

日期：2026-10-16

本文给出 Host C API、Device 计算路径、资源生命周期和自验方法。所列分块参数及优化收益均为设计候选，不作为已通过全量测试或官方验收的证明；实际结果由固定代码版本对应的自测报告提供。

# 需求背景（required）

## 需求来源

本设计对应 10 月社区任务“单精度实数 LU 分解、求解和批量求逆（950）”。需求以任务包 `Atlas950_Sgetrf_Sgetri_SgetriBatched_SmatinvBatched_task_doc.md` 为依据，面向 Ascend 950PR，通过 [cann/ops-solver](https://gitcode.com/cann/ops-solver) 的 Host C API 和 AscendC/CATLASS Kernel 交付。本文按社区任务设计模板组织，四个计算接口与两个 bufferSize 查询作为同一组能力设计。

## 背景介绍

LU 分解与求逆共享三角因子、主元及部分搬运逻辑，但输入契约不同：Sgetrf 消费原矩阵，Sgetri 和 SgetriBatched 消费已有 LU，SmatinvBatched 消费小规模原矩阵。设计首先区分这些契约，再复用数值组件，避免将“共用 Kernel”误解为“每个求逆接口都重新分解”。

源码调研固定在官方 ops-solver 提交 `0bf7eebb6a400ca33c135867eae5ce080ff1b13d`，以下是该基线的差距，不能据此推断后续 master 的状态。

| 调研位置 | 基线行为 | 本设计的处理 |
| --- | --- | --- |
| `include/cann_ops_solver.h` | 现有 Sgetrf/Sgetri 原型与任务书不同，缺少实数批量及 workspace 查询 | 统一声明、定义、导出和调用示例 |
| `src/sgetrf/sgetrf_host.cpp` | Host 矩阵指针、内部数据往返、`lda==n` 和 stream 同步 | 改用 Device 数据、真实列跨度及异步下发 |
| `src/sgetri/sgetri_host.cpp`、`sgetri_kernel.cpp` | 求逆流程包含重新分解，缺少任务书规定的主元输入 | 直接读取 LU，按主元生成右端项并三角求解 |
| `src/utils/lu_host_common.h`、`src/sgetrf/kernel/getf2.hpp` | 有既有面板处理和固定资源配置 | 复核数学逻辑，950 的分核及存储预算单独规划 |
| `src/utils/aclsolver_handle_internal.h`、辅助接口 | handle 保存 stream | 在不暴露内部结构的前提下管理批量临时资源 |
| `CMakeLists.txt`、`src/CMakeLists.txt` | 存在 `ascend950`、`dav-3510`、arch35 接入 | 补充实数批量目录、测试与文档，避免同名符号重复定义 |

现有复数批量接口仅用于参考工程组织和下发方式，不能直接继承其矩阵地址、主元和 info 约定。

# 需求分析（required）

## 需求描述

公开原型保持任务书 §2.3 的参数类型、顺序和内存归属：

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

公共头文件补充 `aclsolverOperation_t`，N/T/C 分别为 0/1/2。上述六个函数没有 trans 或独立 B 参数，不额外推导 getrs 接口。

| 数据 | 所在位置 | 必须保持的契约 |
| --- | --- | --- |
| handle、m/n、batch、ld、Lwork 输入值 | Host | 维数使用 32 位 `int`，内部地址计算另用 checked 64 位整数 |
| A、C、Ainv、Workspace | Device | FLOAT32；矩阵地址为 `base[row+col*ld]`，只访问有效元素 |
| ipiv、info | Device | INT32；主元为 1-based；批量主元起点为 `i*n` |
| 批量矩阵地址数组本身 | Device | 元素可指向分散分配的矩阵，不能当连续三维张量 |
| bufferSize 的 Lwork 输出 | Host | 以 float 元素数返回，不启动数值计算，不读取 A 的值 |

Sgetrf 的有效形状为 m×n，`lda>=max(1,m)`；其余三个计算接口处理 n×n，输入、输出各自满足最小合法 ld。Sgetri 原地写回；两个批量接口保持输入只读，输出与输入不得重叠。

## 需求拆解

| 编号 | 要求与依据 | 实现落点 | 自验关注点 |
| --- | --- | --- | --- |
| R1 | 六个函数、状态码、枚举；§2.2–2.3 | 公共声明与 Host wrapper | C/C++ 编译、链接、签名及导出 |
| R2 | Device 列主序及 padding；§2.1、2.4 | 带 ld 的 CopyIn/CopyOut | 非连续矩阵、ld+8/+32、边界哨兵 |
| R3 | 矩形 LU、有/无主元；§2.1 | 面板分解、整行交换、尾随更新 | 长宽矩形、主元重放、相等候选 |
| R4 | 已有 LU 求逆；§2.1、7 | 只读 LU 与分块 RHS 求解 | 不重新分解、原地覆盖顺序 |
| R5 | 批量逐矩阵语义；§2.4 | 地址数组解引用和矩阵任务划分 | mixed info、尾 batch、n=32/33 |
| R6 | workspace 查询；§2.4 | 查询和执行共用 planner | 精确长度、少一个元素、溢出 |
| R7 | 数值与参数错误；§2.4、3.5 | Host 校验、Device 检查、info 初始化 | 首个零主元、负 info、合法空问题 |
| R8 | 确定性和 stream；§2.1、3.5 | 固定归约、事件式资源复用 | 重复逐位比较、跨流、INFO 复用 |
| R9 | 950PR 与配套 CANN；§3.1 | arch35 资源规划及构建 | Release 真机和环境元数据 |
| R10 | LAPACK 精度及逐用例性能；§3.2–3.3 | 固定 CPU 参考与完整调用采集 | 712 精度、710 有效官方性能基准 |
| R11 | 可复现交付；§4–5 | 用例、报告、命令及原始证据索引 | 代码版本、二进制、输入与结果可关联 |

# 详细设计（required）

## 算子分析

### 数学公式

部分主元 LU 使用 `P*A=L*U`，严格下三角存 L，主对角及以上存 U，L 的单位对角隐含。第 k 步从剩余有效行选主元，交换整行并写 `ipiv[k]=pivotRow+1`；无主元分支令 P=I。

对于已有 LU，求解 `A*C=I` 等价于 `L*Y=P`、`U*C=Y`。先按主元正序交换单位矩阵的行，再前代和回代。精度检查重构原矩阵时，先形成 L*U，再逆序回放相同交换。两种方向分别覆盖测试，不依靠全恒等主元验证。

### 支持数据类型和形状

矩阵为 FLOAT32，主元和 info 为 INT32。正文要求单矩阵及批量 LU 求逆至少覆盖到 n=4096，SmatinvBatched 仅接受 n≤32；正文批量范围到 100 万。附件还包含 Sgetrf n=8168、SmatinvBatched batch=1000 万，因此寻址、循环计数和任务切分按附件实际范围规划，LU 列容量至少覆盖 8192，不因正文下限较小而删除附件。

维数上限不代表任意最大 m、n、batch 组合都能同时驻留 Device。先检查乘法、对齐和容量边界；自构造输入按任务书控制在 4G 内。m/n=0 或 batch=0 的合法空问题单独处理，不能绕过负维数、非法 handle 等必要检查。

## 算子实现

### 实现方案

整体分为契约层、规划层和数值层。契约层统一参数编号、状态码及空问题；规划层只依赖形状、布局和平台资源，生成 workspace 与 Kernel 序列；数值层共享 LU、三角求解和搬运组件。算法分派不依赖官方用例 ID。

```text
Host 标量校验 → 同源 planner → 资源租借 / 调用方 workspace
                             ↓
Sgetrf          原 A → 分块 LU → packed LU / piv / info
Sgetri          原 LU → 保存 LU → 分块三角求解 → 原地逆
SgetriBatched   地址数组中的 LU → 分矩阵求解 → 独立 C / info
SmatinvBatched  地址数组中的原 A → 小矩阵融合分解和求解 → 独立逆 / info
                             ↓
最后一次使用后记录事件 → 可安全复用临时资源
```

#### 3.2.1 Host 侧设计

##### 参数和错误处理

Host 检查 handle、当前设备/context、维数、ld、顶层指针和可知的 workspace 长度，返回对应 `aclsolverStatus_t`。可写的 Device info 在同一 stream 上写负参数编号，编号不计 handle；info 本身为空时仍返回参数错误。非空问题才要求相应数据指针存在。

Device 指针数组中的元素、主元内容和有限值检查属于 Device 内容检查。不能通过 Host 解引用或复制整个 batch 来验证，更不能假称异步 API 已即时获知 Device 检查结果。对此将 Host 返回状态与异步 info 分别记录，严格 INVALID_VALUE 返回要求与同步边界作为待评审问题列出。

每次调用主动初始化本次活跃范围的 info 和控制区，不能依赖调用方预置 0。成功写 0，数值奇异写首个零主元的 1-based 下标；发现零主元后跳过除零。API SUCCESS 表示下发成功，数值结果以同步后 info 为准。批量一项失败不得阻止其他矩阵完成。

主元检查遵循第 k 步的合法行范围；nullptr 表示无主元，不从数据内容猜测因子来源。矩阵有效存储、dtype 和实际分配长度由调用方保证，裸指针不携带这些元数据。输入输出范围使用 checked 64 位乘加与地址区间计算，跨 batch 别名检测边界需在接口文档中明确。

##### workspace 与异步生命周期

查询和执行使用相同 planner，`Lwork=ceil(requiredBytes/sizeof(float))`。内部各段按实际 API 的对齐需求划分，起始对齐余量也计入查询值；结果超出 `int` 时返回错误，不截断。

| 路径 | 拟保留的存储 | 复用或释放时机 |
| --- | --- | --- |
| Sgetrf | 面板暂存、主元候选、tile 与乘法 scratch、控制区 | 面板尾随更新结束后复用临时段 |
| Sgetri | 完整 LU 副本、RHS 列块、三角求解 scratch | 保存全部 LU 后才覆盖 A；RHS 按列块复用 |
| 小矩阵批量 | UB 内 LU、结果、主元和归约向量 | 一个矩阵或矩阵小组完成后复用 |
| 大矩阵批量 | 有界 batch chunk 和 RHS tile 的内部 scratch | 末次使用记录事件后归还池 |

Sgetri 验证 Lwork 不小于查询值。Sgetrf 计算接口没有 Lwork 输入，无法可靠判断任意用户子分配的长度，调用方应按查询值分配；不能新增隐藏完整矩阵缓冲来规避这一约定。

批量接口没有公开 workspace，内部池按设备、context、stream 和容量管理。资源状态依次为“空闲→租借→已下发→事件完成→空闲”；同流复用依靠队列顺序，跨流仅复用已完成块，否则借用独立资源。分配失败返回 ALLOC_FAILED。SetStream 不提前回收旧流在途数据，销毁 handle 仅等待自身资源所需的完成条件，不销毁用户 stream。

Host 函数不为回收临时内存强制同步。tiling/control 按值传入或放到同生命周期槽内，避免异步读取已返回函数的栈内存；池锁仅保护元数据，不决定浮点运算顺序。

##### 分派及资源预算

分派条件包括 op、m/n、batch、ld、是否选主元及平台可用资源。b=16/32/64、RHS 宽度 r=16/32/64 为候选，最终由资源检查和真机数据固定。

| 计算区间 | 设计路径 | 首要瓶颈与处理 |
| --- | --- | --- |
| 小 n、大 batch | AIV 每核循环处理矩阵小组；尽量融合读取、计算与写回 | 降低 launch、指针解引用和中间 GM 往返 |
| 小规模单矩阵 | UB 内分解/求解，减少阶段数 | 启动开销与尾元素屏蔽 |
| 中大 Sgetrf | 面板依赖顺序执行，TRSM 和尾随 GEMM 按输出 tile 分核 | 面板同步、矩形尾块、更新吞吐 |
| 中大已有 LU 求逆 | 只读 LU，按矩阵×RHS 列块划分任务 | LU 复用、三角依赖与输出带宽 |

UB 预算显式计算所有同时存活对象。若每核并行 q 个小矩阵，预算至少包含 `4*q*(LU元素数+结果元素数+向量临时量)`，再加主元、mask、对齐和 API 临时区。小矩阵已有 LU 求逆可以评估紧凑原地三角求解以减少内部复制，但必须先证明覆盖顺序；n=32 和 padded 布局分别核算，不能只根据 n 自动套用同一存储配置。超预算时降低 q 或切换分块路径。

tiling 传递有效维数、ld、batch 起点/数量、b/r、尾长、主元开关和 workspace 偏移；地址偏移用 64 位，尾部不参与主元搜索。SmatinvBatched 的 n>32 直接拒绝，不转调其他接口。

#### 3.2.2 Kernel 侧设计

##### LU 与主元选择

第 k 列在全部剩余有效行中比较绝对值，相等时取行号最小者。各核按固定顺序形成局部候选，再按固定核编号合并，禁止用原子到达顺序选主元。行交换覆盖历史 L 列和尾随列，不能仅交换当前面板。

面板得到 L11/U11 后计算 `U12=L11^{-1}*A12`，更新 `A22=A22-L21*U12`；面板数由 min(m,n) 决定。零主元只记录最早位置，不做非法倒数，后续仍以固定规则写主元和因子。非有限输入在搬运/检查阶段报告，不混入主元比较。

选主元、面板和小三角块由 AIV 处理；尾随 GEMM 可使用满足精度要求的 AIC/CATLASS 或 AscendC Matmul。完整 FP32 路径作为正确性基线，禁止用 CPU fallback 或静默低精度乘法替代。FMA、乘加分离和归约次序会改变病态输入的舍入，因此编译选项及各分支算术序列须固定，不能仅比较普通随机矩阵就认定等价。

##### 已有 LU 的求逆

Sgetri 先完整保存有效 LU，再生成 RHS 列块、正序回放主元、求解单位下三角 L 和上三角 U，将结果写回 A 的对应列。写回不破坏后续仍需读取的 LU。主元交换作用于当前 RHS 块的全部 n 行，不限制在对角块内。

SgetriBatched 读取各 Aarray[i] 的 LU 与 PivotArray+i*n，写 Carray[i]，全程不重新分解、不中途覆盖输入。较大矩阵使用有界 chunk 和列块，不申请 batch*n*n 的完整单位阵集合。每个结果 tile 只有一个写者，三角依赖和 K 块更新保持固定顺序。

##### 小矩阵直接求逆

SmatinvBatched 将原矩阵读取、部分主元 LU、单位阵置换、前代、回代和输出融合为小矩阵任务。每核连续处理分配到的矩阵或矩阵小组；n 非对齐、尾 batch 和任意合法 ld 均使用真实有效范围。奇异矩阵保留 info，不将其输出称为可用逆矩阵；正常矩阵继续执行。

对小 n 优先降低冗余矩阵搬运，而非扩大中间结果。接近零或极小非零主元的处理应避免无保护的倒数溢出；实际除法策略与 FP32 路径通过 SFMIN、subnormal、近奇异和不同缩放用例验证，不把非零主元人为改为 0。

##### 搬运与同步

外部始终按列主序和实际 ld 搬运，内部可转换为适合向量/矩阵指令的 tile；CopyOut 仅写有效行。MTE2、计算和 MTE3 之间建立显式事件依赖。双缓冲预算计入输入、输出、归约和算子临时量。

初版使用同一 stream 的阶段 Kernel 边界表达必要全局依赖；只有验证无死锁后才融合跨核同步。950 的 AIV/AIC 协同按配套文档实现，不沿用其他架构的固定核分组。不使用浮点原子 split-K，info 由唯一负责人按最小失败位置写回，重复调用前重置活跃控制区。

## 支持硬件

| 平台 | 范围 | 配套要求 |
| --- | --- | --- |
| Ascend 950PR（Atlas 950） | 四个计算接口和两个查询接口 | CANN≥9.0.0，并匹配 ops-solver README 已验证版本；`ascend950` / `dav-3510` / arch35 |

报告记录真实 NPU、驱动、CANN、编译器和 CATLASS 版本。编译接入和其他芯片运行结果均不能代替 950PR 真机验证。

## 算子约束限制

不支持 broadcast，不引入图模式或 PyTorch 依赖。调用方保证矩阵、主元、info、workspace 在 stream 完成前有效，workspace 不得与矩阵和输出重叠。批量地址数组及其矩阵在调用期间保持有效；跨矩阵全局别名检测是否要求完全覆盖，需要明确契约后实现。

任务书存在以下需要评审统一的边界：

| 项目 | 本设计的处理与待确认事项 |
| --- | --- |
| 接口数量及 B/nrhs/trans 表项 | 覆盖 §2.3 明确定义的六函数及枚举，不凭表项增加 getrs |
| 正文与附件规模不一致 | 保留 712 条附件，设计覆盖 n=8168 和 batch=1000 万，确认正式范围 |
| 两条 GPU 基准缺失 | 保留 null/error 原值及精度用例，官方性能状态不可自行改成 PASS |
| Device 内容错误与即时返回码 | 区分 Host 标量错误和异步内容错误，确认必要检查同步的边界 |
| 裸指针容量、dtype、跨 batch 别名 | 列明调用方责任和实际能检测的条件，不承诺不可获得的元数据 |
| 多阶段耗时统计 | 保留全部计算阶段，按一次 API 的完整 Kernel 序列统计，提交口径供评审确认 |

# 可维可测分析

## 精度标准/性能标准

### 精度判定与 CPU 参考

矩阵 1-范数为最大列绝对值和，ε 固定为 `2^-24`。LU 残差重构采用主元逆序交换；求逆按 `I-C*A_ref` 的乘法方向，不调换左右。

```text
LU: ratio = ||P^{-1}(L*U)-A_original||_1 / (n*||A_original||_1*ε)
逆: ratio = ||I-C*A_ref||_1 / (n*||A_ref||_1*||C||_1*ε)
```

矩形 LU 的 n 为列数。已有 LU 求逆的 A_ref 从同一 LU 和主元还原；直接批量求逆的 A_ref 为原始输入。CPU 与 NPU 使用同一输入和同式残差，不能把 packed LU 当作原 A。

| 接口 | 原任务阈值 | 粒度 |
| --- | --- | --- |
| Sgetrf | ratio≤max(5*ratio_cpu,0.3) | 单矩阵 |
| Sgetri | ratio≤max(5*ratio_cpu,0.05) | 单矩阵 |
| SgetriBatched | ratio_i≤max(5*ratio_cpu_i,0.05) | 每个矩阵 |
| SmatinvBatched | ratio_i≤max(5*ratio_cpu_i,0.5) | 每个矩阵 |

CPU golden 采用 FLOAT32 LAPACK 链路；本方案将 OpenBLAS 0.3.34、Zen 内核和单 BLAS 线程作为固定自验配置，并记录实际加载库、OpenBLAS 配置和运行时 core，而不是只记录 Python 包版本。运行环境必须支持所选指令路径，不允许静默回落后沿用相同配置标记。该版本与内核属于自验选择，任务书没有指定必须采用它们。

残差中的矩阵运算采用 FLOAT32，范数归约明确采用 FLOAT64 累加；CPU/NPU 两侧使用相同检查程序和顺序。额外高精度分析只作诊断，不替换 FLOAT32 ratio_cpu。CPU 出现非零 info 时保留原输入、info 和数值证据，区分数学奇异与有限精度零主元，不重采样或放宽阈值。

### 性能采集和覆盖

官方有效基准逐条应用 `T_NPU≤T_GPU/0.35`，单位统一为 ms。一次 API 包含初始化、必要校验、布局处理、主元及计算的全部 Kernel；按完整序列合计后除以实际 API 调用数，不能挑最快 Kernel 或只统计尾随 GEMM。Sgetri/Batched 的前置 LU 因子生成不计入本接口时间。

| 附件 | 精度用例 | 有效 GPU 性能基准 |
| --- | --- | --- |
| sgetrf | 148 | 148 |
| sgetri | 200 | 200 |
| sgetribatched | 149 | 149 |
| smatinvbatched | 215 | 213 |
| 合计 | 712 | 710 |

两条 SmatinvBatched 缺失 GPU 平均耗时，仍测试精度并保留采集结果。可为 n=2、batch=1000 万和 n=4、batch=1000 万分别设置 NPU<18 ms、<36 ms 的内部估算目标，独立列示；它们不是官方 GPU 基准，也不能填回原 null 或据此宣称官方性能已补齐。

性能使用单用例 `msprof op` application 采集及 `OpBasicInfo.csv` 的 `Task Duration(us)`。记录实际 profiler 版本、兼容性处理、完整命令、Kernel 序列和实际调用数；工具兼容问题与数值失败分别报告。每次 LU 迭代恢复原 A，getri 恢复原 LU，恢复及 Host/Device 传输不混入 Kernel 时间，但算子自身必需 Kernel 不排除。

优化先针对启动/搬运瓶颈，再调整 b/r 和尾随更新；每次共享组件修改都回归其影响的接口和形状。近门槛波动保留全部采集记录，不能择优替换失败结果。任务书不设内存门槛，仍报告 workspace、内部 scratch 峰值和回收行为。

### 测试及复现安排

1. 接口和内存：六函数签名、合法/非法 ld、非连续 Device 地址数组、原地与非原地契约、workspace 边界及溢出；padding 和只读区用哨兵检查。
2. 数值：正文要求的均匀/正态及对角占优主路径，另测不加对角占优、近奇异、精确奇异、相等主元、需要多次交换的矩阵和极小非零主元。主路径生成后发现问题保留原样分析，不以换输入掩盖失败。
3. 状态：成功/奇异混合 batch、所有零主元位置、非法主元、合法零维与 batch=0、matinv n=33；独立验证 Host 返回与同步后 info。
4. 资源：同一 info 只初始化一次后连续调用，batch 缩小再增大、默认/非默认 stream、延迟同步、SetStream 和 handle 销毁；活跃结果及未活跃区都检查。
5. 确定性：同一输入同 stream 多次逐位比较 A/C、ipiv 和 info；核数、tile 所有权与归约顺序固定，跨 stream 仅使用互不重叠资源。
6. 全量：712 精度逐矩阵覆盖，710 官方性能逐用例比较；两条估算目标单独汇总。按参数与重复记录出现次序关联附件，不覆盖重复参数键，不以局部代表用例替代全量。

多设备可分摊固定版本的独立用例，但每张 NPU 同时只运行一个性能采集任务。各机使用相同代码、Release 产物、输入生成规则、CPU 参考、阈值和采集参数；用例编号互斥且并集恰好为全集。失败留原始证据后继续收集，最终统一核对覆盖、进程终态和结果，不能将“执行结束”当作“全部达标”。

报告逐例保留参数、源码提交、构建信息、CPU 配置、输入/seed、info、ratio/ratio_cpu、阈值、GPU/NPU 耗时、采集调用数、原始日志与结果路径。精度、性能、工具执行三类状态分别呈现。测试工程按 ops-solver 实际构建入口接入，README 提供验收人可直接复现的命令。

## 兼容性分析

旧 Sgetrf/Sgetri 与任务书的参数和返回类型不同，属于公开 API/ABI 变更。声明、定义、测试、示例、接口列表和中文文档必须同步迁移；同名 C 符号不能重载两套 ABI，兼容旧接口若有需求须另经维护者确认。

新增实数批量目录沿用仓库源码收集规则，测试接入相应 CMake。公共修改限于必要的声明、枚举、handle 资源和构建逻辑；复数接口及无关算子不更改数值路径。LU 和三角求解组件设明确契约，修改共享组件后回归所有依赖接口，避免只跑改动接口的测试。

## 参考资料

- 任务包中的任务书、api_list 与四组 cases/bench_result；接口和阈值依据任务书 §2.3、§3.2–3.3。
- [官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)及[社区任务目录说明](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)：2026-10-16 从官方提交 `a73e6af5d857fa890570a0c2330b824a836759c1` 核对。
- [ops-solver 调研基线](https://gitcode.com/cann/ops-solver/tree/0bf7eebb6a400ca33c135867eae5ce080ff1b13d)：公开头文件、Sgetrf/Sgetri Host/Kernel、公共 handle 和 CMake。
- [cuSolver](https://docs.nvidia.com/cuda/cusolver/index.html)、[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)、[LAPACK SGETRI](https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html)：接口和数学行为参考，NPU 实现不依赖 CUDA。
- [社区任务流程](https://gitcode.com/org/cann/discussions/39)：设计 PR、评审、合入和验收是不同状态；提交本设计不表示上述后续流程已完成。
