# 单精度实数 LU 分解、求解和批量求逆（Atlas 950）设计文档

> 作者：PzeYui；日期：2026-10-14。本文为拟实施设计，尚未完成算子实现、编译或 950PR 实测；文中规模、精度及性能数字为任务要求或调优候选，不代表已达标。
>
> 按 2026 社区任务官方设计模板编写。需求依据为随任务包提供的 `Atlas950_Sgetrf_Sgetri_SgetriBatched_SmatinvBatched_task_doc.md`；源码调研基线为 `cann/ops-solver` 的 `0bf7eebb6a400ca33c135867eae5ce080ff1b13d`。

# 需求背景（required）

## 需求来源

10 月社区任务第 1 项“单精度实数 LU 分解、求解和批量求逆（950）”要求在 Ascend 950PR（Atlas 950）上，使用 ops-solver Host C API + AscendC/CATLASS Kernel 直调模式交付四个 FLOAT32 计算接口及两个工作空间查询接口。最终代码仓为 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，本任务代码 fork 为 [PzeYui/ops-solver](https://gitcode.com/PzeYui/ops-solver)。本次 PR 提交设计文档，整组计算能力在实现和验收阶段一并交付。

## 背景介绍

### 现有实现与差距

以下发现均来自上述固定源码基线，可通过对应文件及符号复核。

| 文件或符号 | 当前实现 | 本任务设计变更 |
| --- | --- | --- |
| [`include/cann_ops_solver.h`](https://gitcode.com/cann/ops-solver/blob/0bf7eebb6a400ca33c135867eae5ce080ff1b13d/include/cann_ops_solver.h) | Sgetrf/Sgetri 返回 `aclError`，维数为 `int64_t`；缺少 bufferSize 和实数批量接口 | 按任务书更新为 `aclsolverStatus_t`、32 位 `int`，增加两个查询和两个批量接口 |
| [`src/sgetrf/sgetrf_host.cpp`](https://gitcode.com/cann/ops-solver/blob/0bf7eebb6a400ca33c135867eae5ce080ff1b13d/src/sgetrf/sgetrf_host.cpp)：`aclsolverSgetrf` | Host 读写 A/ipiv/info，内部 H2D/D2H，要求 `lda==n`，计算后同步 stream | Device 列主序输入输出、合法 padding、调用方 workspace、异步下发及真实 Device info |
| [`src/sgetri/sgetri_host.cpp`](https://gitcode.com/cann/ops-solver/blob/0bf7eebb6a400ca33c135867eae5ce080ff1b13d/src/sgetri/sgetri_host.cpp)：`aclsolverSgetri` | 没有主元输入，内部申请矩阵和单位阵缓冲，Host 往返及同步 | 从指定 LU 因子与主元求逆，查询工作空间，保存 LU 后原地写逆矩阵 |
| [`src/sgetri/sgetri_kernel.cpp`](https://gitcode.com/cann/ops-solver/blob/0bf7eebb6a400ca33c135867eae5ce080ff1b13d/src/sgetri/sgetri_kernel.cpp)：`sgetri_kernel` | 调用 `custom_lu` 后进行三角求解，存在重新分解流程 | 删除求逆路径的重新分解，仅执行已有 LU 上的三角求解 |
| `src/utils/lu_host_common.h`、`src/sgetrf/kernel/getf2.hpp` | 共享 LU 常量及面板处理；固定 20 块、16 列面板、512 行 tile，并有 C220 架构分支 | 参考算法与测试框架；950PR 单独规划资源及同步，不照搬固定分核 |
| `src/utils/aclsolver_handle_internal.h`、`aclsolver_auxiliary.cpp` | handle 只保存 stream | 保留公开 handle API；批量内部 scratch 需要新增不透明的池和完成事件管理 |
| `CMakeLists.txt`、`src/CMakeLists.txt` | `ascend950` 对应 `dav-3510`，支持 `arch35` 源码收集 | 增加 950 路径及批量目录，防止重复定义公开符号；编译支持不能替代真机验证 |

现有复数 `cgetri_batched` / `cmatinv_batched` 可参考 Kernel 下发和工程组织，但其连续张量、Host 指针、维数与 info 语义不能作为本任务实数接口的最终约定。已有公共状态码、创建/销毁 handle 和 stream 管理接口复用。

# 需求分析（required）

## 需求描述

公开函数原型严格按任务书 §2.3，不改变参数顺序、类型及内存归属：

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

`cann_ops_solver_common.h` 增加 `aclsolverOperation_t`：`ACLSOLVER_OP_N=0`、`ACLSOLVER_OP_T=1`、`ACLSOLVER_OP_C=2`。四个计算函数本身没有 trans 参数，不为此擅增接口。

| 对象 | 类型及内存归属 | 布局与语义 |
| --- | --- | --- |
| handle、维数、leading dimension | Host 句柄及 `int` 标量 | 本次调用动态生成分派参数 |
| A、C、Ainv 所指矩阵 | Device，FLOAT32 | 列主序，有效地址 `base + row + col*ld`，ld 至少为有效行数与 1 的较大值 |
| devIpiv、PivotArray | Device，INT32 | 1-based 主元；批量第 i 个序列起点为 `i*n`；nullptr 表示未选主元 |
| devInfo、infoArray、info | Device，INT32 | 0 成功，负值非法参数，正值首个零主元位置；批量逐矩阵独立写回 |
| Aarray/Carray/A/Ainv 的数组本身 | Device，地址数组 | 每个元素为独立 Device 矩阵地址，不能假设矩阵连续 |
| Workspace | Device，`float*` | 查询后由调用方提供，长度以 float 元素计 |
| bufferSize 的 `Lwork` 输出 | Host，`int*` | 仅查询所需长度，不读矩阵数据、不启动计算 Kernel |

## 需求拆解

| 编号 | 要求及任务书依据 | 设计落点 | 验证方法 |
| --- | --- | --- | --- |
| R1 | 四个计算接口、两个查询接口、状态码及枚举；§2.2–2.3 | 公开头文件与导出、Host wrapper | 签名、链接和枚举检查 |
| R2 | Device 列主序、合法 padding、Device 地址数组；§2.1、§2.4 | stride 地址计算、搬运和数组解引用 | 非连续矩阵、ld+8/+32、padding 哨兵 |
| R3 | 矩形 LU、有/无主元、1-based ipiv；§2.1 | GETF2 + TRSM + GEMM、确定性选主元 | CPU FLOAT32 LAPACK、重放交换、相等主元用例 |
| R4 | Sgetri 消费已有 LU，原地写逆；§2.1、§7 | LU 保存及分块多右端项求解 | LU 缓冲不被错误重新分解、主元链和残差 |
| R5 | 批量 LU 求逆与 n≤32 直接求逆；§2.1、§2.4 | 两条独立路径、批内错误隔离 | 原始 A/已分解 LU 区分，n=32/33、输入不变 |
| R6 | workspace 查询与边界；§2.1、§2.4 | 与计算同源的 checked planner | 精确查询值、少一个元素、空 workspace、整数溢出 |
| R7 | info、非法参数、奇异、空问题；§2.4、§3.5 | Host 标量检查、Device 内容检查、确定性 info | 正负 info、首个零对角、batch=0、m/n=0 |
| R8 | bit-wise 确定性、非默认 stream；§2.1、§3.5 | 固定归约次序、唯一 tile 写者、事件式资源寿命 | 同输入串行重复、不同 stream 独立调用、延迟同步 |
| R9 | 950PR、配套 CANN≥9.0.0；§3.1 | arch35、平台资源查询、完整 FP32 算法 | Release 真机验证，记录工具链、驱动和设备 |
| R10 | LAPACK 残差、逐 case 性能≥0.35×GPU；§3.2–3.3 | FP32 计算、逐矩阵精度、完整调用计时 | 全量 JSON 关联、CPU ratio、msprof 原始日志 |
| R11 | 设计、测试、文档及可复现报告；§4–5 | 仓内测试和接口 README、交付索引 | 固定源码 commit 对应构建及结果；本稿不声明通过 |

# 详细设计（required）

## 算子分析

### 数学公式和主元方向

LU 分解采用 `P*A=L*U`。L 为单位下三角或梯形，U 为上三角或梯形，打包写回 A：严格下三角为 L，主对角及以上为 U。第 k 步交换 k 行与 `ipiv[k]-1` 行，循环到 `min(m,n)`；无主元时 P=I。

Sgetri 与 SgetriBatched 将已有 LU 和主元视为输入。求解 `A*C=I` 可写为 `L*Y=P`、`U*C=Y`：先对单位矩阵的行按主元**正序**回放，再做前代和回代。用于 LU 残差的重构则对 `L*U` **逆序**回放行交换，得到原 A。两种方向分别测试，避免只用恒等主元掩盖错误。

SmatinvBatched 对原始 A 完成部分主元 LU 和上述两次三角求解，输出独立逆矩阵。SgetriBatched 不做 LU 分解，调用方可循环调用 Sgetrf 构造因子；不强制增加 SgetrfBatched。

### 支持数据类型和形状

仅支持 FLOAT32 矩阵与 INT32 主元/info。正文验收下限：Sgetrf 的 m/n∈[1,4096]，包含矩形；Sgetri 的 n∈[1,4096]；SgetriBatched 的 n∈[1,4096]、batchSize∈[1,1000000]；SmatinvBatched 的 n∈[1,32]、batchSize∈[1,1000000]。合法零维或零 batch 另作空问题处理。

附件 Sgetrf 含 n=8168，SmatinvBatched 含 batchSize=10000000。设计拟将 LU 的列范围扩展到至少 8192，小矩阵批量计数扩展到至少 10000000；通过分块和 64 位寻址避免编译期固定上限。实际可运行组合受 Device 容量约束，不承诺同时驻留所有维数上限的笛卡尔积；自构造用例输入按任务书控制在 4G 内。附件超正文范围的正式验收口径列入评审问题，不自行删用例。

## 算子实现

### 实现方案

```text
Sgetrf: Host 校验/规划 → 初始化 info → 面板选主元/消元
       → TRSM → 尾随 GEMM → 后续面板 → 原地 LU + ipiv/info
Sgetri: Host 校验/规划 → 完整保存 LU → 检查 U 对角与主元
       → 单位阵列块及主元行交换 → 前代 → 回代 → 原地 C
SgetriBatched: Device 地址数组 → 逐矩阵已有 LU → 前代/回代 → Carray/info
SmatinvBatched: n≤32 → Device 地址数组 → 融合 LU/前代/回代 → Ainv/info
```

#### 3.2.1 Host 侧设计

##### 参数校验和返回状态

1. 检查 handle、上下文/设备、维数、leading dimension、所需顶层指针及工作空间。标量参数错误立即返回对应 `aclsolverStatus_t`；可写的 Device info 由调用方 stream 上的短 Kernel 写负参数编号，编号不计 handle。nullptr info 无法写回，仍返回参数错误。
2. 非空矩阵才要求 A、地址数组、输出及所需 workspace 非空。合法空问题返回 SUCCESS；单矩阵 info 写 0，n=0 且 batch>0 时逐矩阵 info 写 0；batch=0 不启动计算、不访问零长数组。具体空问题中的 ld/info 校验顺序按任务书和评审固定，不能让零维绕过非法 handle 或负维数。
3. 数值奇异由 Device 写入首个 `U(k,k)==0` 的 1-based k；API 的 SUCCESS 表示成功下发，不能表示矩阵一定可逆。调用方同步后读取 info。批量仅终止失败矩阵的求逆，其他矩阵继续。
4. 地址数组元素、主元值等 Device 内容在 Kernel 验证；不能通过 Host 解引用或复制整个数组取得。CopyIn 对有限浮点的检查与搬运合并，发现非有限输入按 A 参数位置写负 info，不继续数值计算。需要同步返回 INVALID_VALUE 的 Device 内容错误与异步约定存在冲突，见问题 Q4；不将异步 info 伪装成同步 Host 返回值。
5. 主元需在合法行范围，求逆检查 getrf 第 k 步主元的合法范围；无主元因子由调用方以 nullptr 明确指定。接口没有产生因子的来源标记，无法只凭指针证明“是否来自无主元 getrf”，这属于调用方契约。
6. 全部字节数、ld*n、n*batch 和地址偏移使用 checked 64 位整数，验证乘加和对齐不溢出后再转换为 `size_t`。bufferSize 无法以 `int` 表示的 Lwork 返回错误，不能截断。

##### 工作空间查询和生命周期

两个 bufferSize 与计算使用同一 planner，选择相同算法、面板宽度、RHS 列块及 scratch 布局。Lwork=`ceil(requiredBytes/sizeof(float))`，各区域满足实际搬运/算子 API 对齐要求；必要的起始对齐余量计入查询值。查询不根据 A 数据作算法选择，不修改 A、不分配矩阵缓冲。

| 接口 | 拟规划的 workspace 区域 | 可复用性 |
| --- | --- | --- |
| Sgetrf | 面板暂存 `align(m)*b`、各核候选主元、tile 缓冲、矩阵乘所需 scratch 及必要控制区 | 尾随更新完成后复用面板区域；矩阵主体在调用方 A |
| Sgetri | 一份完整 packed LU `align(n)*align(n)`、一个 RHS 列块 `align(n)*r`、三角求解/矩阵乘 scratch 和控制区 | 所有原 LU 保存完成后才覆盖 A；RHS 逐列块复用 |

Sgetrf 没有计算侧 Lwork 参数，缓冲长度由调用方保证；不能声称可可靠检查任意子分配长度。Sgetri 显式验证 `Lwork>=查询值`。两者都禁止临时隐藏申请完整矩阵来规避查询约定。调用方须保证 A、ipiv、info 和 workspace 活到 stream 完成；workspace 与矩阵/输出不得重叠。

批量接口没有公开 workspace 参数，拟在不透明 handle 内引入按设备/context、stream、容量管理的内部 scratch 池。借用时检查完成事件：同 stream 的复用依靠入队顺序，跨 stream 仅复用已完成资源；未完成则租借独立块。批量大矩阵使用有界 Bchunk，scratch 上界由 Bchunk 和 tile 尺寸决定，不随 batch 总数线性保存全部单位阵。分配失败返回 ALLOC_FAILED。

资源在最后一个使用 Kernel 后记录事件；Host 函数返回前不 free 在途缓冲、不为回收调用 `aclrtSynchronizeStream`。查询完成事件后回收，handle 销毁只等待自身在途资源的必要完成条件，不销毁调用方 stream。SetStream 切换不将旧流缓冲提前复用。Host tiling/control 数据按值传入，或置于与事件同寿命的槽中，避免异步拷贝仍读取已失效的栈内存。池元数据受 Host 锁保护，不用锁序或核竞争决定数值顺序。

##### 分核、分块与分派

按 950PR 平台信息查询 AIV/AIC 数和 UB/L1 等资源。b=16/32/64、RHS 列块 r=16/32/64 为候选值；实际值由 UB 预算与真机数据固定，不能视为已验证配置。

| 路径 | 分核策略 | 分块约束 |
| --- | --- | --- |
| 小 LU/单矩阵逆 | AIV 处理面板或 RHS 块，降低 launch/同步开销 | 小矩阵可在 UB 内完成；尾块 mask |
| 中大 LU | 面板候选搜索按行段分核；TRSM/GEMM 按输出 tile 分核 | 一个 C tile 一个写者，K 方向不做原子 split-K |
| n≤32 批量求逆 | 按矩阵编号均匀分核，每核循环处理多个矩阵或小组 | q 由双缓冲与 `2*n*n` 等实际存活量确定 |
| 较大 SgetriBatched | 以矩阵/RHS 列块组合为任务，在有界 batch chunk 内调度 | n 接近上限时增加单矩阵 tile 并行；保留 Device 指针数组语义 |

Tiling 包含 opKind、m/n、ldA/ldC、batchStart/count、b/r、有效尾长、主元开关、core 数及 workspace 区域偏移。字节 offset 为 64 位。分派由运行时形状和平台决定，不按官方 case ID 特判。SmatinvBatched 的 n>32 在 Host 直接返回 INVALID_VALUE，不转调 SgetriBatched。

#### 3.2.2 Kernel 侧设计

##### LU 面板与尾随更新

每个面板包含 b 个列。第 k 列在剩余全部有效行中选绝对值最大主元，绝对值相同选原矩阵行号最小者；各核先做固定次序局部归约，再由确定的协调者按核编号合并。不能用原子竞争选“第一个到达”的结果。

选定主元后交换对应**整行**，包括已写 L 的历史列和未更新的尾随列，记录 1-based ipiv，再完成面板内消元。无主元分支跳过搜索和交换。零主元时只记录最早失败位置，并跳过除零，仍按固定规则推进分解及 ipiv 写回。

面板完成后计算 `U12=L11^{-1}*A12`，再更新 `A22←A22−L21*U12`。矩形的面板数量由 min(m,n) 决定，边界以有效 m/n 屏蔽，不能把对齐补零参与主元选择。跨面板顺序保持算法依赖。

AIV 负责主元搜索、交换与面板消元；AIC/CATLASS 或 AscendC Matmul 负责满足精度要求的 GEMM。Cube 模式是否支持所需 FP32 乘法/累加及结果精度在 950PR 上先验证，禁止静默使用 FP16/BF16/低有效尾数模式代替 FLOAT32。保留完整 FP32 AIV 实现作为正确性基线，不作为 CPU fallback。

##### 已有 LU 求逆

Sgetri 首先将全部有效 LU 复制到 workspace，完成该阶段后才允许写 A。检查 U 对角线并验证主元，不重新分解。

按 RHS 列块生成单位阵，正序应用主元行交换，执行单位下三角 L 的前代和 U 的回代。三角对角块由 AIV 求解，块间贡献用固定 K 次序 GEMM 更新。每个 RHS 列块依赖同一只读 LU，结果写回 A 的对应列；padding 不写。非空主元可能跨 RHS 列块涉及行交换，交换作用于完整 n 行的当前 RHS 块，而非只交换块内行。

SgetriBatched 采用同样的求解路径，直接读取各 Aarray[i] 的 LU 和 PivotArray+i*n，写独立 Carray[i]。小 n 可将 LU 和 RHS/结果驻留 UB；大 n 分 RHS 列块，不申请 `batch*n*n` 单位阵。Aarray 输入保持不变。

##### 小矩阵直接批量求逆

SmatinvBatched 的每个矩阵独立完成 CopyIn、部分主元 LU、单位阵置换及前代/回代、CopyOut。n≤32 时拟融合进一个 AIV Kernel，通过每核多矩阵循环摊薄启动和地址数组开销；仅在 UB 预算足够时增加 q。

奇异矩阵写首个零主元位置，不输出伪造逆矩阵；失败矩阵输出不作可用性保证。正常矩阵继续执行。尾 batch、n 非对齐值和带 padding 的任意 Device 地址逐矩阵处理，不能假设批内矩阵连续。

##### 搬运、同步及确定性

CopyIn 按列主序真实 ld 装载有效元素，可仅在内部 tile 改为适合计算的布局；CopyOut 只写有效行列，padding 以哨兵验证。大 tile 使用双缓冲，UB 预算计入输入、输出、panel/RHS、归约与中间值，不能只按矩阵大小估算。

MTE2→V、V→MTE3、MTE3→后继读取均显式建立事件依赖；AIV/AIC 协同按 950 文档支持的机制实现。初版依靠同 stream 的阶段 Kernel 边界完成必要跨核依赖，后续只在验证正确后融合。禁止将仅部分核参加的全局 barrier、在途缓存标志或旧 C220 的固定 8+12 分工直接用于 950。

每个输出 tile 由一个任务完整写回，K 块和三角依赖以固定顺序累加，不使用非确定性原子浮点加法。主元候选有固定 tie-break；info 由唯一负责人按最小失败位置写回。重复调用时所有控制/临时区初始化，不依赖前次调用残留。

## 支持硬件

| 芯片 | 本任务范围 | 工具链要求 |
| --- | --- | --- |
| Ascend 950PR（Atlas 950） | 必须完成真实功能、精度与性能验证 | CANN≥9.0.0，匹配 ops-solver 已验证版本；`ascend950` / `dav-3510` / `arch35` |

基线 README 列出 CANN 9.0.0 的已验证软件包，实施时重新核对具体包、驱动、编译器及 CATLASS 配套。A2/A3 的已有运行结果不能替代本任务 950PR 结果。

## 算子约束限制与待评审问题

不支持 broadcast，不要求图融合。Sgetri 输入已分解 LU；Smatinv 输入原始矩阵。批量输出与输入矩阵不重叠，输出矩阵之间也应互不重叠；对有效地址数组逐矩阵检查本矩阵输入/输出范围相交。任意跨矩阵别名的检查范围需统一约定，不承诺只做同下标检查就能检测所有全局别名。

| 问题 | 资料中的差异或接口限制 | 本稿处理与需确认事项 |
| --- | --- | --- |
| Q1 | “五个接口”文字与 §2.3 的四个计算、两个查询不一致，测试表还出现 B/nrhs/trans | 本稿覆盖全部六个已定义函数及枚举；确认是否还有独立 getrs，未定义前不增接口 |
| Q2 | 正文支持下限 4096/100万，附件达到 n=8168/batch=1000万 | 拟扩展实现容量，保留全部附件；请确认正式验收覆盖 |
| Q3 | Smatinv 两条 GPU 基准 `perf=null,error=2` | n=2、4，batch=10000000；保留记录，请补基准或明确性能判定，不能填 0 或判 PASS |
| Q4 | Device 内部指针/主元非法要求返回 INVALID_VALUE，同时要求避免无必要 Host 同步 | Host 可见错误同步返回；Device 内容错误异步写负 info。请确认严格返回码是否需限定错误检查同步，不能同时虚称完全异步与即时获知 Device 结果 |
| Q5 | 未给全矩阵数据 provenance、workspace 分配长度、跨 batch 别名元数据；原始 C 指针不携带 dtype/shape 标签 | 由调用方保证 FLOAT32/INT32、矩阵有效存储、无主元约定及实际缓冲容量；明确可检测条件、跨矩阵重叠的检查和错误归属 |
| Q6 | 按 kernel 名求平均未定义多阶段调用的汇总口径 | 保存每个阶段明细及一次 API 的完整序列累计耗时，拟按完整序列逐次合计后平均；不能挑最快阶段代表整次操作，需评审固定口径 |

# 可维可测分析

## 精度标准/性能标准

### 精度

矩阵 1-范数取最大列绝对值和，FLOAT32 单位舍入使用任务书的 `ε=2^-24`，不能直接用 `np.finfo(float32).eps=2^-23` 代替。

```text
LU:  ratio = ||P^{-1}(L*U)-A_original||_1 / (n*||A_original||_1*ε)
逆:  ratio = ||I-C*A_ref||_1 / (n*||A_ref||_1*||C||_1*ε)
```

Sgetri/SgetriBatched 的 A_ref 按任务书由所输入的 LU 和主元还原，不能把 packed LU 当原矩阵；Smatinv 的 A_ref 是原始输入。LU 矩形分母 n 取列数。残差计算统一计算精度和范数口径，CPU ratio 来自同用例 FLOAT32 LAPACK 链路；可额外用 float64 做诊断，不能用它替换所要求的 FLOAT32 ratio_cpu。

| 接口 | 判定阈值 | 批量规则 |
| --- | --- | --- |
| Sgetrf | `ratio≤max(5*ratio_cpu,0.3)` | 单矩阵 |
| Sgetri | `ratio≤max(5*ratio_cpu,0.05)` | 单矩阵 |
| SgetriBatched | `ratio_i≤max(5*ratio_cpu_i,0.05)` | 每个矩阵都通过才算用例通过 |
| SmatinvBatched | `ratio_i≤max(5*ratio_cpu_i,0.5)` | 每个矩阵都通过才算用例通过 |

奇异及非法参数按状态/主元/info 测试，不强行套用可逆矩阵残差。超阈值保留失败及误差分析，不自行改阈值。

### 性能

同 shape、FLOAT32、列主序及预热/计时条件，逐 case 要求 `T_NPU≤T_GPU/0.35`。GPU 使用附件 `bench_result.json` 的 `perf.avg_ms`；msprof `Task Duration(us)` 比较前统一单位。Sgetri GPU 标杆为已有 LU 后求解 AX=I，不包含 getrf；SgetriBatched 也不将前置 getrf 算入求逆计时。

| 附件目录 | 用例文件 | 用例数 | 有效平均耗时基准 |
| --- | --- | --- | --- |
| sgetrf | cases.json | 148 | 148 |
| sgetri | cases_brief.json | 200 | 200 |
| sgetribatched | cases.json | 149 | 149 |
| smatinvbatched | cases.json | 215 | 213，另 2 条失败 |

共 712 条附件记录；不把 GPU 基准的 failed=0 视为 NPU 测试通过，不以局部 P-01～P-09 替代全量。case 与 benchmark 按参数键和重复记录的出现次序关联，若有重复键不覆盖。计时保持每次迭代输入状态一致：LU 恢复原矩阵，getri 恢复 LU，恢复及 Host/Device 传输不混入 kernel 计时。计时窗口包含一次操作必需的初始化、校验、布局转换、主元及全部计算 Kernel，保存分阶段明细和 Q6 口径。

优化优先级为小矩阵融合/批内循环、固定主元归约降低同步、panel 宽度与 RHS 块调优、尾随 GEMM 和搬运重叠。对近门槛用例重复采集，记录空闲设备与波动；所有方案以全量精度先通过为前提。任务 §3.4 不设内存门槛，但记录 workspace、内部 scratch 峰值及资源回收情况。

### 测试与复现计划

复用 `test/sgetrf`、`test/sgetri`，新增 `test/sgetri_batched`、`test/smatinv_batched`；新增用例读取和报告程序，原始 JSON 保留不修改。现有测试为旧 Host API，需迁移为显式分配 Device A/ipiv/info/workspace、创建 Device 指针数组、调用、同步再读回的流程。

1. 正向覆盖方阵、长/宽矩形、1/2 的幂及幂减一、非对齐尺寸、ld 最小值/+8/+32、有/无主元、batch=1/8/128/1024/3000 和附件大 batch。
2. 负向覆盖空 handle、负维数/batch、非法 ld、空必需指针、workspace 缺失/不足、非法主元、输入输出同址及部分相交、matinv n=33。Device 内容与 Host 返回码按 Q4 分开验证。
3. 数值覆盖 50% 均匀 [-5,5]、50% 指定均值/方差的正态，对精度主路径加对角占优；另构造不加对角占优、需要多次行交换、相等最大主元、奇异及近奇异矩阵。矩形按可适用方式构造，不把“对角+n”当所有随机矩阵可逆的证明。
4. 用哨兵检查 padding、边界及输入只读区；batch 混合成功/奇异；相同输入同 stream 串行多次逐位比较矩阵、ipiv 和 info；非默认 stream、延迟同步、SetStream 切换及 handle 销毁验证 scratch 寿命。
5. 在 950PR、Release 下运行全量精度及性能，输出逐 case 参数、CPU/NPU ratio、info、GPU/NPU 耗时、倍率、判定和原始 profiler 路径。记录代码 commit、CANN/驱动、NPU 型号及 CUDA/cuSolver/cuBLAS 版本。

基线支持 `bash build.sh --soc=ascend950 --ops=sgetrf --run` 类入口，实际新增测试的 CLI 参数在实现阶段写入 README，并按修改后的脚本验证。性能采用 `msprof op --application="<单用例测试命令>" --output=<用例输出目录>`，保留 `OPPROF_*/OpBasicInfo.csv`；本设计不把尚未编写的命令或脚本称为已可运行。

## 兼容性分析

Sgetrf/Sgetri 的旧原型与新原型参数个数和类型不同，属于公开 API/ABI 变更。迁移其声明、定义、示例、测试、docs/zh 及接口列表，说明 Device 内存和 workspace 的新调用方式；同名 C 符号不能依靠重载同时保留两套 ABI。是否保留独立命名的旧接口需仓库维护者确认，本稿不额外承诺兼容封装。

新增批量目录按实际 `src/CMakeLists.txt` 源码收集机制接入；测试补充对应 CMakeLists，避免沿用不存在的算子级 src CMake 文件。公共改动限于状态/枚举、公开声明、handle scratch 管理和必要构建接入；复数接口及其他算子数值逻辑不改。核心算法以本任务专用组件实现，避免为了新接口全量复制公共 Kernel。

## 参考资料与流程记录

- [2026 社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)和仓内提交指南：2026-10-14 从官方 master 读取，设计文档提交到既有 `10-1-aclsolverSgetrf-batch-950/PzeYui/docs/design.md`。
- [社区任务流程及注意事项](https://gitcode.com/org/cann/discussions/39)：2026-10-14 读取动态正文，设计以 PR 评审，验收前须设计审核通过；设计提交不表示评审通过或合入。
- [活动页面](https://www.hiascend.com/developer/activities/cann-community-task)：用于核对流程；不在本 PR 代替开发者报名或登记验收。
- [NVIDIA cuSolver getrf](https://docs.nvidia.com/cuda/cusolver/index.html)、[cuBLAS 批量求逆](https://docs.nvidia.com/cuda/cublas/index.html)、[LAPACK getri](https://netlib.org/lapack/explore-html/d8/ddc/group__getri.html)：按任务书所列接口与功能对标；NPU 实现不依赖 CUDA。
- 原型及构建依据均定位于本文记录的 ops-solver 基线。上述算法、容量扩展和优化尚待实现及真机测试，本设计不构成性能或验收通过证据。
