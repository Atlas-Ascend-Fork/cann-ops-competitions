# aclblasChemv 算子设计文档

提交人：`kongbada`；任务：CANN训练营哈工深-aclblasChemv算子开发（950）。

版本：2026-11-02 设计评审稿。报名已通过，950 算力申请中；本次仅完成文档审查，尚未开展 Chemv 代码开发、950 编译或精度/性能实测。下述实现是待评审方案，待确认事项不视为已获官方批准。

# 需求背景（required）

## 需求来源

本需求来源于“CANN训练营哈工深-aclblasChemv算子开发（950）”社区任务。直接依据任务2压缩包中的 `aclblasChemv_Atlas950PR_task_doc.md`、`test_cases/README.md`、`chemv_test.csv` 与 `gpu_baseline.csv`，采用同目录 `design_template.md` 的章节结构。

开发目标为在 Ascend 950PR、CANN 9.1.0 上使用 Ascend C 实现 `aclblasChemv`，补齐 ops-blas 的 complex64 Hermitian 矩阵向量乘能力。设计文档拟提交至 cann-competitions 的对应任务目录；实现拟贡献到 ops-blas。已核实提交路径为 `04_tasks/01_community-task-2026/tasklist/CANN训练营哈工深-aclblasChemv算子开发（950）/kongbada/docs/design.md`。

任务页面：[CANN训练营哈工深-aclblasChemv算子开发（950）](https://www.hiascend.com/activities/task-center/details/00932c7652e946c7a7184aa98339a35a?menu=tasks)，页面标识为`00932c7652e946c7a7184aa98339a35a`（不将其推定为仓库任务编号）。任务平台截图显示截止日期为2026年11月25日，具体截止时刻仍需官方确认。

## 背景介绍

### Chemv 算子功能

计算 `y = alpha * H * x + beta * y`，其中 H 为 n×n Hermitian 矩阵，满足 `H = conjugate(transpose(H))`；alpha、beta、A、x、y 的数据类型均为单精度复数 complex64。

A 使用列主序完整数组保存，但仅指定的上三角或下三角有效。另一三角由共轭关系隐含，禁止依赖该区域的内容。对角线仅使用实部，虚部不参与计算。

### 现有同族实现与复用边界

已核对 ops-blas `blas/symv/arch35/` 目录存在 `csymv_host.cpp`、`csymv_kernel.cpp`、`csymv_tiling_data.h` 以及实数 ssymv 对应文件；`test/symv/` 存在 csymv、ssymv 测试目录。

本次源码核对固定于 ops-blas master 提交 `bab3ec101cfed2f4c4339e2d5e3919ac35f76a3c`（2026-11-02 从官方仓拉取，提交时间 18:48:52 +08:00），公开头文件及源码目录中尚无 `aclblasChemv` 实现/声明。已阅读 csymv Host、Kernel、Tiling 数据、同族测试与构建配置。2026-11-02再次复核master至`3c851fa1c2596d73d92292902bf6190c260a065c`；相对前次复核`9baad78db8f97673b1966709d810f97a9fba818d`的7个变更文件均不在symv、公共头文件、同族测试或构建依据范围内，相关设计依据未变。后续实现前继续复查上游变化。

| 已核对的机制 | Chemv 设计决策 |
| --- | --- |
| aclblasComplex 为 real、imag 两个 float 字段 | 沿用ABI；SIMT访问采用交错实虚布局，逻辑复数偏移转为float偏移时乘2 |
| Host由handle->stream获取stream，GetAivCoreCount获取AIV核数 | 采用同族入口与启动结构；核数获取失败返回EXECUTION_FAILED |
| csymv支持Host/Device标量，并在部分缺失数据指针场景D2H后同步 | alpha按任务表为Host；beta按任务表保留Host/Device能力，Device beta与Host即时校验的冲突见Q1，不能擅自删去Device支持 |
| csymv通用SIMT路径每线程一行，协作路径每warp一行 | 分别作为小规模/语义回退与并行归约参考，加入共轭、实对角和通用步长 |
| csymv大尺寸寄存器分块路径写workspace，再启动归约Kernel | 作为后续性能候选，首版不依赖workspace；其阈值和限制不直接移植 |
| csymv CPU golden是手写对称复数算法 | Chemv主标杆必须接入Netlib cblas_chemv，不能只改名沿用 |

复数对称与Hermitian语义不同：csymv的镜像元素不取共轭、对角虚部参与乘法，Chemv必须分别修正。现有快路径还限制标量位置、步长、beta及核数，不能视为通用实现。参考代码中的64元素tile、n≥384与核数≥40是其自身条件，不是本任务已验证参数。

本任务采用句柄式 BLAS 接口和 Ascend C Kernel 直调，不套用 ACLNN/TBE 自定义算子工程。

# 需求分析（required）

## 需求描述

新增共享公开接口：

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

声明放入 `include/cann_ops_blas.h`，复数类型和状态码采用 `include/cann_ops_blas_common.h` 的实际定义，不另设 950 专用公开接口。

| 参数 | 位置与类型 | 约束或语义 |
| --- | --- | --- |
| handle | Host 句柄 | 携带已绑定的 stream；空句柄返回 HANDLE_IS_NULLPTR |
| uplo | Host 枚举 | UPPER / LOWER；非法值返回 INVALID_ENUM |
| n | Host int | n≥0，n=0 按规定顺序直接返回成功 |
| alpha | Host complex64 标量指针 | 非空（n=0除外），按任务参数表设计 |
| beta | Host/Device complex64 标量指针 | 非空（n=0除外）；默认Host流程及Device候选流程见下文，Q1待确认 |
| A | Device complex64，lda×n | 列主序，lda≥max(1,n)，仅指定三角有效 |
| x、y | Device complex64 | 逻辑长度 n；物理长度至少 1+(n-1)×abs(inc) |
| incx、incy | Host int | 非零，支持负步长；y 原地更新 |

## 需求拆解

1. 接入公开头文件、Host 入口、Tiling 数据、Kernel 构建及同族测试工程。
2. 实现 Hermitian 共轭补全、实对角、complex64 乘加与原地输出。
3. 实现正负步长、lda padding、非对齐尾块与参数错误码。
4. 正确处理 n=0、alpha=0、beta=0 以及 alpha=0 且 beta=1。
5. 使用 Netlib CBLAS 3.10.0 的 `cblas_chemv` 生成 golden，覆盖随包用例并补充语义专项。
6. 在 950PR 上逐项验证性能，不能用 A2/A3/310P 数据替代验收。

# 详细设计（required）

## 算子分析

### 数学公式

采用零起始索引，列主序地址为 `A[i + j*lda]`。逻辑矩阵元素定义如下：

| 条件 | UPPER | LOWER |
| --- | --- | --- |
| i=j | complex(real(A[i+i*lda]), 0) | 同左 |
| i<j | A[i+j*lda] | conjugate(A[j+i*lda]) |
| i>j | conjugate(A[j+i*lda]) | A[i+j*lda] |

`s_i = sum(j=0..n-1, H_ij * x_j)`，`y_i = alpha*s_i + beta*y_i_old`。

复数乘法按 FP32 实数运算实现：`(a+ib)(c+id) = (ac-bd) + i(ad+bc)`。对角线使用实数乘复数的专门路径，不把内存中的对角虚部先参与乘法再清零。归约次序可以不同，但数学等价不足以保证浮点精度与非有限分类一致，具体基线及优化启用条件见Kernel设计。

### 支持数据类型

仅 complex64：实部和虚部各 float32。首版使用 FP32 乘加及归约，不采用未经验证的低精度转换或矩阵计算近似路径。

### 支持形状

A 为 lda×n，x/y 为逻辑长度 n 的向量，无 batch 和广播。需覆盖 n=0/1、质数、非对齐尺寸、2 的幂及其邻近值，当前性能用例最大 n=4096。实现不得将 4096 写死为 API 上限。测试README的单用例Host≤512MB只是测试预算，实际按A、x/y、golden、备份及哨兵总和计算，不能仅以n*n*8估算。

## 算子实现

### 实现方案

#### 3.2.1 Host 侧设计

**参数校验及快速返回顺序**以任务书 §2.5 为准：

1. handle 判空，失败返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
3. n=0 直接返回成功，不访问 alpha/beta/A/x/y，不继续校验 uplo、lda 和步长。
4. 校验 uplo、lda≥max(1,n)、incx/incy非零、alpha/beta指针非空。uplo错误返回 INVALID_ENUM，其余返回 INVALID_VALUE。
5. 在Host beta场景读取标量；alpha=0 且 beta=1 直接成功，允许 A/x/y 为空。Device beta遵循后述候选流程，不能从Host直接解引用。
6. alpha≠0 时检查 A/x 非空；剩余需要计算的路径检查 y 非空。alpha=0 时不访问 A/x。
7. 构造 Tiling 信息，在 handle 绑定的 stream 上下发 Kernel；结果读回前由调用方同步。

Host beta正常路径不添加设备同步、D2H 标量读取或 CPU 矩阵计算。Device beta不能仅写“后续扩展”：任务表已明确列出Host/Device，设计保留该能力并将Q1作为接口评审重点。

**Device beta候选流程（待Q1确认）**：沿同族方式查询beta位置；查询失败不当作Host指针解引用。alpha仍在Host。alpha非零时A/x/y必须非空，beta值不影响此校验，可直接将Device beta指针传到Kernel；alpha为零且y非空时，在Kernel中读取beta后决定不写、置零或缩放，A/x不访问。alpha为零且y为空时，Host若要立即区分beta=1的SUCCESS与其他beta的INVALID_VALUE，必须先获知Device beta值；候选方案在绑定stream上D2H并同步后判定，与csymv现有机制一致，但这是显式同步例外，需要官方批准。若要求beta=1严格“完全不下发Kernel”，即使y非空也需解决Host读值问题，不能同时承诺任意Device beta、无同步和Host即时判定。不得用异步设备错误替代任务要求的Host错误码。Device beta在消费完成前保持有效；同stream生产者写入beta的依赖须在测试中验证。位置查询失败拟返回INVALID_VALUE，D2H/同步失败拟沿同族返回INTERNAL_ERROR；最终以Q1确认结论为准。

已核对同族GetAivCoreCount()失败时返回EXECUTION_FAILED。Kernel采用`chemv_kernel_do(..., tiling, stream)`内部启动封装，使用AIV任务和`asc_vf_call`调用SIMT计算。csymv启动封装返回void，因此不得宣称返回SUCCESS已证明设备执行完成；异步故障由调用方同步及验收测试检查。

**步长与地址计算**：以调用者传入的分配起始地址为基准，`start_x = incx<0 ? (1-n)*incx : 0`，逻辑元素 j 的偏移为 `start_x+j*incx`；y 同理。乘法和地址偏移在提升到足够宽的有符号类型后计算，避免 int 中间结果溢出；对 `INT_MIN` 不能直接执行 int 类型 abs。额外拒绝规则须经评审确认，不能因测试说明的增强约束擅自改变公开契约。 对lda×n、向量跨度、复数元素转字节偏移等表达式执行有界乘法检查；int入参提升到int64并不自动保证乘8后的地址无溢出。建议在no-op和数据指针判定后，仅检查实际使用的A/x/y跨度，无法表示的布局返回INVALID_VALUE（Q2待确认）；alpha=0时不计算A/x跨度。令L为所选地址算术可表示的字节上限，先检查元素数≤floor(L/8)，再计算字节数；不能先溢出再判断。Host无法仅凭裸指针确认实际分配容量，调用者仍负责提供足够内存。INT_MIN先提升到int64再取负；“n=16、inc=INT_MIN”可用64位表示，不能以32位abs溢出证明该步长在API上必定非法。

**首版分核策略（SIMT）**：

- 通用回退ROW_ORDERED：一个线程负责一行，行号为`blockIdx*threads+threadIdx`，以`blocks*threads`为步长继续分配；不依赖warp通信。优先用于基础语义验证及实测更适合的小规模场景。
- 协作候选WARP_SIMT：一个32-lane warp负责一行。`warpId=threadIdx/32`，`lane=threadIdx%32`，`warpsPerBlock=threads/32`；输出行号为`blockIdx*warpsPerBlock+warpId`，行号步长为`blocks*warpsPerBlock`。
- lane遍历`j=lane, lane+32, ... < n`，分别累加实部和虚部。所有lane以相同调用序列执行宽度32的shuffle归约，只有lane0执行beta处理和写回。尾部无有效j的lane贡献零，不提前退出warp归约；row≥n只能整warp一致跳过。归约前各lane不因对角/三角分支产生不同shuffle调用序列。
- 每行只有一个线程或一个warp拥有写回权，无跨核原子更新。`blocks=min(ceil(n/rowsPerBlock), aivCoreNum)`，threads采用合法的32倍数；已核对同族常量SIMT_MIN_THREAD_NUM=128、SIMT_MAX_THREAD_NUM=2048，初始编译候选threads=128（每块4个warp），仅为与同族一致的起点；具体取值与切换阈值仍需950编译和实测。循环行号/列号及grid-stride更新统一使用64位算术，避免接近int上限时加步长回绕。

**缓冲区与后续分块**：首版SIMT直接按合法元素从GM取数，累加保存在实虚局部变量中，不申请显式矩阵UB tile或Device workspace。之后如引入AIV向量化二维分块，才计算B×T复数矩阵块、T个x及B个实虚累加器的实际UB占用；双缓冲、临时重排及归约空间均须计入。同族头文件UB_SIZE=248×1024是当前工程常量，不能代替目标环境可用容量验证。

**Tiling字段**：建议采用`nthreads/n/lda/uplo`、`alphaReal/alphaImag/betaReal/betaImag`、`int64_t incx/incy`与执行路径。起始偏移可由Kernel用64位算术推导，不重复传输。alpha固定Host，不加alphaIsDevice；beta保留betaIsDevice及Device指针参数，Host beta时存入实虚标量。Q1若正式收窄为Host-only再删除该分支。Device指针按Kernel封装参数传递。

**路径规划**：内部枚举表达SCALE、ROW_ORDERED、WARP_SIMT；UPPER/LOWER采用编译期布尔模板。路径均需支持其声明的全部参数，按n、布局及平台资源选择，不能根据CSV编号识别用例。暂不把寄存器tile方案纳入必选路径。

#### 3.2.2 Kernel 侧设计

**计算基线**：采用输出所有权明确的ROW_ORDERED，WARP_SIMT是待数值验证的优化候选，分核方式见Host设计。数学上的alpha×行点积不保证与Netlib的浮点求值顺序等价：可能改变中间溢出、抵消，以及有限/非有限分类。因此通用回退按每个输出复现Netlib列循环的更新顺序，而非仅将csymv行点积加共轭。

对每行i，先按beta分支初始化输出寄存器v，令t=alpha*x_i。UPPER：按j从0到i-1累加q=conj(A[j,i])*x_j；依次执行v=(v+t*real(A[i,i]))+alpha*q；再按j从i+1到n-1依次加(alpha*x_j)*A[i,j]。LOWER：先按j从0到i-1依次加(alpha*x_j)*A[i,j]；再加t*real(A[i,i])；按j从i+1到n-1累加q=conj(A[j,i])*x_j，最后加alpha*q。q累加从复数零开始，每步使用FP32，复数乘实数对角不引入虚部占位乘法。该顺序来自Netlib CHEMV源码；能否与指定3.10.0构建的复杂乘法和FMA策略一致仍须实测，不承诺bit-exact。记录并固定编译优化、FMA及复数运算策略，不开启会破坏NaN/Inf语义的fast-math；目标工具链支持性在950验证。[Netlib参考](https://www.netlib.org/blas/chemv.f)

WARP_SIMT按行点积树归约计算alpha*s_i+beta*y_i，必须分别通过随机、抵消、大值、Inf/NaN与零实/虚标量测试。没有可证明的适用域或等价性检查前不得默认开启。若出现有限/非有限分类不一致，不能套用“NaN与Inf等效”放行，也不能按CSV名称选择回退；先保留ROW_ORDERED全域可用，再设计可说明的数值判定/回退或修订协作算法。若增加输入扫描或重算，其开销全部计入算子。每个非对角存储元素为生成两个输出可能被读取两次，这是首版的带宽成本，不预先承诺达到性能门槛。以下CopyIn/Compute/CopyOut表示逻辑阶段，首版SIMT不强制使用TQue或显式UB流水。

1. Init：解析参数，建立受控的 GM 访问范围；Device beta在有效线程/warp执行计算前读取并形成一致的标量分支，alpha=0且beta=1整组不写。初始化当前输出组的实虚累加器。
2. CopyIn：按索引表访问唯一合法的三角位置并读取x；仅在非对角时读取A的虚部，对角只加载实部。后续二维块搬运若跨对角线，必须先拆分合法区域，不能整块读取未指定三角后再用计算mask屏蔽。
3. Compute：对需要补全的一侧将虚部取反；对角只取实部。完成复数乘法和 FP32 归约。不以“x_j=0”为由任意跳过矩阵项，避免改变 Inf/NaN 传播语义。
4. Epilogue：ROW_ORDERED按前述顺序直接形成v并写回；warp候选仅lane0计算alpha*s_i并处理beta。beta=0时既不读取旧y，也不执行beta×零占位运算；beta=1时直接加旧y，其余按复数乘法计算beta*y。对角贡献采用实数乘复数路径，避免虚部占位零与Inf相乘引入非预期NaN。
5. CopyOut：仅写当前组的有效输出位置，不改写步长间隙和分配尾部。非对齐尾块使用已验证的精确长度搬运能力或受边界保护的访问，不允许通过向上取整越界。

**连续访存与通用步长**：首版SIMT对incx=1采用连续逻辑x索引；非单位或负步长使用64位物理偏移。行点积在指定三角两侧分别包含连续与跨列访问，warp合并访存不能自动消除全部跨列开销；二维tile优化须另行实测。写回只访问当前y元素，不写步长间隙。

**alpha=0 路径**：只处理 y=beta*y；beta=0 时直接写复数零，不读取旧 y；beta=1 在Host已知时快速返回，Device beta在Kernel读值后不写。该路径不访问 A/x，允许二者为空。

**大尺寸优化候选**：基线正确后，评估列主序下的二维矩阵块搬运、转置/重排及 x 块复用，以降低行点积的跨列访问开销。具体 Ascend C API、对齐条件和 UB 占用须先核对 950 支持情况。若需要拆分归约并增加 workspace/第二 Kernel，须更新设计并统计完整算子所有 Kernel 的成本，不能仅报告其中最快的一个。

**小尺寸优化候选**：减少核数和归约分块开销，尝试单核或少核融合路径。阈值根据全部性能用例确定，不按用例 ID 硬编码行为。

**访存验收不变量**：每次A load都满足被选三角与0≤行/列<n；索引至多(n-1)+(n-1)*lda，padding不读取。x/y偏移均在[0,(n-1)*abs(inc)]内。负步长传入分配起始地址，不能由调用方先挪到逻辑首元素再重复偏移。每个有效y恰有一个写者；x/y重叠可能产生跨行读写竞争，在官方确认前不宣称支持（Q5）。不引入全局原子加或核间自旋同步。尾部必须按实际元素保护load/store，不能仅对计算结果加mask。对角不加载虚部是本方案的实现选择，任务硬约束是其不参与运算。

### 工程文件规划

| 位置 | 拟改动 |
| --- | --- |
| include/cann_ops_blas.h | 新增共享公开接口声明 |
| blas/symv/arch35/chemv_host.cpp | 接口入口、校验、标量分支、分块与启动 |
| blas/symv/arch35/chemv_kernel.cpp | Hermitian 复数计算及缩放路径 |
| blas/symv/arch35/chemv_tiling_data.h | Host/Kernel 共享内部数据 |
| blas/symv/ 下对应说明文档 | 接口语义、约束和 950PR 支持说明，具体文件位置按贡献规范确认 |
| test/symv/chemv/arch35/ | CSV 加载、GTest、CBLAS golden、精度与性能验证 |
| test/symv/chemv/CMakeLists.txt | 参考csymv，以arch35条件调用ops_blas_add_gtest_tests注册chemv_test |
| test/symv/chemv/ 下参数与golden辅助头 | 解析CSV并接入Netlib，不复制csymv对称算法 |
| blas/CMakeLists.txt、test/CMakeLists.txt | 已核对按架构收集源码及发现家族测试目录，原则上复用；用--ops=chemv验证无需额外硬编码 |
| 根CMakeLists.txt及CBLAS测试链接 | 复核chemv编译参数、Netlib头文件和库链接；csymv专属-O3条件不会自动作用于chemv |

文件名为拟定方案，最终遵循仓库约定。测试源码与CSV按任务要求放在arch35子目录，CMake入口放在chemv测试目录。参考当前脚本的拟用命令如下，尚未执行，不能作为已通过构建的证明：

```bash
bash build.sh --soc=ascend950 --ops=chemv
bash build.sh --soc=ascend950 --ops=chemv --run --device=0
build/test/symv/chemv/chemv_test --gtest_filter='-*TC_PF*'
```

Netlib CBLAS 3.10.0需在测试构建中明确定位；不能只因系统存在OpenBLAS就宣称满足指定标杆版本。普通合法数值用例调用cblas_chemv，接口负向与任务特有的n=0先返回用例只断言NPU状态码，不把非法参数传给CBLAS。Netlib本身先检查uplo等再快速返回，与任务Host顺序不同。合法alpha=0且A/x为空用例可给golden安全哑元或按明确缩放语义形成预期值，NPU仍接收任务规定的空指针。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 验证状态 |
| --- | --- | --- |
| Ascend 950PR（arch35），CANN 9.1.0 | √ | 待获得环境后编译、自测与 Profiling |

A2/A3/310P 可用于适合其环境的辅助准备，但不属于本任务声明的已支持平台，不替代 950PR 验收。

## 算子约束限制

- 仅 complex64，不提供 batch、广播或未定义的视图语义；支持 lda、incx、incy 表达的布局。
- A/x 只读，y 原地更新。A/x/y 之间的重叠别名支持范围任务书未明确；不宣称支持 x 与 y 重叠，提交前向评审确认。
- 对角虚部及未指定三角不参与结果；alpha/beta 为零的分支必须避免读取任务明确允许无效的输入。
- 不承诺位级确定性，不预先声明性能达标。
- Host beta及通常Device beta路径保持异步；Device beta空y分支的同步例外待Q1确认。流的创建、绑定和同步沿用库约定。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 设计要求 | 标准来源 |
| --- | --- | --- |
| 精度 golden | Netlib BLAS 3.10.0 cblas_chemv，CblasColMajor；输出实部、虚部分别全量比较 | 任务书 §3.1、§3.2 |
| 有限值精度 | atol=rtol=2^-13；逐元素误差≤atol+rtol×abs(golden)，匹配率≥0.99，同时满足最大绝对误差上限 | 任务书 §3.2 |
| 最大误差 | 已核实固定opbase版本§2.1.2/§2.2：max_abs_error_limit=max(1e-2,32×ULP(g_low))，g_low取最大绝对误差点的FP32 golden；不是任意选择阈值 | 任务书引用的opbase提交28a1ce52 |
| 非有限值 | 同为NaN、同号Inf、NaN与±Inf按任务规则视为等效；有限与非有限不等效；非有限值单独统计且须全部通过 | 任务书 §3.2 |
| 性能 | 每个对应 case 的 NPU 平均耗时≤所给 GPU 耗时的2.5倍，不能以平均加速比掩盖失败 case | 任务书 §3.3、gpu_baseline.csv |
| 采集口径 | 950PR 上使用 msprof op；明确预热、有效采样、同步和单位，排除 Host/CBLAS 比较耗时 | 任务书及测试说明，采样次数待统一 |

有限值比较对实部、虚部各自建掩码并分别判定，不合并以稀释失败。使用FP64计算两个FP32值之差及误差统计，避免比较器自身溢出。ULP边界实现（零、次正规数、二进制幂、最大有限值及并列最大误差点）与官方验证器对齐，禁止直接用nextafter(FLT_MAX,+Inf)得出无限容差。全无有限点时记该项N/A，仅检查所有非有限点；n=0按接口no-op检查。

任务特有非有限规则优先：NaN/NaN、同号Inf/Inf、NaN/任意符号Inf可通过；+Inf/-Inf及有限/非有限必须失败。这是任务测试契约，不解释为通用IEEE等价。CSV的MERE/MARE另行统计，不能替代混合容差；其是否构成额外阻断门槛列Q3。

### 测试覆盖与执行计划

1. 导入随包1200条CSV用例，分开记录1000条精度与200条性能结果。CSV驱动测试工程需实现或接入，Python验证脚本本身不等于完整测试工程。
2. 使用固定种子生成输入，逐项记录参数、分布、golden与NPU结果、实虚误差统计、返回码、设备及构建提交号。补齐任务要求的均匀/正态各50%输入分布，不能只依赖用例列名推定已覆盖。
3. 覆盖UPPER/LOWER、正负步长±1/±2/±3、lda padding、n=0/1及非对齐尺寸、一般与特殊alpha/beta、Inf/NaN、合法空指针快速返回及非法参数。
4. 单独构造未指定三角被污染、对角虚部为非零/NaN、beta=0且旧y为NaN的样本，验证忽略语义。结合内存访问检查核对不合法区域未被批量搬运；数值正确不能单独证明无越界读取。
5. 增加句柄空指针与错误叠加的校验顺序测试，避免CSV默认有效handle遗漏接口负向测试。
6. 给步长间隙和输出缓冲前后设置哨兵，核对只更新逻辑y位置；使用非默认stream验证依赖关系及结果同步。
7. 精度执行时，每轮重新初始化y。性能执行亦须规定y恢复策略，避免反复原地更新导致数值溢出；恢复操作不计入Kernel耗时，具体与验收脚本保持一致。
8. 性能用例与基线按完整参数映射，包含uplo/n/lda/incx/incy及alpha/beta，避免误配。当前200条基线只有198个唯一参数键，其中两组重复且耗时相同；保留200个case身份及全部baseline id，禁止字典覆盖后宣称只需跑198条。遇到同键不同耗时必须报冲突，不选有利值。基线列gpu_ms先转换为us，即允许耗时上限为gpu_ms×1000×2.5。
9. msprof记录各case的有效样本数量、所有相关Kernel的耗时和异常，不以Host时钟或单个Kernel片段替代完整算子成本。若多Kernel方案被采用，须确认各Kernel耗时合计的验收口径。
10. 全有限值为空的特殊样本不计算除零的匹配率；单独报告非有限值全部比较结果。主标杆为CBLAS，独立手算小矩阵只作为排查辅助，不能取代验收标杆。
11. Device beta若按任务表支持，补充同stream前置写标量、beta=0/1/一般值、alpha=0且空A/x/y、查询/同步失败等专项，确认Q1定义的同步边界。
12. ROW_ORDERED与WARP_SIMT各自强制运行全套合法数值用例，补齐n=31/32/33、127/128/129、grid-stride多轮及切换阈值两侧。记录路径和资源配置；优化后回归既有ssymv/csymv构建与测试。
13. 950验证依次进行：构建与基础语义→完整精度→专项边界→200条性能→按失败原因优化→完整回归。提交前完成代码review和仓库要求的pre-commit、构建检查。

### 测试工具修正要求

本轮只审查随包脚本，未修改或运行算子测试。后续测试工程需解决以下问题：

- `verify_accuracy.py`只解析GTest文本，未以进程返回码和预期用例全集判定；超时返回0条也可最终退出0。应将超时、崩溃、零用例、缺失/重复用例、SKIPPED均视为未完成/失败，不输出ALL PASS。
- `verify_performance.py`把GTest整段运行的整数毫秒当作npu_ms，与GPU微秒级数据直接比较；未调用msprof，不能用作正式性能报告。其失败case/0ms可能被归入NO_REF，部分NO_REF也可退出0，应拒绝此类假成功。
- 性能从msprof `Task Duration(us)`提取，按case、调用序号、stream与全部所属Kernel映射；同时要求API返回成功、同步成功、采集完整。FAILED、NO_REF、缺样和异常均不能静默丢弃后宣布达标。
- 保留原始1200条用例；检查实际填充器。当前A/x/y各有1190条使用`RANDOM_NORM_5_5`标签，标签不证明均匀/正态各50%，需在补充用例中显式生成并记录μ/σ和种子。
- 不重新运行gen_csv覆盖随包GPU基线。额外布局/步长性能探索独立报告，无匹配GPU标杆时只报告耗时，不判正式达标。

### 可人工复核的最小数值样本

取`H=[[2, 3+4i], [3-4i, 5]]`、`x=[1, i]`、alpha=1、beta=0，应得`y=[-2+3i, 3+i]`。

- UPPER的列主序A为`[2, poison, 3+4i, 5]`；LOWER为`[2, 3-4i, poison, 5]`。poison可设为NaN，两个布局应给出相同有限结果。
- 在对角线内存中另外填入非零虚部，结果仍应不变；这检验对角虚部不会被错误参与计算。
- incx=-1时物理x改为`[i, 1]`，incy=-1时物理y结果为`[3+i, -2+3i]`，逻辑结果不变。
- 这一小样本仅验证索引和共轭公式，不能代替CBLAS、完整CSV及NPU执行。

### 性能预算与风险定位

已直接读取任务2的200条GPU基线及性能CSV：全部性能case的alpha=(1,0)、beta=(0,0)、incx=incy=1、lda=n。这可作为优先调优路径，但不减少一般alpha/beta和步长的精度覆盖。

| 首四条基线参数 | 允许的NPU平均耗时上限（us） |
| --- | ---: |
| n=512，UPPER | 24.4275 |
| n=1024，UPPER | 35.1775 |
| n=2048，LOWER | 45.7100 |
| n=4096，LOWER | 101.8050 |

所给GPU耗时范围为2.922～44.743us，对应NPU上限为7.305～111.8575us。以上仅是CSV计算出的预算，不是NPU实测结果。小尺寸重点观察启动开销，大尺寸重点观察跨列读取和矩阵重复读取；所有200条均需匹配自己的基线，不能用首四条概括验收。

采样计划分为正式口径与探索口径：任务书§3.3明确为msprof op预热5次后连续10个有效样本取均值；§7另写>10次，测试README写>50次，存在材料冲突，列Q4请官方统一。确认前以§3.3的10次均值作为任务书口径报告，并额外计划独立的预热5次+60次探索采样，保留全部明细和60次均值；不得将探索结果冒充正式验收结论，也不以两组中较快的结果择优判定。核对msprof与测试循环实际共同产生的次数，避免双重预热或重复统计。完整保留异常及补采原因，不因耗时慢而剔除样本。若方案增加Kernel，按同一次调用汇总全部Kernel，不只取最快Kernel；等待Q4确认task duration求和口径。

上述预算是CSV换算，不代表方案能满足要求。大尺寸约O(n²)矩阵读取，行/warp路径逻辑上每个非对角存储元素读两次；实际HBM流量受缓存影响，不能从逻辑读数推算实测带宽。小尺寸先看启动成本，再选择更少核/线程；大尺寸再评估二维分块与x复用。优化阈值只能来自通用参数与实测，不能匹配case编号。

### 自测交付件

设计文档评审及合入证明；验收代码commit、分支与目录；CSV、测试源码和可复现命令；精度/性能报告及日志；msprof原始结果；环境版本及设备信息。报告按任务书§4指定的[自测报告模板](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)和[交付材料包](https://cann-op.obs.cn-north-4.myhuaweicloud.com/task_submission.zip)准备，后续填写前核对模板实际字段，不将模板链接视为已完成交付。设计文档和代码不包含访问令牌、SSH密码或临时算力凭据。

## 兼容性分析

新增接口沿用库中的句柄、stream、复数类型、枚举及状态码。不得改动已有ssymv/csymv行为或公共类型布局。共享头文件声明不等于所有架构均已实现，构建注册、符号导出和支持表须正确限定950PR能力，其他架构兼容处理需依仓库约定确认。

新增测试应与同族测试共存。归约次序可能导致FP32结果与CBLAS不同，按任务容差验收，保留失败样本以定位数值问题。

## 待官方确认事项

| 编号 | 需官方裁定的问题 | 当前设计与影响 |
| --- | --- | --- |
| Q1 | beta的Host/Device是否均为验收要求？Device beta遇alpha=0且y为空时，是否允许D2H+stream同步以满足即时错误码？beta=1是否要求完全不下发Kernel？alpha是否仅Host？ | 保留Device beta设计，不自行缩窄；确认前不能冻结Host契约 |
| Q2 | README/TC_ED_217、218对INT_MIN返回INVALID_VALUE的增强约束是否正式采用？no-op是否仍先返回？不可表示的字节跨度在何阶段返回何错误码？ | 使用64位安全算术；不把实现溢出当作API禁用依据；影响负向用例和Host校验 |
| Q3 | MERE/MARE是否是混合容差之外的额外门槛？ULP边界和最大误差并列点是否有官方比较器？ | “or取max”与锚点已由固定opbase源码解决；两套统计同时保留 |
| Q4 | 正式样本数为10、>10还是>50？msprof与GTest的重复计数如何统一？多Kernel一次调用如何汇总？ | 以§3.3的5次预热+10次均值为任务书口径，另做独立60次探索；两组完整报告，最终正式采样数待统一 |
| Q5 | A/x/y重叠别名是否要求支持？ | 当前只保证不重叠数据；x=y可能跨行竞争，若要求支持需快照/workspace及相应成本 |
| Q6 | 社区任务设计评审是否还需独立SIG需求评审，以及共享API在非arch35构建中的符号策略？ | 任务引导已明确需求Issue在首位通过测试并收到通知后与代码PR一起提交；仅剩SIG衔接和跨架构策略待维护者确认，不提前增设任务步骤 |

Q1/Q2影响实现契约，应在开发前解决；Q3/Q4影响验收报告冻结；Q5影响存储方案；Q6在对应提交环节核对。设计文档可携上述明确问题申请评审，不宣称方案已获批准。

## 提交位置与阶段边界

已核实比赛仓提交 `8ec30dd282b4e8e288aaabfdcf2bfb54efd9f7c0` 的社区任务README，并复核至`8b7d83054724cf8726d1f36beee4585e402bb7fb`，相关规则未变：设计阶段仅要求个人目录下的`docs/design.md`。目标路径：

```text
04_tasks/01_community-task-2026/tasklist/CANN训练营哈工深-aclblasChemv算子开发（950）/kongbada/docs/design.md
```

官方树中该任务已有其他参与者目录，不创建近似名称任务，不覆盖他人文档。`kongbada`作为个人提交目录名。根CONTRIBUTING的通用赛事`submissions/`规范不代替社区任务专属规则；完成代码后的交付按本任务书指向ops-blas，不能套用模板中的ACLNN/op_host工程。

当前`cann-competitions`与`cann-ops-competitions`两个官方Git地址的HEAD相同；浏览器页面/社区README使用后者名称。实际发布时选择该官方仓的master，复查平台重定向和最新HEAD。本次提交范围为设计文档；算子代码及测试结果待设计评审通过并完成开发、自测后另行提交。

参与者补充的《03任务引导》明确以下流程（来源为本次对话粘贴正文，非本轮重新抓取网页）：

1. 报名、设计提交、进展更新和交付件提交统一使用`kongbada`，活动期间不修改账号名。
2. 设计PR标题使用`【CANN社区任务】aclblasChemv算子设计文档`。发布后在平台【更新进展】填写设计PR链接，【代码链接】填“暂无”。
3. 提交或更新设计PR后，在PR评论区提醒`@condfuse_3 @fullt @Ascend-CANN`；修改后再次提醒，以评论中的“评审通过”为审核依据。文档修改在原PR中进行，无需重复更新平台进展。
4. 设计审核通过后再开展代码开发；提交验收前设计PR必须已评审通过并合入。验收材料包含设计PR合入截图、测试用例、结果报告、测试步骤及代码仓/分支/算子目录，私仓邀请Ascend-CANN为开发者。
5. 必须在任务截止时间前提交验收；补充的任务平台截图显示2026年11月25日，具体时刻仍需核对。验收审核期间不能修改当前进展或验收信息。
6. 首位通过测试的开发者收到通知后，须在2个工作日内按任务书提交需求Issue和代码PR，并同步昇腾CANN小助手；随后逐条回复评审意见，修改后提醒对应评审人并附自验证结果。

提交设计PR后按上述流程更新进展及请求评审；未获设计评审通过前不开始算子开发。

## 参考资料

- 本地任务书：任务2压缩包内`aclblasChemv_Atlas950PR_task_doc.md`；测试材料：`test_cases/README.md`、两份CSV、三个Python脚本。任务目录根部与任务2中的同名ZIP的SHA256相同。
- [ops-blas固定提交](https://gitcode.com/cann/ops-blas/tree/bab3ec101cfed2f4c4339e2d5e3919ac35f76a3c)：核对公开头文件、`blas/symv/arch35/csymv_{host,kernel}.cpp`、tiling、`blas/common/helper/kernel_constant.h`、同族测试、根/blas/test CMake、build.sh、CONTRIBUTING.md。
- [社区任务提交规则](https://gitcode.com/cann/cann-ops-competitions/blob/8ec30dd282b4e8e288aaabfdcf2bfb54efd9f7c0/04_tasks/01_community-task-2026/README.md)；[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/8ec30dd282b4e8e288aaabfdcf2bfb54efd9f7c0/04_tasks/01_community-task-2026/resources/design_template.md)。本地模板与固定提交模板规范化换行后文本相同。
- [任务引用的精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)：已通过Git读取固定版本。
- [Netlib CHEMV参考实现](https://www.netlib.org/blas/chemv.f)；[cuBLAS HEMV接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-hemv)。在线参考不能替代指定Netlib 3.10.0构建的实际golden。
- 参与者在本次对话提供的《03任务引导》正文：用于PR标题、账号一致性、指定审核人、平台进展和验收后提交流程；参与者随后补充了上文任务页URL；另有任务平台截图佐证截止日期2026年11月25日，未显示具体时刻。流程正文仍以提供的引导为依据，不声称已独立抓取其网页全文。
- [社区任务流程讨论](https://gitcode.com/org/cann/discussions/39)：此前网页读取仅返回站点外壳；指定审核人等流程现依据参与者补充的引导正文，不声称已读取该讨论正文。

AI辅助：本次提交的最终修订与复核使用OpenAI Codex，模型为gpt-6-astra（GPT-6 Astra），已根据本机会话记录核实。PR描述与提交说明使用相同模型标识；设计和PR文案仍需参与者审阅。
