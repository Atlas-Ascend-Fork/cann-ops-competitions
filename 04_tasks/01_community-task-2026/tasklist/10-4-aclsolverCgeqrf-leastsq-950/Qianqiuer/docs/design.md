# 需求背景（required）

## 需求来源

2026 年 10 月社区任务《Atlas 950 单精度复数 QR 分解与最小二乘任务书》，原文件名 `Atlas950_Cgeqrf_Cungqr_Cunmqr_CCgels_task_doc.md`。参与账号及团队：Qianqiuer。设计日期：2026-10-14。

本任务统一交付 `aclsolverCgeqrf`、`aclsolverCungqr`、`aclsolverCunmqr`、`aclsolverCCgels` 及四个 `_bufferSize` 接口，目标为 Ascend 950PR，代码合入 [cann/ops-solver](https://gitcode.com/cann/ops-solver)。四接口共同验收，不交付公开 `aclsolverXlarft` 接口。

本文以开发前设计为基础，补充实现采用的精度路径；文中性能、精度数字均为验收目标或官方基线统计，实际测试结果见独立自测报告。上游提交 `54fcf6cbdf988bce0efbe6ebe345cec47e2c8ce7` 已将本任务列为 10 月第 4 项，设计文档归入正式目录 `10-4-aclsolverCgeqrf-leastsq-950/Qianqiuer/docs/design.md`。

## 背景介绍

复数 QR 分解是复数最小二乘、正交化和酉变换的基础。采用 Householder 表示可避免为每次乘 Q 显式构造完整矩阵，并将大矩阵更新转换为块矩阵运算。

### 现状与工程差异

以 ops-solver `a29362cfa605c69b1f330feadbff21e686eb9502` 为设计基线：仓库提供 LU、求逆等实现以及 handle/stream 管理，但没有本任务四接口。该任务不是历史 TBE/aclnn 优化，不使用两段式 ACLNN、PyTorch 注册或 CPU fallback。

仓库 `src/CMakeLists.txt` 自动收集算子源码并按 `arch35` 等目录选择架构实现；测试通过 `BUILD_TEST`、`TEST_NAMES` 接入。现有公共头文件包含 C++ 标准库及 `std::complex` 原型，不能仅加 `extern "C"` 就声称支持 C 编译。新增接口必须使用 C POD 复数和现有 `aclsolverStatus_t`，并按兼容性章节处理头文件。

# 需求分析（required）

## 需求描述

| 接口 | 功能和输入/输出 | 合法规模及组合 |
| --- | --- | --- |
| Cgeqrf | A 原地保存 R 和反射器，TAU 输出 min(m,n) 个系数 | m,n≥0；验收覆盖 1～4096，方阵、瘦高、矮宽 |
| Cungqr | 使用 k 个反射器生成 m×n 的 Q，覆盖 A；tau 只读 | m≥n≥k≥0；m 至少覆盖至 4096；含 k<n |
| Cunmqr | 使用反射器计算 Q 或 Qᴴ 对 C 的左/右作用；C 原地输出，A/tau 只读 | m,n≥0，0≤k≤nq；nq=LEFT?m:n；仅 N/C，T 报错 |
| CCgels | 求 min‖AX−B‖F；A 可破坏、B 只读、X 独立输出 | m≥n≥0，nrhs≥0；m,n 至少至 4096，nrhs 至少至 128 |

全部矩阵 COMPLEX64、列主序，元素地址为 `base + row + col*ld`；ld 使用实际传入值，不要求紧凑。Cgeqrf/Cungqr 的 lda≥max(1,m)，Cunmqr 的 lda≥max(1,nq)、ldc≥max(1,m)，CCgels 的 ldda/lddb≥max(1,m)、lddx≥max(1,n)。不支持广播，不要求图融合；Host 按每次维数重新计算布局和分派。

## 需求拆解

1. 八个接口、状态码、Host/Device 参数归属和 workspace 单位与任务书一致。
2. AI Core 上完成全部数值计算，复用已有 handle 管理；调用方 stream 上有序执行。
3. 满足四套 LAPACK 残差门禁；同输入、同 stream 串行重复结果含 tau/info 逐位一致。
4. 官方 694 条用例全部进行功能验证，性能逐例 GPU/NPU≥0.35；没有有效 GPU 基线的用例单独列为基线缺失，不判定达标。
5. 提供异常、零维、padding、确定性、非默认流、自构造分布与边界测试，形成可复现报告及原始证据。

# 详细设计（required）

## 算子分析

### 数学公式与复数约定

令 p=min(m,n)，反射器 `Hj=I−τj vj vjᴴ`，vj 的第 j 个分量隐含为 1。采用 `Q=H1 H2 … Hp`、`R=Qᴴ A`。复数反射器一般不是 Hermitian，不能把 τ 与其共轭混用。

对当前列 `[α;x]`，使用缩放范数计算 `s=‖x‖₂`；若 s=0 且 Im(α)=0，令 τ=0，其余反射器尾向量置零。否则取确定性实数 `β=−sign(Re(α)) hypot(Re(α),Im(α),s)`（Re(α)=0 时符号取正），令 `τ=(β−α)/β`，`v_tail=x/(α−β)`。极小数使用安全缩放再恢复 β，避免直接平方溢出/下溢。该相位和共轭约定参考 [Netlib CLARFG](https://www.netlib.org/lapack/complex/clarfg.f)。零长度尾向量不等于反射器恒为单位阵：α 有虚部时仍需生成反射器。

块反射器写成 `Hblock=I−V T Vᴴ`，T 为上三角；T 的对角是 τ，前 i−1 项按 `−τi Tprev(Vprevᴴ vi)` 构造。仅作为内部实现，不新增 Xlarft 公共 API，数学表示参考 [Netlib CLARFT](https://www.netlib.org/lapack/complex/clarft.f)。

CCgels 使用 QR：先计算 `Y=QᴴB`，再求解 `R(1:n,1:n)X=Y(1:n,:)`。不构造 `AᴴA` 正规方程求解，避免平方条件数；任务书中的正规方程残差仅用于验证。

### 支持数据类型和精度路径

公开复数为两个连续 FLOAT32，大小 8 字节，real 在偏移 0、imag 在偏移 4；以编译期断言和 C/C++ ABI 测试验证。内部拆为实部/虚部运算。考虑 950 的矩阵乘吞吐特征，大矩阵分块更新优先采用 HF32 Cube 路径，输入输出接口仍为 COMPLEX64。复数矩阵乘采用四个实数乘积组合；HF32 对乘法输入进行舍入，不能等同于完整 FP32 乘法精度，需逐例验证任务书规定的残差与确定性。Householder 范数、反射器系数、关键归约及回代采用稳定的 FP32 算法；矩阵更新在必要时采用高低位拆分补偿乘积、补偿累加或修正，不满足门禁时使用经过验证的 FP32 路径。所有补偿和修正开销完整计入性能，不放宽阈值，不默认降为 FP16/BF16。目标 SDK 的 SetHF32Mode 与舍入模式接口须按实际 Cube 调用链设置并验证，最终方案以全部官方精度用例及有效性能基线的实测结果为准。

### 公开类型和接口

以下原型保持任务书参数顺序、const、32 位维数及返回类型，不借用现有 aclError 接口。

```c
typedef struct { float real; float imag; } aclFloatComplex;
typedef enum {
    ACLSOLVER_OP_N = 0, ACLSOLVER_OP_T = 1, ACLSOLVER_OP_C = 2
} aclsolverOperation_t;
typedef enum {
    ACLSOLVER_SIDE_LEFT = 0, ACLSOLVER_SIDE_RIGHT = 1
} aclsolverSideMode_t;

aclsolverStatus_t aclsolverCgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A,
    int lda, int *Lwork);
aclsolverStatus_t aclsolverCgeqrf(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda,
    aclFloatComplex *TAU, aclFloatComplex *Workspace, int Lwork, int *devInfo);
aclsolverStatus_t aclsolverCungqr_bufferSize(
    aclsolverHandle_t handle, int m, int n, int k,
    const aclFloatComplex *A, int lda, const aclFloatComplex *tau, int *lwork);
aclsolverStatus_t aclsolverCungqr(
    aclsolverHandle_t handle, int m, int n, int k,
    aclFloatComplex *A, int lda, const aclFloatComplex *tau,
    aclFloatComplex *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverCunmqr_bufferSize(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda,
    const aclFloatComplex *tau, const aclFloatComplex *C, int ldc, int *lwork);
aclsolverStatus_t aclsolverCunmqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda,
    const aclFloatComplex *tau, aclFloatComplex *C, int ldc,
    aclFloatComplex *work, int lwork, int *devInfo);
aclsolverStatus_t aclsolverCCgels_bufferSize(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    aclFloatComplex *dA, int ldda, aclFloatComplex *dB, int lddb,
    aclFloatComplex *dX, int lddx, void *dwork, size_t *lwork_bytes);
aclsolverStatus_t aclsolverCCgels(
    aclsolverHandle_t handle, int m, int n, int nrhs,
    aclFloatComplex *dA, int ldda, aclFloatComplex *dB, int lddb,
    aclFloatComplex *dX, int lddx, void *dWorkspace, size_t lwork_bytes,
    int *niter, int *dinfo);
```

Lwork/lwork 查询输出和 niter 为 Host 指针；A、TAU/tau、C、B、X、workspace、devInfo/dinfo 为 Device 指针。前三类 workspace 以 8 字节复数元素计，CCgels 以字节计。CCgels 查询的 dwork 不作为已有 workspace 消费，查询允许其为 NULL。查询不执行分解、不读取矩阵内容。

## 算子实现

### Host 侧设计

#### 参数校验和信息写回

复用 aclsolverCreate/Destroy/SetStream/GetStream。先校验 handle，再按参数顺序校验维数、枚举、ld、所需指针、workspace；多个错误取首个非法参数。查询输出指针为空返回 INVALID_VALUE。计算接口的参数编号不计 handle：

| 接口 | 参数编号 |
| --- | --- |
| Cgeqrf | m=1,n=2,A=3,lda=4,TAU=5,Workspace=6,Lwork=7,devInfo=8 |
| Cungqr | m=1,n=2,k=3,A=4,lda=5,tau=6,work=7,lwork=8,devInfo=9 |
| Cunmqr | side=1,trans=2,m=3,n=4,k=5,A=6,lda=7,tau=8,C=9,ldc=10,work=11,lwork=12,devInfo=13 |
| CCgels | m=1,n=2,nrhs=3,dA=4,ldda=5,dB=6,lddb=7,dX=8,lddx=9,dWorkspace=10,lwork_bytes=11,niter=12,dinfo=13 |

非法参数返回 ACLSOLVER_STATUS_INVALID_VALUE；合法 handle 且 info 非空可写时，在该 stream 下发小型写 info kernel 写 −i。成功路径也实际写 0，不能只依赖内存初值。空 handle 返回仓库约定的 HANDLE_IS_NULLPTR，因无合法 stream 不强行写 info；info 本身为空时返回错误且不能写回。异步启动失败映射 EXECUTION_FAILED，资源不足映射 ALLOC_FAILED，不直接强转 ACL 错误码。

有限值约束须单独处理：Host 不能解引用 Device 数据。若要求非法数值在本次 Host 返回中体现 INVALID_VALUE，预检 kernel 只扫描实际消费的输入区域、归约首个非法输入参数，并回传一个标志；该严格检查需要一次有明确理由的 stream 等待，不可谎称完全异步。扫描和信息 kernel 均计入设备性能。评审若确认数值仅为调用者前置条件，可取消该等待并保留设备错误信息；在此确认前按严格校验方案预算，不通过省略校验获取达标结果。缓冲区真实容量由调用方负责，裸指针接口无法从指针推断 dtype、完整分配长度或任意地址可写性；明确区分可检查元数据和调用者存储契约。

#### 空问题语义

零维仍检查非法枚举、负维数、ld 和必需 info/niter；不访问不存在的矩阵元素，相关零长度数据指针可空。m=0 或 n=0 时写 info=0 后返回；CCgels 的 nrhs=0 同样成功，niter=0。Cunmqr k=0 为恒等变换、C 不变。Cungqr 在 m,n>0 且 k=0 时必须写 m×n 的单位矩阵前 n 列，A 仍须有效；不能当作无写入快速返回。其 tau 可空，但 workspace 是否需要仍服从该形状查询结果。

#### Workspace 和 tiling

查询与执行共用同一个布局函数。使用 uint64_t/size_t 做地址、乘加和对齐计算，并逐步检查溢出；前三类返回元素数前检查 INT_MAX，不能截断。各 scratch 分区按 32B 对齐，较强的底层 API 对齐要求由统一布局器升级。分配需求包含最大分支、双缓冲、归约 partial、状态和工具链系统 workspace；执行不隐藏申请与查询不一致的大矩阵内存。

设块宽 b 初始候选为 16/32/64、p=min(m,n)、q=nq，s 为由 UB 与核数确定的输出条带宽。候选是调优参数，不是固定支持上限。

| 接口 | 主要 scratch 分区和生命周期 |
| --- | --- |
| Cgeqrf | panel V：至多 m×b；T：b×b；更新 W：b×s；实虚打包块、固定归约 partial；逐 panel 复用 |
| Cungqr | 保存输入反射器的 q×k 紧凑区，避免初始化 Q 覆盖输入；T、W 及打包区；输出始终为 A 的有效 m×n 区 |
| Cunmqr | V：q×b；T：b×b；LEFT 为 b×s 的 W、RIGHT 为 s×b 的 W；不改写 A/tau |
| CCgels | tau：n；Y：m×nrhs 的 B 副本；QR/酉变换/回代 scratch 按阶段最大值复用；X 最后写独立输出 |

Host tiling 记录接口模式、m/n/k/nrhs、ld、panel/tile 大小、core 数、有效尾块、workspace 偏移、side/trans；设备循环仅处理有效区域。平台查询获得 UB/L1/L0 与核数，按 `输入双缓冲+输出+实虚中间量+归约+API临时空间≤实际可用内存` 校验，不照搬其他芯片内存常数。超出单核容量分块，不因此拒绝任务书内合法 shape。

### Kernel 侧设计

#### Cgeqrf：panel 分解与 trailing 更新

1. 小矩阵采用单核 Vector/SIMT Householder，按固定顺序归约和列更新，减少 launch 与打包开销。
2. 大矩阵每次分解宽 b 的 panel；每列先固定树归约范数，生成 τ/v，使用 Hᴴ 更新 panel 后续列。归约跨核时各核写独占 partial，第二阶段固定顺序合并，禁止浮点原子累加。
3. 生成 T，按 `Atrailing -= V (Tᴴ (Vᴴ Atrailing))` 更新尾矩阵；按列/二维输出 tile 分核，每个输出 tile 只有一个写者。大块用 Cube、Vector 完成实虚组合；小尾块用 Vector。
4. 上三角保存 R，严格下三角保存 v 的尾部，对角隐含 1 不写进 R。矮宽 m<n 时 p=m，仍更新所有剩余列。

#### Cungqr：构造显式 Q

先保存 k 列反射器有效下三角，再初始化输出为矩形单位阵。以块号从后向前左乘 Hblock，块内保留 H 的正向乘积，得到 `H1…Hk E(:,1:n)`。k<n 时新增列必须参与变换。无需为普通瘦 Q 永久分配 m×m 输出；验收需完整 Q 时以 n=m 调用。tau 不修改。

#### Cunmqr：四种酉变换

| side/trans | 执行块顺序 | 单块作用 |
| --- | --- | --- |
| LEFT/N | 从后向前 | C−V(T(VᴴC)) |
| LEFT/C | 从前向后 | C−V(Tᴴ(VᴴC)) |
| RIGHT/N | 从前向后 | C−((CV)T)Vᴴ |
| RIGHT/C | 从后向前 | C−((CV)Tᴴ)Vᴴ |

T 由同一 V/tau 构造，共轭仅施于公式指定位置；RIGHT 的 Q 阶数为 n，不能按 C 的行数 m 取 lda 或计算精度分母。更新按独立行/列条带分核，A/tau 原样保留。trans=T 在 Host 阶段拒绝。

#### CCgels：QR、QᴴB 与回代

在同一 stream 上执行内部 QR、LEFT/C 变换和上三角回代，不在 Host 搬回 A/B 或计算数值。B 先拷入 Y，原始 B 及其 padding 保持不变。回代按 rhs 分核或二维 tile 分核，使用稳定复数除法与固定累加顺序；X 仅写 n×nrhs 有效区。可复用此前 Ctrsm 的数值稳定、尾块及流水同步经验，但不直接依赖 ops-blas 私有源码或复制整套工程。

首版采用直接 COMPLEX64 分解求解，Host niter=0，符合任务书允许的直接路径；不得借该字段伪报迭代结果。对病态输入先分析残差，再决定是否引入有文档和计时覆盖的修正步骤。任务书未定义秩亏时的正 dinfo 语义，拟按首个零对角位置返回正索引并阻止除零，作为评审确认项，不把秩亏输入的任意 X 报成功。

#### 数据搬运、同步与确定性

外部列主序、内部块布局可不同，所有转换都使用实际 ld。GM→UB 尾部显式 rightPadding，按补齐后的 32B 间隔计算 UB stride；UB→GM 只写真实有效长度，不能为对齐扩大用户输出范围。检查 blockCount、blockLen、stride 的接口范围，参数过大时拆搬运。

不同 kernel 通过同 stream 顺序建立依赖；单 kernel 内以对应流水事件同步 CopyIn/Compute/CopyOut 与缓冲复用，不以 PIPE_ALL 掩盖竞争。跨核 scratch 独占且每次初始化，混合 AIC/AIV 的参与和退出分支一致，禁止部分核跳过配对等待。调优固定分派只依赖 shape/模式/资源，不依赖 case 编号；固定 reduction、分块累加次序与数学编译选项，避免调度改变浮点结果。每次调用恢复原始输入后验证确定性，不能反复分解上次输出。

### 工程组织和构建

```text
include/cann_ops_solver.h
include/cann_ops_solver_common.h
src/cgeqrf/arch35/{cgeqrf_host.cpp,cgeqrf_kernel.cpp}
src/cungqr/arch35/{cungqr_host.cpp,cungqr_kernel.cpp}
src/cunmqr/arch35/{cunmqr_host.cpp,cunmqr_kernel.cpp}
src/ccgels/arch35/{ccgels_host.cpp,ccgels_kernel.cpp}
src/utils/                         # 仅放确有多个调用者的复数 QR 声明/布局或公共实现
test/{cgeqrf,cungqr,cunmqr,ccgels}/ # 测试入口、CMakeLists、data、README
docs/zh/{cgeqrf,cungqr,cunmqr,ccgels}.md
docs/api_list.md
README.md
```

具体文件数量按公共原语复用结果收敛，避免每个小函数单独文件或在不同接口复制整套 QR。Host/Kernel 共用 launch 声明，头文件不含全局 `using namespace`。沿用源码自动发现与 arch35 选择；不为单个算子改全局优化选项。新增测试 CMake 使用仓库 TEST_NAMES 机制，完整验证库编译、run 包安装、安装后链接及 `--ops` 单算子选择的真实依赖。CCgels 的内部依赖在单算子构建时也须可链接；若自动选择不能满足，仅调整必要的依赖登记并解释原因。

官方输入目录名 `cgels` 映射到公开接口 CCgels 与仓库目录 `ccgels`，映射写入测试 README，不改原始 cases/bench 数据。仅提交实现、必要测试和使用文档；截图、msprof 原始数据、机器配置、临时工具和交付 ZIP 放仓库外。

## 支持硬件

| 硬件 | 范围 |
| --- | --- |
| Ascend 950PR / Atlas 950 | 本次开发和验收目标，须精度、功能、性能实测 |
| 其他硬件 | 不计入本次验收；不以架构兼容推断已实测 |

CANN 至少 9.0.0，并与所用源码 README 验证配套一致。当前仓列出 9.0.0 的 `20260422000325096`、`20260325000325538`；环境建立时选配套版本并记录实际补丁、编译器、驱动、CPU 架构和 NPU 型号。若改用 9.1.0，先验证配套并取得维护方确认，不能仅凭 Ctrsm 的 9.1.0 经验替代本仓依据。

## 算子约束限制

不新增 rank/broadcast/lda 紧凑限制，不用官方 case 列表限制合法维数。至少实现任务书规定的连续维数范围及零维语义，更大范围需通过溢出和资源检查。原地覆盖仅限任务书声明的 A 或 C；CCgels 要求 X 独立，不强制与 B 别名。调用者负责合法 Device 分配和非冲突别名、直到 stream 完成前保持数据/workspace 存活；同一 handle 并发改 stream 由调用者串行化。输入有效区域要求有限复数，无用 padding 不检查也不改写。

# 可维可测分析

## 精度标准/性能标准

### 精度判据

所有判定以任务书 3.2 为准，ε=2^-24。矩阵 1 范数取列绝对值和最大值，复数模为 hypot(real,imag)，不混用 Frobenius 范数或 `|real|+|imag|`。残差计算在 CPU complex128 下进行；ratio_cpu 必须来自同一 case 的 COMPLEX64 LAPACK 参考链路，不得拿 complex128 求解残差代替它。

| 接口/场景 | 残差 | 阈值 |
| --- | --- | --- |
| Cgeqrf | ‖R−QᴴA₀‖₁/(m‖A₀‖₁ε) | ≤max(5 ratio_cpu,0.1) |
| Cungqr | ‖I−QᴴQ‖₁/(mε) | ≤max(5 ratio_cpu,1.5) |
| Cunmqr | ‖op2−op1‖₁/(nq‖C₀‖₁ε) | ≤max(5 ratio_cpu,0.1) |
| CCgels，m=n | max_j ‖B₀,j−A₀Xj‖₁/(max(m,n)‖A₀‖₁‖Xj‖₁ε) | ≤max(5 ratio_cpu,0.015) |
| CCgels，m>n | ‖(B₀−A₀X)ᴴA₀‖₁/(‖A₀‖₁‖B₀‖₁max(m,n,nrhs)ε) | ≤10 ratio_opt_cpu |

Cgeqrf 的 Q 必须由 NPU 输出的 v/tau 升为 complex128，再用 CPU zungqr 构造 m×m 完整 Q；不能拿独立 CPU 分解的 Q 和 NPU R 混配。矩形 R 构造为 m×n 上梯形。

Cungqr 覆盖 k=n 与 k<n；除正交性门禁外，将输出与同一 v/tau 构造的参考 Q 比较，防止错误返回单位阵也通过正交性指标。Cunmqr 的 op2 使用同一 NPU v/tau，经已验证的 NPU Cungqr 生成 nq×nq 的完整 Q，再按 side/trans 计算显式乘积。依赖验证顺序是 Cgeqrf→Cungqr→Cunmqr→CCgels，不以未通过的 Cungqr 充当可信金标。

零分母单独分支：分子也为零则残差 0，否则判失败；NaN/Inf 残差不得比较后误判通过。超定判据 CPU 残差为零时仍遵循任务书零阈值，不私加 epsilon 下限；若造成平台数值歧义保留双端数据请任务方判定。零维测试以信息、输出和内存语义判断，不强行套除零公式。

### 官方用例和基线完整性

2026-10-14 对用户提供的原始 JSON 统计如下；用例匹配使用全部形状、ld、side/trans、k/nrhs 组成的键，不按行号盲配。side/trans 的数字与文本枚举规范化后匹配。

| 原始目录 | cases 数 | 有效 perf.avg_ms 数 | 处理 |
| --- | --- | --- | --- |
| cgeqrf | 174 | 174 | 全量精度及性能 |
| cungqr | 165 | 165 | 全量精度及性能 |
| cunmqr | 189 | 189 | 全量精度及性能 |
| cgels | 166 | 58 | 166 条功能精度；其余 108 条待补齐 GPU 基线 |
| 合计 | 694 | 586 | 不将基线缺失项计为通过 |

`cgels/bench_result.json` 顶层 failed=108，对应条目 `perf:null,error:2`，例如 m=18,n=8,nrhs=1。该数值错误码不在缺少采集环境信息时自行解释。需任务方提供 108 条有效 GPU 平均耗时及采集版本，或书面确认替代基线；基线补齐前无法证明全部性能达标。GPU 型号、CUDA/cuSolver 版本当前 JSON 也未提供，报告必须补齐来源信息，不编造版本。

### 性能测量与优化闭环

每条满足 `T_NPU_us ≤ 1000*perf.avg_ms/0.35`，等价 GPU/NPU≥0.35。官方 cases 的 iters=25；自测先固定预热次数（初始 10）及 25 次有效调用，记录实际口径并与基线采集方核对。以平均值而非最小值验收，所有用例逐条判定，不用总体均值掩盖慢例。

调用前恢复输入和清理状态，恢复/数据生成/CPU golden/Host 传输不放入算子计时区；bufferSize 单独查询不计 kernel。Cungqr/Cunmqr 的前置 geqrf 不计入其接口耗时；CCgels 内部 QR、变换、回代及必要搬运均计入。使用 msprof op 单 case 采集，按有效迭代归属统计 OpBasicInfo.csv Task Duration(us)。多 kernel 接口必须对每次调用内全部必要 kernel 求和，再对调用取平均，不能对不同 kernel 的单次耗时求平均或仅选最快 kernel；检查预热剔除、异步完成和流水覆盖。另报 ACL Event 端到端耗时用于交叉核查，不替代规定的 msprof 证据。

分别分析 panel 归约串行瓶颈、更新 GEMM 带宽/计算占比、搬运打包、尾块、launch 数和跨核等待；从 b、条带宽、双缓冲、短矩阵融合、布局复用逐项调优。每项改动均先精度/确定性再全量性能回归。记录每条原始 GPU 值、NPU 平均值、倍率、命中分支、采样数、源码 commit、库哈希及 profile 路径；不改官方阈值、金标或排除不利用例。

## 测试设计与可复现性

| 测试类别 | 覆盖和断言 |
| --- | --- |
| 官方全量 | 694 条全部关联参数；验证输出和 info，生成每条残差、CPU 残差、阈值及 PASS/FAIL |
| 小规模解析 | 1×1 纯实/纯虚、零列、单位阵、复数相位；验证反射器方向和 tau=0 路径 |
| 连续形状与尾块 | 1、2、2 的幂及幂−1，tile 边界±1，最大 4096；方阵/瘦高/矮宽；禁止只测对齐尺寸 |
| 属性 | k=0、1、n−1、n；LEFT/RIGHT×N/C；nrhs=0、1、8、32、128；T 和非法枚举报错 |
| padding/越界 | 每个 ld 最小、+8、+32，输入输出 padding 哨兵；合法偏移基址和不对齐尾部；B、tau、unmqr A 只读校验 |
| 异常参数 | 逐参数负维数、m<n 的 CCgels、非法 k、ld 不足、必需空指针、workspace 小于查询值、尺寸运算溢出；逐一核对 −i |
| 零维 | 所有合法零维组合、空数据指针；Cungqr k=0 验证矩形单位阵、Cunmqr k=0 验证 C 不变 |
| 确定性 | 代表形状和全属性重复至少 10 次；每次恢复同一输入，对有效输出、tau、info 逐位比较 |
| 流和资源 | 非默认 stream、两个 handle 独立 workspace；检查顺序和输出；库内无默认流偷跑或静态共享 scratch |
| 头文件/安装 | C 编译包含公共头并调用八接口；C++ 旧接口编译回归；符号、安装库链接和每个单算子构建 |
| 内存 | 无性能以外的内存硬门槛，仍报告 workspace/峰值、重复调用泄漏和越界检查结果 |

自构造数据的实部、虚部分别按任务书混合均匀 [-5,5] 与正态分布（μ∈[-5,5]、σ∈[0.1,2]），各占 50%；固定 seed，记录生成方法。自构造输入总内存控制在 4GB 内，CPU 完整 Q 参考和 profile 分 case 释放。补充有限大/小数、近相关列用于数值诊断，不把额外诊断取代正式分布。随机最小二乘 full-rank 主集与秩亏诊断集分开列示，绝不为了残差达标暗改官方输入构造。

## 兼容性分析

新增 POD、枚举和八接口置于正确的 C linkage 区域。公共类型头以 `<stddef.h>` 等 C 头替代仅 C++ 可用的 include；现有 `std::complex` 声明与 C++ include 用 `#ifdef __cplusplus` 包围，原有 C++ 原型、符号和返回值保持不变。C 调用者可见新增 C 接口与 handle 接口；不声称遗留 std::complex API 也被转换成 C API。针对这种必要的公共头调整单独做 ABI/编译回归，不顺带重排或格式化无关接口。

对标采用任务书明确的 legacy 参数与语义，参考 [NVIDIA cuSolver 文档](https://docs.nvidia.com/cuda/cusolver/index.html)；不因后续 CUDA 增添 generic API 而擅改本次原型。NPU 产品代码不依赖 CUDA，CPU SciPy/LAPACK 只用于测试金标。

## 交付及评审计划

1. 设计评审：本文件提交竞赛仓 Qianqiuer/docs/design.md，四接口统一说明；处理接口、数值与基线口径反馈后进入实现。
2. 工程打通：在配套 950PR 环境完成 C ABI、handle/stream、查询与校验、最小 Householder 链路；先以解析小 case 确认共轭顺序。
3. 完整实现：分块更新、ungqr/unmqr、CCgels、全部边界和确定性；逐接口执行指定残差验证。
4. 性能闭环：694 条均有设备结果，取得完整 GPU 基线后逐条满足 0.35；保存 msprof 明细和截图，不以未测设计估计代替证据。
5. 验收交付：设计 PR、自测代码及 README、按官方模板填写的 XLSX、精度/性能截图、原始日志、内存说明、环境版本、代码 commit、逐例结果、校验和与可复现步骤打包。按平台要求提供个人仓/分支/目录并邀请 Ascend-CANN；设计审核、邀请和实测状态以真实完成记录为准。
6. 上游贡献：验收通过后提交 ops-solver 需求 Issue 和代码 PR，四接口一并合入；保留评审修订和复测记录。

### 需在开发及验收前闭合的评审项

- 补齐 CCgels 的 108 条 GPU 基线及 GPU/CUDA/cuSolver 环境信息，不影响开始实现但影响全量性能结论。
- 确认有限值违规必须同步体现在 Host 状态码，还是作为输入契约通过 Device info 报告；本文按前者给出保守实现及完整计时，不隐去同步成本。
- 确认 CCgels 秩亏正 info 行为；合法 full-rank 的验收流程不依赖该待确认项。
- 任务书 Cungqr 的“前 k 列”文字与 m×n 原型不一致，本文按 m×n 输出、k 个反射器解释；Cunmqr 精度分母使用 Q 阶数 nq，RIGHT 不使用 C 的行数；k=0 的 Cungqr 仍写单位阵列。请评审确认这些与数学定义一致的边界解释。

## 参考材料

- 本任务书第 2～7 章及随附四组 cases.json / bench_result.json。
- [社区任务流程](https://gitcode.com/org/cann/discussions/39)、[设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
- [ops-solver 贡献指南](https://gitcode.com/cann/ops-solver/blob/master/CONTRIBUTING.md)、[目录规范](https://gitcode.com/cann/ops-solver/blob/master/docs/zh/install/dir_structure.md)、[版本配套](https://gitcode.com/cann/ops-solver/blob/master/README.md)。
- [官方自测报告模板](https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2)，开发完成后按模板填写真实结果。
