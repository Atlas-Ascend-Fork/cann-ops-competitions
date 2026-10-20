# 单精度复数 QR 分解与最小二乘（950）接口与验收设计

## 一、需求背景

### 1.1 需求来源

对应[社区任务：单精度复数QR分解与最小二乘（950）](https://www.hiascend.com/activities/task-center/details/d72c98bf72254ca192abb098035d17e4)，开发身份为 `chenpinbo`。以[2026-10-17官方任务附件](https://www.hiascend.com/p/resource/202610/3f6f4855cfd04d0f971282ffc79f2872.zip)中的任务书和各 `op_name/package` 为依据；该附件SHA256为 `8ef06495858beb87f05b2010221efd417e69e07a404f985838f14ff465647268`。

本文沿用[官方模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)的需求背景、需求分析、详细设计及可维可测结构，约定公开接口、输入输出、边界和验收方法。实现方案、Host/Kernel分工及关键取舍见第3.5节链接的私有设计补充，供具有验收仓权限的评审人员审阅。

### 1.2 目标与交付范围

在Ascend 950PR上提供 `aclsolverCgeqrf`、`aclsolverCungqr`、`aclsolverCunmqr`、`aclsolverCCgels` 及各自的 `_bufferSize` 查询接口。四项作为同一任务整体交付，代码目标仓为[ops-solver](https://gitcode.com/cann/ops-solver)。cuSolver仅作为接口、语义与性能标杆，不作为运行时依赖。

## 二、需求分析

### 2.1 数据类型、布局与形状

矩阵类型统一为COMPLEX64，使用列主序。元素地址为 `base[row + col * ld]`。复数实部、虚部均为float32，ABI总大小为8字节。维数、leading dimension和枚举遵循公开原型。

| 接口 | 公开功能与输出 | 形状及关键约束 |
| --- | --- | --- |
| Cgeqrf | A原地输出R和标准反射器表示，TAU输出反射器系数。 | m,n≥0；覆盖m>n、m=n、m<n；lda≥max(1,m)。 |
| Cungqr | 根据A/tau中的标准反射器表示，原地输出Q的前n列。 | m≥n≥k≥0；lda≥max(1,m)；不能将输出限制为前k列。 |
| Cunmqr | C原地输出op(Q)C或Cop(Q)。A/tau为只读反射器输入。 | C为m×n；LEFT时nq=m，RIGHT时nq=n；0≤k≤nq；lda≥max(1,nq)，ldc≥max(1,m)；trans仅支持N/C，T报错。 |
| CCgels | 求最小二乘解X；X为独立n×nrhs输出，B保持不变，A允许被覆盖。 | m≥n≥0，nrhs≥0；ldda、lddb≥max(1,m)，lddx≥max(1,n)。 |

任务覆盖m,n≤4096，CCgels的nrhs≤128。合法padding应保持不变。不增加其他dtype、batch或广播接口；已声明的布局和边界支持不因官方清单未覆盖某个组合而取消。

Cungqr/Cunmqr的A和tau应符合Cgeqrf的输出表示，不能直接传入未分解的原始A。官方测试包准备的兼容反射器输入可直接用于独立接口验收。

### 2.2 环境与工程约定

目标硬件为Ascend 950PR；采用任务书规定的CANN 9.0.0或以上配套工具链，测试报告记录实际CANN、驱动、芯片、Python和数值库版本。NumPy/SciPy仅用于测试端输入准备和数值验证；算子运行时不依赖CUDA。

公开接口、目录、构建和测试遵循ops-solver规范，复用其handle与stream接口。数据准备、CPU参考和测试日志不构成算子运行时接口的一部分。

## 三、详细设计：接口与行为约定

### 3.1 公共类型与函数原型

公共声明位于 `include/cann_ops_solver.h` 和 `include/cann_ops_solver_common.h`，支持C及C++调用。以下为完整八接口原型，参数顺序和const属性与交付头文件一致。

```c
typedef struct {
    float real;
    float imag;
} aclFloatComplex;

typedef enum {
    ACLSOLVER_OP_N = 0,
    ACLSOLVER_OP_T = 1,
    ACLSOLVER_OP_C = 2
} aclsolverOperation_t;

typedef enum {
    ACLSOLVER_SIDE_LEFT = 0,
    ACLSOLVER_SIDE_RIGHT = 1
} aclsolverSideMode_t;

aclsolverStatus_t aclsolverCgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, aclFloatComplex *A, int lda, int *Lwork);

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
    int m, int n, int k, const aclFloatComplex *A, int lda, const aclFloatComplex *tau,
    const aclFloatComplex *C, int ldc, int *lwork);

aclsolverStatus_t aclsolverCunmqr(
    aclsolverHandle_t handle, aclsolverSideMode_t side, aclsolverOperation_t trans,
    int m, int n, int k, const aclFloatComplex *A, int lda, const aclFloatComplex *tau,
    aclFloatComplex *C, int ldc, aclFloatComplex *work, int lwork, int *devInfo);

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

### 3.2 参数归属与工作区

| 参数 | 内存归属及约定 |
| --- | --- |
| handle、维数、ld、side/trans | Host参数；使用handle设置的调用方stream。 |
| A、TAU/tau、C、dA/dB/dX | 合法Device分配，长度和类型符合原型及矩阵形状。 |
| Cgeqrf/Cungqr/Cunmqr的Lwork/lwork查询输出 | Host int指针，容量单位为 `aclFloatComplex` 元素数；对应计算工作区为Device缓冲。 |
| CCgels的lwork_bytes查询输出 | Host size_t指针，容量单位为字节；dWorkspace为Device缓冲。查询参数dwork可为空。 |
| niter | Host int输出，报告实际迭代次数；可以为0，不约定所有成功路径固定返回0。参数错误路径不作为成功的niter结果使用。 |
| devInfo/dinfo | Device int输出；成功为0，非法参数在可写时为-i；数值失败与参数非法分开判定。 |

先查询、再按返回容量分配并调用计算接口。查询不读取矩阵内容；计算不超出已查询容量访问工作区。输入、输出及工作区的有效分配和大小由调用方保证，裸指针不携带可供接口自动识别的dtype或长度。不同写入缓冲不得重叠，X不得与B共用输出存储。

调用方须在读取Device结果前同步对应stream；同一异步操作完成前不得复用其写入缓冲。重复使用工作区、非默认stream和连续调用的语义均应与公开接口一致。

### 3.3 参数错误、数值失败与空问题

非法参数序号不计handle，按计算原型从1开始；API状态码与Device info分别记录，不以预期值代替实际Device输出。

| 接口 | 参数编号 |
| --- | --- |
| Cgeqrf | m=1，n=2，A=3，lda=4，TAU=5，Workspace=6，Lwork=7，devInfo=8。 |
| Cungqr | m=1，n=2，k=3，A=4，lda=5，tau=6，work=7，lwork=8，devInfo=9。 |
| Cunmqr | side=1，trans=2，m=3，n=4，k=5，A=6，lda=7，tau=8，C=9，ldc=10，work=11，lwork=12，devInfo=13。 |
| CCgels | m=1，n=2，nrhs=3，dA=4，ldda=5，dB=6，lddb=7，dX=8，lddx=9，dWorkspace=10，lwork_bytes=11，niter=12，dinfo=13。 |

负维数、非法枚举、非法leading dimension、必需空指针、工作区不足及非有限值按接口要求返回错误。handle为空或Device info不可写时，不承诺写入无法访问的Device输出。

空问题在其标量约束合法时成功返回并写info=0。Cunmqr的k=0保持C；Cungqr在m,n>0且k=0时应输出单位矩阵的前n列。最小二乘数值失败以正info与成功状态区分，此时不保证解有效，不将满秩擅自增加为输入限制。任务书未明示的数值失败编码需在验收时单独对齐。

### 3.4 支持硬件与兼容性

本任务的精度与性能验收在Ascend 950PR执行，其他芯片、CPU计算或仿真结果不替代目标设备验收。新增接口保持既有ops-solver接口的ABI与调用方式；正式构建、正式测试及接口文档遵循目标仓规范。

### 3.5 实现方案与分工

算法、Host/Kernel分工、工作区与并行组织、精度及确定性设计集中在私有验收分支的[实现设计说明](https://gitcode.com/chenpinbo223/ops-solver-complex-qr-950/blob/delivery/complex-qr-least-squares-950/src/ccgels/acceptance/DESIGN.md)。该说明与本文一起构成送审内容，公开稿继续限定为接口与验收约定；文档范围是否满足要求以设计评审结论为准，不以自测通过代替评审。

## 四、可维可测分析

### 4.1 官方用例和输入来源

以2026-10-17附件内有效package清单为本轮自验集合，不使用根目录旧参考表替换package。后续官方清单若变化，应明确记录新版本及差异。

| 官方package | 数值用例 | 派生info检查 | GPU性能基线 |
| --- | ---: | ---: | ---: |
| cgeqrf | 174 | 3 | 174 |
| cungqr | 165 | 3 | 165 |
| cunmqr | 189 | 3 | 189 |
| cgels（仓内接口目录为ccgels） | 166 | 3 | 166 |
| 合计 | 694 | 12 | 694 |

输入使用原始 `canonical_cases.json`、逐case seed及 `gen_data.py`；按原型转换为列主序，不能变更数值内容或矩阵参数。Cungqr/Cunmqr使用包内准备的A32/tau32。冻结CPU参考以 `cases/index.json` 为准，不以本机重算值替换阈值。

12条info检查与694条数值/性能用例分别计数。info检查须真实触发对应非法参数，读取实际API状态及同步后的Device info，并按包内 `k_expected` 判定。

### 4.2 精度标准

严格采用原始 `verify_accuracy.py` 及任务书规定的LAPACK残差判据，ε=2^-24；复数取模并以complex128进行测试端残差计算。CPU基准值取官方冻结记录。

| 接口或场景 | 残差判据 | 通过门槛 |
| --- | --- | --- |
| Cgeqrf | CQRT01，比较R与QᴴA的归一化1-范数残差。 | ratio≤max(5×ratio_cpu,0.1)。 |
| Cungqr | CQRT02，检查Q的酉正交性，并满足输入反射器对应的输出语义。 | ratio≤max(5×ratio_cpu,1.5)。 |
| Cunmqr | CQRT03，比较输出与对应op(Q)施加结果。 | ratio≤max(5×ratio_cpu,0.1)。 |
| CCgels方阵 | CQRT16，各RHS目标残差逐列计算并取最大值。 | ratio≤max(5×ratio_cpu,0.015)。 |
| CCgels超定 | CQRT17，正规方程最优性残差。 | ratio_opt≤10×ratio_opt_cpu，无额外绝对下限。 |

数值精度、真实info、输出布局、只读输入、padding及相同输入的逐位确定性分别检查。不通过精度时，不能仅以性能或更换CPU参考判通过。

### 4.3 性能与内存标准

每个正式case在同shape、dtype、列主序与对齐后的计时口径下，须满足 `T_NPU ≤ T_GPU / 0.35`。GPU基线取对应package的 `perf_baseline.json` 中 `avg_ms`，不以最小值替代平均值。

NPU使用配套CANN官方msopprof采集。一次完整公开接口涉及的全部kernel耗时均计入该次调用，再按实际调用次数计算平均；输入准备、查询、内存分配和结果报告不计入kernel耗时。不得遗漏内部下发、截断采集、漏测case，或用单个内部kernel耗时替代完整接口。

性能检查器 `verify_perf.py` 输出比值供核对；另逐case判断0.35门槛并核对694条完整覆盖，工具退出0不能单独作为全量通过依据。按任务书§3.4，内存不设独立性能门槛，workspace查询正确性和合法内存访问仍属于接口要求。

### 4.4 边界、组合调用与交付测试

在接口约定内检查矩形、部分反射器、LEFT/RIGHT、N/C、多RHS、非法参数、padding、空问题、重复执行和非默认stream。按更新任务书，自构造测试属于补充验证，不将其冒充package必跑数量，也不因不在必跑表内而撤销接口语义。

正式构建与测试使用ops-solver已有入口。任务专用case、自测代码、步骤和验收指南集中在代码分支 `src/ccgels/acceptance/`，在正式测试之外独立存放。私有源码使用[验收分支最新提交](https://gitcode.com/chenpinbo223/ops-solver-complex-qr-950/tree/delivery/complex-qr-least-squares-950)，不要求检出固定SHA；实际实测版本、日志、性能及精度成绩另记自测报告。

验收通过后、提交上游代码PR前删除临时验收辅助目录及其文档引用；正式构建和正式测试不得依赖被删除内容。设计评审、自测通过、提交验收和外部接受分别确认，本文不充当验收结果报告。
