# 需求背景（required）

## 需求来源

本设计对应 [10月社区任务：单精度复数LU分解、求解和批量求逆（950）](https://www.hiascend.com/activities/task-center/details/237f144aab534448b58a123650b2fb35)。
目标代码仓为 [cann/ops-solver](https://gitcode.com/cann/ops-solver)，使用 Host C API 与 Ascend C/CATLASS Kernel 直调工程。

任务包包含 `Atlas950_Cgetrf_Cgetrs_CgetriBatched_CmatinvBatched_task_doc.md` 以及四类计算接口的用例和 GPU 基线。本设计以任务书第 2.3 节的完整原型为依据：四个计算接口和两个 workspace 查询接口。

**待确认的任务书差异**：文件名含 `Cgetrs`，正文部分位置写“五个接口”，并保留 `nrhs`、`B`、`trans` 测试说明；但第 2.1、2.3、7 节列出的计算接口均为 `Cgetrf`、`Cgetri`、`CgetriBatched`、`CmatinvBatched`，未提供 `Cgetrs` 原型及独立用例。本设计不自行增加或更换公开接口；如维护者确认还需 `Cgetrs`，补充其接口、用例和基线后同步扩展设计与实现。

## 背景介绍与现状

复数 LU 是稠密复数线性代数的基础。对矩阵进行部分主元分解后，可复用 LU 因子和主元序列完成单矩阵或批量求逆，避免重复分解。

以 ops-solver 提交 `a29362cfa605c69b1f330feadbff21e686eb9502` 为分析基线。现有 `src/cgetrf/cgetrf_host.cpp` 在 Host 解引用 info、搬运矩阵并同步 stream；`src/cgetri_batched/cgetri_batched_host.cpp` 接收连续批张量，并限制 n、batch 和 leading dimension。它们不能直接满足本任务的 Device 指针数组和异步接口语义。此次需要同时调整公开 ABI、内存归属、列主序、错误反馈和设备计算路径，而非仅修改函数名称。

本文为实现前的设计方案，描述拟采用的路径和验证方法；不将设计中的支持范围、资源预算或性能目标视为已经通过的测试结果。

# 需求分析（required）

## 接口与功能拆解

| 接口 | 输入 | 输出 | 必须满足的语义 |
| --- | --- | --- | --- |
| `aclsolverCgetrf_bufferSize` | m、n、lda、A | Host Lwork | 以 COMPLEX64 元素为单位查询 workspace |
| `aclsolverCgetrf` | 原始 m×n 矩阵 | 原地 LU、可选 ipiv、info | 方阵/矩形；部分主元或不选主元 |
| `aclsolverCgetri_bufferSize` | n、lda、A | Host Lwork | 查询单矩阵求逆 workspace |
| `aclsolverCgetri` | LU 因子及可选 ipiv | 原地逆矩阵、info | 只进行基于 LU 的求逆，不再次分解输入 |
| `aclsolverCgetriBatched` | Device LU 指针数组、可选主元数组 | Device 逆矩阵指针数组、infoArray | out-of-place；每个 batch 独立反馈 |
| `aclsolverCmatinvBatched` | Device 原矩阵指针数组 | Device 逆矩阵指针数组、info | 直接求逆；仅 0≤n≤32，n=33 报错 |

矩阵、workspace、ipiv、info 和指针数组均位于 Device；只有维数、handle、Lwork 查询输出位于 Host。矩阵采用列主序，元素地址为 `base + row + col * ld`，仅访问有效区域，不能覆盖 padding。

## 规模与边界

| 接口 | 最低支持范围 | leading dimension |
| --- | --- | --- |
| Cgetrf | m,n ∈ [1,4096] | lda≥max(1,m) |
| Cgetri | n ∈ [1,4096] | lda≥max(1,n) |
| CgetriBatched | n ∈ [1,4096]，batchSize ∈ [1,1000000] | lda、ldc≥max(1,n) |
| CmatinvBatched | n ∈ [1,32]，batchSize ∈ [1,1000000] | lda、lda_inv≥max(1,n) |

另支持合法空问题；自构造测试的输入总内存控制在任务书规定的 4 GB 范围内。批量维数与矩阵阶数的最大值不意味着需要同时物化其笛卡尔积。

任务包实际包含 Cgetrf 148 条、Cgetri 200 条、CgetriBatched 89 条和 CmatinvBatched 215 条，共 652 条，四份性能金标与各自用例条数一致。其中 Cgetrf 最大列数为 8168，CmatinvBatched 最大 batch 为 10000000，超过正文表中的最低支持范围。实现不得据正文上限直接拒绝这些已提供用例；workspace、索引和 batch chunk 规划同时覆盖上述实际范围，并请维护者确认最终全量验收清单。

复用 `aclsolverCreate/Destroy/SetStream/GetStream`。所有 kernel 下发至调用方 stream；输入和 workspace 的存活期须覆盖该 stream 上的计算。核心计算不能由 CPU fallback 代替。

# 详细设计（required）

## 公开类型与函数

在 `cann_ops_solver_common.h` 中提供 C 兼容类型，复数布局为连续两个 FLOAT32，大小 8 字节：

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
```

公开函数声明与任务书第 2.3 节保持一致，不交换参数或将 int 替换为 int64_t：

```c
aclsolverStatus_t aclsolverCgetrf_bufferSize(
    aclsolverHandle_t handle, int m, int n,
    aclFloatComplex *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverCgetrf(
    aclsolverHandle_t handle, int m, int n,
    aclFloatComplex *A, int lda, aclFloatComplex *Workspace,
    int *devIpiv, int *devInfo);

aclsolverStatus_t aclsolverCgetri_bufferSize(
    aclsolverHandle_t handle, int n,
    aclFloatComplex *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverCgetri(
    aclsolverHandle_t handle, int n, aclFloatComplex *A, int lda,
    int *devIpiv, aclFloatComplex *Workspace, int Lwork, int *devInfo);

aclsolverStatus_t aclsolverCgetriBatched(
    aclsolverHandle_t handle, int n,
    const aclFloatComplex *const Aarray[], int lda, const int *PivotArray,
    aclFloatComplex *const Carray[], int ldc, int *infoArray, int batchSize);

aclsolverStatus_t aclsolverCmatinvBatched(
    aclsolverHandle_t handle, int n,
    const aclFloatComplex *const A[], int lda,
    aclFloatComplex *const Ainv[], int lda_inv, int *info, int batchSize);
```

公共头文件中的 C++ 专用声明使用语言保护，新增类型与上述接口同时支持 C/C++ 调用。旧 Cgetrf/Cgetri 的同名不兼容原型不能与新 C ABI 同时导出；同步迁移原型调用点和测试，不通过重载掩盖 ABI 变化。

## 数学关系

### LU 分解

令 k=min(m,n)，采用 `P A = L U`，L 为 m×k 单位下梯形矩阵，U 为 k×n 上梯形矩阵。打包输出的严格下方存 L，其余存 U。

第 j 步在行 [j,m) 内选主元。初版对齐 LAPACK CGETF2 调用 ICAMAX 的规则，比较 `abs(real)+abs(imag)`，相等时取行号最小者；不能将其悄然换成复数欧氏模，因为两者可能选出不同主元。该规则来源于 [CGETF2](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/cgetf2.f) 和 [ICAMAX](https://www.netlib.org/blas/icamax.f)。主元序列写为 1-based 行号；`devIpiv == nullptr` 时完全跳过选主元和换行。验收以任务书残差为准，不假设不同后端必须生成逐位相同的 LU。

幅值比较采用避免溢出的缩放表示；复除法采用缩放分支，不能无条件计算分母的 `real²+imag²`。零主元按实部和虚部同时等于零判断，不用人为阈值把近奇异矩阵直接判为奇异。

### LU 因子求逆

由 `P A = L U` 可得 `A⁻¹ = U⁻¹ L⁻¹ P`。初始化右端 `X=P`，依次解 `L Y=P` 和 `U X=Y`。主元应用顺序与 getrf 一致；无主元时右端为 I。另一等价实现为先求 `U⁻¹L⁻¹` 再右乘 P，但不能混淆行交换与列交换及其顺序。

单矩阵 getri 固定采用“输入 A 保持 LU、workspace 保存完整输出 X、最后回写 A”的初始方案。所有右端列求解完毕之前不能覆盖 A 中的因子。批量 getri 从只读 LU 读取，以独立 C 保存右端与结果，不隐式执行 getrf。

构造 P 不必物化第二个稠密置换矩阵：初始化整数行映射 p[i]=i，按 j 递增交换 p[j] 与 p[ipiv[j]-1]，再令 X[i,c]=1 当且仅当 p[i]=c，其余为 0。选取至少两次不交换同一对行的主元序列验证次序，不能只用一次换行的测试证明正确。

### 小矩阵直接求逆

对 n≤32 的每个原矩阵，融合部分主元 LU 和两次三角求解。输出不覆盖原输入；首次零主元写入对应 info。n>32 在 Host 拒绝，不能转调 getriBatched。

## Host 设计

### 参数、错误和空问题

先验证 handle 与 Host 标量，再取得 stream。非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`；空 handle 返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。当 info 非空且有有效 stream 时，用轻量设备 kernel 写 `-i`，i 按公开计算接口排除 handle 后的参数顺序计算，不在 Host 解引用 Device info。

计算参数编号如下：

| 接口 | 编号（不含 handle） |
| --- | --- |
| Cgetrf | m=1，n=2，A=3，lda=4，Workspace=5，devIpiv=6，devInfo=7 |
| Cgetri | n=1，A=2，lda=3，devIpiv=4，Workspace=5，Lwork=6，devInfo=7 |
| CgetriBatched | n=1，Aarray=2，lda=3，PivotArray=4，Carray=5，ldc=6，infoArray=7，batchSize=8 |
| CmatinvBatched | n=1，A=2，lda=3，Ainv=4，lda_inv=5，info=6，batchSize=7 |

合法空矩阵不访问矩阵、主元及 workspace；单矩阵 info 写 0，batchSize=0 不写数组，n=0 且 batchSize>0 时对各 info 写 0。仍校验 handle、负维数及 leading dimension 的合法性；info 为空而存在应写的 info 元素时返回错误。

Device 指针数组不能在 Host 展开。错误检查分层处理：

- Host 检查可见的标量、顶层空指针、Lwork 以及尺寸/地址偏移的算术溢出；bufferSize 仅做规划，不读取 A 内容或启动计算 kernel。
- 在指针数组本身可访问的前提下，设备预检可读取其中的地址值，检查空元素、可判定的输入输出区间重叠，以及主元是否落在当前步骤允许的范围；失败矩阵不进入后续计算。
- 仅凭非零地址和对齐不能证明其属于有效 Device 分配，也不能证明分配长度足够。此类属性须依赖运行时可用的内存查询能力与调用方约定；设备 kernel 不通过试读任意地址来“验证指针”。同理，仅检查 A[i]/C[i] 不能证明所有跨 batch 区间互不重叠。
- 主元模式的一致性依赖调用方沿用 getrf 的约定；接口没有携带分解历史，不能声称仅凭 LU 数值就能判定它此前是否选过主元。

所有矩阵跨度、`b*n` 主元偏移和指针数组偏移均先宽化再计算。对空问题先走空分支，避免在 n=0 时计算 `(n-1)*ld`。

**需与维护者确认的错误反馈口径**：任务书要求非法值返回 INVALID_VALUE，同时采用 Device 输入和异步执行；批量 info 的参数表只列 0/k，而总则又要求错误写 -i。建议 Host 可判定错误由返回值报告，Device 数据错误由 info 报告，并确认批量负值约定、跨 batch 重叠的检查范围。如要求依赖 Device 内容的错误也在函数返回前报告，则预检路径必须读取结果并同步；文档、测试和计时均显式记录该行为，不能同时声称全程无同步。

内存分配失败与执行失败分别映射 `ACLSOLVER_STATUS_ALLOC_FAILED`、`ACLSOLVER_STATUS_EXECUTION_FAILED`，不混同为参数非法。返回成功仅表示异步下发成功，不表示尚未完成的设备计算或残差检查已经通过。

### Workspace 与生命周期

bufferSize 与执行端调用同一 workspace 规划函数；中间乘法、加法、对齐使用带溢出检查的 64-bit 无符号算术。查询返回 COMPLEX64 元素数，最终检查是否可由 int 表示。

工作区按字节规划、按复数元素数返回。令 align(x) 为按所选 kernel 实际要求对齐后的字节数，b 为面板宽度，q 为主元归约分组数，s 为归约记录大小；规划器保留以下区域：

| 路径 | 初始资源预算与用途 |
| --- | --- |
| getrf | 面板最多 `8*m*b` 字节；归约记录 `q*s` 字节；L21/U12 的实虚打包上界 `8*(m*b+b*n)` 字节；另加由实际 GEMM tiling 查询得到的临时空间 |
| getri | 完整输出 `8*n*n` 字节；行映射 `4*n` 字节；另加三角求解/GEMM tile 临时空间。不再同时复制第二份完整 LU |
| getriBatched | 复用用户 C 作为右端/结果，内部仅保存有界并行矩阵组的行映射、状态和 tile 暂存 |
| matinvBatched | 片上同时保有 LU 与 X，共 `16*n*n` 字节，再计入主元、行映射、向量临时区和对齐开销 |

上表是设计预算，不是某个硬件的可用容量或性能结论。各区分别对齐后求和，`Lwork=ceil(totalBytes/8)`；查询和计算端共享同一版本的规划器，不以 `lda*n` 代替有效数据所需的暂存。getrf 查询没有主元模式参数，统一预留可选主元路径所需容量。只有生命周期不重叠的区域才可复用，首版不引入跨阶段双缓冲；后续若增加流水重叠，需同步扩大查询值。

getrf 没有容量参数，只能要求调用方按查询值分配；不得声称执行端可仅凭指针发现缓冲区偏小。getri 校验 Lwork 不小于同参数查询值。batch 接口无用户 workspace，优先采用片上工作区；大矩阵以有界 chunk 分配暂存，不按最大 batch 物化全部中间矩阵。

当前 `_aclsolver_handle` 只有 stream 字段，尚无缓存设施。拟在内部实现中增加暂存槽的容量、所属 stream 和完成事件，不改公开 handle ABI：同 stream 的复用服从下发顺序；切换 stream 时旧槽保留到完成事件满足，不能立即交给新 stream 覆写；destroy 必须等待仍引用槽的工作结束再释放。首次扩容、内存不足和事件操作失败均显式返回错误；资源不足时允许缩小 chunk，但不改变数学路径或退回 CPU。多 handle 隔离资源，同一 handle 的并发 Host 调用需要内部互斥保护，不持锁等待无关 stream。

### 分发与 tiling

分发依据为矩阵维数、batchSize、padding、是否选主元和设备资源。初始分为小矩阵独立求解与大矩阵分块求解两类；切换阈值与面板宽度由真机单变量实验确定，不将待测参数宣称为最优值。

tiling 传递 m/n、lda/ldc、batch 范围、面板起点和有效宽度、workspace 偏移。硬件核数与片上容量通过平台查询获得，不硬编码旧原型的核数。尾块按有效行列掩码访问，padding 不参与归约或计算。

## Kernel 设计

### Cgetrf：分块右看 LU

1. 面板分解：处理从当前行至末行的完整高面板，而非仅左上 b×b 方块；逐列选主元、记录全局行号，完成乘子和面板内更新。
2. 换行：按面板内主元产生的顺序，将交换应用到之前已写入的 L 列和右侧未处理列。面板内已经执行过的交换不能再次执行。
3. 三角求解：计算块行 `U12 = L11⁻¹ A12`。
4. 尾矩阵更新：`A22 -= L21 U12`。
5. 进入下一面板直至 min(m,n)，同时处理矩形最后块。

面板内主元依赖串行，各独立行/列向量操作并行；尾矩阵按二维 tile 分配核。初版下发依赖为“预检 → 面板分解 → 面板外换行 → TRSM → 尾更新 → 下一面板”，每一步均在同一 stream 排队，不使用无界全局自旋屏障。每个输出 tile 由一个确定的核负责，避免原子累加次序引入不确定性。

零主元记录第一个 1-based 位置，禁止继续除以该零值；仍按任务语义完成可定义的剩余步骤。不同矩阵的 info 互不覆盖。主元归约使用固定分组和固定合并次序，不以竞争性浮点原子操作决定结果。

### Cgetri / CgetriBatched：分块三角求解

小矩阵将一个或若干完整矩阵分配到一个核，局部缓冲保存解向量和必要的三角块。大矩阵沿右端列分块，各列块独立，但同一右端列块沿行方向存在三角依赖：先从上至下解单位对角 L，再从下至上解 U。每个行块先用已求解的行更新当前右端，再求解对角块；单位对角 L 不读取 packed LU 中实际存放 U 对角线的位置作为 L 对角线。GEMM 仅用于已满足依赖的块更新，不让尚未求解的块参与计算。

先扫描 U 的对角线，记录最小零对角索引；失败矩阵跳过求逆，输出逆矩阵不声明有效。批量输入按 `Aarray[b]`、`Carray[b]` 获取基址，主元偏移为 b*n，不假定相邻矩阵连续存储。

### CmatinvBatched：融合小矩阵路径

对每个矩阵局部完成 LU、置换和求逆，降低多次 kernel 下发开销。n=32 时，单矩阵 LU 与 X 的基本存储已占 16384 字节，必须另计临时区及编译器资源开销，不能仅据矩阵元素数推断可装入的矩阵个数。以实际可用片上容量决定一核保有量；若融合路径资源不足，使用有界 Device 暂存的分阶段路径，不能因此拒绝合法 n≤32 用例。

batch 使用固定静态分配或确定性索引循环，例如核 c 处理 c、c+coreCount、c+2*coreCount 等矩阵；每个矩阵恰好一个拥有者。最后不足一组的矩阵和小于硬件对齐粒度的行列使用有效长度控制，不越界访存。极大 batch 的 chunk 只切分调度范围，不省略矩阵，也不复制整批指针数组到 Host。

### 复数矩阵更新与数值精度

将复数 GEMM 写成四个 FP32 实数积：

```text
Re(AB) = Ar Br - Ai Bi
Im(AB) = Ar Bi + Ai Br
```

优先使用 FP32 计算及累加语义，避免未经残差验证即改用半精度或截断乘法。实虚拆分、布局变换和结果回写都计入算子计算路径。初版建立确定性正确实现，再分别实验面板宽度、打包复用和流水重叠；每次只验证一个主要假设。

## 工程集成与支持硬件

目标硬件仅声明 Ascend 950PR，适配仓库对应的 arch35 编译配置；其他平台保持既有构建行为，不能因新增同名符号产生重复链接。

修改范围主要为 `include/cann_ops_solver*.h`、四个 `src` 算子目录、对应 `test`、`docs/zh`、README 及接口列表。公共复数与 LU/TRSM 辅助逻辑放现有公共目录，避免拷贝出多个高度重复的大文件。保留上游许可证与复用来源。

# 可维可测分析

## 精度标准

按任务书第 3.2.1 节使用 LAPACK 风格残差，ε=2⁻²⁴，参考 ratio_cpu 来自同 case 的 COMPLEX64 CPU 参考链路。complex128 可用于残差求值和诊断，但不能替换同精度参考 ratio_cpu。

| 接口 | NPU 残差比值 | 上限 |
| --- | --- | --- |
| Cgetrf | 逆序回放 ipiv 后的 ‖LU−A‖₁/(n‖A‖₁ε) | max(5 ratio_cpu, 0.5) |
| Cgetri | ‖I−CA‖₁/(n‖A‖₁‖C‖₁ε) | max(5 ratio_cpu, 0.05) |
| CgetriBatched | 每矩阵独立计算同上 | 每矩阵 max(5 ratio_cpu_i, 0.05) |
| CmatinvBatched | 每矩阵独立计算同上 | 每矩阵 max(5 ratio_cpu_i, 0.5) |

getrf 矩形情况分母中的 n 取列数。getri 的 A 从输入 LU 与主元恢复，不把 packed LU 直接当作原矩阵。批量任一矩阵不通过即该用例不通过。非有限结果单独判失败，零分母与奇异矩阵以明确分支处理，不能依靠 NaN 比较得到 PASS。

## 性能标准与测量设计

以各接口目录的 `bench_result.json` 为 GPU 金标，逐条关联 shape、dtype、leading dimension、主元模式及 batch。每个 case 要求 `T_NPU ≤ T_GPU / 0.35`，不是只测任务正文列出的九个示例。

950PR 使用 msprof 获取 kernel 时间，记录预热、迭代数和输入重置；原地接口每轮都从相同输入开始。多 kernel 实现保留全部阶段及每次 API 调用边界，同时报告各 kernel 均值和完整调用内设备计算总和，不以最快的单个 kernel 代替整个接口。计时聚合口径在评审中确认后固定，保持与 GPU 基线可比。

CPU 参考只用于精度，模拟器或 mock 时间不作为 NPU 性能依据。性能比较必须关联候选提交、构建参数、型号、CANN/驱动版本和原始 profiling 证据。

## 覆盖规划

覆盖任务包全部 case，并增加：矩形、2 的幂及减一维数、lda/ldc 的 +8/+32 padding、非连续 Device 指针数组、主元/无主元、奇异/近奇异、非法维数和 workspace、空问题、n=32/33、batch 边界及非默认 stream。

初始化 padding 哨兵并检查保持不变；相同输入在同一 stream 串行重复执行，对输出有效元素、ipiv、info 做 bit-wise 比较。大 batch 采用有界内存测试，并覆盖 chunk 边界。随机数据采用固定种子及任务书要求的均匀/正态分布，同时覆盖非对角占优输入。

关键设计约束与回归项对应如下：

| 设计约束 | 回归方法 |
| --- | --- |
| C ABI 与列主序 | 独立 C/C++ 最小调用工程；编译期检查复数大小和字段偏移；用非对称复矩阵排除隐含转置 |
| 主元及置换顺序 | 相等幅值取最小行号、两次不对易行交换、无主元、首个零主元、近奇异但非零主元 |
| 完整残差链路 | 保留原矩阵和种子；逆序回放 LU 行交换；明确区分原始矩阵与 packed LU |
| 资源与异步 | 查询/执行容量一致；Lwork 少一元素；连续调用、stream 切换、destroy 前未完成工作、多 handle 并行 |
| 设备数组与大 batch | 分散分配的矩阵、允许复用的只读输入、禁止重叠的输出、空元素、非法主元、chunk 前后边界 |
| 全量测试可追踪 | 每条记录保留接口、原始 case 标识、完整参数、种子、CPU/NPU 残差、GPU/NPU 时间和失败原因，不按 shape 去重 |

Host 单测用于参数编号、溢出、workspace 元素单位和空问题；NPU 测试用于设备指针语义、数值结果、确定性、异步依赖和性能。先通过参数/小矩阵、再通过矩形/padding 与分块边界，最后运行全部 652 条用例；性能实验开始前固定可复现的正确性候选。仅 Host 单测通过不能推导 NPU 验收通过。

## 兼容性与风险

旧接口升级属于有意的 ABI 迁移，需同步公开头文件、示例、测试与接口文档。既有非本任务算子保持原语义；不引入 CUDA 运行时依赖。

主要风险是大矩阵主元面板串行瓶颈、异步 Device 错误返回约定、复数打包开销和 workspace 生命周期。通过明确接口约定、分阶段设备实现、残差/确定性回归以及逐 case 性能测量控制风险。涉及 Cgetrs 的任务范围差异需维护者确认后收敛，不能在验收时隐去。
