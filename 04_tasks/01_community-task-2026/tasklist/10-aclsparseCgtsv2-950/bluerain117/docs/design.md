# 需求背景（required）

## 需求来源

- 社区任务：10月社区任务-aclsparseCgtsv2算子开发（950）。
- 任务编号：`3c8d06bce4f54557b895de27e3737e0f`；开发账号：`bluerain117`。
- [官方任务书及测试附件](https://www.hiascend.com/p/resource/202610/0deb445c978f428c9c92c9df6a7625c4.zip)，SHA256：`a57f0cae816c11f241607a73b1a59f3fd67520c82847529ccf4f6e539834a618`。
- [任务讨论帖](https://gitcode.com/cann/ops-sparse/discussions/25)；[社区流程](https://gitcode.com/org/cann/discussions/39)。
- 实现目标仓：[cann/ops-sparse](https://gitcode.com/cann/ops-sparse)，目标分支 `master`；开发起点 `e0580fa8991ab473c5cc06259c2c44f2db412aac`。
- [本修订源码私仓分支](https://gitcode.com/bluerain117/ops-sparse-cgtsv2-950/tree/product-wide64-code)，固定提交 `3fb0af79314968486d1d4a2c0a61c077e426b133`；[独立验证检查点](https://gitcode.com/bluerain117/ops-sparse-cgtsv2-950/tree/wide64-validation/validation_wide64)，固定提交 `fa84de7f4111ce7a594625ad25f2686d40bc3c22`。官方 Ascend-CANN 已具备该私仓 Developer 权限。

## 本次修订

本稿补充已合入的[设计 PR #2388](https://gitcode.com/cann/cann-ops-competitions/pull/2388)，并扩展当前[设计 PR #2515](https://gitcode.com/cann/cann-ops-competitions/pull/2515)的内部指数与大尺寸工作区说明。上游产品集成以 `2457742bc76ad5100ec9d7c088b72e23e1fe4225` 为基线；64 位指数修订源码为 `3fb0af79314968486d1d4a2c0a61c077e426b133`。本次新增大 `m` 分块工作区路径的方案。公开 `aclsparseComplex_t` 仍为 8 字节、float 对齐（4 字节）；函数原型、int32 维度及其范围、部分主元能力和 `Sgtsv2` ABI 均不变，不增加 `m` 上限。`n=0` 的 workspace 查询仍为 0。PR #2515 的 head `184f755d0dc942ac5e531469eedccb22ebd65be8` 曾通过 CI，但本次文档变更需要新的 CI；该 head 的 CI 不代表本次修订已验证。

对公开接口允许的超大维度作范围检查时，发现原 int32 中间指数的累加、差值和最终窄化存在可达的有符号整数溢出路径。修订采用实部/虚部分别携带的 int64 指数及窄化前范围分类，不通过增加 `m<=2^20` 限制规避问题。64 位内部状态使原 workspace 随 `m` 增长；为维持 workspace 查询不超过记录的目标 L2，本稿为旧查询超过 L2 的输入增补固定 tile 的分块 LU 重放。该路径保留部分主元及宽精度回代状态，不将公开数据类型或维度范围改成另一套接口。源码、设备验证和正式采集按候选分别记录；旧版本的未执行大尺寸例不当作设备失败。

## 背景介绍

`aclsparseCgtsv2` 求解一个 complex64 三对角矩阵对应的单个或多个右端项。接口对齐任务书指定的 cuSPARSE `cusparseCgtsv2`，采用 aclsparse Legacy C++ API 与 Ascend C Kernel 直调工程。函数名中的 C 表示单精度复数。

当前仓库的实数 `aclsparseSgtsv2(float*)` 可用于参考句柄、stream 和工程接入方式。本任务新增独立复数类型及 ABI，保留实数接口。核心分解、主元选择、前代和回代均在 Ascend 950 上执行。

# 需求分析（required）

## 需求描述

求解 `A X = B`。A 由长度为 m 的 `dl/d/du` 表示，其中 `dl[0]=0`、`du[m-1]=0`。B 按列主序存储，元素地址为 `B[i+j*ldb]`；输出 X 原地覆盖 B 的有效 m 行，`m<=i<ldb` 的 padding 保持原值。

支持运行时 `m>=3`、`n>=0`、`ldb>=m`。n 为 RHS 数，不是 batchCount。不提供独立 X，不支持广播、转置视图或任意 stride。输入系数只读。

## 需求拆解

1. 新增公开 `aclsparseComplex_t`、workspace 查询和求解函数。
2. 在 NPU 上按复数模长进行相邻行部分主元选取，维护交换产生的第二上对角；不能退化成无主元 Thomas。
3. 查询工作区后由调用方足量分配，并保证 128 字节地址对齐；遵守句柄 stream 的异步执行和生命周期约定。
4. 保证系数只读、B 原地、padding、非奇异系统确定性及完整错误参数校验；真正奇异和特殊值单独记录状态及输出。
5. 使用官方 ATK 验证精度、完整组合和特殊/合同场景；完成全部 203 条逐例性能及内存验证。
6. 公共代码不依赖 arch35 设备实现；与 A2/A3 后续实现共存时复用同一公共 ABI，并保留交叉回归。

# 详细设计（required）

## 算子分析

### 数学公式

`A[i,i]=d[i]`，`A[i,i-1]=dl[i]`，`A[i,i+1]=du[i]`，其余为零。对每一 RHS 列 j，求解 `A*X[:,j]=B[:,j]`。

### 支持数据类型与接口

仅支持 complex64；每个复数由连续的 FLOAT32 实部和虚部组成。CPU Golden 将同一份已量化 complex64 输入升为 complex128，使用 SciPy/LAPACK 带主元带状求解。

```cpp
typedef struct {
    float real;
    float imag;
} aclsparseComplex_t;

static_assert(sizeof(aclsparseComplex_t) == 8);
static_assert(alignof(aclsparseComplex_t) == alignof(float));

aclsparseStatus_t aclsparseCgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t* dl, const aclsparseComplex_t* d,
    const aclsparseComplex_t* du, const aclsparseComplex_t* B,
    int ldb, size_t* pBufferSizeInBytes);

aclsparseStatus_t aclsparseCgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const aclsparseComplex_t* dl, const aclsparseComplex_t* d,
    const aclsparseComplex_t* du, aclsparseComplex_t* B,
    int ldb, void* pBuffer);
```

| 参数 | 数据/布局 | 合同 |
|---|---|---|
| handle | 已创建的 aclsparse 句柄 | 空句柄返回 `ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR` |
| m/n/ldb | int32，运行时属性 | `m>=3,n>=0,ldb>=m`，不合法返回 `ACL_SPARSE_STATUS_INVALID_VALUE` |
| dl/d/du | `[m]` complex64 连续只读 | n>0 时非空；调用方保证边界系数置零及足够存储 |
| B | `[ldb,n]` complex64 列主序 | n>0 时非空；仅覆盖有效 m 行 |
| pBufferSizeInBytes | Host size_t 输出 | 必须非空；合法 n=0 写入 0 |
| pBuffer | 设备工作区 | n>0 时非空、128 字节对齐，并由调用方按查询值足量分配 |

raw C ABI 无法读取实际 dtype、stride、分配长度等元数据。这些由调用方或测试适配层保证；不承诺检测不可见的元数据错误。n=0 仍验证句柄、m/n/ldb 和查询输出指针，但允许数据指针及工作区为空，且不提交求解 Kernel。

## 算子实现

### 3.2.1 Host 侧设计

Host 仅完成可见参数校验、checked size_t 乘加、workspace 计算、取得句柄 stream 和 NPU Dispatch。不会将系数或 RHS 拷回 CPU 求解，不在执行接口中创建或销毁 stream，也不为数值状态执行隐式设备同步。公开 `m/n/ldb` 仍为 int32，`m>=3`、`n>=0`、`ldb>=m`；不新增 `m` 上限。`n=0` 查询 0 字节并跳过 Kernel。

检查 `ldb*n*8`、`m*8`、因子与 RHS scratch 总字节数、证书区及对齐加法是否溢出。查询函数不解引用设备数据。非法参数应在 Dispatch 前返回；查询失败不将未完成计算写成有效 workspace 大小。

对 `n>0`，按 3.2.3 节公式计算 `old_query`。当 `old_query<=134217728` 字节时，查询值和原 kernel/CR 调度保持原有方式；当 `old_query>134217728` 字节时，查询切换到固定 tile 的 bounded LU 工作区，执行也选择同一分支。两个阶段共用相同 Host 分支条件，避免查询与执行对工作区大小作出不同选择。该阈值是当前目标 L2 的字节值，不是公开 `m` 限制。

执行接口新增非清除式 `aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)` 检查：参数校验后、取得 stream 前检查一次，运行时查询后、kernel dispatch 前再检查；bounded 路径每次 kernel 提交后检查，原 void launcher 返回后也检查一次。当前可见的线程错误会映射为 `ACL_SPARSE_STATUS_EXECUTION_FAILED`，检查不清除 last error、不隐式同步。错误在部分 Kernel 入队后才可见时，B 可能已部分更新；调用方仍须在释放或复用输入、B、workspace 前同步对应 stream 并检查异步结果。该检查改变旧 Host 路径的错误返回条件和 Host 开销；原 launcher 内部多个提交无法由外围检查逐次截断，因此不将旧路径描述为行为或耗时完全不变。

内部 tiling 包含 `m,n,ldb,numBlocks,fastEligible,fastBlocks,wideSlots` 七个 int32 字段。每 block 使用 32 个 SIMT 线程；稳定路径按 RHS scratch 槽位和实际 AIV 数决定并行度，线程循环覆盖全部 RHS。并行循环约减（CR）最多使用 32 个 AIV block，且不超过实际可用 AIV 数。`m>=512,n>0` 仅表示值得执行设备证书检查，不能仅凭尺寸进入 CR。

### 3.2.2 Kernel 侧设计

实现包含三种调度结果：旧查询范围内的扩展精度部分主元 LU、安全证书控制的并行 CR，以及超 L2 旧查询触发的分块 LU 重放。三者均由同一公开接口调度；Host 不回读证书、不同步决定算法，也不执行 CPU 求解。

**旧查询范围的完整 LU 第一阶段：部分主元三对角分解。** 单个 SIMT 线程沿行推进，保存 U 的主对角、第一和第二上对角、消元乘子及相邻行交换标志。该路径的分解结果由所有 RHS 共享，不为每列重复分解或分配矩阵副本。

令当前主元、上对角为 p、u，下行系数为 l、q、v：

```text
for i = 0 .. m-2:
    if abs(p) >= abs(l):       # 模长相等时固定选择当前行
        f = l / p
        Udiag[i] = p; U1[i] = u; U2[i] = 0
        swap[i] = 0
        p = q - f*u
        u = v
    else:
        f = p / l
        Udiag[i] = l; U1[i] = q; U2[i] = v
        swap[i] = 1
        p = u - f*q
        u = -f*v
    Lfactor[i] = f
Udiag[m-1] = p
```

上述伪代码中的复数运算在稳定路径使用内部 `ScaledComplex`，不直接以普通 complex64 保存消元和回代中间值。每个实/虚分量分别保存两个 FLOAT32 尾数及独立的有符号 int64 指数，值表示为 `(hi+lo)*2^exponent`；每个内部实数 16 字节、每个内部复数 32 字节，均按 8 字节对齐。公开 `aclsparseComplex_t` 仍为 8 字节、float 对齐。使用 FMA 残差及补偿加法保存额外有效位，除法作商的残差修正；归一化后比较复数模长并选择主元。这样避免中间值超过 FLOAT32 范围后先产生 Inf、再因运算传播变成 NaN。实虚指数独立，不能因为某分量溢出而丢失另一有限分量。仅在最后写回 B 时收窄到 complex64；溢出分类按最终 FLOAT32 输出检查，精度仍按指定 complex128 参考验证。

指数差在排序后以 uint64 差值表示，检查其超过 100 后才将小差值转换为 int32 缩放量，保留原有尾数处理策略。`scale` 及最终 `narrow` 在指数相加和移位前先分类，使实际参与位操作的指数落在小范围内；不先把可能很大的指数窄化或相加。Host 和 Kernel 共享工作区偏移常量与对齐函数，Kernel 编译时明确标注 SIMT callee，避免结构体扩大后仍沿用旧偏移。

**稳定路径第二阶段：并行 RHS 前代与回代。** 每个活动线程使用独立的扩展精度 RHS scratch，读取共享分解，执行相同交换与消元，再用包含第二上对角的 U 回代；完成一列后复用该 scratch 处理后续列。下面 R 为内部暂存，不是新增公开输出：

```text
for j in thread_assigned_columns:
    R[:] = widen(B[0:m,j])
    for i = 0 .. m-2:
        if swap[i] == 0:
            R[i+1] -= Lfactor[i]*R[i]
        else:
            t = R[i]
            R[i] = R[i+1]
            R[i+1] = t - Lfactor[i]*R[i+1]
    R[m-1] /= Udiag[m-1]
    R[m-2] = (R[m-2]-U1[m-2]*R[m-1])/Udiag[m-2]
    for i = m-3 .. 0:
        R[i] = (R[i]-U1[i]*R[i+1]-U2[i]*R[i+2])/Udiag[i]
    B[0:m,j] = narrow_to_complex64(R[:])
```

**大 workspace 路径：分块 LU 重放。** 当 `old_query>134217728` 字节时，Kernel 不分配整列因子和 RHS scratch，也不执行 CR；对每个 RHS 串行地按 tile 顺序扫描，再由右向左重放 tile 并回代。tile 固定 `T=65536` 行，`q=ceil(m/T)`。前向扫描为每个 tile 保存起始 `pivot`、`upper`、前向 RHS 状态 `prev` 三个 `Wide` 值和 tile 首行原始 B（8 字节），不写 B。回代时在 tile 内按原行序重新计算因子和前向 RHS，使用跨 tile 检查点保存的原始边界 B，避免读取已被右侧 tile 覆盖的输出。两个回代后继值一直以 `Wide` 保留，只有最终写回时才收窄为 complex64。

这一路径复用现有 `ScaledComplex`/`Wide` 算术、每个实虚分量独立 int64 指数、部分主元比较和相同行交换逻辑；零下对角保持跳过，零乘子保持直接读取 `next` 的分支，pivot 交换生成的第二上对角在回代中继续使用。系数只读，B 仅写有效行，padding 不变。切换输入原先可能通过 CR 证书，因此 bounded LU 对这些输入的结果仍须按任务书精度标准验证；CPU 行级对照或其它版本 CR 的结果不替代这项验证。

分块入口通过同一 handle stream 按顺序提交。每个 RHS 提交 `q` 个前向 tile 和 `q` 个反向 tile，即 `2*q*n` 次 kernel 提交；当前 tile kernel 以单个活动线程逐行处理。算术复杂度为 `O(n*m)`，Host launch 数量随 `q*n` 增长，性能仍待独立测量。所有已提交 Kernel 的生命周期和错误处理见 3.2.1 节。

**并行 CR 路径。** 每次调用先在 NPU 上检查全部系数和有效 RHS：每个实/虚分量位于 `[-16,16]`，并且每行满足 `L(d[i]) >= U(dl[i])+U(du[i])+0.501`；边界外项视为零。这里 `L(z)=max(abs(real),abs(imag))` 是模长下界，`U(z)=max+0.415*min` 是保守上界。因此 NaN/Inf、界外数值或不满足严格对角占优裕量的输入都会拒绝快路径。

各检查线程独立写入自己的失败标志，再由后续同 stream Kernel 归约为总标志。在证书成功时，初始化 5 个 complex64 工作数组，按步长 `1,2,4,...` 做循环约减：只更新本层偶数活跃行，读取本层保持不变的奇数邻行，保存恢复所需系数。系数更新和 RHS 更新分为有序 Kernel，避免其他线程读取已经覆盖的系数；解出根行后逆序恢复被消去的行。RHS 仍完整保存在原 B 的有效区域，支持非 2 的幂 m 及全部 n，不改变公共问题规模。证书失败时 CR Kernel 不写系数或 B，完整部分主元路径接管求解；不能把无主元算法当作唯一实现。

使用 `simt_api/asc_simt.h` 的 SIMT Kernel、`asc_vf_call`、GM 加载/存储及复数算术辅助函数。公开类型仅保证 float 对齐，因此不能将数据地址擅自当作更高对齐的 float2 指针。行列偏移使用 64 位整数；循环访问范围不进入 B padding。

旧路径的分解、证书发布/消费、各 CR 层及求解通过同 stream Kernel 顺序建立依赖，不使用跨核自旋。每次调用重新写入所有参与归约的线程标志及总标志，不依赖旧 workspace 的状态；本次实际消费的因子和交换项亦重新写入。bounded 路径的检查点、控制区和 tile 临时数组均在调用方工作区中按前向后向提交顺序消费。未消费尾部不要求清零。所有路径的调用方均须在求解完成前保持 B、系数和 workspace 有效；完成后可将同一工作区用于后续调用。

### 3.2.3 Workspace 与本地资源

| 区域（旧查询路径） | 元素数 | 字节数 |
|---|---:|---:|
| U 主对角 | m ScaledComplex | 32m |
| U 第一上对角 | m ScaledComplex | 32m |
| U 第二上对角 | m ScaledComplex | 32m |
| 消元乘子 | m ScaledComplex | 32m |
| 相邻行交换标志 | m int32 | 4m |
| 证书线程标志和总标志 | 1025 int32 | 4100 |
| scratch 前对齐填充 | 紧随证书区 | 奇数 m 为 0，偶数 m 为 4 |
| 独立 RHS scratch | S 列、每列 m ScaledComplex | 32mS |
| `old_query` | `n>0`，仅在 `old_query<=134217728` 时返回 | `align_up(align_up(132*m+4100,8)+32*m*S,128)` |
| 空 RHS | `n=0` | 0 |

`S=min(n,max(1,floor(2097152/(32*m))),1024)`。2 MiB 是并行 RHS 暂存的预算，最低单列空间不可省略；实际活动槽位另受 AIV 数约束。CR 的 5 个 complex64 数组共 40m 字节，复用上述共享因子区，按证书选择与稳定路径互斥使用，不重复申请。证书区从 `132*m` 开始；RHS scratch 起点按 8 字节补齐，不能将四字节对齐的证书尾直接当成内部复数指针。上述完整旧查询公式只用于 `old_query<=134217728` 字节的输入；超过此值时不返回这个大查询，改走以下 bounded 布局。

bounded workspace 的 `T=65536`、`q=ceil(m/T)`，布局如下：

| 区域（bounded 路径） | 布局 | 字节数 |
|---|---|---:|
| tile 临时区 | 四个 `Wide[T]`：U 对角、第一上对角、第二上对角、前向 RHS | `128*T` |
| tile 检查点 | 每 tile 三个 32 B `Wide` 状态及两个原始 B float | `104*q` |
| 固定控制区 | tile 状态及两个未窄化回代后继值 | `4096` |
| bounded 查询 | 末尾按 128 B 对齐 | `align_up(128*T+104*q+4096,128)` |

`m<=INT32_MAX` 时 `q<=32768`，bounded 查询最大为 `11800576` 字节；`m=2^20,n=1` 查询为 `8394368` 字节，`m=7780739,n=1` 查询为 `8405120` 字节。分支由 `old_query` 决定：若它不超过 `134217728` 字节，返回未改变的 `old_query`；若超过该 L2 值，返回 bounded 查询。`n=0` 仍为零。因此，对所有通过现有 Host 参数及 size_t 溢出检查的输入，workspace 查询的算术上界不超过记录的 L2 值；这项结论只针对查询字节，不证明任意动态 shape 的设备正确性，也不代替对实际分配、allocator 增量或进程峰值的测量。

467 环境记录的目标 L2 为 `134217728` 字节。旧布局单独计算时，`m=2^20,n=1,ldb=m` 的 `old_query` 为 `171970688` 字节，`m=7780739,n=1,ldb=m` 为 `1276045312` 字节，二者均选择 bounded 分支；新方案实际返回值分别为 `8394368` 和 `8405120` 字节。`m=818375,n=1` 的 `old_query=134217600` 仍走旧分支；`m=818376,n=1` 的 `old_query=134217856` 切换到 bounded，返回 `8394112` 字节。上述数值用于说明查询选择，不代表这些 shape 均已实际分配或执行。

任务书 3.4 的内存要求为两个条件满足其一：超 500 MB 输入输出的同 Torch API 额外内存不超过 GPU 使用内存的 5%，或方案固有 workspace 不超过目标 L2。本方案的查询上界旨在满足第二个条件。官方[讨论25回复](https://gitcode.com/cann/ops-sparse/discussions/25#discussion-reply-a88a5c98e5654a91843b5418e3616d38)同时表示“内存验收只需通过包内203条性能用例标杆标准即可”和“内存验证标准严格任意合法动态m/n/ldb均需要满足，若203条性能用例不涉及超过500MB的用例，也不需要额外补充”。两段的适用关系仍待官方澄清；本设计不代官方裁决，也不新增无条件验收标准。查询算术显示通过现有 Host 检查的维度请求不超过记录 L2，但实际设备执行正确性、内存占用及资源可行性仍须分开验证。官方已确认包内 GPU CSV 是正式标杆，不需开发者另行提供新 GPU 标杆；NPU 203 条采集与任务书其他交付仍须完成。

SIMT 使用 GM 工作区及每线程标量寄存器，不分配显式 UB 队列：

| UB buffer | 数量 | 使用字节 |
|---|---:|---:|
| 显式输入/输出/临时 UB 队列 | 0 | 0 |

编译器寄存器或溢出资源以构建和设备记录核实，不把“没有显式 UB”解释为零硬件资源占用。此接口只支持 complex64，不引入 FP16/BF16 转换路径。

### 3.2.4 异常与确定性

非奇异输入采用固定的模长比较、相等决策及每列算术顺序，没有跨列浮点归约。恢复原始 B 后，在同一设备、软件和配置重复运行，逐字节比对有效区输出并保留复跑记录。

真正奇异系统不增加 Info 参数或更改 ABI。异步 Host 返回码与设备同步返回码、输出有限性分别记录；零除产生的 NaN 不作为常规精度通过依据。扩展数值类型对 Inf/NaN 另行处理，保持复数 IEEE 传播及零下对角的跳过语义，不把一个无穷分量直接归一成两个 NaN。Inf/-Inf/NaN 按官方特殊值套件和定向位置回归分别判分类、位置及符号，不比较 NaN payload；不能将“不崩溃”解释为特殊值语义通过。

### 3.2.5 性能计划

沿 m 顺序分解和回代存在依赖延迟，因此在经证书确认的数值域内用 CR 提供跨行并行，不安全输入使用完整扩展精度部分主元求解。证书的输入范围是运行时条件，不按用例 ID、种子或模型名称分派；更改数据后必须重新检查，不复用旧证书或旧因子。

`old_query>134217728` 时的 bounded 路径不使用 CR，逐个 RHS 执行 `2*q` 次按序提交，算术工作量为 `O(n*m)`。提交次数、每 kernel 仅单活动线程的当前实现及同 stream 排队成本可能降低大尺寸吞吐；需在该路径的独立候选上采集所有 kernel 总耗时、Event 与内存，当前没有性能通过结论。正式203条若仍在旧查询范围，继续按原路径对整次调用内所有 Kernel 求和。

正式性能对每次调用内的检查、归约、初始化、CR 各层、根求解、恢复以及条件式 LU/求解 Kernel 完整求和，再对 30 个正式样本取中位数。即使某个条件式 Kernel 因设备标志提前返回，其实际耗时仍计入；不能只取最大的 Kernel、丢弃慢项或跨调用复用因子，也不能用 Event 或 Host 墙钟替代 Kernel 耗时。

## 支持硬件

| 芯片/软件 | 本设计范围 |
|---|---|
| Ascend 950，DAV_3510，arch35，A5 | 实现及验收目标 |
| CANN 9.1.0 | 本修订实际构建及设备验证环境；其他配套版本另行验证 |
| Atlas A2/A3 | 不声明本 arch35 Kernel 可在该硬件运行；公共接口与对应实现需完成交叉回归 |

## 工程落点与约束

| 路径 | 内容 |
|---|---|
| `include/cann_ops_sparse.h` | 唯一公开复数类型、布局断言和两个接口 |
| `sparse/cgtsv2/` | 公共说明与可共享逻辑 |
| `sparse/cgtsv2/arch35/` | Host 调度、tiling、Ascend C Kernel |
| `test/cgtsv2/arch35/` | C++ UT/ST、真实 ABI 错误码和完整调用流程 |
| `test/cgtsv2/atk/` 及公共测试说明 | 官方 ATK 设计、适配、精度/性能/内存运行入口 |
| 相关 CMake | 按既有构建规则接入，不新增平行 build 工程 |

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 方法和完成条件 | 来源 |
|---|---|---|
| 发布精度 | 官方 200 条与独立生成的 800 条分别运行；同一 complex64 输入的 complex128 Golden | 任务书 3.2、3.5 |
| 混合容差 | 实/虚分别 `atol=rtol=2^-13`，匹配率均≥0.99；各自最大有限绝对误差≤`max(1e-2,32*ULP(g_low))`；按原比较器处理 RNE 溢出，常规 NaN 不计通过 | 任务书 3.2 |
| 完整组合 | 原生 ATK Tensor/Attr YAML、固定种子、约束插件和只读覆盖审计；覆盖全部有效有限因子域，不能用 200+800 替代组合覆盖 | 任务书 3.5及附件 |
| 特殊/合同 | 合法空 RHS、上下边界、真正奇异及特殊值独立记录；真实 C++ ABI 验证非法指针、对齐、溢出和 Dispatch | 任务书 2.4、3.2、3.5 |
| 性能 | 保留 203 条原始身份；全部 `GPU Kernel总耗时中位数/NPU Kernel总耗时中位数>=0.3`，无均值兜底 | 任务书 3.3 |
| 采样 | 每例至少10次预热、30次正式采样；每次先恢复 B，workspace复用；按一次调用关联全部求解 Kernel 后求和，再取中位数 | 任务书 3.3 |
| Event | 同 stream Event 包围求解、排除 B 恢复，独立列示，不能参与正式倍率判定 | 任务书 3.3 |
| 内存 | 每例记录IO、查询与实际对齐分配workspace、B恢复副本、allocator基线/峰值/增量及free/total；满足任务书两项内存条件之一 | 任务书 3.4 |
| 工程与回归 | 系数只读、原地/padding、确定性、异步stream、红区、泄漏、完整生命周期、Sgtsv2及A2/A3公共路径回归 | 任务书 2.1、3.5、5 |

ATK 使用 26.9.8，对应源码 `9504bee6076820ca74259fb919a790a12ef88cf8`，Python 3.10/3.11、NumPy<2。case 创建、Tensor 分布与序列化由原生 `atk case` 完成，通过 `atk task` 执行；独立生成脚本不能替代正式入口。精度输入遵守四输入正交分布，只置零接口边界系数，不做对角占优变换；性能保留冻结清单的历史输入分支和种子。

三条主场景的 NPU Kernel 总耗时上限分别为：P-01 `640.896/0.3=2136.32 us`，P-02 `628.944/0.3=2096.48 us`，P-03 `1220.048/0.3≈4066.827 us`。这些是验收目标，不是 NPU 实测值。报告保留全部失败、超时、缺失和未执行行；大规格不能因资源限制从覆盖域删除。

## 兼容性分析

公开类型与声明只新增一次，不修改 `aclsparseSgtsv2(float*)` 的声明和语义。后续 A2/A3 Cgtsv2 PR 先合入时，基于最新 master 合并公共 Host 与类型定义，并重新验证受影响公共接口；不将其他硬件或旧二进制的结果当作本 arch35 实测。

## 交付与当前状态

设计评审、源码实现、构建、设备精度、全量性能、全量自验、CI、合并和官方接受分别记录。旧 24 字节布局候选 SO SHA256 `702e61853617fcfc57ec3b74c2bf7c08cda45c4138d6300d4290e513cac0124d` 已完成200/800、特殊8、适配层合同12及203性能/内存，并于2026-10-28 21:48 CST提交验收；正式送验私仓 main 固定提交 `38027a8b8b7bd2d97ddb5abe79d77fa6f801184b` 和材料保持不变；当前仍为一次验收中，进展为2。本次设计及新 bounded 资料尚未替换该送验版本。

64 位指数正式候选源码 `3fb0af79314968486d1d4a2c0a61c077e426b133` 与 SO `b1c081bc9c03304fa61c7b52753f8c2c045eab0bd065d2dfcb72d1bd1d130a55` 已分别通过 200/800、特殊8、合同12、定向特殊27、数值边界5和 m/n=2^20 边界。补充解析例 `m=7780739,n=1` 的全部输出逐位核验无 mismatch。其重启后新完整203条性能/Event/内存采集仍在 `062004` 运行，尚无完整性能结论。上述结果属于 3fb/B1C 候选，不转记到 bounded 候选。

bounded 实验仍在本地隔离工作树，尚未向私仓发布该代码或创建产品代码 PR。源码 `b9cc67727c6b6ab2a4df9fe944534f1782659b8b` 在 467 主机 CPU、CANN `9.1.0 SPC001B243` 下成功构建 CMake `ops_sparse` target 的全部82个编译单元（含 bounded kernel；该构建未编译 GTest 或 torch adapter）。对应 SO SHA256 `fd86d4ae7764e1727d27653bfa314bf732d480c31b3b2182c8e13ee1b1ade132`；另行编译并链接该库的 bridge SHA256 为 `7b64903419aa7148a4ae5c27e51d27bfff925d36b993683dbba84f18266ebba5`。后续 `01314aa797711da094b2a8d453dd65dceb71d49b` 仅修改 GTest 查询期望，其他992项 blob/mode与 b9 相同，可复用 b9 的生产库。013 的真实 SDK GTest 绑定该生产库后为9/9 PASS、另1项 disabled 未运行；一个 small case 另有一次通过。较早 b9 的唯一 GTest 失败是 `VisibleHostArgumentsAndWorkspace` 仍期望旧大尺寸查询值；原始 8 PASS/1 FAIL 记录保留，不覆盖修正后的结果。

bounded 候选已在目标设备执行两个真实 case，均为 `m=818376,n=1,ldb=818392`，分别是对角场景和含两处隔离跨界 pivot 的场景。两例查询均为 `8394112` 字节；execute 成功且 stream 同步返回0。每例818,376个复数输出的实虚位模式均与保存的参考输出 bit-exact、最大误差0；输入系数、16个 padding 值和完整 guard 均未改变。逐 NPZ/raw 复核和五项实物哈希复算记录于 `review/bounded_initial_real_saved_qa_20261029.json`。这两例是补充设备证据，不计入官方200/800精度或203性能分母。467 的 L2 metadata 记录为 `134217728` 字节；记录未保存 reset/finalize 返回码，只能确认后检 idle，不对这些返回码另作结论。

bounded 新候选的八组实际精度验证已完成：200、独立800、特殊8、合同12、定向27、数值5及 m/n=2^20 两项边界均通过组内验证器；父守护退出0，进程结束后两次空闲观察通过，源码、库和 SDK 前后身份一致。完整原始输出仍在取回，独立复核及新版精度报告待完成；正式203性能、Profiler/Event 和内存报告仍为 PENDING。跨块 Wide carry 的 n=1/n=2 及 `m=7846275` 当前仅有 CPU 准备/检查，不列为设备已验证。b9 的旧 query GTest 失败与013修正后的结果分别保留。正式 GPU 基线使用任务包 CSV，官方已确认其为标杆。

本次修订前 PR #2515 的远端 head `184f755d0dc942ac5e531469eedccb22ebd65be8` 曾通过 CI、人审尚未通过；本次设计文档提交后需重新运行 CI，旧 head 结果不沿用。没有创建 bounded 产品代码 PR 或新 Issue。设计变更的新 CI、人审、bounded 独立复核与完整性能交付、正式验收仍为 PENDING。

验收私仓根目录按任务书保存 `task_submission/1 自验证步骤说明.md`、精度/性能/内存三份 XLSX 报告和三份原始日志，并附逐次采样、Profiler、内存与确定性记录。结果绑定最终源码提交、实际加载 SO 完整哈希、调用器、环境和用例清单。设计审核通过并完成充分自验后提交平台验收；代码贡献按官方后续通知及目标仓流程推进。
