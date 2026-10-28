# 需求背景（required）

## 需求来源

- 社区任务：10月社区任务-aclsparseCgtsv2算子开发（950）。
- 任务编号：`3c8d06bce4f54557b895de27e3737e0f`；开发账号：`bluerain117`。
- [官方任务书及测试附件](https://www.hiascend.com/p/resource/202610/0deb445c978f428c9c92c9df6a7625c4.zip)，SHA256：`a57f0cae816c11f241607a73b1a59f3fd67520c82847529ccf4f6e539834a618`。
- [任务讨论帖](https://gitcode.com/cann/ops-sparse/discussions/25)；[社区流程](https://gitcode.com/org/cann/discussions/39)。
- 实现目标仓：[cann/ops-sparse](https://gitcode.com/cann/ops-sparse)，目标分支 `master`；开发起点 `e0580fa8991ab473c5cc06259c2c44f2db412aac`。

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

Host 仅完成可见参数校验、checked size_t 乘加、workspace 计算、取得句柄 stream 和 NPU Dispatch。不会将系数或 RHS 拷回 CPU 求解，不在执行接口中创建或销毁 stream，也不为数值状态执行隐式设备同步。

检查 `ldb*n*8`、`m*8`、因子与 RHS scratch 总字节数、证书区及对齐加法是否溢出。查询函数不解引用设备数据。非法参数应在 Dispatch 前返回；查询失败不将未完成计算写成有效 workspace 大小。

内部 tiling 包含 `m,n,ldb,numBlocks,fastEligible,fastBlocks,wideSlots` 七个 int32 字段。每 block 使用 32 个 SIMT 线程；稳定路径按 RHS scratch 槽位和实际 AIV 数决定并行度，线程循环覆盖全部 RHS。并行循环约减（CR）最多使用 32 个 AIV block，且不超过实际可用 AIV 数。`m>=512,n>0` 仅表示值得执行设备证书检查，不能仅凭尺寸进入 CR。

### 3.2.2 Kernel 侧设计

实现包含两条设备路径：通用的扩展精度部分主元 LU，以及安全证书控制的并行 CR。两条路径均由同一公开接口调度；Host 不回读证书、不同步决定算法，也不执行 CPU 求解。

**稳定路径第一阶段：部分主元三对角 LU 分解。** 单个 SIMT 线程沿行推进，保存 U 的主对角、第一和第二上对角、消元乘子及相邻行交换标志。分解结果由所有 RHS 共享，不为每列重复分解或分配矩阵副本。

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

上述伪代码中的复数运算在稳定路径使用内部 `ScaledComplex`，不直接以普通 complex64 保存消元和回代中间值。每个实/虚分量分别保存两个 FLOAT32 尾数及独立 int32 指数，值表示为 `(hi+lo)*2^exponent`，每个复数共 24 字节。使用 FMA 残差及补偿加法保存额外有效位，除法作商的残差修正；归一化后比较复数模长并选择主元。这样既减少病态系统的前向误差，也避免中间值超过 FLOAT32 范围后先产生 Inf、再因运算传播变成 NaN。实虚指数独立，不能因为某分量溢出而丢失另一有限分量。仅在最后写回 B 时收窄到 complex64；溢出分类按最终 FLOAT32 输出检查。

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

**并行 CR 路径。** 每次调用先在 NPU 上检查全部系数和有效 RHS：每个实/虚分量位于 `[-16,16]`，并且每行满足 `L(d[i]) >= U(dl[i])+U(du[i])+0.501`；边界外项视为零。这里 `L(z)=max(abs(real),abs(imag))` 是模长下界，`U(z)=max+0.415*min` 是保守上界。因此 NaN/Inf、界外数值或不满足严格对角占优裕量的输入都会拒绝快路径。

各检查线程独立写入自己的失败标志，再由后续同 stream Kernel 归约为总标志。在证书成功时，初始化 5 个 complex64 工作数组，按步长 `1,2,4,...` 做循环约减：只更新本层偶数活跃行，读取本层保持不变的奇数邻行，保存恢复所需系数。系数更新和 RHS 更新分为有序 Kernel，避免其他线程读取已经覆盖的系数；解出根行后逆序恢复被消去的行。RHS 仍完整保存在原 B 的有效区域，支持非 2 的幂 m 及全部 n，不改变公共问题规模。证书失败时 CR Kernel 不写系数或 B，完整部分主元路径接管求解；不能把无主元算法当作唯一实现。

使用 `simt_api/asc_simt.h` 的 SIMT Kernel、`asc_vf_call`、GM 加载/存储及复数算术辅助函数。公开类型仅保证 float 对齐，因此不能将数据地址擅自当作更高对齐的 float2 指针。行列偏移使用 64 位整数；循环访问范围不进入 B padding。

分解、证书发布/消费、各 CR 层及求解通过同 stream Kernel 顺序建立依赖，不使用跨核自旋。每次调用重新写入所有参与归约的线程标志及总标志，不依赖旧 workspace 的状态；本次实际消费的因子和交换项亦重新写入。未消费尾部不要求清零。调用方在求解完成前不得复用或释放 B、系数和 workspace；完成后可将同一工作区用于后续调用。

### 3.2.3 Workspace 与本地资源

| 区域 | 元素数 | 字节数 |
|---|---:|---:|
| U 主对角 | m ScaledComplex | 24m |
| U 第一上对角 | m ScaledComplex | 24m |
| U 第二上对角 | m ScaledComplex | 24m |
| 消元乘子 | m ScaledComplex | 24m |
| 相邻行交换标志 | m int32 | 4m |
| 证书线程标志和总标志 | 1025 int32 | 4100 |
| 独立 RHS scratch | S 列、每列 m ScaledComplex | 24mS |
| 总查询大小 | n>0 | `align_up(m*(100+24*S)+4100,128)` |
| 空 RHS | n=0 | 0 |

`S=min(n,max(1,floor(2097152/(24*m))),1024)`。2 MiB 是并行 RHS 暂存的预算，最低单列空间不可省略；实际活动槽位另受 AIV 数约束。CR 的 5 个 complex64 数组共 40m 字节，复用上述共享因子区，按证书选择与稳定路径互斥使用，不重复申请。

当前实板查询 L2 为 `134217728` 字节。按公式，官方 203 条性能清单全部低于 L2，最大查询值为 P-01 的 `2789504` 字节；有限域 `m<=2^20` 的最大查询为 `130027648` 字节。最终逐例报告仍记录真实查询、对齐分配及 allocator/device 数据，公式不替代实际内存采集。公开 ABI 未新增 m 上限：更大合法 m 按查询足量分配，不能宣称其 workspace 恒低于 L2。例如 `m=1082368,n=1,ldb=m` 查询为 `134217856` 字节，超出该设备 L2 128 字节。任务书 3.4 内存条件在有限因子域和正式 203 条以外的适用范围，请在评审中确认。

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

正式性能对每次调用内的检查、归约、初始化、CR 各层、根求解、恢复以及条件式 LU/求解 Kernel 完整求和，再对 30 个正式样本取中位数。即使某个条件式 Kernel 因设备标志提前返回，其实际耗时仍计入；不能只取最大的 Kernel、丢弃慢项或跨调用复用因子，也不能用 Event 或 Host 墙钟替代 Kernel 耗时。

## 支持硬件

| 芯片/软件 | 本设计范围 |
|---|---|
| Ascend 950，DAV_3510，arch35，A5 | 实现及验收目标 |
| CANN 9.0.0 及后续配套版本 | 按实际构建/设备环境记录；当前开发环境 CANN 9.1.0 |
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

设计评审、源码实现、构建、设备精度、全量性能、全量自验、CI、合并和官方接受分别记录。当前统一候选已在 CANN 9.1.0 / Ascend950PR 上通过发布200、独立800、特殊8、适配层合同12、C++常规8项及 m/n=2^20 两个独立边界；另外保留27条非有限位置与5条数值边界补充记录。上述结论绑定该候选的实际 SO、bridge 与原生 ATK 输出；正式203条性能/内存、完整组合的数值未执行项和跨平台回归仍分别记录，不能据此提前声明全任务验收通过。

验收私仓根目录按任务书保存 `task_submission/1 自验证步骤说明.md`、精度/性能/内存三份 XLSX 报告和三份原始日志，并附逐次采样、Profiler、内存与确定性记录。结果绑定最终源码提交、实际加载 SO 完整哈希、调用器、环境和用例清单。设计审核通过并完成充分自验后提交平台验收；代码贡献按官方后续通知及目标仓流程推进。
