# aclblasCsyr2 / aclblasCher2k / aclblasCherk 精度修复设计文档（A2/A3）

# 需求背景（required）

## 需求来源

本设计依据上传任务包中的 `aclblasCsyr2_Cher2k_Cherk精度修复_A2A3_task_doc.md` 及配套测试文件编写，三个算子作为同一任务一并交付；任务书 SHA-256 为 `a41b95e67ab0683e2388647c52912221ec6858e95ce9a79ef83cc3139759d791`。

设计模板与提交说明：

- https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
- https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md

## 背景介绍

### 任务定位

本任务修复已有 BLAS 接口在原地更新三角矩阵时的写回行为。核心问题是：数学上仅更新 `uplo` 指定三角，但某些对齐搬运、整块融合或边界写回路径会触及另一三角，甚至将旧值经浮点运算后重新写回，破坏 NaN payload、负零、非规范数等位模式。

修复须同时保证：

1. 操作区计算符合原有 BLAS 公式和混合容差要求。
2. 非操作三角与前导维 padding 不发生写入，并与调用前逐 bit 一致。
3. 公开接口、标量内存归属、quick return、负步长及异步执行语义保持任务规定的行为。
4. 保留可满足性能门禁的设备计算路径；不回退 CPU。

### 源码基线与实际目录

2026-10-28 核对到的版本如下。以下“源码观察”针对上游固定提交，不代表在 NPU 上已复现全部问题。

| 来源 | 提交 |
| --- | --- |
| 用户 fork 的 `master` | `d9d68732ecd86a62957ba783e6bb8397ee1d6102`（2026-09-30） |
| `cann/ops-blas` 上游 `master` | 首次核对 `05b96ea22cea926ecfd43fc7ac691a795522bd61`（2026-10-28）；提交时复核 `456b3487f939920a103a4f3f1a7d7636c7c94c17`（2026-10-28 19:42） |
| `cann/cann-ops-competitions` 模板与目录参考 | `2ff464e927c4d1957fc5088c1b4a0a327c325329`（2026-10-28） |

用户 fork 的该版本尚无 Csyr2 的 arch22 实现文件及公开声明；当前上游已具备。开发应先选定包含三个已有实现的上游基线，再在个人 fork 的开发分支上修复，避免把精度修复做成重复新增接口。

复核命令与结论：`git diff --stat 05b96ea..456b3487 -- blas/syr2 blas/herk blas/her2k include/cann_ops_blas.h test/syr2/csyr2 test/her2k/cher2k test/herk/cherk` 输出为空，说明两个提交之间本任务相关的实现、测试与公开头文件均无差异，上文的源码观察对当前 master 仍成立；若评审时 master 继续前进，需重跑该命令并把新提交号补入上表。

| 算子 | 任务书所列实现目录 | 核对提交中的实际实现目录 | 实际测试目录 |
| --- | --- | --- | --- |
| Csyr2 | `blas/syr2/arch22/` | 相同 | `test/syr2/csyr2/arch22/` |
| Cher2k | `blas/herk/arch22/` | `blas/her2k/arch22/` | `test/her2k/cher2k/arch22/` |
| Cherk | `blas/herk/arch22/` | 相同 | `test/herk/cherk/arch22/` |

**Cher2k 目录差异说明。**本文按当前已有实现定位修复点，不自行在 `blas/herk/arch22/` 复制第二套 Cher2k；若维护者指定其他验收提交或目录，按确认的基线修订路径表执行，算法与验收契约保持不变。

### 已有实现现状分析

| 位置或函数 | 静态观察 | 对本次设计的影响 |
| --- | --- | --- |
| Csyr2：`csyr2_device_xy.h::CopyOutFloats` 及其调用方 | 已存在按长度搬运路径；UPPER、LOWER、批列、旋转、窗口等分支分别生成地址与 UB 偏移 | 逐分支审计地址、源对齐与有效长度，不预先认定所有分支都有同一缺陷 |
| Cherk：`cherk_direct_kernel.cpp::StoreDirectDiagonalTile` | LOWER 分支对保留前缀使用 `Adds(..., 0.0f, ...)`，随后写出整列有效 tile 高度 | 浮点“加零”不能证明位模式不变，且写出范围包含非操作区；改为裁剪后紧凑写出 |
| Cherk：`StoreDirectTile` | 部分对角完整 tile 可走合并旧值后的二维整块写回 | 完整 tile 也必须进行三角分类，不能只修尾块 |
| Cherk：`cherk_compute_kernel.cpp::StoreSelectedTile` | 已有按列选择区间、对 LOWER 在 UB 中整理后写出的实现思路 | 可复用其设计思路，统一其他路径的写回契约，仍需完整回归 |
| Cher2k：`Cher2kSg3StorePartial`、`Cher2kStorePlanarUnitTile` | LOWER 起点向前对齐，并读取、写回操作区之前的若干元素 | 即便数值保持，也不满足非操作区零写要求；对齐应在 UB 内解决 |
| Cher2k：`Cher2kDirectStoreTile` | 对部分 tile 读取旧 C、合并另一三角后整块写回 | 移除非操作区的读改写，统一到精确区间提交 |
| Hermitian Host 辅助路径 | 存在固定大小的持久 metadata/索引缓存及特殊空输出指针分支 | 单独核对内存与异步约束，不能因为是既有代码便默认符合本任务 |

**目录与平台实现并存说明：** `blas/her2k/` 与 `blas/herk/` 各自同时存在 `arch22/` 与 `arch35/` 两套实现（Cher2k 在 `her2k`，Cherk 在 `herk`），不存在重复新增的平行实现。本次只修改 `arch22/`，不触碰 arch35（950PR）路径。

### 算子功能分析

Csyr2 更新复数**对称**矩阵，使用普通转置；Cher2k、Cherk 更新复数 **Hermitian** 矩阵，使用共轭转置。对称或 Hermitian 性质仅约束逻辑矩阵，不要求物理缓冲区另一三角存有其镜像。

数据统一使用 `aclblasComplex`，实部和虚部各为 FP32，即 complex64。矩阵为列主序全存储，不采用 packed 三角存储，也不引入广播或批量高维接口。

# 需求分析（required）

## 需求描述

在已有 arch22 实现与测试中修复三角写回，对所有合法成功调用保证操作区精度与非操作区位一致同时通过；三个算子共同提交验收。

## 需求拆解

| 编号 | 要求 | 设计措施 | 验证方式 |
| --- | --- | --- | --- |
| R1 | 公开接口冻结 | 复用现有声明、枚举、状态码；不增加平行 API | 头文件 diff、符号与调用样例检查 |
| R2 | 公式与存储语义正确 | 保留列主序、共轭规则、标量类型及负步长 | golden 比对、属性组合与边界测试 |
| R3 | 非操作区逐 bit 一致 | 所有输出路径仅提交 `uplo` 内区间，padding 零写 | `memcmp` / `uint32_t` 比较，代码检查写范围 |
| R4 | quick return 全缓冲不变 | 提前终止，优先于对角虚部清零与输出加载 | 完整缓冲位比较 |
| R5 | stream 异步 | Host 不因读取 Device 标量而同步；各阶段使用句柄 stream | 同一 stream 的标量更新、串联调用测试 |
| R6 | 操作区混合容差 | FP32 分量检查，固定参考库和阈值 | A2、A3 全量功能与精度结果 |
| R7 | 性能达标 | 保留主计算与内部完整 tile 的批量搬运，边界局部整理 | A2 九项显式门禁及全部 `TC_PF_*` |
| R8 | 内存与范围合规 | 不新增保护另一三角的整矩阵备份或持久缓存；只改已有 arch22 路径 | workspace 公式、申请与生命周期检视 |
| R9 | 测试完整 | 保留原 1200 条/算子及已附加哨兵，补充分布、路径和异步覆盖 | case 清单、执行数量与原始日志核对 |

# 详细设计（required）

## 算子分析

### 数学公式

记共轭转置为 `H`，普通转置为 `T`。

**Csyr2：**

```text
A_new = alpha * x * y^T + alpha * y * x^T + A_old
A_new(i,j) = A_old(i,j) + alpha * (x(i)*y(j) + y(i)*x(j))
```

仅计算指定三角；不对 x、y 或 alpha 取共轭；对角虚部按普通复数运算保留。

**Cher2k：**

```text
trans=N: C_new = alpha*A*B^H + conj(alpha)*B*A^H + beta*C_old
trans=C: C_new = alpha*A^H*B + conj(alpha)*B^H*A + beta*C_old
```

alpha 为 complex64，beta 为 FP32。可令 `X=op(A)`、`Y=op(B)`，其中 N 时 `op(A)=A`，C 时 `op(A)=A^H`，则两种情况下均为 `alpha*X*Y^H + conj(alpha)*Y*X^H`。如果复用单个中间乘积 `P=X*Y^H`，组合式为 `alpha*P + conj(alpha)*P^H`；`P^H` 来自中间结果，不能读取用户 C 的另一三角替代。

**Cherk：**

```text
trans=N: C_new = alpha*A*A^H + beta*C_old
trans=C: C_new = alpha*A^H*A + beta*C_old
```

alpha、beta 均为 FP32。两个 Hermitian 算子都只接受 N/C；实际执行更新时忽略旧对角虚部，并将新对角虚部直接置为 `+0.0f`。合法 quick return 不触碰任何输出字节，因此不执行对角清零。

### 接口定义

沿用 `include/cann_ops_blas.h`，不修改声明：

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);

aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo,
    aclblasOperation_t trans, int n, int k,
    const float* alpha, const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc);
```

### 支持数据类型与形状

| 参数 | 类型及位置 | 形状/约束 |
| --- | --- | --- |
| handle | Host 句柄 | 有效句柄，执行流来自 `aclblasSetStream` |
| uplo | Host 枚举 | `ACLBLAS_UPPER=121` / `ACLBLAS_LOWER=122` |
| trans | Host 枚举，仅 Hermitian 族 | `ACLBLAS_OP_N=111` / `ACLBLAS_OP_C=113`；`OP_T=112` 不支持 |
| n、k | Host int | n≥0；Hermitian 族 k≥0 |
| Csyr2 alpha | Host complex64 指针 | 不可为空；值为零允许 quick return |
| Cher2k alpha / beta | Device complex64 / FP32 指针 | 均不可为空；不改变任务规定的 Device 归属 |
| Cherk alpha / beta | Device FP32 指针 | 均不可为空 |
| Csyr2 x、y | Device complex64，只读 | n>0 时物理长度分别为 `1+(n-1)*abs(incx)`、`1+(n-1)*abs(incy)` |
| incx、incy | Host int | 非零，可为负；n=0 时不使用上述长度公式申请负长度 |
| Csyr2 A | Device complex64，原地读写 | 物理 `lda*n`，逻辑 n×n；`lda≥max(1,n)` |
| Hermitian A、B | Device complex64，只读；Cherk 无 B | N：逻辑 n×k、物理 `ld*k`；C：逻辑 k×n、物理 `ld*n` |
| lda、ldb | Host int | N 时 ≥max(1,n)，C 时 ≥max(1,k) |
| Hermitian C | Device complex64，原地输出 | 物理 `ldc*n`，逻辑 n×n；`ldc≥max(1,n)` |

仅支持由 inc 和 leading dimension 描述的步长；不增加广播、任意 stride tensor、batch 或 ≥3 维 ND 语义。用户矩阵之间的额外重叠/别名不作为新增能力承诺。

### 操作区和非操作区定义

令输出矩阵为 `M`，Csyr2 的 `M=A, ld=lda`，Hermitian 族的 `M=C, ld=ldc`。零基地址为：

```text
complex_index(i,j) = uint64(i) + uint64(j) * uint64(ld)
byte_offset(i,j)   = 8 * complex_index(i,j)
```

| 区域 | UPPER | LOWER |
| --- | --- | --- |
| 操作区 S | `0≤i≤j<n` | `0≤j≤i<n` |
| 非操作三角 | `0≤j<i<n` | `0≤i<j<n` |
| padding | `n≤i<ld, 0≤j<n` | 相同 |

非操作区 U 是整个 `ld*n` 物理矩阵扣除 S 的部分。要求 `bytes(M_after[U]) == bytes(M_before[U])`，且 Kernel 对 U 零写。仅在寄存器中掩掉计算结果、随后整块写回，不能满足该要求。

## 算子实现

### 实现方案

保留 Csyr2 的 AIV 向量化与数据复用；保留 Cher2k/Cherk 已有适配 arch22 的主乘积计算和必要 workspace。重点收敛最终读取旧输出与写回的边界行为，使通用、小 shape、快速标量、LOWER/UPPER、N/C、尾块和 beta 缩放路径都遵守同一契约。

对输出 tile 分三类：

| tile 类型 | 旧输出读取 | 最终写回 |
| --- | --- | --- |
| 完全位于非操作三角 | 不读 | 不写 |
| 完全位于操作三角 | 需要旧值时读真实范围 | 可保留二维批量搬运；行数/列数裁剪到逻辑 n |
| 跨越对角或有逻辑尾部 | 按列裁剪；beta=0 时不读旧 C | 有效区间在 UB 内重新整理后，按精确字节数搬出 |

Cube 中间结果允许存在完整矩形 workspace，因为它不是用户输出。禁止把完整矩形临时结果直接写入用户 M，再尝试恢复非操作区。

### Host 侧设计（对应模板 3.2.1）

#### 参数检查与返回码

先检查 handle、枚举、维度、leading dimension、必需标量指针，再按各接口既有规则处理 quick return 与数据指针。保持基线对多项同时非法时的检查先后，不借本次修复重新定义报错优先级。

| 条件 | 预期返回 |
| --- | --- |
| handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo 或 trans 为非法枚举值 | `ACLBLAS_STATUS_INVALID_ENUM` |
| Hermitian 族 trans=`ACLBLAS_OP_T` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 负维度、零向量步长、前导维不足、必需标量指针为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 合法计算或合法 quick return | `ACLBLAS_STATUS_SUCCESS` |

Csyr2 在 n>0 且 alpha 非零时校验 x/y/A；Hermitian 族按任务书在 n>0、k>0 时校验 A/B。需要写输出时必须有有效的 C 存储。`beta=0` 表示不依赖 C 的旧内容，不能一般性地解释为允许向空地址输出。任务书与既有代码对少数空 C 场景的表述不同，实现按任务书条款执行，并以仓库现有返回码检查先后为准。

维度、列偏移、步长起点及字节数计算在乘法之前提升为 64 位；`abs(INT_MIN)` 不在 int32 上求值；workspace 大小做乘法溢出检查。保留仓库已有资源不足/内部错误返回机制，不把无法申请的规模伪装成成功。

#### quick return 与标量归属

| 算子 | 条件 | 行为 |
| --- | --- | --- |
| Csyr2 | n=0 或 alpha=(0,0) | Host 直接成功返回，不 launch、不读写 A |
| Cher2k / Cherk | n=0 | 完成必需合法性校验后成功返回，不 launch |
| Cher2k / Cherk | `(alpha=0 或 k=0) 且 beta=1` | Device 分支直接退出，整个 C 位模式保持不变 |
| Cher2k / Cherk | `(alpha=0 或 k=0) 且 beta≠1` | 只缩放指定三角；beta=0 时直接置零；对角虚部置零 |
| Cher2k / Cherk | 存在乘积且 beta=0 | 不加载旧 C，直接计算和提交指定三角 |

Hermitian Device 标量值由设备在用户 stream 上读取。Host 只根据已知 shape、枚举及资源生成调度，不为判断 alpha/beta 做同步 D2H。需要多 Kernel 时，所有会读取 A/B/C 的阶段都执行一致的设备分支；跨核同步前的退出条件必须一致，避免部分生产者退出而消费者等待。

对于 Device 标量决定的 no-op，允许下发仅执行条件判断的设备入口，但不访问矩阵、不执行乘积或对角修正。调用方在同一 stream 上先更新 alpha/beta 后调用算子时，必须使用本次值，不能按指针地址缓存旧标量。

#### Tiling 与分核

1. 以当前已通过构建的分派框架为基础，保留与主乘积计算有关的 tiling；补齐每个输出 tile 的 `rowBase/colBase/validRows/validCols` 和三角分类。
2. Csyr2 可保留已有按列循环分配及列内分块；每个输出区间只归属一个 AIV。负步长使用 `start = inc<0 ? -(n-1)*int64(inc) : 0`，逻辑元素地址为 `start+t*inc`。
3. Cher2k/Cherk 保留已有输出 tile 所有权，内部 tile 快路径与对角边界路径分开。不得引入多个核对同一 C 元素的无序写入；本修复不新增 split-K 原子累加。
4. 核数、UB、L1 等容量使用平台信息；不把 A2 的资源常量当作 A3 的通用参数。
5. 内部分派至少覆盖普通更新、纯 beta 缩放、完整操作 tile、对角/尾块。沿用已有 tiling key 编号规则；这些分类不是新增公开 API。

#### UB 与 workspace 预算

边界列使用 32 字节对齐的 UB staging，令最大有效列段长度为 `B_r` 个 complex64，单缓冲容量为 `align32(8*B_r)`。双缓冲额外容量上界为 `2*align32(8*B_r)`；例如 `B_r=128` 时为 2048 字节，仅是设计验算值。

实际必须满足：

```text
UB_existing + UB_staging + UB_indices + UB_pipeline_reserve <= UB_available
```

优先复用已释放生命周期的输出缓冲；若容量不足，缩小局部列段或关闭局部双缓冲，不放大 GM 写出范围。索引尽量在 UB 内生成。已有固定大小的持久 metadata / 索引缓存（如 Cher2k 的 interleave offsets）属于既有性能路径，本次保留其用途与规模；按任务书 §3.4，只需保证本次修复不新增“为保护非操作区”而与问题规模无关的持久 Device 缓存，不以“既有实现”作为豁免。

Csyr2 写回修复不需要新增 GM workspace。Hermitian 族继续使用随 n/k 和实际计算路径变化的中间计算空间，给出其大小公式并由仓库机制管理。新增 staging 为核内临时 UB，不申请 `ld*n` 备份。异步执行所用 workspace 在 stream 完成前不得释放或被另一并发调用复用。

### Kernel 侧设计（对应模板 3.2.2）

#### 统一的三角裁剪规则

对 tile 行区间 `[r0,r1)` 与列区间 `[c0,c1)`，先把上界裁剪到 n。对列 j：

```text
UPPER: begin = r0;         end = min(r1, j+1)
LOWER: begin = max(r0, j); end = r1
若 begin >= end，跳过该列段。
否则 length = end - begin，写回地址为 M[begin + j*ld]。
```

区间计算使用有符号或已验证的宽整数，先判空再做无符号减法，避免非操作 tile 的下溢导致超大拷贝。

完全有效 tile 的判定使用全局坐标：UPPER 为 `r1-1≤c0`，LOWER 为 `r0≥c1-1`；完全无效 tile 的判定分别为 `r0≥c1`、`r1≤c0`。其余进入逐列裁剪，不依赖 tile 恰好为正方形或对角恰好落在局部原点。

#### 精确字节写回与 LOWER 源对齐

边界实现按以下顺序执行：

1. 取得 `[begin,end)` 内已经计算好的复数结果。
2. 若源结果起点不满足 UB 搬出对齐要求，使用 Gather/重排把这一段交织实虚部整理到**32 字节对齐**的 staging 起点；Gather 的源基址也保持接口要求的对齐，以索引表达段内偏移，不把任意切片当作对齐基址。只整理有效结果，不拼入旧的另一三角。
3. 确认 Vector 计算或重排完成后，使用 CANN 9.1 对目标平台支持的 UB→GM 非对齐搬运接口，指定 `blockLen = 8*length` 字节、`blockCount=1`。
4. 等待该缓冲的 MTE3 使用结束，再复用、修改或释放 staging。双缓冲分别追踪读写依赖。

`DataCopyPad` 的 GM 目的起点与传输长度可非 32 字节对齐，但 UB 源起点仍须对齐。**不能把 `src[offset]` 的任意非对齐切片直接交给该 API。**也不能为了 UB 对齐而把 GM 起点向前移动、把长度向上取整。

以 `rowBase=0, col=5` 的 LOWER 列为例，合法起点是行 5。若旧路径将起点改成行 4，会触及非操作区；新路径把行 5 起的结果整理到对齐 staging，再从 GM 的行 5 开始写出。

禁止用浮点 `Adds(x,0)`、乘 1 或格式转换“保存”未引用值；禁止使用 `GlobalTensor::SetValue` 逐点替代边界 DMA，因为其 DCache 回写粒度与多核一致性需要额外保证。小 shape、纯缩放与快速路径同样使用精确范围的输出搬运，不留下另一套未经证明的 scalar GM 写回。

对于内部完整 tile，可继续二维 `DataCopyPad`：`blockLen` 为真实有效行字节数，GM `dstStride` 按字节填写；UB pitch 按接口的 32 字节单位换算。尾部 padding 只能存在于 UB/workspace，不能写入用户 leading dimension 的 padding。

#### 读旧输出的范围

Csyr2 只读取待更新的 A 列段。Hermitian 族 beta 非零时只读取操作区 C 列段；对角旧虚部不参与计算。beta 为零时完全跳过旧 C 加载，不使用 `0*C_old` 代替，以避免 NaN/Inf 污染。

在跨对角 tile 中，不能先读整块旧 C 再通过 Vector mask 丢弃另一三角；正确的读取范围与提交范围采用同一裁剪函数。需要对齐的输入拷贝在 UB 侧 padding，不越过 GM 的合法区间取数。

#### 三个算子的计算路径

| 算子 | 保留的主要计算组织 | 修复重点 |
| --- | --- | --- |
| Csyr2 | AIV 上复数逐元素更新，复用列标量 x(j)/y(j) 与向量行窗口 | 审计所有 `CopyOutFloats` / 批列调用的范围和源对齐；统一 LOWER 与尾段写出 |
| Cher2k | 现有 AIC 乘积与 AIV 组合路径，保留 N/C 和标量分派 | 组合阶段只读/写指定 C 三角；替换 Direct、SG3、Planar 等边界回写 |
| Cherk | 现有 AIC 乘积与 AIV epilogue、小 shape 路径 | 移除对角 tile 合并另一三角的写回；统一 Direct、通用、缩放等路径 |

复数乘法按 `(a+ib)(c+id)=(ac-bd)+i(ad+bc)` 解释；共轭符号由算子公式决定。修复优先不改变主乘积的数值顺序。若实际发现 3M/HF32 等路径在大 k、抵消或正态分布下不满足容差，则在现有 arch22 内选择更准确的乘积/累加或补偿路径，按参数域验证后再启用，不能通过修改 golden、放宽阈值或按 case 名特判解决。

FP32 累加并不自动证明全部中间乘积具有 FP32 精度。最终报告需记录实际数学模式；不为了性能把所有 FP32 输入无条件降为 FP16/BF16。

#### 非操作区零写的正确性论证

每个最终 store 的目标集合为 `{(i,j) | begin≤i<end}`。由裁剪定义，UPPER 时始终 `i≤j`，LOWER 时始终 `i≥j`，且 `i<n、j<n`，故目标集合是 S 的子集，与非操作三角及 padding 无交集。

所有输出 tile 覆盖 S，且每个输出元素只归属一个最终写回任务；quick return 的写集合为空。若所有输出路径遵守该约束，U 中任意元素从未进入写集合，位模式与调用前相同。此结论仍须由 A2/A3 测试和对全部写回调用点的检查共同验证，单次 bit 比对不能证明中间过程从未写过。

### 代码修改范围

| 范围 | 计划变更 |
| --- | --- |
| `blas/syr2/arch22/` | `csyr2_device_xy.h`、边界/批列输出相关文件；必要时调整已有 host tiling 的 UB 预算 |
| `blas/herk/arch22/` | Cherk 的 Direct/通用/tiny/scale 写回与旧 C 读取；移除相关保护性 metadata 缓存依赖 |
| `blas/her2k/arch22/` | 在目录评审确认后修改 Cher2k 的 postprocess、cube 写回辅助及必要 tiling；不新建平行实现 |
| 三个已有 arch22 测试目录 | 安装任务 CSV、接入 bit 检查、补充边界与分布测试、修正计时和结果统计 |
| 算子 README 与对应测试说明 | 记录约束、编译运行方式、验收口径和来源差异 |

公开接口头文件与 arch35 计算实现不属于计划修改范围。共享构建配置/框架仅在接入确有必要时做最小修改，并补做受影响产品回归；不为了本任务全局改变其他算子的精度判定。

## 支持硬件

| 平台 | 本任务范围 | 验证要求 |
| --- | --- | --- |
| Atlas A2（arch22） | 支持 | 全量功能、精度、bit-exact；910B3 执行性能门禁 |
| Atlas A3（arch22） | 支持 | 全量功能、精度、bit-exact；性能跑通并记录 |
| Ascend 950 / arch35 | 不新增或修改其计算能力 | 若涉及共享配置，验证不受影响 |

A3 使用环境实际型号对应的受支持 `--soc` 值，不凭名称推定编译参数。开发和验收记录 CANN 9.1.0、驱动、固件、NPU 型号、编译器、仓库提交及 Netlib 3.10.0。

## 算子约束限制

不新增 API、dtype、broadcast、batch、CPU fallback 或确定性承诺。不保证本设计阶段已经达到性能上限。既有输入范围和返回语义按任务书与冻结基线核对；资料不一致时先记录并完成评审，不擅自放宽或收紧接口。

# 可维可测分析

## 精度标准/性能标准

### 操作区精度

Csyr2 的 golden 按任务要求依据 Netlib `ssyr2` 语义自实现复数版本，因为标准 CBLAS 无 complex syr2；Cher2k/Cherk 使用 Netlib BLAS 3.10.0 的 CBLAS 复数接口生成参考。另可用 complex128 直接公式定位数值问题，但不擅自替换验收指定参考。

仅收集操作区，complex64 实部、虚部分别按 FP32 评估并分别记录，不能只比较模长，不能把非操作区混入容差统计来提高通过率。

| 指标 | 要求 |
| --- | --- |
| rtol | `2^-13 = 0.0001220703125` |
| atol | `2^-13 = 0.0001220703125` |
| 逐点混合容差 | `abs(actual-golden) ≤ atol + rtol*abs(golden)` |
| 通过率 | `matched_ratio ≥ 0.99` |
| 绝对误差硬上限 | `max_abs_error ≤ max(1e-2, 32*ULP(g_low))` |

按已读取的混合容差标准 §2.1.2，`g_low` 是**最大有限绝对误差点**的 golden 收窄到输出 dtype 后的值，不是全张量最大幅值点。有限误差用更高精度计算；所有点为特殊值时，不强行定义不存在的有限误差最大点。

NaN/Inf 先分类，避免直接进行未定义的误差运算。匹配 NaN、同号 Inf 按测试框架的对应规则记通过；异号 Inf、无法匹配的 NaN/有限值等记失败，不借 99% 通过率掩盖特殊值失配。实际值为 Inf、golden 为高精度有限值时，先按标准以 round-to-nearest-even 收窄 golden，再判断同号溢出是否匹配。非操作区的 NaN payload 则仍须完全相同。

当前上游 `test/frame/verify.h` 使用逐元素 ULP 硬上限，与所读标准的“最大误差点”口径存在差异。计划保留仓库判定并同时输出任务标准判定，两者均满足时再申报；若结果不一致，单列失败原因供维护者确认，不静默放宽共享框架。

### 非操作区及 no-op 检查

| 场景 | 比较范围 | 判定 |
| --- | --- | --- |
| 正常成功更新 | 未引用三角 + ld padding | 实虚部的 uint32 位模式均相同，匹配率 100% |
| 合法 quick return | 完整有效物理输出缓冲 | 逐 bit 相同，包括对角和操作区 |
| n=0 | 逻辑输出为空；如测试另提供已分配缓冲，确认无写入 | 不执行无意义的空指针解引用 |
| 合法 Hermitian 更新 | 另外检查对角虚部 | 输出为 +0；不使用乘 0 处理旧 NaN |
| 参数错误 | 检查预期状态码；已提供缓冲时检查未修改 | 不将未执行用例计为数值精度通过 |

在 GTest 中接入任务提供的 `check_untouched_bitwise.h`：填充输入后覆盖非操作区哨兵，完成初始化并取得完整 `ld*n` 原始字节快照；调用算子；由测试侧同步 stream 后读回；最后分别进行操作区混合容差、非操作区 bit 检查。Kernel 内不得为此增加快照或恢复步骤。

哨兵采用 uint32 原始位模式填入，至少包括 `0x00000000`、`0x80000000`、不同 payload 的 NaN、正负 Inf、最小非规范数及随机有限值。诊断输出算子、case、uplo/trans、坐标、区域类别、before/after 实虚部十六进制位模式。quick return 另外使用全缓冲比较，因为原 `CheckUntouchedBitwise` 只检查 U，不能替代 no-op 验证。

### 用例复用及补充

本次读取上传 CSV 得到如下实际数量，不以 README 中的概述代替统计：

| 算子 | 原功能/精度/负向 | 性能 | 已附加 TC_BIT | 总计 |
| --- | ---: | ---: | ---: | ---: |
| Csyr2 | 1000 | 200 | 22 | 1222 |
| Cher2k | 1000 | 200 | 24 | 1224 |
| Cherk | 1000 | 200 | 24 | 1224 |
| 合计 | 3000 | 600 | 70 | 3670 |

已有 CSV 不删除、不以 `gen_csv.py` 覆盖。`append_bit_cases.py` 已添加的条目不重复追加。仓库原 CSV 中 `TC_ED_151` 把非法 `uplo`（999）的期望结果写成 `INVALID_VALUE`，与任务书 §2.4.1 的 `INVALID_ENUM` 冲突；提交 CSV 时按任务书修订该行并保留原值与修改理由，Host 实现不改为 `INVALID_VALUE` 以迎合 CSV。保留仓内现有额外回归，用独立补充 CSV/测试组覆盖以下场景：

| 补充项 | 覆盖设计 |
| --- | --- |
| 对齐和分派边界 | n/k 位于 7/8/9、15/16/17、31/32/33、63/64/65、127/128/129；再对实际 tiling/分派阈值取前后邻点 |
| leading dimension | 紧凑、n+1、n+3、n+8、n+32；N/C 下输入 ld 按对应真实行数取值 |
| 所有路径 | UPPER/LOWER × N/C；普通/小 shape/Direct/SG3/通用/纯缩放；记录实际命中路径 |
| Csyr2 步长 | incx/incy 为 ±1、±2、±3，混合正负；结合非对齐 n 与 padding |
| quick return | n=0；alpha=0；k=0、beta=1；对角旧虚部含非零/NaN，仍要求全缓冲不变 |
| beta 特例 | 0、1、负值；beta=0 时旧 C 操作区放 NaN，验证不读取其旧内容 |
| Hermitian 对角 | 非零/NaN 旧虚部被忽略；真实更新后置零；Csyr2 不误清对角虚部 |
| 随机分布 | 采用固定种子，将同一组 shape/attr 分别生成均匀与正态输入，形成 1:1 成对补充集 |
| 异步与重入 | 同一 stream 上更新 Device alpha/beta 后立即调用；重复调用变更标量；多个独立句柄/stream 使用独立缓冲 |

上传扩展用例使用 `RANDOM_NORM_5_5`；该命名中的 NORM 是布局模式，不能当作正态分布证据。补充集按任务要求生成均匀 `[-5,5]` 与正态 `μ∈[-5,5]、σ∈[0.1,2]`，实虚部独立采样。当前公共 Gauss 生成器还会截断输出，若直接复用则不能声称实现了任务规定的未额外截断正态分布；在本任务测试目录实现或明确配置对应生成规则，不通过截断规避精度失败。

所有成功且有有效输出缓冲的计算 case 均运行 bit 检查，不仅检查 `TC_BIT_*`。性能 case 的正确性和 bit 检查放在计时区外；负向、no-op、正常计算分别统计数量。

### 性能目标及测量

以下来自任务书 §3.3，单位为微秒（us）。

| 算子 | uplo | trans | n | k | A2 平均耗时上限 |
| --- | --- | --- | ---: | ---: | ---: |
| Csyr2 | UPPER | — | 512 | — | 7.54 |
| Csyr2 | LOWER | — | 1024 | — | 9.72 |
| Csyr2 | UPPER | — | 2048 | — | 19.05 |
| Cher2k | UPPER | N | 1024 | 1024 | 1458.91 |
| Cher2k | UPPER | N | 2048 | 2048 | 5240.68 |
| Cher2k | LOWER | C | 1024 | 1024 | 1081.90 |
| Cherk | UPPER | N | 1024 | 1024 | 738.09 |
| Cherk | UPPER | N | 2048 | 2048 | 2616.28 |
| Cherk | LOWER | C | 1024 | 1024 | 485.94 |

Csyr2 上述 case 使用 alpha=(1,0)、incx=incy=1、lda=n；Hermitian 典型 case 按配套 CSV 使用 alpha=1、beta=0 及合法紧凑前导维。三个算子的全部 200 条性能 case 仍需运行，包括未列入此九行表的其他规模。

上传 `gpu_baseline.csv` 三份各 200 行，`gpu_ms` 均已有值，并非全空占位。配套脚本使用 `baseline_ms / actual_ms ≥ 0.8`。例如 Csyr2 n=512 的基线为 0.006029 ms，对应上限 7.53625 us，与任务书 7.54 us 为显示精度差异。对显式门禁按任务书上限判断，同时报告 CSV 的未取整倍率；不能再对任务书上限除一次 0.8。

测量计划：

1. 初始化、申请内存、生成 golden 与必要预处理在计时外完成。建议 warmup 10 次、有效采样 60 次，满足任务要求的有效采样 >50。
2. 以同一绑定 stream 上的设备事件包围一次完整算子调用；若包含多个 Kernel，包含本次调用所有 pack/计算/epilogue 阶段。不能只报告其中最快的主 GEMM。
3. 原地输出在各样本前恢复为相同初值，恢复操作在 start event 之前完成，避免累积更新造成溢出或改变输入。恢复仅存在于测试侧。
4. 事件同步与结果读回在测试侧执行；不写入被测 API。汇总有效样本的算术平均值，保留原始样本，不用最小值或中位数替代门禁平均值。
5. 可用 msprof 分析 Kernel 构成与瓶颈，保留调用边界和单位换算记录。

配套 `verify_performance.py` 当前从 `[ OK ] ... (整数 ms)` 读取 GTest 整例耗时，包含 CPU golden、分配与比较，不能作为微秒级算子计时。计划只在三个算子的 arch22 测试目录内（不改 `test/frame/` 公共框架）用设备事件包围一次完整调用，输出结构化 `case_name/avg_us/sample_count/warmup_count` 行，脚本读取这些字段并校验完整性；缺失样本、测试失败、重复映射、无基线或零条匹配都不能输出 ALL PASS。

性能风险主要集中在边界重排和小 shape launch 开销。先复用原完整 tile 快路径，边界整理优先融入现有 epilogue；双缓冲和批列优化仅在精度、bit 与流水依赖验证通过后启用。任何单 case 性能失败均保留并分析，不用平均加速比抵消。

边界改造的性能回退顺序：先按精确 `DataCopyPad` 实现并实测；若门禁不达标，再评估“UB 内掩码整理 + 精确搬出”或调整 tile 划分，并以实测数据说明取舍，不以降低非操作区要求换取性能。

### 测试执行与报告

复现说明应提供从冻结提交到结果的完整步骤：配置 CANN 9.1.0 和 Netlib 3.10.0 → 编译三个算子及 GTest → 安装任务 CSV → 全量精度/bit → 专项回归 → 性能 → 导出报告。

任务包的入口示例为：

```bash
# 在任务包的 test_cases/csyr2 目录，先确保 GTest 已接入 bit 检查。
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --csv ./syr2_test.csv
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --filter TC_BIT --skip-build
```

该示例为计划复现命令，本文未执行 NPU 测试。Cher2k 的配套安装路径、查找二进制路径须先适配实际 `her2k` 目录；A3 的 `--soc` 值及脚本映射须按真实环境验证。性能入口在结构化计时修正完成后使用。

结果清单至少包含：环境/commit、case ID 与完整参数、随机种子、返回码、命中路径、real/imag matched ratio、最大误差及锚点、非操作区检查数量与失配数、对角结果、有效采样数、平均 us、基线、倍率、最终结论。空检查集合明确标注，不以 0 个元素伪造实测 100%。

## 兼容性分析

| 项目 | 处理方式 |
| --- | --- |
| API / ABI | 保持三个已有公开签名、结构体、枚举和状态码；不增加并行入口 |
| 布局 | 保持列主序和 leading dimension 语义；另一三角仍非输出 |
| 数值 | 不更改公式、参考版本和阈值；快速路径同样接受完整精度检查 |
| stream | 各阶段使用句柄绑定 stream；Device 标量变化按 stream 顺序生效 |
| 多核 | 每个输出元素唯一写回任务，禁止原子加或 cache line 回写侵入 U |
| arch35 | 不修改其计算路径；共享构建/框架变更单独检查影响 |
| 基线升级 | 冻结上游提交后复核全部分支，已有修复不重复覆盖，差异可追溯 |