# aclblasHgemmStridedBatched 算子设计（Ascend 950PR）

| 项目 | 内容 |
| --- | --- |
| 任务 | 10 月社区任务：aclblasHgemmStridedBatched 算子开发 |
| 提交者 | c15503545287 |
| 目标平台 | Ascend 950PR，arch35，CANN 9.1.0 |
| 实现形式 | Ascend C kernel 直调，复用 ops-blas GEMM 族工程 |
| 文档阶段 | 设计评审；本文的分块参数为候选值，测试结果待实现后实测 |

# 1. 需求背景

## 1.1 需求来源

依据任务包中的 `aclblasHgemmStridedBatched_Atlas950PR_task_doc.md`、`test_cases/README.md`、精度用例 CSV 和 GPU 性能基线，补齐 950PR 上半精度固定步长批量 GEMM 的句柄式接口。文档按社区设计模板组织，覆盖功能、Host 调度、设备端计算、资源约束与验收方法。

本次交付为设计文档。实现代码、自测报告以及性能数据属于后续交付，不将设计估算作为验收结果。

## 1.2 背景与设计目标

固定步长批量 GEMM 的各批矩阵具有相同尺寸、转置模式和前导维度。调用者只提供三块缓冲区及批间步长，无需构造设备指针数组。这适合同时处理多个独立的小矩阵，也能减少逐矩阵调用带来的 Host 下发开销。

本设计将批次号与输出矩阵的二维分块共同编码为工作项：矩阵较小时，通过 batch 维提供并行度；矩阵较大时，通过输出块提供并行度。K 维在单个输出块内部完成累加，以避免跨核归约及重复应用 beta。通用路径使用有界 FP32 中间缓冲，常见 `alpha=1、beta=0` 路径在满足写回约束时直接输出 FP16。

ops-blas 中已有 GEMM 和 strided-batched GEMM 的 arch35 实现可供复用。复用范围为布局转换、矩阵乘模板、平台资源查询及 stream 下发方式；FP16 数据类型、批次调度和本任务的参数约束需要单独适配，不直接沿用 FP32 的字节对齐常量。

# 2. 需求分析

## 2.1 数学语义与数据布局

令批次号为 `b`，输出行列为 `r、c`，归约下标为 `p`：

```text
Ab = A + b * strideA
Bb = B + b * strideB
Cb = C + b * strideC

Cb[r, c] = alpha * sum(op(Ab)[r, p] * op(Bb)[p, c], p=0..k-1)
           + beta * Cb_old[r, c]
```

所有矩阵均为列主序；`strideA/B/C` 以元素数计。A、B、C 及两个标量均采用 IEEE 754 binary16。N 表示不转置，T 表示转置，C 表示共轭转置；实数类型下 C 与 T 采用同一计算分支。

| 对象 | N 时物理形状 | T/C 时物理形状 | 前导维约束 |
| --- | --- | --- | --- |
| A | m × k | k × m | `lda >= max(1, transa==N ? m : k)` |
| B | k × n | n × k | `ldb >= max(1, transb==N ? k : n)` |
| C | m × n | 不涉及 | `ldc >= max(1, m)` |

精确的元素偏移是实现和测试共同使用的基准：

```text
A_offset(b,r,p) = b*strideA + (transa==N ? r+p*lda : p+r*lda)
B_offset(b,p,c) = b*strideB + (transb==N ? p+c*ldb : c+p*ldb)
C_offset(b,r,c) = b*strideC + r+c*ldc
```

不把 `lda` 当作逻辑行数，也不把批间步长重新计算为矩阵紧凑大小。行间 padding 与批间空隙必须保留。

## 2.2 公共接口

接口声明进入 `include/cann_ops_blas.h`，不增加 950PR 专属 API，也不引入 aclnn 两阶段接口。

```cpp
aclblasStatus_t aclblasHgemmStridedBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, long long strideA,
    const aclblasHalf* B, int ldb, long long strideB,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc, long long strideC,
    int batchCount);
```

`alpha、beta` 是 Host 标量指针，在调用期间读取并转换为 FP32 值后随 launch 参数传递；不得将 Host 指针交给设备端解引用。A/B 为 Device 只读缓冲，C 为 Device 原地输入输出缓冲。kernel 使用 handle 绑定的 stream。

`aclblasHalf` 应在公共类型层统一定义或复用已有定义，满足 2 字节 binary16 存储、Host/Device 一致及 C/C++ 头文件兼容性；不在 arch35 私有头中单独声明同名类型。若以 16 位存储类型承载半精度，标量转换必须解析其半精度位表示，不能把位模式当整数进行数值转换。具体类型接入与同族 Hgemm 接口统一，作为代码实现阶段的编译检查项。

## 2.3 功能拆解

| 需求 | 设计落点 | 验证重点 |
| --- | --- | --- |
| 九种 N/T/C 组合 | Host 将 C 归一化为 T，设备选择四种布局 | 非方阵及独立 padding |
| 固定步长批量计算 | 64 位批偏移和 batch × 输出块调度 | 批次边界、零步长、非紧凑步长 |
| FP16 输入输出 | FP32 累加及后处理，最终转 FP16 | 大 K、抵消误差、溢出 |
| 任意 alpha/beta | FP32 epilogue，beta=0 不读旧 C | C 初始为 NaN 的 beta=0 用例 |
| 空规模 | 参数检查后 Host 返回 | 不启动 kernel、不访问矩阵 |
| k=0 | 独立批量缩放/清零路径 | A/B 可为空，padding 不变 |
| 异常参数 | 共用错误码、固定检查顺序 | no-op 与非法参数组合 |
| 性能 | 批量下发、分块流水和直接写回 | 每项倍率与全体平均倍率 |

## 2.4 需求口径与明确约定

任务材料存在以下差异，设计评审时需要统一，避免后续实现和验收采用不同解释。

| 差异 | 本设计采用的口径 |
| --- | --- |
| 任务书写“不支持 broadcast”，测试包包含 strideA/B=0 正向用例 | 不支持张量维度广播；允许固定地址步长为零带来的只读矩阵复用，也允许 A/B 的批间重叠读 |
| 任务书要求 k=0、beta=0 清零，但 C 判空条件遗漏这一分支 | 非空输出需要有效 C 才能清零；设计建议 C 为空返回 INVALID_VALUE。该项作为评审确认点，代码验收前应同步任务表和测试期望，不静默对空地址清零或宣称产生了输出 |
| strideC 仅要求非负，测试说明排除输出重叠 | 非空 C 的不同 batch 逻辑输出不得重叠；C 也不得与仍会读取的 A/B 区域重叠。重叠输出不属于支持范围，不按批次串行定义隐含递推语义 |
| §3.3 要求单项 0.4×、平均 0.5×，§7 又写“达到 GPU 数据” | 以专门验收章节 §3.3 和测试脚本的两项量化阈值为设计目标；保留原始倍率供评审，不把更高指标写成已达成 |
| 性能采样要求分别出现 10 次、超过 10 次、超过 50 次 | 标准记录保留 5 次 warmup 后的 10 个有效样本；另执行超过 50 次采样用于稳定性复核，两组结果单独标注 |

上述差异不影响设计文档提交。尚未统一的行为在实现前固化为 API 说明和测试断言。

# 3. 详细设计

## 3.1 Host 处理流程

```text
公共 API
  -> 句柄、枚举、尺寸、前导维、步长、标量指针检查
  -> 空规模返回
  -> 按非空分支检查矩阵指针
  -> 安全地址跨度计算，读取半精度标量
  -> k=0 / alpha=0：批量缩放或清零
  -> 选择输出布局、分块和工作空间容量
  -> 直接输出：批量 GEMM
     通用输出：分组 GEMM -> epilogue，按 stream 顺序复用工作空间
```

### 3.1.1 校验顺序与边界分支

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；无效枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`。
2. 校验 `m/n/k/batchCount >= 0`，再依据转置模式校验三个前导维。
3. 校验三个 stride 非负，以及 alpha/beta 非空；失败返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 若 `m==0 || n==0 || batchCount==0`，返回 SUCCESS；不读取矩阵，不启动 kernel，不进行 Device 地址偏移计算。
5. 非空输出检查 C；`k>0` 时检查 A/B，即使 alpha 为零也执行 A/B 判空。`k==0` 时不要求 A/B 有效。
6. 检查实际访问跨度及工作项计数的整数溢出，再进入计算分支。

例如 `batchCount=0、strideA=-1` 必须返回 INVALID_VALUE，不能被空批次短路掩盖。`alpha=nullptr、m=0` 同样返回 INVALID_VALUE。C 空指针与清零分支的冲突按 §2.4 提交评审。

在完成必须的判空后，`alpha=0` 可跳过 A/B 数据读取，仅计算 `beta*C`。`k=0` 不使用 alpha 的数值决定输出；即使 alpha 为 NaN，也执行任务规定的缩放语义。beta 为零时直接生成零，不通过 `0*C_old` 清零。

### 3.1.2 地址与容量计算

维度类型保留接口要求的 int，内部乘加先提升至 64 位。对物理形状为 `rows × cols` 的非空矩阵，最后一个逻辑元素之前的跨度为：

```text
spanElements = (cols - 1) * ld + rows
requiredElements = (batchCount - 1) * stride + spanElements
requiredBytes = requiredElements * sizeof(aclblasHalf)
```

逐步执行有检查的乘法和加法，不能先溢出再比较。Host 验证结果可由设备地址类型表示；实际 Device 分配长度由调用者保证，普通指针接口无法仅凭入参证明分配足够大。A/B 零步长不应被“stride 必须大于矩阵大小”的错误检查拒绝。

例如 `m=3、n=2、k=4、transa=N、lda=5、strideA=23` 时，A 的逻辑跨度为 18 个元素，第二批从元素 23 开始。A 的存储间隙不参与矩阵乘，不能因搬运对齐而当作有效数据。

### 3.1.3 参数下发与工程组织

Host 为同一次调用建立一个共享计划，设备端通过 batch 编号定位数据。拟下发字段如下，最终定义采用固定宽度整数及无指针悬挂的按值结构。

| 字段组 | 内容 |
| --- | --- |
| 逻辑尺寸 | m、n、k、batchCount |
| 物理布局 | lda、ldb、ldc、三个 64 位 stride、归一化转置标志 |
| 调度 | tileM/N/K、输出块数、启动核数、工作项总数 |
| 分组 | batchBase、groupBatchCount、列条带起点与长度 |
| 精度及分支 | FP32 alpha/beta、直接输出/通用输出/缩放标志、是否读取旧 C |
| 临时输出 | 工作空间起址、临时前导维、每批槽位步长 |

实现文件按任务书放入 `blas/gemm/arch35/`，优先扩展同族 Host/kernel/tiling 组织；公共声明放入 `include/cann_ops_blas.h`。现有 `blas/gemm_strided_batched/arch35/` 的布局与调度方法可抽取复用，但不复制出第二套公共 API。构建注册只增加必要的 FP16 实例。

任务书给出的测试目标目录为 `test/gemm/hgemmStridedBatched/arch35/`，随包脚本则查找 `test/gemm_strided_batched/hgemm_strided_batched/...`。实现阶段需使构建目标与脚本查找路径一致，通过参数或适配入口统一，不维护两份 CSV。

## 3.2 列主序到计算引擎的映射

若复用的 GEMM 模板以行主序表达结果，将列主序 C 的同一缓冲解释为行主序 `C^T`：

```text
C^T = alpha * op(B)^T * op(A)^T + beta * C_old^T
engineM = n, engineN = m, engineK = k
left = B, ldLeft = ldb, transLeft = transb
right = A, ldRight = lda, transRight = transa
```

这里转置标志与原始标志相同，是因为原始列主序缓冲被重新解释为其转置的行主序矩阵。设备模板根据标志选择对应的 ND/DN 访问布局，不先生成整矩阵转置副本。采用直接列主序模板时则使用 §2.1 的偏移式，二者不能在同一路径上重复交换。

以 `m=3、n=5、k=7、transa=T、transb=N` 为例，A 物理形状为 7×3，B 为 7×5。计算引擎得到 5×7 的左矩阵与 7×3 的右矩阵，结果为 5×3，与 C 的列主序缓冲一致。该非方阵映射作为专门回归项。

## 3.3 batch 与输出块联合调度

以下用原始 C 的行列坐标说明，映射到计算引擎后同步交换坐标。设输出块大小为 `BM × BN`：

```text
tilesM = ceil_div(m, BM)
tilesN = ceil_div(n, BN)
tilesPerBatch = tilesM * tilesN
totalJobs = groupBatchCount * tilesPerBatch

for job = coreId; job < totalJobs; job += activeCoreCount:
    localBatch = job / tilesPerBatch
    tile = job % tilesPerBatch
    tileRow = tile % tilesM
    tileCol = tile / tilesM
    b = batchBase + localBatch
    compute C[b, tileRow*BM : ..., tileCol*BN : ...]
```

`activeCoreCount = min(可用 AIC 数, totalJobs)`，不假定固定核数。小矩阵每批仅一个工作项时，不同核处理不同 batch；大矩阵单 batch 也能通过二维输出块占满核心。每个输出块由一个核负责全部 K 累加，确保 beta 仅应用一次。

第一版不采用 split-K 或全局原子累加。对长 K、窄 M/N 且 batch 很小的场景，记录利用率作为后续优化依据；只有实测证明必要时才引入额外归约，并独立评估误差和临时空间。

## 3.4 分块选择与片上资源

初始候选输出块为 32×64、64×64、64×128、128×64，K 块为 32、64、128。候选值属于调优起点，必须同时满足平台资源、矩阵指令及搬运约束才可实例化。选择顺序为：先确保资源合法，再控制尾块比例，最后以实测耗时确定分派阈值。

以 FP16 输入、FP32 累加估算：

```text
L1_AB_bytes ~= 2 * (BM*BK + BK*BN) * 2      # 双缓冲、每输入元素 2B
L0A_bytes   ~= BM * baseK * 2               # 若双缓冲需再乘 2
L0B_bytes   ~= baseK * BN * 2               # 若双缓冲需再乘 2
L0C_bytes  ~= BM * BN * 4
```

这些是未计布局填充的下界。实际预算使用各维按硬件布局对齐后的值，并预留模板临时对象和同步资源。例如 64×64、BK=64 时，输入双缓冲为 32 KiB、累加块为 16 KiB；能否采用还取决于 L0、L1 及所选模板的约束，不能仅据此断言可运行。

对很小的 m/n 优先减小输出块，避免大量无效 MMAD；对规则大矩阵优先扩大有效数据复用。输入地址、前导维或 stride 不满足快速搬运要求时，采用有边界保护的局部打包，不能拒绝任务范围内的非对齐合法输入。

## 3.5 设备端计算与流水

每个输出工作项执行以下步骤：

1. 计算 batch 首地址及当前输出块的有效行列数。
2. 通过布局描述将 A/B 的当前 K 分块从 GM 搬入 L1；尾块只读取有效元素，片上补零。
3. 将 L1 分块送入 L0A/L0B，使用 FP16 乘法输入、FP32 L0C 累加。首个 K 块初始化累加器，后续块持续累加，不在 K 块之间转回 FP16。
4. K 归约完成后，直接写回或写入 FP32 临时输出。
5. 通用路径在同一 stream 的后处理 kernel 中读取临时输出，计算 alpha/beta 并将逻辑结果写回 C。

双缓冲槽位遵循“搬入完成后才计算、消费完成后才覆盖”的事件配对。输出块结束时释放所属同步资源；跨 batch 复用缓冲必须重新初始化累加器。异步流水只在所用模板保证依赖正确时启用，不依赖未声明的跨核屏障。

### 3.5.1 直接写回路径

`alpha==1 && beta==0` 时无需读取旧 C。若所选 FP16 写回指令支持当前输出布局、尾块和地址对齐，累加结果从 FP32 转 FP16 后直接写 C，省去中间 GM 数据及 epilogue 启动。

快速路径还必须验证 C 首地址、每批首地址、ldc 和输出块边界满足实际写回粒度。32 字节对于 FP16 是 16 个元素，不能照搬 FP32 的 8 元素判断。对于完整对齐的性能用例，此路径作为主要优化对象；非对齐输出走通用路径。

### 3.5.2 通用后处理与有界工作空间

一般 alpha/beta 采用 FP32 临时结果 `P=op(A)*op(B)`。设备后处理执行：

```text
v = float(alpha) * P
if beta != 0:
    v += float(beta) * float(C_old)
C = round_to_nearest_even_fp16(v)
```

只在最终结果处转换 FP16，避免乘积先转半精度、再缩放所产生的额外舍入。beta 为零的分支必须真正跳过 C 的读取，防止未初始化值和 NaN 污染结果。

工作空间由 handle 的既有机制提供。设 `Mp` 为按临时输出要求对齐后的 m，当前条带包含 `Ns` 个 C 列，同时处理 `G` 个 batch：

```text
tempBatchStride = align_up(Mp * Ns * 4, workspaceAlignment)
workspaceRequired = G * tempBatchStride
```

先尝试完整 n，再缩减每组 batch 数；若单 batch 仍放不下，则缩小列条带。每个槽位对应一个 batch，GEMM 与 epilogue 的索引共用 `Mp、Ns、tempBatchStride`。每组按“GEMM 写入 -> epilogue 读出 -> 下一组复用”顺序提交到同一 stream，避免覆盖仍在使用的结果。不得按 `batchCount*m*n` 无上限申请全量 FP32 临时空间。

最小合法条带也无法容纳时，遵循库的资源错误返回约定，在任何输出写入前结束，不发生整数回绕或零长度调度。不同 stream 的并发调用应采用独立 handle/工作空间，不能共享尚未消费的槽位。

### 3.5.3 尾块、padding 与输出所有权

M/N/K 尾块的有效长度分别是 `min(BM,m-r0)`、`min(BN,n-c0)`、`min(BK,k-p0)`。K 尾部补零在片上完成，不读取调用者未分配的 GM 元素；C 仅更新有效坐标，保留每列 padding 和批间空隙。

逻辑块不重叠还不足以证明窄粒度写入安全：两个核可能写入同一搬运对齐块中的不同 FP16 元素。快速路径通过地址与分块对齐排除此情况。通用 epilogue 以互不重叠的物理写回区域分配所有权；无法证明区域互斥的非对齐布局，第一版使用单 AIV 顺序处理该组输出，完成后再复用临时缓冲。精确尾部写回应采用目标 SDK 支持的有界存储，不以扩大 GM 写入长度实现对齐。该保守分支可后续优化，但首先保证 padding 和邻接 batch 不受损。

## 3.6 退化路径

`k=0` 或校验完成后的 `alpha=0` 使用批量缩放 kernel，不启动 Cube 计算。工作项覆盖 `(batch,column,row-range)`，地址仍按原始 ldc/strideC 计算。

| beta | 行为 |
| --- | --- |
| 0，包括 -0 | 对逻辑 C 写零，不读旧 C |
| 1 | 完成参数校验后返回，C 保持原值 |
| 其他有限值、Inf、NaN | FP32 执行缩放，再转换为 FP16 |

不能对包含 padding 的整个存储跨度进行 memset。只在已证明逻辑区域紧凑连续时合并清零；其他情况采用有界列段写入，并遵守 §3.5.3 的物理输出所有权规则。

## 3.7 错误处理与兼容性

设备能力、分块资源和工作空间规划在 kernel 提交前完成。Host 下发失败映射到 ops-blas 既有状态码并记录操作上下文；SUCCESS 表示异步任务成功提交，不表示执行已经同步完成。测试端通过 stream 同步检查异步错误，正常 API 不为返回结果而强制全设备同步。

公共函数签名与任务书一致，其他产品线可以复用同一个声明。arch35 的新增实例不改变既有 Sgemm/Cgemm 的参数、精度或分派行为。C 与 T 在本实数接口中归一化，不将该处理扩展到复数接口。

| 硬件 | 本设计范围 |
| --- | --- |
| Ascend 950PR | 支持目标 |
| A2/A3 等其他产品线 | 本次不新增实现，沿用仓库平台分派约定 |

# 4. 可维可测分析

## 4.1 功能与内存正确性

以随包 1000 条精度用例为基础，不遗漏负向用例。当前 CSV 的类别数为：L0 18、MN 47、AB 10、LD 22、FL 18、CV 21、ED 41、EX 823。每条记录保留参数、随机种子、预期状态码及各 batch 的误差结果。

补充验证关注算法容易出错的位置：

| 检查 | 用例构造与通过条件 |
| --- | --- |
| 转置寻址 | 九种组合 × 非方阵 × lda/ldb 独立 padding，与元素偏移公式一致 |
| 批次寻址 | stride 为紧凑值、紧凑值+1/+16、A/B 零步长及重叠只读，结果逐批匹配 |
| 旧 C 访问 | beta=0 时 C 预填 NaN，有限 A/B 的输出不受旧 C 影响 |
| 尾部保护 | m/n/k 取 1、质数、分块边界±1；padding/批间空隙填哨兵，运行前后逐字节比较 |
| 调度覆盖 | batchCount 取 1、3、核数附近、非整倍数及大批量，输出块无遗漏、无重复 |
| 工作空间分组 | 改变可用容量触发 G/Ns 不同选择，结果与全量布局一致，守护区不变 |
| 校验顺序 | 空规模与负 stride、空标量、非法枚举交叉组合，返回指定错误码 |
| 退化分支 | k=0 时 A/B 为空；alpha=0、k>0 时空 A/B 仍报错；清零不触碰 padding |
| 溢出校验 | 极大 stride 与 batchCount 的乘加，Host 拒绝不可表示的跨度，不实际分配超大内存 |
| stream 行为 | 同 stream 连续调用及独立 handle 多 stream 调用，验证依赖和临时空间隔离 |

API 不支持的 C 重叠写入不作为数值正确性的正向用例。对 C 为空且 k=0、beta=0 的测试，在 §2.4 结论确认后补充唯一明确的期望值。

测试包当前主要为均匀分布，需增加任务要求的 50% 正态分布；仅改变 CSV 条数不足以补齐此项。大规模覆盖按实际物理跨度估算 `2*(A_elements+B_elements+C_elements)`，再计入 Host golden 和临时缓冲，不能仅用 `3*m*n*batchCount*2` 估算非方阵内存。16384 维度和 65536 批次采用分批生成、流式比对等方式补充，不以 512 MB 生成器预算作为 API 限制。

## 4.2 精度验证

golden 从 FP16 输入解码至 FP32，逐 batch 调用 Netlib BLAS 3.10.0 的 `cblas_sgemm`，完成 alpha/beta 计算后按 round-to-nearest-even 转成 FP16。空规模与缩放分支按 API 规则建立参考结果，避免参考库的 quick-return 行为替代本任务的参数校验。

对有限值，采用任务书的混合容差：

```text
abs(actual - golden) <= 2^-10 + 2^-10 * abs(golden)
matched_ratio >= 0.99
```

并同时满足最大绝对误差限制。任务材料写作“0.1 或 32 ULP”，其中 ULP 档位的具体计算与适用条件应直接复用指定版本 opbase 标准实现，不能事后选择更宽阈值；验收记录保留最大绝对误差、最大误差所在 golden 值及对应 ULP。每个 batch 独立报告，避免大 batch 掩盖某个 batch 的错误。

非有限值按任务书单独判定：双方为 NaN、同号 Inf，或 NaN 与任一 Inf 的组合视为等效；异号 Inf 不等效；有限值与非有限值不等效。非有限元素不进入有限值统计，必须全部满足该规则。没有有限值的输出单独标记“有限值统计不适用”，避免除零；同时仍需通过非有限值检查。

CSV 中的 MERE/MARE 配置属于辅助检查，不能替代上述混合容差与最大误差要求。发生大 K 精度失败时，保留 FP32 累加配置、输入分布、误差位置及随 K 的曲线，通过小规模 FP64 诊断区分实现错误和参考路径舍入差异，不放宽阈值掩盖问题。

## 4.3 性能验证

以随包 200 条性能用例和 `gpu_baseline.csv` 为基线。性能倍率统一定义为 `GPU耗时/NPU耗时`，各 case 均需不低于 0.4，全部 case 的倍率算术平均需不低于 0.5。不能用总耗时之比或几何平均替代算术平均。

以下为任务包提供的前四项 GPU 数据及由 0.4× 门槛推导的 NPU 耗时上限，均不是 NPU 实测结果。

| m=n=k | batchCount | transa/transb | GPU 基线（μs） | NPU 门槛（μs） |
| --- | --- | --- | --- | --- |
| 256 | 16 | N/N | 5.285 | 13.2125 |
| 512 | 8 | N/N | 9.086 | 22.715 |
| 1024 | 4 | T/N | 25.346 | 63.365 |
| 2048 | 2 | N/T | 57.154 | 142.885 |

采集时固定软件版本、设备型号、时钟/功耗设置和输入参数，warmup 后用 msprof op 记录 task duration。单次 API 若产生多个 GEMM、epilogue 或缩放 task，必须按调用关联记录全部任务，分别给出 task 耗时和调用区间耗时；不只取最短 kernel 或第一组 batch。CPU golden、初始化和 GTest 进程开销不计入算子耗时。

随包 `verify_performance.py` 当前从 GTest 的毫秒级整条测试耗时解析结果，其中可能包含 Host 准备和 golden 计算；该值只适合流程检查，不能直接作为微秒级 kernel 的达标证据。实现阶段需要接入 profiler 数据，关联 case 与完整调用后再计算倍率，缺失数据标记待补测，不以零耗时或跳过失败项抬高平均值。

调优顺序为：先修正错误与越界，再观察小矩阵的启动开销和大矩阵的有效吞吐，随后比较候选 tile、双缓冲及批量调度。记录每项改动的输入类别、耗时变化和精度结果，性能收益必须通过相同条件的对照实验确认。

## 4.4 可维护性与交付检查

实现保留少量按职责组织的组件：参数检查与 launch plan、GEMM 计算实例、缩放/后处理实例、共享 tiling 定义。优先复用库已有能力，不为每种转置或尺寸生成独立文件。日志包含形状、转置、步长、所选路径及工作空间需求，便于复现分派问题。

设计转入代码验收时检查：公共 API 与半精度类型可编译；校验顺序与评审结论一致；1000 条精度用例及补充边界通过；200 条性能数据完整；输出 padding 与守护区验证通过；原有 GEMM 族回归不受影响。实现、精度和性能结论以实际日志及自测报告为准。

# 5. 参考资料

1. 随任务提供的 `aclblasHgemmStridedBatched_Atlas950PR_task_doc.md` 与 `test_cases/` 测试材料（2026-10-28 任务包）。
2. [社区任务流程与目录要求](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)。
3. [社区算子设计文档模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
4. [ops-blas GEMM arch35 实现](https://gitcode.com/cann/ops-blas/tree/master/blas/gemm/arch35)及[同族 strided-batched 实现](https://gitcode.com/cann/ops-blas/tree/master/blas/gemm_strided_batched/arch35)，用于工程接口与布局复用分析。
5. [cuBLAS gemmStridedBatched 接口](https://docs.nvidia.com/cuda/cublas/index.html#cublas-gemmstridedbatched)，用于接口参数和列主序语义对照；任务明确规定的边界行为以任务书为准。
6. [任务指定版本的生态混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
