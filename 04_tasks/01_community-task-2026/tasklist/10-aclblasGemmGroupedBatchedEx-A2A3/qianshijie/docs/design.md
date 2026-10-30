# aclblasGemmGroupedBatchedEx 算子设计文档（Atlas A2/A3）

> 任务：10 月社区任务 `aclblasGemmGroupedBatchedEx`（A2/A3）
>
> 目标源码仓：`cann/ops-blas`
>
> 设计目标路径：`blas/gemm_grouped_batched_ex/arch22/`
>
> 测试目标路径：`test/gemm_grouped_batched_ex/arch22/`
>
> 本文仅提交开发前设计，不包含算子源码、二进制、测试运行结果或性能通过声明。

# 一、需求背景（required）

## 1.1 需求来源

任务书为 `aclblasGemmGroupedBatchedEx_A2_task_doc.md`，本地资料包为
`aclblasGemmGroupedBatchedEx算子开发.zip`，SHA256 为
`f594aa4fc0ed8bde8c39e6791b9dd006c50771ee851599accd42ce452f252d89`。

任务要求在 Atlas A2/A3 系列产品上，基于
[ops-blas](https://gitcode.com/cann/ops-blas) 的句柄式 BLAS 工程框架，使用 Ascend C
实现 FLOAT32 的 `aclblasGemmGroupedBatchedEx`。接口语义对标
`cublasGemmGroupedBatchedEx`，实现代码落在
`blas/gemm_grouped_batched_ex/arch22/`。

设计文档按
[社区任务设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)
组织，并提交到
`cann/cann-ops-competitions`。

## 1.2 背景介绍

`ops-blas` 的公共头文件
`include/cann_ops_blas.h` 已声明 `aclblasGemmGroupedBatchedEx`，供不同产品线共用。
远端 `cann/ops-blas`（核对基线 `05b96ea22cea926ecfd43fc7ac691a795522bd61`）
当前已有面向 Ascend 950PR/950DT 的
`blas/gemm_grouped_batched_ex/arch35/` 实现与测试；Atlas A2/A3 尚无
`arch22/` 实现，因此本任务不新增平行 API，也不复制产品私有接口。

A2/A3 侧需要补齐以下能力：

1. 一次调用处理 `groupCount` 个分组，每组包含 `groupSize[g]` 个独立矩阵乘；
2. 各组独立拥有 `m/n/k`、`transa/transb`、`lda/ldb/ldc`、`alpha/beta`；
3. A/B/C 通过 Host 侧指针数组指向 Device 矩阵，全部采用列主序；
4. FLOAT32 输入、FLOAT32 累加、FLOAT32 原位输出；
5. 正确实现空组、零维、`k=0`、`alpha=0`、`beta=0/1` 和 Inf/NaN 传播；
6. 在 Atlas 800T A2（910B3）上满足任务书精度和性能门槛。

本设计沿用现有公共接口和 `arch35` 的组语义，但 `arch22` 的计算主体必须落到 AIC
Cube。`arch35` 的 FP8、双 kernel Cube/Vector FFTS 机制不能直接照搬到 910-class；
A2/A3 使用 `arch22` 的 `MatmulImpl`/ND 布局、MTE 搬运、Cube 累加与 AIV
epilogue 路线。

## 1.3 本次范围

| 项目 | 本次设计范围 |
| --- | --- |
| 硬件 | Atlas A2、Atlas A3（`arch22`） |
| CANN | 9.1.0 |
| 主数据类型 | `ACL_FLOAT` + `ACLBLAS_COMPUTE_32F` |
| 接口 | 已存在于 `include/cann_ops_blas.h`，不修改 ABI |
| 转置 | `ACLBLAS_OP_N/T/C`，实数路径下 C 与 T 等价 |
| 布局 | 列主序，仅支持 lda/ldb/ldc 表达的子矩阵 |
| 已有 950 路径 | 保留，本任务不改变 `arch35` 行为 |
| 代码交付 | 设计评审通过后新增 `arch22/` 与 CSV 驱动测试 |

FP16、BF16、FP8 和 INT8 输入属于扩展接口的候选能力，但没有随本任务提供正向测试
数据。`arch22` 第一阶段只声明并验证 FLOAT32；其他组合必须在 host 校验阶段返回
`ACLBLAS_STATUS_NOT_SUPPORTED`，不得调用 reference、CPU、vendor、另一个架构目录或
其他后端完成输出。README 的产品支持表必须同时表达“平台”和“dtype 子集”，不能因
接口是 `*Ex` 就宣称所有扩展 dtype 均已支持。

## 1.4 随包资料与证据角色

| 资料 | 角色 | SHA256 |
| --- | --- | --- |
| `aclblasGemmGroupedBatchedEx_A2_task_doc.md` | 语义和验收第一来源 | `db87649f405bbd1a5b550dda19685aa30b37e4c840f7d7db7d2a2a9b90827f5b` |
| `test_cases/gemm_grouped_batched_ex_test.csv` | 公开精度和性能工作负载 | `ec8dada8fe62b09d8cee719247fa9993a144c9334480e3fb68968087c4839660` |
| `test_cases/gpu_baseline.csv` | 外部 GPU 性能参考，仅作验收比较 | `5b6310088b598f86c661a855dcf1e8ea33889f4725438e06f0c3fecc6731a4c2` |
| `test_cases/gen_csv.py` | 固定种子用例生成器 | `dd7dc73ad65021ee39d30dc3951758953ac9f89def7ca40c1d823d140a71d15a` |
| `test_cases/verify_accuracy.py` | 精度运行入口 | `0b51de854c3c73f202338f2ae512bb9fcbd3c1528b1bb3c1dcacae045bc12c94` |
| `test_cases/verify_performance.py` | 性能运行入口 | `97cacf97c75b3f2d2466bd8522ed050aa024c2ca9aeb78685b25177c5a91d7b6` |

当前 CSV 包含 1200 条用例：1000 条精度和 200 条性能；`gpu_baseline.csv` 包含
200 条性能参考。任务书“本任务书 200 条”与资料包总数不一致，设计按以下解释执行：

1. 代码验收覆盖资料包中的全部 1000 条精度用例和 200 条性能用例；
2. 任务书所称“200 条”至少由 200 条性能用例完整覆盖；
3. 不以某一个公开 case id、CSV 行号或完整公开形状集合作为运行时分派依据。

## 1.5 资料包待确认项

设计评审需要先冻结以下边界；代码阶段不得自行选择更宽松或更窄的解释。

| 项目 | 已观察事实 | 设计决定 |
| --- | --- | --- |
| `alphaArray`/`betaArray` 内存 | 任务书写“Device 或 Host 内存”；cuBLAS 参数表与 ops-blas 现有 README 均为 Host | 按 Host 侧连续标量数组设计，host 直接读取每组标量；若要求同时接受 Device 标量数组，需要新增可判别的 ABI/属性，不能在现有 `const void*` 上可靠猜测 |
| 数组长度不一致 | 测试 README 要求 `len_mismatch_kArray` 返回 `INVALID_VALUE`，但接口不传任何组数组长度 | 列为 API 无法检测的调用方前置条件；不得以越界读取探测长度。该自动用例若保留，必须先由任务方补充可表达的 ABI 或改为 host 侧测试桩 |
| GPU 基线来源 | 测试 README 一处写 A100 回填，另一处写 NVIDIA P102-100；CSV 本身没有硬件、CUDA 和锁频元数据 | 设计只引用 CSV 数值，不宣称其硬件或吞吐来源；最终性能验收前由任务方确认基线的设备、时钟、采样和单位 |
| 采样次数 | 任务书要求 5 次 warmup、10 个有效样本；测试 README 写 warmup 后采样超过 50 次 | 正式验收按任务书 5/10；若测试工程另跑 50 次，只能作为稳定性补充，不能替换任务书写明的指标 |

# 二、需求分析（required）

## 2.1 功能语义

对分组 `g = 0..groupCount-1`，组内下标 `j = 0..groupSize[g]-1`，令扁平问题下标
`idx = groupSize[0] + ... + groupSize[g-1] + j`，计算：

```text
C[idx] = alpha[g] * op(A[idx]) * op(B[idx]) + beta[g] * C[idx]
```

其中：

```text
op(X) = X     （trans = ACLBLAS_OP_N）
op(X) = X^T   （trans = ACLBLAS_OP_T）
op(X) = X^H   （trans = ACLBLAS_OP_C；FLOAT32 下 X^H = X^T）
```

`op(A[idx])` 为 `m[g] x k[g]`，`op(B[idx])` 为 `k[g] x n[g]`，
`C[idx]` 为 `m[g] x n[g]`。全部矩阵采用列主序。

## 2.2 存储和转置映射

| 操作 | 物理存储 | 最小前导维 |
| --- | --- | --- |
| `transa=N` | `A = lda x k`，有效区 `m x k` | `lda >= max(1,m)` |
| `transa=T/C` | `A = lda x m`，有效区 `k x m` | `lda >= max(1,k)` |
| `transb=N` | `B = ldb x n`，有效区 `k x n` | `ldb >= max(1,k)` |
| `transb=T/C` | `B = ldb x k`，有效区 `n x k` | `ldb >= max(1,n)` |
| `C` | `C = ldc x n`，有效区 `m x n` | `ldc >= max(1,m)` |

列主序 `C(m,n)` 可视为行主序 `C^T(n,m)`。`arch22` kernel从该视角完成
`C^T = op(B)^T * op(A)^T`，因而可以直接使用 Ascend C 的 ND Matmul 数据流，
无需对 A/B 做大范围转置物化。四种公开转置组合分别映射到四个静态 Matmul
特化；C 与 T 在 FLOAT32 路径共用同一特化。

## 2.3 边界语义

| 条件 | 行为 |
| --- | --- |
| `groupCount=0` | 校验 handle、dtype 和 `groupCount >= 0` 后直接返回 SUCCESS，不再检查其余参数 |
| `groupSize[g]=0` | 该组没有 matrix instance，不产生工作项 |
| `m[g]=0` 或 `n[g]=0` | 该组为合法 no-op，不读 A/B/C，不产生工作项 |
| `k[g]=0` | 不读 A/B，仅计算 `C = beta * C`；`beta=1` 时不发射 kernel |
| `alpha[g]=0` | 不读 A/B，仅计算 `C = beta * C`；`beta=1` 时不写 C |
| `beta[g]=0` | 不读取 C 旧值，直接写 `alpha * op(A) * op(B)` |
| `beta[g]=1` | epilogue 计算 `alpha * product + C_old`，必须读取 C 旧值 |
| 输入含 Inf/NaN | 不做清洗；除 `beta=0` 不读取 C 外，按 IEEE 浮点运算自然传播 |

`alpha[g]=0` 且 `beta[g]=1` 是位精确 no-op。`alpha[g]=0` 的其他 beta 值路径只允许
使用向量 `Muls` 缩放，不得经过额外矩阵乘或类型转换。

## 2.4 接口

```cpp
aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[],
    const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[],
    const void* alphaArray,
    const void* const Aarray[], aclDataType Atype, const int ldaArray[],
    const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray,
    void* const Carray[], aclDataType Ctype, const int ldcArray[],
    int groupCount, const int groupSize[],
    aclblasComputeType_t computeType);
```

Host 侧组数组长度为 `groupCount`；`alphaArray`/`betaArray` 是长度
`groupCount` 的连续 FLOAT32 Host 数组；`Aarray`/`Barray`/`Carray` 是 Host 侧
指针数组，长度为 `problemCount = sum(groupSize[g])`，元素指向 Device 矩阵。

`Carray[idx]` 必须指向可写 Device 内存；不同 `idx` 的 C 矩阵不得重叠。矩阵地址对齐、
各 Host 数组实际长度、C 重叠和输入生命周期是调用方前置条件，C ABI 无法在所有情况下
检测。

## 2.5 需求拆解

1. 保留 `include/cann_ops_blas.h` 现有声明，不新增 ABI、不修改参数顺序；
2. 在 `blas/gemm_grouped_batched_ex/arch22/` 增加 host 校验、tiling 和 kernel；
3. 以 AIC Cube 覆盖主要 FLOAT32 GEMM，以 AIV 覆盖 beta-only 和 tiny-shape；
4. 支持组间异构形状、转置、scale、batch 以及 ld padding；
5. 复用 `test/gemm_grouped_batched_ex/` 的公共 param/golden 测试框架，新增
   `arch22/` CSV 驱动入口；
6. 覆盖资料包 1200 条用例，并记录精度、性能、采样次数、设备和 CANN 版本；
7. 更新 `blas/gemm_grouped_batched_ex/README.md` 的 A2/A3 产品与 dtype 支持矩阵；
8. 明示无 fallback；任一未支持 dtype、布局或容量条件必须报错，不能更换计算路径偷偷
   完成任务。

# 三、详细设计（required）

## 3.1 总体方案

采用“列主序转行主视角 + AIC Cube 主计算 + AIV epilogue + tiny-shape 向量路径”：

```text
Host 参数校验
  -> 组元数据和问题任务展开
  -> 指针数组/元数据 H2D
  -> beta-only / tiny-shape 工作项（如存在）
  -> 分轮 AIC FP32 Cube GEMM 到 workspace
  -> 同轮 AIV epilogue 写回 C
  -> handle stream 异步返回
```

正常 FLOAT32 GEMM 的主要算力由 AIC Cube 承担。AIV 只处理 O(m*n) 级的 beta 缩放、
alpha/beta 合成和 tiny-shape 直算，避免以 AIV 承担 O(m*n*k) 的主体计算。

## 3.2 Host 侧设计

### 3.2.1 参数校验顺序

为避免对空指针解引用和不同状态码竞争，统一按下列顺序校验：

1. `handle == nullptr`：`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. dtype 组合不在当前 A2/A3 支持集：`ACLBLAS_STATUS_NOT_SUPPORTED`；
3. `groupCount < 0`：`ACLBLAS_STATUS_INVALID_VALUE`；
4. `groupCount == 0`：`ACLBLAS_STATUS_SUCCESS`，不再访问其他参数；
5. `groupCount > 0` 时检查各 Host 组数组和 alpha/beta 指针非空；空指针返回
   `ACLBLAS_STATUS_INVALID_VALUE`；
6. 逐组校验 transa/transb 枚举。非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；
7. 逐组校验 `m/n/k/groupSize >= 0` 和 lda/ldb/ldc 前导维约束，非法返回
   `ACLBLAS_STATUS_INVALID_VALUE`；
8. 计算 `problemCount = sum(groupSize)`；当 `problemCount > 0` 时检查
   `Aarray/Barray/Carray` 外层数组非空，空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`；
9. 所有校验通过后再判断 quick return 和 kernel 发射。

前导维检查必须按转置前的物理存储计算，不能只使用逻辑 `m/k`：

```text
minLda = max(1, transa == N ? m : k)
minLdb = max(1, transb == N ? k : n)
minLdc = max(1, m)
```

不支持一组非零维度而另一组零维度时省略整体工作项的检查。每个分组独立生成工作项，
某个组的 no-op 不影响其他组。

### 3.2.2 组描述和任务展开

Host 为每组构建固定布局 `GroupParam`：

```text
transa/transb, m/n/k, lda/ldb/ldc,
alpha/beta, groupOffset/groupSize,
path (skip / beta-only / tiny / cube),
tileM/tileN/baseK,
logical M/N/ldc for transpose view
```

再将每个 A/B/C Host 指针值分别复制为 Device 侧 `uint64_t` 数组。Host 不读取矩阵数据，
只在需要时按行检查指针值。组描述和任务表使用 POD 定义并由 Host/Device 共用同一头文件，
字段宽度、对齐和打包方式保持一致。

工作项定义为：

```text
(group, batchInGroup, tileRow, tileCol, path)
```

工作项只依据合法运行时元数据和实现能力生成。Host 按估算工作量降序将工作项放入确定性的
任务表，再按 AIC/AIV 核数进行连续切片；不得按 case id、CSV 行号或完整公开形状集合分派。

### 3.2.3 执行路径

| 路径 | 条件 | 实现 |
| --- | --- | --- |
| skip | 组尺寸为零，或 `k=0/alpha=0` 且 `beta=1` | 不发射工作项 |
| beta-only | `k=0` 或 `alpha=0`，且 `beta!=1` | AIV 按列块执行 `C = beta*C`；`beta=0` 时不读 C |
| tiny | 输出规模低于经设备实测确定的 Cube 启动阈值，且输入满足向量装载约束 | AIV 在 UB 中完成分组 GEMM |
| cube-direct | `alpha=1`、`beta=0`、tile 与 ldc 均满足安全直写条件 | AIC 直接写 C，省去 temp 与 epilogue |
| cube-temp | 其余有效 FLOAT32 问题 | AIC 写 FLOAT32 temp，AIV 执行 `alpha*temp + beta*C` |

tiny 阈值不是公开 case 边界，而是由 `m*n*k`、问题数量、AIC/AIV 发射开销和设备实测形成的
通用容量判断。阈值变化必须重新验证所有覆盖分支，不能按 case 名增加例外。

固定长度 device 参数区最多承载 8 个组。Host 只在 `groupCount <= 8` 时选择一次下发的
Matmul 快速路径；device 入口在使用 `groupCount` 索引组数组前再次将其钳制到 `0..8`，
对负数和超出容量的外部计数均不越界读取。

### 3.2.4 Tiling 与分核

主 Cube tile 推荐为 `BM x BN = 128 x 128`、`BK = 64`：

- 一个 FP32 L0C tile 为 `128 * 128 * 4 = 64 KiB`，在 arch22 的 L0C 容量内；
- A/B 各 `128 x 64 x 4 = 32 KiB` 的 tile 可通过 L1/L0 流水复用；
- K 维在单任务内循环累加，不把完整 K 维写入 GM；
- 对较小 m/n，Host 选择 `64 x 64` 或 `32 x 32` tile，减少无效计算与尾块；
- 对大于一个输出 tile 的矩阵，任务按 `(rowTile, colTile)` 展开；
- Host 联合使用 `GetAicCoreCount()` 和 `GetAivCoreCount()`；获取为零视为环境错误，
  返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。

分组负载按任务数和工作量排序后分配，使用“整块 round-robin + 尾部按核再平衡”，避免一个
大组与多个小组同轮导致固定长尾。分核参数全部来自运行时元数据，不写死公开 case 形状。

### 3.2.5 Workspace

workspace 由 handle 管理并跨调用复用，按以下区间布局：

```text
[TilingData][GroupParam[groupCount]]
[aPtr[problemCount]][bPtr[problemCount]][cPtr[problemCount]]
[per-round FP32 temp tiles]
```

每个 chunk 的 temp 槽数由可用 AIC 核数和每轮任务数决定；AIC 完成后，同一 stream 上的
AIV epilogue 读取并释放该槽。扩容仅在 stream 已排空或调用处于无并发状态时进行；容量不足
且无法安全扩容时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，并记录所需字节数。

### 3.2.6 发射顺序

同一 `handle->stream` 上按以下顺序发射：

1. 所有有效 beta-only 工作项；
2. 所有 tiny 工作项；
3. Cube 工作项按固定 chunk 循环，每轮先 AIC GEMM，后 AIV epilogue。

不同问题之间的 C 不得重叠，因此组之间没有数据依赖。同一 stream 的顺序只用于保证
temp 槽在 AIC 写完后才被 AIV 读取，不允许依赖概率性时序。

## 3.3 Kernel 侧设计

### 3.3.1 AIC FP32 Cube 路径

列主序坐标先转为内部行主视角：

```text
internalA = op(B), internalB = op(A), internalC = C^T
internalM = n, internalN = m, internalK = k
```

Host 根据 A/B 的原始 N/T/C 组合选择四个静态 `MatmulImpl` 特化之一，并设置
`SetOrgShape/SetTail` 或等价接口，使每个输出 tile 的原矩阵行跨度和 `ld` 保持一致。
kernel 对每个 cube 任务执行：

```text
GM(A/B) -> L1/ND -> L0A/L0B -> FP32 Cube accumulation -> L0C
```

只在 L0C 中保留当前输出 tile。K 超过 `BK` 时循环累加；不将每个 K 分块写入 GM。
`FLOAT32` 路径显式关闭 HF32 截断模式，避免长 K 累加相对 `cblas_sgemm` 产生系统性偏差。

对齐部分：

- 完整 tile 使用 ND DataCopy/LoadData 快路径；
- M/N 尾块使用 `DataCopyPad` 或等价的有界读写，不越过 `m/n`；
- LD padding 只用于寻址，不写回；
- k 尾块按实现要求补零，但补零值不进入逻辑输出；
- 转置只改变读取索引和 Matmul 特化，不应为公开转置组合创建输入值分支。

### 3.3.2 AIV epilogue

generic Cube 路径写 FLOAT32 temp 后，AIV 对每个输出列块执行：

```text
C_tile = alpha * temp_tile + beta * C_old_tile
```

- `beta=0`：跳过 C_old 读取；
- `beta=1`：计算 `alpha * temp_tile + C_old_tile`，不能省略 C 旧值读取；
- `alpha=1`：取消乘以 1 的临时操作；
- 统一在 FLOAT32 域完成合成，再写回 FLOAT32 C；
- 只写 `[0,m) x [0,n)` 的有效区域，不覆盖 ld padding；
- 采用 2D 尾块安全的搬运方式，读取和写回使用同一有效范围。

### 3.3.3 beta-only 路径

按每个 problem 的列块生成 AIV 工作项。`beta=0` 使用填充/置零，不读取 C；其他 beta
值按有效 `m x n` 区域读取、`Muls`、原位写回。`k=0` 与 `alpha=0` 共用该路径，但在
host 侧先消除 `beta=1` 的纯 no-op。

### 3.3.4 tiny-shape 路径

tiny 路径用于避免小矩阵与大量小组触发 Cube 启动和数据搬运开销。Host 只有在尺寸、
对齐和累加容量全部满足时选择该路径。AIV 按输出块装载 A/B tile，使用有界的
`Axpy/Fma` 累加；alpha 可先折叠到 A 或 B tile，beta 在最后合成。

此路径仍由 `arch22/` 自有 kernel 完成，不得调用 cblas、PyTorch、官方 whole-op 或
其他 backend。超过 tiny 阈值的问题必须走 Cube，不得因性能不好从 Cube 路由到 CPU。

### 3.3.5 浮点与边界

1. FLOAT32 乘加不承诺逐位一致，但必须满足任务书混合容差；
2. `alpha=0` 的 `C=beta*C` 与 `k=0` 的缩放路径逐元素可由单次乘法和存储完成；
3. `beta=0` 路径不得读取 C，避免未初始化值或 0×Inf 污染结果；
4. A/B 中出现 Inf/NaN 时按 IEEE 规则传播，不做 sanitize、clamp 或随机修复；
5. 所有尾块必须有有效范围掩码和 padding 隔离；
6. `C` 的不同 problem 区域不得重叠；重叠行为由调用方契约排除。

### 3.3.6 同步与资源

AIC、AIV 和 Host 之间的可见性只由 handle 绑定的 stream 顺序与显式设备同步建立：

- temp 写入、epilogue 读取和 C 写回之间不得依赖 wall-clock 或隐式 CPU sleep；
- 同一 temp slot 在 epilogue 完成后才可在下一轮复用；
- 一次调用内不创建额外 CUDA/NPU 线程池或 Python 回调；
- 返回前只保证异步任务已成功下发；调用方读回结果前负责 stream 同步；
- kernel 退出前排空本 kernel 自有搬运和写出管道，不能在 GM 写尚未完成时释放临时资源。

## 3.4 支持硬件

| 产品 | 架构路径 | 本任务状态 |
| --- | --- | --- |
| Atlas 800T A2（910B3） | `arch22` | 本次实现与性能自验证目标 |
| Atlas A2 训练/推理系列 | `arch22` | 本次实现目标 |
| Atlas A3 训练/推理系列 | `arch22` | 本次实现目标 |
| Ascend 950PR/950DT | `arch35` | 保留现有实现，本任务不修改 |

## 3.5 约束限制

- 仅支持列主序和 lda/ldb/ldc 表达的子矩阵；不支持要求额外指数/stride 的非连续布局；
- 第一阶段只声明 FLOAT32；FP16/BF16/FP8/INT8 没有被本任务的正向资料验证；
- A/B/C 矩阵位于 Device，组数组和 alpha/beta 位于 Host；
- C 原位覆写，不返回视图，不允许不同 problem 的 C 区域重叠；
- `Carray` 元素由调用方保证可写；beta 非零时 C 旧值必须有效；
- 不支持 broadcast，不支持跨 problem 的 A/B 复用假设；
- 确定性计算不要求；`alpha=0` 且 `beta=1` 的 no-op 必须位保持不变；
- CANN 版本以任务书 9.1.0 为验收环境，其他版本需单独验证。

# 四、可维可测分析（required）

## 4.1 精度标准

golden 由 Netlib BLAS 3.10.0 的 `cblas_sgemm` 对每个 problem 独立生成，逐元素比较
逻辑 `m x n` 输出。

| 项目 | FLOAT32 验收口径 |
| --- | --- |
| 相对容差 `rtol` | `2^-13` |
| 绝对容差 `atol` | `2^-13` |
| matched ratio | `>= 0.99` |
| 最大绝对误差 | `<= 1e-2` 或 `32 * ULP` |
| 逐元素公式 | `abs(actual-golden) <= atol + rtol * abs(golden)` |

Inf/NaN 不套用有限值误差公式。双方同为 NaN、双方为同号 Inf，或一方为 NaN 而另一方
为 ±Inf，按任务书视为等价；异号 Inf 和有限值与非有限值判为失败。非有限元素不计入
matched ratio 和 max-abs 统计，单独记录数量与通过数。

`alpha=0`、`beta=1` 的 no-op 使用位精确比较；其他 FLOAT32 GEMM 结果因累加顺序不同，
不做 bit-exact 声明。

## 4.2 性能与内存标准

正式性能设备为 Atlas 800T A2（910B3），输入 FLOAT32，调用方矩阵在 Device 内存。
任务书判据为：

```text
NpuAverageUs <= GpuBaselineUs / 0.8
```

即 NPU 平均单次耗时不超过 GPU 标杆耗时的 1.25 倍。采样按任务书执行：5 次 warmup 后
连续 10 次有效样本取算术平均，非 SUCCESS 样本不计入且必须保留原因。报告必须同时记录：

| 字段 | 要求 |
| --- | --- |
| 设备 | 具体 NPU 型号、物理/Card/Chip/Logical 标识 |
| 软件 | CANN 9.1.0、编译器与测试工程提交 |
| 每 case | groupCount、groupSize、m/n/k、转置、ld、alpha/beta |
| 时间 | GPU 参考、NPU Avg time、单位 `us`、倍数 `x` |
| 采样 | warmup=5、repeat=10，失败样本和过滤规则 |
| 产物 | 原始日志、CSV、汇总表；截图不是唯一证据 |

当前 `gpu_baseline.csv` 前四条为 1024³、2048³、4096³ 和
`groupCount=2, groupSize=[1,1], 1024³`。其 GPU 型号和采样来源尚不一致，代码验收前
必须确认来源；本设计不据此声称任何性能已经达标。

任务书声明“不涉及”显式内存上限，但实现仍必须：

- 在 host 计算 workspace 总字节数前检查整数溢出；
- 不把完整 A/B/C 的副本作为 workspace 常驻；
- temp 以输出 tile 为单位复用；
- 所有 malloc/memcpy/launch 失败都返回失败码并保留日志，不静默降级。

## 4.3 测试设计

测试落在 `test/gemm_grouped_batched_ex/arch22/`，复用
`test/gemm_grouped_batched_ex/` 的 param、golden 和 CSV 加载框架，新增 A2/A3 CSV
驱动入口。测试不读取 OpForge 的 task/golden/private 数据。

| 类别 | 覆盖 |
| --- | --- |
| L0 基础 | N/N、N/T、T/N、T/T、含 C 的实数等价组合；小尺寸多组 |
| shape 扫描 | 1、小质数、2 的幂及 ±1、非对齐、矩形、到 4096 级 |
| scalar | 组内与组间不同 alpha/beta；0、1、-1、大值 |
| leading dimension | compact、+4/+8 padding、转置混合、仅 lda 或 ldc padding |
| fill | 均匀、正态、零、交替、极大、Inf、NaN |
| group | `groupCount` 0/1/小/中/2 的幂；`groupSize` 0/1/小/中/大 |
| 边界/负向 | 空 handle、非法枚举、非法 dtype/computeType、负维度、负 group、负 groupSize、非法 ld、外层数组空指针 |
| 精度 | 资料包全部 1000 条精度用例 |
| 性能 | 资料包全部 200 条性能用例，并记录 workspace 峰值 |

测试程序需逐 problem 生成独立 golden，并按组传播同一 alpha/beta。负向用例只检查返回码；
正向用例对全 `m x n` 有效输出做精度比较。CSV 中若存在与设计冻结项不一致的用例，先修
CSV 或其生成器，再实现依赖该期望的分支。

## 4.4 自验交付与复现

代码阶段需要提交以下证据：

1. `test/gemm_grouped_batched_ex/arch22/` 的测试源码、CSV 与 README；
2. `test_cases/verify_accuracy.py` 等入口实际调用的命令、退出码和 pass marker；
3. 精度汇总：总数、PASS/FAIL、失败 case、最大误差和 matched ratio；
4. 性能汇总：每个 case 的 benchmark、warmup/repeat、平均 us、倍率和 PASS/FAIL；
5. 具体 NPU、CANN、编译器版本，构造/运行时间与 git commit；
6. 原始日志和 CSV；截图如需，从原日志确定性生成并保留源文本；
7. 源文件 SHA256 清单，并在提交后复核远端分支和固定 commit。

未运行 NPU 性能前只能报告“设计完成”或“正确性待验证”，不能使用 GPU 参考值替代 NPU
实测，也不能把部分 case PASS 描述为完整性能达标。

# 五、兼容性、安全与维护性

## 5.1 接口和架构兼容

- `include/cann_ops_blas.h` 不改签名；A2/A3 与 950PR 继续共用接口；
- 只新增 `arch22/`，按 SoC 编译期选择，不改变 `arch35/` 源码和已有测试；
- README 的平台与 dtype 表分开描述，避免把 950PR 的支持范围误写成 A2/A3 的支持范围；
- 不增加 `OpV2`、私有 `aclblasSgemmGroupedBatchedEx` 或其他平行 API；
- 文件命名沿用 `gemm_grouped_batched_ex_` 前缀，复用仓内测试框架，不为单个 case 建文件。

## 5.2 无回退边界

被评估的 `arch22` 路径只允许：

1. 编译并执行本任务 `blas/gemm_grouped_batched_ex/arch22/` 中的 Ascend C 代码；
2. 使用经 CANN 预检的公共 CANN/Ascend C/ACL runtime 依赖；
3. 使用本任务构建出的 tiling、kernel、wrapper 和测试程序。

禁止通过 cblas、CPU/host、PyTorch、torch_npu、官方 whole-op、`arch35/` 二进制、
其他 backend、peer solution、baseline cache 或任务 golden 生成 A2/A3 的 evaluated
输出。未支持条件必须 fail loud，不能返回看似成功的零值。

## 5.3 风险和维护策略

| 风险 | 设计控制 |
| --- | --- |
| 组间异构导致核间长尾 | 统一任务表按估算工作量排序，再按连续区间分核；用 200 个性能样本验证，不按 case 特殊化 |
| 任意 lda/ldb/ldc 造成越界 | host 先校验最小前导维；kernel 仅按有效 m/n/k 和运行时 ld 寻址；尾块有界读写 |
| FP32 Cube 的 HF32 截断 | 显式关闭 HF32，精度失败时先检查累加模式和 k 尾块，不通过放宽阈值修复 |
| 指针数组是 Host 数组但元素指向 Device | Host 只复制指针值，不在 Host 解引用矩阵；kernel 通过 `uint64_t` 地址解引用 |
| tiny 和 Cube 阈值形成公开 case 拟合 | 阈值只由维度、工作量和设备实测决定；新增或修改阈值必须覆盖边界和 private/generalization 形状 |
| 2 次 launch 导致小问题开销 | `alpha=1,beta=0` 直写快路径；tiny AIV 路径；其余保持 temp 语义正确 |
| 资料包口径冲突 | 在 PR 正文和评审回复中显式列出，要求维护者冻结后再进入代码验收 |

# 六、Checklist 覆盖

- [x] 需求背景、需求来源和现有实现分析
- [x] 数学公式、列主序布局与 N/T/C 语义
- [x] 输入、输出、属性的完整约束表
- [x] host 参数校验顺序、错误码与 no-op 边界
- [x] AIC/AIV 执行路径、tiling、分核、workspace 和同步
- [x] 对齐、尾块、ld padding、Inf/NaN 和 bit-exact 特例
- [x] Atlas A2/A3、CANN 9.1.0 和 arch35 共存关系
- [x] 精度阈值、性能设备、采样和证据产物
- [x] 1200 条随包用例和负向用例设计
- [x] 无 fallback、无 benchmark overfitting、无私有接口
- [x] 已列出需要任务方确认的资料包冲突

# 参考资料

1. `aclblasGemmGroupedBatchedEx_A2_task_doc.md`
2. [CANN 社区任务 2026 目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)
3. [ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)
4. [cuBLAS cublasGemmGroupedBatchedEx](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmGroupedBatchedEx)
5. [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
6. Netlib BLAS 3.10.0 `sgemm.f`
