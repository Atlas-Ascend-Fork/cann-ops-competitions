# 需求背景（required）

## 需求来源

**11月社区任务：aclblasSgemmBatched / aclblasHgemmBatched 算子开发（A2/A3）**。交付对象为 ops-blas 的两个既有 C 接口，不是新增 aclnn 算子。本文为待实现、待上板验证的设计；tile、缓冲区和调度参数是设计值，不是性能测量值。

| 依据 | 本设计采用的口径 |
|---|---|
| 任务书 §2.1–2.5 | 数学语义、公开 ABI、结构校验优先级、stream 异步执行 |
| 任务书 §3.1–3.5 | CANN 9.1.0、精度双门限、kernel 性能正式判据、自验规模 |
| `test_cases/{s,h}gemm_batched/README.md` 及各自 CSV | 输入 token、用例标识、计时日志和 GPU 对照键 |
| `ops-blas` 的 `origin/master`，版本 `55ef791b77f3` | 下述源码现状及测试接入点；不以其他分支工作区作为基线 |

## 背景介绍

BLAS 的 pointer-batched 接口适合在一个 stream 上计算形状相同、地址独立的多组矩阵。与 strided-batched 不同，知道第 0 个矩阵的位置并不能推导第 1 个矩阵的位置。若按 batch 在 host 循环调用普通 GEMM，既增加下发次数，又需要 host 知道 Device 上的地址表内容。本设计将地址解析和循环都放在 Device。

### 现状分析

| 仓内对象 | 已有机制 | 在 arch22 中的取舍 |
|---|---|---|
| `arch35/gemm_batched_host.cpp`、`hgemmBatched_host.cpp` | S 校验、`ExecuteSgemmBatched`；H 独立入口和 `HgemmLaunchContext` | 合并实数校验模板，不继承 950 对空矩阵地址表的限制 |
| `arch35/gemm_batched_host_common.h` | 核数校验、MN 切分、列主序交换、临时空间、alpha/beta 后处理 | 保留列主序交换的数学关系；不在每次调用申请整批临时矩阵 |
| `arch35/gemm_batched_cube_impl.h` | `GbProcessKL1Loop`、L0 循环、按 task 解码、half 输出 | 保留 K 留在一核累加；不用 arch35 的 Te 搬运接口 |
| `arch35/gemm_batched_kernel.cpp`、`*_kernel.h`、`*_tiling_data.h` | GEMM 与按列 epilogue/early-exit 分工，多个 tiling 结构 | 改为按值传 POD、编译期区分 AIC/AIV，不照搬 strided 入口 |
| `arch22/cgemm_batched_host.cpp`、`*_kernel.h`、`*_tiling_data.h` | `ValidateCgemmBatchedParams`、工作区规划、`*_do` 直调 | 沿用同一 stream 和类型化 launch 约定，不复用复数工作区公式 |
| `arch22/cgemm_batched_kernel.cpp`、`*_interleaved.cpp` | 向量、拆分、Cube、合并等路径；`MatmulImpl` 静态实例与向量 tiny | 只需要一次实数乘积，不拆实部虚部，不做四路实数归并 |
| `arch22/cgemm_batched_device_utils.h`、`*_float.h` | `CgbtBatchPtr`、物理地址与复数数值辅助 | 借鉴按 batch 读地址，不移植特殊复数乘法 |
| `test/gemm_batched/sgemm_batched` | 公共 param/golden、arch35 GTest 和 wrapper | 使用已有测试框架，arch22 单独归一化输入列并收集混合容差统计 |
| `test/gemm_batched/cgemm_batched/arch22` | 空 handle、偏移指针、stream 顺序、并发工作区回归、预分配性能缓冲 | 补非等距指针、独立 stream 与 padding 的可执行断言 |
| `test/gemm/hgemmBatched/arch35` | half 生成、指针数组 wrapper、`CsvPerformance` | half 位模式转换及预置缓冲思路可用；本任务 H 不另建测试目标 |

### 功能分析

输入为 handle、两项转置、尺寸、两个 host 标量、三组 ld 和三个 Device 地址表；输出为地址表 C 指向的原矩阵，不返回新 Tensor。一般路径读 A、B，按 beta 决定是否读 C；无乘积路径只访问 C；无输出路径不访问矩阵。

# 需求分析（required）

## 需求描述

为 Atlas A2/A3 实现与 cuBLAS 同名接口参数序列一致的 S/H batched GEMM。保持 `include/cann_ops_blas.h` L606、L616 声明及 `aclblasStatus_t` ABI；只支持 host pointer mode 的标量。非法枚举按 ops-blas 返回 `INVALID_ENUM`，不是 cuBLAS 的 `INVALID_VALUE`。

## 需求拆解

| 子问题 | 实现边界 | 可观察结果 |
|---|---|---|
| 校验与空操作 | 在 host 依调用约定决定哪些地址必须存在 | 错误码优先级正确；合法 no-op 无 kernel |
| 地址与转置 | 在 Device 解引用每批地址；由 ld 计算列偏移 | 非方阵、非等距、9 种转置组合结果正确 |
| 可扩展计算 | batch 和 MN tile 并行；K 局部循环 | 无跨核 K 原子归约，不依赖 batchCount 个 launch |
| 精度与搬出 | FP32 累加，在写回阶段组合标量项 | beta=0 不传播旧 C 的 NaN；H 不提前转 half |
| 工程验收 | 双 dtype 的同一套 GTest、两套计时证据 | 每个 case 可通过 op、参数、seed 复现 |

# 详细设计（required）

## 算子分析

### 数学公式

设 `b ∈ [0,batchCount)`，`0 ≤ i < m`，`0 ≤ j < n`。下标以元素计，不是字节计。

```text
Aop(i,p) = A[b][i+p*lda]  (N) ; A[b][p+i*lda]  (T/C)
Bop(p,j) = B[b][p+j*ldb]  (N) ; B[b][j+p*ldb]  (T/C)
P(i,j)   = sum(p=0..k-1, Aop(i,p)*Bop(p,j))
C[b][i+j*ldc] <- alpha*P(i,j) + beta*C[b][i+j*ldc]
```

公式表示非退化数学结果，不要求用 `0*NaN` 实现丢弃项。执行决策（**先完成全部结构校验**）：

| 条件，按从上往下首次命中 | 读矩阵 | 写矩阵 | 下发 |
|---|---|---|---|
| batchCount=0 或 m=0 或 n=0 | 无 | 无 | 无 |
| (k=0 或 alpha=0) 且 beta=1 | 无 | 无 | 无；仍要求非空输出的 Carray 本体有效 |
| (k=0 或 alpha=0) 且 beta=0 | 无 | C 逻辑区写零 | 1 次 AIV |
| (k=0 或 alpha=0) 且 beta 为其他值 | C | C 逻辑区乘 beta | 1 次 AIV |
| 有乘积且 beta=0 | A、B | C 逻辑区 | 1 次 |
| 有乘积且 beta≠0 | A、B、C | C 逻辑区 | 1 次 |

### 支持数据类型

| 入口 | A/B/C 与 alpha/beta | 运算域 | 写回 |
|---|---|---|---|
| `aclblasSgemmBatched` | float，4 B | FP32 输入，禁用 HF32，FP32 累加与 epilogue | FP32，不量化 |
| `aclblasHgemmBatched` | `aclblasHalf`，2 B | 按 IEEE binary16 位模式解码；乘积累加及一般 epilogue 为 FP32 | 最终一次 round-to-nearest-even 到 FP16 |

`aclblasHalf` 不按同宽整数数值转换。比较零和一用解码后的数值，正负零均命中零分支。无 mixed-input、复数及 BF16 公开入口。

### 支持形状

每个 batch 为二维列主序 ND，有且只有前导维可携带填充。各 batch 共用 shape、ld、标量和 trans，无 batch 间广播。令 `ta=(transa!=N)`、`tb=(transb!=N)`，仅在枚举校验成功后归一化。

| 矩阵 | 物理分配（行距×列数） | 逻辑参与区 | ld 最小值 |
|---|---|---|---|
| A，N / T,C | lda×k / lda×m | m×k / k×m | max(1,m) / max(1,k) |
| B，N / T,C | ldb×n / ldb×k | k×n / n×k | max(1,k) / max(1,n) |
| C | ldc×n | m×n | max(1,m) |

#### 参数契约（任务书 §2.4）

记 `T=float | aclblasHalf`（分别对应 S、H），`D=(batchCount>0 && m>0 && n>0)`，`G=D && k>0 && decodedAlpha!=0`。下表状态名省略 `ACLBLAS_STATUS_`；正常有限数据按各 dtype 可表示域给入，特殊浮点值按未读/传播规则测试，不扫描有限性。

| 参数 | 方向、类型、位置与维度 | 含义 / 有效域 | 异常及允许空的精确条件 |
|---|---|---|---|
| handle | 入，`aclblasHandle_t` 标量 | 已创建，含绑定 stream | 空 → HANDLE_IS_NULLPTR |
| transa | 属性，`aclblasOperation_t` 标量 | A 的 N/T/C | 其他 → INVALID_ENUM |
| transb | 属性，`aclblasOperation_t` 标量 | B 的 N/T/C | 其他 → INVALID_ENUM |
| m | 属性，int 标量 | op(A)、C 行数，≥0 | 负 → INVALID_VALUE；零为空输出 |
| n | 属性，int 标量 | op(B)、C 列数，≥0 | 负 → INVALID_VALUE；零为空输出 |
| k | 属性，int 标量 | 点积长度，≥0 | 负 → INVALID_VALUE；零不读 A/B |
| alpha | 入，`const T*`，Host，1 元素 | 乘积系数，H 以 half 可表示值给入 | 任何路径空 → INVALID_VALUE |
| Aarray | 入，`const T* const[]`，Device，[batchCount] | 各元素指向上述 A 物理矩阵 | 本体仅 `!G` 可空，否则空 → INVALID_VALUE |
| lda | 属性，int 标量 | A 列距，≥max(1,ta?k:m) | 不足 → INVALID_VALUE，即使无乘积 |
| Barray | 入，`const T* const[]`，Device，[batchCount] | 各元素指向上述 B 物理矩阵 | 本体仅 `!G` 可空，否则空 → INVALID_VALUE |
| ldb | 属性，int 标量 | B 列距，≥max(1,tb?n:k) | 不足 → INVALID_VALUE，即使无乘积 |
| beta | 入，`const T*`，Host，1 元素 | 旧 C 系数；零时旧 C 可未初始化 | 任何路径空 → INVALID_VALUE |
| Carray | 出/入出，`T* const[]`，Device，[batchCount] | 各元素指向可写 ldc×n 矩阵 | 本体仅 `!D` 可空，否则空 → INVALID_VALUE，含 beta-only 恒等 |
| ldc | 属性，int 标量 | C 列距，≥max(1,m) | 不足 → INVALID_VALUE |
| batchCount | 属性，int 标量 | 独立矩阵数，≥0 | 负 → INVALID_VALUE；零允许三个数组为空 |

各必用数组元素的非空、Device 可达性及其容量由调用者保证；数组本体不是 host 指针列表。对任何 i、j，C[i] 的可写区不得与 A[j]/B[j] 的可读区重叠；不同 C 不重叠。禁止用读取 padding 或回写邻列实现尾块。

## 算子实现

### 实现方案

**有乘积两次下发，FP32 tile 经 GM 中转。** 主路径由 AIC 用 Ascend C Matmul 高层 API 算出 FP32 tile 落到 GM 工作区，再由同一 stream 上的 AIV epilogue 入口按列做 `alpha`/`beta` 与半精度收窄后写回 C；无乘积只下发一次 AIV 的 beta-only 入口。两次下发之间不插 host 同步，工作区按 `min(任务数, 槽数)` 分轮复用。batch 地址表始终留在 Device。

**可行性门已实测（V1 结论）。** arch22 上 `MatmulType<TPosition::VECCALC, CubeFormat::ND, float>` + `IterateAll(localTensor)` 在 CANN 9.1.0 / bisheng `--npu-arch=dav-2201` 下编译失败：`kernel_operator_fixpipe_v2_impl.h:314:47: error: the 1st parameter maybe need a type '__cbuf__ half *'`（8 errors），即该直出路径要求 half 目的类型，FP32 累加结果不能直落 UB。故主路径定为「AIC 出 FP32 tile 到 GM 工作区 + AIV epilogue」，不再宣称零工作区或单次下发；工作区 = `槽数 × BM × BN × 4 B`，槽数由 handle 现有工作区容量决定（默认 32 MiB → 2048 槽），不随 batchCount 线性增长，也不在调用热路 malloc。校验、key、任务切分、精度与性能判据不受此影响。

#### 3.2.1 host 侧设计

两个公开名字、参数序列、状态类型保持不变。S 文件以 float 实例化共同模板，H 文件以 `aclblasHalf` 实例化；不在 H 入口将 half 地址表转换为 float 地址表。
共同头只在实现内部可见，负责校验、标量解码、分派、POD 和 launch。模板按实参类型生成代码，不以 `void*` 抹去标量类型。

**校验先于 quick return。** `ValidateCgemmBatchedParams` 的实际顺序是 handle → alpha/beta 地址 → m/n/k/batchCount → trans → ld → 三张表（batchCount>0 即要求非空）。沿用其 `CHECK_RET`、`OP_LOGE` 和分项返回写法；S/H 的优先级和必用地址按 §2.4，不能原样调用复数校验。

| 顺序 | S/H 检查或动作 | 首错处理 |
|---:|---|---|
| 1 | handle，尚不访问 stream | HANDLE_IS_NULLPTR |
| 2 | transa、transb 分别只取 N/T/C；成功才把 OP_C 归一化为 OP_T | INVALID_ENUM |
| 3 | m、n、k 非负 | INVALID_VALUE |
| 4 | 按归一化转置检查 lda、ldb，再检查 ldc；零维也需要合法 ld | INVALID_VALUE |
| 5 | batchCount 非负 | INVALID_VALUE |
| 6 | alpha、beta 地址非空；S 取 float，H 用 `blas_common::HalfToFloat` 解码 | INVALID_VALUE |
| 7 | 按参数契约的 G 检查 A/B 本体，按 D 检查 C 本体 | INVALID_VALUE |
| 8 | 无输出或无乘积且 beta=1：SUCCESS | 不查询资源，不下发 |

每个错误记录接口、出错参数和值/下界；不读表内指针作诊断。校验通过后才取 `_aclblas_handle` 的 `h->stream`，不创建、切换或同步 stream。Host 标量的两个 float 快照按值随 POD 下发，不延长 host 标量的生命周期。

**尺寸分派依据。** 先列出 `CgbtDispatch` 的实际优先级，避免把复数 small 的 GM 槽误作实数 tiny 的一部分。下表 `s=max(m,n,k)`、`v=可用 AIV 数`，所有乘法先扩为 64 位。

| arch22 C 参考顺序 | 条件 | 原有选择 |
|---|---|---|
| direct | 复数 ld 字节距装不进 uint32，或 `CgbtUseDirect` 成立 | 先于 vector16/small；不规划复数工作区 |
| direct 的尺寸判断 | `s≤8`，或 `m*n≤4096` 且不是下面两种 Cube 候选 | 排除 N/N 的 16、32、64 立方及 `m*n>256 && k≤64` |
| vector16 尝试 | 未命中 direct，`s≤16 && alpha=1 && beta=0`；`v>0` 且 batch≤2v，或同一 v>0 下紧凑 N/N 16 立方且 batch≤8v | `CgbtTryVector`；未处理才继续 |
| small | 余下 `s≤64 && alpha=1 && beta=0` | `CgbtRunSmall` 按 16/32/64 面板规划、分批 |
| general | 其余 | `CgbtPlanWorkspace` 按两对输入平面和四个乘积平面求容量及 batchChunk |

**S/H 决策表（按行首次命中）。** 保留先排除无乘积、先试整块小向量再进 Cube 的取舍；不照搬复数专用的误差修正、函数句柄或工作区容量条件。表中阈值是设计边界，性能分界待实测。

| 条件（已排除 quick return） | path / shapeId | 下发与核类型 |
|---|---|---|
| `k==0 || alpha==0` | 1 / 0 | `gemm_batched_beta_scale_kernel_do`，仅 AIV；此处 beta 必不等于 1 |
| `s≤8`，含 1×1×1、batchCount=65536 | 2 / 0 | `gemm_batched_tiny_kernel_do`；取 direct 的极小尺寸边界，batch 在核内循环 |
| `8<s≤16` | 2 / 0 | 同一 tiny 入口；覆盖 vector16 的尺寸，非默认标量走同一 FP32 epilogue |
| `s>16`，包括原 small 的 32/64 立方 | 0 / 0 | S 或 H 的 Cube 入口，按 MN tile 发给核组 |

16 的依据是整矩阵三维都不超过一个 16×16 基础块的相应边；不能仅凭 batch 很大就把大矩阵改走 AIV。小尺寸与大 batch 同时出现时，避免反复启动利用率低的 Cube。原 vector16 的 2v/8v 是**选特化实现**的限制，不是「超过就不用 tiny」的数学条件；本设计通用 tiny 不假定紧凑 NN、默认标量或连续 batch。

**工作空间及直调。** C 参考的 `CgbtRunBetaScale` 将 `CgbtBetaScaleTilingData scale{}` 填入 m、n、ldc、batchCount 和 beta 信息，再以 `(blocks,h->stream,Carray,scale)` 直调；一般乘积的 GEMM POD 则带 workspace 内平面偏移。这里只沿用前一种按值传参与同 stream 下发的方式。

| 状态 | 对 handle 工作区的动作 | 理由 |
|---|---|---|
| 主路径 | 借用 handle 现有工作区，按 `min(任务数, 槽数)` 轮转，每槽 `BM×BN×4 B`（64×64 → 16 KiB） | AIC 写、AIV 读同一 stream 的 FP32 tile；不按 batchCount 线性增长，不逐调用 malloc |
| beta-only | 0 字节 | 单次 AIV 下发，逐列 `Duplicate(0)` 或 `Muls(beta)` |
| 如果某个实例只能输出 GM | 不能宣称该实例达到主路径要求 | 至少增加 FP32 tile 的一次写和一次读，以及消费者可见性同步；须另行评审，不作为运行时后备 |

对需下发的路径，从 handle 对应设备取得有效核数（参照 `CgbtAicCoreCount` / `CgbtAivCoreCount`）；Cube 计可配对消费结果的核组，纯向量计 AIV。零核报 INTERNAL_ERROR，不将零强行钳成一。
Host 按 dtype/path 只调用一次对应的 `*_kernel_do(blockDim, h->stream, ..., tiling)`。beta-only 不传 A/B；其余传三个数组本体的 GM 地址，不在 host 循环调用。若封装可报告同步 launch 错误则记 key、blockDim 与错误码并返回 EXECUTION_FAILED，不为获得异步状态而同步。

**key 与 ABI。** `gemm_batched_tiling_data.h` 是两端唯一来源；字段枚举、移位常量、掩码与 POD 一起发布，不使用 C++ 位域。编码以无符号 32 位进行：

```text
key = dtype | (transA << 1) | (transB << 2) | (path << 3) | (shapeId << 5)
```

| 字段 | 宽度、移位、掩码 | 枚举 |
|---|---|---|
| dtype | 1，0，0x01 | 0=float；1=half |
| transA | 1，1，0x02 | 0=N；1=T（含 C） |
| transB | 1，2，0x04 | 同上，按 BLAS 顺序，不按 Cube 左右顺序 |
| path | 2，3，0x18 | 0=cube；1=beta-only；2=tiny；3 保留 |
| shapeId | 2，5，0x60 | 0=各路径的初始形状；1–3 保留，未验证不得下发 |

高位一律为零。解码为 `(key & mask) >> shift`；例如 H、T/N、cube、初始形状为 `0x03`，相同输入的 tiny 为 `0x13`。测试遍历 2×2×2 个基本组合，检查 host 编码与模板实参回读一致。

**Tiling POD。** 下表自上而下即内存顺序；同一行多字段按书写顺序连续存储。对象零初始化，不放指针、bool、size_t 或 C++ 模板对象；数组地址是独立 launch 参数。布局按 8 字节对齐。

| 字段名 | 类型 | 写者 → 读者 | 字节偏移 / 含义 |
|---|---|---|---|
| m,n,k | uint32_t 各一 | host 校验 → kernel | 0,4,8；BLAS 原轴，不是交换后的形状 |
| lda,ldb,ldc | uint32_t 各一 | host 校验 → 地址计算 | 12,16,20；元素距，不是 DMA 字节距 |
| batchCount,blockDim | uint32_t 各一 | host 分核 → kernel | 24,28；总批数、有效核数 |
| BM,BN,BK | uint32_t 各一 | host 选形 → kernel | 32,36,40；原轴输出 tile、静态 K 基块；不是 K 分核数 |
| tilesM,tilesN | uint32_t 各一 | host 切分 → 解码 | 44,48；两方向向上整除 |
| key | uint32_t | host 分派 → kernel | 52；上述枚举 |
| tilesPerBatch | uint64_t | host 切分 → 解码 | 56；先提升再乘 |
| jobsPerBlock | uint64_t | host 分核 → 循环 | 64；tasks/blockDim |
| extra | uint32_t | host 分核 → 循环 | 72；tasks%blockDim |
| alpha,beta | float 各一 | host 解码 → 向量运算 | 76,80；H 也存数值 FP32，不存 half 原位模式 |
| tinyMaxDim | uint32_t | host 阈值 → tiny | 84；当前为 16，约束 UB 最大片及边界检查 |

共 88 B。两端编译同一标准布局定义，对实际 POD 作 `static_assert(sizeof(...) == 88)`，以 `offsetof` 逐字段断言上表数值，另断言对齐与可平凡复制；不可按 host 编译器的默认 packing 猜测偏移。

**块数及分工。** 初始 cube 静态形状拟用 BM=BN=64、BK=`CBLAS3_BASE_K`（64），兼顾一块 FP32 结果 tile 的槽容量与 AIV 列缓冲；不是声称复数默认 tile 为 64。beta-only 用 64×1、BK=0（无 K 循环）。两者均使用同一任务坐标。

```text
tilesM = m/BM + (m%BM != 0); tilesN = n/BN + (n%BN != 0)
tilesPerBatch = uint64(tilesM) * tilesN
tasks = uint64(batchCount) * tilesPerBatch
blockDim = min(tasks, 可用核数)       // 仅非空工作到达这里
q = tasks / blockDim; r = tasks % blockDim
start(u) = uint64(u)*q + min(u,r); count(u) = q + (u<r)
```

求积、最大元素偏移及其字节换算检查可表示性，不能溢出后再比较；异常跨度在访问前报错。特别是 ld 本身合法但转为字节 stride 超宽时，以逐列 64 位地址计算代替窄 stride 转换。只有 `GetBlockIdx() < blockDim` 才干活；混合实例按 API 的逻辑核组而非两个独立任务生产同一 tile。

#### 3.2.2 kernel 侧设计

**任务与地址。** 每核处理 `[start,start+count)` 连续区间，首尾可在某个 batch 中间。对任务号 t：

```text
b = t / tilesPerBatch; tile = t % tilesPerBatch
i0 = (tile % tilesM)*BM; j0 = (tile / tilesM)*BN
mm = min(BM,m-i0); nn = min(BN,n-j0)
```

相邻核交界满足 `start(u)+count(u)=start(u+1)`，最后一个区间终于 tasks；一个 C 元素只有一个写者。K 不跨核切分，无 atomic。
`CgbtBatchPtr` 以 `GlobalTensor<uint64_t>` 绑定数组，`GetValue(b)` 读第 b 个 64 位地址。进入新 b 才读一次各必用表，缓存基址；每个 tile 以该批基址加偏移重绑 `SetTensorA/B`，绝不由上一批首址加矩阵大小。beta-only 只读 C 的地址项。

**Cube 类型与转置。** 列主序在 ND 行主序 Matmul 中解释为 `Pᵀ = op(B)ᵀ × op(A)ᵀ`。左 X 是 BLAS B 的行主序视图，右 Y 是 BLAS A 的行主序视图；交换运算次序而非物理转置整块 GM。

| BLAS A/B | Cube X/Y（左、右的编译期转置） | X 起始元素 | Y 起始元素 |
|---|---|---|---|
| N/N | plain/plain | j0×ldb | i0 |
| N/T | trans/plain | j0 | i0 |
| T/N | plain/trans | j0×ldb | i0×lda |
| T/T | trans/trans | j0 | i0×lda |

同一模板取 `<T,ta,tb>`：左 `MatmulType<TPosition::GM,CubeFormat::ND,T,tb>`，右同型末参为 `ta`，目标类型拟为 `MatmulType<TPosition::VECCALC,CubeFormat::ND,float>`，bias 禁用。`GetMMConfig` 的静态形状配置交给 `GetMatmulApiTiling` 生成 `MatmulApiStaticTiling`，作为 `MatmulImpl` 的静态实参。不能直接用 `cblas3::MmPlain` 作 C，因为它固定为 GM float。

入口以 dtype 选择 float/half 模板，再按 ta、tb 的 if/else 选择 `<false,false>`、`<false,true>`、`<true,false>`、`<true,true>`；模板内部在编译期交换左右类型。合计 8 个乘积实例，共用任务解码、尾块、指针重绑与 epilogue，不复制八份循环。
`cblas3::ShapeParams()` 的默认 tile 是 128×128、baseK=64、singleCoreK=8192；`CgbtGemmBodyT` 使用局部 `TPipe`、`Init`、`DisableBias`，且 `SetHF32(false)`。本设计保留 S 全 mantissa；H 为原生 half 输入、FP32 累加。静态 singleCoreK 不是公开 k 上限；超过配置时在本核沿 K 分段，以 FP32 合并在 UB，不跨核、不提前缩放或转 half。

**物理形状与尾块。** 对照 `CgbtSetOrgShape` 的五参（orgM,orgN,orgKa,orgKb,orgKc），令 `ldX=ldb, ldY=lda`，Cube 逻辑形状为 n×m，`L` 为 UB 行距（元素，至少 mm，按 local 输出要求对齐），不是 ldc。

| X/Y | SetOrgShape 五参 |
|---|---|
| plain/plain | n, ldY, ldX, ldX, L |
| trans/plain | ldX, ldY, k, k, L |
| plain/trans | n, m, ldX, ldY, L |
| trans/trans | ldX, m, k, ldY, L |

按该批真实 pitch 定位输入，而 `SetSingleShape(nn,mm,kCount)` 给出本 tile 的有效行、列、K；参考体用的是它，不是 `SetTail`。K 分段时 X 偏移再加 `p0*(tb?ldb:1)`，Y 加 `p0*(ta?1:lda)`；最后一段只算剩余 K。库管理搬入和 Cube 内部补齐，不把补齐数当作逻辑尺寸。
主路径调用 `IterateAll(gmTile, enAtomic)` 输出到 GM 工作区的 FP32 tile；K 超过 `singleCoreK` 时后续段以原子加累加，首段清写。实例结束按参考调用 `End()`。tile 布局为 nn 行、行距 `BN`、每行前 mm 个 float 有效，AIV 段按此布局读回。

**AIV epilogue（每个有效列 j）。** 第一行对应 BLAS 第 j0 列，列内行 i 连续。以 `off=uint64(j0+j)*ldc+i0` 定位 GM，按 `mm*sizeof(T)` 搬运，而非按 ldc 搬运；不从工作区跨列连读。

| 顺序 | 运算与类型 | 边界 |
|---:|---|---|
| 1 | `Muls`：FP32 乘积列 × alpha | 直到全部 K 完成才缩放 |
| 2 | beta≠0 才 `DataCopyPad` 读旧 C；H 用 `Cast` 升 FP32 | beta=0 不构造旧 C 读请求 |
| 3 | `Muls`：旧 C × beta；`Add` 到结果列 | 都在 FP32，有效长度 mm |
| 4 | S 保持 float；H 用 `Cast` 一次 RNE 回 half | 半途不写回 half；用相邻 half 中点验证偶舍 |
| 5 | `DataCopyPad` 将一列 `[i0,i0+mm)` 写至 C[b]+off | 下一列另加 ldc；不回写对齐填充 |

UB 核算按静态上界：AIV 每次只处理一列，需要 FP32 结果列、旧 C 列（原 dtype）、FP32 缩放列与 half 输出列；实现取 `3 × GB_TILE × 4 B`（768 B）并以 `constexpr` + `static_assert` 与 192 KiB 对账。Matmul 自身的 L1/L0 由库管理，不并入 UB 预算。
同 stream 的两次下发之间由 stream 顺序保证 AIC 结果对 AIV 可见；同一轮内 tile 槽只被本轮 AIV 消费，下一轮覆盖前已按 stream 顺序完成。若后续改成单 kernel 混合核组，必须补 `CrossCoreSetFlag/WaitFlag` 握手，不以删同步换性能。

**两个退化 kernel。** 不把无乘积乘法伪装成 Matmul，也不为 tiny 留 GM 缓冲。

| 路径 | 计算方法 | 不能做的操作 |
|---|---|---|
| beta-only，beta=0 | 每段 UB `Duplicate(0)`；按列搬出 | 不读取 C 旧值，不绑定 A/B |
| beta-only，其他 beta | 按列读 C；H 升 FP32，`Muls(beta)`，最终按 dtype 写回 | 不对 `[m,ldc)` 缩放；beta=1 在 host 已返回 |
| tiny | 各批将有效 A/B 列搬入 UB；按 ta/tb 的物理视图重排为逻辑列 | 不推定不同批地址连续，也不把 ld 的间隙当数据 |
| tiny 的点积 | 对每个 j 清零 FP32 累加列，逐 p 广播 Bop(p,j)，向量乘 Aop(:,p) 并加到累加列；H 先升 FP32 | 不按元素反复读 GM，不移植复数四积和纠错 |
| tiny 的输出 | 调用相同的 alpha/beta 与列写回逻辑，切换到下一批 | 不因小尺寸降低精度、覆盖哨兵或改变 C/T 等价性 |

`CgbtTinyRead/Prepare/Accumulate/Store` 提供读入、广播、按 p 累加和按地址表存回的参照。`CgbtLocalLoad` 则是局部面板的 LoadData 参数构造器，不是读取 batch 表的接口；本设计不复制其面板搬运到 Matmul 之外。

### 与 arch35 实现的差异

| 维度 | arch35 S/H 源码 | arch22 设计及不能直接搬的原因 |
|---|---|---|
| 入口 | `ExecuteSgemmBatched` / `ExecuteHgemmBatched` 分开，共用 host_common 的 tiling、临时空间、early-exit 与后处理辅助 | 两入口共用校验和计划模板；任务的表可空契约不同 |
| 标量与输出 | H 在 host `HalfToFloat`；默认标量可直接写 half，否则 FP32 临时结果由 AIV 读回、缩放、Cast | 保留解码和最终收窄位置，不需要默认标量专属 GM 快路 |
| 批调度 | Cube 按 tiling 处理 batch/MN，列主序交换输入；后处理按自身 AIV tiling 调度 | 保留交换关系，由同一核组拥有整 K 和输出 tile，切批时在核内重绑 GM 指针 |
| 输出路径 | 一般标量时规划临时地址表和 GM FP32 输出区，在同一 stream 追加 alpha/beta kernel | 同构：AIC 出 FP32 tile 到 GM 工作区 + 同 stream 的 AIV epilogue；差异是本设计按 `min(任务数,槽数)` 分轮复用槽，不调 `batchCount*m*n` 大小的临时矩阵 |

## 支持硬件

| 产品 | 架构/软件 | 验证安排 |
|---|---|---|
| Atlas A2 系列（含 Atlas 800T A2） | arch22，CANN 9.1.0 | 性能正式设备 910B3；精度及编译待验证 |
| Atlas A3 系列（含 Atlas 800T A3） | arch22，CANN 9.1.0 | 同一资源查询和代码路径；以实际 SoC 配置编译，建议补精度回归 |

任务允许自验覆盖一种款型。950PR/arch35 仅作接口、结构及回归参考，不是本任务性能设备。

## 算子约束限制

- 不是任意 stride/view 算子，不调用 AutoContiguous/ViewCopy，不做维度补全或广播。
- 不接收 0 维 Tensor。运行时 int 尺寸经校验后 tiling，无动态 shape 图编译要求。
- 不复制或扫描 Device 地址表到 host，不验证元素内容或尝试检查跨 batch 别名。
- 输出原地，不返回视图；调用方在 stream 完成前不得释放/修改正在使用的地址表和矩阵。
- 不要求逐位确定性，不依测试编号、随机种子、golden 或输入分布分派；按混合容差验收。
- 任务书 §3.4 无额外内存性能门限；上文 UB/GM 容量为实现安全边界，不是内存性能实测。

# 可维可测分析

## 精度标准/性能标准

### 数值参考与精度判定（§3.2）

在调用前快照旧 C，各 batch 只在逻辑区生成参考。S 对齐 Netlib BLAS 3.10.0 `cblas_sgemm(CblasColMajor,...)` 的数学语义；冻结 golden 将已量化的 FP32 输入及标量扩为 float64，逐点积累加、缩放、最后收窄 FP32。直接调用单精度 cblas 是允许的独立对照，不声称库内部使用 double。H 没有 Netlib `cblas_hgemm`，先从真实 half 位模式解码，参考按 float64 累加、缩放，收窄到 FP32，再按 IEEE RNE 到 FP16。报告标记使用的 golden 类型，不用设备输出生成 golden。

| dtype | atol | rtol | required_matched_ratio | max_abs_error_limit |
|---|---:|---:|---:|---|
| FP32 | 2^-13 | 2^-13 | 0.99 | 1e-2 或 32 ULP（FP32） |
| FP16 | 2^-10 | 2^-10 | 0.99 | 1e-1 或 32 ULP（FP16） |

```text
finite_err(i) = abs(double(actual[i])-double(golden[i]))
matched(i) = finite_err(i) <= atol + rtol*abs(golden[i])
matched_ratio = matched 个数 / 全部逻辑元素数（含所有 batch）
PASS = (matched_ratio >= 0.99) AND 最大误差门通过 AND 特殊值/哨兵检查通过
```

绝对上限与 32 ULP 是备用口径而非相加：报告最大绝对误差和对应参考值，同时列出逐点 ULP 距离；采用 ULP 档时用输出 dtype 相邻可表示值作尺度，不用别处大值的 ULP 放宽近零点门限。非有限值先按分类比较（NaN 对 NaN，Inf 同号），不把 NaN 转成零参与 max；类别不符直接失败。空输出断言状态、零写入、零下发，不算 0/0 的比例。CSV 中 `mere_threshold/mare_multiplier` 供框架诊断，不代替上述联合判据。

误差主要来自 FP32 乘加舍入、Cube 微块内部加法结合顺序、缩放及抵消，H 还多一次最终 half 舍入。切 K 不增加 panel 间的格式转换，保留 FP32 L0C 且不跨核合并 partial；因而不会随 panel 数重复引入 half 量化误差。对通常量级数据，FP32 单次舍入单位约 2^-24，小于两档相对容差；H 的最终输出按同一 RNE 规则舍入。**这不是任意长 K 或严重抵消时的相对误差证明**：以 K=4095/4096、正负抵消、近零、极值/次正规数专项检验；超阈值时按首个错误的 b/i/j、K panel 和部分和比对，定位累加或收窄误差后重验，不改阈值。所有通过率待实测。

### 性能验收与优化（§3.3）

**正式判据只比较 kernel 与 kernel。** Atlas 800T A2/910B3 上用 `msprof op`，逐 PF case 过滤，核对实际下发名、stream 和次数。5 次 warmup 不计；接着 10 次 SUCCESS 的有效采样取算术平均。发生异常说明原因并重采足 10 个，不挑最短 10 个。取 `OPPROF_*/OpBasicInfo.csv` 中 `Task Duration(us)`，同一次混合下发按一次逻辑任务关联，不把单个子核时长当整算子时间。

```text
npu_kernel_ms = sum(Task Duration(us) of 10 samples) / 10 / 1000
逐例达标目标：npu_kernel_ms <= 1.25 * gpu_ms
等价性能倍率：gpu_ms / npu_kernel_ms >= 0.8
```

以下仅为自验包 A100 cuBLAS、nsys/profiling 的 kernel-only 原始标杆，不是 NPU 耗时；两边使用相同 trans、尺寸、ld、标量和 batchCount。

| 来源（`test_cases/` 下） | id | trans | m,n,k | batchCount | alpha,beta | lda,ldb,ldc | gpu_ms |
|---|---|---|---|---:|---|---|---:|
| `sgemm_batched/gpu_baseline.csv` | sgemmBatched-base-000 | N,N | 256,256,256 | 16 | 1,0 | 256,256,256 | 0.0716800019 |
| `sgemm_batched/gpu_baseline.csv` | sgemmBatched-base-001 | N,N | 512,512,512 | 8 | 1,0 | 512,512,512 | 0.247603197 |
| `hgemm_batched/gpu_baseline.csv` | hgemmBatched-base-000 | N,N | 256,256,256 | 16 | 1,0 | 256,256,256 | 0.0183296004 |

**辅助数据。** GTest 先预分配 Device 矩阵及表并 H2D；warmup 5 次，随后用同 stream 的 `aclrtEvent` 包围连续 10 次下发，在结束事件完成后取 elapsed/10 并统一为 us。打印 `[PERF] case=<case_name> ... avg_us=<数值>`。含下发影响，不含造数、H2D/D2H、golden；只给 `verify_performance.py` 交叉验证，不以 GTest 整数毫秒替代。未解析到该行判 FAIL；缺有效 GPU 基线记 NO_REF，正式验收不得放行。非默认标量须使用已复核的同参标杆，不沿用默认标量值。

| 优化点 | 实施动作 | 预期收益方向（均待实测） |
|---|---|---|
| 去每调用申请/同步 | 按值参数，handle/stream 资源复用，不申请 GM 临时区 | 减少事件计时中的下发间隙，不将 host 节省计入 kernel 收益 |
| Matmul 与 tile 中转 | 核对静态配置、FP32 tile 列缓冲与槽复用开销 | 减少 MTE 等待，短 K 另看流水填充代价 |
| 核数对齐 | P=min(资源,任务数)，商余均分 | 减少空核和末波长尾 |
| tile 形状 | tiny 初始边长 16、Cube 初始边长 64 | 小问题少补零，大问题复用 panel、提高 Cube 忙碌度 |
| 无旧 C | beta=0 跳过旧 C 读入与缩放 | 减少逻辑列读流量，仍从 UB 按列写回 |
| batch 核内循环 | 同一 plan 多次解码、同 batch 缓存 3 个地址 | 减少 launch 与重复 tiling，不依赖等距地址 |

若小矩阵慢，分别看 event/kernel 差、地址表读和空闲核；大矩阵慢则看 MTE2、Cube、FIX、AIV 等待及 UB tile 的交付开销。调参先比较相同数据/标杆和精度，再记录 tile/key/blocks/workspace。本文没有 NPU 延迟、吞吐或提升比例实测值。

### 自验方案（§3.5）

**同一 harness，两份输入。** `test/gemm_batched/sgemm_batched/arch22/` 注册一个参数化测试。中间行模型含 `op` 列并按 `op=S/H` 分派类型；本包两份 CSV 实际表头都没有 `op`，按输入文件名严格注入该列，若未来有显式列则校验一致。用 `(op,case_name)` 作唯一键，两包都有 `TC_L0_001`，不可仅按 case_name 合并。单包验收时选择相应 op、保持 `[PERF] case=` 后的原名以适配脚本。

```text
case_name,description,transA,transB,m,n,k,alpha,aarray_fill,lda,barray_fill,ldb,
beta,carray_fill,ldc,batchCount,expect_result,mere_threshold,mare_multiplier,random_seed
```

arch22 adapter 以表头名取值，将 `aarray_fill/barray_fill/carray_fill` 映射为内部 fill 三元组；`NULLPTR` 表示数组本体空，不是矩阵填零。仓内 arch35 `sgemm_batched_param.h` 读取 `a_fill/b_fill/c_fill`，其缺省值、独立 null 标志与 CSV 行为均不改动。仅在 arch22 adapter 兼容旧名；同列两种名称冲突报 schema 错误，不静默使用随机值。空 ld 推导紧凑下界，显式非法 ld 原样传递，`null` 标量传空，`999` 转非法枚举，`description=handle_null` 显式传空 handle。负向行先安全构造最小参数，不按负尺寸申请缓冲。

| 测试维度 | 必须验证 |
|---|---|
| 数量与分类 | S/H 各 600 行：各 500 条精度/状态（非 PF），各 100 条 `TC_PF`；合计精度全量 1000，性能子集 200。对照执行清单，MISSING 不算通过 |
| 规模 | small ≤64 KiB，medium ≤2 MiB，large 更大；按单批 A/B/C 最大逻辑占用×batchCount。性能排除零维及零 batch，约 12% 非默认标量（alpha≠0） |
| 尺寸 | 0、1、2、3、5、7、16、31/32/33、63/64/65、127/128/129、255/256、511/512、1023/1024、2047/2048、4095/4096；含非方阵和两种最大立方 |
| batchCount | 0/1/2/3/8/16/64/256/1024/4096/16384/65536；补后四档 1×1×1，检查任务覆盖、首尾地址、无重写 |
| 分布 | 一般随机输入整例交替：`RANDOM_NORM_5_5` 均匀 [-5,5] 50%，`GAUSS_NORM_0_1` 正态 N(0,1) 50%；保留 seed。不把任务书分布描述误当 `RANDOM_GAUSS_*` token |
| 标量与值域 | alpha/beta 取 0、±1、0.5、±1e4、可表示随机值；H 先转 half，另测 5E-8、65504、Inf/NaN、溢出与次正规 |
| 地址 | 故意打乱或偏移各批首址，A/B/C 独立分配；与连续等距输入对照；不构造重叠输出作为正常用例 |
| 转置 | 3×3 全覆盖；给同一输入分别传 OP_C、OP_T，保持物理 ld 一致，结果逐元素一致 |
| padding | ld 取下界与 +1/+8/+64；各列 `[m,ldc)` 及缓冲边界置位模式哨兵，D2H 后逐字节比对，独立于浮点阈值 |
| 未读分支 | beta=0、C 含 NaN；alpha=0、A/B 含 NaN 或表空；k=0、beta≠1；k=0、beta=1 且 Carray 空仍报错 |
| 零输出 | m/n/batchCount=0 与三个空表组合，标量有效、ld 合法应 SUCCESS，无 Device 任务；换空 alpha 应 INVALID_VALUE |
| 负向 | 空 handle；分别非法 transa/transb；逐个负 m/n/k；逐个 ld 低于相应下界；负 batchCount；空 alpha/beta、必用 A/B/C 表；只验状态码 |
| 多错优先级 | 空 handle+非法 trans；非法 trans+负尺寸；空输出+非法 ld；负 batch+空标量；确保先校验后 quick return |
| 异步 | 绑定非默认 stream，前置写入输入、后置读出；不同 stream/handle 并发，确认工作区不窜写 |

用例分类核对：两种 dtype 的 L0/MN/AB/LD/CV 分别 18/61/12/15/24；S 的 FL/ED/EX 为 21/63/286，H 为 27/69/274。额外定向回归与 CSV 清单分别报告，不挤占 500 条。4096³ golden 应分列/块生成以控制 host 峰值，任务包单例预算 512 MB，不同时保留完整 double 输入、结果及其全部副本。

**执行和留证。** 准备 CANN 环境、设备、Netlib、ATK/GTest；先校验每包的 schema、行数、PF 参数键唯一性。使用 `gemm_batched_npu_common.h` 的 buffer/context 辅助完成分配、地址表 H2D、同步读回并清理；性能 fixture 单独预置，不能把同步 wrapper 放入计时循环。

```bash
bash build.sh --ops=sgemm_batched,hgemm_batched --soc=ascend910b3
# 从各自自验包运行；H 的脚本路径映射到共同的 arch22 测试目录/二进制
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b3 --csv ./sgemm_batched_test.csv
# H 包改传 ./hgemm_batched_test.csv，选择 H 行；分别核对 500 条
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend910b3
msprof op --application="./build/test/gemm_batched/sgemm_batched/arch22/sgemm_batched_test --gtest_filter=<单个PF实例>" --output=./prof_case
```

精度日志逐例保留 op、参数、seed、返回码、matched_ratio、max_abs/ULP 和最坏下标；错误码用例记录实际值与期望值。性能按 `(op,transA,transB,m,n,k,batchCount,alpha,beta,lda,ldb,ldc)` 对照各包 baseline，保留 10 个原始 us、均值、GPU ms、倍率和辅助 event 日志。归档设备/SoC、CANN 版本、构建参数和 profiler 导出文件；未采集不填通过。

## 兼容性分析

不新增公共接口、私有平行 API 或新 pointer mode；arch35 保留原 kernel、CSV 与构建行为。arch22 新增 S/H 导出时按 SoC 选择源文件，避免一个库同时链接两套同名符号；原 cgemm_batched 源和测试保持独立。README 明确 A2/A3 空矩阵语义及调用方生命周期责任。

### 分阶段实现与验收

| 阶段 | 做什么 | 离开阶段所需的证据 |
|---|---|---|
| V1 可行性 | ~~在目标 CANN 上编译并运行两种 dtype × 四种转置的 local 输出实例~~ **已完成前置探针**：FP32 直出 UB 在 arch22 编译失败（见 §3.2.1），主路径改为 GM tile 中转；后续只需确认两次下发在四种转置 × 两种 dtype 下数值一致 | 已存：编译错误原文与探针日志；待存：8 个实例的 FP32 tile 对照与列写回哨兵 |
| V1 正确性 | 实现共同校验、POD、商余分核、两个退化入口及最终 RNE；接入两包统一测试 | 1000 条精度/状态用例全过；另附指针偏移、负向首错、未读 NaN 与 padding 哨兵逐字节结果 |
| V2 性能 | 在 V1 固定的契约下，分析 tiny/Cube 尺寸边界与核数；只按同参数对照调整，不按 case 名特化 | 200 条 PF 逐例给出 msprof kernel 均值、GPU id、倍率及辅助 event 数据；按 §3.3 逐例验收，当前均待实测 |

验收前最小检查清单：

- [ ] A2/A3 构建配置选中 arch22 S/H 源，arch35 原目标仍能构建；公共头 ABI 未改。
- [ ] 两侧 POD 大小、字段偏移、key 往返测试通过；主路径无 workspace 请求或隐藏中转。
- [ ] 核对 K 尾、两种 ld 方向、末批余数、beta=0 未读和 beta=1 恒等零下发。
- [ ] S/H 各 500 个非 PF 键完整，不因同名合并；负向状态、特殊值、C 列尾与前后保护区分别核对。
- [ ] S/H 各 100 个 PF 键完整；逐例同参标杆、10 个有效样本、stream 与下发次数可追溯。
- [ ] 归档输入、seed、设备、CANN、构建版本、原始日志与 profiler 文件；无基线、无样本或含异常的记录不记 PASS。

### 文件清单（拟实现，不是本次文档提交中的代码修改）

| 动作 | 文件 | 预计行数 | 职责 |
|---|---|---:|---|
| 新增 | `blas/gemm_batched/arch22/sgemm_batched_host.cpp` | 45–70 | S ABI、float 实例化；经共用模板完成校验、计划与下发 |
| 新增 | `blas/gemm_batched/arch22/hgemm_batched_host.cpp` | 45–70 | H ABI、half 实例化；解码标量，不转换 Device 指针表 |
| 新增 | `blas/gemm_batched/arch22/gemm_batched_host_common.h` | 230–320 | 共用模板：参数首错、工作区决策、切块、按值 launch |
| 新增 | `blas/gemm_batched/arch22/gemm_batched_tiling_data.h` | 70–110 | POD、key 枚举、大小及偏移断言 |
| 新增 | `blas/gemm_batched/arch22/gemm_batched_kernel.h` | 35–55 | 声明 `sgemm_batched_kernel_do`、`hgemm_batched_kernel_do`、`gemm_batched_beta_scale_kernel_do`、`gemm_batched_tiny_kernel_do` |
| 新增 | `blas/gemm_batched/arch22/gemm_batched_kernel.cpp` | 380–480 | 上述 launch 入口及模板实例化、指针重绑、Matmul 与 UB 后处理、两个 AIV 退化实现 |
| 修改 | `blas/gemm_batched/README.md` | 35–55 | 在既有 S/H 节按产品支持情况、函数原型、参数说明、约束说明、精度验证、调用示例的次序更新；保留 arch35 说明 |
| 修改 | `test/gemm_batched/sgemm_batched/CMakeLists.txt` | 5–15 | 经 `ops_blas_add_gtest_tests` 注册 arch22 目标，arch35 不变 |
| 新增 | `test/gemm_batched/sgemm_batched/arch22/sgemm_batched_test.cpp` | 280–390 | 同一 GTest 按 op 驱动状态、精度、性能，打印可关联日志 |
| 新增 | `test/gemm_batched/sgemm_batched/arch22/sgemm_batched_npu_wrapper.h` | 140–210 | 分配、地址表 H2D、stream/event 计时与资源回收 |
| 新增 | `test/gemm_batched/sgemm_batched/arch22/gemm_batched_case.h` | 120–180 | 解析两套 CSV，归一化字段，保留显式非法参数 |
| 新增 | `test/gemm_batched/sgemm_batched/arch22/gemm_batched_oracle.h` | 160–230 | S/H 参考、双门限、特殊值和 padding 字节比较 |

行数为新增/改动预算。沿用仓内许可证头和 LF；每个头文件上限 500 行。两份 CSV 直接取任务包，不覆盖 arch35 数据。提交前分别核对实现、测试、README 和自测证据。

### 本设计引用到的仓内符号 → 出处文件

均相对 `ops-blas` 的 `origin/master`（`55ef791b77f3`）（提交号见需求来源）。为缩短路径，以下 `C22=blas/gemm_batched/arch22`、`G35=blas/gemm_batched/arch35`、`TC=test/gemm_batched/cgemm_batched/arch22`、`TS=test/gemm_batched/sgemm_batched`。表列的是已有调用或定义，不含文件清单中拟新增的入口。

| 仓内符号 | 出处文件 | 可核对的事实 |
|---|---|---|
| `ValidateCgemmBatchedParams`、`CHECK_RET`、`OP_LOGE` | `C22/cgemm_batched_host.cpp` | 分项校验、状态与日志 |
| `CgbtAicCoreCount`、`CgbtAivCoreCount` | `C22/cgemm_batched_host.cpp` | 查询并在 handle 缓存核数 |
| `CgbtUseDirect`、`CgbtDispatch` | `C22/cgemm_batched_host.cpp` | direct 优先级与尺寸排除条件 |
| `CgbtTryVector`、`CgbtRunSmall` | `C22/cgemm_batched_host.cpp` | packedSquare、2v/8v、16/32/64 |
| `CgbtPlanWorkspace`、`CgbtAllocateWorkspace` | `C22/cgemm_batched_host.cpp` | 复数平面、溢出检查、按容量分轮 |
| `CgbtBetaScaleTilingData`、`CgbtGemmTilingData` | `C22/cgemm_batched_tiling_data.h` | 按值传递的结构 |
| `CgbtRunBetaScale`、`cgemm_batched_beta_scale_kernel_do` | `C22/cgemm_batched_host.cpp` | 填 POD、block 数、h->stream 直调 |
| `CBLAS3_TILE_M`、`CBLAS3_TILE_N`、`CBLAS3_BASE_K` | `blas/common/helper/complex_blas3_tiling_data.h` | 128、128、64 |
| `cblas3::MmPlain`、`cblas3::MmTrans` | `blas/common/helper/complex_blas3_arch22.h` | GM/ND/float 类型，不是 UB 输出类型 |
| `cblas3::SINGLE_K`、`cblas3::ShapeParams` | `blas/common/helper/complex_blas3_arch22.h` | 静态 singleCoreK 与基块配置 |
| `GetMMConfig`、`GetMatmulApiTiling`、`MatmulApiStaticTiling` | `C22/cgemm_batched_kernel.cpp` | 编译期 config → tiling |
| `MatmulType`、`MatmulImpl`、`MatmulBothTrans` | `C22/cgemm_batched_kernel.cpp` | ND 操作数及双转置实例 |
| `CgbtSetOrgShape`、`SetOrgShape` | `C22/cgemm_batched_kernel.cpp` | 四组 stride 槽位 |
| `CgbtGemmBodyT`、`TPipe`、`Init`、`DisableBias`、`SetHF32` | `C22/cgemm_batched_kernel.cpp` | 局部 pipe 和精度配置 |
| `GetBlockIdx`、`SetSingleShape`、`SetTensorA`、`SetTensorB` | `C22/cgemm_batched_kernel.cpp` | 编号、尾块与重绑 |
| `IterateAll`、`End` | `C22/cgemm_batched_kernel.cpp` | 参考目标为 dGlobal；local 重载/后端能力须另证 |
| `CgbtBatchPtr`、`SetGlobalBuffer`、`GetValue` | `C22/cgemm_batched_device_utils.h` | 每批一次 64 位表项读取 |
| `CgbtLocalLoad` | `C22/cgemm_batched_device_utils.h` | 局部 LoadData 参数，与读表不同 |
| `CgbtTinyRead`、`CgbtTinyPrepare`、`CgbtTinyAccumulate`、`CgbtTinyStore` | `C22/cgemm_batched_device_utils.h` | 复数 tiny 向量循环先例 |
| `TPosition::VECCALC`、`LocalTensor` | `C22/cgemm_batched_device_utils.h` | UB 缓冲先例；不是 Matmul 输出证明 |
| `ExecuteSgemmBatched` | `G35/gemm_batched_host.cpp` | 默认标量直写与一般标量二次下发 |
| `ExecuteHgemmBatched`、`blas_common::HalfToFloat` | `G35/hgemmBatched_host.cpp` | 独立 H 入口、host 解码 |
| `GemmBatchedPrepareTempWorkspace`、`GemmBatchedLaunchAlphaBetaPostProcess` | `G35/gemm_batched_host_common.h` | arch35 共用工作区和后处理 |
| `DataCopyPad`、`Cast`、`Muls`、`Add`、`Duplicate` | `G35/gemm_batched_kernel.cpp` | 旧 C 升精度、按列缩放、跳读与搬出 |
| `Cast`、`RoundMode::CAST_RINT` | `blas/gemv_batched/arch22/gemv_batched_kernel.cpp` | arch22 从 FP32 收窄；H 中点另验 |
| `RunCgemmBatchedPerformanceCase` | `TC/cgemm_batched_test.cpp` | 预分配、warmup；原例用 steady_clock 加 stream 同步，不是 event |
| `aclrtRecordEvent`、`aclrtEventElapsedTime` | `test/frame/npu_event_timer.h` | 同 stream 事件计时依据 |
| `SetPointers`、`OffsetPointersAndStreamOrdering` | `TC/cgemm_batched_pointer_regression.h` | 偏移地址构造、prefix/suffix/padding 哨兵 |
| `SgemmBatchedParam`、`sgemmBatchedPhysRows` | `TS/sgemm_batched_param.h` | 公共参数及物理维度 |
| `ApplyAlphaZeroGolden`、`cblas_sgemm` | `TS/sgemm_batched_golden.h` | 无乘积参考、列主序 CPU 参考 |
| `ComputeSgemmBatchedBufferSizes` | `TS/arch35/sgemm_batched_npu_wrapper.h` | wrapper 的缓冲尺寸 |
| `SgemmBatchedArch35Test`、`CsvDriven` | `TS/arch35/sgemm_batched_test.cpp` | 参数化测试接入方式 |
| `ops_blas_add_gtest_tests` | `TS/CMakeLists.txt` | 现有测试注册 |
