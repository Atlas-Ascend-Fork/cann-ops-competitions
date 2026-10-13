# aclblasStrsmBatched 算子设计文档（Atlas A2/A3）

# 需求背景（required）

## 需求来源

本设计对应 CANN 社区任务 2026 年 9 月第 78 项：`aclblasStrsmBatched` 算子开发（A2/A3）。依据任务书 `aclblasStrsmBatched_A2A3_task_doc.md`、随任务提供的测试包，以及 `cann/ops-blas` 的实际接口和工程结构制定。提交账号及个人团队目录为 `maincourses`。

目标是在 Atlas A2/A3 系列产品上，以 Ascend C kernel 直调方式实现 FP32 批量三角求解，完成配套 **1000 条精度用例和 200 条性能用例**。本文中的阈值是验收目标，tile 参数是待真机验证的初始候选值；本文不包含已实现、已通过测试或已达到性能目标的声明。

## 背景介绍

三角求解用于矩阵分解后的线性系统求解。不同 batch 之间相互独立；同一系统的三角维存在前代或回代依赖，但多个右端向量可并行。实现需要同时利用 batch 并行和右端并行，不能把有递推依赖的三角行分配给互不协作的核心。

本任务参考 cuBLAS `cublasStrsmBatched`，同时规定三项差异：B 只读并增加独立输出 C；alpha 仅支持 Host 指针；实数 trans 仅支持 N/T，OP_C 必须返回非法参数。

### 现有实现及差异

源码调研固定在 `cann/ops-blas` 提交 `617bb065849dc60fbc543c364dcf36f6e7660905`。下表描述该提交的现状，而非本任务已经完成的能力。

| 位置 / 符号 | 已有能力 | 本任务需要处理的差异 |
| --- | --- | --- |
| `include/cann_ops_blas.h::aclblasStrsmBatched` | 已有同名原地接口，无 Carray/ldc，声明位于 `extern "C"` 范围 | 任务要求独立输出；涉及公开 API/ABI，不能同时声明两个不同的 C 原型 |
| `blas/trsmbatched/arch35/strsmbatched_host.cpp` | 已有参数检查、Device 地址数组 D2H、逐 batch 调度、分块求解 | 该实现允许 OP_C、拒绝 batchCount=0；与本任务部分语义不同 |
| `blas/trsmbatched/README.md` | 现有支持表为 Ascend 950，A2/A3 不支持 | 补充 arch22；交付时依据实际验证结果更新支持表 |
| `experimental/aclblasTrsmBatched/` | 有 AIV panel 求解、AIC GEMM 更新及 MIX_AIC_1_2 事件协作原型 | 原型原地写 B，Host 内部同步后释放临时内存；不能直接作为独立输出异步接口使用 |
| `test/trsmbatched/strsmbatched/` | 有 CSV/GTest、CBLAS golden、arch35 NPU wrapper | 新增 arch22 wrapper 和用例，保留 B、独立分配 C，并落实任务负向场景 |
| `blas/common/helper/aclblas_handle_internal.h` | handle 管理 stream 和 workspace；默认 32 MiB，上限 2 GiB；扩容会同步已有 stream 工作 | 复用现有生命周期；不能在 kernel 尚未完成时释放临时缓冲区 |

实现目录采用任务书交付章节和上游现有家族目录：`blas/trsmbatched/arch22/`、`test/trsmbatched/strsmbatched/arch22/`。任务书前文出现的 `blas/trsm/` 不用于新建第二套重复家族。

# 需求分析（required）

## 需求描述

对每个 batch `b ∈ [0, batchCount)` 求解：

| side | 方程 | A[b] 形状 | B[b]、C[b] 形状 |
| --- | --- | --- | --- |
| LEFT | `op(A[b]) X[b] = alpha B[b]` | m × m | m × n |
| RIGHT | `X[b] op(A[b]) = alpha B[b]` | n × n | m × n |

`C[b] = X[b]`，所有矩阵采用列主序 FP32，地址按独立 lda/ldb/ldc 计算。各 batch 共享尺寸、枚举和标量，矩阵地址独立，不进行广播。

## 需求拆解

| 编号 | 需求及任务书依据 | 设计和验证落点 |
| --- | --- | --- |
| R1 | FP32、LEFT/RIGHT、UPPER/LOWER、N/T、UNIT/NON_UNIT，§2.1 | 16 组属性组合；直接与 CBLAS 逐 batch 比对 |
| R2 | B 只读、C 独立输出，§2.3、§2.5 | 工作区保存中间解；独立 ldc 寻址；检查 A/B 保持不变及 C 的写入范围 |
| R3 | 前导维 padding 与非对齐尺寸，§2.4、§3.5 | 64 位地址计算、真实尾块长度；补充 ldc≠ldb 和缓冲区偏移场景 |
| R4 | 零维或零 batch 为合法 no-op，§2.1、§2.5 | 快速返回；不读取矩阵地址数组、不下发计算 |
| R5 | alpha=0 不引用 A/B、C 清零；UNIT 不读取对角，§2.5、§7 | 专用清零路径；有效三角按需搬运，单位对角在片上构造 |
| R6 | 非法输入与元素级空指针，§2.4 | Host 标量/枚举校验及地址元数据校验；返回规定状态码 |
| R7 | handle 绑定 stream、异步数值计算，§2.5 | 数据准备、计算、输出均在同一 stream；workspace 生存期覆盖完成事件 |
| R8 | 官方 1000 精度 + 200 性能及补充覆盖，§3.5 | 保留原始包，核对执行清单、退出码、无漏跑/跳过；补充用例单独列出 |
| R9 | 精度与 kernel 时间门限，§3.2、§3.3 | FP32 运算、关闭 HF32；统一混合容差、按完整计算调用统计性能 |
| R10 | Atlas A2/A3、CANN 9.1.0，§3.1、§7 | arch22 接入，查询平台资源；910B3 性能验证，记录 A2/A3 验收覆盖 |
| R11 | 文档、复现、报告及日志，§4 | 交付阶段使用固定提交；设计阶段仅定义测试方法，不填写虚构结果 |

# 详细设计（required）

## 算子分析

### 接口原型与参数

以下为任务书要求的目标原型：

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const float* alpha,
    const float* const* Aarray, int lda,
    const float* const* Barray, int ldb,
    float* const* Carray, int ldc,
    int batchCount);
```

| 参数 | 类型与位置 | 约束 / 行为 |
| --- | --- | --- |
| handle | Host 句柄 | nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；携带 stream/workspace |
| side/uplo/trans/diag | Host 枚举 | LEFT/RIGHT、UPPER/LOWER、N/T、NON_UNIT/UNIT；其他值包括 OP_C 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| m/n/batchCount | Host int | 非负；任一个为零为 no-op；不以测试尺寸上限硬编码合法输入范围 |
| alpha | Host FP32 标量指针 | 普通计算须非空；读取本次值，不缓存旧标量；正零/负零走清零路径 |
| Aarray | Device 指针数组，元素指向 Device A | A 阶数 `q=(side==LEFT?m:n)`；只访问选定三角，UNIT 不读对角 |
| Barray | Device 指针数组，元素指向 Device B | 输入只读；alpha=0 时不引用数组及矩阵 |
| Carray | Device 指针数组，元素指向 Device C | 非空输出；与任意输入矩阵不能重叠，各输出之间不得发生写冲突 |
| lda | Host int | `lda≥max(1,q)` |
| ldb/ldc | Host int | 各自 `≥max(1,m)`，分别用于 B/C，不假定相等 |

现有公共枚举名字为 `ACLBLAS_UPPER/LOWER`、`ACLBLAS_NON_UNIT/UNIT`。任务书中的长名称在 CSV 层映射到对应已有值，不复制一套枚举。是否公开添加长名称别名由维护者确认，不作为 kernel 依赖。

### 数学公式与统一索引

令 `q` 为三角阶数，`r` 为独立右端数，将计算统一为 `T Y = D`：

| 模式 | q/r | T | Y/D |
| --- | --- | --- | --- |
| LEFT | q=m，r=n | op(A) | Y=X，D=alpha B |
| RIGHT | q=n，r=m | op(A)ᵀ | Y=Xᵀ，D=alpha Bᵀ |

定义 `right=(side==RIGHT)`，`effectiveTranspose=(trans==T) XOR right`，`effectiveLower=(uplo==LOWER) XOR effectiveTranspose`。effectiveLower 为真按 0→q−1 前代，为假按 q−1→0 回代。RIGHT 的转置是逻辑索引映射，不要求完整转置或复制 A。

对逻辑元素 `T[i,j]`，原始列主序地址为：

```text
effectiveTranspose == false: A[i + j * lda]
effectiveTranspose == true : A[j + i * lda]
LEFT  D[i,s] -> alpha * B[i + s * ldb]
RIGHT D[i,s] -> alpha * B[s + i * ldb]
LEFT  Y[i,s] -> C[i + s * ldc]
RIGHT Y[i,s] -> C[s + i * ldc]
```

在读取地址前先判断三角有效性与 diag：未选三角不读取；UNIT 对角直接使用 1。不能先搬整块再屏蔽无效位置。索引乘加和 byte span 使用经过溢出检查的 64 位整数。

### 支持形状与边界

接口支持满足前导维和可寻址内存约束的非负 int 尺寸。官方用例精度尺寸至 2048、性能尺寸至 4096，batch 至 1024，是当前验收覆盖范围，不是接口新增的硬上限。超出片上容量时按 tile 切分；不可表示的地址范围或无法分配的 workspace 返回相应错误，不截断参数。

NON_UNIT 的测试生成器会对非零对角进行符号保持的 boost，golden 与 NPU 必须使用同一份处理后的输入。算子本身不修改 A、不添加 boost、不预求逆，也不把奇异矩阵伪装成可解系统。该接口不提供逐 batch 的奇异性 info 输出。

## 算子实现

### 实现方案

主方案为小问题 Vector 直接求解，以及中大问题 Vector panel 求解 + Cube trailing update。首先建立可核验的纯 FP32 Vector 路径，再接入混合核优化。算法选择只依据尺寸、属性、平台容量和真实数据数值条件，不读取测试 case ID 或 golden。

```text
参数与地址元数据校验
  ├─ no-op：SUCCESS，无数值 kernel
  ├─ alpha=0：按 ldc 批量清零
  ├─ 小 q / 窄 RHS：AIV 直接前代或回代
  └─ 中大 q + 足够 RHS：AIV panel solve + AIC GEMM update
                         └─ 非有限数路径：受控切换至 Vector 语义路径
```

#### 3.2.1 Host 侧设计

**校验顺序与错误返回。** 拟采用：handle → 枚举和负维度 → no-op → 前导维与 alpha → 必需数组及地址元素 → span/重叠/容量 → 下发。任务包的 m=0/m=n=0 用例包含 lda=0 或 ldb=0，因此 no-op 不按普通非空矩阵继续校验 leading dimension。混合非法参数的优先级在补充测试中明确，保持基础非法枚举和负维度不能被零维掩盖。

对非空输出，始终检查 Carray/Carray[b]；alpha=0 时不校验或读取 Aarray/Barray 的元素，不引用 A/B 数据，清零只写 C 的 m×n 逻辑元素。非零 alpha 时校验全部 A/B/C 地址。数组复制/API 查询失败必须返回库错误码，不继续启动 kernel。

**Device 指针数组与同步边界。** 沿用上游 Device 指针数组契约。为兑现“数组元素为 nullptr 时本次 API 返回 INVALID_VALUE”，需要将少量地址元数据读到 Host。主方案在 handle stream 上进行地址数组 D2H，记录并等待该复制的完成事件，随后校验；不把 A/B 矩阵内容回传 Host。该过程可能等待同一 stream 中此前已经排队的工作，不能描述为 Host 全程无阻塞。通过校验后下发的本次数值计算保持异步，调用方读回 C 前仍需同步。

如果评审要求包括地址元数据在内的严格 Host 非阻塞，当前“Device 地址数组 + 同步返回元素级错误”组合需调整契约，例如额外提供 Host 地址元数据，或定义设备错误状态异步回读；不能仅在 Device 检测后仍返回 SUCCESS 来冒充完成负向校验。此项列入评审确认清单。

**输出重叠。** 根据 `sizeof(float)*((ld*(cols−1))+rows)` 计算矩阵存储 span，并检查地址加法溢出。把 C 的区间与全部 A/B 区间比较，覆盖跨 batch 的 C[b] 与 A[k]/B[k] 重叠；检查 C 输出间的写冲突。元数据数量较大时使用排序扫描，不引入逐元素 Host 拷贝。对仅涉及 padding 的区间重叠采用保守拒绝规则，并在接口说明中注明。

**Tiling 与分派。** 查询实际 AIC/AIV 核数、UB/L1 等容量；不硬编码某款 NPU 的核心数量。初始候选 panel 大小 `p∈{16,32,64}`、RHS stripe 宽度 `w∈{32,64,128}`、trailing 行 tile `h∈{64,128}`。按 UB、Matmul tiling 可行性和工作区限制选取组合，最终阈值由 200 条性能用例及相关精度回归确定。

| 参数 | 含义 |
| --- | --- |
| q/r、m/n、lda/ldb/ldc、batchCount、alpha | 运行时尺寸、寻址和标量 |
| effectiveTranspose/effectiveLower、diag | 数学映射及求解方向 |
| path、panelSize、rhsWidth、trailRows | 零值/Vector/分块分派与 tile 候选 |
| taskCount、activeGroups、taskStride | `(batch,RHS stripe)` 静态均衡分配 |
| workspaceStride、区域 offsets | 每个活动组的中间解、panel、packed A 和乘积槽 |
| validRows/validCols、Cube tiling | 非对齐尾块的真实尺寸和矩阵库参数 |

**分核。** LEFT 按 batch 和 B/C 列的 RHS stripe 切分；RIGHT 按 batch 和 B/C 行的 RHS stripe 切分。总任务数为 `batchCount*ceil(r/w)`，活动组不超过实际资源数；group g 处理任务 `g,g+activeGroups,...`。每个任务在三角维上独立完成全部 q 步，不跨组共享可写中间结果。batch 很小时增大 RHS 切分以填充核心，batch 很大时每个组循环处理多任务，避免 Host 逐 batch 下发。

**Workspace 生命周期。** 使用 handle 的 `GetEffectiveWorkspace/EnsureDefaultWorkspace` 机制。先算并验证所需容量，再下发本次数值工作；已有库工作区扩容所需同步属于容量准备阶段，记录首次调用开销。用户 workspace 不自行释放或重新分配；容量不足时尝试减小活动组数、stripe 或走更省空间的 Vector 路径，仍不足则返回 ALLOC_FAILED。

同一 handle/stream 上的调用按 stream 顺序复用工作区；不同 stream 并发使用不同 handle 或独立 workspace。不可在尚有计算读写旧缓冲区时切换/释放它。Host tiling 优先按值传参；必须放入 Device 的控制信息占用受 handle 管理的区域。地址数组由调用方保持有效到计算完成；若采用地址快照，H2D 使用 Host 缓冲区必须延长生存期，不能把栈上 vector 交给异步复制后立即销毁。

#### 3.2.2 Kernel 侧设计

**清零路径。** 将 `(batch,列,有效行区间)` 映射到 AIV 任务，只读取 Carray。连续访问按 32 字节对齐分段、尾段按真实 m 处理；不读 A/B，不写 C padding。不将所有 batch 假定为同一连续矩阵分配。

**Vector 直接路径。** 对每个 RHS stripe，通过逻辑映射读取 B，缩放并执行前代/回代：

```text
for i in solve_order:
    value = alpha * B_logical[i, stripe]
    for k in already_solved_rows:
        value -= T[i,k] * Y[k, stripe]
    if diag == NON_UNIT:
        value /= T[i,i]
    Y[i,stripe] = value
```

沿 RHS 维使用 FP32 Vector 运算；短 RHS 或很小 q 可将有效三角及中间 RHS 留在 UB。超过 UB 容量时流式读 T、已求解 Y 和分段 RHS，不能按 q² 分配片上缓冲区。最终按 LEFT/RIGHT 映射和 ldc 写 C，A/B 全程只读。性能分派不强制每个小矩阵使用全部核心。

**分块路径。** 将 T 按求解方向切为对角 panel。每个 panel 对应的递推为：

```text
Y_panel = solve(T_panel,panel, D_panel)
D_trailing = D_trailing - T_trailing,panel * Y_panel
```

工作区的当前 RHS 初值为 alpha B，按当前任务 stripe 转成便于 Vector/Cube 使用的内部 ND 排布。AIV 处理 panel 内三角递推，AIC 用 FP32 Matmul 计算 trailing update；原始 A 仅按需打包 panel/rectangular tile，不为每个 RHS stripe 完整复制 q×q 的 A。上三角回代按反向 panel 顺序及真实尾 panel 计算。

初始实现采用上游可参考的 MIX_AIC_1_2 编排：一个 AIC 和两个关联 AIV 为一个协作组，两 AIV 各处理 RHS stripe 的一半。原型注释与事件行为需在 CANN 9.1.0 上确认，不能把 arch35 SIMT 代码直接移植到 arch22。

**事件与写入所有权。** 同组内固定三个阶段，等待/发送次数必须匹配：

1. 两 AIV 将各自 panel 解写入互不重叠列区间；AIV0 打包本次 A tile；双方完成 MTE/Vector 到 GM 的依赖后发送输入就绪事件。AIC 等待两个 AIV 都就绪再读取。
2. AIC 完成该 tile Matmul 和输出搬运后发送乘积就绪事件；两 AIV 各更新 RHS 的一半，C/Y 无重叠写入。
3. 两 AIV 发送 tile 已消费事件；AIC 确认双方完成后，才允许重复使用乘积槽或进入下一轮。

无有效列的 AIV 仍参与全部事件；最后一个 panel 无 trailing rows 时 AIC/AIV 走一致的空操作序列或一致终止分支。不存在一个参与者提前 return、其他参与者永久等待的路径。事件限于物理协作组，不在不同调度组之间使用全局自旋栅栏；每个组完成前一任务后复用自己的 workspace 和事件。

第一版使用单槽以便验证事件协议；双缓冲属于后续可选优化，需增加槽编号/代次并证明不会提前覆盖尚未消费的数据。所有 flag 指令前后按相应生产/消费管线完成缓存和搬运可见性，不能用单独的 Host 同步掩盖核内竞争。

**UB、GM 及尾块。** 片上空间预算计入 panel A、两份 panel/RHS 搬运缓冲区、索引/转置临时区和 API scratch；优先选满足平台 UB 容量的 tile。内部按矩阵库要求补齐存储，但计算传入真实 tail 形状；补齐元素不映射到用户地址。UNIT 对角和非选定三角在临时 tile 内构造，无输入读取。对 LEFT 的列主序 RHS 按小块转置，对 RIGHT 的逻辑转置直接利用原有行内连续访存。

**数值路径。** panel 及累加保留 FP32，不使用 FP16/BF16 压缩 A 或 RHS，Cube Matmul 显式关闭 HF32。具体配置参考已有 arch22 FP32 Matmul 的 `SetHF32(false)`，由 CANN 9.1.0 编译/真机验证实际 API 支持。

NaN/Inf 或递推溢出不能通过“当成零”“钳位”或提前删除用例处理。Vector 路径按实际 BLAS 语义处理有效三角和零值分支；分块路径在 RHS 准备/panel 处理阶段检测非有限数，通过协作组一致的状态与事件终止 Cube 工作，再从原始 A/B 重算该任务的 Vector 路径。检测范围覆盖所引用 A 元素以及 RHS；不得让某 AIV 单方面退出。为特殊值路径单独验证 CBLAS 的分类、符号和 NaN 位置，避免 padding 中的额外 `0*Inf` 污染结果。

不要求不同归约顺序逐位一致。对有限值误差较大的形状，先检查索引、求解方向、对角、HF32 配置和分块误差，再调整 panel 或使用数值更稳健的 Vector 方案；是否加入残差修正需有独立实验依据，不作为未经证明的性能承诺。

#### 3.2.3 空间与性能优化

令 S 为活动协作组数，q_pad/w_pad/p_pad/h_pad 为合法存储对齐值。单槽分块路径的主要 GM scratch 预算为：

```text
Wgroup ≈ 4 * (q_pad*w_pad + p_pad*p_pad
              + p_pad*w_pad + h_pad*p_pad + h_pad*w_pad)
Wtotal = align(control_bytes) + S * align(Wgroup) + Matmul_system_workspace
```

上述为容量规划公式，不是实测内存。每组循环复用 scratch，空间随活动组数增长，不随全部 batch 与 RHS stripe 任务数同时增长。Matmul 系统工作区和同步控制区必须单独计入；额外流水槽按真实数量计入，不假定为零。没有全 batch 的 A 规范化副本。

优化顺序为减少 Host 下发与冷分配开销、选择合适 batch/RHS 并行、减少列主序搬运、提高 panel 与矩阵库 tile 利用率、最后验证双缓冲等流水优化。大 batch 不采用 1 batch = 1 次 Host API 启动的串行循环；小 batch 的 2048/4096 场景优先利用 RHS stripes。

## 支持硬件

| 目标产品 | 设计支持范围 | 验证计划 |
| --- | --- | --- |
| Atlas A2 训练/推理系列，含 Atlas 800I A2 | arch22 | 910B3 完成官方精度与性能；记录实际具体设备 |
| Atlas A3 训练/推理系列，含 Atlas 800I A3 | arch22 | 完成对应硬件运行回归并记录验收覆盖 |
| Ascend 950 / arch35 | 已有实现，非本任务新增硬件目标 | 公共接口若变更，必须检查该架构的编译与调用适配 |

CANN 要求为 9.1.0。设备实际驱动、固件、toolkit、编译器与核资源在开发环境核验后登记；本文不把截图中的连接状态等同于版本已满足。

## 算子约束限制

- 仅 FP32、列主序；不支持 OP_C、广播及 lda/ldb/ldc 语义之外的任意非连续视图。
- 输出 C 不与输入重叠；内部临时视图不对外返回；不缓存矩阵或地址数组中可变化的值。
- 调用方负责有效 Device 内存及数组长度，保持地址数组/矩阵/workspace 生命周期至 stream 计算完成；错误地址并非所有情况下都能由 Host 地址属性查询证明安全。
- 无确定性逐位一致要求；无数学求逆接口和奇异性检测承诺。
- Device 地址元素同步校验与严格 Host 非阻塞之间存在契约选择，见 Host 设计及评审确认项。

# 可维可测分析

## 精度标准/性能标准

### 精度判定

golden 使用 Netlib `cblas_strsm` 逐 batch 生成：将 B 的逻辑元素复制到独立 golden C，再在这份副本上进行 CBLAS 原地求解；B 原输入保留，golden 使用自身的前导维。返回码单独按任务定义校验，不能套用已有 arch35 golden 对 OP_C/batchCount=0 的规则。

对有限值逐元素检查：`|actual−golden| ≤ 2^−13 + 2^−13*|golden|`；全用例同时满足 matched_ratio≥0.99 和最大绝对误差门限。2026-10-12 读取的官方混合容差标准将表中的“1e−2 或 32 ULP”明确为：在最大有限值误差点，使用 `max(1e−2,32*ULP(golden_fp32))`。记录所用标准版本与 ULP 锚点，不把所有形状固定放宽到某个任意阈值。

NaN/±Inf 不参与有限值 max_abs_error；按 FP32 收窄后的参考结果检查相应非有限分类/符号，NaN 位置另外列出。未匹配的特殊值不得被忽略或通过分母调整掩盖。完整矩阵的各 batch 逻辑输出都参加验证，padding 不属于数值输出但检查没有被写坏。

### 性能判定

200 条 `gpu_baseline.csv` 均已有 gpu_ms 数值，逐条按参数与 TC_PF 用例关联，单位转换必须显式：

```text
GPU_us = gpu_ms * 1000
NPU_avg_us <= GPU_us / 0.8
```

任务书五条指定用例还需满足下列上限；同时检查对应 CSV 条件，避免舍入差异导致两套口径之一未达标。

| case | batchCount | m × n | side/uplo/trans/diag | 任务书 NPU 上限（μs） |
| --- | --- | --- | --- | --- |
| 1 | 64 | 256 × 256 | LEFT/LOWER/N/NON_UNIT | 630.7 |
| 2 | 128 | 384 × 512 | LEFT/UPPER/T/NON_UNIT | 3062 |
| 3 | 32 | 1024 × 1024 | RIGHT/LOWER/N/UNIT | 5134 |
| 4 | 8 | 2048 × 2048 | LEFT/UPPER/T/NON_UNIT | 8680 |
| 5 | 2 | 4096 × 4096 | RIGHT/UPPER/N/UNIT | 14361 |

使用 `msprof op` 采集真实计算 kernel 耗时，先完成 warmup，选择至少 20 次有效采样以覆盖任务书“>10 次”要求。每次测量前准备相同输入、完成前一轮计算，golden 和回读放在计时外。GTest 耗时以及 Host 分配、地址检查、数据拷贝等开销单独报告，不代替 kernel 时间。

若一次算子调用下发多个数值 kernel，包含必要的初始化/转置/求解/回写，而非只选择最快 GEMM。混合核的 AIC/AIV 记录按同一逻辑 kernel 调用关联，避免把协同参与者重复计算为多次调用；多 kernel 按调用与时间关系归并，保留明细和归并方法。记录 Avg、采样数、波动及每 case 是否满足门限；200 条逐条达标，不能用整体平均掩盖失败项。

### 内存与资源

任务书 §3.4 没有单独内存数值门限，§4 仍要求报告和日志。因此记录输入/输出分配、库 workspace、算子峰值 scratch、扩容前后容量与释放行为，不虚构“内存已经通过”。反复调用和销毁 handle 后验证无持续增长；首次扩容和稳态性能分别报告。测试生成器的 512 MiB Host 预算是其生成策略，不是 kernel scratch 的官方验收上限。

### 测试计划与可复现入口

配套原始 CSV 共 1200 条：1000 条精度（其中 20 条负向，27 条边界分类）及 200 条性能。负向按预期状态码通过计数，不要求 1000 条都返回 SUCCESS。

| 测试集合 | 覆盖及检查 |
| --- | --- |
| 官方 L0/SQ/AB/BC/WS/TH/LD/FL/CV/ED/EX | 16 组组合、尺寸、alpha、batch、矩形、padding、填充与所有负向；严格核对执行 case 清单 |
| 官方 PF 200 条 | 参数关联、数值基本正确性、5 个硬指标及全部基线 ratio；固定源码和构建配置 |
| 独立输出补充 | ldb≠ldc、C padding 哨兵、A/B 未修改、C 与本 batch/其他 batch A/B 重叠、输出写冲突 |
| 未引用元素补充 | UNIT 对角填 NaN/Inf、非选三角填异常值；alpha=0 的 A/B 数组及元素可为空 |
| 泛化与数值补充 | 单独正态分布、非对齐地址、所有 side/trans 的非方阵、较大/较小有限值、A 的特殊值 |
| stream/workspace 补充 | 同 stream 连续调用、跨 stream 独立 handle、输出读回同步、workspace 切换/扩容/不足、重复释放和泄漏检查 |

官方 `gen_csv.py --dist mixed` 当前只是预留参数，实际仍生成均匀输入。原始官方集合保留；50% 均匀/50% 正态等缺失覆盖在独立补充集合实现并记录 seed，不重新生成 CSV 覆盖已有 gpu_baseline。现有 CSV 没有 ldc 列，负向通过 description 触发；需要在 arch22 测试中落实这些触发规则，并为补充测试显式传入独立 ldc。

现有精度脚本的路径优先级含 `test/trsm/`，但实际家族是 `test/trsmbatched/`；运行时核对实际安装路径。脚本解析汇总不能替代子进程退出码和执行数验证，0 条、超时、崩溃、漏跑或跳过不得判定通过。

预期工程入口如下，测试接入后按真实二进制名称复核：

```bash
bash build.sh --ops=strsmbatched --soc=ascend910b3
./build/test/trsmbatched/strsmbatched/strsmbatched_test --gtest_filter=-*TC_PF*
msprof op --application="./build/test/trsmbatched/strsmbatched/strsmbatched_test --gtest_filter=*TC_PF_1001*" --output=./prof_strsmbatched
```

性能采集逐 case 收敛，20 次有效采样由测试计时循环/Profiling 配置真实实现；不能仅凭上述命令声称已经采样足够次数。保存原始 GTest 输出、msprof 明细、参数映射和计算报告；自验报告在固定代码版本有真实结果后制作。

## 兼容性分析

### 公共 API/ABI

当前公开 C 接口为 13 个参数的原地形式，任务目标为增加 Carray/ldc 的 15 个参数形式。二者同名，不能通过普通 C++ 重载同时导出两个 C ABI；旧二进制调用也不能凭“新增参数”保持兼容。此问题属于本设计的主要评审项。

本设计以任务书目标原型为提案：公共声明统一为独立输出形式；在代码提交中枚举 arch35 Host、NPU wrapper、golden 和样例等同名调用点，做最小必要适配，重建这些调用方。arch35 原地 kernel 可在 B→C 拷贝后继续以内核 C 缓冲区运算，保留其已有数值实现；不会自动获得旧二进制 ABI 兼容。OP_C 与 no-op 的跨架构规范一并在文档中注明并评审。

若维护者要求保留已发布的原地 ABI，应选择维护者批准的新符号或版本化接口/适配层，并与任务验收接口协同确认。不能仅为 arch22 改公共头文件而使 arch35 不再编译，也不能用私有测试头绕过公开声明。最终接口选择在评审后落实。

### 工程接入与文件范围

```text
include/cann_ops_blas.h                        # 经评审统一的声明
blas/trsmbatched/README.md                     # 接口语义与产品支持
blas/trsmbatched/arch22/
    strsmbatched_host.cpp                     # 校验、tiling、下发
    strsmbatched_kernel.cpp                   # Vector/混合核入口
    strsmbatched_kernel.h                     # 必需的设备实现
    strsmbatched_tiling_data.h                # Host/Device 共用参数
test/trsmbatched/strsmbatched/
    CMakeLists.txt                            # 复用 ops_blas_add_gtest_tests
    strsmbatched_param.h                      # 兼容原 CSV 与负向字段
    strsmbatched_golden.h                      # 独立输出的 CBLAS 参考
    arch22/strsmbatched_npu_wrapper.h
    arch22/strsmbatched_test.cpp
    arch22/strsmbatched_test.csv
```

文件数量随必要职责调整，复用上游自动发现 arch 目录的构建方式；不为 blas 源目录增加无用 CMakeLists，不复制整个 experimental 工程。跨架构公共调用点适配仅限接口一致性所需内容。修改公共接口时除 arch22 本任务测试外，必须执行 arch35 相关构建和接口回归，不能用一种架构的编译替代另一种。

## 评审确认与风险处置

| 项目 | 当前提案 / 后续处置 |
| --- | --- |
| 同名原地接口与任务独立输出接口冲突 | 以 15 参数目标接口设计，明确 ABI 影响；请评审选择统一迁移或版本化方案 |
| Device 数组元素校验与严格 Host 非阻塞 | 地址元数据允许事件等待，本次数值 kernel 异步；若要求更严格则需明确替代错误反馈契约 |
| no-op 与 alpha=0 的参数校验优先级 | 依照原始用例和退化语义制定上述顺序；补充混合非法参数测试供评审核对 |
| 硬件覆盖 | §3.1 自验一种款型与 §7 A2/A3 均验收分别记录；按 A2/A3 均需覆盖准备，不把单机结果写成双机通过 |
| 精度与高条件数 / 特殊值 | 使用官方明确的动态 ULP 上限，保留全部 case；失败按原因整改，禁止改阈值或过滤伪装达标 |
| 混合核事件 / FP32 Cube API | CANN 9.1.0 真机确认；用小规模事件、尾块和不均衡任务回归后再扩展到全量 |
| 流程状态 | 设计 PR 提交、审核通过、合入分别记录；提交设计不代表通过，开发和验收流程按官方要求衔接 |

## 参考依据

- 任务附件：`aclblasStrsmBatched_A2A3_task_doc.md`、`test_cases/README.md`、`strsmbatched_test.csv`、`gpu_baseline.csv`、`gen_csv.py`、`verify_accuracy.py`。
- [社区任务目录与提交规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)；[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。2026-10-12 通过仓库内容接口读取，模板 blob 为 `b2d9e2c47e2afe307a9f303735fb045011642b50`。
- [公共接口基线](https://gitcode.com/cann/ops-blas/blob/617bb065849dc60fbc543c364dcf36f6e7660905/include/cann_ops_blas.h)；[现有 arch35 Host](https://gitcode.com/cann/ops-blas/blob/617bb065849dc60fbc543c364dcf36f6e7660905/blas/trsmbatched/arch35/strsmbatched_host.cpp)；[handle/workspace 管理](https://gitcode.com/cann/ops-blas/blob/617bb065849dc60fbc543c364dcf36f6e7660905/blas/common/helper/aclblas_handle_internal.h)。
- [实验性混合核参考](https://gitcode.com/cann/ops-blas/tree/617bb065849dc60fbc543c364dcf36f6e7660905/experimental/aclblasTrsmBatched)；[现有测试 golden](https://gitcode.com/cann/ops-blas/blob/617bb065849dc60fbc543c364dcf36f6e7660905/test/trsmbatched/strsmbatched/strsmbatched_golden.h)。
- [官方混合容差标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)，2026-10-12 读取的 blob 为 `0f8ed375ba70cd426d81620a88553dafd8f7792a`。
- [cuBLAS trsmBatched 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-trsmbatched)；[Ascend C SetHF32 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0646.html)。API 的目标版本支持仍以 CANN 9.1.0 编译和设备验证为准。
- [活动流程入口](https://www.hiascend.com/developer/activities/details/11014a50a8794171a4a08688fd398774#tab2)；[社区任务注意事项](https://gitcode.com/org/cann/discussions/39)。本次非 UI 读取未获得这两个动态页面的完整正文，提交目录和模板使用已实时读取的官方仓库规则，不声称已核验动态页面全部最新要求。
