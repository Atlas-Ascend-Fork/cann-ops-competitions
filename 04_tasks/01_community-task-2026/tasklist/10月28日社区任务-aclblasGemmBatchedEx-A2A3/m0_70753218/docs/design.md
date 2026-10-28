# aclblasGemmBatchedEx 算子设计文档（Atlas A2/A3）

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 10月28日社区任务-aclblasGemmBatchedEx-A2A3 |
| 团队名称 | [m0_70753218](https://gitcode.com/m0_70753218) |
| 需求来源 | `aclblasGemmBatchedEx_A2_task_doc.md` 及随附测试资料 |
| 目标仓库 | [cann/ops-blas](https://gitcode.com/cann/ops-blas) |
| 实现目录 | `blas/gemm_batched_ex/arch22/` |
| 测试目录 | `test/gemm_batched_ex/arch22/` |
| 公共接口 | `include/cann_ops_blas.h` 中已有 `aclblasGemmBatchedEx` 声明 |
| 开发语言与调用方式 | Ascend C，handle 绑定 stream，kernel 直调 |
| 目标硬件 | Atlas A2/A3 系列，`arch22`；性能基准设备为 Atlas 800T A2（910B3） |
| 验收软件版本 | CANN 9.1.0；CPU golden 使用 Netlib BLAS 3.10.0 的 `cblas_sgemm` |
| 实现范围 | FP32 输入、计算与输出，`ACLBLAS_COMPUTE_32F`，`ACLBLAS_GEMM_DEFAULT` |

本文描述接口契约、实现设计及验证方法。实测结果、环境版本、日志和 profiler 数据在自测报告中记录。

代码检查点为 [addd2c7](https://gitcode.com/m0_70753218/ops-blas/commit/addd2c7d5b9c8cfd3ef7bcebdb787576b0711fe9)。已在 910B3/CANN 9.1.0 编译，通过 32 项冒烟测试和原失败的 43 条用例复测；其余原始用例与最终性能验收仍在进行。

# 1. 需求背景（required）

## 1.1 需求来源

任务要求补齐 `ops-blas` 在 Atlas A2/A3 上的扩展批量矩阵乘能力。现有公共头已声明该接口，上游同算子具有 `arch35` 实现；本任务增加 `arch22` Host 与 Device 实现，沿用公共 handle、stream、枚举、状态码和函数原型。

设计依据为任务书第 2 节的功能与接口要求、第 3 节的验收标准，以及第 7 节的补充要求。工程组织采用 `ops-blas/CONTRIBUTING.md` 的目录与构建规范，设计文档章节采用社区 `design_template.md`。

## 1.2 背景介绍

批量 GEMM 对多个形状相同、地址独立的矩阵执行乘法与累加，可用于多个小型线性变换、分组计算等场景。每个 batch 使用独立的 A、B、C 指针，全部 batch 共享维度、转置属性、前导维度、标量和数据类型。

对每个 \(i\in[0,\mathrm{batchCount})\)，运算为 \(C_i\leftarrow\alpha\operatorname{op}(A_i)\operatorname{op}(B_i)+\beta C_i\)。其中 \(\operatorname{op}(A_i)\) 为 \(m\times k\)，\(\operatorname{op}(B_i)\) 为 \(k\times n\)，\(C_i\) 为 \(m\times n\)。A、B、C 均采用列主序存储。

本次 A2/A3 实现支持以下组合：

| Atype | Btype | Ctype | computeType | algo | alpha/beta 类型 |
| --- | --- | --- | --- | --- | --- |
| `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` | `ACLBLAS_COMPUTE_32F` | `ACLBLAS_GEMM_DEFAULT` | `float` |

FP16、BF16、FP8、INT8 等其他输入输出或计算类型组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。其他架构已有的类型能力继续由相应架构实现和 README 的分架构说明定义。

# 2. 需求分析（required）

## 2.1 需求描述

实现 FP32 列主序批量 GEMM，覆盖 N/T/C 全部转置组合、运行时维度、非紧凑前导维度、任意独立 batch 地址、Host/Device 标量、原地 C 更新及边界返回行为。计算通过 handle 所绑定的 stream 提交，调用方在读回结果前同步该 stream。

`ACLBLAS_OP_N` 表示原矩阵，`ACLBLAS_OP_T` 表示转置，`ACLBLAS_OP_C` 表示共轭转置；实数 FP32 下 T 与 C 使用相同寻址。每个 batch 的输出具有独立写区。

## 2.2 公共接口

```cpp
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);
```

| 参数 | 存储位置与语义 | 合法范围与返回行为 |
| --- | --- | --- |
| `handle` | Host 句柄，携带 stream 与 workspace | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `transa/transb` | Host 枚举 | N/T/C；其他值返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `m/n/k` | Host 矩阵维度 | 非负；负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `batchCount` | Host 批数 | 非负；负值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha/beta` | Host 或 Device 上的 FP32 标量指针 | 空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Aarray/Barray` | Device 指针数组；元素为各 batch 的只读 Device 矩阵地址 | `batchCount>0` 时数组指针须非空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Carray` | Device 指针数组；元素为各 batch 的 Device 输入/输出矩阵地址 | `batchCount>0` 时数组指针须非空，否则返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `Atype/Btype/Ctype` | Host 数据类型枚举 | 本实现均取 `ACL_FLOAT`，其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `lda` | A 的列主序前导维度 | N 时至少 `max(1,m)`，T/C 时至少 `max(1,k)`；不满足返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldb` | B 的列主序前导维度 | N 时至少 `max(1,k)`，T/C 时至少 `max(1,n)`；不满足返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldc` | C 的列主序前导维度 | 至少 `max(1,m)`；不满足返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `computeType` | Host 计算类型枚举 | `ACLBLAS_COMPUTE_32F`；其他值返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `algo` | Host 算法枚举 | `ACLBLAS_GEMM_DEFAULT`；其他值返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |

A/B/C 的数组位于 Device，Host 分发过程传递数组地址；Device kernel 按 batch 读取对应矩阵地址。每个矩阵按实际 `lda/ldb/ldc` 计算元素位置。

## 2.3 边界契约

参数校验先于 no-op 判定。包括空维在内的调用均检查 handle、枚举、维度、前导维度、类型、算法及标量非空；`batchCount>0` 时继续要求 A/B/C 三个数组指针非空。

| 条件 | 计算行为 |
| --- | --- |
| `m==0`、`n==0` 或 `batchCount==0` | 校验通过后直接返回 `ACLBLAS_STATUS_SUCCESS`，不启动计算 |
| `k==0` | 不读取 A/B 矩阵，执行 C 的 beta 缩放 |
| `alpha==0 && beta==1` | C 保持原值，不读取 A/B 矩阵 |
| `alpha==0 && beta==0` | C 的逻辑元素写入正零，不读取 A/B 矩阵及 C 旧值 |
| `alpha==0` 且其他 beta | 对 C 的逻辑元素逐元素乘 beta，不读取 A/B 矩阵 |
| `beta==0` 且执行乘法 | 计算输出时跳过 C 旧值读取；C 输出地址仍须有效 |

只读或只写逻辑矩阵区域；C 的 leading-dimension padding、分配前后保护区保持原值。对齐偏移与尾块按真实地址和有效长度处理。

## 2.4 需求拆解

| 模块 | 职责 |
| --- | --- |
| 公共 API | 复用同名声明与参数顺序，供 A2/A3 和其他产品线共用 |
| Host 校验与 tiling | 校验参数、识别标量位置、查询核数、选择执行路径、划分 workspace 条带 |
| Cube kernel | FP32 矩阵乘、转置与前导维度适配、输出 tile 分核 |
| Vector kernel | 小矩阵乘、beta-only 缩放、alpha/beta 融合、非对齐写回 |
| CSV/GTest | 载入全部任务用例，构造独立 Device 矩阵与指针数组，检查状态码 |
| Golden 与精度统计 | 逐 batch 调用 Netlib `cblas_sgemm`，统计有限值、非有限值、EXACT 和保护区 |
| Profiler 汇总 | 从 msprof task 数据重建每次接口调用耗时，计算平均值和 GPU 比值 |

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 列主序寻址

记原矩阵的行列坐标为 \((r,c)\)，列主序元素偏移为 \(r+c\cdot ld\)。计算输出 \(C_i(r,c)\) 时，对归约坐标 \(p\) 使用以下元素偏移：

| 矩阵 | N | T/C |
| --- | --- | --- |
| A | \(r+p\cdot lda\) | \(p+r\cdot lda\) |
| B | \(p+c\cdot ldb\) | \(c+p\cdot ldb\) |
| C | \(r+c\cdot ldc\) | 与转置属性无关 |

实际存储列数分别为：A 在 N 时为 k、T/C 时为 m；B 在 N 时为 n、T/C 时为 k；C 为 n。每个 batch 通过对应数组元素取得基地址。偏移、容量和任务数量使用 64 位中间值计算，资源大小进入分配或 tiling 字段前检查范围。

### 3.1.2 支持形状与布局

维度和 batch 数由运行时参数给出，覆盖方阵、矩形、奇数维度、非对齐尾块及前导维度 padding。batch 指针之间没有固定 stride 要求；指针数组中的各矩阵可独立分配。超出 leading dimension 语义的任意多维 stride 不属于本接口的布局描述。

### 3.1.3 FP32 计算要求

输入与累加保持 FP32，Cube 显式设置 `SetHF32(false)`。普通矩阵乘允许浮点归约顺序差异，最终按任务书混合容差、最大绝对误差及非有限值规则判定。`alpha==0` 的 C 缩放使用位精确检查。

## 3.2 Host 侧设计

### 3.2.1 校验与分发顺序

1. 检查 handle。
2. 检查 N/T/C 转置枚举。
3. 检查非负维度、批数、三个前导维度和 alpha/beta 非空。
4. 检查 FP32 数据类型组合、计算类型和算法。
5. 当 `batchCount>0` 时检查三个数组指针非空。
6. 对零输出维或零 batch 返回成功。
7. 获取标量位置、核数与必要资源，选择纯缩放、Ordered、小矩阵 Vector 或带输入检查的 Cube 路径。

该顺序使 no-op 调用遵循任务书规定的校验契约。平台查询返回零核数时返回内部错误；库管理 workspace 分配失败时返回资源错误。用户 workspace 容量不足以容纳一列及检查标志时，采用无需全局临时区的 Vector 运算。

### 3.2.2 Host/Device 标量

Host 使用公共 `CheckPtrLocation` 查询 alpha/beta 所在位置。Host 标量的值写入 tiling；Device 标量将地址写入 tiling，由各 kernel 在执行时读取。两个标量分别解析。

Device 标量值通过绑定 stream 上的先行写入生效，kernel 按执行时的值决定是否跳过乘法或 C 读取。正常提交路径保持 kernel 对 Device 标量的直接读取。workspace 首次分配或扩容沿用公共 handle 的资源管理语义，扩容过程中所需同步由公共管理函数执行。

### 3.2.3 路径选择

| 路径 | 选择条件与用途 |
| --- | --- |
| Host no-op | 输出维或 batch 为零；或 Host 已知无需乘法且 beta 为 1 |
| Vector beta-only | `k==0` 或 Host 已知 `alpha==0` |
| Ordered Vector | 乘法路径的 `k>2048`、任一 Device 标量或 Host 标量超出 \([-1,1]\)；按 Netlib 的 K 顺序与系数位置计算 |
| 紧凑小方阵 Vector | Host 标量 1/0、NN、紧凑 8×8、16×16、32×32；一次搬入整矩阵，向量 Gather 与乘加计算全部输出 |
| 小矩阵 Vector | 单矩阵 \(m n\max(1,k)\le4096\) |
| Cube 直接输出 | Host 标量 `alpha==1 && beta==0`，且 `ldc` 满足 8 个 FP32 元素的输出对齐条件；先检查输入范围，异常 batch 由 Vector 参考路径计算 |
| Cube+Vector | 其他普通乘法：检查输入范围，正常 batch 的 Cube 乘积写入临时区；Vector 完成 alpha/beta 或异常 batch 的参考计算 |

4096 是小矩阵路径的单矩阵乘加项数量阈值，batch 维度由核间分工处理。Ordered 在执行时读取 Device 标量；alpha 为零时直接选择 beta-only 行为。

### 3.2.4 分核与 tiling

Cube 的输出基础 tile 为 \(128\times128\)，Host 将归约长度超过 2048 的乘法交给 Ordered。紧凑 NN 方阵的 16×16 与 32×32 使用对应静态 Matmul tiling。以 \((batch,tileN,tileM)\) 展平后的任务编号进行分核，各核以实际 block 数为步长领取任务。Vector 按 \((batch,column,rowTile)\) 分工，rowTile 最多包含 256 个 FP32 元素。直接输出路径的修复阶段按 batch 分工，正常 batch 读取检查标志后直接结束。各阶段按照实际任务数限制 block 数。

Tiling 记录矩阵维度、leading dimensions、转置标记、有效 batch、临时输出 leading dimension、标量值或 Device 地址、检查标志地址、检查分核数、直接输出标记及列条带范围。

### 3.2.5 有界 workspace

通用路径按 batch 组和输出列条带复用 workspace，默认预算为 32 MiB。临时行数向 8 个 FP32 元素对齐，记为 \(m_a=8\lceil m/8\rceil\)。检查阶段每 batch、每检查核独占 32 字节标志。对检查分核数 \(g\)、条带列数 \(n_s\)、组内批数 \(b_s\)，空间为 \(W=32gb_s+4m_a n_s b_s\) 字节；直接输出路径仅使用标志区域。

Host 依据预算及有效 workspace 容量选择 \(g,n_s,b_s\)，依次提交该条带的输入检查、Cube、Vector；同一 stream 保证后续条带复用空间前，前一条带已经完成使用。单列及一个检查标志超过默认预算时，库管理 workspace 按单列需求扩容；超过公共分配上限或用户容量不足时使用纯 Vector 路径。分配沿用公共 handle 的所有权及扩容规则。

条带化保持原矩阵的 `lda/ldb/ldc`：每条带只改变输出列起点、相应 B 地址偏移和局部范围，输入物理行跨度仍使用调用方给出的前导维度。完整输出矩阵由各条带的逻辑区域组成。

## 3.3 Kernel 侧设计

### 3.3.1 Cube 数据流

列主序输出可按行主序转置结果解释，即 \(C^T=\operatorname{op}(B)^T\operatorname{op}(A)^T\)。Cube kernel 交换 A/B 在 Matmul API 中的角色，使用转置模板匹配 N 与 T/C，并由 `SetOrgShape` 描述真实物理跨度。

单个 tile 先读取该 batch 的检查标志；异常 batch 留给 Vector 参考计算。正常 batch 读取 Device 指针数组中的地址，设置原始形状与有效 tile 大小，执行 FP32 Matmul，将乘积写入 C 或条带临时区。

每次写回使用剩余的有效 M/N 范围，输出列跨度由 `ldc` 或临时 `tempLd` 指定。模板实例与 `SetTensorA/SetTensorB` 的转置标志保持一致。

### 3.3.2 Vector 数据流

Vector kernel 先读取本次 alpha/beta，再选择以下流程：

- beta-only：beta 为 1 时直接结束；beta 为 0 时生成零；其他值搬入 C 并乘 beta，对 subnormal 使用软件 IEEE-754 乘法，对 NaN 保留符号与 payload 并按 ARM 规则 quiet signaling NaN。
- 小矩阵：根据 N/T/C 和真实前导维度读取 A/B；范围安全的 NN 使用向量乘加，其他情况按参考顺序归约。
- 乘积融合：从 workspace 搬入该条带乘积并乘 alpha；beta 非零时读取 C 并相加。
- 异常 batch：按 Netlib 的 K 顺序与标量缩放位置计算，极值乘加保留完整乘积后做一次 FP32 舍入。

GM 与 UB 的非对齐搬运使用 `DataCopyPad` 和有效字节数。UB 起始地址按搬运要求对齐，输出只覆盖实际 count。beta 为零时跳过 C 输入搬运。

Host 已知 alpha=1、beta=0 且需要临时乘积时，写回按最多 8 列×256 行分组，使用双向二维搬运保持真实 `ldc`。该路径跳过标量乘法与旧 C 读取；异常 batch 在写回前执行参考计算。行列尾块及 workspace 列条带分别保留有效长度与列起点。

### 3.3.3 快速路径与特殊值

`k==0` 或执行时 alpha 为零的路径在读取 A/B 数组元素及矩阵内容前决定缩放行为。beta 为零采用直接写零，从而覆盖 C 旧值中可能存在的 Inf/NaN。

输入检查仅扫描逻辑元素，避开 padding。每核使用向量 Abs 与 `CompareScalar(..., 2^30, LE)` 检查 A、B，以及 beta 非零时的 C；NaN、Inf 与超界元素均使该 batch 进入参考路径。alpha、beta 同样检查有限范围。检查标志由各核独占写入，避免跨核原子竞争。

当全部相关值的绝对值不超过 \(2^{30}\) 时，使用 \(k<2^{31}\) 可得乘积归约、alpha 缩放与 beta 项的绝对中间值小于 \(2^{122}\)，普通路径保持有限值分类。参考路径按 Netlib 3.10.0 的缩放和 K 归约顺序处理超范围数据；非转置 A 的路径先计算 alpha×B，再与旧 C 累加。该版本每个 K 项均参与计算，包含零因子与 Inf/NaN 的组合。重点回归包括：

| 输入情形 | Netlib 路径关注点 |
| --- | --- |
| NN、1×1、alpha=0.5、A=FLT_MAX、B=2、beta=0 | 非转置 A 的路径先缩放 B；先求 A×B 会提前溢出 |
| NN、m=n=1、k=2、alpha=beta=1、A=[1,1]、B=[FLT_MAX,FLT_MAX]、C=-FLT_MAX | 旧 C 与各归约项的相加顺序影响中间溢出 |
| Inf/NaN 输入、beta=0、alpha=0 | 验证跳过读取与任务书非有限值等效规则 |

这些情形分别覆盖小矩阵与 Cube 分发范围。测试记录保留有限/非有限差异、位精确缩放差异及具体失败参数。

### 3.3.4 Ordered 数据流

每个向量 lane 对应一个输出行，单任务处理一列的最多 256 行。A 以 64 个 K 项为一段搬入 UB；N 路径按原列跨度搬运，T/C 路径按物理列搬运后 Gather。B 的 N 路径连续搬入，T/C 路径按实际 ldb 提取各 K 项。显式零填充保持尾块检查区域有效。局部缓冲总计 82,976 字节（约 81 KiB），无需全局临时区。

A 为 N 时先形成 beta×C，再按递增 K 计算 alpha×B 并进行逐项乘加。A 为 T/C 时先按递增 K 做乘加归约，再舍入 alpha×sum，最后将 beta×C 融合到该乘积；该顺序与实测 Netlib 编译结果一致。两个独立结果/操作数缓冲交替使用，跨 K 段保留累加器。

设备独立探针对四种向量乘加 API 的实测均为先舍入乘积。Ordered 因而用 12 位尾数拆分恢复乘积残差，再用 TwoSum 恢复加法残差，最后合并修正。CPU 的 20 万组普通指数范围和 20 万组近抵消输入与 `fmaf` 比对均无差异；设备上的长归约及放大系数回归逐位对齐。极值参考路径使用软件 IEEE-754 FMA。

每段检查 A/B，beta 非零时检查旧 C；超范围、非有限值或 DMA stride 超过接口字段范围时，在写回 C 前改用逐元素参考运算。alpha 为零或 K 为零时先处理缩放，再解析 A/B 数组元素。新增回归覆盖 257 行、65/4097 个 K 项、九种转置组合、Host/Device 标量及 NaN payload 缩放，已通过设备冒烟验证。

## 3.4 支持硬件

| 产品 | 本设计实现路径 |
| --- | --- |
| Atlas A2 系列，含 Atlas 800T A2（910B3） | 支持，`arch22` |
| Atlas A3 系列，含 Atlas 800T A3 | 支持，与 A2 共用 `arch22` 源码 |

验收以 CANN 9.1.0 编译和运行结果为准，报告记录具体 SoC、驱动、固件、CANN、编译器和设备编号。A2 性能在任务指定 910B3 上采集，A3 用于同源代码的适配与精度验证。

## 3.5 算子约束限制

接口实现范围为 FP32 的列主序矩阵与 N/T/C 转置；各 batch 共享维度、标量与前导维度。A/B/C 地址、分配范围与 C 的 batch 独立写区由调用参数描述。任意高维非连续 tensor、动态广播与确定性保证均不属于本任务需求。C 原地覆写，调用方同步绑定 stream 后读取结果。

# 4. 可维可测分析

## 4.1 测试工程与覆盖

任务包实际包含 1200 条 CSV 用例，分为 1000 条精度/状态码用例和 200 条性能用例。测试保留全部 1200 条，并对性能形状执行精度检查。用例分类如下：

| 类别 | 前缀 | 条数 |
| --- | --- | ---: |
| 基础转置组合 | `TC_L0` | 9 |
| shape 扫描 | `TC_MN` | 116 |
| alpha/beta | `TC_AB` | 24 |
| 前导维度 | `TC_LD` | 12 |
| 数据填充与特殊值 | `TC_FL` | 18 |
| 中等形状覆盖 | `TC_CV` | 16 |
| batch 扫描 | `TC_GRP` | 16 |
| 边界与负向参数 | `TC_ED` | 36 |
| 扩展组合 | `TC_EX` | 753 |
| 性能形状 | `TC_PF` | 200 |

CSV parser 按任务字段读取 `transa/transb`、`aarray_fill/barray_fill/carray_fill`，处理标量字面量 `null`、数组字面量 `NULLPTR` 和非法枚举。负向参数在构造测试缓冲时采用安全容量，接口仍接收原始非法参数。

除 CSV 外补充 Device 标量、同 stream 先行标量写入、跳过 A/B 读取、独立 batch 分配、非对齐首地址、padding 保护、极值与归约溢出顺序测试。每次运行检查 ACL 返回值和 stream 同步结果。

## 4.2 精度标准

普通乘法使用 Netlib BLAS 3.10.0 `cblas_sgemm(CblasColMajor,...)` 对每个 batch 独立生成 golden，覆盖全部 \(m\times n\) 逻辑输出。测试记录实际链接的 BLAS 库和版本。

对 finite 元素，令误差为 \(e=|actual-golden|\)。逐元素匹配条件为 \(e\le 2^{-13}+2^{-13}|golden|\)，用例的有限值匹配率至少 0.99；同时检查最大绝对误差要求，逐元素满足 \(e\le10^{-2}\) 或 \(e\le32\operatorname{ULP}(golden)\)。报告输出有限值总数、匹配数、匹配率与最大绝对误差。

出现非有限值时，独立统计以下规则：

| actual 与 golden | 判定 |
| --- | --- |
| 同为 NaN | 等效 |
| 同为同号 Inf | 等效 |
| 一方 NaN，另一方为正或负 Inf | 等效 |
| 相反符号的 Inf | 不等效 |
| 一方 finite，另一方 Inf/NaN | 不等效 |

非有限元素从有限值统计中排除，非有限元素须全部等效。alpha 为零的缩放路径逐元素比较 FP32 位模式；k 为零的 beta-only 路径也执行位模式检查。C padding、前后保护区及 A/B 只读性独立检查。

## 4.3 性能标准与采样方法

性能测试每个 case 单独运行应用，先执行 5 次 warmup，再执行 20 次有效调用。每次调用检查返回 `ACLBLAS_STATUS_SUCCESS` 并同步绑定 stream，以建立 API 调用与 profiler task 的对应关系。失败调用不作为有效样本，报告保存失败参数与原因。

采集使用 msprof op，耗时来自导出的 `Task Duration(us)`。一次接口调用若包含多个 Cube、Vector 或 workspace 条带 kernel，先将同次调用的所有相关 task duration 相加，得到该次接口计算耗时；排除 warmup 后，对 20 个有效值取算术平均。日志保留 case 名、样本序号及调用开始/结束标记，汇总脚本核对样本数量、kernel 序列与单位。

任务书有两种性能表述：第 3.3 节要求平均耗时不超过 GPU 的 1.25 倍，第 7 节补充要求 A2 达到 GPU 性能。报告并列给出以下判定：

| 口径 | 判定 |
| --- | --- |
| 第 3.3 节，至少 0.8× GPU 性能 | \(T_{NPU}\le1.25T_{GPU}\)，即 \(T_{GPU}/T_{NPU}\ge0.8\) |
| 第 7 节，至少 1× GPU 性能 | \(T_{NPU}\le T_{GPU}\)，即 \(T_{GPU}/T_{NPU}\ge1\) |

原 `gpu_baseline.csv` 的 200 条 `gpu_ms` 数据按毫秒保存，比较前转换为微秒。配套 README 一处写 A100，另一处写 NVIDIA P102-100（CUDA 12.5、未锁频）；CSV 本身没有设备元数据。目前无法从该文件核实真实 GPU 型号。报告保留原始文件与数值，明确这项来源冲突，并按提供数值计算两个比值门槛。

## 4.4 内存与 workspace 记录

任务书第 3.4 节为“内存要求不涉及”，交付件表另要求 workspace 相对 GPU 数据。报告记录各矩阵物理字节数、Device 指针数组字节数、实际有效 workspace 容量与本算子使用量。GPU CSV 未提供 workspace 数值，因此 GPU workspace 比值标记为“缺少 GPU workspace 数据”，并保留 NPU 的实际记录。

A/B/C 的物理元素数分别采用真实 leading dimension 乘实际存储列数，再逐 batch 汇总；测试保护区和 Host golden 缓冲单列统计。workspace 设计预算与运行时实际分配容量分别记录。

## 4.5 构建、复现与交付

源码由 `blas/CMakeLists.txt` 自动收集，测试通过 `ops_blas_add_gtest_tests` 注册。基本构建入口为：

```bash
bash build.sh --ops=gemm_batched_ex --soc=ascend910b3
bash build.sh --ops=gemm_batched_ex --soc=ascend910b3 --run
```

详细数据生成、设备选择、全量精度、Device 标量及单 case profiler 命令以 `test/gemm_batched_ex/README.md` 为复现入口。交付保存源码版本、原始 CSV、各 case 参数与状态码、逐 batch 精度结果、EXACT 结果、原始日志、profiler 导出、20 次有效采样明细和汇总表。

设计评审通过并合入后按社区流程提交验收；代码提交位置为 `blas/gemm_batched_ex/arch22/`，测试代码位置为 `test/gemm_batched_ex/`。设计文档位于 `04_tasks/01_community-task-2026/tasklist/10月28日社区任务-aclblasGemmBatchedEx-A2A3/m0_70753218/docs/design.md`，通过[现有 PR #2433](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2433)评审。

## 4.6 当前验证状态

编译与各类验证结果由随交付保存的自测报告、原始日志和 profiler CSV 记录，并绑定最终源码版本。设计文档描述实现方案，验收状态以这些实际记录为依据。

## 4.7 兼容性分析

本实现通过架构目录选择参与构建，A2/A3 共用 `arch22` 源码。公共 API 名称、参数顺序、类型和符号保持与现有头文件一致；`arch35` 的同名实现继续由其目标 SoC 构建选择。新增测试使用架构目录隔离本任务的 FP32 范围与 CSV 契约。

# 5. 参考资料

1. 本任务资料包：`aclblasGemmBatchedEx_A2_task_doc.md`、`test_cases/README.md`、全部 CSV 与指导脚本。
2. [ops-blas 公共接口](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h)。
3. [ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)。
4. [社区设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
5. [社区任务流程与注意事项](https://gitcode.com/org/cann/discussions/39)。
6. [cuBLAS cublasGemmBatchedEx](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmBatchedEx)。
7. [Netlib sgemm](https://www.netlib.org/blas/sgemm.f)。
8. [混合容差精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
9. [Netlib BLAS 3.10.0 SGEMM 源码](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/BLAS/SRC/sgemm.f)。
