# 需求背景（required）

## 需求来源

本设计对应 **11月社区任务-aclblasChemv算子开发（A2/A3）**，为 `cann/ops-blas` 补齐单精度复数 Hermitian 矩阵与向量乘能力。

| 项目 | 依据 |
| --- | --- |
| 官方任务 | [Chemv 任务页面](https://www.hiascend.com/activities/task-center/details/cbfd47808791498d94595dd9e48500aa) |
| 官方任务总览 | [Ascend-CANN 发布的11月清单](https://gitcode.com/org/cann/discussions/22)，2026-11-10 更新 |
| 任务书与测试包 | [2026年11月任务附件](https://www.hiascend.com/p/resource/202611/6faee17b760b4accb74904f9ec97ed3f.zip)，其中 `aclblasChemv_AtlasA2_A3_task_doc.md`、`test_cases/README.md`、CSV 及验证脚本 |
| 设计模板 | [社区任务官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/f90187f2646933961fa7e3c15fca2036f6c6fdc6/04_tasks/01_community-task-2026/resources/design_template.md) |
| 源码调研基线 | `cann/ops-blas`，[commit 5bd95145a9c499e43ebad44a313a1029c63d54bc](https://gitcode.com/cann/ops-blas/tree/5bd95145a9c499e43ebad44a313a1029c63d54bc) |
| 调研日期 | 2026-11-11 |

本文为设计方案及验证计划。核数、tile 参数属于运行时选择或待调优项；精度与性能尚未实测，不以设计目标代替通过结论。

## 背景介绍

### aclblasChemv 算子实现需求

在 Atlas A2/A3 上，以 Ascend C kernel 直调方式实现句柄式 BLAS 接口：

```text
y ← alpha × H × x + beta × y
```

`H` 是由 `A` 的指定三角隐含定义的 Hermitian 矩阵；全部输入输出为 `complex64`。对外使用 `aclblasHandle_t` 绑定的 stream，输出原地写入 `y`。

### aclblasChemv 算子现状分析

以下为调研基线的实际情况，不代表本任务已完成实现。

| 文件/符号 | 当前能力 | 本任务处理 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | 有 `aclblasSsymv`、`aclblasCsymv`；无 `aclblasChemv` 声明 | 新增统一公共接口，不另设私有 API |
| `include/cann_ops_blas_common.h` | 定义 `aclblasComplex`（`float real`、`float imag`）、fill mode 和 BLAS 状态码 | 复用既有类型与枚举，不修改 ABI |
| `blas/symv/arch22/ssymv_host.cpp`、`ssymv_kernel.cpp` | A2/A3 实数对称矩阵乘向量；Host 有跨度守卫，计算分为 beta 处理与累加阶段 | 参考工程接入；不照搬其校验顺序、固定核数及原子累加方案 |
| `blas/symv/arch35/csymv_host.cpp` 中 `ComplexScalarSource` | 复数标量支持 Host/Device 来源；特殊缺指针场景可读回标量 | 参考标量与指针检查方法；不移植 950 专属 SIMT/RegBase kernel |
| `blas/common/helper/host_utils.h` | `GetAivCoreCount`、`CheckPtrLocation`、`ReadScalarFromHostOrDevice` 等 | 优先复用经环境核对可用的 helper |
| `blas/common/helper/aclblas_handle_internal.h` | handle 的 stream 及 workspace 管理 | 基础方案无需额外 GM workspace，不新增 handle 生命周期状态 |
| `test/symv/ssymv/CMakeLists.txt` | 仅 arch35 调用 `ops_blas_add_gtest_tests`，其他架构走普通 runner | 为新 Chemv 测试显式注册 arch22 GTest，不能依赖整数毫秒 runner |

### aclblasChemv 算子功能分析

矩阵采用列主序全存储，物理布局为 `lda × n`。仅引用 `uplo` 指定三角，另一侧通过共轭对称性取得，主对角元素只取实部。与复数对称的 `Csymv` 相比，反射侧必须共轭；不能只替换算子名。

| 参数 | 含义/位置 | 类型和物理形状 | 主要约束 |
| --- | --- | --- | --- |
| handle | Host 上的库句柄 | `aclblasHandle_t` | 有效非空，使用其 stream |
| uplo | Host 上的三角选择 | `aclblasFillMode_t` | `ACLBLAS_UPPER` / `ACLBLAS_LOWER` |
| n | Host 上的矩阵阶数 | `int` | `n ≥ 0` |
| alpha、beta | 复数标量指针 | `const aclblasComplex*`，Host 或 Device | `n > 0` 时非空，按指针属性识别来源 |
| A | Device 上的只读矩阵 | `complex64[lda × n]` | `lda ≥ max(1,n)`，仅指定三角可读 |
| x | Device 上的只读向量 | 至少 `1+(n-1)×abs(incx)` 个复数 | 步长可正可负；alpha 非零且 n>0 时非空 |
| y | Device 上的输入/原地输出向量 | 至少 `1+(n-1)×abs(incy)` 个复数 | 除合法 no-op 外非空；beta=0 不读旧值 |
| incx、incy | Host 上的逻辑元素步长 | `int` | 非 0、非 `INT32_MIN`；跨度和地址计算不得溢出 |

# 需求分析（required）

## 需求描述

新增 `aclblasChemv`，计算语义与 cuBLAS `cublasChemv`、Netlib `chemv` 对齐，支持 A2/A3（arch22）、上下三角、非紧凑 `lda`、正负步长、Host/Device 复数标量和任务规定的退化语义。

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

## 需求拆解

| 编号 | 需求 | 任务书依据 | 设计/验证落点 |
| --- | --- | --- | --- |
| R1 | 公共 complex64 BLAS 接口、handle stream、arch22 直调 | §2.1～2.3 | 公共头文件、Chemv Host/Kernel、链接符号与 stream 测试 |
| R2 | 仅引用指定三角、反射共轭、忽略对角虚部 | §2.1、§2.5 | 分块访存、对角块专门处理、哨兵/NaN/Inf 测试 |
| R3 | 支持 lda padding、±1/±2/±3 及合法大跨度 | §2.4、§3.5 | 有符号逻辑索引与宽位偏移；padding、负步长和越界检查 |
| R4 | 严格校验顺序、no-op、alpha=0、beta=0 | §2.4～2.5 | Host 校验与 Device 分支、错误码及不读取测试 |
| R5 | alpha/beta 支持 Host/Device | §2.4（issue #869） | 双来源标量；同 stream 先写标量再调用及连续改值回归 |
| R6 | 实/虚部分别满足 FLOAT32 精度和特殊值规则 | §3.2 | Netlib BLAS 3.10.0 golden；保留逐用例误差指标 |
| R7 | 910B3 上每个性能用例达到 GPU 标杆的 0.8 倍 | §3.1、§3.3 | 200 条标杆匹配，`ratio=GPU/NPU ≥ 0.8` |
| R8 | 全量自验、补齐分布与接口级覆盖 | §3.5 | 1000 条精度/边界、200 条性能，另补正态和接口单测 |
| R9 | 设计、代码及严格七文件验收材料可追溯 | §4～5 | 固定 commit、原始日志、报告与复现步骤；本 PR 仅交设计 |

# 详细设计（required）

## 算子分析

### 数学公式

以下索引均从 0 开始。存储元素 `A(i,j)` 的复数索引为 `i+j×lda`。定义实际参与计算的矩阵：

```text
UPPER:
  H(i,j) = A(i,j)           , i < j
  H(i,j) = real(A(i,i))     , i = j
  H(i,j) = conj(A(j,i))     , i > j
LOWER:
  H(i,j) = A(i,j)           , i > j
  H(i,j) = real(A(i,i))     , i = j
  H(i,j) = conj(A(j,i))     , i < j
```

对任一向量步长 `inc`：

```text
start(inc) = (inc < 0) ? (1-n)×inc : 0
physical(t,inc) = start(inc) + t×inc
s(i) = Σ[j=0..n-1] H(i,j) × x[physical(j,incx)]
y[physical(i,incy)] = alpha×s(i) + beta×y_old[physical(i,incy)]
```

复数乘法采用 FP32 实部/虚部拆分：`(a+ib)(c+id)=(ac-bd)+i(ad+bc)`。对角虚部直接替换为字面值 0，不能通过“虚部乘 0”忽略 NaN/Inf。alpha=0 时直接走退化分支，不计算 `0×A×x`；beta=0 时不加载或参与运算的 y 旧值。

### 支持数据类型

输入、标量及输出仅 `complex64`；内部实/虚分量与累加使用 FP32，不将输入降为 FP16/BF16，不使用低精度 Cube 结果替代规定精度。

### 支持形状

`n ≥ 0`、`lda ≥ max(1,n)`。`n=0` 无 kernel 计算；正阶数分块处理，不将随包的 `n≤4096` 人为写成 API 上限。`n`、`lda` 对外保持 int，乘积、起始位置、绝对步长及字节偏移先提升至 int64/uint64，并在转换到具体 DMA/索引字段前检查范围。

不涉及广播、batch、额外 trans 参数或超出 lda/incx/incy 语义的任意视图。仅 y 原地更新；A、x 与 y 任意相互重叠不在本任务支持范围。

## 算子实现

### 实现方案

采用 **输出所有权固定的 SIMD 分块方案**：各核拥有互不重叠的输出行块，独立遍历输入维并完成复数累加，最后将 alpha/beta 融入同一次写回。基础路径不依赖跨核原子加、不要求全局同步，也不建立完整 Hermitian 矩阵副本。

#### 3.2.1 host侧设计：

**参数检查与返回顺序**按任务书执行，不直接复用实数 Ssymv 的既有顺序：

1. `handle == nullptr`：`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n < 0`：`ACLBLAS_STATUS_INVALID_VALUE`。
3. `n == 0`：直接 `ACLBLAS_STATUS_SUCCESS`，不要求其余参数有效，不下发 kernel。
4. 检查 uplo、lda、incx/incy、alpha/beta 指针：非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`；其余非法返回 `ACLBLAS_STATUS_INVALID_VALUE`。拒绝零步长与 INT32_MIN；跨度在 int64 域计算，沿用 arch22 同族 uint32 元素索引守卫，并校验后续字节地址计算。不能先求 int32 的 abs。
5. 确定标量来源及必要的值：`alpha=0 && beta=1` 时成功返回；否则按值检查 A/x/y。alpha=0 时 A/x 可以为空；需要写输出时 y 必须非空，beta=0 不豁免输出存储。
6. 获取实际 AIV 数和 UB 容量，建立本次调用的分派参数，在 handle stream 下发；资源查询或运行时 API 失败按仓库既有错误码映射，不返回虚假成功。

**Host/Device 标量方案**：Host 指针读取一次并随 tiling 传值；Device 指针保留来源标志与地址，在本次 handle stream 的 kernel 中读取，保证此前同 stream 写入已生效，不在 Host 直接解引用 Device 地址，也不跨调用缓存标量数值。当数据指针缺失、Host 必须基于标量值决定空指针是否合法时，参考基线 `ComplexScalarSource`，将需要的 Device 标量在同 stream 读回并同步后判断，明确这一特殊路径可能同步。普通数据指针完整的 Device 标量路径保持异步，由 kernel 在读 A/x/y 前处理退化分支。

**拟新增/修改工程范围**：

| 路径 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增公共 Chemv 声明 |
| `blas/symv/arch22/chemv_host.cpp` | 参数校验、标量来源、平台资源查询、分派与启动 |
| `blas/symv/arch22/chemv_kernel.cpp` | SIMD kernel 与直调启动入口 |
| `blas/symv/arch22/chemv_kernel.h`、`chemv_tiling_data.h` | 按仓内同族模式组织入口及标量 tiling 字段 |
| `blas/symv/README.md` | 补充 Chemv 接口、产品支持和约束 |
| `test/symv/chemv/` | CMake、参数读取、CBLAS golden、arch22 GTest 与 CSV |

基线 `blas/CMakeLists.txt` 按架构收集源文件，并支持按 `chemv_` 前缀筛选；实现阶段检查 `--ops=chemv` 的最终源列表、公共符号导出及链接，不为新增算子重写无关构建规则。测试必须在 arch22 分支调用 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)`。

tiling 策略：

##### 1. 分核策略：

设单个输出 tile 含 `B` 个逻辑行，输入分块长度为 `K`；`Q=ceil(n/B)`，实际使用核数 `C=min(Q,GetAivCoreCount())`。核 c 处理 `q=c,c+C,...` 的输出块，块内有效行数 `min(B,n-qB)`。最后不足 B 的行只计算有效部分，不产生越界写入。小尺寸根据任务数收敛核数，避免对单个输出启动满核。

每个输出行只归一个核所有，核内完成整个 j 维归约，再融合 alpha/beta 写回，因此不存在“某核缩放 y、其他核向 y 累加”的竞态。大 n 通过输入维多次循环完成，不因输入分块而增加独立 kernel 启动。

##### 2. 数据分块和内存优化策略：

列主序下优先搬运连续列片段。对输出行块 I、输入块 J：

- 位于所存三角内的非对角矩形块：读取 `A[I,J]`，按列对 x 的对应分量进行复数乘加，向输出 B 行累加。
- 位于反射侧的块：读取实际存储侧 `A[J,I]`，在 UB 中按共轭转置含义计算；连续读取每列的输入行段，再归约至对应输出行，不能从未存储侧直接取值。
- 跨主对角线的块：按每一列与有效三角的交集拆分 DMA，先仅搬入合法片段，再在 UB 内构造共轭反射。不得先 DMA 整个矩形、再靠 mask 隐藏未引用三角的读取。
- 对角实部单独参与运算，虚部直接置零；lda padding 与不对应有效元素的区域不参与加载或归约。

采用 `DataCopyPad` 处理非对齐有效字节长度，UB 每段起始地址按 32B 对齐；严格满足条件时才使用普通 `DataCopy`。AoS 复数按浮点分量解释，通过批量重排/GatherMask 分离实虚分量；不使用 GM 上逐元素 GetValue/SetValue 作为计算主体。

**UB 预算**采用以下保守上界，`B`、`K` 均指分配容量而非尾块有效长度；各子段再按 32B 对齐：

| 缓冲区 | 字节上界 | 生命周期与用途 |
| --- | --- | --- |
| A 搬入双缓冲 | `16BK` | 两份 complex64 tile，用于 CopyIn/Compute 重叠 |
| A 实虚平面 | `8BK` | 一份实部与一份虚部，计算后复用 |
| x 搬入双缓冲 | `16K` | 两份 complex64 输入块 |
| x 实虚平面 | `8K` | 计算各个输出行共用同一输入块 |
| 乘积/分块归约暂存 | `8max(B,K)` | 实虚工作数组，阶段之间复用 |
| 输出累加 | `8B` | B 个复数的 FP32 实虚累加 |
| y 输入/输出 staging | `64B` | 两份按每行 32B 保守预留的步长暂存，连续场景可缩减 |
| y 旧值实虚平面 | `8B` | beta 非零时使用，与纯计算临时区按生命周期复用 |
| 归约 API 临时区 | `Treduce` | 依实际 CANN 9.1 API 的临时空间要求计算，不假设为零 |
| 对齐/掩码等保留量 | 至少 4096 | 设计预留，若实际元数据更大则相应增加 |

保守总量：`U(B,K)=24BK+24K+8max(B,K)+80B+Treduce+4096`，所有乘积在宽位整数域计算。Host 使用实际 UB 容量枚举候选 B/K，选取满足分配和 API 字段限制的较大 K，并对最终逐段对齐后的总量复核，不能仅用未对齐公式判断。

例如 `B=16,K=384` 时，不含 Treduce 及逐段额外对齐的部分为 165120 字节；此数仅用于说明预算方法，不是固定参数或实测最优值。A、x 两个双缓冲加 y 输入/输出的事件资源需一并核对。若小尺寸无需流水，切换单缓冲并复用不重叠临时区；容量不足时缩小 K/B，不能溢出 UB。

##### 3. tilingkey规划策略：

kernel 直调不使用 aclnn executor 或图引擎的 tilingkey 注册，拟用普通标量 `mode` 及明确启动入口表达分派，字段不包含变长数组。

| mode/场景 | 决策条件 | 处理方式 |
| --- | --- | --- |
| Host no-op | n=0，或已知 alpha=0/beta=1 | 不下发 kernel |
| Scale/Zero | alpha=0 | y=beta×y；beta=0 直接写零，beta=1 不访问 y |
| Small | n 可在预算内全载，且小规模启动更合适 | 减少核数/循环与初始化开销，阈值经 B3 实测确定 |
| Tiled contiguous | incx=incy=1 | 采用输出分块与连续列片段搬运 |
| Tiled strided | 其余合法步长 | 与相同数学/归约路径共用，额外处理打包和写回位置 |

拟传字段包含 n、lda、uplo、incx/incy、x/y 起始偏移、B/K、核数、mode、标量值及来源标志；具体宽度以已核实的最大地址与 API 字段范围确定，不将逻辑长度与对齐容量混用。UPPER/LOWER 使用相同接口和各自的合法三角访问规则，不按用例编号分派。

#### 3.2.2 kernel侧设计：

执行顺序为 Init → 必要的 Device 标量加载 → 退化分支 → 对每个自有输出块执行 CopyIn/Compute/CopyOut。

1. **Init**：使用外部构造的 TPipe 及本次 tiling 初始化队列/缓冲。设置 GM 的实际物理跨度与 64 位偏移；尾块只以有效长度参与计算。
2. **CopyIn**：按合法三角区间读取 A，读取逻辑 x 块并拆实虚。incx<0 时从对应物理区间正向搬运后在 UB 反序/重排，DMA 不传负 stride；abs(incx)>1 时分批搬运或重排提取逻辑元素，不将间隙加入归约。
3. **Compute**：按直接块/共轭反射块采用向量乘加及 FP32 局部归约，保持多个局部和后分层合并。K 分块间合并实虚部分和，避免将长向量退化为大量 GM 标量访问；性能与误差以实测决定是否进一步调整累加次序。
4. **Epilogue**：beta 非零才加载旧 y；融合复数 alpha/beta，避免独立 beta kernel。beta=0 直接覆盖输出，不能读 NaN 旧值再乘零。
5. **CopyOut**：只写属于本核的逻辑 y 元素。负 incy 映射至其物理位置；跨步写回只覆盖目标复数，保持间隙与 guard 区不变。非连续 DMA 的 UB 端按每条记录 32B staging，必要时分批，避免以未对齐的 8B 偏移作为 UB 源起址。

**同步与生命周期**：A/x 搬入采用 TQue EnQue/DeQue 建立 MTE2→Vector 依赖，输出队列建立 Vector→MTE3 依赖，缓冲在消费/写回完成后再回收。仅有 Vector 内部依赖时采用适用的向量流水同步；不将同一流水线 barrier 当成跨引擎数据就绪的替代。双缓冲只在生产/消费顺序和事件资源满足时启用。无跨核共享写，无需 SyncAll 或原子加。

**workspace**：基础方案额外 GM workspace=0，不改变 handle workspace 的所有权、大小或释放规则。x 的打包、A 的平面拆分和 y staging 均为分块 UB 临时数据，不分配 n×n 的完整矩阵或持久缓存。性能优化若需增加分阶段 workspace，须另行给出容量、stream 顺序和不足时回退方案后更新本设计；当前方案不以该优化作为正确性前提。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 | 实现路径/验证定位 |
| --- | --- | --- |
| Atlas A2 系列产品（含 Atlas 800T A2） | √ | arch22；910B3 为任务指定性能设备 |
| Atlas A3 系列产品（含 Atlas 800T A3） | √ | arch22；同源构建和适用精度验证 |
| Atlas 950 / arch35 | 不在本任务范围 | 不将 arch35 编译通过或测试结果作为 A2/A3 证据 |

## 算子约束限制

- 实际开发/测试环境须核对 CANN **9.1.0**、Netlib BLAS **3.10.0**、真实芯片、编译类型和实际加载库。任务 README 的“指定版本及以上”不能覆盖任务书的明确版本。
- 同步前不得读 Device 输出；普通 Device 标量路径的值按 stream 顺序读取，特殊缺指针校验路径可能为判定合法性同步 stream。
- n=0 优先于其余参数校验，但不能优先于 handle 判空和 n<0 判断。Host 已知 alpha=0/beta=1 的 no-op 不跳过 uplo/lda/inc/标量指针的规定检查。
- 只读指定三角，不写 A/x，不通过完整矩形加载引入未指定三角的数据依赖；对角虚部及 padding 不参与计算。
- 不要求位级确定性。不同 tile/核配置下仍须满足相同精度标准，不得按测试 case 或 golden 特判输出。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | Netlib `cblas_chemv`、列主序、单标杆；实部和虚部分别采用 `atol=rtol=2^-13`，matched_ratio≥0.99，最大绝对误差满足任务书 `1e-2 或 32×ULP` 上限 | 任务书 §3.2、其固定版本精度标准链接 |
| 特殊值 | 按任务书单独判定非有限值；不能把 NaN 简单计为匹配或漏出统计 | 任务书 §3.2 |
| 性能标准 | 910B3 上，200 个性能用例逐条要求 `T_NPU≤1.25×T_GPU`，等价于 `ratio=T_GPU/T_NPU≥0.8`；两端必须同 case、同单位 | 任务书 §3.1、§3.3；随包 gpu_baseline.csv |
| 内存标准 | 不涉及独立内存指标；仍检查越界、间隙写保护和 UB 预算，七文件材料中内存报告/日志明确填不适用及依据 | 任务书 §3.4、§4 |

### 精度方法与特殊值处理

保留官方 Netlib 3.10.0 `cblas_chemv` golden，不擅自改用 Chbmv 的双精度 golden 口径。对角虚部在 golden 与 kernel 两侧一致忽略；取有效的逻辑 y，全向量、实虚部分别比较：

```text
finite_match = abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)
pass = 两个分量分别满足有限值比例和最大误差上限
       且所有非有限值元素满足任务书的特殊值等效规则
```

非有限值不计入有限值 matched_ratio/max_abs_error，另报总数与通过数：同为 NaN、同号 Inf、以及任务书明确允许的 NaN 与 ±Inf 组合按其规则通过；异号 Inf 或有限值与非有限值组合不通过。有限集合为空时报告 N/A，不除以零、不制造 matched_ratio=0 的失败；仍须全部非有限值符合规则。32×ULP 的具体实现与任务固定版本标准保持一致，并输出阈值依据。

测试工程保留 CSV 的 MERE/MARE 配置与输出，但不能用 MERE/MARE 通过替代任务书混合容差判定。若库内默认阈值或通用文档不同，分别记录；本任务显式 `2^-13` 的要求不被较宽默认值覆盖。

### 测试覆盖与可复现计划

随包 `chemv_test.csv` 实读 **1200** 行：1000 条非性能用例及 200 条 `TC_PF_` 用例；有 1180 条期望 SUCCESS、18 条期望 INVALID_VALUE、2 条期望 INVALID_ENUM。CSV 期望值不能代替执行结果。

| 覆盖项 | 设计验证方法 |
| --- | --- |
| 基础/规模 | 上下三角、小尺寸、质数、2 的幂及邻接、非对齐、大规模；性能 n 覆盖 1～4096 |
| 复数语义 | 非零实/虚标量、纯虚数、负数；反射侧共轭、非对称填充的未引用三角 |
| 三角与对角 | 未引用三角填 NaN/毒值并对照；对角虚部为 NaN/Inf；配合代码访存审计，不能仅凭数值通过声称物理未读 |
| 内存布局 | lda=n/n+padding；incx/incy 的 ±1/±2/±3 交叉；对 A/x 只读区、y 间隙与 guard 区做前后比较 |
| no-op/退化 | n=0；alpha=0/beta=1 且 A/x/y 为空；alpha=0 的 scale/zero；beta=0 且 y 旧值为 NaN |
| 错误/边界 | handle 判空优先级、非法 uplo、负 n、非法 lda、零步长、INT32_MIN、跨度溢出、条件空指针 |
| 标量来源与 stream | Host/Host、Host/Device、Device/Host、Device/Device；同 stream 先修改 Device 标量，连续调用修改同一地址，验证没有数值缓存 |
| 输入分布 | 保留原始用例，追加明确种子与 μ/σ 的 `GAUSS_NORM_5_2`；使随机分布覆盖满足均匀/正态各 50% 要求，并记录每类数量，不能只把 token 改名 |
| 特殊值 | Inf/NaN、大值、抵消、全零；分别输出有限误差及非有限等效统计 |

负向用例在接口应拒绝前不按巨大 n/inc 分配内存，以 dummy 或空指针装配；合法的非零规模负步长计算仍按真实物理跨度分配。handle 为空及复杂校验优先级不能只靠 CSV 表达，须补独立 GTest。新增分布/接口测试与原始 CSV 分开统计，不删改失败用例，不扩展性能用例后继续套用失配的旧标杆。

拟在匹配环境执行以下入口；命令为计划，未执行：

```bash
# 当前 shell 先加载本机实际的 CANN 9.1.0 环境并确认 Netlib 3.10.0 动态库。
python3 test_cases/verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b --csv test_cases/chemv_test.csv
python3 test_cases/verify_performance.py --repo /path/to/ops-blas --soc ascend910b --timeout 3600
```

构建及 profiler 仅在远端匹配环境进行。每轮记录源码 commit/未提交差异、实际 SOC、库路径与版本、构建类型、命令、退出码、总数/通过/失败/跳过及官方原始输出；报告每个用例的参数、实虚精度、NPU/GPU 耗时与 ratio。不能以汇总出现 PASS 或退出码 0 推断全部要求达标。

### 性能采集与风险处理

随包 200 条 GPU 标杆已填数值，README 声明来源为 NVIDIA A100-SXM4-80GB、CUDA 12.8 / cuBLAS 12.8.4；本设计引用附件来源，不声称重新实测 GPU。`gpu_ms` 转为微秒须乘 1000；Chemv 按 uplo、n、alpha/beta 实虚、incx/incy、lda 联合匹配，不能仅按行号匹配。

任务书 §3.3 要求 `msprof op` 的 kernel task duration，而随包 README/脚本用 `aclrtEvent` 的 `[PERF] ... avg_us=`，且明确 msprof 不参与其脚本判定，二者存在实质差异。本设计保留原脚本结果，同时按任务书采集另一套数据，报告分别列出，不修改脚本阈值或用较优计时项覆盖另一套。最终验收采用哪一计时口径，作为需维护者明确的事项；这不阻止功能与双口径采集准备。

按任务书先 warmup 5 次、有效样本至少 10 次取算术平均，失败调用不能作为有效样本。若一逻辑调用包含多段任务，要汇总该调用全部相关任务，而非只选最快 kernel；记录 profiler 分组/过滤依据。正确性检查和 Host 造数/H2D/D2H 不混入纯算子计时；脚本自身的 event 原始结果照实保留。

小尺寸受下发、指针属性查询与初始化开销影响：附件最小标杆约 6.144 μs，对应 NPU 预算约 7.680 μs。优先采用单启动、少核、融合 epilogue 与无额外 workspace 的方案；大规模重点分析列片段搬运、反射块归约和 x 复用。瓶颈必须依据 B3 测量判定，不能预先承诺 200 条性能全部达标。

### 可维护性及交付

沿用同族目录和命名，仅扩充 Chemv 必需接口、源文件、测试与 README；不修改其他算子的数值契约。实现与设计偏离时更新本设计，并保留对应源码和测试证据。

设计评审交付仅包含本文件。后续算子交付固定代码 commit，七个验收文件按任务模板制作并放入私仓 `task_submission/`；内存两项填写不适用，原始测试产物和内部索引保存在正式七文件目录外。本文不把设计 PR 的提交视为评审通过、合入或算子验收通过。

## 兼容性分析

新增 `aclblasChemv` 公共声明，复用既有 complex、handle、uplo 与状态码，不改变现有 `Ssymv/Csymv` 接口、结构体布局和行为。A2/A3 使用 arch22 同源实现，950 的同名 API 如后续补齐应共享公共声明，不能创建私有平行接口。

既有 Ssymv 的 `n<=0`、非法 uplo 返回值及固定核数不作为 Chemv 契约；Chemv 严格按本任务的 n<0、n=0、INVALID_ENUM 与动态资源获取实现，并通过接口测试防止与其他产品线混淆。

需维护者澄清但不伪造结论的事项：任务书与脚本的性能计时口径。CANN 版本、精度阈值和 golden 已按任务书明确要求设计；采集结果保留两套原始数据及判定。

修订记录：2026-11-11，wangyuyan_44，V1.0，基于官方模板完成 Chemv A2/A3 方案与验证计划；AI 辅助：Codex / GPT-6.1 Sol。
