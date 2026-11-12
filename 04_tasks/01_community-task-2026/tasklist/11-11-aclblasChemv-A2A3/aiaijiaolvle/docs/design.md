# aclblasChemv 算子设计（Atlas A2/A3）

## 需求背景（required）

### 需求来源

本设计对应 [11 月社区任务：aclblasChemv 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/cbfd47808791498d94595dd9e48500aa)，在 [ops-blas](https://gitcode.com/cann/ops-blas) 中补齐 Atlas A2/A3 的单精度复数 Hermitian 矩阵向量乘能力。

语义对标 cuBLAS `cublasChemv`，参考 [Netlib CHEMV](https://www.netlib.org/blas/chemv.f)。计算为 `y = alpha * A * x + beta * y`；A 仅引用指定三角，另一侧由共轭对称性恢复，对角元素的虚部不参与计算。

### 现状分析

已固定的 ops-blas 开发基线未提供 `aclblasChemv` 公共声明及 arch22 实现。同族 `aclblasSsymv` 已有 arch22 工程，可参考句柄、平台核数查询及构建组织；复数 Hermitian 的共轭访问与复数乘加由本任务实现。

本任务沿用仓内句柄式 BLAS API 与 Ascend C kernel 直调模式，不增加私有平行接口，不修改既有算子的计算路径。

## 需求分析（required）

### 需求描述

新增公共接口：

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

`aclblasComplex` 使用仓内公共类型，实部、虚部分别为 FLOAT32。A、x、y 位于 Device；alpha、beta 可以位于 Host 或 Device。y 为原地输入输出。

### 需求拆解

| 能力 | 设计要求 |
| --- | --- |
| Hermitian 语义 | 支持 UPPER、LOWER；只读指定三角；镜像元素取共轭；忽略对角虚部 |
| 数据布局 | A 为列主序 `lda × n`，`lda >= max(1,n)`；x、y 支持非零正负步长 |
| 标量 | 支持 Host/Device alpha、beta，不缓存跨调用的标量值 |
| 退化路径 | `n=0` 不启动计算；`alpha=0,beta=1` 不访问 A/x/y；`alpha=0` 不读 A/x；`beta=0` 不读旧 y |
| 异常处理 | 按任务书顺序校验 handle、维度、枚举、跨度及必要指针，返回仓内状态码 |
| 工程集成 | 声明放入公共头文件；实现和测试放入指定 arch22 目录，补齐 GTest 注册 |
| 自验证 | 覆盖官方 1000 条精度和 200 条性能用例，并补充正态分布等场景 |

## 详细设计（required）

### 算子分析

#### 数学公式

以零基逻辑下标表示：

```text
y[i] = alpha * sum(A_eff[i,j] * x[j], j=0..n-1) + beta * y[i]
```

对 UPPER，`i < j` 时直接读取 `A[i + j*lda]`，`i > j` 时读取 `conj(A[j + i*lda])`；LOWER 的直接半区与镜像半区相反。对角元素使用 `real(A[i + i*lda])`。

向量逻辑位置 i 对应的物理元素偏移为：

```text
inc > 0: i * inc
inc < 0: (n - 1 - i) * abs(inc)
```

所有物理地址乘加在 64 位域内计算。负步长表示反向遍历，仍执行正常计算，不按归约类 BLAS 接口的 `inc<=0` 快速返回规则处理。

#### 支持数据类型

本次仅支持 COMPLEX64。复数乘加由 FLOAT32 实、虚部分解实现，不额外引入低精度计算路径。

#### 支持形状

- A 的逻辑形状为 `[n,n]`，列主序物理存储为 `lda × n`。
- x、y 的逻辑长度均为 n，物理长度至少为 `1+(n-1)*abs(inc)`。
- 支持紧凑 lda、padding lda、非对齐长度，以及正负步长。
- n 为运行时入参，host 根据本次调用生成 tiling；不涉及图模式动态 shape 和广播。

### 算子实现

#### 实现方案

当前实现采用输出行独占、列维每块 256 个复数的向量计算方案。每行只由一个 AIV block 更新，不使用跨 block 原子累加，也不需要全局 y 缩放与归约之间的同步。矩阵的直接半区按 lda 跨列加载，镜像半区从指定三角连续加载并取共轭，对角只读取实部。

在向量结果之外增加通用的数值稳定路径，按 Netlib CHEMV 的列更新顺序重建一行结果，并显式实现与 ARM 参考库一致的 binary32 融合乘加舍入。普通 `alpha=1,beta=0,incx=incy=1` 参数先走向量计算；当某一输出分量接近零时，按参考运算顺序重算该分量，降低消去导致的相对误差。其他 alpha/beta 或步长组合使用完整参考顺序结果。该分派只依赖接口参数与本次计算值，不依赖用例编号、固定输入模式或 golden 数据。

连续矩形 DMA、多行共享 x 等方案作为后续性能调优计划单独验证，不作为当前已交付计算路径。最终纳入代码的方案需同时通过同版本的全量精度和正式性能检查。

#### 3.2.1 host 侧设计

**参数校验与快速返回**

入口按以下顺序执行：

1. handle 为空，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. `n<0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；`n=0` 返回 SUCCESS，不访问其他参数指向的数据。
3. 校验 uplo、lda、incx/incy、alpha/beta 指针。非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`；其余非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 复用仓内 `ReadScalarFromHostOrDevice` 读取标量。Host 指针直接读取；Device 指针通过 handle stream 复制和同步。读取失败向上传播，不启动 kernel。
5. `alpha=(0,0),beta=(1,0)` 直接返回 SUCCESS，此时 A/x/y 可为空。
6. 非零 alpha 要求 A、x 非空；会写入输出时要求 y 非空。`beta=0` 仅免除旧 y 值的有效性要求，不免除输出指针非空要求。

步长拒绝 0 和 INT32_MIN。`(n-1)*abs(inc)` 用 int64 计算，并沿用任务书引用的 arch22 Ssymv 大跨度守卫，拒绝超出 uint32 元素偏移域的调用。

**分核与 tiling**

从平台查询可用 AIV 核数，使用 `min(n, availableAivCores)` 个 block，不写死设备核数。block b 处理逻辑行 `b, b+coreCount, ...`，小 n 不启动没有输出行的 block。核数查询失败时返回执行失败。

`ChemvTilingData` 按值传递 n、lda、incx/incy、三角标志、实际 block 数和已读取的复数标量。列块为 256，尾块按实际元素数搬运和计算，不要求输入额外 padding。不引入图模式 TilingKey 接口，不额外申请 Device workspace。

#### 3.2.2 kernel 侧设计

kernel 分为 Init 和 Process 两个阶段。

**Init 与资源组织**

绑定 A/x/y 的 GlobalTensor，接收本次 tiling。为矩阵块、x 块、输出、计算临时量、Gather 索引和参考顺序重算缓存分配 UB buffer。连续与分块布局的正向 Gather 索引每 block 生成一次，小 n 只生成实际需要的条目。

当前源码显式分配的 UB buffer 合计 44,064 字节，其中参考顺序重算的 A/x 缓存为 16,384 字节。缓存按列块复用，避免逐个元素重复发起 GM 搬运。该数值仅为显式 buffer 之和，不代替编译器资源报告或实际占用数据。

**向量计算路径**

1. 为当前输出行初始化实、虚部累加器。alpha 非零时，单独读取对角实部及对应 x。
2. 将两个非对角区间按 256 复数分块。连续部分使用连续 DMA；跨步部分使用带 stride 的 DMA，并为每个复数保留一个 32 字节 UB 槽位。
3. x 为负步长时先按物理正向搬入，再通过逆序 Gather 恢复逻辑顺序。逆序偏移按实际块长度缓存。
4. Gather 拆出 A、x 的实、虚部，执行向量乘法、加减及 ReduceSum，块结果累加到该行。
5. 完成 alpha 缩放及 beta*y。beta=1 使用直接加法，避免无意义的 `0*Inf`；beta=0 不读取旧 y。
6. 根据通用数值稳定规则决定是否重算分量，最后按 incy 写回逻辑输出位置，不写向量间隙。

**参考顺序与软件 FMA**

参考顺序路径按 UPPER/LOWER 分别重建 Netlib 的更新顺序：先处理 beta 缩放，计算 `temp1=alpha*x[j]`，保持对应的三角贡献、对角实数贡献和共轭 `temp2` 贡献的加法次序。Reader 仅从指定三角和逻辑 x 位置读取数据，沿用同一负步长和快速返回语义。

复数乘法中需要的融合乘加通过整数有效数运算、带粘滞位的移位和一次最终舍入实现，目标为 IEEE binary32 的 round-to-nearest、ties-to-even；同时处理零、非规格化数、Inf/NaN。独立乘法和加减保留显式 FLOAT32 舍入，并用源文件编译选项约束隐式融合，避免编译器把参考运算顺序重新收缩。相关实现集中在 `chemv_reference_order.h` 与 `software_fma32.h`。

标准连续参数下，当前近零重算阈值为分量绝对值小于 1；只替换需要重算的实部或虚部。非标准 alpha/beta 或步长组合使用完整参考顺序结果。此路径为了满足输入域内的数值要求，成本计入算子实际耗时，不在性能报告中排除。

**依赖与性能优化计划**

当前代码采用单队列及明确的 CopyIn/Compute/CopyOut 依赖，不宣称已有双缓冲重叠流水。后续重点评估多输出行共享矩形搬运和 x 数据的方案，包括 8 行／K 块 512 的候选；同时保留通用步长回退及数值稳定规则。候选须单独绑定实际加载的库、完整复测精度，再以 Task Duration 判断是否纳入最终版本，不能继承当前数值通过结论。

### 支持硬件

| 产品系列 | 本次适配范围 | 自验证计划 |
| --- | --- | --- |
| Atlas A2 | arch22 | 优先在 910B3、CANN 9.1.0 上验证 |
| Atlas A3 | arch22 | 同一实现覆盖，任务书允许任选一种款型自验证，不额外要求双平台全测 |

任务书性能测试设备列为 Atlas 800T A2 910B3。CANN 9.0 环境上的检查单独记录为兼容性检查，不替代 CANN 9.1.0 的正式自验证。

### 算子约束限制

- `n>=0`；正常计算要求 `lda>=max(1,n)`，非零且非 INT32_MIN 的 incx/incy，以及不溢出的向量跨度。
- 只支持任务书定义的 lda、incx、incy 非连续访问，不增加任意视图或广播支持。
- A、x 为只读输入，y 原地更新；各操作数按任务书独立矩阵/向量语义使用。
- Device 计算下发到 handle 绑定的 stream；读取结果前由调用者同步。Device 标量解析阶段可同步该 stream，不承诺入口完全不发生 Host 等待。
- 输出采用浮点容差验收，不要求不同归约顺序的结果逐位一致。

### 可维可测分析

新增实现集中在 `blas/symv/arch22` 的 host、kernel、tiling、参考顺序与软件 FMA 文件，并在公共头文件新增声明。测试使用 `test/symv/chemv/arch22` 的 CSV 驱动 GTest，在 `test/symv/chemv/CMakeLists.txt` 的 arch22 分支注册 `chemv_test`。

官方 1200 条 CSV 保持字节不变：1000 条精度用例及 200 条性能用例。原始 `RANDOM_NORM_5_5` 始终使用随包的均匀分布语义，不按 seed 奇偶改成其他分布。另提供 48 条显式 token 补充用例，24 条 uniform、24 条 Gaussian，分别覆盖 alpha/beta/A/x/y 的分布，以及 padding、正负步长和分块边界。正式报告列明分布生成方式及实际数量，不把测试输入分布与算子随机性混淆。

测试还包括：

- 给未指定三角、lda padding、对角虚部填 NaN，验证结果不依赖这些位置。
- beta=0 时给旧 y 填 NaN；alpha=0 的合法路径使用空 A/x；无操作路径允许空 A/x/y。
- 检查步长间隙字节不变，以及 Host/Host、Host/Device、Device/Host、Device/Device 四类标量位置组合。
- 负向用例仅分配 dummy buffer，检查校验优先级、状态码和超大跨度拒绝行为，避免测试本身先发生巨量分配。

#### 精度标准 / 性能标准

| 验收项目 | 判据 | 标准来源 |
| --- | --- | --- |
| 精度参考 | Netlib BLAS/CBLAS 3.10.0 的 `cblas_chemv`，列主序、全逻辑输出向量；记录实际库版本和身份 | 任务书 §3.1、§3.2 |
| 有限值精度 | 实部、虚部分别检查，rtol=atol=2^-13、matched ratio≥0.99，并满足最大绝对误差≤1e-2 或 32 ULP 的限制 | 任务书 §3.2 |
| 非有限值 | NaN/NaN、同号 Inf、NaN/Inf 按任务书视为等效；有限/非有限、异号 Inf 不等效；非有限值全部符合规则才通过 | 任务书 §3.2 |
| 测试框架精度 | 另外按 CSV 的 `mere_threshold`、`mare_multiplier` 报告有限分量 MERE/MARE 检查，不改写原阈值列 | 配套测试指导 |
| 正式性能 | 每条性能用例采集 `msprof op` 单算子 Task Duration，预热 5 次后取至少 10 个有效样本，平均耗时≤对应 GPU 标杆的 1.25 倍 | 任务书 §3.3、`gpu_baseline.csv` |
| 内存 | 无独立内存验收门槛；报告中按模板注明 N/A，仍说明实际实现资源需求 | 任务书 §3.4 |

实现过程另以官方耗时上限的 95% 作为工程目标，即 `mean <= GPU*1.25*0.95`，留出 5% 性能余量。这是开发过程的额外目标，正式标准仍为任务书的 1.25 倍上限。不得使用墙钟或 GTest 整数毫秒耗时替代 Task Duration。

测试先覆盖代表性上下三角、负步长、特殊标量与最紧性能点，再扩大到同一份最终源码的全量自验证。保存源码/库身份、原始日志、XML、msprof 原始结果、失败和未执行项。

### 兼容性分析

这是新增公共接口，不改变既有接口签名、句柄布局及其他算子路径。接口声明与其他产品线共用，当前计算实现仅加入 arch22。已通过主仓标准 CMake 流程下的 Chemv 库和测试目标构建、链接；不将该算子子集构建表述为全部算子的全仓回归。

### 验证状态

本节对应产品代码 [PR #945](https://gitcode.com/cann/ops-blas/pull/945) 的公开 R8 提交 [4b7f5a7ce2047a9001ade34d08a2a7e3d0384530](https://gitcode.com/aiaijiaolvle/ops-blas/commit/4b7f5a7ce2047a9001ade34d08a2a7e3d0384530?ref=feat-chemv-a2a3-aiaijiaolvle-20261111)。以下完整功能自测仅对应该冻结源；尚未纳入本提交的性能候选单独验证，不继承这些通过数。

本次同步修复后的源码已在 Atlas A2 910B2、CANN 9.1.0 上，通过主仓标准 CMake 的 Chemv 库与测试目标构建。官方 1000 条功能用例全部通过，其中 976 条执行了实际数值对照，24 条为状态码或快速返回检查；数值项同时满足任务书 mixed tolerance 和配套 CSV 的 MERE/MARE。另有 48 条显式分布补充用例全部通过（24 uniform、24 Gaussian），以及同输入敏感用例额外重复 10 次全部通过。原 RANDOM_NORM_5_5 仍按 uniform[-5,5] 生成；Gaussian 使用独立显式 token。

该修复在 A/X 从输入队列取出后、向量计算前增加显式 MTE2_V 同步。上述结果来自修复源码自身的完整构建和测试，不继承修复前版本的通过数。当前提交实现仍为输出行独占的计算与必要参考顺序补偿；多行、流水和 Cube 等性能候选尚未作为本提交的验收结论。

性能尚未满足验收要求。正式交付前将选择最终源码，固定对应库、测试程序、模型输入规则和 golden 库，完成全部 200 项性能自验并补齐报告、截图和原始采集数据。按任务书 §3.1，在所用的一种受支持款型完成自验证；如自验使用 910B2 或 910B4，将明确记录实际硬件，另行区分 §3.3 给出的 910B3 性能测试平台，不将不同设备数据混称为 910B3 结果。

浮点编译选项仅作用于 Chemv kernel，不将本算子目标的构建与测试表述为全部 ops-blas 算子的全仓回归。当前不宣称全部验收条件已经满足。
