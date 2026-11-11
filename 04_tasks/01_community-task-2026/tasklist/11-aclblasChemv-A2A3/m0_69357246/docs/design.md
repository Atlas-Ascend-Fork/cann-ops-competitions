# 需求背景（required）

## 需求来源

2026 年 11 月社区任务：aclblasChemv 算子开发（A2/A3）。目标代码仓为 [cann/ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

`aclblasChemv` 计算单精度复数 Hermitian 矩阵与向量的乘加。Hermitian 矩阵满足共轭对称，只需引用指定的一个三角。算子采用 Ascend C kernel 直调，复用 ops-blas 句柄与 stream。

# 需求分析（required）

## 需求描述

计算 `y = alpha * A * x + beta * y`。A 为列主序，物理空间为 `lda * n` 个 `aclblasComplex`。x、y 分别支持非零正负步长。仅使用 `uplo` 指定的三角，对角元素仅使用实部。

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

## 需求拆解

1. 在公共头文件新增上述共用接口，不新增平行 API。
2. 实现 complex64 的复数乘加、Hermitian 共轭与对角实数语义。
3. 支持 alpha、beta 独立位于 Host 或 Device。
4. 支持紧凑和 padding 的 lda，支持 x、y 的正负步长。
5. 实现参数错误、空规模及标量快捷分支。
6. 在 arch22 注册 GTest，使用 Netlib BLAS 3.10.0 生成 golden。
7. 性能目标为每个验收性能用例的 NPU 平均耗时不超过 GPU 标杆的 1.25 倍。

# 详细设计（required）

## 算子分析

### 数学公式

设 `H(i,j)` 为完整 Hermitian 矩阵。对 UPPER：`i<j` 时 `H(i,j)=A[i+j*lda]`，`i>j` 时 `H(i,j)=conj(A[j+i*lda])`。LOWER 的条件相反。`H(i,i)=real(A[i+i*lda])`。

对每个逻辑输出 i，计算 `s_i = sum_j H(i,j)*x_j`，然后计算 `y_i = alpha*s_i + beta*y_i`。复数乘法使用 FP32 实部、虚部运算：`(a+ib)(c+id)=(ac-bd)+i(ad+bc)`。归约和中间结果保留 FP32，不转为 FP16/BF16。

例如 UPPER 存储 `A00=2+9i`、`A01=3+4i`、`A11=5-7i`。有效完整矩阵为 `[[2,3+4i],[3-4i,5]]`。对角虚部 9、-7 不参与计算，A10 不读取。

### 数据类型、形状与地址

alpha、beta、A、x、y 均为 complex64，按公共 `aclblasComplex` 的两个 FP32 分量存储。`n>=0`，`lda>=max(1,n)`。逻辑向量长度为 n，物理长度为 `1+(n-1)*abs(inc)`。

负步长的逻辑首地址偏移为 `(1-n)*inc`，正步长为 0。逻辑元素 i 的物理偏移为 `base+i*inc`。Host 跨度计算、kernel 地址乘法使用 int64，先提升类型再乘法。拒绝 `INT32_MIN` 步长及同族接口规定的超限跨度。

### 计算与数据流特点

矩阵访问量随 n² 增长，向量与输出随 n 增长。列主序中同一列的连续行适合 DMA 与向量计算。大规模路径复用一个已存储非对角元素，同时更新两个逻辑输出，避免补全整矩阵。小规模路径优先减少 kernel 启动和跨核合并开销。

## 算子实现

### Host 侧设计

#### 校验与快捷分支

按以下顺序执行边界检查：

1. handle 为空，返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0，返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0，直接成功。
3. 检查 uplo，非法值返回 `ACLBLAS_STATUS_INVALID_ENUM`。检查 lda、incx、incy、跨度与 alpha/beta 指针，非法值返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. 根据指针属性解析两个标量。alpha=0 且 beta=1 时直接成功。
5. alpha 非零时要求 A、x 非空。剩余分支要求输出 y 非空。beta=0 只豁免旧 y 的读取，不豁免输出指针。

标量按指针属性分别选择读取路径，不假定两者地址类型相同。Host 标量按值传入 tiling，Device 标量在正常路径由 kernel 读取，复用同族 `ComplexScalarSource` 口径。仅在 A/x/y 缺失时，Host 为判定条件空指针是否合法，按 stream 顺序读取 Device 标量并同步绑定 stream，不做设备全局同步。指针探测失败与搬运错误按仓内状态码传播。此路径的同步代价单独列入接口行为说明，不将其计作 kernel task duration。

Host alpha=0 时选择单 kernel 缩放路径：beta=0 时仅写零，其他 beta 时仅缩放 y。Device alpha=0 时由设备读取标量后跳过 A/x；若形状与步长选择了三角路径，仍执行清零 partial 与归并两个 kernel。beta=0 的所有分支均不加载旧 y，也不采用 `0*y` 代替跳过读取。

#### 初始几何调度

路径按规模和步长选择。边界值是设计参数，性能结论须来自同一套验收流程。

| 条件 | 路径 | 并行与块大小 | workspace |
|---|---|---|---|
| n=0，或 Host 已知 alpha=0,beta=1 | Host 快捷返回 | 无 kernel | 0 |
| Host alpha=0 | y 缩放/清零 | 复用输出所有者，每个任务独占输出行 | 0 |
| 1<=n<=64 | 输出所有者 | 单 kernel，每个任务独占输出行，SIMD 分段 dot | 0 |
| 65<=n<=4096 且 incx=incy=1 | 三角分块+合并 | 64×64 矩阵块，三角任务按任务编号循环分给 AIV 核 | P*8*pitch，pitch=ceil(n/64)*64 |
| incx!=1 或 incy!=1 | 输出所有者分段归约 | 单 kernel，向量按逻辑顺序映射，SIMD 分段 dot | 0 |
| n>4096 | 输出所有者分段归约 | 单 kernel，归约轴按 UB 预算分段 | 0 |

P 从平台实际可用 AIV 数与任务数量计算，不硬编码芯片核数。输出所有者路径同时支持步长，不因步长不等于 1 而固定单核。连续中大规模路径复用三角数据，其他规模和步长路径省去 workspace 与第二次 kernel 启动。

Device 标量且 A/x/y 均非空时，Host 不提前读取标量。alpha=0,beta=1 由 kernel 判定：输出所有者直接返回；三角路径不读取 A/x，归并阶段直接返回而不写 y。

#### Tiling 字段

`ChemvTilingData` 传递 n、lda、incx、incy、uplo、path、alpha/beta 分量、alphaIsDevice/betaIsDevice、useCoreNum 和 pitch。incx/incy 使用 int64，地址与跨度先提升到 64 位后运算。64×64 tile、256 元素段和 128 输出归并块为 kernel 常量，尾块长度与负步长起点在 kernel 计算，不重复存入 tiling。

### Kernel 侧设计

#### 小规模与输出所有者路径

每个任务独占输出区间。对每个输出，使用指定三角直接读取一侧，使用指定三角转置位置并取共轭得到另一侧。每个输出只写一次，不对 y 做原子更新。该路径会为不同输出重复读取同一个已存储非对角元素，换取单 kernel、零 workspace 和简单输出依赖。

输出行按 `row=blockIdx+k*blockNum` 循环分配。每行拆为对角前段、单独对角项和对角后段。非对角段按最多 256 个复数分块，每段同时搬入对应 x，使用 SIMD 复数乘法与 ReduceSum，再累加段结果。连续镜像段按连续 DMA 读取，跨列段使用 blockCount 和 srcStride 批量读取，DMA 在 UB 中的 32B 间隔由 Gather 消除。srcStride 超过 DMA 字段上限时，仅逐次提交 DMA，不使用逐元素 GM GetValue 做数据计算。row 路径 UB 总分配为 24 KiB。

#### 三角分块与双向复用路径

只枚举已存储三角中的矩阵块。对于非对角块中的元素 `a=A(i,j)`，同时产生 `s_i += a*x_j` 和 `s_j += conj(a)*x_i`。一份 A 数据服务两个方向。对角元素只产生一次 `s_i += real(Aii)*x_i`。

每个核将贡献累积到自己的 n 长复数 partial，实部与虚部分开存储。partial 在核入口清零，在该核所有矩阵任务完成后写入独占的 workspace 行。核间不共享写地址，不使用 AtomicAdd。矩阵按 64×64 分块；对角块逐列取合法行范围，在 UB 内 gather 构建完整 Hermitian tile。三角计算热循环不使用数据 GetValue 标量循环。

UPPER 的列 j 只读取行 `[0,j]`，LOWER 只读取 `[j,n)`。每次 DMA 的 blockLen 等于合法字节数。尾部补零只发生在 UB，GM 地址不向三角外或 lda padding 扩张。完整对角矩形搬入后再 mask 会读取未引用三角，因此不采用该方式。

对角项在任何乘法前用 0 覆盖虚部分量。复数乘法还单独 mask 掉对角虚部交叉项，避免 x 含 Inf 时由 `0*Inf` 产生无效值。该规则同时覆盖对角虚部为 NaN/Inf 的情况。补零尾部不参与有效元素归约。

#### 合并与写回

同一 stream 顺序启动合并 kernel，以 kernel 边界保证所有 partial 写完。合并 kernel 的每个任务独占一段输出，按核布局读取 P 行 partial，将核槽数量补到二的幂，空槽清零，再做 FP32 二分树归约。每个归并任务处理最多 128 个输出。随后融合 alpha 乘法和 beta*y，写回正确步长位置。beta=0 分支直接写 alpha*s，不读 y。

partial 的有效行全部由生产核初始化并写出，尾部不读取未初始化数据。workspace 复用 handle 的 `EnsureDefaultWorkspace` 与 `GetEffectiveWorkspace`。每核占用 `8*pitch` 字节，总容量为 `P*8*pitch`。生产与消费在绑定 stream 上顺序执行；同一 handle 的 workspace 生命周期遵循仓内既有管理约束，不引入每次调用的临时分配与阻塞释放。

#### UB 布局

以 n=4096、64×64 tile 为例，x 与核私有 partial 各 32768 B。矩阵块、复数拆分、向量乘积、归约暂存和输出区按生命周期复用，矩阵 raw、planar 和乘积暂存各 32 KiB，Gather 索引 16 KiB，x 与 partial 各 32 KiB，比较 mask 为 512 B，总分配为 176.5 KiB（180736 B）。剩余容量用于队列元数据、API 所需暂存和对齐，不直接当作可用 tile 空间。实际占用按 CANN 9.1.0 API 的暂存查询结果核算，并与平台 UB 容量比较。

| 区域 | 生命周期 | 用途 |
|---|---|---|
| xResident | 整个核执行 | 逻辑顺序 x，实虚分离或交错格式 |
| partialResident | 整个核执行 | 当前核 n 个复数贡献 |
| matrix | 单 tile | 连续列搬入、实虚分离和两向计算；生命周期结束后复用 |
| temporary | 单列/单 tile | 拆分、复数乘法、局部归约；串行阶段复用 |
| output | 最后写出阶段 | 复用已结束生命周期的矩阵或临时区 |

复数交错转实虚分量在 UB 完成。偏移表在核初始化时构建并复用。不同活跃操作数保持适当对齐与地址错位，具体 padding 按实际资源和性能证据选择，不照搬其他架构的 bank 结论。

#### 同步与竞态约束

矩阵搬入完成后，以对应方向事件或队列同步表达 MTE2→Vector 数据依赖，计算完成后再复用输入 buffer。输出在 Vector→MTE3 同步后搬出，MTE3 完成后才能复用同一物理 buffer。常驻 TBuf 的标量、向量和搬运跨流水访问使用对应方向事件。每个复用区域的生产者、消费者和回收者使用同一个 tile 索引。

partial 只由本核写，y 只由合并任务或输出所有者写。不在单 kernel 内做全核 SyncAll，避免尾核提前返回与全核屏障不配对。两阶段路径的跨核依赖由同 stream 的 kernel 边界表达。

### 工程集成

| 位置 | 改动 |
|---|---|
| `include/cann_ops_blas.h` | 共用 aclblasChemv 声明 |
| `blas/symv/arch22/` | Chemv host、kernel、tiling 与现有构建接线 |
| `blas/symv/README.md` 或仓内对应说明 | 接口、参数、Hermitian 与步长语义、A2/A3 支持表 |
| `test/symv/chemv/arch22/` | GTest、NPU wrapper、CSV 与复现说明 |
| `test/symv/chemv/CMakeLists.txt` | arch22 注册 `ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)` |

保留 arch35 实现与注册逻辑。公共 API 采用 int 维数，避免为不同产品线定义重复接口。

## 支持硬件

| 产品 | 设计支持 |
|---|---|
| Atlas A2 系列 | 支持 |
| Atlas A3 系列 | 支持 |

工具链为 CANN 9.1.0。性能目标设备为 Atlas 800T A2（910B3）。自验证按任务要求至少覆盖其中一种款型，报告明确实际设备。

## 算子约束限制

仅支持 complex64、列主序全存储，以及由 lda/incx/incy 描述的布局。不涉及广播。n 为运行时参数。y 原地更新。x 与 y 的重叠别名不作为新增扩展能力。Device 标量在缺失 A/x/y 的条件校验路径中需要读取到 Host，并同步绑定 stream。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 设计要求 | 来源 |
|---|---|---|
| Golden | Netlib BLAS 3.10.0 `cblas_chemv`，`CblasColMajor`，比较完整逻辑 y | 任务书 |
| 有限值精度 | 实虚分别按 atol=rtol=2^-13，matched_ratio>=0.99，并满足任务书最大绝对误差上限 `1e-2 or 32*ULP` | 任务书及生态混合容差标准 |
| 非有限值 | 同 NaN、同号 Inf、NaN 与 ±Inf 等效；有限与非有限不等效；非有限单独统计 | 任务书 |
| 性能 | 每个验收 case 的 NPU Avg time<=1.25*GPU Avg time | 任务书 |
| 采样 | msprof op 的 5 次预热不计入样本，随后至少 10 次有效执行取算术平均，单位 us | 任务书 |

两 kernel 路径将生产与合并两个 task duration 相加，作为一次 Chemv 的设备耗时。性能用例的重复调用须控制 y 的输入状态，保持与 GPU 标杆一致，避免 beta 非零时连续更新改变数值规模。接口同步耗时与设备 task duration 分开记录。

开发调试使用普通 msprof 定位搬运、向量计算和调度开销。最终验收使用 msprof op，严格采用上表的预热、有效采样与算术平均。两类证据分开标注，调试结果不替代最终验收数据。

## 用例与验证设计

附件 CSV 包含 1200 条：200 条 PF 性能用例，其余 1000 条为精度、边界和参数场景。期望 SUCCESS 为 1180 条，INVALID_VALUE 为 18 条，INVALID_ENUM 为 2 条。n 最大为 4096，覆盖小质数、2 的幂及相邻值、padding、正负步长和随机非对齐规模。

完整保留附件用例，并补充下列可维护回归：

1. Host/Device alpha 与 beta 的四种组合，以及绑定非默认 stream 的先写标量再调用顺序。
2. 未引用三角填 NaN/Inf，对角虚部填 NaN/Inf，验证 Hermitian 数学语义。读越界工具用于补充检查 DMA 边界；仅数值通过不证明未发生读取。
3. beta=0 时旧 y 填 NaN；alpha=0 时 A/x 为空；alpha=0,beta=1 时 A/x/y 为空。
4. n=0 时非法其他参数的校验优先级，以及 n<0、INT32_MIN、跨度超限、非法枚举与空指针。
5. ±1/±2/±3 的步长组合，输出间隙使用哨兵，验证只写逻辑 y。
6. 在各调度边界两侧及矩阵 tile 的尾部验证 UPPER/LOWER，验证每个非对角元素贡献两次、对角贡献一次。
7. 补齐 `GAUSS_NORM_5_2` 正态输入，使均匀与正态各占随机分布覆盖的一半，固定 seed 并记录数量。

负向用例使用单元素 dummy 或空指针，不按非法大跨度分配。CSV 参数解析使用足够宽的整数类型，在构造缓冲区前识别期望错误与快捷返回分支。测试代码使用仓内框架，不以测试分配异常代替接口返回校验。

先完成所有设计路径，再对全量用例统一执行精度和性能验证。报告分别记录用例数量、实际设备、工具链、提交号、有限/非有限结果、有效采样数、算术平均与逐例 GPU 比值。设计目标不作为测试通过结论。

## 兼容性分析

新增共用 API 声明与 arch22 实现。alpha/beta 属性探测对齐同族 complex 标量口径。参数状态与任务书一致，不沿用旧参考中 n<0 当作成功、非法 uplo 返回 INVALID_VALUE 等差异。编译阶段分别检查 arch22 新路径及公共头文件与已有产品线声明的一致性。
