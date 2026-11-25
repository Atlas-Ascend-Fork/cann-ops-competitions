# aclblasChpr2 算子设计（Atlas A2/A3）

# 需求背景

## 需求来源

2026 年 11 月社区任务「aclblasChpr2 算子开发（A2A3）」，以任务包中的 `aclblasChpr2_AtlasA2_A3_task_doc.md` 及附带 CSV、GPU 基线为接口与验收依据。目标为 Atlas A2/A3、arch22，复用 ops-blas 公共 BLAS API；golden 使用 Netlib BLAS 3.10.0 `cblas_chpr2`。

实现位于 `ops-blas/blas/spr2/arch22/`，测试位于 `ops-blas/test/spr2/chpr2/arch22/`，开发环境使用 CANN 9.1.0。设计涵盖接口、算法、资源分配、同步及验证方法。

## 背景介绍

Chpr2 对单精度复数 Hermitian 矩阵执行秩 2 更新：

```text
A ← A + alpha · x · yᴴ + conj(alpha) · y · xᴴ
```

其中 `ᴴ` 表示共轭转置。矩阵按列保存上三角或下三角的 `n(n+1)/2` 个复数，无需展开完整矩阵。对角元素只取实部，更新后虚部置零；`alpha=0` 的完整 quick return 除外。

ops-blas 已有公共接口及 arch22 实现。本设计围绕 packed 存储的负载不均、向量输入复用、非对齐列首、小矩阵调度和 FP32 运算顺序优化已有能力。arch22 采用 AIV 向量指令、UB 缓存和 MTE 搬运；950 的 SIMT lane 索引及 RegBase 接口不直接复用。

# 需求分析

## 功能范围

| 项目 | 设计要求 |
| --- | --- |
| 数据类型 | COMPLEX64；实部、虚部均为 FP32，交错存储，每个复数 8 B |
| 矩阵存储 | 列主序 Hermitian packed，支持 UPPER/LOWER |
| 标量 | alpha 支持 Host / Device 复数指针，运行时查询其位置 |
| 向量与矩阵 | x、y 为只读 Device 指针；ap 为 Device 指针，原地更新 |
| 步长 | incx、incy 以复数元素为单位，支持正负值 |
| 执行 | 在 handle 绑定的 stream 上异步执行；有效调用启动一个 AIV kernel |
| 资源 | 按实际 AIV 数与 UB 容量分派，无额外 GM workspace 或跨核归约 |
| 产品 | 目标为 Atlas A2/A3、arch22；资源预算参考本机 Ascend910B3 |

复用现有句柄、状态码及复数类型，不增加 lda、batch 或广播参数。

## 接口定义

沿用 `include/cann_ops_blas.h` 的声明：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy,
    aclblasComplex* ap);
```

| 参数 | 说明与约束 |
| --- | --- |
| handle | 有效句柄，提供执行 stream |
| uplo | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | 矩阵阶数，`n≥0` |
| alpha | Host / Device 上的单个 complex64 标量；包括 n=0 在内均要求非空 |
| x、incx | x 为 Device 地址，至少 `1+(n-1)·abs(incx)` 个复数；步长非 0、非 `INT_MIN` |
| y、incy | y 为 Device 地址，长度与步长约束同 x |
| ap | Device 地址，至少 `n(n+1)/2` 个复数 |

quick return 的校验优先级见 Host 设计。调用方提供满足跨度要求的有效 Device 缓冲区；Host 检查字节跨度溢出。

# 详细设计

## 存储映射与边界语义

下标从 0 开始，packed 列首和元素地址为：

```text
UPPER: Base(j) = j(j+1)/2
       AP[Base(j)+i],       0 ≤ i ≤ j < n
LOWER: Base(j) = j(2n-j+1)/2
       AP[Base(j)+(i-j)],   0 ≤ j ≤ i < n
```

正步长起点为 0，负步长起点为 `(1-n)·inc`；第 k 个逻辑元素的物理下标为 `start+k·inc`。通用路径和行条带路径的 packed 偏移、向量起点及跨度乘积使用 64 位计算。列分组路径限定 `n≤2048`，其列首递推可安全使用 32 位运算。

每列复用两个复数系数：

```text
t1 = alpha · conj(y[j])
t2 = conj(alpha · x[j])
u  = x[i] · t1
v  = y[i] · t2

非对角：AP' = (AP + u) + v
对角：  Re(AP') = Re(AP) + (Re(u) + Re(v))，Im(AP') = +0
```

旧对角虚部不参与计算。列系数及对角实乘积显式保存 FP32 乘积再加减，保持 Netlib 的舍入分组，避免编译器收缩改变结果。对角的两项增量先相加，再与原对角实部相加。

当 x[j]、y[j] 的四个实虚分量都为零时，保留该列非对角值，只清零对角虚部。`alpha=(0,0)` 则为完整 no-op，包括保留对角虚部。单位 alpha 模板保留必要的零乘法和独立舍入，以维持 `0×Inf`、NaN 及对角的计算语义。

## Host侧设计

### 参数校验与返回值

按以下顺序处理，先命中的条件决定返回值：

| 顺序 | 条件或动作 | 结果 |
| --- | --- | --- |
| 1 | handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | n<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | alpha 为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | uplo 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| 5 | incx、incy 为 0 或 `INT_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | n=0 | `ACLBLAS_STATUS_SUCCESS`，不检查 x/y/ap，不读取 alpha 数值，不启动 kernel |
| 7 | x、y、ap 任一为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | alpha 实虚部均为零 | `ACLBLAS_STATUS_SUCCESS`，不读写 x/y/AP |
| 9 | 向量或 AP 字节跨度超过 `size_t` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | AIV 数获取失败或 UB 不足以容纳最小通用 tile | `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 11 | 生成 tiling，在 handle stream 上启动 kernel | 返回 `ACLBLAS_STATUS_SUCCESS`，保持异步语义 |

alpha 通过公共 `CheckPtrLocation` 查询位置；Host 标量直接读取，Device 标量在 handle stream 上复制 8 B 到 Host 并同步该次读取，再进行零值判断和 tiling。Device 标量读取遵循此前同一 stream 的写入顺序；后续计算 kernel 异步下发。`n=1` 支持 `±INT_MAX` 步长。资源校验使用 `GetAivCoreCount()` 和平台 `GetCoreMemSize(UB)`，不以“910”名称假设具体核数或 UB 容量。

### 分派与分核

启动核数 `K=min(n,实际 AIV 数)`。Host 从五个入口中选择一个：

| 入口 | 选择条件 | 工作方式 |
| --- | --- | --- |
| `chpr2_kernel_rows` | 优先选择：单位正步长、UB≥180224 B，且 UPPER n>1024 或 LOWER n≥897 | UPPER 2048 行正向条带，LOWER 1024 行反向条带，8 列组镜像分核 |
| `chpr2_kernel_small` | 列分组启用，n≤256，UPPER | 8 列分组、流式系数、对角掩码 |
| `chpr2_kernel_small_lower` | 列分组启用，n≤256，LOWER，一般 alpha | AP 左补齐、独立输出槽 |
| `chpr2_kernel_small_unit_lower` | 上述小 LOWER 条件且 alpha=(1,0) | 单位标量模板 |
| `chpr2_kernel` | 其余情况 | 通用窗口或中尺寸列分组 |

列分组要求 `incx=incy=1`、`1≤n≤2048` 且 UB 预算满足；UPPER 另限 `n≤1024`。行条带条件满足时优先使用行条带。分组的缓存偏移数 P 和列组宽 B 为：

| 范围 | P | B |
| --- | --- | --- |
| n≤256，UPPER/LOWER | 1 | 8 |
| 257≤n<512，UPPER | 1 | 16 |
| 257≤n<512，LOWER | 4 | 16 |
| 512≤n≤1024，UPPER | 1 | 16 |
| 512≤n≤2048，LOWER，未选择行条带时 | 按 UB 选择 2/3/4 | P=4 且预算允许时为 8，否则为 4 |

特化条件不满足时回退通用路径。非行条带路径在 n≤256 时按列数均分，余数分配给较短列的一端：UPPER 前端、LOWER 后端；n>256 时按累计成本 `Cost(j)=Base(j)+256·j` 划分连续列区间，同时考虑元素量和逐列开销。

逻辑列边界映射到 packed 后向上对齐 4 个复数，即 32 B，并裁剪至 packed 长度。相邻核共享同一边界计算，处理半开区间 `[begin,end)`。即使边界切入列内，也仅更新本核所属行段，末尾按实际字节数写回；每个对角只有一个核负责。

行条带路径以 8 列为一组，组号 `g=round·K+rank`；偶数轮 `rank=blockIdx`，奇数轮 `rank=K−1−blockIdx`，将较长与较短列组配对。UPPER/LOWER 的 `Base(8q)·8` 均为 32 B 的整数倍，同一列所有条带始终属于同一核；尾组裁剪至 n。

### Tiling信息

tiling 包含 n、uplo、alpha 实虚部、64 位步长与起点、packed 长度、输入缓存容量、缓存偏移数、列组宽及小矩阵分列商余。Host 与 kernel 共享结构和 tile 常量。行条带模式以专用分派值标识，kernel 再按 UPPER/LOWER 选择对应条带尺寸。

## Kernel侧设计

### 计算流程

每核确定所属列或列组，加载 x/y 逻辑窗口到 UB，并生成 i·x、i·y 缓存。随后计算列系数，搬入对应 AP 段，依次累加两项复数乘积，单独处理对角，再按有效长度写回。AP 使用双缓冲，输入缓存可在多列间复用；跨核无归约、原子写回或共享 workspace。

### 通用窗口与小矩阵

通用路径按 512 个复数处理 AP tile，以 DataCopyPad 搬运尾块。单位步长使用连续输入缓存；非单位步长按绝对步长选择输入搬运方式：步长≤32时连续搬入包含间隙的物理窗口，再Gather有效复数；更大步长采用分块DMA，每个有效复数搬入一个32 B的UB块，再压紧到输入缓存。负步长按物理地址升序搬入，通过反向索引恢复逻辑顺序；超过DMA stride字段范围时回退标量读取。

跨步输入每批最多 512 个复数，临时搬运区占 16384 B，复用四个连续工作 tile；索引与反向索引工作区复用后续工作 tile，不增加 UB 总预算或 GM workspace。该暂存区与 AP 双缓冲分离，Gather 完成后通过 V_MTE2 同步再复用。对齐偏移直接使用缓存，其他偏移通过 Gather 整理。大尺寸分窗口重载输入，缓存容量不构成 n 的接口上限。

列分组路径同时组织多列 AP 搬运，摊薄同步与指令开销。LOWER 中尺寸保存 2/3/4 种偏移的缓存，减少重复 Gather。小矩阵和 LOWER P=4/B=4 使用流式列循环，其余分组路径保存一组列系数。输入加载与索引生成、首批 AP 搬入与缓存转换可重叠。

小 LOWER 的 `SHIFT_AP` 路径保存一种输入偏移，将 AP 左补齐到对齐输入位置；保留旧对角实部，依次累加两项乘积，恢复单独计算的对角，最后 Gather 到独立输出槽。小 UPPER 以向量掩码保护对角。单位 alpha 特化在 Host 分派，减少列内重复判断。

### 大矩阵行条带

UPPER 从前向后遍历，每次最多 2048 行；LOWER 从后向前遍历，每次最多 1024 行，内部条带起点按 4 个复数对齐。每核先判断当前窗口是否与自己的列组相交，无交集时跳过输入加载。

窗口中的 x、y、i·x、i·y 被该核所有相交列组复用。列系数所需元素位于窗口内时从 UB 读取，否则按逻辑索引从 GM 读取；无需为每列重新加载完整向量窗口。

LOWER 非对齐段从对角开始时，将 AP 在 UB 内左补齐到对齐输入位置。两项乘积直接使用同一输入窗口，按 `(AP+u)+v` 累加并恢复对角后，只对最终结果做一次 Gather，写入交替输出槽。补齐前缀不写回 GM，搬出长度仍为原有效段长度。

### 数值运算

复数向量乘法拆分实虚部，使用向量乘法和加减指令。小分组与 n≥512 路径采用 Muls/Axpy 组合，其余较小通用或中尺寸路径采用分立乘加；AP 的两项更新仍顺序执行。列系数和对角使用独立的 FP32 舍入控制，不随向量路径选择改变分组。

对角虚部污染、相消、Inf/NaN、subnormal 和正负零按前述语义处理，不以只对有限值成立的代数化简替代原运算。

### UB布局与同步

令 `R=ceil(n/32)·32`、P 为缓存偏移数、B 为列组宽，UBBytes 为平台查询值。四组复数缓存每个逻辑元素合计 32 B。通用 scratch 为 `11·512·8=45056 B`。

| 路径 | UB 字节数 |
| --- | --- |
| UPPER 行条带，T=2048 | `11·T·8=180224`；四组输入缓存、两个 AP 槽、三个工作槽、两张索引表 |
| LOWER 行条带，T=1024 | `13·T·8=106496`；在上述布局上增加两个最终输出槽 |
| 通用，P=1 | `32·C+45056`，`C=align_down(min(4096,floor((UBBytes−45056)/32)),32)` |
| 通用，P>1 | `32·P·R+45056` |
| 普通列分组 | `(32·P+8·(B+2))·R+8192` |
| 小 LOWER，SHIFT_AP | `(32+8·(2B+2))·R+8192` |

Host 对两个行条带模板使用统一的 180224 B 容量门槛。预算参考设备 Ascend910B3（dav-2201）为 40 AIV、UB 196352 B；其他设备按查询结果选择路径。上述型号只说明开发环境，不替代其他 A2/A3 产品的验证。

| 数据依赖 | 同步要求 |
| --- | --- |
| MTE2 搬入后标量读取 | `MTE2_S` |
| MTE2 搬入后向量使用 | `MTE2_V` |
| 标量读写完成后向量使用 | `S_V` |
| 向量结果供标量处理对角 | `V_S` |
| 标量/向量写完后搬出 | 按实际生产者满足 `S_MTE3`、`V_MTE3` |
| 槽内数据搬出完成后重新加载 | `MTE3_MTE2`，按双缓冲槽等待 |

保留编译器自动同步并显式处理跨流水线依赖。预加载后无工作区间的核须消费对应事件。counter 模式结束恢复普通 mask；五个入口经过统一结束处理，完成流水屏障和 mask 恢复，支持后续连续调用。

### 输出搬运

所有路径只写所属 packed 有效范围，尾块通过 DataCopyPad 按实际字节数搬出。行条带的输出槽与 AP 槽遵守同一复用等待关系，避免尚未完成 MTE3 时覆盖数据。x/y 及步长间隙不写入。

# 可维可测分析

## 验证方法

精度使用 Netlib BLAS 3.10.0 `cblas_chpr2` 单精度复数输出作为 golden，实虚分量分别比较。按任务书 §3.2 使用 `atol=rtol=2^-13`、`matched_ratio≥0.99`，并满足最大绝对误差 `1e-2` 或 `32×ULP` 的限制；非有限值按任务引用的精度标准处理。FP64 仅辅助诊断。任务未要求逐位一致或新增其他 dtype。

原始 CSV 共 1200 条：1000 条精度用例和 200 条性能用例。按 CSV token 分别生成均匀分布 [-5,5] 与正态分布 N(0,1)，保持给定种子；`RANDOM_DIAG_IMAG` 保留非零输入对角虚部。边界用例实际为 29 条，以 CSV 为准。

| 测试类别 | 覆盖重点 |
| --- | --- |
| 任务用例 | 完整保留任务 CSV、生成规则和种子；性能用例也检查数值 |
| 接口 | handle 优先级、n<0/n=0、空指针、非法 uplo、0/INT_MIN 步长及字节跨度 |
| 数学语义 | UPPER/LOWER、一般复数 alpha、alpha=0/1、零列、正负步长及对角虚部污染 |
| 数值边界 | 极值、Inf/NaN、相消、subnormal、正负零，对角与非对角分别检查 |
| 分派边界 | n=63/64、256/257、511/512、896/897、1024/1025、2048/2049；8 列尾组与多条带 |
| 资源与状态 | UB 导致的 P/B 切换、尾块、输入输出有效边界、连续调用和空核退出 |

GTest 接入 arch22 构建，记录实际执行的用例及结果，跳过或未生成测试目标不作为通过。保留已通过且未变化路径的验证记录，针对实现变化及正式任务新增要求补测。

性能按任务书 §3.3 在 910B3、CANN 9.1.0 环境使用 `msprof op` 的 kernel task duration：每个 case 预热 5 次，再连续采集 10 次有效样本取算术平均，覆盖一次 API 的全部 kernel。以任务原始 `gpu_baseline.csv` 的 `gpu_ms`（GPU kernel-only）为基线，要求 `T_GPU/T_NPU≥0.8`，即 `T_NPU≤1.25·T_GPU`。事件计时仅供交叉验证，不能作为正式判据。

任务书 §3.4 内存要求为“不涉及”。已有 sanitizer 插桩检查作为开发验证资料保留，新增内存访问路径按需补充；正式报告如实填写“不涉及”，不将抽样记录描述成全量内存验收。性能使用普通构建。

## 兼容性分析

沿用 ops-blas 公共 API、架构分发及同族 CMake，Host、kernel 和 tiling 成套构建。测试注册在既有 Chpr2 arch22 目录；公共头、arch35 接口及调用方式保持兼容。

资源分派、预算公式和共享 tile 常量同步维护，每个特化具备通用回退及边界用例。源码遵循仓库许可证与注释规范；设计文档、源码和自验证报告按社区任务流程分别交付。

# 参考资料

1. [ops-blas 公共接口与算子实现](https://atomgit.com/cann/ops-blas)。
2. [社区任务设计模板](https://atomgit.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
3. [Netlib CHPR2](https://www.netlib.org/blas/chpr2.f)。
4. 2026 年 11 月社区任务「aclblasChpr2 算子开发（A2A3）」：`aclblasChpr2_AtlasA2_A3_task_doc.md` 及附带 `test_cases/`。
