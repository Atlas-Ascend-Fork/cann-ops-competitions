# aclblasCher2 算子设计文档（Ascend 950PR）

| 文档版本 | 作者/团队 | 目标平台 | 工具链 |
| --- | --- | --- | --- |
| V1.0 | BaoBao2076 | Ascend 950PR（arch35） | CANN 9.1.0 |

# 一、需求背景

## 1.1 需求来源

本设计对应 [2026 年 10 月社区任务 aclblasCher2 算子开发（950）](https://www.hiascend.com/activities/task-center/details/a02b1193767b4b26a31e7a198375c25c?menu=guide)。基于 [ops-blas](https://gitcode.com/cann/ops-blas) 的句柄式 BLAS 框架，使用 Ascend C kernel 直调实现单精度复数 Hermitian 秩 2 更新。

接口与参数语义对齐 cuBLAS `cublasCher2`。精度 Golden 使用 Netlib BLAS 3.10.0 的 `cblas_cher2`，设置 `CblasColMajor`。本任务补齐 `arch35` 能力，不涉及历史 TBE 算子迁移。

## 1.2 算子功能

计算公式为：

~~~text
A = A + alpha * x * y^H + conj(alpha) * y * x^H
~~~

`x`、`y` 是逻辑长度为 n 的复数向量，`alpha` 是复数标量，A 是列主序 n×n Hermitian 矩阵。上标 H 表示共轭转置。输出原地覆盖 A 的指定三角。

只引用和更新 `uplo` 指定的上三角或下三角。另一三角由共轭对称性隐含，不补写。对角输入虚部不参与数值计算；非 no-op 调用后，对角虚部设为 0。未引用三角和 `lda` padding 保持原样。

# 二、需求分析

## 2.1 公共接口

在 `include/cann_ops_blas.h` 新增公共声明，复用仓库句柄、复数类型和状态码。

~~~cpp
aclblasStatus_t aclblasCher2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy,
    aclblasComplex* A, const int lda);
~~~

| 参数 | 所在位置、类型及约束 |
| --- | --- |
| handle | Host 句柄；使用其绑定的 stream |
| uplo | Host 枚举；`ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | Host int；n≥0，指定矩阵阶数和向量逻辑长度 |
| alpha | Host `const aclblasComplex*`；非空，直接在 Host 解引用 |
| x | Device 只读 complex64 向量；n>0 时物理长度至少 `1+(n-1)*abs(incx)` |
| incx | Host int；支持正负步长，排除 0 和 `INT_MIN` |
| y | Device 只读 complex64 向量；n>0 时物理长度至少 `1+(n-1)*abs(incy)` |
| incy | Host int；支持正负步长，排除 0 和 `INT_MIN` |
| A | Device complex64 矩阵；物理存储 `lda*n` 个元素，输入旧值并原地输出 |
| lda | Host int；`lda >= max(1,n)` |

`aclblasComplex` 采用 `include/cann_ops_blas_common.h` 的定义，实部和虚部各为 float32。步长以复数元素为单位。alpha 不支持 Device 指针识别或同步读回。

## 2.2 校验与边界语义

| 条件 | 返回值或行为 |
| --- | --- |
| handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n<0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| uplo 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| alpha 为空、incx/incy 为 0 或 `INT_MIN`、lda 非法 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 结构参数合法，n=0 或 alpha=(0,0) | `ACLBLAS_STATUS_SUCCESS`；不下发 kernel，不访问 x/y/A |
| 非 no-op，x/y/A 任一为空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 设备核数查询失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` |

校验顺序为：handle → n → uplo → 步长/lda/alpha 指针 → quick return → x/y/A 指针。结构参数错误优先于 quick return。合法 no-op 允许 x/y/A 为空，也不清零 A 的对角虚部。

步长 `INT_MIN` 按附件异常用例 `TC_ED_219`、`TC_ED_220` 拒绝。其他合法正负 int 步长按 64 位中间量寻址。4096 是优化路径边界，不是公共接口的 n 上限。

## 2.3 需求拆解与工程组织

| 模块/路径 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 公共 API 声明，供各产品线共用 |
| `blas/syr2/arch35/cher2_host.cpp` | 参数校验、平台查询、路径选择、stream 下发 |
| `blas/syr2/arch35/cher2_kernel.h` | Tiling 参数与内部启动接口 |
| `blas/syr2/arch35/cher2_aiv_kernel.cpp` | 连续向量的批量搬运和向量计算 |
| `blas/syr2/arch35/cher2_kernel.cpp` | 通用步长和较大 n 的 SIMT 计算 |
| `test/syr2/cher2/arch35/` | CSV、GTest、Netlib Golden、精度及性能检查 |
| `blas/syr2/README.md` | 公共接口、调用约束和产品支持说明 |

复用同族算子的构建和测试组织，不添加独立构建框架。测试日志、性能采样、临时脚本与构建产物独立保存，不进入产品源码。

# 三、详细设计

## 3.1 总体流程与执行路径

~~~mermaid
flowchart TD
    A[校验 handle 与结构参数] --> B{n 为零或 alpha 为零}
    B -->|是| C[返回成功]
    B -->|否| D[校验设备指针并查询核数]
    D --> E{incx 和 incy 为 1 且 n 不超过 4096}
    E -->|是| F[AIV 连续向量路径]
    E -->|否| G[SIMT 通用步长路径]
    F --> H[更新指定三角并清零对角虚部]
    G --> H
~~~

AIV 路径利用向量指令批量处理连续行。SIMT 路径由多个设备线程计算独立元素，直接支持间隔和反向索引。两条路径均在句柄 stream 上提交一个 kernel，不申请额外 Device workspace，不使用跨调用缓存。

| 条件 | 路径 | 设计原因 |
| --- | --- | --- |
| `incx=incy=1` 且 `1<=n<=4096` | AIV | x/y 可驻留 UB，矩阵按列连续搬运；复用行向量 |
| 任一合法非单位步长，或 n>4096 | SIMT | 按逻辑索引寻址，避免整向量驻留要求 |
| `lda>n` | 仍按上述条件选择 | 前导维影响列地址，不改变单列有效行的连续性 |

路径选择只依赖参数和资源约束，不依赖用例编号、输入值或 GPU 标杆值。

## 3.2 索引与三角边界

矩阵元素地址为 `uint64_t(col)*lda+row`。UPPER 的有效行范围是 `[0,col]`，LOWER 是 `[col,n-1]`。

~~~text
VectorOffset(i, n, inc):
    inc > 0: uint64(i) * int64(inc)
    inc < 0: uint64(n-1-i) * (-int64(inc))
~~~

x、y 的传入地址指向各自物理存储起点，负步长时逻辑首元素位于高地址端。元素偏移及复数到字节的换算采用 64 位计算。调用方提供满足 shape、步长和前导维要求的有效存储。

搬运长度使用当前列的有效行数；只在 UB 内补齐对齐和向量尾块。GM 读取和写回不扩展到未引用三角、padding 或向量物理末尾之外。

## 3.3 数值计算顺序

每列生成两个复数系数：

~~~text
p = alpha * conj(y[col])
q = conj(alpha) * conj(x[col])

非对角：A[row,col] = (A[row,col] + x[row]*p) + y[row]*q
对角实部：A[col,col].real += real(x[col]*p) + real(y[col]*q)
对角虚部：A[col,col].imag = 0
~~~

复数乘法按 FP32 实虚分量展开：`(ar*br-ai*bi, ar*bi+ai*br)`。保留每个乘积和加减的舍入顺序，禁止编译器随意把乘法和加法合并为 FMA，以维持溢出、Inf/NaN 和抵消场景的语义。

当 `x[col]` 与 `y[col]` 均为复数零时，跳过该列数值更新，仍清零对角虚部。该分支对应 Netlib 的零列行为，避免以无效的 `0*Inf` 改写原值。AIV 和 SIMT 共用相同的数学规则；对角不使用原虚部参与运算。

## 3.4 Host 分核与 Tiling

从平台查询 AIV 核数，不固定核数常量。AIV 使用 `min(n, floor(coreCount/8)*8)` 个 block；SIMT 使用 `min(n,coreCount)` 个 block，每个 block 配置 256 个线程。目标为 Ascend 950PR 的有效平台配置。

第 k 个 block 处理 `k, k+blockCount, k+2*blockCount, ...` 列。循环分列分散三角长列与短列，避免将连续长列集中到少数核。每列及每个输出元素只有一个写入者，无原子累加和跨核归约。

Tiling 传递 n、lda、uplo、incx、incy、alpha 的实虚部及线程数。Host 仅处理元数据和 Host alpha，不同步读回设备输入。正常提交后异步返回；调用方在读取结果前同步该 stream。

## 3.5 AIV 数据布局与容量

UB 仅保留 x/y 的实部和虚部四个平面。列系数 p/q 在寄存器中生成，不再保存四个完整的系数平面。矩阵缓冲保存每列的交错复数数据。

设 `capacity=ceil(n/64)*64+64`，每个平面按 float32 分配。额外 64 个元素用于向量尾块和对齐余量。算法 UB 工作集预算为 200704 字节，包含这些余量，不将该预算当作芯片总 UB 容量。

~~~text
availableColumns = floor((floor(200704/(capacity*4))-4)/2)
maxColumns = ceil(n/blockCount)
bufferCount = 1, 当 maxColumns<=availableColumns 且 maxColumns<=16
bufferCount = 2, 其他情况
batch = min(16, floor(availableColumns/bufferCount))
UB 字节数 = (4 + 2*bufferCount*batch) * capacity * 4 <= 200704
~~~

单批容纳本核全部列时使用单缓冲，减少事件与切换成本。多批时使用双缓冲，在当前批计算期间预取下一批。初始化阶段借用矩阵槽位暂存 x/y 的交错数据，拆分完成后才复用这些槽位。

LOWER 按 `blockIdx%8` 平移向量缓存起点。存在后续列时，核数为 8 的倍数，使各列行视图保持 32 字节对齐。每批只进入一次向量计算作用域，在内部循环处理多列。对角专用计算只在含对角元素的 64 元素寄存器块执行。

## 3.6 搬运与同步

每个矩阵缓冲对象绑定数据地址以及 ready、computed、released 三类事件。预取、计算和写回传递同一个对象，避免缓冲序号与事件序号分离。

| 依赖 | 事件 | 等待位置 |
| --- | --- | --- |
| GM 搬入完成后计算 | `MTE2_V` | 向量核消费该批前 |
| 向量计算完成后写回 | `V_MTE3` | MTE3 搬出前 |
| 写回完成后复用缓冲 | `MTE3_MTE2` | 同一缓冲下一次搬入前 |
| 初始化暂存区拆分完成后复用 | `V_MTE2` | 后续矩阵搬入前 |

事件从 TPipe 分配并成对释放。首批不等待不存在的复用事件，末批不发出无人消费的复用事件。kernel 收尾完成流水同步后释放事件；不在每列计算中插入全流水屏障。

## 3.7 SIMT 通用路径

每个 block 循环处理分配到的列，线程沿有效行区间以 256 为步长遍历。x/y 按 64 位步长公式读取，A 沿列连续访问。UPPER/LOWER 在模板实例中分别确定行界限，省去逐元素三角判定。

SIMT 保留合法稀疏步长和大 n 的功能覆盖，其性能需单独按用例验证。若性能瓶颈集中在这一类输入，进一步分块或增加专用路径须同时验证索引、特殊值和三角保护；不以缩小接口范围替代优化。

## 3.8 支持范围与设计取舍

本设计适配 Ascend 950PR、CANN 9.1.0。公共签名供其他产品线复用，但本设计不声明已完成 A2/A3 适配。类型仅为 complex64，不引入低精度计算或隐式类型转换。

A 是原地输出，不返回视图。不涉及广播，也不扩展 incx/incy/lda 之外的 Tensor 非连续语义。n 为运行时参数，不引入图模式动态 shape 接口。选择向量/SIMT 而非 Cube 分解，旨在避免秩 2 更新的额外格式转换、中间矩阵和跨核同步。

# 四、可维可测分析

## 4.1 测试覆盖

采用仓库 CSV 驱动的 C++ GTest 框架。原附件包含 1000 条精度用例及 200 条性能用例，保留原始参数和 GPU baseline 的逐条对应关系。

| 分类 | 覆盖内容 |
| --- | --- |
| 接口 | 空 handle、非法 uplo、负 n、空 alpha、零步长、`INT_MIN`、非法 lda、非 no-op 空设备指针 |
| Quick return | n=0、alpha=(0,0)、允许空 x/y/A；非法结构参数与 no-op 组合 |
| 尺寸 | 1、小质数、2 的幂及相邻值、非对齐尾块、大规模、4096 路径边界两侧 |
| 步长与存储 | 正负步长交叉组合、±1/±2/±3、紧凑及 padding lda、UPPER/LOWER |
| 数据 | 实虚独立采样，均匀/正态各 50%；零、交替值、大值、抵消、Inf/NaN |
| 不变量 | 未引用三角、padding、向量输入和保护区保持；对角虚部为 0 |
| 流水 | 单批/多批、单缓冲/双缓冲、首批/末批、异步 stream 和重复调用 |

均匀分布为 [-5,5]，正态分布的均值在 [-5,5]、标准差在 [0.1,2] 中选择。记录随机 seed 与实际参数。A 按 Hermitian 约束生成，另以非零或非有限对角虚部检查其不参与计算。

## 4.2 精度判据

Netlib BLAS 3.10.0 `cblas_cher2(CblasColMajor,...)` 生成 Golden。仅在指定三角内，将实部、虚部分开按 FLOAT32 标准比较。

| 项目 | 判据 |
| --- | --- |
| 有限值逐元素匹配 | `abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)` |
| 有限值匹配率 | 每个分量不低于 0.99 |
| 绝对误差上限 | 每个有限值元素误差不超过 `max(1e-2,32*ULP(golden))` |
| 非有限值等效 | NaN/NaN、同号 Inf/Inf、NaN/±Inf |
| 非有限值失败 | 有限/非有限混合，或正负 Inf 相反 |
| 非有限值统计 | 从有限值统计中剔除，单独计数并要求全部等效；无有限值时有限值项不适用 |

未引用三角、padding 和输入保护区使用哨兵逐字节核对，不套用浮点容差。哨兵不变只能证明未改写；未引用区域不读取还需结合地址边界审查与内存访问检测验证。对角虚部必须为零。性能用例也执行 Golden 比对，精度标准不随路径或规模放宽。

## 4.3 性能采样与优化验证

使用 **`msprof op`** 采集单算子 **Task Duration**，单位为 us。预热 5 次，不计入有效样本。正式记录 60 次有效采样的算术均值，同时列出同序列前 10 次均值：前者满足附件超过 50 次及任务书第 7 节超过 10 次的要求，后者对应第 3.3 节的 10 次口径。

仅成功执行的调用计入有效样本；执行异常单独记录。全部有效样本保留，不删除耗时尖峰。发生外部任务干扰时，将整组标为受干扰，另行在空闲设备复测，不将受干扰数据当作正式验收结论。

逐条对照 `gpu_baseline.csv` 的 uplo、n、incx、incy、lda 和数据类型。GPU 环境为 NVIDIA 驱动 550.54、CUDA 12.4。若 GPU 基线单位为 ms，则性能上限为：

~~~text
NPU 平均 Task Duration(us) <= GPU 基线(ms) * 2500
~~~

先完成一次全量基线，再按不达标用例所属路径和尺寸集中优化。修改后定向复测不达标项及相邻边界；各类问题处理完成后，再执行全量回归检查遗漏和退化。报告同时保留基线值、修改后值、阈值、全部采样和源码版本，定向结果不表述为全量通过。

设备测试每卡使用独立进程，通过 `ASCEND_RT_VISIBLE_DEVICES` 仅暴露一张卡，进程内使用逻辑设备 0。正式性能采样检查设备占用，避免与其他任务混跑。

## 4.4 资源检查、兼容性与交付

任务书的内存占用指标为“不涉及”。补充使用匹配工具链的 mssanitizer 插桩构建检查越界和同步问题，结合静态容量计算、事件配对和尾块审查验证实现。插桩性能不作为正式耗时。

公共 API 参数顺序、类型、状态码和句柄 stream 语义与仓库保持一致。工程评审对照同族 `syr2/arch35`，逐个说明新增文件用途，排除实验残留和重复框架。

本文件描述算法和验证方案，不作为全部用例达标证明。自测报告独立记录与待验收源码一致的精度、性能及资源检查结果。每次修改 kernel 后重新绑定证据版本，历史通过数不直接沿用到新版本。
