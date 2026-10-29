# aclblasCgemvBatched 算子设计文档（Ascend 950PR）

## 1 需求背景（required）

### 1.1 需求来源

CANN 社区 `aclblasCgemvBatched` 算子开发任务。基于 Ascend C 和 ops-blas 的 kernel
直调框架，面向 Ascend 950PR、CANN 9.1.0 补齐单精度复数批量矩阵向量乘能力。
接口对标 cuBLAS `cublasCgemvBatched`，精度 golden 采用 Netlib BLAS 3.10.0 的
`cblas_cgemv`，对每个 batch 独立计算。

### 1.2 背景与功能

对每个 `0 ≤ b < batchCount`，计算：

```text
y_b = alpha * op(A_b) * x_b + beta * y_b
```

所有 batch 共享形状、前导维、步长、转置方式和复数标量，各矩阵和向量通过 Device
指针数组传入，可以位于不同地址。A 为列主序，y 原地更新，不做维度广播。

| trans | op(A) | A 原始形状 | x 逻辑长度 | y 逻辑长度 |
| --- | --- | --- | ---: | ---: |
| ACLBLAS_OP_N | A | m×n | n | m |
| ACLBLAS_OP_T | A 的转置 | m×n | m | n |
| ACLBLAS_OP_C | A 的共轭转置 | m×n | m | n |

### 1.3 现状与工程范围

仓内既有 `aclblasCgemvBatched` 单指针声明无法表达任务要求的 Device 指针数组接口。
本设计统一修正公共声明、arch22 定义和调用方，在 `blas/gemv_batched/arch35/` 实现
950PR 路径，不增加平台私有重载。测试沿用 `test/gemv_batched/cgemv_batched/`
的 GTest 与 CSV 框架。

批量处理融合在一次 arch35 计算 kernel 调用中，避免 Host 逐 batch 下发造成小矩阵
启动开销累积。计算使用 AIV 向量能力，保持 complex64 的两个 FP32 分量精度。

## 2 需求分析（required）

### 2.1 接口与参数约束

```cpp
aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const xarray[], int incx, const aclblasComplex* beta,
    aclblasComplex* const yarray[], int incy, int batchCount);
```

| 参数 | 位置与数据类型 | 约束 |
| --- | --- | --- |
| handle | Host 句柄 | 非空，使用其绑定的 stream |
| trans | Host 枚举 | 只接受 N、T、C |
| m、n | Host int | 均大于 0，表示原始 A 的行数和列数 |
| alpha、beta | Host complex64 标量指针 | 均非空，所有 batch 共用 |
| Aarray | Device 指针数组 | 长度为 batchCount，每个元素非空，指向列主序矩阵 |
| lda | Host int | lda≥m，与 trans 无关 |
| xarray、yarray | Device 指针数组 | 长度为 batchCount，每个元素非空，指向相应 Device 向量 |
| incx、incy | Host int | 支持正负步长，拒绝 0 和 INT_MIN |
| batchCount | Host int | 大于 0 |

complex64 按 `real, imag` 交错存储，两个分量均为 float32。公共 ABI 只要求其本身的
对齐，不能额外假定所有 Device 地址均为 8 字节或 32 字节对齐。

### 2.2 需求拆解

1. 实现 N/T/C、正负步长、非方阵、lda padding、复数标量和 y 原地输出。
2. 按固定顺序校验 Host 参数，在任何计算前完成三个 Device 指针数组的全部元素预检。
3. 在 batch 和输出维度并行，使用连续搬运、UB 数据复用与双缓冲降低访存开销。
4. 保持每个输出的固定 FP32 累加次序，不使用跨核浮点原子累加。
5. 只更新 y 的逻辑元素；A、x、指针数组、padding 和向量步长空洞保持不变。
6. 原样覆盖任务提供的 1200 条 CSV 和 200 条性能基线，补充边界、确定性和内存保护测试。

### 2.3 支持硬件与限制

| 项目 | 设计范围 |
| --- | --- |
| 目标硬件 | Ascend 950PR，arch35 |
| 软件环境 | CANN 9.1.0 |
| 支持类型 | COMPLEX64 输入、输出与标量，FP32 分量计算 |
| 布局 | A 列主序；x/y 支持 incx/incy；不支持其他 Tensor stride 或广播 |
| 形状 | m/n/batchCount 为运行时参数；不涉及图编译动态 shape |
| 零维、零 batch | INVALID_VALUE，不启动 kernel，不写 y |
| 输出别名 | 各 batch 的 y 存储互不重叠，且不能覆盖仍需读取的 A/x |
| 生命周期 | 调用方保证 Device 数据、指针数组及分配容量有效，直到相关 stream 工作完成 |
| 其他架构 | arch22 做公共签名兼容迁移，其验证与 950PR 分开记录 |

## 3 详细设计（required）

### 3.1 数学展开与地址计算

复数乘法采用 `(a+bi)(c+di)=(ac-bd)+(ad+bc)i`。C 模式只对 A 的虚部取反，
不对 x 或标量取共轭。矩阵元素地址偏移为 `uint64_t(col)*lda+row`。
逻辑长度为 L、逻辑位置为 i 的向量元素偏移为：

```text
offset(i,L,inc) = int64_t(i) * inc                   inc > 0
                  int64_t(L-1-i) * (-int64_t(inc))   inc < 0
```

向量物理跨度为 `1+(L-1)*abs(int64_t(inc))`。取负、乘法和任务数计算先提升到
64 位，防止 int 溢出后再转换。数据跨度还需检查乘以元素字节数后的溢出：

```text
maxElements = SIZE_MAX / sizeof(aclblasComplex)
矩阵：length=n，step=lda，tail=m
向量：length=L，step=abs(int64_t(inc))，tail=1
要求 tail <= maxElements 且 length-1 <= (maxElements-tail)/step
```

矩阵按最后一个逻辑元素检查，不把最后一列未访问的尾部 padding 计入必要跨度。
指针表大小同样在分配前检查乘法溢出。跨度检查安排在固定 Host 校验之后、D2H 之前，
失败返回 INVALID_VALUE；它不替代调用方对实际分配容量和有效 Device 地址的保证。

### 3.2 Host 侧校验与 stream

Host 按任务书规定的优先级检查，状态名统一带 `ACLBLAS_STATUS_` 前缀：

| 顺序 | 条件 | 返回值 |
| ---: | --- | --- |
| 1 | handle 为空 | HANDLE_IS_NULLPTR |
| 2 | m≤0 或 n≤0 | INVALID_VALUE |
| 3 | lda<m | INVALID_VALUE |
| 4 | incx/incy 为 0 或 INT_MIN，或 batchCount≤0 | INVALID_VALUE |
| 5 | Aarray/xarray/yarray 任一外层指针为空 | INVALID_VALUE |
| 6 | alpha/beta 任一指针为空 | INVALID_VALUE |
| 7 | trans 非 N/T/C | INVALID_ENUM |

三个 Device 指针数组不能由 Host 直接解引用。预检流程为：

1. 通过 `aclrtMallocHost` 分配完整列表所需的锁页暂存；分配失败返回 ALLOC_FAILED。
2. 在 handle 绑定的 stream 上依次提交三个完整列表的 `aclrtMemcpyAsync` D2H 拷贝。
3. 同步该 stream，再读取 Host 暂存；拷贝或同步错误返回 EXECUTION_FAILED。
   即使后续拷贝提交失败，也需等待已提交的拷贝完成后再释放暂存。
4. 遍历所有 batch 的 A/x/y 地址，任一内层指针为空均返回 INVALID_VALUE；此时尚未
   启动计算，因此不会发生前几个 batch 已写 y、后一个 batch 才报错的部分更新。

内层指针检查不因 alpha=0 而省略，也不缓存前次调用的列表来跳过后续预检。校验后，
alpha/beta 的值复制进 tiling 参数，计算提交到同一 stream。kernel 从 Device
指针表读取每个 batch 的地址，调用方不得在预检后并发改写列表。

接口包含指针预检同步，计算提交保持异步；调用方读取最终结果前仍需同步 stream。
Host 暂存的资源管理覆盖所有错误路径，无计算用的全局部分和 workspace。

### 3.3 标量分支与精度语义

所有参数和指针数组校验成功后，再进入以下分支：

| 条件 | 行为 |
| --- | --- |
| alpha=0、beta=1 | 返回成功，不写 y |
| alpha=0、beta=0 | 将逻辑 y 写零，不读 A/x/y 数据 |
| alpha=0、其他 beta | 只计算 beta*y，不读 A/x 数据 |
| alpha 非零、beta=0 | 计算乘积，不读旧 y |
| 其他 | 完整 GEMV |

N 模式沿列方向按 Netlib 的更新次序计算：

```text
acc = beta*y                  # beta=0 置零，beta=1 读取原值
for col = 0 .. n-1:
    temp = alpha*x[col]
    acc += A(row,col)*temp
```

T/C 模式沿行方向计算单一复数累加器，再执行 `alpha*sum + beta*y`。
跨搬运块继续更新同一个累加器，块边界不重置、不合并独立部分和。SIMD lane 对应
不同输出，而非同一输出的不同归约项；SIMT 回退路径保持相同的递推顺序。

复数乘法保留四个 FP32 乘积及加减的舍入边界，控制编译器 FMA 收缩，避免重排导致
抵消误差或非有限分类变化。例如 T/C 的 alpha=1 不能无条件省略末尾复乘：
当 sum 实部溢出为 Inf 时，`0*Inf` 仍可能影响虚部分量。也不能仅因 x=0 就省略
与非有限 A 的乘法。特殊值行为纳入 Netlib golden 回归，而非只用实数恒等式推导。

### 3.4 分核与 tiling 策略

定义输出长度 `O=(trans==N ? m : n)`，归约长度 `R=(trans==N ? n : m)`。
对输出块宽 B，`tilesPerBatch=ceil(O/B)`，`totalTiles=batchCount*tilesPerBatch`。
使用平台查询的 AIV 核数，`blockDim=min(totalTiles, aivCoreCount)`；核数为零返回
INTERNAL_ERROR。每个核按 `tileId=blockIdx+k*blockDim` 遍历工作：

```text
batch = tileId / tilesPerBatch
outputBegin = (tileId % tilesPerBatch) * B
outputCount = min(B, O-outputBegin)
```

每个输出仅由一个计算单元负责，无跨核同步或浮点 atomic。将 batch 与输出块共同
分核，改善单 batch 窄矩阵的核利用率；不拆分归约维到不同核。

| 计算路径 | 适用条件 | 分块与搬运方案 |
| --- | --- | --- |
| N 连续 SIMD | incx=incy=1 | 物理块为 128 输出行×64 归约列；同列连续行 DMA，列块内复用 x |
| N 窄输出 SIMD | incx=incy=1，输出块≤64 | 物理块为 64 输出行×128 归约列，减少长归约的搬运与同步次数 |
| T/C SIMD | incy=1，incx∈{±1,±2} | 64 输出列×96 归约行；连续搬入每列，在 UB 转置为按行排列的实虚平面 |
| 通用 SIMT | 其他合法步长或不适用上述配置的形状 | 一个线程负责一个输出，按逻辑顺序执行完整归约，覆盖任意合法正负步长 |
| 仅缩放 y | alpha=0 且 beta≠1 | 按逻辑输出分块，不搬运 A/x |

N 的输出块宽在 16/32/64/128 中选取，T/C 在 16/32/64 中选取；物理 UB 布局由
对应配置固定，逻辑尾块使用 mask。选择同时考虑 `totalTiles`、调度波数
`ceil(totalTiles/aivCoreCount)`、每块向量组数和搬运次数，避免为填满核心反而增加
整轮归约。小矩阵、高 batchCount 场景同时比较 SIMT 融合路径，减少每 batch 的
流水准备成本。具体分段阈值通过参数化基准确定，计算路径按形状、步长和平台能力
选择，不依赖用例编号、A/x/y 的数值或测试环境；alpha/beta 仅用于规定的标量语义分支。

Tiling 数据包含 m/n/lda、batchCount、trans、int64 的 incx/incy、alpha/beta
四个 float 分量、标量零值标志、输出块宽及计算路径标识。路径标识区分 N 宽块、
N 窄块、T/C UB 转置、通用 SIMT 与仅缩放；T/C 共轭标志独立保留。Host 与 kernel
共享同一定义，启动参数、缓冲配置和文档保持一致。

### 3.5 UB 预算与数据流水

各路径使用不同的局部布局，预算均包含双缓冲和对齐预留；下表按最大物理块计算，
单位为字节：

| 路径 | A 双缓冲 | x 双缓冲 | 累加/输出区 | 转置区 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N：128×64 | 131072 | 2048 | 1024 | 0 | 134144 |
| N：64×128 | 131072 | 3072 | 512 | 0 | 134656 |
| T/C：64×96 | 102400 | 4096 | 1024 | 49152 | 156672 |

N 的 A 双缓冲为 `2*输出物理块*归约物理块*8`，x 每个 slot 除逻辑数据外预留
512 字节广播加载空间。T/C 每列使用 200 个 float 的 pitch（96 个复数与
32 字节 padding），A 为 `2*64*200*4`；x 覆盖最大绝对步长 2 并保留广播空间。
转置区为 `64*96*8`。这些配置限制在 192 KiB 的设计预算内，仍需结合平台 UB
查询、编译器占用及 API 临时空间核对；容量不满足时选择更小配置或通用路径。

流水分为 CopyIn、Compute、CopyOut：

1. 用两个 A/x slot 交替搬入归约块，通过 MTE2→V 事件等待数据就绪。
2. N 路径在同一列上并行更新多个输出行；T/C 使用 `TransDataTo5HD` 将 UB 中
   `[输出列][归约行实虚分量]` 变成 `[归约行实虚分量][输出列]`，以连续向量加载
   代替逐行离散 gather。转置只改变数据布局，不改变 FP32 累加次序。
3. 用 V→MTE2 事件确保计算完成后再覆盖输入 slot；转置区写入、读取及复用之间
   使用向量流水依赖。负 incx 从连续 span 的末端按逻辑顺序读取。
4. 最后一个归约块后写出逻辑 y，等待搬出完成再复用累加区或切换 batch。

矩阵 DMA 仅搬运有效行段；非单位步长的 x 可搬入逻辑首尾之间已分配的连续 span，
跨度内空洞不参与计算。GM 访问不超出所需物理跨度，UB 对齐填充不扩大 GM 范围，
写回仅覆盖逻辑 y 元素。转置尾列只引用已初始化的合法 UB 行，失效 lane 由 mask
屏蔽；广播加载的完整范围必须在已分配 UB 内。通用路径按 float 分量访问，覆盖
4 字节对齐地址、大步长和离散输出。

### 3.6 性能分析与优化方向

每个矩阵元素包含 8 字节数据，复数乘加约 8 个实数运算；大尺寸 GEMV 的主要数据量
为 A 的读取。设计优先保证列主序连续搬运、x 块复用和双缓冲重叠，不额外写出部分和。
T/C 的 UB 转置增加局部处理成本，但能将后续逐行计算改为连续加载，需结合归约长度
和输出宽度选择。长归约通过加大搬运块、预取和保持计算次序的循环展开降低传输等待
与循环控制开销。

小矩阵主要关注启动和调度波数，整个 batch 共用一次计算 kernel；大 batch 的小输出
避免机械地每 batch 使用一个完整 SIMD 计算块。非单位步长同样纳入性能目标，例如
任务性能集中的 incx=2 路径使用连续 span 搬运和 UB 内索引，不能排除在性能验收之外。

### 3.7 工程集成与兼容性分析

| 文件或目录 | 职责 |
| --- | --- |
| `include/cann_ops_blas.h` | 统一 Device 指针数组 API |
| `blas/gemv_batched/common/` | Host 顺序校验、跨度检查与完整指针列表预检 |
| `blas/gemv_batched/arch35/` | tiling、入口、SIMD/SIMT 计算和局部缓冲管理 |
| `blas/gemv_batched/arch22/` | 公共签名迁移与既有 Cgemv 兼容调用 |
| `test/gemv_batched/cgemv_batched/` | CSV、Netlib golden、GTest、复现脚本及接口迁移测试 |

arch22 采用相同的完整列表预检，再逐 batch 调用既有 `aclblasCgemv`；若既有路径
仅支持正步长，则以 Device gather/scatter 适配负步长，临时缓冲释放前等待绑定
stream 完成。同步更新所有旧单指针调用方与示例，不把指针表地址当成矩阵地址。
此次接口修正需要调用方重新编译，不保留错误签名的重载。arch22 的编译、运行和性能
结论与 arch35 分开记录；950PR 验收数据不能替代其他平台验证。

## 4 可维可测分析

### 4.1 精度与确定性标准

实部、虚部分别按 FLOAT32 判定。有限值逐元素满足：

```text
abs(actual-golden) <= 2^-13 + 2^-13*abs(golden)
```

每个分量的 matched_ratio≥0.99，且误差上限满足任务规定的 `1e-2 or 32*ULP`。
MERE/MARE 可用于分析，不替代混合容差标准。非有限分量单独统计：NaN/NaN、同号
Inf/Inf、NaN/Inf 按任务规则等效；有限/非有限和异号 Inf 不等效。非有限分量全部
匹配后，才结合有限分量指标判断通过。

同一输入和初始 y 复位后重复执行，比较完整输出字节验证确定性。每次计算逐 batch
执行 Netlib golden，不能把性能 replay 后的原地输出与单次调用结果混用。

### 4.2 测试覆盖与内存保护

| 类别 | 覆盖方法 |
| --- | --- |
| 任务原始用例 | 保留全部 1200 条 CSV，其中包含 200 条性能用例；检查唯一覆盖、返回码与完整输出 |
| 形状与布局 | N/T/C、非方阵、1、小质数、2 的幂及 ±1、输出块和归约块边界、lda=m/m+1/m+8 |
| 数值分布 | 实虚部独立生成；均匀 [-5,5] 与正态分布各半，另含零、±1、纯虚数、大值、Inf/NaN |
| 步长 | ±1/±2/±3、±1000；n 或 m 取 1 覆盖 INT_MAX 与 INT_MIN+1 的成功路径；0/INT_MIN 拒绝 |
| 负向契约 | 空句柄、非法枚举、零/负维度与 batch、非法 lda、空标量、外层与后一个 batch 内层空指针 |
| 校验顺序 | 多个错误同时存在、alpha=0 仍检查内层指针、失败调用不写任何 batch 输出 |
| 地址与保护 | 4 字节对齐、64 位索引、跨度溢出、A/x 不变、指针表不变、padding/步长空洞不变 |
| 调度边界 | 各输出块宽、一个核的多个 batch、尾波、负步长和转置尾块组合 |

采用前后 guard 检查越界写；所有 batch 使用独立 Device 数据，不以重复指针降低
实际计算规模。大矩阵逐 batch 构造和上传 Host 数据，验证时再逐 batch 重建 golden，
控制 Host 内存峰值。任务内存验收项标注“不涉及”；另外使用 mssanitizer 插桩构建
检查典型计算、步长、尾块、转置及负向场景，不把未产生 kernel 的负向返回当作计算
路径已完成内存检查。

收集器检查进程退出码、GTest XML、实际用例名单和非有限值统计，缺失、重复、跳过
或日志截断均不能计作通过。测试原始日志和结果随验收材料交付，不进入生产源码仓。

### 4.3 性能标准与采集方法

| 验收项目 | 标准及采集方式 | 来源 |
| --- | --- | --- |
| 精度 | COMPLEX64 的实虚部独立满足混合容差与非有限判定 | 任务书 §3.2 |
| 正式性能 | msprof op 自带 5 次 warmup 后，10 个有效 Task Duration(us) 取算术均值；每个 case 的 GPU_us/NPU_us≥0.4 | 任务书 §3.3 |
| 补充性能 | 独立执行 5 次 warmup、51 个有效样本并报告均值与实际 GPU/NPU 比值；优化目标取 GPU/NPU≥1.0 | 任务书 §7 及随附 README |
| 确定性 | 相同输入和初始 y 两次执行，输出逐位一致 | 任务书 §2.5 |

任务书 §3.3 的专用性能判据与 §7 的倍率、样本数表述不同。因此正式 10 样本与补充
51 样本分别采集、分别列明结果，以更严格的 ≥GPU 作为优化目标，最终验收口径由
设计评审确认。不能把达到 0.4×GPU 描述为达到 1.0×GPU。

性能集逐项匹配 200 条 GPU 基线的 trans/m/n/lda/incx/incy/batchCount、标量与
数据填充参数。每个 case 独立判定，不以平均加速比替代失败项。下表列出典型预算，
均为任务 GPU 数据及推导上限：

| 用例参数 | GPU 基线 us | §3.3 的 NPU 上限 us |
| --- | ---: | ---: |
| N，256×256，batch=8 | 13.306 | 33.265 |
| N，1024×2048，lda=2048，batch=16 | 89.696 | 224.240 |
| T，4096×4096，batch=32 | 1352.200 | 3380.500 |
| C，8192×1024，lda=8192，incx=2，batch=64 | 1339.830 | 3349.575 |
| N，16×16，batch=256 | 3.929 | 9.8225 |

采集保留每次原始样本、工具版本、运行状态与代码 commit；只将 SUCCESS 且工具
执行完整的调用计为有效样本，异常须记录原因并重新完整采集。校验 warmup、有效
样本数、kernel 名称与记录顺序，并检测额外 kernel。若一个 API 包含多个计算 task，
须汇总该调用的全部计算 task，不能只选最快的一条。

Host 指针表预检同步不属于 Device kernel Task Duration，报告明确区分；GTest
整体耗时也不能代替 kernel 时间。正式数据使用 msprof op，普通 msprof 补充数据
单独标识。精度、性能和内存材料对应同一个固定代码版本。

### 4.4 交付与维护

设计文档按社区模板提交至本仓对应任务的 `docs/design.md`；评审通过且合入后，
按任务流程提交自验步骤、精度与性能报告、原始日志及私仓验收 commit。代码侧保留
可复现脚本、参数说明和测试数据定义，机器相关结果与临时调优材料不入生产仓。
后续修改公共签名、tiling 路由、UB 配置或数值次序时，同步更新设计与对应验证范围。
