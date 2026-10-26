# aclblasCsymv A2/A3 算子设计

# 需求背景

## 需求来源

本设计对应 [aclblasCsymv A2/A3 社区任务](https://www.hiascend.com/activities/task-center/details/de583b4362aa48fab76510369d78d0c6)，为 ops-blas 增加 Atlas A2、Atlas A3 上的单精度复数对称矩阵-向量乘实现。

## 背景介绍

Csymv 是 Level-2 BLAS 算子。它只读取矩阵的上三角或下三角，再利用对称关系完成矩阵-向量乘法。

实现采用两种计算方式。小中矩阵让每个核独立算完若干输出行，避免核间交换部分和；大矩阵则复用三角块，一次读取同时计算两个对称方向，减少矩阵搬运。

# 需求分析

## 需求描述

计算 `y = alpha * A * x + beta * y`。输入输出类型为 COMPLEX64；支持上下三角、列主序、lda padding、正负向量步长和 Host/Device 标量。

## 需求拆解

1. 复用公开的 `aclblasCsymv` 接口，在 handle 绑定的 stream 上直调 arch22 kernel。
2. 实现参数检查，以及 n=0、alpha=0、beta=0 等特殊情况。
3. 根据矩阵规模、步长和已有 workspace 选择计算路径，处理尾块和核间归约。
4. 提供 1000 条精度用例、200 条性能输入及补充接口与边界测试，测量全部性能用例，并对任务书指定的五项断言上限。

# 详细设计

## 算子分析

### 数学公式

```text
y[i] = alpha * sum(A_sym[i,j] * x[j], j=0..n-1) + beta * y[i]
A_sym = A_sym^T
```

A_sym 由 `uplo` 指定的三角及其转置构成。复数乘法为：

```text
(a+bi)(c+di) = (ac-bd) + (ad+bc)i
```

### 支持数据类型

A、x、y、alpha、beta 均使用 `aclblasComplex`（COMPLEX64），实部和虚部各占一个 float32。

### 支持形状

A 的有效形状为 n×n，按列主序存储在 lda×n 数组中；x、y 的逻辑长度为 n。要求 `n>=0`、`lda>=max(1,n)`，incx、incy 均为非零整数。

n>0 时，向量至少需要 `1+(n-1)*abs(inc)` 个元素的存储空间。传入指针指向存储基址；步长为负时，从 `(n-1)*abs(inc)` 处开始遍历。

## 算子实现

### 实现方案

Host 检查参数和标量位置，查询实际可用的 AIV 数量，然后选择 kernel 和切分参数。设备函数按 handle 绑定的设备缓存；当前设备改变时重新查询。

小中矩阵按输出行分工：一个 AIV 负责若干行，遍历完整 K 维后直接写回。大矩阵按三角块分工：先计算两个对称方向的部分和，写入 workspace，再归约成最终输出。非单位 incx、超大 lda 等输入使用 64 位地址计算的通用路径。

#### Host 侧设计

公共头文件中已有以下声明，A2/A3 直接复用。

```cpp
aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

| 参数 | 含义 |
|---|---|
| handle | 携带执行 stream 的上下文句柄 |
| uplo | `ACLBLAS_UPPER` 读取上三角，`ACLBLAS_LOWER` 读取下三角 |
| n | A 的行列数及 x、y 的逻辑长度，`n>=0` |
| alpha、beta | COMPLEX64 乘数，指向 Host 或 Device 标量 |
| A | Device 上的列主序矩阵，存储大小为 `lda*n` 个复数 |
| lda | 矩阵前导维度，`lda>=max(1,n)` |
| x、y | Device 上的输入向量和原地更新的输出向量 |
| incx、incy | 以复数元素为单位的非零步长，支持正负值 |

正常计算在 handle 绑定的 stream 上异步执行，调用方在读回结果前同步 stream。Host 标量随 tiling 参数传入；Device 标量由 kernel 读取。如果数据指针为空，Host 需要先在同一 stream 上读回 Device 标量并同步，判断该指针是否允许为空。

##### 参数与短路行为

- handle 为空时返回 `HANDLE_IS_NULLPTR`；n<0 返回 `INVALID_VALUE`；n=0 返回 `SUCCESS`。
- n>0 时检查 uplo、lda、incx、incy 及 alpha、beta 指针。非法 uplo 返回 `INVALID_ENUM`，其余非法参数返回 `INVALID_VALUE`。
- alpha=0 时不读 A、x；若同时 beta=1，则不读写 y。
- beta=0 时不读原 y；beta=1 时直接使用原 y，避免额外的零乘法影响 Inf 分量。

##### 分核策略

除 generic 外，各路径要求 `incx=1`、`16<=n<=8192`，且 `lda*sizeof(aclblasComplex)<=UINT32_MAX`。incy 仍可使用任意合法正负步长。满足这些条件后，按下表选择路径。

| 输入条件 | 路径 | 工作划分 | workspace |
|---|---|---|---|
| `16<=n<=240`，`n%16==0` | small | 每组 16 行，在核内完成输出 | 无 |
| `256<=n<=512`，`n%16==0` | adaptive / R16 owner | 按可用核数选择 R4、R8、R12 或 R16 输出矩形 | 无 |
| `528<=n<=2048`，`n%16==0` | wide owner | 连续输出按 16 行分组，以 R16/R32/R48 矩形计算 | 无 |
| `2176<=n<=4096`，`n%128==0`，现有 workspace 足够 | T128 large fused | 三角 tile 复用，核间归约 | `n²/16` 字节 |
| 其余 `2048<n<=8192`，`n%16==0` | tiled | 每组 16 行，独占完整输出 | 无 |
| `257<=n<=8192`，`n%16!=0` | tail | 逻辑范围搬运，屏蔽无效 lane | 无 |
| 其他合法输入 | generic | 按输出行划分，使用 64 位地址计算 | 无 |

Owner 使用实际可用的 AIV 数量分配输出。小矩阵以 4 行为单位选择输出矩形，最多 16 行；中矩阵的单个矩形最多 48 行，超出时拆为多个矩形。大矩阵按 64×64 微块的数量分配三角任务，使各核工作量接近。

##### 数据分块和内存优化策略

adaptive 路径每次处理 128 个 K 元素，wide 路径每次处理 64 个。两张矩阵原始数据缓冲交替预取；一张缓冲的向量读取全部结束后，MTE2 才能覆盖它。

T128 路径将指定三角划分为 128×128 的块，再按 64×64 微块计算。非对角块同时产生两个输出方向的部分和；对角块只计算有效三角。实部、虚部部分和分别连续存入 workspace，所需空间为：

```text
2 * n * (n / 128) * sizeof(float) = n² / 16 字节
```

该路径只使用 handle 已有的 workspace。指针为空或容量不足时，改用 tiled 路径。

##### Kernel 选择与参数传递

Host 根据 n、lda、incx、AIV 数量和 workspace 选择入口，并通过 tiling 数据传入切分参数。

#### Kernel 侧设计

复数按实部、虚部交错的 AoS 布局存储。直接读取区域通过 b32 转置拆开实部和虚部；镜像区域保留交错布局计算复乘。每次先形成 FP32 复数乘积，再按输出行和 K 方向的向量通道累加，最后做水平归约并应用 alpha、beta。

adaptive 从对齐地址读取 x，将计算宽度向 16 取整，并屏蔽无效 K 元素。wide 将独立输出行安排在 repeat 轴，复用两张乘积缓冲；x 的读取范围按该核实际访问的直接区域和镜像区域准备。

Owner 和 triangle 显式管理流水同步，其余路径使用编译器自动同步。Owner 使用 `AIV_ONLY` 入口。T128 的部分和计算与归约放在同一个 `MIX_AIV_1_0` kernel 中，采用同驻留调度；所有参与核通过 `SyncAll<true>` 同步后，再读取部分和并写回 y。

尾块只搬运逻辑范围内的数据。从 64 个 float 对齐的基址开始，按 mask 清理缓存尾部，并在 K 维归约前屏蔽无效乘积，避免 padding 中的 `0*Inf` 污染结果。输出只修改 y 的逻辑元素。矩阵-向量乘主体的计算量为 O(n²)。

## 支持硬件

| 产品 | 支持情况 | 已验证环境 |
|---|---|---|
| Atlas A2 | 支持 | Atlas A2（910B3），CANN 9.1.0 |
| Atlas A3 | 支持 | Ascend910_9391，CANN 9.1.0 |

## 算子约束限制

A、x、y 位于 Device，alpha、beta 可位于 Host 或 Device。调用方负责分配有效存储，并在读取输出前同步 stream。列主序元素地址为 `A[row+col*lda]`，lda padding 不参与计算。

handle 必须有效。n=0 时不访问 alpha、beta、A、x、y；n>0 时 alpha、beta 必须有效。alpha=0 时 A、x 可为空；若同时 beta=1，y 也可为空。其余需要写回结果的情况，y 必须指向有效存储；beta=0 仅免去读取旧值。

| 场景 | 返回码 |
|---|---|
| handle 为空 | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| n、lda 或步长非法；缺少必要指针；标量地址查询失败 | `ACLBLAS_STATUS_INVALID_VALUE` |
| Device 标量读回或同步失败 | `ACLBLAS_STATUS_INTERNAL_ERROR` |
| 获取计算资源失败；arch22 kernel 启动失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 参数合法且完成短路返回或提交计算 | `ACLBLAS_STATUS_SUCCESS` |

# 可维可测分析

## 精度标准/性能标准

参考实现按 Netlib ssymv 的对称语义扩展复数计算。复数乘积使用 FP32，保留其舍入、溢出和 NaN 行为；求和使用 double，再转回 float 并应用 alpha、beta。

| 验收项 | 标准 | 来源 |
|---|---|---|
| 精度 | 实部、虚部分别按 rtol=atol=2^-13 比较，匹配率≥0.99，最大绝对误差≤max(1e-2,32*ULP)；NaN 分类及 Inf 符号一致 | 任务书§3.2 |
| 性能 | 预热 5 次，一对 ACL event 包住连续 100 次调用，总区间除以 100；五项上限见下表 | 任务书§3.3；连续调用测量协议 |
| 内存 | 不涉及 | 任务书§3.4 |

1000 条精度用例覆盖上下三角、尺寸变化、lda padding、正负步长、Host/Device 标量、非法参数和 Inf/NaN。未使用的三角填入 NaN，以检测误读；测试同时检查 y 的步长空隙是否保持原值。另有 9 项补充接口与边界测试，检查空句柄、短路、仅缩放 y、对称不共轭，以及 workspace 和备用路径。

### 编译与验收

在装有 CANN 9.1.0、C++ 工具链、CMake、GTest 和参考 BLAS/LAPACK 的 NPU 环境中，加载 CANN 安装目录的 `set_env.sh` 后，在 ops-blas 仓库根目录执行：

```bash
bash build.sh --soc=ascend910b3 --ops=csymv --run --device=0
```

Atlas A3 将 `--soc` 改为 `Ascend910_9391`。`--device` 指定运行时逻辑设备编号；设置 `ASCEND_RT_VISIBLE_DEVICES` 时使用映射后的编号。

### 连续调用计时

200 条性能用例均在计时前后检查实部、虚部。内存分配、输入传输、y 恢复和 golden 计算均在事件区间外。

每项先调用 `aclblasCsymv` 5 次预热并逐次同步。随后用同一 stream 上一对 ACL event 包住连续 100 次公开接口调用，记录结束事件后同步一次。通过 `aclrtEventElapsedTime` 读取总区间并除以 100，得到单次调用均值；五项任务书用例与下表上限比较。每次调用均保留 Host 准备；连续下发允许 Host 与设备执行重叠。200 条用例均为 `beta=0`，重复调用不读取上次的 y。

| 用例 | n / uplo | 平均耗时上限 us |
|---|---|---:|
| TC_PF_1001 | 512 UPPER | 15.45 |
| TC_PF_1002 | 2048 UPPER | 46.14 |
| TC_PF_1003 | 2048 LOWER | 38.57 |
| TC_PF_1004 | 4096 UPPER | 146 |
| TC_PF_1005 | 4096 LOWER | 99.55 |

200 条 `[PERF]` 日志均包含 `metric=acl_event_batch warmup=5 repeats=100 batches=1`、`total_us` 和 `avg_us`。五项任务用例另记录上限及通过判定，其余 195 条输出 `verdict=REFERENCE`，用于性能回归比较。

### A2/A3 实测结果

两平台均使用 CANN 9.1.0、Release 构建。每个平台 1009 项精度与接口检查、200 条性能输入全部通过。

| n / uplo | A2（910B3）us/次 | A3 us/次 | 上限 us/次 |
|---|---:|---:|---:|
| 512 UPPER | 6.738 | 12.498 | 15.45 |
| 2048 UPPER | 26.413 | 25.494 | 46.14 |
| 2048 LOWER | 27.148 | 25.668 | 38.57 |
| 4096 UPPER | 79.257 | 66.954 | 146 |
| 4096 LOWER | 79.034 | 66.662 | 99.55 |

## 兼容性分析

A2/A3 沿用已有公开接口和类型；arch22、arch35 按目标架构分别构建。新增路径支持接口规定的步长、padding、标量位置和 stream 语义。

arch35 kernel、精度阈值和 CPU 参考入口保持原有行为。A2/A3 测试通过独立入口使用 double 累加，复数乘积仍为 FP32；950 测试继续使用原来的 float 累加入口。
