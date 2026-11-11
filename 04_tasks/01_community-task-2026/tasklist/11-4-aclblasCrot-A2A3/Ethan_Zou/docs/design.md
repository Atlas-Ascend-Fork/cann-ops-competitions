# aclblasCrot 算子设计（Atlas A2/A3）

## 1. 需求背景

### 1.1 需求来源

CANN 社区任务 `11-4-aclblasCrot-A2A3` 要求在 Atlas A2/A3（arch22）上实现 BLAS Level 1 算子 `aclblasCrot`。算子经 ops-blas 句柄接口提交到绑定的 stream，由 Ascend C AIV kernel 执行。

### 1.2 背景介绍

Givens 平面旋转同时更新两个向量。这里的向量为单精度复数，余弦为实数，正弦为复数。第二个输出使用正弦的共轭以及旋转前的第一个输入。数学语义参考 [cuBLAS rot](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-rot) 和 [LAPACK CROT](https://github.com/Reference-LAPACK/lapack/blob/master/SRC/crot.f)。

## 2. 需求分析

### 2.1 算子原型

复用 ops-blas 公共头文件 `include/cann_ops_blas.h` 中的声明，由 arch22 提供实现。复数类型沿用 `include/cann_ops_blas_common.h` 的 `aclblasComplex`，实部和虚部为顺序排列的两个 float32。

```cpp
aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle,
    int n,
    aclblasComplex* x,
    int incx,
    aclblasComplex* y,
    int incy,
    const float* c,
    const aclblasComplex* s);
```

| 参数 | 方向 | 类型与位置 | 含义 |
| --- | --- | --- | --- |
| `handle` | 输入 | Host 句柄 | 有效 ops-blas 上下文，提供调用 stream |
| `n` | 输入 | Host int | 向量的逻辑元素数 |
| `x`、`y` | 输入／输出 | Device complex64，ND | 各为逻辑形状 `[n]` 的一维向量，原地更新 |
| `incx`、`incy` | 输入 | Host int | 以复数元素为单位的有符号步长 |
| `c` | 输入 | Host 或 Device float32 指针 | 单个实数余弦 |
| `s` | 输入 | Host 或 Device complex64 指针 | 单个复数正弦 |

### 2.2 需求拆解

1. 按 arch22 契约处理空句柄、空向量、非法步长、跨度溢出和空数据指针。
2. 分别识别 `c` 与 `s` 的存储位置，覆盖 Host/Host、Host/Device、Device/Host 和 Device/Device 四种组合。
3. 连续向量通过 AIV 向量流水线并行处理。带步长向量按逻辑下标取数，写回仅触及被选中的复数元素。
4. 调用通过 `aclblasSetStream` 绑定的 stream 异步提交。调用方在读回结果或释放 Device 缓冲区前同步。

## 3. 详细设计

### 3.1 算子分析

对每个逻辑下标 `i ∈ [0, n)`，向量基址指向物理下标 0。令 `span(inc) = (n - 1) × abs(inc)`，则 `n > 0` 时，每个向量的物理容量至少为 `1 + span(inc)` 个复数。

```text
phys(i, inc) = (inc < 0 ? span(inc) : 0) + i × inc
k = phys(i, incx)
j = phys(i, incy)

X = x[k]                  Y = y[j]                  s = sr + i·si
x[k] = c·X + s·Y          y[j] = c·Y − conj(s)·X

outX.real = c·X.real + (sr·Y.real − si·Y.imag)
outX.imag = c·X.imag + (sr·Y.imag + si·Y.real)
outY.real = c·Y.real − (sr·X.real + si·X.imag)
outY.imag = c·Y.imag − (sr·X.imag − si·X.real)
```

每一对元素先读入原始 `X`、`Y`，计算完两个结果再写回。接口接受任意 FP32 标量值，直接按给定系数计算。`c=0` 或 `s=0` 时也保留公式的全部运算项，以保留 Inf/NaN 的传播行为。

### 3.2 Host 侧设计

```text
aclblasCrot
  │
  ├─ handle 为空 ──────────────────► HANDLE_IS_NULLPTR
  ├─ n < 0 ────────────────────────► INVALID_VALUE
  ├─ n = 0 ────────────────────────► SUCCESS
  ├─ 步长、跨度或数据指针非法 ─────► INVALID_VALUE
  │
  ▼
分别解析 c、s 的指针属性
  ├─ 属性查询失败 ─────────────────► INVALID_VALUE
  ▼
按访存条件选路径，计算分核与 tile 参数
  │
  ▼
在 handle 的 stream 下发 AIV kernel
  └─ 提交成功 ─────────────────────► SUCCESS
```

图中的返回值为 `aclblasStatus_t`，状态名带 `ACLBLAS_STATUS_` 前缀。

- **参数校验**。按图中顺序执行。有效句柄且 `n == 0` 时直接返回，跳过数据指针、步长和标量属性检查。对 `n > 0` 拦截任一步长为 0 或 `INT32_MIN`、任一跨度大于 `INT32_MAX`、以及 `x/y/c/s` 中任一空指针。使用 int64 计算绝对值与跨度。容量、字节数和交织后的浮点下标同样使用宽整数，避免合法跨度在 `span + 1`、`2 × phys + 1` 或换算为字节时溢出。
- **标量传递**。对 `c`、`s` 分别调用 `aclrtPointerGetAttributes`。Host 值在调用期间读取，以 `cValue/sReal/sImag` 随值传递。Device 地址通过独立的 kernel 参数传入，由各 AIV 在处理数据前从 GM 读取。标量标志独立设置，因而混合位置也能异步下发。Device 标量由同一 stream 上的上游操作写入时，stream 顺序保证取值时机。
- **Tiling 数据**。携带复数元素数 `n`、两个有符号步长、负步长起点、路径标志、标量来源标志、Host 标量值与每个 tile 的容量。采用 ROT 直调框架的按值 kernel 参数方式传递小型 tiling，AIV 核数通过下发配置给出。输入和输出共用原有缓冲区，额外存储位于各核 UB。
- **路径选择**。两个步长均为 +1 且两个基址均满足 32 B 对齐时，走连续向量路径。其余合法输入走按元素访问路径。AIV 核数或 UB 容量查询失败时返回 `ACLBLAS_STATUS_INTERNAL_ERROR`。立即可见的下发错误按 ops-blas 约定转换，异步执行错误由 stream 同步报告。

#### 分核

连续路径以 4 个复数（32 B）为分配单位，最后一个单位可以不足 4 个。令 `G = ceil(n / 4)`，`K = min(AIV 核数, ceil(n / tileElems))`。将 `G` 个单位除以 `K`，前 `G mod K` 个核多取一个单位。核内将起止单位乘以 4 后裁剪到 `[0, n)`，保持核间切点对齐且无重叠。

按元素路径采用同样的商余分配，单位改为单个逻辑元素，核数取 `min(AIV 核数, ceil(n / 32))`。每个合法的非零步长使同一向量的逻辑下标映射一一对应，因此同一向量的不同核写入不同元素。

### 3.3 Kernel 侧设计

```text
每个 AIV 确定逻辑区间并取得 c、sr、si
  │
  ├─ 连续且对齐
  │    搬入两段原始交织数据 → 交换相邻实虚分量
  │    → 向量计算两个输出 → 精确搬出有效元素
  │    → 还有 tile 则继续
  │
  └─ 按元素访问
       计算 x/y 物理下标 → 读取原始复数对
       → 标量计算两个输出 → 经 UB 暂存写回各 8 B
       → 还有逻辑元素则继续
```

#### 连续向量路径

1. UB 中保留交织布局。每个 tile 同时搬入 `x`、`y`，完整对齐的块使用连续搬运。最后一个不对齐的尾块使用 `DataCopyPad`，GM 搬入和搬出长度均为有效复数个数乘以 8 B。补齐仅发生在 UB，向量计算只处理有效的 float32 lane。
2. 用 `Gather` 交换每对相邻的 float32 lane，字节偏移为 `(lane XOR 1) × sizeof(float)`。每核复用偏移表和交替系数向量 `[-si, +si, -si, +si, …]`。定义 `cross(v) = swap(v) × [-si, +si, …]`，则 `s·Y = sr·Y + cross(Y)`，`conj(s)·X = sr·X − cross(X)`。使用向量乘、加、减计算两个独立的输出缓冲区。
3. 输入、输出各自采用两级队列。利用队列或等价事件保证搬入完成后计算、计算完成后搬出、搬出完成后复用对应缓冲。四组双缓冲、交换索引和系数表，以及交换缓冲与两个临时缓冲共占 `13 × 8 × tileElems` 字节。候选 `tileElems = 1024` 需约 104 KiB UB，另预留 8 KiB 给队列和对齐开销。根据平台 UB 容量向下调整，保持容量为 4 个复数的倍数。

#### 按元素访问路径

每核以最多 32 对为一批，使用 int64 计算两个物理下标，读取各自的实部和虚部。计算后的两个 complex64 结果分别存入 32 B 对齐的 UB 槽位，每个槽位只写回有效的 8 B，避免覆盖步长间隙。复用 arch22 `csrot` 的标量读取、精确字节搬出模式。标量写 UB 与 MTE3 之间、MTE3 与下批槽位复用之间设置同步。负步长与正负混用只改变下标映射。

### 3.4 工程接线与约束

| 项目 | 设计 |
| --- | --- |
| 支持硬件 | Atlas 800T A2、Atlas 800T A3（arch22） |
| 算子源码 | `blas/rot/arch22/` 增加 Crot Host 与 kernel，沿用 ROT 的架构选源机制 |
| 公共接口 | `include/cann_ops_blas.h` 共享现有原型，`blas/rot/README.md` 补充 arch22 支持及参数行为 |
| 测试接线 | `test/rot/crot/arch22/` 提供 CSV 驱动用例，核对 `test/rot/crot/CMakeLists.txt` 现有 `ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)` 在 arch22 选中正确源文件 |

两向量均为一维原地更新。调用方须提供有效的 Device 缓冲区，且两个输出向量的访问元素不重叠，Device 标量在 kernel 执行期间保持可读并独立于被修改的元素。该设计无需广播、视图或跨核归约。

## 4. 可维可测分析

### 4.1 精度与功能验收

基于 LAPACK CROT 公式独立计算参考值，按 `x`、`y` 的实部和虚部分别统计。有限值按 `abs(actual − golden) ≤ atol + rtol × abs(golden)` 逐项判断，`atol = rtol = 2^-13`。用例要求匹配比例至少 0.99，最大绝对误差满足任务书的 `1e-2` 或 `32 ULP` 上限。非有限值单独统计，同为 NaN、同号 Inf 或 NaN 与 Inf 配对视为通过，有限值与非有限值配对及异号 Inf 配对视为失败。非有限值须全部通过。

| 验证点 | 预期 |
| --- | --- |
| 纯虚 `s`，输入含非零实部和虚部 | `y` 使用共轭和旧 `x`。例如 `c=0, s=i, x=1+2i, y=3+4i` 时得到 `x=-4+3i, y=-2+i` |
| 步长 `±1/±2/±3` 组合、错位基址、对齐与尾块边界 | 尾端起算、逐对旋转、间隔元素与缓冲区哨兵保持原值 |
| `n=0` 携带空数据指针和非法步长、空句柄携带 `n=0` | 分别返回 `SUCCESS`、`HANDLE_IS_NULLPTR`，无 kernel 下发 |
| `n<0`、零步长、`INT32_MIN`、跨度超限、正长度且数据指针为空 | 返回 `INVALID_VALUE`。负向用例使用单元素或空指针占位 |
| 标量位置组合、同 stream 生产 Device 标量 | 使用对应位置及流顺序下的标量值 |
| 小尺寸、规模扫描、均匀和正态填充、Inf/NaN | 四个输出分量通过任务书精度判据 |

### 4.2 性能验收

目标为 Atlas 800T A2（910B3）上各性能用例的平均 kernel 耗时不高于对应 GPU 基线的 1.25 倍。以 `msprof op` 的 kernel task duration（µs）为口径，排除 5 次预热，取至少 10 次成功调用的算术平均。`gpu_baseline.csv` 的 `gpu_ms` 换算为 µs 后逐用例比较。

连续路径每对复数至少读 16 B、写 16 B。优先优化连续带宽和小规模下发成本，单次直调融合两路更新，以双缓冲覆盖搬运延迟。按元素路径优先保证非连续访问正确和写回范围精确。

### 4.3 兼容性分析

arch22 复用公共 ABI 并按任务书返回负长度、非法步长的错误码。现有 arch35 路径及其已发布参数契约保持独立，回归覆盖公共头文件与两套构建选源。
