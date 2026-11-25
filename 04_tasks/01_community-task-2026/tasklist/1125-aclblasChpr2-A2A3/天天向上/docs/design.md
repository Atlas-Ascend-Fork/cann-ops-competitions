# aclblasChpr2（arch22 / Atlas 800T A2 910B3）算子设计文档

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasChpr2` |
| 文档版本 | V1.0 |
| 提交者 | `harrynospot` |
| 单位 | 个人开发者 |
| 目标硬件 | arch22，Atlas 800T A2（910B3） |
| 软件版本 | CANN 9.1.0 |
| 数据类型 | COMPLEX64（实部、虚部均为 FLOAT32） |
| 开发方式 | Ascend C Kernel 直调，句柄绑定 Stream，AIV SIMD |
| 代码规划目录 | `blas/spr2/arch22/` |
| 文档状态 | 设计完成，代码与真机数据待实现和验证 |

# 1 需求背景（required）

## 1.1 需求来源

本任务来源于 CANN 社区算子开发任务，要求基于 `cann/ops-blas` 工程，在 Atlas A2/A3 系列产品上实现单精度复数 Hermitian 压缩存储秩-2 更新接口 `aclblasChpr2`。本设计面向 arch22，性能验证设备为 Atlas 800T A2（910B3）；接口语义对齐 cuBLAS `cublasChpr2` 与 Netlib BLAS `chpr2`，工程形态沿用句柄绑定 Stream 的 BLAS 直调模式。

## 1.2 背景介绍

HPR2 是 BLAS Level 2 中的 Hermitian packed rank-2 更新算子。输入矩阵并不保存完整的 `n × n` 个元素，而只保存上三角或下三角，未保存的一半由 Hermitian 共轭对称关系隐含得到。`aclblasChpr2` 在原地更新压缩数组 `AP`：

```text
AP = alpha * x * y^H + conj(alpha) * y * x^H + AP
```

其中 `alpha`、`x`、`y`、`AP` 均为 COMPLEX64；`x^H`、`y^H` 表示共轭转置。两项更新互为共轭转置，因此更新部分天然为 Hermitian 矩阵。计算后仅重写 `uplo` 指定的压缩三角，并将对角元素虚部强制置为 0。

该算子的关键不只是复数乘加，还包括压缩下标与矩阵行列下标的精确换算、任意非零正负步长、严格的 Host 校验顺序，以及在 arch22 不支持 SIMT 的约束下用 SIMD 高效处理长短不一的三角形列。

## 1.3 现状与问题

指定参考实现与本设计的关系如下：

| 参考位置 | 可复用内容 | 不能直接照抄的部分 |
| --- | --- | --- |
| `blas/spr2/arch35/chpr2_*` | 秩-2 数学分解、packed 列基址、负步长映射、对角处理 | 使用 SIMT 线程及 packed 线性区间反查列；arch22 不支持该 SIMT 路径 |
| `blas/syr2/arch22/cher2_*` | AIV SIMD kernel、zigzag 列分核、1024 行 tile、4096 复数窗口、流水同步 | 处理带 `lda` 的稠密矩阵，需改为 packed 列基址与列内偏移 |
| `blas/hpr/arch22/chpr_*` | packed 寻址、AIV 数据搬运、Host/Device 标量解析模式 | 仅为实数 `alpha`、单向量的秩-1 更新，不能直接表达 chpr2 两项复数更新 |
| `blas/spr2/arch22/chpr2_*` | 与本算子对应的 Host、tiling、SIMD kernel 骨架 | 仍须以任务书为准统一校验顺序、状态码、Host/Device `alpha`、tiling 单测和真机证据；文件存在不等于验收通过 |

因此，本设计选择“arch22 现有 SIMD 列更新框架 + chpr2 packed 数学语义”，而不是移植 arch35 的 SIMT 实现。重点解决以下问题：

1. 用 64 位整数计算 packed 基址、物理向量下标和字节偏移，避免三角形前缀和及大步长乘法溢出。
2. 让 zigzag 分核后的每一列且仅每一列归属一个 Core，避免 AP 写竞争并改善三角列长度不均造成的负载倾斜。
3. 用 1024 行 tile 限制常驻 UB 的向量长度，用 4096 复数窗口合并非单位步长搬运，超出窗口时走有界逐复数回退。
4. 严格实现 `n == 0` 与 `alpha == 0` 的不同校验边界，并同时支持 Host、Device 两种 `alpha` 指针位置。
5. 分开正式 msprof kernel-only 性能口径与事件计时快速回归口径，不把设计或模拟数据写成已通过结果。

# 2 需求分析（required）

## 2.1 需求描述

对逻辑阶数 `n > 0` 且 `alpha != (0,0)` 的调用，算子按列更新 Hermitian 压缩矩阵。对任意被存储的矩阵位置 `(i,j)`，执行：

```text
A'(i,j) = A(i,j) + alpha * x[i] * conj(y[j])
                   + conj(alpha) * y[i] * conj(x[j])
```

其中 `x[i]`、`y[i]` 表示第 `i` 个逻辑元素，不一定等于底层连续数组的第 `i` 项。若 `i == j`，只保留更新后的实部并令虚部为 0；若 `i != j`，实部和虚部均按公式累加。

接口还需满足：

- `uplo` 支持 `ACLBLAS_UPPER`、`ACLBLAS_LOWER`，只读写对应 packed 三角；
- `incx`、`incy` 支持任意非零 `int`，包括负步长，但 0 和 `INT_MIN` 非法；
- `n == 0` 是合法 no-op，但不能跳过 `alpha` 指针、`uplo`、`incx/incy` 的结构校验；
- `alpha == (0,0)` 是合法 no-op，但仅在 `n > 0` 的 `x/y/AP` 指针检查通过后返回；
- `alpha` 可位于 Host 或 Device，`x/y/AP` 为 Device 数据；
- Kernel 在 `handle` 绑定的 Stream 上异步执行，不申请完整矩阵临时空间，不做广播或自动连续化。

## 2.2 接口定义

使用 `include/cann_ops_blas.h` 已有公共声明，不增加 arch22 私有平行 API：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* AP);
```

返回 `aclblasStatus_t`。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法 `uplo` 返回 `ACLBLAS_STATUS_INVALID_ENUM`；其余非法阶数、步长、空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`；平台信息或 Kernel 启动失败映射为相应执行错误。packed 接口没有 `lda` 参数。

## 2.3 参数说明

| 参数 | 位置 | 含义 | 约束与异常行为 |
| --- | --- | --- | --- |
| `handle` | Host | ops-blas 句柄，携带执行 Stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | Host | packed 上三角或下三角模式 | 仅允许 `ACLBLAS_UPPER/LOWER`，其他值返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | Host | Hermitian 方阵阶数 | `n >= 0`；`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | Host 或 Device | COMPLEX64 复数标量 | 指针必须非空，包括 `n == 0`；空指针返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | Device | 逻辑长度 `n` 的只读复数向量 | `n > 0` 时指针必须非空；物理元素数至少 `1+(n-1)*abs(incx)` |
| `incx` | Host | `x` 的逻辑步长 | 非零且不为 `INT_MIN`，可为负 |
| `y` | Device | 逻辑长度 `n` 的只读复数向量 | `n > 0` 时指针必须非空；物理元素数至少 `1+(n-1)*abs(incy)` |
| `incy` | Host | `y` 的逻辑步长 | 非零且不为 `INT_MIN`，可为负 |
| `AP` | Device | 输入并原地输出的 packed Hermitian 矩阵 | `n > 0` 时指针必须非空；长度为 `n(n+1)/2` 个复数，无 `lda` |

调用方保证 `AP` 的可写区域与 `x/y` 的可读区域不重叠。`n == 0` 时 `x/y/AP` 逻辑长度均为 0，允许为空；此时 `alpha` 仍须是非空指针，但不读取其数值。

## 2.4 需求拆解

1. 实现 COMPLEX64 Hermitian packed 秩-2 更新，实部、虚部均按 FLOAT32 计算。
2. 正确实现 UPPER、LOWER 两种列主序无间隙 packed 布局及对角虚部清零。
3. 支持运行时 `n`，覆盖 0、1、非对齐尺寸、1024 行 tile 边界与大尺寸。
4. 支持 `incx/incy` 的正负号、非单位步长和大步长组合，不越界读取物理缓冲区。
5. Host 侧按任务书固定顺序校验，解析 Host/Device `alpha`，生成 tiling 并启动 Kernel。
6. Device 侧只采用 arch22 可用的 AIV SIMD 能力，不依赖 SIMT。
7. 通过 packed 基址、分核双射、窗口边界三类 tiling 单测先行固定地址契约。
8. 以 cblas/Netlib `chpr2` 为 golden，按任务书精度与 msprof kernel-only 性能口径验证。

## 2.5 边界语义

Host 校验顺序是对外契约，不能因提前 quick return 或合并判断而改变：

| 顺序 | 检查或动作 | 结果 |
| ---: | --- | --- |
| 1 | `handle == nullptr` | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n < 0` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `alpha == nullptr` | 返回 `ACLBLAS_STATUS_INVALID_VALUE`；此步不读 `alpha` 数值 |
| 4 | `uplo` 非 UPPER/LOWER | 返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| 5 | 依次检查 `incx`、`incy` 为 0 或 `INT_MIN` | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `n == 0` | 返回 SUCCESS；不检查 `x/y/AP`，不读 `alpha`，不生成 tiling，不启动 Kernel |
| 7 | `n > 0` 时检查 `x/y/AP` 空指针 | 任一空指针均返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | 解析 Host/Device `alpha`，再判断实部、虚部是否都为 0 | 为 `(0,0)` 时返回 SUCCESS，不读 `x/y/AP` 数值、不启动 Kernel |
| 9 | 非零 `alpha` | 做长度溢出保护、tiling 和异步 Kernel 下发 |

由此得到几个必须保留的反例语义：

- `n == 0` 叠加空 `alpha`、非法 `uplo`、零步长或 `INT_MIN`，仍返回对应结构错误；
- `n == 0` 且结构参数合法时，即使 `x/y/AP == nullptr` 也返回 SUCCESS；
- `n > 0`、`alpha == (0,0)` 且任一 `x/y/AP` 为空时，仍返回 INVALID_VALUE；
- `n > 0`、`alpha == (0,0)` 且指针均有效时，即使 `x/y` 数值为 NaN，也直接成功且不读取这些数值；
- 额外的尺寸或溢出检查放在零 `alpha` quick return 之后，避免改变上述状态优先级。

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

记虚数单位为 `I`，`I²=-1`。复数 `z=a+I*b` 的共轭为 `conj(z)=a-I*b`；两个复数相乘为：

```text
(a + I*b)(c + I*d) = (a*c - b*d) + I*(a*d + b*c)
```

`x`、`y` 是 `n × 1` 列向量，`y^H` 是 `1 × n` 共轭行向量，所以 `x*y^H` 是 `n × n` 矩阵，其第 `(i,j)` 个元素为 `x[i]*conj(y[j])`。因此逐元素公式为：

```text
P(i,j) = alpha * x[i] * conj(y[j])
Q(i,j) = conj(alpha) * y[i] * conj(x[j])
A'(i,j) = A(i,j) + P(i,j) + Q(i,j)
```

对 `P=alpha*x*y^H` 做共轭转置，得到 `P^H=conj(alpha)*y*x^H=Q`，故 `P+Q` 为 Hermitian。对角线上还有更直接的解释：令 `z=alpha*x[j]*conj(y[j])`，则 `Q(j,j)=conj(z)`，所以：

```text
P(j,j) + Q(j,j) = z + conj(z) = 2 * real(z)
```

更新部分的对角虚部必为 0。最终写回时强制 `imag(A'(j,j))=0`，输入对角旧虚部不保留。

为了映射到 SIMD 标量系数，固定列 `j`，定义：

```text
t1 = alpha * conj(y[j])
t2 = conj(alpha * x[j]) = conj(alpha) * conj(x[j])
```

则该列所有被存储的行 `i` 只需计算：

```text
A'(i,j) = A(i,j) + x[i]*t1 + y[i]*t2
```

这把每列的 `alpha` 与列标量运算降到一次，把列内运算变成两个复数向量乘标量并累加。

### 3.1.2 数据类型与存储

`aclblasComplex` 由两个 FLOAT32 分量组成，Device 内存采用实部、虚部交错的 AoS 布局：

```text
[real0, imag0, real1, imag1, ...]
```

每个复数占 8 Byte，第 `p` 个复数对应的 FLOAT32 下标为 `2*p` 与 `2*p+1`。所有 GM 元素偏移和字节偏移先以 64 位无符号整数计算，再交给地址接口。

packed 数组复数元素总数为：

```text
packedLength = n * (n + 1) / 2
```

`x/y` 的物理元素跨度分别为 `1+(n-1)*abs(incx)`、`1+(n-1)*abs(incy)`；只对逻辑下标指向的元素取值，不处理跨度中的空洞。

### 3.1.3 packed 行列映射

采用 0 基行列下标。设 `B(j)` 是第 `j` 列第一个元素在 `AP` 中的复数下标。

| 模式 | 本列保存范围 | 前列元素和 `B(j)` | 元素 `(i,j)` 的 packed 下标 `p` | 本列长度 |
| --- | --- | --- | --- | --- |
| UPPER | `0 <= i <= j` | `j*(j+1)/2` | `B(j)+i = i+j*(j+1)/2` | `j+1` |
| LOWER | `j <= i < n` | `j*(2*n-j+1)/2` | `B(j)+(i-j) = i+j*(2*n-j-1)/2` | `n-j` |

LOWER 的两种写法完全等价，第二种与任务书及 Netlib 的线性下标形式一致。对角下标分别为 `B_UPPER(j)+j` 和 `B_LOWER(j)`。所有合法 `p` 均满足 `0 <= p < packedLength`。

以 `n=4` 为例：

```text
UPPER AP = [A00, A01, A11, A02, A12, A22, A03, A13, A23, A33]
LOWER AP = [A00, A10, A20, A30, A11, A21, A31, A22, A32, A33]
```

未存储的另一半只在数学上通过 `A(i,j)=conj(A(j,i))` 隐含存在；Kernel 不展开、不读写这部分 GM。

### 3.1.4 负步长地址映射

传入指针指向物理缓冲区的第 0 个元素。对逻辑下标 `k`，定义物理复数下标：

```text
inc > 0: physicalIndex(k, n, inc) = k * inc
inc < 0: physicalIndex(k, n, inc) = (n - 1 - k) * (-inc)
```

也可写成“逻辑起点加带符号步长”：正步长起点为 0，负步长起点为 `(n-1)*abs(inc)`，之后每次加 `inc`。负步长因此自然表示反向遍历，且所有结果仍落在 `[0, 1+(n-1)*abs(inc))` 内。

计算 `abs(inc)` 前先将 `inc` 提升为 `int64_t`；由于 `INT_MIN` 已在 Host 拒绝，不存在 32 位取负溢出。x、y 各自独立使用该映射，不能因二者步长不同而共用物理下标。

## 3.2 总体方案

```text
用户调用 aclblasChpr2
        |
        v
Host 按固定顺序校验 handle/n/alpha指针/uplo/incx/incy
        |
        +-- n == 0 ------------------------------> SUCCESS
        |
        v
校验 x/y/AP 指针 -> 解析 Host/Device alpha
        |
        +-- alpha == (0,0) ----------------------> SUCCESS
        |
        v
64 位长度保护、获取 AIV Core 数、生成 Chpr2TilingData
        |
        v
在 handle->stream 异步启动 AIV SIMD Kernel
        |
        v
zigzag 列分核
        |
        v
按 1024 行 tile 装载 x/y
  |-- inc == 1：连续 DataCopyPad
  |-- 窗口 <= 4096 复数：连续窗口搬运 + UB 重排
  |-- 窗口 > 4096 复数：逐逻辑复数 8 Byte 有界回退
        |
        v
逐列求有效行交集 -> packed 段搬入 -> SIMD 两项复数乘加
        |
        v
对角虚部置 0 -> 精确长度写回 AP
```

该方案不创建完整 `n × n` 矩阵、不执行跨列归约，也不需要额外 Device Workspace。列之间没有写依赖，行 tile 只是同一列内部的连续分段。

## 3.3 Host 侧设计

### 3.3.1 参数校验

Host 将“结构校验”“张量指针校验”“标量解析与 quick return”“启动准备”分开，严格按照 2.5 节顺序调用，不将多个条件合并成可能改变优先级的快速路径。

实现时特别固定：

- `uplo` 单独返回 `ACLBLAS_STATUS_INVALID_ENUM`，不与步长错误合并；
- `n == 0` 判断只能位于结构校验之后、张量指针检查之前；
- `alpha` 指针判空位于 `uplo` 之前，`alpha` 数值读取位于张量指针检查之后；
- 只有进入非零 `alpha` 启动路径后才获取 AIV Core 数和构造 tiling。

### 3.3.2 Host/Device alpha 解析

复用 `blas/common/helper/host_utils.h` 的指针位置与标量读取机制，行为与 `blas/hpr/arch22/chpr_host.cpp` 的 Host/Device 双位置模式一致：

1. 用 `CheckPtrLocation` 判断 `alpha` 的位置，查询失败直接返回 helper 对应错误；
2. Host 指针直接读取 `real/imag` 两个 FLOAT32，正常路径不做 Stream 同步；
3. Device 指针通过 `ReadScalarFromHostOrDevice<aclblasComplex>` 或等价的 8 Byte D2H 流程，在 `handle->stream` 上回读并同步，以获得 quick return 判断和 tiling 所需的标量值；失败时向上返回错误，不启动 Kernel；
4. 将解析后的值复制到 tiling 的 `alphaReal/alphaImag`，Kernel 不再保存或解引用调用者的标量指针。

Device 标量回读是取得 Host 侧 quick return 决策所必需的同步步骤；Kernel 本身仍按句柄 Stream 异步执行。`n == 0` 不进入本流程，因此只检查 `alpha` 非空，不查询位置、不读取数值。

### 3.3.3 Tiling 数据

tiling 以固定值类型传递，计划保持以下等价信息：

| 字段或常量 | 类型 | 含义与取值 |
| --- | --- | --- |
| `n` | `uint32_t` | 非零逻辑阶数，由已校验的正 `int` 转换 |
| `uplo` | `uint32_t` | UPPER/LOWER 枚举值 |
| `numBlocks` | `uint32_t` | `min(n, GetAivCoreCount())` |
| `alphaReal/alphaImag` | `float` | 已解析的复数标量分量 |
| `incx/incy` | `int64_t` | 保留符号的逻辑步长 |
| `CHPR2_ARCH22_TILE_ROWS` | 编译期常量 | 固定为 1024 个逻辑行 |
| `CHPR2_ARCH22_WIN_MAX_CPLX` | 编译期常量 | 固定为 4096 个物理复数 |

启动前用 64 位整数检查 `x/y` 物理跨度和 `packedLength` 的字节表示能力。`GetAivCoreCount()==0` 或启动失败返回执行错误。`numBlocks` 不超过 `n`，因此没有仅因核数多于列数而产生的空核。

### 3.3.4 zigzag 列分核

设 Core 数为 `C=numBlocks`，Core 编号为 `b∈[0,C)`，轮次为 `r∈[0,ceil(n/C))`。该 Core 在轮次 `r` 处理列：

```text
base = r * C
j(b,r) = base + b                 ，r 为偶数
j(b,r) = base + (C - 1 - b)       ，r 为奇数
```

若 `j >= n`，说明是不完整末轮的无效槽位，直接跳过。偶数轮正序、奇数轮逆序，使相邻两组三角列的长短在不同 Core 间互补；UPPER 的列长递增与 LOWER 的列长递减都可受益。

映射具有可证明的逆映射。令 `q=floor(j/C)`、`t=j-q*C`：

```text
q 为偶数：r=q，b=t
q 为奇数：r=q，b=C-1-t
```

故每个有效列 `j` 恰好映射到一个 `(b,r)`，既无遗漏也无重复。每个 Core 的列数只相差至多 1；列长均衡是 zigzag 的负载优化目标，不把“各列元素数完全相等”作为正确性假设。

### 3.3.5 1024 行 tile 与 packed 段

对 `tileStart=0,1024,2048,...` 逐块处理：

```text
tileEnd = min(n, tileStart + 1024)
L       = tileEnd - tileStart，1 <= L <= 1024
```

每个 Core 将本 tile 的 `x/y` 逻辑行装载到 UB。对本 Core 的每一列 `j`，先求列存储范围与当前 tile 的交集：

```text
UPPER: colLo=0，colHi=j+1
LOWER: colLo=j，colHi=n
rowLo=max(tileStart,colLo)
rowHi=min(tileEnd,colHi)
```

仅在 `rowLo<rowHi` 时处理，`count=rowHi-rowLo`。该段在 AP 中严格连续：

```text
segmentPackedBase = B(j) + (rowLo-colLo)
segmentGmFloat    = 2 * segmentPackedBase
segmentBytes      = count * 8
```

列标量 `x[j]/y[j]` 若属于当前 tile，直接从 UB 原始向量副本读取；否则按负步长通用映射做一次 8 Byte 标量读取。这样既复用 tile 数据，又不为列标量构造额外大缓冲区。

### 3.3.6 4096 复数窗口

对非单位步长的某个 tile，首末逻辑行的物理下标为：

```text
first = physicalIndex(tileStart, n, inc)
last  = physicalIndex(tileStart + L - 1, n, inc)
winLo = min(first,last)
W     = abs(first-last) + 1 = (L-1)*abs(inc) + 1
```

物理下标随逻辑下标单调变化，因此首末两点即可确定覆盖本 tile 全部逻辑元素的最小连续物理窗口 `[winLo,winLo+W)`。

- `W <= 4096`：一次搬入该连续窗口，最多 `4096*8=32768 Byte`；随后按 `physicalIndex-winLo` 在 UB 选取逻辑元素。正步长顺序收集，负步长自然逆序收集。
- `W > 4096`：不扩大 UB，也不截断窗口；改为逐逻辑元素搬运 8 Byte 到连续 tile 缓冲区，保证任意合法大步长正确。
- `inc == 1`：无需窗口，直接搬运 `L*8 Byte` 的连续逻辑向量。
- `L == 1`：无论步长多大，`W=1`，只需一个复数。

窗口仅覆盖物理下标最小值到最大值，跨度中的空洞可以随窗口读入 UB，但不会被当作逻辑元素参与计算；所有读取均位于调用方物理缓冲跨度内。

### 3.3.7 Stream 与错误处理

Kernel 启动入口接收 `handle->stream`，正常 Host `alpha` 路径只完成校验、tiling 与下发，不主动等待 Kernel 完成。调用者读回 AP 前负责同步 Stream。启动封装返回实际 `aclError`，Host 将非 SUCCESS 映射为 `ACLBLAS_STATUS_EXECUTION_FAILED`，不能在启动失败后仍报告成功。

## 3.4 Kernel 侧设计

### 3.4.1 AIV SIMD 复数乘法

Kernel 声明为 `KERNEL_TYPE_AIV_ONLY`，复用 `cher2/chpr2` 的 `TPipe+TBuf+DataCopyPad+Gather+Muls+Axpy+Add` 模式。arch22 不包含 arch35 的 `asc_simt`、`asc_vf_call` 或逐线程 packed 反查路径。

对 UB 中交错保存的复数 `v=vr+I*vi`，构造辅助序列：

```text
S(v) = -vi + I*vr
```

即每个复数的两个 FLOAT32 位置由 `[vr,vi]` 变为 `[-vi,vr]`。对复数标量 `t=tr+I*ti`，先对原序列乘 `tr`，再以 `S(v)` 和 `ti` 做 Axpy：

```text
real = vr*tr - vi*ti
imag = vi*tr + vr*ti
```

这恰好是 `v*t`。`S(x)`、`S(y)` 在每个行 tile 装载后生成一次，供该 Core 的多列复用。

### 3.4.2 向量装载与重排

单位步长使用精确字节数的连续 `DataCopyPad`。非单位步长按 3.3.6 节选择窗口或逐复数回退，先在 UB 得到按逻辑行顺序连续排列的 `[xr,xi]`、`[yr,yi]`，再通过 Gather 分离实部、虚部并生成 `S(x)`、`S(y)`。

窗口路径的 GM 搬运是连续的，逻辑顺序恢复只发生在 UB；回退路径的每次 GM 请求固定为一个复数，避免跨大步长产生越界或超硬件stride限制。两条路径进入后续 SIMD 计算时使用完全相同的数据契约。

### 3.4.3 packed 列更新

每列先计算标量系数：

```text
t1r = ar*yjr + ai*yji
t1i = ai*yjr - ar*yji
t2r = ar*xjr - ai*xji
t2i = -(ar*xji + ai*xjr)
```

其中 `alpha=ar+I*ai`、`x[j]=xjr+I*xji`、`y[j]=yjr+I*yji`。对当前 packed 行段执行：

1. 按 `segmentGmFloat` 和 `segmentBytes` 将旧 AP 段搬入 UB；
2. 用 `x` 与 `S(x)` 计算 `D1[i]=x[i]*t1`；
3. 用 `y` 与 `S(y)` 计算 `D2[i]=y[i]*t2`；
4. 以 SIMD Add 完成 `AP=AP+D1+D2`；
5. 若本段含 `i==j`，对角实部按相同公式更新，对角虚部直接写 0；旧对角虚部不参与输出；
6. 以原 `segmentBytes` 精确写回，不触碰相邻列或 packed 末尾之外的元素。

当 `x[j]` 与 `y[j]` 同时为 `(0,0)` 时，`t1=t2=0`，整列更新量为 0，可跳过非对角读写；若该段包含对角，仍须把对角虚部清 0。NaN 不参与该零值优化判断。

### 3.4.4 对齐、尾块与流水

复数起始地址按 8 Byte 递增，而 SIMD 对齐粒度为 32 Byte，因此用 `localRow mod 4` 判断列段在 UB 中的对齐位置：

- 对齐时直接对 `2*count` 个 FLOAT32 执行 Muls/Axpy；
- 非对齐时向较小行号向下对齐到本 tile 内已初始化的位置，扩展本地计算后用 Gather 丢弃前导数据，GM 读写范围仍保持原行段不变；
- 不足完整 SIMD 重复块的尾部只在 UB 预留区补齐，GM 写回始终为 `count*8 Byte`，不覆盖有效范围外数据。

沿用参考 SIMD kernel 的资源规划：`WORK_STRIDE=1032` FLOAT32、常规缓冲区 `BUF_FLOATS=2112` FLOAT32。8 个常规缓冲区、3 个索引表和标量区合计约 88,832 Byte；非单位步长再启用 32,768 Byte 窗口，逻辑总量约 121,600 Byte。实际编译前仍需按平台 UB 可用容量和编译器保留区复核，不能仅凭逻辑求和宣称资源验证通过。

流水中明确保持以下依赖：MTE2 搬入完成后才能 Vector 读取；Vector 写 UB 完成后才能标量修正或 MTE3 搬出；下一轮换入复用同一 UB 前等待上一轮 MTE3 完成。各 Core 只写自己拥有的 packed 列，不需要全局同步或原子操作。

## 3.5 文件结构设计

计划在 `ops-blas` 中核对、实现或修正：

```text
include/cann_ops_blas.h                       # 只核对已有 aclblasChpr2 公共声明
blas/spr2/arch22/chpr2_host.cpp               # 校验、alpha 解析、tiling、启动
blas/spr2/arch22/chpr2_kernel.cpp             # arch22 AIV SIMD kernel
blas/spr2/arch22/chpr2_kernel.h               # Kernel 启动声明
blas/spr2/arch22/chpr2_tiling_data.h          # tiling 字段及 1024/4096 常量
test/spr2/chpr2/CMakeLists.txt                 # 测试构建
test/spr2/chpr2/chpr2_param.h                 # CSV 参数与状态映射
test/spr2/chpr2/chpr2_golden.h                # cblas/Netlib packed golden
test/spr2/chpr2/arch22/chpr2_npu_wrapper.h    # NPU 调用与 Host/Device alpha 封装
test/spr2/chpr2/arch22/chpr2_test.cpp         # GTest 功能、精度、性能入口
test/spr2/chpr2/arch22/chpr2_tiling_test.cpp  # packed/分核/窗口纯计算单测
test/spr2/chpr2/arch22/chpr2_test.csv         # 任务配套 1000 精度 + 200 性能用例
```

CSV 的 `ReadMap` 键统一使用 `x_fill/y_fill/ap_fill`。公共声明已存在，不另建私有接口；测试工程实现 `RANDOM_GAUSS_0_1`、`RANDOM_DIAG_IMAG` 等任务配套标记。

## 3.6 兼容性与资源分析

- 公共接口与其他产品线共用，数据类型固定为 COMPLEX64，不进行隐式降精度或扩展到其他 dtype。
- AP 原地更新，仅 `uplo` 指定三角被读写；不返回视图，不支持 AP 与 x/y 重叠。
- x/y 的任意非零步长由地址映射表达，不使用 AutoContiguous/ViewCopy，不产生全向量连续化临时 GM。
- 不展开 Hermitian 另一半，不分配 `n²` 工作区；正常执行无需额外 Workspace，tiling 随 Kernel 参数传递。
- Host `alpha` 直接取值，Device `alpha` 通过既有 helper 回读；二者最终使用同一份 tiling 标量，避免 Device Kernel 的指针位置分支。
- Kernel 在 handle Stream 上执行；Host `alpha` 路径无主动同步，Device `alpha` 仅在启动前为标量决策执行既有 D2H 同步。
- packed 基址、列分核、物理跨度与 GM 字节地址使用 64 位中间值；UB 内局部长度保持 32 位且受 1024/4096 上限约束。
- arch22 仅使用 AIV SIMD，不依赖 SIMT；arch35 只作为数学与 packed 语义参考。

# 4 可维可测分析

## 4.1 精度标准

COMPLEX64 的实部、虚部分别按 FLOAT32 与 cblas/Netlib `chpr2` golden 比对，golden 使用 `CblasColMajor` packed 单精度复数结果。任务书验收阈值为：

| 指标 | 要求 |
| --- | --- |
| `rtol` | `2^-13 = 0.0001220703125`，约 `1.2207e-4` |
| `atol` | `2^-13 = 0.0001220703125`，约 `1.2207e-4` |
| 匹配比例 | `matched_ratio >= 0.99` |
| 最大绝对误差 | `max_abs_error <= 1e-2 or 32 * ULP`，按任务书生态标准口径执行 |

单个实部或虚部分量的匹配判据为：

```text
abs(actual - golden) <= atol + rtol * abs(golden)
```

一个用例须同时满足匹配比例和最大绝对误差限制。实部、虚部分别记录误差与匹配统计，对角虚部另做“精确为 0”的语义检查。测试工程的 MERE/MARE 输出可作为辅助诊断，不替代上述任务书阈值，也不沿用参考模板中其他算子的精度数值。

有限随机输入按任务要求覆盖 50% 均匀分布 `[-5,5]` 与 50% 标准正态分布，实部、虚部独立采样。Inf/NaN、非零输入对角虚部、零 `alpha` 加 NaN 向量等场景作为独立语义用例核对，不能预先把它们写成已通过。

## 4.2 功能与异常测试

采用任务配套 CSV 驱动的 C++ GTest，规划 1000 条精度/功能/异常用例和 200 条性能用例。至少覆盖：

| 类别 | 重点内容 |
| --- | --- |
| 基础与尺寸 | UPPER/LOWER；`n=0/1/2/3/5/7`、2 的幂及相邻值、1023/1024/1025、2048、4095/4096 |
| `alpha` | Host 与 Device 两种位置；零、1、-1、纯虚数、一般复数；零值 quick return |
| 步长 | `incx×incy` 的 ±1/±2/±3 组合，大步长 7/8/16/31/64/127，负步长与窗口回退 |
| packed 语义 | 两种三角基址、最后一列、对角位置、非零旧对角虚部被清除、未选三角不被访问 |
| 数据填充 | 均匀、正态、全零、交替、极值、Inf/NaN；实部与虚部独立检查 |
| 异常与顺序 | 空 handle、负 `n`、空 `alpha`、非法 `uplo`、零步长、`INT_MIN`、空 `x/y/AP` 的组合优先级 |
| no-op | `n==0` 的结构错误不豁免、合法时 tensor 可空；零 `alpha` 前仍检查 tensor 指针且不读 NaN 数值 |
| 地址与尾块 | tile 首尾、非 32 Byte 对齐列段、packed 末尾保护区、物理向量首尾保护区 |

tiling 纯计算单测不依赖真机 Kernel，先固定三类核心不变量：

1. **packed 基址单测**：对 UPPER/LOWER 逐列比较 `B(j)` 与前列长度累加；检查 `n=1`、首/末列、对角、`packedLength-1` 和 64 位大下标。`n=4` 时 UPPER 基址必须为 `[0,1,3,6]`，LOWER 基址必须为 `[0,4,7,9]`，两种模式最后有效下标均为 9；另验证 1024 tile 切分后 `segmentPackedBase` 与不切分公式一致。
2. **分核双射单测**：枚举 `C<=n` 的多种组合，特别是 `n=1`、`n=C-1/C/C+1`、`2C-1/2C/2C+1` 及不完整末轮；对所有有效 `(b,r)` 检查 `j<n`，对每列用逆映射找回唯一 `(b,r)`，断言全集无重复、无遗漏，并检查每核列数仅取 `floor(n/C)` 或 `ceil(n/C)`。另在 Host 单测中构造物理核数大于 `n` 的情形，验证先归一化为 `C=min(n,物理核数)`，再进入上述双射检查。
3. **窗口边界单测**：对正负步长分别验证 `winLo/W` 与逐元素物理下标集合。构造 `L=179,abs(inc)=23` 得 `W=4095`，`L=820,abs(inc)=5` 得 `W=4096`，`L=257,abs(inc)=16` 得 `W=4097`；前两者选择窗口，第三者选择逐复数回退。另覆盖 `L=1`、非 1024 尾 tile、单位步长以及物理缓冲区首末地址，断言窗口不超过 4096 复数且所有逻辑读取不越界。

这些单测是设计要求的验证项，当前不填写“通过”结论；实现后需保留测试日志。

## 4.3 性能标准

正式性能验证在 Atlas 800T A2（910B3）、CANN 9.1.0 上进行，使用与 `gpu_baseline.csv` 完全相同的 case、`uplo/n/incx/incy` 和 kernel-only 口径：

1. 用 GTest 收敛到单条 `TC_PF` 用例，先 warmup 5 次，不计入样本；随后连续执行 10 次，只有返回 SUCCESS 的执行才可计入有效样本。
2. 用 `msprof op` 采集，例如：
   ```bash
   msprof op --application="./build/test/spr2/chpr2/chpr2_test --gtest_filter=*TC_PF*" --output=./prof_chpr2
   ```
3. 在输出目录 `OPPROF_*/OpBasicInfo.csv` 中按本算子 kernel 名读取 `Task Duration(us)`。剔除 5 次 warmup，取后续 10 个有效 kernel task 的算术平均；异常样本不计入并在报告中说明。
4. 单位换算与正式判据为：
   ```text
   npu_kernel_ms = avg(Task Duration(us)) / 1000
   通过条件      = npu_kernel_ms <= gpu_ms / 0.8
                  = npu_kernel_ms <= 1.25 * gpu_ms
   等价性能倍率  = gpu_ms / npu_kernel_ms >= 0.8
   ```
5. `gpu_ms` 必须来自 cuBLAS profiling/nsys 的 kernel duration，不使用 GPU 端到端时间；缺失基线的 case 标记为无参考，不伪造结论。

`verify_performance.py` 的 `aclrtEvent` 路径同样 warmup 5 次、连续 10 次下发取平均，但其 `avg_us` 含下发开销，只用于快速回归和趋势判断。不得将事件时间、GTest 整数毫秒或端到端时间替代 msprof kernel-only 正式结果。当前文档不预填任何真机耗时或性能通过率。

## 4.4 测试执行与证据

实现与验证完成后，测试报告需要记录：

1. 硬件型号、arch22、CANN 版本、代码提交号和完整测试命令。
2. 精度、功能、异常、tiling 单测的计划数、实际执行数、通过数和失败数。
3. 实部、虚部的 matched ratio、最大绝对误差及对角虚部检查；失败用例保留输入参数和 golden 对照。
4. 每条性能 case 的 5 次 warmup 与 10 个有效样本、msprof 原始 `Task Duration(us)`、平均值、GPU kernel-only 基线及单位换算。
5. Host/Device `alpha` 两类测试证据，以及负步长、窗口回退、packed 边界和保护区检查日志。
6. 编译日志、测试日志、msprof 文件和真实设备截图；异常样本与未执行项明确列出。

本文档只描述设计和验证计划，不将已有参考代码、未执行测试或占位基线标记为通过。

# 5 风险与应对

| 风险 | 影响 | 应对措施 |
| --- | --- | --- |
| LOWER 基址与 `i-j` 偏移混用 | 整列错位、对角写错 | 同时固定 `B(j)` 与任务书展开式，使用 `n=4` 显式布局及逐列单测 |
| 提前 `n==0` 或零 `alpha` 返回 | 状态优先级与 cuBLAS/任务书不符 | 按 2.5 表逐步校验，用组合负向用例锁定顺序 |
| 非法 `uplo` 返回普通参数错误 | 状态码不符合任务书 | 单独返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| Device `alpha` 被 Host 直接解引用 | 非法访问或读到错误标量 | 使用指针位置检查和 8 Byte D2H helper，覆盖两种位置与零值 |
| 负步长首地址理解错误 | 逻辑向量倒序或 GM 越界 | 统一物理下标公式，覆盖四种步长符号组合和首尾保护区 |
| zigzag 分核遗漏或重复列 | AP 未更新或跨核写竞争 | 用逆映射证明并对小规模全枚举做双射单测 |
| 1024 行 tile 与列范围交集错误 | 尾块漏算、重复累加 | 使用半开区间和 `max/min` 交集，覆盖 1023/1024/1025 与最后尾 tile |
| 窗口刚好越界或大步长溢出 UB | 结果错误或运行失败 | 4095/4096/4097 边界单测；`W>4096` 逐复数回退，64 位跨度计算 |
| 对角旧虚部被保留 | 输出不再满足 Hermitian 对角实数约束 | 所有列更新路径末尾显式清 0，包括零列向量优化路径 |
| 非对齐段及流水竞争 | 数据污染、偶发错误 | UB 内对齐扩展、精确 GM 字节数、MTE2/V/MTE3 事件依赖和尾块测试 |
| 误用 arch35 SIMT | arch22 编译或运行失败 | 只使用 AIV SIMD 指令与现有 arch22 kernel 框架 |
| 性能计时口径混用 | 性能结论失真 | 正式只采用 msprof Task Duration，warmup 5 + 有效 10，与 GPU kernel-only 对比 |
| 将计划或参考数据写成实测 | 验收证据失真 | 文档保持“待实现和验证”，实测后附原始日志与截图 |

# 6 结论

本设计采用“严格 Host 契约 + Host/Device 标量归一 + arch22 AIV SIMD 列更新”的方案。Host 按任务书固定顺序处理校验和 quick return；Device 以 zigzag 列分核保证 packed 列唯一归属，以 1024 行 tile 控制 UB 常驻规模，以 4096 复数窗口兼顾非单位步长搬运效率和任意合法大步长的正确回退。

packed 基址、负步长物理映射、两项复数秩更新、对角虚部清零均有明确数学定义，配套三类 tiling 纯计算单测可在真机验证前发现主要寻址问题。精度与性能验证分别遵循 `2^-13` FLOAT32 分量阈值和 msprof kernel-only 正式口径；当前仅为设计完成状态，代码实现与真机结果仍需后续验证。

# 7 参考资料

1. `task-spec/aclblasChpr2_AtlasA2_A3_task_doc.md`，aclblasChpr2 算子开发任务书。
2. `task-spec/test_cases/README.md`，测试用例、精度与 msprof 性能口径。
3. `blas/spr2/arch22/chpr2_host.cpp`、`chpr2_kernel.cpp`、`chpr2_kernel.h`、`chpr2_tiling_data.h`。
4. `blas/spr2/arch35/chpr2_*`，chpr2 数学、packed 与负步长参考。
5. `blas/syr2/arch22/cher2_*`，arch22 SIMD、zigzag、行 tile 与窗口参考。
6. `blas/hpr/arch22/chpr_*`，packed 与 Host/Device 标量处理参考。
7. Netlib BLAS `chpr2`：<https://www.netlib.org/blas/chpr2.f>。
8. NVIDIA cuBLAS `cublasChpr2`：<https://docs.nvidia.com/cuda/cublas/index.html>。
9. CANN 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>。
10. Ascend C 算子开发文档：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html>。
