# aclblasChpr 算子设计文档（Ascend 950PR / arch35）

## 1 需求背景（required）

### 1.1 需求来源

2026 年 10 月 CANN 社区任务「aclblasChpr 算子开发（950）」，任务要求使用 Ascend C 编程语言在昇腾 NPU（Atlas 950PR）上补齐 `aclblasChpr` 的 arch35 实现，完成算子设计、开发、测试全流程，验收通过后合入昇腾算子开源仓 [ops-blas](https://gitcode.com/cann/ops-blas)。

### 1.2 背景介绍

`aclblasChpr` 是 BLAS `CHPR` 的等效实现，执行单精度复数（complex64）Hermitian 压缩存储秩 1 更新：

```
A = alpha * x * conjg(x^T) + A
```

其中 `alpha` 为**实数** float 标量（Hermitian 矩阵在实数标量缩放与 Hermitian 外积叠加下仍保持 Hermitian，这是 alpha 必须是实数的数学原因，与 cuBLAS `cublasChpr` 及 Netlib `ALPHA is REAL` 一致）。`x` 为 n 元素复数向量，`A` 为 n×n Hermitian 矩阵，以 packed 压缩格式存储于 `ap`。

ops-blas 仓 `blas/hpr/` 算子族已有 arch22 实现（`blas/hpr/arch22/chpr_host.cpp`），本任务只新增 arch35（Ascend 950PR）实现，归属同一 Hpr 算子族，不新建 spr 族、不另立 950PR 私有接口。公开接口 `include/cann_ops_blas.h:406-408` 已存在，须直接复用，禁止重复声明或定义平行 API。

## 2 需求分析（required）

### 2.1 需求描述

在 Ascend 950PR 上实现 `aclblasChpr`，行为与 cuBLAS `cublasChpr` 逐参数对齐：

```cpp
aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* ap);
```

### 2.2 需求拆解

| 编号 | 需求项 | 说明 |
| --- | --- | --- |
| R1 | packed 压缩格式 | `UPPER` 上三角按列无间隙压缩，`A(i,j)`（i ≤ j）存于 `ap[i + j(j+1)/2]`；`LOWER` 下三角按列压缩，列基址 `j(2n-j+1)/2` 加列内偏移 `i-j`。仅 `uplo` 指定三角被引用与更新 |
| R2 | Hermitian 语义 | 对角元素虚部假定为 0；**计算完成后 ap 对角虚部须置为 0**（含 alpha=0 与 x 分量为零的分支） |
| R3 | 参数校验顺序 | `n<0` → `INVALID_VALUE`；`n=0` → `SUCCESS` 且不再校验其余参数；`n>0` 依次校验 `uplo`、`incx`、`alpha/x/ap` 非空；`handle` 为空 → `HANDLE_IS_NULLPTR` |
| R4 | alpha 语义 | `n>0 && alpha=0` 时先完成参数与空指针校验，再返回 `SUCCESS` 且**不更新 ap** |
| R5 | incx 语义 | 支持任意非零 incx，含负步长（按 Netlib 语义反向遍历 x）；不支持 `incx=0` 与 `incx=INT_MIN` |
| R6 | 精度 | 实部、虚部分别按 FLOAT32 标准判定：`rtol=atol=2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2`（或 32×ULP） |
| R7 | 性能 | 平均单次耗时 ≤ 2.5× GPU 基线（即达到基线 0.4 倍以上性能），采样口径见任务书 §3.3.1 |
| R8 | 工程约束 | 实现代码置于 `blas/hpr/arch35/`，测试代码置于 `test/hpr/chpr/arch35/`；代码仓规范，无冗余文件 |

### 2.3 接口声明与参数说明

接口已在 `include/cann_ops_blas.h` 声明，本任务直接复用，不新增声明：

| 参数 | 方向 | 类型 | 说明 |
| --- | --- | --- | --- |
| `handle` | in | `aclblasHandle_t` | 算子句柄，携带 stream；为空返回 `HANDLE_IS_NULLPTR` |
| `uplo` | in | `aclblasFillMode_t` | `ACLBLAS_UPPER`（121）或 `ACLBLAS_LOWER`（122），指定存储与更新的三角 |
| `n` | in | `const int` | 矩阵阶数，`n ≥ 0`；`n = 0` 直接返回 `SUCCESS` |
| `alpha` | in | `const float*` | 实数标量（Host 内存），`*alpha = 0` 时不更新 `ap` |
| `x` | in | `const aclblasComplex*` | Device 内存，长度为 `n` 的复数向量，只读 |
| `incx` | in | `const int` | `x` 的元素步长，支持负值；`0` 与 `INT_MIN` 非法 |
| `ap` | in/out | `aclblasComplex*` | Device 内存，`n(n+1)/2` 个复数，原地读改写 |

返回值：`aclblasStatus_t`，取值 `SUCCESS` / `INVALID_VALUE` / `HANDLE_IS_NULLPTR` / `ARCH_MISMATCH`（非 arch35 设备）。

### 2.4 调用方式

```cpp
#include "cann_ops_blas.h"

aclblasHandle_t handle;
aclblasCreate(&handle);

int n = 4;
int incx = 1;
float alpha = 1.0f;
size_t nAp = (size_t)n * (n + 1) / 2;

aclblasComplex *dX = nullptr, *dAp = nullptr;
aclblasMalloc((void **)&dX,  n     * sizeof(aclblasComplex));
aclblasMalloc((void **)&dAp, nAp   * sizeof(aclblasComplex));
// ... 填充 dX / dAp（dAp 按 uplo 指定的三角 packed 存放）...

aclblasStatus_t st = aclblasChpr(handle, ACLBLAS_UPPER, n, &alpha, dX, incx, dAp);

aclblasStreamSynchronize(handle->stream);   // 异步接口，读回结果前需同步
```

若使用自定义 stream，先 `aclblasSetStream(handle, stream)`，kernel 会下发到该 stream 上，与仓内其他 blas 算子行为一致。

## 3 详细设计（required）

### 3.1 算子分析

#### 3.1.1 数学公式

对 `uplo = UPPER`，列 j 满足 `0 ≤ j < n`，行 i 满足 `0 ≤ i ≤ j`：

```
ap[i + j(j+1)/2] += alpha * x(i) * conj(x(j))
```

对 `uplo = LOWER`，列 j 满足 `0 ≤ j < n`，行 i 满足 `j ≤ i < n`：

```
ap[j(2n-j+1)/2 + (i-j)] += alpha * x(i) * conj(x(j))
```

复数展开（记 `x(k) = xr_k + i·xi_k`，`x(j) = xr_c + i·xi_c`）：

```
Re(增量) = alpha · (xr_i · xr_c + xi_i · xi_c)
Im(增量) = alpha · (xi_i · xr_c - xr_i · xi_c)
```

实现上把列乘子并入缩放系数，一次遍历同时产出实部与虚部：

```
s = alpha · xr_c          t = -alpha · xi_c
x    = [xr_i, xi_i, ...]  （AoS 交错）
ix   = [-xi_i, xr_i, ...] （即 -i·x 的实值表示）
增量 = s · x + t · ix
     = [ s·xr_i + t·(-xi_i),  s·xi_i + t·xr_i ]
     = [ alpha(xr_i·xr_c + xi_i·xi_c),  alpha(xi_i·xr_c - xr_i·xi_c) ]
```

与上式逐项一致。

#### 3.1.2 数值稳定性设计

复数增量必须**先在寄存器内构造完整，再整体累加到 ap**：

```
increment = Muls(x, realScale) + Muls(ix, imagScale)   // 先合成
ap        = ap + increment                              // 后累加
```

若改为把两个实数乘积分别累加进 ap（`ap += s·x` 后 `ap += t·ix`），在大数相反符号相消的场景下，中间态的 ap 会因单精度舍入而丢失原有值，导致最终结果偏离 golden。这正是自验中 `TC_FL_085`（`fill_x_extreme`，n=32，UPPER，`RANDOM_EXTREME`）在早期版本虚部被判为 0、`matched_ratio` 差 0.0013 未达 0.99 的根因；改为先合成增量后，该用例通过。

#### 3.1.3 支持数据类型

| 数据 | 类型 | 说明 |
| --- | --- | --- |
| `alpha` | `float`（FLOAT32） | 实数标量，Host 内存，与 cuBLAS 默认 pointer mode = HOST 一致 |
| `x` | `aclblasComplex`（COMPLEX64） | 实部/虚部各 float32，Device 内存，只读 |
| `ap` | `aclblasComplex`（COMPLEX64） | 实部/虚部各 float32，Device 内存，原地输入输出 |

计算全程 float32，不升精度中间态（异构计算与 golden 的舍入路径差异由 §3.5 判据吸收）。

#### 3.1.4 支持形状

| 参数 | 约束 |
| --- | --- |
| `n` | `n ≥ 0`；`n=0` 为合法 no-op |
| `x` 物理长度 | `≥ 1 + (n-1)·|incx|`（复数元素） |
| `ap` 长度 | `n(n+1)/2`（复数元素），无前导维度参数 |

### 3.2 算子实现

#### 3.2.1 host 侧设计（`blas/hpr/arch35/chpr_host.cpp`）

**（1）参数校验**

顺序严格按 §2.5：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `n == 0` → `ACLBLAS_STATUS_SUCCESS`（quick return，不校验其余参数）
4. `alpha == nullptr` → `INVALID_VALUE`
5. `uplo` 不属于 `{ACLBLAS_UPPER, ACLBLAS_LOWER}` → `INVALID_VALUE`
6. `incx == 0` 或 `incx == INT_MIN` → `INVALID_VALUE`
7. `x == nullptr` 或 `ap == nullptr` → `INVALID_VALUE`

`handle` 与 `n<0` 同时不合法时返回 `HANDLE_IS_NULLPTR`（句柄为最外层上下文，任务书 §2.4 亦将 handle 列为首个参数）；该组合在官方 CSV 中不存在，独立用例 `NullHandle` 单独覆盖 handle 为空。

**（2）alpha = 0 短路**

校验全部通过后若 `*alpha == 0.0f`，直接返回 `SUCCESS`，不发 kernel、不触碰 ap。满足 R4 的「先校验、后 no-op」。

**（3）tiling 策略**

| 字段 | 取值 | 依据 |
| --- | --- | --- |
| `n` | `uint32_t` | 阶数 |
| `useCoreNum` | `min(n, aivCoreNum)`，下限 1 | 每个 AIV 核至少承担一列，列数少于核数时不空跑 |
| `uplo` | 枚举原值（121/122） | kernel 侧单分支选择，避免模板膨胀 |
| `incx` | `int64_t` | 负步长与 INT_MAX 边界安全 |
| `alpha` | `float` | 直接标量传参，不进 GM |

不使用 tilingkey：分核只由 `n` 与核数决定，UPPER/LOWER 由运行期单分支选择，无编译期特化需求。

**（4）下发**

`chpr_kernel<<<useCoreNum, nullptr, stream>>>(x, ap, tiling)`，stream 取自 `handle->stream`，与 `aclblasSetStream` 绑定，异步执行，调用方读回前需同步 stream。

#### 3.2.2 kernel 侧设计（`blas/hpr/arch35/chpr_kernel.cpp`）

**（1）核函数类型**

`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`：纯向量计算，不涉及矩阵单元。

**（2）列分核（均衡策略）**

按 packed 存储的**列**切分，保证每个核只写自己负责的列区间，核间无写冲突、无需原子操作：

```
baseCols  = n / useCoreNum
extraCols = n % useCoreNum
cols      = baseCols + (blockIdx < extraCols ? 1 : 0)
start     = blockIdx * baseCols + min(blockIdx, extraCols)
end       = start + cols
```

前 `extraCols` 个核各多承担一列，列数与计算量近似均衡。UPPER 的列 j 含 `j+1` 个元素、LOWER 的列 j 含 `n-j` 个元素，因此 UPPER 前段列更轻、LOWER 前段列更重；采用连续列区间而非交错分配，使每个核访问的 ap 区间连续，利于合并访存。

**（3）UB 内存布局**

按 192 KB UB 规划，三分区等长 + 三块 256 B 标量暂存：

```
ixLocal   @ 0
xLocal    @ 2·BUF_COMPLEX
apLocal   @ 4·BUF_COMPLEX
scratchX  @ UB_FLOATS
scratchIX @ UB_FLOATS + 64
apHead    @ UB_FLOATS + 128
```

其中 `UB_FLOATS = 192KB / 4`，`BUF_COMPLEX = (UB_FLOATS / 6) & ~31`（复数元素数，32 元素对齐）。三块 scratch 各 256 B，因为 `LoadAlign` 读满 256 B 向量且 `Muls` 目标需 32 B 对齐，列头不足 4 行时须借独立缓冲拼接。`static_assert` 校验分区不重叠。

**（4）x 与 ix 的加载分档**

`ix = -i·x` 的构造方式随规模与步长分五档，避免为小规模付出大规模开销、也为大规模避免 UB 溢出：

| 档 | 触发条件 | 做法 |
| --- | --- | --- |
| `LoadXIxUnitStride` | `incx == 1` 且规模允许 | x 与 x 右移一个 float 的一维拷贝在 UB 内拼出 `[-xi, xr]`，无需跨步搬运 |
| `LoadXIxScaler` | 单列/极窄行块 | 标量逐元素构造 |
| `LoadXIxStridedLe2048` | 跨步行块 ≤ 2048 | `DataCopyPad` 分块跨步搬运 + 就地取负/交换 |
| `LoadXIxStridedLe4096` | 2048 < 行块 ≤ 4096 | 分两段构造，借用 `apLocal` 作临时区 |
| `LoadXIxStridedLe8192` | 行块更大 | 分段 + 双缓冲复用，控制 UB 峰值 |

分档只影响搬运动作，不影响数值路径，因此各档结果一致（由定向用例 `D01`–`D28` 覆盖各档交接点）。

**（5）UPPER 路径（`ProcessUpperCols`）**

外层遍历行块（步长 `ROW_TILE_STEP = BUF_COMPLEX`），内层遍历本核负责的列。对列 j，行范围为 `[rowTileStart, min(j+1, rowTileEnd))`：

- 末元素是否为对角：`col < rowTileEnd` 说明该块包含第 col 行，即对角元素在块末
- 用 `CopyInAos(..., skipLastFloat = skipDiagFloat)` 搬入时跳过对角虚部（该值不使用）
- 计算后若含对角，经 `V→S→MTE3` 用 `SetValue` 把对角虚部写为 `+0.0f`（即使公式产出 NaN/Inf 也强制归零，满足 R2）
- 列首值 `xrCol/xiCol` 若落在本行块内从 UB 取，否则从 GM 单点读，避免为单列再搬一次

**（6）LOWER 路径（`ProcessLowerCols`）**

与 UPPER 对称，行范围 `[col, rowTileEnd)`。额外处理列头对齐：

```
delta = col - rowTileStart
head  = (4 - (delta & 3)) & 3      // 至多 3 行未对齐头
bulk  = chunk - head                // 4 行对齐主体
```

列头经 `scratchX/scratchIX/apHead` 以 32 B 对齐方式单独 `Axpy`，主体一次向量化完成。这样既满足对齐要求，又不把整列退化到标量路径。`head > 0` 蕴含 `delta > 0`，此时列头必然是下三角的首元素（对角），由后续 `V→S` 保证 scratch 在下一列复用前已完成读取。

**（7）同步与事件**

用固定事件 ID `PIPE_EVT`，按数据依赖逐段设置：

| 边 | 作用 |
| --- | --- |
| `MTE2→V` | 搬入的 ap 可被向量单元读取 |
| `V→S` | 向量结果可被标量 `SetValue` 覆盖（对角虚部归零） |
| `S→MTE3` | 标量写对后搬出 |
| `V→MTE3` | 无需标量修正时的搬出 |
| `MTE3→MTE2` | **每列末必须等本列搬出完成**，否则下一列 `CopyInAos` 的 MTE2 写与上一列的 MTE3 读竞争同一片 `apLocal` |
| `S→MTE2` / `MTE2→S` | LOWER 路径列头 scratch 的复用保护 |

**（8）零列分支**

若某列 `x(j) = 0`（实虚部同为 0），增量恒为 0，但 Netlib `CHPR` 的 ELSE 分支仍要求 `ap(diag) = REAL(ap(diag))`。实现只写对角虚部为 0，其余字节不读不写、不参与秩 1 更新。该分支同时省掉一次向量计算。

### 3.3 性能优化方案

本算子的性能瓶颈不在算力而在**访存与流水**：packed 布局使同一列的行区间连续、但相邻列的起点不同（UPPER 列首地址按 `j(j+1)/2` 递增，LOWER 按 `j(2n-j+1)/2` 递增），因此列内可向量化、列间天然不对齐。优化围绕四点展开：

**（1）列内连续访存，一列一次搬入搬出**

UPPER 的列 j 行区间 `[0, j+1)` 与 LOWER 的列 j 行区间 `[j, n)` 在 packed 向量里都是**连续区间**，因此每列只需一次 `DataCopyPad` 搬入、一次向量计算、一次搬出，不产生跨列拼接。列与列之间用 `MTE3→MTE2` 事件串行化，保证同一个 `apLocal` 缓冲安全复用，避免额外 ping-pong 缓冲占用 UB。

**（2）x 与 ix 一次加载、多列复用**

对第 j 列，`x(i)` 的取值与列无关，只有列首标量 `x(j)` 变化。因此按**行块**（`ROW_TILE_STEP = BUF_COMPLEX`）加载一次 x 与 ix，块内所有列共用，把 `O(n²/2)` 次元素读取降为 `O(n)` 次搬运。列首标量若不在当前行块内，用 `GlobalTensor::GetValue` 单点取两个 float，不为一个元素再搬一次。

**（3）ix 的构造按规模与步长分五档，避免"为小规模付大规模开销"**

复数乘需要 `ix = -i·x`。这个派生量若一律走跨步搬运，`incx=1` 时也要付一次带 stride 的 `DataCopyPad`；若一律标量拼装，大 n 时会退化成逐元素写 UB。因此按触发条件分五档（见 §3.2.2（4））：`incx=1` 走"整体搬入 + UB 内右移一个 float"的零跨步方案；小规模走标量；跨步按 2048 / 4096 / 更大分三段，段内用 `DataCopyPad` 的 blockCount/blockLen/stride 一次描述多行，避免逐行循环。分档只影响搬运动作，数值路径完全一致。

**（4）LOWER 列头拆分，保住主体对齐**

LOWER 的列 j 起始行是 j，落在 UB 里的偏移 `delta = j - rowTileStart` 往往不是 4 的倍数，而 `Axpy`/`LoadAlign` 要求 32 B（8 个 float）对齐。若整列退化为标量处理，大 n 时损失显著。实现把每列切成至多 3 行的未对齐头 + 4 行对齐的主体：头部经 256 B 的 `scratchX/scratchIX/apHead` 中转，主体一次向量化。头部开销上界固定为 3 行，与 n 无关。

**（5）同步归并，减少流水气泡**

同一批内的两次搬入（如列头与主体）背靠背下发、共用一次 `MTE2→V` 等待；两次搬出共用一次 `MTE3→MTE2` 等待。这样在列内把流水等待次数从"每次搬运一次"降到"每批一次"。

**（6）计算链保持寄存器内闭合**

`increment = s·x + t·ix` 全部在向量寄存器内完成，只有最终结果落 UB 并搬出。除了保证 §3.1.2 的数值稳定性，也避免了中间量在 UB 与寄存器之间往返。

上述优化使 6 条先行例（n = 1 / 256 / 512 / 1024 / 2048 / 4096）的 R 值达到 0.741 ~ 5.649（门槛 0.4），即 NPU 耗时最差也只有 GPU 基线的 1.35 倍、最好快 5.6 倍。

### 3.4 支持硬件

| 芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |
| Atlas A3 训练/推理系列产品（arch22） | 仓内既有实现，本次不改动 |
| Atlas A2 训练/推理系列产品（arch22） | 仓内既有实现，本次不改动 |

### 3.5 算子约束限制

1. `ap` 无前导维度参数，调用方须保证实际分配长度不小于 `n(n+1)/2` 个复数，实现不做越界检查。
2. 仅 `uplo` 指定三角被引用与更新；未存储的另一半不作任何读写。
3. 输入 `ap` 的对角元素虚部视为 0；输出强制为 `+0.0f`（含 NaN/Inf 输入场景）。
4. 不支持 `incx = 0` 与 `incx = INT_MIN`；不支持超出 `incx` 语义的非连续内存访问。
5. 不涉及 broadcast、dynamic shape、确定性计算特殊要求；`ap` 原地更新，不返回视图。
6. `n` 上限受 `ap` 分配长度与 UB 分块策略约束，UB 侧按行块循环处理，不因 `n` 增大而增大 UB 占用。

### 3.6 可维可测分析

**日志**：host 侧 `OP_LOGD` 打印 tiling（n / uplo / incx / cores / alpha），`OP_LOGI` 打印 block 与核数；参数不合法时 `OP_LOGE` 打印具体参数名与取值，便于定位调用方错误。

**符号唯一性**：`nm -D --defined-only libops_blas.so` 中 `aclblasChpr` 计数应为 1，可直接验证未重复定义或引入私有平行接口。

**测试可定位性**：测试实例名使用 CSV 原始 `case_name`（如 `TC_FL_085`），失败可逐条回溯到用例；补充用例使用独立前缀（`D`/`S`/`Y`），不占用官方实例名。

**结构化自检**：除数值比对外，附加检查对角虚部为 0、输入 `x` 字节不变、ap 前后哨兵不变，用于捕获越界写与原地污染。

**性能可复现**：性能采集入口与判定脚本随测试代码提供，输出 `performance.csv`、`samples.csv`、`execution-manifest.csv`、`validation.json` 与身份哈希；证据缺失或样本数不符时标 `INCOMPLETE`，不补值、不择优。

### 3.7 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32：`rtol = atol = 2^-13`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2`（或 32×ULP）；非有限值按 NaN/Inf 等效规则独立判定 | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | 平均单次耗时 ≤ 2.5× GPU 基线（等效 R = gpu/Avg ≥ 0.4）；msprof `Task Duration(us)`，warmup 5 次不计入，有效样本 10 次算术平均 | 任务书 §3.3 |

### 3.8 兼容性分析

本任务为已有接口 `aclblasChpr` 新增 arch35 实现，接口签名、参数语义、状态码均沿用仓内既有声明，不改动 `include/cann_ops_blas.h`，不影响 arch22 及其他产品线实现，不涉及兼容性破坏。`blas/hpr/README.md` 的产品支持表由「Ascend 950PR / Ascend 950DT：不支持」更新为「Ascend 950PR：支持（arch35）」与「Ascend 950DT：本次未验证，暂不声明支持」，如实反映验证范围。
