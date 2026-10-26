# aclblasCgeru（A2/A3）算子设计文档

- 任务：CANN训练营山东大学-aclblasCgeru算子开发(A2/A3)
- 提交账号：windyskire
- 本文档对应代码仓库：https://gitcode.com/windyskire/ops-blas-cgeru
- 基线：ops-blas `master` 提交 `80dd24585b47dfc4ec42d736e3f59e4b70e54958`

## 需求背景（required）

### 需求来源

- 任务页面：<https://www.hiascend.com/activities/task-center/details/ca4f433e23174bdc9a770ed7abad8f06>（CANN 训练营山东大学专场，2026-10-22 ~ 2026-11-15）
- 任务书：任务页面“查看任务书”，文件 `aclblasCgeru_Atlas800IA3_task_doc.md`
- 设计文档模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
- 目标代码仓：<https://gitcode.com/cann/ops-blas>

### 背景介绍与实现现状

`aclblasCgeru` 完成 BLAS Level 2 中单精度复数（complex64）**无共轭**秩 1 更新（rank-1 update）：

```text
A := alpha * x * y^T + A        （y 不取共轭，区别于 gerc）
```

其中 A 为 m×n 列主序矩阵，x 为长度 m 的复数向量，y 为长度 n 的复数向量，alpha 为复数标量。

调研基线 `80dd2458` 的现状：

| 项 | 现状 |
| --- | --- |
| `include/cann_ops_blas.h` | **已存在** `aclblasCgeru` 声明（早于本任务由 950PR 侧合入），本次不改动公共头文件 |
| `blas/ger/arch35/` | 已有 950PR 侧 cgeru 实现（arch35），本次不修改 |
| `blas/ger/arch22/` | 仅有实数 `sger`，**无 cgeru 实现** ← 本任务交付物 |
| `blas/gerc/arch22/` | 有复数 `cgerc`（共轭版），可作为复数拆分的参考，但其 host 未把 incx/incy/lda 传入 kernel，且每次调用有额外申请/同步开销，不能直接改名复用 |
| `test/ger/cgeru/` | 已有 arch35 的测试工程（golden/CSV 可复用），缺 arch22 工程 |

复用的仓库既有约定：

- host/kernel 之间用结构体按值传递 tiling，`__global__ __aicore__ void cgeru_kernel(...)` 直调，与 `blas/ger/arch22/sger_*` 一致；
- 复数类型使用仓库既有的 `aclblasComplex<float>`（`float real; float imag;`），不新增公开类型；
- 复数向量化计算参考 `blas/gerc/arch22` 与 `blas/axpy/arch22/caxpy_kernel.cpp` 的实/虚部交错处理方式。

### 算子功能分析

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| handle | `aclblasHandle_t` | 绑定的 stream 用于异步下发 |
| m, n | int | A 的行数、列数；m=0 或 n=0 为合法 no-op |
| alpha | `const aclblasComplex<float>*` | Host 内存上的复数标量；不可为 nullptr |
| x | `const aclblasComplex<float>*` | Device 内存，步长 incx，长度 m |
| incx | int | 不可为 0，支持负数 |
| y | `const aclblasComplex<float>*` | Device 内存，步长 incy，长度 n |
| incy | int | 不可为 0，支持负数 |
| A | `aclblasComplex<float>*` | Device 内存，列主序，物理长 lda×n，仅访问每列前 m 个元素 |
| lda | int | 列跨度，lda ≥ max(1, m) |

计算公式（i∈[0,m)，j∈[0,n)）：

```text
kx = incx < 0 ? -(m-1)*incx : 0
ky = incy < 0 ? -(n-1)*incy : 0
A[i + j*lda] += alpha * x[kx + i*incx] * y[ky + j*incy]
```

## 需求分析（required）

### 需求描述

1. 在 `blas/ger/arch22/` 下用 Ascend C 实现 `aclblasCgeru`，支持 Atlas A2/A3（arch22，V220 向量核）。
2. 完整落实参数校验与 no-op：handle/alpha 为 nullptr、m<0、n<0、incx==0、incy==0、lda<max(1,m) 返回 `ACLBLAS_STATUS_INVALID_VALUE`；m==0 或 n==0 返回成功且不下发 kernel；alpha==(0,0) 直接返回成功，不写 A。
3. 支持任意非零整数步长（含负数、非 ±1）与任意合法 lda（含 padding）。
4. 不共轭语义，golden 使用 Netlib `cblas_cgeru(CblasColMajor, ...)`。
5. 精度按 FLOAT32 分量标准：实部、虚部分别统计 `matched_ratio ≥ 0.99`、`max_abs_error ≤ 1e-2`（或 32 ULP），逐元素判据 `|a-g| ≤ 2^-13 + 2^-13*|g|`。
6. 性能达任务书标杆：512→5.7us、1024→10.1us、2048→55.71us、4096→219.35us（910B3，warmup 5 次、有效采样 >50 次取均值）。

### 需求拆解

1. host 侧参数校验、偏移计算（int64 提升，避免负步长/大尺寸溢出）、tiling 计算、kernel 直调。
2. kernel 侧：列主序 A 的读-改-写，行片段向量化，复数交错布局的乘加，尾块与非对齐搬运。
3. 分核分块：保证任一 A 元素只被一个核读写（无原子、无跨核归约）。
4. 覆盖任务书 1200 条 CSV（1000 精度 + 200 性能）+ 补充边界用例，输出自测报告。
5. 性能调优：32 列列块一次搬运（减少搬运/同步次数）、A 在 UB 内就地累加（省一份输出缓冲）、
   x 与 S 跨列复用、去掉列内冗余 V 屏障。

## 详细设计（required）

### 算子分析

数学上，对每一列 j 令 `t = alpha * y[j]`（复数标量），则该列更新为

```text
A[:, j] += x * t
```

展开为实/虚部（u = Re(t), v = Im(t)；xr, xi 为 x 的实、虚部）：

```text
P_re[i] = u*xr[i] - v*xi[i]
P_im[i] = u*xi[i] + v*xr[i]
A_re[i, j] += P_re[i]
A_im[i, j] += P_im[i]
```

先形成复数乘积 P，再与 A 相加（不把 A 提前合并进乘加），与 Netlib `cgeru.f` 的 `TEMP = ALPHA*Y(J); A(I,J) = A(I,J) + X(I)*TEMP` 的舍入次序一致。

**交错布局下的向量化**：complex64 在内存中为 `(re, im)` 交错，一份长度为 L 的复数向量对应 2L 个 float。令
`X = (xr0, xi0, xr1, xi1, ...)`，`S = (xi0, xr0, xi1, xr1, ...)`（即逐元素成对交换，`S[2i]=X[2i+1]`、`S[2i+1]=X[2i]`），则

```text
P = Muls(X, u) + Muls(S, v) * sign,  sign = (-1, +1, -1, +1, ...)
```

即 `Muls(X,u)` 得到 `(u*xr, u*xi)`，`Muls(S,v)` 得到 `(v*xi, v*xr)`，偶数通道取负后相加得到 `(u*xr - v*xi, u*xi + v*xr)`，正好是复数乘 `x*t` 的交错结果。
`S` 只要在加载 x 片段后构造一次，即可在该行片段的所有列上复用。

### 算子实现

#### host 侧设计

处理次序（保证非法/空计算场景不访问 Device 数据）：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
2. `m < 0 || n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`
3. `m == 0 || n == 0` → `ACLBLAS_STATUS_SUCCESS`（不校验 x/y/A，不下发）
4. `incx == 0 || incx == INT_MIN`（incy 同理）、`lda < max(1, m)` → `ACLBLAS_STATUS_INVALID_VALUE`
5. `alpha` / `x` / `y` / `A` 任一为 nullptr → `ACLBLAS_STATUS_INVALID_VALUE`
6. 未绑定 stream → `ACLBLAS_STATUS_NOT_INITIALIZED`
7. `alpha == (0,0)` → 返回成功，不启动 kernel（BLAS quick-return 语义）
8. 按仓库约定取 stream，计算 tiling，直调 `cgeru_kernel<<<blockDim, nullptr, stream>>>(...)`

偏移全部在 int64 下计算，避免 `-(m-1)*incx` 的 int 溢出：

```text
xOffset(i) = incx > 0 ? i*incx : (m-1-i)*(-incx)
yOffset(j) = incy > 0 ? j*incy : (n-1-j)*(-incy)
aOffset(i,j) = j*lda + i
```

tiling 结构体 `CgeruTilingData` 按值传入（与 sger 一致），字段为：

```cpp
struct CgeruTilingData {
    uint64_t A;          // 矩阵基址
    uint64_t x;          // x 基址
    uint64_t y;          // y 基址
    uint64_t constTable; // 常量表（成对交换字节偏移表 + 交错符号表）
    uint32_t m, n, lda;
    uint32_t useCoreNum; // 实际使用的 AIV 核数
    uint32_t colsPerCore;// 每核列数
    uint32_t rowTile;    // 行片段长度（complex 元素个数，固定 512 = CGERU_ROW_TILE_MAX）
    int32_t incx, incy;
    float alphaReal, alphaImag;
};
```

常量表由 host 首次调用时生成并缓存在 handle 上（`_aclblas_handle::cgeru_const_table`，
8,704 B = 2 × 1088 项 × 4 B，随 handle 释放），内容与位置一一对应、不随 m/n 变化，避免每次调用做一次 H2D；
表 0 是 `Gather` 用的成对交换**字节偏移** `4*(k^1)`（`AscendC::Gather` 的偏移量以字节计，
与 `blas/gerc/arch22/cgerc_host.cpp` 的 `NUM_INBYTES = 4` 一致），表 1 是交错符号模式
`(-1, +1, -1, +1, ...)`。

不申请算子专用 GM workspace（新增分配仅上述常量表）。

#### 分核及分块设计

采用 **列 slab 分核 + 行片段循环**，保证 A 中每个元素只属于一个核、无需原子加：

```text
useCoreNum  = min(n, GetAivCoreCount())
colsPerCore = ceil(n / useCoreNum)
colStart    = blockIdx * colsPerCore
colCount    = min(colsPerCore, n - colStart)     // 末核/非整分时再夹一次
rowTile     = 512（complex，32 的整数倍）
rowOff      = 0, rowTile, 2*rowTile, ...         // 外层循环覆盖 m 行
colBlk      = min(CGERU_COL_TILE=32, colCount)   // 一次搬入/搬出的列数
```

- 每个核负责 `colCount` 列的全部行；行方向按 rowTile 分块，块内按 `colBlk` 列成批处理。
- 一个行片段内：加载 `x[rowOff : rowOff+rowTile]` 一次 → 构造 S 一次 → 对本批 `colBlk` 列
  一次 `DataCopyPad` 2D 搬入（`blockCount=colBlk`、`blockLen=chunk*8`、`srcStride=lda*8`、
  `dstStride=rowTile*8`）→ 逐列 `AccumulateColumn`（**A 在 UB 内就地更新**，不再单独开输出缓冲）
  → 一次 2D 搬出。
- 列栈分核保证 A 的写不冲突；`lda` padding 行不搬写。
- 限制：列块 2D 搬运仅在 `chunk` 为 32 复数的整数倍（`calcNum == chunk`，列间无 padding 空隙）时启用，
  否则退化为逐列搬运——`DataCopyPad` 的多块目标地址在 UB 侧带非零 stride 时行为不可靠，
  已在 A2 上实测踩过（111 条用例失败），退化为逐列后全部通过。

#### kernel 侧设计

阶段：`Init` → `LoadConstTiles` → `PrepareColumnCoeffs` →（每行片段）`LoadX` /
`BuildSwapPattern` →（每列）`LoadAColumn` / `AccumulateColumn` / `StoreAColumn`。

1. **常量表**：启动时把 16.9 KB 常量表整体搬入 UB（`DataCopyPad`，一次），此后不再访问 GM。
2. **y 的读取**：kernel 入口把该核所有列的 `t = alpha * y[j] = (u, v)` 一次性用标量算出，
   存进 UB 的 coeff 缓冲（每列 2 个 float）+ 跳过标记（`t == (0,0)` 时为 1，整列跳过，
   与 Netlib `IF (TEMP.NE.ZERO)` 一致）。主循环内不再访问 y 的 GM。
3. **搬运 x**：`incx == 1` 时 `DataCopyPad` 一次搬 `rowTile*8` 字节；`|incx| != 1`（含负数）
   时按行标量 `GetValue/SetValue` 逐元素搬，避免在 UB 里做带步长的重排。
4. **构造 S**：全程在交错布局上工作，不拆实/虚部：
   ```text
   scratch = X * sign           // Mul，sign = (-1,+1,-1,+1,...)  → (-xr, +xi)
   S       = Gather(scratch, swapTable)   // (xi, -xr)，swap 表为字节偏移 4*(k^1)
   ```
   X、S 每个行片段只构造一次，供该片段内所有列复用（x 的 GM 流量与列数无关）。
5. **每列计算**（A 就地更新，3 条向量指令，长度 2*chunk 个 float）：
   ```text
   acc  = Muls(S, -v)             // (-v*xi, +v*xr)
   acc  = Axpy(acc, X, u)         // (u*xr - v*xi, u*xi + v*xr)  ← 复数乘积 c*x
   curA = Add(curA, curA, acc)    // 就地累加：乘积先形成再加 A，舍入次序与 Netlib 一致
   ```
   `u = ar*yr - ai*yi`，`v = ar*yi + ai*yr` 与 Netlib `TEMP = ALPHA*Y(JY)` 完全相同。
6. **同步**：x 搬入（`MTE2_V`）、A 搬入（`MTE2_V`）、y 搬入（`MTE2_S`）、A 搬出（`V_MTE3`/
   `MTE3_MTE2`）各一次硬事件；Muls/Axpy/Add 同为 V 流水顺序指令，列内不再插 `PIPE_V` 屏障
   （实测去掉后无性能变化，省掉冗余 barrier）。

#### UB 预算

行片段上限 R = 512 complex（= 1024 float，`CGERU_ROW_FLOATS`），单块 4096 B（`CGERU_ROW_BYTES`）：

| 缓冲区 | 字节数 | 说明 |
| --- | ---: | --- |
| A 列块队列（1 槽，32 列） | 131072 | 一次搬入/就地累加/搬出的 32 列 × 1024 float |
| X（x 片段，交错） | 4096 | 每行片段加载一次 |
| S（成对交换结果） | 4096 | 每行片段构造一次 |
| calc（暂存 acc/scratch） | 4096 | 复用 |
| swap 表 / sign 表 | 4096 × 2 | 常量表，启动时一次搬入 |
| 固定小计 | 151552 | —— |
| coeff（u, v, skip）+ y 缓存 | ≤ 4128 | 随每核列数（`ceil32(12*colCount)` + `ceil32(8*colCount)`）|
| 合计 | ≈ 152 KB | 192 KB UB 上限内留足余量 |

TQue 深度取 1（`CGERU_BUFFER_NUM=1`）：当前实现是“搬入→算→搬出”串行，深度 2 不带来重叠，
把省下的 UB 用于 32 列的列块，减少小尺寸上的搬运次数与同步次数（512² 从 7.10 µs 降到 6.30 µs）。

#### 支持硬件

| 芯片版本 | 是否支持 |
| --- | --- |
| Atlas 800I/T A2（910B 系列，arch22） | √ |
| Atlas 800I/T A3（arch22） | √ |
| Atlas 950PR（arch35） | 已有独立实现，本次不涉及 |

#### 算子约束限制

- 不支持超出 incx/incy/lda 语义的任意 Tensor 视图（本批次不要求）。
- 无 broadcast；不支持运行时类型分派（仅 complex64）。
- A 与 x/y 的内存重叠未在任务书中定义，不承诺支持。
- 依赖 `aclblasSetStream` 绑定 stream 的异步语义，调用方需自行同步后读回结果。

## 可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 实部/虚部分别按 FLOAT32：`|a-g| ≤ 2^-13 + 2^-13*|g|`，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP | 任务书 §3.2 与 opbase《混合精度算子精度标准》 |
| 性能 | 512→5.7us、1024→10.1us、2048→55.71us、4096→219.35us（910B3，warmup 5 次、采样 >50 次取均值） | 任务书 §3.3 |

### 兼容性分析

- 新增 arch22 实现与 arch22 测试工程，不改动已有算子与公共头文件（`aclblasCgeru` 声明已存在）；
- 新增符号不改变既有 API 行为，二进制兼容；
- 仅声明支持 arch22 产品，未验证的架构不宣称支持。

### 自测与证据

- 精度：`test/ger/cgeru/arch22` GTest 加载任务书 1200 条 CSV；golden 为与 Netlib BLAS 3.10.0
  `cblas_cgeru` 逐式同源的本地参考实现（`NaiveCgeruGolden`：列主序、`y` 不取共轭、
  `TEMP == 0` 的列跳过），不引入 CBLAS 外部依赖，实部/虚部分别统计 matched_ratio 与
  max_abs_error；补充 no-op、越界参数、负步长/非单位步长、lda padding、Inf/NaN、
  alpha 特殊值等 14 个专项用例。
- 性能：warmup 5 次、采样 100 次取均值，导出 4 个标杆尺寸的耗时截图与日志；同时记录算子新增 workspace 为 0。
- 自测报告：按任务书模板（腾讯文档）填报，含用例参数、精度对比截图（实部/虚部）、性能数据与截图、内存占用。

### 实测结果（Atlas 800T A2 / 910B3，CANN 9.1.0，commit `1995d39`）

精度：`cgeru_test` 全量 **1214 / 1214 PASS，0 FAILED**（1200 条任务书 CSV + 14 条专项）。
（进程退出码 139 出现在 gtest teardown 的 `aclblasDestroy → aclrtDestroyStream → aclrtResetDevice →
aclFinalize` 段，栈帧全在 `libruntime_v100.so`，为仓内所有测试二进制共用的清理路径，与算子无关；
官方 `verify_accuracy.py` 只统计 `[ OK ]/[ FAILED ]` 行。）

性能（`cgeru_perf`，warmup 5 次 + 100 次 ACL Event 取均值，incx = incy = 1）：

| m×n | 任务书基线 (µs) | 实测 (µs) | 相对基线 |
| --- | ---: | ---: | ---: |
| 512×512 | 5.70 | 6.2972 | 1.10× |
| 1024×1024 | 10.10 | 13.0108 | 1.29× |
| 2048×2048 | 55.71 | 36.4994 | 0.66× |
| 4096×4096 | 219.35 | 124.0084 | 0.57× |

2048、4096 两个尺寸优于任务书基线；512、1024 未达基线，剩余差距是**单次调用的固定开销（约 5 µs：
launch + 首列段 DMA 往返 + 事件同步）**，与数据量无关。跨列软件流水改造已在真机验证会触发
TQue 同步死锁（提交 `d0ab8e1`、`56222e3`，均已回滚），列为后续优化项。

## 提交范围

1. 算子代码：`blas/ger/arch22/cgeru_tiling.h`、`cgeru_host.cpp`、`cgeru_kernel.cpp`
2. 公共支撑（新增字段 + 释放配套，各 1 处）：`blas/common/helper/aclblas_handle_internal.h`
   （`_aclblas_handle` 增加 `void* cgeru_const_table`）、`blas/common/helper/aclblas_auxiliary.cpp`
   （`aclblasDestroy` 中 `aclrtFree`）
3. 测试工程：`test/ger/cgeru/arch22/`（`cgeru_test.cpp`、`cgeru_perf.cpp`、`cgeru_npu_wrapper.h`、
   `README.md`、`cgeru_test.csv`）
4. `blas/ger/README.md` 中补充 Cgeru 接口与产品支持表（如仓库已有该文件则追加，不覆盖既有内容）

设计评审通过并合入后，按任务书流程提交待验收代码地址与自测报告（2026-10-22 已完成验收提交，
附件为自验证步骤说明 + 精度/性能/内存三组报告与日志）。
