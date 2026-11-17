# 需求背景（required）

## 需求来源

本需求来源于 2026 年 11 月 CANN 社区任务《aclblasCdgmm 算子开发（950）》（华为昇腾社区任务中心任务 ID：dcfe51817ba0422a90b643f3e611cfb7），目标仓库为 `cann/ops-blas`，目标硬件为 Atlas 950PR（arch35），目标软件版本为 CANN 9.1.0。

任务要求在 `ops-blas` 仓 `blas/dgmm/arch35/` 目录下，使用 Ascend C 编程语言实现单精度复数（complex64）对角矩阵缩放接口 `aclblasCdgmm` 的 Host 与 Kernel 代码，并配套搭建 `test/dgmm/cdgmm/arch35/` 测试工程，完成精度与性能自测后申请合入。

本文档以任务书、社区设计文档模板以及 `ops-blas` 当前源码为依据，描述接口语义、实现方案、测试方案和验收标准。

本文档设计基线如下：

| 项目 | 基线 |
| --- | --- |
| 目标仓库 | `cann/ops-blas` |
| 源码基线 | `25e4a37`（个人仓 `2402_89190804/ops-blas` fork 点） |
| 目标产品 | Atlas 950PR（arch35，NPU_ARCH dav-3510）；Atlas 950DT 同架构适用 |
| CANN 版本 | 9.1.0（asc-devkit >= 9.1） |
| 算子接口 | `aclblasCdgmm`（沿用 `include/cann_ops_blas.h` 已有声明，不修改声明） |
| 数据类型 | `aclblasComplex`（实部/虚部各 FP32，即 complex64，定义于 `include/cann_ops_blas_common.h`） |
| 数据布局 | 列主序（column-major） |
| 计算方式 | Ascend C Kernel 直调，异步提交到 handle 当前 stream |

## 背景介绍

DGMM（Diagonal Matrix-Matrix Multiplication）用向量 x 构造对角矩阵，按 mode 对矩阵 A 的行或列做逐元素缩放，是 BLAS 扩展接口中一类轻量但高频的算子，常用于机器学习中的行/列归一化、attention 缩放、复数信号处理等场景。

`ops-blas` 已提供实数同族算子 `aclblasSdgmm` 的 arch35 实现（`blas/dgmm/arch35/`，列主序、LEFT/RIGHT、正负 incx），作为本任务的同族模板；`aclblasCdgmm` 的声明已存在于 `include/cann_ops_blas.h`，现由 arch22（Atlas A2/A3）行主序、仅 LEFT 的旧实现使用。本任务在 arch35 上新增复数实现，语义与 cuBLAS `cublasCdgmm` 完全对齐：

- 列主序存储：A(i,j) 位于 `A[i + j·lda]`，C(i,j) 位于 `C[i + j·ldc]`，`lda/ldc >= max(1, m)`；
- 同时支持 LEFT 与 RIGHT 两种 mode；
- incx 允许负数（从 x 尾部反向访问），仅拒绝 incx == 0；
- 非法 mode 返回 `ACLBLAS_STATUS_INVALID_VALUE`（与 cuBLAS 对齐，不使用 INVALID_ENUM）。

复数版本不能简单复用实数 `aclblasSdgmm`：复数乘法按 `(ar·xr − ai·xi, ar·xi + ai·xr)` 展开，实部/虚部存在耦合，kernel 需要将 interleaved 复数拆分为实部/虚部两个平面计算后再交织写回。arch22 的 `blas/dgmm/arch22/cdgmm*` 为行主序、仅 LEFT 的旧产品线实现，按任务书要求不得参考、复用或照抄。

# 需求分析（required）

## 需求描述

在 `blas/dgmm/arch35/` 下新增 `aclblasCdgmm` 的 arch35 实现，接口签名沿用仓内已有声明（与 cuBLAS `cublasCdgmm` 逐参数对应，与同族 `aclblasSdgmm` 同构）：

```cpp
aclblasStatus_t aclblasCdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    aclblasComplex* C, int ldc);
```

计算语义（列主序）：

```
LEFT （ACLBLAS_SIDE_LEFT） ： C(i,j) = x(idx(i)) · A(i,j)
RIGHT（ACLBLAS_SIDE_RIGHT）： C(i,j) = A(i,j) · x(idx(j))
idx(k) = k·incx            （incx > 0）
       = (L−1−k)·|incx|    （incx < 0），L = m（LEFT）/ n（RIGHT）
```

复数乘法按 `(ar·xr − ai·xi, ar·xi + ai·xr)` 展开，乘加均按 float32 计算；A、C 为列主序 m×n complex64 矩阵，x 在 LEFT 时长度为 m、RIGHT 时长度为 n，第 k 个逻辑元素按 idx(k) 定位，0 ≤ i < m，0 ≤ j < n。A 与 C 各自按 lda、ldc 定位列首地址，分别占用 lda×n、ldc×n 存储，lda、ldc 的 padding 行不参与运算也不得被写。

参数定义与任务书 §2.4 保持一致：

| 参数 | 内存位置 | 属性 | 类型/合法值 | Shape/约束 | 异常行为 |
| --- | --- | --- | --- | --- | --- |
| `handle` | Host | 输入 | 已创建的 `aclblasHandle_t` | 绑定当前执行 stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `mode` | Host | 属性 | `ACLBLAS_SIDE_LEFT(141)`、`ACLBLAS_SIDE_RIGHT(142)` | 决定对角缩放一侧 | 其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `m` | Host | 输入 | `int`，`m >= 0` | 矩阵 A/C 的行数 | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE`；m = 0 为合法 no-op |
| `n` | Host | 输入 | `int`，`n >= 0` | 矩阵 A/C 的列数 | 负值返回 `ACLBLAS_STATUS_INVALID_VALUE`；n = 0 为合法 no-op |
| `A` | Device | 输入、只读 | `const aclblasComplex*` | `[lda, n]` 列主序，实/虚部为 FP32 | m > 0 且 n > 0 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `lda` | Host | 输入 | `int` | `lda >= max(1, m)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `x` | Device | 输入、只读 | `const aclblasComplex*` | LEFT 长 m / RIGHT 长 n，按 incx 步长存取 | m > 0 且 n > 0 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `incx` | Host | 输入 | `int`，`incx != 0` | x 的元素存储步长，可为负数（反向访问） | incx = 0 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `C` | Device | 输出 | `aclblasComplex*` | `[ldc, n]` 列主序；可与 A 指向同一内存（原地） | m > 0 且 n > 0 时为空返回 `ACLBLAS_STATUS_INVALID_VALUE`；C == A 且 lda != ldc 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `ldc` | Host | 输入 | `int` | `ldc >= max(1, m)` | 不满足约束返回 `ACLBLAS_STATUS_INVALID_VALUE` |

接口为异步接口：计算任务提交到 handle 当前 stream，正常路径不主动同步；调用方在 Host 读回 C 或跨 stream 使用结果前，必须同步该 stream 或建立正确的事件依赖。

### 参数校验与状态码

参数检查按以下顺序执行（任务书 §2.5，前导维校验先于 m/n 为 0 的早返回）：

1. `handle == nullptr` 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `mode` 不在 `{ACLBLAS_SIDE_LEFT, ACLBLAS_SIDE_RIGHT}` 返回 `ACLBLAS_STATUS_INVALID_VALUE`（与 cuBLAS 对齐，不使用 INVALID_ENUM；注意与仓内 sdgmm 旧实现的差异）；
3. `m < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
4. `n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
5. `incx == 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
6. `lda < max(1, m)` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
7. `ldc < max(1, m)` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
8. `m == 0 || n == 0` 为合法 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`，不读 A/x/C、不下发 Kernel；
9. 非空计算时 `x == nullptr || A == nullptr || C == nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`；
10. `A == C` 且 `lda != ldc` 返回 `ACLBLAS_STATUS_INVALID_VALUE`（Host 侧拒绝，不下发 Kernel）。

Tensor 元素值不做扫描（含 Inf/NaN 的输入按 IEEE 规则参与计算）。

## 需求拆解

### 接口与语义

1. 沿用 `include/cann_ops_blas.h` 已有声明，不新增 950PR 私有平行接口，不改变 arch22 既有实现的签名；
2. 支持 LEFT / RIGHT 两种 mode、incx 正负（反向访问）、独立 lda/ldc 与 padding 行；
3. 支持 `A == C && lda == ldc` 的原地执行，输出与独立输出一致；
4. 不支持任意 stride 的非连续 Tensor、broadcast、batched 与视图返回。

### Host 端

1. 按上文顺序完成参数校验与早返回；
2. 获取 AIV core 数，按 n、m 与 UB 容量计算 2D block 分解 tiling（详见 §算子实现）；
3. 将 tiling 与原始指针打包下发，异步 launch kernel。

### Kernel 端

1. AIV 向量路径：列段按 interleaved float 对搬入 UB，`DeInterleave` 拆为实/虚平面；
2. 复数乘法用 `Mul/Muls/Axpy/Add/Sub` 在实/虚平面上展开：
   `cr = ar·xr − ai·xi`，`ci = ar·xi + ai·xr`；
3. `Interleave` 交织回 complex64 后搬出到 C；
4. mode=R 时 x 为按列标量（GetValue 取复数值），mode=L 时 x 段向量常驻 UB 跨列复用；
5. 负 incx 统一换算为非负下标访问（与 CPU golden 同一公式），x 指针不做预偏移。

### 测试工程

1. `test/dgmm/cdgmm/arch35/` 下新增 `cdgmm_test.cpp`、`cdgmm_npu_wrapper.h`、`cdgmm_param.h`、`cdgmm_test.csv`，golden 使用随任务提供的 `test_cases/cdgmm_golden.h`（`aclblasCdgmm_cpu`，float64 计算）拷贝至该目录并 include；
2. CSV 覆盖 1200 条用例（1000 精度 + 200 性能），补齐原地执行等 CSV 无法表达的指针别名用例；
3. 精度按 COMPLEX64=FLOAT32 档混合容差判定（实部/虚部分别比对），性能按 msprof op 采样（5 次 warmup + ≥10 次有效采样）对比 H100 基线。

### 文档

1. 更新 `blas/dgmm/README.md` 的 sdgmm/cdgmm 差异表与 `aclblasCdgmm` 产品支持表（950PR：支持）；
2. 更新 `docs/zh/api_list.md` 中 `aclblasCdgmm` 的"行主序…左乘"描述，避免文档与 950PR 实现矛盾。

# 详细设计（required）

## 算子分析

### 数学公式

```
LEFT ： C(i,j) = x(idx(i)) · A(i,j)，0 <= i < m，0 <= j < n
RIGHT： C(i,j) = A(i,j) · x(idx(j))，0 <= i < m，0 <= j < n
idx(k) = k·incx           （incx > 0）
       = (L−1−k)·|incx|   （incx < 0），L = m（LEFT）/ n（RIGHT）

复数乘法（实部/虚部各 FP32）：
    cr = ar·xr − ai·xi
    ci = ar·xi + ai·xr
```

### 支持数据类型

- `aclblasComplex`：实部/虚部各 float32（complex64），定义以 `include/cann_ops_blas_common.h` 为准；
- 全部乘加按 float32 计算，无类型转换。

### 支持形状

- 任意 `m >= 0`、`n >= 0`（m = 0 或 n = 0 为合法 no-op）；`lda/ldc >= max(1, m)`；
- 列主序，不解析任意 stride；x 按 incx 步长访问，incx 可为任意非 0 整数（含负数）。

### 算子特性

- 确定性：同一设备、同一输入、同一 tiling 下多次调用输出逐元素一致（无随机数、无归约顺序差异）；
- 异步：依赖 `aclblasSetStream` 绑定 stream，Host 不主动同步；
- 原地：C == A 且 lda == ldc 合法（按列/按块独立读写，先读后写保证正确性）；C 与 x 不复用内存；部分重叠行为未定义。

## 算子实现

### 总体结构

代码全部放在 `blas/dgmm/arch35/` 目录（与 sdgmm 同目录族），文件规划：

| 文件 | 内容 |
| --- | --- |
| `cdgmm_tiling_data.h` | Host/Kernel 共享的 tiling 结构体与 mode 归一化常量 |
| `cdgmm_host.cpp` | 参数校验、tiling 计算、kernel launch、`aclblasCdgmm` 入口 |
| `cdgmm_kernel.h` | kernel launcher 声明 |
| `cdgmm_kernel.cpp` | AIV kernel：LEFT/RIGHT 两套向量化处理流程 |
| `cdgmm_*.cpp` 编译集成 | `blas/CMakeLists.txt` 按目录 GLOB 自动收集 `blas/*/arch35/*.cpp`；在 `cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` 列表新增一行 `"CDGMM|/dgmm/arch35/cdgmm|cdgmm"`（低版本 devkit 门控与测试注册自动生效） |

### Host 侧设计

#### 3.2.1 参数校验

按 §需求分析 的顺序执行校验与早返回，非法输入不下发 Kernel，日志走仓内 `log/log.h`。

#### 3.2.2 Tiling 策略

沿用 sdgmm 的 2D block 分解（colBlocks × mBlocks），元素宽度换成 complex64（8B，按 interleaved float 对处理）：

1. **tileM（单列块的行分片粒度，单位：复数元素）**
   UB 预算 = `UB_SIZE − UB_RESERVE(256B)`。kernel 内 UB buffer 均按 float 平面计：
   - mode=L：aIn(2t)、ar(t)、ai(t)、cr(t)、ci(t)、cOut(2t)、xIn(2t)、xr(t)、xi(t)，共 12t 个 float；
   - mode=R：aIn(2t)、ar(t)、ai(t)、cr(t)、ci(t)、cOut(2t)，共 8t 个 float。
   其中 t = tileM（复数元素数），每个复数占 2 个 float。
   `tileM = floor((UB_SIZE − 256) / (bufsPerElement × 4))`，再向下取整到 8 的倍数
   （保证 32B 对齐与向量指令 8-float 重复粒度）；bufsPerElement 为每复数元素占用的
   UB float 数（mode=L 为 12，mode=R 为 8）。按 mode 分别计算，随 tiling 下发。

2. **列分解（n 维）**
   `colBlocks = min(n, aivCoreNum)`；`perCoreN = n / colBlocks`；`remainder = n − perCoreN × colBlocks`。
   前 `remainder` 个 col-block 各多处理 1 列（余数均衡分配，不堆积到最后一个 block）。

3. **行分解（m 维）**
   `mTiles = ceil(m / tileM)`。仅当 `colBlocks < aivCoreNum`（存在空闲核）且 `mTiles > 1` 时启用：
   `mBlocks = min(aivCoreNum / colBlocks, mTiles)`；`perCoreMTile = mTiles / mBlocks`；
   `mTileRemainder = mTiles − perCoreMTile × mBlocks`（前几个 m-block 各多 1 个 tile）。
   `mBlocks == 1` 时每核处理全部 m-tiles。

4. **block 索引布局**
   `blockIdx = colBlock × mBlocks + mBlock`，总 block 数 `numBlocks = colBlocks × mBlocks`。
   block 内负责 `[startCol, endCol) × [startM, endM)`：
   - col 区间按 remainder 规则：`colBlock < remainder` 时区间长 `perCoreN + 1`，否则 `perCoreN`；
   - m 区间由 `startMTile/endMTile`（同样按 mTileRemainder 规则）× tileM 得出，`endM` 截断到 m。

5. **负 incx 与指针**
   x 原始指针直接下发；kernel 内按 `idx(k)` 公式换算非负下标（与 CPU golden 一致），避免 Host 预偏移带来的负数下标与 `−incx` 溢出风险。

#### 3.2.3 下发

`cdgmm_kernel_do(x, A, C, tiling, numBlocks, stream)` 以 `<<<numBlocks, nullptr, stream>>>` 异步 launch，tiling 按值拷贝到 device。入口 `aclblasCdgmm` 从 `_aclblas_handle` 取 stream。

### Kernel 侧设计

Kernel 为 AIV-only 向量实现（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），GM 数据按 interleaved float 对（`GlobalTensor<float>`）视图访问，complex64 元素 i 对应 float 下标 2i、2i+1。

UB 流水（每列块）：

```
GM --(DataCopyPad)--> aIn[2t] --(DeInterleave)--> ar[t], ai[t]
                                                    |
                     x（标量/向量，见下）--(Mul/Muls/Add/Sub)--> cr[t], ci[t]
                                                    |
cOut[2t] <--(Interleave)-- cr, ci --(DataCopyPad)--> GM(C 列段)
```

同步采用 sdgmm 同款事件模式：搬入后 `SetFlag/WaitFlag<MTE2_V>`，向量计算后 `PipeBarrier<PIPE_V>` + `SetFlag/WaitFlag<V_MTE3>` 再搬出；跨列循环以 `V_MTE2`/`MTE3_V` 互锁复用 buffer。

#### 3.2.1 mode=R（按列标量缩放）

- 外层遍历本 block 的列 j：`xjRe/xjIm = GetValue(xIdx)`（xIdx 按 incx 符号换算，见 §需求描述）；
- 内层遍历 m-tile：
  1. `DataCopyPad` 搬 A 列段 `[j·lda + startM, j·lda + endM)`（2·curM 个 float）到 aIn；
  2. `DeInterleave(ar, ai, aIn)`；
  3. 标量复数乘（`Muls` + 标量 `Axpy`，共 4 条 V 指令）：
     `cr = Muls(ar, xjRe); Axpy(cr, ai, −xjIm)`，`ci = Muls(ar, xjIm); Axpy(ci, ai, xjRe)`；
  4. `Interleave(cOut, cr, ci)`，`DataCopyPad` 搬出到 C 列段。

#### 3.2.2 mode=L（按行向量缩放，x 段常驻 UB）

- 外层遍历 m-tile：
  1. 载入 x 段 `[startM, endM)`（2·curM 个 float）：
     - `incx == 1` 快路径：`DataCopyPad` 连续搬入；
     - 其余（含负 incx）：`GetValue/SetValue` 逐元素搬入（实/虚各一次，下标按 idx 公式换算）；
  2. `DeInterleave(xr, xi, xIn)`，一次拆解、跨全部列复用；
- 内层遍历列 j：与 mode=R 相同地搬入 A 列段、拆解，然后向量复数乘：
  `cr = Mul(ar, xr) − Mul(ai, xi)`，`ci = Mul(ar, xi) + Mul(ai, xr)`，
  交织后搬出。

#### 3.2.3 边界处理

- 尾 tile（curM < tileM）：V 指令按精确 curM 计数执行（arch35 上 DeInterleave/Interleave/Mul/Muls/Add/Sub/Axpy 均支持任意计数，仓内 caxpy/saxpy 尾块先例）；搬入用 `DataCopyPad`，blockLen = 2·curM·4 字节（精确有效数据，不越界读 GM），rightPadding 补零至 8-float 对齐（≤ 6 float = 24B，满足 DataCopyPad 的 32B 填充上限）；搬出仅写 2·curM 个 float，填充位不落 GM；
- 空区间（startCol >= endCol 或 startM >= endM）直接 return；
- 原地执行：列段"先搬入、后搬出"，同一 block 内不重叠，正确性成立。

### 构建集成

`blas/CMakeLists.txt` 的 GLOB 规则自动收集 `blas/dgmm/arch35/cdgmm_*.cpp`；`cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` 新增一行后，四个 foreach（版本门控、源过滤、测试 skip、测试注册）自动生效，无需其他 CMake 改动。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（arch35） | √ |
| Atlas 950DT（arch35） | √ |
| Atlas A2/A3（arch22，行主序旧实现） | 维持既有行为，本实现不编译到 arch22 |

> 950PR/950DT 上的 cdgmm 依赖 CANN asc-devkit >= 9.1，低于该版本时编译与运行跳过此算子（与 sdgmm 一致）。

## 算子约束限制

- 仅支持列主序 + lda/ldc 前导维访问，不支持任意 stride 的非连续矩阵；
- 不涉及 broadcast；x 与 A 的行/列按 mode 一一对应；
- 原地执行仅限 `A == C && lda == ldc`；C 与 x 复用内存不支持；部分重叠行为未定义；
- 元素值不扫描，Inf/NaN 按 IEEE 规则传播；
- 不要求 dynamic shape 能力，m、n 为运行时入参，Host 按实际维度计算 tiling。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | COMPLEX64 按 FLOAT32 档判定：rtol = atol = 2^-13（1.2e-4），required_matched_ratio >= 0.99，max_abs_error <= 1e-2 或 32×ULP；实部、虚部分别比对，仅比对 m×n 逻辑块，padding 行保持哨兵值；Inf/NaN 按任务书 §3.2 规则判定；golden 为随任务提供的 `test_cases/cdgmm_golden.h`（`aclblasCdgmm_cpu`，float64 计算） | 生态算子开源精度标准（mixed tolerance）；任务书 §3.2 |
| 性能标准 | msprof op 采样单算子 task duration：5 次 warmup + >=10 次有效采样取平均；每个性能 case 平均耗时不高于 H100 标杆的 2.5 倍（>=0.4×），全部 case 倍率算术平均 >= 0.5×；标杆为 `test_cases/gpu_baseline.csv`（H100 80GB / CUDA 13.2 / cuBLAS 13.3） | 任务书 §3.3 |

**性能可行性预估**：最大性能 case（8192×1024 LEFT / 1024×8192 RIGHT）A+C 总流量约 2×8192×1024×8B ≈ 134MB，按 950PR HBM 带宽约 3TB/s 估算搬运算耗时约 45us，低于 2.5×53.2us ≈ 133us 的预算；小 case（1×1 ~ 128×128）标杆约 10us（H100 launch 主导），预算 25us，单 block 小 kernel 的 task duration 可满足。

## 测试方案

1. **CSV 驱动 GTest**（`test/dgmm/cdgmm/arch35/`）：随任务提供的 1200 条用例（1000 精度 TC_* + 200 性能 TC_PF），覆盖两种 mode、m/n 扫描（0/1/质数/2 的幂±1/非对齐直至 8192）、lda/ldc padding、incx ∈ {1,2,3,−1,−2}、空矩阵、非法枚举、负维度、非法前导维、空指针等边界与负向场景；mode 列按 int 解析以支持 255 非法枚举用例；
2. **手工用例**：原地执行（A == C && lda == ldc 输出正确、A == C && lda != ldc 拒绝）、nullptr handle，在 `cdgmm_test.cpp` 中以 TEST_F 单独构造；
3. **精度验证**：`verify_accuracy.py` + 混合容差（ops-precision-standard），实部/虚部分别输出；
4. **性能验证**：`verify_performance.py` + msprof op 模式，5 warmup + >=10 样本；
5. **回归**：同族 `sdgmm_test` 不受影响；arch22 的 cdgmm 测试与实现不编译到 950，两套契约互不混合。

## 兼容性分析

- 接口签名、类型（`aclblasComplex`）与状态码定义均沿用仓内既有声明，A2/A3 产品线的 `aclblasCdgmm`（arch22，行主序、仅 LEFT）继续编译运行，行为不变；
- 布局语义按产品线区分：950PR 列主序 + LEFT/RIGHT + 正负 incx，A2/A3 行主序 + 仅 LEFT（文档同步更新 `blas/dgmm/README.md` 差异表与 `docs/zh/api_list.md`，见 §需求拆解）；
- tiling 与 kernel 全部位于 arch35 目录，不引入跨架构符号或新公开 API。
