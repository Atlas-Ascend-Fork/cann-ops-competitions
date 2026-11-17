# aclblasCdgmm 算子设计（Ascend 950PR / arch35）

# 需求背景（required）

## 需求来源

11 月社区任务：aclblasCdgmm 算子开发（950）。目标仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，算子目录 `blas/dgmm/arch35/`，测试目录 `test/dgmm/cdgmm/arch35/`。适配硬件 Ascend 950PR（arch35）、CANN 9.1.0，以 Ascend C kernel 直调方式实现句柄式 BLAS 接口 `aclblasCdgmm`（complex64 对角缩放）。

## 背景介绍

DGMM（Diagonal Matrix-Matrix Multiplication）用一个向量构造对角矩阵，与普通矩阵逐元素相乘：`mode=ACLBLAS_SIDE_LEFT` 时按行缩放（`C = diag(x)·A`），`mode=ACLBLAS_SIDE_RIGHT` 时按列缩放（`C = A·diag(x)`）。对标基线为 cuBLAS 扩展接口 `cublasCdgmm`，Netlib BLAS / cblas 无对应实现，精度 golden 因此随任务提供（`test_cases/cdgmm_golden.h`，函数 `aclblasCdgmm_cpu`）。

仓内已有两个可直接参考的工程事实：

1. 同族实数实现 `aclblasSdgmm`（`blas/dgmm/arch35/`）已建立 arch35 的句柄、stream、列主序、正负 `incx`、2D 分核与 `tiling` 传值范式，本设计的工程组织与之同族对齐；
2. 复数逐元素类算子的 arch35 范式为「AIV 分块向量路径 + SIMT 通用路径」两极（参考 `blas/geam/arch35/cgeam_kernel.cpp`：紧凑布局走 packed SIMD，任意 leading dimension / 标量语义走 SIMT；`blas/axpy/arch35/caxpy_kernel.cpp` 同构），本设计沿用该范式。

`blas/dgmm/arch22/cdgmm*` 属 Atlas A2/A3 旧产品线：**行主序、仅 LEFT（RIGHT 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`）、`lda/ldc ≥ max(1,n)`、非法 mode 返回 `INVALID_ENUM`、Host 侧 `aclrtSynchronizeStream`、额外分配 aug/workspace/tiling 三块 Device 内存**。这些语义与本任务（列主序、LEFT/RIGHT、正负 incx、`lda/ldc ≥ max(1,m)`、非法 mode 返回 `INVALID_VALUE`、异步下发、无 workspace）逐项冲突，本设计不参考、不复用、不照抄其布局与校验顺序。

公共声明已存在于 `include/cann_ops_blas.h`（现由 arch22 实现使用），本任务沿用同一符号与签名，**不修改声明、不定义 950PR 私有平行接口**；arch35 源码仅在 950 目标下参与编译（`SOC_ARCH_DIRS` 与 `blas/CMakeLists.txt` 的 `archXX` 过滤机制），与 arch22 实现互为替代、不会同时链接。

# 需求分析（required）

## 需求描述

函数原型（逐字取自 `include/cann_ops_blas.h`，本次不改动）：

```cpp
aclblasStatus_t aclblasCdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    aclblasComplex* C, int ldc);
```

计算语义（列主序，`A(i,j) = A[i + j·lda]`，`C(i,j) = C[i + j·ldc]`，`0 ≤ i < m`，`0 ≤ j < n`）：

```text
LEFT （ACLBLAS_SIDE_LEFT） ：C(i,j) = x(idx(i)) · A(i,j)，x 长度为 m
RIGHT（ACLBLAS_SIDE_RIGHT）：C(i,j) = A(i,j) · x(idx(j))，x 长度为 n
idx(k) = k · incx                    (incx > 0)
idx(k) = (L − 1 − k) · |incx|        (incx < 0)，L = m（LEFT）/ n（RIGHT）
```

复数乘法按 `(ar·xr − ai·xi, ar·xi + ai·xr)` 展开，全程 float32；`aclblasComplex` 为 `{float real; float imag;}`（实部、虚部各 32 bit，共 8 字节，在 Device 上等价于一个 `float2`）。

功能与契约：

- 两种 mode 均为正向功能，`x` 的长度由 mode 决定（m 或 n），两种 mode 的 x 下标规则不同，不可复用同一套索引；
- `incx` 只要求非 0，**允许负数**：`incx < 0` 时按 `idx(k)` 从 x 尾部反向访问；
- `lda`、`ldc` 各自 `≥ max(1, m)`；A/C 的 padding 行既不参与运算也不被写入；
- `m == 0` 或 `n == 0` 是合法 no-op，返回 `ACLBLAS_STATUS_SUCCESS` 且不写 C，此时 A/x/C 可为 `nullptr`；
- `C` 可与 `A` 指向同一内存（原地执行），要求 `lda == ldc`，否则 Host 侧拒绝；`C` 与 `x` 不复用内存；
- 状态码对齐 cuBLAS 13.5 `cublasCdgmm`：非法 `mode`、`m < 0`、`n < 0`、`incx == 0`、`lda < max(1,m)`、`ldc < max(1,m)`、空指针、原地 `lda != ldc` 均返回 `ACLBLAS_STATUS_INVALID_VALUE`（**不使用** `ACLBLAS_STATUS_INVALID_ENUM`）；空 handle 沿用仓内约定返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
- 异步：依赖 `aclblasSetStream` 绑定的 stream，接口只下发不同步，调用方在读取 Device 输出前自行同步 stream。

## 需求拆解

| 项目 | 实现要求 |
|---|---|
| 参数校验 | `handle` → `mode` → `m < 0` → `n < 0` → `incx == 0` → `lda` → `ldc` → no-op 早返回 → `A/x/C` 判空 → `C == A` 时 `lda == ldc`；**前导维校验先于早返回**（`n = 0` 且 `lda = 0` 仍返回 INVALID_VALUE） |
| 错误码 | 非法 mode 返回 `INVALID_VALUE`（与 cuBLAS 一致）；空 handle 返回 `HANDLE_IS_NULLPTR` |
| 数据语义 | A/x/C 统一 complex64；LEFT 行缩放（x 长度 m）/ RIGHT 列缩放（x 长度 n） |
| 步长 | `incx ≠ 0`，正负号均合法；索引统一按 `idx(k)` 计算，地址用 64 位整数 |
| 布局 | 任意 `lda/ldc ≥ max(1,m)`；按列首地址定位，padding 不读不写；不支持任意 stride、不涉及 broadcast |
| quick return | `m == 0` 或 `n == 0` 在结构校验之后返回 SUCCESS，不启动 Kernel |
| 原地执行 | `A == C && lda == ldc` 允许；逐元素独立、无跨元素依赖，读写同一地址安全 |
| 异步与内存 | 仅在 handle stream 上下发 1 个 Kernel；无 workspace、无 Host↔Device 标量搬运、无流同步 |
| 确定性 | 无原子、无归约、无跨核依赖，逐元素独立计算；同设备同输入同 tiling 多次调用结果逐位一致 |
| 自验 | 随包 1200 条 CSV（1000 精度 + 200 性能）全量通过；golden 用随任务提供的 `cdgmm_golden.h`（float64 计算、支持 LEFT/RIGHT 与正负 incx） |

# 详细设计（required）

## 算子分析

### 数学公式

对每个逻辑元素，标量缩放对象由 mode 决定：

```text
LEFT ： c_re = a_re·x_re − a_im·x_im ,  c_im = a_re·x_im + a_im·x_re ,  x = x(idx(i))
RIGHT： c_re = a_re·x_re − a_im·x_im ,  c_im = a_re·x_im + a_im·x_re ,  x = x(idx(j))
```

其中 `a = A[i + j·lda]`、`c = C[i + j·ldc]`、`x` 取址下标为

```text
idx(k) = k·incx               (incx > 0)
idx(k) = (L − 1 − k)·|incx|   (incx < 0)，L = m (LEFT) / n (RIGHT)
```

`incx < 0` 的物理含义是：逻辑第 k 个元素位于 x 尾部起第 k 个跨步位置，等价于把 x 的物理顺序反转后再按 `|incx|` 取样。

**示例（m = 2，n = 3，lda = ldc = 3，复元素按列主序存于 3×3 缓冲区，第 3 行为 padding）**：

```text
A 缓冲区（每列 3 个复元素，末位 PAD 不参与运算）：
  A[0,0]=1+1i  A[1,0]=2−1i  PAD | A[0,1]=3      A[1,1]=−1+2i  PAD | A[0,2]=0.5−0.5i  A[1,2]=4+4i  PAD

LEFT，x = (2+1i, 1−3i)：           RIGHT，x = (2+1i, 1−3i, −2i)：
  C(0,0)=1+3i    C(1,0)=−1−7i        C(0,0)=1+3i     C(1,0)=5
  C(0,1)=6+3i    C(1,1)=5+5i         C(0,1)=3−9i     C(1,1)=5+5i
  C(0,2)=1.5−0.5i C(1,2)=16−8i       C(0,2)=−1−1i    C(1,2)=8−8i
```

`m = 2`、`incx = −1` 时 `idx(0) = (2−1−0)·1 = 1`、`idx(1) = 0`，即 `x(0) = x[1]`、`x(1) = x[0]`。

### 支持数据类型

A、x、C 统一为 `aclblasComplex`（complex64，实部/虚部各 float32）；中间乘加按 float32 计算。golden 用 float64 求值后收窄到 float32，因此本实现的 float32 逐步舍入结果与 golden 的偏差上界约为 2 ULP，远小于任务书规定的混合容差（rtol = atol = 2⁻¹³）。kernel 内不复用半精度中间量，也不改变乘加顺序（先乘后加/减，不使用 FMA 收缩），使 AIV 与 SIMT 两条路径的舍入行为一致、结果可预测。

### 支持布局与形状

- A 物理跨度 `lda × n` 复元素，C 物理跨度 `ldc × n` 复元素，`lda, ldc ≥ max(1, m)`；元素地址 `i + j·lda`（A）/ `i + j·ldc`（C）；
- x 物理长度至少 `(L − 1)·|incx| + 1` 个复元素，`L = m`（LEFT）/ `n`（RIGHT）；原始 C ABI 无法推断真实分配长度，由调用方保证存储有效；
- `m`、`n` 为运行时入参，覆盖 0、1、质数、2 的幂及 2 的幂 ±1、非对齐值，直至 8192；`m·n` 与 `lda·n` 使用 64 位计算，Host 侧检查字节跨度不溢出；
- 不支持任意 stride、不涉及 broadcast、不返回视图；`m == 0` / `n == 0` 时不启动 Kernel。

### 边界与特殊值

- 输入实部/虚部覆盖 FLOAT32 全集（含 Inf/NaN）。缩放按 IEEE 逐元素传播：`0·Inf`、`Inf − Inf` 会产生 NaN，属允许输出，按任务书 §3.2 的 Inf/NaN 等价规则判定；
- Host 与 Kernel 都**不扫描** Tensor 元素值，不做特殊值预处理，不因此改变 tiling 或分派；
- C 的 padding 行（`i ≥ m`）在任何路径下都不被写入；测试侧以哨兵值校验；
- `m`/`n` 恰为 tile 整数倍、tile 余 1、余 3 等尾块场景由尾块长度处理，不需单独路径。

### 原地执行与别名

`C == A && lda == ldc` 时原地更新。每条路径对同一元素都是「读出 → 计算 → 写回」，且 A、C 按列首地址与物理跨度一致，元素间无依赖、分核后各核的列区间互不重叠，因此原地与独立输出结果一致；`lda != ldc` 的原地调用在 Host 侧直接拒绝（否则会写坏相邻元素）。`C` 与 `x` 复用内存属调用方违约，部分重叠行为未定义。

### 参数与错误码契约（含校验顺序）

Host 侧校验顺序固定（与任务书 §2.5 及随包 golden 的 `CdgmmValidateParams` 逐条一致）：

| 顺序 | 检查 | 返回 |
|---|---|---|
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `mode ∉ {ACLBLAS_SIDE_LEFT(141), ACLBLAS_SIDE_RIGHT(142)}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `m < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 5 | `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `lda < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `ldc < max(1, m)` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `m == 0 \|\| n == 0` | `ACLBLAS_STATUS_SUCCESS`（quick return，不下发） |
| 9 | `A`、`x`、`C` 任一为空（仅在 `m > 0 && n > 0` 时） | `ACLBLAS_STATUS_INVALID_VALUE` |
| 10 | `C == A && lda != ldc` | `ACLBLAS_STATUS_INVALID_VALUE` |

要点：前导维校验（6/7）位于早返回（8）之前，因此 `n = 0 && lda = 0`、`m = 0 && ldc = 0` 仍返回 `INVALID_VALUE`；非法 `mode` 先于维度与指针检查（`mode = 255, m = -1` 返回 `INVALID_VALUE`）；空指针检查只在 `m > 0 && n > 0` 时进行（`m = 0, n = 16` 且 A/x/C 全空仍返回 `SUCCESS`）。

`incx = INT32_MIN` 不属于任务书的拒绝范围（golden 也不拒绝），实现用 int64 计算 `|incx|` 以避免取绝对值溢出；该值本身会导出无法成立的 x 寻址，属调用方非法用法。

## 算子实现

### 实现方案

Host 侧完成校验、mode 归一化、tiling 计算与分派；Device 侧 `cdgmm_kernel` 一个 Kernel 入口，按 `tiling.useSimt` 分派两条互斥路径：

| 路径 | 适用条件 | 主要手段 | 覆盖范围 |
|---|---|---|---|
| AIV 列块路径（主路径） | `m ≥ 32` | 按列 `DataCopyPad` 搬入 UB、`DeInterleave` 拆实虚平面、4 次 `Mul` + 2 次 `Add/Sub`、`Interleave` 回写 | 200 条性能用例中的 189 条（含 4 条验收基准形状）与绝大多数精度用例 |
| SIMT 通用路径 | `m < 32` | 每线程处理一个复元素，`float2`/`uint64` 打包读写、列内行连续合并访存 | 宽条形状（`m` 很小、`n` 很大）与 `1×1`、`1×N` 等延迟区用例 |

选择阈值只依赖形状（`m`）与平台信息，不依赖数据值，因此同一输入恒走同一路径，满足确定性要求；阈值与 tiling 常量集中在 `cdgmm_tiling_data.h`，便于回归调整。

> 说明：性能用例全部是紧凑布局（`lda = ldc = max(1,m)`）且 `incx = 1`，其中仅 11 条满足 `m < 32` 走 SIMT；带 padding 前导维与负步长的精度用例两条路径都能正确执行，AIV 路径按列搬移因此天然支持任意 `lda/ldc`。

#### 3.2.1 host侧设计

1. `aclblasCdgmm` 按上表 1~10 的顺序校验，`m == 0 || n == 0` 直接 `SUCCESS`。
2. 取 `aivCoreNum = GetAivCoreCount()`；为 0 说明平台信息获取失败，返回 `ACLBLAS_STATUS_INTERNAL_ERROR`（与 `sdgmm` 一致）。
3. mode 归一化为 `CDGMM_MODE_LEFT = 0` / `CDGMM_MODE_RIGHT = 1` 存入 tiling，避免 Device 侧比较枚举值。
4. 路径与分核：

   - `m ≥ CDGMM_AIV_MIN_M(32)` → AIV 列块路径
     ```text
     tileM        = ((UB_SIZE − UB_RESERVE) / (CDGMM_UB_FLOATS_PER_COMPLEX · 4)) 向下取 4 的倍数
                    // 15 float/复元素、UB_SIZE = 248KB、UB_RESERVE = 8KB ⇒ tileM = 4096
                    // 取 4 的倍数保证每个 tile 的平面起点落在 32B 块边界
     totalMTiles  = ceil(m / tileM)                       // 行块数
     colBlocks    = min(n, aivCoreNum)                    // 沿 n 分核
     mBlocks      = (colBlocks < aivCoreNum && totalMTiles > 1)
                    ? min(aivCoreNum / colBlocks, totalMTiles) : 1   // n 很小（竖长）时沿 m 分核
     numBlocks    = colBlocks · mBlocks
     perCoreN     = n / colBlocks ,     remainder     = n % colBlocks          // 前 remainder 个列块多 1 列
     perCoreMTile = totalMTiles / mBlocks , mTileRemainder = totalMTiles % mBlocks // 前 mTileRemainder 个行块多 1 块
     ```
     分核只用「每核基数 + 余数」的标量字段表达（符合仓内 R4：TilingData 不使用数组），Kernel 侧自行按 `blockIdx` 还原区间；核数在运行时查询（符合 R2：不硬编码核数）。
   - `m < CDGMM_AIV_MIN_M` → SIMT 路径
     ```text
     total      = m · n                       // 64 位
     numBlocks  = min(aivCoreNum, ceil(total / SIMT_MIN_THREAD_NUM))     // SIMT_MIN_THREAD_NUM = 128
     threadCount = min(align(ceil(total / numBlocks), 128), SIMT_MAX_THREAD_NUM)  // 上限 2048
     ```
5. 组装 `CdgmmTilingData`（全标量 14 个字段，56 字节），调用 `cdgmm_kernel_do(A, x, C, tiling, numBlocks, handle->stream)` 下发；tiling 随 `<<<>>>` 按值传入，**不申请 workspace、不做 D2H、不同步 stream**。
6. 设备跨度安全：`n·lda·8`、`n·ldc·8` 用 64 位计算；若超出 `size_t` 可表示范围则返回 `ACLBLAS_STATUS_INVALID_VALUE`（不产生越界地址）。

`CdgmmTilingData` 字段（`blas/dgmm/arch35/cdgmm_tiling_data.h`）：

```cpp
struct CdgmmTilingData {
    uint32_t mode;           // 归一化：0 = LEFT，1 = RIGHT
    uint32_t m, n;           // 逻辑行列
    int32_t  incx;           // 原始步长（可负）
    uint32_t lda, ldc;       // 列主序前导维
    uint32_t tileM;          // 行块长度（复元素），4 的倍数
    uint32_t perCoreN;       // 每个列块的基础列数
    uint32_t remainder;      // 前 remainder 个列块各多 1 列
    uint32_t mBlocks;        // 行方向分块数
    uint32_t perCoreMTile;   // 每个行块的基础行块数
    uint32_t mTileRemainder; // 前 mTileRemainder 个行块各多 1 个行块
    uint32_t useSimt;        // 1 = SIMT 路径，0 = AIV 路径
    uint32_t threadCount;    // SIMT 每 block 线程数
};
```

#### 3.2.2 kernel侧设计

Kernel 入口（两条路径共用一次下发；`TPipe` 为局部变量，符合 R3）：

```cpp
extern "C" __global__ __aicore__ void cdgmm_kernel(
    GM_ADDR A, GM_ADDR x, GM_ADDR C, CdgmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.m == 0 || tiling.n == 0) {
        return;                                             // 双保险：Host 侧已在 no-op 早返回
    }
    if (tiling.useSimt != 0) {
        asc_vf_call<CdgmmElementwiseSimt>(
            dim3{tiling.threadCount, 1, 1},
            static_cast<uint64_t>(tiling.m) * tiling.n, tiling.m, tiling.n, tiling.incx,
            tiling.lda, tiling.ldc, tiling.mode, A, x, C);
        return;
    }
    TPipe pipe;
    CdgmmAiv op;
    op.Init(A, x, C, tiling, &pipe);
    op.Process();
}
```

两条路径共用同一 `<<<numBlocks, nullptr, stream>>>` 下发；Kernel 内用 `GetBlockIdx()` 定位自身区间，不读取 Host 侧每核数组。

**AIV 列块路径**

1. 由 `blockIdx` 还原区间（与 `sdgmm` 相同的 2D 解码，`blockIdx = colBlock · mBlocks + mBlock`）：
   ```text
   colBlock < remainder ? [colBlock·(perCoreN+1), +perCoreN+1) : [colBlock·perCoreN + remainder, +perCoreN)
   mBlock < mTileRemainder ? 多 1 个行块 : 基础行块数
   startM = startMTile · tileM ， endM = min(endMTile · tileM, m)
   ```
   `startCol >= endCol || startM >= endM || tileM == 0` 时直接返回（空核）。
2. **行块外层、列内层** 的双层循环，使 LEFT 模式的 x 段在行块内被该核全部列复用：
   ```text
   for tile in [startMTile, endMTile):
       rowStart = tile · tileM ; curM = min(tileM, m − rowStart)
       if mode == LEFT: 装配本行块的 x 平面 Xr/Xi（每核每行块一次）
       for col in [startCol, endCol):
           CopyInColumn(col, rowStart, curM)     // 1 次 DataCopyPad
           Compute(col, curM)                    // 平面复数乘法
           CopyOutColumn(col, rowStart, curM)    // 1 次 DataCopyPad
   ```
3. 逐列搬入（支持任意 `lda/ldc`，无需 32B 对齐的列首地址；与 `blas/geam/arch35/sgeam_kernel.cpp` 的单列 `DataCopyPad` 用法一致）：
   ```cpp
   // A 列首物理偏移为 (col·lda + rowStart) 个复元素，块长按字节给 8·curM
   uint64_t aOff = static_cast<uint64_t>(col) * tiling.lda + rowStart;
   DataCopyExtParams cp{1, static_cast<uint32_t>(curM * sizeof(aclblasComplex)), 0, 0, 0};
   DataCopyPad(aAos, aGm_[aOff * 2], cp, noPad);        // aGm_ 为 float 视图，偏移 ×2
   ```
   尾块 `curM < tileM` 由 `DataCopyPad` 的字节长度直接支持，不需要对齐前缀/尾块分支。
4. 复数乘法的平面实现（UB 内，全部为 32B 对齐的整体张量，无逐元素操作）：
   ```text
   DeInterleave(Ar, Ai, aAos, 2·curM)                    // 交错 → 实/虚平面
   // LEFT：Xr/Xi 来自本行块 x 段；RIGHT：Xr/Xi 由单列标量广播（Duplicate）
   tmp = Ai · Xi ;  Cr = Ar · Xr ;  Cr = Cr − tmp         // 实部：ar·xr − ai·xi
   tmp = Ai · Xr ;  Ci = Ar · Xi ;  Ci = Ci + tmp         // 虚部：ar·xi + ai·xr
   Interleave(cAos, cAos[curM], Cr, Ci, curM)             // 平面 → 交错输出
   ```
   `Mul`/`Sub`/`Add` 各操作数均为长度 `curM` 的整段张量；每列 8 条向量指令（RIGHT 另加 2 条 `Duplicate` 做标量广播），`curM ≥ 32` 时每条指令至少覆盖一个完整 repeat，向量利用率与访存带宽匹配。
5. x 段装配（仅 LEFT 需要行向量；实部、虚部各走一路跨步搬移，直接落进 `Xr`/`Xi` 平面，不额外占用 AoS 暂存缓冲）：
   - `incx == 1`：`blockCount = curM`、`blockLen = 4` 字节、`srcStride = 4` 字节，实部基址偏移 0、虚部基址偏移 1 个 float，各一次 `DataCopyPad`；
   - `incx > 1`：同结构，`srcStride = (incx−1)·8 + 4` 字节，把逻辑行序的实部/虚部紧凑装入平面（`sgeam` 的转置路径同样是 `blockLen = 4` 字节的跨步搬移）；
   - `incx < 0`：物理下标随逻辑行递减，块间步长字段不支持递减方向，改用标量 `GetValue` 按 `idx(k)` 直接写平面（每核每行块一次，复杂度 `O(curM)`，与矩阵元素数无关）。性能用例全部 `incx = 1`，该分支只影响负步长精度用例的绝对耗时，不影响性能判定；
   - RIGHT：每列 1 个复数标量（2 次标量读），`Duplicate` 成 `Xr/Xi` 平面，`curM` 长度一次广播。
6. 原地执行：`C == A && lda == ldc` 时同一列块「先读后写」，行块间、列间互不重叠，无需额外处理；分核只按列区间与行块区间切分，保证同一元素只有一个写者。
7. 同步与流水：搬入/搬出分别用 `TQue<VECIN, 2>` / `TQue<VECOUT, 2>`，`EnQue/DeQue` 自动插入 MTE2→V、V→MTE3 同步；`TBuf<VECCALC>` 承载实/虚平面，只被 V 流水访问。VECOUT 双缓冲使上一 tile 的异步回写与下一 tile 的搬入/计算重叠（MTE3 不阻塞后续 MTE2）。`TPipe` 在 Kernel 入口定义为局部变量并以指针传递（符合 R3：不作为类成员）。

**UB 预算**（以 1 个复元素计的 float 数）：

| 缓冲 | float / 复元素 | 说明 |
|---|---|---|
| AoS 搬入（`TQue<VECIN,2>`） | 2 × 2 = 4 | 双缓冲 |
| AoS 搬出（`TQue<VECOUT,2>`） | 2 × 2 = 4 | 双缓冲 |
| `Ar`、`Ai` | 2 | 实/虚输入平面 |
| `Cr`、`Ci` | 2 | 实/虚输出平面（与输入平面分离，保证原地写回前读完成） |
| `tmp` | 1 | 乘法中间量 |
| `Xr`、`Xi` | 2 | LEFT 行块向量或 RIGHT 列标量广播 |
| **合计** | **15 float = 60 B** | `tileM = ((248KB − 8KB) / 60) 向下取 4 的倍数 = 4096` |

`tileM` 由 `UB_SIZE` 反算而非硬编码（`blas/common/helper/kernel_constant.h`，当前 248KB）；若平台 UB 变小导致 `tileM < 4`，Host 侧返回 `ACLBLAS_STATUS_INTERNAL_ERROR`，不做静默降级。

**SIMT 通用路径（`m < 32`）**

采用 `blas/geam/arch35/cgeam_kernel.cpp` 的元素级范式：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CdgmmElementwiseSimt(
    uint64_t total, uint32_t m, uint32_t n, int32_t incx, uint32_t lda, uint32_t ldc, uint32_t mode,
    __gm__ const uint64_t* a, __gm__ const uint64_t* x, __gm__ uint64_t* c)
{
    const uint64_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = gridDim.x * blockDim.x;
    uint32_t row = static_cast<uint32_t>(thread % m);      // 循环外算一次
    uint32_t col = static_cast<uint32_t>(thread / m);
    const uint32_t rowStep = static_cast<uint32_t>(stride % m);
    const uint32_t colStep = static_cast<uint32_t>(stride / m);
    for (uint64_t k = thread; k < total; k += stride) {
        // idx：LEFT 用 row，RIGHT 用 col；incx<0 时按 (L−1−k)·|incx| 反算，64 位运算
        // 复数乘法 4 乘 2 加，实/虚部独立舍入，打包为 uint64_t packedResult
        c[static_cast<uint64_t>(col) * ldc + row] = packedResult;
        // row += rowStep, col += colStep 并处理进位（避免循环内整数除法）
    }
}
```

- 打包读写：一个复元素就是 8 字节，按 `uint64_t`（等价 `float2`）一次访存，复数乘法在寄存器内展开为 4 次 float 乘法与 2 次加减；
- 合并访存：同一 warp 内线程的逻辑下标连续，紧凑布局下相邻线程访问相邻 8 字节，合并为 256 字节请求；`lda/ldc` 带 padding 时仅在列边界断开；
- x 访问：LEFT 下同线程 `row` 固定、相邻线程 `x[idx(row)]` 连续；RIGHT 下同列所有线程读同一个 `x[idx(col)]`，属广播，不放大访存；
- 该路径没有 UB 容量约束、没有列首对齐约束、也没有 tile 余数分支，是任意布局与极小 `m` 的通用兜底。

### 工程组织

| 位置 | 内容 |
|---|---|
| `include/cann_ops_blas.h` | 复用已有唯一声明，本次不改动 |
| `blas/dgmm/arch35/cdgmm_host.cpp` | 参数契约、mode 归一化、tiling、路径分派、异步下发 |
| `blas/dgmm/arch35/cdgmm_tiling_data.h` | Tiling 结构与模式常量、UB/阈值常量（Host 与 Kernel 共享） |
| `blas/dgmm/arch35/cdgmm_kernel.h` | `cdgmm_kernel_do` 声明与 `GM_ADDR` 约定 |
| `blas/dgmm/arch35/cdgmm_kernel.cpp` | AIV 列块路径 + SIMT 通用路径 + Kernel 入口 + launcher |
| `test/dgmm/cdgmm/arch35/cdgmm_test.cpp` | CSV 驱动的 GTest（含手工 `TEST_F`：原地别名、空 handle） |
| `test/dgmm/cdgmm/arch35/cdgmm_param.h` | 17 列 CSV 逐列显式读取（缺列/空值抛异常），`mode` 按整数解析 |
| `test/dgmm/cdgmm/arch35/cdgmm_npu_wrapper.h` | Device 内存/句柄/stream 封装，调用 `aclblasCdgmm` |
| `test/dgmm/cdgmm/arch35/cdgmm_golden.h` | 随任务提供的 float64 参考实现（原样提交） |
| `test/dgmm/cdgmm/arch35/cdgmm_test.csv` | 1200 条用例（1000 精度 + 200 性能） |
| `test/dgmm/cdgmm/arch35/README.md` | 复现步骤（编译、精度、性能、msprof 采集） |

构建无需改动 CMake：`blas/CMakeLists.txt` 递归收集 `blas/*/arch35/*.cpp`；测试由既有 `test/dgmm/cdgmm/CMakeLists.txt`（`ops_blas_add_gtest_tests(${OPS_BLAS} cdgmm_test)`）注册，`ops_blas_get_test_target_sources` 在 950 目标下用 `arch35/cdgmm_test.cpp` 替换 arch22 根目录实现，CSV/golden 由 `_ops_blas_copy_test_config_files` 一并部署。arch35 测试源用引号包含 `cdgmm_golden.h`/`cdgmm_param.h`，编译器优先命中同目录文件，不会误用 arch22 根目录的旧 golden。

同一 PR 内需同步的文档（任务书 §4 要求）：

| 文档 | 修改内容 |
|---|---|
| `blas/dgmm/README.md` | `aclblasCdgmm` 产品支持表标注 Ascend 950PR：支持；参数/约束改为列主序、LEFT/RIGHT、`incx ≠ 0`（含负数）、`lda/ldc ≥ max(1,m)`、非法 mode 返回 `INVALID_VALUE`；补 sdgmm/cdgmm 与 arch22/arch35 的差异表 |
| `docs/zh/api_list.md` | `aclblasCdgmm` 描述由「行主序单精度复数对角矩阵左乘」改为「列主序单精度复数对角矩阵乘法（LEFT/RIGHT）」 |

## 支持硬件

| 芯片版本 | 支持情况 | 说明 |
|---|---|---|
| Ascend 950PR | 支持 | 本任务验收目标，arch35 |
| Ascend 950DT | 支持 | 同 arch35 代码路径 |
| Atlas A2 / A3 训练推理系列 | 由 arch22 旧实现承载 | 本设计不涉及，两个 arch 目录按 SOC 互斥编译 |

依赖 CANN 9.1.0（asc-devkit ≥ 9.1）。实现只使用经典 AscendC 接口（`GlobalTensor`、`TBuf/TQue`、`DataCopyPad`、`DeInterleave/Interleave`、`Mul/Add/Sub/Duplicate`、`asc_vf_call`），不依赖 `tensor_api`，因此无需在 `blas/CMakeLists.txt` 增加 `TENSOR_API_OPS` 版本门控。

## 算子约束限制

- 仅支持 complex64、列主序、`lda/ldc ≥ max(1,m)` 的常规 BLAS 布局；不支持任意 stride、不支持 broadcast；
- `x` 的长度由 mode 决定（LEFT 为 m，RIGHT 为 n），调用方必须保证 `(L−1)·|incx| + 1` 个复元素有效；`C` 的写入范围严格为 m×n 逻辑块，padding 不写；
- `C == A` 仅在 `lda == ldc` 时允许，`lda != ldc` 的原地调用返回 `INVALID_VALUE`；`C` 与 `x` 不得复用内存，部分重叠行为未定义；
- 接口异步，调用方在 stream 完成前必须保持 A/x/C 存储有效，读取结果前需自行同步；
- Host 不扫描 Tensor 元素值；Inf/NaN 按 IEEE 传播，判定交给测试侧的等价规则；
- 不支持 Device 侧标量（本接口无标量入参），无 workspace 语义。

# 可维可测分析

## 精度标准/性能标准

| 标准 | 判定 | 来源 |
|---|---|---|
| Golden | 随任务提供的 `cdgmm_golden.h`（`aclblasCdgmm_cpu`）：列主序、float64 计算、支持 LEFT/RIGHT 与正负 `incx`、校验顺序与错误码对齐 §2.5 | 任务书 §3.1/§3.2 |
| 普通精度 | 实部、虚部分别判定：`abs(actual − golden) ≤ atol + rtol·abs(golden)`，`atol = rtol = 2⁻¹³ (≈1.2207e-4)`，`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2 或 32·ULP` | 任务书 §3.2、混合容差标准 |
| 矩阵完整性 | 只比较 m×n 逻辑块；C 的 `ldc` padding 行调用前写哨兵、调用后校验未变；C 的 m×n 元素必须全被写入 | 任务书 §3.2 第 4 条 |
| 非有限值 | golden 与 actual 同为 NaN、同为同号 Inf，或 NaN 对 ±Inf 视为等效；有限对 Inf/NaN 不通过；非有限值不计入 matched_ratio/max_abs，单独统计且须全部等效 | 任务书 §3.2 |
| 行只读 | A 与 x 不被改写（测试侧逐元素比对调用前后） | 任务书 §3.2 自验要求 |
| 性能 | 每个性能 case：`GPU 标杆耗时 / NPU 实测耗时 ≥ 0.4`；全部可比 case 的倍率算术平均 `≥ 0.5`；两项须同时满足 | 任务书 §3.3 |
| 采样 | `msprof op` 采集单算子 task duration；5 次 warmup 不计入，warmup 后连续执行 ≥10 次取算术平均，仅统计返回 SUCCESS 的样本 | 任务书 §3.3 |
| 确定性 | 同设备、同输入、同 tiling 连续多次调用，输出逐元素一致 | 任务书 §2.5 |
| 内存 | 任务书不要求内存验收；本实现不申请 workspace，Host 侧峰值仅为 tiling 结构体与日志 | 任务书 §3.4 |

**性能预算反算**（字节量 = 读 A + 写 C = 16 字节/复元素；倍率 0.4 对应的单例门限）：

| 用例（mode, m×n, lda=ldc） | 数据量 | H100 标杆 | 0.4× 门限（NPU 上限） | 门限对应有效带宽 |
|---|---|---|---|---|
| LEFT, 512×512, 512 | 4.0 MiB | 10.531 us | 26.33 us | ≈ 0.16 TB/s |
| RIGHT, 1024×1024, 1024 | 16.0 MiB | 11.651 us | 29.13 us | ≈ 0.58 TB/s |
| LEFT, 8192×1024, 8192 | 128.0 MiB | 53.242 us | 133.1 us | ≈ 1.01 TB/s |
| RIGHT, 1024×8192, 1024 | 128.0 MiB | 53.453 us | 133.6 us | ≈ 1.00 TB/s |

H100 在大尺寸下的实测有效带宽约 2.8~2.9 TB/s（`gpu_baseline.csv` 反算），200 条用例的平均耗时为 53.4 us；平均倍率 0.5 对应 NPU 平均耗时 ≤ 106.8 us，即大尺寸平均有效带宽需达到标杆的 0.5 倍（约 1.4 TB/s）。小尺寸用例的标杆耗时已被 CUDA 侧启动开销（约 8~10 us）主导，NPU 侧启动开销与之同量级，倍率通常大于 1，可拉高平均值。实现侧对应措施：单次搬移 32 KB（`tileM = 4096`）、列内与行块内连续访存、VECOUT 双缓冲让回写与下一 tile 搬入重叠、2D 分核保证 8192 级尺寸吃满 AIV 核数。

**自验方法**：

- 精度：CSV 驱动 GTest（`test/dgmm/cdgmm/arch35/cdgmm_test.cpp`）。1000 条精度用例覆盖 `mode` 全组合、m/n 扫描（0、1、质数、2 的幂及 ±1、非对齐值，至 8192）、长条/宽条、`lda/ldc` padding（`max(1,m)`、+1、+3、2·max(1,m)）、`incx ∈ {1,2,3,−1,−2,−3,−4}`、均匀/正态混合填充与 7 种特殊填充、边界负向（非法 mode=255、负维度、`lda/ldc = 0`、`incx = 0`、空指针、no-op 全空指针、前导维先于早返回）；200 条 `TC_PF` 只做性能采集，精度脚本按前缀排除。指针别名（`C == A && lda == ldc`）与空 handle 无法用 CSV 表达，以 `TEST_F` 手工构造并断言 `SUCCESS`、且原地结果与独立输出逐元素一致；
- 性能：200 条 `TC_PF` 用例跑 `msprof op` 取 task duration，与 `gpu_baseline.csv` 逐例比对倍率并求平均，判定脚本随任务提供（`verify_accuracy.py` / `verify_performance.py`）；
- 报告：精度/性能/内存自验报告与原始日志按任务模板提交到个人私仓 `task_submission/`（本实现不申请 workspace，内存报告按模板注明「不涉及」并给出 Host 峰值），自测报告记录 golden 源码路径与填充分布口径（`GAUSS_NORM_5_2` 为 μ∈[−5,5]、σ∈[0.1,2] 随机取值后截断到 [−5,5]，与生态标准建议的 N(0,1) 钳制存在差异，需注明）。

## 兼容性分析

- 新增 `blas/dgmm/arch35/` 实现，复用 `include/cann_ops_blas.h` 中已有声明与公共 complex64 类型，**不修改 ABI**；`aclblasSdgmm` 与 arch22 的 `aclblasCdgmm` 实现均不受影响（arch35 源文件仅在 950 目标参与编译，符号不会重复定义）；
- 不引入 `tensor_api`，不增加 CMake 版本门控，`asc-devkit ≥ 9.1` 的判定沿用仓内既有机制；
- 行为差异属产品线差异而非回归：A2/A3 的行主序、仅 LEFT、`INVALID_ENUM` 与 `lda/ldc ≥ max(1,n)` 语义保持不变；950PR 按本设计执行列主序、LEFT/RIGHT、`INVALID_VALUE` 与 `lda/ldc ≥ max(1,m)`，通过 `blas/dgmm/README.md` 的差异表与 `docs/zh/api_list.md` 的描述同步对外说明；
- 测试侧只在 `test/dgmm/cdgmm/arch35/` 新增文件，arch22 的 `cdgmm_test.cpp`、`cdgmm_golden.h`、`cdgmm_param.h` 与 `cdgmm_test.csv` 保持原样。

## 风险与对策

| 风险 | 对策 |
|---|---|
| 小 `m` 形状下按列搬移退化为 8~248 字节 DMA | `m < 32` 走 SIMT 路径（一次 8 字节打包读写、线程级合并访存）；`m ≥ 32` 时单列搬移 ≥ 256 字节，DMA 效率可接受 |
| 负 `incx` 的 x 段装配需逐元素读取 | 只发生在「每核每行块一次」的 x 段上，复杂度 `O(curM)` 与矩阵规模无关；性能用例全部 `incx = 1`，不影响性能判定 |
| 8192 级尺寸的带宽目标（≥1.0 TB/s 单例、≈1.4 TB/s 平均） | 32 KB 大块搬移、`lda/ldc` 连续访存、VECOUT 双缓冲重叠回写、2D 分核与余数均衡；实测不达标时优先增大 `tileM`/提高搬入深度，而非拆分 kernel |
| UB 容量或头文件常量变化导致 `tileM` 失效 | `tileM` 由 `UB_SIZE` 动态反算并向下取 4 的倍数；`tileM < 4` 时 Host 返回 `INTERNAL_ERROR`，不静默降级 |
| 特殊值（Inf/NaN）与 `0·Inf` 产生的 NaN | 逐元素 IEEE 传播，不做特殊值预处理；判定按任务书等价规则，有限元素与非有限元素分开统计 |
| `incx = INT32_MIN` 取绝对值溢出 | 统一用 int64 计算 `abs(incx)` 与地址；该值按任务书不属于拒绝范围，故不额外返回错误，行为等同调用方非法寻址 |
| 列首地址在 `lda % 4 != 0` 时非 32B 对齐，若目标平台 `DataCopyPad` 对 GM 起始地址另有约束 | 路径选择只看形状与布局：实现时若真机验证发现该约束，对 `lda % 4 != 0` 或 `ldc % 4 != 0` 的布局把 `useSimt` 置 1（SIMT 无对齐要求），精度与性能用例按实际覆盖重新统计；该回退不改接口、不改 tiling 结构 |
| 原地执行（`C == A`）读写冲突 | 元素级「先读后写」、分核区间不重叠、同核列/行块不重叠，保证唯一写者；`lda != ldc` 在 Host 侧拒绝 |

## 交付与复现

按任务书 §4 与模板交付：设计文档在 cann-competitions 仓以 PR 提交并在评审通过后合入；代码在个人仓开发后向 ops-blas 提交 PR（`blas/dgmm/arch35/`、`test/dgmm/cdgmm/arch35/`、`blas/dgmm/README.md`、`docs/zh/api_list.md` 同一 PR）；自验材料按 `task_submission/` 模板命名（`1 自验证步骤说明.md`、精度/性能/内存自验证报告与日志），并在任务 IT 系统填写验收 commit；个人仓邀请 `Ascend-CANN` 为开发者。

复现步骤见 `test/dgmm/cdgmm/arch35/README.md`：安装依赖 → `./build.sh` 编译 950 目标 → 运行 `cdgmm_test --gtest_filter=*TC_*` 做精度自验 → `msprof op` 采集 `TC_PF` 单算子耗时并与 `gpu_baseline.csv` 比对。

参考：[cublasCdgmm](https://docs.nvidia.com/cuda/cublas/index.html)、[ops-blas 贡献指南](https://gitcode.com/cann/ops-blas/blob/master/CONTRIBUTING.md)、[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)、[社区任务流程](https://gitcode.com/org/cann/discussions/39)。本设计中的性能数值为验收目标与反算门限，实际结果以真机自验报告为准。
