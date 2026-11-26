# 需求背景（required）

## 需求来源

CANN 社区任务「11月社区任务-aclblasChpr2算子开发（A2/A3）」，任务 ID `7a1275ee6b654459957fc27c8dde32b0`。

任务要求在 Atlas A2 / A3（arch22）上用 Ascend C 完成 `aclblasChpr2` 算子，接口与语义对标 cuBLAS `cublasChpr2` / Netlib BLAS `chpr2.f`。

## 背景介绍

### aclblasChpr2 算子功能定位

`aclblasChpr2` 属于 BLAS Level-2 的秩-2 更新类算子（Hermitian rank-2 update），与 `aclblasSspr2`、`aclblasCher2`、`aclblasCher` 同族。区别在于：

| 算子 | 矩阵性质 | 存储方式 | 标量 |
| --- | --- | --- | --- |
| `aclblasCher` | Hermitian | 压缩存储 | 实数 |
| `aclblasCher2` | Hermitian | 压缩存储 | 复数 |
| `aclblasChpr` | Hermitian | 压缩存储 | 实数 |
| **`aclblasChpr2`** | **Hermitian** | **压缩存储** | **复数** |
| `aclblasCsyr2` | 一般复数 | 全矩阵（有 lda） | 复数 |

`aclblasChpr2` 的难点集中在三点：复数标量带来的共轭非对称性、压缩存储导致的非线性寻址、以及 Hermitian 性质要求对角虚部必须归零。

### 参考实现现状分析

- Netlib `chpr2.f`：纯标量三重循环实现，语义清晰，作为精度标杆。
- cuBLAS `cublasChpr2`：闭源，但公开文档与行为一致——**校验顺序与状态码约定是显式契约**，负向用例按此判定。
- ops-blas `arch35`（Ascend 950PR）：已有 SIMT 实现，按核划分连续 packed 区间，各线程更新独立复数元素。
- ops-blas `arch22`（Atlas A2/A3）：本任务的目标实现，采用 Vector Core + 按行分块策略。

### 参考实现与本任务的差异

arch35 的 SIMT 思路（线程粒度追平元素）在 arch22 的 Vector 编程模型下不适用：arch22 以向量指令为单位，逐元素 scatter/gather 代价高。因此 arch22 需要独立设计，核心是把"列内连续"这一压缩存储的天然性质用足。

### 现状问题

按任务书负向与边界用例集在 Atlas A2（NPU 910B4 / CANN 9.1.0）上验收时，既有 arch22 实现暴露两处缺陷：

1. **校验顺序与状态码不符**：`n == 0` 在所有校验之前直接返回 `SUCCESS`，导致 `n=0` 且 `uplo` 非法时不报错；非法 `uplo` 返回 `INVALID_VALUE` 而非 `INVALID_ENUM`。
2. **特殊值污染**：`x` 或 `y` 含 `inf` 且 `alpha` 某分量为 0 时，朴素复数乘法算出 `0 * inf = NaN`，设备侧输出全 NaN，而参考实现按 C99 Annex G 语义返回 `inf`。

# 需求分析（required）

## 需求描述

实现 `aclblasChpr2` 的 arch22（Atlas A2 / A3）版本，满足：

```cpp
aclblasStatus_t aclblasChpr2(aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
                             const aclblasComplex* alpha, const aclblasComplex* x, int incx,
                             const aclblasComplex* y, int incy, aclblasComplex* ap);
```

计算语义：

```
AP := alpha * x * y^H + conj(alpha) * y * x^H + AP
```

其中 `ap` 为压缩存储的 Hermitian 矩阵，`uplo` 指定上/下三角，`H` 表示共轭转置。

## 需求拆解

1. 支持 `ACLBLAS_UPPER` / `ACLBLAS_LOWER` 两种压缩存储布局，无 `lda` 参数。
2. 支持 `incx` / `incy` 为任意非零整数（含负数，反向取数），不支持 0 与 `INT_MIN`。
3. Hermitian 语义：计算完成后对角元素虚部强制为 0。
4. 校验顺序与状态码与 cuBLAS / 任务书 §2.5 逐条对齐，`n == 0` 仅豁免指针检查与 kernel 启动。
5. 特殊值行为遵循 C99 Annex G：`alpha` 零分量不得产生 `0 * inf = NaN`。
6. 精度：实部/虚部分别按 `atol = rtol = 2^-13`、`matched_ratio ≥ 0.99`、单元素上限 `max(1e-2, 32×ULP)` 判定。
7. 不申请额外 workspace，原地更新 `ap`。

# 详细设计（required）

## 算子分析

### 数学公式

```
AP = alpha * x * y^H + conj(alpha) * y * x^H + AP
```

展开到元素级（以 UPPER 为例，`i ≤ j`）：

```
AP[i,j] += alpha * x[i] * conj(y[j]) + conj(alpha) * y[i] * conj(x[j])
```

按列看，列 `j` 只需要两个列标量：

```
t1(j) = alpha * conj(y[j])
t2(j) = conj(alpha * x[j])
```

于是列 `j` 的更新退化为：

```
AP[:,j] += x * t1(j) + y * t2(j)
```

这一步把"两个外积"化归为"两次向量缩放 + 一次向量加"，是整个 arch22 设计的支点。

### 压缩存储布局

| uplo | 存储条件 | 下标 |
| --- | --- | --- |
| `ACLBLAS_UPPER` | `i ≤ j` | `AP[i + j(j+1)/2]` |
| `ACLBLAS_LOWER` | `i ≥ j` | `AP[i + j(2n-j-1)/2]` |

列 `j` 的起始偏移 `colBase(j)` 只依赖 `j` 与 `n`，且**列内元素连续**——这允许整列一次 `DataCopyPad` 搬入 UB，无需逐元素寻址。

### 支持数据类型

`complex64`（实部、虚部均为 float32）。

### 支持形状

一维向量 `x`、`y`（长度 `n`）与压缩数组 `ap`（长度 `n(n+1)/2`），无广播语义。

## 算子实现

### 实现方案

#### 3.1 host 侧设计

**1）校验顺序与状态码**

严格按任务书 §2.5（与 cuBLAS 一致）短路返回：

| 顺序 | 检查项 | 失败返回 |
| --- | --- | --- |
| 1 | `handle != nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n >= 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 3 | `alpha != nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `uplo ∈ {UPPER, LOWER}` | `ACLBLAS_STATUS_INVALID_ENUM` |
| 5 | `incx` / `incy` 非 0 且非 `INT_MIN` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `n == 0` → quick return | `ACLBLAS_STATUS_SUCCESS` |
| 7 | `x` / `y` / `ap` 非空 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `alpha == (0,0)` → quick return | `ACLBLAS_STATUS_SUCCESS` |

关键点：第 6 步位于结构参数校验**之后**，只豁免张量指针检查与 kernel 启动。`n=0` 且 `uplo` 非法时仍返回 `INVALID_ENUM`。

**2）溢出校验**

`n` 与步长组合可能溢出：按 `(n-1) * |inc| + 1` 计算所需元素数并用 64 位中间量校验，超限返回 `ACLBLAS_STATUS_INVALID_VALUE`，抽成 `ValidateChpr2Shape(n, incx, incy)`。

**3）分核策略**

```
numBlocks = min(n, aivCoreNum)
每核处理列数 = ceil(n / numBlocks)
```

按列分核而非按元素分核：列内连续、跨列不连续，按列切分使每核拿到若干完整的列，搬运始终是连续段。

列归属采用蛇形映射 `Chpr2ColOf`，让相邻核处理的列落在同一行区间内，改善 AP 访问局部性。

**4）tiling 参数**

| 字段 | 含义 |
| --- | --- |
| `n` | 矩阵阶数 |
| `uplo` | 三角方向 |
| `alphaReal` / `alphaImag` | 复数标量，拆成两个 float 传入 kernel，避免 kernel 侧解引用 host 指针 |
| `incx` / `incy` | 步长，kernel 内换算物理下标 |

不申请 workspace。

#### 3.2 kernel 侧设计

整体为 `Init` + `Process` 两阶段，`Process` 按行块循环：**搬入 x/y 行块 → 遍历该行区间内的列 → 累加搬出**。

**1）行方向分块**

每轮处理 `CHPR2_ARCH22_TILE_ROWS = 1024` 行。先把该行区间的 `x`、`y` 载入 UB，之后所有落在该区间内的列共用这份数据，避免每列重复搬运向量。

**2）复数交织与乘加**

complex64 在内存中是 `(re, im)` 交织。设待乘复标量 `t = (tr, ti)`，用两条向量指令完成复乘：

```
dst = raw * tr              // Muls
dst = dst + sprime * ti     // Axpy，其中 sprime = (-im, re) 交织
```

`sprime` 在载入阶段用 `Gather` 一次构造（实部取原值、虚部取负后重排），后续每列复用，不重复计算。

**3）列标量计算：零分量分支**

`alpha` 的某个分量为 0 时，朴素复数乘法会产生 `0 * inf = NaN`。参考实现遵循 C99 Annex G，在 0 分量上跳过该乘积、保留 `inf`。因此标量乘改为：

| 条件 | 实部 | 虚部 |
| --- | --- | --- |
| `ai == 0` | `ar * vr` | `ar * vi` |
| `ar == 0` | `-(ai * vi)` | `ai * vr` |
| 一般 | `ar*vr - ai*vi` | `ar*vi + ai*vr` |

对应 `Chpr2MulComplexScalar`，用于计算 `t1 = alpha * conj(y_j)` 与 `t2 = conj(alpha * x_j)`。

**4）对角处理**

对角元素 `A(j,j)` 的虚部必须为 0。向量路径算出 `x_j*t1 + y_j*t2` 的实部后，与 GM 旧值相加写回，虚部强制写 0。

优化：当 `x_j` 与 `y_j` 四个分量全为 0 时，整列贡献为 0，只需把对角虚部清零，其余整段跳过搬运。

**5）非单位步长**

`incx != 1` 或 `incy != 1` 时：

- 反向遍历语义：逻辑下标 `l` 映射到物理下标 `(n-1-l) * |inc|`（`Chpr2VecPhysIdx`）；
- 跨度窗口 `≤ CHPR2_ARCH22_WIN_MAX_CPLX = 4096` 时整窗搬入 UB，再用 `SetValue` 重排；
- 超窗时逐元素 `DataCopyPad` 单点搬 8 Byte。

**6）UB 缓冲规划**

| 缓冲 | 槽位（float） | 用途 |
| --- | --- | --- |
| `bufXraw_` / `bufXsprime_` | 2112 × 2 | x 行块原始交织 / 共轭重排 |
| `bufYraw_` / `bufYsprime_` | 2112 × 2 | y 行块 |
| `bufAP_` | 2112 | 列数据搬入搬出 |
| `bufD1_` / `bufD2_` | 2112 × 2 | 两次缩放中间结果 |
| `bufWork_` | 2112 | 累加工作区 |
| `bufShiftIdx_` / `bufPackIdx_` | 2112 + 2112 | 地址偏移 / 打包索引 |
| `bufSplitIdx_` | 1024 | 行块内拆分索引 |
| `bufScalar_` | 64 | 标量暂存 |
| `bufWin_` | 4096 × 2 | 非单位步长整窗 |

固定占用 30400 个 float 槽位 = 121600 B ≈ **118.75 KiB**，与 `n` 无关。

**7）数据流**

```
GM(x, y)                 GM(AP 列 j)
   │                         │
   ├─ DataCopyPad ──> UB     ├─ DataCopyPad ──> bufAP_
   │                         │
   ├─ Muls/Axpy(t1, t2) ─────┤
   │                         │
   └────> 累加 ──────────────┴─ DataCopyPad ──> GM(AP 列 j)
```

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品（910B3 / 910B4） | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品（910_93） | √ |

编译 soc：`ascend910b4` / `ascend910b3` / `ascend910_93`。

## 算子约束限制

1. `incx`、`incy` 不得为 0 或 `INT_MIN`；可为负数（反向取数）。
2. `ap` 不得与 `x`、`y` 发生有害重叠（与 BLAS 约定一致）。
3. `alpha` 位于 Host 内存，不支持 Device 标量指针。
4. 不支持广播。
5. 接口异步执行，读取结果前须同步 handle 绑定的 stream。
6. 浮点归约顺序由分块方式决定，多次执行不保证逐位一致，按精度阈值判定。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部、虚部分别对比；`atol = rtol = 2^-13 = 1.2207e-04`；`matched_ratio ≥ 0.99`；单元素上限 `max(1e-2, 32×ULP)`；`NaN/NaN`、同号 `inf/inf` 视为一致 | 任务书 §3.5、任务包 `verify_accuracy.py` |
| 性能标准 | 先 warmup 5 次，再有效采样 ≥10 次取平均；与 `gpu_baseline.csv` 的 GPU 基线比对 | 任务书 §7.4、任务包 `verify_performance.py` |

自验证结果：Atlas A2（NPU 910B4，CANN 9.1.0），1000 条精度用例 + handle 判空用例共 **1001 passed / 0 failed**，耗时 307.9 s。

## 兼容性分析

| 产品线 | 实现位置 | 说明 |
| --- | --- | --- |
| Ascend 950PR / 950DT | `blas/spr2/arch35/` | 既有 SIMT 实现，本次不改动 |
| Atlas A2 / A3 | `blas/spr2/arch22/` | 本任务实现 |

- 接口声明已在 `include/cann_ops_blas.h` 中存在，不新增私有 API。
- arch22 与 arch35 共用同一份接口契约，行为保持一致（含校验顺序与状态码）。
- 与 `aclblasSspr2`（实数版本）共用 `blas/spr2/` 目录与文档，实数版本不受本次改动影响。
