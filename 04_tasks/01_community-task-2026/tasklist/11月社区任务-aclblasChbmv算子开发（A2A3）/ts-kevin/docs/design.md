# 需求背景（required）

## 需求来源

本设计文档对应 2026 年 11 月社区任务 `aclblasChbmv`（Atlas A2/A3 系列产品）。任务要求在昇腾 NPU（Atlas A2/A3 系列产品，性能验收设备 Atlas 800T A2 / 910B3，arch22）上，基于 Ascend C 编程语言、依托 ops-blas 开源仓工程框架，以 kernel 直调（handle 绑定 stream 下发 NPU kernel）方式开发单精度复数（COMPLEX64）Hermitian 带状矩阵-向量乘算子 `aclblasChbmv`，功能与参数语义对齐 cuBLAS `cublasChbmv`，完成算子设计、开发、测试全流程。验收通过后实现代码合入 `blas/sbmv/arch22/`，测试代码合入 `test/sbmv/chbmv/arch22/`，接口声明放入 `include/cann_ops_blas.h`。CANN 版本 9.1.0。

依据材料：任务书 `aclblasChbmv_AtlasA2_A3_task_doc.md`、配套测试用例目录（`chbmv_test.csv`、`gpu_baseline.csv` 200 条性能基线、`gen_csv.py`、`verify_accuracy.py`、`verify_performance.py`、测试指导 README）。

## 背景介绍

### aclblasChbmv 算子功能

`aclblasChbmv` 为 BLAS Level-2 单精度复数 Hermitian 带状矩阵-向量乘，计算公式：

```text
y = alpha * A * x + beta * y
```

其中：

- `A` 为 n×n Hermitian 带状矩阵，含 k 条次对角线与 k 条超对角线；
- `x`、`y` 为 n 元素复数向量；
- `alpha`、`beta` 为 COMPLEX64 标量；
- 仅 `uplo` 指定的带形三角被存储并引用，另一侧由 Hermitian 共轭对称性隐含补全；
- 对角元素虚部按 Hermitian 性质假定为 0，实现只取对角元素实部参与计算。

`A` 按列主序带状格式存储于 `lda×n` 数组：

- `uplo = ACLBLAS_UPPER`：主对角存于第 `k+1` 行，第 1 条超对角存于第 `k` 行（从第 2 列开始），元素 `A(i,j)` 存于 `A(1+k+i-j, j)`，其中 `j=1..n`、`i∈[max(1,j-k), j]`；数组左上 `k×k` 三角不被引用。
- `uplo = ACLBLAS_LOWER`：主对角存于第 1 行，第 1 条次对角存于第 2 行（从第 1 列开始），元素 `A(i,j)` 存于 `A(1+i-j, j)`，其中 `j=1..n`、`i∈[j, min(n,j+k)]`；数组右下 `k×k` 三角不被引用。

### 标杆算子行为（Netlib chbmv / cuBLAS cublasChbmv）

精度 golden 由 Netlib BLAS `chbmv`（CBLAS，`CblasColMajor`，fp64 真值计算、按输出 FLOAT32 收窄落盘）生成，与 `cublasChbmv` 语义一致。参考实现逐输出行 `i` 在有效列区间 `j∈[max(1,i-k), min(n,i+k)]` 上累加：

```text
y_i = beta * y_i + alpha * sum_j A(i,j) * x_j
```

其中存储带外的元素按 Hermitian 共轭对称补全，对角元素只取实部。实现必须保持两条关键语义：

1. **未存储三角区不被引用**：不属于 `uplo` 指定带形三角的存储位置（如 UPPER 时左上 `k×k`、LOWER 时右下 `k×k`）不应影响结果。
2. **镜像侧按共轭读取**：对需要隐式补全的带内元素 `A(i,j)`，通过已存储的 `A(j,i)` 取共轭参与乘加。

### 仓内现状与可复用资产

`blas/sbmv/` 当前已有实数同族算子 `aclblasSsbmv`（arch22 与 arch35）。本任务在 arch22 上新增复数版本，并直接复用现有工程框架：

| 层次 | 关键文件 | 现有职责 | 本算子复用点 |
| --- | --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` | 全部 aclblas 接口对外声明 | 新增 `aclblasChbmv` 声明，签名与 `cublasChbmv` 逐参数对应 |
| Host 基础设施 | `common/helper/` | AIV/AIC 核数查询、handle/stream 解析、日志 | 直接复用 |
| 同族参考 | `blas/sbmv/arch22/`（ssbmv） | 带状矩阵-向量乘的 row-dot 算法骨架、分核、负步长处理 | 算法结构参照，复数运算与 Hermitian 镜像语义新写 |
| 测试框架 | `test/sbmv/ssbmv/` | CSV 用例 + C++ GTest + golden + npu wrapper | 新建 `test/sbmv/chbmv/` 对齐该结构 |

### 参数语义（任务书 §2.4）

| 参数 | 方向 | 类型与约束 |
| --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t`，nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | 输入 attr | `ACLBLAS_UPPER` / `ACLBLAS_LOWER`，非法值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| `n` | 输入 | 方阵阶数，非负；`n=0` 为合法 quick return |
| `k` | 输入 | 次/超对角线条数，非负；`k > n-1` 时按有效带宽 `min(k, n-1)` 计算 |
| `alpha` | 输入 | COMPLEX64 标量指针（Host 内存），`n>0` 时不可为 nullptr |
| `A` | 输入 | COMPLEX64 Device，列主序带状存储，`lda×n`，只读；`lda ≥ k+1` |
| `lda` | 输入 | 带状数组 A 的前导维度，`lda ≥ k+1`；比较在 int64 域完成 |
| `x` | 输入 | COMPLEX64 Device，长度至少 `1+(n-1)×abs(incx)`，只读 |
| `incx` | 输入 | 向量步长，可正可负；`0`、`INT32_MIN` 或跨度溢出非法 |
| `beta` | 输入 | COMPLEX64 标量指针（Host 内存）；`beta=(0,0)` 时 y 旧值不被读取 |
| `y` | 输入/输出 | COMPLEX64 Device，原地覆写，长度至少 `1+(n-1)×abs(incy)` |
| `incy` | 输入 | 向量步长，可正可负；`0`、`INT32_MIN` 或跨度溢出非法 |

非法参数统一返回 `ACLBLAS_STATUS_INVALID_VALUE`（`handle` 空指针单独返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n=0` 按校验顺序优先返回成功）。

# 需求分析（required）

## 需求描述

在 `blas/sbmv/arch22/` 以 kernel 直调方式实现 `aclblasChbmv`：handle 绑定 stream 异步下发，覆盖 `uplo` 全枚举、列主序带状存储与前导维（含 padding）、`n≥0` 全尺寸、`k≥0` 全带宽、标量特殊值、正负向量步长、边界与负向用例；精度满足 COMPLEX64 混合容差标准（rtol=atol=2^-13，matched_ratio≥0.99，max_abs_error≤1e-2 或 32ULP），性能在 910B3 上满足任务书 §3.3 的 200 条 `TC_PF` 用例达标耗时。

## 需求拆解

1. **接口与校验闭环**：`include/cann_ops_blas.h` 新增声明；host 按确定顺序校验，失败返回对应状态码且不发起 kernel。关键顺序约束：`n=0` 是合法 quick return，必须位于 `uplo/k/lda/incx/incy/alpha/A/x/beta/y` 指针校验之前（仅 handle 判空更优先）。
2. **Hermitian 带状语义闭环**：每行只累加有效列区间 `[max(1,i-k), min(n,i+k)]`；直接存储侧按存储值读取，镜像侧对存储值取共轭；对角元素虚部强制为 0；未引用三角区不触碰。
3. **复数乘加与数值行为闭环**：按 `(a_r+i·a_i)(x_r+i·x_i)` 展开为实数乘加，逐 chunk 使用 Ascend C 向量指令与 `ReduceSum`；`alpha`、`beta` 严格按 IEEE 复数乘法公式处理，`beta=(0,0)` 时不读 y 旧值。
4. **步长与寻址闭环**：`incx/incy` 支持正负步长，Host 侧校验溢出，Kernel 侧统一映射逻辑坐标到物理地址。
5. **性能闭环**：单 AIV kernel 按行循环分配，将长度为 `O(min(n,2k+1))` 的行点积向量化，减少多 kernel 下发开销；小带宽与满带宽路径共享同一套行点积逻辑，按列片段分块搬运和归约。
6. **测试与合入闭环**：新建 `test/sbmv/chbmv/` 测试工程，覆盖全部官方用例并补充正态分布用例；`blas/sbmv/README.md` 增加复数接口说明，产品支持表标注 Atlas A2/A3 支持。

# 详细设计（required）

## 算子分析

### 数学公式

设 n 阶 Hermitian 带状矩阵 A 与 n 元素向量 x、y，输出向量第 i 行（0-based 行号 `i`）：

```text
dot_i = sum_{j=j0}^{j1} A(i,j) * x_j
y_i   = alpha * dot_i + beta * y_i
j0 = max(0, i-k)
j1 = min(n-1, i+k)
```

其中 `A(i,j)` 的取值规则：

- 若 `A(i,j)` 位于 `uplo` 指定的存储三角带内，则按列主序带状地址直接读取；
- 若 `A(i,j)` 位于镜像侧，则读取已存储的 `A(j,i)` 并取共轭；
- `i=j` 时只取实部，虚部按 0 参与计算。

复数乘法展开：

```text
(a_r + i*a_i) * (x_r + i*x_i)
  = (a_r*x_r - a_i*x_i) + i*(a_r*x_i + a_i*x_r)

conj(a) * x
  = (a_r*x_r + a_i*x_i) + i*(a_r*x_i - a_i*x_r)
```

复数 `alpha` 缩放：

```text
alpha * (r + i*s)
  = (alpha_r*r - alpha_i*s) + i*(alpha_r*s + alpha_i*r)
```

复数 `beta` 项同理。所有实/虚部均按 FLOAT32 运算，不引入更高精度中间格式，精度判定采用任务书混合容差。

### 数据排布与寻址

`A` 为 COMPLEX64（`aclblasComplex`，AoS 交织存储，实部/虚部各 fp32），逻辑位置 `(i,j)` 到物理复数元素偏移 `aComplexOffset`：

```text
UPPER 直接段（j ≥ i，i 固定）:
  aComplexOffset(i,j) = (k + i - j) + j * lda

LOWER 直接段（j ≤ i，i 固定）:
  aComplexOffset(i,j) = (i - j) + j * lda
```

对镜像段，不按上式逐个计算，而是直接定位到“同行 i、已存储的对称元素”所在列，复用 `uplo` 的存储规律：

- UPPER 时 `j < i` 的镜像段来自上三角存储区；
- LOWER 时 `j > i` 的镜像段来自下三角存储区；
- 镜像段传入 `conjugateA=true`，计算时对 A 虚部取负。

`x`、`y` 的物理地址由步长决定：

```text
inc >= 0: physical(logical) = logical * abs(inc)
inc <  0: physical(logical) = (n-1-logical) * abs(inc)
```

### 支持数据类型

| dtype | 元素宽度 | 计算/搬运语义 |
| --- | --- | --- |
| COMPLEX64（`aclblasComplex`） | 8B（fp32 实部 + fp32 虚部，AoS 交织存储） | GM 搬运按 2×fp32 交织；进入 UB 后用向量指令解交织为实/虚两个平面（AoS→SoA），乘加、归约均在 fp32 实数平面完成；写回 y 时按 2×fp32 直接覆写 |

### 支持形状与参数组合

- `n≥0`，`k≥0`；`lda≥k+1`；`incx/incy` 非 0 且非 `INT32_MIN`。
- `uplo` 覆盖 `ACLBLAS_UPPER`、`ACLBLAS_LOWER`，共 2 组枚举。
- 官方精度用例尺寸覆盖 0、1、小质数、2 的幂及 2 的幂 ±1、非对齐值直至大规模；带宽覆盖 0、1、小带宽、`k=n-1` 满带退化、典型大带宽；性能用例含 200 条 `TC_PF`。
- 仅支持任务书语义内的 `lda/incx/incy` 非连续访问，不扩展任意 stride、无 broadcast、无 dynamic shape 特化。

### 核心算法：按输出行点积 + Hermitian 镜像补全

本算子采用 **one-kernel row-dot** 算法，不调用 Matmul API：

1. 每个 AIV core 负责若干输出行，行号 `row = blockIdx, blockIdx+useCoreNum, ...`。
2. 对每个 `row`，计算有效列区间 `[j0, j1]`。
3. 将列区间拆成两个连续段：
   - **直接段**：位于 `uplo` 存储侧的列，A 元素直接读取；
   - **镜像段**：位于 Hermitian 隐式补全侧的列，A 元素取共轭读取。
   - UPPER 时先处理 `[j0, row-1]` 镜像段，再处理 `[row, j1]` 直接段；LOWER 时先处理 `[j0, row]` 直接段，再处理 `[row+1, j1]` 镜像段。
4. 每段按 `CHBMV_DOT_CHUNK` 分片：
   - A 片按“连续列”或“跨列固定 `lda-1` 步长”搬入 UB；
   - x 片按 `incx` 的连续或步长方式搬入 UB；
   - A、x 解交织为实/虚平面后，按上述复数乘法公式做向量乘加；
   - 使用 `ReduceSum` 对当前片归约，实部/虚部分别累加到行级标量 `sumReal/sumImag`。
5. 行点积完成后，计算 `alpha * dot`，按需读入 y 旧值计算 `beta * y_i`，最后写回 `y_i`。

该算法优势：

- 单 kernel、无 workspace、无多段流水，小尺寸下发开销低；
- 每个点积项只按有效带宽遍历，`k` 小的工作量与稀疏带宽成比例；
- 直接段与镜像段复用同一套向量乘加/归约函数，代码单份；
- 行级循环天然隔离不同输出行，便于按 AIV 核数灵活切分。

### 分块与向量化

- `CHBMV_DOT_CHUNK = 1024`，一个 chunk 内最多处理 1024 个复数元素。
- A/x 搬运优先使用 `DataCopyPad`；当 A 列步长为 `lda-1` 或 x 步长不为 1 时，退化到逐元素 `GetValue/SetValue` 搬运，保证寻址正确。
- 解交织复用 `vreducev2` 的 interleaved→split 语义，将 AoS 复数缓冲拆成实/虚两个 LocalTensor。
- 乘加使用 `Mul/Sub/Add` 向量指令，归约使用 `ReduceSum`；实部、虚部各自独立归约后写回标量累加器。
- 对角元素落在当前 chunk 时，在乘法前强制把 A 虚部对应位置写 0，等价“对角虚部为 0”的 Hermitian 约束。

### 特殊值与边界语义

- `n=0`：Host 直接返回 `ACLBLAS_STATUS_SUCCESS`，不启动 kernel、不校验后续参数。
- `beta=(0,0)`：Kernel 不读取 y 旧值，`oldReal/oldImag` 保持 0，等价 `y = alpha*A*x`。
- `alpha=(0,0)`：按通用公式计算，不承诺跳过 A/x 读取；含 Inf/NaN 的 x 用例不得与 `alpha=(0,0)` 组合（任务书 §2.1）。
- 非有限值：任务书 §3.2 规定 Inf/NaN 单独判定；本实现按 IEEE fp32 向量乘加和 `ReduceSum` 顺序计算，不引入额外分支，最终由精度验证器按特殊值规则比对。
- 未引用三角区：Host/Kernel 只按 `j0..j1` 与 `uplo` 存储规律取地址，不访问左上/右下未引用三角。

## 算子实现

### 实现方案

整体执行流程（kernel 直调，kernel 于 handle 绑定 stream 顺序下发，函数返回即入队完成；读回前由调用方同步）：

```text
aclblasChbmv
  ├─ ValidateChbmvArgs
  │    handle == nullptr            → HANDLE_IS_NULLPTR
  │    n < 0                        → INVALID_VALUE
  │    n == 0                       → SUCCESS（不启动 kernel）
  │    uplo 非法                    → INVALID_VALUE
  │    k < 0                        → INVALID_VALUE
  │    lda < k+1（int64 比较）      → INVALID_VALUE
  │    alpha/A/x/beta/y 为空        → INVALID_VALUE
  │    incx/incy = 0 或 INT32_MIN   → INVALID_VALUE
  │    (n-1)*|incx| 或 (n-1)*|incy| 超出 int32 → INVALID_VALUE
  ├─ 读取 AIV 核数，计算 useCoreNum
  ├─ 填充 ChbmvTilingData
  └─ chbmv_kernel<<<useCoreNum, nullptr, handle->stream>>>(A, x, y, nullptr, tiling)
```

### Host 侧设计

#### 接口声明

`include/cann_ops_blas.h` 新增：

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

签名与 `cublasChbmv` 参数序列一致，维数参数采用 `int`，与仓内 `aclblasSsbmv` 口径一致。

#### 校验顺序

按任务书 §2.5 的维护者口径实现，任一失败即返回对应状态码且不发起 kernel：

1. `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
2. `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；
3. `n == 0` → `ACLBLAS_STATUS_SUCCESS`；
4. `uplo` 非法 → `ACLBLAS_STATUS_INVALID_VALUE`；
5. `k < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；
6. `lda < k+1` → `ACLBLAS_STATUS_INVALID_VALUE`（`k+1` 在 int64 域计算，避免 `k=INT32_MAX` 回绕穿透）；
7. `alpha/A/x/beta/y` 任一为空 → `ACLBLAS_STATUS_INVALID_VALUE`；
8. `incx/incy` 为 0 或 `INT32_MIN` → `ACLBLAS_STATUS_INVALID_VALUE`；
9. `(n-1)×abs(incx)` 或 `(n-1)×abs(incy)` 超出 int32 表示范围 → `ACLBLAS_STATUS_INVALID_VALUE`（Host 侧 int64 计算）。

#### 分核策略

- `useCoreNum = min(aivCoreNum, CHBMV_MAX_CORE_NUM, n)`；
- `CHBMV_MAX_CORE_NUM = 50`；
- 当 `n==0` 时已提前返回，因此 `useCoreNum` 至少为 1；
- 支持 `CHBMV_FORCE_CORE` 环境变量覆盖核数，仅用于调试；调试开启时额外申请 256 个 float 的 GM workspace 用于写入核号/块数/tiling 关键值，并在同步后回读打印。

#### Tiling 数据结构

```cpp
constexpr uint32_t CHBMV_MAX_CORE_NUM = 50;

struct ChbmvTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;
    uint32_t useCoreNum;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    int64_t incx;
    int64_t incy;
};
```

Host 从 `alpha`、`beta` Host 内存直接读取实/虚部快照，随 tiling 传入 kernel，避免 kernel 从 GM 读 Host 标量。

### Kernel 侧设计

文件组织：

```text
blas/sbmv/arch22/
├── chbmv_tiling_data.h
├── chbmv_kernel.h
├── chbmv_kernel.cpp
└── chbmv_host.cpp
```

Kernel 为 `__global__ __aicore__ __vector__` 的 AIV kernel，标准 `Init → Process` 结构。

#### Init 阶段

- 从 tiling 恢复 `n/k/lda/uplo/useCoreNum/incx/incy/alpha/beta`；
- 对 `useCoreNum` 做防御性夹取，避免 0 或超过上限；
- 计算 `absIncx/absIncy`；
- 设置 A/x/y GlobalTensor；
- 初始化 UB 缓冲。

#### Process 阶段

```text
blockIdx = GetBlockIdx()
for row in [blockIdx, blockIdx+useCoreNum, ...]:
    sumReal = sumImag = 0
    j0 = max(0, row-k)
    j1 = min(n-1, row+k)

    if uplo == UPPER:
        mirror segment [j0, row-1]: conj(A) * x
        direct segment [row, j1]:   A * x（对角虚部置 0）
    else:
        direct segment [j0, row]:    A * x（对角虚部置 0）
        mirror segment [row+1, j1]:  conj(A) * x

    dot = sumReal + i*sumImag
    y[row] = alpha * dot + beta * y_old[row]
```

#### 复数乘加实现

对当前 chunk 内的 A 实/虚平面 `aReal/aImag` 与 x 实/虚平面 `xReal/xImag`：

```text
pRR = aReal * xReal
pII = aImag * xImag
pRI = aReal * xImag
pIR = aImag * xReal

real = pRR - pII
imag = pRI + pIR
```

镜像段在乘法前对 `aImag` 取负：

```text
real = pRR + pII
imag = pRI - pIR
```

随后对 `real`、`imag` 分别 `ReduceSum`，累加进 `sumReal/sumImag`。

#### 行结果写回

```text
axReal = alphaReal * sumReal - alphaImag * sumImag
axImag = alphaReal * sumImag + alphaImag * sumReal

if beta != (0,0):
    read oldReal, oldImag from y[row]
else:
    oldReal = oldImag = 0

betaTermReal = betaReal * oldReal - betaImag * oldImag
betaTermImag = betaReal * oldImag + betaImag * oldReal

yReal[row] = axReal + betaTermReal
yImag[row] = axImag + betaTermImag
```

#### UB 容量估算

以 `CHBMV_DOT_CHUNK = 1024` 计，主要 UB 缓冲占用：

| 缓冲 | 大小 |
| --- | --- |
| `aCplxBuf` / `xCplxBuf` | 2 × 1024 × 2 × 4B = 16 KiB |
| `aReal/aImag/xReal/xImag` | 4 × (1024+64) × 4B ≈ 17 KiB |
| `pRR/pII/pRI/pIR` | 4 × 1024 × 4B = 16 KiB |
| `tmpReal/tmpImag` | 2 × 1024 × 4B = 8 KiB |
| `reduceWorkBuf` | (2×1024+64) × 4B ≈ 8.25 KiB |
| `dotReal/dotImag` 及其他 | < 0.1 KiB |

合计约 65.3 KiB。若目标 UB 更小，可下调 `CHBMV_DOT_CHUNK` 线性减少峰值占用；当前取值在满足典型 AIV UB 容量的同时保证向量化效率。

## 支持硬件

| 支持的芯片版本 | 本设计说明 |
| --- | --- |
| Atlas 800T A2（910B3，arch22） | 精度/性能验收设备；单 AIV row-dot kernel 直调 |
| Atlas 800I A2（arch22） | 同 arch22 数据流，随产品包构建回归 |
| Atlas 800I A3（arch22） | 同 arch22 数据流（任务书 §2.2：A2/A3 对应架构目录 arch22），自验后按目标包构建回归 |

按任务书：自验证覆盖一种款型即可（精度建议 910B3 或 A3，性能建议 910B3）；验收阶段 Atlas 800T A2 与 Atlas 800I A3 均需完成。

## 算子约束限制

1. `handle` 空 → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；`n<0`、非法 `uplo`、`k<0`、`lda<k+1`、`n>0` 时空 `alpha/A/x/beta/y`、非法 `incx/incy`、跨度溢出 → `ACLBLAS_STATUS_INVALID_VALUE`。
2. `n=0` 合法 quick return，且优先于除 handle 判空外的其他校验。
3. 仅 COMPLEX64；列主序带状存储；无 broadcast、无视图返回、无 dynamic shape 特化。
4. 仅引用 `uplo` 指定带形三角；对角元素虚部假定为 0；未引用三角区不访问。
5. `beta=(0,0)` 不读 y 旧值；`alpha=(0,0)` 不承诺跳过 A/x 读取。
6. 异步语义：`aclblasSetStream` 绑定 stream；读回前须同步。
7. 接口声明放入 `include/cann_ops_blas.h`，不设私有平行接口。
8. `y` 原地覆写；除 y 自身输入/输出外，不涉及其他内存重叠语义。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 为 Netlib `cblas_chbmv`（随测试工程提供，fp64 真值、按输出 FLOAT32 收窄），输出 y 全量验证，实部/虚部分别按 FLOAT32 混合容差判定：`|actual−golden| ≤ atol + rtol·|golden|`，rtol=atol=2^-13（1.2e-4），matched_ratio≥0.99，max_abs_error≤1e-2 或 32ULP；Inf/NaN 按任务书 §3.2 单独匹配 | 任务书 §3.2、生态算子开源精度标准 |
| 性能标准 | Atlas 800T A2（910B3），msprof op Device 计时或测试工程 `aclrtEvent` 口径；5 次 warmup + 有效采样 ≥10 次取平均；判定 NPU 平均单次耗时 ≤ `gpu_baseline.csv` 的 `gpu_ms` / 0.8 | 任务书 §3.3、测试指导 README |

算法复杂度：每个输出行遍历有效带宽 `j1-j0+1 ≤ min(n, 2k+1)` 个复数乘加。总复数乘加量约为：

```text
sum_{i=0}^{n-1} (j1(i) - j0(i) + 1)
```

当 `k` 较小时为 `O(n(2k+1))`，当 `k=n-1` 满带退化为 `O(n^2)`。实现通过 1024 元素 chunk 向量化、A/x 的连续/步长搬运和行级多核并行提升有效带宽利用率，不引入额外 workspace 往返。

## 自验用例覆盖

以官方 `chbmv_test.csv` 为主集，另按任务书 §3.5 入参生成规则补充正态分布用例（`GAUSS_NORM_5_2`）：

- `uplo` 全组合的小尺寸基础用例；
- 尺寸扫描：0、1、小质数、2 的幂及 2 的幂 ±1、非对齐值直至大规模；
- 带宽扫描：`k=0`、`1`、小带宽、`k=n-1` 满带退化、典型大带宽；
- 标量特殊值：`alpha=(0,0)`、`(1,0)`、纯虚数、大值；`beta=(0,0)` 等；
- 前导维 `lda` 紧凑与多个 padding 场景；
- `incx/incy` 覆盖 `±1、±2、±3` 及零步长负向用例；
- 边界与负向用例：零维、空指针、非法 `uplo`、非法 `lda`、负维度、零步长、`INT32_MIN` 步长、跨度溢出；其中 `n=0` 叠加非法参数期望 `ACLBLAS_STATUS_SUCCESS`；
- Inf/NaN 特殊值用例；
- 200 条 `TC_PF` 性能用例，按 `gpu_baseline.csv` 的 `(uplo,n,k,incx,incy)` 联合匹配。

测试工程：`test/sbmv/chbmv/`（`CMakeLists.txt`、`chbmv_golden.h`、`chbmv_param.h`），`test/sbmv/chbmv/arch22/`（`chbmv_test.cpp`、`chbmv_test.csv`）；CSV 列与官方文件一一对应，枚举大小写不敏感解析。

## 兼容性分析

新增接口与新增目录，不修改任何既有接口行为；`blas/sbmv/arch35` 实数实现互不影响；arch22 源文件仅随 A2/A3 目标构建。接口签名与 `cublasChbmv` 参数一一对应（handle 及参数顺序一致），无需额外映射。

# 交付与合入

| 交付件 | 说明 |
| --- | --- |
| 设计文档 | 本文档，经 PR 评审合入 tasklist 对应目录 |
| 实现代码 | `blas/sbmv/arch22/`（个人 fork 仓，邀 Ascend-CANN 为开发者） |
| 测试代码 | `test/sbmv/chbmv/` + `test/sbmv/chbmv/arch22/`（readme 说明复现步骤，覆盖全部官方用例并标注精度/性能 case） |
| 自测报告 | `task_submission/` 目录按任务书 §4 模板提交：1 自验证步骤说明.md、2.1 精度自验证报告.xlsx、2.2 精度自验证日志.log、3.1 性能自验证报告.xlsx、3.2 性能自验证日志.log、4.1 内存自验证报告.xlsx、4.2 内存自验证日志.log（精度/性能含用例参数、对比结果与截图、有效采样 ≥10 次平均值） |
| 算子 README | `blas/sbmv/README.md` 增加 `aclblasChbmv` 接口说明，产品支持表标注 Atlas A2/A3 系列产品（含 Atlas 800I A2/A3）：支持 |
