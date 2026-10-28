# aclblasCher 算子设计文档

- 算子：`aclblasCher`（单精度复数厄米特秩-1更新，COMPLEX64）
- 平台：Ascend 950PR（arch35），CANN 9.1.0
- 目标仓：`cann/ops-blas`，实现目录 `blas/syr/arch35/`，测试目录 `test/syr/cher/arch35/`
- 对标接口：cuBLAS `cublasCher`；语义参考 Netlib `cher`

---

# 1 需求背景

## 1.1 需求来源

10 月社区任务《10月社区任务-aclblasCher算子开发（950）》。任务目标是补齐 `cann/ops-blas` 的 BLAS-2 能力：
在 Ascend 950PR 上以 Ascend C（SIMT/AIV）实现句柄式 `aclblasCher`，完成设计、开发、自验与合入。

## 1.2 背景介绍

BLAS 的 Hermitian 秩-1 更新族包含 `cher`（复数厄米特）与实数/复数对称版本 `ssyr`/`csyr`。
`ops-blas` 仓已有 `aclblasSsyr`（实数对称）与 `aclblasCsyr`（复数对称，不共轭），但缺少共轭版本 `aclblasCher`，
调用方无法在 NPU 上完成 `A := alpha·x·x^H + A` 这一 Hermitian 更新原语。

数学定义（列主序，仅 `uplo` 指定三角被引用和更新）：

```
A(i,j) := A(i,j) + alpha · x(i) · conj(x(j))     i,j 在 uplo 指定三角内
A(i,i) := real(A(i,i)) + alpha · |x(i)|²         对角：只累加实部，imag 置 0
```

其中 `alpha` 为 **FLOAT32 实数标量**（cuBLAS `cublasCher` 的默认 pointer mode = HOST）。

## 1.3 同族实现现状（一手源码核对）

依据 `cann/ops-blas` master（提交 `05b96ea`，2026-10-28）逐目录核对：

| 同族资产 | 路径 | 现状 | 对本设计的作用 |
|---|---|---|---|
| aclblasSsyr（实数对称） | `blas/syr/arch35/syr_host.cpp`、`syr_kernel.cpp` | arch35 已有 | 句柄/校验/返回码风格的参照 |
| aclblasCsyr（复数对称，不共轭） | `blas/syr/arch35/csyr_host.cpp`、`csyr_kernel.cpp`、`csyr_kernel.h`、`csyr_tiling_data.h` | arch35 已有（SIMT，56 AIV block 调优） | **主复用对象**：仅“共轭 + 实数 alpha + 对角置零”三处语义差异 |
| aclblasCgeru（复数 rank-1，非共轭） | `blas/ger/arch35/cgeru_kernel.cpp` | arch35 已有 | SIMT 调度与索引写法的第二参照 |
| aclblasCherk / Cher2k / Cherkx（L3） | `blas/herk/`、`blas/her2k/`、`blas/herkx/` | arch35 已有 | 仅确认命名不冲突（L3 算子，与本 L2 算子不同） |
| 复数 CSV 驱动 ST | `test/syr/csyr/`（param/golden/arch35 test+wrapper） | 已有 | 测试工程骨架（参数、golden、guard 字节、边界矩阵） |

`include/cann_ops_blas.h` 当前**没有** `aclblasCher` 声明（只有 `aclblasCherk`），本设计新增声明。

## 1.4 与 aclblasCsyr 的差异表

| # | 差异 | aclblasCsyr | aclblasCher | 本设计处理 |
|---|---|---|---|---|
| 1 | 向量转置方式 | `x·x^T`（不共轭） | `x·x^H`（共轭转置） | 逐元素做 `conj(x_col)`：`delta = (alpha·x_row)·conj(x_col)` |
| 2 | alpha 类型 | COMPLEX64（实部+虚部） | **FLOAT32 实数** | tiling 只携带 `alphaReal`；kernel 不再处理 `alphaImag` |
| 3 | 对角元素 | 普通复数（虚部照常累加） | 只累加 `alpha·|x(i)|²`，虚部**强制置 0** | 元素级更新函数按 `row == col` 分支写 `imag = 0` |
| 4 | 输入矩阵假设 | 对称（无对角约束） | 厄米特（对角虚部假定 0） | 实现不读未引用三角；对角旧虚部按 Netlib 语义丢弃 |
| 5 | golden 来源 | 自实现（标准 CBLAS 无 `csyr`） | `cblas_cher`（Netlib 语义，标准 CBLAS 提供） | 测试工程直接调用 `cblas_cher` 现算 |
| 6 | 性能基线 | — | 任务书 §3.3 要求 ≤ 2.5× GPU | 沿用同一套 SIMT 分档 + 逐 case 上限表（§3.4） |

## 1.5 需求拆解（R01–R10，每条给设计落点）

| 编号 | 需求（任务书 §2） | 设计落点 |
|---|---|---|
| R01 | 功能：`A = alpha·x·x^H + A`，列主序，仅 uplo 三角更新 | §3.1 数学定义；`cher_kernel.cpp` 全部路径只写三角内元素 |
| R02 | 参数语义与 cuBLAS `cublasCher` 逐参数对齐 | §2.1 接口原型；§2.2 参数契约表 |
| R03 | 对角元素更新后虚部置 0 | §3.2.2 `CherUpdateValue(..., diagonal)`：`value.y = 0` |
| R04 | 未引用三角与 lda padding **不读不写** | 全部路径按三角枚举；ST 用哨兵值逐字节比对（§4.1） |
| R05 | `n = 0`、`alpha = 0.0` 合法 no-op | §2.3 校验顺序：`n == 0` 早退；`alpha = 0` 早退 |
| R06 | 支持任意非零 `incx`（含负步长） | §3.1 索引映射；`incx = INT_MIN` 单独拒绝（§2.6 口径表） |
| R07 | 接口声明入 `include/cann_ops_blas.h`，不设 950PR 私有平行接口 | 头文件仅新增一段声明；实现按 SOC 目录 `arch35/` 组织 |
| R08 | 句柄式 BLAS 接口，handle 绑定 stream 直调 kernel | `cher_host.cpp`：`handle->stream` 直接下发，异步入队 |
| R09 | 1200 条 CSV 用例（1000 精度 + 200 性能）全量精度达标 | §4 测试矩阵与判据 |
| R10 | 性能：平均单次 ≤ 2.5 × GPU 标杆（≥ 0.4×） | §3.4 性能方案与逐 case 上限 |

---

# 2 需求分析

## 2.1 公共接口原型

`include/cann_ops_blas.h` 新增（与 `aclblasSsyr` 同构，alpha 为 FLOAT32 实数指针）：

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

## 2.2 参数契约表

| 参数 | 方向 | 类型 | 形状/长度 | 值域 | 异常行为 |
|---|---|---|---|---|---|
| handle | 入 | aclblasHandle_t（Host） | — | 已创建句柄 | nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 入 | aclblasFillMode_t（Host） | — | `ACLBLAS_UPPER`(121) / `ACLBLAS_LOWER`(122) | 其它值 → `ACLBLAS_STATUS_INVALID_ENUM` |
| n | 入 | int（Host） | — | `n >= 0` | `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE`；`n == 0` → SUCCESS |
| alpha | 入 | `const float*`（Host；兼容 Device） | 1 个 FP32 | FLOAT32 全集 | nullptr → INVALID_VALUE；`alpha == 0.0` → SUCCESS（不更新 A） |
| x | 入 | `const aclblasComplex*`（Device） | 物理 ≥ `1+(n-1)·abs(incx)` | 实/虚部 FLOAT32 全集 | `n>0 且 alpha≠0` 时 nullptr → INVALID_VALUE |
| incx | 入 | int（Host） | — | `incx != 0`，可负 | `0` 或 `INT_MIN` → INVALID_VALUE |
| A | 入/出 | `aclblasComplex*`（Device） | `lda × n` 列主序 | 实/虚部 FLOAT32 全集 | `n>0 且 alpha≠0` 时 nullptr → INVALID_VALUE |
| lda | 入 | int（Host） | — | `lda >= max(1, n)` | 不满足 → INVALID_VALUE |

地址公式（第 `j` 列、第 `i` 行，0 基）：
`A(i,j)` 复数下标 = `j · lda + i`；`x(i)` 复数下标 = `incx > 0 ? i·incx : (n-1-i)·|incx|`。

## 2.3 校验顺序与边界裁定

采用与 `aclblasCsyr` 完全一致的顺序（任务书 §2.5 逐字要求）：

```
handle 判空
  → n < 0  → INVALID_VALUE
  → n == 0 → SUCCESS（不再校验 uplo/lda/incx/alpha/x/A）
  → uplo   → INVALID_ENUM
  → lda >= max(1, n) → INVALID_VALUE
  → incx != 0（并拒绝 INT_MIN） → INVALID_VALUE
  → alpha != nullptr → INVALID_VALUE
  → alpha == 0.0 → SUCCESS（不再校验 x/A）
  → x、A 非空 → INVALID_VALUE
  → 派发 kernel
```

**多种非法参数同时存在时，以上表顺序为准。** 例如 `n = 0` 且 uplo/lda/incx/alpha/x/A 全非法时返回 SUCCESS；
`alpha = 0` 但 `incx = 0` 或 `lda < n` 时仍返回结构错误码（结构校验早于 alpha no-op）。

## 2.4 异常行为表

| 场景 | 判定 | 行为 |
|---|---|---|
| handle 为空 | 结构性 | INVALID_VALUE 之前先返回 HANDLE_IS_NULLPTR |
| n < 0 | 结构性 | 不访问任何指针 |
| n == 0 | 合法 | SUCCESS，A 不变 |
| uplo 非 121/122 | 结构性 | INVALID_ENUM |
| lda < max(1,n) | 结构性 | INVALID_VALUE |
| incx == 0 / INT_MIN | 结构性 | INVALID_VALUE |
| alpha 指针空 | 结构性 | INVALID_VALUE |
| alpha == ±0.0 | 合法 no-op | SUCCESS，A 不变，x/A 允许为空 |
| n > 0、alpha ≠ 0、x 或 A 为空 | 结构性 | INVALID_VALUE |
| x/A 含 Inf/NaN | 合法输入 | 按 IEEE 传播；ST 按 §4.2 的非有限值规则判定 |
| 设备查询失败 / alpha 读回失败 | 运行期 | EXECUTION_FAILED / INTERNAL_ERROR |

## 2.5 支持域

- `n ∈ [0, 2^31)`（接口无上限；SIMT 快路径按 `n` 分档，4096 为调优阈值而非尺寸限制）
- `lda ∈ [max(1,n), 2^31)`；快路径要求 `lda <= 4096`，否则走标量 GM 路径
- `incx != 0` 且 `incx != INT_MIN`
- 不涉及非连续 tensor、broadcast、dynamic shape、视图返回

## 2.6 任务书口径差异与本设计采用口径（冻结表）

| # | 差异点 | 任务书/测试包原文 | 本设计采用 | 依据 |
|---|---|---|---|---|
| Q1 | `incx = INT_MIN` | §2.4 仅约束 `incx ≠ 0`；随任务测试包 README 明确“另加 INT_MIN 负向用例，期望 INVALID_VALUE，属实现侧增强约束” | 拒绝 `INT_MIN`（INVALID_VALUE） | 测试包原文 + `1+(n-1)·|incx|` 的 int32 长度估算溢出 |
| Q2 | alpha 指针位置 | §2.4 写 “Host 内存（与 cuBLAS 默认 pointer mode = HOST 一致）” | **契约要求 Host**；额外兼容 Device 指针（D2H 读回后建 tiling） | 与 `aclblasSsyr`/`aclblasCsyr` 仓内惯例一致，不改变 Host 契约 |
| Q3 | 输入分布 | §3.5 要求均匀/正态各 50% | 采用随包 `cher_test.csv` 的均匀 `RANDOM_NORM_5_5`（[-5,5]） | 随包 `gen_csv.py` 明示“正态分布依赖测试工程扩展，本次以均匀生成”；任务书 §3.5 生成的输入域为测试数据规则，非算子行为 |
| Q4 | 对角虚部 | §2.1 要求“更新后强制置为 0”；Netlib `cher`：`A(j,j) = REAL(A(j,j)) + alpha·|x(j)|²` | 只累加实部，旧虚部丢弃，写 0 | 任务书与 Netlib 原文一致 |
| Q5 | 性能判据 | §3.3：`msprof op` Task Duration，5 次 warmup 不计入，10 个有效样本取平均；≤ 2.5×GPU | 按此口径；另用 event 计时 runner 做同口径复核 | 任务书 §3.3 逐字 |
| Q6 | 未引用三角/padding | §3.2 第 1 条：以哨兵值比对确认未被读取或改写 | ST 在调用前写哨兵、调用后逐字节比对；另加首尾 guard 字节越界检测 | 任务书 §3.2 + 仓内 `test/syr/csyr` 工程惯例 |
| Q7 | 复数乘法舍入 | 生态精度标准 `atol=rtol=2^-13`、`matched_ratio>=0.99`、`max(1e-2, 32·ULP)` | 每个乘积独立舍入（`__fma(a,b,0)`），不做乘加收缩；保证 Inf/NaN 分类一致 | 精度标准 + 同族 kernel 既有写法 |

---

# 3 详细设计

## 3.1 算子分析

### 3.1.1 逐元素数学

记 `x(i) = xr_i + i·xi_i`，`alpha = a`（实数）：

```
t(i)   = a·x(i)     = (a·xr_i, a·xi_i)                 （行标量）
delta  = t(i)·conj(x(j))
       = ( a·xr_i·xr_j + a·xi_i·xi_j ,  a·xi_i·xr_j − a·xr_i·xi_j )
A(i,j) += delta                                          （i,j 在 uplo 三角内）
A(i,i)  = ( A(i,i).real + delta.real , 0 )
```

与 Netlib `cher`（`temp = alpha·conjg(x(j))`，`A(i,j) += x(i)·temp`）逐项等价。

### 3.1.2 复数算术与数值边界

- 每个乘积独立舍入：写 `__fma(a, b, 0.0f)` 而非 `a*b` 的裸表达式，避免编译器把 `p1 ± p2` 收缩成单次 FMA
  ——收缩会改变 `Inf`/`NaN` 的产生位置（有限乘积上溢时 `Inf − Inf` 与两次舍入的 `Inf` 不同）。
- **不做零分量捷径**：`x(i)=0` 或 `alpha.x=0` 时仍按上述公式计算，保证 `0 × Inf = NaN` 的传播与 golden 一致。
- 对角：`imag` 直接写 0（不累加），与 Netlib 丢弃旧对角虚部的语义一致。
- `incx < 0`：逻辑下标 `i` 映射到物理存储 `(n−1−i)·|incx|`，与 Netlib 反向遍历一致。

### 3.1.3 复杂度

三角内元素数 `n(n+1)/2`，每元素 1 次读 + 1 次写（各 8 B）+ 常数次浮点乘加。
时间复杂度 `O(n²)`，访存主导：总流量 `≈ 16·n(n+1)/2 = 8n(n+1)` 字节。

## 3.2 算子实现

### 3.2.1 Host 侧设计

`blas/syr/arch35/cher_host.cpp`：

1. `aclblasCher` 按 §2.3 顺序校验；
2. alpha 解析：`CheckPtrLocation` 判定 Host/Device；Device 指针经 handle stream `D2H + 同步` 读回
   （复用仓内 `ReadHostScalarFromDevice`）；
3. `alpha == 0.0` 早退（在 x/A 判空之前）；
4. `GetAivCoreCount()` 查询 AIV 核数，`numBlocks = min(n, 核数)`；线程数 `n <= 512 ? 512 : 1024`；
5. 填 `CherTilingData`（`n/lda/uplo/alphaReal/alphaIsDevice/incx/numThreads`），调用 `cher_kernel_do`
   异步入队（不做隐式同步，符合 arch35 异步契约）。

### 3.2.2 Kernel 侧设计

`blas/syr/arch35/cher_kernel.cpp`：SIMT（`__simt_vf__` + `asc_vf_call`）AIV kernel，**直接在 GM 上原地读改写**，
不分配 UB tile、不需要 workspace。元素级更新函数：

```cpp
float2 CherUpdateValue(float2 value, float2 xCol, float axReal, float axImag, bool diagonal) {
    float dR = fma(axReal, xCol.x, 0) + fma(axImag, xCol.y, 0);
    float dI = fma(axImag, xCol.x, 0) - fma(axReal, xCol.y, 0);
    value.x += dR;
    value.y = diagonal ? 0.0f : (value.y + dI);
    return value;
}
```

其中 `axReal/axImag = alpha·x(row)`（单位 alpha 时直接取 `x(row)` 分量）。全部路径只调用这一个更新函数，
因此“共轭 + 实数 alpha + 对角置零”的差异集中在**一处**，其余调度逻辑与 `aclblasCsyr` 逐行同构。

## 3.2.3 路径状态表

| 路径 | 触发条件 | 状态 |
|---|---|---|
| 标量 GM 路径（`CherGm`） | 负步长、4B 对齐、`n < 32`、`n > 4096`、`lda > 4096`、Device alpha | 已有代码（**待 NPU 验证**） |
| tiny 单位三角（`CherTinyUnitTriangle`） | 8B 对齐 + `incx=1` + `lda=n` + `n < 32` + Host `alpha=1` | 已有代码（待 NPU 验证） |
| 分组列（`CherUnitAlphaGroupedColumns`） | `incx=1` + `n ∈ (512, 1024]` + Host `alpha=1` | 已有代码（待 NPU 验证） |
| 配对列（`CSyr` 同构，`n ∈ (1024, 2048]`） | 同上 | 已有代码（待 NPU 验证） |
| 64×64 三角分块（`CherTriangleTiles`） | `incx=1` + `1024 < n <= 4096` + `lda <= 4096` + Host alpha ≠ 0 + 56 blocks × 1024 threads | 已有代码（待 NPU 验证） |
| 精确尺寸特化（`CherUnitAlphaUpperTriangle<9>/<11>`、`CherUnitAlphaLowerTriangle1024`） | `alpha=1` 且 `n = lda = 512 / 1024 / 2048` 对应 uplo | 已有代码（待 NPU 验证） |
| 循环列（`CherCyclicColumns`） | 其余 `32 <= n <= 4096` 的 8B 对齐情形（含非单位 alpha） | 已有代码（待 NPU 验证） |

> 上表“待 NPU 验证”表示代码已按 `aclblasCsyr` 已验证结构落地、尚未在本任务设备上跑过全量；
> 验证结论将回填自测报告，不把设想写成已实现。

### 3.2.4 实现流程图

现状（master）：`aclblasCher` 不存在，调用方无法在 NPU 上完成 Hermitian 秩-1 更新；同族 `aclblasCsyr`
走“句柄校验 → tiling → SIMT 分档 kernel”的链路。本设计的实现流程与 `aclblasCsyr` 同构，只在元素级
更新函数上引入共轭与对角置零：

```mermaid
flowchart TD
    A[调用 aclblasCher] --> B{handle 非空?}
    B -- 否 --> B1[返回 HANDLE_IS_NULLPTR]
    B -- 是 --> C{n < 0?}
    C -- 是 --> C1[返回 INVALID_VALUE]
    C -- 否 --> D{n == 0?}
    D -- 是 --> D1[返回 SUCCESS 不更新 A]
    D -- 否 --> E{uplo 合法?}
    E -- 否 --> E1[返回 INVALID_ENUM]
    E -- 是 --> F{lda >= max 1,n?}
    F -- 否 --> F1[返回 INVALID_VALUE]
    F -- 是 --> G{incx != 0 且 != INT_MIN?}
    G -- 否 --> G1[返回 INVALID_VALUE]
    G -- 是 --> H{alpha 指针非空?}
    H -- 否 --> H1[返回 INVALID_VALUE]
    H -- 是 --> I{Host/Device 指针解析}
    I --> J{alpha == 0.0?}
    J -- 是 --> J1[返回 SUCCESS 不更新 A]
    J -- 否 --> K{x、A 非空?}
    K -- 否 --> K1[返回 INVALID_VALUE]
    K -- 是 --> L[GetAivCoreCount + CalCherTilingData]
    L --> M{pairAligned 且 incx==1?}
    M -- 否 --> N[CherGm 标量 GM 路径]
    M -- 是 --> O{n < 32?}
    O -- 是 --> P[CherTinyUnitTriangle]
    O -- 否 --> Q{n > 4096 或 lda > 4096?}
    Q -- 是 --> N
    Q -- 否 --> R{alph==1 且 n==lda==512/1024/2048?}
    R -- 是 --> S[精确尺寸特化 packed]
    R -- 否 --> T{1024 < n <= 4096 且 56 block x 1024 thread?}
    T -- 是 --> U[CherTriangleTiles 64x64 三角分块]
    T -- 否 --> V[CherCyclicColumns / GroupedColumns]
    N --> W[CherUpdateValue: A += alpha*x(i)*conj(x(j)); 对角 imag=0]
    P --> W
    S --> W
    U --> W
    V --> W
    W --> X[异步入队返回 SUCCESS]
```

与同族 `aclblasCsyr` 的流程差异表：

| 环节 | aclblasCsyr | aclblasCher（本设计） |
|---|---|---|
| 元素级更新 | `A += alpha*x(i)*x(j)`（普通复乘） | `A += alpha*x(i)*conj(x(j))`（共轭乘） |
| alpha 解析 | 复数标量（实部+虚部） | 实数标量（仅 `alphaReal`），无 `alphaImag` 分支 |
| 对角元素 | 普通复数累加 | 只累加 `alpha·|x(i)|²` 到实部，虚部写 0 |
| 校验顺序/分档/特化 | 基线 | 完全沿用（见 §3.2.3 路径表） |


### 3.2.5 忠实移植声明

本 kernel 以 `blas/syr/arch35/csyr_kernel.cpp`（master 已合入）为母本**逐函数移植**，差异仅限：

1. 符号重命名（`CSYR_*`/`Csyr*` → `CHER_*`/`Cher*`）；
2. 复数乘改为共轭乘（`x_col` 虚部参与符号翻转），由 `CherUpdateValue` 统一承担；
3. 对角元素虚部写 0（同上函数 `diagonal` 分支）；
4. 删除 `alphaImag` 相关分支（alpha 为实数；单位 alpha 判定只剩 `alphaReal == 1.0f`）。

调度阈值、分块尺寸、packed 特化、三角分块编号反解、`float2` 配对装载、56 block 固定步长特化**均未改动**。
逐文件 diff 与“对齐证明”将在代码 PR 中给出。

## 3.3 内存 / workspace

- 显式设备内存分配：**0 字节**（不使用 workspace，不需要 host 侧 staging buffer）。
- 该算子直接在 GM 原地读改写，`x` 只读；句柄仅携带 stream。

## 3.4 性能方案与逐 case 上限

判据（任务书 §3.3）：`msprof op` 采集单算子 `Task Duration`，warmup 5 次不计入，取 10 个有效样本均值；
平均单次耗时 ≤ GPU 标杆 / 0.4（即 ≥ 0.4× GPU 吞吐）。随包 `gpu_baseline.csv` 前 4 条即任务书典型 case：

| case | uplo | n | GPU 标杆 (µs) | NPU 上限 (µs) = 标杆/0.4 | 设计手段 |
|---|---|---|---|---|---|
| cher-base-000 | UPPER | 512 | 5.574 | 13.94 | 精确尺寸特化 `<9>`（alpha=1、紧凑 lda） |
| cher-base-001 | LOWER | 1024 | 5.937 | 14.84 | 精确尺寸特化 `LowerTriangle1024` |
| cher-base-002 | UPPER | 2048 | 11.113 | 27.78 | 精确尺寸特化 `<11>` / 配对列 |
| cher-base-003 | LOWER | 4096 | 54.050 | 135.10 | 64×64 三角分块（56 blocks × 1024 threads） |

Roofline 核算（按 910B3 实测 HBM ≈ 1.6 TB/s 的下界估计，950PR 不低于该值）：
`n = 4096` 三角流量 `8·4096·4097 ≈ 134 MB` → 上限约 `84 µs < 135 µs`，判定物理可达；
`n = 2048` 约 `33.6 MB → 21 µs < 27.8 µs`；`n = 1024` 约 `8.4 MB → 5.2 µs < 14.8 µs`。
小尺寸（`n <= 512`）受 launch 固定开销影响，上限 13.9 µs 远高于典型 2–5 µs 的空 kernel 开销。

其余 196 条性能用例以同一 CSV 逐条判定（判据同上），报告给出**最小值**与**保守复测值**，不用均值替代。

## 3.5 兼容性分析

- 新增接口：`include/cann_ops_blas.h` 仅新增 1 段声明，不改动既有声明（逐 token diff 可证）。
- 新算子无历史版本，不涉及行为兼容；`aclblasCherk` 为 L3 秩-k 更新，与本 L2 算子名字与语义均不同，不冲突。
- arch22（A2/A3）本版本不提供实现：README 产品支持表标注“Ascend 950PR：支持 / A2/A3：不支持”，
  不定义 950PR 私有平行接口，接口签名供其他产品线后续直接实现。

## 3.6 风险与回退

| 风险 | 触发条件 | 影响 | 缓解/回退 |
|---|---|---|---|
| SIMT kernel 首次上机不可编译 | arch35 编译器对 `__simt_vf__`/`float2` 的约束差异 | 无法交付 | 按 `csyr_kernel` 逐字保留写法；编译失败即退回与该文件同构的最小改动 |
| 8B 对齐快路径越界写 | x/A 仅 4B 对齐却被 `float2` 访问 | 破坏 GM | kernel 入口检查 `pairAligned`，不满足即走标量 GM 路径；ST 用 4B 对齐用例 + 首尾 guard 字节覆盖 |
| 溢出边界与参考实现分歧 | `x(i)=±FLT_MAX` 导致实部饱和 Inf | 精度判定争议 | golden 采用 Netlib 顺序自实现（任务书指定 Netlib 语义）；ST 保留该用例并附对照实验 |
| 性能不达标 | 小尺寸固定开销 / 大尺寸带宽 | 验收失败 | 先算 roofline（§3.4）；小尺寸先拆固定开销，大尺寸已接近带宽上限则按上限披露 |
| alpha=0/占用空指针的 quick return 误判 | 校验顺序写错 | 与任务书不符 | 校验顺序固定（§2.3），ST 用 25 条边界用例覆盖 |
| 回退开关 | 任何路径异常 | — | 单文件实现，`git revert` 对应 commit 即可完全回退；不改动 csyr/其他算子任何字节 |

---

# 4 可维可测分析

## 4.1 测试矩阵（随包 CSV 逐前缀统计）

`test/syr/cher/arch35/cher_test.csv` 共 **1200** 条，逐前缀计数（由 CSV 直接统计，不以 README 分类估计代替）：

| 前缀 | 含义 | 条数 |
|---|---|---|
| TC_L0 | 小尺寸基础（uplo × n∈{4,8}） | 4 |
| TC_SQ | 尺寸扫描（n∈{1..2048} 含质数/2 的幂 ±1/非对齐） × uplo | 46 |
| TC_AB | alpha 特殊值（0/±1/±0.5/±2/3/100/0.001） × uplo | 20 |
| TC_LD | lda padding（n+1/n+8/n+16） × n∈{16,32,64} × uplo | 18 |
| TC_INC | incx ∈ ±1/±2/±3 × uplo | 12 |
| TC_FL | 填充模式（均匀/全零/交替/极端/Inf/NaN，分别作用于 A 或 x） | 12 |
| TC_CV | 中等尺寸覆盖（10~400） × uplo | 16 |
| TC_ED | 边界与负向（n=0/alpha=0/空指针/非法枚举/非法 lda/负 n/incx=0/INT_MIN/特殊标量） | 25 |
| TC_EX | 扩展精度（尺寸池 × uplo × alpha × 步长 × padding 确定性采样） | 847 |
| **精度小计** | | **1000** |
| TC_PF | 性能（4 条任务书典型 case + 小尺寸 + 对数扫描 + 预算内混合） | 200 |

uplo 覆盖：UPPER 607 / LOWER 589 / 非法枚举 4；incx 覆盖：1(938) / -1(59) / 2(53) / -3(51) / 3(50) / -2(43) / 0(5) / INT_MIN(1)。

除 CSV 外，`cher_test.cpp` 另含固定用例：null handle、n<0、n=0 全非法叠加、alpha=0 空指针、
非法枚举/lda/incx/alpha 指针、**alpha 分布 100 组**（均匀 50 + 正态 50）、
边界 200+ 组（n 跨 31/32/33/511/512/513/1023/1024/1025/2047/2048/2049/4096/4097，
含 Inf/NaN/上溢/次正规、负步长、4B 对齐与非对齐地址、guard 字节）。

## 4.2 精度判据

- golden：`cblas_cher`（`CblasColMajor`，Netlib 语义）对**同一份实际输入**现算；不复用被测实现。
  `incx < 0` 时按逻辑序物化后以单位步长调用，等价于 Netlib 的索引语义。
- 比对范围：仅 uplo 指定三角（含对角），**实部、虚部分别**按 FLOAT32 判定。
- 逐元素：`|actual − golden| ≤ atol + rtol·|golden|`，`atol = rtol = 2^-13`；
  逐元素硬上限 `max(1e-2, 32·ULP(|golden|))`；`matched_ratio ≥ 0.99`。
- 非有限值：golden 与 actual 同为 NaN、同为同号 Inf、或一方 NaN 一方 ±Inf 视为等效；一方有限一方非有限判失败；
  非有限元素不计入 `matched_ratio` 与 `max_abs_error` 统计，单独记录总数与通过数。
- 对角：额外断言 `imag == 0`。
- 写保护：未引用三角与 `lda` padding 行调用前后逐字节一致；x 缓冲逐字节一致；x/A 首尾各 40 B guard 未被改写。

## 4.3 性能判据与计时边界

- 主判据：`msprof op` 的 `Task Duration(us)`（warmup 5 次不计入，10 个有效样本均值），`≤ GPU/0.4`。
- `cher_perf`（独立 runner）：每 case warmup 5 次 → 10 次连发，`aclrt` event 夹住测量窗口取均值；
  A 的清零、golden 计算、H2D/D2H 全部在计时窗口外。
- **不使用 GTest 总耗时替代设备调用耗时**（GTest 时间含 host 准备、golden、比对，仅作保守上界参考）。

## 4.4 支撑资产

| 资产 | 路径 |
|---|---|
| 算子实现 | `blas/syr/arch35/cher_host.cpp`、`cher_kernel.cpp`、`cher_kernel.h`、`cher_tiling_data.h` |
| 测试工程 | `test/syr/cher/{cher_param.h, cher_golden.h, README.md, CMakeLists.txt}`、`test/syr/cher/arch35/{cher_test.cpp, cher_npu_wrapper.h, cher_perf.cpp, cher_test.csv, gpu_baseline.csv}` |
| 接口文档 | `blas/syr/README.md#aclblascher`（含产品支持表、约束、返回值、调用示例） |
| 接口声明 | `include/cann_ops_blas.h`（新增 `aclblasCher`） |

---

# 5 支持硬件

| 芯片版本 | 涉及勾选 |
|---|---|
| Ascend 950PR / 950DT（arch35） | √ |
| Atlas A3 训练/推理系列（arch22） | 不支持（本版本仅 arch35 实现） |
| Atlas A2 训练/推理系列（arch22） | 不支持（本版本仅 arch35 实现） |

# 6 参考依据及核对版本

1. 任务书：`aclblasCher_Atlas950PR_task_doc.md`（随任务发放，2026-10-28）
2. `cann/ops-blas` master `05b96ea`（2026-10-28）：`blas/syr/arch35/csyr_*`、`blas/ger/arch35/cgeru_*`、`test/syr/csyr/*`、`include/cann_ops_blas.h`
3. 生态算子开源精度标准：`cann/opbase` `docs/zh/ops_precision_standard/mixed_tolerance_standard.md`（任务书链接版本）
4. Netlib BLAS `cher.f`：https://www.netlib.org/blas/cher.f
5. cuBLAS `cublasCher`：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-her
6. 随包测试资产：`cher_test.csv`（1200 条）、`gpu_baseline.csv`（200 条 H100 基线）、`gen_csv.py`、`verify_accuracy.py`、`verify_performance.py`、`README.md`
