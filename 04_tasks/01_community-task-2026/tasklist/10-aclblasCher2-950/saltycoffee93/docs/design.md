# 需求背景（required）

## 需求来源

社区任务【10月社区任务-aclblasCher2算子开发（950）】，任务书：`aclblasCher2_Atlas950PR_task_doc.md`。目标是在昇腾 NPU（Atlas 950PR）上使用 Ascend C 编程语言开发单精度复数 Hermitian 秩2更新算子 `aclblasCher2`，补齐 ops-blas 开源仓（https://gitcode.com/cann/ops-blas ）BLAS 能力，对标 cuBLAS `cublasCher2` 与 Netlib `cher2.f`。

## 背景介绍

### aclblasCher2 算子说明

Hermitian 秩2更新（Hermitian rank-2 update）是 BLAS Level 2 中的核心例程，广泛用于 Cholesky 分解、对称特征值问题、共轭梯度类迭代法等线性代数流程中。计算公式为：

```
A = alpha * x * y^H + conjg(alpha) * y * x^H + A
```

其中：

- `alpha` 为单精度复数标量（complex64）；
- `x`、`y` 为 n 元素复数向量（complex64，支持任意非零步长 incx/incy，含负步长）；
- `A` 为 n×n Hermitian 复数矩阵（complex64，列主序存储），原地更新；
- `y^H`、`x^H` 表示共轭转置，`conjg(alpha)` 为 alpha 的复共轭；
- 仅引用/更新 `uplo` 指定的三角（`ACLBLAS_UPPER` 上三角 / `ACLBLAS_LOWER` 下三角），未引用三角不读不写；对角元素虚部假定为 0，更新后强制置 0（语义参考 Netlib `cher2.f`）。

### ops-blas 仓现状分析

- ops-blas 仓 `include/cann_ops_blas.h` 当前**尚无** `aclblasCher2` 声明，头文件需新增该声明；接口声明放入公共头文件，与其他产品线共用同一 API，禁止定义 950PR 私有平行接口。
- 仓内同族实数算子 `aclblasSsyr2` 已在 `blas/syr2/arch35/` 目录实现，本算子采用相同的工程模式（Kernel 直调、handle 绑定 stream 直调 NPU kernel），实现代码放于 `blas/syr2/arch35/`，与同族算子同目录。
- 复数类型 `aclblasComplex` 以仓 `include/cann_ops_blas_common.h` 定义为准（实部/虚部各 float32）。
- 精度 golden 由 `cblas_cher2`（Netlib CBLAS 复数单精度，`CblasColMajor`）随测试工程提供；测试用例 CSV（1200 条：1000 精度 + 200 性能）与 GPU H100 性能基线（`gpu_baseline.csv`）已随任务提供。

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言在 Atlas 950PR 上实现 `aclblasCher2` 算子（complex64，Kernel 直调模式），功能、参数语义与 cuBLAS `cublasCher2` 完全对齐；性能不低于 GPU H100 标杆的 0.4 倍（NPU 平均单次耗时 ≤ GPU 标杆 × 2.5）。

## 需求拆解

1. 接口实现：按任务书 §2.3 签名在 `include/cann_ops_blas.h` 新增 `aclblasCher2` 声明，host 侧完成参数校验、tiling 计算与 kernel 下发。
2. 功能实现：
   - Hermitian 秩2更新主计算 `A = alpha·x·y^H + conjg(alpha)·y·x^H + A`；
   - 仅读/写 uplo 指定三角，未引用三角不读不写；对角元素更新后虚部强制为 0；
   - 支持任意非零 incx/incy（含负步长，按 Netlib 语义反向遍历）；
   - 支持 lda ≥ max(1,n) 的 padding 场景（列主序，未引用的 padding 行不读写）。
3. 参数校验：结构参数校验先于 no-op 判定——n ≥ 0、uplo ∈ {UPPER, LOWER}（非法返回 `ACLBLAS_STATUS_INVALID_ENUM`）、incx ≠ 0、incy ≠ 0、lda ≥ max(1,n)、alpha 指针非空（非法返回 `ACLBLAS_STATUS_INVALID_VALUE`）；handle 为 nullptr 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；校验通过后 `n = 0` 或 `alpha = (0,0)` 为合法 no-op（返回 `ACLBLAS_STATUS_SUCCESS`，不访问 x/y/A）；`n > 0` 且 `alpha ≠ (0,0)` 时 x/y/A 不得为空。
4. 性能实现：合理分核与 UB 分块，A 读写为访存主体（n=4096 时 A 为 4096×4096×8B = 128MB），目标达到 GPU H100 标杆 0.4 倍以上性能。
5. 测试自验：基于 ops-blas 仓 test 框架加载 `cher2_test.csv`（1200 条）完成精度与性能自测，输出自测报告。

# 详细设计（required）

## 算子分析

### 数学公式

对 uplo 指定三角内的每个元素（列主序，0-based）：

```
A(i,j) += alpha·x(i)·conj(y(j)) + conj(alpha)·y(i)·conj(x(j))    i ≤ j（UPPER）或 i ≥ j（LOWER）
A(j,j) += 2·Re(alpha·x(j)·conj(y(j)))                             对角元（结果为实数，虚部置 0）
```

展开为实部/虚部运算（各分量 float32）：

```
令 x = (xr, xi), y = (yr, yi), alpha = (ar, ai)
w1 = alpha·x   = (ar·xr − ai·xi, ar·xi + ai·xr)     // alpha·x(i)
w2 = conj(alpha)·y = (ar·yr + ai·yi, ar·yi − ai·yr)  // conjg(alpha)·y(i)
A(i,j).re += w1.re·yr(j) + w1.im·yi(j) + w2.re·xr(j) + w2.im·xi(j)
A(i,j).im += w1.im·yr(j) − w1.re·yi(j) + w2.im·xr(j) − w2.re·xi(j)
```

说明：`alpha·x(i)·conj(y(j))` 中 `alpha·x(i)` 仅依赖 i、`conj(y(j))` 仅依赖 j，故将 `alpha·x(i)` 与 `conjg(alpha)·y(i)` 的复数乘法先行算好（每核每向量元素一次），列内逐元素只需 4 次实数乘加 ×2 项，避免重复计算复数乘法。

### 支持数据类型

| 操作数 | 数据类型 | 说明 |
| --- | --- | --- |
| alpha | complex64（`aclblasComplex`） | Host 内存标量指针，host 侧直接解引用，不做设备指针识别与 D2H 读回 |
| x / y | complex64（`aclblasComplex`） | Device 内存向量，ND，步长由 incx/incy 描述 |
| A | complex64（`aclblasComplex`） | Device 内存 n×n 矩阵，列主序，原地更新 |

### 支持形状

- x/y：逻辑长度 n（n ∈ [0, 4096]），物理长度 ≥ 1+(n−1)×|inc|；
- A：lda×n（lda ≥ max(1,n)）；
- 不涉及广播、dynamic shape（n 为运行时入参）、确定性计算要求。

## 算子实现

### 实现方案

采用 ops-blas 仓 Kernel 直调工程模式：host 侧实现句柄式 BLAS 接口（handle 绑定 stream 直调 NPU kernel），代码位于 `blas/syr2/arch35/`，与 `aclblasSsyr2` 同目录族。整体流程为：host 侧参数校验 → quick return 判定 → tiling 计算 → 通过 handle 的 stream 直调 kernel（`<<<>>>` 方式）→ 异步返回。

#### 3.2.1 host 侧设计

**1. 参数校验（顺序固定，结构校验先于 no-op）**

```
1) handle == nullptr                      → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
2) uplo ∉ {UPPER, LOWER}                  → ACLBLAS_STATUS_INVALID_ENUM
3) n < 0 / incx == 0 / incy == 0 /
   alpha == nullptr /
   lda < max(1, n)                        → ACLBLAS_STATUS_INVALID_VALUE
4) n == 0 或 alpha == (0,0)               → ACLBLAS_STATUS_SUCCESS（quick return，不访问 x/y/A）
5) n > 0 且 alpha ≠ (0,0) 时 x/y/A 为空   → ACLBLAS_STATUS_INVALID_VALUE
```

alpha = (0,0) 的判定在 host 侧解引用后进行；alpha = (0,0) 但结构参数非法（如 lda 非法、incx=0）仍返回对应错误码，与测试用例 `alpha0_invalid_lda` / `alpha0_incx0` 口径一致。

**2. 负步长规整（host 侧，kernel 不感知负步长）**

采用标准 BLAS 指针调整技巧：当 incx < 0（incy 同理）时，将首元素指针调整为 `x + (n−1)·incx`，步长取绝对值，kernel 内部始终按正步长从逻辑第 0 个元素遍历到第 n−1 个元素。这样等价于 Netlib 语义的反向遍历，kernel 侧消除符号分支。

**3. tiling 策略**

A 的读写是本算子访存主体（每个被更新元素 8B 读 + 8B 写），x/y 各仅读一遍，tiling 以"按列分块、满核均分"为原则：

- **分核策略（按列分核）**：将 n 列按核数 coreNum 均分，优先满核；不能整除时余列分配给前几个核（大核/小核模式），每核处理连续的一段列 `[colBegin, colEnd)`。核间无数据依赖（不同列写不同 A 列，互不重叠），无需核间同步。
- **核内分块（按 UB 容量）**：核内对每列按行方向分 tile。设每 tile 处理 tileRow 行元素，需加载 x、y 的对应行段（`2 × tileRow × 8B`）与 A 的对应列段（`tileRow × 8B`）。通过 `GetCoreMemSize` 获取 UB 大小，结合 `BLOCK_SIZE` 与 `BUFFER_NUM`（double buffer 取 2）计算 tileRow，使单次搬入 `x 段 + y 段 + A 段 + 输出缓冲`不超出 UB，并尽量占满 UB 提高搬运效率。
- **uplo 三角处理**：UPPER 时列 j 的有效行区间为 `[0, j]`，LOWER 时为 `[j, n−1]`。核内对每列只在该区间内按 tile 搬入/搬出 A（区间外不发起任何读写，满足"未引用三角不读不写"与 lda padding 行写保护）；对角元素所在 tile 单独处理或计算后强制将虚部置 0。
- **tilingKey 规划**：本算子数据类型单一（complex64）、无广播、无分支感知需求，规划单一 tilingKey（0），host 侧通过 TilingData 传递 n、lda、uplo、列区间等参数，kernel 侧不设多分支。若后续验证中 uplo 两个分支的向量化路径差异显著，可扩展 tilingKey 区分 UPPER/LOWER（预留，不默认启用）。

**4. TilingData 结构**

```
struct Cher2TilingData {
    uint32_t n;            // 矩阵阶数
    uint32_t lda;          // 前导维度
    uint32_t uplo;         // 0: UPPER, 1: LOWER
    uint32_t coreColBegin; // 本核起始列
    uint32_t coreColNum;   // 本核列数
    uint32_t tileRow;      // 核内行方向 tile 大小（元素数）
    uint32_t bufferNum;    // double buffer 深度
};
alpha 以 host 侧解引用后的 4 个 float（ar/ai）打包传入 kernel（常量参数，不经 Device 内存）。
```

#### 3.2.2 kernel 侧设计

kernel 采用 Init / Process 两阶段，Process 内为 CopyIn → Compute → CopyOut 三段流水（Que/EnQue/DeQue 同步，double buffer 覆盖搬运与计算）：

1. **Init**：初始化 TPipe，按 TilingData 申请输入缓冲（x 段、y 段）、A 列段输入/输出缓冲与中间结果缓冲；解析 alpha 实部/虚部。
2. **CopyIn**：对列 j 的当前行 tile `[r0, r1)`：
   - 按步长 gather 或分段拷贝方式搬入 `x(r0..r1)`、`y(r0..r1)`（连续步长 inc=1 时用 DataCopy 整段拷贝；非连续步长时按元素步长拷贝，950PR 上复数元素为 8B 对齐粒度）；
   - 搬入 `A(r0..r1, j)` 列段（列主序下 A 列内连续，DataCopy 整段）。
3. **Compute**：
   - 核内先对本核 x、y 向量段各做一次复数乘法：`wx = alpha·x`、`wy = conjg(alpha)·y`（用 Mul + 复数展开或 Conj 后 Mul 实现，结果驻留 UB 复用）；
   - 逐 tile 计算 `A' = A + wx·conj(y) + wy·conj(x)`：conj 通过交换虚部符号（或专用 Conj 接口）实现，乘加用 Mul/Mul/Add（或 FMA 类接口）按实部/虚部分量展开；
   - 对角元素（i == j）：`A(j,j) += 2·Re(wx(j)·conj(y(j)))`，计算后将虚部强制写 0（`A(j,j).im = 0`），保证 Hermitian 语义。
4. **CopyOut**：仅将更新后的 A 列段 tile 写回原地址；区间外与 padding 行不发起写回。

**流水与同步**：同一列内行 tile 间通过 double buffer（`BUFFER_NUM=2`）重叠 CopyIn 与 Compute/CopyOut；EnQue/DeQue 保证 Compute 消费的数据已就绪；不同列由不同核并行处理，无核间同步。kernel 结束由 stream 语义保证后续同步（读回结果前由调用方同步 stream）。

**性能设计要点**：

- A 列段在列主序下连续访存，尽量使用大粒度 DataCopy 提升 MTE2/MTE3 带宽利用率；
- `wx`/`wy` 每向量元素只算一次并驻留 UB，列内 tile 复用，浮点计算量为每元素 8 次实数 FMA，非带宽瓶颈；
- 按列满核均分，n 较大（≥2048）时各核负载均匀；n 较小时列数少，核内行方向 tile 增大以保证每核有足够工作量；
- 单核 UB 中 x/y 段按 tileRow 加载，tileRow ≥ 128（元素数）时 MTE 拷贝效率接近峰值。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR | √ |

## 算子约束限制

- 仅支持 complex64（`aclblasComplex`）数据类型；
- 仅支持列主序；不支持超出 incx/incy/lda 步长语义的非连续内存访问；
- 不涉及广播；A 原地更新，不返回视图；
- 仅 uplo 指定三角被引用/更新，未引用三角不读不写；对角虚部更新后强制为 0；
- incx/incy 须非 0；负步长按 Netlib 语义反向遍历；
- 不要求确定性计算；`n = 0` 或 `alpha = (0,0)` 时为合法 no-op（结构参数校验先行）。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 `cblas_cher2`（Netlib CBLAS，CblasColMajor）单标杆生成；complex64 实部/虚部分别按 FLOAT32 判定：atol = rtol = 2⁻¹³（1.22e-4），matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32×ULP；Inf/NaN 按 §3.2 等效规则单独判定；同时以哨兵值比对确认未引用三角与 lda padding 行未被读写、对角虚部 \|imag\| ≤ atol | 任务书 §3.2 / opbase 生态算子开源精度标准 |
| 性能标准 | Ascend 950PR 上 msprof op 模式采集单算子 task duration，warmup 5 次后有效采样 ≥10 次取平均；NPU 平均单次耗时 ≤ GPU H100 标杆（`gpu_baseline.csv`，CUDA 12.4 cuBLAS `cublasCher2`）耗时 × 2.5（即 ≥ 0.4× GPU），单 case 达标 | 任务书 §3.3 |

测试方法：ops-blas 仓 test 框架加载 `cher2_test.csv`（1200 条：1000 精度 + 200 性能），C++ GTest 驱动，覆盖 uplo 全组合、尺寸扫描（0→2048，含 0/1/质数/2 的幂±1/非对齐）、alpha 特殊值（(0,0) quick return 等 8 组）、步长 ±1/±2/±3 全组合、lda padding（n+1/n+8/n+16）、填充模式（均匀/正态/全零/交替/极端值/Inf/NaN）、边界负向（空指针、非法枚举、非法 lda、负维度、零步长、INT_MIN）与性能用例（4 条任务书典型 case：n=512 UPPER / n=1024 LOWER / n=2048 UPPER / n=4096 LOWER）。

## 兼容性分析

- 本算子为 ops-blas 仓**新增接口**：`include/cann_ops_blas.h` 新增 `aclblasCher2` 声明，签名与其他产品线共用，不引入 950PR 私有平行 API，对既有接口无任何改动，不涉及兼容性风险。
- 状态码复用 `include/cann_ops_blas_common.h` 既有 `aclblasStatus_t` 定义，不新增错误码。
- 实现目录 `blas/syr2/arch35/` 与同族 `aclblasSsyr2` 共存同目录，遵循仓内目录组织规范，不影响其他算子编译。
