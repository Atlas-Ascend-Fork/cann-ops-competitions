# 需求背景（required）

## 需求来源

- 来源：CANN 社区任务 2026「10月社区任务-aclblasCher算子开发（950）」
- 任务目标：在 Ascend 950PR（arch35）上使用 Ascend C 实现单精度复数 Hermitian 秩-1 更新算子 `aclblasCher`，验收通过后合入昇腾算子开源仓 https://gitcode.com/cann/ops-blas 的 `blas/syr/arch35/`
- 对标基线：cuBLAS `cublasCher`；语义参考 Netlib `cher.f`
- 上游规格：`aclblasCher_Atlas950PR_task_doc.md`（随任务提供）、`test_cases/README.md`

## 背景介绍

### 算子在 BLAS 体系中的位置

`cher` 属 BLAS Level-2 的 Hermitian rank-1 update，与实数同族 `ssyr`、复数不共轭同族 `csyr` 并列：

```
A := alpha * x * x^H + A            (alpha 为实数, x 为复数向量, A 为 Hermitian 方阵)
```

与 `csyr` 的两个本质差异决定了实现侧的特殊处理：

1. **共轭**：`csyr` 计算 `x * x^T`，`cher` 计算 `x * x^H`，乘加中虚部符号相反；对角元素理论上退化为实数 `alpha * |x(i)|^2`。
2. **标量类型**：`alpha` 为 **FLOAT32 实数**（不是 complex），Host 指针；因此 Host 侧无须 `alpha` 的 D2H 读回分支（`csyr` 有 `alphaIsDevice` 异步路径）。

### 仓内现状

| 能力 | 仓内现状 |
| --- | --- |
| L2 Hermitian rank-1（`cher`） | **缺失**，本任务补齐 |
| L2 对称 rank-1（`ssyr` / `csyr`） | 已有，`blas/syr/arch35/`，可复用其 SIMT 行并行框架 |
| L3 Hermitian rank-k（`cherk` / `cher2k` / `cherkx`） | 已有，与本算子接口、算法均不同，**不可混淆** |
| 头文件声明 | `include/cann_ops_blas.h` 中**无** `aclblasCher`，需新增（与 `aclblasCherk` 是两个接口） |

# 需求分析（required）

## 需求描述

在 `ops-blas` 仓工程框架内，以 Ascend C kernel 直调方式实现句柄式 BLAS 接口 `aclblasCher`：通过 `aclblasHandle_t` 绑定的 stream 下发 kernel，在 950PR 上原地更新 Hermitian 矩阵 A 的指定三角。

接口签名（与 cuBLAS `cublasCher` 逐参数对应，与仓内 `aclblasSsyr` 同构）：

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha,
    const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

## 需求拆解

1. **接口声明**：`include/cann_ops_blas.h` 新增 `aclblasCher`（Doxygen 注释 + 声明），与其他产品线共用，禁止 950PR 私有平行 API。
2. **参数校验顺序**（严格对齐 `aclblasCsyr` 口径，顺序不可调换）：
   handle 判空 → `n < 0` → `n == 0` 直接 SUCCESS（不再校验 uplo/lda/incx/alpha/x/A）→ uplo 枚举 → `lda >= max(1,n)` → `incx != 0` → `alpha` 非空 → `alpha == 0.0f` 无操作返回 → `x`/`A` 非空。
3. **no-op 语义**：`n == 0` 或 `alpha == 0.0f` 为合法 quick return，返回 `ACLBLAS_STATUS_SUCCESS` 且不触碰 A。
4. **Hermitian 语义**：只引用/更新 `uplo` 指定三角；另一三角与 `lda` padding 不读不写；对角虚部更新后强制为 0。
5. **步长语义**：支持任意非零 `incx`，负步长按 Netlib 语义反向遍历 x。
6. **精度**：complex64 实部/虚部分别按 FLOAT32 档判定（atol = rtol = 2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 或 32 ULP）。
7. **性能**：平均单次耗时 ≤ 2.5 × GPU（H100）标杆，即 ≥ 0.4 × 标杆；msprof 采集，warmup 5 次后 10 个有效样本取均值。
8. **测试工程**：CSV 驱动 GTest，落到 `test/syr/cher/arch35/`，golden 用 `cblas_cher`（`CblasColMajor`），含哨兵值保护校验与负向用例。
9. **文档**：`blas/syr/README.md` 补充 `aclblasCher` 接口说明与产品支持表（Ascend 950PR：支持）。

# 详细设计（required）

## 算子分析

### 数学公式

列主序下，仅计算 `uplo` 指定三角：

```
UPPER:  A(i,j) += alpha * x(i) * conj(x(j)),  0 <= i <= j < n
LOWER:  A(i,j) += alpha * x(i) * conj(x(j)),  0 <= j <= i < n
```

记 `x(i) = (xr_i, xi_i)`，`alpha` 为实数，复乘展开为：

```
real:  A(i,j).real += alpha * (xr_i * xr_j + xi_i * xi_j)
imag:  A(i,j).imag += alpha * (xi_i * xr_j - xr_i * xi_j)
```

对角 `i == j` 时 `imag` 项恒为 `alpha * (xi*xr - xr*xi) = 0`，故：

```
A(i,i).real += alpha * (xr_i^2 + xi_i^2)
A(i,i).imag  = 0        // 直接写 0，不读旧虚部，保证 Hermitian 语义
```

### 支持数据类型

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `alpha` | FLOAT32（Host 指针） | 实数标量，非复数 |
| `x` | COMPLEX64（`aclblasComplex`） | 实部/虚部各 float32，Device 只读 |
| `A` | COMPLEX64（`aclblasComplex`） | Device 原地读写 |

### 支持形状

- `A`：`lda × n`，列主序，`lda >= max(1, n)`，允许 `lda > n` 的 padding
- `x`：逻辑长度 `n`，物理长度 `1 + (n-1) × |incx|`
- `n`：`n >= 0`；`n = 0` 为合法 no-op

## 算子实现

### 实现方案

整体沿用 `blas/syr/arch35/` 的 **SIMT 直调**框架（`__simt_vf__` + `asc_vf_call` + `KERNEL_TYPE_AIV_ONLY`），按行并行、按三角裁剪，新增文件与 `csyr_*` 同族并列。

#### host 侧设计

1. **参数校验**：`ValidateCherParams` 按需求拆解第 2 条的顺序实现，错误码 `ACLBLAS_STATUS_INVALID_ENUM` / `ACLBLAS_STATUS_INVALID_VALUE` / `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. **alpha 处理**：`alpha` 恒为 Host 指针，直接解引用取值；**不实现** `csyr` 的 `alphaIsDevice` D2H 分支（任务书 §2.4 明确 Host 内存）。`alpha == 0.0f` 时直接返回 SUCCESS，不下发 kernel。
3. **tiling 计算**：
   - `useNumBlocks = min(n, GetAivCoreCount())`，核数向运行时查询，不写死
   - `numThreads`：`n <= 512` 取 512，否则取 1024（SIMT 线程上限 2048，阈值待实测调优）
   - `uplo` 以 `uint32_t` 传入，kernel 侧再由模板参数消解分支
   - `incx == INT_MIN` 额外拒绝为 `INVALID_VALUE`：`|incx|` 溢出导致物理长度无法表示（任务书 §2.4 仅约束 `incx != 0`，此为测试工程约定的实现侧增强约束）
4. **kernel 下发**：`cher_kernel<<<useNumBlocks, nullptr, stream>>>(x, A, tiling)`，与 handle 绑定的 stream 一致，Host 侧不做同步（异步语义由调用方 `aclrtSynchronizeStream` 保证）。

`CherTilingData` 定义：

```cpp
struct CherTilingData {
    uint32_t numThreads;    // SIMT 线程数/block
    uint32_t n;             // 矩阵阶数
    uint32_t lda;           // 前导维
    uint32_t uplo;          // ACLBLAS_UPPER / ACLBLAS_LOWER
    float    alpha;         // 实数标量（csyr 是复数对，此处只有实部）
    int64_t  incx;          // 支持负步长，用 64 位避免乘法溢出
};
```

#### kernel 侧设计

1. **编译期三角特化**：`template <bool UPLO_IS_UPPER>` 实例化两份 kernel 体，`GetColRange` 计算列区间（UPPER：`[row, n)`；LOWER：`[0, row]`），避免运行期分支进入内层循环。
2. **行并行映射**：`row = blockIdx.x * blockDim.x + threadIdx.x`，按 `gridDim.x * blockDim.x` 跨步推进。列主序下 A 地址为 `col * lda + row`，warp 内 `threadIdx` 相邻 ⇒ `row` 相邻 ⇒ 同一列的地址连续，天然合并访存。
3. **两条执行路径**（与 `ssyr` 同构，按需选择）：
   - **UB 缓存路径**（`incx == 1 && 32 <= n <= 2048`）：把整段 x（≤ 16 KiB）搬到 `__ubuf__`，`asc_syncthreads()` 后复用；每个 block 只额外读 n 个复数，内层列循环全部命中 UB。
   - **GM 直算路径**（`incx != 1 || n < 32 || n > 2048`）：直接在 GM 上按步长取 `x`，保持通用性，保证任意 `|incx|` 与超大 n 正确。
4. **负步长索引**：统一用 `k → (incx > 0 ? k * incx : (n-1-k) * (-incx))` 映射逻辑下标到物理下标，覆盖 `incx = -1/-2/-3` 等场景。
5. **对角特化**：`row == col` 时走独立分支——实部用 `alpha*(xr²+xi²)`，虚部**直接写 0**。既省一次读取，也满足“对角虚部更新后为 0”的硬性语义。
6. **写保护**：循环边界严格限制在 `uplo` 三角内（含对角），另一三角与 `lda` padding 行/列不产生任何 load/store，测试工程用哨兵值验证。
7. **A 的读改写**：单元素 load `(real, imag)` → 计算 → 单元素 store，不经 UB 中转，避免 `n=4096`（128 MiB）场景下的搬运放大。

#### tilingkey 规划

不使用运行期 tilingkey：`uplo` 通过模板参数在编译期特化，`incx` 路径由 kernel 入口的两次分支决定，无额外 ABI 分支成本。

### 性能优化方案

| 优化点 | 作用 |
| --- | --- |
| 三角裁剪 | 工作量从 `n²` 降为 `n(n+1)/2`，约 50%，同时减少一半无效访存 |
| 行并行 + warp 内地址连续 | A 的访问在 warp 内合并，避免非合并 8B 事务 |
| UB 缓存 x | 内层列循环不再反复访问 GM，x 的 GM 流量从 `O(n²)` 降到 `O(n)` |
| 小尺寸降线程（512） | 减少小 `n` 场景的调度/启动开销 |
| 对角特化 | 省掉对角元素的虚部读取与冗余乘加 |
| 后续可选项 | `float2` 一次 8B 读写替代两次 4B、按列分块做双精度累加 | 

### 支持硬件

| 芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

`arch22`（A2/A3）不新增本算子文件；如需支持，复用同目录 `syr/csyr` 的 arch22 结构另行评估。

### 算子约束限制

1. `alpha` 仅支持 Host 指针（pointer mode = HOST），不支持 Device 侧标量。
2. 仅支持列主序、`lda >= max(1,n)`，不支持额外 stride/非连续视图。
3. `A` 必须为合法 Hermitian 输入（对角虚部为 0），实现侧不校验、直接按语义覆盖对角虚部。
4. `n` 上限受设备内存约束（`lda × n × 8B`），性能用例上限 4096。
5. 只保证 `uplo` 指定三角的读写，另一三角内容由调用方按 Hermitian 对称性推导。

### 改动点清单

| 文件 | 变更 |
| --- | --- |
| `include/cann_ops_blas.h` | **新增** `aclblasCher` 声明与 Doxygen 注释 |
| `blas/syr/arch35/cher_host.cpp` | 新增 Host 实现：校验、alpha 解析、tiling、下发 |
| `blas/syr/arch35/cher_kernel.cpp` | 新增 kernel 实现：SIMT 行并行 + 双路径 |
| `blas/syr/arch35/cher_kernel.h` | 新增 launcher 声明 |
| `blas/syr/arch35/cher_tiling_data.h` | 新增 tiling 结构体 |
| `blas/syr/README.md` | 新增接口章节与产品支持表 |
| `test/syr/cher/CMakeLists.txt` | 新增测试目标 `cher_test` |
| `test/syr/cher/{cher_param.h, cher_golden.h, cher_fill_ext.h, cher_verify_ext.h}` | 新增用例解析、golden、填充与校验 |
| `test/syr/cher/arch35/{cher_test.cpp, cher_npu_wrapper.h, cher_test.csv}` | 新增 arch35 测试与用例 |

> 源文件由 `blas/CMakeLists.txt` 的 `GLOB_RECURSE */arch35/*.cpp` 自动收集，`--ops=cher` 时按 `^cher_` 前缀保留，**无需修改 CMake 算子注册表**。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | complex64 实部/虚部分别按 FLOAT32 档：`atol = rtol = 2⁻¹³ ≈ 1.2207e-4`，`matched_ratio ≥ 0.99`，`max_abs_error ≤ 1e-2` 或 32 ULP；对角虚部额外断言 `|imag| ≤ atol`；未引用三角与 padding 用哨兵值校验不被改写 | 生态算子开源精度标准 `mixed_tolerance_standard.md` |
| 性能标准 | 平均单次耗时 ≤ 2.5 × GPU(H100) 标杆，即 ≥ 0.4 × 标杆；msprof 采 task duration，warmup 5 次 + 10 个有效样本取均值 | 任务书 §3.3 + `test_cases/gpu_baseline.csv` |

性能基线（`gpu_baseline.csv`，alpha=1.0、incx=1、lda=n）：

| case | uplo | n | GPU 标杆 | 达标上限（2.5×） |
| --- | --- | --- | --- | --- |
| cher-base-000 | UPPER | 512 | 5.574 us | 13.94 us |
| cher-base-001 | LOWER | 1024 | 5.937 us | 14.84 us |
| cher-base-002 | UPPER | 2048 | 11.113 us | 27.78 us |
| cher-base-003 | LOWER | 4096 | 54.050 us | 135.13 us |

## 兼容性分析

- **纯新增接口**：仅在 `include/cann_ops_blas.h` 追加声明，不改动任何既有函数签名/枚举/结构体，无 ABI 破坏。
- **与 `aclblasCherk` 区分**：后者为 L3 Hermitian rank-k，参数列表不同（含 `k`、`trans`、`beta`），二者并存不冲突。
- **产品线共用**：接口为全产品线统一 API，950PR 仅作为本批次首个实现，其他产品线可复用同一声明。
- **测试工程隔离**：新增 `test/syr/cher/` 目标，不影响既有 `ssyr_test` / `csyr_test`。

## 可测试性

1. **CSV 驱动**：`cher_test.csv` 1200 条（1000 精度 + 200 性能），列 `case_name,description,uplo,n,alpha,x_fill,incx,a_fill,lda,expect_result,mere_threshold,mare_multiplier,random_seed`。
2. **golden**：`cblas_cher`（`CblasColMajor`），与 cuBLAS/Netlib 同源语义，避免自证。
3. **覆盖面**：L0 基础、L1 尺寸扫描（1→2048，含质数/2 的幂±1/非对齐）、L2 alpha 特殊值（0/±1/±0.5/±2/3/100/0.001）、L4 前导维 padding、L4 步长（±1/±2/±3）、L5 填充（全零/交替/极端/Inf/NaN）、L6 边界负向（n=0 与 alpha=0 quick return、空指针、非法枚举 999、非法/零/负 lda、负 n、incx=0 与 INT_MIN）、EX 扩展、PF 性能。
4. **保护性校验**：调用前在未引用三角与 padding 写哨兵，调用后比对未被改写。
5. **验收脚本**：`verify_accuracy.py` / `verify_performance.py`（`--soc ascend950`），以及 `msprof op` 采集 kernel 级耗时。

## 风险与待实测项

1. **大规模带宽受限**：`n = 4096` 需读写约 134 MiB（半矩阵），达标上限 135 us 对应约 1 TB/s 有效带宽量级，需在 950PR 实测确认；不足时优先做 `float2` 向量化与行块/列块二维分块。
2. **UB 容量与路径切换阈值**：`incx == 1` 时的 UB 可缓存复数个数需按实际 UB 大小标定（初值 2048 complex = 16 KiB），超限回退 GM 路径。
3. **`numThreads` 阈值（512/1024）**：沿用 `csyr` 经验值，需按 950PR 实测微调。
4. **负步长 + padding 组合**：用例已覆盖，若发现 golden 与实现口径不一致，以 Netlib `cher.f` 为准并记录说明。
