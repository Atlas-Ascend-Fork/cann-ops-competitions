# aclblasCsyrk3mEx 算子设计文档（arch35，Ascend 950PR）

> 11月社区任务-aclblasCsyrk3mEx算子开发（950）设计文档。本 PR 仅含本文件一个新增 md；
> 算子实现与测试代码后续按任务书合入 [ops-blas](https://gitcode.com/cann/ops-blas)
> （`blas/syrk/arch35/`、`test/syrk/csyrk3m_ex/arch35/`）。性能实测数据以上机 950PR `msprof op` 采集为准【待填】，不预填。

# 需求背景（required）

## 需求来源

- 任务：[11月社区任务-aclblasCsyrk3mEx算子开发（950）](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6)，任务 ID `3d67a5f992894ea8b43e472342c122e6`
- 任务讨论帖：https://gitcode.com/cann/ops-blas/discussions/88
- 任务书：`aclblasCsyrk3mEx_Atlas950PR_task_doc.md`（接口契约、边界语义、精度与性能验收口径均以任务书为准）
- 对标接口：cuBLAS `cublasCsyrk3mEx`（数学语义参考 Netlib `csyrk`，https://www.netlib.org/blas/csyrk.f ）
- 目标仓：https://gitcode.com/cann/ops-blas ，实现目录 `blas/syrk/arch35/`，测试目录 `test/syrk/csyrk3m_ex/arch35/`

## 背景介绍

单精度复数对称矩阵秩-k 更新（CSYRK）是线性代数求解、电磁场有限元分析、量子物理与信号处理中的核心 Level-3 BLAS 计算单元，执行 `C = alpha * op(A) * op(A)^T + beta * C`，仅读写 C 的 `uplo` 指定三角（另一三角由对称性 `C = C^T` 隐含）。相较于传统 4M 复数乘法（4 次实数矩阵乘），3M 算法通过高斯复数乘法分解将 `X · X^T` 归约为 3 次实数对称外积矩阵乘，理论节省约 25% 的矩阵乘计算量与中间累加器带宽；扩展接口（`Ex`）进一步支持混合精度输入（`Atype` 支持 `ACL_COMPLEX64` 与交错 2 字节 complex-int8 `ACL_INT8`，`Ctype` 为 `ACL_COMPLEX64`）。cuBLAS 通过 `cublasCsyrk3mEx` 提供该能力，`ops-blas` 仓 Ascend 950PR（`arch35`）当前空缺，本任务予以补齐。

**仓内现状（基于 `ops-blas` `master` 核实）**：

| 项 | 现状 | 对本任务的意义 |
| --- | --- | --- |
| 接口声明 | `include/cann_ops_blas.h` 已有 `aclblasCsyrk`、`aclblasSsyrk`、`aclblasCsyrkx`、`aclblasCgemm3m`、`aclblasCgemmEx`，尚无 `aclblasCsyrk3mEx`（1125 之前无 `syrk` `Ex` 系列接口） | 本任务在该公共头新增 `aclblasCsyrk3mEx` 声明，签名与 `cublasCsyrk3mEx` 逐参对齐，供各产品线共用，禁止私有平行 API；与同批新增的 `aclblasCsyrkEx` 彼此独立、互不依赖 |
| `blas/syrk/arch35/` 参考 | 已有非 Ex 的 `csyrk`（4M 融合 Cube + HF32x3 补偿 + SIMT 直算 + AIV Combine）与 `ssyrk` 实现 | 可参考参数校验、三角跳块、`te::` Cube 流水与对称语义（对角虚部保留、`OP_C` 不共轭）；但 `csyrk` 为 4M 路径且不含 `ACL_INT8` 解码，本任务需独立实现单次发射的对称 3M 融合路径 |
| `blas/gemm/arch35/` 参考 | 已有 `cgemm3m_host.cpp`、`cgemm3m_rounding_kernel.cpp`、`cgemm3m_strided_batched_kernel.cpp` | 可参考 3M 分解与舍入控制思路；需注意通用 GEMM 为 `A · B`，而对称秩-k 为 `X · X^T`，不得套用 Hermitian 的 `X · X^H` 公式 |
| 构建与测试现状 | `blas/CMakeLists.txt` 在 `--ops=csyrk3m_ex` 单算子构建时按 `^csyrk3m_ex_` 文件前缀过滤源文件；`test/syrk/` 仅有 `csyrk/` 与 `ssyrk/` | 实现文件统一采用 `csyrk3m_ex_*` 命名自动接入 `--ops=csyrk3m_ex` 构建；新建 `test/syrk/csyrk3m_ex/arch35/` 与 `test/syrk/csyrk3m_ex/CMakeLists.txt` 注册 GTest 目标；涉及 `te::`（tensor_api）的源文件同步纳入 `ASC_DEVKIT_GE_9_1` 版本守卫 |

# 需求分析（required）

## 需求描述

在 Ascend 950PR（`arch35`，CANN 9.1.0 及以上）上基于 `ops-blas` 句柄式工程框架，采用 Ascend C kernel 直调方式实现单精度复数对称矩阵秩-k 更新扩展算子 `aclblasCsyrk3mEx`：

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta,
    aclblasComplex* C, aclDataType Ctype, int ldc);
```

支持对称转置语义（`OP_C` 等价于 `OP_T`，不共轭）、`uplo` 三角原地更新（对角虚部作为有效数据保留）、`Atype ∈ {ACL_COMPLEX64, ACL_INT8}` 与 `Ctype = ACL_COMPLEX64` 两种输入组合、Host/Device 双位置复数标量指针；采用对称专用 3M 算法且三次实数矩阵乘在同一次 kernel launch 内完成；精度满足 FLOAT32 分量混合容差标准，kernel-to-kernel 性能达到 GPU `cublasCsyrk3mEx` 标杆单 case ≥ 0.4×、全部用例算术平均 ≥ 0.5×。

## 需求拆解

1. **参数校验（顺序与错误码严格对齐任务书 §2.4/§2.5）**：
   - `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - `uplo ∉ {ACLBLAS_UPPER, ACLBLAS_LOWER}` 或 `trans ∉ {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}` → `ACLBLAS_STATUS_INVALID_ENUM`（`ops-blas` 既定约定，与 cuBLAS 返回 `INVALID_VALUE` 属有意差异）；
   - `n < 0` 或 `k < 0`、`lda < max(1, trans == ACLBLAS_OP_N ? n : k)`、`ldc < max(1, n)` → `ACLBLAS_STATUS_INVALID_VALUE`；
   - `alpha == nullptr` 或 `beta == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
   - `(Atype, Ctype) ∉ {(ACL_COMPLEX64, ACL_COMPLEX64), (ACL_INT8, ACL_COMPLEX64)}` → `ACLBLAS_STATUS_NOT_SUPPORTED`；
   - `n > 0 && k > 0 && A == nullptr` 或 `n > 0 && C == nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
   - 所有结构参数校验全部通过后再执行 quick return；不对 Device 张量做 Host 侧元素级扫描。
2. **对称复数语义与三角写保护**：
   - `trans = ACLBLAS_OP_N` 时 A 物理维度 n×k，`op(A) = A`；`trans = ACLBLAS_OP_T` 或 `ACLBLAS_OP_C` 时 A 物理维度 k×n，`op(A) = A^T`（`OP_C` 不做共轭）；
   - 仅 `uplo` 指定三角（含对角线）被读写，另一三角及 `ldc` padding 区域严格不被改写；
   - 对角元素 `C(i,i)` 的虚部为有效数据，计算后原样保留，不得按 Hermitian 语义置 0。
3. **对称专用 3M 算法（单次 kernel launch 约束）**：
   - 令 `X = op(A) = Xr + i·Xi`（`ACL_INT8` 时先将交错 `(int8 real, int8 imag)` 精确提升为 float32），计算 `P0 = Xr·Xr^T`、`P1 = Xi·Xi^T`、`P2 = (Xr + Xi)·(Xr + Xi)^T`，复数积实部 `= P0 − P1`、虚部 `= P2 − P0 − P1`；
   - 三次实数矩阵乘必须在同一次 kernel launch 内完成，严禁拼接三次独立 GEMM 调用，严禁套用 Hermitian 的 `X·X^H` 分解。
4. **混合精度输入（`ACL_INT8` / `ACL_COMPLEX64`）**：
   - `Atype = ACL_COMPLEX64`：每复元素 8 字节 `(float real, float imag)`，列主序步距 `lda` 个复元素；
   - `Atype = ACL_INT8`：对应 cuBLAS `CUDA_C_8I`，每复元素 2 字节交错 `(int8_t real, int8_t imag)`，列主序步距 `lda` 个复元素（即 `2 * lda` 字节），精确提升为 float32 后参与 3M 计算，输出 C 为 `ACL_COMPLEX64`。
5. **边界与退化分支**：
   - `n = 0`：合法 no-op，不引用 `A`/`C`（允许为 `nullptr`），不启动 kernel，返回 `ACLBLAS_STATUS_SUCCESS`；
   - `k = 0` 或 `alpha = (0,0)`：秩-k 贡献为 0，不引用 `A`（允许 `A` 为 `nullptr` 或含 NaN）；若 `beta = (1,0)` 则 quick return（C 不变），若 `beta ≠ (1,0)` 则仅对 C 的 `uplo` 三角执行 `C = beta · C`；
   - `beta = (0,0)`：不读取旧 C（旧 C 中的 NaN/Inf 不得污染 `uplo` 结果），直接写入新值；
   - `alpha`、`beta` 支持 Host 与 Device 两种指针位置，均按 `aclblasComplex`（实部/虚部各 float32）解析。
6. **精度与性能达标**：
   - 精度：对标 Netlib CBLAS `cblas_csyrk`（float64 参考可用 `cblas_zsyrk`，`ACL_INT8` 精确提升后调用），实部/虚部分别按 FLOAT32 阈值判定（`atol = rtol = 2^-13`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ 1e-2` 或 `32×ULP`）；
   - 性能：Ascend 950PR 上以 `msprof op` 采集 kernel `Task Duration(us)`（5 次 warmup + 有效采样 10 次取均值），与 `gpu_baseline.csv` 的 GPU kernel-only 耗时 `gpu_ms` 按 kernel-to-kernel 口径比对，满足单 case ≤ `gpu_ms / 0.4` 且全部用例加速比算术平均 ≥ 0.5；同步自补采集 `ACL_INT8` 路径性能数据。
7. **工程接入**：
   - `include/cann_ops_blas.h` 新增公开声明；`blas/syrk/arch35/` 新增 `csyrk3m_ex_*` 实现；`test/syrk/csyrk3m_ex/arch35/` 与 `test/syrk/csyrk3m_ex/CMakeLists.txt` 新建 CSV 驱动 GTest 测试工程；`blas/syrk/README.md` 补充 `aclblasCsyrk3mEx` 文档章节；支持 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 单算子编译与测试。

# 详细设计（required）

## 算子分析

### 数学公式

设输入矩阵经转置选择后为 `X = op(A) ∈ C^(n×k)`，写作 `X = Xr + i·Xi`（其中 `Xr, Xi ∈ R^(n×k)`）：

- 当 `trans = ACLBLAS_OP_N` 时，`X(p, t) = A(p, t)`；
- 当 `trans = ACLBLAS_OP_T` 或 `ACLBLAS_OP_C` 时，`X(p, t) = A(t, p)`（对称语义不共轭）。

对称秩-k 更新定义为（仅对 `uplo` 指定三角中的 `(p, q)`，即 `UPPER` 时 `0 ≤ p ≤ q < n`、`LOWER` 时 `0 ≤ q ≤ p < n`）：

```text
C_new(p, q) = alpha * sum_{t=0..k-1} X(p, t) * X(q, t) + beta * C_old(p, q)
```

#### 对称专用 3M 分解

记 `Xs = Xr + Xi ∈ R^(n×k)`，定义三个实对称矩阵乘积：

```text
P0 = Xr * Xr^T
P1 = Xi * Xi^T
P2 = Xs * Xs^T = (Xr + Xi) * (Xr + Xi)^T

Re(X * X^T) = P0 - P1
Im(X * X^T) = P2 - P0 - P1 = Xr * Xi^T + Xi * Xr^T
```

> **与 Hermitian（`cherk3m`）的关键区别**：Hermitian 乘积 `X · X^H` 的实部为 `P0 + P1`、虚部为 `Xi·Xr^T − Xr·Xi^T` 且对角虚部恒为 0；本算子为对称 `X · X^T`，实部为 `P0 − P1`、虚部为 `P2 − P0 − P1`，且对角元素 `p = q` 处 `Im((X·X^T)_{p,p}) = 2 * sum_t Xr(p,t)·Xi(p,t)` 一般非零，必须完整计算并保留。

#### 复数标量更新

记 `R = P0 − P1`、`I = P2 − P0 − P1`，`alpha = alpha_r + i·alpha_i`，`beta = beta_r + i·beta_i`，`C_old = c_r + i·c_i`：

```text
prod_r = alpha_r * R - alpha_i * I
prod_i = alpha_r * I + alpha_i * R
if (beta != (0, 0)):
    out_r = prod_r + (beta_r * c_r - beta_i * c_i)
    out_i = prod_i + (beta_r * c_i + beta_i * c_r)
else:
    out_r = prod_r
    out_i = prod_i
```

当 `k == 0` 或 `alpha == (0, 0)` 时，`prod_r = prod_i = 0` 且不引用 A；当 `beta == (0, 0)` 时不读取 `C_old`。

#### 数值与硬件特性分析

1. **`ACL_INT8` 在 HF32 Cube 上的零截断特性**：当 `Atype = ACL_INT8` 时，`Xr, Xi ∈ [-128, 127]`，`Xs = Xr + Xi ∈ [-256, 254]`，三者均为绝对值不超过 256 的精确整数，有效尾数最多 9 bit，小于 Ascend 950PR Cube HF32 模式的 10 bit 尾数宽度。因此在 `ACL_INT8` 路径下，直接启用单次 HF32 `Mmad`（FP32 L0C 累加）即可做到**无输入尾数截断误差**，无需 HF32x3 残差拆分，单 tile 仅需 3 次 `Mmad`。
2. **`ACL_COMPLEX64` 的精度分层**：FP32 具有 23 bit 尾数，直接落入 10 bit HF32 会引入约 `2^-10` 的输入截断并在 `P2 − P0 − P1` 抵消时放大相对误差。对中小规模（`n ≤ 256 && k ≤ 512`）使用原生 FP32 Cube（禁用 HF32）或单发射 SIMT 保持完整 23 bit 精度；对大规模（`n > 256 || k > 512`）采用 HF32x3 残差补偿（将 `Xr, Xi, Xs` 各自拆分为高 10 bit `xh = HF32(x)` 与低位残差 `xl = x − xh`，每乘积执行 `xh·yh + xl·yh + xh·yl` 共 3 次 HF32 `Mmad`，三乘积合计 9 次 `Mmad`，较 4M 的 12 次 `Mmad` 节省 25% 算力），将有效尾数恢复至 ≥ 20 bit。
3. **3M 抵消与对角线精度**：对于大 k 下对角元素 `Re((X·X^T)_{i,i}) = sum_t (Xr(i,t)^2 − Xi(i,t)^2)` 存在的大数相减抵消（及 `TC_3M_*` 交替/极值填充），在 Combine 阶段支持按阈值开启对角线直算保护（仅 `O(n·k)` 代价，逐项计算 `Xr^2 − Xi^2` 与 `2·Xr·Xi` 后累加），避免大矩阵乘相减放大对角误差。

### 支持数据类型

| 参数 | 角色 | 支持数据类型 | 内存位置与排布 | 说明 |
| --- | --- | --- | --- | --- |
| `handle` | 输入 | `aclblasHandle_t` | Host | 携带执行 stream 的上下文句柄 |
| `uplo` / `trans` | 属性 | `aclblasFillMode_t` / `aclblasOperation_t` | Host | `{UPPER, LOWER}` / `{OP_N, OP_T, OP_C}` |
| `n` / `k` / `lda` / `ldc` | 属性 | `int`（int32） | Host | 矩阵阶数、秩与前导维 |
| `alpha` / `beta` | 输入 | `const aclblasComplex*`（`ACL_COMPLEX64`） | Host 或 Device | 实部/虚部各 FP32，共 8 字节 |
| `A` | 输入 | `ACL_COMPLEX64` 或 `ACL_INT8`（complex-int8） | Device，列主序 ND | `ACL_COMPLEX64` 为 8B `(float, float)`；`ACL_INT8` 为 2B 交错 `(int8, int8)` |
| `Atype` | 属性 | `aclDataType` | Host | 仅支持 `ACL_COMPLEX64`、`ACL_INT8` |
| `C` | 原地输入/输出 | `aclblasComplex*`（`ACL_COMPLEX64`） | Device，列主序 ND | 仅 `uplo` 指定三角读写，原地覆写 |
| `Ctype` | 属性 | `aclDataType` | Host | 仅支持 `ACL_COMPLEX64` |

支持的 `(Atype, Ctype)` 组合为 `(ACL_COMPLEX64, ACL_COMPLEX64)` 与 `(ACL_INT8, ACL_COMPLEX64)`，其余任意组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

### 支持形状

- A 逻辑形状为 `n×k`（`trans = ACLBLAS_OP_N`，`lda ≥ max(1, n)`）或 `k×n`（`trans = ACLBLAS_OP_T/C`，`lda ≥ max(1, k)`），物理步距由 `lda` 描述；
- C 逻辑形状为 `n×n`，物理存储 `ldc × n`（`ldc ≥ max(1, n)`），仅更新 `uplo` 三角；
- `n, k ≥ 0` 为运行时入参，Host 侧按实际维度完成路径分派与 tiling，无编译期固定 shape 假设；测试范围覆盖 `n, k ∈ [0, 4096]`（含 0、1、质数、2 的幂及 ±1、`k ≫ n` 宽矩阵、`n ≫ k` 窄矩阵及 `lda`/`ldc` `+1/+8/+64` padding）；
- 不支持 broadcast、不支持超出前导维语义的任意非连续视图、不接受 0 维 Tensor。

## 算子实现

### 实现方案

在 `ops-blas` 仓 `blas/syrk/arch35/` 下独立新增 `aclblasCsyrk3mEx` 实现文件集（前缀统一为 `csyrk3m_ex_`，与现有 `csyrk_*` 彻底解耦且自动匹配 `blas/CMakeLists.txt` 的 `--ops=csyrk3m_ex` 单算子构建规则）：

- `blas/syrk/arch35/csyrk3m_ex_host.cpp`：公开 C API 入口、参数校验、标量指针解析、路径分派、tiling 与 workspace 管理；
- `blas/syrk/arch35/csyrk3m_ex_tiling_data.h`：Host/Kernel 共享 tiling 结构体与片上容量常量；
- `blas/syrk/arch35/csyrk3m_ex_kernel.h`：各 kernel 的发射声明；
- `blas/syrk/arch35/csyrk3m_ex_direct_kernel.cpp`：小规模单次发射 SIMT 3M 直算 kernel（兼顾 `ACL_COMPLEX64` 与 `ACL_INT8`）；
- `blas/syrk/arch35/csyrk3m_ex_kernel.cpp`：AIV 预处理（解交错 / int8 精确提升 / `Xs = Xr + Xi` / HF32 残差融合生成）与 AIV Combine/Beta-only 写回 kernel；
- `blas/syrk/arch35/csyrk3m_ex_cube_kernel.cpp` & `csyrk3m_ex_gemm_fused.h`：AIC 单次发射 3M 融合 Cube GEMM kernel（`te::` tensor_api，单 launch 内完成 `P0, P1, P2` 三次实数矩阵乘，支持三角跳块、INT8 单次 HF32、COMPLEX64 原生 FP32 与 HF32x3 补偿三种子模式）。

#### Host 侧设计

入口 `aclblasCsyrk3mEx` 位于 `blas/syrk/arch35/csyrk3m_ex_host.cpp`，公开声明新增入 `include/cann_ops_blas.h`：

1. **参数校验（固定顺序，任务书 §2.5）**：
   - `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
   - `uplo`、`trans` 枚举合法性 → `ACLBLAS_STATUS_INVALID_ENUM`；
   - `n >= 0`、`k >= 0`、`lda >= max(1, isTransN ? n : k)`、`ldc >= max(1, n)` → `ACLBLAS_STATUS_INVALID_VALUE`；
   - `alpha != nullptr`、`beta != nullptr` → `ACLBLAS_STATUS_INVALID_VALUE`；
   - `(Atype, Ctype)` 组合检查 → `ACLBLAS_STATUS_NOT_SUPPORTED`；
   - `!(n > 0 && k > 0 && A == nullptr)`、`!(n > 0 && C == nullptr)` → `ACLBLAS_STATUS_INVALID_VALUE`；
   - 校验全部通过后，若 `n == 0` 直接返回 `ACLBLAS_STATUS_SUCCESS`（不引用 `A`/`C`、不启动 kernel）。
2. **标量位置解析与短路逻辑**：
   - 使用仓内公共 `CheckPtrLocation`（底层 `aclrtPointerGetAttributes`）分别探测 `alpha`、`beta` 的 Host/Device 位置；Host 指针直接解引用零同步开销，Device 指针支持在 `direct` / `beta-only` 路径由 kernel 直接从 GM 读取或按流异步解析（复用 `ops-blas` 既定处理方式）；
   - 将 `ACLBLAS_OP_C` 统一归一为 `ACLBLAS_OP_T`（不共轭）；
   - 当 `k == 0` 或 `alpha == (0, 0)` 时标记 `skipRankK = true`：若 `beta == (1, 0)` 直接 quick return（`ACLBLAS_STATUS_SUCCESS`，不修改 C、不引用 A）；否则直接分派至 **Path A（Beta-only Scale）**，零 workspace 且不解引用 A。
3. **路径分派与 Tiling 策略（仅依赖运行时形状、dtype 与片上容量）**：

| 执行路径 | 触发条件 | 启动 Kernel 数 | Workspace 需求 | 设计目标 |
| --- | --- | --- | --- | --- |
| **Path A：Beta-only Scale** | `k == 0` 或 `alpha == (0,0)`，且 `beta != (1,0)` | 1（AIV） | 0 | 跳过矩阵乘，不引用 `A`，仅对 `uplo` 三角执行 `C = beta·C`（`beta=(0,0)` 时直接写零不读旧 `C`） |
| **Path B：小形状单发射 SIMT 3M** | `n <= 64`（且非 Path A） | 1（AIV SIMT） | 0 | 消除三阶段发射开销，对标 GPU 2.5~5.1 us 小形状基线；单 kernel 内以 FP32 精确完成 3M 内积（支持 `COMPLEX64`/`INT8` 与任意 `lda`/`ldc` padding 及特殊值） |
| **Path C：中大形状融合 3M 三阶段** | `n > 64`（且非 Path A） | 3（AIV Pack → AIC 3M Cube → AIV Combine） | `Ar/Ai/As`（+ 可选 `ArL/AiL/AsL`）+ `temp[3 × n × tempLdc]` | 满载 Ascend 950PR AIC Cube 算力；单次 Cube kernel launch 内融合完成 `P0/P1/P2` 三次实数矩阵乘与三角跳块 |

4. **Workspace 布局与内存预算**：
   - 令 `arRows = isTransN ? n : k`，`arCols = isTransN ? k : n`，`arLdc = CeilAlign(arRows, 8)`（32B 对齐），`tempLdc = CeilAlign(n, 8)`；
   - `Ar`、`Ai`、`As` 各占 `arLdc × arCols × 4B`；当启用 `useHf32x3`（仅 `Atype == ACL_COMPLEX64 && (n > 256 || k > 512)`）时追加 `ArLow`、`AiLow`、`AsLow` 三个残差平面；`temp` 占 `3 × n × tempLdc × 4B`（较 4M 减少 25%）；
   - 统一通过 `_aclblas_handle` 内置 workspace 缓冲按 512B 对齐一次性切分，避免逐次 `aclrtMalloc`/`aclrtFree` 开销，全部在 `h->stream` 上异步下发。

#### Kernel 侧设计

##### 1. Path B：小形状单发射 SIMT 3M 直算 Kernel（`csyrk3m_ex_direct_kernel.cpp`）

- 标记 `KERNEL_TYPE_AIV_ONLY`，经 `asc_vf_call` 启动 256 线程/block，按 `uplo` 三角有效元素总数 `n*(n+1)/2` 均匀映射到 AIV 核与线程（每个线程负责若干上/下三角元素 `(p, q)`，不访问非 `uplo` 元素）；
- 每个元素 `(p, q)` 沿 `t = 0..k-1` 单遍读取 `X(p, t)` 与 `X(q, t)`（`Atype == ACL_INT8` 时读取 2B `(int8_t, int8_t)` 精确转为 `float`；`Atype == ACL_COMPLEX64` 时读取 8B `(float, float)`），按对称 3M 公式同步累加三个 FP32 累加器（长 k 辅以 Kahan 补偿）：
  ```text
  p0 += xr_p * xr_q
  p1 += xi_p * xi_q
  p2 += (xr_p + xi_p) * (xr_q + xi_q)
  ```
- 规约完成后计算 `R = p0 - p1`、`I = (p2 - p0) - p1`，乘复数标量 `alpha`，当 `beta != (0,0)` 时读取 `C(p, q)` 累加 `beta * C_old`（`beta == (0,0)` 不读旧 `C`），原地写回 `C[q * ldc + p]`，对角虚部原样保留。

##### 2. Path C - Phase 0：AIV 融合预处理 Kernel（`csyrk3m_ex_kernel.cpp`）

- 标记 `KERNEL_TYPE_AIV_ONLY`，按物理列 `arCols` 均分到全部 AIV 核，UB 采用 ping-pong 双缓冲掩盖 GM 搬运；
- **单遍融合输出**：
  - `Atype == ACL_COMPLEX64`：以 `DataCopyPad` 搬入列切片（兼容任意 `lda` padding），经解交错提取 `Ar`、`Ai`，向量加法计算 `As = Ar + Ai`；当 `useHf32x3 == 1` 时，在 UB 内同步对 `Ar`、`Ai`、`As` 计算 HF32 舍入残差（对齐硬件 10-bit 尾数舍入得 `xh`，计算 `xl = x - xh`）并一并写回 `ArLow`/`AiLow`/`AsLow`，相较拆分为两个独立预处理 kernel 减少一次 launch 与一次完整 GM 读写；
  - `Atype == ACL_INT8`：每复元素仅 2 字节，搬入 UB 后拆出交错的 `int8` 实部/虚部并精确 `Cast` 为 `float32`（`[-128, 127]` 全程零误差），计算 `As = Ar + Ai`（`[-256, 254]`），直接写出 3 个 32B 对齐平面 `Ar`/`Ai`/`As`，不产生残差平面（`useHf32x3 = 0`）。

##### 3. Path C - Phase 1：AIC 单次发射 3M 融合 Cube GEMM Kernel（`csyrk3m_ex_cube_kernel.cpp` & `csyrk3m_ex_gemm_fused.h`）

- 标记 `KERNEL_TYPE_AIC_ONLY`，采用 Ascend C `te::`（tensor_api）编程范式，在**同一次 kernel launch** 内完成 `P0 = Xr·Xr^T`、`P1 = Xi·Xi^T`、`P2 = Xs·Xs^T` 三次实数矩阵乘，写入 `temp` 的三个连续象限 `[P0 | P1 | P2]`；
- **三角跳块与负载均衡**：输出矩阵按 `BM = BN = 64` 切分，按蛇形（serpentine）行偶正向、行奇反向遍历 tile 以增强 L2 复用；`triangleMode == 1`（`UPPER`）跳过 `mStart > nStart + 63` 的严格下三角块，`triangleMode == 2`（`LOWER`）跳过 `nStart > mStart + 63` 的严格上三角块，直接削减约 50% Cube 计算量；对保留的有效块采用连续计数取模 `(kept % blockNum) == coreIdx` 静态均分至全部 AIC 核，避免按行切分导致的三角负载倾斜；
- **统一 L0C 双组 Ping-Pong**：单块 `64×64` FP32 NZ 累加器占 `16 KB`，三个乘积 `P0/P1/P2` 合计仅占 `48 KB`（低于 4M 的 `64 KB`）。L0C 开辟两组累加器（偏移 `0` 与 `48 KB`，共 `96 KB`），tile t 的三次 `CopyL0C2GM` Fixpipe 写回与 tile t+1 的 `Mmad` 严格并行重叠；
- **GM Layout 零转置搬运**：`trans = N` 时左矩阵采用 `DNExtLayoutPtn(arLdc, k)`、右矩阵采用 `NDExtLayoutPtn(k, arLdc)`；`trans = T/C` 时左矩阵 `NDExtLayoutPtn(n, arLdc)`、右矩阵 `DNExtLayoutPtn(arLdc, n)`，由 MTE2/MTE1 硬件在搬运时完成转置，无需显式转置 kernel；
- **三种 Cube 精度子模式（依据 `Atype` 与规模分派）**：

| Cube 子模式 | 适用条件 | `SetHF32Mode` | 单 tile K 块参数与片上 SRAM 规划 | 每 K 块 `Mmad` 次数 |
| --- | --- | --- | --- | --- |
| **INT8-HF32 精确快路径** | `Atype == ACL_INT8` | `ENABLE` | `BM=BN=64, CHUNK_K=128, BASE_K=64`；L1 2-buf 双缓冲（每 buf 6 个 `64×128` 面板 = 192 KB，共 384 KB ≤ 512 KB）；L0A/L0B 按乘积双槽 ping-pong（每槽 16 KB） | 3（`P0/P1/P2` 各 1 次，整数 `[-256, 254]` 无截断误差） |
| **FP32 原生路径** | `Atype == ACL_COMPLEX64` 且 `n <= 256 && k <= 512` | `DISABLE` | `BM=BN=64, CHUNK_K=128, BASE_K=64`；L1 2-buf 双缓冲（384 KB）；L0A/L0B 双槽 ping-pong（每槽 `64×64` = 16 KB），逐乘积轮转发射 | 3（完整 FP32 23-bit 尾数） |
| **HF32x3 补偿路径** | `Atype == ACL_COMPLEX64` 且 `(n > 256 || k > 512)` | `ENABLE` | `BM=BN=64, CHUNK_K=64`；L1 2-buf 双缓冲（每 buf 6 个 M 面板 + 6 个 N 面板，每面板 `64×64` = 16 KB，共 192 KB/buf，合计 384 KB ≤ 512 KB）；L0A/L0B 各设 2 个 32 KB ping-pong 槽（每槽放单个乘积的 `(H, L)` 一对面板），`P0 → P1 → P2` 交替预取与计算 | 9（`P0/P1/P2` 各 3 次：`xh·yh + xl·yh + xh·yl`，较 4M 节省 25%） |

- 每个 `csyrk3m_ex_cube_kernel` 入口显式调用 `AscendC::SetHF32Mode(...)`，避免受同一流上先前算子残留的全局 HF32 状态影响。

##### 4. Path C - Phase 2 & Path A：AIV Combine / Beta-only Kernel（`csyrk3m_ex_kernel.cpp`）

- 标记 `KERNEL_TYPE_AIV_ONLY`，严格只读写 `uplo` 三角有效元素（按行/列分块在 `uplo` 边界处按实际有效长度裁剪 `DataCopyPad`，确保非 `uplo` 三角与 `ldc` padding 区域**零写入**）；
- **3M 还原与复数缩放**：从 `temp` 读取同一三角块的 `P0`、`P1`、`P2`，向量计算 `R = P0 - P1`、`I = (P2 - P0) - P1`；
  - 快速配置（`alpha == (1,0) && beta == (0,0)`）：直接交错写出 `(R, I)` 至 `C`（小中等规模 `n ≤ 256` 且无需对角重算时走无 `TPipe` 同步开销的 SIMT Combine 路径）；
  - 通用配置：按复数乘公式计算 `alpha * (R + i·I)`，仅在 `isBetaZero == 0` 时读取 `C_old` 并累加 `beta * C_old`（`isBetaZero == 1` 时完全跳过读 `C`，满足 `beta=(0,0)` 且 `C=NaN` 语义）；
- **对角线虚部保留与大 k 抵消保护**：对角元素 `C(i,i)` 虚部作为有效结果正常写入，绝不置 0；当 `Atype == ACL_COMPLEX64` 且 `k > 512`（或 `n <= 64 && k > 128`）时开启 `useDirectDiag`，对 n 个对角元直接读取 `Ar`/`Ai` 逐项计算 `sum_t (Xr^2 - Xi^2)` 与 `2 * sum_t (Xr * Xi)` 覆盖 `P0 - P1` 的对角分量，消除大数相减的相对误差放大；
- **Path A（Beta-only）**：复用同一 Combine kernel 的 `isAlphaZero || isKZero` 分支，不读取 `temp`/`Ar`/`Ai`（空指针安全），仅对 `uplo` 三角做 `C = beta * C`（`isBetaZero == 1` 时直接填零）。

### 精度设计

1. **Golden 参考与判定标准**：测试工程以 CPU 侧 Netlib CBLAS `cblas_csyrk`（`CblasColMajor`；float64 参考可用 `cblas_zsyrk`）生成 golden：`Atype = ACL_INT8` 时先将每个交错 `(int8 real, int8 imag)` 精确提升为 float 后调用，`Atype = ACL_COMPLEX64` 直接调用。输出 `C` 仅比对 `uplo` 指定三角（含对角线），实部与虚部分别按 FLOAT32 混合容差标准判定：`atol = rtol = 2^-13`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ 1e-2`（或 `32×ULP`）。
2. **3M 舍入差异说明**：3M 算法通过 `(Xr + Xi)·(Xr + Xi)^T − Xr·Xr^T − Xi·Xi^T` 间接构造虚部，其中间 FP32 舍入路径与非 3M 直接复数乘（`cblas_csyrk`）不同，不要求逐位相等，统一按任务书 §3.2 阈值验收。
3. **关键数值风险与控制措施**：
   - **HF32 截断风险**：`ACL_COMPLEX64` 大规模对 `Xr, Xi, Xs` 三个平面同步做 HF32x3 残差补偿，保证 `P0, P1, P2` 同等精度级，避免相减后放大单边截断误差；中小规模采用原生 FP32 Cube 或 SIMT；
   - **INT8 极值（`VALUE_INT8_EXTREME`，`-128/127`）**：`-128 + (-128) = -256` 在 HF32 与 FP32 中均精确可表示，L0C 采用 FP32 累加，最大单元素累加幅值远低于 FP32 动态范围上限，无精度损失；
   - **3M 抵消与对角虚部（`TC_3M_414~417`）**：对 `RANDOM_ALTER` / `RANDOM_EXTREME` / `RANDOM_DIAG_IMAG` 场景，通过原生 FP32（`n ≤ 256, k ≤ 512`）、HF32x3 + `useDirectDiag`（大 k）与对角虚部无清零写入保障精度；
   - **特殊值（`Inf`/`NaN`）与无 Host 扫描约束**：严守任务书 §2.5（不对 Device 张量做 Host 侧元素级扫描）；`alpha = (0,0)` 且 `A = NaN` 由 Path A/quick return 完全不读 `A` 保证；`beta = (0,0)` 且 `C = NaN` 由 Combine `isBetaZero` 分支完全不读 `C` 保证；输入矩阵含 `Inf`/`NaN` 时在 Device 侧按 IEEE-754 语义自然传播（小规模由 Path B 逐元素 3M 精确执行，避免 Cube 侧冗余分支）。

### 性能设计

1. **采集与验收口径**：在 Ascend 950PR 上通过 `msprof op` 对 GTest `TC_PF*` 用例采集单算子 kernel `Task Duration(us)`（5 次 warmup 不计入 + 10 次有效样本算术平均，剔除非 `SUCCESS` 样本并说明），除以 1000 转为 ms 后与 `test_cases/gpu_baseline.csv` 的 GPU kernel-only 基线 `gpu_ms` 做 **kernel-to-kernel** 对比：单 case ≤ `gpu_ms / 0.4`（≥ 0.4×），且全部性能 case 加速比算术平均 ≥ 0.5×。GTest `[PERF] avg_us` 与 `verify_performance.py` 仅用于快速回归趋势判断，不替代 `msprof op` 正式判据。
2. **核心性能优化手段**：
   - **小规模（`n ≤ 64`，对应 `gpu_ms` 2.5~5.1 us）**：Path B 单次发射 SIMT 3M 直算 kernel，零 workspace、省去三阶段 launch 同步开销，直接贴近硬件发射下限；
   - **中大规模（`n ∈ [128, 4096]`，典型 case 如 512×512、1024×1024、2048×2048、1024×256）**：
     - 3M 融合 Cube 将单 tile 实数矩阵乘从 4 次降至 3 次（HF32x3 模式下从 12 次 `Mmad` 降至 9 次），理论算力节省 25%；
     - 对称三角跳块省去约 50% 无效 tile，蛇形遍历 + 保留块取模分配实现多 AIC 核负载均衡与高 L2 命中率；
     - L1 双缓冲（384 KB）+ L0A/L0B 乘积级双槽 ping-pong（64 KB）+ L0C 双组累加器 ping-pong（96 KB）全面隐藏 MTE2/MTE1/Fixpipe 搬运延迟；
     - Phase 0 将解交错、`As = Ar + Ai` 与 HF32 残差计算融合在单次 AIV launch 内，消除独立残差 kernel；
   - **`ACL_INT8` 路径性能优势**：利用 `[-256, 254]` 在 HF32 下精确可表示的特性走单次 HF32 `Mmad`（3 次 `Mmad`/tile 而非 9 次）且无需残差平面搬运，吞吐显著优于 `ACL_COMPLEX64` 路径；开发者将按任务书 §3.3.6 补充采集 `ACL_INT8` 性能用例并与 `ACL_COMPLEX64` 的 `gpu_ms` 基线对比。
3. **实测性能数据**：【待填】（待上机 Ascend 950PR 完成 `msprof op` 逐 case 采集后填入自测报告）。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（`arch35`） | √ |
| Ascend 950DT（`arch35` 同源架构） | √ |

- 软件环境：CANN 9.1.0 及以上（`ASC_DEVKIT_GE_9_1`）；
- 三方参考软件：Netlib BLAS 3.10.0 / CBLAS（`cblas_csyrk` / `cblas_zsyrk`），随测试工程提供；
- `blas/syrk/README.md` 同步补充 `aclblasCsyrk3mEx` 章节，产品支持表标注 Ascend 950PR / Ascend 950DT：支持。

## 算子约束限制

- **参数校验顺序与错误码**：严格按 `handle`（`HANDLE_IS_NULLPTR`）→ `uplo`/`trans`（`INVALID_ENUM`）→ `n`/`k` 与 `lda`/`ldc`（`INVALID_VALUE`）→ `alpha`/`beta`（`INVALID_VALUE`）→ `Atype`/`Ctype`（`NOT_SUPPORTED`）→ `A`/`C` 指针（`INVALID_VALUE`）顺序执行；结构合法后再进入 quick return；
- **数据类型组合**：仅支持 `(Atype=ACL_COMPLEX64, Ctype=ACL_COMPLEX64)` 与 `(Atype=ACL_INT8, Ctype=ACL_COMPLEX64)`；`ACL_INT8` 按每复元素 `(int8 real, int8 imag)` 交错 2 字节解释；
- **对称语义**：`ACLBLAS_OP_C` 等价于 `ACLBLAS_OP_T`（不共轭）；仅 `uplo` 指定三角被读写，另一三角及 padding 区域不被改写；对角元素虚部为有效数据，计算后原样保留不置 0；
- **边界语义**：`n = 0` 为合法 no-op（允许 `A`/`C` 为 `nullptr`）；`k = 0` 或 `alpha = (0,0)` 不引用 `A`（允许 `A` 为 `nullptr` 或含 NaN），`beta = (1,0)` 时 quick return，`beta ≠ (1,0)` 时仅做 `C = beta·C`；`beta = (0,0)` 时不读取旧 `C`；
- **内存与别名**：`C` 原地覆写，不返回视图；调用方须保证 `C` 的可写区域与 `A` 的可读区域不重叠；不对 Device 张量做 Host 侧元素级扫描；`alpha`/`beta` 支持 Host/Device 双指针位置，`A`/`C` 必须为 Device 内存；
- **排布与功能边界**：仅支持列主序 `lda`/`ldc` 跨度排布，不支持任意非连续视图（不使用 AutoContiguous/ViewCopy）、不支持 broadcast、不接受 0 维 Tensor；
- **异步与确定性**：依赖 `aclblasSetStream` 绑定 stream 异步执行，Host 读回结果前须同步 stream；不要求确定性逐位一致，3M 与 `cblas_csyrk` 的舍入差异按 §3.2 阈值判定。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 输出 `COMPLEX64` 按实部/虚部 FLOAT32 分量分别比对：`atol = rtol = 2^-13`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ 1e-2`（或 `32×ULP`）；测试工程同时校验 `mere_threshold = 2^-13` 与 `mare_multiplier = 10.0`；golden 为 CBLAS `cblas_csyrk`（`ACL_INT8` 精确提升后调用） | 任务书 §3.2 / [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能标准 | Ascend 950PR `msprof op` kernel `Task Duration(us)`（5 次 warmup + 10 次有效样本均值）转 ms 后对比 `gpu_baseline.csv` `gpu_ms`（kernel-to-kernel）：单 case ≤ `gpu_ms / 0.4`（≥ 0.4×），全部性能用例加速比算术平均 ≥ 0.5×；开发者同步补充 `ACL_INT8` 性能采集 | 任务书 §3.3 / `test_cases/gpu_baseline.csv` |
| 内存标准 | 不涉及（任务书 §3.4 明确标注不涉及） | 任务书 §3.4 |

### 测试方案

1. **测试工程接入**：
   - 在 `ops-blas` 仓新建 `test/syrk/csyrk3m_ex/arch35/` 与 `test/syrk/csyrk3m_ex/CMakeLists.txt`（通过 `ops_blas_add_gtest_tests(${OPS_BLAS} csyrk3m_ex_test)` 注册构建目标，`test/CMakeLists.txt` 按家族子目录自动发现），支持 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 一键编译；
   - 实现 CSV 驱动 GTest 执行器，完整支持 `RANDOM_NORM_5_5`、`RANDOM_GAUSS_0_1`、`RANDOM_INT8`、`VALUE_INT8_EXTREME`、`RANDOM_DIAG_IMAG`、`RANDOM_ALTER`、`RANDOM_EXTREME`、`NULLPTR`、`null` 等测试 token；
   - 测试前向非 `uplo` 三角及 `ldc` padding 区域填充哨兵值，调用后严格校验未被改写；同时在 C++ 侧独立覆盖 `handle == nullptr`（期望 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）负向用例；
   - 提供完整可复现测试 README，配合 `verify_accuracy.py`、`verify_performance.py` 及 `msprof op` 命令完成自验。
2. **精度用例覆盖（任务书自带 `csyrk3m_ex_test.csv` 精度全量 1000 条）**：
   - 正向 982 条 + 负向 18 条（另加 C++ 装配层 `handle == nullptr` 用例），`Atype` 覆盖 505 条 `ACL_COMPLEX64` + 493 条 `ACL_INT8` + 2 条 `ACL_FLOAT`（不支持类型负向）；
   - 覆盖 6 组 `uplo × trans` 组合、`n, k ∈ [0, 4096]` 全尺寸扫描（含 2 的幂及 ±1、宽窄矩形、`+1/+8/+64` padding）、均匀/正态/全零/交替/极值/Inf/NaN 填充；
   - 重点覆盖边界与 3M 专项用例：`TC_3M_414~417`（交替/极值抵消、`VALUE_INT8_EXTREME`、`RANDOM_DIAG_IMAG` 对角虚部保留）、`k = 0 & beta ≠ (1,0)` 缩放、`alpha = (0,0) & A = NaN` 不读 `A`、`beta = (0,0) & C = NaN` 不读旧 `C`、非法枚举（`INVALID_ENUM`）、不支持 dtype（`NOT_SUPPORTED`）、非法前导维与负维度（`INVALID_VALUE`）。
3. **性能自测与采样**：
   - 使用 `msprof op --application="./build/test/syrk/csyrk3m_ex/csyrk3m_ex_test --gtest_filter=*TC_PF*" --output=./prof_csyrk3m_ex` 全量采集 `gpu_baseline.csv` 的 200 条 `TC_PF_*` `ACL_COMPLEX64` 用例（含前 5 条典型 case：N/512/512、N/1024/1024、N/2048/2048、T/2048/2048、C/1024/256），按 5 次 warmup + 10 次有效样本均值统计 `Task Duration(us)` 并回填自测报告；
   - 按任务书 §3.3.6 补充采集 `Atype = ACL_INT8` 性能用例（沿用 `ACL_COMPLEX64` `gpu_ms` 基线对比），并辅以 `verify_performance.py`（解析 `[PERF] case=... avg_us=...`）做开发期快速回归。

## 兼容性分析

- **公共接口兼容**：`include/cann_ops_blas.h` 仅新增 `aclblasCsyrk3mEx` 声明，风格与参数约定完全对齐同族 `aclblasCsyrk` / `aclblasCgemmEx`，不修改任何既有声明与行为，可供其他产品线复用；错误码、handle、stream 与 `aclblasComplex` 定义全部复用 `include/cann_ops_blas_common.h` 与公共 handle 设施；
- **与同批任务及存量算子解耦**：本实现完全独立于同批新增的 `aclblasCsyrkEx`，互不依赖；实现文件以 `csyrk3m_ex_*` 命名并置于 `blas/syrk/arch35/`，不影响既有 `aclblasCsyrk` / `aclblasSsyrk` 的编译、符号与单算子构建；
- **版本与架构兼容**：依赖 `te::`（tensor_api）的 arch35 Cube 路径使用 `ASC_DEVKIT_GE_9_1` 守卫（与仓内 `cgemm3m` / `ssyrk` 一致），低于 CANN 9.1 时安全跳过不破坏全仓构建；
- **向后兼容**：新增算子，无存量调用方，不涉及向后兼容风险；非法 `uplo`/`trans` 返回 `ACLBLAS_STATUS_INVALID_ENUM` 与 `ops-blas` 全仓约定一致，并在 `blas/syrk/README.md` 明确说明与 cuBLAS 的有意差异。
