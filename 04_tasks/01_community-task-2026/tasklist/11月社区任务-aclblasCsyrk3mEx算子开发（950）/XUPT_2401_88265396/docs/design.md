# aclblasCsyrk3mEx 算子设计文档（Ascend 950PR，arch35）

> 依据《aclblasCsyrk3mEx_Atlas950PR 任务书》与 CANN 社区任务算子设计文档模板编写，四个必需一级章节齐全。
> 文中的实现描述以 2026-11-25 本地源码快照为准；快照之外的能力（Cube 接通、真机精度、性能）均显式标注为待验证。
> 设计评审通过不等于算子通过验收：文档评审、算子精度/性能自验、平台验收分别引用各自的证据。

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `aclblasCsyrk3mEx` |
| 所属任务 | 11 月社区任务 aclblasCsyrk3mEx 算子开发（950），讨论帖 [#88](https://gitcode.com/cann/ops-blas/discussions/88) |
| 目标硬件 | Ascend 950PR（arch35，DAV-3510） |
| 软件版本 | CANN 9.1.0 及以上；开发实例基线 CANN 9.2.1 |
| 精度标杆 | Netlib CBLAS 3.10.0 `cblas_csyrk`；float64 参考可用 `cblas_zsyrk` |
| 实现仓库／目录 | `cann/ops-blas`，实现 `blas/syrk/arch35/`，测试 `test/syrk/csyrk3m_ex/arch35/` |
| 代码基线 | `6df9aa04a1d37288966e1b97a20ae8033ca2ce3c` |
| 文档状态 | v2 修订稿待复审（v1 为 PR #3306 `2be350d0`）；Host 当前调用 SIMT 路径，Cube 路径待接通，真机精度与性能待采集 |

**实现描述对应的源码快照**（SHA256 前 12 位，完整值见本地快照目录）：

| 文件 | SHA256 前缀 | 状态 |
| --- | --- | --- |
| `include/cann_ops_blas.h` | `83ea5ced76ad` | 已修改（新增声明） |
| `blas/CMakeLists.txt` | `19ec00d2743b` | 已修改（单算子构建保留 Cube 源） |
| `blas/gemm/arch35/cgemm_compute_kernel.cpp` | `c61893739b58` | 已修改（包含 Cube 实现头） |
| `blas/syrk/arch35/csyrk3m_ex_host.cpp` | `5e8f8f7865d2` | 新增 |
| `blas/syrk/arch35/csyrk3m_ex_tiling.h` | `7eebcc7da7ca` | 新增 |
| `blas/syrk/arch35/csyrk3m_ex_kernel.cpp` | `f515706cf414` | 新增 |
| `blas/syrk/arch35/csyrk3m_ex_cube_impl.h` | `ec608b1fc7e4` | 新增（未接通） |

**修订记录**

| 版本 | 日期 | 依据 | 主要变更 |
| --- | --- | --- | --- |
| v1 | 2026-11-25 | 初稿（PR #3306，`2be350d0`） | 按官方模板整理接口、3M、SIMT/Cube 与验证方案 |
| v2 | 2026-11-25 | PR #3306 复审意见 | 非有限回退收窄到 Inf/NaN，有限输入一律 3M；新增 `(xr+xi)` 吸收问题的 TwoSum 低位补偿与反例回归；三角边界改为逐 lane 过滤；修正 3M 转置表述；补充交付物去向 |

# 1. 需求背景（required）

## 1.1 需求来源

昇腾 CANN 2026 年 11 月社区任务要求在 `cann/ops-blas` 新增单精度复数对称矩阵秩-k 更新接口 `aclblasCsyrk3mEx`，对标 cuBLAS `cublasCsyrk3mEx`：

```text
C = alpha · op(A) · op(A)^T + beta · C
```

`C` 为 `n×n` 列主序对称复数矩阵，仅 `uplo` 指定三角被读写；`A` 为列主序矩阵，支持 `COMPLEX64` 与按复元素交错的 complex-int8；`alpha`、`beta` 为 `COMPLEX64` 标量。任务书同时规定：三次实数矩阵乘必须在**同一次 kernel launch** 内完成，不得用三次独立 GEMM 调用拼接，也不得依赖同批新增的 `aclblasCsyrkEx`。

本任务分三处交付：设计文档提交到 `cann/cann-ops-competitions` 的任务目录（本 PR）；算子实现、测试代码与算子 README 提交到 `cann/ops-blas`；正式验收材料（自验证步骤、精度/性能/内存自验报告与日志）提交任务 IT 系统，并同步到开发者私仓的 `task_submission/`。本文只覆盖设计文档这一交付件，完整去向见 §4.6。

## 1.2 背景介绍

### 1.2.1 仓内可复用实现

| 参考位置 | 可复用内容 | 本任务用途 |
| --- | --- | --- |
| `blas/syrk/arch35/csyrk_host.cpp`、`csyrk_kernel.cpp` | 参数校验、前导维度、三角范围、复数缩放、SIMT 地址映射 | Host 校验顺序与 SIMT 直接路径的基础 |
| `blas/syrk/arch35/csyrk_reference_device.h` | 逐元素直接复数乘、非有限值处理 | 特殊 IEEE 输入的回退分支 |
| `blas/syrk/arch35/csyrk_cube_kernel.cpp`、`csyrk_gemm_fused.h` | 三角 tile 划分、Cube 访存、尾块处理 | 三角 tile 调度思路 |
| `blas/gemm/arch35/cgemm_onchip_kernel.h` | AIV 打包、AIC 三乘积、AIC→AIV 回传、L1 双缓冲与跨核同步 | Cube 性能路径的共用设备入口 |
| `blas/gemm/arch35/cgemm3m_rounding_kernel.cpp` | 3M 分解与 FP32 舍入处理 | 舍入与补偿策略参考 |
| `blas/gemm/arch35/cgemm_compute_kernel.cpp` | 复数 GEMM 的分段流水与复数合并 | Cube 实现的编译宿主 |

### 1.2.2 与仓内相近实现的差异

1. **对称而非 Hermitian。** 乘积是 `X·X^T`，虚部为 `Xr·Xi^T + Xi·Xr^T`；若误用 Hermitian 的 `X·X^H` 分解，虚部会变成 `Xi·Xr^T − Xr·Xi^T`，符号错误。
2. **3M 对称分解。** 复用 `P0=Xr·Xr^T`、`P1=Xi·Xi^T`、`P2=(Xr+Xi)(Xr+Xi)^T` 三次实数乘，比标准复数 GEMM 的四乘积少一次；`Xr·Xi^T` 与 `Xi·Xr^T` **互为转置**，`P2` 展开后同时含这两项，`P2−P0−P1` 得到二者之和（对称矩阵），因此不需要第四次矩阵乘。三次乘必须同次下发。
3. **`OP_C` 不共轭。** 对称语义下 `trans=ACLBLAS_OP_C` 与 `ACLBLAS_OP_T` 等价，装载与寻址完全一致，仅枚举校验保留三分。
4. **对角虚部有效。** 对称矩阵的 `C(i,i)` 虚部是数据，必须按公式计算并保留，不能按 Hermitian 习惯置 0。
5. **INT8 交错输入。** `Atype=ACL_INT8` 时每个复元素为 `(int8 real, int8 imag)` 两字节，对应 cuBLAS `CUDA_C_8I`；`lda` 仍以复元素计数。
6. **写保护范围。** 只读写 `uplo` 三角与对角，另一三角和 `ldc` padding 必须逐字节保持；这一约束同时决定 Cube 的三角 tile 调度与对角 tile 的掩码写回。

### 1.2.3 设计边界

- 只新增接口与 arch35 实现，不修改既有 `aclblasCsyrk`、`cgemm` 的行为与签名。
- 所有**有限**输入一律走 3M，不按量级分流；只有分量本身为 Inf/NaN 的特殊 IEEE 输入才走逐元素直接复数乘的回退分支，触发条件与证据要求见 §3.2.4。
- Cube 路径是拟用的性能方案，设备入口已存在但 Host 未调用；本文把它与已生效的 SIMT 路径分开描述，避免把未验证能力写成现状。
- 任务书 §3.4 明确“内存要求：不涉及”，本文据此不设计额外内存校验项。

# 2. 需求分析（required）

## 2.1 需求描述

在 `blas/syrk/arch35/` 新增 `aclblasCsyrk3mEx` 入口与 3M 计算路径，覆盖：

- `uplo ∈ {UPPER, LOWER}`，`trans ∈ {N, T, C}`（`C` 等价 `T`，不共轭）；
- `n, k ≥ 0`；`n=0` 合法 no-op；`k=0` 或 `alpha=0` 时秩-k 贡献为零；
- `Atype ∈ {ACL_COMPLEX64, ACL_INT8}`，`Ctype=ACL_COMPLEX64`，其他组合返回 `NOT_SUPPORTED`；
- `lda` / `ldc` 按 `trans` 与 `n` 取最小下界，支持 padding；
- `alpha`、`beta` 支持 Host 与 Device 指针，Device 标量按 handle 绑定 stream 顺序读取；
- 精度按 FLOAT32 分量阈值判定，性能按 kernel-to-kernel 口径判定（详见 §4.1）。

## 2.2 需求拆解

| 编号 | 需求 | 来源 | 设计对应 |
| --- | --- | --- | --- |
| R01 | 公开签名、状态码与参数语义对齐 cuBLAS `cublasCsyrk3mEx` | 任务书 §2.1/§2.3/§2.4 | §2.3 |
| R02 | 列主序，`lda`/`ldc` 以复元素计，支持 padding | §2.1/§2.4 | §2.3.2、§3.1.4 |
| R03 | 仅 `uplo` 三角与对角读写，对角虚部保留，另一三角与 padding 不变 | §2.1/§2.5 | §2.3.2、§3.2.2 |
| R04 | `OP_C` 按 `OP_T` 等价处理，不做共轭 | §2.1 | §2.3.1、§3.1.4 |
| R05 | INT8 复元素两字节交错，实虚分量精确提升后计算 | §2.1/§2.5 | §2.3.2、§3.2.2 |
| R06 | 校验顺序与 `n=0`、`k=0`、`alpha=0`、`beta=0/1`、空指针组合 | §2.5 | §2.4 |
| R07 | Host 标量按值、Device 标量按 stream 顺序；`A=nullptr` 时的校验例外 | §2.4/§2.5 | §3.2.1 |
| R08 | 三次实数乘在同一次 launch 内完成，正常路径为 3M | §2.1/§2.2 | §3.1.1、§3.2.2、§3.2.3 |
| R09 | 大规格提供 Cube 性能路径，具备三角 tile、跨核同步、尾块与写保护 | 任务书 §3.3 | §3.2.3 |
| R10 | 精度：1000 条用例，实虚分量分别判定 | §3.2/§3.5 | §4.1、§4.2.2 |
| R11 | 性能：200 条 COMPLEX64 用例，单例 ≥0.4×、均值 ≥0.5×，INT8 另补 | §3.3/§3.5 | §4.1、§4.2.3 |
| R12 | 兼容性、约束与不支持场景显式声明 | §2.5/§3.1 | §3.4、§4.3 |

## 2.3 功能与接口

### 2.3.1 公开签名与参数语义

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

| 参数 | 语义与约束 | 违反时的返回 |
| --- | --- | --- |
| `handle` | 已创建的 ops-blas 句柄，提供 stream | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `uplo` | `ACLBLAS_UPPER` 更新 `i ≤ j`；`ACLBLAS_LOWER` 更新 `i ≥ j` | `ACLBLAS_STATUS_INVALID_ENUM` |
| `trans` | `OP_N`：`op(A)=A`；`OP_T`/`OP_C`：`op(A)=A^T`，`OP_C` 不共轭 | `ACLBLAS_STATUS_INVALID_ENUM` |
| `n` | `C` 的阶数，`n ≥ 0` | `n < 0` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `k` | 秩-k 归约长度，`k ≥ 0` | `k < 0` → `ACLBLAS_STATUS_INVALID_VALUE` |
| `alpha` | 复数标量，Host 或 Device 指针 | 空指针 → `ACLBLAS_STATUS_INVALID_VALUE` |
| `A` | Device 列主序矩阵；`OP_N` 为 `n×k`，否则 `k×n` | 需要乘积且为空 → `ACLBLAS_STATUS_INVALID_VALUE` |
| `Atype` | `ACL_COMPLEX64` 或 `ACL_INT8` | 其他值 → `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `lda` | `OP_N`：`lda ≥ max(1,n)`；否则 `lda ≥ max(1,k)` | 越界 → `ACLBLAS_STATUS_INVALID_VALUE` |
| `beta` | 复数标量，Host 或 Device 指针 | 空指针 → `ACLBLAS_STATUS_INVALID_VALUE` |
| `C` | Device 列主序 `n×n`，原地覆写 | `n>0` 且为空 → `ACLBLAS_STATUS_INVALID_VALUE` |
| `Ctype` | 仅 `ACL_COMPLEX64` | 其他值 → `ACLBLAS_STATUS_NOT_SUPPORTED` |
| `ldc` | `ldc ≥ max(1,n)` | 越界 → `ACLBLAS_STATUS_INVALID_VALUE` |

`OP_C` 在接口层归一为 `OP_T`：不读取共轭信息，也不改变虚部符号；枚举校验仍保留三值，便于与 cuBLAS 参数序列一一对应。

### 2.3.2 数据类型与内存布局

| 对象 | 类型 | 单复元素字节 | 布局 |
| --- | --- | --- | --- |
| `C` | `ACL_COMPLEX64` | 8 | `(float real, float imag)` 交错，列主序，`C(i,j)` 位于 `j·ldc + i` |
| `A`（`ACL_COMPLEX64`） | `ACL_COMPLEX64` | 8 | 列主序存储；`OP_N` 时逻辑 `A(i,j)` 位于 `j·lda + i`，转置时按 §3.1.4 的映射访问 |
| `A`（`ACL_INT8`） | complex-int8 | 2 | `(int8 real, int8 imag)` 交错，对应 `CUDA_C_8I`，`A(i,j)` 位于 `2·(j·lda + i)` 字节 |
| `alpha`、`beta` | `ACL_COMPLEX64` | 8 | 2 个连续 float；Device 指针时同样按 `[real, imag]` 排列 |

要点：

- `lda`、`ldc` 的单位是**复元素**，不是标量 float 或字节。INT8 输入的 `lda` 同样以复元素计，字节跨度自动为 `2·lda`。
- `trans=OP_N` 时 `A` 的逻辑形状是 `n×k`；`trans=OP_T/OP_C` 时以 `k×n` 存储，`op(A)` 仍是 `n×k`。
- `C` 的 padding 位于每列末尾（`i ≥ n` 且 `i < ldc`）以及最后一个复元素之后；接口不写这些位置。

### 2.3.3 状态码

| 状态码 | 触发条件 |
| --- | --- |
| `ACLBLAS_STATUS_SUCCESS` | 正常下发；或合法的 no-op / quick return |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | `handle == nullptr` |
| `ACLBLAS_STATUS_INVALID_ENUM` | `uplo`、`trans` 不在合法枚举内 |
| `ACLBLAS_STATUS_INVALID_VALUE` | `n<0`、`k<0`、`lda`/`ldc` 低于下界、`alpha`/`beta`/`C` 空指针、需要乘积时 `A` 为空、标量指针位置查询失败、地址上界溢出 |
| `ACLBLAS_STATUS_NOT_SUPPORTED` | `Atype`/`Ctype` 不在支持集 |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | Device alpha 读取失败、AIV 核数为 0、kernel 下发失败 |

`INVALID_ENUM` 是 ops-blas 对非法 `uplo`/`trans` 的约定；cuBLAS 对同类错误返回 `CUBLAS_STATUS_INVALID_VALUE`，属有意差异，已在任务书 §2.3 说明。

## 2.4 校验顺序与退化组合

### 2.4.1 校验顺序

实现按下列顺序短路，先命中者决定返回值：

1. `handle == nullptr` → `HANDLE_IS_NULLPTR`；
2. `uplo` 不在 `{UPPER, LOWER}`，或 `trans` 不在 `{N, T, C}` → `INVALID_ENUM`；
3. `n < 0`、`k < 0`、`lda < max(1, trans==N ? n : k)`、`ldc < max(1, n)` → `INVALID_VALUE`；
4. `alpha == nullptr` 或 `beta == nullptr` → `INVALID_VALUE`；
5. `Atype ∉ {COMPLEX64, INT8}` 或 `Ctype ≠ COMPLEX64` → `NOT_SUPPORTED`；
6. `n == 0` → `SUCCESS`，此后不引用 `A`/`C`、不读取标量、不 launch；
7. `C == nullptr`，或 `C` 的地址上界溢出（`(n−1)·ldc + n > SIZE_MAX/8`）→ `INVALID_VALUE`；
8. `alpha`（仅 `k≠0` 时）与 `beta` 的指针位置查询失败 → `INVALID_VALUE`；
9. `k≠0` 且 `A==nullptr` 且 `alpha` 是 Device 指针：在 handle stream 上做一次 D2H 拷贝并同步，读取 alpha 当前值（失败 → `INTERNAL_ERROR`）；
10. 需要乘积时（`k≠0` 且未命中零 alpha 快捷判定）`A == nullptr`，或 `A` 的地址上界溢出 → `INVALID_VALUE`；
11. 秩-k 贡献为零、`beta` 为 Host 指针且 `beta==(1,0)` → `SUCCESS`，不 launch；
12. 其余情况组装 tiling 并下发 SIMT kernel；下发后 `aclrtGetLastError` 非 `SUCCESS` → `INTERNAL_ERROR`。

顺序上有两点需要说明：

- `alpha`/`beta` 判空与 dtype 校验发生在 `n==0` **之前**，因此 `n=0` 时 `alpha=nullptr` 仍返回 `INVALID_VALUE`，不支持的 dtype 仍返回 `NOT_SUPPORTED`；这与任务书 §2.4 的异常行为一致。
- 第 9 步是唯一会在 Host 侧同步 stream 的常规路径，只用于“`A` 为空但需要判断 Device alpha 是否为零”的结构校验；正常 Device 标量调用不做同步。

### 2.4.2 退化与空指针组合

| 场景 | 读 `A` | 读旧 `C` | 行为 | launch |
| --- | --- | --- | --- | --- |
| `n=0` | 否 | 否 | `SUCCESS`，C 不变 | 否 |
| `k=0`，Host `beta=(1,0)` | 否 | 否 | quick return | 否 |
| `k=0`，Host `beta≠(1,0)` | 否 | `beta≠0` 时读 | 仅三角 `C=beta·C` | 是（缩放） |
| `k=0`，Device `beta` | 否 | kernel 内按 beta | kernel 判 `beta==1` 则直接返回，否则缩放 | 是（kernel 内快捷） |
| Host `alpha=(0,0)`，Host `beta=(1,0)` | 否 | 否 | quick return | 否 |
| Host `alpha=(0,0)`，Host `beta≠(1,0)` | 否 | `beta≠0` 时读 | 仅三角 `C=beta·C` | 是 |
| Device `alpha=(0,0)`，`A≠nullptr` | 否（kernel 提前转缩放） | 视 beta | kernel 内 `IsZero(alpha)` 后缩放 | 是 |
| Device `alpha`，`A=nullptr`，alpha 为零 | 否 | 视 beta | 第 9 步同 stream 读取后缩放 | 视 beta |
| Device `alpha`，`A=nullptr`，alpha 非零 | — | — | `INVALID_VALUE` | 否 |
| Host `beta=(0,0)`，旧 C 含 NaN | 视 alpha | 否 | 跳过旧 C 加载，写 `alpha·product` | 是 |
| `beta=(1,0)`，乘积非空 | 是 | 是 | `C = alpha·product + C` | 是 |
| `A=nullptr`、`k>0`、Host alpha 非零 | — | — | `INVALID_VALUE` | 否 |
| `C=nullptr`、`n>0` | — | — | `INVALID_VALUE` | 否 |

补充：`n=0` 时仍要求 `lda`、`ldc` 满足下界（`lda ≥ max(1, trans==N?n:k)`、`ldc ≥ max(1,n)`），因为这两项在第 3 步就已校验。`beta=(0,0)` 的 NaN 安全依赖 kernel 内“beta 为零则不加载旧 C”，Host 不做数据面判断。

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

令 `X = op(A) = Xr + i·Xi`。`Atype=ACL_INT8` 时先把每个 int8 分量精确提升为 float。对称 3M 分解为：

```text
P0 = Xr · Xr^T
P1 = Xi · Xi^T
P2 = (Xr + Xi) · (Xr + Xi)^T

Pr = P0 − P1
Pi = (P2 − P0) − P1
C  = alpha · (Pr + i·Pi) + beta · C
```

推导：`(Xr+iXi)(Xr+iXi)^T = XrXr^T − XiXi^T + i(XrXi^T + XiXr^T)`，而 `P2 = XrXr^T + XrXi^T + XiXr^T + XiXi^T`，故 `Pi = P2 − P0 − P1`。注意这里没有共轭项，`Pi` 是本征值意义上的对称虚部，`C(i,i)` 的虚部因此非零且有效。

三项实数乘积在同一 kernel 内完成：SIMT 路径在同一次归约循环中累加三组乘积；Cube 路径在同一次 MIX 下发中由 AIC 连续完成三个实数乘。

### 3.1.2 支持数据类型

| 位置 | 支持 | 说明 |
| --- | --- | --- |
| `A` | `ACL_COMPLEX64`、`ACL_INT8` | INT8 为 `(int8 real, int8 imag)` 交错两字节 |
| `C` | `ACL_COMPLEX64` | 输出始终 complex64 |
| `alpha`、`beta` | `ACL_COMPLEX64` | Host 或 Device 指针 |
| 合法组合 | `(COMPLEX64, COMPLEX64)`、`(INT8, COMPLEX64)` | 其他组合 `NOT_SUPPORTED` |

INT8 分量的提升是精确的：int8 的 256 个取值都能被 float 精确表示，因此提升不引入额外舍入；`-128` 与 `127` 同样适用。

### 3.1.3 支持形状与前导维度

| 项 | 约束 |
| --- | --- |
| `n`、`k` | 非负整数；测试包覆盖 0 与 1…4096 的代表尺寸 |
| `A` 形状 | `OP_N`：`n×k`；`OP_T/OP_C`：`k×n` |
| `C` 形状 | 逻辑 `n×n`，物理 `ldc×n` |
| `lda` | `OP_N`：`max(1,n)` 起；否则 `max(1,k)` 起 |
| `ldc` | `max(1,n)` 起 |
| padding | `lda`/`ldc` 可大于最小下界，支持 `+1/+8/+64` 等填充；接口不写 padding |

接口不设硬编码尺寸上限；实际可用规模受 64 位寻址、Device 内存与测试包 512 MB 预算约束。测试包按 `8·(n·k + n²) ≤ 512 MB` 选取性能用例，`n ≤ 4096`，这是测试设计选择而非硬件限制。

### 3.1.4 寻址与溢出保护

- 逻辑索引 `(i, j)`：`OP_N` 时 `A` 元素位于 `(j·lda + i)`，`OP_T/OP_C` 时位于 `(i·lda + j)`；`C` 元素位于 `(j·ldc + i)`。
- 字节地址：`COMPLEX64` 为 `8·index`，INT8 为 `2·index`。所有索引乘加使用 `uint64_t`，避免 `n=4096`、`lda` 较大时 32 位溢出。
- 地址上界检查在 Host 完成：`elements = (cols−1)·leading + rows`，要求 `elements ≤ SIZE_MAX / elementBytes`；`C` 用 `(n,n,ldc,8)`，`A` 用 `(rows, cols, lda, 2 或 8)`。检查失败返回 `INVALID_VALUE`。
- 三角判定：`UPPER` 为 `i ≤ j`，`LOWER` 为 `i ≥ j`；判定在 kernel 内按逻辑索引进行，与 `ldc` padding 无关。

## 3.2 算子实现

### 3.2.0 文件布局

| 文件 | 作用 |
| --- | --- |
| `include/cann_ops_blas.h` | 新增公开声明（与既有 `aclblasCsyrk` 相邻） |
| `blas/syrk/arch35/csyrk3m_ex_host.cpp` | 参数校验、标量处理、tiling 组装、kernel 下发 |
| `blas/syrk/arch35/csyrk3m_ex_tiling.h` | `Csyrk3mExTiling` 与两个入口声明 |
| `blas/syrk/arch35/csyrk3m_ex_kernel.cpp` | AIV/SIMT 直接路径（当前生效） |
| `blas/syrk/arch35/csyrk3m_ex_cube_impl.h` | Cube 设备入口（被 `cgemm_compute_kernel.cpp` 包含，未接通） |
| `blas/CMakeLists.txt` | 单算子构建时保留 `cgemm_compute_kernel.cpp` |
| `test/syrk/csyrk3m_ex/` | CMake 目标、CSV 驱动的 C++ GTest、CSV 副本 |

### 3.2.1 Host 侧设计

**（1）结构校验与归一化。** 按 §2.4.1 顺序短路；`trans` 只保留“是否转置”一位，`OP_C` 在入口处即与 `OP_T` 合并。`uplo` 转为布尔 `upper`。不做任何 Device 张量元素级扫描。

**（2）标量处理。** 分三种情况：

| 标量位置 | Host 行为 | kernel 行为 |
| --- | --- | --- |
| Host `alpha`/`beta` | 解引用取值，写入 tiling 的 `alphaReal/alphaImag/betaReal/betaImag` | 直接使用 tiling 值 |
| Device `alpha`（常规） | 只做指针位置查询，不读值、不同步 | 从 GM 读取 `[real, imag]`，按 stream 顺序可见 |
| Device `alpha` 且 `A==nullptr` | 同 stream D2H + `aclrtSynchronizeStream` 读取当前值，用于判断是否需要 `A` | 用 tiling 中已读到的值（零）；非零时 Host 已返回 `INVALID_VALUE`，不会进入乘积分支 |
| Device `beta` | 只做指针位置查询，tiling 写占位零 | 从 GM 读取 `[real, imag]`，自行判断 `beta==1`/`beta==0` |

“按 stream 顺序读取”指 Device 标量由 kernel 在 Device 侧读取，Host 不插入额外同步；上游在**同一 stream** 上先写入标量的异步操作，其效果对 kernel 可见。若上游在别的 stream 写标量，调用方需自行建链，接口不做跨 stream 同步。

`A=nullptr` 的例外：Host 必须先知道 alpha 是否为零，才能判定“允许 A 为空”还是“返回 `INVALID_VALUE`”，因此这里做一次阻塞读取。失败返回 `INTERNAL_ERROR`；读取成功且 alpha 为零时继续走缩放路径。该分支是本接口唯一的 Host 同步点，正常调用不受影响。

**（3）tiling。** `Csyrk3mExTiling` 为 48 字节 POD，按值传给 kernel：

```cpp
struct Csyrk3mExTiling {
    uint32_t n, k, lda, ldc;
    uint32_t transposed, upper, packedInt8, rowLanes;
    float alphaReal, alphaImag, betaReal, betaImag;
};
```

`rowLanes` 是 SIMT 的行分组数：`!transposed && n ≥ 32` 时取 4（一个 warp 处理 4 行、每行 8 lane 归约），否则取 1（整 warp 归约一行）。

**（4）分核与下发。** 记 `product = !skipProduct`，即 `k≠0` 且 alpha 非零；Device alpha 在未触发第 9 步读取时按非零处理（Host 不预判，kernel 内再判）：

```text
rowGroups   = ceil(n / rowLanes)
tasks       = n · (product ? rowGroups : n)
tasksPerCore= product ? 8 : 256
blocks      = min(AIV 核数, ceil(tasks / tasksPerCore))
```

缩放路径按输出元素计任务（每线程一个元素粒度），乘积路径按 warp 计任务（每 warp 一个 `(行组, 列)`）。`blocks` 上限为平台 AIV 核数，避免空转 block。

下发形式为 `csyrk3m_ex_simt_kernel<<<blocks, nullptr, stream>>>(...)`，kernel 内 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，每个 block 通过 `asc_vf_call` 启动 256 个 SIMT 线程。下发前清零本线程错误码，下发后检查 `aclrtGetLastError(ACL_RT_THREAD_LEVEL)`，非 `ACL_SUCCESS` 返回 `INTERNAL_ERROR`。

**（5）资源。** 不申请 GM workspace，不创建临时 Device 缓冲，不改 handle 的 stream。除 §2.4.1 第 9 步外不做同步。

### 3.2.2 AIV/SIMT 直接路径（当前生效）

**任务映射。** 每个 block 有 8 个 warp（256 线程）。warp 以 `index = core·warps + warpId` 起步，按 `cores·warps` 跨步遍历全部 `(行组, 列)`：

```text
row = (index mod rowGroups) · rowLanes + (lane mod rowLanes)
col = index / rowGroups
```

每个 lane 用自己的 `(row, col)` 独立判定 `InTriangle`。同一 warp 覆盖的 4 行可能只有部分落在 `uplo` 三角内（上三角 `col` 固定、`row` 跨越 `col` 时尤其明显），因此**不能按 warp 整组跳过**：`active` 为假的 lane 不参与该元素的归约，写回条件是 `active && lane < rowLanes`，其余 lane 仍参与归约但不写。这样对角附近的有效元素不会被整组漏写，每个输出位置仍只有一个写者。

**三项乘积与补偿归约。** 每个 lane 在 K 方向以 `step = 32 / rowLanes` 跨步，累计三组主乘积与一组低位修正：

```text
(sx, ex) = TwoSum(xr, xi)      // sx + ex = xr + xi，精确
(sy, ey) = TwoSum(yr, yi)      // sy + ey = yr + yi，精确
p0 += xr·yr ; p1 += xi·yi ; p2 += sx·sy
low += sx·ey + ex·sy           // P2 被吸收的一阶交叉项
```

补偿采用 Neumaier 形式：每次 `Add(v)` 计算 `corrected = v − error`、`next = total + corrected`、`error = (next − total) − corrected`，`total = next`。每个 lane 先在本地合成复部分和，再用 `asc_shfl_down` 蝶式归约到行组内的 8 或 32 个 lane：`real = (p0−p1)+(p1e−p0e)`、`imag = ((p2−p0)−p1)+(p0e+p1e−p2e)+(low−lowe)`。先合成再归约，避免三个大累加器在整 warp 上做相减放大误差。

**`(xr+xi)` 的吸收与低位补偿（v2 新增）。** 3M 的 `P2` 必须先形成 `xr+xi`。当 `|xi| ≪ |xr|` 时，FP32 加法会把 `xi` 整个舍掉（例如 `fl(1+1e-8)=1`），`P2−P0−P1` 随之为 0，虚部丢失 `2·xr·xi`。可复算的反例：`n=k=1`、`A=(1, 1e-8)`、`alpha=(1e4, 0)`、`beta=0`，真值虚部约 `2e-4`，未补偿的 3M 只得到约 `−1e-12`，超出 `1.221e-4` 的匹配容差。

处理方式：用 `TwoSum` 把两个和分别表示为 `(sx, ex)`、`(sy, ey)`，三次主乘积仍只做 `xr·yr`、`xi·yi`、`sx·sy`；被吸收的一阶项 `sx·ey + ex·sy` 累加到独立的 `low`（同样用 Neumaier 补偿），二阶项 `ex·ey` 量级约 `1e-16`，忽略。重构时把 `low` 并入虚部。该方案仍是对 3M 恒等式的补偿求值，**不增加第四次矩阵乘**；`TwoSum` 与重构段需关闭 contraction/reassociation（`#pragma clang fp contract(off) reassociate(off)`），否则编译器可能把残差消掉。

代价是每 K 项增加约 2 次 `TwoSum`（约 12 次浮点加减）与 2 次乘加；SIMT 路径以访存为主，实际影响需用 `msprof op` 在 512/1024/2048 方阵上实测（列入 §4.2.3 与 §4.2.2 回归）。同一反例也适用于 Cube 的 FP32 Gauss 打包：`(xr+xi)` 在打包阶段已经丢位，而给 Cube 增加残差面会引入额外实数矩阵乘、超出 3M 约束，因此 Cube 的适用数据域必须由实测精度界定（见 §3.2.3(7)）。

INT8 路径用 `int64_t` 精确累加（`-128…127` 的乘积与 `k ≤ 4096` 的和不会溢出），归约后用 `ShuffleDownInt64` 合并，最后一次性转 float：整数累加本身无舍入，转 float 时可能损失低位，之后 alpha/beta 融合仍按 FP32 计算。

**alpha/beta 融合与写回。** 归约结果为 `product = Pr + i·Pi`，随后：

- `alpha==(1,0)` 时直接使用，否则做一次复数乘；
- `beta==(0,0)` 时**不加载**旧 `C`，只写 `alpha·product`（旧 C 含 NaN 也安全）；
- `beta==(1,0)` 时取旧 `C` 原值相加，否则做复数乘后相加；
- 写回 `C(j·ldc+i)` 的实部与虚部，`j·ldc+i` 之外的地址不触碰。

**缩放路径。** `k==0` 或 alpha 为零时，按 `index = core·256 + threadIdx.x` 起步、`cores·256` 跨步遍历 `n·n` 个逻辑位置，仅对三角内位置执行 `C = beta·C`；`beta==(1,0)` 直接返回，`beta==(0,0)` 跳过旧值加载。该路径与乘积路径共用 `Store`，行为一致。

**写保护。** 三角外元素既不计算也不写；`ldc` padding 与最后一个元素之后的哨兵区不被访问。测试用哨兵验证（见 §4.2.2）。

### 3.2.3 MIX AIC/AIV 性能路径（Cube，待接通）

**（1）设备入口与构建接入。**

```cpp
void csyrk3m_ex_cube_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* c,
                        uint32_t n, uint32_t k, uint32_t lda, uint32_t ldc,
                        bool transposed, bool upper);
```

实现位于 `csyrk3m_ex_cube_impl.h`，通过 `cgemm_compute_kernel.cpp` 末尾的 `#include` 进入共用设备编译单元，复用其中的 `cgemm_onchip`（AIV 打包、AIC 乘、跨核同步）与 `cgemm_fused`（输出面、写回）。`blas/CMakeLists.txt` 在 `TEST_NAMES` 含 `csyrk3m_ex` 时保留该源文件，保证窄目标构建也能链接到设备入口。kernel 类型为 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC 配 2 AIV），AIC 侧关闭 HF32。

入口组装 `CgemmOnchipTilingData{n, n, k, lda, lda, ldc, transA=transposed, transB=!transposed, 0, 0}`，把 `A` 同时作为左右操作数，实现 `X·X^T`。

**（2）三角 tile 划分。** 输出按 64×64 tile 切分（`CGEMM_FUSED_TILE = 64`，由 `CGEMM_MAX_SQUARE_TILE/2` 推出）。只调度 `uplo` 三角内的 tile：`T = ceil(n/64)`，tile 数 `T(T+1)/2`，`SelectTile` 用三角编号二分把线性 tile 号映射为 `(row, col)`；上三角取 `row ≤ col`，下三角交换行列。每个 tile 由单个 block 独占，跨 block 不做归约，因此每个输出位置只有一个最终写者。整块位于另一三角的 tile 不进入调度，天然不写。

**（3）K 方向流水。** K 按 64 行一个 panel 装载（`PANEL == TILE`，编译期断言），每 2 个 panel（`REDUCTION_SPAN = 128`）构成一个归约跨度：

- AIV 调 `ProducePanel` 把复数输入拆成 3 个 Gauss 变换面：左操作数（`A`）为 `(xr+xi, xr, xi)`，右操作数（同一 `A` 的转置视图）为 `(yr, yi−yr, yr+yi)`；连同 2 个原始分量暂存面，在 UB 中占 5 个面，随后写入 L1 的 ping-pong 阶段（2 级）；
- AIC 调 `MultiplyPanel` 对同一输出 tile 连续做 3 个实数乘（`p0=(xr+xi)·yr`、`p1=xr·(yi−yr)`、`p2=xi·(yr+yi)`），L0C 累加 2 个 panel 后由 `PublishProducts` 回传到 UB 的 6 个输出面；
- AIV 在跨度边界调 `MergeRankK`，按 `real = p0 − p2`、`imag = p0 + p1` 把三个乘积面合成实部/虚部；第一个跨度初始化，后续跨度用 Neumaier 补偿累加；
- 最后一个跨度写回：对角 tile 走 `StoreDiagonal`（128 线程，按 `i ≤ j` / `i ≥ j` 掩码逐元素写），非对角 tile 走 `InterleaveOutput` + `StoreConsumedPart`（整块都在三角内，无需掩码）。

**（4）跨核同步。** 使用 `CrossCoreSetFlag/WaitFlag` 的 mode-4 事件：L1 输入用 `INPUT_FREE/INPUT_READY` 两级，输出面用 `FREE_FLAG/READY_FLAG` 两级，分别绑在 `PIPE_MTE1`、`PIPE_MTE3`、`PIPE_FIX` 上；复用缓冲前必须等消费者释放。kernel 结束时 `DrainPipeline` 回收全部在途事件。单缓冲/双缓冲由阶段数决定，先用现有 2 级 ping-pong 验证正确性。

**（5）尾块。** N 方向：`rows = min(n−row, 64)`、`cols = min(n−col, 64)`，行列按 16 对齐（`paddedRows`），对角写回按 `rows/cols` 截断，padding 列不写。K 方向：最后一个 panel 的 `depth < 64`，`ClearInput` 把缺失行清零，`MultiplyPanel` 只累加 `depth` 行，避免把未初始化数据带入乘积。

**（6）片上预算。** 每个 AIV 的 UB 预算 192 KiB：

| 区域 | 计算 | 大小 |
| --- | --- | --- |
| 输出面 | 2 阶段 × 6 面 × 8 KiB | 96 KiB |
| 输入暂存 | (2 原始分量 + 3 Gauss 变换) × 16 KiB | 80 KiB |
| 补偿面 | 2 × 2048 float | 16 KiB |
| 合计 | — | 192 KiB（`static_assert` 恰好不超） |

L1 每阶段 2 个操作数 × 3 个 Gauss 面 × 16 KiB = 96 KiB，双阶段 192 KiB，低于 512 KiB 上限。不使用 GM workspace。

**（7）当前缺口（必须显式说明）。**

- Host **尚未调用** Cube 入口，实际执行路径仍是 SIMT；
- 设备入口只写 3M 乘积，没有 alpha/beta 融合，也没有 INT8 打包与 Device 标量读取；
- 三角 tile 的写保护、K 尾块、跨核同步只经过代码审查，**没有真机正确性对比**；
- FP32 Gauss 打包在 `(xr+xi)` 上与 SIMT 有同样的吸收问题（§3.2.2 的反例同样成立）；补残差面会引入额外实数矩阵乘、超出 3M 约束，因此 Cube 的适用数据域必须由实测精度界定，未达标形状不得启用；
- 切换阈值、精度、性能均未实测。在这些证据齐备前，Cube 不得作为默认路径，小形状、INT8、Device 标量与非平凡 alpha/beta 继续走 SIMT。

### 3.2.4 非有限值与特殊 IEEE 输入（v2 收窄）

**分流规则。** 只有输入分量本身是 **Inf 或 NaN** 时，该输出元素才转到逐元素直接复数乘；判断用 `__builtin_isinf` / `__builtin_isnan`（或等价的指数位全 1 检查）。不再使用幅值阈值，也不把“部分和或乘积出现非有限值”作为触发条件——那会把有限输入的中间溢出也送去回退。所有**有限**输入——包括 `1e-20`、`1e-30`、次正规数以及大但未溢出的值——一律走 3M，不按量级分流。

当前源码的 `CsyrkUnsafeValue` 用幅值阈值 `0x5a0e1bca`（约 `9×10^15`）与 `0x24e69595`（约 `1×10^-16`）判断，会把 `1e-20` 这类合法有限值判为“不安全”，从而走直接复数乘。这与任务书的 3M 约束不符：v2 要求把判断收窄为仅 Inf/NaN，实现须在代码 PR 中同步修改，并以 §4.2.2 的有限极端值与 `TC_3M_*` 用例证明 **有限输入的回退计数为 0**。

**回退分支行为。** 触发后：

- `CsyrkReferenceElement` 按 K 顺序做**直接复数乘**（`(xr·yr − xi·yi, xr·yi + xi·yr)`，关闭 FP contraction），再做 alpha/beta；
- 复数乘实现带 `__mulsc3` 风格的 Inf/NaN 恢复：两个分量同时为 NaN 且存在无穷时，按符号重建 `∞·有符号量`，避免 `∞−∞` 把可判定的结果吞成 NaN；
- 回退是**逐元素**的，不影响同一 tile 中其他元素继续走 3M，也不新增 launch。

**有限值溢出。** 有限但量级接近 FP32 上限的输入（如 `1e20`）在 3M 中仍可能出现中间乘积溢出，结果按 IEEE 传播为 Inf/NaN；这不是分流分支，而是 FP32 3M 的固有语义。任务书测试数据（A ∈ [-5,5] 或 N(0,1)，|alpha| ≤ 1e4，k ≤ 4096）远低于该量级；若后续要求覆盖，按分量分类单独评估，不引入有限值回退。

**例外范围与证据。** 仅 Inf/NaN 的回退仍是任务书未明列的偏离，需在评审中请任务方确认其范围。若任务方不接受任何回退，则关闭该分支、让 3M 直接传播 Inf/NaN，并按分量分类记录与 golden 的差异；两种方案都必须在 §4.2.2 的 Inf/NaN 用例上给出实测对比。关闭条件：有限输入误触发为 0，且非有限用例的分类差异有书面结论。

### 3.2.5 数值与性能取舍

- **为什么用 3M。** 标准复数 GEMM 需要 4 次实数乘（`XrYr`、`XiYi`、`XrYi`、`XiYr`）；对称 `X·X^T` 下 `XrXi^T` 与 `XiXr^T` **互为转置**，`P2` 展开后同时含这两项，`P2 − P0 − P1` 得到二者之和，因此把 4 次降到 3 次。代价是 `P2 − P0 − P1` 存在消去，必须靠补偿兜底。
- **补偿策略。** SIMT 做三件事：lane 内 Neumaier 求和补偿、warp 内蝶式归约、以及 v2 新增的 `(xr+xi)` 低位补偿（TwoSum + `low`）。三者都不改变 3M 的数学定义，只降低 K 方向累加误差与和运算的吸收误差。Cube 在跨度合并时做求和补偿，但打包阶段的 `(xr+xi)` 吸收无法在不增加矩阵乘的前提下修复，见 §3.2.3(7)。
- **性能瓶颈。** SIMT 每个输出元素要读 `A` 的两行（`2k` 个复数），总访存 `O(n²k)`，且没有片上复用；Cube 通过 64×64 tile 与 K panel 复用把访存降到 `O(n²k/64)` 量级，并用 AIC 承担主要乘加。目标是把 512/1024/2048 方阵从 SIMT 的访存受限区间拉到大核利用率更高的区间。
- **切换原则（计划）。** 在代表形状（`n,k ∈ {64,128,256,512,1024,2048,4096}`，两种 `uplo`、三种 `trans`）上对同一输入比较 SIMT、Cube 与 CPU golden，再用 `msprof op` 的 kernel 时间选阈值；只有正确性与性能同时成立的形状才切 Cube。阈值本身不写死在本设计中，待实测后回填。

### 3.2.6 待接通项与风险

| 项 | 现状 | 关闭条件 |
| --- | --- | --- |
| Cube alpha/beta 融合 | 设备入口只输出 3M 乘积 | 在 AIV 写回阶段加入复数缩放与 beta 累加，并与 SIMT 同输入比对 |
| Cube INT8 打包 | 只按 complex64 装载 | 增加两字节交错的提升面，覆盖 `-128/127` |
| Cube Device 标量 | 入口无 alpha/beta 参数 | 与 SIMT 共用 GM 标量读取 |
| 三角 tile 写保护 | 代码审查通过，未上机 | 对角/非对角 tile 的哨兵测试 |
| K 尾块与 N 尾块 | 代码审查通过，未上机 | `k` 非 64 倍数、`n` 非 64 倍数的对比 |
| 非有限分支范围 | 源码仍按幅值阈值判断，`1e-20` 会误触发 | 收窄为仅 Inf/NaN；有限输入回退计数为 0；任务方确认例外范围 |
| SIMT `(xr+xi)` 低位补偿 | v2 设计新增，未实现 | `TwoSum` + `low` 落地，§4.2.2 反例回归通过，性能影响实测 |
| Cube 打包吸收 | `(xr+xi)` 在 FP32 打包阶段丢位，补残差需额外矩阵乘 | 用实测精度界定适用域；未达标形状保持关闭 |
| 性能阈值 | 未测 | 代表形状的 SIMT/Cube/golden 三方对比 + msprof |

## 3.3 支持硬件

| 芯片版本 | 是否支持 | 说明 |
| --- | --- | --- |
| Ascend 950PR（arch35，DAV-3510） | √ | 本任务目标；`bash build.sh --ops=csyrk3m_ex --soc=ascend950` |
| Atlas A2/A3（arch22） | × | 不在任务范围，不注册 arch22 实现 |
| 其他产品线 | × | 接口声明可共用，但当前只有 arch35 实现 |

## 3.4 算子约束限制

1. **布局**：仅支持列主序稠密矩阵，`A`/`C` 的跨度由 `lda`/`ldc` 描述；不支持任意 stride 视图、广播、非连续 Tensor 或 AutoContiguous 语义。
2. **三角语义**：只读写 `uplo` 指定三角与对角；另一三角、`ldc` padding 及尾部哨兵区必须保持逐字节不变，调用方不得依赖接口去初始化它们。
3. **原地与别名**：`C` 为原地覆写；`A` 与 `C` 的可读写区域不得重叠，接口不做别名检测。
4. **数值一致性**：3M 与直接复数乘的舍入过程不同，结果与 `aclblasCsyrk`/`cublasCsyrk` 不保证逐位一致；按 §4.1 阈值判定。接口也不承诺同一输入两次执行逐位一致（当前 SIMT 路径实际按固定分核执行，但这不是契约）。
5. **异步**：kernel 在 handle 绑定的 stream 上异步执行，接口返回不代表计算完成；Host 读回结果前须同步该 stream。
6. **标量位置**：`alpha`/`beta` 支持 Host 与 Device 指针；Device 标量按同 stream 顺序可见，跨 stream 需调用方自行建链。
7. **资源**：不使用 GM workspace，不申请临时 Device 缓冲。
8. **版本**：仅 arch35 注册，CANN 9.1.0 及以上；低版本不保证可编译或可运行。

# 4. 可维可测分析

## 4.1 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 对 `C` 的 `uplo` 三角（含对角）逐元素、逐分量（实部/虚部）比对：`\|actual−golden\| ≤ atol + rtol·\|golden\|`，`rtol = atol = 2^-13`；用例需同时满足 `matched_ratio ≥ 0.99` 与 `max_abs_error ≤ 1e-2`（或 32×ULP）。golden 为 Netlib 3.10.0 `cblas_csyrk`，INT8 分量精确提升后调用 | 任务书 §3.2；生态算子开源精度标准 |
| 性能标准 | kernel-to-kernel：`msprof op` 采集单算子 `Task Duration(us)`，5 次 warmup 后取 10 次有效样本均值，除以 1000 得 ms；单 case ≤ `gpu_ms`/0.4（即 ≥0.4×），全部正式用例算术平均 ≥0.5×，两项同时满足 | 任务书 §3.3、§3.3.1 |
| 内存标准 | 不涉及 | 任务书 §3.4 |

补充口径：

- 非有限输出（Inf/NaN）按分量的分类（NaN/Inf/有限）与 golden 比对并单独记录，不参与有限值的 `matched_ratio` 统计。
- 未引用区域（另一三角、padding、尾部哨兵）按位比对，必须完全一致；这类检查不计入 `matched_ratio`，单独判定。
- 测试 CSV 的 `mere_threshold`/`mare_multiplier` 只用于测试工程内部诊断，正式结论以任务书 §3.2 为准。

## 4.2 验证方案

### 4.2.1 环境

| 项 | 要求／现状 |
| --- | --- |
| 设备 | Ascend 950PR（arch35），单卡 |
| CANN | ≥ 9.1.0；已验证实例为 9.2.1，新实例以阶段 1 环境核对为准 |
| OS | Ubuntu 22.04.5 LTS |
| Golden | Netlib 3.10.0 CBLAS/LAPACK（项目内构建，避免覆盖系统包）；`cblas_zsyrk` 仅用于定位参考精度域 |
| 构建 | `bash build.sh --ops=csyrk3m_ex --soc=ascend950`；GTest 目标 `csyrk3m_ex_test` |
| 工具 | `msprof op`（正式性能）、`aclrtEvent`（快速回归，不作正式判据） |

### 4.2.2 精度验证

**用例规模。** 测试包共 1200 条，其中精度 1000 条、性能 200 条；精度用例分布为：L0 基础 12、SQ 尺寸 204、AB 标量 48、WS/TH 矩形 24、LD 前导维 27、FL 填充 12、CV 覆盖 48、ED 边界 38、3M 专项 4、EX 扩展 583。INT8 用例 493 条，接近一半。

**必测项。**

1. `uplo × trans` 六组组合，含 `OP_C` 与 `OP_T` 结果一致（同一输入下逐位相同）；
2. 尺寸 0、1、2、3、5、7、16、31/32/33、63/64/65、127/128/129、255/256、511/512、1023/1024、2047/2048、4095/4096；
3. `lda`/`ldc` 紧凑与 `+1/+8/+64` padding；低于下界、0、负数的负向用例；
4. `alpha`/`beta` 特殊值 `(0,0)`、`(1,0)`、`(0,1)`、`±1e4` 与随机值；
5. INT8 极值 `-128/127`、零值、随机交错输入；
6. `TC_3M_*` 四项：交替/极值填充下的实虚部抵消、INT8 极值、对角虚部保留；
7. 退化：`n=0`、`k=0` 且 `beta≠1`、`alpha=0` 且 `A=NaN`、`beta=0` 且旧 `C=NaN`；
8. 写保护：另一三角与 padding 哨兵、尾部 `GUARD_ELEMENTS` 哨兵；
9. 标量：Host/Device 两种位置，Device 标量由同 stream 上游 kernel 写入后再调用，验证 stream 顺序；
10. 空指针：`alpha`、`beta`、`C`、`A`（在需要乘积时）分别置空，核对状态码；
11. 非法枚举与不支持的 `Atype`/`Ctype`，核对 `INVALID_ENUM`/`NOT_SUPPORTED`；
12. `TC_ED_403`（`beta=(0,0)`、`c_fill=VALUE_NORM_NAN`）验证不读旧 C；`TC_ED_404`（`alpha=(0,0)`、`a_fill=VALUE_NORM_NAN`）验证不读 A。

**复审回归项（v2 新增，必须执行）。**

1. **P2 吸收反例**：`n=k=1`、UPPER/N、`A=(1, 1e-8)`、`alpha=(1e4, 0)`、`beta=0`。真值虚部约 `2e-4`；未补偿的 3M 约 `−1e-12`，补偿后应回到 `2e-4` 并通过 §4.1 判据。报告需同时给出补偿前/后的实测值，证明 `low` 生效。
2. **比例扫描**：`|xi/xr| ∈ {1e-4, 1e-5, 1e-6, 1e-7, 1e-8}`，正负号各一，`k ∈ {1, 8, 64, 1024}`，`alpha ∈ {(1,0), (1e4,0)}`；比对实部/虚部分量。
3. **有限小值**：`1e-20`、`1e-30`、`1e-38`、次正规 `1e-40`，确认全部走 3M（回退计数为 0），并与 `cblas_csyrk` 比对。
4. **有限大值**：`1e20`、`1e30`（中间乘积溢出），记录 Inf/NaN 分量分类，确认为 IEEE 传播而非回退；不作为精度通过项。
5. **非有限**：A/C 含 Inf、NaN 的用例，确认只有这些输入触发回退，回退元素数与预期一致。
6. **三角 lane 过滤**：`n` 非 4 倍数、`col ∈ [0,3]` 的行组，分别验证 UPPER/LOWER 的写回完整性与另一三角哨兵；防止按 warp 整组跳过造成漏写。

**补测项（当前未覆盖，需补）。**

- `A=nullptr` 且 Device alpha 由同 stream 上游写入的两支（零值放行、非零拒绝）；CSV 无法表达 Device 标量位置，只能由测试工程装配；
- Host/Device 混合标量（`alpha` Host + `beta` Device、`alpha` Device + `beta` Host），核对两条读取路径不会互相覆盖；
- 非有限分支的触发边界：仅 Inf/NaN；`1e-20`/`1e-30` 等有限值不得触发（并入复审回归项 3）；
- INT8 与 COMPLEX64 在同形状下的结果一致性（提升后应完全相同）；
- Cube 接通后的同输入三方比对（SIMT/Cube/golden），当前不适用。

**判定与记录。** 每个用例记录：`case_name`、实际状态码、实部/虚部分量的 `matched_ratio`、`max_abs_error`、失败样例的前若干个索引。报告绑定源码 commit、CANN 版本、设备型号、测试二进制 SHA256、CSV SHA256（`a356831d…`）与 golden 版本。

### 4.2.3 性能验证

**用例。** 正式性能 200 条，全部为 `ACL_COMPLEX64`；覆盖六组 `uplo × trans`、`n ∈ [1,4096]`、方阵与宽窄矩阵、紧凑 `lda`/`ldc`。测试包 README 曾写“含 2 条 INT8 性能用例”，但实际 CSV 与生成脚本的自检均为 **0 条 INT8**；本设计按实际数据描述：INT8 性能由开发者按任务书 §3.3 第 6 条另行补采，并沿用同参数 COMPLEX64 的 `gpu_ms` 作为参考基线。

**采集。** 逐用例 `msprof op`：

```bash
msprof op --application="./build/test/syrk/csyrk3m_ex/csyrk3m_ex_test --gtest_filter=*TC_PF_1001*" \
          --output=./prof_csyrk3m_ex
```

在 `OPPROF_*/OpBasicInfo*.csv` 中按 kernel 名取 `Task Duration(us)`：warmup 5 次不计，取后续 10 次有效样本的算术平均；执行返回非 `SUCCESS` 的样本剔除并记录。不得用 GTest `[PERF] avg_us` 代替。

**判据。** `ratio = gpu_ms / (npu_mean_us / 1000)`；单 case `ratio ≥ 0.4`，200 条算术平均 `≥ 0.5`。

**补偿开销对照（v2）。** `TwoSum` + `low` 会增加 SIMT 的 ALU 操作。在 512/1024/2048 方阵、UPPER/N 上分别采集补偿开启与关闭的 kernel 时间，确认仍满足 0.4×/0.5×；若代价显著，优先把大形状切到 Cube，但前提是 Cube 精度达标（§3.2.3(7)）。

**重复基线的逐 ID 映射。** `gpu_baseline.csv` 有两组同参数但不同 `gpu_ms` 的记录，快速脚本按参数建字典会覆盖；正式对照按用例 ID 与基线 ID 一一配对：

| 用例 | 基线 ID | `uplo/trans/n/k` | `gpu_ms` |
| --- | --- | --- | --- |
| `TC_PF_1001` | `csyrk3m_ex-base-000` | UPPER/N/512/512 | 0.0453060 |
| `TC_PF_1002` | `csyrk3m_ex-base-001` | UPPER/N/1024/1024 | 0.2543576 |
| `TC_PF_1006` | `csyrk3m_ex-perf-005` | UPPER/N/512/512 | 0.0453154 |
| `TC_PF_1007` | `csyrk3m_ex-perf-006` | UPPER/N/1024/1024 | 0.2549783 |

`verify_performance.py` 只解析 `[PERF] avg_us`（事件计时，含下发开销），用于快速回归与趋势；其字典键覆盖问题使它无法区分上述两组，正式分析必须从原始 `msprof` 样本复算。

**INT8 补测。** 选取与 COMPLEX64 性能用例同形状的 INT8 输入（两字节交错），逐条采集 kernel 时间，标注“参考基线借用同参数 COMPLEX64”，不改变 0.4×/0.5× 判据。

### 4.2.4 内存验证

任务书 §3.4 明确“内存要求：不涉及”，本算子也不申请 GM workspace，因此不设计内存用例。若平台模板仍要求内存项，按“不涉及”填写并引用 §3.4，不伪造数据。

### 4.2.5 回归与报告绑定

顺序：参数与小规格冒烟（`*TC_L0_*`、`*HandlesStructuralErrors*`、`*DeviceScalars*`、`*NontrivialScaleWithNullA*`）→ 单组 `uplo×trans` → 1000 条精度全量 → 200 条性能。任一阶段失败先修真实缺陷，不放宽判据。

报告分别给出：环境基线、代码 commit、测试二进制与 CSV 哈希、逐用例机器可读结果、失败/跳过/缺失清单、复现命令。精度报告与性能报告引用同一代码快照；设计评审只引用本文，不引用工程数据。

## 4.3 兼容性分析

- **接口兼容**：`aclblasCsyrk3mEx` 是新增符号，不改动 `aclblasCsyrk`、`aclblasCgemm` 等既有声明与行为；头文件仍由 `include/cann_ops_blas.h` 统一暴露，不引入私有平行 API。
- **状态码兼容**：沿用 ops-blas 既有状态码；`INVALID_ENUM` 与 cuBLAS `INVALID_VALUE` 的差异属仓内约定，调用方按 ops-blas 语义处理。
- **数据布局兼容**：complex64 与 complex-int8 的布局、`lda`/`ldc` 以复元素计的口径与 cuBLAS `cublasCsyrk3mEx`（`CUDA_C_8I`）一致，便于参数级迁移。
- **数值兼容**：3M 与直接复数乘不保证逐位一致，这是算法定义决定的；与 `aclblasCsyrk`/`aclblasCsyrkEx` 的差异在 §3.4 与 §4.1 已声明。
- **平台兼容**：仅 arch35 注册；在 arch22 或其他产品线上不提供实现，调用应返回不支持或链接失败，不静默走错路径。
- **构建兼容**：单算子构建新增一条 `cgemm_compute_kernel.cpp` 保留规则，只在 `TEST_NAMES` 含 `csyrk3m_ex` 时生效，不影响其他目标的源集合。

## 4.4 证据与当前状态

| 证据 | 内容 | 能说明 | 不能说明 |
| --- | --- | --- | --- |
| 环境基线（旧实例） | Ubuntu 22.04.5 + 950PR + CANN 9.2.1 + Netlib 3.10.0；现有 `csyrk` 在上游 `55ef791` 的 14 条冒烟全通过；独立加法 kernel 的 `msprof op` 采集到 15 份记录，去前 5 次后均值 2.3392 µs | 旧实例的 CANN、设备、Netlib、msprof 链路可用 | **不说明** `aclblasCsyrk3mEx` 的编译、精度或性能；加法 kernel 与 3M 无关 |
| 本地源码快照 | ops-blas `6df9aa04` + 4 个新文件 + 3 处跟踪改动；SHA256 见表头 | 本文实现描述的依据 | 未编译、未运行、未采样 |
| 新增算子验证 | 尚未开始；阶段 1 已整理本地快照与同步包。v2 的“仅 Inf/NaN 分流”与 `TwoSum` 低位补偿目前只是设计，代码未改、未上机 | — | 不能声称任何精度/性能结论，也不能声称反例已修复 |
| 设计评审 | v1（`2be350d0`）复审提出 2 项 P1 与 3 项文档问题；本文为 v2 修订稿 | 接口、算法、验证方案可评审 | 不代表算子通过平台验收，也不代表 P1 已在代码中关闭 |

三者的引用边界必须分开：设计评审引用本文；精度/性能报告引用真机日志与原始 `msprof` 样本；平台验收引用绑定同一 commit 的验收包。旧实例的 `csyrk` 冒烟与加法采样只能作为“环境可用”的旁证，不能折算成本算子的任何指标。

## 4.5 未决问题

1. **任务目录与团队名**：官方 `tasklist` 目前没有本任务的子目录，现有同名 PR 使用了四种命名（`11-aclblasCsyrk3mEx-950`、`11-aclblasCsyrk3mEx_950`、`3d67a5f9…-aclblasCsyrk3mEx-950`、`11月社区任务-aclblasCsyrk3mEx算子开发（950）`）。复审已确认当前写法有同批次先例，保留在 PR #3306 中等待维护者确认；团队目录 `XUPT_2401_88265396` 有同账号历史提交作为依据，本次报名记录仍待核对。
2. **Cube 接通**：alpha/beta 融合、INT8、Device 标量、切换阈值均未完成，见 §3.2.6；FP32 打包的吸收问题使 Cube 的适用域需由实测界定。
3. **非有限分支范围**：v2 已将触发条件收窄为仅 Inf/NaN，但源码仍按幅值阈值判断，需在代码 PR 中同步修改；有限输入回退为 0 待实测，例外范围待任务方确认。
4. **P2 低位补偿**：v2 设计新增了 `TwoSum` + `low` 方案，尚未实现；§4.2.2 的反例回归与 §4.2.3 的开销对照是关闭条件。
5. **INT8 性能**：正式 200 条性能用例全是 COMPLEX64，INT8 需按 §3.3 第 6 条补采。
6. **冻结用例包**：任务书 §4 要求 `cases.json`、`inputs/`、`golden/`、`facts.json`，当前本地只有 CSV 与脚本；提交前需确认官方提供方式。
7. **`msprof op` 平台通道**：讨论帖中有同任务开发者报告 `StartFFTSTask` 失败导致 `Task Duration` 为 `NA`。本设计仍以 `msprof op` 为正式口径；若平台通道不可用，按任务书向社区确认替代采集方式，不擅自改用事件计时。
8. **3M 抵消裕度**：`TC_3M_*` 的实测误差与阈值余量待真机数据回填。

## 4.6 交付物去向

| 交付件 | 去向 | 内容 |
| --- | --- | --- |
| 设计文档 | `cann/cann-ops-competitions` 任务目录（本 PR） | `docs/design.md` |
| 算子实现与算子 README | `cann/ops-blas`：`blas/syrk/arch35/`、`include/cann_ops_blas.h`、`blas/syrk/README.md` | host、tiling、SIMT/Cube kernel、接口声明、支持范围 |
| 测试代码 | `cann/ops-blas`：`test/syrk/csyrk3m_ex/arch35/` | CMake 目标、CSV 驱动 GTest、原始 CSV |
| 正式验收材料 | 任务 IT 系统（提交验收） | 自验证步骤、精度/性能/内存自验报告与日志 |
| 验收材料副本 | 开发者私仓 `task_submission/`（按任务书 §4 目录结构） | `1 自验证步骤说明.md`、`2.1/2.2 精度`、`3.1/3.2 性能`、`4.1/4.2 内存` |
| 权限 | 开发者私仓邀请 `Ascend-CANN` 为开发者 | 任务书 §4 |
| 任务进展 | 任务讨论帖 #88，按报名账号至少每 3 个工作日更新 | 进展与问题 |

内存项按任务书 §3.4 为“不涉及”，验收材料中按模板填“不涉及”并说明原因，不伪造数据。

# 5. 参考链接

1. 任务页：<https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6?menu=guide>
2. 任务讨论帖：<https://gitcode.com/cann/ops-blas/discussions/88>
3. 任务书：`aclblasCsyrk3mEx_Atlas950PR_task_doc.md`（本地随任务下发）
4. 设计模板：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md>
5. 提交说明：<https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md>
6. cuBLAS `cublasCsyrk3mEx`：<https://docs.nvidia.com/cuda/cublas/index.html#cublascsyrk3mex>
7. Netlib `csyrk`：<https://www.netlib.org/blas/csyrk.f>
8. 生态算子开源精度标准：<https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md>
9. ops-blas 仓：<https://gitcode.com/cann/ops-blas>
