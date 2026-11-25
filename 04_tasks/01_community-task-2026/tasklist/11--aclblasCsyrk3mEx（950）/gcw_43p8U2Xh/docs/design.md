# aclblasCsyrk3mEx 算子设计文档

## 需求背景（required）

### 需求来源

CANN 社区任务 2026，任务书《aclblasCsyrk3mEx 算子开发任务书》。目标代码仓为昇腾算子开源仓 `ops-blas`（https://gitcode.com/cann/ops-blas ），实现目录为 `blas/syrk/arch35/`，测试代码合入 `test/syrk/csyrk3m_ex/arch35/`。

本任务为补齐 Blas 仓能力：在昇腾 950PR（arch35）平台上使用 Ascend C 编程语言开发单精度复数对称矩阵秩-k更新（扩展版）算子 `aclblasCsyrk3mEx`，对标 cuBLAS `cublasCsyrk3mEx`，完成算子设计、开发、测试全流程。验收通过后合入昇腾算子开源仓。

### 背景介绍

`aclblasCsyrk3mEx` 是 Level-3 BLAS 中对称秩-k更新算子的扩展版本（Ex），其核心数学公式为：

```text
C = alpha * op(A) * op(A)^T + beta * C
```

其中 `A` 为 n×k（或 k×n）矩阵，`C` 为 n×n 对称复数矩阵。该接口与普通 `csyrk` 的核心区别在于引入了 **3M 算法（对称专用分解）** 与**混合输入类型（Atype/Ctype）** 支持：

1. **3M 算法**：将复数矩阵乘分解为三次实数矩阵乘。相比直接复数乘法，3M 算法能减少复数乘法的计算量（复数乘法需要 4 次实数乘和 2 次实数加，3M 只需 3 次实数乘和若干次实数加），在部分硬件上能获得更好的吞吐。
2. **混合精度/类型**：`Atype` 支持 `ACL_COMPLEX64` 与 `ACL_INT8`（complex-int8，实部/虚部各 1 字节交错存储），`Ctype` 固定为 `ACL_COMPLEX64`。对于 `ACL_INT8` 输入，需精确提升为 float32 后再进行实数矩阵乘。
3. **对称语义**：区别于 Hermitian 语义，对称语义下 `op(A) * op(A)^T` 不使用共轭，且对称矩阵 `C` 对角元素的虚部是有效数据，计算后保留，不强制置 0。`trans = ACLBLAS_OP_C` 在本算子中等价于 `ACLBLAS_OP_T`。

本任务要求 3M 的三次实数矩阵乘必须在**同一次 kernel launch 内完成**，禁止使用三次独立的 GEMM 调用拼接，且不得套用 Hermitian 的 \(X \cdot X^H\) 分解公式。

## 需求分析（required）

### 需求描述

在 Ascend 950PR 上使用 Ascend C 实现 `aclblasCsyrk3mEx`，严格对齐 cuBLAS `cublasCsyrk3mEx` 的接口和参数语义，支持 `ACL_COMPLEX64` 和 `ACL_INT8` 两种 A 输入类型，仅写入 `uplo` 指定的三角区域，在单次 kernel launch 内融合 3M 算法完成计算，满足任务书的精度与性能验收标准。

### 需求拆解

| 编号 | 需求 | 验收方式 |
| --- | --- | --- |
| R1 | 在 `include/cann_ops_blas.h` 新增 `aclblasCsyrk3mEx` 声明，不定义私有平行接口 | 代码审查；头文件比对 |
| R2 | 实现对称语义的 `C = alpha * op(A) * op(A)^T + beta * C`，对角虚部不置 0 | 功能用例；golden 比对 |
| R3 | 支持 Atype ∈ {ACL_COMPLEX64, ACL_INT8}，int8 按交错字节精确提升为 float | int8 专项用例；精度比对 |
| R4 | **3M 算法融合**：在同一次 kernel launch 内计算 P0, P1, P2，不拼接多次 GEMM | 代码审查；性能剖析 |
| R5 | 边界行为：`n=0` no-op；`k=0` 或 `alpha=(0,0)` 时仅做 `C=beta*C`（或 quick return） | 边界用例；状态码断言 |
| R6 | 参数校验及状态码符合 §2.4/§2.5，特别是整数/枚举非法与 unsupported dtype 的区分 | 负向用例 |
| R7 | 精度满足 COMPLEX64 实/虚部分别按 FLOAT32 判定；golden 采用 `cblas_csyrk` | 自测报告；1000 条用例 |
| R8 | 性能满足 kernel-to-kernel 采集口径，单 case ≥0.4× GPU，平均 ≥0.5× GPU | msprof op 采集；自测报告 |

## 详细设计（required）

### 算子分析

#### 数学公式

```text
C = alpha * op(A) * op(A)^T + beta * C
```

**3M 算法展开**：
令 \(X = op(A) = X_r + i \cdot X_i\)（`Atype=ACL_INT8` 时先精确提升），则乘积 \(X \cdot X^T\) 的实部和虚部可表示为：
- \(P_0 = X_r \cdot X_r^T\)
- \(P_1 = X_i \cdot X_i^T\)
- \(P_2 = (X_r + X_i) \cdot (X_r + X_i)^T\)

由此得到：
- 实部 = \(P_0 - P_1\)
- 虚部 = \(P_2 - P_0 - P_1\)

最终结果与 `beta * C` 按复数规则相加并写回 `C` 矩阵的 `uplo` 指定三角区域。

#### 接口定义

在 `include/cann_ops_blas.h` 新增：

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

#### 支持数据类型与形状

- **Atype/Ctype 组合**：
  - `Atype = ACL_COMPLEX64, Ctype = ACL_COMPLEX64`
  - `Atype = ACL_INT8, Ctype = ACL_COMPLEX64`（int8 按 `(int8 real, int8 imag)` 交错存储）
  - 其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
- **维度与约束**：
  - `n ≥ 0, k ≥ 0`。
  - `trans = ACLBLAS_OP_N` 时 `lda ≥ max(1, n)`，否则 `lda ≥ max(1, k)`；`ldc ≥ max(1, n)`。

#### 语义说明

1. **对称性**：`trans = ACLBLAS_OP_C` 等价于 `ACLBLAS_OP_T`，不做共轭。对 `C` 的访问仅限 `uplo` 三角，另一侧不更新。
2. **对角虚部**：对称矩阵 `C` 对角元素的虚部为有效数据，计算后原样保留。
3. **Quick Return**：
   - `n = 0`：直接返回 SUCCESS。
   - `k = 0` 或 `alpha = (0,0)` 时，若 `beta = (1,0)`，返回 SUCCESS 且不修改 C；若 `beta ≠ (1,0)`，仅执行 `C = beta * C`（`uplo` 三角内）。
   - 发生上述 no-op 时，允许 A 指针为 NULL。

### 算子实现

#### Host 侧设计

1. **参数校验（优先级顺序）**：
   - `handle == nullptr` → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
   - `uplo` 和 `trans` 枚举非法 → `ACLBLAS_STATUS_INVALID_ENUM`
   - `n/k < 0`、`lda/ldc` 低于下界 → `ACLBLAS_STATUS_INVALID_VALUE`
   - `Atype/Ctype` 不支持组合 → `ACLBLAS_STATUS_NOT_SUPPORTED`
   - `alpha/beta` 判空，`n>0` 时 `C` 判空
   - **Quick Return 逻辑**在结构体/指针合法性检查通过后执行。

2. **Tiling 策略**：
   - 根据 `n, k, trans, Atype` 及 950PR 架构的核数和 UB 容量生成 tiling。
   - 由于 3M 算法需要同时计算 \(P_0, P_1, P_2\)，Host 侧需划分出三路计算所需的 workspace 或在 UB 内的分块策略。建议将 \(X_r\)、\(X_i\)、\(X_r+X_i\) 作为三路输入，在单次 launch 的 Kernel 内按 panel 或 tile 依次或并发执行三次矩阵乘。
   - `Atype=ACL_INT8` 时，Tiling 需要考虑转换所需的空间；由于要求“精确提升”，Kernel 在读取 int8 数据后会直接 Cast 成 float 参与后续实数乘法。

#### Kernel 侧设计

1. **融合 3M 数据流**：
   - Kernel 内部读取 `A` 的数据。若为 `ACL_INT8`，从 Global Memory 搬入时直接按交错字节解析出实部/虚部，转换为 float。
   - 在 UB 中构建三路矩阵：`Xr`、`Xi`、`X_sum = Xr + Xi`。
   - 利用 Cube 单元或 Vector 单元（950PR特性）实现三次实数矩阵乘 \(P_0, P_1, P_2\) 的计算。必须在同一次 Kernel 下发中调度完成，例如通过循环展开或在同级代码中分配不同的计算块。
2. **Epilogue 计算**：
   - 计算实部：\(Acc_{real} = P_0 - P_1\)
   - 计算虚部：\(Acc_{imag} = P_2 - P_0 - P_1\)
   - 读取原 `C` 的值（如果 `beta ≠ (0,0)`），执行复数的 `alpha * Acc + beta * C` 运算。
   - 写回时，通过掩码控制只写 `uplo` 指定的三角区域（包含对角），对于对角元素，实部正常写，虚部也正常写（不置 0）。
3. **边界处理**：
   - 当 `k = 0` 或 `alpha = (0,0)` 时，跳过 3M 计算，直接执行仅包含 `beta * C` 的轻量 Kernel 路径。
   - 当 `n=0` 时，由 Host 直接返回，不下发 Kernel。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |

实现目录为 `blas/syrk/arch35/`；README 产品支持表标注 Ascend 950PR：支持。

### 算子约束限制

- 仅支持 `Atype` 为 `ACL_COMPLEX64` / `ACL_INT8` 且 `Ctype` 为 `ACL_COMPLEX64`。
- `trans` 仅支持 N/T/C，其中 C 按 T 处理。
- 3M 计算必须在单次 Kernel 内完成，禁止 Host 侧发起三次 GEMM 拼接。
- 不支持广播；不支持超出 `lda/ldc` 语义的任意非连续内存。
- 结果不保证与直接复数乘（如 `aclblasCsyrk`）逐位一致，按精度阈值判定。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | golden 由 CPU 侧非 3M 的 `cblas_csyrk` 生成（int8 精确提升后调用，或使用 `cblas_zsyrk` 保证高精度）。COMPLEX64 实部/虚部分别按 FLOAT32 判定：rtol=atol=2^-13、matched_ratio≥0.99、max_abs_error≤1e-2 或 32 ULP。3M 的中间舍入差异允许在阈值内。 | 任务书 §3.2；混合容差标准 |
| Inf/NaN 规则 | 同标准 2.3 节收窄规则处理非有限值。 | 任务书 §3.2 |
| 性能标准 | 采用 kernel-to-kernel 口径。NPU 侧用 `msprof op` 采集单算子 kernel `Task Duration`（warmup 5 + 有效采样 10 取平均）；对比 `gpu_ms`（GPU cuBLAS kernel-only）。单 case ≤ `gpu_ms` / 0.4，全部 case 平均 ≥ 0.5。不得混用端到端 `[PERF] avg_us`。 | 任务书 §3.3 / §3.3.1 |
| 内存标准 | 不涉及额外内存门禁。 | 任务书 §3.4 |

自验要求：使用 ops-blas 仓 `test/` 目录下 ATK 测试框架，加载 1000 条精度用例与 200 条性能用例。需覆盖：`n`/`k` 尺寸扫描（0, 1, 幂次及非对齐值）、uplo/trans 全覆盖、alpha/beta 特殊值、Atype 包含 COMPLEX64/INT8 及不支持枚举、对称对角虚部非零专项、`TC_3M_*` 极值填充与抵消测试。

### 兼容性分析

- 本算子为 ops-blas 仓内**新增接口**与**新增实现**：在 `include/cann_ops_blas.h` 新增 `aclblasCsyrk3mEx` 声明，在 `blas/syrk/arch35/` 新增实现。
- 接口签名独立，非 Ex 的 `csyrk` 保持原样，`aclblasCsyrkEx`（若存在）互不依赖。
- 测试代码放在 `test/syrk/csyrk3m_ex/arch35/`，独立编译目标，不影响其他算子。
- 遵循 `CONTRIBUTING.md` 规范，PR 合入前确保 CI 与实机测试通过。