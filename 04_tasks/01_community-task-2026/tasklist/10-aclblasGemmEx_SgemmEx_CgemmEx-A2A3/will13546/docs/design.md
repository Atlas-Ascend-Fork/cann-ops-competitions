# 需求背景

## 需求来源

CANN 社区任务 2026 年 10 月「aclblasGemmEx / aclblasSgemmEx / aclblasCgemmEx 算子开发（A2A3）」。
验收通过后合入昇腾算子开源仓 [cann/ops-blas](https://gitcode.com/cann/ops-blas) 的 `blas/gemm_ex/arch22/`。

## 背景介绍

ops-blas 仓的扩展 GEMM 接口现状：

| 接口 | 仓内现状 |
| --- | --- |
| `aclblasGemmEx` | 已有声明（19 参，与 `cublasGemmEx` 同构）与 arch35 实现；Atlas A2/A3 无实现 |
| `aclblasSgemmEx` | 声明为 15 参强类型变体（带 `algo`、无 `Atype/Btype/Ctype`），与 `cublasSgemmEx` 不同构 |
| `aclblasCgemmEx` | 仅有 950PR 的 `int typeA/B/C` 形态（取值见 `aclblasCgemmExType_t`），无 cuBLAS 口径实现 |

`blas/gemm_ex/README.md` 产品支持表中 Atlas A2/A3 标注为「不支持」。本任务补齐 A2/A3（arch22）的三接口，
并把 `aclblasSgemmEx`、`aclblasCgemmEx` 的公共声明对齐到 cuBLAS 口径，同时保证 950PR（arch35）既有实现
不受影响。

## 功能分析

三接口均计算 `C = alpha * op(A) * op(B) + beta * C`，A/B/C 列主序，维度 op(A) 为 m×k、op(B) 为 k×n、C 为 m×n。
主路径 dtype：`aclblasGemmEx` / `aclblasSgemmEx` 为 FLOAT32，`aclblasCgemmEx` 为 COMPLEX64（实部/虚部交错存储）。

# 需求分析

## 需求描述

在 Atlas A2/A3（arch22）上提供三个句柄式接口的实现，参数序列与 cuBLAS 一一对应；接口声明统一放在
`include/cann_ops_blas.h`，与 950PR 共用；`blas/gemm_ex/README.md` 产品支持表将 A2/A3 更新为「支持」。

## 需求拆解

1. `aclblasSgemmEx` 声明由 15 参改为 cuBLAS 的 17 参（`Atype/Btype/Ctype` 取代 `algo`），并同步迁移 950 侧实现与测试；
2. `aclblasCgemmEx` 声明统一为 `aclDataType` 17 参（与 `aclblasGemmEx` 同构、各产品线共用），950 侧入口把类型标签桥接到内部的 `aclblasCgemmExType_t` 编码；
3. arch22 实现 `aclblasGemmEx` / `aclblasSgemmEx`（FLOAT32 主路径）；
4. arch22 实现 `aclblasCgemmEx`（COMPLEX64 主路径）；
5. 三套 CSV 驱动测试与 README，覆盖任务提供的 1200 条用例与性能用例。

# 详细设计

## 算子分析

### 数学公式

```
C = alpha * op(A) * op(B) + beta * C
op(X) = X        (transA/transB = ACLBLAS_OP_N)
op(X) = X^T      (ACLBLAS_OP_T)
op(X) = X^H      (ACLBLAS_OP_C，实数路径等价于 T)
```

### 支持数据类型

| 接口 | Atype / Btype / Ctype | 计算 | 说明 |
| --- | --- | --- | --- |
| aclblasGemmEx | ACL_FLOAT | COMPUTE_32F | 其余 dtype / computeType 返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| aclblasSgemmEx | ACL_FLOAT | FP32 | 同上 |
| aclblasCgemmEx | ACL_COMPLEX64 | FP32（实/虚分别累加） | 复数按实部/虚部交错 |

### 支持形状

m、n、k ≥ 0；列主序；前导维 `lda ≥ max(1, m|k)`、`ldb ≥ max(1, k|n)`、`ldc ≥ max(1, m)`（按转置标志取对应维度）。
m = 0 或 n = 0 为合法 no-op；k = 0 或 alpha = 0 退化为 `C = beta * C`。

## 算子实现

### 实现方案

#### 3.2.1 host 侧设计：

**参数校验（三个接口一致，先于 no-op 判定）**

1. `handle` 为空 → `HANDLE_IS_NULLPTR`；
2. `transA/transB` 不在 {N, T, C} → `INVALID_ENUM`；
3. m/n/k 为负、前导维不足 → `INVALID_VALUE`；
4. dtype 不支持、`computeType`/`algo` 不支持 → `NOT_SUPPORTED`；
5. `alpha/beta` 为空 → `INVALID_VALUE`；`k > 0` 且 alpha 非零时 A/B 为空 → `INVALID_VALUE`；
   `beta` 非零时 C 为空 → `INVALID_VALUE`，`beta == 0` 时 C 允许为空（不读不写）。

**实数路径（GemmEx / SgemmEx）**

两个接口只差参数形态，归一化后直接复用 arch22 已有的 `aclblasSgemm`：由它完成 tiling、分核、L1/L0 搬运与
fixpipe 写回，因此本任务不新增 kernel。C 为空且 beta == 0 时直接返回成功，避免 SGEMM 对空指针的强校验。

**复数路径（CgemmEx）**

arch22 没有复数 cube 路径，按「拆平面 → 四次实数 GEMM → 复数向量收尾」实现：

1. **拆平面**：`aclrtMemcpy2D` 以源 pitch 8 字节、目的 pitch 4 字节、宽度 4 字节、高度 = 元素个数，
   把交错的 A、B 分别拆成实部平面与虚部平面（纯 Device 间搬运，不新增 kernel）；
2. **四次实数 GEMM**：记 sA = (transA == C) ? -1 : 1，sB 同理，则
   `Re(P) = Ar·Br + (−sA·sB)·Ai·Bi`、`Im(P) = sB·Ar·Bi + sA·Ai·Br`；
   将 P 视为列宽 2n 的实数矩阵（`ldc' = 2·ldc`），实部写入偶数列、虚部写入奇数列（`P + 1`），
   四次 `aclblasSgemm` 用 alpha = ±1、beta = 0/1 累加，共轭符号折进 alpha，无需额外取反或合并 kernel；
3. **收尾**：`aclblasCscal(C, beta)` 先做 `C = beta·C`（beta = 0 时同时清零），再
   `aclblasCaxpy(P, alpha, C)` 完成 `C = alpha·P + beta·C`，两次调用均作用在交错的 C 上，前导维与 P 一致。

#### 3.2.2 kernel 侧设计：

本期不新增设备 kernel：

- 实数路径复用 `blas/gemm/arch22` 的 SGEMM cube kernel（含 host 分块、多核划分、直接写回 C 的 fast path）；
- 复数路径复用上述 SGEMM kernel 四次，以及 `blas/scal/arch22`、`blas/axpy/arch22` 的复数向量 kernel。

后续优化方向（不影响接口与精度）：Karatsuba 三乘法把四次 GEMM 降到三次；把拆平面与收尾合并为单个 AIV kernel，
减少两次全矩阵遍历。

#### 分核与内存

- 分核沿用 `aclblasSgemm` 的 host 侧切块（`hostTileM/hostTileN`）与 cube 核数选取，不引入新的调度；
- 复数路径的临时内存：两块实平面（A、B 各 2 份）+ 一块 2×m×n 的乘积缓冲，均在调用内申请、返回前释放，
  不留常驻 Device 缓存。

## 支持硬件

- Atlas A2 训练/推理系列、Atlas A3 训练/推理系列（arch22）：本任务新增；
- Ascend 950PR / 950DT（arch35）：保持原有实现，`aclblasCgemmEx` 继续使用 `int typeA/B/C` 形态。

三个接口使用同一份公共声明，不定义产品私有平行接口：`aclblasCgemmEx` 的 `aclDataType` 参数在 arch35 入口
经 `FromAclDataType()` 映射为内部元素类别（ACL_FLOAT→R_32、ACL_COMPLEX64→C_32、ACL_FLOAT16→H_R_32），
其余产品线在 arch22 内直接按 `ACL_FLOAT` / `ACL_COMPLEX64` 处理。

## 算子约束限制

- 仅列主序；不支持非连续 Tensor、广播、dynamic shape 之外的扩展语义；
- `aclblasGemmEx` 的 `computeType` 仅支持 `ACLBLAS_COMPUTE_32F`、`algo` 仅支持 `ACLBLAS_GEMM_DEFAULT`；
- `aclblasCgemmEx` 仅支持 `ACL_COMPLEX64`；
- 参数校验先于零维 quick return：m = 0 / n = 0 时标量与数组指针仍须满足非空约束。

# 可维可测分析

## 精度标准/性能标准

**精度**（任务书 §3.2）：FLOAT32 档 `rtol = atol = 2^-13`、`matched_ratio ≥ 0.99`、`max_abs_error ≤ max(1e-2, 32·ULP)`；
COMPLEX64 按实部、虚部分别判定；Inf/NaN 按任务书收窄规则；alpha = 0 且 beta 参与时结果为位精确缩放。
golden 由 Netlib BLAS 3.10.0 的 `cblas_sgemm` / `cblas_cgemm` 生成。

**性能**（任务书 §3.3）：`msprof op` 采单算子 task duration，5 次 warmup + 10 次有效采样取平均，单 case ≥ 0.8× GPU 标杆。

**测试**：三个 CSV 各 1200 条（1000 精度 + 200 性能），负向用例以 `expect_result` 判定返回码，`handle_null` 由 description 标记。

## 兼容性分析

- `aclblasSgemmEx` 由 15 参改为 17 参属于公共声明变更：950 侧 `blas/gemm_ex/arch35/sgemm_ex_host.cpp`
  与其测试 wrapper 同步迁移，行为（FLOAT32 主路径）不变；
- `aclblasCgemmEx` 统一为 `aclDataType` 声明后，950 侧 `blas/gemm/arch35/cgemm_ex_host.cpp` 在入口做类型桥接，
  其 `aclblasCgemmExType_t` 内部语义（含半精度复数）不受影响；
- `blas/CMakeLists.txt` 将三个接口登记为 arch22 GEMM host/kernel 的依赖，保证单算子构建（`--ops`）链路完整；
- 其余算子与共享缓存代码未改动。
