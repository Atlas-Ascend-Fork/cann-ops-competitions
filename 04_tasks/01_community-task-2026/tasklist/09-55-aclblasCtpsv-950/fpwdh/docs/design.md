# aclblasCtpsv 算子设计文档

> **任务来源**：CANN 社区任务 2026 — aclblasCtpsv 950 算子开发（Atlas 950PR / arch35）
> **目标仓库**：[gitcode.com/cann/ops-blas](https://gitcode.com/cann/ops-blas)
> **源码目录**：`blas/tpsv/arch35/`（与同族 `aclblasStpsv` 同目录）
> **测试目录**：`test/tpsv/ctpsv/arch35/`
> **对标接口**：cuBLAS `cublasCtpsv`（数学语义参考 Netlib [ctpsv.f](https://www.netlib.org/blas/ctpsv.f)）

---

# 1. 需求背景（required）

## 1.1 需求来源

昇腾 CANN 社区任务 2026（`04_tasks/01_community-task-2026`）— 算子实操工坊杭州站，任务编号 #6：在 Ascend 950PR 上使用 Ascend C 编程语言开发单精度复数（COMPLEX64）三角压缩存储（packed）求解算子 `aclblasCtpsv`，并合入 ops-blas 开源仓。

## 1.2 背景介绍

### 1.2.1 aclblasCtpsv 算子功能

`tpsv`（Triangular Packed matrix Solve）求解三角线性方程组：

```
op(A) · x = b
```

其中：
- `A` 是 `n × n` 上/下三角矩阵，以 **packed（压缩）** 格式存储，仅存 `n(n+1)/2` 个元素；
- `op(A)` 由 `trans` 决定：`A` / `A^T` / `A^H`；
- `x` 入口时存放右端向量 `b`，出口时**原地覆写**为解向量 `x`。

`aclblasCtpsv` 是 `tpsv` 的 **单精度复数（COMPLEX64）** 版本，与 cuBLAS `cublasCtpsv` 语义一致。

### 1.2.2 ops-blas 仓现状分析

| 已有资源 | 位置 | 说明 |
|---------|------|------|
| `aclblasStpsv` arch35 实数版 | `blas/tpsv/arch35/` | 完整参考实现，含 scalar + SIMT VF 双路径 |
| `aclblasComplex` 类型定义 | `include/cann_ops_blas_common.h` | POD struct `{float real; float imag;}` |
| Host 侧复数运算符重载 | `blas/common/helper/complex.h` | `+ - * / += -= *= /= == !=` 全套 |
| Kernel 常量 | `blas/common/helper/kernel_constant.h` | `BUFFER_NUM=2`, `UB_SIZE=248KB`, `SIMT_MIN=128`, `SIMT_MAX=2048` |
| Handle 内部结构 | `blas/common/helper/aclblas_handle_internal.h` | `_aclblas_handle { stream, workspace, ... }` |
| Host 工具 | `blas/common/helper/host_utils.h` | `CHECK_RET`, `CeilDiv`, `CeilAlign`, `GetAivCoreCount` |
| arch35 复数算子参考 | `blas/herk/arch35/cherk_*.{cpp,h}` | 采用"分离实虚 + Deinterleave/Combine"策略 |
| ST 测试框架 | `test/frame/*.h` | `csv_loader` / `blas_test` / `fill` / `verify` / `types` / `data` / `device` |
| Tpsv README | `blas/tpsv/README.md` | 产品支持表 / 参数说明 / 调用示例模板 |

**缺口**：
- `include/cann_ops_blas.h` 中 **无 `aclblasCtpsv` 声明**（需新增）；
- `blas/tpsv/arch35/` 下 **无复数版实现**（需新增 5 个文件）；
- `test/tpsv/ctpsv/arch35/` 目录 **不存在**（需新建）；
- `blas/tpsv/README.md` 的接口列表 **无 aclblasCtpsv**（需追加）。

### 1.2.3 arch35 复数算子可行性

同 arch35 上已有 `aclblasCherk`（复数 Hermitian rank-k update）通过 SIMT/AIV 混合流水实现，验证了 **Ascend 950PR 支持 complex64 数据类型**。cherk 采用"分离实虚 → 4M 分解 GEMM → Combine"策略，属于 BLAS-3 矩阵类算子；**ctpsv 是 BLAS-2 三角求解，串行依赖强，不适用 GEMM 分解**，需要独立设计。

## 1.3 与 aclblasStpsv（实数版）的差异

| 维度 | Stpsv（实数） | Ctpsv（复数） |
|------|---------------|---------------|
| 元素类型 | `float`（4B） | `aclblasComplex`（8B = 2×float） |
| `trans` 有效值 | OP_N / OP_T / OP_C（**OP_C 与 OP_T 等价**） | OP_N / OP_T / OP_C（**OP_C 与 OP_T 语义不同**） |
| 模板组合数 | 2 uplo × 2 trans × 2 diag = **8 组** | 2 uplo × **3** trans × 2 diag = **12 组** |
| 计算基本操作 | 实数加减乘除 | 复数加减乘除 + 共轭 |
| 内存占用 | 每元素 4B | 每元素 8B（翻倍） |
| SIMT partialSums | `__ubuf__ float[2048]` | `__ubuf__ float partialRe[2048] + partialIm[2048]`（分离实虚） |
| Packed 索引 | 以 `float` 为单位 | 以 `aclblasComplex` 为单位（**索引公式相同**） |

---

# 2. 需求分析（required）

## 2.1 需求描述

使用 Ascend C 编程语言在 ops-blas 仓 `blas/tpsv/arch35/` 下实现 `aclblasCtpsv`：
1. 支持 COMPLEX64 数据类型（实部/虚部各 float32）；
2. 支持 12 组枚举组合（UPLO × TRANS × DIAG）；
3. 支持 n ≥ 0（n=0 合法 no-op）；
4. 支持 incx ≠ 0（可正可负）；
5. 精度满足 FLOAT32 生态算子开源标准（rtol=2⁻¹⁰, atol=2⁻¹⁶, matched_ratio ≥ 0.99）；
6. 性能满足 4 条典型 case 的标杆耗时（ratio ≥ 0.4）；
7. 头文件 `include/cann_ops_blas.h` 中新增声明，禁止定义 950PR 私有平行接口。

## 2.2 需求拆解

| 编号 | 需求 | 交付物 |
|-----|------|--------|
| R1 | 接口签名与 Stpsv 逐参数对应，元素类型扩展为 aclblasComplex | `cann_ops_blas.h` patch |
| R2 | 参数合法性检查（handle/uplo/trans/diag/n/incx/AP/x） | `ctpsv_host.cpp` |
| R3 | n=0 合法 no-op | `ctpsv_host.cpp` |
| R4 | 12 组枚举全覆盖 | `ctpsv_kernel.cpp` + `ctpsv_kernel_simt.cpp` 模板 |
| R5 | Packed 索引 0-based 换算 | `ctpsv_kernel_utils.h` |
| R6 | UNIT diag 时对角元不访问 | kernel 模板 `if constexpr` 分支 |
| R7 | NON_UNIT 时对角元从 AP 读取，不做奇异检测 | kernel 模板 `if constexpr` 分支 |
| R8 | OP_T / OP_C 严格区分（复数下 A^T ≠ A^H） | kernel 模板参数扩到 3 值 enum |
| R9 | incx < 0 时 x 反向存储 | host 侧 x 指针偏移 + kernel 索引换算 |
| R10 | 双路径：n < 128 scalar，n ≥ 128 SIMT VF | host 侧 tiling 分派 |
| R11 | 精度对齐 cuBLAS + Netlib ctpsv | 测试 golden 用 CPU 复数逐元素求解 |
| R12 | 性能 ratio ≥ 0.4 vs H100 基线 × 2.5 | SIMT VF tree reduction 并行度 |
| R13 | ST 测试工程符合仓内规范 | `test/tpsv/ctpsv/{param.h,golden.h,arch35/{npu_wrapper.h,test.cpp,test.csv,CMakeLists.txt}}` |
| R14 | README 产品支持表标注 Ascend 950PR：支持 | `blas/tpsv/README.md` 追加 |

---

# 3. 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

给定：
- `A ∈ C^(n×n)` 三角矩阵（上/下），packed 存储 `n(n+1)/2` 个复数元素；
- `b ∈ C^n` 右端向量（存于 `x` 入口）；
- `uplo ∈ {UPPER, LOWER}`，`trans ∈ {OP_N, OP_T, OP_C}`，`diag ∈ {UNIT, NON_UNIT}`。

求解 `x ∈ C^n` 使得 `op(A) · x = b`，其中：

```
op(A) = A           if trans == OP_N
op(A) = A^T         if trans == OP_T   （转置：A^T[i,j] = A[j,i]）
op(A) = A^H         if trans == OP_C   （共轭转置：A^H[i,j] = conj(A[j,i])）
```

**求解算法**（前向/后向代入）：

```
if (UPLO=LOWER 且 OP_N) 或 (UPLO=UPPER 且 OP_T/OP_C):
    # 前向代入：从 row=0 到 n-1
    for row in [0, n):
        sum = b[row]
        for j in [0, row):
            sum -= op(A)[row, j] * x[j]
        if diag == NON_UNIT:
            sum /= op(A)[row, row]
        x[row] = sum
else:
    # 后向代入：从 row=n-1 到 0
    for row in [n-1, -1, -1]:
        sum = b[row]
        for j in (row, n-1]:
            sum -= op(A)[row, j] * x[j]
        if diag == NON_UNIT:
            sum /= op(A)[row, row]
        x[row] = sum
```

**复数除法**（`a / b`，b ≠ 0）：

```
denom = b.real² + b.imag²
result.real = (a.real·b.real + a.imag·b.imag) / denom
result.imag = (a.imag·b.real - a.real·b.imag) / denom
```

**复数乘法**（`a · b`）：

```
result.real = a.real·b.real - a.imag·b.imag
result.imag = a.real·b.imag + a.imag·b.real
```

**共轭**（`conj(a)`）：

```
result.real =  a.real
result.imag = -a.imag
```

### 3.1.2 Packed 索引公式（0-based，以元素为单位）

| uplo | 元素 A(i,j) 索引 | 约束 |
|------|-----------------|------|
| UPPER | `i + j·(j+1)/2` | i ≤ j |
| LOWER | `i + (2n - j - 1)·j/2` | i ≥ j |

**⚠️ 任务书 §2.1 LOWER 公式 typo 说明**：

任务书原文写作 `AP[i + ((2*n-j+1)*j)/2]`，实测校验（n=3, LOWER）：

| (i,j) | 期望索引 | 任务书公式 | 仓内 stpsv_kernel_utils.h 公式 |
|-------|---------|-----------|------------------------------|
| (0,0) | 0 | 0 ✓ | 0 ✓ |
| (1,0) | 1 | 1 ✓ | 1 ✓ |
| (2,0) | 2 | 2 ✓ | 2 ✓ |
| (1,1) | 3 | **4 ✗** | 3 ✓ |
| (2,1) | 4 | **5 ✗** | 4 ✓ |
| (2,2) | 5 | **7 ✗** | 5 ✓ |

**结论**：任务书 LOWER 公式 `(2n-j+1)` 应为 `(2n-j-1)`。本设计按仓内 `stpsv_kernel_utils.h` 的正确公式（与 Netlib ctpsv.f、cuBLAS ctpsv 一致）实现。README 中同步标注正确公式。

### 3.1.3 支持数据类型

| 参数 | 数据类型 | dtype | 说明 |
|-----|---------|-------|------|
| AP | tensor | COMPLEX64 | packed 三角矩阵，`n(n+1)/2` 个元素，Device 内存 |
| x | tensor | COMPLEX64 | 右端向量 b / 解向量 x（原地覆写），`1 + (n-1)·|incx|` 个元素，Device 内存 |
| handle | scalar | aclblasHandle_t | ops-blas 上下文句柄，Host 内存 |
| uplo / trans / diag | attr | int（枚举） | Host 内存 |
| n / incx | scalar | int | Host 内存 |

### 3.1.4 支持形状

- `n ≥ 0`（n=0 合法 no-op，直接返回 SUCCESS）
- `incx ≠ 0`（可正可负，|incx| 无上界）
- AP 长度：`n(n+1)/2` 复数元素
- x 长度：`1 + (n-1)·|incx|` 复数元素

## 3.2 算子实现

### 3.2.0 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  include/cann_ops_blas.h                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  aclblasStatus_t aclblasCtpsv(handle, uplo, trans, diag,  │  │
│  │                                n, AP, x, incx);           │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  blas/tpsv/arch35/ctpsv_host.cpp                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. handle 空指针检查 → HANDLE_IS_NULLPTR                 │  │
│  │  2. n < 0 检查 → INVALID_VALUE                            │  │
│  │  3. n == 0 → SUCCESS (no-op)                              │  │
│  │  4. uplo/trans/diag/incx/AP/x 合法性 → INVALID_VALUE      │  │
│  │  5. tiling 组装 (CtpsvTilingData)                         │  │
│  │  6. numThreads 决策：n < 128 → 0 (scalar)                 │  │
│  │                        n ≥ 128 → min(CeilAlign(n,128),2048)│  │
│  │  7. ctpsv_kernel_do(tiling, stream)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  blas/tpsv/arch35/ctpsv_kernel.cpp (dispatcher)                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  extern "C" __global__ __aicore__ ctpsv_kernel(GM_ADDR)   │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  if (tiling.numThreads > 0) → SIMT VF 路径          │  │  │
│  │  │    (ctpsv_kernel_simt.cpp: asc_vf_call<12 组模板>)  │  │  │
│  │  │  else → Scalar 路径                                  │  │  │
│  │  │    (12 组模板 switch-case)                          │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2.1 Host 侧设计

#### 3.2.1.1 接口签名

`include/cann_ops_blas.h` 中新增声明（插入位置：`aclblasStpsv` 与 `aclblasStpmv` 之间）：

```cpp
/**
 * @brief 单精度复数三角 packed 矩阵求解 (COMPLEX64)
 *
 * 求解 op(A) · x = b，其中 A 为 n×n 上/下三角矩阵，以 packed 格式存储；
 * x 入口时存放 b，出口时原地覆写为解向量 x。
 *
 * @param handle  ops-blas 句柄，携带 stream；nullptr 返回 HANDLE_IS_NULLPTR
 * @param uplo    ACLBLAS_UPPER / ACLBLAS_LOWER
 * @param trans   ACLBLAS_OP_N (A) / ACLBLAS_OP_T (A^T) / ACLBLAS_OP_C (A^H)
 * @param diag    ACLBLAS_NON_UNIT (对角从 AP 读) / ACLBLAS_UNIT (对角恒为 1)
 * @param n       矩阵阶数，n ≥ 0；n=0 为合法 no-op
 * @param AP      packed 三角矩阵，n(n+1)/2 个 aclblasComplex，Device 内存
 * @param x       入口 b、出口解 x（原地覆写），Device 内存
 * @param incx    x 步长，incx ≠ 0（可正可负）
 * @return ACLBLAS_STATUS_SUCCESS / INVALID_VALUE / HANDLE_IS_NULLPTR
 */
aclblasStatus_t aclblasCtpsv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    const aclblasComplex* AP,
    aclblasComplex* x,
    int incx);
```

#### 3.2.1.2 参数验证矩阵

| 检查项 | 条件 | 返回值 |
|-------|------|--------|
| handle 空 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n 负 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| n 零 | `n == 0` | `ACLBLAS_STATUS_SUCCESS`（no-op，不再做后续检查） |
| uplo 非法 | `uplo ∉ {UPPER, LOWER}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| trans 非法 | `trans ∉ {OP_N, OP_T, OP_C}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| diag 非法 | `diag ∉ {NON_UNIT, UNIT}` | `ACLBLAS_STATUS_INVALID_VALUE` |
| incx 零 | `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| AP 空 | `n > 0 && AP == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| x 空 | `n > 0 && x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |

**检查顺序**：handle → n<0 → n==0 短路 → uplo → trans → diag → incx → AP → x（与 stpsv_host.cpp 完全一致）。

#### 3.2.1.3 Tiling 策略

**分派阈值**：`SIMT_THRESHOLD = 128`（与 stpsv 一致）。

```cpp
uint32_t numThreads = 0;
if (n >= SIMT_THRESHOLD) {
    numThreads = std::min(
        CeilAlign<uint32_t>(n, SIMT_MIN_THREAD_NUM),  // CeilAlign(n, 128)
        SIMT_MAX_THREAD_NUM                            // 2048
    );
}
```

**分派结果**：
- `numThreads == 0` → scalar 路径（单核 AIV 串行）
- `numThreads > 0` → SIMT VF 路径（多核 AIV + tree reduction）

**incx 处理**：Host 侧不做 x 指针偏移，将 incx 原值传给 kernel，kernel 侧用 `XOffset(j) = (incx > 0 ? j : (n-1-j)) * |incx|` 计算偏移（与 stpsv 一致）。

**Workspace**：本算子无需额外 workspace（AP/x 已在 Device 内存，SIMT VF partialSums 在 UB），跳过 `EnsureDefaultWorkspace` 调用。

#### 3.2.1.4 数据结构（CtpsvTilingData）

```cpp
// blas/tpsv/arch35/ctpsv_tiling_data.h
#pragma once
#include <cstdint>

struct CtpsvTilingData {
    uint64_t ap;          // Device 指针（aclblasComplex*）
    uint64_t x;           // Device 指针（aclblasComplex*）
    uint32_t n;           // 矩阵阶数
    uint32_t uplo;        // ACLBLAS_UPPER (121) / ACLBLAS_LOWER (122)
    uint32_t trans;       // ACLBLAS_OP_N (111) / OP_T (112) / OP_C (113)
    uint32_t diag;        // ACLBLAS_NON_UNIT (131) / UNIT (132)
    int64_t  incx;        // x 步长（可正可负）
    uint32_t numThreads;  // 0 = scalar, >0 = SIMT VF 线程数
};
```

**注**：结构与 `StpsvTilingData` 字段完全一致（都是 POD），仅名称不同以支持双算子共存。

#### 3.2.1.5 Kernel 分派入口

```cpp
// ctpsv_host.cpp 内部
void ctpsv_kernel_do(const CtpsvTilingData& tiling, void* stream);
```

对应 kernel 侧 `ctpsv_kernel.cpp` 里 `extern "C" __global__ __aicore__ void ctpsv_kernel(GM_ADDR tilingGM)`，通过 Ascend C 标准的 kernel launch 宏展开。

### 3.2.2 Kernel 侧设计

#### 3.2.2.1 复数运算 inline 函数（kernel 侧）

**决策**：kernel 侧**手写** `__aicore__ inline` 复数运算函数，**不 include** `blas/common/helper/complex.h`（该头文件的运算符重载在 `#ifdef __cplusplus` 内是 host inline，kernel 编译期是否能内联不确定，保守起见手写）。

```cpp
// blas/tpsv/arch35/ctpsv_kernel_utils.h
#pragma once
#include <cstdint>
#include "cann_ops_blas_common.h"   // aclblasComplex 定义

// ===== Packed 索引（0-based，以 aclblasComplex 为单位） =====
__aicore__ inline uint32_t CtpsvPackedUpperIdx(uint32_t i, uint32_t j) {
    return i + j * (j + 1) / 2;
}
__aicore__ inline uint32_t CtpsvPackedLowerIdx(uint32_t i, uint32_t j, uint32_t n) {
    return i + (2 * n - j - 1) * j / 2;
}

// ===== 复数运算（kernel 侧） =====
__aicore__ inline aclblasComplex CtpsvCAdd(aclblasComplex a, aclblasComplex b) {
    return {a.real + b.real, a.imag + b.imag};
}
__aicore__ inline aclblasComplex CtpsvCSub(aclblasComplex a, aclblasComplex b) {
    return {a.real - b.real, a.imag - b.imag};
}
__aicore__ inline aclblasComplex CtpsvCMul(aclblasComplex a, aclblasComplex b) {
    return {a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real};
}
__aicore__ inline aclblasComplex CtpsvCDiv(aclblasComplex a, aclblasComplex b) {
    float denom = b.real * b.real + b.imag * b.imag;
    // NON_UNIT 时对角元由调用方保证非零，此处不做奇异检测（与 stpsv 一致）
    return {(a.real * b.real + a.imag * b.imag) / denom,
            (a.imag * b.real - a.real * b.imag) / denom};
}
__aicore__ inline aclblasComplex CtpsvCConj(aclblasComplex a) {
    return {a.real, -a.imag};
}

// ===== SIMT VF 版本（同名加 Simt 后缀） =====
__simt_callee__ inline uint32_t CtpsvPackedUpperIdxSimt(uint32_t i, uint32_t j) {
    return i + j * (j + 1) / 2;
}
__simt_callee__ inline uint32_t CtpsvPackedLowerIdxSimt(uint32_t i, uint32_t j, uint32_t n) {
    return i + (2 * n - j - 1) * j / 2;
}
// 复数运算 SIMT 版本同上（略）
```

#### 3.2.2.2 模板参数与 12 组展开

```cpp
enum class CtpsvUplo  { UPPER, LOWER };
enum class CtpsvTrans { NO_TRANS, TRANS, CONJ_TRANS };   // 比实数版多一个 CONJ_TRANS
enum class CtpsvDiag  { UNIT, NON_UNIT };

template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
class CtpsvKernel { ... };
```

**12 组模板实例化**：

| # | UPLO | TRANS | DIAG | kForward |
|---|------|-------|------|---------|
| 1 | UPPER | NO_TRANS | NON_UNIT | 后向 |
| 2 | UPPER | NO_TRANS | UNIT | 后向 |
| 3 | UPPER | TRANS | NON_UNIT | 前向 |
| 4 | UPPER | TRANS | UNIT | 前向 |
| 5 | UPPER | CONJ_TRANS | NON_UNIT | 前向 |
| 6 | UPPER | CONJ_TRANS | UNIT | 前向 |
| 7 | LOWER | NO_TRANS | NON_UNIT | 前向 |
| 8 | LOWER | NO_TRANS | UNIT | 前向 |
| 9 | LOWER | TRANS | NON_UNIT | 后向 |
| 10 | LOWER | TRANS | UNIT | 后向 |
| 11 | LOWER | CONJ_TRANS | NON_UNIT | 后向 |
| 12 | LOWER | CONJ_TRANS | UNIT | 后向 |

**kForward 判定**（复数版扩展）：

```cpp
static constexpr bool kForward =
    (UPLO == CtpsvUplo::LOWER && TRANS == CtpsvTrans::NO_TRANS) ||
    (UPLO == CtpsvUplo::UPPER && (TRANS == CtpsvTrans::TRANS || TRANS == CtpsvTrans::CONJ_TRANS));
```

#### 3.2.2.3 元素读取（含 CONJ_TRANS 处理）

```cpp
// 从 AP 读取 op(A)[row, j]（考虑 uplo/trans/conj）
__aicore__ inline aclblasComplex GetOpElem(uint32_t row, uint32_t j) {
    aclblasComplex v;
    if constexpr (UPLO == CtpsvUplo::UPPER) {
        if constexpr (TRANS == CtpsvTrans::NO_TRANS) {
            // op(A)[row,j] = A[row,j]， Upper 存 (i ≤ j)，此处需要 row ≤ j
            // 但代入求解中 row 是当前行，j < row（前向）或 j > row（后向）
            // UPPER + NO_TRANS 是后向代入，j > row → A[row,j] 存于 packed
            v = apGM.GetValue(CtpsvPackedUpperIdx(row, j));
        } else {
            // op(A)[row,j] = A[j,row] (TRANS) 或 conj(A[j,row]) (CONJ_TRANS)
            // UPPER 存 i ≤ j，此处 j ≤ row → A[j,row] 存在
            v = apGM.GetValue(CtpsvPackedUpperIdx(j, row));
            if constexpr (TRANS == CtpsvTrans::CONJ_TRANS) v = CtpsvCConj(v);
        }
    } else {  // LOWER
        if constexpr (TRANS == CtpsvTrans::NO_TRANS) {
            // LOWER 存 i ≥ j，前向代入 j < row → A[row,j] 存在
            v = apGM.GetValue(CtpsvPackedLowerIdx(row, j, n));
        } else {
            // op(A)[row,j] = A[j,row] 或 conj(A[j,row])
            // LOWER 存 i ≥ j，此处 j ≥ row → A[j,row] 存在
            v = apGM.GetValue(CtpsvPackedLowerIdx(j, row, n));
            if constexpr (TRANS == CtpsvTrans::CONJ_TRANS) v = CtpsvCConj(v);
        }
    }
    return v;
}

// 对角元 op(A)[row, row]
__aicore__ inline aclblasComplex GetOpDiag(uint32_t row) {
    if constexpr (DIAG == CtpsvDiag::UNIT) {
        return {1.0f, 0.0f};   // UNIT 时恒为 (1, 0)，不访问 AP
    } else {
        aclblasComplex v;
        if constexpr (UPLO == CtpsvUplo::UPPER) {
            v = apGM.GetValue(CtpsvPackedUpperIdx(row, row));
        } else {
            v = apGM.GetValue(CtpsvPackedLowerIdx(row, row, n));
        }
        // 对角元共轭：OP_C 时 op(A)[row,row] = conj(A[row,row])
        if constexpr (TRANS == CtpsvTrans::CONJ_TRANS) v = CtpsvCConj(v);
        return v;
    }
}
```

#### 3.2.2.4 Scalar 路径（n < 128）

```cpp
template <CtpsvUplo UPLO, CtpsvTrans TRANS, CtpsvDiag DIAG>
class CtpsvKernel {
public:
    __aicore__ inline CtpsvKernel() {}
    __aicore__ inline void Init(const CtpsvTilingData& tiling) {
        n = tiling.n;
        incx = tiling.incx;
        apGM.SetAddress(reinterpret_cast<GM_ADDR>(tiling.ap));
        xGM.SetAddress(reinterpret_cast<GM_ADDR>(tiling.x));
    }
    __aicore__ inline void Process() {
        if constexpr (kForward) {
            for (uint32_t row = 0; row < n; ++row) Compute(row);
        } else {
            for (uint32_t row = n; row-- > 0;) Compute(row);
        }
    }
private:
    __aicore__ inline uint32_t XOffset(uint32_t j) const {
        // incx < 0 时 x 反向存储：x[j] 位于 base + (n-1-j)*|incx|
        return (incx > 0 ? j : (n - 1 - j)) * static_cast<uint32_t>(
            incx > 0 ? incx : -incx);
    }
    __aicore__ inline void Compute(uint32_t row) {
        aclblasComplex sum = xGM.GetValue(XOffset(row));
        if constexpr (kForward) {
            for (uint32_t j = 0; j < row; ++j) {
                sum = CtpsvCSub(sum, CtpsvCMul(GetOpElem(row, j), xGM.GetValue(XOffset(j))));
            }
        } else {
            for (uint32_t j = row + 1; j < n; ++j) {
                sum = CtpsvCSub(sum, CtpsvCMul(GetOpElem(row, j), xGM.GetValue(XOffset(j))));
            }
        }
        if constexpr (DIAG == CtpsvDiag::NON_UNIT) {
            sum = CtpsvCDiv(sum, GetOpDiag(row));
        }
        xGM.SetValue(XOffset(row), sum);
    }
    // ... 成员：n, incx, apGM, xGM
};
```

**关键点**：
1. `x` 原地覆写（读入 b → 计算 → 写回 x），无需额外 buffer；
2. `GetValue/SetValue` 是 GlobalTensor 的**标量** GM 读写 API（与 stpsv 一致）；
3. **arch22 复盘教训**：多核 AIV 标量 GM 写不可靠 → **scalar 路径只在单核 AIV 执行**（tpsv 天然串行，无多核并行），无此风险；
4. UNIT diag 时对角元**完全不访问 AP**，符合任务书 §2.1 第 3 点。

#### 3.2.2.5 SIMT VF 路径（n ≥ 128）

**设计要点**：
- **partialSums 分离实虚双数组**（保守方案）：SIMT API 官方支持类型清单不含 complex，且结构体在 UB 上的对齐/访问行为不确定，故用两个 `__ubuf__ float*` 分别存 partial 的实部和虚部；
- **tree reduction** 合并所有线程的 partial；
- **thread 0 写回 GM**（SIMT VF 语义下由单线程执行标量 GM 写是可靠的，参考 stpsv_kernel_simt.cpp）。

```cpp
// blas/tpsv/arch35/ctpsv_kernel_simt.cpp
#include "simt_api/asc_simt.h"
#include "ctpsv_kernel_utils.h"
#include "kernel_constant.h"

constexpr uint32_t UB_PARTIALS_FLOATS = 2048;   // = SIMT_MAX_THREAD_NUM

// Reduce 一个 row 的 partial sum（分离实虚双 tree reduction）
template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT>
__simt_callee__ inline void CtpsvSimtReduceAndUpdate(
    float partialRe, float partialIm,
    float sumRe, float sumIm,
    uint32_t xOff, uint32_t row, uint32_t n,
    __gm__ const aclblasComplex* apGm,
    __gm__ aclblasComplex* xGm,
    __ubuf__ float* partialReBuf,
    __ubuf__ float* partialImBuf)
{
    partialReBuf[threadIdx.x] = partialRe;
    partialImBuf[threadIdx.x] = partialIm;
    asc_syncthreads();
    // Tree reduction（分离实虚）
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialReBuf[threadIdx.x] += partialReBuf[threadIdx.x + stride];
            partialImBuf[threadIdx.x] += partialImBuf[threadIdx.x + stride];
        }
        asc_syncthreads();
    }
    if (threadIdx.x == 0) {
        aclblasComplex sum = {sumRe - partialReBuf[0], sumIm - partialImBuf[0]};
        if constexpr (!DIAG_IS_UNIT) {
            aclblasComplex diag = /* 从 apGm 读 op(A)[row,row] */;
            sum = CtpsvCDivSimt(sum, diag);
        }
        xGm[xOff] = sum;
    }
    asc_syncthreads();
}

// 主 SIMT VF kernel（12 组模板）
template <bool UPLO_IS_UPPER, int TRANS_MODE, bool DIAG_IS_UNIT>
// TRANS_MODE: 0=NO_TRANS, 1=TRANS, 2=CONJ_TRANS
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM)
inline void CtpsvSimt(
    uint32_t n, int64_t incx,
    __gm__ const aclblasComplex* apGm,
    __gm__ aclblasComplex* xGm)
{
    __ubuf__ float partialReBuf[UB_PARTIALS_FLOATS];
    __ubuf__ float partialImBuf[UB_PARTIALS_FLOATS];

    // 每 thread 处理一个 j 分片：thread t 处理 j = t, t+blockDim, t+2*blockDim, ...
    // row 循环由所有 thread 同步推进（串行依赖）
    bool forward = /* kForward 判定 */;
    for (uint32_t row = forward ? 0 : n; forward ? row < n : row-- > 0;) {
        // 每 thread 累加自己分到的 j 的 partial
        float partRe = 0.0f, partIm = 0.0f;
        uint32_t jStart = forward ? threadIdx.x : row + 1 + threadIdx.x;
        uint32_t jEnd   = forward ? row : n;
        for (uint32_t j = jStart; j < jEnd; j += blockDim.x) {
            aclblasComplex a = /* op(A)[row, j] */;
            aclblasComplex xj = xGm[XOffset(j)];
            aclblasComplex prod = CtpsvCMulSimt(a, xj);
            partRe += prod.real;
            partIm += prod.imag;
        }
        // 当前 x[row] 的初值（b[row]）
        float sumRe = 0.0f, sumIm = 0.0f;
        if (threadIdx.x == 0) {
            aclblasComplex b = xGm[XOffset(row)];
            sumRe = b.real; sumIm = b.imag;
        }
        // Reduce + Update（含除法与 GM 写回）
        CtpsvSimtReduceAndUpdate<UPLO_IS_UPPER, DIAG_IS_UNIT>(
            partRe, partIm, sumRe, sumIm,
            XOffset(row), row, n, apGm, xGm,
            partialReBuf, partialImBuf);
    }
}
```

**关键点**：
1. **UB 内存占用**：`2 × 2048 × 4B = 16 KB`，UB_SIZE 248 KB 富余 232 KB，无风险；
2. **LAUNCH_BOUND(SIMT_MAX_THREAD_NUM)**：编译器告知最大线程数，用于寄存器分配；
3. **`asc_syncthreads()`** 在每次 tree reduction 步与最终写回后调用，保证 UB→GM 的顺序；
4. **thread 0 独占 GM 写**：避免多核标量 GM 写不可靠（arch22 复盘教训）；
5. **`sumRe/sumIm` 初值仅 thread 0 有意义**：其他 thread 传 0，reduce 时用 `sum - partial` 只在 thread 0 执行；
6. **12 组模板实例化**：用 `TRANS_MODE ∈ {0,1,2}` 而非 bool，配合 `if constexpr` 分派 CONJ。

#### 3.2.2.6 分派入口（ctpsv_kernel.cpp）

```cpp
extern "C" __global__ __aicore__ void ctpsv_kernel(GM_ADDR tilingGM)
{
    CtpsvTilingData tiling = *reinterpret_cast<CtpsvTilingData*>(tilingGM);
    if (tiling.numThreads > 0) {
        // SIMT VF 路径：12 组模板 switch
        #define DISPATCH_SIMT(UPLO, TRANS, DIAG, TRANS_MODE) \
            asc_vf_call<CtpsvSimt<UPLO, TRANS_MODE, DIAG>>( \
                dim3{tiling.numThreads, 1, 1}, \
                tiling.n, tiling.incx, \
                reinterpret_cast<__gm__ const aclblasComplex*>(tiling.ap), \
                reinterpret_cast<__gm__ aclblasComplex*>(tiling.x));
        // ... 12 组 case
    } else {
        // Scalar 路径：12 组模板 switch
        // ... 12 组 case，实例化 CtpsvKernel<UPLO, TRANS, DIAG>
    }
}
```

### 3.2.3 支持硬件

| 芯片版本 | 支持情况 |
|---------|---------|
| Ascend 950PR | **支持** |
| Ascend 950DT | **支持**（同 arch35） |
| Atlas A3 训练系列 / 推理系列 | 不支持 |
| Atlas A2 训练系列 / 推理系列 | 不支持 |

（与同族 `aclblasStpsv` README 一致。）

### 3.2.4 算子约束限制

- 仅支持 COMPLEX64（实部/虚部各 float32）；
- 仅支持 packed 存储（无 lda 参数）；
- 仅支持 incx 步长（不支持其他非连续内存布局）；
- 不做奇异/近奇异检测（NON_UNIT 时调用方保证对角非零）；
- 不要求确定性计算（SIMT tree reduction 顺序确定，实际结果 bitwise 一致）；
- 不支持 dynamic shape（n 为运行时入参，kernel 每次 launch 重新分派）；
- x 原地覆写（入口 b、出口解 x），不返回视图；
- n = 0 合法 no-op。

---

# 4. 可维可测分析

## 4.1 精度标准

**来源**：[opbase experimental_standard.md](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) + 任务书 §3.2。

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|---------|------|------|------------------------|---------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ (1.53e-5) | 0.99 | 1e-2 或 32 × ULP |

**逐元素通过条件**：`|actual - golden| ≤ atol + rtol × |golden|`
**整体通过条件**：`matched_ratio ≥ 0.99` **且** `max_abs_error ≤ max_abs_error_limit`

**补充说明**：
- 实部、虚部**分别独立**按 FLOAT32 标准判定；
- 三角求解含除法与前/后向代入，非 bit-exact；
- NON_UNIT 场景测试输入保证对角非零（近零对角修正）；
- 算子本身不含随机数生成，正态/均匀分布是测试输入生成规则。

**MERE / MARE 阈值**（任务方 CSV 提供）：
- `mere_threshold = 0.00012207031 = 2⁻¹³`
- `mare_multiplier = 10.0`

## 4.2 性能标准

**测试设备**：Ascend 950PR。
**采样方式**：warmup ≥ 5 次，有效采样 ≥ 50 次取平均（Avg time，单位 us）。
**判据**：`ratio = 标杆耗时 / NPU 实际耗时 ≥ 0.4`。

| case | n | uplo | trans | diag | incx | 标杆耗时 (us) | H100 基线 (us) |
|------|---|------|-------|------|------|--------------|----------------|
| 1 | 512 | LOWER | N | NON_UNIT | 1 | 971.85 | 388.74 |
| 2 | 1024 | UPPER | N | NON_UNIT | 1 | 2074.42 | 829.77 |
| 3 | 2048 | LOWER | T | NON_UNIT | 1 | 4012.85 | 1605.14 |
| 4 | 4096 | UPPER | C | NON_UNIT | 1 | 12481.55 | 4992.62 |

**NPU 实际耗时上限**（ratio = 0.4 反推）：
- case 1: ≤ 2429.6 us
- case 2: ≤ 5186.1 us
- case 3: ≤ 10032.1 us
- case 4: ≤ 31203.9 us

## 4.3 测试工程结构

**遵循仓内 ST 开发指南**（`docs/zh/develop/st_develop_guide.md`）：

```
test/tpsv/ctpsv/
├── ctpsv_param.h              # 继承 BlasTestParamBase，字段按 API 参数顺序
├── ctpsv_golden.h             # CPU golden：aclblasCtpsv_cpu()，逐元素复数求解
├── CMakeLists.txt             # ops_blas_add_gtest_tests(${OPS_BLAS} ctpsv_test)
└── arch35/
    ├── ctpsv_npu_wrapper.h    # NPU wrapper：aclblasCtpsv_npu()，封装 ACL malloc/H2D/kernel/D2H/free
    ├── ctpsv_test.cpp         # GTest 主入口（TEST_F NullHandle + TEST_P CsvDriven）
    ├── ctpsv_test.csv         # 复用任务提供的 1200 行 CSV（1000 精度 + 200 性能）
    └── CMakeLists.txt         # 可选
```

**CSV 列格式**（任务方 `gen_csv.py` 输出）：

```
case_name,description,uplo,trans,diag,n,incx,ap_fill,x_fill,random_seed,expect_result,mere_threshold,mare_multiplier
```

**GTest 模板**：

```cpp
class CtpsvArch35Test : public BlasTest<CtpsvParam> {};

TEST_F(CtpsvArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasCtpsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N,
                                            ACLBLAS_NON_UNIT, 5, nullptr, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ctpsv, CtpsvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<CtpsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtpsvParam>);

TEST_P(CtpsvArch35Test, CsvDriven) {
    const auto& p = GetParam();
    // 1. makeBlasTriangular (复数版) 生成 AP host 数据
    // 2. makeBlasArray (复数版) 生成 x host 数据
    // 3. H2D → aclblasCtpsv_npu → D2H
    // 4. EXPECT_EQ(ret, p.expectResult)
    // 5. 若 SUCCESS：aclblasCtpsv_cpu golden + Verifier::verifyVector（实虚分别比对）
}
```

## 4.4 测试用例覆盖

**任务方提供**（`test_cases/ctpsv_test.csv`，1200 行）：
- 精度用例 1000 条
- 性能用例 200 条

**覆盖矩阵**（对齐任务书 §3.5 + `experimental_standard.md` §1）：

| 维度 | 覆盖 |
|-----|------|
| uplo | UPPER / LOWER 全覆盖 |
| trans | OP_N / OP_T / OP_C 全覆盖 |
| diag | NON_UNIT / UNIT 全覆盖 |
| 组合数 | 2 × 3 × 2 = **12 组全覆盖** |
| n | 0 / 1 / 小质数 / 2^k / 2^k±1 / 非对齐 / 大规模（≥4096） |
| AP 填充 | RANDOM_NORM_5_5 / RANDOM_ALTER / RANDOM_EXTREME / VALUE_NORM_0 / VALUE_NORM_INF / VALUE_NORM_NAN / NULLPTR |
| x 填充 | 同上 |
| incx | ±1 / ±2 / ±3 全组合 |
| 边界 | n=0（no-op）、n=1（单元素）、AP/x nullptr（预期 INVALID_VALUE） |
| 特殊值 | Inf / -Inf / NaN |

**自补充用例**（若任务方未覆盖）：
- incx 大绝对值（如 ±100）验证步长正确性；
- UNIT diag 时 AP 对角元填 NaN/Inf 验证"不访问"语义；
- OP_C 与 OP_T 对同一 AP/x 的结果**必须不同**（复数下 A^T ≠ A^H）。

## 4.5 兼容性分析

**新增接口**，无兼容性问题：
- `include/cann_ops_blas.h` 中新增 `aclblasCtpsv` 声明，不影响已有接口；
- `blas/tpsv/arch35/` 下新增 5 个源文件，不修改已有 `stpsv_*.{cpp,h}`；
- `blas/tpsv/README.md` 追加 aclblasCtpsv 章节，不删除已有 aclblasStpsv 章节；
- `test/tpsv/ctpsv/` 目录全新，不影响 `test/tpsv/stpsv/`；
- Kernel 命名 `ctpsv_kernel` 与已有 `stpsv_kernel` 独立，无符号冲突。

**接口一致性**：
- 与 cuBLAS `cublasCtpsv` 参数序列**逐参数对应**（handle/uplo/trans/diag/n/AP/x/incx）；
- 与同族 `aclblasStpsv` 参数序列**逐参数对应**，仅元素类型扩展为 `aclblasComplex`；
- 禁止定义 950PR 私有平行接口（任务书 §7.3）。

---

# 5. 交付物清单与开发排期

## 5.1 交付物清单

| # | 交付物 | 路径 | 说明 |
|---|-------|------|------|
| D1 | 设计文档（本文档） | `cann-ops-competitions/04_tasks/01_community-task-2026/tasklist/XX-aclblasCtpsv/design.md` | PR 到 cann-ops-competitions 仓 |
| D2 | 源码 | `ops-blas/blas/tpsv/arch35/ctpsv_{tiling_data.h, kernel_utils.h, kernel.cpp, kernel_simt.cpp, host.cpp}` | 5 个新文件 |
| D3 | 头文件 patch | `ops-blas/include/cann_ops_blas.h` | 插入 aclblasCtpsv 声明 |
| D4 | 测试工程 | `ops-blas/test/tpsv/ctpsv/{ctpsv_param.h, ctpsv_golden.h, CMakeLists.txt, arch35/{ctpsv_npu_wrapper.h, ctpsv_test.cpp, ctpsv_test.csv}}` | 6 个新文件 |
| D5 | README patch | `ops-blas/blas/tpsv/README.md` | 追加 aclblasCtpsv 章节 |
| D6 | 自测报告 | cann-ops-competitions 指定 sheet 模板 | 用例参数 + 精度截图（实/虚分开）+ 性能截图 + 内存 |
| D7 | 个人代码仓 | 邀请 Ascend-CANN 为开发者 | 分支：`feature/aclblasCtpsv-arch35` |

## 5.2 开发排期

| 阶段 | 预估耗时 | 交付 |
|-----|---------|------|
| 0. 工程骨架 | 10 min | scratch/{design,src,test,report} |
| 1. 设计文档 | 30 min | 本文档 |
| 2. 源码（5 文件 + 头文件 patch） | 60 min | D2 + D3 |
| 3. 测试工程（6 文件 + README patch） | 45 min | D4 + D5 |
| 4. HiDevLab webIDE 编译冒烟 | 20 min | `bash build.sh --ops=ctpsv --soc=ascend950` 0 error |
| 5. 真机精度验证 | 30 min | 12 组枚举 + 1000 条 CSV，matched_ratio ≥ 0.99 |
| 6. 真机性能验证 | 20 min | 4 条 case，ratio ≥ 0.4 |
| 7. 自测报告 + PR | 30 min | D1 + D6 + D7 + ops-blas PR |

**总计**：约 4-5 小时（含真机验证）。

---

# 6. 风险与缓解

| # | 风险 | 概率 | 影响 | 缓解方案 |
|---|------|-----|------|---------|
| R1 | `complex.h` 运算符重载在 kernel 侧不可用 | 中 | 编译失败 | **保守方案**：kernel 侧手写 `__aicore__ inline` 复数运算，不 include complex.h |
| R2 | SIMT VF 不支持 `aclblasComplex` 结构体 partialSums | 中 | 编译失败或运行异常 | **保守方案**：partialSums 分离为 `float partialRe[2048] + float partialIm[2048]` |
| R3 | 12 组模板展开导致代码体积膨胀 | 低 | 编译时间增加 | 用宏 `DEFINE_CTPSV_KERNEL(uplo, trans, diag, name)` 生成，控制重复代码 |
| R4 | n=4096 SIMT 路径 UB 不足 | 低 | kernel launch 失败 | 静态断言 `2 × SIMT_MAX × sizeof(float) ≤ UB_SIZE`（16KB ≤ 248KB） |
| R5 | NON_UNIT 对角元近零导致除法溢出 | 低 | Inf/NaN 输出 | 与 stpsv 一致——不做奇异检测；测试用例 gen_csv.py 保证对角非零 |
| R6 | Debug 编译 175× 慢（任务2 血泪） | 中 | 编译超时 | CMake 强制 `-O2 Release`，编译前 `grep CMAKE_BUILD_TYPE` 校验 |
| R7 | arch22 多核 AIV 标量 GM 写不可靠 | 低 | 结果错误 | 本任务 arch35；SIMT VF 路径由 thread 0 独占 GM 写，规避多核竞争 |
| R8 | HiDevLab webIDE 100h 免费时长耗尽 | 低 | 无法完成真机验证 | 编译前先本地静态检查，真机时段集中做精度/性能验证 |
| R9 | 任务书 §2.1 LOWER 索引公式 typo | 已确认 | 若按 typo 实现则全错 | 按仓内 stpsv_kernel_utils.h 正确公式（Netlib/cuBLAS 一致） |
| R10 | OP_C 与 OP_T 混淆（实数下等价） | 中 | CONJ_TRANS 结果错误 | kernel 模板 3 值 enum 严格区分；测试用例专门覆盖 OP_C 与 OP_T 结果**不同**的场景 |

---

# 7. 附录

## 7.1 参考实现文件（本地镜像路径）

- `scratch/mirror/cann-ops/ops-blas/blas/tpsv/arch35/stpsv_kernel.cpp` (264 行)
- `scratch/mirror/cann-ops/ops-blas/blas/tpsv/arch35/stpsv_kernel_simt.cpp` (189 行)
- `scratch/mirror/cann-ops/ops-blas/blas/tpsv/arch35/stpsv_kernel_utils.h` (39 行)
- `scratch/mirror/cann-ops/ops-blas/blas/tpsv/arch35/stpsv_host.cpp` (96 行)
- `scratch/mirror/cann-ops/ops-blas/blas/tpsv/arch35/stpsv_tiling_data.h` (30 行)
- `scratch/mirror/cann-ops/ops-blas/blas/common/helper/complex.h` (119 行)
- `scratch/mirror/cann-ops/ops-blas/include/cann_ops_blas.h` (26418B)
- `scratch/mirror/cann-ops/ops-blas/include/cann_ops_blas_common.h` (137 行)
- `scratch/mirror/cann-ops/ops-blas/blas/common/helper/kernel_constant.h` (20 行)
- `scratch/mirror/cann-ops/ops-blas/blas/common/helper/host_utils.h` (96 行)
- `scratch/mirror/cann-ops/ops-blas/blas/common/helper/aclblas_handle_internal.h` (210 行)
- `scratch/mirror/cann-ops/ops-blas/blas/herk/arch35/cherk_host.cpp` (431 行，arch35 复数算子参考)
- `scratch/mirror/cann-ops/ops-blas/blas/herk/arch35/cherk_kernel.cpp` (25448B)

## 7.2 参考文档（本地镜像路径）

- `scratch/mirror/cann-ops__ops-blas__blas__tpsv__README.md` — Tpsv README 官方模板
- `scratch/mirror/cann-ops__ops-blas__docs__zh__develop__st_develop_guide.md` — ST 开发指南
- `scratch/mirror/cann-ops__ops-blas__docs__zh__develop__compile_and_run_example.md` — build.sh 用法
- `scratch/mirror/experimental_standard.md` — 生态算子开源精度标准（FLOAT32）
- `design_template.md` — 官方设计文档模板

## 7.3 关键枚举值（ACLBLAS_*）

| 名称 | 值 |
|------|---|
| ACLBLAS_UPPER | 121 |
| ACLBLAS_LOWER | 122 |
| ACLBLAS_OP_N | 111 |
| ACLBLAS_OP_T | 112 |
| ACLBLAS_OP_C | 113 |
| ACLBLAS_NON_UNIT | 131 |
| ACLBLAS_UNIT | 132 |
| ACLBLAS_STATUS_SUCCESS | 0 |
| ACLBLAS_STATUS_INVALID_VALUE | 3 |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | 9 |

## 7.4 构建命令

```bash
# 环境变量
source /usr/local/Ascend/cann/set_env.sh

# 仅编译（不跑测试）
bash build.sh --ops=ctpsv --soc=ascend950

# 编译 + 跑测试
bash build.sh --ops=ctpsv --run --soc=ascend950

# 指定 device（多卡机器）
bash build.sh --ops=ctpsv --run --soc=ascend950 --device=1

# 打 run 包（部署）
bash build.sh --pkg --soc=ascend950
```

---

**文档版本**：v1.0
**作者**：（待填 gitcode 用户名）
**最后更新**：2026-09-16
