# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-aclblasCgemm3m |
| 提交团队 | Eason_Cat |
| 任务讨论 | https://gitcode.com/cann/ops-blas/discussions/93 |
| 适配硬件 | Atlas A2/A3，arch22 |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/gemm/arch22/` |
| 测试目标目录 | `test/gemm/cgemm3m/arch22/` |
| 文档状态 | 设计文档提交 |

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务。目标：复数 GEMM 3m；三次实乘分解；禁 SIMT/Te。

## 背景介绍

复数 GEMM 3m；三次实乘分解；禁 SIMT/Te。

### 约束

1. 按官方任务书接口与精度/性能口径验收。
2. A2/A3（arch22）禁止照搬 arch35 的 SIMT / Te。
3. Host 校验与 quick-return 对齐 cuBLAS / Netlib / 任务书。

# 技术方案（required）

## 总体方案

- Host：参数校验、tiling、kernel launch。
- Device：Ascend C 直调。
- 测试：CSV + GTest；精度全量 + 性能子集。

## 关键模块

1. tiling
2. kernel
3. ST golden

# 验收与风险

- 精度 / 性能按任务书；风险含非单位 stride 与复数口径。

# 任务书摘录

```text
# aclblasCgemm3m 算子开发任务书（Atlas A2/A3）

> **本地静默草稿**：由官方「10月社区任务-aclblasCgemm3m（950）」改写为 A2/A3。未对外发布。  
> 用例：精度 1000 + 性能 200。`test_cases/gpu_baseline.csv` 待 A100 自测（PENDING）。

## 1 任务概述

aclblasCgemm3m 算子开发（Atlas A2/A3），对标 cuBLAS legacy `cublasCgemm3m`（3m 算法），补齐 Blas 仓 A2/A3 复数 GEMM-3m 能力；语义依据 cuBLAS、Netlib/CBLAS 与仓内 950PR 既有实现，代码与测试合入 https://gitcode.com/cann/ops-blas。

开发语言与工程：Ascend C、kernel 直调；工程结构对齐同仓 `arch22` 兄弟算子。**禁止照搬 arch35 的 SIMT / Te 路径**。建议 A2 机器开发。

讨论帖参考：https://gitcode.com/cann/ops-blas/discussions/93

## 2 核心开发要求

### 2.1 功能实现要求

实现 `aclblasCgemm3m`，对标：

```cpp
cublasStatus_t cublasCgemm3m(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha,
    const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const cuComplex* beta,
    cuComplex* C, int ldc);
```

**数学公式**：`C = alpha * op(A) * op(B) + beta * C`，op ∈ {N, T, C}；列主序。

**算法说明**：

1. **3m 分解**：复数乘拆为三次实数乘的线性组合得到实部/虚部；设计与实现须说明走 3m，不得退化为未声明的 4M 路径冒充验收。
2. **实虚部分离**：拆出 op(A)/op(B) 实部与虚部，按 3m 公式重组写入 C。
3. **枚举**：transa/transb 支持 `ACLBLAS_OP_N` / `T` / `C`，其它返回 `ACLBLAS_STATUS_INVALID_VALUE`。
4. **确定性**：同输入多次执行结果一致（见 §2.5）。

### 2.2 算子工程模式

kernel 直调；属 Level-3 GEMM 族。实现目录：`blas/gemm/arch22/`（新增 `cgemm3m_*` 或扩展既有 gemm 分派）。测试：`test/gemm/cgemm3m/arch22/`。

**A2 约束**：禁止 SIMT / Te；可复用 arch35 host 校验与 3m 算法语义，device 按 arch22 重写。

### 2.3 接口定义

```cpp
aclblasStatus_t aclblasCgemm3m(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

（`include/cann_ops_blas.h` 既有声明，与 950PR 共用，禁止平行 API）

### 2.4 参数说明

与官方 950 任务书 §2.4 一致（handle / transa / transb / m,n,k / alpha / A,lda / B,ldb / beta / C,ldc）。负维度、非法枚举、空指针、lda/ldb/ldc 不足等返回 `ACLBLAS_STATUS_INVALID_VALUE`；handle 空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。

### 2.5 算子实现约束

| 约束项 | 内容 |
| --- | --- |
| 非连续 Tensor | 不支持（列主序前导维） |
| broadcast | 不支持 |
| dynamic shape | 不要求 |
| 原地与视图 | C 为 in/out；不复用 A/B 内存 |
| 确定性计算 | 要求，同输入两次执行逐位一致 |
| 空/0 维 | m=0 或 n=0：quick return SUCCESS；k=0：仅 beta 缩放 C；beta=0 不读 C 旧值 |
| 架构能力 | **仅 arch22**：禁止 SIMT / Te |

## 3 验收标准
```
