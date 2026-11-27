# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-gemvBatched类5算子 |
| 提交团队 | Eason_Cat |
| 任务讨论 | https://gitcode.com/cann/ops-blas/discussions |
| 适配硬件 | Atlas A2/A3，arch22 |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/gemv/arch22/` |
| 测试目标目录 | `test/gemv/*/arch22/` |
| 文档状态 | 设计文档提交 |

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务。目标：gemvBatched 五接口族；禁 SIMT/Te。

## 背景介绍

gemvBatched 五接口族；禁 SIMT/Te。

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
# aclblasCgemvBatched 算子开发任务书（Atlas A2/A3）

> **本地静默草稿**：由官方「10月 CgemvBatched（950）」改写为 A2。  
> 讨论帖为 **gemvBatched×5**；本文件先覆盖 **C**；其余精度变体另补。  
> **仅 A2 测 A100**；满暖场 w≥100 / s≥200。禁止对外 PR。

## 1 任务概述

aclblasCgemvBatched（Atlas A2/A3），对标 cuBLAS legacy `cublasCgemvBatched`。合入 ops-blas。  
**禁止 SIMT/Te 照搬 arch35**。

## 2 核心要求

每 batch：`y_i = alpha * op(A_i) * x_i + beta * y_i`，COMPLEX64；op∈{N,T,C}。  
指针数组 `Aarray`/`xarray`/`yarray`。实现：`blas/gemv_batched/arch22/`。

```cpp
aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans,
    int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* const* Aarray, int lda,
    const aclblasComplex* const* xarray, int incx,
    const aclblasComplex* beta,
    aclblasComplex* const* yarray, int incy, int batchCount);
```

## 3 验收

- 硬件：A2/A3，自验 910B3；CANN 9.1.0  
- 精度：ACC1000；mixed_tolerance FLOAT32 分量  
- 性能：≥0.8× GPU；标杆 **A100@2201** 满暖场（cold + w≥100 + s≥200）  
- 用例：`cgemv_batched_acc1000.csv` / `cgemv_batched_pf200.csv`

## 4 私仓

`qq_53202098/202611-aclblasGemvBatched5-A2A3-code`：`acceptance` / `testing`。
```
