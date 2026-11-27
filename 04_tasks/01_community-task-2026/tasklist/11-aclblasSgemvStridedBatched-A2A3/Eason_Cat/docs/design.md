# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-gemvStridedBatched类5算子 |
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

2026 年 11 月 CANN 社区任务。目标：gemvStridedBatched 五接口族；A2 直调。

## 背景介绍

gemvStridedBatched 五接口族；A2 直调。

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
# aclblasSgemvStridedBatched 算子开发任务书（Atlas A2/A3）

> 讨论帖 #91：gemvStridedBatched×5。本文件先覆盖 **S（float32）**；其余 dtype 变体另补。  
> 仅 A2 测 A100。禁止对外 PR。

## 1 概述

对标 cuBLAS `cublasSgemvStridedBatched`。单基址 + strideA/stridex/stridey。  
实现：`blas/gemv_strided_batched/arch22/`。禁止 SIMT/Te。

## 2 接口

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey,
    int batchCount);
```

## 3 验收

- 910B3；ACC1000 + PF200；≥0.8× GPU  
- A100@2201 满暖场 w≥100 / s≥200  
- 用例：`sgemv_strided_batched_acc1000.csv` / `sgemv_strided_batched_pf200.csv`

## 4 私仓

`qq_53202098/202611-aclblasGemvStridedBatched5-A2A3-code`：`acceptance` / `testing`。
```
