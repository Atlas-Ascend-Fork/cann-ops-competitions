# 设计文档头信息

| 项目 | 内容 |
| --- | --- |
| 任务名称 | 202611社区任务-aclblasSgemmBatched/HgemmBatched |
| 提交团队 | Eason_Cat |
| 任务讨论 | https://gitcode.com/cann/ops-blas/discussions/94 |
| 适配硬件 | Atlas A2/A3，arch22 |
| CANN 版本 | 9.1.0 |
| 相关代码仓 | `cann/ops-blas` |
| 代码目标目录 | `blas/gemm/arch22/` |
| 测试目标目录 | `test/gemm/*/arch22/` |
| 文档状态 | 设计文档提交 |

# 需求背景（required）

## 需求来源

2026 年 11 月 CANN 社区任务。目标：Batched GEMM FP32/FP16；A2 直调。

## 背景介绍

Batched GEMM FP32/FP16；A2 直调。

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
# aclblasSgemmBatched 算子开发任务书（Atlas A2/A3）

> 与 `aclblasHgemmBatched` 同讨论帖（#94）。本地静默。仅 A2 测 A100。禁止对外 PR。

## 1 概述

FLOAT32 批量 GEMM，对标 `cublasSgemmBatched`。目录：`blas/gemm/arch22/`；测试：`test/gemm_batched/sgemm_batched/arch22/`。禁止 SIMT/Te。

## 2 接口

```cpp
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount);
```

## 3 验收

- 910B3；ACC1000 + PF200；≥0.8× GPU  
- A100@2201 满暖场（w≥100,s≥200）  
- 用例：`sgemm_batched_acc1000.csv` / `sgemm_batched_pf200.csv`（由同规模 H 用例衍生）

## 4 私仓

`qq_53202098/202611-aclblasS_HgemmBatched-A2A3-code`：`acceptance` / `testing`。
```
