# 需求背景（required）

## 需求来源

本设计对应“10 月社区任务 aclblasCgemm3mStridedBatched 算子开发（950）”，参与账号为 **T350380**。目标代码仓为 [ops-blas](https://gitcode.com/cann/ops-blas)，个人开发仓为 [T350380/ops-blas](https://gitcode.com/T350380/ops-blas)。

## 背景介绍

面向 Atlas 950PR，使用 Ascend C 补齐单精度复数跨步批量矩阵乘能力。接口对标 cuBLAS 的 `cublasCgemm3mStridedBatched`，采用三次实数矩阵乘完成复数矩阵乘，并复用仓内 GEMM 族的直调工程和计算组件。

# 需求分析（required）

## 需求描述

对每个批次 `i`，计算 `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`。矩阵为列主序 COMPLEX64，`op` 支持不转置、转置和共轭转置。A、B、C 的批次基址分别为 `A + i*strideA`、`B + i*strideB`、`C + i*strideC`。

公共接口在 `include/cann_ops_blas.h` 中声明：

```cpp
aclblasStatus_t aclblasCgemm3mStridedBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa,
    aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, long long strideA,
    const aclblasComplex* B, int ldb, long long strideB,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc, long long strideC,
    int batchCount);
```

| 参数 | 语义与约束 |
| --- | --- |
| handle | 非空 BLAS 句柄，使用其绑定的 stream |
| transa、transb | `ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C` |
| m、n、k、batchCount | 非负 `int`；运算后 A 为 m×k，B 为 k×n，C 为 m×n |
| alpha、beta | 非空 COMPLEX64 标量指针 |
| lda、ldb、ldc | `int`；分别不小于 `max(1, transa=N ? m : k)`、`max(1, transb=N ? k : n)`、`max(1, m)` |
| strideA、strideB、strideC | `long long` 元素偏移，支持零步长及负步长；调用方保证寻址范围有效 |
| A、B、C | COMPLEX64 矩阵基址；C 原地更新，C 不与 A/B 共用存储 |

## 需求拆解

1. 保持任务书规定的公共接口、参数顺序、类型和返回码。
2. 实现 3m 分解、N/T/C 组合、固定步长批处理及复数 alpha/beta 运算。
3. 覆盖空维度、零归约维、标量特殊值、非法参数及非有限值。
4. 相同输入重复执行逐位一致，并按任务书进行完整矩阵精度和性能验证。

# 详细设计（required）

## 算子分析

设完成转置或共轭处理后的输入为 `A = Ar + i*Ai`、`B = Br + i*Bi`。采用以下 3m 形式：

```text
P0 = (Ar + Ai) @ Br
P1 = Ar @ (Bi - Br)
P2 = Ai @ (Br + Bi)
Re(A @ B) = P0 - P2
Im(A @ B) = P0 + P1
C = alpha * (Re(A @ B) + i*Im(A @ B)) + beta * C
```

实数乘积使用 FLOAT32 累加。小矩阵与窄矩阵采用原生 FLOAT32 3m；大矩阵复用同族的高位与残差补偿乘法。长归约按固定顺序分段合并，控制 3m 合成时的舍入误差。非有限值、大幅值有序归约或严重消减等特殊数值场景允许四乘积精度修正，普通路径保持 3m。

## 算子实现

### Host 侧设计

先校验句柄、枚举、维度、前导维及必要指针，再处理 quick return。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举、负维度、非法前导维和必要指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。

当 m、n 或 batchCount 为零时返回 SUCCESS，C 保持原值。当 k=0 或 alpha=0 时仅执行 `C_i = beta*C_i`，不读取 A/B；alpha=0 时允许空 A/B，遵循附件边界用例。beta=0 时直接写零或乘积结果，beta=1 的纯缩放路径直接返回。

依据矩阵形状、批次数和片上容量选择分块及核数，通过 handle 的 stream 直调 kernel。workspace 复用仓内句柄管理机制，保存必要的实数分量和分块乘积。批次工作区超过预算时分批调度，单批超过预算时使用片上 3m 路径。

### Kernel 侧设计

小矩阵采用轻量并行路径，减少启动和中间存储开销。短 K、窄 M 且 B 转置的矩阵采用向量面板，在寄存器中复用 B 并广播 A。一般矩阵采用“实虚部分离与线性组合 → 三次实数矩阵乘 → 复数结果合成”的流程，复用 GEMM 族的 Cube 计算、搬运和同步组件。输入预处理使用双缓冲；高位与残差在 L1 中驻留，减少重复搬运。

按 batch 与输出分块组织并行任务，在数据独立时合并批次调度。stride=0 的 A/B 复用同一输入；strideC=0 时按批次顺序更新 C，保持 beta 累加语义。尾块只读写有效元素，保留前导维 padding 与批次间隙。

### 性能优化思路

优先优化整体数据流：减少逐批次 kernel 启动，扩大可并行的输出块，复用输入分量，并重叠数据搬运与 Cube 计算。对小矩阵关注启动成本和核利用率，对大矩阵关注分块复用与流水效率。路径选择依据形状和资源容量，不按测试用例编号分派。

## 支持硬件与约束

目标硬件为 Atlas 950PR，软件基线为 CANN 9.1.0。实现置于 `blas/gemm/arch35/`，测试沿用仓内 GTest 与 CSV 框架。支持列主序前导维和固定批次步长，不增加任意维度广播或任意非连续视图接口。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 验证计划 |
| --- | --- |
| 功能与边界 | 覆盖 N/T/C、大小矩阵、非对齐维度、padding、正负及零 stride、复数标量、空维度和非法参数 |
| 精度 | 使用 Netlib BLAS 3.10.0 的 cblas 生成完整矩阵 Golden；实部、虚部分别按任务书 FLOAT32 混合容差标准判定 |
| 非有限值 | 同为 NaN、同号 Inf，或 NaN 与 Inf 配对按任务书视为等效；有限值与非有限值配对判失败，并单独统计 |
| 确定性与内存 | 相同输入重复执行逐位比较，检查 padding、保护区、workspace 和同步依赖 |
| 性能 | 按任务书 §3.3 使用 `msprof op`，预热 5 次后取 10 个有效样本；每个样本汇总完整 API 的全部 kernel 与 batch 耗时 |

性能目标为逐例 `GPU 标杆耗时 / NPU 平均耗时 >= 0.4`，全部性能用例倍率的算术平均值不低于 0.5。异常或不完整采样单独记录。本文描述设计与验证计划，实测结果放入独立自测报告。

## 兼容性分析

使用 ops-blas 统一句柄式接口，按仓内 GEMM 族的目录及编译规则接入。新增功能限定于本算子；复用公共组件时，对受影响的同族路径进行回归验证。

参考：[官方设计模板](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)、[任务提交目录规范](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/README.md)、[任务书引用的混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
