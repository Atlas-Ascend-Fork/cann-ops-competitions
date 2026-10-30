# aclblasGemmBatchedEx 算子设计文档

# 1 需求背景

## 1.1 需求来源

完成“10月社区任务-aclblasGemmBatchedEx算子开发（A2A3）”，使用 Ascend C 补齐 Atlas A2/A3 的 FLOAT32 批量矩阵乘能力。
产品实现位于 `ops-blas/blas/gemm_batched_ex/arch22/`，测试复用仓库 CSV/GTest 框架。

## 1.2 背景介绍

批量 GEMM 对多个独立矩阵执行同一组乘加运算。
各 batch 共用维度、前导维和标量，通过 Device 指针数组分别指定矩阵地址。
设计重点是复用现有 BLAS 接口、支持列主序与转置，并平衡小矩阵调度和大矩阵计算效率。

# 2 需求分析

## 2.1 需求描述

复用 `include/cann_ops_blas.h` 中已有声明，不新增产品私有接口：

```cpp
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);
```

| 参数 | 约定 |
| --- | --- |
| handle | 有效 BLAS 句柄，使用句柄绑定的 stream 异步执行 |
| transa/transb | N、T、C；实数路径中 C 与 T 等价 |
| m/n/k/batchCount | 非负整数；op(A) 为 m×k，op(B) 为 k×n |
| alpha/beta | 非空 FLOAT32 标量指针，支持 Host 或 Device 内存 |
| Aarray/Barray/Carray | Device 指针数组，各矩阵按列主序存储，C 原地更新 |
| Atype/Btype/Ctype | 本次 A2/A3 实现支持 ACL_FLOAT |
| lda/ldb/ldc | 满足任务书的前导维约束，支持前导维 padding |
| computeType/algo | ACLBLAS_COMPUTE_32F / ACLBLAS_GEMM_DEFAULT |

## 2.2 需求拆解

1. 按任务书校验句柄、枚举、维度、前导维、指针和支持组合，校验先于 no-op。
2. 覆盖 N/T/C 组合、独立 batch 地址和 C 的原地更新。
3. 覆盖零维、k=0、alpha=0、beta=0/1，以及 Device 标量的流内更新。
4. 以 Netlib BLAS 3.10.0 为精度基准，以 `msprof op` 采集性能。

# 3 详细设计

## 3.1 算子分析

### 数学公式

对 i=0,…,batchCount−1：

`C_i = alpha × op(A_i) × op(B_i) + beta × C_i`。

### 支持数据类型与形状

输入、输出及标量为 FLOAT32。各 batch 共享 m/n/k，矩阵之间不广播。
`lda >= max(1, transa=N ? m : k)`，`ldb >= max(1, transb=N ? k : n)`，`ldc >= max(1,m)`。

## 3.2 算子实现

### Host 侧设计

先完成参数校验，再处理 m=0、n=0 或 batchCount=0 的合法 no-op。
Host 标量可用于路径选择；Device 标量由 Kernel 在绑定 stream 上读取，避免在 Host 侧提前缓存其值。

将列主序计算映射为行主序 `Cᵀ = op(B)ᵀ × op(A)ᵀ`，保留调用方的前导维和指针数组。
按输出矩阵划分 tile，把 batch 与输出 tile 合并为任务序列，根据可用 Cube 核数分配任务。
小输出优先增加并行任务，大输出优先复用输入并减少调度开销。

按路径计算句柄 workspace：直接输出路径减少中间存储；输入整理和乘积后处理路径按实际需要分配。
workspace 及各 batch 的数据区域相互隔离。

### Kernel 侧设计

Cube 核根据任务编号读取对应 batch 的矩阵地址，完成分块乘法和尾块处理，使用 FLOAT32 累加。
针对不规则前导维与归约维，可先将输入整理为适合 Cube 的连续分块布局，并显式补零归约尾部。
通过扩大基础计算块、平衡各核任务和直接写回 C，减少重复搬运及额外 Kernel。

alpha=1、beta=0 的路径优先直接输出乘积。一般标量路径将乘积与旧 C 在 Vector 核中合并。
beta=0 时跳过旧 C 的读取。所有写回只覆盖 m×n 的有效矩阵区域，保留 padding。

k=0 或 alpha=0 时仅执行 beta 缩放，不读取 A/B 矩阵元素；beta=1 时保持 C，beta=0 时直接置零。
不同路径均遵循同一接口和精度约定，性能优化不得改变参数语义。

## 3.3 支持硬件

| 产品 | 支持范围 |
| --- | --- |
| Atlas A2 系列 | FLOAT32，arch22 |
| Atlas A3 系列 | FLOAT32，arch22 |

开发与自验证环境按任务书使用 CANN 9.1.0，性能验证使用 910B3。

## 3.4 算子约束限制

本次不扩展混合精度组合、广播或前导维以外的非连续布局。
不支持的数据类型、计算类型和算法组合返回 NOT_SUPPORTED；其余错误按任务书返回对应状态码。
调用方须在读回结果前同步绑定 stream，并保证有效矩阵地址及异步输入的生命周期。

# 4 可维可测分析

## 4.1 精度标准与性能标准

| 项目 | 验证方法 |
| --- | --- |
| Golden | Netlib BLAS 3.10.0 的 cblas_sgemm，逐 batch 全矩阵比对 |
| FLOAT32 | atol=rtol=2^-13，matched_ratio≥0.99，max_abs_error≤max(0.01,32×ULP) |
| 特殊值 | NaN/Inf 按任务书单独判定，不混入有限值误差统计 |
| alpha=0 | beta 缩放按 EXACT 校验 |
| 性能 | 使用 msprof op 的 Task Duration，排除预热样本，按任务书采样和逐例比较 GPU 基线；多 Kernel 路径合计完整调用耗时 |
| 覆盖范围 | 配套 CSV、转置、padding、Host/Device 标量、零维、非法参数及 Inf/NaN |

先打通 baseline 并完成一次全量性能采样，再针对未达标用例优化。
优化收敛后执行全量性能与精度回归，检查遗漏和性能分化。
实测数据、原始日志与 profiler 产物单独保存在自测材料中。

## 4.2 兼容性分析

复用公共接口、句柄和构建框架，新增内容限定在 arch22 实现与对应测试目录。
950 产品线保留既有实现和类型支持范围。本设计不新增平行 API 或独立编译工程。
