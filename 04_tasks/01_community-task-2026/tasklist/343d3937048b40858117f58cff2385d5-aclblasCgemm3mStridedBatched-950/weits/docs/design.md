# aclblasCgemm3mStridedBatched 算子设计

# 需求背景（required）

## 需求来源

[10月社区任务-aclblasCgemm3mStridedBatched算子开发（950）](https://www.hiascend.com/activities/task-center/details/343d3937048b40858117f58cff2385d5)。实现对标 cuBLAS `cublasCgemm3mStridedBatched` 的句柄式 BLAS 接口，为 ops-blas 补齐 Atlas 950PR 的固定步长批量复数矩阵乘能力。

## 背景介绍

复数矩阵乘可将实部、虚部分解后交给实数 GEMM。3m 算法将通常的四次实数矩阵乘降为三次，适合复用仓内 arch35 的 Pack、Cube、Combine 流程。此次新增统一 API，并在共享实现中增加明确的 3m 路由；已有 `aclblasCgemmStridedBatched` 继续使用其原有路由。

# 需求分析（required）

## 需求描述

在 Atlas 950PR、CANN 9.1.0 上，使用 Ascend C 实现 complex64、列主序、N/T/C 运算、复数 alpha/beta、前导维 padding 和 int64 固定步长批处理，支持确定性计算。通过 handle 绑定的 stream 直调 NPU kernel。

```cpp
aclblasStatus_t aclblasCgemm3mStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa,
    aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, long long strideA,
    const aclblasComplex* B, int ldb, long long strideB,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc, long long strideC, int batchCount);
```

## 需求拆解

1. 实现标准 API、参数检查和零维/零乘积处理。
2. 复用 `CgemmSbTiling` 与现有批量 GEMM 数据搬运、FP32 Cube 和结果合并代码，增加强制 ThreeM 选择。
3. 为小矩阵提供融合 SIMT 3m 计算，减少多 kernel 调度和临时数据读写。
4. 接入官方 CSV、Netlib BLAS golden、精度及确定性检查，采集 msprof 性能并形成可复现报告。

# 详细设计（required）

## 算子分析

### 数学公式

对第 i 个 batch，令 `X=op(A+i*strideA)`、`Y=op(B+i*strideB)`，有：

```text
P0 = Re(X) × Re(Y)
P1 = Im(X) × Im(Y)
P2 = (Re(X) + Im(X)) × (Re(Y) + Im(Y))
Re(Z) = P0 - P1
Im(Z) = P2 - P0 - P1
C_i = alpha × Z + beta × C_i
```

三条乘积路径均为实数 GEMM。通用路径在32≤K≤512时将FP32分量拆为高位与残差，复用补偿FP16乘法并以FP32累加；其他区间使用FP32乘积或下述长K分段路径。小尺寸SIMT路径按相同三积分解和固定K顺序累加。共轭操作在读取/打包时对虚部取负。

### 数据类型与存储

输入和输出均为 `aclblasComplex`（complex64，两个 FP32 分量），alpha/beta 为复数标量指针。矩阵按列主序存储，元素偏移如下：

| 矩阵 | N | T/C |
| --- | --- | --- |
| op(A) 的 (r,p) | `r+p*lda` | `p+r*lda` |
| op(B) 的 (p,c) | `p+c*ldb` | `c+p*ldb` |
| C 的 (r,c) | `r+c*ldc` | 不适用 |

每批基址再加 `i*strideX`。stride 单位为复数元素，使用有符号 int64 计算；零步长复用首矩阵，负步长允许调用方提供对应的合法起始地址。padding 和批次间空隙不参与计算、不写入输出。

## 算子实现

### Host 侧设计

新增声明放在 `include/cann_ops_blas.h`，入口放在 `blas/gemm/arch35/cgemm3m_strided_batched_host.cpp`，公共计算复用 `blas/gemm_strided_batched/arch35/`。入口传递强制 ThreeM 标志，普通复数批量 GEMM 不设置该标志。

参数检查顺序为 handle、枚举、非负维数/batchCount、前导维、alpha/beta，再处理计算分支：

| 场景 | 行为 |
| --- | --- |
| 空 handle | 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 非法 N/T/C 枚举、负维数、负 batchCount、非法 LD、空 alpha/beta | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| m=0、n=0 或 batchCount=0 | 返回 SUCCESS，不启动计算、不写 C |
| k=0 或 alpha=0 | 不读取 A/B，仅执行 beta×C |
| beta=0 | 不读取 C 旧值，写入计算结果或零 |
| 无乘积且 beta=1 | 保持 C 原值 |
| 有实际矩阵乘且 A/B 为空，或非空输出的 C 为空 | 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

该检查顺序覆盖官方 `m0_a_null`、`n0_b_null`、`alpha0_a_null` 等 SUCCESS 用例。`lda>=max(1, N?m:k)`、`ldb>=max(1, N?k:n)`、`ldc>=max(1,m)`；这些结构性检查先于 quick return。

`CgemmSbTiling` 传递维数、LD、三个有符号 stride、标量、对齐尺寸、tile、batch 分组与 workspace 偏移。Host 根据平台核心数分配任务，按维数、总计算量和临时存储成本选择小尺寸或通用路径。

### Kernel 侧设计

**小尺寸融合 SIMT 路径。** 将 batch 和输出元素映射到线程，每个线程负责独立输出元素并按固定 K 顺序执行三个 FP32 实数乘积累加，随后直接应用 alpha/beta 写回。复用已有 SIMT 入口与索引逻辑，不生成 Pack 和产品平面。Host按尺寸、批量和总计算量选择SIMT或Cube路径。

**小矩阵L1批量路径。** 对alpha=1、beta=0，M/N/K均为16倍数且不超过64、batch≥32的独立输出，复用已有L1环形槽和AIC:AIV=1:2批次流水线。AIV将实部、虚部、实虚和的高位与残差直接打包到L1；AIC用三个补偿实数GEMM生成P0/P1/P2，AIV再合并为复数C。每批只通过GM保存三个FP32乘积，不写全局打包输入。64³的每批L1数据为96KiB，每个AIV的256KiB区域使用两个128KiB槽，UB最多约145KiB。数值范围标志由各AIV归约，异常批次复用参考顺序计算。产品空间按3×M×N×batch个FP32分配，并计入既有2GiB预算。

**长 K 分段路径。** 对 alpha=1、beta=0、K≥4096 且输出可按64行×32列分块的矩阵，复用仓内Cube前缀与顺序SIMD后缀框架。前缀长度取K的2/3、按128向下对齐，并限制在2688以内。前缀的三个实数乘积各拆为FP16高位与残差交叉项，FP32累加并恢复复数结果；后缀逐K执行三个FP32实数乘法恢复复数乘积，按顺序加入前缀结果。N/T/C输入按需转换为列主序NN数据，包含共轭及前导维处理。独立批次在同一stream依次执行，单个4096方阵的临时空间约1.25GiB，包含打包数据、产品平面和归一化输入。超出既有2GiB工作区预算时使用完整顺序3m路径。SIMD的A/B双缓冲占192KiB、输出区16KiB，共208KiB；每次微核处理64行×4列，复用A向量及广播B值。输入超出补偿半精度的数值范围时复用参考顺序回退。

**通用 Pack→Cube→Combine 路径。**

1. Pack 按 N/T/C、LD、stride 读取交错复数数据，生成实部、虚部、实虚和三个分量；补偿路径保存各分量的FP16高位和残差。边界padding清零，A/B零步长只打包一次；数值范围和非有限值标志随Pack写出。
2. Cube 将 `(batch, product, tileM, tileN)` 映射到核心，分别计算 P0/P1/P2。FP32路径复用`GemmSbProcessTile`；补偿路径对每个实数乘积计算高位×高位、高位×残差、残差×高位，再以FP32合并。两条路径复用GM→L1→L0A/L0B、Mmad及Fixpipe。FP32长K分为128深度的部分乘积，由同一核心按固定顺序加入产品平面；不使用跨核心原子归约。超范围或非有限值按标志进入参考顺序回退。
3. Combine 由三个乘积恢复复数结果，融合复数 alpha/beta：`outR=ar*zR-ai*zI+br*cR-bi*cI`，`outI=ar*zI+ai*zR+br*cI+bi*cR`。beta=0 路径跳过旧 C 读取，按列主序只写有效 m×n 区域。

对m、n≤256且K≤512的补偿路径，复用仓内AIC:AIV=1:2的融合kernel，将Pack、三积Cube和Combine放在一次调度中，按既有跨核标志完成阶段同步。其余通用路径分阶段调度。

**输出别名。** 当 strideC=0，或合法调用产生跨 batch 的 C 区域重叠时，以 batch 索引升序在同一 stream 执行，保证后一批读取前一批更新后的 C。该路径按 K 顺序累加，N 形态先应用 beta，再逐项加入 alpha×B 与 A 的乘积。当 `|beta|>1` 时，重复输出形成放大递推，采用参考复数乘加顺序，避免三积恢复的舍入差异被跨 batch 放大。无重叠输出采用三积 GEMM，通用路径在一次调度中并行处理批次，长K分段路径按工作区预算复用同一缓冲。A/B 复用不改变输出顺序。

**确定性。** 每个输出归属于唯一计算任务，乘积、K 累加和 Combine 顺序固定；不采用竞争写、跨核心浮点原子累加或依赖运行时完成顺序的归约。两次验证从同一份初始 C 恢复后执行，并逐位比较。

### 分块与资源预算

分块与资源预算沿用现有 arch35 常量和同步协议。

| 资源 | 规划 |
| --- | --- |
| 基本 FP32 Cube tile | M=N=128、baseK=64，边缘块按有效尺寸裁剪 |
| L0A/L0B | 每侧 128×64×4B=32KiB；两个交替 bank 共64KiB，等待 M 释放后复用 |
| L0C | 128×128×4B=64KiB，放在现有128KiB结果槽预算内 |
| L1 | 现有总预算512KiB，分两个256KiB槽；每槽 A/B 的实际对齐块总和不得越界 |
| AIV UB | 沿用224KiB预算，包含Pack/Combine缓冲、尾块和临时标志；只保留当前阶段存活数据 |
| SIMT | 复用256线程配置及512 launch bound；结合编译结果检查寄存器与溢出 |

通用 FP32 3m 输入平面为三份 A 和三份 B，产品为三份 C。复用现有 workspace 布局时允许保留四份 C 的分配上界，以避免改变普通接口的布局；本接口只计算三份产品。各分区512B对齐，容量估算上界为 `4*(3*aSize+3*bSize+4*cSize+flagSize)` 字节，其中 size 为对齐后的FP32元素数。Host 检查 workspace 容量计算并按现有 workspace 接口分配；批次复用和分组布局沿用现有实现。

补偿路径沿用仓内高位/残差布局，输入分配按`6*aSize+6*bSize`个FP32存储单元计算。Host先核算完整空间，再分配打包、产品和标志区域；长K的归一化输入另外计入同一预算。

### 性能优化方案

- 小矩阵融合索引、3m和epilogue，优先降低启动与workspace流量。
- 大矩阵复用Cube分块和双缓冲搬运，用三个产品替代四个产品；独立任务按batch和输出tile展开。
- strideA/B=0复用打包数据；紧凑列主序使用连续搬运，padding及转置使用通用打包。
- 三积结果使用FP32累加，长K前缀复用高位与残差补偿；优化以官方逐case耗时和全部倍率均值衡量。

## 支持硬件

| 芯片 | 支持 |
| --- | --- |
| Atlas 950PR（arch35） | √ |

## 算子约束限制

支持列主序前导维及固定批次步长，不支持任意维度的非连续Tensor。C为in/out，不复用A/B存储；调用方保证地址及可访问范围合法。维数在调用时已知，不要求编译期未知shape。输入可含Inf/NaN，按任务书特殊值规则验证输出。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 判据 | 来源 |
| --- | --- | --- |
| 精度 | 实/虚FP32分量atol=rtol=2^-13；matched_ratio≥0.99；max_abs_error≤1e-2或32ULP | 任务书§3.2及生态混合容差标准 |
| 非有限值 | 同NaN、同号Inf、NaN与Inf按规定等效；有限/非有限不等效，单独统计 | 任务书§3.2第4条 |
| 确定性 | 相同初始输入两次输出逐位一致 | 任务书§2.5 |
| 单case性能 | GPU耗时/NPU平均耗时≥0.4 | 任务书§3.3 |
| L3平均性能 | 全部性能case倍率算术平均≥0.5 | 任务书§3.3 |

GTest读取官方1200条CSV（1000精度、200性能），golden逐batch调用Netlib BLAS 3.10.0 `cblas_cgemm`。精度检查覆盖全输出矩阵，分别统计实部/虚部的容差匹配、最大误差、非有限值等效情况；额外覆盖普通接口回归、空handle、有符号stride、数据分布和确定性。官方测试CSV及已回填的H100基线保留原件。

性能使用msprof op的Task Duration(us)，记录5次warmup后10个有效样本均值。多kernel路径将一次API调用的相关计算kernel耗时求和。报告列明200个case参数、GPU/NPU耗时、倍率、平均倍率、采样数量及截图，并保留原始profiler数据，按§3.3逐项展示。

## 兼容性分析

新增统一头文件API及arch35入口；共享tiling的ThreeM标志在既有接口初始化为0。已有普通复数GEMM、句柄、stream、workspace和其他架构接口保持各自行为，通过普通接口回归验证共享文件的影响。

# 当前 PR 是否有 AI 参与

- 是。
- AI Agent 平台：Codex。
- AI 模型：GPT-6。
- Prompt 上下文：本题官方任务书及随附CSV、精度与性能指导；ops-blas上游源码中的arch35复数固定步长批量GEMM、Pack/Cube/Combine、tiling及测试框架；要求复用现有工程完成标准3m接口、实现和自验证。
