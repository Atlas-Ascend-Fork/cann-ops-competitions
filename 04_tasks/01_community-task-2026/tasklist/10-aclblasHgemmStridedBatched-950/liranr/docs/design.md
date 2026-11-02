# 需求背景（required）

## 需求来源

2026 年 10 月社区任务“aclblasHgemmStridedBatched 算子开发（950）”，贡献者：liranr。依据该任务书，在 ops-blas 仓补齐 Atlas 950PR 上的半精度固定步长批量矩阵乘能力。

## 背景介绍

### aclblasHgemmStridedBatched 算子功能

该算子对一组形状相同的矩阵执行通用矩阵乘加。A、B、C 分别位于各自的单一 Device 缓冲区，相邻批次通过固定元素步长定位，所有批次共用转置模式、尺寸、前导维和标量 alpha、beta。输出原地写回 C，可用于多组小矩阵的并行计算。

与逐批从 Host 调用普通 GEMM 相比，固定步长批处理可以在一次任务调度中分配多个批次的计算块，减少重复启动和参数处理开销。实现采用 Ascend C kernel 直调方式，复用 ops-blas 同族 GEMM 模板，通过 handle 绑定的 stream 执行。

# 需求分析（required）

## 需求描述

新增公共接口 `aclblasHgemmStridedBatched`，支持 FLOAT16 输入、输出及 Host 标量；支持列主序矩阵、N/T/C 转置模式、前导维 padding、固定批次步长、运行时尺寸和批次数。实数的 C 模式与 T 模式等价。

接口声明放入 `include/cann_ops_blas.h`，实现放入 `blas/gemm/arch35/`，测试放入 `test/gemm/hgemmStridedBatched/arch35/`。共用 ops-blas 的句柄、状态码、构建和测试框架，不新增 950PR 私有平行接口。

## 需求拆解

1. 实现公共接口及参数校验，覆盖 no-op、k=0、非法枚举、非法尺寸、前导维、步长和空指针。
2. 基于 GEMM 模板实现批次与矩阵分块联合调度，保证转置、padding、尾块及 C 原地更新的正确性。
3. 使用 FP32 累加，按 FLOAT16 混合容差标准验证精度；beta=0 时不读取 C 旧值。
4. 完成任务提供的 1000 条精度、200 条性能用例，补充参数边界、特殊值和输入分布覆盖。
5. 通过 msprof op 采集内核耗时，按任务书性能判据验证，并提供可复现的测试步骤、报告和日志。

# 详细设计（required）

## 算子分析

### 数学公式

对批次 b，0 ≤ b < batchCount：

```text
A_b = A + b * strideA
B_b = B + b * strideB
C_b = C + b * strideC
C_b = alpha * op(A_b) * op(B_b) + beta * C_b
```

步长以 FLOAT16 元素为单位。列主序下，逻辑输出元素 `(r, c)` 的地址为 `C + b*strideC + r + c*ldc`。乘法输入的元素偏移如下：

| 输入 | N 模式 | T/C 模式 |
| --- | --- | --- |
| op(A_b)[r, p] | b*strideA + r + p*lda | b*strideA + p + r*lda |
| op(B_b)[p, c] | b*strideB + p + c*ldb | b*strideB + c + p*ldb |

地址计算和批次偏移使用 64 位整数，避免先以 32 位执行乘法再转换造成溢出。Host 对偏移、存储跨度及字节数计算做溢出检查。

### 支持数据类型

A、B、C、alpha、beta 均为 FLOAT16。alpha、beta 从 Host 指针读取并作为启动参数传入，乘加与结果缩放采用 FP32 中间计算，最终转换为 FLOAT16 写回。

### 支持形状

`op(A_b)` 为 m×k，`op(B_b)` 为 k×n，C_b 为 m×n。所有批次形状一致，m/n/k/batchCount 为运行时参数，不要求固定编译期 shape。

| 参数 | 约束 |
| --- | --- |
| m、n、k、batchCount | 非负整数 |
| lda | N 模式 ≥ max(1,m)，T/C 模式 ≥ max(1,k) |
| ldb | N 模式 ≥ max(1,k)，T/C 模式 ≥ max(1,n) |
| ldc | ≥ max(1,m) |
| strideA、strideB、strideC | 非负 long long，单位为元素 |

测试覆盖任务规定的尺寸上限 16384 和 batchCount 上限 65536，分别构造可容纳的边界用例，不将各参数最大值机械组合成不可分配的缓冲。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

公共接口使用任务书定义的参数序列：

```cpp
aclblasStatus_t aclblasHgemmStridedBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda, long long strideA,
    const aclblasHalf* B, int ldb, long long strideB,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc, long long strideC,
    int batchCount);
```

**参数检查与分支：**

1. 检查 handle、转置枚举、m/n/k、batchCount、lda/ldb/ldc、strideA/B/C 及 alpha/beta 指针；在 no-op 返回前完成这些检查。分别返回任务指定的 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、`ACLBLAS_STATUS_INVALID_ENUM` 或 `ACLBLAS_STATUS_INVALID_VALUE`。
2. m=0、n=0 或 batchCount=0 时返回成功，不读取 A/B/C，不启动 kernel。例如 batchCount=0 且 strideA<0 仍须报错。
3. 非空计算且 k>0 时校验 A/B/C，alpha=0 也不跳过任务规定的 A/B 判空。beta=0 允许 C 旧值无效，但仍需有效的输出地址。
4. k=0 时不访问 A/B；beta=0 写零，beta=1 保持 C，其他值执行缩放。针对任务表格对 k=0、beta=0 时 C 判空描述与“写零”要求的差异，在接口约束中列为待澄清项，不对空地址执行写操作。
5. 使用 handle 中的 stream 启动内核，沿用仓库错误处理机制；正常计算路径不额外执行设备级同步。

**分块与分核：**

将每个输出矩阵划分为 BM×BN 的块，沿 K 维按 BK 分段累加。令 `tilesM=ceil(m/BM)`、`tilesN=ceil(n/BN)`，将 `batchCount*tilesM*tilesN` 个输出块展平分配给计算核。由块序号反解批次 b 和矩阵内块坐标，使一个输出块由一个计算任务独占，避免对 C 的原子累加。

小矩阵、大 batch 场景优先跨批次分核；大矩阵场景同时利用批次和 M/N 块并行。任务较多时，每核按核数步进领取后续块。依据矩阵尺寸、转置组合、对齐情况和平台资源选择模板分块参数，具体 BM/BN/BK 和启动核数通过 950PR 实测确定，不按测试 case ID 分派。

**模板与资源复用：**

复用 arch35 GEMM 模板的矩阵加载、Cube 计算及同步机制，扩展批次基地址计算。按模板要求约束 L1/L0 等片上资源用量；在资源允许时使用流水和双缓冲。优先直接加载带 leading dimension 的矩阵，不为每批矩阵额外创建全量转置副本。转置组合归并为 NN、NT、TN、TT 四类，C 模式映射为 T。

#### 3.2.2 kernel侧设计：

1. **初始化与定位：**读取尺寸、前导维、步长、分块和标量，确定当前任务的 batch 及输出子块，建立 A/B/C 的批次基地址。
2. **输入搬运：**根据转置模式和 leading dimension 加载当前 K 分块。M/N/K 尾块通过模板支持的尾块处理和补零参与计算，不读取 padding 之外的无效地址。
3. **矩阵乘累加：**使用 Cube 完成 FLOAT16 乘法、FP32 累加，累加器跨 K 分块保留，避免每次分块后转回 FLOAT16。
4. **结果处理：**在 FP32 域计算 alpha*acc；仅 beta≠0 时读取 C 旧值并计算 beta*C。beta=0 独立分支避免 `0*NaN` 污染输出。alpha=0 的快捷路径是否使用，需与 Inf/NaN 对照测试确认后决定。
5. **输出写回：**转换为 FLOAT16，只写有效 m×n 元素，保持 leading dimension padding 和批次间隙不变。A/B 始终只读。
6. **退化分支：**k=0 使用轻量缩放/置零路径，按 batch、列和有效行定位；beta=1 可以直接返回。该路径不加载 A/B，也不调用矩阵乘内核。

### 步长与存储处理

输入 A/B 只读，按实际元素步长寻址。依据配套测试说明，设计支持 strideA/B=0 复用输入及输入批次读区间重叠，不进行一般张量维度广播。缓冲跨度由最后批次偏移和单矩阵实际访问跨度计算，不简单使用 batchCount*m*n。

并行输出路径要求各批次 C 的实际写入区域互不重叠。任务正文“各 batch 独立”与 strideC 覆盖规则、配套测试对重叠输出的说明存在差异，因此不将 strideC=0 且 batchCount>1 的重叠写入作为已定义的并行计算行为；具体返回或支持规则按任务澄清结果补齐。不得将未定义的重叠写入跑通视为正确性通过。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 950PR（arch35），CANN 9.1.0 | √ |

## 算子约束限制

仅支持 FLOAT16、列主序及固定批次步长，不支持任意维度的非连续张量或不同批次具有不同形状。C 原地输出，A/B 与输出缓冲不应发生会破坏输入的别名重叠。API 为公共接口，本任务实现范围为 950PR，不改变其他平台已有实现。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述（不涉及说明原因） | 标准来源 |
| --- | --- | --- |
| 精度标准 | FLOAT32 Netlib BLAS 3.10.0 `cblas_sgemm` 逐 batch 生成 golden，再就近舍入为 FLOAT16；rtol=atol=2^-10，matched_ratio≥0.99，最大绝对误差按 0.1 或 32 ULP 条款判定 | 任务书 §3.2 |
| 性能标准 | 每条 GPU耗时/NPU耗时≥0.4，全部性能 case 倍率算术平均≥0.5；使用 msprof op task duration，单位 us | 任务书 §3.3 |
| 内存标准 | 无独立内存性能验收；仍检查越界、padding、输入只读及资源释放 | 任务书 §3.4 |

**精度与功能验证：**

- 执行附件全部精度用例，覆盖转置全组合、不同 batchCount、前导维及步长组合。补充尺寸和批次上限、非法参数、no-op、k=0、alpha/beta 特殊值、beta=0 且 C 为 NaN 等场景。
- 按任务书生成均匀/正态分布各占 50% 的随机输入，固定并记录种子；以实际共享的底层存储构造 strideA/B=0 或输入重叠用例，保证 golden 读取相同数据。
- 每个 batch 的有效输出独立验证，检查输入和输出 padding 的哨兵值。Inf/NaN 按 §3.2 单独判定和记录，不混入有限值误差统计；全为非有限值的情况避免除零。
- 对大 K 的误差问题记录累加方式、golden 和误差变化，不自行放宽阈值。测试执行异常、超时、缺失用例或无结果均不得作为通过。

**性能验证：**

执行附件全部 200 条性能用例，与 GPU CSV 按完整参数对应。按 §3.3 预热 5 次后采集 10 次有效样本，保存原始采样及均值；保持每次测量的输入条件一致，避免 beta≠0 时反复覆盖 C 导致后续输入变化。数据恢复与同步不计入目标内核耗时，多内核实现须完整记录其执行耗时，不使用 GTest 总运行时间代替内核时间。

逐条报告 GPU/NPU 倍率和全体倍率算术平均，不以平均耗时之比替代平均倍率，不遗漏失败或慢用例。分别观察小矩阵启动开销、大 batch 负载均衡、转置加载和大矩阵吞吐，再调整模板分块与流水配置。

任务书 §7 的“有效采样 >10 次”“达到 ≥GPU数据”与 §3.3 的采样和倍率条款不一致；本设计引用具体的 §3.3 作为当前测试方案，提交验收前须依据官方澄清统一测试及报告口径，不将冲突条款默认为豁免。

## 兼容性分析

保持任务书规定的公共函数签名和 ops-blas 状态码，复用现有 handle/stream 管理，不改变其他 GEMM 接口行为。实现、测试和示例均接入仓库现有工程；README 标注 Atlas 950PR 支持范围和运行步骤。交付设计文档、实现及测试代码、自验证步骤、精度/性能报告和日志，记录实际环境和代码版本，便于复现。
