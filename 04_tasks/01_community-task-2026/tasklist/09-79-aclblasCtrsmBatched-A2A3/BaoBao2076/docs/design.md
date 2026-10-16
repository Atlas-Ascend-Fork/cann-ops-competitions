# aclblasCtrsmBatched 算子设计文档（Atlas A2/A3）

| 文档版本 | 作者/团队 | 适用产品 |
| --- | --- | --- |
| V1.0 | BaoBao2076 | Atlas A2/A3 系列产品，含 Atlas 800I A2/A3 |

# 一、需求背景

## 1.1 需求来源

本设计对应 2026 年 9 月社区任务第 79 项：aclblasCtrsmBatched 算子开发（A2/A3）。基于 ops-blas 和 Ascend C 实现 complex64 批量三角求解，接口语义对齐 cuBLAS cublasCtrsmBatched，精度标杆采用 Netlib CBLAS ctrsm。

## 1.2 算子功能

每个 batch 独立求解，解矩阵 X 原地覆盖输入矩阵 B：

- LEFT：op(A[i]) × X[i] = alpha × B[i]。
- RIGHT：X[i] × op(A[i]) = alpha × B[i]。

op(A) 支持原矩阵、转置和共轭转置。矩阵采用列主序，所有 batch 共享维度、前导维和属性。A、B 为设备侧指针数组，各矩阵地址可以独立分配。

# 二、需求分析

## 2.1 接口定义

~~~cpp
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m, int n,
    const aclblasComplex* alpha,
    const aclblasComplex* const A[], int lda,
    aclblasComplex* const B[], int ldb,
    int batchCount);
~~~

| 参数 | 语义与约束 |
| --- | --- |
| handle | 有效的 ops-blas 句柄，计算使用句柄绑定的 stream |
| side | LEFT 或 RIGHT；A 的阶数 k 分别为 m 或 n |
| uplo | UPPER 或 LOWER，只引用指定三角 |
| trans | N、T 或 C，分别表示不转置、转置、共轭转置 |
| diag | UNIT 或 NON_UNIT；UNIT 视对角为 1，跳过对角读取 |
| m、n | B 的行数和列数，均大于等于 0 |
| alpha | Host complex64 标量指针 |
| A | 只读的设备指针数组；每个矩阵按 lda × k 存储 |
| lda | 大于等于 max(1, k) |
| B | 设备指针数组；每个矩阵按 ldb × n 存储，输出原地覆盖有效区域 |
| ldb | 大于等于 max(1, m) |
| batchCount | 大于等于 1 |

## 2.2 边界与异常语义

- 空 handle 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR。非法枚举、负维度、非法前导维及其他非法参数返回 ACLBLAS_STATUS_INVALID_VALUE。
- m=0 或 n=0 为合法 no-op。按公共接口约定完成基本参数校验后返回成功，不下发计算。
- alpha=0 时只清零 B 的有效区域，跳过 A 和原 B 数值读取。B 的 padding 保持不变。
- NON_UNIT 要求被引用的对角元非零。接口不执行奇异性检测。
- 各 B[i] 的有效存储区域相互独立。调用方在 stream 完成前保持输入、输出和句柄有效。
- 指针数组整体判空属于接口校验；数组元素级 Host 判空按任务书列为可选增强，不将同步读回数组作为常规计算前置步骤。

## 2.3 工程组织与依赖

| 模块 | 职责 |
| --- | --- |
| 公共接口 | 在 include/cann_ops_blas.h 的 C ABI 区域声明接口 |
| Host/Tiling | 参数校验、平台查询、任务划分、workspace 规划与 kernel 下发 |
| Kernel | 数据转换、复数三角递推、块更新与原地写回 |
| 测试 | CSV 用例、Netlib Golden、NPU 调用及精度与性能判定 |

算子实现纳入 blas/trsmbatched/arch22，测试遵循 test/trsmbatched/ctrsmbatched 的仓库结构。工具链使用 CANN 9.1.0，A2/A3 由同一份数学规范和源码组织适配。

# 三、详细设计

## 3.1 总体流程

~~~mermaid
flowchart TD
    A[接口与参数校验] --> B{空维度或 alpha 为零}
    B -->|空维度| C[返回成功]
    B -->|alpha 为零| D[清零 B 有效区域]
    B -->|常规输入| E[统一求解方向并规划任务]
    E --> F[选择直接递推或分块路径]
    F --> G[按依赖顺序求解并更新]
    G --> H[原地写回 B]
~~~

Host 只处理元数据与 Host 标量。设备指针数组由 kernel 读取。数据转换、求解、更新及写回均进入同一 stream，阶段间利用 stream 顺序或显式事件保证依赖。

## 3.2 求解方向与坐标映射

LEFT 直接求解 op(A)X=alpha B。RIGHT 通过转置关系转换为 op(A)^T X^T=alpha B^T。共轭转置的坐标交换与虚部取反分别表达，避免把转置与共轭混用。

内部统一记录有效三角方向、转置和共轭标志。下三角按前向依赖处理，上三角按后向依赖处理。原始列主序地址使用 row + col × leadingDimension，乘法采用足够宽的整数中间量。

## 3.3 Host 与 Tiling

TilingData 记录维度、前导维、batchCount、属性、Host alpha、行块和 RHS 分组、尾块长度及 workspace 偏移。RHS 表示右端向量组，即可独立求解的列或转置后的行。

任务首先沿 batch 和 RHS 分组拆分。批量较小时增加 RHS 并行；批量较大时优先复用同一矩阵系数。tile 大小由 UB、L1、L0 容量、数据类型和对齐要求共同约束。

路径选择只依赖矩阵几何、属性、资源容量与芯片能力。相邻边界尺寸采用相同规则。实现检查所有 workspace 字节计算的加法与乘法溢出，分配或容量检查失败时向上返回状态。

## 3.4 直接递推路径

小阶矩阵及需要保留递推顺序的场景采用向量求解。一个任务负责独立的 batch/RHS 分组，系数块与右端块保存在片上缓冲中，按三角依赖顺序处理 pivot。

每步先计算当前解，再更新尚未求解的右端。UNIT 跳过对角读取与除法。NON_UNIT 执行复数除法，针对实部或虚部占优选择缩放形式，避免直接平方导致额外的溢出或下溢。

复数更新以 FP32 实部和虚部分量执行。递推、乘加和特殊值处理遵循统一数值约定；低精度转换与重排累计不得替代精度验证。

## 3.5 分块求解与矩阵更新

对适合分块的系统，将三角矩阵划分为对角 panel 和非对角块。panel 表示一段连续的主元行或列。以有效下三角为例：

~~~text
R_p = alpha × B_p - Σ(q<p) T_pq × X_q
T_pp × X_p = R_p
~~~

向量核完成复数布局处理与对角 panel 求解。Cube 核承担非对角矩阵乘更新。对角块可直接求解，或在满足精度要求的路径中构造局部逆后批量应用。局部逆仅在当前调用内复用。

更新可按两个通用方案组织：求解一个 panel 后更新后续块，或求解当前 panel 前合并所有已解块贡献。选择时比较任务并行度、矩阵乘数量、长 K 累加开销和工作空间，不只比较单个矩阵乘耗时。

若一个核独占同一 batch/RHS 组，则可在一次 kernel 内连续推进多个 panel。存在跨核生产与消费时，使用明确的完成事件和一致的任务映射，消费方在数据可见后再读取。

## 3.6 复数布局与存储规划

复数乘法可展开为实数块矩阵：

~~~text
[ real(Y) ]   [ real(A)  -imag(A) ] [ real(X) ]
[ imag(Y) ] = [ imag(A)   real(A) ] [ imag(X) ]
~~~

布局转换只处理后续计算会消费的块。对角块与非对角块分别管理，尾块补零并保持有效尺寸信息。实数展开增加存储与搬运开销，只有整体收益覆盖转换和回写成本时才选择该路径。

| 存储层级 | 内容与生命周期 |
| --- | --- |
| GM workspace | 必要的布局转换块、局部逆、右端暂存及中间解；仅在本次调用内使用 |
| UB | 复数实虚分离、panel 递推、索引、mask 与输出暂存 |
| L1/L0 | Cube 操作数 tile 和累加器，按目标架构实际支持的数据通路规划 |

每个缓冲记录字节数、对齐、写入者、读取者和最后使用位置。GM 写回完成后才允许依赖读取。向量 mask、事件与寄存器状态在正常和提前退出路径恢复。

## 3.7 搬运、尾块与异步语义

按矩阵连续方向批量搬运，避免逐元素 GM 标量访问。非对齐行使用有效长度与 stride 描述，输入补零只作用于片上或独占 workspace 区域。写回严格限定 B 的 m × n 有效区域。

API 返回成功表示任务已提交。调用方同步句柄 stream 后读取结果。workspace 复用遵守句柄生命周期及同一 stream 的顺序，跨调用保留 A/B 数据不属于本设计。

## 3.8 性能设计原则

1. 小尺寸控制 kernel 数和核数，降低启动与空任务成本。
2. 多个 RHS 共享 A tile，提高系数复用率。
3. 优先消除重复转换、无效清零、细碎搬运和多余矩阵乘下发。
4. 以完整接口数据流评估性能，覆盖转换、求解、更新、同步与回写。
5. 双缓冲与并行调度必须同时满足容量、数据依赖和精度约束。

# 四、可维可测分析

## 4.1 测试覆盖

| 分类 | 验证内容 |
| --- | --- |
| 属性 | LEFT/RIGHT × UPPER/LOWER × N/T/C × UNIT/NON_UNIT 共 24 组 |
| 维度 | 零维、1、小质数、非方阵、2 的幂及相邻尺寸、尾块和大尺寸 |
| 数据 | alpha 为零、单位值、负值及复数；随机、抵消、Inf/NaN 等规格允许输入 |
| 存储 | 独立设备数组、leading dimension padding、未使用三角及 UNIT 对角填 NaN |
| 接口 | 空 handle、空指针、非法枚举、负维度、非法前导维和 batchCount |
| 资源 | workspace 容量、保护区、异步 stream、内存插桩与同步状态 |

## 4.2 精度标准

使用 Netlib CBLAS ctrsm 逐 batch 生成 Golden。实部与虚部分别采用 atol=rtol=2^-13 的混合容差，匹配率不低于 0.99，最大误差按任务书规定的 1e-2 或 ULP 条件判定。NaN/Inf 按生态精度标准单独比较。

完整测试包含原始 1000 条精度用例与 200 条性能用例。性能用例同样先做 Golden 比对，禁止通过修改输入、容差或 Golden 掩盖差异。

## 4.3 性能与资源验证

在指定设备和 CANN 版本下预热后采样，正式报告提供超过 10 次有效采样的平均值。使用 msprof 关联同一次接口调用的全部 kernel，完整保留数据转换和回写阶段。逐条与原始 GPU baseline 对应，目标为 NPU 平均耗时不高于 gpu_ms × 1000 / 0.8，单位为微秒。

内存检测采用匹配工具链的独立插桩构建，检查实际扫描的 kernel 数、错误和告警。保护区测试用于补充验证。性能、精度与资源检测均绑定同一待验收源码版本。

## 4.4 兼容性与交付

复用 ops-blas 句柄、状态码、workspace 与构建机制。公共 ABI 保持参数顺序和类型一致。A2/A3 平台差异集中在平台查询和 tiling，数学语义保持一致。

交付包括设计文档、公共接口与实现、正式测试、调用说明及自测报告。设计文档描述需求、算法和验证方法；自测报告独立记录对应源码版本的验证结论。
