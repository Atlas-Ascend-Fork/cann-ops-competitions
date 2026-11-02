# 需求背景（required）

## 需求来源

[10月社区任务-aclblasHgemmStridedBatched算子开发（950）](https://www.hiascend.com/activities/task-center/details/39ca031614a04829b74c47a68265db99)。实现提交至 [cann/ops-blas](https://gitcode.com/cann/ops-blas)，使用 Ascend C、CANN 9.1.0，适配 Ascend 950PR。

## 背景介绍

`aclblasHgemmStridedBatched` 对固定步长存储的多组半精度矩阵执行 GEMM，适用于具有相同形状和前导维的批量矩阵计算。与逐矩阵调用相比，批量调度可以减少 Host 下发并利用 batch 维增加并行度；固定步长无需构造设备指针数组。

基于 ops-blas 的句柄式 BLAS 接口，在 `blas/gemm/arch35/` 扩展半精度步长批处理，复用 GEMM 的分块、布局和流水结构。现有 FP32 GEMM 不能直接替代本接口，需要独立处理 FLOAT16 存储、FP32 累加、批次偏移及 FLOAT16 输出。采用 Kernel 直调工程，不新增 aclnn 接口。

# 需求分析（required）

## 需求描述与拆解

1. 提供与任务书一致的公共 API，支持 FLOAT16 输入、输出和 Host 标量，列主序及 N/T/C 全部组合。
2. 支持运行时 m/n/k/batchCount、合法前导维和非负 64 位元素步长；保持只读输入和 C 原地更新语义。
3. 正确处理参数错误、空输出、k=0、beta=0、padding 和非有限值，保持 handle stream 的异步执行方式。
4. 采用 FP16 乘法输入、FP32 累加与乘加后处理、最终 RNE 转为 FP16，按任务书执行精度与性能验证。

## 算子原型

在 `include/cann_ops_blas.h` 增加以下公共声明，与其他产品线共用，不定义 950PR 私有平行接口：

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

公共半精度类型 `aclblasHalf` 保存 16 位 IEEE 754 binary16 编码，拟在公共类型头统一定义为 `uint16_t`；若其他 Hgemm 接口先行引入该类型则直接复用。Host 使用半精度解码函数读取标量，不能将编码按整数数值转换为 float；Device 按 half 访问相同存储。

| 参数 | 位置和含义 | 约束 |
| --- | --- | --- |
| handle | Host 库句柄，提供 stream | 必须有效 |
| transa/transb | Host 转置属性 | N/T/C；实数 C 与 T 等价 |
| m/n/k、batchCount | Host int，输出形状、归约长度与批数 | 非负 |
| alpha/beta | Host FLOAT16 标量指针，全批共用 | 始终非空，包括 no-op |
| A/B | Device FLOAT16，只读 | 非空输出且 k>0 时非空，包括 alpha=0 |
| C | Device FLOAT16，原地输入输出 | 非空输出且 k>0 或 beta≠0 时非空；beta=0 不读取旧 C |
| lda | A 的物理列间距 | N 时≥max(1,m)，T/C 时≥max(1,k) |
| ldb | B 的物理列间距 | N 时≥max(1,k)，T/C 时≥max(1,n) |
| ldc | C 的物理列间距 | ≥max(1,m) |
| strideA/B/C | 相邻批次首地址的元素偏移，long long | 非负，不是字节数 |

不提供任意 Tensor stride 或维度广播。`strideA=0` 或 `strideB=0` 按地址公式复用同一只读矩阵；只读输入批次部分重叠也按实际地址读取。输出仅写各矩阵有效 m×n 区域，不改写列间 padding 或批次空隙。调用方提供足够容量，输入与输出不得发生破坏只读输入的别名。

## 参数校验与边界

依次检查 handle、转置枚举、m/n/k/batchCount、lda/ldb/ldc、strideA/B/C、alpha/beta 指针。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；其余非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。这些检查全部先于 no-op，因此 batchCount=0 且 strideA<0 仍报错。

结构校验通过后，m=0、n=0 或 batchCount=0 返回 SUCCESS，不访问 A/B/C。其余情况再检查矩阵指针：k>0 时 A/B/C 均须非空，不能因 alpha=0 免除任务书规定的 A/B 判空。k=0 不读取 A/B，C 非空时执行 `C=beta*C`；beta=0 直接置零，beta=1 可直接返回。alpha=0 的有效指针场景也使用缩放路径，避免无用 GEMM。所有退化路径只处理逻辑输出区域。

任务书的两项边界需要统一解释：

- §2.4/§2.5 的判空条件允许 k=0、beta=0、C=nullptr，而置零描述隐含输出地址存在。拟对该空指针组合返回 SUCCESS 且不访问内存；C 非空时仍置零。
- §3.5 对 strideC 的文字包含 0 和小于矩阵存储宽度的情况，但附件 `gen_csv.py::ld_stride` 明确说明 C 重叠写入未定义且不构造。拟沿附件与 cuBLAS 的独立输出矩阵约定，支持非重叠有效输出；需要支持输出重叠时，须进一步确定覆盖顺序及旧 C 的读取时点。输入步长 0 和输入重叠不受此项影响。

上述两项提交评审确认；不通过修改任务书或测试数据自行消除差异。

# 详细设计（required）

## 数学表达与存储映射

采用批次下标 b、行下标 r、列下标 c：

\[
C[b\,s_C+r+c\,ldc]\leftarrow
\operatorname{RNE}_{16}\left(\alpha\sum_{p=0}^{k-1}
A[b\,s_A+a(r,p)]B[b\,s_B+d(p,c)]+\beta C[b\,s_C+r+c\,ldc]\right).
\]

其中求和和标量后处理采用 FP32，输入 FP16 可精确提升为 FP32；地址函数为：

\[
a(r,p)=\begin{cases}r+p\,lda,&transa=N\\p+r\,lda,&transa=T/C\end{cases},\qquad
d(p,c)=\begin{cases}p+c\,ldb,&transb=N\\c+p\,ldb,&transb=T/C\end{cases}.
\]

beta=0 时不计算或装载旧 C 项，避免旧值 NaN/Inf 污染结果。中间乘积不提前转为 FP16；仅最终输出舍入，避免大 k 与相消场景的额外误差。

总计算量约为 `2*batchCount*m*n*k`。以 batch 与二维输出 tile 共同形成工作空间：

\[
Q=batchCount\lceil m/B_M\rceil\lceil n/B_N\rceil.
\]

核心按步长遍历 Q，每个任务沿完整 K 轴归约并独占一个输出块。小矩阵利用批次并行，大矩阵同时切分 m/n；常规路径不依赖跨核原子累加。地址、元素跨度、分块数量及 workspace 大小使用宽整数，乘加前检查可表示性，避免先在 int 中溢出。

## Host 侧设计

1. 完成校验和 no-op 返回，读取 Host 标量并将值写入调度参数。
2. 根据形状、batchCount、转置、前导维、步长对齐和设备资源选择缩放、小矩阵或 Cube 路径；不按用例编号、随机种子或预知数据分派。
3. 计算输出分块、K 分段、尾块与缓冲布局。复用句柄 workspace 机制，检查容量后再下发；避免逐 batch 申请内存或循环启动 GEMM。
4. 所有计算阶段在 handle 绑定的 stream 上执行，不将矩阵复制到 Host 计算，不插入无必要的设备同步。调用方在读取结果或释放输入前完成 stream 同步。

## Kernel 侧设计

**Cube 主路径。** 使用 `C^T=op(B)^T*op(A)^T` 的行主序计算视角，交换内部 A/B 和 m/n，外部仍保持列主序。复用 GEMM 模板组织 FP16 的 GM→L1→L0A/L0B→Cube、FP32 L0C 累加及输出链路；按转置和 leading dimension 建立布局，尾块只搬有效数据并对运算填充区域补零。FP16 的布局粒度与 FP32 不同，不能只替换数据类型而沿用其字节偏移。

沿 K 分段装载面板，在片上容量允许时对 L1/L0 使用双缓冲，重叠搬运与 Cube 计算。缓冲重用前确保上一消费者完成，输出前确保全部 K 段累加结束。beta=0、alpha=1 的兼容布局可直接输出 FP16；一般 alpha/beta 优先在块内完成 FP32 后处理，必要时先输出 FP32 workspace，再由同 stream 的 AIV 完成乘加与 FP16 回写。该分阶段路径的全部耗时计入一次算子调用。

**小矩阵与缩放路径。** 小矩阵比较 Cube 填充及启动成本与 AIV/SIMT 直接计算成本，按通用形状和批数选取。AIV/SIMT 路径直接使用地址公式并进行 FP32 累加；k=0 或 alpha=0 时只做 C 的缩放/置零。多个批次合并下发，非对齐输出按有效范围写回，避免覆盖相邻输出。

**步长处理。** Kernel 用 64 位 `b*strideX` 定位本批次，不把 stride 截断为 int。只读矩阵可重复使用；对非对齐首地址、leading dimension 和尾块采用目标 SDK 支持的搬运方式，必要时仅整理当前面板，不默认生成整个 batch 的转置副本。

## 硬件与工程组织

| 项目 | 位置或支持范围 |
| --- | --- |
| 产品 | Ascend 950PR，arch35，CANN 9.1.0 |
| 公共 API/类型 | `include/cann_ops_blas.h`、`include/cann_ops_blas_common.h` |
| Host 与 Kernel | `blas/gemm/arch35/`，复用 GEMM 模板 |
| 专项测试 | `test/gemm/hgemmStridedBatched/arch35/`，CSV 与 C++ GTest |
| 使用说明 | GEMM README 中补充 API、支持产品、约束和测试入口 |

# 可维可测分析

## 精度标准与测试方法

按任务书附件的 1000 条精度与 200 条性能用例进行自验。Golden 使用 Netlib BLAS 3.10.0 的 `cblas_sgemm` 逐 batch 计算，输入和标量从 FP16 提升为 FP32，输出按 IEEE 754 RNE 转为 FP16。分别核对每个 batch 全部有效元素，并用哨兵检查 padding 和批次空隙。

有限值逐元素满足 `|actual-golden|≤2^-10+2^-10*|golden|`，匹配比例≥0.99。最大绝对误差上限为 `max(0.1, 32*ULP(golden))`，其中 golden 取最大绝对误差所在位置的 FP16 参考值，遵循任务引用的固定版本精度标准。自测分别记录最大绝对误差、该位置的 golden 和上限、ULP 距离及匹配比例；全局最大 ULP 距离仅用于诊断，不替代上述判据，也不把 CSV 的 MERE/MARE 字段当作完整 checker。非有限值单独统计：同为 NaN、同号 Inf、NaN 与任意 Inf 视为等效；异号 Inf、有限与非有限配对失败；非有限值不加入有限值统计，并须全部通过特殊值判定。无有限元素时仅检查特殊值集合。

覆盖九种转置组合、非方阵、奇数尾块、批数、前导维 padding、步长 padding、输入零步长与重叠、零尺寸、k=0、alpha=0、beta=0/1、大标量、相消和 Inf/NaN。beta=0 专门使用旧 C 含 NaN 的输入；负向用例核对状态码和校验顺序，不启动非法内存访问。大 k 不通过时保留实际误差和 FP32 累加证据，不放宽阈值。

## 性能标准与测试方法

在 Ascend 950PR、CANN 9.1.0 上逐项对齐附件 `gpu_baseline.csv` 的 200 条性能记录，核对转置、形状、批数、leading dimension 和步长，将 gpu_ms 换算为 us。任务书 §3.3 要求每项 `T_GPU/T_NPU≥0.4`，且全部倍率的算术平均≥0.5；两项同时检查，不能用聚合成绩掩盖逐例退化。

使用官方 `msprof op` 采集 task duration，预热 5 次后获取有效样本，给出 §3.3 规定的 10 次算术平均。一次调用含多个 Kernel 时先汇总完整调用的阶段耗时，异常调用单独记录。官方 `msopprof` 用于分析搬运、Cube/Vector 流水及核间负载；GTest 整体耗时包含分配、Golden 与比对，不能替代算子 task duration。

任务书 §7.4 另写“有效采样>10次、达到≥GPU数据”，附件脚本注释还出现“>50次”。采样次数按负责人在[任务讨论帖](https://gitcode.com/cann/ops-blas/discussions/61)中的回复，统一采用工具预热 5 次后连续 10 个有效样本取算术平均。性能表同时列出 §3.3 门槛与逐例≥1.0倍率的结果，开发以较严格目标推进；“达到≥GPU数据”采用逐例还是平均倍率的含义仍请评审明确，不将其视为已获确认。

## 兼容性分析

新增公共声明与 arch35 实现，不改变既有 GEMM API、状态码数值和其他产品线实现。公共类型由相关 Hgemm 接口共享，避免重复定义。沿用仓内依赖、构建和测试注册方式，检查专项构建确实包含新入口，不能将被跳过的测试视为通过。
