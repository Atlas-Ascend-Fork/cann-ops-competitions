# 需求背景

## 需求来源

[10月社区任务-aclblasGemmBatchedEx算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/d945b8eabdc44fa98939d23f5cef0517)要求基于 [ops-blas](https://gitcode.com/cann/ops-blas)，使用 Ascend C 补齐 Atlas A2/A3 的批量 GEMM Ex 能力。接口语义参考 [cuBLAS cublasGemmBatchedEx](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmBatchedEx)，验收范围以任务书及随附测试用例为准。

## 背景介绍

算子对一组形状相同、地址独立的矩阵执行乘加。矩阵采用列主序，通过 Device 指针数组传入，不要求不同 batch 连续存放。各 batch 共享维度、转置方式、前导维和标量系数。实现复用 `include/cann_ops_blas.h` 中的公共声明，在 `blas/gemm_batched_ex/arch22/` 增加 Host 与 Kernel，不改动 arch35 接口或实现。

# 需求分析

## 功能与接口

对批次 `b∈[0,batchCount)` 计算：

\[
C_b(r,c)\leftarrow\alpha\sum_{p=0}^{k-1}\operatorname{op}(A_b)(r,p)\operatorname{op}(B_b)(p,c)+\beta C_b(r,c),
\quad 0\le r<m,\;0\le c<n.
\]

`op(X)` 在 N 模式取原矩阵，在 T 模式取转置，在 C 模式取共轭转置。实数路径中 C 与 T 等价。`op(A)` 为 `m×k`，`op(B)` 为 `k×n`，输出 C 为 `m×n`，原地覆写。

```cpp
aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const void* alpha,
    const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta, void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo);
```

| 参数 | 语义与范围 |
| --- | --- |
| handle | Host 上的有效句柄，使用其绑定的 stream。 |
| transa、transb | `ACLBLAS_OP_N/T/C`。 |
| m、n、k、batchCount | 非负 int，运行时传入。 |
| alpha、beta | 非空 FP32 标量指针，支持 Host 或 Device 内存。Device 值在绑定 stream 中读取。 |
| Aarray、Barray | Device 上的矩阵指针数组；每项指向对应 batch 的只读 Device 矩阵。 |
| Carray | Device 上的输出指针数组；每项指向对应 batch 可写的 Device 矩阵。各输出矩阵不重叠。 |
| Atype、Btype、Ctype | 本设计的 arch22 支持组合为 `ACL_FLOAT/ACL_FLOAT/ACL_FLOAT`。 |
| computeType、algo | 支持 `ACLBLAS_COMPUTE_32F` 和 `ACLBLAS_GEMM_DEFAULT`；其余组合返回 `NOT_SUPPORTED`。 |
| lda | N 模式下至少 `max(1,m)`；T/C 模式下至少 `max(1,k)`。 |
| ldb | N 模式下至少 `max(1,k)`；T/C 模式下至少 `max(1,n)`。 |
| ldc | 至少 `max(1,m)`。只写有效 m 行，保留列间 padding。 |

FP16、BF16、FP8、INT8 等扩展组合不属于本次 arch22 支持集，返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；README 按产品分别标明支持范围，保留 arch35 原有能力。无需广播、额外 stride、视图返回或确定性保证。

## 参数校验与边界行为

Host 先校验参数，再判断 no-op。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法转置返回 `ACLBLAS_STATUS_INVALID_ENUM`；负维度、非法前导维、空 alpha/beta，以及 `batchCount>0` 时为空的 Aarray/Barray/Carray 返回 `ACLBLAS_STATUS_INVALID_VALUE`。不支持的数据类型、计算类型或算法返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

上述指针校验检查数组本身，不在 Host 解引用 Device 指针数组。参与计算的矩阵内存须有效；即使矩阵乘被跳过，`batchCount>0` 时三个数组本身仍须非空。

| 条件（完成参数校验后） | 行为 | 访存要求 |
| --- | --- | --- |
| m=0、n=0 或 batchCount=0 | 返回成功，不下发计算。 | 不读取矩阵或标量值。 |
| k=0 或 alpha=0，beta=1 | C 保持原位模式。 | 不读取 A/B 矩阵，不写 C。 |
| k=0 或 alpha=0，beta=0 | C 有效元素写正零。 | 不读取 A/B 或旧 C。 |
| k=0 或 alpha=0，其余 beta | 逐元素执行 FP32 `C=beta*C`。 | 不读取 A/B，保留 C padding。 |
| 常规乘加，beta=0 | 仅写 `alpha*op(A)*op(B)`。 | 不读取旧 C，避免 `0*Inf/NaN` 污染。 |

`alpha=0` 的缩放分支按 EXACT 校验：beta=1 不写回，beta=0 写零，其余 beta 只做一次 FP32 乘法，不引入与零的加法或融合乘加。Device 标量也在 Kernel 中按该顺序处理，不能先搬 A/B 再判断 alpha。接口按 stream 异步执行，结果读回前由调用方同步。

# 详细设计

## 算子分析

### 地址映射

以元素为单位，列主序的访问偏移为：

\[
a(r,p)=\begin{cases}r+p\,lda,&transa=N\\p+r\,lda,&transa=T/C\end{cases},\qquad
b(p,c)=\begin{cases}p+c\,ldb,&transb=N\\c+p\,ldb,&transb=T/C\end{cases},
\quad c(r,c)=r+c\,ldc.
\]

指针数组先按 batch 取出矩阵基址，再应用以上偏移。地址、分块数和 workspace 字节数采用经过溢出检查的宽整数运算。尾块仅访问有效行列，不把 padding 纳入矩阵乘或结果校验。

### 计算结构

总矩阵乘工作量为 `2*batchCount*m*n*k`。将 batch 和二维输出分块共同组成任务空间：

\[
Q=batchCount\left\lceil m/B_M\right\rceil\left\lceil n/B_N\right\rceil.
\]

各核心按 `q=coreId, coreId+coreNum, ...` 领取输出块，每块沿完整 K 轴归约。这样小矩阵可借助 batch 提供并行度，大矩阵可在 batch 内继续分核；不需要 Host 逐 batch 循环下发，也不需要全局 split-K 归约。

## 算子实现

### Host 侧设计

1. 完成接口校验与零尺寸返回。用仓内指针属性查询区分 Host/Device 标量；Host 标量按值写入 tiling，Device 标量保持设备地址，不能直接在 Host 解引用，也不缓存其数值。
2. 根据 m/n/k、batchCount、转置、前导维与实际可用核数选择缩放、AIV 小矩阵或 Cube 路径。分派只依赖通用输入属性，不依赖用例编号、随机种子或矩阵内容。
3. 设置分块、有效尾块、地址步长、标量位置和 workspace 布局。使用句柄既有 workspace 管理；用户 workspace 不足时，在改写 C 前选择零 workspace 的通用路径或返回仓内约定错误。复用稳定容量，避免常规调用反复申请、释放内存和同步。
4. 在 handle 绑定的 stream 上下发全部阶段，保持前序生产 Device 标量、矩阵和指针数组的依赖关系；运行时失败按仓内状态码传播。

### Kernel 侧设计

**缩放与小矩阵路径。** AIV 读取 Device 指针数组与标量，先执行退化分支。小矩阵按 batch 与 C 的有效元素分配工作，在 UB 内分块装载、FP32 乘加并回写。N/T/C 分别使用正确地址映射，尾部用掩码或非对齐搬运处理。这条路径无需全矩阵临时空间，也用于快速覆盖边界和特殊值。

**Cube 主路径。** 采用 `C^T=op(B)^T*op(A)^T` 的行主序计算视角，交换内部左右操作数及 m/n，保持公开列主序接口。先读取本输出块所属 batch 的指针，按转置和 leading dim 搬运对应矩阵块。紧凑且满足 Matmul 约束时直接装载；否则将必要面板整理成对齐布局，尾部补零，不读取无效元素。

每核沿 K 轴分块，使用 GM→L1→L0A/L0B→Cube→L0C→GM 的矩阵计算链，L1/L0 缓冲与搬运按容量分配，使用 Ascend C Matmul 或其基础 API 管理依赖。FP32 主路径显式采用完整 FP32 乘积与累加，关闭 HF32 截断；不直接将输入转换为 FP16/BF16 代替本任务 FP32 计算。

**乘加与原地回写。** 对计算结果应用 alpha；beta 非零时先读取对应旧 C 再写回，beta=0 完全跳过旧 C 搬运。优先在输出块内完成乘加；需要独立 epilogue 时，将完整乘积暂存于句柄 workspace，在同一 stream 下发 AIV 回写。临时数据不与输入或 C 重叠，workspace 复用须满足前序读写依赖。

### 性能优化

- 小形状避免全量布局转换与多次 Kernel 启动；将多 batch 纳入一次下发，提高核利用率。
- 大形状复用 L1 面板，以 K 轴双缓冲重叠搬运与矩阵计算；根据 m/n/k 与容量选择分块，不为所有形状固定同一参数。
- 对紧凑布局减少中间整理；非紧凑布局保持 lda/ldb/ldc 语义，以必要面板搬运限制额外空间和流量。
- 性能采集覆盖一次公开调用触发的全部计算阶段。使用官方 `msopprof` 分析流水、访存及核间负载，依据实际瓶颈调整分块、搬运或输出融合。

## 支持硬件与工程组织

| 产品 | 计划支持范围 |
| --- | --- |
| Atlas A2 系列（含 Atlas 800T A2、910B3） | FLOAT32 主路径，arch22。 |
| Atlas A3 系列（含 Atlas 800T A3） | FLOAT32 主路径，arch22。 |

实现位于 `blas/gemm_batched_ex/arch22/`，测试位于 `test/gemm_batched_ex/`，沿仓内 GTest、CSV 与构建注册机制接入。使用公共头中的既有函数签名，不新增产品私有 API。README 按架构标明类型、转置、标量位置与 workspace 支持情况。

# 可维可测分析

## 精度标准与测试方法

以 Netlib BLAS 3.10.0 实数 `cblas_sgemm` 逐 batch 生成 golden，比较每个输出矩阵的有效 `m×n` 元素，另以哨兵核对 padding 未改写。FLOAT32 采用 `rtol=atol=2^-13`，匹配比例至少 0.99，并执行任务书要求的最大绝对误差上限（固定限值 `1e-2` 或标准规定的 `32*ULP`）。CSV 中 MERE/MARE 配置不替代该判据。

非有限值单独统计：同为 NaN、同号 Inf、NaN 与任意 Inf 按任务书视为等效；异号 Inf、有限值与非有限值配对不通过。有限值仍需同时满足匹配比例和最大误差门槛。alpha=0 的缩放结果额外执行 EXACT 位比较。

附件实际为 1000 条精度与 200 条性能用例，按附件全量自验，并补充 Host/Device 标量、同 stream 动态修改 Device 标量、非连续批次地址、九组转置、leading dim padding、尾块、零尺寸、k=0、alpha=0、beta=0/1、空指针、非法枚举、非法类型与前导维。beta=0 以旧 C 含 NaN/Inf 检查禁止读取语义。负向用例只比较状态码，不对其构造非法内存访问。

## 性能标准与测试方法

性能设备为 Atlas 800T A2（910B3），使用 CANN 9.1.0。以附件 `gpu_baseline.csv` 的 200 条记录逐例对齐转置、m/n/k、batchCount、标量与 leading dim，并将 `gpu_ms` 换算成 us。任务书正文要求 `T_NPU≤1.25*T_GPU`，末章提出 `T_NPU≤T_GPU`，开发目标取后者，报告同时列出两项逐例结果。

使用官方 `msprof op` 采集 task duration，预热 5 次后保留 60 次有效采样，提供正文要求的 10 次样本平均值及全体有效样本平均值，满足附件和交付章节对更多样本的要求。一次调用有多阶段时，先合计该次调用全部阶段耗时，再求算术平均，不只选择最快 Kernel；异常调用不计入有效样本并在报告中说明。

附件 README 对 GPU 型号存在不一致描述，性能报告保留原始 CSV 来源，不据其文字推定基线设备。额外记录 Host 下发、同步与 workspace 开销。任务书交付表提出 workspace 对比，而附件未提供 GPU workspace 数据；报告列明实际 NPU 峰值与缺失的基线字段，不能据空值判断该项达标。

## 兼容性分析

仅增加 arch22 实现及相应测试，公共接口、状态码和 arch35 支持范围保持兼容。A2/A3 分别按仓内 SoC 配置编译；自验证按任务要求覆盖至少一种款型，性能结论以 910B3 实测为依据。
