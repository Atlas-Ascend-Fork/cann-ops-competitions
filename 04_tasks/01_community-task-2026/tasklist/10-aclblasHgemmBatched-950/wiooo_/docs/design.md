# aclblasHgemmBatched 算子设计（Ascend 950PR）

## 1 需求背景

本设计对应 [10 月社区任务 aclblasHgemmBatched（950）](https://www.hiascend.com/activities/task-center/details/fe48759fdaaf4afb80ecbf5a758a1123?menu=tasks)，在 CANN 9.1.0、Ascend 950PR 上以 Ascend C 补齐半精度指针数组批量 GEMM。接口和计算语义参照 cuBLAS `cublasHgemmBatched`，实现接入 `cann/ops-blas` 的句柄式 BLAS 工程。

## 2 需求分析

### 2.1 接口与数据约定

在 `include/cann_ops_blas.h` 声明公共接口，供各产品线共用：

```cpp
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* const* Aarray, int lda,
    const aclblasHalf* const* Barray, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* const* Carray, int ldc,
    int batchCount);
```

`aclblasHalf` 为与 ACL FLOAT16 一致的 16 位半精度存储类型；公共类型统一定义，不引入 950PR 专用接口。标量和属性位于 Host，三个指针数组本身及其指向的矩阵均位于 Device。矩阵地址可不连续，不能由首个指针推断批间步长。

| 参数 | 约定 |
| --- | --- |
| `handle` | 有效的 ops-blas 句柄，使用其绑定的 stream 下发计算。 |
| `transa/transb` | `ACLBLAS_OP_N/T/C`；实数下 C 与 T 等价。 |
| `m/n/k/batchCount` | 非负 `int`，运行时给定。每批维度相同。 |
| `alpha/beta` | 非空 Host FLOAT16 标量指针，全批共用。 |
| `Aarray` | `batchCount` 个只读矩阵地址；物理 shape 在 N 时为 `(m,k)`，T/C 时为 `(k,m)`。 |
| `Barray` | `batchCount` 个只读矩阵地址；物理 shape 在 N 时为 `(k,n)`，T/C 时为 `(n,k)`。 |
| `Carray` | `batchCount` 个 `(m,n)` 矩阵地址，原地更新。`beta=0` 时不读取矩阵旧值。 |
| `lda/ldb/ldc` | 列主序前导维。`lda≥max(1,transa=N?m:k)`，`ldb≥max(1,transb=N?k:n)`，`ldc≥max(1,m)`。允许 padding，仅更新 C 的有效矩阵区域。 |

支持 FLOAT16、N/T/C 全组合和前导维 padding；不涉及广播、额外 stride 视图或跨批输出重叠。不要求确定性计算。实现代码位于 `blas/gemm/arch35/`，测试位于 `test/gemm/hgemmBatched/arch35/`。

### 2.2 参数检查与边界

1. 空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法转置枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；负维度、负 batchCount、非法前导维返回 `ACLBLAS_STATUS_INVALID_VALUE`。
2. alpha/beta 判空，以及 `batchCount>0` 时三个数组指针判空，均先于 no-op 短路；空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。因此 `m=0` 或 `n=0` 且 batchCount>0 仍需非空数组指针，batchCount=0 不检查数组指针。
3. 合法参数下，`m=0`、`n=0` 或 `batchCount=0` 返回 SUCCESS，不解引用设备数组、不启动计算。
4. `k=0` 时只执行 `C_b=beta*C_b`，不解引用 A/B 的数组元素或矩阵；beta=0 直接写零，beta=1 保留原值。alpha=0 采用 BLAS 的退化缩放路径，仍执行上述参数检查。
5. 正常路径中，beta=0 不加载旧 C，避免无效旧值或 NaN 污染结果。异步任务遵循 handle 的 stream 顺序，调用者保证设备数组与矩阵在计算结束前有效。

## 3 详细设计

### 3.1 数学表达

对批号 `b∈[0,batchCount)`、`r∈[0,m)`、`c∈[0,n)`：

\[
C_b^{out}(r,c)=\operatorname{RN}_{16}\left(\alpha\sum_{p=0}^{k-1}
\operatorname{op}(A_b)(r,p)\operatorname{op}(B_b)(p,c)+\beta C_b^{in}(r,c)\right).
\]

输入提升为 FP32 累加，缩放与加法保留 FP32，最终按就近舍入转换为 FLOAT16。列主序地址分别为：

| 元素 | N | T/C |
| --- | --- | --- |
| `op(A_b)(r,p)` | `A_b[r+p*lda]` | `A_b[p+r*lda]` |
| `op(B_b)(p,c)` | `B_b[p+c*ldb]` | `B_b[c+p*ldb]` |
| `C_b(r,c)` | `C_b[r+c*ldc]` | 同左 |

数学等价不要求与 CPU 逐位一致，精度按第 4 节逐批验证。索引、工作量与工作区字节计算使用足够宽的整数，并检查溢出。

### 3.2 Host 设计

Host 完成参数检查、读取半精度标量、获取核数、计算 tiling 和 kernel 直调。设备指针数组由 kernel 读取，Host 不将数组搬回逐批下发。利用 `Cᵀ=op(B)ᵀop(A)ᵀ` 交换操作数及 m/n、lda/ldb、转置属性，使现有行主序 GEMM 模板直接处理列主序数据，无需实体转置。

计算任务由 batch 和 M/N 输出块共同划分，以一维任务序号分配到计算核。大矩阵切分 M/N 维获得并行度，小矩阵优先利用 batch 维，尾块按真实长度处理。分派仅依赖维度、转置、前导维和标量等合法入参，不依赖测试编号或指针地址。

alpha=1、beta=0 路径直接输出 FLOAT16；通用标量路径保留 FP32 中间结果后完成 epilogue，不能先把乘积截断到 FLOAT16 再乘 alpha。工作区复用 handle 既有机制，按容量划分批次或输出块，避免默认工作区大小限制合法批量规模；每次调用中的全部内部阶段均纳入性能统计。

### 3.3 Kernel 设计

复用 ops-blas 同族 batch GEMM 模板及搬运、矩阵计算组件。首选 Cube 的 FP16×FP16→FP32 累加，K 分块在 L0C 累加；GM→L1 与 L1→L0 使用双缓冲和事件同步，输出块由唯一任务写入。

| 阶段 | 执行单元与处理 |
| --- | --- |
| 读取批地址、定位输出块 | Scalar 读取设备指针表，计算本批基址及 64 位元素偏移。 |
| 搬入 A/B | MTE 按转置布局及前导维搬入 L1，转换为矩阵引擎需要的分块布局，尾块按真实范围访问。 |
| 矩阵乘累加 | Cube 分块计算，L0C 保存 FP32 部分和，完成全部 K 后才收窄输出。 |
| alpha/beta 后处理 | Vector/SIMT 在 FP32 中计算缩放与加法；beta=0 分支省略旧 C 读取。 |
| 输出 | alpha=1、beta=0 优先通过 Fixpipe 直接转换写回；通用路径在后处理后写回有效 C 区域，保持 padding。 |

对于极小矩阵，可比较一核处理多个 batch 或 Vector/SIMT 直接计算与 Cube 的成本，避免填充和多阶段启动开销。最终分派依据目标设备测量确定，保持同一接口和精度要求。性能分析使用官方 msopprof，结合搬运、流水等待、矩阵引擎利用率和核间负载定位问题。

## 4 可维可测分析

### 4.1 精度与功能

使用 ops-blas CSV/GTest 框架，逐 batch 以 Netlib BLAS 3.10.0 `cblas_sgemm` 生成 FP32 golden，再就近舍入到 FLOAT16。每批完整 `(m,n)` 输出独立检查：有限值按 `|actual-golden|≤2^-10+2^-10*|golden|` 判断，匹配率至少 0.99，最大绝对误差满足任务书规定的 `1e-1 或 32×ULP` 限制，不能以其他误差指标替代。

非有限值单独检查：双方 NaN、同号 Inf、或一方 NaN 另一方 Inf 视为等效；有限值与非有限值不等效，异号 Inf 不等效。非有限元素不进入有限值统计，且须全部满足上述规则。大 K 导致的精度问题保留 golden、误差增长和累加方式证据，不放宽阈值。

覆盖官方 1000 条精度用例及 200 条性能用例，并补齐任务书要求而附件未覆盖的正态/均匀各占 50% 输入、16384 维度与 65536 batch 边界。重点检查 N/T/C 全组合、矩形和奇数尾块、独立非连续批地址、padding 哨兵、beta=0 且旧 C 为 NaN、k=0、空维度和错误码优先级。负向测试在分配矩阵前处理负维度、空标量与非法枚举，防止测试程序先失败。

### 4.2 性能

基线使用任务包 `gpu_baseline.csv` 的 200 条 H100 数据。按任务书 §3.3，预热 5 次后使用官方工具采集 10 次有效 task duration，取算术平均；多 kernel 路径统计一次接口调用的全部计算阶段，不使用含 CPU golden 的 GTest 总耗时替代。逐例要求 `GPU耗时/NPU耗时≥0.4`，全部性能用例倍率的算术平均值要求 ≥0.5，两项同时满足。

任务书 §7.4 另写“有效采样 >10 次、达到 ≥GPU”，与 §3.3 及配套性能脚本不一致。评审时需确认最终验收采用的阈值和采样次数；本设计完整列明两处条款，不将其默认为豁免。脚本旧提示、测试目录和计时方式如与任务书不一致，测试接入按确认后的任务要求处理。

### 4.3 兼容性与维护

新增共用声明与 arch35 实现，不改变现有 GEMM 接口语义。沿用标准构建和 GTest 注册方式，复用模板时检查受影响接口的编译与代表性回归。算子 README 标注 Ascend 950PR 支持情况、列主序和设备指针表约束，并提供 handle、stream、分配、调用及同步示例。

## 5 参考资料

1. [官方任务包](https://www.hiascend.com/p/resource/202610/135324b6762b4fa6bc20cf26b23ebad6.zip)。
2. [ops-blas 源码与贡献指南](https://gitcode.com/cann/ops-blas)。
3. [cuBLAS batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublas-gemmbatched)。
4. [生态算子混合容差标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)。
