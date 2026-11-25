# 需求背景（required）

本设计面向在 Atlas A2/A3 上调用单矩阵 BLAS 接口的应用。调用者提供列主序矩阵、两个 Host 标量和绑定了 stream 的句柄。三个入口分别接受 FP32、由两个 FP32 分量组成的 COMPLEX64、以及以 IEEE FP16 位模式存储的 `aclblasHalf`。[任务详情](https://www.hiascend.com/activities/task-center/details/4b45f7b048224317a2ec62eab1e8df52?menu=tasks) · [任务书与自验包](https://www.hiascend.com/p/resource/202611/65f1c7ed628f416ebbb68e9dc9595ce0.zip)

上游 arch22 已包含 Sgemm 的 FP32 分块计算，以及 Cgemm 拆分实部和虚部后调用四次实数矩阵乘的路径。Sgemm 和 Cgemm 沿用现有声明。Hgemm 新增同族单矩阵入口及 arch22 实现，批量接口保持独立。[上游基线](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/gemm/arch22)

# 需求分析（required）

计算 `C ← alpha · op(A) · op(B) + beta · C`。逻辑结果为 m 行 n 列，内积长度为 k。第 j 列第 i 行在 C 内的元素偏移是 `j * ldc + i`。写回只覆盖 `0 ≤ i < m`，保留列尾填充。调用者保证 C 的写入区域与 A/B 的读取区域无重叠。

| 参数 | 约束 |
| --- | --- |
| handle、transa、transb | handle 提供执行 stream。两个枚举值各为 N、T 或 C。实数的 C 与 T 等价，复数的 C 对转置后的虚部取反。 |
| m、n、k | 三者均为非负整数。op(A) 为 m×k，op(B) 为 k×n。 |
| A、lda | 输入在 Device。N 时 lda ≥ max(1,m)，物理列数为 k。T/C 时 lda ≥ max(1,k)，物理列数为 m。 |
| B、ldb | 输入在 Device。N 时 ldb ≥ max(1,k)，物理列数为 n。T/C 时 ldb ≥ max(1,n)，物理列数为 k。 |
| C、ldc | 原地输出在 Device，ldc ≥ max(1,m)。m 和 n 同时为正时 C 必须有效。 |
| alpha、beta | 两个指针均指向 Host 内存中的本算子标量，必须非空。零维输出允许 A/B/C 为空；非空输出在 k=0 或 alpha=0 时允许 A/B 为空。 |

# 详细设计（required）

## 算子原型

```cpp
aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);

aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasHalf* alpha, const aclblasHalf* A, int lda,
    const aclblasHalf* B, int ldb, const aclblasHalf* beta, aclblasHalf* C, int ldc);
```

## Host 校验与下发

按 handle、转置枚举、维度、前导维度、标量指针、所需矩阵指针的次序校验。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法维度或必需指针为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。结构参数合法且 m=0 或 n=0 时返回成功，设备无工作。对于非空输出，k=0 或 alpha=0 时仅执行 C 的缩放，beta=1 可直接返回，beta=0 直接写零。一般乘法的 beta=0 分支也跳过旧 C 的读取，以免将未初始化值或 NaN 引入结果。

Host 捕获本次调用的标量值，将布局、有效尾块大小和缩放参数随 kernel 下发。全部阶段在 handle 的 stream 上依序执行，接口异步返回。Cgemm 的 arch22 测试封装需与公开的 Host 标量语义一致；核对依赖现有 Device 标量行为的旧调用点，再确定兼容处理的范围。其他复数 BLAS 接口共用的标量读取函数不做全局替换。

## Kernel 分工

| 入口 | 计算与写回 |
| --- | --- |
| Sgemm | 复用 arch22 的 FP32 分块矩阵乘。紧凑且对齐的矩阵沿用直接加载；转置、带填充及尾块按各自的逻辑坐标读取。分块累加完成后，只对有效 C 执行一次 alpha/beta 更新。 |
| Cgemm | 按列拆分实部和虚部。在转置后的实矩阵上计算四个乘积。设 sA、sB 在各自共轭时为 −1，其余为 +1。实部取 ArBr−sA·sB·AiBi，虚部取 sB·ArBi+sA·AiBr。完成全部 K 分块后，用复数 alpha、beta 合并到交错存储的 C。复用现有四乘与 workspace 管理，逐阶段测量拆分、实数乘和合并开销。 |
| Hgemm | Cube 使用 FP16 输入、FP32 累加。对不满足直接加载条件的输入，AIV 按前导维读取有效行列，在 UB 中补零，再将对齐的 FP16 临时矩阵写入 handle workspace 供 Cube 读取。alpha=1、beta=0 且 C 满足写回对齐要求时，Cube 直接写入 FP16 输出。其余情形先将 FP32 乘积写入 workspace，再由 AIV 按 alpha、beta 缩放，收窄为 FP16 并只写 C 的有效行列。 |

实现文件放在 `blas/gemm/arch22/`。公开声明放在 `include/cann_ops_blas.h`。扩展 `test/gemm/arch22/` 的 GTest 入口，按三个算子各自的附件字段读入用例。文档在 `blas/gemm/README.md` 说明接口、产品支持、Host 标量和 stream 语义。所有偏移和工作区大小用经溢出检查的宽整数计算。

# 可维可测分析

目标软件版本为 CANN 9.1.0。精度自验覆盖任务书允许的一种 A2/A3 款型。正式性能在 Atlas 800T A2（910B3）按用例采集。

| 验证项 | 输入、方法和判据 |
| --- | --- |
| 用例及参考结果 | 附件 S/C/H 分别有 334/333/333 条精度用例和 67/67/66 条性能用例。S/C 以 Netlib BLAS 3.10.0 对应的 cblas 结果为独立参考，按任务书保留双精度累加后收窄参考与 cblas 参考的差异。H 输入与标量从 FP16 解码，高精度累加后先转为 FP32，再以最近偶数舍入转 FP16。 |
| 精度 | 用例内满足 `abs(actual−golden) ≤ atol + rtol·abs(golden)` 的比例至少 99%。S/C 的 atol、rtol 均为 2⁻¹³，最大绝对误差按 10⁻² 或 32 ULP 判定；H 分别为 2⁻¹⁰ 和 10⁻¹ 或 32 ULP。复数对实部、虚部分别判定。为 NaN、Inf、FP16 次正规值和溢出保留专门输入与分类比较。 |
| 共轭与输出保护 | 对逻辑矩阵 A=[[1+i,2],[3i,4−i]]、B=[1,i]ᵀ，取 alpha=1、beta=0。转置结果为 [−2+i,3+4i]ᵀ，共轭转置结果为 [4−i,1+4i]ᵀ。九种转置组合均覆盖；对每列的 padding 做逐位哨兵检查。 |
| 退化与错误路径 | 在 m=0 的同一输入上依次使用合法及非法 lda，分别期待成功与 INVALID_VALUE。另测 alpha=0 且 A/B 为空或含 NaN、k=0 且 beta 非 1、beta=0 且旧 C 含 NaN，以及空标量和非法枚举。 |
| 性能 | 每例先预热 5 次，再执行 10 次成功调用。用 msprof op 采集 kernel Task Duration，换算为毫秒后取十次均值，每例要求 ≤ 附件对应 `gpu_ms × 1.25`。对于一次接口调用下发多个 kernel 的情形，保留逐个任务及调用归属；报告另列所有阶段的累计耗时。任务书未单列多 kernel 的聚合规则，需在设计评审时确认正式判定口径。按转置、形状、前导维度及标量匹配基线。另以 aclrtEvent 采集预置 Device 缓冲上的 10 次平均，输出 `[PERF] case=… avg_us=…`，供交叉验证。 |
