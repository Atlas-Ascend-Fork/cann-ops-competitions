# 需求背景（required）

本设计面向 11 月社区任务，在 Atlas A2/A3 的 arch22 路径上实现与当期任务书一致的 `aclblasChpr2`。该 BLAS Level 2 接口只存储 Hermitian 矩阵的一半，可直接更新压缩矩阵，避免展开为完整方阵。上游已有 arch22 内核，本次沿用其向量计算和列分配，重点补齐入口校验、Device 标量以及本期验收用例。[任务与附件](https://www.hiascend.com/activities/task-center/details/7a1275ee6b654459957fc27c8dde32b0?menu=tasks) · [上游 arch22 实现](https://gitcode.com/cann/ops-blas/tree/55ef791b77f3677b45da3468b0a8b17941424f84/blas/spr2/arch22)

# 需求分析（required）

令 `H` 表示共轭转置，计算 `A ← A + αxyᴴ + conj(α)yxᴴ`。A、x、y 和 α 均为 complex64。AP 为 A 在列主序下逐列紧密排列的上三角或下三角，输出覆盖原 AP。正常更新忽略输入对角虚部，输出对角虚部为零。零阶或零标量的合法调用保持整个 AP 原值，包括原有对角虚部。

# 详细设计（required）

## 原型与输入约束

沿用 `include/cann_ops_blas.h` 的导出接口。

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* ap);
```

| 参数 | 方向与要求 |
| --- | --- |
| handle | 已创建的 BLAS 句柄，使用绑定的 stream 执行。 |
| uplo、n | 取 ACLBLAS_UPPER 或 ACLBLAS_LOWER，阶数 n ≥ 0。 |
| alpha | 输入复数标量指针，支持 Host 和 Device 地址。 |
| x、incx；y、incy | 两个 Device 输入向量。n > 0 时各至少有 1 + (n − 1) × abs(inc) 个复数元素。步长可以为负，0 和 INT_MIN 非法。 |
| ap | Device 输入输出缓冲区，共 n(n+1)/2 个复数元素。调用方保证可写范围与 x、y 可读范围不重叠。 |

## Host 入口

按任务书 §2.5 顺序判断 handle、n 是否为负、alpha 是否为空、uplo、两个步长。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其余上述错误返回 `ACLBLAS_STATUS_INVALID_VALUE`。此后 n 为零即成功返回，无须引用任何张量或启动内核。n 大于零时先校验 x、y、ap 均非空，再处理标量值。

Host 标量可直接复制到 tiling。Device 标量先识别其地址类型，将地址传给内核，由各核在同一 stream 上读取一次；若两个分量均为零，在读取任何向量或 AP 前返回。这样同一 stream 先前写入的标量可见，普通调用无需为读取标量同步整条 stream。Host 零标量可在发射前返回。所有物理长度和偏移按 64 位计算并检查可表示范围。沿用内核发射错误的状态上传。

## Kernel 计算与访存

每个向量核负责一组互不重叠的列。按组交替正序和反序分配列，使长列与短列交错。每列先算 `p = α·conj(y[j])`、`q = conj(α·x[j])`，再逐块计算 `AP[k] += x[i]·p + y[i]·q`。上三角的索引为 `k = j(j+1)/2 + i`，取 0 ≤ i ≤ j；下三角为 `k = j(2n−j+1)/2 + (i−j)`，取 j ≤ i < n。对角元素写实部并将虚部置零。每个压缩元素只由一个核写回。

参考实现中，复乘的两个分量各有一个乘积先舍入为 FP32，另一个乘积通过融合乘加累加。非对角元素按 `(AP + x[i]·p) + y[i]·q` 的顺序舍入；对角实部先合并两个更新项，再加原实部。当更新项与原值相消时，常规向量乘加的舍入差异会被放大。内核用块内向量分量与列系数的幅值估计舍入影响，对这些分量按 Netlib 的顺序重新计算。验证保留相同输入、原始 AP 及独立 CPU 参考，逐项检查实、虚部。

沿用 UB 行块缓存复用 x、y。单位步长时连续搬入；其他步长按逻辑下标取值，负步长将逻辑下标 i 映射至物理下标 (n−1−i)×abs(inc)。尾块读写按有效字节数处理。正常计算核和测试工程分别位于 `blas/spr2/arch22/` 和 `test/spr2/chpr2/arch22/`，使用 CANN 9.1.0。

# 可维可测分析

| 项目 | 验证方法与判据 |
| --- | --- |
| 精度 | 逐条执行附件的 1,000 条精度用例。用 Netlib BLAS 3.10.0 的 `cblas_chpr2(CblasColMajor, …)` 生成 golden；正向用例比较全部 AP 实部与虚部，检查对角虚部。按分量取 atol = rtol = 2⁻¹³、匹配比例 ≥ 0.99，最大绝对误差按任务书的 1e−2 或 32 ULP 限值判定。另记录 CSV 指定的 MERE、MARE 口径及 NaN、Inf 匹配。 |
| 接口边界 | 验证上下三角、负步长、大步长及非对齐尾部。用 n=0 同时传非法枚举、零步长、INT_MIN 或空 alpha 检查错误优先级。用有效指针和零标量验证 AP 字节保持不变。补测空 handle、Device 非零复标量及同 stream 写入零标量，检查 x/y 及 AP 保护区。 |
| 性能 | 在 Atlas 800T A2（910B3）上按 200 条性能用例分别采集 `msprof op` 的 `Task Duration(us)`。每例预热 5 次，取后续 10 次成功调用的平均值并换算为 ms，以该例的 uplo、n、incx、incy 匹配附件 `gpu_baseline.csv`。每例须满足 NPU 平均时间 ≤ 1.25 × GPU 时间。GTest 的事件计时用于快速回归。 |
| 兼容性 | 保留公开 ABI，回归既有 arch22 测试。测试驱动使用本期 CSV 原始种子、标准正态及均匀分布标签。对 Device 标量的 stream 顺序做独立覆盖。 |

例如 n=2、下三角时，三个槽位依次为 A₀₀、A₁₀、A₁₁。取 x=(1,i)、y=(1,1)、α=i、初始 AP=(0,0,0)，期望输出为 (0,−1−i,−2)。该输入同时检查列压缩顺序、第二项的共轭以及两个对角元素的虚部。
