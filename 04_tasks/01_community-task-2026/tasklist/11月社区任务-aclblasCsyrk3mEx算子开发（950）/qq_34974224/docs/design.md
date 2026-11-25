# aclblasCsyrk3mEx 算子设计

## 需求背景

为 ops-blas 的 Atlas 950PR（arch35）增加句柄式扩展复数对称秩-k 更新接口。输入矩阵 A 可为交错 complex64 或交错有符号 complex-int8，输出矩阵 C 与系数 alpha、beta 均为 complex64。开发环境为 CANN 9.1.0 及以上。[当期官方任务书](https://www.hiascend.com/p/resource/202611/a34a01ae360a451bb9abfcbc4421dcb5.zip)

## 需求分析

令 X 为逻辑上的 n×k 矩阵。trans=N 时 X=A，trans=T 或 C 时 X 是 A 的普通转置。接口在 uplo 指定的三角原地计算 C ← αXXᵀ + βC。对角虚部参与乘加。只访问指定三角中的逻辑元素。

例如 n=2、k=1、X 的两行分别为 1+i 和 2−i，取 α=1、β=0、uplo=UPPER。两个对角输出分别是 2i 和 3−4i，上三角非对角元素为 3+i。把下三角及每列填充预置为哨兵，调用后应逐字节相同。将同一 X 按 1×2 存储并选择 OP_C，应得到相同的三个输出值。

## 详细设计

### 算子原型

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const void* A, aclDataType Atype,
    int lda, const aclblasComplex* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

- handle 提供执行 stream。alpha 与 beta 支持任务书规定的 Host 和 Device 指针位置。Host 值应在入口复制，Device 值按句柄 stream 上的顺序取得，以便处理前序写入和快速返回。
- N 模式下 lda ≥ max(1,n)，T、C 模式下 lda ≥ max(1,k)。ldc ≥ max(1,n)。所有前导维度以复元素计。ACL_INT8 输入的一个元素占两个有符号字节，按 real、imag 顺序读取，再精确提升到 FP32。Ctype 取 ACL_COMPLEX64。
- Host 按 handle、枚举、尺寸和前导维度、标量与数据类型及矩阵指针的次序校验。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，尺寸、跨度或所需指针非法时返回 `ACLBLAS_STATUS_INVALID_VALUE`，不支持的类型组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。结构参数合法时，n=0 允许 A/C 为空且直接成功。n>0 时 C 必须有效。k=0 或 α=0 允许 A 为空，此时 β=1 直接返回，否则只更新所选三角。β=0 应直接写入新值，确保旧 C 中的 NaN 不会传播。

### 实现方案

接口声明进入 `include/cann_ops_blas.h`，arch35 的 Host 与 kernel 进入 `blas/syrk/arch35/`。对照当前 `csyrk_host.cpp` 的标量读取及转置归一化逻辑，核对新接口要求的校验顺序与数据类型分派。Host 传递输入类型、列跨度、三角和分块参数。对于正常乘加，拟将指定半区分成互不重叠的输出块，每个块由一个核负责完整的 K 归约。跨对角线的块逐行截取有效列区间，或逐列截取有效行区间。列偏移使用 64 位整型计算。对于逻辑坐标 (i,t)，N 模式读取 A[i+t·lda]，T 或 C 模式读取 A[t+i·lda]。C(i,j) 使用 C[i+j·ldc]；上三角选择 i≤j，下三角选择 i≥j。

同一 kernel 按 K 分段加载两片输入。设 X=R+iI，一种对称 3M 分解使用 P=RRᵀ、Q=IIᵀ、S=(R+I)(R+I)ᵀ，每段重构复数乘积 (P−Q)+i(S−P−Q)。也可使用仓内 cgemm3m 的等价三乘积公式。较大矩阵在 Cube 上计算三次实乘，向量核在片上重构结果。对长 K，三个部分和可能远大于最终复数分量；将重构移到每个 K 段末尾，再累计各段的复数贡献，可减小抵消误差。950PR 开发中已验证过每段 64 项的方案，具体分块按片上容量、误差和实际耗时选择。每个输出块由同一核完成归约，在所选三角写回一次。

小矩阵、缩放和数值敏感的归约可在向量核内直接计算。对于可能在形成三乘积中间量时先溢出的元素，采用二进制幂缩放并恢复指数，或在同一次 kernel 内按项重算该元素。α、β 的应用需覆盖普通转置和非转置两种情况下的 NaN/Inf 行为。β=0 时直接生成新值，跳过旧 C 的读取。

交错 complex-int8 在装入片上时分别将两个有符号字节精确转换为 FP32，复用三乘积计算。依据各部分积的上界、K 长度及 α、β 判断累加是否处于 FP32 的连续整数范围。超出该范围时采用较短的归约或逐项累加。回归同时覆盖正负端点、长 K、对角虚部和列填充。

本接口仅新增 arch35 符号并复用仓内 `aclblasComplex` 与句柄定义。调用方提供列主序的矩阵，A 的可读区域与 C 的可写区域应互不重叠。计算在句柄绑定的 stream 上执行，读取输出前由调用方同步该 stream。

## 可维可测分析

| 验证项 | 执行与判据 |
| --- | --- |
| 接口和输入边界 | 在 `test/syrk/csyrk3m_ex/arch35/` 接入 GTest 与随包 CSV。先运行上面的手算用例，再组合 UPPER/LOWER、N/T/C、两类 A、非紧凑 lda/ldc、Host/Device 系数以及 n=0、k=0、α=0、β=0 和非法入参。检查状态码、对角虚部及保护区。 |
| 精度 | 950PR 上运行官方 1,000 例，依据任务书采用 Netlib BLAS 3.10.0 的 cblas_csyrk 作为独立参考，INT8 先精确提升。比较所选三角的 FP32 实虚分量，atol=rtol=2⁻¹³、matched_ratio ≥ 0.99，同时满足最大绝对误差 10⁻² 或 32 ULP 的条款。逐例记录四个 TC_3M 专项结果，另检 NaN/Inf、未选三角和列填充。 |
| 性能 | 对 200 个官方性能例分别预热 5 次，记录随后 10 次有效 msprof Task Duration 的均值。每例 GPU/NPU 耗时比 ≥ 0.4，全体加速比算术平均 ≥ 0.5。保留每例真实 kernel 下发数量，另报告 aclrtEvent 时间。按任务书补采 INT8 并注明对应的 complex64 GPU 基线。 |
| 工程接入 | 目标命令为 `bash build.sh --ops=csyrk3m_ex --soc=ascend950`。新增公开声明、注册构建和测试、在 `blas/syrk/README.md` 补充设备及 dtype 支持。回归现有 csyrk 构建与用例。 |
