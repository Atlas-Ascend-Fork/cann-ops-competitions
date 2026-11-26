# aclblasCsyrk3mEx 算子设计

## 需求背景

11 月社区任务要求在 Atlas 950PR 上提供单精度复数对称秩 k 更新接口，支持复数浮点和复数整数两种输入。现有 `aclblasCsyrk` 可供参考列主序寻址与三角写回，`blas/gemm/arch35` 的乘积内核支持在一次下发中调度三个实数乘积。新接口复用该内核，按一个输入矩阵生成三个平面，减少两侧操作数重复打包。[任务详情](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6)

## 需求分析

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const void* A, aclDataType Atype,
    int lda, const aclblasComplex* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

令 X 为 n 行 k 列的逻辑矩阵。计算 C ← αXXᵀ + βC。上三角更新行号小于等于列号的元素，下三角更新行号大于等于列号的元素。对角的实部、虚部都参与计算。A 只读，C 在原地址更新，调用方保证两者的有效区域互不重叠。

| trans | A 的存储形状 | X(i,l) 的复元素下标 | 最小 lda |
| --- | --- | --- | --- |
| ACLBLAS_OP_N | n × k | l × lda + i | max(1,n) |
| ACLBLAS_OP_T、ACLBLAS_OP_C | k × n | i × lda + l | max(1,k) |

两种转置标志均保留输入虚部符号。C(i,j) 的复元素下标为 j × ldc + i，ldc 至少为 max(1,n)。A 为 ACL_COMPLEX64 时每个复元素占 8 字节；为 ACL_INT8 时占 2 字节，依次读取有符号实部、虚部并精确提升到 float。C、α、β 均为单精度复数。

## 详细设计

### Host 侧

公共声明放在 `include/cann_ops_blas.h`，实现在 `blas/syrk/arch35/`。空句柄返回 HANDLE_IS_NULLPTR，非法三角或转置枚举返回 INVALID_ENUM，负维度、无效前导维和必需指针为空返回 INVALID_VALUE，类型组合超出支持范围返回 NOT_SUPPORTED。Host 使用 64 位算术核对矩阵访问跨度，从句柄取得 stream 和可用工作区。

| 条件 | 执行方式 |
| --- | --- |
| n = 0 | 校验后返回，设备侧无访存 |
| k = 0 或 α = 0 | β = 1 时直接返回；其余情况按选定三角缩放 C。β = 0 时直接写零 |
| 小矩阵 | 每个线程负责独立输出元素，沿 k 累计三个实数乘积，在寄存器中合并并写回 |
| 一般矩阵 | 在同一 stream 上依次打包、计算三角乘积、合并写回；工作区不足时使用直接计算路径 |

### Kernel 侧

1. **打包。** 按 A 的实际列数分配任务，按 lda 定位有效元素，将实部 R、虚部 I、两者之和 S 写入三个 FP32 平面。内部平面的前导维向上对齐到 16，额外行由打包内核写零。三个平面同时作为左右操作数。
2. **三角分块计算。** 依据 n 和分块边长构造三角块网格。若一边有 t 块，共有 t(t+1)/2 个三角块，每块有三个乘积任务。在一次 Cube kernel 下发中按核心数分配任务，分别计算 P = RRᵀ、Q = IIᵀ、U = SSᵀ。行主序乘积内核换轴后翻转三角方向。对角块允许完整计算，尾块按实际长度搬运。
3. **合并。** 对选定三角中的每个元素，实部取 P−Q，虚部取 U−P−Q，应用复数 α、β 后写回。β = 0 时跳过旧值读取。另一半三角与列尾填充保持原字节值。

设打包平面的前导维为 L、存储列数为 J、乘积平面的前导维为 D。一般路径使用三个 L×J 输入平面和三个 D×D 乘积平面，均为 FP32。各平面起始地址按 512 字节对齐，Host 在申请前计入平面间及基地址对齐开销。直接计算路径无需全局临时平面。

### 支持硬件与约束

目标为 Atlas 950PR，使用 CANN 9.1.0 及以上版本。n、k 和前导维在运行时传入。分派仅依赖形状、有效标量和工作区容量。接口覆盖列主序带填充矩阵；调用方通过句柄绑定 stream，读取结果前完成同步。

## 可维可测分析

测试目录为 `test/syrk/csyrk3m_ex/arch35/`，构建入口为 `bash build.sh --ops=csyrk3m_ex --soc=ascend950`。以下为计划采用的验证方法。

- **精度。** 使用 Netlib BLAS 3.10.0 的 `cblas_csyrk` 生成独立参考值；整数分量先精确提升。对指定三角含对角的实部、虚部分别统计混合容差，rtol、atol 均为 2⁻¹³，匹配率至少 99%，最大绝对误差按任务书的 0.01 或 32 ULP 判定。同时检查 CSV 指定的 MERE、MARE，并逐字节比较只读输入、另一半三角和列尾填充。
- **边界。** 覆盖 N/T/C、上下三角、空尺寸、零标量、非法指针和枚举、带填充的前导维、非对齐尾块、整数极值及三乘积消减。构造虚部非零的矩阵分别检验普通转置和对角虚部保留。
- **性能。** 对官方 200 条性能用例逐例采集 `msprof op` 的 Task Duration，预热 5 次，取随后 10 次有效调用的平均耗时。每次恢复原始 C，按一次调用归集完整内核序列。单例耗时至多为相应基线的 2.5 倍，全部可比用例的“基线耗时 / 实测耗时”算术平均至少为 0.5。另测复数 int8 路径；重复参数保留各自用例和基线记录。
- **兼容性与内存。** 单独构建算子目标并检查聚合测试入口。使用 msSanitizer 核对端点访问、工作区分区、非整块写回和退化路径；保存设备、工具版本与用例标识以复现结果。
