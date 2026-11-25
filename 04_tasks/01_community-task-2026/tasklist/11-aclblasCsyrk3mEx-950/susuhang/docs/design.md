# aclblasCsyrk3mEx 算子设计文档

# 需求背景

## 需求来源

11 月 CANN 社区任务要求为 Atlas 950PR（arch35）新增 Ascend C 算子 `aclblasCsyrk3mEx`，在 `cann/ops-blas` 的 BLAS 句柄和 stream 上执行单精度复数对称秩-k 更新。设计依据为任务页面发布的任务书及同包自验附件，附件未提供可核对的任务书 Git commit。 [任务页面](https://www.hiascend.com/activities/task-center/details/3d67a5f992894ea8b43e472342c122e6) · [任务书与附件](https://www.hiascend.com/p/resource/202611/a34a01ae360a451bb9abfcbc4421dcb5.zip) · [任务讨论帖](https://gitcode.com/cann/ops-blas/discussions/88)

## 背景介绍

复数对称矩阵的两个三角区域具有转置关系。用实数矩阵乘处理秩-k 更新时，可以跳过输出未指定的三角块，并通过高斯 3M 将计算从四组实数乘积降为三组。Host 侧沿用仓库 `csyrk` 的句柄、stream 和参数形状约定。设备侧复用 Ascend C Tensor API 的矩阵分块与搬运方式，参考既有 `cgemm3m` 的 3M 计算。 [ops-blas csyrk](https://gitcode.com/cann/ops-blas/tree/master/blas/syrk/arch35) · [ops-blas GEMM](https://gitcode.com/cann/ops-blas/tree/master/blas/gemm/arch35)

# 需求分析

## 需求描述

```cpp
aclblasStatus_t aclblasCsyrk3mEx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha, const void* A, aclDataType Atype, int lda,
    const aclblasComplex* beta, aclblasComplex* C, aclDataType Ctype, int ldc);
```

| 参数 | 语义和范围 |
| --- | --- |
| `handle` | BLAS 句柄，输出更新在绑定的 stream 上执行 |
| `uplo` | `ACLBLAS_UPPER` 更新 i ≤ j，`ACLBLAS_LOWER` 更新 i ≥ j |
| `trans` | `OP_N` 直接使用 A。`OP_T` 与 `OP_C` 均做普通转置 |
| `n, k` | n ≥ 0、k ≥ 0，C 的阶数与更新秩 |
| `alpha, beta` | `aclblasComplex` 标量，接受 Host 或 Device 地址 |
| `A, Atype, lda` | 列主序。N 时逻辑形状 n × k，`lda >= max(1,n))。T/C 时逻辑形状 k × n，`lda >= max(1,k))。`ACL_COMPLEX64` 每个元素两个 FP32，`ACL_INT8` 每个元素两个有符号 int8，实部在前。前导维单位为复元素 |
| `C, Ctype, ldc` | 原地更新列主序 C，`Ctype = ACL_COMPLEX64`，`ldc >= max(1,n)`。对角虚部参与计算，未指定三角及填充保持原字节 |

## 需求拆解

1. 参数按 handle、枚举、尺寸和前导维、类型及指针的顺序检查。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。非法 uplo/trans 返回 `ACLBLAS_STATUS_INVALID_ENUM`。非法尺寸、前导维及必需指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。类型组合超出支持集返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。
2. 结构合法且 `n=0` 时直接成功。若 `k=0` 或 `alpha=(0,0)`，A 可为空，且仅当 `beta!=(1,0)` 时更新 C。此时及普通秩-k 更新中，`beta=(0,0)` 允许省去对旧 C 的读取。非空矩阵调用始终要求 C 指针有效。
3. 非退化路径把三个实数乘积放在同一次设备 kernel 发射中，保留复数缩放、列主序前导维和三角存储语义。公开入口放在 `include/cann_ops_blas.h`，实现和测试分别放在 `blas/syrk/arch35/` 与 `test/syrk/csyrk3m_ex/arch35/`。
4. 精度按任务书 FLOAT32 分量阈值验收。性能按 950PR kernel 耗时和附件 cuBLAS 基线验收。交付 CSV 及其冻结数据、测试代码、报告和从完整 commit 开始的复现步骤。

# 详细设计

## 算子分析

令 `X=op(A)=R+iI`。N 时 `X[i,t]=A[i,t]`，T/C 时 `X[i,t]=A[t,i]`。定义 `P0=RRᵀ`、`P1=IIᵀ`、`P2=(R+I)(R+I)ᵀ`，则乘积实部为 `P0-P1`，虚部为 `(P2-P0)-P1`。虚部展开为 `RIᵀ+IRᵀ`，包括对角上的 `2 Σₜ Rᵢₜ Iᵢₜ`。对每个选定的 `C[i,j]`，以复数乘法完成 `alpha * (real + i * imag) + beta * C_old`。

三角内有效算术量为 `O(n²k)`，存储访问除输入读入外主要来自实数平面和部分乘积。小规模优先减少 launch 和临时访存。大规模优先减少三角外计算以及 AIC 面板重复搬运。3M 重组中的差分会放大舍入误差，尤其是对角实部和两项相消时。FP32 路径使用原精度或等效补偿精度，针对有损抵消时按 K 分段重组或增加精度；INT8 在扩展为可精确表示的实数后，也可按安全长度分段累加，再用宽整数归并，避免用低精度近似整型产品。参考实现使用普通复数乘法，精度比较允许合理的舍入差异。 [Netlib CSYRK](https://www.netlib.org/blas/csyrk.f)

## 算子实现

### Host/Tiling 设计

- 按上文约定检查结构。空阶数无需读取标量。其余情况使用仓内的指针位置检测，以值传入 Host 标量，在需要 Device 值判定短路或校验空 A 时通过句柄 stream 有序读取。结构校验完成后再执行快返。调用方通过该 stream 获取计算完成状态。
- `n`、`k`、`lda`、`ldc`、类型、三角、转置和标量分支随 tiling 传入 kernel。网格只枚举与选定三角相交的输出 tile，每块由一个逻辑所有者完整写回。使用 64 位地址及容量运算，分配临时空间前检查对齐与乘加溢出。复用 handle 工作区与其 stream 生命周期。
- 退化缩放和小问题使用 AIV，按列主序三角线性索引或局部矩形分发。大问题用 AIC 对三角 tile 做 K 方向巡回，初始候选为 64 × 64 输出 tile 与最多 64 个连续 K 元素的面板。核数取有效 tile 数与平台可用核数的较小值。阈值按形状、dtype 和片上资源选取，输出 tile 的最后一列和最后一行均按逻辑范围裁切。

### Kernel 设计

1. AIV 小规模路径直接解码两个输入元素的实虚部，在同一 kernel 内沿 K 累加三个实乘并组合。AIV 缩放路径复用三角所有权和写回规则。两个路径均先处理 `alpha`、`beta` 快捷分支。
2. AIC 路径前将交错 A 解码为 R、I、R+I 实数平面。按原始列主序处理转置和 stride，输入尾块补零，补零限于工作区。AIC 从相同平面分别装载左右面板，复用仓库 Tensor API 的 GM→L1→L0 搬运与 Mmad。三个累加器在同一 kernel 内完成 K 方向更新，由 Fixpipe 产出 tile 的三个实数结果。流水事件约束缓冲区再次写入的时机。
3. AIV 在结果就绪后组合实虚部，读取需要的旧 C 区间，完成复数标量缩放并写回。对角 tile 内逐元素校验 uplo，未选中三角、前导维填充及矩阵外的地址均跳过。输入准备、矩阵乘和收尾的依赖由 stream 或同核流水事件保证。
4. COMPLEX64 非有限值、极端值和抵消是专项验证对象。运算重排会改变非有限值传播时，设备侧可沿非 Ex `csyrk` 的参考求值顺序处理受影响的元素。普通有限区间仍走 3M。对 beta 为零且旧 C 含 NaN、alpha 为零且 A 含 NaN 的场景验证短路生效。

## 支持硬件

| 芯片 | 工具链 | 资源约束 |
| --- | --- | --- |
| Atlas 950PR（arch35） | CANN ≥ 9.1.0 | AIV 负责拆分、缩放及写回，AIC 负责实矩阵乘。参考仓内 512 KiB L1 和 Tensor API 的块对齐约束 |

以 FP32 64 × 64 面板作为预算示例，每个面板 16 KiB。三个乘积各有左右面板，L1 双缓冲约需 192 KiB，L0C 单组累加器约需 48 KiB。每次只将当前乘积所需面板传到 L0A/L0B，避免三组面板挤占其容量。实现时按具体 devkit 的布局、尾块取整、L0 容量及事件约束核查此预算。

## 算子约束限制

仅支持任务书指定的类型组合、三角原地更新与由前导维描述的列主序带填充布局。A 的可读区与 C 的可写区需互不重叠。同一三角 tile 在普通路径中只由一个所有者输出，若按 K 分段则在写回前归并。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 计划 |
| --- | --- |
| 正确性 | 在 Atlas 950PR 跑全量 1000 条 CSV 精度用例，附加空 handle、Device 标量、空 A 短路及 T/C 等价检查。用 Netlib BLAS 3.10.0 的 `cblas_csyrk` 或 `cblas_zsyrk` 提供独立 golden。对三角实部、虚部分别统计 `rtol=atol=2^-13`、匹配率 ≥ 0.99、最大绝对误差 ≤ `1e-2` 或 32 ULP，保留非有限值判定、对角结果及未选三角/padding 的逐字节检查。 |
| 性能 | 保留附件全部 200 个 COMPLEX64 性能用例及其逐行 GPU kernel 基线，重复形状按用例对应行匹配。补采 INT8，与对应形状的 COMPLEX64 GPU 基线比较。在空闲设备用 `msprof op` 逐例记录，预热 5 次、随后 10 个有效样本取平均。逐例保存主 kernel 与调用内各阶段的 `Task Duration(us)`，多阶段同时给出合计，核对单例 GPU/NPU ≥ 0.4 和全量平均 ≥ 0.5。事件时延单独作为趋势参考。 |
| 可复现性 | 用 `bash build.sh --ops=csyrk3m_ex --soc=ascend950` 构建。保留用例配置、随机种子、冻结输入、golden、精度日志和原始 profiling，独立检出后按版本执行自验，在报告中分列实部、虚部及逐例性能。 |

## 兼容性分析

公开新增 `aclblasCsyrk3mEx`，复用 `aclblasComplex`、`aclDataType` 与仓库状态码。类型检查、编译条件和测试目标限定在 950 实现范围。数值参考按对称 CSYRK 映射 T/C；上层调用方式和任务书签名一致。 [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html) · [混合误差精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)
