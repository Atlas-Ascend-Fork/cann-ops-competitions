# Csyr2 / Cher2k / Cherk 精度修复设计文档（A2/A3）

# 1. 需求背景

## 1.1 需求来源

本需求来自 CANN 10 月社区任务“aclblasCsyr2/aclblasCher2k/aclblasCherk 精度修复-A2/A3”。基于 ops-blas 已有实现，使用 Ascend C 完成修复，适配 CANN 9.1.0 和 Atlas A2/A3。

任务要求以任务书为准。本文说明整体设计，测试结果在自测报告中单独记录。

## 1.2 背景介绍

三个算子均对列主序矩阵进行原地更新，且只允许更新 `uplo` 指定的三角区域。任务书指出，已有实现可能在分块写回时覆盖另一侧三角或前导维 padding，导致这些位置与输入不再二进制一致。

本次修复围绕三角区域裁剪、边界写回和特殊分支展开，在保持公开接口及计算语义的基础上，保证非操作区原样保留。

# 2. 需求分析

## 2.1 需求描述

完成 Csyr2、Cher2k、Cherk 三个算子的 A2/A3 精度修复，并一并交付实现、测试代码、设计文档和自测报告。

## 2.2 需求拆解

1. 操作区计算结果满足任务书规定的混合容差标准。
2. 非操作三角及 `lda/ldc` padding 不发生写入，调用前后逐 bit 一致。
3. 保持 quick return、负步长、Hermitian 对角虚部及参数校验语义。
4. 保持句柄绑定 stream 的异步执行方式，不引入 Host 强制同步或 CPU 计算回退。
5. 性能满足任务书标杆，三个算子均完成 A2/A3 验证。

# 3. 详细设计

## 3.1 算子分析

### 数学公式

以下 `T` 表示普通转置，`H` 表示共轭转置。

| 算子 | 计算公式 |
| --- | --- |
| Csyr2 | `A ← alpha·x·yᵀ + alpha·y·xᵀ + A` |
| Cher2k，trans=N | `C ← alpha·A·Bᴴ + conj(alpha)·B·Aᴴ + beta·C` |
| Cher2k，trans=C | `C ← alpha·Aᴴ·B + conj(alpha)·Bᴴ·A + beta·C` |
| Cherk，trans=N | `C ← alpha·A·Aᴴ + beta·C` |
| Cherk，trans=C | `C ← alpha·Aᴴ·A + beta·C` |

Csyr2 为复数对称更新，不做共轭，对角虚部正常参与计算。Cher2k/Cherk 为 Hermitian 更新，计算分支将对角虚部置零，且仅支持 `trans=N/C`。

### 数据类型与形状

矩阵及向量使用 `complex64`，实部、虚部均为 float32。Csyr2 的 alpha 为复数；Cher2k 的 alpha 为复数、beta 为实数；Cherk 的 alpha、beta 均为实数。

输出矩阵逻辑形状为 `n×n`，物理存储为 `ld×n`，元素地址为 `i+j·ld`。Cher2k/Cherk 的输入矩阵在 `trans=N` 时为 `n×k`，在 `trans=C` 时为 `k×n`。Csyr2 支持非零正负步长。

## 3.2 算子实现

### Host 侧设计

沿用已有公开接口和 Ascend C kernel 直调方式，保留参数校验、返回码以及标量指针的 Host/Device 归属。

根据 `uplo`、`trans`、矩阵尺寸和前导维组织计算任务。划分任务时同时考虑三角区域的工作量，保证各核心负责的输出区域互不重叠。完整操作区块与跨对角线、尾部边界块分别处理，具体块大小结合现有实现和设备资源确定。

Host 可直接判定的 quick return 在校验后返回；依赖 Device 标量的分支在 Device 侧处理，避免为读取标量增加 Host 同步。

### Kernel 侧设计

计算流程包括输入搬运、复数计算和结果写回。Csyr2 按步长读取向量，负步长采用反向起点；Cher2k/Cherk 按 `trans` 选择对应的共轭和矩阵访问方式。

写回时按列确定有效行范围，并与当前 tile 范围取交集：

| uplo | 第 j 列允许写入的行范围 |
| --- | --- |
| UPPER | `0 ≤ i ≤ j` |
| LOWER | `j ≤ i < n` |

完整位于操作区的块使用批量搬运；跨对角线块和尾块采用有效区间裁剪、分段或精确长度写回。全局输出写入不得因搬运对齐而扩展到另一三角或 padding，也不得通过“先整块覆盖再恢复”实现保护。

旧输出矩阵只读取当前计算需要的操作区。`beta=0` 时不使用旧 C 参与计算，避免旧值中的 NaN/Inf 影响结果。Hermitian 对角计算忽略旧对角虚部，输出虚部置零。

### 特殊分支

- Csyr2：`n=0` 或 `alpha=0` 时不更新矩阵。
- Cher2k/Cherk：`n=0`，或 `(alpha=0 或 k=0) 且 beta=1` 时不更新矩阵，包括对角虚部。
- Cher2k/Cherk：`alpha=0` 或 `k=0` 且未触发 quick return 时，仅对指定三角执行 beta 缩放；对角虚部置零。

以上分支均遵循任务书规定的参数校验要求。

### 修改范围

以下路径均相对于 ops-blas 仓库：

| 范围 | 路径 |
| --- | --- |
| Csyr2 实现 | `blas/syr2/arch22/` |
| Cher2k/Cherk 实现 | `blas/herk/arch22/` |
| Csyr2 测试 | `test/syr2/csyr2/arch22/` |
| Cher2k 测试 | `test/herk/cher2k/arch22/` |
| Cherk 测试 | `test/herk/cherk/arch22/` |

不修改公开接口签名，不新增平行 API，不修改 arch35 实现。不为保护非操作区引入与问题规模无关的持久 Device 缓存。

## 3.3 支持硬件与约束

支持 Atlas A2、Atlas A3，性能主验收设备为 Atlas 800I A2（910B3）。沿用列主序、前导维、原地更新及异步 stream 语义，不扩展广播或超出既有接口范围的非连续访问能力。

# 4. 可维可测分析

## 4.1 精度标准

| 检查对象 | 验收要求 |
| --- | --- |
| 操作区 | 实部、虚部分别按 float32 混合容差判定；`rtol=atol=2^-13`，匹配率不低于 99%，最大绝对误差满足任务书规定的 `max(1e-2, 32×ULP(g_low))` 限制 |
| 非操作三角及 padding | 与调用前快照逐 bit 比较，匹配率必须为 100% |
| quick return | 整个输出缓冲区逐 bit 不变 |

Cher2k/Cherk 使用 Netlib BLAS 3.10.0 生成 golden；Csyr2 按 Netlib ssyr2 语义实现复数参考。NaN/Inf 的操作区判定遵循任务书引用的混合容差标准。

复用测试包的全部原始用例，并保留 Csyr2 的 22 条、Cher2k/Cherk 各 24 条 `TC_BIT_*` 用例。将 `check_untouched_bitwise.h` 接入 GTest，对成功计算用例检查另一三角与 padding；quick return 另做全缓冲区比较。非操作区哨兵覆盖 NaN payload、±0、±Inf、非规范数和有限值。

## 4.2 性能标准

在 A2（910B3）上对照任务书 §3.3 标杆及配套性能用例验收。预热后有效采样超过 50 次，统计平均单次耗时，A3 同时跑通并记录结果。

性能计时限定在算子执行区间，数据准备、golden 计算及精度比对放在计时区间之外。配套 Python 脚本目前解析 GTest 总耗时，测试接入时需调整为读取独立的算子计时结果，不以 GTest 总耗时作为最终性能依据。

## 4.3 兼容性分析

本次属于已有算子精度修复。公开接口、数据类型、参数顺序、枚举及返回码保持不变，调用方无需调整接口使用方式。通过原有用例和新增 bit 检查验证修复后的行为。
