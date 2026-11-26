# aclblasCgemm3m 算子设计（Atlas A2/A3）

## 需求背景

### 需求来源

11 月 CANN 社区任务要求在 Atlas A2/A3 上提供 `aclblasCgemm3m`，以 cuBLAS `cublasCgemm3m` 为接口和性能对标。算子在 ops-blas 的 arch22 实现中用 Ascend C kernel 直调完成计算。[昇腾任务](https://www.hiascend.com/activities/task-center/details/cae74cb900484862a2b43b24c0f7d9d3) · [任务讨论](https://gitcode.com/cann/ops-blas/discussions/93)

### 背景介绍

复数矩阵乘可拆成实部与虚部的四次实数矩阵乘。3M 将其中两项的和通过 Gauss 恒等式求出，只需三次实数矩阵乘。三乘法的点积和复数组合采用不同的舍入顺序，因此输出按任务书的浮点容差验证。

## 需求分析

### 需求描述

`aclblasCgemm3m` 计算 `C = alpha × op(A) × op(B) + beta × C`。`op(A)` 的逻辑形状为 `m × k`，`op(B)` 为 `k × n`，`C` 为 `m × n`。输入和输出在 device 内存中按列主序保存，实部和虚部交错。`alpha` 和 `beta` 为 host 上的 COMPLEX64 标量。

### 需求拆解

1. 复用 ops-blas 的句柄、stream、前导维校验和工作区管理。覆盖 N、T、C 的九种组合。
2. 复用实数 GEMM 计算三组乘积，按列交错写回输出。支持 `alpha=0`、`beta=0`、`k=0`、零尺寸及合法 padding。
3. 用官方 CSV 在 A2/A3 验证精度。在 Atlas 800T A2（910B3）采集设备事件和 `msprof op` 耗时。

## 详细设计

### 算子分析

对 `op(A)` 记 `Ar + iAi`，对 `op(B)` 记 `Br + iBi`。当操作为共轭转置时，打包阶段取反相应虚部，实数 GEMM 使用转置布局。

```text
P0 = Ar × Br
P1 = Ai × Bi
P2 = (Ar + Ai) × (Br + Bi)
product.real = P0 − P1
product.imag = P2 − P0 − P1
C = alpha × product + beta × C
```

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `handle` | `aclblasHandle_t` | 有效句柄，决定使用的 stream 和工作区 |
| `transa`、`transb` | `aclblasOperation_t` | `ACLBLAS_OP_N`、`ACLBLAS_OP_T`、`ACLBLAS_OP_C` |
| `m`、`n`、`k` | `int` | 非负 |
| `alpha`、`beta` | `const aclblasComplex*` | host 上有效标量地址 |
| `A`、`B` | `const aclblasComplex*` | device 上列主序复数矩阵，实际参与内积时要求有效地址 |
| `C` | `aclblasComplex*` | 非空输出要求有效 device 地址 |
| `lda` | `int` | N 时至少 `max(1,m)`，T/C 时至少 `max(1,k)` |
| `ldb` | `int` | N 时至少 `max(1,k)`，T/C 时至少 `max(1,n)` |
| `ldc` | `int` | 至少 `max(1,m)` |

### 算子实现

#### Host 侧设计

`blas/gemm/arch22/cgemm3m_host.cpp` 提供 C ABI 入口，调用 `RunCgemmArch22`。接口先校验句柄、枚举、维度、前导维和标量，再处理空输出，最后检查实际访问的矩阵指针。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法维度及必需指针缺失返回 `ACLBLAS_STATUS_INVALID_VALUE`。

`m=0` 或 `n=0` 时直接返回成功。`k=0` 或 `alpha=0` 时只执行 C 的缩放，`beta=1` 时直接返回。`beta=0` 的写回路径只使用新结果。普通路径在同一 stream 依次启动范围扫描、A/B 打包、分段实数 GEMM 和复数写回。

Host 根据实际存储的行列数规划各平面的前导维。打包平面按 16 个 FP32 元素对齐。若 `PA`、`PB`、`PR` 为对齐后的 A、B、输出单平面元素数，持久工作区以 FP32 元素计为 `3 × PA + 3 × PB + 5 × PR + 16 × R`，其中 `R` 是每个输入的范围标记数。实数 GEMM 的临时区域单独切片为 `2 × (PA + PB + 4 × PR)` 加固定余量，子句柄以用户工作区方式借用该切片。尺寸采用检查溢出的 64 位运算，工作区申请失败时传回对应状态码。

#### Kernel 侧设计

对 NN 且 m、n、k 均不超过 32 的有效乘积，单个 AIV kernel 完成 3M 计算与写回。按 C 的列分核，把 A 和对应的 B 列搬入 UB，累加三组实数乘积后交错写回。极值输入或较大的 alpha 使用带残差的计算路径。beta 为零时跳过读取历史 C。

| 阶段 | 分核与计算 |
| --- | --- |
| 范围扫描 | AIV 按输入列分工，统计两个操作数的绝对幅值。每核写入 32 字节对齐的范围标记。极大幅值以 2 的幂缩放，避免打包求和溢出。 |
| 解交织和打包 | AIV 块数上限取硬件可用核数和存储列数。每核轮转处理列，按最多 4096 个复数元素分段。`GatherMask` 拆出实部和虚部，同时生成第三平面。范围标记一次连续搬入 UB 后用向量归约。|
| 三次实数 GEMM | 实数 GEMM 沿 K 分段，当前实现使用最多 512 项的小段，多个分段累加到对应结果平面。对已知存在地址对齐风险的转置路径，子段大小避开会触发该路径的对齐条件。 |
| 复数合并 | AIV 按列分工，每次最多处理 1024 个输出元素。由三个结果平面重建实部、虚部，应用复数 alpha 和 beta，交错写回 `C`。只写列中的前 `m` 个元素。 |

当输入进行了指数缩放，或复数标量会放大中间抵消误差时，写回阶段用两份 FP32 数保存高位和残差，重新计算对应内积。此路径覆盖极值输入，也造成大 alpha 用例的额外耗时。

#### 支持硬件及约束

| 机型 | 验证用途 |
| --- | --- |
| Atlas 800T A2，910B3 | 精度及正式性能自测，CANN 9.1.0 |
| Atlas A3，910_9382 | 单卡精度交叉验证 |

接口支持 COMPLEX64、列主序和普通单矩阵 GEMM。A/B 与 C 的重叠存储、批量 GEMM 和广播不在本接口范围内。

## 可维可测分析

### 精度标准和性能标准

| 项目 | 标准与现状 |
| --- | --- |
| 精度 | 用 Netlib BLAS 3.10.0 的 FP64 累加收窄至 COMPLEX64，按任务书分别比对实部、虚部。`atol=rtol=2^-13`，分量匹配率至少 99%，最大绝对误差不超过 `max(0.01, 32 ULP)`。A2 上 1000 条均通过。 |
| 性能 | 910B3 单卡每条预热 5 次，再采样 10 次。正式数据为 `msprof op` 下每次调用内全部 kernel task duration 之和，单用例平均值上限为同参数 GPU kernel 耗时的 1.25 倍。应用事件计时供交叉验证。小矩阵 V2 的六条正式样本均达标。其中 16³ 为 6.560 µs，门槛为 13.312 µs。全量正式复验继续进行，当前 256³ 样本超出门槛。 |
| 用例问题 | 随包 200 条性能用例中有 60 条 `lda` 或 `ldb` 低于参数下限。此类用例返回 `ACLBLAS_STATUS_INVALID_VALUE`。待任务方确认修订参数及对应 GPU 基准。 |

### 兼容性分析

新入口使用原有 ops-blas 句柄和公开签名，arch22 的 Cgemm 调用继续使用四乘法。A2 上的 Cgemm 独立冒烟覆盖九种转置、长 K、`k=0`、`beta=0` 和 padding 检查。arch35 由该架构原有实现提供。
