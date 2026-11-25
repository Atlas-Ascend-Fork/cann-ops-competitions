# Sgemm、Hgemm、Cgemm 算子设计（A2/A3）

# 需求背景（required）

## 需求来源

11 月社区任务「GEMM 类 3 个算子开发（A2/A3）」。接口与计算语义参考 [cuBLAS GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublas-gemm)，参数校验与验收按《aclblasSgemm / aclblasCgemm / aclblasHgemm 类算子开发任务书》执行。

## 背景介绍

在 ops-blas 的 arch22 实现中扩展 Sgemm，补齐 Hgemm 和 Cgemm。三个接口都计算单个输出矩阵。常规乘积由 Cube 计算，AIV 负责输入打包、部分积归约和输出更新。Sgemm 和 Cgemm 的小尺寸路径由 AIV 直接计算。

# 需求分析（required）

## 需求描述

计算 `C ← alpha × op(A) × op(B) + beta × C`。A、B、C 按列主序存储，允许前导维填充，C 原地更新。三个接口均从 Host 读取 alpha 和 beta。

| 接口 | 矩阵和标量类型 | 设备计算 |
| --- | --- | --- |
| `aclblasSgemm` | `float`（FP32） | FP32 乘加与缩放 |
| `aclblasHgemm` | `aclblasHalf`（IEEE binary16 位模式） | FP16 输入，FP32 累加与缩放，结果就近舍入为 FP16 |
| `aclblasCgemm` | `aclblasComplex`（实部、虚部交错存储，各为 FP32） | 按实部和虚部计算复数乘加 |

## 需求拆解

1. 复用 Sgemm、Cgemm 的公共函数原型，在同一头文件新增 Hgemm 声明。
2. 校验参数，处理空输出和零乘积。输出只写 C 的逻辑元素，保留每列填充区。
3. 根据矩阵形状、转置和对齐条件选择计算路径，在句柄绑定的 stream 上提交 kernel。
4. 用任务附件的精度用例和性能子集验收，精度参考与计时方法见「可维可测分析」。

# 详细设计（required）

## 算子分析

### 数学公式

对 `0 ≤ i < m`、`0 ≤ j < n`，计算

```text
C[i,j] = alpha × Σ(p=0…k−1) op(A)[i,p] × op(B)[p,j] + beta × C[i,j]
```

`N` 使用原矩阵，`T` 转置，`C` 共轭转置。对实数矩阵，`C` 与 `T` 等价。

### 支持形状与地址映射

列主序元素下标为 `col × ld + row`。下表中的形状为填充前的矩阵行列数。存储跨度按前导维乘列数计算。

| 矩阵 | N 下的行列数 | T/C 下的行列数 | 最小前导维 |
| --- | --- | --- | --- |
| A | m × k | k × m | `max(1, transa == N ? m : k)` |
| B | k × n | n × k | `max(1, transb == N ? k : n)` |
| C | m × n | m × n | `max(1, m)` |

## 算子实现

### 公共接口

三个接口在 `include/cann_ops_blas.h` 声明，返回 `aclblasStatus_t`，参数顺序均为 `handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc`。转置参数使用 `aclblasOperation_t`。尺寸和前导维使用 `int`。alpha、A、B、beta 为 `const T*`，C 为 `T*`，T 由接口决定。

### Host 侧设计

依次校验句柄、转置枚举、非负尺寸、前导维和标量指针。空句柄返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法转置枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，非法尺寸、前导维或缺少必需指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。

| 条件 | 指针要求 | 操作 |
| --- | --- | --- |
| `m=0` 或 `n=0` | A、B、C 可为空 | 结构校验通过后返回成功 |
| 输出非空，且 `k=0` 或 `alpha=0` | A、B 可为空，C 必须有效 | `beta=1` 时保留 C，`beta=0` 时写零，其余按 beta 缩放 |
| 输出非空，且 `k>0`、`alpha≠0` | A、B、C 均须有效 | 计算乘积并更新 C。beta 为零时跳过旧 C 的读取 |

Host 读取 alpha、beta，将标量值随 tiling 参数传入 kernel。打包、Cube 计算、归约和写回按依赖顺序提交到同一 stream，后续分块在前一分块用完后复用工作区。

### Kernel 侧设计

#### Sgemm

| 路径 | 分派条件 | 计算与写回 |
| --- | --- | --- |
| 小尺寸 AIV | m、n、k 均不超过 64，总乘积项数不超过 4096，且 A、B 的对齐读取跨度合计不超过 2048 个 FP32 元素 | AIV 直接计算和写回，无需全局工作区 |
| 整矩阵 Cube | 小尺寸路径之外，m、n、k 均不超过 8192 | 紧凑且对齐的输入可直接读取。其余输入按前导维打包，或由流水 kernel 直接处理符合搬运条件的原布局。按二维输出 tile 分核 |
| 流水 Split-K | 整矩阵进入 CATLASS 流水，A 为 N、B 为 T/C，且 `m≤128`、`n≤256`、`k≥1024` | Cube 将 K 维切为 8 段，分别写部分积。AIV 归约后写 C |
| 外层分块 | 任一维超过 8192 | M、N、K 各按最多 2048 个元素分块。同一输出块的首段使用原 beta，后续 K 段以 `beta=1` 累加到 C |

Cube 按行主序计算 `op(B)ᵀ × op(A)ᵀ`，其输出布局对应列主序乘积。CATLASS 流水使用 L1 双缓冲。当 `alpha=1`、`beta=0` 且输出布局满足直写条件时，Cube 直接写 C。工作区按所选路径分配 A、B 打包区和 FP32 乘积区，流水 Split-K 为各段部分积保留独立空间。

#### Hgemm

Host 将 M、N、K 各按最多 1024 个元素分块，块内各维向上对齐到 16 个元素。指针对齐、前导维紧凑且各维无需补齐的输入直接送入 Cube。其余输入由 AIV 按物理列搬运，对补齐的行和列填零。

Cube 按 64 × 64 的输出 tile 分核，根据 A、B 的转置组合选择 FP16 输入、FP32 输出的 Matmul 实例。首段 K 写 FP32 累加区，后续段写独立乘积区，再由 AIV 累加。最后一段完成后，AIV 在 FP32 中应用 alpha、beta，将结果舍入为 FP16 并按逻辑长度写回。C 地址按 32 字节对齐时，根据 `ldc` 将相邻列分给同一核，避免两个核写同一个 32 字节块。C 地址未对齐时，由单核写回当前输出块。

工作区含两块 FP16 输入缓冲、一块 FP32 乘积缓冲，以及 K 分段时的第二块 FP32 缓冲，上限为 12 MiB。打包和写回阶段每核的 UB 列缓冲上限分别为 2 KiB 和 10 KiB。Cube 的 L1/L0 分配由静态 Matmul tiling 管理。

#### Cgemm

| 路径 | 分派条件 | 计算与写回 |
| --- | --- | --- |
| 小尺寸 AIV | m、n、k 均不超过 4，前导维满足搬运步长限制，且 `alpha=(1,0)`、`beta=(0,0)` | AIV 直接做复数乘加并写回，无需全局工作区 |
| 窄输出 Cube | `transa=T`、`transb=T/C`、`n≤32`、`256≤m≤4096`、`128≤k≤1024`，且 A 满足基址对齐和前导维限制 | A 保持原布局，AIV 展开 B，Cube 分别输出实部行和虚部行，AIV 交错写回 |
| 通用 Cube | 其余有乘积的输入 | K 按最多 4096 个元素分段。AIV 展开 A，并在 B 的布局或对齐不满足直接读取条件时打包 B |

通用路径按转置或共轭要求取复数分量。对一对复数，Cube 中对应的实数乘法为

```text
[br, bi] × [[ar, ai], [-ai, ar]] = [br·ar − bi·ai, br·ai + bi·ar]
```

工作区保存展开的操作数、需要打包的输入和 FP32 乘积。串行 K 分段及通用路径中的并行 K 归约均为部分积预留空间。AIV 合并部分积，应用复数 alpha、beta，再按 `ldc` 写回。一般布局在无需归约、输出满足对齐条件且 `alpha=(1,0)`、`beta=(0,0)` 时可直接写 C。输出更新跳过复数标量中为零的分量乘法，避免把 `0 × Inf` 引入另一分量。

## 支持硬件

| 产品 | SoC 系列 | 架构 |
| --- | --- | --- |
| Atlas A2 系列 | `ascend910b` | `arch22` |
| Atlas A3 系列 | `ascend910_93` | `arch22` |

## 算子约束限制

A、B、C 为当前设备上的 ND 列主序缓冲，只允许前导维填充。C 的可写区域与 A、B 的可读区域不得重叠。调用方在读取 C、释放设备缓冲或更换工作区前须同步句柄绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

### 精度

| 接口 | CPU 参考 | atol / rtol | 固定绝对误差上限 |
| --- | --- | --- | --- |
| Sgemm | Netlib BLAS 3.10.0 `cblas_sgemm` | 2⁻¹³ / 2⁻¹³ | 0.01 |
| Hgemm | 将 FP16 输入和标量转为 float64，以 float64 累加，结果先收窄到 FP32，再按 IEEE 就近舍入到 FP16 | 2⁻¹⁰ / 2⁻¹⁰ | 0.1 |
| Cgemm | Netlib BLAS 3.10.0 `cblas_cgemm`，实部、虚部分别比较 | 2⁻¹³ / 2⁻¹³ | 0.01 |

Sgemm 与 Cgemm 也可用 float64 累加后收窄到 FP32，并在自测报告中注明参考口径。对逻辑 C 做全量比较。有限值满足 `|actual − golden| ≤ atol + rtol × |golden|` 时计为匹配。用例通过要求匹配率至少为 0.99，且各项绝对误差均不超过固定上限与参考值处 32 ULP 的较大者。NaN、Inf 按任务书引用的 [生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) 判定。

任务附件提供 1,000 条精度用例，Sgemm 334 条、Hgemm 和 Cgemm 各 333 条。覆盖九种转置组合、尾块、前导维填充、零尺寸、零乘积、非法参数和特殊值。用哨兵检查 C 的填充区，使用含 NaN 的旧 C 检查 `beta=0` 路径。Hgemm 单独覆盖 FP16 次正规数、接近上限和溢出场景。

### 性能

任务附件提供 200 条性能用例，Sgemm 67 条、Hgemm 66 条、Cgemm 67 条，其中 24 条使用非默认标量。在 Atlas 800T A2（910B3）、CANN 9.1.0 上采集以下两项指标，每项预热 5 次，连续采集 10 次有效样本并取算术平均。

| 指标 | 采集与判定 |
| --- | --- |
| Kernel 耗时 | 用 `msprof op` 采集 kernel Task Duration。一次接口调用包含多个 kernel 时，按调用汇总参与计算的任务耗时。GPU 基线按接口、转置、尺寸、前导维及 alpha/beta 匹配，将 `gpu_ms` 换算为微秒，逐例判定 NPU 平均耗时是否在 cuBLAS kernel 耗时的 1.25 倍以内 |
| 端到端耗时 | GTest 在预置 Device 缓冲上用 `aclrtEvent` 夹住连续下发，输出 `[PERF] case=<case_name> ... avg_us=<数值>`，供 `verify_performance.py` 交叉验证。正式性能结论取 kernel 耗时 |

任务书 §3.4 的内存验收项为「不涉及」。

## 兼容性分析

实现放在 `blas/gemm/arch22/`，使用 ops-blas 的直调构建。Sgemm 和 Cgemm 的流水路径共用固定版本的 CATLASS 头文件，由 `catlass.cmake` 接入。离线构建可通过 `OPS_BLAS_CATLASS_SOURCE_DIR` 指定对应的本地源码。

任务验收通过 `test/gemm/arch22/` 的 GTest 入口加载用例。A2 构建命令为 `bash build.sh --ops=sgemm,cgemm,hgemm --soc=ascend910b3`，A3 使用对应的 SoC 参数。
