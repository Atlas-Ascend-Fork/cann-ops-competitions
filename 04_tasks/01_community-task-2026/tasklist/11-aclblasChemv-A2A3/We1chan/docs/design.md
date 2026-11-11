# 需求背景

## 需求来源

社区算子开发任务《aclblasChemv 算子开发任务书（Atlas A2/A3）》。

## 背景介绍

为 `cann/ops-blas` 补齐 Atlas A2/A3 的单精度复数 Hermitian 矩阵向量乘。公共接口 `aclblasChemv` 的参数顺序与 cuBLAS `cublasChemv` 对齐，语义以 Netlib `chemv` 为参考。

# 需求分析

## 需求描述

使用 Ascend C kernel 直调实现 `y = alpha * A * x + beta * y`。`A` 为 `n × n` 的 Hermitian 矩阵，以列主序存于 `lda × n` 的 Device 缓冲区。`uplo` 决定读取哪一半，另一半由共轭关系给出，对角元素只参与实部计算。`x` 和 `y` 支持正负步长，`y` 原地覆写。

### 接口原型

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy);
```

## 需求拆解

1. 在 `include/cann_ops_blas.h` 声明共享接口。复数使用 `aclblasComplex`，实部和虚部各为一个 `float32`。
2. 支持 Host 和 Device 标量、`n=0` 和 `alpha=0, beta=1` 短路、`alpha=0` 的向量缩放以及 `beta=0` 的覆盖写。
3. 在 `test/symv/chemv/arch22/` 注册 CSV 驱动的 GTest。使用 Netlib BLAS 3.10.0 对照全向量，使用 `msprof op` 在 910B3 上验证性能。

# 详细设计

## 算子分析

对逻辑位置 `(i,j)`，若 `i=j`，矩阵值取 `real(A[j*lda+i])`。若在 `uplo` 指定三角内，使用 `A[j*lda+i]`。其余位置使用 `conj(A[i*lda+j])`。负步长的向量从物理位置 `(n-1)*abs(inc)` 开始遍历。

### 数据类型和形状

输入与输出均为交错存储的 complex64。矩阵尺寸为 n×n，列间步幅为 lda。x、y 各包含 n 个逻辑元素，物理跨度由有符号 incx、incy 决定。两种三角方向共用该布局。

## 算子实现

### Host 侧设计

先检查 handle 和 `n`，对有效 handle 的 `n=0` 返回成功。正 `n` 下校验 `uplo`、`lda`、步长、元素跨度和标量指针。需要根据标量值判定空数据指针时，在 handle 的 stream 上读回 Device 标量。其余 Device 标量由 kernel 在同一 stream 上读取。Host 标量值通过 tiling 参数下发。

小尺寸和 DMA 步长无法表示的场景按单行分组。其余按 8、16 或 32 行分组，每核循环领取行组，且每个输出元素只由一个核写回。tiling 参数包含 `n`、`lda`、步长、三角方向、标量来源和参与核数。

对于 Host 标量 α=1、β=0、单位步长、矩阵地址对齐且 handle 已有足够工作区的输入，可使用三角分块路径。中小尺寸若能把完整且对齐的 64×64 三角块一次分给各 AIV，优先使用纯 AIV；从 `n=320` 起，其他适合分块的尺寸可使用混合核。常规分支处理其他参数和工作区不足的情况。两条路径都在 handle 绑定的 stream 上异步执行。

### Kernel 侧设计

输出行组两侧按最多 64 列分块。矩形块均完全位于选中三角，通过 `DataCopyPad` 搬到 UB。直接项与共轭项分别使用对应的地址表展开。向量指令将每一行放在独立 lane 中，按列序累加。对角块按列搬运所选三角，并用掩码分别累加对角线两侧的项。对角元素用实数乘法计算。

直接项先计算 `alpha*x[j]` 后乘矩阵元素，累加到缩放后的 `y[i]`。共轭项先乘 `x[j]` 并累加，再整体乘 `alpha`。上三角在处理对角时合并共轭项，下三角在处理完该行后合并。此顺序与 Netlib 一致，可控制大尺寸偏置输入的浮点误差。

当直接项与共轭项的相消使输出可能超过混合容差的绝对误差上限时，按原顺序对该行重算。重算通过尾数拆分补偿复数乘加中的乘积误差。

`alpha=0` 时只计算 `beta*y`。`beta=0` 时跳过 `y` 的输入读取。常规分支使用原步长覆写 `y`，跨核无共享累加，也无需 GM 临时工作区。

### 三角分块优化

纯 AIV 路径按 64×64 处理选中三角。每个非对角块一次搬入 UB，同时算它对行段的直接贡献和对列段的共轭贡献。各核把局部和写入已有工作区，同步后按输出行合并。

混合路径在中大尺寸用 Cube 处理 64×64 或 128×128 的非对角块。AIV 把 X 打包成两组实数向量；Cube 把交错存储的原矩阵块分别视作行主序和转置后的列主序，在 L1 共用一次搬运，用实数矩阵乘得到直接项和共轭项。尾部非对角块只从 GM 搬运有效矩形，未填满的部分在 L1 补零。L1 双缓冲让下一个块的搬运与本块计算重叠。Cube 关闭 HF32，使用完整 FP32 输入精度。

打包完成后，AIV 在 Cube 计算期间处理对角块，只搬运 `uplo` 允许读取的元素。可用核数充足时，额外的 AIV 分担 128×128 对角块内的第二个 64×64 对角子块。对 `n≥3072` 且尾段不超过 64 行的矩阵，AIV 也同时处理与尾段相连的窄矩形，节省 Cube 的整块补齐和计算。第二次核间同步后，各 AIV 合并自己输出段的局部和并写回。`n=4096` 时使用已有工作区约 2.13 MiB，其中大部分用于 Cube 的分块局部和。

在混合路径上，如果 X 多个分块存在明显分量偏置，重排求和可能与 Netlib 的顺序累加产生较大差异，输出按原顺序计算。合并阶段也检查非有限值，按输出行段复算可疑结果。两类复算都在同一次 kernel 启动中完成。

## 支持硬件

Atlas A2 和 Atlas A3 系列产品（arch22）。性能验收设备为 Atlas 800T A2（910B3）。

## 算子约束限制

`lda ≥ max(1,n)`。步长不得为零或 `INT32_MIN`，向量最大元素偏移不得超过 `UINT32_MAX`。`n>0` 时标量指针必须有效。`alpha` 非零时要求 `A` 和 `x` 有效。除无操作情形外，`y` 必须提供 Device 输出缓冲区。读取 Device 结果前应同步 stream。

# 可维可测分析

## 精度标准与性能标准

使用 `cblas_chemv(CblasColMajor, ...)` 生成 golden，对实部和虚部分别检查 `atol = rtol = 2^-13`、至少 99% 的有限值匹配率及 `max(0.01,32*ULP(golden))` 逐元素硬上限。按任务书单独判定 Inf 和 NaN。测试给未引用三角、对角虚部和列尾 padding 填入 NaN，并检查 `y` 的步长间隙保持原值。

性能在 910B3 上用 `msprof op` 排除 5 次预热，采集至少 10 个 `Task Duration(us)`，逐例取平均并与 A100 基线的 1.25 倍比较。

## 兼容性分析

实现与已有 Symv 算子共享公共头文件和 handle 类型。测试在 arch22 注册，运行时使用当前 handle 绑定的 stream。

## 自验证结果

Atlas 800T A2（910B3），CANN 9.1.0，Netlib BLAS 3.10.0。功能与精度 2,127/2,127 通过，200 条性能形状数值对照全通过。200 条性能用例逐条达标，共采集 2,000 个有效样本。n=4096 实测 76.994 μs，上限 81.621 μs。

[昇腾社区任务](https://www.hiascend.com/activities/task-center/details/cbfd47808791498d94595dd9e48500aa) · [任务讨论](https://gitcode.com/cann/ops-blas/discussions/80)
