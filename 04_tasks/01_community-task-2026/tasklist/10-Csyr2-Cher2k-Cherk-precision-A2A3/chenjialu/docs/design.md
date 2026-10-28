# 需求背景（required）

## 需求来源

10月社区任务-Csyr2/cher2k/cherk算子精度修复（A2/A3）：
https://www.hiascend.com/activities/task-center/details/fc6145cabc15459ab08b3996cad3b8f9?menu=tasks

任务书及测试附件：
https://www.hiascend.com/p/resource/202610/d7d6833b9324453688ba4f76cbef5f4d.zip

目标仓库：https://gitcode.com/cann/ops-blas

## 背景介绍

本任务修复既有 `aclblasCsyr2`、`aclblasCher2k`、`aclblasCherk` 的 A2/A3 实现。三个接口均为列主序原地三角更新。修复需要同时满足操作区混合容差和非操作区二进制一致，三个接口作为同一任务交付。

对齐搬运、原子加或整块写回即使在非操作区写入数值上相等的结果，也可能改变 NaN 载荷、负零或非规范数。设计以实际写地址属于操作区为约束，不通过事后恢复另一三角实现保护。

# 需求分析（required）

## 需求描述

保持 `include/cann_ops_blas.h` 中的接口声明、参数顺序、枚举数值、Host/Device 归属和返回码不变；在 CANN 9.1.0、Atlas A2/A3 上修复 Ascend C kernel 直调路径。handle 绑定 stream，Host 不增加流同步，不回退 CPU，不增加平行 API。

## 需求拆解

| 项目 | 要求 |
| --- | --- |
| 操作区 | UPPER：`0 <= i <= j < n`；LOWER：`0 <= j <= i < n`。实部、虚部分别按 FLOAT32 混合容差检查。 |
| 非操作区 | 严格另一三角，以及每列 `n <= i < ld` 的 padding。不得写入，与调用前输入逐 bit 相同。 |
| 复数布局 | `aclblasComplex`，实部和虚部各 float32；矩阵地址为 `M[i + j * ld]`，物理输出容量为 `ld * n` 个复数元素。 |
| Csyr2 | 普通转置、不共轭，保留对角虚部的正常计算；支持正负非零 incx/incy。 |
| Cher2k / Cherk | 只支持 OP_N、OP_C；执行更新时对角虚部置正零，旧对角虚部不参与计算。 |
| 快速返回 | 不改变整个输出缓冲，包括原对角虚部和所有特殊值位模式。 |
| 性能 | 满足任务书 §3.3 的 A2（910B3）逐例耗时上限，补充全部附件性能用例；A3 跑通并记录。 |

# 详细设计（required）

## 算子分析

### 数学公式

Csyr2：

```text
A(i,j) <- A(i,j) + alpha*x(i)*y(j) + alpha*y(i)*x(j)
```

x/y 的零基访问起点为 `inc < 0 ? (1-n)*inc : 0`，第 i 个逻辑元素位于 `start + i*inc`。仅更新选中三角，不生成另一三角。

Cher2k：

```text
OP_N: C <- alpha*A*B^H + conj(alpha)*B*A^H + beta*C
OP_C: C <- alpha*A^H*B + conj(alpha)*B^H*A + beta*C
```

Cherk：

```text
OP_N: C <- alpha*A*A^H + beta*C
OP_C: C <- alpha*A^H*A + beta*C
```

Cher2k 的 alpha 为复数、beta 为实数；Cherk 的 alpha/beta 均为实数。Hermitian 性质由选中三角隐含，不向另一三角写入共轭值。

### 接口原型

以下原型复用公开头文件，不增加接口：

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda);

aclblasStatus_t aclblasCher2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);

aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const aclblasComplex* A, int lda,
    const float* beta, aclblasComplex* C, int ldc);
```

### 参数、形状及边界

| 接口 | 输入与输出 | 约束 |
| --- | --- | --- |
| Csyr2 | alpha 为 Host COMPLEX64 标量；x/y 为 Device 复数向量，物理长度分别为 `1+(n-1)*abs(incx/incy)`（n>0）；A 为 Device 原地输出，物理形状 `lda*n`。 | `n>=0`，`incx!=0`、`incy!=0`，`lda>=max(1,n)`。 |
| Cher2k | alpha/beta 为 Device 标量；A/B 为 Device 只读矩阵；C 为 Device 原地输出，物理形状 `ldc*n`。 | `n,k>=0`；N：A/B 逻辑形状 `n*k`、`lda/ldb>=max(1,n)`；C：A/B 逻辑形状 `k*n`、`lda/ldb>=max(1,k)`；`ldc>=max(1,n)`。 |
| Cherk | alpha/beta 为 Device FLOAT32 标量；A 为 Device 只读矩阵；C 为 Device 原地输出，物理形状 `ldc*n`。 | `n,k>=0`；N：A 逻辑形状 `n*k`、`lda>=max(1,n)`；C：A 逻辑形状 `k*n`、`lda>=max(1,k)`；`ldc>=max(1,n)`。 |

向量和矩阵只支持步长、前导维描述的存储，不引入额外广播或高维 Tensor 语义。所有维数、偏移和容量计算检查溢出；地址乘法使用足够宽的整数。

按任务书 §2.4/§2.5 校验参数：空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；合法但不支持的 OP_T 返回 `ACLBLAS_STATUS_INVALID_VALUE`；非法维数、前导维、零步长及所需的空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。alpha、beta 指针不得为空。`beta=0` 表示输出 C 的旧值无需初始化，不表示正常更新可以缺少可写输出存储。

### 快速返回及标量分支

| 条件 | 行为 |
| --- | --- |
| Csyr2：n=0 或 alpha=(0,0) | 返回成功，整个 A 不变。完成所需参数校验后，不访问不需要的 x/y/A。 |
| Cher2k / Cherk：n=0 | 返回成功，不下发更新 kernel。 |
| Cher2k / Cherk：(alpha=0 或 k=0) 且 beta=1 | Device 标量判断后直接结束，不读取乘法操作数、不写 C；不对原对角虚部清零。 |
| Cher2k / Cherk：alpha=0 或 k=0，beta!=1 | 仅缩放选中三角；对角只使用旧实部，输出虚部为 0。 |
| Cher2k / Cherk：beta=0 且需更新 | 不读取旧 C 参与运算；选中三角写入乘积或零值，非操作区仍零写。 |

Device 标量不经同步 D2H 传回 Host。各生产、计算和写回阶段在 Device 上遵循同一标量分支，防止 no-op 路径误写或读取未生成的工作区。

## 算子实现

### Host 侧设计与修改范围

复用既有 arch22 分派、句柄、stream 与工作区管理。Host 校验枚举、维数、前导维、步长和指针，按形状、uplo、trans 划分任务，向 kernel 传递逻辑边界及实际 ld。保留现有公共接口与 arch35 分派。

| 算子 | 实现范围 | 正式测试范围 |
| --- | --- | --- |
| Csyr2 | `blas/syr2/arch22/` 中 csyr2 文件。 | `test/syr2/csyr2/arch22/` |
| Cher2k | `blas/her2k/arch22/` | `test/her2k/cher2k/arch22/` |
| Cherk | `blas/herk/arch22/` | `test/herk/cherk/arch22/` |

路径说明：任务书表格将 Cher2k 写在 herk 目录；当前上游实际实现和测试位于 her2k，修复沿用该既有位置，不新建重复算子目录。公开头文件签名、arch35 实现和其他精度接口不在本次修改范围内。

### Kernel 侧设计

Csyr2 复用 Vector 外积更新。每列 j 的合法行段为 UPPER `[0,j+1)`、LOWER `[j,n)`，按连续行段分块。x/y 按各自有符号步长装载，在 UB 中计算复数乘加；只有该行段的旧 A 可以参与原地更新。移除通过对齐前缀原子加零或浮点复制保留另一三角的写回方式。

Cher2k / Cherk 复用现有 Vector 预处理、Cube 分块乘积与 Vector 后处理。输入转换和中间乘积可在按 n/k 计量的临时工作区中按完整 tile 计算，但原地输出不得由 Cube 整块覆盖。后处理同时裁剪逻辑尾块和 uplo 边界，并在更新时单独处理 Hermitian 对角虚部。

对输出 tile `[r0,r1) x [c0,c1)`，r1/c1 先裁剪至 n；第 j 列的唯一可写行段定义为：

```text
UPPER: [r0, min(r1, j+1))，下界小于上界时写入。
LOWER: [max(r0, j), r1)，下界小于上界时写入。
GM地址: (j*ld + row_begin) * sizeof(aclblasComplex)
字节数: (row_end-row_begin) * sizeof(aclblasComplex)
```

完全处于非操作三角的 tile 不写输出，完全处于操作三角的 tile 可成批搬出，跨对角 tile 按列提交合法行段。GM 起始地址和长度保持精确，不向前对齐到非操作区，也不向后扩展至 padding。UB 对齐所需的边界元素先在局部缓冲重新排列，再使用目标版本支持的精确字节搬运；不能把 UB 对齐需求转化成扩大 GM 写范围。

旧 C 只在 beta 非零且该元素属于操作区时装载；旧对角虚部不参与乘加，避免 NaN 虚部污染实部。所有写路径，包括小矩阵、缩放、直接输出及融合路径，遵循同一个行段约束。

### 性能与资源

保留有效的分块计算、数据复用和流水结构；只对跨三角边界或尾块进行细分，降低写回修复对主计算路径的影响。Csyr2 重点减少多列向量装载与下发成本；Hermitian 族重点复用实数 Cube 乘积与后处理，避免整矩阵恢复拷贝。

不为保护非操作区申请与 n/k 无关的持久 Device 缓存。局部对齐缓冲位于 UB；临时工作区按问题规模计算并遵循现有生命周期。性能分析使用官方 msopprof，分别检查计算、搬运和写回开销。

## 支持硬件

| 硬件 | 软件与验证范围 |
| --- | --- |
| Atlas A2 | CANN 9.1.0；功能、混合容差、bit-exact；Atlas 800I A2（910B3）性能门禁。 |
| Atlas A3 | CANN 9.1.0；功能、混合容差、bit-exact 与性能运行记录。 |

# 可维可测分析

## 精度标准 / 性能标准

操作区 golden：Cher2k/Cherk 使用 Netlib BLAS 3.10.0 的 CBLAS；Csyr2 按 Netlib ssyr2 的普通转置语义生成复数参考。COMPLEX64 实部/虚部分别采用 FLOAT32 混合容差：`rtol=atol=2^-13`、匹配比例至少 0.99，并满足 `max_abs_error_limit=max(1e-2,32*ULP(g_low))`。Inf/NaN 按标准 §2.3 检查，不沿用旧 MERE/MARE 判据替代验收标准。

标准来源：https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md

非操作区逐元素按 8 字节或两个 uint32 比较，匹配率必须为 100%。quick return 额外比较整个输出缓冲，保留原对角位模式；这时不要求对角虚部置零。

| 算子 | 任务书主性能用例 | A2 平均单次耗时上限（us） |
| --- | --- | --- |
| Csyr2 | UPPER n=512；LOWER n=1024；UPPER n=2048，alpha=(1,0)、单位步长、紧凑 lda。 | 7.54；9.72；19.05 |
| Cher2k | UPPER/N n=k=1024；UPPER/N n=k=2048；LOWER/C n=k=1024。 | 1458.91；5240.68；1081.90 |
| Cherk | 同上三组 uplo/trans/n/k。 | 738.09；2616.28；485.94 |

性能先 warmup，再有效采样超过 50 次，按任务附件脚本和对应 gpu_baseline.csv 检查全部 TC_PF 用例；计时使用绑定 stream，输入准备、Host 快照和结果比对不计入算子耗时。

## 测试方法

沿用 ops-blas `build.sh --soc=<目标芯片> --ops=<接口名>` 和 `test/` 下 CSV/GTest 流程。将附件三个原始 CSV 安装到各自 arch22 测试目录，保留每算子原 1000 条精度与 200 条性能用例。使用附件追加脚本增加 Csyr2 22 条、Cher2k/Cherk 各 24 条 TC_BIT，原用例不得删减；按任务要求补齐附件未覆盖的场景。

所有成功计算用例在调用前保存完整 `ld*n` 缓冲，调用并同步 stream 后回读：操作区走混合容差，另一三角与 padding 走 `check_untouched_bitwise.h` 的 memcmp。仅测试边界允许同步，不改变库函数异步行为。

哨兵通过 uint32 位模式写入非操作区，覆盖随机有限数、正负零、不同 payload 的 NaN、正负 Inf 和非规范数。覆盖 UPPER/LOWER、N/C、紧凑和带 padding 的 ld，以及 n=1、7、8、16、63、64、128 和计算分派边界；补充 N/C 的不等 n/k 与非对齐尾块。

边界测试包括 n=0、alpha=0、k=0、beta=0/1、对角虚部为特殊值、正负步长、非法 handle/枚举/维数/前导维/零步长/空指针。针对 beta=0 的操作区填入 NaN，验证未读取旧 C；针对 quick return 对整个输出做 bit 比较。两个平台均记录 CANN、NPU、驱动、Netlib 版本与逐例结果。

## 兼容性分析

三个公开接口、复数结构、返回码、列主序、原地语义和 stream 归属不变。修复限定 arch22，不改变 arch35 和其他 BLAS 接口。开发验证先覆盖实际修改路径，再对三个接口在 A2/A3 上执行完整任务附件；设计评审与代码验收分别完成。
