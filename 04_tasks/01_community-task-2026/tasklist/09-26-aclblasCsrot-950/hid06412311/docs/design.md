aclblasCsrot 算子设计文档

一、需求背景（required）

1.1 需求来源
CANN 社区任务 2026 — 算子开发任务 aclblasCsrot（Ascend 950PR）。
任务书参考：aclblasCsrot_Atlas950PR_task_doc.md。
上游仓库：https://gitcode.com/cann/ops-blas ，目标目录 blas/rot/arch35/。
接口声明：include/cann_ops_blas.h。

1.2 背景介绍
aclblasCsrot 算子实现优化。
ops-blas 仓的 rot 算子族提供 Givens 平面旋转（Givens rotation）。实数版本 aclblasSrot 已完成 arch35（Ascend 950PR）适配；复数版本 aclblasCsrot 此前只有 arch22（Atlas A2/A3）实现，本任务基于 Ascend C 为 arch35 重新实现。

1.3 算子实现路径和 API 路径
算子实现路径：https://gitcode.com/cann/ops-blas，目标目录 blas/rot/arch35/

接口声明路径：include/cann_ops_blas.h

测试代码路径：test/rot/csrot/arch35/

1.4 aclblasCsrot 算子实现现状分析
arch22 的实现不能直接复用到 arch35，原因有三：

舍入语义不符：arch22 kernel 用 AscendC::Axpy（FMA 语义）计算 cx + sy，而 netlib csrot.f 对每个乘积、每个加法分别舍入。二者在一般情况下只差半个操作数 ULP，但当两项相消时（c == s 且 x == y），参考结果是精确 0 而 FMA 形式留下非零残差，相对误差无界。

入口校验缺失：arch22 的 host 不校验 handle，也不区分 n <= 0 的快速返回，与对齐 cublas 的边界语义不一致。

并行策略：arch22 按单核 SIMD 设计，未利用 arch35 的多 AIV 核与 SIMT 能力。

本实现在 arch35 目录下新建 csrot_host.cpp、csrot_kernel.cpp 等文件，沿用同族 srot 的双路径架构。

1.5 aclblasCsrot 算子功能分析
对两个等长复数向量 x、y 做原地 Givens 旋转，旋转因子 c、s 均为实数（与 crot 的区别：crot 的 s 为复数）。旋转后：

x[i] = c * x[i] + s * y[i]

y[i] = c * y[i] - s * x[i] （使用旋转前的 x[i]）

典型用途是 QR 分解中消去下三角元素，属 BLAS Level-1 auxiliary 例程。

二、需求分析（required）

2.1 需求描述
使用 Ascend C 编程语言实现 aclblasCsrot 算子：

数据类型：COMPLEX64（单精度复数向量）+ FLOAT32 实数旋转因子。

适配硬件：Ascend 950PR（arch35），CANN 9.1.0。

完整对齐 netlib csrot.f 的边界语义与舍入语义。

性能不低于任务书 §3.3 的标杆耗时。

接口定义（与 include/cann_ops_blas.h 中已有声明一致，禁止定义 950PR 私有平行接口）：

    aclblasStatus_t aclblasCsrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y,
    int incy, const float* c, const float* s);
2.2 需求拆解
双路径实现：incx == 1 && incy == 1 走连续 SIMD 路径；其余步长组合（含正非 1、负、零及其混合）走 stride SIMT 路径。

原地运算：x、y 既是输入又是输出，且允许 x/y 指向同一元素（如 incx == incy），需保证读到全部原始分量后再写。

边界语义对齐 netlib：n <= 0 为 no-op 且不校验其余参数；inc == 0 不拦截（同一元素串行旋转 n 次）；负步长从尾端起算；c == 1 && s == 0 不短路。

标量指针位置自动判定：c/s 可位于 host 或 device，算子运行时通过 aclrtPointerGetAttributes 自动判定。

性能不低于任务书 §3.3 标杆。

三、详细设计（required）

3.1 算子分析

3.1.1 数学公式
x'[k] = c * x[k] + s * y[j]
y'[j] = c * y[j] - s * x[k] (x[k] 为旋转前的值)
k = (i-1) * incx, j = (i-1) * incy, i = 1..n

c、s 为实数，实部与虚部按同一组 (c, s) 独立旋转，不存在复数乘法交叉项：
x'[k].real = c * x[k].real + s * y[j].real
x'[k].imag = c * x[k].imag + s * y[j].imag
y'[j].real = c * y[j].real - s * x[k].real
y'[j].imag = c * y[j].imag - s * x[k].imag
这一性质是整个 kernel 设计的基础：一个 Muls 指令即可同时旋转实部与虚部。

3.1.2 支持数据类型
x, y：COMPLEX64（aclblasComplex，8 字节：float real + float imag）

c, s：FLOAT32

计算精度：FLOAT32

3.1.3 支持形状
逻辑上一维 [n]，物理跨度 1 + (n-1) * |inc|。n 任意整数（含 0、负值）；incx/incy 任意整数（含正、负、零）。

3.2 算子实现
3.2.1 host侧设计
内存布局约定：aclblasComplex { float real; float imag; } 是一个 8 字节对，因此 n 个 complex 元素的数组等价于 2n 个 float lane 的数组。kernel 全程以 float 视图操作，向量指令作用在 2 * elementCount 个 lane 上。

tiling策略：

tilingKey：0 表示连续（SIMD），1 表示 stride（SIMT）。

totalN：元素总数。

cosValue / sinValue：c/s 位于 host 时的标量值；位于 device 时置 0 占位。

cIsDevice / sIsDevice：c/s 各自的位置标志。

perCoreN / remainder：连续路径每核元素数 / 前 remainder 个核各多 1 个。

tileSize：连续路径单次搬运的 UB 上限。

incx / incy：stride 路径步长。

nthreads：stride 路径每 block 的 SIMT 线程数。

分核策略：

连续路径：numBlocks = min(n, aivCoreNum)，均分 perCoreN，余数分配给前若干个核。刻意不做 block 对齐取整，使小 n 不会饿死任何核。

stride 路径：numBlocks = min(ceilDiv(n, SIMT_MIN_THREAD_NUM), aivCoreNum)。kernel 内用 grid-stride 循环遍历 n 个元素。

数据分块和内存优化策略：

连续路径是 TQue 三级流水（CopyIn / Compute / CopyOut），使用 5 个 UB tile：inQueueX、inQueueY（VECIN）、outNewX、outNewY（VECOUT）、workBuf（VECCALC）。

tileSize 保持 32 字节 block 对齐，使每个 tile 都起始于对齐边界。

原地安全性：把原始数据在整个 Compute 阶段保留在 VECIN 队列中，结果写入独立的 VECOUT 队列，四路分量全部读完才落盘，因此 x 与 y 指向同一元素也是良定义的。

tilingkey规划策略：
需要 tilingkey 感知 host 侧信息对 kernel 走不同分支。incx == 1 && incy == 1 走 tilingKey=0 的 SIMD 连续路径；其余全部（含负、零、混合）走 tilingKey=1 的 SIMT stride 路径。c/s 的指针位置不进入 tilingKey，kernel 侧统一读 cIsDevice/sIsDevice 标志。

3.2.2 kernel侧设计
与 srot 同构，分为 Init 与 Process 两个阶段，Process 含 CopyIn / Compute / CopyOut 三段。

1. 连续路径（tilingKey 0）：

CopyIn：DataCopyPad 搬入 x/y tile，字节数 dataCount * 8，lane 偏移 curOffset * 2。

Compute：laneCount = dataCount * 2。

Muls(newX, xLocal, cosValue, laneCount) // round(c*x)

Muls(work, yLocal, sinValue, laneCount) // round(s*y)

Add(newX, newX, work, laneCount) // round(cx) + round(sy)

Muls(newY, yLocal, cosValue, laneCount) // round(c*y)

Muls(work, xLocal, -sinValue, laneCount) // round(-s*x)

Add(newY, newY, work, laneCount) // round(cy) + round(-sx)

CopyOut：DataCopyPad 写回。

2. stride 路径（tilingKey 1）：

按 netlib 的访问序定位元素：inc > 0 时 jx/jy = i，inc < 0 时 jx/jy = n-1-i。

舍入：stride 路径的 c * xr + s * yr 在 -O3 下会被编译器收缩成 FMA，省掉一次乘积舍入。因此每个乘积都过一层 CsrotRoundedMul（__simt_callee__ 内联函数，用 volatile 局部量钉住乘积）强制单次舍入。

原地安全性：四个分量（xr/xi/yr/yi）先全部读入寄存器，再执行四条写入，因此 x/y 别名不会污染源操作数。

零步长特例：当 zeroIncX || zeroIncY 时，只有 block0.thread0 按严格 netlib 顺序串行执行 n 次迭代，其余线程与 block 立即返回，保证被复用的地址只被单一程序序触碰。

四、支持硬件

支持的芯片版本    	  涉及勾选

Ascend 950PR（arch35）     	√

Atlas A3 / A2 训练/推理系列   	√（沿用 arch22 既有实现，本任务不改动）

五、算子约束限制
n 为 int。

stride 路径要求 (n-1) * incx 与 (n-1) * incy 均落在 int32 范围内，否则返回 ACLBLAS_STATUS_INVALID_VALUE。

c、s 必须为 float（实数）；复数正弦的 crot 不在本算子范围内。

原地运算，x/y 必须位于 Device 内存。

n <= 0 为合法 no-op，返回成功；n > 0 时 x、y 不得为 nullptr；c、s 不得为 nullptr。

六、可维可测分析

6.1 精度标准/性能标准

验收标准	

1.精度标准	

实部/虚部分别按 FLOAT32 判定：atol = 2^-13，rtol = 2^-13，required_matched_ratio ≥ 0.99，max_abs_error ≤ 1e-2 or 32 × ULP	

标准来源：

任务书 §3.2；生态算子开源精度标准

2.性能标准

n=1048576 / 2097152 / 4194304（incx=incy=1）平均单次耗时 ≤ 21.69 / 43.19 / 119.38 us

标准来源：

任务书 §3.3

精度说明：本算子的误差来源仅为每元素两次实数乘加的浮点舍入。为避免 golden 与算子对“分步舍入 vs FMA”的理解不一致，两侧都显式固定了分步舍入：算子连续路径用 Muls+Muls+Add（本身即单次舍入），stride 路径用 volatile 包装的乘法；golden 由 cblas 的 csrot 生成，编译 cblas 时固定 -ffp-contract=off（或等价选项），确保 golden 也按分步舍入执行，与算子语义一致。

七、兼容性分析

接口 aclblasCsrot 的参数序列与 cublasCsrot 一一对应（handle 及参数顺序），无需额外映射说明。arch22 的旧实现保留在 blas/rot/arch22/ 目录下，按 SOC 分派，不影响既有产品线。新算子作为增量特性，不涉及向前兼容问题。