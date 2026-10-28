# aclblasChpmv（Ascend 950PR）算子设计

## 需求背景（required）

### 需求来源

社区任务《aclblasChpmv 算子开发任务书》，提交团队/账号：`sss-hust`。目标是在 `cann/ops-blas` 补齐 complex64 Hermitian packed 矩阵向量乘，使用 Ascend C kernel 直调和仓库现有 handle/stream 机制。

任务环境要求 Ascend 950PR、CANN 9.1.0，golden 为 Netlib BLAS 3.10.0 的 `cblas_chpmv`。本设计描述当前候选实现，完整硬件精度与性能验收仍在进行；不同工具链的开发测试单独标注，不替代指定版本验收。

### 当前工程基线

检查官方 `cann/ops-blas` 提交 `05b96ea22cea926ecfd43fc7ac691a795522bd61`：`blas/spmv/arch35/` 已有 Sspmv host、tiling 和 SIMT AIV kernel，公共头中尚无 Chpmv 声明。复用该家族的工程注册、分核和 stream 传递方式，不将实对称运算直接当作复 Hermitian 运算。

参考源码：https://gitcode.com/cann/ops-blas/tree/05b96ea22cea926ecfd43fc7ac691a795522bd61/blas/spmv/arch35

## 需求分析（required）

### 功能与接口

计算 `y = alpha * A * x + beta * y`；AP 为列主序 packed 上三角或下三角，长度为 `n*(n+1)/2`，没有 lda。复数使用公共 `aclblasComplex` 类型，实部和虚部各为 float32。

```cpp
aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* AP,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta,
    aclblasComplex* y, int incy);
```

公共声明放在 `include/cann_ops_blas.h`；不新建产品专用平行 API。alpha/beta 按任务书为 Host 指针；AP/x/y 为 Device 指针。实现位于 `blas/spmv/arch35/`，测试位于 `test/spmv/chpmv/arch35/`。

### 参数与行为约束

| 项目 | 约束 |
|---|---|
| n | int，n>=0；负数返回 INVALID_VALUE |
| handle | n 检查之后检查；空句柄返回 HANDLE_IS_NULLPTR |
| n=0 | 有效 handle 下直接 SUCCESS；不检查其余参数，不引用 AP/x/y，不启动 kernel |
| uplo | 仅 UPPER/LOWER |
| incx/incy | 支持正负非零步长，零返回 INVALID_VALUE；不照搬其他任务对 INT_MIN 的特殊拒绝 |
| 指针 | n>0 时 alpha/beta/AP/x/y 均非空，否则 INVALID_VALUE |
| Hermitian | 对角虚部忽略，不参与计算；另一半取共轭 |
| beta=0 | 不读取旧 y；只写逻辑输出位置，保留 stride 间隙 |
| alpha=0 | 可仅计算 beta*y，但仍按上述规则验证参数；含非有限 AP/x 不与 alpha=0 组合测试 |
| stream | 从 handle 获取，异步启动；测试读回前同步该 stream |

检查顺序固定为 `n → handle → n==0 → uplo/stride/pointers`。地址计算先扩展为 64 位再进行乘法与取负；检查字节偏移/地址范围溢出，不因先做 int 乘法或 abs(INT_MIN) 引入未定义行为。

## 详细设计（required）

### 索引及数学公式

索引均从 0 开始。上三角存储 `A(i,j), i<=j` 对应 `i+j*(j+1)/2`；下三角 `A(i,j), i>=j` 对应 `j*(2*n-j+1)/2+(i-j)`。

对输出行 r、归约列 c：

| 模式 | 直接读取区域 | 另一半 |
|---|---|---|
| UPPER | r<=c：AP[r+c*(c+1)/2] | r>c：conj(AP[c+r*(r+1)/2]) |
| LOWER | r>=c：AP[c*(2*n-c+1)/2+r-c] | r<c：conj(AP[r*(2*n-r+1)/2+c-r]) |

当 r==c 时只读取对角实部参与运算，虚部无条件视为零。逻辑向量元素 i 的物理偏移为 `i*inc`（inc>0）或 `(n-1-i)*(-int64_t(inc))`（inc<0），指针仍指向物理分配首地址。

复乘用四个实乘积：`pr=ar*xr-ai*xi`、`pi=ar*xi+ai*xr`。实乘均先舍入到 binary32，再进行加减；SIMT 使用公开的 `fmaf(lhs,rhs,-0.0f)` intrinsic 建立舍入边界，SIMD 使用分离的 `Reg::Mul`、`Reg::Sub`、`Reg::Add`。没有三实乘法重构、HF32 降精度或输入相关的精度阈值调整。

每个输出严格保留 Netlib CHPMV 的更新顺序，不能把 alpha 从各列更新中提取到整行求和之后，也不使用改变归约顺序的线程树。先按 beta=0、beta=1 或通用复乘初始化 y。对 UPPER 的行 r，先顺序累加 c<r 的共轭 `temp2`，再依次执行 `y+=alpha*x[r]*real(diag)`、`y+=alpha*temp2`，最后按 c>r 逐列执行 `y+=(alpha*x[c])*A(r,c)`。对 LOWER，先按 c<r 更新直接贡献，再执行实对角更新，随后顺序累加 c>r 的共轭 `temp2`，最后执行 `y+=alpha*temp2`。独立输出行之间可以并行，单行有效实乘与累加次序保持不变。

### Host 设计

1. 按约定顺序检查参数，合法 no-op 直接返回。
2. 验证 packed 元素数、stride 范围和最大字节偏移可表示；不擅自限制合法的大步长。
3. 读取 Host alpha/beta，形成结构化 tiling（n、uplo、64 位步长、复数标量、线程/分核参数）。
4. 查询实际 AIV 核数，按输出行任务数限定 block 数，使用 handle stream 直接启动设备 kernel。
5. 首版不构造 n×n 临时矩阵、不进行 Host 数值计算、不申请 O(n²) workspace。若后续引入临时空间，另记录分配、同流生命周期与完整调用代价。

### Kernel 设计

实现使用 AIV，按输出行独占状态和最终写回，避免原子更新、跨核归约和 GM 中间矩阵。UPPER/LOWER 通过模板分派。

- n<=128：SIMT 每线程顺序计算独立输出；按实际 AIV 核数分配任务。
- 128<n<=8192：每 block 负责 64 行；512 个 SIMT 线程按列优先协作装载 64×128 的 AP 分块，恢复 Hermitian 镜像共轭。SIMD lane 各负责一行，逐列消费分块，按直接、镜像与对角三个互斥掩码更新状态。累加使用显式 MERGING，保留非当前掩码的行。
- x 与 alpha*x 按逻辑索引各缓存一次，正负步长映射为连续 UB 向量；计算 alpha*x 时仍保留原复乘顺序。
- n>8192：保留不依赖固定 UB 容量的 SIMT GM 通用路径。8192 来自向量缓存容量，不来自用例 ID；边界两侧必须回归。
- beta 为零时在加载 y 前分支，alpha 为零时不读 AP/x。对角装载仅读取实部，不将存储的虚部先乘零。
- AP 真实地址满足 8 字节对齐时可合并读取非对角复数；仅 4 字节对齐仍走两个 float 读取，不增加接口对齐约束。对角始终使用实部读取。
- 所有访问以真实 n 和 stride 为界，尾部行/列显式处理，y 的 padding、AP 与 x 保持不变。

中等规模路径的单个编译器可见静态 arena 包含 raw x 64 KiB、alpha*x 64 KiB、AP tile 64 KiB、64 行的 y/temp2/temp1 状态 1.5 KiB，总计 193.5 KiB。官方 UB 预算为 248 KiB，余量 54.5 KiB 用于编译器临时量及溢出。SIMT/SIMD VF 间使用 `PipeBarrier<PIPE_ALL>()` 和 `DataSyncBarrier<MemDsbT::UB>()` 维护 UB 生产消费依赖；这不代表双缓冲流水重叠。

LOWER 的最后 alpha*temp2 更新在外层列循环结束后，通过单独 SIMD VF 执行一次；不在每个列块消费函数中传递最终块布尔标记。普通调用与 profiler kernel replay 均需验证。

SIMD 路径与旧 SIMT 缓存路径使用不同 kernel entry，隔离各自的共享内存注册资源；Host 按上述 n 范围选择，每次 API 调用仍只启动一个 kernel。资源预算必须同时经实际编译和运行验证，不能只依据源码数组之和判断可运行。

### 优化路径与验证边界

packed 访存同时包含连续与跨列访问。优化以单变量实测比较分块大小、协作装载及 SIMD 消费开销；任何后续路径仍须保持单行 Netlib 更新顺序。性能按完整 API 的 kernel 总时长判定，不只取最快阶段。

所有性能结论均以本算子设备数据为依据。通用路径必须覆盖所有合法参数，不以跳过输出、硬编码结果或放宽容差获得通过。

### 文件计划与兼容性

- `blas/spmv/arch35/chpmv_host.cpp`、`chpmv_kernel.cpp`、`chpmv_tiling_data.h`：参数校验、分派、设备计算。
- `include/cann_ops_blas.h`：新增公共接口声明，沿用公共复数和状态类型。
- `blas/spmv/README.md`：增加 Chpmv API、packed/stride 说明、Ascend 950PR 支持信息。
- `test/spmv/chpmv/arch35/`：GTest、CSV、独立 golden、边界与特殊值检查。

不改动已有 Sspmv 的接口或算法；新接口的其他架构支持不得误标为已完成。

## 可维可测分析

### 功能与精度

原附件 CSV 共 1200 条，其中 200 条 TC_PF；不修改原始参数或覆盖用例。不把测试进程退出 0 当作用例全通过，核对逐 case 身份、执行数、失败/跳过/无结果数。另补 handle 空、n=0 与非法参数叠加、INT_MIN 宽位算术、beta=0 的 y 毒化、对角虚部毒化、非默认 stream、stride gap 和输入只读检查。

golden 使用 Netlib BLAS 3.10.0 的 `cblas_chpmv(CblasColMajor,...)`，独立构造 packed 输入，不复用设备索引实现生成 golden。全向量实/虚分别计算：rtol=atol=2^-13，matched_ratio>=0.99，最大绝对误差符合任务书 `1e-2 or 32*ULP` 约束，逐项保存最大误差及 ULP。保留任务书混合容差与 README 的 MERE/MARE 两套结果及来源，不混为一个通过标志。

非有限值遵循本任务特定规则：同为 NaN、同号 Inf，或 NaN 与任意符号 Inf 等效；有限/非有限不等效；反号 Inf 不等效。非有限元素单独计数，不加入有限 matched_ratio/最大误差；必须全部按规则通过。只用于本任务，不扩展为通用数值规则。

原 CSV 的 RANDOM_NORM_5_5 名称不证明已有正态分布。原始输入路径保留，另记录均匀/正态各 50% 的补充覆盖（固定 seed、μ/σ、实虚独立采样）；不覆盖原 CSV。对角虚部非零或 NaN 的毒化输入是额外语义测试，常规随机输入按 Hermitian 约束生成。

### 性能及材料差异

门槛为每条性能用例 `GPU_ms*1000 / NPU_us >= 0.4`。基线来自附件 `gpu_baseline.csv`，GPU 环境为驱动 550.54、CUDA 12.4。200 行基线与 200 条 TC_PF 建立显式映射，重复参数仍保留其各自 case ID，不使用参数字典去重。

任务书 §3.3 指定 msprof op 5 次 warmup 后 10 次有效样本均值；§7 写大于 10 次，README 写大于 50 次。本实现计划保存 5 次 warmup 后至少 51 次样本，单列前 10 次与全部 51 次的均值/逐项判定以及各自来源，交付中显式记录采样口径差异，由评审确认正式统计列；不得静默择取最有利口径。warmup、replay/launch 身份和多 kernel 调用归属必须可核对，不能把同一 kernel 的 replay 当成不同 API 样本而不披露。

附带 verify_performance.py 使用 GTest 整数毫秒整条测试时间，包含准备/golden/比对，不能用作 task duration。正式统计从原始 msprof CSV 获取 kernel duration，统一 us；返回失败、采集缺失、零时长、NO_REF、共享设备或截断均不判通过。异常样本与原因全部保留，不靠反复重测选择最快值。

### 交付与状态

冻结源码 commit、库哈希、原始 1200 条清单及 200 行基线，记录 CANN/驱动/硬件/golden 版本。提交源码、README、自验证步骤、精度/性能/内存报告及对应日志；本任务内存门槛不涉及，不宣称未经执行的专项泄漏测试通过。

本 MR 仅提交设计文档。当前候选已完成部分设备验证，完整 1200 项原始精度用例和 200 项性能用例须绑定同一冻结版本逐项验证；设计 MR 状态不等同于任务验收通过。
