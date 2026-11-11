# 需求背景（required）

## 需求来源

11月社区任务：[aclblasChemv 算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/cbfd47808791498d94595dd9e48500aa)。
依据 2026 年 11 月 11 日发布的 `aclblasChemv_AtlasA2_A3_task_doc.md` 和随包测试指导设计。
相关讨论：[ops-blas #80](https://gitcode.com/cann/ops-blas/discussions/80)。

## 背景介绍

在 ops-blas 中补齐单精度复数 Hermitian 矩阵与向量乘法。
Hermitian 矩阵满足 `A = A^H`，只存储并引用指定的一个三角，另一三角由共轭关系推导。
不能直接沿用复数对称矩阵的非共轭语义，也不能把对角虚部参与运算。

使用仓库句柄式 BLAS API 和 Ascend C kernel 直调方式，公共声明放入
`include/cann_ops_blas.h`，A2/A3 实现位于 `blas/symv/arch22/`，
测试位于 `test/symv/chemv/arch22/`。

# 需求分析（required）

## 需求描述

计算 `y = alpha * A * x + beta * y`，在 handle 绑定的 stream 上异步执行。
alpha、beta、A、x、y 均为 complex64，实部和虚部各为 float32。

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

| 参数 | 类型及内存 | 约束与语义 |
| --- | --- | --- |
| handle | Host 句柄 | 非空，携带执行 stream |
| uplo | Host 枚举 | ACLBLAS_UPPER 或 ACLBLAS_LOWER |
| n、lda | Host int | n≥0；非空计算时 lda≥max(1,n) |
| alpha、beta | complex64，Host 或 Device | 运行时探测指针属性；n>0 时非空 |
| A | complex64，Device | 列主序 lda×n，仅引用指定三角；对角只读实部 |
| x | complex64，Device | 逻辑长度 n，支持正负 incx |
| y | complex64，Device，原位输出 | 逻辑长度 n，支持正负 incy；只覆写逻辑元素 |
| incx、incy | Host int | 非零且非 INT32_MIN；跨度以 int64 计算并检查可表示范围 |

## 需求拆解

1. 共用唯一公共 API，不引入按产品区分的平行接口。
2. 支持两种 uplo、非对齐 n、lda padding、正负步长，以及 Host/Device 标量组合。
3. 保持任务书定义的校验顺序和错误码，负向测试在读取输入前返回。
4. `n=0` 不访问任何数据；`alpha=0,beta=1` 不访问 A/x/y；
   `alpha=0` 不读 A/x；`beta=0` 不读旧 y，但实际输出仍须有有效 y 指针。
5. 使用 GTest 和 CSV 覆盖全部正式用例，并补齐正态分布和读取边界测试。
6. 以正式 A100 基线逐例检查 0.8 倍性能要求，输出可复现的精度、性能和材料。

# 详细设计（required）

## 算子分析

### 数学和地址映射

令 `a(i,j)` 为逻辑矩阵元素。UPPER 模式下：

```text
i < j : a(i,j) = A[j*lda+i]
i = j : a(i,i) = real(A[i*lda+i])
i > j : a(i,j) = conjugate(A[i*lda+j])
```

LOWER 模式交换直接读取侧和共轭侧。任何模式都不先把整矩阵读入，避免读取未引用三角。
对角以单个 float 读取实部，避免对角虚部的 NaN/Inf 传播。

向量逻辑元素 i 的物理偏移为：

```text
inc > 0 : i*inc
inc < 0 : (n-1-i)*(-int64(inc))
```

在 Host 侧先用 int64 检查 `(n-1)*abs(inc)`，再转换 kernel 参数。
拒绝 INT32_MIN 和超出既有跨度表示范围的参数，不能先在 int 中取绝对值或做乘法。

复数乘法采用 `(ar*br-ai*bi, ar*bi+ai*br)`。
每个输出只由一个核拥有，避免跨核原子累加、额外 workspace 和全局同步。
计算复杂度为 O(n²)，矩阵请求读流量约为 `8*n²` 字节。向量和输出的理想流量下界为 O(n)；分组路径在每核缓存 x，通用逐行路径可能重复读取 x，实际请求流量可为 O(n²)。请求字节数不等同于经过 L2 后的 HBM 实际流量。

## 算子实现

### Host 侧设计

参数校验按任务书执行：

1. handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`。
2. n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE`；n=0 立即返回 SUCCESS。
3. 检查 uplo、lda、步长、跨度和 alpha/beta 指针。非法 uplo 返回 INVALID_ENUM，
   其余非法参数返回 INVALID_VALUE。
4. 使用仓内标量读取辅助逻辑，在绑定 stream 上读取 Host/Device alpha/beta，
   检查运行时错误；不直接在 Host 解引用 Device 指针。
5. alpha=0 且 beta=1 立即返回成功；其他情况按实际依赖检查 A/x/y。
6. 获取 AIV 核数，计算分组和尾行，调用 Ascend C kernel。错误不能被当作 SUCCESS 忽略。

分核保持每核输出独立。通用路径将行分配到核；连续 x/y 的中等及大矩阵使用多行分组，
把不同输出映射到 SIMD lane。核数不超过设备可用核数和实际输出任务数。

### Kernel 侧设计

通用路径将一行拆为直接侧、实对角和共轭侧，按块读取合法元素。
连续区间使用 DMA，跨列读取使用带步长搬运；当硬件 DMA 字段不能表示跨度时采用
合法的小块访问路径，不能截断地址。非单位步长 x/y 均按上述 64 位偏移处理。

连续 x/y 使用分组路径：16/32/64 行为一组、列块宽度 16。
同一 lane 持有同一输出，批量计算各列的复数乘积，再按原列序加到对应输出。
这样利用输出间并行，避免为了归约改变单输出累加顺序。
直接侧先形成 alpha*x，共轭侧先形成累加和后乘 alpha；对角与两侧的组合次序
按 UPPER/LOWER 对齐 Netlib 参考算法，以降低抵消和中间溢出带来的精度风险。

对角块使用合法三角片段 DMA 加独立实对角读入，再用行掩码区分不同 lane 的累加边界。
尾行和尾列使用实际有效长度，不能把 padding、另一三角或步长间隙纳入运算。

通用路径的 UB 使用量与 n 无关。分组路径缓存连续 x，当前分组上限为 n=8192，
最大 UB 预算约 173KiB；更大规模或不满足布局条件时回退通用路径。
分组门限是实现选择，接口不将 n=8192 作为合法尺寸上限。
所有 DMA/Vector/Scalar 事件由所属 TPipe 获取 ID 并成对设置、等待；
不假设固定事件 ID 在嵌套流水中可安全复用。

### 特殊值和输出

alpha=0 时直接执行 beta*y 缩放，不读取 A/x；beta=0 时从零初始化输出，
不能通过 `0*old_y` 规避读取，否则 NaN 会污染结果。
每行只写回对应逻辑 y，padding 和步长间隙保持不变。
alpha=0,beta=1 和 n=0 在 Host 已短路，不下发无用 kernel。

### 性能分析和迭代边界

优先合并矩阵搬运、缓存 x、提高多行 SIMD 利用率，并按规模选择分组，
同时保留负步长和小规模的轻量路径。
正式 4096 场景的矩阵读流量估计约为 128MiB；以 A100 对应基线换算，
需要检查访存带宽、Vector/Scalar 指令开销及下发开销各自是否能满足逐例预算。
公式是成本估算，不代表任何设备已达到该带宽或性能门槛。

优化候选先验证误差、合法读取和受影响边界，再测代表性能及完整性能集。
不以重排累加、放宽精度、删除正式慢用例或改变计时范围获得通过。
性能尚未满足要求时如实记录失败，不将参考准备结果写成正式验收通过。

## 支持硬件

| 产品 | 实现与构建路径 | 验证安排 |
| --- | --- | --- |
| Atlas A2 系列，包括 Atlas 800T A2 | arch22 | 优先在 910B3 完成正式精度及性能验证 |
| Atlas A3 系列，包括 Atlas 800T A3 | arch22 | 保留编译和兼容回归，按需要补实际设备结果 |

任务书 §3.1 规定自验证只需覆盖一种款型；开发记录分别保存实际芯片、CANN、驱动和命令。
以 CANN 9.1.0 为正式环境要求，不用其他版本或其他款型结果替代其环境证明。

## 算子约束限制

无广播、额外 dtype、稀疏格式或超出 lda/incx/incy 的视图语义；y 原位更新。
不新增非任务要求的 Host/Device 别名支持，不假设 A、x 与 y 可以重叠。
Device 结果由调用者在绑定 stream 同步后读取。

# 可维可测分析

## 工程集成和精度验证

在 `test/symv/chemv/CMakeLists.txt` 为 arch22 注册
`ops_blas_add_gtest_tests(${OPS_BLAS} chemv_test)`，使用仓库根构建入口，
不能退回整数毫秒的非 GTest runner。

以任务包的 1000 条精度和 200 条性能用例为固定输入，保留原 CSV 哈希。
为精度用例补均匀和正态数据，正态参数覆盖 μ∈[-5,5]、σ∈[0.1,2]，
alpha/beta 的实部、虚部分别采样；新增 Host/Device 标量组合、
强抵消、中间溢出和非对齐行列控制。

负向用例使用单元素 dummy 或空指针；不得按非法 n/步长申请巨量缓冲。
quick return 用例不为不会读取的输入申请与步长对应的存储。
正常的负步长用例仍按合法物理跨度分配和填充，覆盖 ±1、±2、±3。

Golden 使用未修改的 Netlib BLAS/CBLAS 3.10.0 `cblas_chemv`，按 ColMajor 调用。
对实、虚分量分别统计 atol=rtol=2^-13、matched_ratio≥0.99、
max_abs_error≤max(0.01,32*ULP)，按标准规定的位置取 ULP。
非有限值单独统计：NaN/NaN、同号 Inf/Inf、NaN/Inf 等效，
有限/非有限不等效；全部非有限元素必须满足规则。
CSV 的 MERE/MARE 列另行保留，不用于放宽生态精度标准。

通过 NaN 填充、访问模型和设备输出守卫检查未引用三角、对角虚部、padding、
alpha=0 的 A/x、beta=0 的旧 y，以及步长间隙。
原生测试检查返回码、完整输出、同步状态和未改写间隙。

## 性能标准和测量

逐例要求 `NPU平均耗时(us) <= A100 gpu_ms*1000/0.8`。
预热至少 5 次，固定有效样本数且不少于 10，异常和失败调用单独报告。
设备内存预置，H2D/D2H、Host 造数和 golden 比较在计时范围外。

正式任务书 §3.3 指定 msprof op Task Duration，随包 README 和
`verify_performance.py` 则要求 aclrtEvent 连续下发均值 `[PERF] ... avg_us=`。
两种口径分别采集并标注，分别与正式基线比较，不混成一个耗时；
GTest 按脚本所需字段输出事件耗时，msprof 保留原始逐 kernel 数据用于核对正文要求。
验收报告按两种口径分别列出全部用例；存在不一致时保留原始数据，不能择优替换判据。
不能以 GTest 整数毫秒、CPU 墙钟或 profiler 启动总时间替代任一测量。
每条性能用例同时检查成功返回、数值输出和缓冲区守卫，保留全部 200 条结果。

## 兼容性、复现和交付

保持 Ssymv/Csymv 和其他架构的既有接口及构建注册；公共头文件仅补共用声明。
使用公开、支持的 SDK API，不依赖其他算子的内部实现或安装目录。
测试说明给出从固定提交构建、运行全部套件和收集结果的命令，
记录精确源码提交、正式输入哈希、环境、命令和完整原始日志。

正式验收前完成设计评审合入、全部必要验证和审核代码访问核查；
review 分支包含测试脚本、复现说明以及根目录 task_submission 的七项模板材料。
验收说明使用完整 review commit 链接，材料包与同一冻结候选绑定。
实现迭代、生产合入候选和验收入口分别使用 develop、submission、review；
完整交付件交给官方测试审核后，按社区流程从 submission 提出代码合入 PR。
