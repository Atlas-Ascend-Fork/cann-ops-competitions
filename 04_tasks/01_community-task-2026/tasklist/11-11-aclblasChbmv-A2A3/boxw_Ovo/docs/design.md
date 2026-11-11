# 需求背景（required）

## 需求来源

11月社区任务「aclblasChbmv算子开发（A2/A3）」，开发账号 boxw_Ovo：

- [官方任务页](https://www.hiascend.com/activities/task-center/details/ea8efcc9f98c4d54930f8b8e94a59366)。
- [正式任务书与用例包](https://www.hiascend.com/p/resource/202611/87d7ffb2d85c4a3eba8727ec14683bec.zip)。
- [官方讨论](https://gitcode.com/cann/ops-blas/discussions/81)。
- [ops-blas](https://gitcode.com/cann/ops-blas)。

目标环境为 CANN 9.1.0，适配 Atlas A2/A3。任务书 §3.1 规定自验证只需一种款型，计划在 Atlas 800T A2（910B3）完成自验证。

## 背景介绍

Chbmv 完成单精度复数 Hermitian 带状矩阵与向量的乘加。仅保存上三角带或下三角带，另一侧按共轭对称补全，减少窄带矩阵的存储。实现采用 ops-blas 句柄式接口和 Ascend C kernel 直调，与同族 Ssbmv 共用目录族。

# 需求分析（required）

## 需求描述

实现 `y = alpha * A * x + beta * y`，参数序列与 cuBLAS cublasChbmv 对齐，维数为 int，公共 API 声明位于 include/cann_ops_blas.h。alpha、beta 位于 Host，A、x、y 位于 Device；y 原地更新。数据均为 aclblasComplex，实部和虚部各为 float32。

```cpp
aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

## 需求拆解

| 项目 | 要求 |
| --- | --- |
| 存储 | A 为列主序 lda×n 带状数组，支持 UPPER/LOWER，不展开 n×n |
| Hermitian | 镜像侧取共轭，对角只引用实部，未定义带外存储不参与运算 |
| 带宽 | k≥0，有效范围 min(k,n−1)，UPPER 物理对角仍在原始 k 行 |
| 向量 | 支持正负 incx/incy；0、INT32_MIN 和跨度超 int32 返回 INVALID_VALUE |
| 前导维 | int64 计算和比较 lda≥k+1，避免 int32 溢出 |
| 校验顺序 | handle 非空 → n≥0 → n=0 成功返回 → 其余参数 |
| 标量边界 | beta=0 不读旧 y；alpha=0 计算 beta*y；n=0 不启动 kernel |
| 异步 | 使用 handle 绑定的 stream，读回前由调用方同步 |
| 工程 | 实现位于 blas/sbmv/arch22/，测试位于 test/sbmv/chbmv/arch22/ |

# 详细设计（required）

## 算子分析

### 数学公式与存储

令 B=min(k,n−1)，输出行 i 只访问 j∈[max(0,i−B),min(n−1,i+B)]：

`y[i] = alpha * sum(A(i,j) * x[j]) + beta * y[i]`。

下表使用从 0 开始的下标：

| 模式 | 直接存储元素 | 复数元素偏移 |
| --- | --- | --- |
| UPPER | i≤j 且 j−i≤B | j*lda + k+i−j |
| LOWER | j≤i 且 i−j≤B | j*lda + i−j |

未存储侧从 A(j,i) 读取后取共轭。主对角仅读取对应地址的 real 分量。负步长时，逻辑 i 映射到物理位置 `(n−1−i)*abs(inc)`。逻辑带宽与原始 k/lda 分别传递，地址中间量使用 64 位。

### 支持数据类型和形状

仅支持 complex64。n≥0，k≥0，lda≥k+1；x/y 各至少有 `1+(n−1)*abs(inc)` 个复数元素。API 不以公开 CSV 最大尺寸为上限。无 trans、batch、广播或额外视图参数。

## 算子实现

### Host 侧设计

先检查 handle，为空时返回 HANDLE_IS_NULLPTR；再检查负 n，对 n=0 返回 SUCCESS。非空场景校验 uplo/k/lda、步长与跨度、标量及 A/x/y 指针，非法返回 INVALID_VALUE。beta=0 仍要求输出 y 非空。

k+1、abs(inc)、(n−1)*abs(inc) 在 int64 域计算；拒绝 INT32_MIN 和跨度超过 INT32_MAX 的参数。读取 Host 标量，把数值、n、原始 storageK、effectiveK、lda、步长及三角模式传入 kernel。

按实际可用 AIV 数分核。沿物理地址排列输出，每组最多四个 complex64 元素。y 起点 32 字节对齐时按组分核，使核间输出缓存行不相交；非对齐时用单核。负 incy 只改变物理组到逻辑行的映射。

### Kernel 侧设计

按输出行独立计算，无核间归约或原子写入。每行拆为直接存储侧、实对角和共轭侧，按带内范围分块，每块最多 128 个复数元素。

1. 连续 A/x 使用连续 DMA；固定间隔输入使用 stride 搬运，在 UB 提取实部/虚部。DMA 字段无法表示的间隔使用 64 位地址标量读取。
2. 在 UB 将复数乘积表示为高低两部分，以乘法残差和补偿归约降低舍入误差，镜像侧虚部变号。对角做实数缩放，不读取其虚部。
3. beta 非零时读取旧 y，beta=0 从零组合。每次只写该输出元素的 8 字节，保护步长空隙和尾部。
4. 有效带宽≤4 时使用短循环，省去通用向量准备和多余事件。
5. 检查输入范围与每行误差界；非有限、中间溢出风险或误差界超出容差时，使用整数实现的宽精度标量路径，分别验证有限/非有限输出行为。

不分配 n×n 临时矩阵和全局 workspace。128 元素分块的主要缓冲为 A/x 各 4 KiB、16 个计算平面共 8 KiB、3 个索引平面共 1.5 KiB、32 字节输出区，总计 17952 字节。事件由 TPipe 管理，建立搬入、向量计算、标量访问和搬出的依赖。

### 数值策略

输出为 FP32；golden 按正式要求将原始 FP32 输入逐值转 FP64，由 Netlib BLAS 3.10.0 cblas_zhbmv 计算，再收窄为 FP32。golden 独立于 kernel 的乘加顺序，不以 FP32 参考误差放宽阈值。

一般有限值在 FP32 中保留乘法低位残差，用高低两部分完成分块归约与行组合，并计算保守误差界；超出容差的行回退到整数实现的 binary64 运算。回退按 Netlib 的直接侧、实对角与共轭侧顺序处理，避免提前溢出改变有限结果。首轮原生验证先检查 FusedMulAdd 残差位值及整行抵消控制，再执行完整数值、正态及性能集；不能用 CPU 模型或编译结果替代硬件舍入证据。

### 性能分析与优化

逻辑带内元素数 `L=n*(2B+1)−B*(B+1)`，核心计算量 O(L)。按行读取时，每个非对角存储元素服务于两个输出，A 和 x 的有效读取流量各约 8L 字节；y 输出 8n 字节，beta 非零另读 8n。DMA 对齐、下发、事件和标量累加会增加成本，该模型不等于实测性能。

优先优化窄带准备、连续搬运、x 数据复用和宽带归约。每个改动进行独立正确性检查和性能对照，最终结果绑定到同一冻结源码。目标为全部正式性能用例达到 A100 0.8 倍以上，由 910B3/CANN 9.1 实测确认。

## 支持硬件

| 产品 | 适配方案 | 自验证计划 |
| --- | --- | --- |
| Atlas A2（910B） | arch22 公共实现 | 910B3 / CANN 9.1.0 |
| Atlas A3 | arch22 公共实现与对应编译目标 | 按 §3.1 一种款型自验证，编译与实际运行分别记录 |

## 算子约束限制

仅支持任务规定的列主序带状布局及 lda/incx/incy 语义。alpha/beta 为 Host 指针，A/x/y 为 Device 指针。非法参数在 Host 返回错误。无额外确定性要求，按正式混合容差评价。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 方法 | 来源 |
| --- | --- | --- |
| 精度 | 实/虚部分别 atol=rtol=2^-13，有限元素匹配率≥0.99；最大绝对误差≤max(0.01,32×ULP)，ULP 锚定最大误差点 golden | 正式任务书 §3.2 |
| 非有限值 | 同为 NaN、同号 Inf、NaN/Inf 交叉按任务等效；有限/非有限和异号 Inf 不匹配，单独统计 | 正式任务书 §3.2 |
| 性能 | 每例 Avg_us≤1000×gpu_ms/0.8（等价于 Avg_ms≤gpu_ms/0.8）；A100-SXM4-80GB / CUDA 12.8 / cuBLAS 12.8.4 基线 | §3.3 及正式 gpu_baseline.csv |
| 内存 | 无独立性能门槛，检查边界、输入不变和步长空隙 | §3.4 |

正式包存在计时冲突：任务书 §3.3/§7.4 规定 msprof op TaskDuration，README 与 verify_performance.py 按 GTest ACL event avg_us 判定。两种结果分别记录，预置 Device 缓冲调用算子，先 warmup 5 次；event 按连续 10 次下发取均值，TaskDuration 保留至少 10 个有效样本。H2D/D2H、输入生成和 golden 不进入计时。采集前明确应用与 profiler warmup 的逐次映射，只把返回 SUCCESS 的调用计入有效样本，异常样本单独记录。两种口径不混用，最终验收口径以官方澄清为准。

使用仓库 CSV/GTest 框架，执行正式 1000 条精度和 200 条性能参数。补充覆盖 handle 优先级、INT32_MIN/跨度溢出、非法 uplo、空指针、零维、k=0/满带/k>n−1、负步长、实对角、beta=0 不读旧 y 和污染填充。负向用例采用最小 dummy 缓冲，避免测试在接口校验前分配溢出尺寸。含 Inf/NaN 的 x 不与 alpha=0 组合；alpha=0 不保证跳过 A/x 读取。

均匀/正态按任务要求各补齐 50%，实/虚独立生成，正态 μ∈[-5,5]、σ∈[0.1,2]，保存种子和参数。正式原始 CSV 与补充套件分别报告。检查 A/x 不变、y 空隙与前后 guard。构建、逐例结果、原始日志和报告绑定同一源码提交。

## 兼容性分析

新增公共 aclblasChbmv API，不改动其他同族算子或其他产品线实现。非法 uplo 返回任务要求的 INVALID_VALUE，与其他接口的 INVALID_ENUM 分开。代码及测试按 ops-blas 原生 CMake 和算子登记方式集成。

## 交付安排

设计评审合入后完成测试 README、全量精度/性能报告、日志、截图和个人仓固定代码链接。七个模板 task_submission 文件置于交付分支根目录，确认 Ascend-CANN 审核访问。develop 形成固定生产候选后建立 submission，由同一生产提交派生 review 并加入复现和交付材料；官方测试审核通过后再发起代码 PR。
