# aclblasCher 算子设计（Ascend 950PR）

# 需求背景（required）

## 需求来源

[10月社区任务-aclblasCher算子开发（950）](https://www.hiascend.com/activities/task-center/details/56b693d9d8ef417da7715f76a3f50cf8)
及其配套任务书、自测用例。实现目标仓为 [ops-blas](https://gitcode.com/cann/ops-blas)。

## 背景介绍

Cher 是 BLAS Level 2 的单精度复数 Hermitian 秩-1 更新操作。它以实数标量和复数向量
更新矩阵的指定三角区域，适用于协方差矩阵构造和 Hermitian 矩阵更新。
本设计采用 ops-blas 句柄接口与 Ascend C kernel 直调方式，实现位于 `blas/syr/arch35/`，
测试位于 `test/syr/cher/arch35/`。

# 需求分析（required）

## 需求描述

接口及参数序列与 cuBLAS `cublasCher` 对齐：

```cpp
aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const float* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

声明加入 `include/cann_ops_blas.h`。复数类型复用 `cann_ops_blas_common.h` 中的
`aclblasComplex`，实部和虚部各为一个 FLOAT32。

| 参数 | 类型与位置 | 约束 |
| --- | --- | --- |
| handle | Host，aclblasHandle_t | 有效句柄，携带 stream |
| uplo | Host，aclblasFillMode_t | ACLBLAS_UPPER 或 ACLBLAS_LOWER |
| n | Host，int | n ≥ 0 |
| alpha | Host，const float* | FLOAT32 实数标量，非空 |
| x | Device，const aclblasComplex* | COMPLEX64，物理长度至少 1+(n−1)×abs(incx) |
| incx | Host，int | 任意非零步长，支持反向遍历 |
| A | Device，aclblasComplex* | COMPLEX64，列主序，分配长度至少 lda×n |
| lda | Host，int | lda ≥ max(1,n) |

## 需求拆解

1. 计算 `A = alpha*x*x^H + A`，A 原地更新，x 只读。
2. 只引用和更新 uplo 指定三角，保留另一三角及 lda padding。
3. 对角旧虚部按零处理，非零更新后虚部置零。
4. 支持正、负及非单位步长；n=0、alpha=0 按规定顺序快返。
5. 通过 handle 绑定的 stream 异步执行，读回由调用方同步。
6. 在 Ascend 950PR、CANN 9.1.0、Netlib BLAS 3.10.0 环境完成全量自测。
7. 精度按 FLOAT32 分量判定，性能逐例达到 GPU 标杆的 0.4 倍以上。

# 详细设计（required）

## 算子分析

### 数学公式

设 `x(i)=xr(i)+i*xi(i)`，只对选定三角中的 `(i,j)` 更新：

```text
cr = alpha * xr(j)
ci = -alpha * xi(j)
deltaReal = xr(i)*cr - xi(i)*ci
deltaImag = xr(i)*ci + xi(i)*cr
A(i,j).real += deltaReal
A(i,j).imag += deltaImag
```

对角实部按 Hermitian 秩-1 更新计算，虚部写零。复数乘积先合并，再加旧矩阵值，
避免抵消时先加旧值丢失有效分量。乘积使用独立 FP32 舍入，数值分支与 `cblas_cher` 对照。
当 x(j) 的实、虚部均为零时，沿用 Netlib 的零列分支，保留该列非对角元素，
仅清除对角虚部；相应特殊值组合纳入精度回归。

### 支持数据类型

| alpha | x | A | 计算类型 |
| --- | --- | --- | --- |
| FLOAT32 实数 | COMPLEX64 | COMPLEX64 | FLOAT32 实、虚分量 |

### 支持形状

A 的逻辑形状为 `[n,n]`，存储形状为 `[lda,n]`；x 的逻辑形状为 `[n]`。
地址以复数元素为单位：`A(i,j)=A[j*lda+i]`。
正步长时 `x(i)=x[i*incx]`；负步长时 `x(i)=x[(n-1-i)*abs(incx)]`，
x 指向分配区域起点。偏移先提升为 64 位，再计算乘积和绝对值。
块大小、缓存上限均为内部调优参数，不改变接口的合法尺寸范围。

## 算子实现

### Host 侧设计

参数校验和快返按下列顺序组织：

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | handle 为空 | HANDLE_IS_NULLPTR |
| 2 | n < 0 | INVALID_VALUE |
| 3 | n = 0 | SUCCESS，不访问后续参数和数据 |
| 4 | uplo 非法 | INVALID_ENUM |
| 5 | lda < max(1,n)、incx=0 或 alpha 为空 | INVALID_VALUE |
| 6 | alpha=0.0 | SUCCESS，不更新 A，允许 x/A 为空 |
| 7 | x 或 A 为空 | INVALID_VALUE |
| 8 | 参数有效且需更新 | 提交 kernel |

通过仓库公共平台接口查询 AIV 核数，block 数取 `min(n,AIV核数)`。
初始调度按列循环分配，各 block 处理 `blockId, blockId+blockNum, ...` 列；
每列按三角范围切成行 tile，每个输出元素由一个 block 写入。

Host 将 n、lda、uplo、block 数、64 位 incx 和 alpha 写入共享 tiling。
以静态断言固定结构尺寸、对齐与关键字段偏移；标量在调用时读取，数据指针传给 kernel。
调用内不申请库工作区，不等待 stream。

### Kernel 侧设计

kernel 使用 AIV 向量路径，由 Init 和 Process 两阶段组成。

1. Init 绑定 x/A 的 GM 地址，划分 UB 工作平面。
2. UPPER 列 j 处理 `[0,j]` 行，LOWER 列 j 处理 `[j,n)` 行。
3. 读取并预乘本列标量，在列内复用。
4. 搬入行 tile 的 x 与 A，解交织复数实、虚分量。
5. 用 RegBase/MicroAPI 完成 FP32 乘法、加减及复数交织。
6. 对角虚部置零，按 tile 有效字节数写回。

初始行 tile 为 1024 个复数元素。GM 搬运采用带有效长度的 DataCopyPad，
只访问选定三角；向量尾块采用 mask，UB 预留完整向量访问需要的尾部空间。
非单位步长用 Compact 搬运有效复数；负步长先读对应物理地址，再在 UB 内反排。
LOWER 的非对齐行起点通过 Gather 搬入对齐工作平面。

### 数据复用与性能优化

选定三角元素数为 `P=n*(n+1)/2`，主要计算量约为 `8P+2n` 次 FP32 运算，
A 的最低读写量为 `16P` 字节。优化重点是连续搬运、输入复用和流水重叠。

- 当 `n <= 4096` 且 `n >= 2*blockNum` 时，各 block 可预加载 x 并缓存解交织结果。
  行向量读取量由 P 个复数降为 `n*blockNum` 个复数，缓存条件结合搬运量和 UB 容量选择。
- 缓存完成后建立向量写到标量读的事件依赖，列标量也从缓存获取。
- UB 规划为 `(9*1088+2*4160)*4 = 72448` 字节，包含输入、矩阵、实虚工作面、
  反排索引与 x 缓存，无需全矩阵中间副本。
- 对正式性能形状，比较循环列分配与长短列配对的最大单核工作量。
- 在确认跨流水依赖后，评估矩阵双缓冲与按流水事件同步，重叠搬入、计算和写回。
  优化候选同时复测精度、非更新区保护和全部性能用例。

缓存、调度及流水参数由正式用例的设备测量确定；成本模型用于筛选方案，
性能结论以 CANN 9.1.0 下任务规定的计时结果为准。

## 支持硬件

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR | 支持 |

## 算子约束限制

数据为列主序，步长由 incx、lda 表达，不涉及额外广播或视图语义。
alpha 位于 Host，x/A 位于 Device；调用方保证缓冲区容量及生命周期。
A 原地更新，仅指定三角参与计算；快返时不更新 A。
实现复用 ops-blas 公共接口和 Ascend C API，保持已有算子的行为与接口。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 标准 | 来源 |
| --- | --- | --- |
| Golden | Netlib BLAS 3.10.0 的 cblas_cher，CblasColMajor | 任务书 §3.1、§3.2 |
| 有限值精度 | 实、虚部分别满足 atol=rtol=2^-13，matched_ratio≥0.99；绝对误差限按 1e-2 或 32 ULP | 任务书 §3.2 |
| 非有限值 | 同为 NaN、同号 Inf、NaN 与 Inf 配对等效；有限与非有限、异号 Inf 不匹配，单独计数 | 任务书 §3.2 |
| 对角与保护区 | 对角虚部为零；另一三角及 padding 哨兵不变，x 不变 | 任务书 §2.1、§3.2 |
| 性能 | 每条 NPU 平均 kernel 耗时≤gpu_ms×1000/0.4，单位 μs | 任务书 §3.3、gpu_baseline.csv |
| 内存 | 无独立内存验收门槛 | 任务书 §3.4 |

精度将有限值和非有限值分开，分别统计实、虚部匹配率和最大误差。
绝对误差硬上限为 `max(1e-2,32*ULP(g_low))`；g_low 为该分量最大绝对误差点的
FLOAT32 golden，按任务书引用的生态标准 §2.1.2 取锚点。
保留 CSV 配置的仓库 MERE/MARE 检查，与任务书混合容差分别报告。

复用任务包全部 1200 条用例：1000 条精度、200 条性能。覆盖基础组合、尺寸扫描、
alpha 特殊值、正负步长、padding、上下三角、空指针、参数校验及 Inf/NaN。
对随机精度参数分别执行均匀 `[-5,5]` 和正态分布两批，覆盖比例各为 50%。
正态分布先采样 mu∈[-5,5]、sigma∈[0.1,2]，alpha 与 x/A 的实、虚分量独立生成。
保留原始 CSV 参数及边界用例，另补充随机 alpha 的组合回归；记录生成器、种子和输入来源。
测试采用仓库 CSV/GTest 入口和 CBLAS golden。

仓库根目录的构建及测试入口为：

```bash
bash build.sh --ops=cher --soc=ascend950 --run
```

性能全量逐例关联 GPU 基线，四条典型用例上限如下：

| n | uplo | alpha | incx | lda | NPU 上限（μs） |
| --- | --- | --- | --- | --- | --- |
| 512 | UPPER | 1.0 | 1 | 512 | 13.935 |
| 1024 | LOWER | 1.0 | 1 | 1024 | 14.8425 |
| 2048 | UPPER | 1.0 | 1 | 2048 | 27.7825 |
| 4096 | LOWER | 1.0 | 1 | 4096 | 135.125 |

正式性能使用 msprof op 的 task duration，排除预热，逐例保存样本及算术平均值。
任务书 §3.3 写 5 次预热、10 个有效样本；§7 写有效采样大于 10 次，
配套测试指导写大于 50 次。自测保留 10 样本与大于 50 样本的分组统计，
分别标注口径，最终验收口径以组织方澄清为准。

任务书规定任意非零 incx；配套 TC_ED_150 将 INT_MIN 列为 INVALID_VALUE 负例。
接口设计保留 64 位偏移处理，其用例预期与非零步长范围需在验收前统一。

## 兼容性分析

新增 aclblasCher 公共声明和 arch35 实现，不改变既有 Ssyr、Csyr 等接口。
目录、构建和测试沿用 syr 家族的仓库约定；README 产品表标注 Ascend 950PR 支持。
生产代码合入 ops-blas，设计文档按模板提交至社区任务设计仓对应目录。
