# 1 需求背景（required）

## 1.1 需求来源

依据 10 月社区任务《aclblasChpr2 算子开发任务书》，在 Ascend 950PR、CANN 9.1.0 环境中实现复数 Hermitian 压缩存储秩 2 更新。

## 1.2 背景介绍

### 1.2.1 实现方案

基于 ops-blas，使用 Ascend C kernel 直调，通过 handle 绑定的 stream 异步更新 packed 向量 ap。

### 1.2.2 实现现状

参考基线尚无 `aclblasChpr2` 接口；同族 `blas/spr2/arch35/` 已有实数 Sspr2，可借鉴其工程、索引与 SIMT 启动方式。

### 1.2.3 算子功能

计算 `A = alpha*x*y^H + conj(alpha)*y*x^H + A`，支持上下三角和独立正负步长，结果原地写回 ap。

# 2 需求分析（required）

## 2.1 需求描述

在 `include/cann_ops_blas.h` 中新增：

```cpp
aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha,
    const aclblasComplex* x, const int incx,
    const aclblasComplex* y, const int incy,
    aclblasComplex* ap);
```

handle 提供 stream，uplo 选择上/下三角，n≥0。alpha 为 Host 复数标量；x/y 为 Device 只读向量；incx/incy 以复数元素为单位，支持正负值，排除 0 和 INT_MIN。ap 为 Device 输入输出缓冲。

## 2.2 需求拆解

1. 完成 COMPLEX64 秩 2 更新及 packed、正负步长寻址。
2. 正常更新忽略输入对角虚部并将其置零；合法快速返回保持 ap 不变。
3. 满足参数顺序、特殊值、精度和逐例性能要求，提供可复现测试。

# 3 详细设计（required）

## 3.1 算子分析

### 3.1.1 数学公式

沿用 Netlib CHPR2 运算顺序，避免浮点重结合改变 Inf/NaN 结果：

```text
t1 = alpha * conj(y[j])
t2 = conj(alpha * x[j])
u = x[i] * t1; v = y[i] * t2
非对角：ap[p] = (old_ap[p] + u) + v
对角：  ap[p] = (old_ap[p].real + (u.real + v.real), +0)
```

### 3.1.2 支持数据类型

alpha、x、y、ap 使用已有 `aclblasComplex`，每个 COMPLEX64 元素为两个 FLOAT32 分量，共 8 字节。核内按 FLOAT32 分量计算和访存。

### 3.1.3 支持形状

x/y 逻辑长度为 n；n>0 时物理长度至少 `1+(n-1)*abs(inc)`。ap 长度为 `n(n+1)/2`，无 lda。n 为运行时参数，测试最大尺寸不构成接口上限。

所有索引乘法先提升至 64 位；输入向量指针指向物理缓冲起始地址：

```text
UPPER：p = int64(j)*(j+1)/2 + i                  # 0<=i<=j<n
LOWER：p = int64(j)*(2*int64(n)-j+1)/2 + (i-j)  # 0<=j<=i<n
start = inc>0 ? 0 : (1-int64(n))*int64(inc)
vectorIndex(i) = start + int64(i)*int64(inc)
```

## 3.2 算子实现

### 3.2.1 Host 侧设计

**参数处理顺序**如下，状态码省略 `ACLBLAS_STATUS_` 前缀：

| 顺序 | 条件 | 处理 |
| --- | --- | --- |
| 1 | handle 为空 | HANDLE_IS_NULLPTR |
| 2 | n<0 | INVALID_VALUE |
| 3 | n=0 | SUCCESS，不访问其余参数或 Device 数据 |
| 4 | alpha 为空 | INVALID_VALUE |
| 5 | uplo 非法 | INVALID_VALUE |
| 6 | 任一步长为 0 或 INT_MIN | INVALID_VALUE |
| 7 | x、y、ap 任一为空 | INVALID_VALUE |
| 8 | alpha=(0,0) | SUCCESS，ap 逐字节不变，不下发 Kernel |
| 9 | 有效更新 | 计算 Tiling，通过 handle stream 下发 |

**分核与 Tiling：** 读取实际 AIV 核数 C，C=0 返回 EXECUTION_FAILED。首版 n≤32 时 B=1，否则 B=min(n,C)，每块 128 个 SIMT 线程；阈值和线程数按实测调整。

Host/Device 共用 Tiling：`n/uplo/numThreads` 为 uint32_t，`alphaReal/alphaImag` 为 float，`incx/incy/xStart/yStart` 为 int64_t。块数经启动参数传入，uplo 用模板或分支区分，不设置额外 tilingKey。Host 参数校验与下发分为两个函数，不主动同步 stream。

### 3.2.2 Kernel 侧设计

1. 块 b 遍历列 `j=b,b+B,…`；UPPER 的行区间为 `[0,j+1)`，LOWER 为 `[j,n)`。
2. 若 x[j]、y[j] 均为复数零，先由线程 0 清零对角虚部并保留实部，随后跳过该列计算。±0 视为零，NaN 不视为零。
3. 其他列按 3.1.1 计算 t1/t2；线程 t 遍历行 `i=begin+t,begin+t+numThreads,…`，检查边界后更新。对角分支不读取旧虚部。

各列 packed 区间互不重叠，每个元素仅由一个线程写入，无需原子操作、跨核同步或额外 GM 工作区。首版采用 GM 分量访存；仅在 profiling 显示瓶颈时评估列分段、SIMD 或 UB 缓存，优化后回归精度与索引覆盖。

### 3.2.3 工程组织

| 目录（相对 ops-blas） | 内容 |
| --- | --- |
| `blas/spr2/arch35/` | `chpr2_host.cpp`、`chpr2_kernel.h`、`chpr2_kernel.cpp`、`chpr2_tiling_data.h` |
| `test/spr2/chpr2/` | 参数、golden、独立精度判定、CMake、验证脚本与 README |
| `test/spr2/chpr2/arch35/` | NPU wrapper、GTest 与官方 CSV |
| `blas/spr2/README.md` | API、约束、示例与平台支持 |

## 3.3 支持硬件

目标平台为 Ascend 950PR（arch35），软件环境为 CANN 9.1.0。

## 3.4 算子约束限制

仅支持 COMPLEX64，不涉及广播、batch 或超出步长语义的一般非连续视图。调用方保证缓冲长度、地址有效且 x/y 不与可写 ap 重叠；读取结果前须同步 handle stream 并检查异步错误。

# 4 可维可测分析

## 4.1 精度标准/性能标准

### 4.1.1 精度验证

golden 使用 Netlib BLAS 3.10.0 `cblas_chpr2`，记录实际链接库；ap 的实部、虚部全量按[固定版本混合容差规范](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md)分别判定：

| 检查项 | 通过条件 |
| --- | --- |
| 有限值 | `abs(actual-golden) ≤ 2^-13 + 2^-13*abs(golden)` 的比例≥0.99，最大误差≤`max(1e-2,32*ULP(g_low))`；g_low 为最大误差位置的 golden FP32 值 |
| 非有限值 | NaN/NaN、同号 Inf/Inf、NaN/任意符号 Inf 等效；异号 Inf 或有限/非有限混合不通过；全部非有限点须等效，无有限点时不计算有限值比例 |
| 边界语义 | 正常更新对角虚部为零；快速返回 ap 逐字节不变；非法参数检查错误码及内存不变，不调用 golden |

覆盖官方 1,000 条精度用例，并补充空 handle、参数组合、对角虚部 NaN、零列特殊值、保护区、输入不变和 stream 测试。每个精度 case 成对执行均匀/正态两轮，保留指定 alpha 与特殊填充；随机 alpha 在 n=1/3/33、上下三角和负步长组合中补测。固定种子，实虚部独立采样，正态参数分别采样 μ、σ；alpha=0 另检查 NaN/非零对角虚部保持不变。

### 4.1.2 性能验证

200 条性能用例按 ID 奇偶各分配 100 条均匀/正态输入。使用 msprof op 的 `Task Duration(us)`，逐例满足 `T_NPU_us ≤ 2.5*T_GPU_ms*1000`。每次计时前恢复 ap，恢复操作不计入 Kernel 耗时。

自测采用 5 次 warmup、60 次有效采样并保留全部样本。官方资料存在 10 次、>10 次、>50 次口径差异，评审时确认。检查接口/同步状态、进程退出码、完整 case 集合、基线和采样数量，缺项即失败；不以 GTest 整体耗时代替算子耗时。

## 4.2 兼容性分析

保持 `aclblasComplex` ABI 与现有 Sspr2 行为不变，复用 CMake/GTest，不修改公共精度判据。实现后回归 Sspr2，按官方模板交付测试代码、报告、复现步骤、版本/commit、日志和截图；内存专项要求为“不涉及”。当前已完成本地索引与代数核对，NPU 结果以实际自测报告为准。
