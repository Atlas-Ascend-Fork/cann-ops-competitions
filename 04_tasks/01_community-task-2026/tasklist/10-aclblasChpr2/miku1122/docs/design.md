# 需求背景（required）

## 需求来源

2026 年 10 月社区任务 aclblasChpr2（950），任务编号
8e6e450ec9a44869a4956675200cc05b。本文按 cann-competitions 官方设计模板组织。

## 背景介绍

ops-blas 的 spr2 家族已有 arch35 实数 packed 秩 2 更新，本次补充 complex64
Hermitian packed 秩 2 更新公共接口。Host 参考 Sspr2 的校验与 stream，
Kernel 参考 Csyr2 的 SIMT 并行，复数运算语义以 Netlib CHPR2 为准。

# 需求分析（required）

## 需求描述

实现 `A := alpha*x*y^H + conj(alpha)*y*x^H + A`，alpha 位于 Host，x/y/ap
位于 Device。上下三角 packed 原地更新，负步长、非有限数和 quick return
按任务书处理。公共 API 放入 include/cann_ops_blas.h。

## 需求拆解

1. 完成 Host 校验、Tiling、异步 kernel launch 与公共 API。
2. 保持 packed 列索引、共轭、对角虚部和 Netlib 零列分支的语义。
3. 原样接入 1200 条官方测试，固定 Netlib 3.10.0 golden，补充接口和随机分布覆盖。
4. 在 950PR / CANN 9.1.0 完成完整精度与 200 条 msprof 性能自验。

# 详细设计（required）

## 算子分析

### 数学公式

每列 j 计算 `t1=alpha*conj(y[j])`、`t2=conj(alpha*x[j])`。
非对角更新采用 `(ap+x[i]*t1)+y[i]*t2`，对角采用
`real(ap)+real(x[j]*t1+y[j]*t2)`，输出虚部为 +0。
x[j]/y[j] 同时为复数零时，保留整列非对角，只清零对角虚部；
这样可避免其他行含 Inf/NaN 时引入参考实现未执行的乘法。

### 支持数据类型

complex64，实部和虚部各为 FP32。标量与向量均不进行降精度。
普通复数乘法显式拆为四次乘法、一次减法和一次加法，禁用 FP contraction。

### 支持形状

n 为非负 int，无 lda；ap 元素数为 n(n+1)/2，x/y 的物理元素数为
1+(n-1)*abs(inc)。支持任意非零 int 步长，INT_MIN 非法，4096 只是官方性能用例
最大尺寸，不限制公共 API。

## 算子实现

### 实现方案

#### 3.2.1 host侧设计

依次检查 handle、负 n、零 n、alpha、uplo、步长、数据指针，然后判断零 alpha。
零 n 在访问其余参数前返回，零 alpha 在数据指针判空后返回，均不修改 ap。
通过平台信息动态取得 AIV 核数，使用 `min(n, aivCoreNum)` 个核。
线程数按 n 向 SIMT 最小线程粒度取整并截断到 SIMT 最大线程数。
Tiling 仅携带 n、uplo、线程数、alpha 的实虚部及两个 int64 步长。
只进行一次 launch，复用 handle->stream，不申请 workspace，不同步设备。

##### 分核策略

核 b 处理 j=b, b+coreNum, ... 的列。列间独立，每个存储元素只有一个写入者，
无需核间通信或原子操作。轮转列分配减轻三角列长递增/递减导致的不均衡。

##### 数据分块和内存策略

首版直接访问 GM，列内 SIMT 线程访问连续 packed 行，线程以 blockDim 为步幅
循环处理该列。无 UB tile 或 double buffer，避免在没有性能证据时增加缓存分支。
除线程寄存器和设备编译器栈外无额外工作区，x/y 只读、ap 原地读写。

##### tilingkey规划

按 uplo 分为两个模板实例，步长符号在通用向量索引函数中处理；不引入额外
shape 特化。合法大 n、非连续与负步长均走同一通用实现。

#### 3.2.2 kernel侧设计

UPPER 列起点为 j(j+1)/2，行范围 [0,j]；LOWER 列起点为
j(2n-j+1)/2，行范围 [j,n)。元素索引为列起点加列内行偏移，最后乘 2
转换为 float 分量索引。所有乘法在提升为 64 位后进行。
负步长逻辑 i 映射到物理 `(n-1-i)*(-inc)`，避免 int abs(INT_MIN) 或窄整数溢出。
先检查零列，再构造 t1/t2；对角只读取旧实部，避免脏虚部影响结果。

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR / arch35 / CANN 9.1.0 | √ |

## 算子约束限制

alpha 为 Host 标量指针，不扩展 Device pointer mode。用户应提供满足物理长度的
设备内存；API 不包含长度参数，不验证实际分配大小。x/y 与正在更新的 ap 不重叠。
不增加广播、矩阵 leading dimension 或额外确定性接口。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度 | 完整 packed 实部/虚部分别检查，atol=rtol=2^-13，匹配率≥0.99；绝对误差≤max(1e-2,32ULP)，非有限数单独按任务规则等效判断 | 官方任务书 §3.2 |
| 性能 | 200 条每条 GPU/NPU kernel 耗时≥0.4，msprof task duration，5 次预热后 60 次有效采样，并保留前 10 次均值 | 官方任务书 §3.3、§7及测试 README |

测试入口与实际环境配置见 test/spr2/chpr2/README.md，逐 case 原始结果和性能
证据随自测报告提供。参数错误和 quick return 用例单独验证状态及 ap 原样保留。

## 兼容性分析

新增共用 aclblasChpr2 声明，其他接口签名不变。源码与测试通过现有按名称发现
机制注册，无需修改顶层 CMake 或其他架构实现。未新增 CI 或自动化框架。
