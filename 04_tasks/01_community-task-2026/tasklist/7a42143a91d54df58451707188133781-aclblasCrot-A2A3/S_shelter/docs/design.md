# aclblasCrot（Atlas A2/A3）算子设计

## 需求背景（required）

### 需求来源

[11 月社区任务 aclblasCrot 算子开发（A2A3）](https://www.hiascend.com/activities/task-center/details/7a42143a91d54df58451707188133781?menu=tasks)。贡献者：S_shelter。代码目标仓库为 `cann/ops-blas`，实现目录 `blas/rot/arch22/`，测试目录 `test/rot/crot/arch22/`。

### 背景介绍

为 Atlas A2/A3 补齐单精度复数向量 Givens 旋转能力，采用 Ascend C kernel 直调、句柄绑定 stream 的工程方式。输入 x、y 以两个相邻 float 表示一个 complex64，均原地更新。c 是实数，s 是复数；标量指针支持 Host、Device 及混合位置。

## 需求分析（required）

公开接口复用 `include/cann_ops_blas.h` 中的 `aclblasCrot`，不新增平行接口：

```cpp
aclblasStatus_t aclblasCrot(aclblasHandle_t handle, int n,
    aclblasComplex* x, int incx, aclblasComplex* y, int incy,
    const float* c, const aclblasComplex* s);
```

校验顺序为 handle 判空、n<0 拒绝、n==0 快速返回，再检查步长、数据及标量指针。零步长、INT32_MIN 步长、以 int64 计算的 `(n-1)*abs(inc)` 超过 INT32_MAX 均返回 INVALID_VALUE。负步长从物理尾端开始遍历。quick return 不读取数据指针。

x、y 的逻辑形状为 [n]，物理长度为 `1+(n-1)*abs(inc)`；n==0 时不要求缓冲区。c 为 FLOAT32，s、x、y 为 COMPLEX64。不涉及广播、转置、leading dimension 或批维度。支持值域包含 Inf/NaN；特殊值传播必须符合任务书精度规则，不得仅根据整块同值填充简化语义。

## 详细设计（required）

### 数学公式

使用旋转前的 x、y：

```text
x' = c*x + s*y
y' = c*y - conj(s)*x
x'r = c*xr + (sr*yr - si*yi)
x'i = c*xi + (sr*yi + si*yr)
y'r = c*yr - (sr*xr + si*xi)
y'i = c*yi - (sr*xi - si*xr)
```

Golden 根据 Netlib/LAPACK crot 语义实现。乘法与加减分开舍入，避免大系数及抵消场景下融合运算产生精度差异。

### Host 侧

通过 `aclrtPointerGetAttributes` 判断 c、s 位置。Host 标量写入按值传递的 tiling 参数；Device 标量由 kernel 读取。tiling 包含 n、步长、标量值与位置标志，避免逐次上传 tiling GM 缓冲区。

按设备 AIV 核数划分任务，核数在 handle 中缓存；小 n 限制活动核数。交换索引及系数符号常量表在 handle 内一次创建，后续复用，在 handle 销毁时释放。临时资源初始化及 kernel 启动错误向调用方传递。使用 handle 绑定的 stream 异步启动；调用方读回结果前同步。

### 连续路径：incx=incy=1

以 32 字节 DMA 块（4 个复数）分核；仅末核允许不完整 DMA 块，防止核间边界共享写块。每核按 2400 个复数分块，尾块使用实际元素数搬运。该参数由 910B3 实测选择。

数据保持 AoS 交织布局。用两次 neighbour gather 分别交换 y、x 的实虚部，配合交替正负系数重建复乘。先保存涉及原始 x 的乘积及交换结果，再覆盖 x，保证 y 使用原始 x。每个 tile 仅设置一次 64-lane FP32 mask，后续向量指令复用，减少重复设置开销。

si==0 时使用 6080 元素的大块单缓冲路径（四个数组，共 194560 字节），以减少检查次数。先在 UB 内计算 0*(x+y)，以 vcadd 归约并读取累加寄存器判定 tile 是否全为有限值。检测为零才使用省略 gather 的快速路径；非有限值或有限加法溢出均保守回退完整交叉公式。该检测避免将另一复数分量的 Inf/NaN 传播错误地当作当前分量的传播，混合分量场景有独立回归用例。

### 流水及内存

两个 UB 工作组各持有 x、y；swap、part、tmp 三个临时数组由顺序执行的向量流水共享。下一块输入在当前块计算前入队；每槽 MTE3→MTE2 事件保证输出读取完成后才能重用。MTE2→V 保护输入，V→MTE3 保护输出；必要的 V barrier 保留中间缓冲区重用依赖。

首块输入提前入队，辅助表采用对齐 DMA；相邻两张系数表用一次双 repeat Muls 生成。UB 布局总计 154880 字节（151.25 KiB），编译期断言不超过 192 KiB。尾部向量 repeat 只处理 UB 内的补齐区域，GM 搬出严格使用有效长度。

### 非连续路径

按逻辑元素分核，正步长索引为 `i*inc`，负步长索引为 `(n-1-i)*(-inc)`。按批读取到 UB，保存原始 x/y 后采用分离的标量乘加；每元素使用 32 字节对齐暂存槽，以 8 字节 DataCopyPad 写回。步长间隙不修改。

### 支持硬件与兼容性

| 项目 | 说明 |
|---|---|
| Atlas A2 / A3 | arch22 实现目标 |
| 已自测设备 | Atlas 910B3，CANN 9.1.0 |
| 其他架构 | 不修改 arch35 算子实现或契约 |
| 接口 | 与既有公开声明一致 |

## 可维可测分析

| 标准 | 验收口径 | 来源 |
|---|---|---|
| 精度 | 实虚部分别 atol=rtol=2^-13，matched ratio≥0.99，最大绝对误差满足 1e-2 或逐元素 32 ULP；非有限值单独按任务书判定 | 任务书 §3.2 |
| 性能 | 910B3，msprof op warmup 5 次后有效采样≥10，Task Duration 算术均值≤对应 GPU 基线/0.8 | 任务书 §3.3 |
| 内存专项 | 不涉及；报告说明 UB、辅助表及资源释放 | 任务书 §3.4 |

测试保留任务配套 CSV 全部用例；以任务书优先修正随附旧脚本中的 n<0、零步长状态码。补充正态分布、Host/Device/mixed 标量、负步长、空指针、跨度溢出、步长间隙和流水边界。性能用例单独执行；同 shape 的 GPU 记录按 case ID 对齐，禁止覆盖成一个 shape 的单一基线。

验收前以全新构建复测，检查非有限值混合分量、实际分块及 DMA 边界，留存原始日志、逐 case 参数、profiling 明细及对应代码 Commit。ACL Event 数据仅为辅助；`msprof op` 采集失败或样本不足不得判定正式性能通过。

当前已有实现及辅助 Event 测试数据；正式 msprof op 全量采集与提交前审查进行中。已定位当前 CANN profiler 在容器内将逻辑号再次转换为物理号的问题；本机采集使用与上游逻辑转换一致的兼容适配，仅更正设备号，不修改计时或样本。此设计 PR 不代表验收已完成，最终性能和精度以对应最终 Commit 的交付报告为准。
