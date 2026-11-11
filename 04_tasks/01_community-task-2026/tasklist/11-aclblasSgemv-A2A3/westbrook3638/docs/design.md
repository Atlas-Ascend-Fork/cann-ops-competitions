# aclblasSgemv（A2/A3）算子设计文档

| 项目 | 内容 |
| --- | --- |
| 任务 | 11月社区任务-aclblasSgemv算子开发（A2/A3） |
| 参与账号 | westbrook3638 |
| 目标仓库 | `cann/ops-blas`，master，arch22 |
| 实现目录 | `blas/gemv/arch22/` |
| 测试目录 | `test/gemv/sgemv/arch22/` |
| 硬件与版本 | Atlas A2 910B3，CANN 9.1.0 |
| 任务讨论 | https://gitcode.com/cann/ops-blas/discussions/79 |

# 需求背景（required）

## 需求来源

依据《11月社区任务 aclblasSgemv算子开发（A2/A3）任务书》，使用 Ascend C 和 kernel 直调方式补齐 ops-blas 在 Atlas A2/A3 上的单精度 BLAS-2 矩阵向量乘法。公共接口保持 `include/cann_ops_blas.h` 中已有声明，host、tiling、device kernel 和测试均按 arch22 工程组织。

## 背景介绍

Sgemv 计算

$$
y \leftarrow \alpha\,\mathrm{op}(A)x+\beta y.
$$

矩阵 \(A\) 按列主序存储，元素 \(A[i+j\cdot lda]\) 对应第 \(i\) 行、第 \(j\) 列。N 模式下 \(x\) 长度为 \(n\)、\(y\) 长度为 \(m\)；T/C 模式下 \(x\) 长度为 \(m\)、\(y\) 长度为 \(n\)。实数矩阵的 C 与 T 使用同一计算路径。

仓内已有 arch35 SIMT 实现、公共测试框架和 arch22 复数 Cgemv。本任务复用公共 ABI、handle/stream、golden 与 CSV 框架，在 arch22 新增实数 AIV kernel。arch35 的 SIMT 调用方式不用于 arch22。

# 需求分析（required）

## 需求描述

实现 FLOAT32 的 N/T/C 三种模式，支持列主序 `lda` 空洞、正负 `incx/incy`、运行时尺寸、原地输出和异步 stream。参数错误码、快速返回和特殊值行为按任务书执行。

## 需求拆解

1. 新增 arch22 host、tiling、kernel 和 README。
2. 参数校验先于 `m=0` 或 `n=0` 的快速返回；host 使用 64 位数值检查向量跨度。
3. `alpha=1,beta=0,incx=incy=1` 使用 Cube/Float32 Matmul、AIV 列块归约和受控 split-K；其他参数组合使用与任务机 Netlib 舍入顺序一致的参考路径。
4. `beta=0` 不读取旧 \(y\)；`alpha=0` 按 Netlib 快速返回只计算 \(\beta y\)，不读取 A/x。
5. 新建 arch22 GTest 和 wrapper，加载官方 1200 条 CSV，并补充正态分布输入。
6. 在 910B3 上执行 1000 条精度和 200 条性能用例，生成模板要求的报告与日志。

# 详细设计（required）

## 算子分析

### 接口

```cpp
aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);
```

### 参数与支持范围

| 参数 | 类型与位置 | 设计处理 |
| --- | --- | --- |
| handle | Host 句柄，携带 stream | 空指针返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| trans | Host 枚举 | N/T/C；其他值返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| m, n | Host `int` | 非负；参数校验后任一为零则成功返回且不启动 kernel |
| alpha, beta | Host FLOAT32 指针 | 非空；值域含 Inf/NaN，不做数值限制 |
| A | Device FLOAT32 | 列主序，物理范围 \(lda\times n\)；m、n 均大于零时非空 |
| lda | Host `int` | \(lda\geq\max(1,m)\) |
| x, y | Device FLOAT32 | 逻辑长度由 trans 决定；对应长度为非零时非空 |
| incx, incy | Host `int` | 非零且不为 `INT32_MIN`；正负步长均支持 |

### 数学与地址映射

以零基索引表示：

\[
\mathrm{out}[r]=\alpha\sum_{k=0}^{K-1}\mathrm{op}(A)[r,k]\,x_{\mathrm{logical}}[k]+\beta y_{\mathrm{old}}[r].
\]

N 模式 \(r=0,\ldots,m-1\)、\(K=n\)，矩阵地址为 \(r+k\cdot lda\)。T/C 模式 \(r=0,\ldots,n-1\)、\(K=m\)，矩阵地址为 \(k+r\cdot lda\)。

向量逻辑位置 \(i\) 的物理偏移为

\[
\mathrm{phys}(i,L,inc)=
\begin{cases}
i\cdot inc, & inc>0,\\
(L-1-i)\cdot |inc|, & inc<0.
\end{cases}
\]

host 用 `int64_t` 计算 \((L-1)\cdot|\!inc\!|\)，超过 `INT32_MAX` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`。

## 算子实现

### Host 侧设计

参数检查顺序为：

1. 检查 handle。
2. 检查 trans、m、n、lda、incx、incy。
3. 检查 alpha、beta 指针。
4. 计算 \(L_x,L_y\)，检查 A、x、y 指针和向量跨度。
5. 参数合法且 \(m=0\) 或 \(n=0\) 时返回成功。
6. 读取 Host alpha/beta，生成 tiling，向 handle 的 stream 下发 kernel。

host 返回成功表示下发完成，调用方读取 \(y\) 前同步 stream。tiling 直接按值传入 kernel，不额外申请 tiling GM，也不在算子内同步。

### 分核与 tiling

| 路径 | 输出分核 | 单核内切分 | 数据访问 |
| --- | --- | --- | --- |
| N，快速路径，\(m\geq256\) | AIC 输出列块 | Matmul 单行结果，N tile 最多 512 | 把 \(x^T\) 视作 \(1\times n\)，A 的列主序存储视作行主序 \(n\times m\) |
| N，快速路径，\(m\leq16\) | AIV 归约列分段 | 最多 4096 列，连续 A span | 初始化 \(y=0\) 后按输出行原子累加 |
| T/C，快速路径，\(m\leq4224\) | AIV 输出列块 | 每批 8 列、每批最多 2032 行 | 列内连续搬入 A，批量 `Mul + WholeReduceSum` |
| T/C，快速路径，\(m>4224\) | AIC 输出列块 | Matmul 单行结果 | Cube 回退，保持大 K 可用 |
| T/C，\(m\leq32\) | AIV 输出列块 | 最多 64 列 | Gather/Axpy 小形状路径 |
| 其他参数组合 | 单 kernel | 严格输出顺序 | Netlib 舍入顺序的 FLOAT32 FMA 参考路径 |

N 的 AIC 路径按输出块数和 AIC 波次利用率选择 1 至 4 段 split-K。split-K 部分结果写入 handle workspace，再由 AIV kernel 合并到 \(y\)。workspace 上界为 `4 * outDim * sizeof(float)`。小 \(m\) 的 N 路径使用初始化 kernel 加原子加，不申请额外 workspace。

tiling 字段包含 m、n、lda、带符号 incx/incy、alpha/beta、路径、输出块数、split-K 段数和每段 K。所有字段显式初始化，索引与字节偏移使用 64 位中间量。

### Kernel 侧设计

#### N 模式

快速路径把 Sgemv 改写为

$$
y^T = x^T A^T.
$$

列主序 A 的连续物理存储可直接解释为 \(A^T\) 的行主序视图，因此 Cube 只做 \(1\times n\) 与 \(n\times m\) 的 Matmul，不复制 A 或 \(x\)。每个 AIC block 拥有一个连续输出列片；主机根据输出片数和 AIC 波次估算选择 split-K。split-K 段按至少 32 个 K 元素对齐，部分结果先落 workspace，再由 AIV 相加。

当 \(m\leq16\) 时，AIV 按列分段连续搬入 A span，避免每列一次小 DMA；初始化 kernel 清零 \(y\)，计算 kernel 对每个输出行原子累加。该路径覆盖 `16 x 65536` 和 `4 x 1048576`。

其他参数组合走严格参考路径。N 的舍入顺序与任务机 Netlib `sgemv_` 一致：先计算 `round(alpha*x[j])`，再初始化 `round(beta*y[i])`，然后逐列执行 `fma(A[i,j],alpha*x[j],y[i])`。软件 FMA 使用 Binary32 误差变换；乘积溢出再与加数抵消的输入使用整数对齐回退。

#### T/C 模式

快速路径每批处理 8 个输出列，并把 \(m\) 行均分为最多 3 个行批。每批连续搬入 A 列片段，执行一次批量 `Mul` 和 `WholeReduceSum`，块间使用 FLOAT32 Kahan 补偿。该路径减少逐列 DMA 和 Scalar/Vector 同步。

\(m>4224\) 时回落 Cube；\(m\leq32\) 时使用小形状 Gather/Axpy 路径。其他参数组合走严格参考路径：先执行 `fma(A[i,j],x[i],sum)` 点积，再用 `fma(alpha,sum,round(beta*y[j]))` 写回。

#### 步长、尾块与同步

连续数据使用 `DataCopyPad` 或 `gm_to_ub_align` 精确搬入有效长度。跨步向量先按物理步长搬入 UB，再整理为逻辑连续序列；负步长从物理跨度尾端开始。跨步 \(y\) 先构造零初始化暂存区，再把有效逻辑元素写入对应位置；交集写入使用原子加，交集外写入零，保持间隙不变。

MTE2、Vector、Scalar、MTE3 之间的跨流水依赖用显式队列或事件同步。尾块按有效长度搬运，GM 地址范围保持在调用者提供的缓冲区内。kernel 使用 handle stream 异步执行，不在 kernel 内同步或动态分配。

### 工程结构

| 内容 | 路径 |
| --- | --- |
| Host 与校验 | `blas/gemv/arch22/sgemv_host.cpp` |
| Tiling 结构 | `blas/gemv/arch22/sgemv_tiling_data.h` |
| Kernel | `blas/gemv/arch22/sgemv_kernel.cpp` |
| Kernel 声明 | `blas/gemv/arch22/sgemv_kernel.h` |
| GTest、wrapper、CSV | `test/gemv/sgemv/arch22/` |
| GTest 注册 | `test/gemv/sgemv/CMakeLists.txt` |
| 接口说明 | `blas/gemv/README.md` |
| 交付材料 | 个人代码仓根目录 `task_submission/` |

## 支持硬件

| 产品 | 设计支持 | 正式自验证 |
| --- | --- | --- |
| Atlas A2 训练/推理系列（910B3） | arch22，支持 | 精度、性能和构建 |
| Atlas A3 训练/推理系列 | arch22，支持 | 构建适配 |

## 算子约束限制：§2.5 逐项核对

| 约束 | 实现结论 | 验证 |
| --- | --- | --- |
| 参数合法性与校验层级 | 按任务书顺序和返回码，快速返回在参数校验之后 | 参数 GTest 与官方负向 CSV |
| 非连续 Tensor | 支持 \(lda>m\)、正负 incx/incy，拒绝零、`INT32_MIN` 和跨度溢出 | 空洞、正负步长、哨兵 |
| broadcast | 不涉及 | 不生成广播路径 |
| dynamic shape | m/n/lda/inc 为运行时入参，host 当次计算 tiling | 多形状连续调用 |
| 原地与视图 | \(y\) 原地更新；beta=0 不读旧值；A/x 只读 | 特殊初值、只读与间隙检查 |
| 确定性 | 无跨次位级一致性要求；小 \(m\) 的 N 路径原子累加顺序可变化 | 固定输入重复执行并记录误差 |
| 空 Tensor | m=0 或 n=0 合法快速返回，不启动 kernel | 空矩阵与空指针组合 |
| 0 维 Tensor | 不涉及 | alpha/beta 为 Host 标量 |
| 异步执行 | handle stream 异步下发 | 同步后读取结果 |

不新增 A/x/y 互相别名的支持承诺；\(y\) 的输入与输出复用属于本算子接口语义。

# 可维可测分析

## 精度标准/性能标准

| 验收项 | 判定 |
| --- | --- |
| Golden | 随附 `sgemv_golden.h` 的 FLOAT32 `cblas_sgemv` |
| 精度用例 | 官方 1000 条精度用例；补充正态分布输入，保持 1000 条正式规模 |
| 有限值阈值 | \(atol=rtol=2^{-13}\)；matched_ratio ≥ 0.99；逐元素误差不超过 \(\max(10^{-2},32\,\mathrm{ULP})\) |
| 非有限值 | 同为 NaN、同为同号 Inf、一侧 NaN 与另一侧任意 Inf 视为等效；有限与非有限不等效 |
| 性能用例 | 官方 200 条 TC_PF |
| 性能采集 | GTest 在预置 device 缓冲上 warmup 5 次，再连续执行不少于 10 次，输出 `[PERF] avg_us` |
| 性能判据 | 逐 case `NPU_us <= gpu_ms * 1000 / 0.8` |
| 内存 | 任务书 §3.4 为不涉及，交付模板填写“不涉及” |

## 测试设计

arch22 GTest 复制官方 `sgemv_test.csv`，复用 `sgemv_param.h` 和 `sgemv_golden.h`。wrapper 在 `INT32_MIN`、非法尺寸和空指针用例上直接透传 dummy 指针，避免测试工具先计算超大跨度。合法负步长和非单位步长按完整物理跨度分配缓冲区。

精度测试流程：

1. 根据 CSV 生成 A、x、y。
2. 将数据复制到 device，调用 `aclblasSgemv`。
3. 同步 stream，复制 \(y\) 回 host。
4. 用相同输入调用 CPU golden。
5. 按物理到逻辑映射提取输出。
6. 分别统计有限元素和特殊值，应用任务书精度规则。

性能测试复用同一 wrapper 和 CSV。每个 TC_PF case 恢复 \(y\) 初值，执行 5 次 warmup，再用两个 `aclrtEvent` 夹住连续 20 或 50 次下发，计算平均 device 时间并输出 `[PERF]`。H2D/D2H、host 造数和 golden 不计时。正式判定只使用该 `[PERF] avg_us`；`msprof` 不作为门槛。

## 用例覆盖

官方 CSV 覆盖 N/T/C、方阵、非方阵、极端矩形、m/n 为 0、lda 空洞、正负步长、alpha/beta 全值、Inf/NaN 和负向参数。实现保留原 case 名称和参数，不按历史通过结果筛选。

正态分布补充方式为在相同 case 参数下增加 `GAUSS_NORM_5_2` 输入，固定随机种子和分布参数，使随机数值输入覆盖均匀与正态两类。边界和负向用例继续保留独立值。

## 实现状态与自验证

实现代码位于个人仓库分支：

https://gitcode.com/westbrook3638/ops-blas/commit/e54e91b3cf053104c0aad1cdbd8e90feaa67f067

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 环境 | Atlas 800I A2，910B3；CANN 9.1.0 | `npu-smi`、构建日志 |
| 官方精度 | PASS=1000，FAIL=0 | `task_submission/2.2 精度自验证日志.log` |
| 正态混合分布 | PASS=1000，FAIL=0 | 同一精度日志的混合分布附录 |
| 官方性能 | PASS=200，FAIL=0，NO_REF=0 | `task_submission/3.2 性能自验证日志.log` |
| 分类回归 | 152/152 PASS | `categories_latest.log` |
| 额外非对齐回归 | 2/2 PASS | `regression.log` |
| 内存 | 任务书 §3.4 不涉及 | `4.1`、`4.2` 模板填写“不涉及” |

性能正式证据使用 GTest `[PERF] avg_us`。复测时设备 0/1 被其他任务占用，未将资源争用下的结果覆盖为正式性能结论；正式全通过结果保存在 `results_npu_sgemv_20261111_045457.csv`。

任务书口径澄清已在 discussions/79 提问，评论 ID 为 `dd139a14f16745adbbdb9b23ea97df3d`，覆盖 FLOAT32 精度真值、GTest/msprof 主判据和 `alpha=0` 语义。收到维护者明确答复前不新增其他验收门槛。

## 兼容性分析

不修改公共接口签名、参数顺序或 arch35 实现。arch22 源文件由 `SOC_ARCH_DIRS` 选择，Sgemv GTest 在 arch22 显式注册。README 更新 A2/A3 支持状态和约束。
