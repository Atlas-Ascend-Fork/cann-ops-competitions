# 需求背景

## 需求来源

11月社区任务《aclblasChemv 算子开发任务书（A2/A3）》及随附测试包。实现提交至 [ops-blas](https://gitcode.com/cann/ops-blas)，本文件为开发前设计，尚无实机达标结论。

## 背景介绍

在 Atlas A2/A3 的 arch22 后端补充 complex64 Hermitian 矩阵向量乘。复用句柄、stream 和构建框架，与同族 Ssymv 共用目录；不修改其他算子的计算路径。

# 需求分析

## 接口与语义

在 `include/cann_ops_blas.h` 增加公共声明，复数类型使用 `cann_ops_blas_common.h` 中的 `aclblasComplex`：

```cpp
aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

计算 `y = alpha * A * x + beta * y`。A 按列主序存于 `lda × n`，只读取 uplo 指定三角；另一半通过共轭恢复，对角虚部忽略。x/y 支持正负非零步长，负步长起点为 `(1-n)*inc`，地址计算使用 int64。标量支持 Host/Device 指针，按运行时指针属性处理，参考同族 ComplexScalarSource 的实现。

校验顺序按任务书：handle 判空；拒绝负 n；n=0 直接成功；校验 uplo、lda、步长和标量指针；读取标量后判断 alpha=0 且 beta=1 的 quick return；最后按取值检查 A/x/y。拒绝 INT32_MIN 步长及超出允许跨度的索引。alpha=0 不读取 A/x；beta=0 不读取 y 旧值，但实际输出仍须非空。非法 handle、枚举和数值参数分别使用任务书对应状态码。

# 详细设计

## Host 与工程集成

实现放在 `blas/symv/arch22/`，测试放在 `test/symv/chemv/arch22/`，补齐 `test/symv/chemv/CMakeLists.txt` 的 arch22 GTest 注册。采用与 Ssymv 相近的 host、kernel 和参数组织，不为不同 shape 复制工程。入口从 handle 取得 stream，所有搬运与 kernel 使用该 stream。Device 标量的读取须保证同一 stream 上先前写入可见，错误向调用方传递。

Host 根据 n、uplo、lda、incx/incy 和设备资源选择分块。alpha=0 时使用向量缩放路径；alpha=0、beta=1 或 n=0 不启动 kernel。通用索引和快速连续路径共享数学语义，选择不依赖测试编号或 GPU 基线。

## Kernel 计算与三角访问

初始方案按输出行块分工，每个输出元素由一个核负责，避免对 y 的跨核原子累加。将每个行块的 K 区间按对角线拆分：直接三角读取 `A[i+j*lda]`，镜像三角读取 `conj(A[j+i*lda])`。搬运仅覆盖合法三角区间，不能先把未引用三角搬入再掩码。对角读取后只使用实部。

complex64 在 UB 内拆为实部和虚部，执行 FP32 乘加：实部 `ar*xr-ai*xi`，虚部 `ar*xi+ai*xr`。镜像项翻转 ai 符号。分块归约后再乘 alpha，并在 beta 非零时读取、缩放旧 y。尾块和步长写回使用有效长度，保留向量间隙中的内容。

按列主序安排二维 tile，在所选三角内合并连续搬运，减少逐元素标量读取。x 可按 K 块复用；非单位步长按逻辑索引收集。双缓冲仅在 MTE/Vector 依赖明确后启用，通过队列或事件保证搬运完成后再计算、计算完成后再复用缓冲。不同输出块不重叠；末尾不足一个写回粒度时避免覆盖相邻核的数据。

## 性能迭代

先覆盖上/下三角、小 n、非对齐、较大 n，核实 profiler 的完整采集范围。小 n 减少核数和启动次数；较大 n 调整行块、K 块和 x 复用。若行块方案的跨步访问成为瓶颈，再评估三角块局部累加与独立归并 kernel，计时必须包含归并，不能只计主 kernel。额外 workspace 通过仓内机制分配，保证其生命周期覆盖异步执行。

只有 profiling 证明收益才引入额外路径。保留版本、编译参数与逐次采样，不以最快一批替代平均值。归约顺序及 FMA 的变化均需检查抵消、大标量和非有限值用例，不以放宽容差换性能。

## 支持硬件与限制

| 产品 | 设计目标 |
|---|---|
| Atlas A2 系列（910B3） | arch22，作为本次自验证设备 |
| Atlas A3 系列 | arch22 支持，未实测时不声称完成 A3 验证 |

CANN 固定为任务书要求的 9.1.0。当前 DevEnv 的 9.0.0 仅为已有环境，正式测试前在独立目录准备 9.1.0 并记录实际加载库。输入仅支持 complex64、列主序与 lda/incx/incy 描述的步长；不扩展广播或其他 dtype。无确定性计算和专项内存验收要求。

# 可维可测分析

## 精度与功能

使用 Netlib BLAS 3.10.0 的 `cblas_chemv`，记录源码版本、Release 编译参数、编译器和实际链接库。全部逻辑 y 元素的实部/虚部分别验证，避免误用对称但不共轭的 csymv golden。

执行随包 1000 条精度用例，并检查 200 条性能用例的输出。另补 `GAUSS_NORM_5_2` 用例覆盖上下三角、尺寸、标量、padding 和正负步长，使均匀/正态分布覆盖与任务书一致；正式报告列出补充数量及覆盖，不将附件缺项隐藏。负向参数使用 dummy 缓冲区，不能按非法跨度分配巨量内存；合法负步长用例仍按实际跨度分配。

生态精度按任务书固定标准：atol=rtol=2^-13，matched_ratio≥0.99，最大绝对误差按 1e-2 或 32 ULP 判定。MERE/MARE 作为测试框架的另一个结果保留，不替换生态判据。NaN/Inf 按任务书等效规则单独统计；有限/非有限不一致失败。补充未引用三角污染、对角虚部污染、beta=0 旧 y 为 NaN、Host/Device 标量及空指针的检查。

## 性能标准与附件差异

任务书 §3.3 指定 `msprof op` 的 task duration，附件 README/verify_performance.py 则以 aclrtEvent avg_us 判定，二者不一致。本设计以任务书为准，event 结果仅作诊断对照；保留原附件并在复现说明记录采集方式调整。

正式采集显式使用 `--replay-mode=kernel`，核验 CANN 9.1.0 随附工具的参数、预热和清 L2 行为。工具预热 5 次，正式计划取 64 次有效样本的算术平均（满足任务书至少10次）；应用预热、工具预热分别记录。每次原地计算须恢复初始 y，恢复动作不计入算子 kernel 耗时，并验证 replay 的输入恢复语义。多 kernel 实现统计一次公共接口调用的全部必要 kernel。

200 条性能用例逐项对照随包 `gpu_baseline.csv` 的 A100 数据，倍率定义为 GPU 平均时间/NPU 平均时间，要求每例≥0.8，ms/us 单位转换显式处理。临界项做相同口径的独立复测，保存全部结果及失败原因。

## 交付与维护

提供算子 README、GTest/CSV、复现命令、逐例 XLSX 报告、原始日志和真实终端截图。实部/虚部数据分别呈现。验收 fork 根目录按模板保存 `task_submission`；专项内存项标记不涉及，不伪填通过。实现代码 PR 按任务书提交到 ops-blas，包含必要测试 CSV，与设计 PR 分开记录。

先完成本设计评审合入，再开始设备性能迭代。最终以实际结果填写报告，当前设计不预先承诺全部用例通过。
