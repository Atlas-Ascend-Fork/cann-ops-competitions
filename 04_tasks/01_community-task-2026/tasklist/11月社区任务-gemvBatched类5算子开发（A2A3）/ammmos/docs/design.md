# 需求背景（required）

## 需求来源

《aclblasSgemvBatched 类算子开发任务书》，2026-11-24 修订版；适配 Atlas A2/A3、CANN 9.1.0，代码基于 ops-blas 的句柄式 BLAS 工程。

## 背景介绍

为 S/HSH/HSS/TST/TSS 五种实数批量 GEMV 接口提供 arch22 支持。开发基线为 ops-blas `55ef791b77f3677b45da3468b0a8b17941424f84`，该版本已存在五种 arch22 实现，本次在该实现上补齐任务书规定的参数和测试语义。公开声明保持仓内接口。

# 需求分析（required）

## 需求描述

逐 batch 执行 `y_i = alpha * op(A_i) * x_i + beta * y_i`。Device 指针数组分别存放各 batch 的地址，不要求 batch 连续。A 为列主序 `lda × n`；转置只改变计算方向，`lda >= max(1,m)`。支持 N/T/C，实数 C 等价于 T。

| 变体 | A/x | y | alpha/beta | 计算 |
| --- | --- | --- | --- | --- |
| S | FP32 | FP32 | Host FP32 | FP32 |
| HSH | FP16 | FP16 | Host FP32 | FP32 |
| HSS | FP16 | FP32 | Host FP32 | FP32 |
| TST | BF16 | BF16 | Host FP32 | FP32 |
| TSS | BF16 | FP32 | Host FP32 | FP32 |

## 需求拆解

1. 五个标准入口共用参数校验、tiling 和 kernel，通过 dtype 分派。
2. 支持带 padding 的 lda、正负步长、非连续的 batch。
3. 结构合法且 m/n/batchCount 任一为 0 时不读取数组、不启动 kernel。
4. alpha=0 不读取 A/x；beta=0 不读取旧 y，避免无关 NaN 污染。
5. 5000 条精度及 1000 条性能用例覆盖全部变体；准确报告实际验证状态。

# 详细设计（required）

## 算子分析

N 情况下输出长度 m、归约长度 n；T/C 情况下输出长度 n、归约长度 m。物理矩阵地址恒为 `A[col*lda + row]`。

负步长按照 BLAS 约定，从 `(1-length)*inc` 对应的位置开始，再逐次加 inc。调用方传入分配区起始地址。Host 采用 64 位中间值校验 A/x/y 的有效跨度，避免内核索引越界。

## 算子实现

### Host 侧设计

入口顺序：handle → trans → m/n、lda、incx/incy、batchCount → alpha/beta → 非空操作的数组 → launch。空操作的数组允许为空。非法 handle 返回 HANDLE_IS_NULLPTR，非法 trans 返回 INVALID_ENUM，其余非法参数返回 INVALID_VALUE。

五个接口共用 `GemvBatchedImpl<TIn,TOut>`。结构合法的空操作直接返回，不查询设备核数。普通操作从 handle 取得 stream，异步下发一个 AIV kernel，不在库接口内部同步。

按 batch 与输出区间共同分核。batch 数不足核数时，将输出划为不重叠区间，N 路径按 32 个元素对齐，T/C 路径按 8 个元素对齐；每项工作对应一个 batch 的一个输出区间。N路径在batch不足核数时进一步限制输出区间不超过256行，以保持多列块处理；batch已足够占满核时保留原分区，避免额外工作项回退。核间不共享累加结果，不需要跨核归约。batch 足够多时退化为按 batch 分核。

Tiling 包含 dtype、N/转置标记、FP32 alpha/beta、m/n、工作项切分、outputTiles/outputTileSize、lda、incx/incy。N/T/C 中 T/C 归一为同一计算路径。

### Kernel 侧设计

使用 Ascend C DataCopyPad、Cast、Muls、Mul、Add/Sub、Compares/Select 和归约。向量 tile 为 2048 元素；基础 UB 为十个 2048×FP32 缓冲加掩码、归约、指针、标量缓冲，共 84992 字节。N路径增加8192字节的广播索引；n≤4096时再分配16384字节的x缓存。非单位步长或极窄N场景增加16384字节窗口与8192字节步长索引。上述条件全部满足时合计134144字节。

N 路径先计算点积，再乘 alpha 并与 beta*y 合并。输出行片段不超过 1024 元素时，以多列 DataCopyPad 合并搬运，GM 仅读取有效行，UB 行间距补齐到 32 个元素；块间间隔及右侧零填充支持奇数和极小行数，补零 Duplicate 使用对齐地址。UB内按列位置保留高位和补偿累计，全部列块完成后再用TwoSum树合并列位置，减少每次搬运后的重复归约；其余片段保留逐列路径。可支持的n≤4096向量按batch缓存在UB。广播索引按行数缓存：2次幂行数使用位移，其余32倍数行数使用带0.25偏置的FP32序号除行宽再向下取整，最后转换为字节偏移；所有块统一Gather广播。以 TwoSum 保存加法残差，FP32 输入额外以 Dekker 分解保存乘积残差；半精度/BF16 输入的乘积可由 FP32 精确表示。

T/C 路径使用同样的乘积残差，并以对齐的向量 TwoSum 树归约及高低位累加合并 tile。m 为 16/32/64/128 时，最多 16 个输出列共用一次矩阵搬运和按 repeat 分列的向量归约。最后八项归约使用 Gather 打包和向量 TwoSum 合并，已通过完整精度回归。普通路径保留标量高低位尾部合并。极大 FP32 输入回退到逐列二次幂动态缩放路径，避免有限输入在归约途中溢出。补偿中的非有限项清零，实际点积保持 Inf/NaN 传播。

单位步长使用连续 DMA；非单位步长在跨度不超过 4096 时搬运窗口后按步长 gather，否则逐元素读取。输出非单位步长在支持范围内用Gather扩展与分块DataCopyPad写回，其余情况逐元素scatter；只写逻辑输出位置，保留padding。

FP16/BF16 输入转 FP32；最终按目标 dtype 舍入。alpha=0 时不解引用 Aarray/xarray 的成员，也不进入矩阵计算；beta=0 使用零初始化累加缓冲。DMA、标量与向量流水间采用事件和 barrier 保证可见性。

## 支持硬件

| 硬件 | 实现路径 | 验证状态 |
| --- | --- | --- |
| Atlas A2 | arch22 | 910B4/CANN9.1.0：代码6a15925：5000条CSV精度及3项固定/合同/相消测试通过，293.262秒；319项冒烟与14项性能探针通过 |
| Atlas A3 | arch22 | 尚未实机验证 |

## 算子约束限制

- 不支持广播，不接受重叠的输出 batch；输入与输出不得重叠，由调用方保证。
- 不扫描 Device 指针数组成员的合法性。
- 当前跨度保护拒绝超过 INT32_MAX 的有效寻址范围，以及 INT_MIN 步长。任务书“任意非零 int”与自验包 INT_MIN 负向用例口径存在差异，采用自验包要求，并明确记录限制。
- 大矩阵性能需按实际采集结果判断；输出区间并行不等于性能验收通过。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 方法 | 标准 |
| --- | --- | --- |
| 精度 | 主验收域为任务书§3.2.4允许的 cblas_dgemv 逐 batch FP64 累加后收窄；另保存 FP32 参考域差异 | 按输出 dtype 的混合容差，matched_ratio≥0.99 并满足最大误差上限 |
| 正式性能 | 910B3、msprof op Task Duration | warmup 5，取随后 10 次均值，NPU≤gpu_ms×1.25 |
| 辅助性能 | 预分配 Device buffer、aclrtEvent | warmup 5，连续 10 次均值，输出 [PERF] avg_us |
| 内存验收 | 任务书未要求独立内存测试 | 不编造内存报告通过结论 |

测试侧修复 FP16/BF16 转换，复用仓内 dtype_compat；HSS/TSS 以 FP32 输出阈值判定，HSH/TST 的参考输出先量化后比较。负向用例直接调用对应真实 API，不在非法维度上分配或计算 abs(INT_MIN)。默认参考域为FP64；显式GEMV_REFERENCE_FP64=0仅用于FP32诊断。合同补充测试覆盖五接口的错误优先级及空操作。新增乘积舍入相消回归：前1024项(8+2^-10)*(8-2^-10)、后1024项(-8)*8，精确点积为-2^-10；舍弃乘积残差会错误输出0，不采用仅凭乘积幅值判定的低精度快速路径。

当前提供的 gpu_baseline.csv 中 T/C 行按任务书修订要求需重测。当前6a15925在910B4上完成全量1000条性能子集采集，1257.74秒，普通msprof的task-time通道取得 16000 条真实 kernel 记录。每例按串行调用顺序跳过 1 次精度调用及 5 次 warmup，保留随后 10 次 Task Duration 样本，共 10000 个有效样本。解析器拒绝失败、重复、缺失或非法耗时，不猜测对应关系。`msprof op` 的采集通道仍失败（0 success / 10 failed），因此保留硬件与采集方式差异，不能冒称原任务正式性能验收完成。

当前6a15925的374条有效N基线开发对比为140条≤1.25倍、234条超限；626条T/C旧基线不作有效比较。相对2a8eac5的N用例耗时合计下降0.67%，中位下降0.50%，整体收益有限。社区CI #4642全部通过（10分29秒）。原始10000个样本已本地独立解析，逐例逐样本与远端一致。

6000条精度与性能用例的原生输入及golden共24477361024字节已冻结，5890个有数据用例完成尺寸与SHA256校验，110个负向/空操作用例记录无张量要求。已整理cases.json、perf/cases.json、inputs/、golden/和facts.json，但完整张量约24.48GB仍在远程，尚未全量交付。附件未提供repo-task-case-gen，现有工具仅为已注明来源的布局适配器，正式格式契约仍需确认。正式910B3性能仍待完成。

## 兼容性分析

不修改公开接口、dtype 编码与架构选择。arch35 实现不变；测试侧共享参考函数保持已有行为。Host 对 OP_C 和合法空操作扩展支持范围，非法枚举优先返回 INVALID_ENUM。
