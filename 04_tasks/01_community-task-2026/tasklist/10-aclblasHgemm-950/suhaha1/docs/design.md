# aclblasHgemm（Ascend 950PR）设计文档

# 需求背景（required）

## 需求来源

本设计对应[10月社区任务-aclblasHgemm算子开发（950）](https://www.hiascend.com/activities/task-center/details/f0160348759e46018f2609cfb2b7f9df)，依据该网页提供的[任务书及测试附件](https://www.hiascend.com/p/resource/202610/160a8d320b4a416584a74e12630bd93c.zip)。任务书与附件未提供可核实的源仓库 Git commit，版本来源尚待确认；不以下载时间或文件哈希代替任务书 commit。

工程分析基于 [ops-blas commit 05b96ea22cea926ecfd43fc7ac691a795522bd61](https://gitcode.com/cann/ops-blas/commit/05b96ea22cea926ecfd43fc7ac691a795522bd61)，文档结构依据任务指定的[设计模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。

## 背景介绍

HGEMM 是 BLAS Level-3 半精度实数通用矩阵乘接口，用于将半精度矩阵的乘加计算提交给设备。目标仓库已有 arch35 GEMM 的参数描述、列主序映射、分核和 Cube 流水框架，需要在同一家族补齐 FLOAT16 输入输出的公共接口。实现沿用句柄绑定 stream 的 Ascend C kernel 直调方式，供其他产品线共用 `aclblasHgemm` 声明。

# 需求分析（required）

## 接口与数据

公开接口位于 `include/cann_ops_blas.h`：

```cpp
aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle,
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k,
    const aclblasHalf* alpha,
    const aclblasHalf* A, int lda,
    const aclblasHalf* B, int ldb,
    const aclblasHalf* beta,
    aclblasHalf* C, int ldc);
```

alpha/beta 是 Host 上的 FLOAT16 标量指针；A/B/C 是 Device 上的 FLOAT16 矩阵。C 原地更新。矩阵按 BLAS 列主序存储，允许前导维 padding，无 batch、广播、任意 stride 或视图返回语义。N、T、C 三种转置模式全部支持，实数的 C 与 T 等价。

| 对象 | op=N 的物理形状 | op=T/C 的物理形状 | 前导维最小值 |
| --- | --- | --- | --- |
| A | m×k | k×m | max(1, transa=N ? m : k) |
| B | k×n | n×k | max(1, transb=N ? k : n) |
| C | m×n | 不适用 | max(1, m) |

## 边界与异常

Host 首先验证 handle、枚举、非负维度、前导维和 alpha/beta 指针；alpha/beta 判空必须先于 m/n 的 no-op 返回。handle 为空返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`，其他非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。

合法的 `m=0` 或 `n=0` 返回 SUCCESS，不访问 A/B/C。非空输出场景下，A/B 仅在 `k>0 && alpha!=0` 时必须非空；C 在 `k>0 || beta!=0` 时必须非空。按任务约定，`k=0 && beta=0` 时允许 C 为空；C 非空则将有效输出区域置零。`k=0` 和 `alpha=0` 均跳过 A/B 读取，仅执行 beta 缩放；beta=0 不读取旧 C，beta=1 可直接保留 C。所有写入只覆盖 C 的 m×n 有效元素，保持列间 padding 不变。

## 验收目标

新增公共 `aclblasHalf` 类型别名及 `aclblasHgemm` 声明，保持各产品线共用API。目标为 Ascend 950PR、CANN 9.1.0。精度 golden 用 Netlib BLAS/CBLAS 3.10.0 的 FP32 `cblas_sgemm`，再按 round-to-nearest-even 转为 FLOAT16。按任务书 §3.2 使用 rtol=atol=2^-10、有限元素匹配率≥0.99、最大误差保护以及规定的 Inf/NaN 等效规则。不得通过放宽阈值处理误差。

性能按 task duration 计时，与附件 `gpu_baseline.csv` 对齐：每个性能 case 的 GPU/NPU 耗时倍率≥0.4，全部性能 case 倍率算术平均≥0.5。内存不设单独性能指标，但实现必须防止越界、工作区尺寸溢出和缓冲区复用冲突。

# 详细设计（required）

## 算子分析

计算定义为：

`C[i,j] = alpha * sum(p=0..k-1, op(A)[i,p] * op(B)[p,j]) + beta * C[i,j]`。

列主序访问规则如下：

- A：N 时为 `A[p*lda+i]`，T/C 时为 `A[i*lda+p]`。
- B：N 时为 `B[j*ldb+p]`，T/C 时为 `B[p*ldb+j]`。
- C：`C[j*ldc+i]`。

正常路径计算复杂度为 O(mnk)，输入输出空间为 O(mk+kn+mn)。大矩阵主要受 Cube 算力、分块重用和搬运流水影响；小矩阵主要受任务启动、低核利用率和尾块开销影响。采用 FP16 乘法输入、FP32 累加和 FP32 标量运算，最终只转换一次到 FP16，避免在 K 循环中反复截断为 FP16。

## Host/Tiling 设计

实现位于 `blas/gemm/arch35/`。公共入口负责参数检查、读取 Host 标量、选择执行路径、从 handle 取 stream、计算 Tiling 和工作区需求，再提交 kernel。工作区已满足需求的常规入口异步提交；由 handle 管理的默认工作区扩容按现有仓库契约同步后替换，用户显式工作区不足则返回失败。测试侧负责同步及报告运行错误。

复用同族 `GemmTilingData`、列主序交换映射和分核思路，将 `C^T = op(B)^T * op(A)^T` 映射到内部行主序 GEMM，交换 m/n、A/B、lda/ldb 和转置标志，避免额外整矩阵转置。FP16 专用参数通过明确的 Tiling 字段或类型特化传递，不把 FP16 地址直接交给读取 FP32 的现有 kernel。

执行路径为：

| 条件 | 路径 | 工作区 |
| --- | --- | --- |
| m=0 或 n=0 | Host no-op | 无 |
| k=0 或 alpha=0 | FP16 beta 缩放/置零 kernel；beta=1 可返回 | 无 |
| alpha=1 且 beta=0 | FP16 Cube GEMM，FP32 累加后直接转换并写 C | 无整矩阵中间结果 |
| 其他合法标量组合 | Cube 写 FP32 乘积，AIV epilogue 应用 alpha/beta 后写 FP16 C | 对齐后的 FP32 乘积矩阵 |

一般路径的工作区按内部输出布局计算，参考同族的 `align_up(m,16)*n*sizeof(float)` 方式与 handle 的工作区管理契合。尺寸乘法用 64 位并检查 size_t 溢出；工作区不足返回仓库规定的失败状态，禁止静默采用低精度结果。stream 内核顺序保证乘积先于 epilogue 完成，工作区生命周期遵循 handle 约定。

输出 M/N 逻辑tile展开成一维编号，按核号循环分配，使每个tile仅由一个核更新，每核最多ceil(总tile数/核数)个tile，不使用输出原子累加。核数不超过设备 AIC 核数和有效 tile 数，小规模减少启动核数。K 维先保持单核完整累加；只有性能分析证明需要时才引入有明确 FP32 归并语义的 K 切分。

FP16 候选采用小块 BM=64、BN=64、BK=32、K-panel=64；当平台 L1 容量满足要求时，中等尺寸候选采用128×128、BK=128、K-panel=512，大尺寸候选采用256×256、BK=64、K-panel=256。分核通过一维循环分配均衡tile数量；小规模按有效tile数减少启动核数。候选根据精度和性能验证选择，并重新核对资源及尾块。tile 和分核策略按输入维度/转置/对齐特征选择，不按 case ID 或测试数据取值选择。尾块使用真实 m/n/k 长度参与搬运和计算，内部补齐元素显式置零，最终仅写有效输出。

## Kernel 设计

复用同族 Gemm 的 GM→L1→L0A/L0B→L0C→GM 流程和 tensor_api 数据布局，补齐半精度输入及输出特化。输入 FP16 使用 16 元素 fractal 宽度，L0C 用 FP32；使用原生 FP16×FP16→FP32 Cube 运算。T/C 的实现只改变源矩阵视图和搬运布局，禁止对实数执行额外共轭操作。

L1 A/B panel 与 L0A/L0B 使用双缓冲，大L1路径预取下一段K数据，与当前段搬运和计算形成流水。MTE2/MTE1/M/FIX 之间沿用清晰的生产/消费事件；缓冲区覆盖前须等待前一次使用完成，kernel 退出前回收已设置的事件。L0C 首个 K 块清零并初始化累加，后续块累加，末块完成后再输出。不同 M/N tile 复用 L0C 前等待 Fixpipe 完成。

快速路径将 FP32 累加结果按 round-to-nearest-even 写为 FP16。一般路径保存 FP32 乘积，AIV epilogue 计算 `float(alpha)*product + float(beta)*float(C_old)`，最后转换为 FP16；beta=0 分支不搬入旧 C。beta 缩放路径按列或有效元素 tile 处理，避免把 padding 当作输出矩阵的一部分清零。FP16 次正规数、Inf/NaN 和溢出行为在测试中单列核对，不把所有非有限值一概判为通过。

## 硬件与约束

仅承诺任务要求的 Ascend 950PR（arch35/dav-3510），使用 CANN/asc-devkit 9.1.0 的 Ascend C 与 Tensor API 能力。README 明确标注 Ascend 950PR 支持；其他产品线的支持不由本任务推断。

资源规划保留仓库通用 `HardwareInfo<ASCEND_V350>` 的L1 32 KiB默认预算；Hgemm的大L1模板在Host查询确认L1≥512 KiB后启用。L0A/L0B各64 KiB、L0C 256 KiB、UB 248 KiB，并与平台查询及实际编译资源报告核对。小块L1双缓冲占 `2*(64+64)*64*2=32768` 字节，L0A/L0B各8192字节，单tile FP32 L0C 16384字节。中块L1双缓冲占 `2*(128+128)*512*2=524288` 字节，L0A/L0B各65536字节，L0C 65536字节。大块L1双缓冲占 `2*(256+256)*256*2=524288` 字节，L0A/L0B各65536字节，L0C 262144字节。各模板由编译期不等式约束，容量不足时回退小块，不超额分配。

AIV epilogue 的输入、FP32临时值、FP16输出和队列缓冲合计不得超过UB预算。数据搬运按接口要求对齐，非对齐前导维与M/N尾块用支持真实字节数的搬运路径处理，不以向上取整读取超出矩阵分配范围的数据。索引与偏移使用64位，不依赖int乘法中间值。

# 可维可测分析

## 模块与兼容性

公开API声明、Host检查、Tiling、Cube计算、AIV epilogue和测试适配分别维护。优先抽取/扩展同族可复用模板，保留已有SGEMM/CGEMM行为。共享模板改动须回归其转置、padding、标量与边界路径；新增Hgemm按单算子构建过滤规则登记依赖，不引入与任务无关的工程文件。

自测工程放在 `test/gemm/hgemm/`，包含参数解析、CPU golden和架构专用GTest/CSV。非法枚举不得被解析器默认转换为N；NULLPTR、null、handle_null等附件语义须实际作用于调用参数。测试README提供构建、数据生成、全部精度测试、性能采集与结果解析命令。

## 精度验证计划

完整执行附件1000条精度case，清晰区分200条性能case；覆盖9种转置组合、尺寸扫描、前导维padding、零维、k=0、alpha/beta特殊值、空指针、非法参数及非有限值。为任务要求补充真实的均匀/正态混合分布输入，固定种子，先随机μ∈[-5,5]、σ∈[0.1,2]再采样，记录正态分布μ/σ；不能让 `--dist mixed` 只改变提示文字。对16384维补充预算内的瘦长矩阵和边界用例，保留官方1200条用例原始来源及全部覆盖。

输入FP16先提升到FP32，使用Netlib 3.10.0 `cblas_sgemm` 产生FP32结果，再舍入为FP16。有限元素按 `abs(actual-golden)<=2^-10+2^-10*abs(golden)` 统计匹配率；选择有限点中最大绝对误差的位置（并列取列主序首点），以该点golden为锚，检查最大误差不超过 `max(0.1,32*ULP_FP16(golden_at_max_error))`；逐点ULP检查仅作额外诊断。非有限元素单独记录总数/等效数，按任务书NaN/同号Inf/NaN与Inf规则判断；一方有限而另一方非有限判失败。报告输出匹配率、最大误差及位置、超限数量、非有限值统计和返回码。

beta=0且旧C含NaN的用例验证不读C；padding设置哨兵并验证未被覆盖；零维和空指针组合验证短路顺序。必要时增加m/n/k尾块、极小数和大k误差分析，不用更宽阈值替代定位。

## 性能验证计划

性能集按case参数与官方 `gpu_baseline.csv` 逐条关联，保留全部200个case，GPU的ms转换为us后与NPU比较。使用 `msprof op` 的task duration，不使用GTest包含数据准备/golden/校验的整用例毫秒值冒充kernel时间。若接口包含多个串行kernel，统计完整接口对应的全部计算task时长，不能只选乘法kernel忽略必要epilogue。

任务书§3.3写明5次warmup、10个有效样本，§7与附件分别要求超过10次、超过50次。计划预先固定采集5次warmup后60个有效样本，同时报告前10次均值和全部60次均值，两种口径均应用逐case≥0.4和全体平均倍率≥0.5的要求。保留完整原始样本、采集配置、设备负载和异常样本；若审核方统一样本口径则按其明确要求执行。只将调用SUCCESS且执行无误的样本纳入统计。

## 复现与交付

固定最终验收完整commit及依赖版本，提供从独立检出、部署Netlib和工具链、准备CSV/基线、执行全部自验到生成报告/截图的连续命令。保留输入生成方式、种子、配置、实际输出、日志与报告之间的对应关系；代码外文件随验收包提供并编写复现附件说明。测量值从本次原始输出解析，不手填或复制历史结论。交付前从原网页刷新任务书、处理变化，再以独立检出完整复现并核对ZIP内容、压缩算法、CRC、解压一致性和100M限制。

### 口径待确认事项

任务书§7写“达到≥GPU数据”，与§3.3反复列明的单case≥0.4×、平均≥0.5×存在差异。以§3.3的明确公式计算并完整保留所有case倍率，另列≥1.0×的实际达成情况；未取得任务方澄清前不将不同阈值写成同一条件。该事项不通过改变基线或删减用例处理。
