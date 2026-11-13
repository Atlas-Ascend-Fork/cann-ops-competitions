# aclblasIsamin（Atlas A2/A3）算子设计

## 1 需求背景

### 1.1 需求来源

任务：[11月社区任务-aclblasIsamin算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/b8d4f23bc41c4c58b64414e16dcf9e34)。依据官网任务包中的 `aclblasIsamin_AtlasA2_A3_task_doc.md`、测试 CSV、A100 基线及测试指导。官网下载包未给出可核实的任务书 Git commit，故不以代码版本或下载哈希代替任务书版本。

代码基点：[ops-blas 5bd95145a9c499e43ebad44a313a1029c63d54bc](https://gitcode.com/cann/ops-blas/commit/5bd95145a9c499e43ebad44a313a1029c63d54bc)。开发目标为该仓 `master`，新增 `blas/iamin/arch22/` 和 `test/isamin/arch22/`，复用公开头文件中的 `aclblasIsamin` 声明。

设计章节遵循本仓 `resources/design_template.md`。任务月份为 11 月；官网与任务包未提供独立数字编号，沿用本任务已有设计 PR 使用的 `11-aclblasIsamin-A2A3` 目录，不编造编号。

### 1.2 背景介绍

BLAS-1 的最小绝对值索引用于向量分析和数值计算。当前仓库存在 arch35 的 Isamin，arch22 尚缺实现；不能把 arch35 的 SIMT 路径直接用于 A2/A3。采用 Ascend C、kernel 直调工程，补齐 Host 校验、Tiling、AIV 向量核和 GTest 注册。

## 2 需求分析

### 2.1 接口与数学语义

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n,
                            const float* x, int incx, int* result);
```

正常域 `n > 0 && incx >= 1`：对逻辑元素 `x[i * incx]`（`0 <= i < n`）取绝对值最小的最先出现位置，返回 `i + 1`。输入为 Device FLOAT32 ND 向量；输出为独立 Device INT32 标量。handle、n、incx 为 Host 参数；无广播，无图模式接口，无输入原地修改。

| 条件，按顺序检查 | 行为 |
| --- | --- |
| handle 为空 | HANDLE_IS_NULLPTR |
| n < 0 | INVALID_VALUE |
| result 为空，包括快速返回路径 | INVALID_VALUE |
| n == 0 或 incx < 1 | 在 handle stream 上以 aclrtMemsetAsync 写 result=0，返回 SUCCESS；失败返回 EXECUTION_FAILED；不读 x、不启动计算核 |
| 正常域 x 为空 | INVALID_VALUE |
| int64(n-1) * incx > INT32_MAX | INVALID_VALUE |
| 正常域 | 异步提交计算，读结果前同步同一 stream |

`n == 1` 且任意正 incx 合法，不按 incx 无谓读取数据。`incx == INT32_MIN` 属快速返回，不调用 `abs(incx)`。跨度、分块与字节偏移采用足够宽的整数计算，禁止溢出后再检查。

### 2.2 特殊值与依据优先级

与任务书及仓内 `isamin_golden` 对齐：初始索引为 1，初始值为 `fabs(x[0])`；若首元素是 NaN，则初始值替换为 FLT_MAX。其后跳过 NaN，仅在绝对值严格更小时更新。全 NaN、全 Inf、全零均返回 1；首元素 NaN 后接 FLT_MAX/Inf 仍返回 1；首元素 Inf 后接 FLT_MAX 则选后者。正负零并列取先出现者。索引始终为整数，不转换为 float 数值保存，避免大索引丢精度。

cuBLAS 文档给出通用索引语义；本任务对 n<0、Device 输出和 NaN 作了明确约束，按任务正文及当前 arch35 Host 校验落地。附件 README/生成器中“快速返回允许 result 空指针”的旧描述与正文不一致，采用正文要求。性能正文要求 msprof op，附件脚本采用 aclrtEvent，正式统计采用正文，event 数据仅作为辅助诊断。

### 2.3 验收目标

- 精度：原始 1000 条用例全部覆盖，INT32 输出与 CPU golden 精确一致；另补正态分布、incx=16、混合 NaN/Inf、空 result、跨核并列与大索引等用例。
- 输入分布：对正常随机输入按相同 shape/stride 配对生成均匀和正态样本，两个分布数量相等；边界/特殊值用例单独统计。
- 性能：原始 200 条 TC_PF，A2 910B3、CANN 9.1.0；msprof op 自带 5 次预热后至少 10 个有效样本，平均耗时不大于对应 A100 gpu_ms × 1000 / 0.8（us）。逐 case 判定，不用总平均掩盖失败。
- A3 用于开发和辅助验证；A2 910B3 数据是正式自验的必需项。任务 §3.1/§3.3 的服务器名称分别为 800I/800T A2，交付记录实际型号与芯片，不擅自认定任意 A3 等效。
- 无独立内存指标；仍检查尾块越界、负向用例分配、输出范围及工作区容量。模板内存表如实填写不适用。

## 3 详细设计

### 3.1 算子分析

连续输入为流式读取加归约，时间复杂度 O(n)，额外空间 O(核数+UB)。小向量受下发延迟影响，大向量受访存与归约吞吐影响。步长路径只访问合法逻辑元素，不以跳过元素参与归约。

归约状态为 `(value, index)`，比较先按绝对值升序、并列按逻辑索引升序。NaN 处理在向量归约前完成：普通 NaN 替换为正无穷；全局首元素若为 NaN，则为其索引 0 注入 FLT_MAX 候选。最终候选以索引 0 的初始状态参与比较，保证全无效和 Inf 边界与顺序 CPU 语义一致。分块、分核结果均采用同一比较规则。

### 3.2 Host / Tiling 设计

1. 先执行 §2.1 的参数校验；正常域才查询当前设备 AIV 核数和 UB 容量。查询失败返回内部错误，不以其他设备的固定核数替代。
2. 小规模单核计算并直接输出，避免跨核工作区与第二次下发。初始分界选择每核至少约 16K 元素，以实机 profiling 调整；阈值只依赖规模、步长和平台资源，不依赖用例名或输入值。
3. 大规模使用 `min(AIV核数, ceil(n/目标每核元素数))` 核。每核处理连续逻辑区间，区间起止采用整数比例分割或对齐分块，所有合法元素恰好覆盖一次。
4. Tiling 按值传递紧凑字段：n、incx、核数、tile 长度；不引入每次 Host→Device Tiling 分配和同步。
5. 多核每核写一个对齐的局部 `(value, uint32 index)` 记录，使用 handle 既有 workspace。空间按 32B 每核对齐；不足时按仓内错误码返回，不越界使用用户工作区。
6. 多核先提交局部归约核，再在同一 stream 上提交单核合并核；stream 顺序保证可见性，避免自旋锁和跨核死锁。合并核按索引处理并列值，最终只写 4B 输出。单核路径只下发一个核。

多核方案以两次 task 的耗时之和作为一次算子调用耗时，不只计第一阶段。若 profiling 表明二次下发成为主要开销，可在保留语义的前提下调整单核区间或采用经验证的设备全局同步方案；实现与测试材料记录最终选择。

### 3.3 Kernel 设计

#### 连续输入

使用 MTE2 将有界 tile 搬入 UB，尾部用 DataCopyPad 指定真实字节数，避免读超过 `1+(n-1)*incx` 的分配。Abs 后先过滤 NaN，再 ReduceMin 得到最小值与首索引；核对所用 ReduceMin API 的并列首索引保证，并加入同一 tile 不同 repeat/lane 的并列测试；若 API 不保证首索引，则先归约值，再定位最小匹配整数索引。局部索引以位解释取得整数，与全局起点相加。仅用真实元素 count 归约，补齐区域不得影响结果。

对 NaN 的向量比较与 Select 显式建立 PIPE_V 顺序；MTE2→V、V→S、S→MTE3 和缓冲区复用均使用事件或队列建立依赖。读取最小值/索引后才复用 UB。首元素 NaN 的特殊候选在标量状态中保留，普通 NaN 置 Inf 不改变初始状态规则。

#### 正步长输入

中小步长按受 UB 容量约束的连续物理片段搬运，Gather 抽取逻辑元素后复用连续归约。tile 大小满足 `(tileCount-1)*incx+1` 的物理元素数上限，Gather 的偏移为局部字节偏移。极大步长不搬运跨越的大空洞，采用逐逻辑元素的有界读取作为正确性回退；不使用 arch35 SIMT 指令。

#### 特殊值与资源预算

保持 FLOAT32 位模式直到 Abs，不降精度、不启用破坏 NaN 语义的 fast-math。在 Abs 或其他浮点运算前保留输入原始位模式。对 FP32 次正规数验证真实硬件行为；若向量最小值落入零/次正规区间，则从保留的原始位模式恢复该 tile 的精确最小绝对值与首索引。tile 间与核间状态统一保存去符号后的 uint32 IEEE754 key 并用整数比较，避免回退后又经浮点比较丢失次正规数。NaN 先分类排除，首 NaN 使用 FLT_MAX key 的索引 0 候选；正负零均映射为 key 0。以专门用例验证所有归约层级。

初始 tile 最大 8192 个 float；连续输入、规整数据、索引、掩码和 ReduceMin 临时缓冲合计按实际使用计算并限制在查询的 UB 预算以内，预留对齐及临时空间。步长搬运物理片段上限同为 8192 float。双缓冲仅在资源核算与性能测量支持时启用；初版采用显式依赖的单缓冲保证正确性。

### 3.4 支持硬件与约束逐项核对

| 任务 §2.5 约束 | 设计处理 |
| --- | --- |
| 参数合法性与校验层级 | §2.1，Host 前置校验，所有负向路径在读输入前返回 |
| 非连续 Tensor | BLAS 正 incx 逻辑访问；incx<1 写零 |
| broadcast | 无，单向量归约 |
| dynamic shape | n/stride 运行时 Tiling，无需图模式 |
| 原地/视图 | x 只读，result 独立，调用者提供不重叠存储 |
| 确定性 | 每层并列取最小逻辑索引；跨核顺序不影响输出 |
| 空 Tensor | n=0 不读输入、不启动计算核，异步写零 |
| 0维 Tensor | 无；标量只作为参数或独立输出 |
| 异步执行 | 全部写入及核下发在 handle stream；不在正常计算中同步设备 |

目标为 Atlas A2/A3 arch22、CANN 9.1.0。不能因工具包含 SIMT 头文件就认定 arch22 硬件支持 SIMT。使用平台实际 AIV/UB 信息，未用 Cube、L1 或 L0 矩阵计算资源。

## 4 可维可测分析

### 4.1 代码组织与兼容性

`blas/iamin/arch22/isamin_host.cpp` 保存校验及调度，`isamin_kernel.cpp` 保存设备实现，紧凑 Tiling 结构放在同目录头文件。沿用顶层源码自动收集，不添加多余算子 CMake。`test/isamin/CMakeLists.txt` 为 arch22 注册 `ops_blas_add_gtest_tests`，arch35 既有分支保留。

测试直接调用公开接口，不能由包装器代替返回空 handle 错误码。设备输入/输出与 CPU golden 分离；任何预期错误和快速返回用例仅分配固定小缓冲，不按非法 n/stride 分配。保持 Device result 语义，避免旧包装器把 Host 栈地址传给异步设备写零。

### 4.2 测试计划

保留任务原始 CSV 和基线，并生成独立补充 CSV，记录种子、分布参数及用例数量。CPU golden 按任务的顺序扫描实现，不能复用设备归约逻辑生成答案。每条输出记录 case ID、n/stride、返回码、actual/golden、成功或失败与输入生成参数。

专项测试包括：null handle/result/x、快速返回、负 n、INT32_MIN 步长、跨度溢出、n=1 且 incx=INT32_MAX、incx=16、正负零、首 NaN+FLT_MAX/Inf、后置 NaN、全 NaN、全 Inf、有限值与 Inf 混合、次正规数、tile/核边界并列，以及索引超过 2^24 的非首最小值。另验证连续异步调用和输出前后哨兵，检查 workspace 复用与单个 4B 写回。

### 4.3 性能及复现材料

精度与性能分开运行；性能前记录设备负载并暂停文件回传，预热后采集至少 10 次成功调用。按单 case 隔离 profiling，保留原始 OpBasicInfo.csv；多 kernel 实现按一次调用配对相加再平均。异常、缺失样本、无基线分别报错，不按零耗时或 GTest 整数毫秒替代。

从完整 commit 提供环境初始化、构建、输入生成、全量精度、自验 profiling、解析及报告生成脚本；输入规则、实际输出与模板均可取得，不依赖开发机私有工具。按官方模板生成步骤说明、精度和性能 Excel/日志及不适用的内存报告；根目录 `task_submission` 保存任务要求文件。复现附件说明列明固定 CSV、基线、配置、输出及生成方式，不附与任务无关的媒体。

提交验收前刷新任务书及附件，核对差异；独立检出验证复现步骤与材料生成。A3 自测与 A2 910B3 正式数据分别保存；未取得 910B3 结果前不宣称正式性能验收通过。
