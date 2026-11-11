# aclblasIcamax Atlas A2/A3 设计文档

## 需求背景

### 需求来源

[2026 年 11 月昇腾社区 aclblasIcamax（A2/A3）任务](https://www.hiascend.com/activities/task-center/details/3187f64ea116468190cabc7779a7cb03)。[待验收代码仓](https://gitcode.com/S_FF/aclblas_icamax_a2a3)的 `main` 分支包含完整交付件；实现位于 `blas/iamax/arch22`，公共接口使用 `include/cann_ops_blas.h`。本文文件路径均相对于该代码仓。设计章节参照 cann-competitions 的 `04_tasks/01_community-task-2026/resources/design_template.md`。

### 背景介绍

Icamax 从单精度复数向量中返回 L1 比较量最大的第一个元素的逻辑索引。A2/A3 实现采用 Ascend C Vector 指令和单次 kernel 启动，覆盖连续与正步长输入，并遵循 Netlib 的索引与特殊值语义。

## 需求分析

### 需求描述

接口：`aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)`。

普通输入令 `v[i] = abs(real(x[i*incx])) + abs(imag(x[i*incx]))`，输出 `1 + min(argmax(v))`。输入为交错存储的 COMPLEX64，输出为设备端 INT32 标量；不是复数模长平方。调用异步入流，由调用方同步后读取结果。

### 需求拆解

| 项目 | 实现与验证 |
|---|---|
| 参数与快速返回 | null handle、n<0、null result、正常路径 null x、跨度溢出检查；n=0 或 incx≤0 异步写 0 |
| 连续归约 | 8,192 个复数一块，Abs → PairReduceSum → NaN 掩码 → ReduceMax |
| 正步长 | DMA 分段采集到 UB，按逻辑顺序标量比较 |
| 多核归并 | 每核保存候选值与逻辑索引；SyncAll 后 core 0 归并 |
| 精度 | 参考 Netlib BLAS 3.10.0，整数索引完全相等 |
| 性能 | 全部 200 条原始基线逐项对照，不合并重复 shape |

## 详细设计

### 算子分析

支持一维逻辑长度 n 与 int 步长 incx。正常路径要求 `(int64_t)(n-1)*incx <= INT32_MAX`，且调用方分配足够的实际跨度。n<0 返回非法参数；参数检查顺序见 host 源码和边界测试。第一项含 NaN 时结果固定为 1；首项不含 NaN 时后续 NaN 不更新最大值。Inf 和相等最大值均选择最小逻辑索引，与指定 Netlib 参考一致。

### 算子实现

**Host tiling。** 读取平台可用 AIV 核数；目标核数为 `min(AIV, ceil(n/2048))`。每核逻辑长度向上对齐至 32 个复数，再计算实际核数，避免启动空核。Tiling 字段为 n、incx、coreCount、perCoreN。通过 incx 选择 kernel 内连续或步长分支，不额外注册 tiling key。

**分块与内存。** 每块最多 8,192 个复数，输入 UB 65,536 B、比较量 32,768 B、mask 1,024 B、ReduceMax scratch 32,768 B、归约结果 32 B、输出队列 32 B，源码显式申请合计 132,160 B/核。本次平台查询 UB 可用容量为 196,352 B/核。单核不需要算子 workspace；多核需要 `coreCount*32 B`，本机 40 AIV 时最多 1,280 B。框架 handle 默认分配 32 MiB workspace，此数与算子实际使用量分别报告。算子接口内部没有逐次内存分配或主机同步。

**连续 kernel。** 输入交错实部/虚部直接 Abs，再两两求和，省去两次 gather。8,192 项对应 256 次 pair-reduce repeat，拆为 255+1，满足单指令限制。尾部按 64 个 FP32 比较项补齐；填充 NaN，经掩码替换为 -1，保证不参与有效最大值竞争。各块按顺序只在严格更大时替换候选，保留最早索引。

**步长 kernel。** DataCopyPad 每次采集最多 256 个复数，每个复数占一个 32 B UB 块，再标量读取实虚部。超出 DMA stride 字段时降为单项采集。读取路径经过 UB，避免重复调用中的 GM 标量缓存陈旧问题。

**归并与同步。** 每核向 workspace 的独立 32 B 槽写 8 B 候选；PIPE_ALL 屏障和 SyncAll 后由 core 0 读取，按值更大、相等时索引更小的规则归并，最终仅向 result 写 4 B。整个正常调用只有一个 kernel；host 限制 block 数不超过 AIV 数。

### 支持硬件

实现编译目标为 arch22，A2 使用 `ascend910b`，A3 使用 `ascend910_93`。本次实机验证仅在 **Ascend 910B3（A2）** 完成，不能据此声称已做 A3 实测。CANN 9.1.0，驱动 25.5.0；完整配置和查询原文见 `task_submission/验证配置与设备参数.md` 与 evidence。

### 算子约束限制

使用默认 workspace 或调用方提供的足够 workspace；多核不足时返回执行失败。输入地址和输出地址必须有效、分配跨度足够。支持偏移地址以及非 32 B 整数倍长度。并发调用若共享同一 handle/workspace，应由调用方保证不发生数据竞争。

## 可维可测分析

### 精度标准与结果

Netlib `cblas_icamax` 的 0-based 结果加 1，按 INT32 完全相等判定；quick return 为 0。原始 1,000 精度/边界、434 正态补充、200 性能用例精度、6 个 GTest 边界组共 **1,640/1,640 通过**。原始实际随机归约精度用例 434 条均匀，与 434 条正态补充一一对应；quick return、非法参数和特殊值不混入分布占比。边界组另列 55 条有输出索引的实际断言，不把组计数重复累加。

### 性能标准、方法与结果

本次使用通用 `msprof --task-time=l0 --ai-core=off --aicpu=off --runtime-api=on`。每用例 1 次正确性调用、5 次预热、10 次采样。原始 task_time 表按时间排序后每 16 条映射一个用例，去掉前 6 条，计算剩余 10 个设备任务耗时的算术平均。保留原始 GPU 基线及 2,000 个有效样本；以 `Ascend平均us <= GPU_ms*1000*1.25` 判定。

结果 **200/200 通过（100%）**。n=1,048,576 / 2,097,152 / 4,194,304 的 TC_PF_1001/1002/1003 分别为 9.674 / 15.190 / 25.800 μs。最小门槛余量约 18.44%。这些是本次配置下的通用 msprof 实测值，不是 aclrtEvent 日志值。

任务书指定的 `msprof op` 因物理设备 6 与逻辑设备 0 映射问题采集失败；0 μs 记录无效。因此不能标为该指定采集方式的验收通过。当前报告完整披露已完成的方法、数据和限制。

### 内存与兼容性分析

任务书 §3.4 内存指标为“不涉及”。额外进行 9 组实机缓冲区守卫检查，每组预热 5 次、再调用 20 次，结果与守卫均通过，调用前后设备可用内存差为 0。`aclrtGetMemInfo` 是设备级快照，不是算子峰值；不构造 GPU 内存基线或比例。

保留现有公共声明与 arch35 路径；新增 arch22 实现和对应测试。原始附件 CSV 按字节保留。交付件按官方模板形成三份 Excel，附完整日志、XML、源码/二进制 SHA256、截图及可复现脚本。
