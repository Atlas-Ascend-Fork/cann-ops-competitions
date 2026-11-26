# aclblasScasum（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务书](https://www.hiascend.com/activities/task-center/details/a698b640487f434ca332f08fb23eb899)要求支持复数单精度 ASUM。以 ops-blas `60d9d0b153e2` 的 `blas/asum/arch22/` 为实现基线，评估归约与跨步读取。本稿用于历史任务设计练习。

## 需求分析

`aclblasScasum(handle,n,x,incx,result)` 计算 `Σ(|Re(x[k·incx])|+|Im(x[k·incx])|)`，输入为交错 complex64，输出为 Device FP32 标量。n≤0 或 incx≤0 时结果为 +0，非正增量没有反向遍历语义。

## 详细设计

Host 先检查句柄和 result，普通路径检查 x 与 `(n−1)·incx≤INT32_MAX`，在 int64 计算。快速路径使用 handle stream 上的异步零填充，只写 4 B，省略归约。结果写入以设备 stream 顺序可见，无 Host 结果缓冲。

单核路径适用于短向量，批量读取交错实虚并在 UB 做绝对值、向量归约；对连续输入可以直接对 2n 个 float 取绝对值归约。中大规模对逻辑索引均分，分别归约每核有效元素；每核写一个独立的 32 B 对齐局部和槽，第二个 kernel 按核号固定顺序合并。C 个核的 workspace 为 32C 字节，向量有效读量为 8n 字节。若 workspace 不足使用单核或降低核数；实际跨步读流量和双 kernel 启动成本单列。大步长只搬运逻辑点，尾块以有效长度归约，避免读取向量末端之外的数据。

## 可维可测分析

`test/asum/scasum/arch22/` 使用附件精度和性能用例，以 CBLAS scasum 做标量比对；补测实虚符号相反、仅虚部非零、奇数尾、负 n/非正增量返回 +0、result 为空、溢出及同一 stream 上覆盖结果。910B3 用 `msprof op` 预热 5 次、至少 10 次有效调用，双阶段累计 kernel 耗时，比较 NPU≤1.25×GPU。板上精度与性能待实测。
