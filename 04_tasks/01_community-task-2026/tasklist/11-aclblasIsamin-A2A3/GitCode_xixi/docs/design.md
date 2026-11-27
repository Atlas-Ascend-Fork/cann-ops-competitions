# aclblasIsamin（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务书](https://www.hiascend.com/activities/task-center/details/b8d4f23bc41c4c58b64414e16dcf9e34)规定 FP32 最小绝对值索引语义。以 ops-blas `60d9d0b153e2` 中 arch35 Isamin 为接口参考。本稿用于历史任务设计练习。

## 需求分析

`aclblasIsamin(handle,n,x,incx,result)` 寻找 FP32 向量中最先出现的最小绝对值，写 1-based Device INT32。n=0 或 incx<1 时写 0。

## 详细设计

Host 检查 handle、n<0 与非空 result。快速路径按 stream 异步写零；普通路径校验 x 并以 int64 计算 `(n−1)·incx`，超过 INT32_MAX 返回 INVALID_VALUE。按 k·incx 访问向量。小形状用单核，大形状按逻辑区间分核，单位步长连续搬运，稀疏步长直接读取有效元素。

首值非 NaN 时按其绝对值初始化全局比较种子，首值为 NaN 时以 FLT_MAX 作种子、索引保持 1。块内跳过 NaN，合并候选时先比绝对值、再取最小逻辑索引；候选严格小于种子才替换索引 1。全 NaN 返回 1，首项 NaN 且后续等于 FLT_MAX 也返回 1。归约核按同一 stream 顺序写一个标量。

## 可维可测分析

按任务书搭建 CPU 参考函数，对附件 1000 条精度用例做整数精确匹配。覆盖重复最小值、±0、首项 NaN 后跟有限数、全 NaN、Inf/FLT_MAX 和跨度上界。910B3 上对 200 条性能用例采集 `msprof op`，5 次预热后至少 10 次有效采样取平均，目标 NPU≤1.25×GPU。上板结果待测。
