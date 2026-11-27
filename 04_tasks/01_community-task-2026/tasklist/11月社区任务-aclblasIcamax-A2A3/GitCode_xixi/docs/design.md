# aclblasIcamax（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务书](https://www.hiascend.com/activities/task-center/details/3187f64ea116468190cabc7779a7cb03)要求在 A2/A3 提供单精度复数向量的最大模索引。参照 ops-blas `60d9d0b153e2` 的 arch35 Icamax 和 arch22 Icamin。本稿用于历史任务设计练习。

## 需求分析

`aclblasIcamax(handle,n,x,incx,result)` 在交错 complex64 向量中比较 `abs(Re)+abs(Im)`，并列时取首个逻辑元素，输出 1-based Device INT32。正步长元素 k 的物理位置为 k·incx。

## 详细设计

Host 优先检查 handle，再检查 n<0 与 result 为空。n=0 或 incx≤0 时在 handle stream 上异步清零 result，省略归约。普通路径要求 x 非空，以 int64 检查 `(n−1)·incx≤INT32_MAX`。

小向量单核完成；大向量按逻辑元素均分，块内解交错、向量计算 FP32 模值。使用带有效位的候选 `(value,index)`，先比模值，再取较小逻辑索引。全局首元素为 NaN 时显式返回 1，其余 NaN 跳过，Inf 保持相同并列规则。连续输入按块搬运，密集跨步比较窗口提取与逐点加载，稀疏步长只读取逻辑元素。多核各写独立的对齐候选槽，再由同 stream 上的归约核写 result；workspace 不足回退单核。

## 可维可测分析

`test/iamax/icamax/arch22/` 使用附件全量数据，对有效输入将 CBLAS 的结果转换成 1-based 后精确比较。定向验证首项 NaN、后续 NaN、Inf、跨核并列、非正步长、跨度溢出及空结果指针。910B3 用 `msprof op` 预热 5 次后至少 10 次有效采样，计入所有计算阶段，目标 NPU≤1.25×GPU。按任务书记录 workspace 峰值，上板结果待测。
