# aclblasChpr2（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/7a1275ee6b654459957fc27c8dde32b0)要求在 ops-blas 增加 arch22 的复数 Hermitian 压缩秩-2 更新，保留现有 arch35 实现及接口。

## 需求分析

`aclblasChpr2(handle,uplo,n,alpha,x,incx,y,incy,AP)` 将所选三角原地更新为 A + αxyᴴ + conj(α)yxᴴ。AP、x、y、α 为交错 FP32 复数。上三角按列存 0…j 行，下三角存 j…n−1 行；AP 总长 n(n+1)/2。写入对角时只累加原实部和 2·Re(α xᵢ conj(yᵢ))，虚部写 +0；α=0 或 n=0 时保持 AP 位模式。允许 x/y 以负步长反向遍历。

## 详细设计

### Host 与分核

保留公共 ABI，arch22 放在 `blas/spr2/arch22/`。检查句柄、n、α 指针、uplo、incx/incy（含 INT_MIN）；合法的 n=0 返回成功，随后检查 x/y/AP 与 α=0。采用 64 位计算上三角列起点 j(j+1)/2、下三角起点 j(2n−j+1)/2 和有符号向量地址。Host 标量按值传入；Device 标量在内核中判零。始终用 handle 的 stream 异步启动。

将 AP 按连续且互不重叠的元素区间分核，边界倾向对齐写回块，同一列可横跨多个核。核对区间起点使用整数二分找列，再增量更新列界，不对每个元素做逆三角索引。当末尾不足一个搬运块时走精确长度搬运；不得用读改写触碰其他核的所有权区间。

### Kernel 流水

每个连续列段缓存 xⱼ/yⱼ，先算列系数 α·conj(yⱼ) 与 conj(α)·conj(xⱼ)，随后用 FP32 向量乘加并行计算 AP 段。连续向量直接搬运，跨步向量走逻辑索引加载；同一 UB tile 中重复使用列系数。对角采用专门的实部表达式；AP 输入按块拆分实虚，掩码写回有效项。按 UB 容量选择段长与双缓冲，小时延路径关闭双缓冲并合并短列。分核只依据 n、对齐与设备资源。

## 可维可测分析

`test/spr2/chpr2/arch22/` 使用附件 1000 条精度数据、200 条性能数据；对照 CBLAS chpr2，独立统计实虚分量（FP32 rtol=atol=2^-13、匹配率≥0.99，并按任务书检查最大误差）。用负步长、首末列边界、零 α、对角虚部预填哨兵和空入参检查写入范围及错误优先级。正式性能使用 `msprof op` kernel duration，与 GPU kernel-only 逐 case 比较 NPU≤1.25×GPU。记录非单位步长和列段尾部的性能分布；结果待实测。
