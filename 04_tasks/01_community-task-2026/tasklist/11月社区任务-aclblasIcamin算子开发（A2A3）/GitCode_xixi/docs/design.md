# aclblasIcamin（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务书](https://www.hiascend.com/activities/task-center/details/6ff54aa152cf4edbbaf516fa36024893)规定复数最小模索引语义。以 ops-blas `60d9d0b153e2` 的 `blas/iamin/arch22/icamin_host.cpp` 为基线。本稿用于历史任务设计练习。

## 需求分析

`aclblasIcamin(handle,n,x,incx,result)` 返回复数向量中模值最小且最先出现的元素的 1-based 逻辑索引。输入 x 是 Device 上只读的交错 complex64 向量，每个元素占两个 FP32；result 是独立的 Device INT32 标量。n 和 incx 由 Host 传入，incx 以复数元素为单位。沿用公共 ABI 与 handle 绑定的 stream。

## 算子分析

正常路径中，逻辑索引 k∈[0,n) 对应物理复数位置 `p=k·incx`，其模值 `v[k]=abs(x[2p])+abs(x[2p+1])`（此处 x 按 FP32 分量编号）。比较采用 FP32 的实部、虚部绝对值之和。对非 NaN 候选按 `(v[k],k)` 字典序取最小值，归约结束后写 `k+1`，不把物理位置 p 当成结果。例如 incx=2、逻辑模值为 [4,1,1,3] 时，两个最小值位于物理位置 2、4，输出为 2。

任务书的串行参考以首值为初始最小值，只在严格小于时更新。首值为 NaN 时后续比较均为假，输出 1；首值有效时跳过后续 NaN。基线在向量归约前把 NaN 替换为 +Inf，并单独保存首值：若有有限最小值，替换的 NaN 不会被选中；若首值为 +Inf 且所有剩余值也是 +Inf 或 NaN，索引 0 已是最早候选；若首值为 NaN，最终强制写 1。先按 tile 顺序、再按核号合并等值候选，等价于选较小逻辑索引。

有效输入只需扫描一次，计算量 O(n)，最低有效读取 8n 字节，结果写入 4 字节。密集跨步会多读窗口中的步长空位；稀疏跨步减少有效槽之间的 GM 读量，代价是更多 DMA 子块和更大的 UB 槽。

## Host/Tiling 设计

先检查 handle，再检查 n<0 和 result 判空。空句柄返回 HANDLE_IS_NULLPTR；负 n、空 result、正常路径空 x 或跨度越界返回 INVALID_VALUE。n=0 或 incx<1 时在 handle stream 上异步写 4 字节零，不读取 x，不启动归约。普通路径用 int64 检查 `(n−1)·incx≤INT32_MAX`；Kernel 用 64 位计算实际 FP32 分量偏移。

设可用 AIV 核数为 C。非空输入先取 `requested=min(C,ceil(n/4096))`，再令 `perCore=align_up(ceil(n/requested),64)`、`cores=ceil(n/perCore)`。第 b 核处理 `[b·perCore,min(n,(b+1)·perCore))`，各区间连续且不重叠。Tiling 下发 n、incx、cores、perCore、tile、inputFloats、buffers、slots 以及布局和归约模式。incx=1 时使用连续路径，2≤incx≤4 时使用密集跨步路径，incx>4 使用稀疏路径。连续短切片采用单缓冲（tile 上限 14336），其余采用双缓冲（连续 tile 上限 7680，跨步上限 2048）；这些是源码基线值。

## Kernel 设计

### 搬运与模值计算

- **连续**：每 tile 连续搬运 `8·valid` 字节，在 UB 对交错实虚分量取绝对值，再用 `ReducePairElem` 把每对分量相加。大 tile 拆成满足向量指令 repeat 上限的批次，末尾按有效分量数提交。
- **密集跨步**：DMA 连续读取 `8·((valid−1)·incx+1)` 字节，按 `8·incx` 字节间距通过 Gather 分离实部与虚部，逐项 Abs 后 Add。
- **稀疏跨步**：每个 8 字节复数读到独立的 32 字节 UB 槽，以 32 字节为 Gather 间距。若 `8·(incx−1)` 超出 DMA 源间隔字段，则逐元素提交精确 8 字节搬运。

双缓冲路径预取下一 tile，再计算当前 tile；单缓冲串行搬运和计算。任一 tile 的 `valid=min(tile,区间剩余长度)`；DMA 按有效窗口读取，向量暂存补齐的 lane 不参加 ReduceMin。首逻辑元素的原始模值在 NaN 替换之前保存。

### 归约与结果写回

每个 tile 输出模值和 tile 内最早索引；在 INT32 域加上 tile 全局起点得到逻辑索引，避免 FP32 在 2²⁴ 之后丢失整数精度。每 32 个 tile 或到达尾部时，按候选顺序做小组归约。核内以严格小于更新，保留等值时较早的区间。核间的候选槽按核号排列，最小值并列时取最早槽。

- **单核**：直接归约并写结果，不分配 GM workspace。
- **两阶段**：多核、连续布局且每核最多 32 个 tile 时，第一阶段每核写独立的 32 字节候选槽，核 0 额外写 256 字节首值暂存；同 stream 的第二个 kernel 读取并合并。workspace 为 `32·cores+256` 字节。
- **融合**：其余多核算例每核写独立候选槽，所有核经过 `SyncAll` 后由核 0 读槽归约；workspace 为 `32·cores` 字节。

输出路径先检查首值 NaN，再选择 1 或归约索引加 1。单一写者在向量计算完成后，仅 DMA 写 result 的 4 字节，避免触碰相邻位置。

## 硬件与约束

目标为 Atlas A2/A3（arch22）AIV，使用 GM/UB 搬运和向量归约。Host 读取实际 AIV 核数及 UB 容量。UB 预算计入对齐后的输入队列、FP32 模值、跨步实虚拆分及 Gather 偏移、候选槽、每 256 项占 32 字节的比较掩码、归约与 Select scratch。基线在平台信息或 UB 不足时返回 INTERNAL_ERROR，多核 workspace 不足或启动失败返回 EXECUTION_FAILED。调用异步，Host 读回前同步对应 stream。

## 可维可测分析

`test/iamin/icamin/arch22/` 使用任务书配套精度与性能集，CPU 严格小于的参考循环生成 1-based golden，使用 `EXPECT_EQ` 比较 INT32。定向用例覆盖并列跨 tile/核、首项与后续 NaN、全部 +Inf、步长 1/2/4/5、DMA 源间隔上界、超 2²⁴ 的逻辑索引、非整 tile、空形状异步清零和非法跨度。对 result 邻接位置填哨兵检查精确写回。

910B3 上以 `msprof op` 预热 5 次再取至少 10 次有效样本，计入单次 API 调用的所有计算阶段，目标 NPU≤1.25×GPU。按布局与归约模式分开记录时延及 workspace 用量，便于比较融合与两阶段路径。上板结果待测。
