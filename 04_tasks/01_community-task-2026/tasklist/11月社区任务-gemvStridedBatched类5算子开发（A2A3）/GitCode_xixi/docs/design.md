# GEMV Strided Batched 五接口（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/404250b6e18c4c3b853c73526ed0570f)要求 arch22 实现 `aclblas{S,HSH,HSS,TST,TSS}gemvStridedBatched`，采用固定间距批地址。

## 需求分析

第 b 批执行 yᵦ ← α·op(Aᵦ)xᵦ + βyᵦ；三个基址分别加 b×各自 stride（元素数）。A 是列主序 m×n，N 时 x/y 长度 n/m，T/C 时 m/n；C 与 T 对实数等价。S 的输入输出是 FP32；HSH/HSS 输入 FP16，输出 FP16/FP32；TST/TSS 输入 BF16，输出 BF16/FP32。五种接口均用 FP32 标量和累加，窄类型以 `uint16_t` 传递。

## 详细设计

### Host 与地址计算

沿用公共签名：handle、trans/m/n、α、A/lda/strideA、x/incx/stridex、β、y/incy/stridey、batchCount。校验 handle、trans、尺寸、lda、步长、stride、batchCount 及所需指针。stride 为 int64 元素数，必须为正；非空数组的一批占用至少覆盖 lda×n 或 1+(len−1)×|inc|，禁止相邻批重叠。用受检的 64 位运算计算负步长起点、末批偏移和总字节跨度，避免整数溢出。m/n/batchCount 为零时免读数据并允许数据基址为空；非法结构仍按仓库状态码返回。使用 handle 绑定的 stream。

### Kernel 与批调度

将 (batch, 输出 tile) 映射到核，每个输出 tile 单写者。先按 dtype 缩放批偏移，再按有符号 inc 处理逻辑向量坐标。N 路径分列段累计一组连续行输出，x 块在 UB 广播；T/C 路径读取连续行向量，做每个原始列的分段点积。tail 搬运采用有效范围描述，任何 stride 空隙和负步长洞均不覆盖。小矩阵将相邻 batch 连续分配到同核，复用调度和队列；大矩阵以输出范围铺核，避免跨核回写竞争。FP16/BF16 按位装载并转换到 FP32，最后一次舍入。α=0 跳过 A/x，β=0 直接写新值。依据 UB 实际容量切分两个输入缓冲与 FP32 累加缓冲。

## 可维可测分析

按任务书附件 1000 条精度、200 条性能输入覆盖五个接口。CPU golden 对各 batch 独立计算、按输出类型舍入，并执行 §3.2 的匹配率≥0.99 与最大误差联合判定。专项验证最小合法间距、间距多留一元素、重叠拒绝、偏移溢出、负步长、尾块与跨批哨兵。用 `msprof op` 取后 10 次有效 kernel 平均耗时（5 次热身），逐 case 检查 NPU≤1.25×GPU；端到端耗时另报。保持 arch35 ABI 不变，性能和精度待设备验证。
