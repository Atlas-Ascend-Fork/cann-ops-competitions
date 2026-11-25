# GEMV Batched 五接口（Atlas A2/A3）设计文档

## 需求背景

[11 月社区任务](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)要求在 arch22 实现 `aclblas{S,HSH,HSS,TST,TSS}gemvBatched`，沿用 ops-blas 公共接口。

## 需求分析

每个 batch 独立计算 y ← α·op(A)x + βy。矩阵 A 为 m×n 列主序，lda≥max(1,m)。N 分支输出 m、归约 n；T/C 分支输出 n、归约 m，实数 C 与 T 相同。A/x/y 由设备上的指针数组分别寻址，x/y 支持非零正负步长，各 y 区域互不重叠。

| 接口前缀 | A/x | y | α/β 与累加 |
| --- | --- | --- | --- |
| S | FP32 | FP32 | FP32 |
| HSH | FP16 | FP16 | FP32 |
| HSS | FP16 | FP32 | FP32 |
| TST | BF16 | BF16 | FP32 |
| TSS | BF16 | FP32 | FP32 |

窄类型按公开签名用 `uint16_t` 表示原始位模式；载入提升为 FP32，写回按 y 类型一次舍入。

## 详细设计

### Host 入口

参数顺序保持 handle、trans/m/n、α、Aarray/lda、xarray/incx、β、yarray/incy、batchCount。校验 handle、trans、维数、lda、非零步长、batchCount、标量和所需数组本体。零 m/n/batch 合法退出并豁免矩阵与向量数组；非空时按任务书校验数组本体，不读 Device 数组元素。负步长采用 64 位起点 (1−len)·inc。通过 handle 的 stream 异步执行；Host 标量按约定封装，Device 数组留在设备侧。

### Kernel 与调度

只发射一次批量 kernel，以 (batch, 输出段) 为并行单位。每个核取得所拥有的 batch 指针后复用，独占 y 逻辑段。N 路径按列段读取并用 x 标量/向量更新一段连续输出；T/C 路径沿每个原始列的连续行段计算点积，并在同核归约到输出项。片上输入块和累加块分配双缓冲及尾部掩码；单位步长走连续搬运，大步长逐逻辑项加载/写回。小输入优先一个核处理多个 batch，宽输出按段铺核。归约维很长且输出不足时先调大每核归约块，避免引入跨核部分和；按芯片 UB 预算确定块形。α=0 走 y 缩放、跳过 A/x 数据；β=0 跳过旧 y。

## 可维可测分析

保持五个公开 ABI 与 arch35 分发兼容。按附件每精度 200 条精度、40 条性能输入，FP32/FP16/BF16 输出用对应任务书 §3.2 混合容差和最大误差阈值（匹配率≥0.99）。CPU golden 使用 FP32 计算并按输出类型舍入；额外覆盖离散指针、负步长、矩阵 padding、跨批输出哨兵与空输入。性能采集 `msprof op`（5 次热身，后 10 次有效 kernel 耗时均值），逐 case 对比 NPU≤1.25×GPU；GTest 端到端另记。设计阶段无设备实测值。
