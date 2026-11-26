# 需求背景

## 需求来源

11 月社区任务：aclblasChpr2 算子开发（A2/A3）。接口、精度与性能要求来自任务附件。

## 背景介绍

Chpr2 是单精度复数 Hermitian 压缩矩阵的秩二更新。本工程以 ops-blas 提交
`6df9aa04a1d37288966e1b97a20ae8033ca2ce3c` 为基线，保留既有 arch22 向量核与多核分列结构，完善新任务要求的标量位置、参数顺序、特殊值语义和测试覆盖，并优化步长输入搬运。

# 需求分析

## 需求描述

`AP ← AP + alpha*x*yᴴ + conj(alpha)*y*xᴴ`。
复用公开接口 `aclblasChpr2`，数据类型为 COMPLEX64。矩阵阶数 n 为非负 int，向量步长支持正负非零 int，排除 INT_MIN。AP 为列主序上三角或下三角 packed 格式。

## 需求拆解

1. Host alpha 按值下发，Device alpha 在绑定 stream 的 kernel 内读取，保留异步语义。
2. 参数检查按 handle、n、alpha、uplo、incx/incy、n=0、tensor 指针、alpha=0 的顺序执行。
3. 只读写压缩三角。非零更新忽略对角输入虚部并将输出虚部置零，零 alpha 保持 AP 原样。
4. 对照 Netlib BLAS 3.10.0 的 cblas_chpr2 逐分量验证。
5. 正式性能以任务指定 910B3 上的 msprof op kernel duration 为准。

# 详细设计

## 算子分析

### 地址映射

对上三角 i≤j，packed 偏移为 `j*(j+1)/2 + i`。
对下三角 i≥j，列起点为 `j*(2*n-j+1)/2`，元素偏移为列起点加 `i-j`。
所有 GM 偏移使用 64 位。负步长时逻辑元素 i 的物理下标为 `(n-1-i)*abs(inc)`。

### 数学分解

每列预计算 `t1=alpha*conj(y[j])`、`t2=conj(alpha*x[j])`，然后更新列片段
`AP[i,j] += x[i]*t1 + y[i]*t2`，避免每个输出重复计算列系数。
对角采用参考实现的加法次序：`real(AP) + real(x[j]*t1 + y[j]*t2)`。

## 算子实现

### Host 设计

通过仓库公共 CheckPtrLocation 查询 alpha 所在位置。Host 零 alpha 直接返回。
Device alpha 指针随 kernel 参数下发，在设备侧判断零值后再访问 x/y/AP，不进行阻塞的标量 D2H。

启动核数为 `min(n, AIV核心数)`。沿列序号蛇形分配，将长列与短列配对，缓解三角区域负载差异。

Gather 索引表按 n≤64、n≤256、其余三档选择编译期长度，减少小矩阵不需要的初始化指令。分档只影响内部索引表覆盖范围，数学计算和地址规则相同；n>256 使用原有1024行分块。
每个核以 1,024 行为计算 tile，片段起止同时受矩阵三角边界、tile 边界与 n 限制。

### Kernel 搬运设计

| 输入特征 | 搬运方法 | 依据 |
| --- | --- | --- |
| inc=1 | 连续 DataCopyPad | 输入紧凑，直接搬运最省指令 |
| 跨度可放入 8,192 complex 的 UB 窗口 | 连续窗口加载，再 Gather 去除间隔或反转顺序 | 减少小事务与逐元素等待 |
| 中等步长 abs(inc)≤16、窗口不足 | 分段加载窗口，每段 Gather 到同一 1,024 行计算 tile | 维持列系数复用，避免缩小计算 tile |
| 更大步长、DMA stride 字段可表示 | 多块 DataCopyPad，UB 每个 complex 占 32 字节，再 Gather 去填充 | 避免每个元素独立 DMA 和同步 |
| DMA stride 字段不可表示 | 64 位地址逐元素加载 | 保留整个非零 int 步长范围的正确行为 |

上述条件来自步长、UB 容量和 DMA 参数范围，不引用测试编号或预置答案。
非单位步长时 UB 约占 154,368 字节，包含计算、索引、标量和 65,536 字节窗口缓冲。

### 数值处理

普通输入使用向量实数运算完成交错复数乘加。
每个输入 tile 做一次绝对值最大值检查；Inf/NaN 或幅值极大的输入进入 NPU 标量计算分支，以保留复数乘法的特殊值语义和中间溢出处理。
列系数独立处理复数乘法双 NaN 时的无穷大恢复。该路径仍在 NPU 上执行，不是 CPU 回退。

### 同步与边界

使用 MTE2→V/S、V→S、S→MTE3 等事件保证搬运、向量与标量操作之间的数据可见性。
AP 的每个元素由唯一列所属核写入。尾部用 DataCopyPad 精确控制有效字节，避免写入相邻列或分配区外。

## 支持硬件

面向 A2/A3 的 arch22。当前自测环境为 910B3、CANN 9.1.0、Ubuntu 22.04，未在 A3 上实测。

# 可维可测分析

## 精度标准

实部与虚部分别满足 atol=rtol=2^-13、matched_ratio≥0.99，以及逐元素绝对误差上限 `max(1e-2,32*ULP(golden))`。
1,200 行原始 CSV 分别在 Host 和 Device alpha 下运行，加四项专项测试，共 2,404 项。
输入输出均附保护区，检查越界写入、只读参数、空输入、零 alpha、非零对角虚部、负步长以及 Inf/NaN。
CSV 内容与官方附件一致，报告保留 SHA256 与测试二进制 SHA256。

## 性能标准

正式采集预热 5 次，取 10 个有效 kernel duration 的平均，目标为不超过 GPU kernel-only 标杆的 1.25 倍。
完整基线的 200 行按顺序对应性能子集，核验 uplo/n/incx/incy，重复 shape 不合并。
事件计时仅用于趋势比较。当前 HiDevLab 的 msprof op 设备编号转换失败，正式性能结果保持“待测”，不以零耗时或事件计时替代。
另提供 application 模式的 msprof op_summary 核耗时辅助分析：每条用例按一次精度调用、5 次预热、10 次采样执行，检查完整执行次序及每组16条记录后只取最后10条。此口径与事件计时独立记录，不声称完成任务指定的 msprof op 验收。

## 兼容性

公开函数签名和 arch35 实现不变。新增参数仅在 arch22 内部 Host/Kernel 之间传递。
以单算子目标完成构建，避免依赖其他算子测试的执行结果。复跑脚本会核验完整测试集合，失败和遗漏不会静默忽略。

## 来源

- 任务附件：aclblasChpr2_AtlasA2_A3_task_doc.md、chpr2_test.csv、gpu_baseline.csv。
- [ops-blas 源码](https://gitcode.com/cann/ops-blas)。
- [Netlib CHPR2 参考算法](https://netlib.org/blas/chpr2.f)。
- [社区任务设计文档模板](https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md)。
