# aclblasGemmGroupedBatchedEx A2/A3 设计文档

## 需求背景

需求来源为 [10 月社区任务](https://www.hiascend.com/activities/task-center/details/36dbb2d3e38d4b4c96fc71b4869cd3e7)及其配套任务书。本次补齐 ops-blas 的 arch22 实现，使用公共头中已有的 aclblasGemmGroupedBatchedEx 接口。950 的 arch35 实现继续独立构建。

同一个调用包含多个形状、转置与批数可以不同的 GEMM 分组。每个分组共享 alpha、beta、m/n/k 和 leading dimension；矩阵指针数组按分组顺序展开，保留在设备上。

## 需求分析

每个实例计算：

`C[g,i] = alpha[g] * op(A[g,i]) * op(B[g,i]) + beta[g] * C[g,i]`

矩阵使用列主序。实数类型下 OP_C 与 OP_T 等价。主路径支持 ACL_FLOAT 输入输出和 ACLBLAS_COMPUTE_32F，不宣称 FP16、BF16、FP8 或 INT8 扩展能力。

实现覆盖不同分组和组内批数、九种 N/T/C 组合、leading dimension 填充、Host/Device 标量，以及 m/n/batch 为零、k 为零、alpha 为零等边界。非法入参遵循任务书规定的返回顺序。

## 详细设计

### Host 侧

1. 检查 handle、数据类型、groupCount。合法类型下 groupCount=0 直接成功。
2. 检查必要的 Host 数组，并逐组验证操作枚举、非负维度、非负批数和 leading dimension。在任一输出下发前完成所有组的校验。
3. 按总实例数检查设备指针数组。C 数组本身没有长度信息，因此调用者必须保证数组长度；不能从 C ABI 检出多传或少传的数组元素。
4. 查询 alpha/beta 的内存位置。Host 标量按值写入 tiling；Device 标量由 kernel 按分组索引读取。
5. 合并相邻、Host 标量及其余参数均相同的分组。仅合并展开设备指针数组中连续的实例，不改变实例顺序。Device 标量分组不合并。
6. 计算各分组的工作区布局，使用最大分组需求作为可复用工作区，在 handle 绑定的同一 stream 顺序下发各组任务。

公共接口没有额外 D2H 拷贝或显式 stream 同步。工作区分配沿用库内 EnsureDefaultWorkspace；默认工作区扩容可能按该公共辅助函数的规则同步。调用者读回结果前同步 handle stream。

### 分核与分块

一般矩阵使用 256×256 输出分块，baseM×baseN 为 64×256；双转置场景使用 128×128、baseM×baseN 为 64×128。FP32 累加保留 L0C 余量。K 维每段最多 8192，超长 K 分段累加。NN/NT/TT 的 baseK 为 32，特定转置路径使用 64。关闭 HF32，避免舍入到较短尾数。

将列主序结果视为行主序转置：`C^T = op(B)^T * op(A)^T`。四个实数转置特化共享同一实现模板。对于 transA=N 且 n≤32 的窄输出，另用 32×512 分块、baseM×baseN=32×256，减少同一核心重复初始化小输出 tile 的开销。每个输出 tile 由一个核负责，各实例依次展开到 tile 任务空间，任务按 blockIdx 跨步分配。

m≤16、n≤16、k≤32 的小矩阵使用 AIV FLOAT32 递推，避免 Cube 补齐与启动开销。A 一次搬入 UB，B 的当前列保存于 UB。每个任务并行计算最多四个输出列；极端有限值使用幂次缩放保护乘加中的中间溢出，不改变比较阈值或 reference。普通输入复用本地数据，只在有真实流水依赖时同步。

### 数据搬运与边界

M/N 补齐到 16，K 按转置路径补齐到 32 或 64。输入物理方向保持不变，去掉 leading dimension 间隙。补齐区域显式写零；DataCopyPad 的尾部使用显式零填充。

窄行在同一任务内合并多行搬运，以减少小 K 矩阵的逐行开销。每次最多搬运 1024 个 FLOAT32 元素，过宽行继续分块。对原始输出只写 m×n 逻辑区域，保留 ldc 填充字节。

alpha=1、beta=0、输出紧凑且 M/N 对齐时，Cube 直接输出 C。其他情况将乘积放入工作区，由 AIV 完成 alpha/beta 处理和合法区域回写。beta=0 时不读取旧 C；alpha=0 或 k=0 时不引用 A/B 数据。无乘积且 beta=1 直接返回，保留 C 的全部位模式。

### 工作区

需要补齐时，分配各实例的 padded A/B；非直接输出时另分配 padded product。组之间顺序复用工作区，需求为各个合并分组的最大值。分配前进行乘法和累加溢出检查，遵守库内 2 GiB 工作区上限；超限返回 ALLOC_FAILED。

## 支持硬件

| 产品 | 实现 | 自验证 |
| --- | --- | --- |
| Atlas A2 系列 | arch22 | Ascend910B3 |
| Atlas A3 系列 | arch22 | 本次未单独上板 |

自验证按任务书允许的一种款型完成。软件环境为 CANN 9.1.0，golden 链接 Netlib BLAS 3.10.0 的 cblas_sgemm。

## 约束

输入输出为 FLOAT32，各矩阵仅支持 leading dimension 描述的列主序布局。调用者负责设备地址、存储空间、数组长度和有效句柄的生命周期。支持运行时形状，不支持跨矩阵广播。C 原地更新。非 FLOAT32 组合返回 NOT_SUPPORTED。

## 可维可测分析

精度用例从随任务包提供的 CSV 加载，逐组、逐实例比对 Netlib cblas_sgemm。有限值要求 rtol=atol=2^-13、匹配比例≥0.99，同时最大绝对误差≤max(0.01,32×ULP)。非有限值按任务书单独比对，检查 C 填充区及 no-op 的位模式。负向状态用例与 CSV 输入格式校验单独统计。

性能每例先进行一次准备调用，再预热 5 次，记录后续 60 次有效调用。各次调用中所有计算 kernel 的 task duration 相加，作为完整接口的设备计算耗时。性能判据使用任务包 gpu_baseline.csv，NPU 平均耗时≤标杆/0.8；同时列出是否达到标杆原值。报告保留逐次样本、采集日志和库文件 SHA256。

本环境 msprof op 在设备虚实编号转换时失败。使用 CANN 9.1 的 msprof task-time 采集真实 kernel duration，记录采集方式差异，供验收方复核。没有使用 GTest 总墙钟耗时代替设备耗时。

任务包的 GPU 型号表述不一致，报告仅称“任务提供的 GPU 基线”。任务书 §3.4 写明内存不涉及，但交付表另有 workspace 相对 GPU 要求；缺少 GPU workspace 基线时不虚填达标结论，需在正式验收前确认口径。

## 兼容性

公共 API 签名与头文件不变。新增 arch22 文件通过现有 SOC 构建选择机制参与编译。arch35 的实现与支持类型保持原有行为。测试目录按架构隔离，随任务提供的原始 CSV 保持不变。
