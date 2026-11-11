# 需求背景（required）

## 需求来源

昇腾社区 2026 年 11 月 `aclblasChbmv` A2/A3 算子开发任务，补齐 ops-blas 单精度复数 Hermitian 带状矩阵向量乘能力。[任务入口](https://www.hiascend.com/activities/task-center/details/ea8efcc9f98c4d54930f8b8e94a59366?menu=tasks)

## 背景介绍

计算 `y = alpha × A × x + beta × y`。A 使用列主序带状存储，实际内存为 `lda×n` 个 complex64。上三角模式的主对角位于带状数组第 k 行，下三角模式位于第 0 行。未存储的一侧通过共轭对称性恢复。主对角只使用实部。

# 需求分析（required）

## 需求描述

在 `include/cann_ops_blas.h` 新增公共 `aclblasChbmv` 声明，通过 handle 绑定的 stream 异步直调 Ascend C kernel。实现位于 `blas/sbmv/arch22`，测试位于 `test/sbmv/chbmv/arch22`。

## 需求拆解

1. 支持 UPPER/LOWER、正负非零步长、lda padding、k 超过 n−1、零维 quick return。
2. 按任务书顺序验证 handle、n 和其余参数。跨度及 `k+1` 在 int64 域校验，拒绝零步长、INT32_MIN 步长及超过 INT32_MAX 的向量跨度。
3. beta=0 时跳过旧 y 的读取。仅输出 y 的逻辑元素，保留步长间隙。
4. 通过 Netlib 3.10.0 FP64 Golden 精度测试，并对照随包 GPU 基线逐例验证性能。

# 详细设计（required）

## 算子分析

对输出行 i，只计算 `j∈[max(0,i−k),min(n−1,i+k)]`。UPPER 的直接元素地址为 `j×lda+k+i−j`，镜像元素地址为 `i×lda+k+j−i`。LOWER 采用对应的 `i−j` 或 `j−i` 带内偏移。镜像元素的虚部取反。

`aclblasComplex` 实部、虚部各为 float32。所有全局地址乘积使用 int64，循环上界采用剩余长度计算，避免接近 INT32_MAX 时回绕。

## 算子实现

### Host 设计

参数验证顺序为 handle 非空、n 非负、n=0 成功返回，再验证 uplo、k、lda、步长及数据指针。非法 uplo 返回 `ACLBLAS_STATUS_INVALID_VALUE`。

Host 查询可用 AIV 核数和 UB 容量。Tiling 按值传入，包含尺寸、步长、复系数、核间分工和路径选择。各次调用使用本次传入的系数。

### 通用逐行归约

各核处理独立输出行。每侧带状区间按最多 1024 个复数分块。连续区间采用单块 DMA，跨列区间通过 DataCopyPad 的源 stride 搬运。Gather 解交织为实部和虚部，向量乘加后分别 ReduceSum。

左右两侧分别归约，缩短大均值正态分布输入在复系数相消时的误差传播。负 incx 在 UB 中调整元素顺序。常见连续向量在核内缓存 x、旧 y 和主对角，输出按核批量写回。

基础 UB 为 `(26×1024+16)×4` 字节。连续缓存仅在 `n≤4096` 且单核行数不超过 1024 时启用，额外空间包括对齐后的 `2n` 个 float、`2×rowsPerCore` 个 float 和 `8×rowsPerCore` 个 float，总量低于 A2/A3 的 192 KiB。

### 对角方向向量化

连续向量场景每块处理 64 行。一个向量指令同时更新 64 个输出，矩阵数据通过 96 列滑动窗口进入 UB。X 窗口保留带宽范围内的邻域。

八组独立部分和在结束时按二叉树合并，减小连续数百次累加的舍入误差。全内部块使用固定有效掩码，边界块使用逐对角有效掩码。

Host 以 `ChbmvDiagUbBytes` 计算矩阵窗口、X 窗口和 36 个 64-float 向量的空间需求，并预留 1024 字节。满足预算的形状才进入该路径。

在 `16≤n≤4096` 范围内，较长的带宽分成八份，由全部 AIV 核处理行块与带宽块的组合。小矩阵从 k=16 起采用分带，n≥128 时从 k=96 起采用分带。部分结果写入 handle 的现有 workspace，随后在同一 stream 上启动归并 kernel，合并部分和并执行 alpha/beta 更新。所需 workspace 为 `n×8×sizeof(aclblasComplex)`。容量不足时使用单 kernel 路径。

### 连续列路径

对 `alpha=(1,0)`、`beta=(0,0)` 的连续向量，较大矩阵按存储列分工。每个核循环处理间隔相同的列，连续读取已存储三角带。

每个非对角元素同时贡献 `A(i,j)×x(j)` 和 `conj(A(i,j))×x(i)`。各核在 UB 保存自己的输出部分和，末尾写入独立 workspace 区域。归并 kernel 批量搬入所有核的部分和，采用八组部分和及树形合并，统一写回 y。该路径覆盖 `2048≤n≤4096`、`k≥128`，workspace 为 `n×blockDim×sizeof(aclblasComplex)`。

该路径最大 UB 为 `(4×align8(n)+11×1024+32)×4` 字节，n=4096 时为 110720 字节。各核 workspace 区域互不重叠，两个 kernel 通过同一 stream 的执行顺序建立依赖。

### 特殊数值处理

主对角乘法省略虚系数项，避免无意义的 `0×Inf`。普通向量路径产生非有限结果时，对受影响输出行采用带二进制缩放和乘积误差补偿的标量恢复路径，避免中间溢出把本可表示的最终结果破坏。

### 同步与资源生命周期

核内使用 TPipe/TBuf 管理 UB。MTE2→V 和 V→MTE2 通过配对 HardEvent 同步，涉及标量读取、特殊值恢复及输出写回的阶段使用完整流水同步。归并前的 workspace 由前一个 kernel 完整写入。所有输出行由唯一的归并线程块写回。

实现复用 handle 既有 workspace。空间不足时选择无 workspace 的路径。输入设备内存、handle 和 stream 的生命周期遵循公共 BLAS 接口约定。

## 支持硬件

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2 训练/推理系列 | 支持 |
| Atlas A3 训练/推理系列 | 支持 |

自验证使用 Ascend 910B3、CANN 9.1.0。A3 采用相同 arch22 实现。

## 算子约束限制

支持的内存布局由 lda、incx、incy 描述。alpha/beta 为 Host 指针，A/x/y 为 Device 指针。读取 y 的结果前同步绑定的 stream。

# 可维可测分析

## 精度标准/性能标准

| 项目 | 验证方式 |
| --- | --- |
| 精度 | 任务包 1000 例、正态补充 1000 例、5 个补充边界及空 handle 单测，实部/虚部分别使用任务书 FLOAT32 标准 |
| 性能用例正确性 | 额外验证全部 200 个性能用例输出 |
| 性能 | 910B3 上 msprof op，5 次内部 warmup，每例 10 次公共接口调用，汇总全部 kernel 的 Task Duration 后取算术平均 |
| 性能门槛 | 每例平均耗时 ≤ 随包 GPU 基线的 1.25 倍 |
| 内存 | 任务书未设置验收指标 |

每个用例保留固定随机种子。未引用槽位与对角虚部采用 NaN 注入，测试负步长、padding、参数错误和非有限值。测试步骤及工具修复说明见测试目录 README，结果与日志置于 `task_submission`。

## 兼容性分析

新增公共接口采用同族 int 维数签名，保留现有 Ssbmv 接口及调用流程。A2/A3 共用 arch22 实现。测试框架补充 CANN 9.1.0 头文件枚举探测，保证该版本可构建 CSV 驱动测试。
