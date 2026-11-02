# 需求背景（required）

## 需求来源

CANN 训练营哈尔滨工业大学（深圳）aclblasChbmv（950）社区任务，任务中心：https://www.hiascend.com/activities/task-center/details/47b20ecf102b45d39a549baf1bedaebf?menu=tasks 。

## 背景介绍

为 ops-blas 补齐 complex64 Hermitian 带状矩阵向量乘。基于源码提交 5c336989ca707c998e45fa1100558e94883370d7，复用句柄、stream、Ascend C kernel 直调及 CSV/GTest 工程。代码位于 blas/sbmv/arch35，统一 API 位于 include/cann_ops_blas.h。

# 需求分析（required）

计算 y = alpha A x + beta y，A 为 n×n Hermitian 带状矩阵，设备存储为列主序 lda×n 的 aclblasComplex。alpha/beta 在 Host，A/x/y 在 Device。支持上下三角、正负非零步长、padding、k 超过 n-1、n=0。

参数检查顺序：n<0 返回 INVALID_VALUE；handle 空返回 HANDLE_IS_NULLPTR；n=0 返回 SUCCESS；随后校验 uplo、k、lda、incx、incy 以及五个指针。lda 比较提升到 int64_t，避免 k+1 溢出。

# 详细设计（required）

## 算子分析

有效带宽 w=min(k,n-1)。每个输出行 i 只遍历 j∈[max(0,i-w),min(n-1,i+w)]。访问原始带存储时继续使用输入 k，防止 k>n-1 的上三角对角位置改变。

上三角：i≤j 使用 lda*j+k+i-j；i>j 使用 lda*i+k+j-i 并取共轭。下三角：i≥j 使用 lda*j+i-j；i<j 使用 lda*i+j-i 并取共轭。对角仅加载实部并分别乘 x 实虚部，避免被忽略的虚部 NaN 污染。

向量索引为 inc>0 时 i*inc，否则 (n-1-i)*(-inc)。步长和乘法使用 64 位整数，覆盖 INT_MIN 步长在 n=1 场景下的合法语义。

## 算子实现

### Host 侧设计

复用 GetAivCoreCount 和仓内 SIMT 线程常量，按输出行数分配核心。tiling 传递 n/k/lda/uplo、复数 alpha/beta、64 位步长。使用 handle->stream 异步发射一个 AIV kernel；不分配 workspace，不做设备全局同步。

### Kernel 侧设计

首版采用每 SIMT 线程计算一行、grid-stride 遍历输出行，无原子写、无核间同步。每行复数累加保存在寄存器。按Netlib将直接列更新与共轭点积分别累加，保持alpha作用顺序，避免在大值抵消时扩大与单精度golden的差异。上下三角为模板特化。alpha=0 时计算 beta*y；beta=0 时完全跳过 y 的加载。输出只更新逻辑向量位置，保留步长空隙。

性能优化将在真实 profile 后决定：大带宽可按行内分块进行并行归约，x 可缓存到 UB；使用实测时间指导修改，保持当前首版为正确性参照。当前runtime task-time实测200条全部达到GPU/NPU≥0.4，最低约0.475。任务指定msprof op存在物理/逻辑设备编号映射失败，正式采集方式待平台处理或验收方确认。

## 支持硬件

| 硬件 | 实现目标 |
| --- | --- |
| Ascend 950PR | arch35，CANN 9.1.0 |

## 算子约束限制

不涉及广播或一般非连续 tensor。A/x/y 的存储容量由调用者保证。y 原地更新；跨 A/x/y 的重叠别名不作为本次支持契约。接口与其他产品线共用；此次只增加 arch35 实现。

# 可维可测分析

## 精度标准/性能标准

原始 CSV 1200 条完整保留：1000 精度、200 性能。GTest 直接调用公共 API，负向用例不由包装器提前返回。golden 为 cblas_chbmv(CblasColMajor)，以 Netlib 3.10.0 为验收环境目标，实际加载库版本记录在报告。

实部/虚部分别统计：rtol=atol=2^-13，matched_ratio≥0.99 且 max_abs_error≤max(1e-2,32×ULP(golden在最大误差点的值))。非有限值单独比对，同 NaN、同号 Inf、NaN 与 Inf 视为等效，有限/非有限混合判失败。

测试对 padding、未引用带和对角虚部写入 NaN；beta=0 时把旧 y 设为 NaN，验证实际访存语义。每个随机用例按种子奇偶在均匀与正态分布间切换，保留 CSV 指定标量。另增 API 校验顺序与整数边界测试，以及40条固定种子的均匀/正态随机复数标量补充测试。

性能要求 NPU 平均 kernel 时间≤H100 gpu_ms×1000/0.4，单位统一为 us。原材料有 10、>10、>50 采样表述；计划提供 5 次预热、60 次有效样本，并给出前 10 次与全部60次统计。每次恢复相同 y 输入，恢复开销不计入 kernel task duration；只统计 msprof 的 chbmv kernel task。

## 兼容性分析

新增 aclblasChbmv 声明不修改其他 API。复用 sbmv 目录和构建自动发现机制。独立分支仅包含相关源码、CSV、测试和产品文档。

## 验证状态

已完成950原生构建。原始1000精度用例、200性能场景精度及接口边界测试通过。200条runtime task-time性能实测数值达标。40条随机标量补充测试全部通过。设计评审、任务验收和代码合入待官方处理。
