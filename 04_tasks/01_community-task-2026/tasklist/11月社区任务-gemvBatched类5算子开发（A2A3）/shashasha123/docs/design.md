# 需求背景（required）

## 需求来源

[11 月社区任务 · gemvBatched 类 5 个算子开发（A2/A3）](https://www.hiascend.com/activities/task-center/details/d82eff6522c34a1a89710f93bb8926b7)，依据 2026-11-24 修订任务书。网页下载包提供修订日期，任务书 Git commit 待发布方确认。

## 背景介绍

批量 GEMV 将多个独立的矩阵向量乘合并到一次句柄式 API 调用。A、x、y 各由一个 Device 指针表定位，批间可以分散存放。

`ops-blas` 上游已在 [arch22 初始实现](https://gitcode.com/cann/ops-blas/commit/feb4f9c4c737ee5987781d4fb560872ad4dd9e09) 中提供五个入口。本设计沿用公开 API、stream 下发和 SIMD 实现，补齐任务书规定的参数校验、转置和空操作语义，改进向量核调度，并扩展按任务用例和标杆验证的能力。

# 需求分析（required）

## 需求描述

| 接口 | A / x | 输出 y | 累加与标量 alpha、beta |
| --- | --- | --- | --- |
| `aclblasSgemvBatched` | FP32 | FP32 | FP32 |
| `aclblasHSHgemvBatched` | FP16 | FP16 | FP32 |
| `aclblasHSSgemvBatched` | FP16 | FP32 | FP32 |
| `aclblasTSTgemvBatched` | BF16 | BF16 | FP32 |
| `aclblasTSSgemvBatched` | BF16 | FP32 | FP32 |

沿用 [公开声明](https://gitcode.com/cann/ops-blas/blob/master/include/cann_ops_blas.h) 的参数顺序和 `uint16_t` 半精度位承载方式。两个标量从 Host 指针取值。A 为列主序 `lda × n` 的存储，逻辑矩阵为 `m × n`，所有转置模式均要求 `lda ≥ max(1, m)`。N 模式读长 `n` 的 x、写长 `m` 的 y。T/C 模式读长 `m` 的 x、写长 `n` 的 y，实数 C 与 T 语义相同。

## 需求拆解

1. 五个 API 共用参数校验和 tiling，支持 N/T/C、非紧凑 `lda`、有符号跨步和非连续批地址。
2. 合法零维或零 batch 在 Host 返回成功。有效计算中 `alpha=0` 只缩放 y，`beta=0` 跳过 y 旧值读取。
3. 按输出 dtype 执行精度门限，按同参 GPU kernel 基线验收性能，并保留 ACL event 数据作对照。
4. 在 `blas/gemv_batched/arch22/` 和 `test/gemv_batched/arch22/` 内开发，更新算子产品支持说明，沿用 ops-blas 的构建和 GTest/CSV 入口。

# 详细设计（required）

## 算子分析

对批 `b` 的每个逻辑输出 `o`，先计算点积 `s`，再得到 `y[b,o] = alpha × s + beta × y[b,o]`。N 模式取 `A[b][k*lda+o]`，T/C 模式取 `A[b][o*lda+k]`。向量长度为 `L`、步长为 `inc` 时，逻辑位置 `i` 映射到 `max(0, (1-L)*inc) + i*inc`。Host 和 Device 在乘法前扩位，避免有符号绝对值和索引乘积溢出。

常规计算含 `batchCount × m × n` 次乘加。N 模式沿列累加行向量，T/C 模式将连续的列片段与 x 分段相乘后归约。矩阵每个逻辑元素参与一次，优势来自按输出分摊访存、批量搬运列片段和复用 x。标量缩放放在点积完成后，避免量级较大的 alpha 放大逐项舍入。FP16/BF16 的中间过程在 FP32 完成，写回时按最近偶数规则收窄。

## 算子实现

### Host 和 Tiling 设计

校验顺序按任务书为 handle、trans、m/n、lda、步长、batchCount、标量、指针数组，再判定 quick return。分别返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`、`ACLBLAS_STATUS_INVALID_ENUM` 或 `ACLBLAS_STATUS_INVALID_VALUE`。空问题允许三个 Device 指针表为空，标量指针仍须有效。附件负向用例将 `INT_MIN` 步长列为非法值，沿用该极值约定。非空计算先校验受 kernel 索引域约束的矩阵和向量跨度，再查有效 AIV 核数。

同一结构传递 dtype、N/T 类别、维度、步长、标量和输出分片。小问题优先按 batch 分核。batch 数量不足时，按 `batch × 输出段` 分配独立任务，每个输出保留完整归约维。单位步长路径按 32 字节搬运粒度选择输出段边界，最后一段单独裁剪。零 alpha 的问题按输出长度分派。计算使用 handle 绑定 stream 异步下发，传递 launch 错误，调用方同步 stream 后读取结果。

### Kernel 设计

每个 AIV 读取本任务的 y 指针。`alpha=0` 路径仅处理 y 的初始化、缩放和写回，常规路径再读取 A/x 指针。`beta=0` 时清零或直接以点积结果开始，避免加载 y；无效区和跨步间隙保持原状。

- N 路径以行片段为输出块，用 `DataCopyPad` 读取列片段并保留真实 `lda`，加载 x 的一段，按列乘加并归约到同一行块。规则形状使用多列搬运，归约尾部仅处理有效列。通用分支适配短行、填充和跨步。
- T/C 路径以列片段为输出块。K 维分段，x 在同一列组内复用，逐列乘法和向量归约得到一组 FP32 点积。最后一段按有效行数裁剪，中间对齐填充值为零。
- 连续向量使用成块 DMA，跨步向量按有符号索引取出或写回。有限 UB 窗口允许合并跨步读，窗口超限时按元素精确访问。输出切分保证每个逻辑位置由一个任务写入，跨步写按实际元素发出。
- UB 容量预算包括 A/x 输入缓冲及其 FP32 数据、累加、归约和 dtype 写回。以运行架构可用 UB 和 32 字节对齐为上界限制 tile，处理尾块时 DMA 使用实际长度。MTE2、Vector、Scalar、MTE3 在读取依赖和复用缓冲前同步。

## 支持硬件

| 产品 | 架构 | 验证计划 |
| --- | --- | --- |
| Atlas A2（含 800T A2 / 910B3） | `arch22` | CANN 9.1.0 上完成精度和性能自验 |
| Atlas A3 | `arch22` | 使用同一份 Ascend C SIMD 实现，按任务书允许自验覆盖一种款型 |

## 算子约束限制

y 的有效写入位置由调用者保证在 batch 间互斥，并与本次 A/x 读取区域互斥。Host 只检查表指针，Device 表元素由调用方提供。无广播，A 的物理布局同一批内保持列主序。本任务保持 arch35 路径和公开签名兼容。

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 验证计划 | 标准来源 |
| --- | --- | --- |
| 精度 | 每变体 1,000 条，另跑仓内回归与专项校验。逐 batch 使用 Netlib `cblas_sgemv`，按输出 dtype 舍入；检查 NaN 对 NaN、同号 Inf、跨步间隙与 quick return。有限结果匹配率至少 99%，同时执行任务书绝对误差界及 ULP 规则。FP32 的 atol/rtol 为 `2^-13`，FP16 为 `2^-10`，BF16 为 `2^-7`。 | 任务书 §3.2、§3.5，[生态精度标准](https://gitcode.com/cann/opbase/blob/28a1ce52ae856353589841226a0002af73314a6b/docs/zh/ops_precision_standard/mixed_tolerance_standard.md) |
| 性能 | 每变体 200 条，预热 5 次，记录随后 10 次成功执行。`msprof op` 的 kernel Task Duration 算术平均与同变体、同转置、形状、lda、步长、batch、alpha/beta 的 `gpu_ms` 比较，要求 NPU ≤ GPU × 1.25。另记 ACL event `avg_us`。检查基线采集参数和版本，记录缺少可比 GPU 实测的用例。 | 任务书 §3.3、§3.5 |
| 内存 | 任务书 §3.4 无独立内存指标。通过跨步哨兵、奇数尾块、非连续批地址、非零填充和边界用例检查写入范围。 | 任务书 §2.5、§3.4 |

## 兼容性分析

arch22 的 OP_C、合法零维和零 batch 行为按任务书补齐。测试覆盖多入参同时非法时的校验次序、`alpha=0` 的 NaN 输入、`beta=0` 的 NaN 旧输出、最小跨度和 batch 尾部。验收从固定 commit 重新构建，保留编译命令、测试输入与 golden 来源、结构化日志、GPU 基线和两套性能计时数据。
